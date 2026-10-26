"""Stage 4 单专家 annotation project、草稿与可恢复人工核验工作流。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
    stable_hash,
)
from oa_groundrag.artifacts.io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    first_symlink_component,
)
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
    safe_join,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.grounding.messages import build_mask_grounded_region_messages
from oa_groundrag.grounding.outputs import (
    RegionDraftQualityStatus,
    assess_region_draft_quality,
    parse_region_model_output,
    region_output_template,
)

from ..contracts import fail
from ..region_contracts import (
    ANNOTATION_QUEUE_SCHEMA,
    EVAL_MANIFEST_SCHEMA,
    REGION_MANIFEST_SCHEMA,
    REGION_RECORD_SCHEMA,
)
from ..region import region_asset_identity


ANNOTATION_PROJECT_SCHEMA = "oa_groundrag.mask_grounded_region.annotation_project.v1"
ANNOTATION_ASSIGNMENT_SCHEMA = "oa_groundrag.mask_grounded_region.annotation_assignment.v1"
ANNOTATION_WORK_SCHEMA = "oa_groundrag.mask_grounded_region.annotation_work.v1"
MODEL_DRAFT_SCHEMA = "oa_groundrag.mask_grounded_region.model_draft.v1"
MODEL_DRAFT_FAILURE_SCHEMA = "oa_groundrag.mask_grounded_region.model_draft_failure.v1"
MODEL_DRAFT_RUN_SCHEMA = "oa_groundrag.mask_grounded_region.model_draft_run.v1"
VERIFIED_ANNOTATION_SCHEMA = (
    "oa_groundrag.mask_grounded_region.expert_verified_annotation.v1"
)
DRAFT_CONFIG_SCHEMA = "oa_groundrag.mask_grounded_region.draft_config.v1"

CALIBRATION_SEED = 20260804
CALIBRATION_COUNT = 20
TRAIN_ANNOTATION_COUNT = 500
DEV_REFERENCE_COUNT = 100
VERIFICATION_STATUS = "expert_verified"
SINGLE_EXPERT_ANNOTATOR = "expert"
DRAFT_MODEL_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
DRAFT_MODEL_REVISION = "60595ebc30ec8e3b1d3b9e65d4943ca011c0006a"


class AnnotationIntendedUse(StrEnum):
    """单专家资产只允许训练监督或开发参考，不提供 Gold 语义。"""

    TRAIN_SUPERVISION = "expert_verified_train_supervision"
    DEV_REFERENCE = "single_expert_dev_reference"


PROJECT_FIELDS = (
    "schema_version",
    "project_id",
    "asset_root",
    "asset_manifest_sha256",
    "asset_schema_version",
    "split",
    "intended_use",
    "seed",
    "baseline_only",
    "calibration_record_ids",
    "annotation_record_ids",
    "frozen_prompt_sha256",
    "frozen_draft_config_sha256",
    "formal_acceptance",
)
ASSIGNMENT_FIELDS = (
    "schema_version",
    "ordinal",
    "record_id",
    "source",
    "split",
    "target_status",
    "asset_identity_sha256",
    "partition",
)
DRAFT_FIELDS = (
    "schema_version",
    "draft_id",
    "draft_run_id",
    "record_id",
    "asset_identity_sha256",
    "messages_sha256",
    "raw_output",
    "parse_status",
    "description",
    "failure",
)
DRAFT_FAILURE_FIELDS = ("schema_version", "code", "message", "details")
DRAFT_RUN_FIELDS = (
    "schema_version",
    "draft_run_id",
    "config_sha256",
    "config_semantic_sha256",
    "config",
    "model_repository",
    "model_revision",
    "model_identity",
    "processor_identity",
    "prompt_text",
    "prompt_sha256",
    "generation",
    "partition",
    "record_ids",
    "record_ids_sha256",
    "formal_acceptance",
)
VERIFIED_FIELDS = (
    "schema_version",
    "record_id",
    "asset_identity_sha256",
    "draft_id",
    "annotator",
    "verification_status",
    "description",
)
WORK_FIELDS = (
    "schema_version",
    "record_id",
    "annotator",
    "editor_text",
)
STATUS_FIELDS = (
    "schema_version", "total", "drafted", "valid_drafts", "failed_drafts",
    "verified", "calibration_total", "calibration_verified", "complete",
    "formal_acceptance",
)
@dataclass(frozen=True)
class AnnotationAssetContext:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    records: tuple[Mapping[str, Any], ...]
    queue: tuple[Mapping[str, Any], ...]

def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("ANNOTATION_INVALID", f"{location} 必须是字符串键对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: Sequence[str], location: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        fail(
            "ANNOTATION_INVALID",
            f"{location} 字段不匹配",
            details={
                "missing": sorted(expected - set(value)),
                "unknown": sorted(set(value) - expected),
            },
        )


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("ANNOTATION_INVALID", f"{location} 必须是非空字符串")
    return value.strip()


def _sha256(value: Any, location: str) -> str:
    result = _text(value, location)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        fail("ANNOTATION_INVALID", f"{location} 必须是小写 SHA-256")
    return result


def _ordinary_file(path: Path, *, location: str) -> None:
    if not path.is_file() or path.is_symlink():
        fail("ANNOTATION_INVALID", f"{location} 必须是普通文件：{path}")
    if path.stat().st_nlink != 1:
        fail("ASSET_HARDLINK", f"{location} 不得是 hardlink：{path}")


def _ordinary_root(value: Path | str, *, location: str) -> Path:
    root = Path(os.path.abspath(Path(value)))
    linked = first_symlink_component(root)
    if linked is not None or not root.is_dir():
        fail("ANNOTATION_INVALID", f"{location} 必须是无 symlink 的目录：{root}")
    return root


def _safe_record_filename(record_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", record_id):
        fail("ANNOTATION_INVALID", f"record_id 不能安全映射为工作文件名：{record_id}")
    return f"{record_id}.json"


def _manifest_file_identity(
    manifest: Mapping[str, Any],
    *,
    key: str,
    path: Path,
) -> None:
    contract = manifest.get(key)
    if not isinstance(contract, dict):
        fail("ANNOTATION_INVALID", f"manifest.{key} 缺少文件合同")
    if contract.get("path") != path.name or contract.get("sha256") != sha256_file(path):
        fail("ANNOTATION_INVALID", f"manifest.{key} 与现场文件身份不一致")


def load_annotation_asset(
    asset_root: Path | str,
    *,
    verify_asset_identity: bool = True,
) -> AnnotationAssetContext:
    """只读装载已发布 Region/Eval 资产并重算 annotation 所需身份。"""

    root = _ordinary_root(asset_root, location="asset_root")
    paths = {
        "manifest": root / "manifest.json",
        "records": root / "records.jsonl",
        "queue": root / "annotation_queue.jsonl",
    }
    for name, path in paths.items():
        _ordinary_file(path, location=f"asset_root.{name}")
    manifest = _mapping(read_json(paths["manifest"]), "asset manifest")
    schema = manifest.get("schema_version")
    if schema not in {REGION_MANIFEST_SCHEMA, EVAL_MANIFEST_SCHEMA}:
        fail("ANNOTATION_INVALID", f"不支持的 Stage 4 asset schema：{schema}")
    _manifest_file_identity(manifest, key="records", path=paths["records"])
    _manifest_file_identity(manifest, key="annotation_queue", path=paths["queue"])
    records = tuple(_mapping(row, f"records[{index}]") for index, row in enumerate(read_jsonl(paths["records"])))
    queue = tuple(_mapping(row, f"queue[{index}]") for index, row in enumerate(read_jsonl(paths["queue"])))
    if len(records) != len(queue):
        fail("ANNOTATION_INVALID", "records 与 annotation queue 数量不一致")
    record_by_id = {str(row.get("record_id")): row for row in records}
    if len(record_by_id) != len(records):
        fail("ANNOTATION_INVALID", "records 存在重复 record_id")
    seen: set[str] = set()
    for index, row in enumerate(queue):
        if row.get("schema_version") != ANNOTATION_QUEUE_SCHEMA:
            fail("ANNOTATION_INVALID", f"queue[{index}] schema 非法")
        record_id = _text(row.get("record_id"), f"queue[{index}].record_id")
        record = record_by_id.get(record_id)
        if record is None or record_id in seen:
            fail("ANNOTATION_INVALID", f"queue[{index}] record_id 未知或重复")
        seen.add(record_id)
        if record.get("schema_version") != REGION_RECORD_SCHEMA:
            fail("ANNOTATION_INVALID", f"records[{index}] schema 非法")
        if (
            row.get("assets") != record.get("assets")
            or row.get("program_facts") != record.get("program_facts")
            or row.get("target_status") != record.get("target_status")
            or row.get("split") != record.get("split")
        ):
            fail("ANNOTATION_INVALID", f"queue[{index}] 未绑定对应 Region record")
        if verify_asset_identity:
            expected_asset_id = region_asset_identity(
                root,
                _mapping(record.get("assets"), "record.assets"),
            )
            if row.get("asset_identity_sha256") != expected_asset_id:
                fail("ANNOTATION_INVALID", f"queue[{index}] asset identity 漂移")
    return AnnotationAssetContext(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(paths["manifest"]),
        records=records,
        queue=queue,
    )


def _foreground_bin(row: Mapping[str, Any]) -> str:
    mask = row.get("program_facts", {}).get("mask", {})
    ratio = mask.get("area_ratio") if isinstance(mask, dict) else None
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        fail("ANNOTATION_INVALID", "target record 缺少合法 area_ratio")
    return "small" if ratio < 0.01 else "medium" if ratio < 0.10 else "large"


def _target_novelty(
    row: Mapping[str, Any],
    *,
    seen_components: set[str],
    seen_locations: set[str],
) -> tuple[int, int]:
    mask = row.get("program_facts", {}).get("mask", {})
    if not isinstance(mask, dict):
        return (0, 0)
    count = mask.get("fragment_count")
    component = "single" if count == 1 else "multiple"
    location = str(mask.get("location_3x3", "unknown"))
    return (int(component not in seen_components), int(location not in seen_locations))


def select_calibration_record_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = CALIBRATION_SEED,
) -> tuple[str, ...]:
    """每来源四条；无 no-target 的来源使用第四条 target 补充组件/位置覆盖。"""

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_source.setdefault(_text(row.get("source"), "record.source"), []).append(row)
    if len(by_source) * 4 != CALIBRATION_COUNT:
        fail("ANNOTATION_INVALID", "v1 calibration 要求五个来源且每来源四条")
    selected: list[str] = []
    for source in sorted(by_source):
        rows = by_source[source]
        target = [row for row in rows if row.get("target_status") == "target_present"]
        no_target = [row for row in rows if row.get("target_status") == "no_target"]
        chosen: list[Mapping[str, Any]] = []
        seen_components: set[str] = set()
        seen_locations: set[str] = set()

        def choose(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
            if not candidates:
                fail("ANNOTATION_INVALID", f"calibration cell 无候选：{source}")
            ranked = sorted(
                candidates,
                key=lambda row: (
                    -_target_novelty(
                        row,
                        seen_components=seen_components,
                        seen_locations=seen_locations,
                    )[0],
                    -_target_novelty(
                        row,
                        seen_components=seen_components,
                        seen_locations=seen_locations,
                    )[1],
                    stable_hash(seed, source, row["record_id"]),
                ),
            )
            result = ranked[0]
            if result.get("target_status") == "target_present":
                mask = result["program_facts"]["mask"]
                seen_components.add("single" if mask["fragment_count"] == 1 else "multiple")
                seen_locations.add(str(mask["location_3x3"]))
            return result

        if no_target:
            chosen.append(min(no_target, key=lambda row: stable_hash(seed, source, row["record_id"])))
        for size in ("small", "medium", "large"):
            chosen.append(choose([row for row in target if _foreground_bin(row) == size and row not in chosen]))
        while len(chosen) < 4:
            chosen.append(choose([row for row in target if row not in chosen]))
        if len(chosen) != 4:
            fail("ANNOTATION_INVALID", f"calibration source 配额异常：{source}")
        selected.extend(str(row["record_id"]) for row in chosen)
    if len(selected) != CALIBRATION_COUNT or len(set(selected)) != CALIBRATION_COUNT:
        fail("ANNOTATION_INVALID", "calibration 必须是 20 个唯一 train records")
    return tuple(selected)


def _selected_records(
    context: AnnotationAssetContext,
    intended_use: AnnotationIntendedUse,
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    if intended_use is AnnotationIntendedUse.TRAIN_SUPERVISION:
        if context.manifest.get("schema_version") != REGION_MANIFEST_SCHEMA:
            fail("SPLIT_FORBIDDEN", "训练标注只允许 Region Corpus")
        if len(context.records) != TRAIN_ANNOTATION_COUNT or any(row.get("split") != "train" for row in context.records):
            fail("SPLIT_FORBIDDEN", "训练标注必须精确使用 500 条 train records")
        return context.records, False
    if context.manifest.get("schema_version") != EVAL_MANIFEST_SCHEMA:
        fail("SPLIT_FORBIDDEN", "开发参考只允许 OA-GroundedEval-dev")
    selection = context.manifest.get("selection")
    baseline_ids = selection.get("baseline_record_ids") if isinstance(selection, dict) else None
    if (
        not isinstance(baseline_ids, list)
        or len(baseline_ids) != DEV_REFERENCE_COUNT
        or len(set(baseline_ids)) != DEV_REFERENCE_COUNT
    ):
        fail("ANNOTATION_INVALID", "Eval-dev baseline_record_ids 必须精确为 100 条")
    by_id = {row["record_id"]: row for row in context.records}
    selected = []
    for record_id in baseline_ids:
        row = by_id.get(record_id)
        counterfactual = None if row is None else row.get("program_facts", {}).get("counterfactual")
        if (
            row is None
            or row.get("split") != "val"
            or not isinstance(counterfactual, dict)
            or counterfactual.get("kind") != "baseline_correct_mask"
        ):
            fail("SPLIT_FORBIDDEN", "开发参考只能包含 100 条 val baseline")
        selected.append(row)
    return tuple(selected), True


def _status_payload(
    assignments: Sequence[Mapping[str, Any]],
    drafts: Mapping[str, Mapping[str, Any]],
    *,
    verified_count: int,
    calibration_verified: int,
) -> dict[str, Any]:
    calibration = {row["record_id"] for row in assignments if row["partition"] == "calibration"}
    if not 0 <= verified_count <= len(assignments) or not 0 <= calibration_verified <= len(calibration):
        fail("ANNOTATION_INVALID", "annotation status count 非法")
    return {
        "schema_version": "oa_groundrag.mask_grounded_region.annotation_status.v1",
        "total": len(assignments),
        "drafted": len(drafts),
        "valid_drafts": sum(row["parse_status"] == "valid" for row in drafts.values()),
        "failed_drafts": sum(row["parse_status"] == "invalid" for row in drafts.values()),
        "verified": verified_count,
        "calibration_total": len(calibration),
        "calibration_verified": calibration_verified,
        "complete": verified_count == len(assignments),
        "formal_acceptance": False,
    }


def _write_status(project_root: Path) -> dict[str, Any]:
    _, assignments = load_annotation_project(project_root)
    drafts = load_model_drafts(project_root, assignments=assignments)
    verified = load_verified_work(project_root, assignments=assignments, drafts=drafts)
    calibration = {row["record_id"] for row in assignments if row["partition"] == "calibration"}
    status = _status_payload(
        assignments,
        drafts,
        verified_count=len(verified),
        calibration_verified=sum(record_id in verified for record_id in calibration),
    )
    atomic_write_json(project_root / "status.json", status)
    return status


def _status_snapshot(project_root: Path, assignments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = project_root / "status.json"
    _ordinary_file(path, location="status")
    status = _mapping(read_json(path), "status")
    _exact(status, STATUS_FIELDS, "status")
    count_fields = (
        "total", "drafted", "valid_drafts", "failed_drafts", "verified",
        "calibration_total", "calibration_verified",
    )
    if any(
        isinstance(status[field], bool)
        or not isinstance(status[field], int)
        or status[field] < 0
        for field in count_fields
    ):
        fail("ANNOTATION_INVALID", "annotation status count 类型非法")
    if (
        status["schema_version"] != "oa_groundrag.mask_grounded_region.annotation_status.v1"
        or status["total"] != len(assignments)
        or status["drafted"] > status["total"]
        or status["valid_drafts"] + status["failed_drafts"] != status["drafted"]
        or status["verified"] > status["drafted"]
        or status["calibration_verified"] > status["calibration_total"]
        or not isinstance(status["complete"], bool)
        or status["complete"] != (status["verified"] == status["total"])
        or status["formal_acceptance"] is not False
    ):
        fail("ANNOTATION_INVALID", "annotation status identity 非法")
    return status


def _write_status_after_drafts(
    project_root: Path,
    assignments: Sequence[Mapping[str, Any]],
    drafts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    previous = _status_snapshot(project_root, assignments)
    status = _status_payload(
        assignments,
        drafts,
        verified_count=int(previous["verified"]),
        calibration_verified=int(previous["calibration_verified"]),
    )
    atomic_write_json(project_root / "status.json", status)
    return status


def create_annotation_project(
    *,
    asset_root: Path | str,
    output_root: Path | str,
    intended_use: str,
    prompt_path: Path | str | None = None,
    train_project_root: Path | str | None = None,
    seed: int = CALIBRATION_SEED,
) -> dict[str, Any]:
    """创建可恢复工作根；源 Region/Eval 资产始终只读。"""

    try:
        use = AnnotationIntendedUse(intended_use)
    except ValueError as error:
        fail("ANNOTATION_INVALID", f"intended_use 非法：{intended_use}")
        raise AssertionError from error
    if isinstance(seed, bool) or seed != CALIBRATION_SEED:
        fail("ANNOTATION_INVALID", f"v1 calibration seed 必须为 {CALIBRATION_SEED}")
    # annotation 入口先绑定发布 ledger，避免在被替换或遗漏的源资产上开始人工工作。
    from ..region_validation import validate_region_asset_files

    validate_region_asset_files(asset_root)
    context = load_annotation_asset(asset_root)
    records, baseline_only = _selected_records(context, use)
    if use is AnnotationIntendedUse.TRAIN_SUPERVISION:
        if prompt_path is None or train_project_root is not None:
            fail("ANNOTATION_INVALID", "train project 必须只提供 prompt_path")
        prompt = Path(os.path.abspath(Path(prompt_path)))
        _ordinary_file(prompt, location="prompt_path")
        prompt_text = prompt.read_text(encoding="utf-8")
        frozen_prompt_sha: str | None = None
        frozen_config_sha: str | None = None
        selected_calibration_ids = set(select_calibration_record_ids(records, seed=seed))
        # 成员由分层选择器决定，落盘顺序仍服从冻结 Corpus 的 ordered IDs。
        calibration_ids = tuple(
            str(row["record_id"])
            for row in records
            if str(row["record_id"]) in selected_calibration_ids
        )
    else:
        if prompt_path is not None or train_project_root is None:
            fail("ANNOTATION_INVALID", "dev reference 必须只绑定 train_project_root 的冻结 prompt")
        train_root = _ordinary_root(train_project_root, location="train_project_root")
        train_project, _ = load_annotation_project(train_root)
        if train_project["intended_use"] != AnnotationIntendedUse.TRAIN_SUPERVISION.value:
            fail("ANNOTATION_INVALID", "train_project_root intended_use 非法")
        frozen_prompt_sha = train_project["frozen_prompt_sha256"]
        frozen_config_sha = train_project["frozen_draft_config_sha256"]
        if not isinstance(frozen_prompt_sha, str) or not isinstance(frozen_config_sha, str):
            fail("ANNOTATION_INVALID", "train project 尚未冻结 remaining prompt/config")
        prompt_file = train_root / "prompt.txt"
        _ordinary_file(prompt_file, location="train_project.prompt")
        if sha256_file(prompt_file) != frozen_prompt_sha:
            fail("ANNOTATION_INVALID", "train project prompt 已在冻结后漂移")
        prompt_text = prompt_file.read_text(encoding="utf-8")
        calibration_ids = ()
    if not prompt_text.strip():
        fail("ANNOTATION_INVALID", "annotation prompt 不能为空")
    record_ids = tuple(str(row["record_id"]) for row in records)
    queue_by_id = {row["record_id"]: row for row in context.queue}
    assignments = []
    for ordinal, row in enumerate(records):
        record_id = str(row["record_id"])
        queue_row = queue_by_id[record_id]
        assignments.append({
            "schema_version": ANNOTATION_ASSIGNMENT_SCHEMA,
            "ordinal": ordinal,
            "record_id": record_id,
            "source": row["source"],
            "split": row["split"],
            "target_status": row["target_status"],
            "asset_identity_sha256": queue_row["asset_identity_sha256"],
            "partition": (
                "calibration" if record_id in calibration_ids
                else "remaining" if use is AnnotationIntendedUse.TRAIN_SUPERVISION
                else "all"
            ),
        })
    project_id = "annotation_project_" + sha256_text(canonical_json({
        "asset_manifest_sha256": context.manifest_sha256,
        "intended_use": use.value,
        "seed": seed,
        "record_ids": list(record_ids),
    }))[:24]
    project = {
        "schema_version": ANNOTATION_PROJECT_SCHEMA,
        "project_id": project_id,
        "asset_root": str(context.root),
        "asset_manifest_sha256": context.manifest_sha256,
        "asset_schema_version": context.manifest["schema_version"],
        "split": "train" if use is AnnotationIntendedUse.TRAIN_SUPERVISION else "val",
        "intended_use": use.value,
        "seed": seed,
        "baseline_only": baseline_only,
        "calibration_record_ids": list(calibration_ids),
        "annotation_record_ids": list(record_ids),
        "frozen_prompt_sha256": frozen_prompt_sha,
        "frozen_draft_config_sha256": frozen_config_sha,
        "formal_acceptance": False,
    }
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_json("project.json", project)
        writer.write_jsonl("assignments.jsonl", assignments)
        atomic_write_text(writer.path("prompt.txt"), prompt_text)
        writer.path("verified").mkdir(parents=True, exist_ok=False)
        writer.path("work").mkdir(parents=True, exist_ok=False)
        writer.path("draft_runs").mkdir(parents=True, exist_ok=False)
        writer.write_jsonl("drafts.jsonl", [])
        root = writer.publish()
    status = _write_status(root)
    return {
        "ok": True,
        "root": str(root),
        "project_id": project_id,
        "record_count": len(assignments),
        "calibration_count": len(calibration_ids),
        "status": status,
        "formal_acceptance": False,
    }


def validate_assignment_row(value: Any, *, location: str) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, ASSIGNMENT_FIELDS, location)
    if row["schema_version"] != ANNOTATION_ASSIGNMENT_SCHEMA:
        fail("ANNOTATION_INVALID", f"{location} schema 非法")
    if isinstance(row["ordinal"], bool) or not isinstance(row["ordinal"], int) or row["ordinal"] < 0:
        fail("ANNOTATION_INVALID", f"{location}.ordinal 非法")
    for field in ("record_id", "source", "split", "target_status", "partition"):
        _text(row[field], f"{location}.{field}")
    _sha256(row["asset_identity_sha256"], f"{location}.asset_identity_sha256")
    if row["split"] not in {"train", "val"} or row["partition"] not in {"calibration", "remaining", "all"}:
        fail("SPLIT_FORBIDDEN", f"{location} split/partition 非法")
    if row["target_status"] not in {"target_present", "no_target"}:
        fail("ANNOTATION_INVALID", f"{location}.target_status 非法")
    return row


def load_annotation_project(
    project_root: Path | str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = _ordinary_root(project_root, location="project_root")
    project_path = root / "project.json"
    assignments_path = root / "assignments.jsonl"
    prompt_path = root / "prompt.txt"
    for location, path in (
        ("project", project_path), ("assignments", assignments_path), ("prompt", prompt_path)
    ):
        _ordinary_file(path, location=location)
    project = _mapping(read_json(project_path), "project")
    _exact(project, PROJECT_FIELDS, "project")
    if project["schema_version"] != ANNOTATION_PROJECT_SCHEMA or project["formal_acceptance"] is not False:
        fail("ANNOTATION_INVALID", "annotation project schema/formal_acceptance 非法")
    try:
        use = AnnotationIntendedUse(project["intended_use"])
    except (TypeError, ValueError) as error:
        fail("ANNOTATION_INVALID", "project intended_use 非法")
        raise AssertionError from error
    expected_split = "train" if use is AnnotationIntendedUse.TRAIN_SUPERVISION else "val"
    if project["split"] != expected_split:
        fail("SPLIT_FORBIDDEN", "project split 与 intended_use 不一致")
    _text(project["project_id"], "project.project_id")
    _sha256(project["asset_manifest_sha256"], "project.asset_manifest_sha256")
    if isinstance(project["seed"], bool) or project["seed"] != CALIBRATION_SEED:
        fail("ANNOTATION_INVALID", "project seed 非法")
    if (
        not isinstance(project["baseline_only"], bool)
        or project["baseline_only"] != (use is AnnotationIntendedUse.DEV_REFERENCE)
    ):
        fail("ANNOTATION_INVALID", "project baseline_only 与 intended_use 不一致")
    asset_root = _ordinary_root(_text(project["asset_root"], "project.asset_root"), location="project.asset_root")
    asset_manifest_path = asset_root / "manifest.json"
    _ordinary_file(asset_manifest_path, location="project.asset_manifest")
    asset_manifest = _mapping(read_json(asset_manifest_path), "project.asset_manifest")
    expected_asset_schema = (
        REGION_MANIFEST_SCHEMA
        if use is AnnotationIntendedUse.TRAIN_SUPERVISION
        else EVAL_MANIFEST_SCHEMA
    )
    if (
        project["asset_schema_version"] != expected_asset_schema
        or asset_manifest.get("schema_version") != expected_asset_schema
        or sha256_file(asset_manifest_path) != project["asset_manifest_sha256"]
    ):
        fail("ANNOTATION_INVALID", "project source manifest 漂移")
    if not (root / "prompt.txt").read_text(encoding="utf-8").strip():
        fail("ANNOTATION_INVALID", "project prompt 不能为空")
    assignments = tuple(
        validate_assignment_row(row, location=f"assignments[{index}]")
        for index, row in enumerate(read_jsonl(assignments_path))
    )
    if [row["ordinal"] for row in assignments] != list(range(len(assignments))):
        fail("ANNOTATION_INVALID", "assignment ordinal 不连续")
    ids = [row["record_id"] for row in assignments]
    if ids != project["annotation_record_ids"] or len(ids) != len(set(ids)):
        fail("ANNOTATION_INVALID", "project 与 assignments ordered IDs 不一致")
    expected_project_id = "annotation_project_" + sha256_text(canonical_json({
        "asset_manifest_sha256": project["asset_manifest_sha256"],
        "intended_use": use.value,
        "seed": project["seed"],
        "record_ids": ids,
    }))[:24]
    if project["project_id"] != expected_project_id:
        fail("ANNOTATION_INVALID", "project_id 重算不一致")
    expected_count = TRAIN_ANNOTATION_COUNT if expected_split == "train" else DEV_REFERENCE_COUNT
    if len(assignments) != expected_count:
        fail("ANNOTATION_INVALID", f"project 必须精确包含 {expected_count} 条")
    if expected_split == "train":
        calibration = [row["record_id"] for row in assignments if row["partition"] == "calibration"]
        if calibration != project["calibration_record_ids"] or len(calibration) != CALIBRATION_COUNT:
            fail("ANNOTATION_INVALID", "train calibration identity 不一致")
    elif project["calibration_record_ids"] or not project["baseline_only"]:
        fail("ANNOTATION_INVALID", "dev project 必须 baseline-only 且无 calibration")
    frozen = project["frozen_prompt_sha256"]
    if frozen is not None:
        if _sha256(frozen, "project.frozen_prompt_sha256") != sha256_file(prompt_path):
            fail("ANNOTATION_INVALID", "冻结 prompt 已漂移")
    frozen_config = project["frozen_draft_config_sha256"]
    if frozen_config is not None:
        _sha256(frozen_config, "project.frozen_draft_config_sha256")
    if use is AnnotationIntendedUse.DEV_REFERENCE and frozen_config is None:
        fail("ANNOTATION_INVALID", "dev project 必须绑定 train 冻结 config")
    return project, assignments


def validate_model_draft_row(
    value: Any,
    *,
    assignment: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, DRAFT_FIELDS, location)
    if row["schema_version"] != MODEL_DRAFT_SCHEMA:
        fail("ANNOTATION_INVALID", f"{location} schema 非法")
    for field in ("draft_id", "draft_run_id", "record_id", "parse_status"):
        _text(row[field], f"{location}.{field}")
    if (
        row["record_id"] != assignment["record_id"]
        or row["asset_identity_sha256"] != assignment["asset_identity_sha256"]
    ):
        fail("ANNOTATION_INVALID", f"{location} record/asset identity 不匹配")
    _sha256(row["messages_sha256"], f"{location}.messages_sha256")
    if row["parse_status"] == "valid":
        if not isinstance(row["raw_output"], str) or row["failure"] is not None:
            fail("ANNOTATION_INVALID", f"{location} valid draft 载荷非法")
        parsed = parse_region_model_output(row["description"])
        parsed_raw = parse_region_model_output(row["raw_output"])
        if parsed.target_status.value != assignment["target_status"]:
            fail("ANNOTATION_INVALID", f"{location} target_status 与程序事实不一致")
        if canonical_json(parsed_raw.to_dict()) != canonical_json(parsed.to_dict()):
            fail("ANNOTATION_INVALID", f"{location} raw_output 与 description 不一致")
    elif row["parse_status"] == "invalid":
        if (
            not isinstance(row["raw_output"], str)
            or row["description"] is not None
            or not isinstance(row["failure"], dict)
        ):
            fail("ANNOTATION_INVALID", f"{location} invalid draft 载荷非法")
        failure = _mapping(row["failure"], f"{location}.failure")
        _exact(failure, DRAFT_FAILURE_FIELDS, f"{location}.failure")
        if failure["schema_version"] != MODEL_DRAFT_FAILURE_SCHEMA:
            fail("ANNOTATION_INVALID", f"{location}.failure schema 非法")
        _text(failure["code"], f"{location}.failure.code")
        _text(failure["message"], f"{location}.failure.message")
        if not isinstance(failure["details"], dict):
            fail("ANNOTATION_INVALID", f"{location}.failure.details 必须是对象")
        try:
            parsed_raw = parse_region_model_output(row["raw_output"])
        except VLMError:
            pass
        else:
            if parsed_raw.target_status.value == assignment["target_status"]:
                fail("ANNOTATION_INVALID", f"{location} 可解析 draft 不得伪装为 failure")
    else:
        fail("ANNOTATION_INVALID", f"{location}.parse_status 非法")
    return row


def load_model_drafts(
    project_root: Path | str,
    *,
    assignments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    root = _ordinary_root(project_root, location="project_root")
    path = root / "drafts.jsonl"
    _ordinary_file(path, location="drafts")
    if assignments is None:
        _, loaded = load_annotation_project(root)
        assignments = loaded
    by_id = {row["record_id"]: row for row in assignments}
    drafts: dict[str, dict[str, Any]] = {}
    draft_ids: set[str] = set()
    for index, value in enumerate(read_jsonl(path)):
        record_id = value.get("record_id")
        assignment = by_id.get(record_id)
        if assignment is None or record_id in drafts:
            fail("ANNOTATION_INVALID", f"drafts[{index}] record_id 未知或重复")
        validated = validate_model_draft_row(
            value,
            assignment=assignment,
            location=f"drafts[{index}]",
        )
        if validated["draft_id"] in draft_ids:
            fail("ANNOTATION_INVALID", f"drafts[{index}] draft_id 重复")
        draft_ids.add(validated["draft_id"])
        drafts[str(record_id)] = validated
    return drafts


def validate_draft_run_row(value: Any, *, location: str) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, DRAFT_RUN_FIELDS, location)
    if row["schema_version"] != MODEL_DRAFT_RUN_SCHEMA or row["formal_acceptance"] is not False:
        fail("ANNOTATION_INVALID", f"{location} schema/formal_acceptance 非法")
    for field in (
        "draft_run_id", "model_repository", "model_revision", "prompt_sha256",
        "record_ids_sha256", "partition",
    ):
        _text(row[field], f"{location}.{field}")
    for field in ("config_sha256", "config_semantic_sha256", "prompt_sha256", "record_ids_sha256"):
        _sha256(row[field], f"{location}.{field}")
    if (
        row["model_repository"] != DRAFT_MODEL_REPOSITORY
        or row["model_revision"] != DRAFT_MODEL_REVISION
    ):
        fail("ANNOTATION_INVALID", f"{location} 只允许冻结的本地 Qwen3-VL-8B")
    config = _mapping(row["config"], f"{location}.config")
    if row["config_semantic_sha256"] != sha256_text(canonical_json(config)):
        fail("ANNOTATION_INVALID", f"{location} config semantic identity 非法")
    _exact(config, ("schema_version", "model", "processor", "generation"), f"{location}.config")
    if config["schema_version"] != DRAFT_CONFIG_SCHEMA:
        fail("ANNOTATION_INVALID", f"{location} config schema 非法")
    config_model = _mapping(config["model"], f"{location}.config.model")
    _exact(
        config_model,
        (
            "path", "processor_path", "repository", "revision", "local_files_only",
            "trust_remote_code", "dtype", "attn_implementation",
        ),
        f"{location}.config.model",
    )
    if (
        config_model["repository"] != DRAFT_MODEL_REPOSITORY
        or config_model["revision"] != DRAFT_MODEL_REVISION
        or config_model["local_files_only"] is not True
        or config_model["trust_remote_code"] is not False
        or config_model["dtype"] != "bfloat16"
        or config_model["attn_implementation"] != "sdpa"
        or not isinstance(config_model["path"], str)
        or not config_model["path"]
        or not isinstance(config_model["processor_path"], str)
        or not config_model["processor_path"]
    ):
        fail("ANNOTATION_INVALID", f"{location} config model 身份非法")
    config_processor = _mapping(config["processor"], f"{location}.config.processor")
    _exact(
        config_processor,
        ("min_pixels", "max_pixels", "max_images", "max_input_tokens"),
        f"{location}.config.processor",
    )
    if (
        isinstance(config_processor["min_pixels"], bool)
        or not isinstance(config_processor["min_pixels"], int)
        or isinstance(config_processor["max_pixels"], bool)
        or not isinstance(config_processor["max_pixels"], int)
        or not 3136 <= config_processor["min_pixels"] <= config_processor["max_pixels"] <= 401408
        or config_processor["max_images"] != 3
        or config_processor["max_input_tokens"] != 4096
    ):
        fail("ANNOTATION_INVALID", f"{location} config processor limits 非法")
    config_generation = _mapping(config["generation"], f"{location}.config.generation")
    _exact(
        config_generation,
        ("max_new_tokens", "do_sample", "temperature", "top_p", "seed"),
        f"{location}.config.generation",
    )
    prompt_text = row["prompt_text"]
    if (
        not isinstance(prompt_text, str)
        or not prompt_text.strip()
        or row["prompt_sha256"] != sha256_text(prompt_text)
    ):
        fail("ANNOTATION_INVALID", f"{location} prompt identity 非法")
    if (
        not isinstance(row["model_identity"], dict)
        or not row["model_identity"]
        or not isinstance(row["processor_identity"], dict)
        or not row["processor_identity"]
    ):
        fail("ANNOTATION_INVALID", f"{location} model/processor identity 非法")
    if not isinstance(row["generation"], dict) or row["partition"] not in {"calibration", "remaining", "all"}:
        fail("ANNOTATION_INVALID", f"{location} generation/partition 非法")
    generation = _mapping(row["generation"], f"{location}.generation")
    _exact(
        generation,
        ("max_new_tokens", "do_sample", "temperature", "top_p", "seed", "single_attempt"),
        f"{location}.generation",
    )
    if (
        isinstance(generation["max_new_tokens"], bool)
        or not isinstance(generation["max_new_tokens"], int)
        or not 256 <= generation["max_new_tokens"] <= 1024
        or generation["do_sample"] is not False
        or generation["temperature"] != 0.0
        or generation["top_p"] != 1.0
        or generation["seed"] != CALIBRATION_SEED
        or generation["single_attempt"] is not True
    ):
        fail("ANNOTATION_INVALID", f"{location} generation 未满足确定性单次生成合同")
    if any(
        generation[key] != config_generation[key]
        for key in ("max_new_tokens", "do_sample", "temperature", "top_p", "seed")
    ):
        fail("ANNOTATION_INVALID", f"{location} generation 与 config 不一致")
    record_ids = row["record_ids"]
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or not all(isinstance(item, str) and item for item in record_ids)
        or len(record_ids) != len(set(record_ids))
        or row["record_ids_sha256"] != sha256_text(canonical_json(record_ids))
    ):
        fail("ANNOTATION_INVALID", f"{location} record_ids 身份非法")
    return row


def load_draft_runs(project_root: Path | str) -> tuple[dict[str, Any], ...]:
    root = _ordinary_root(project_root, location="project_root")
    run_root = root / "draft_runs"
    if not run_root.is_dir() or run_root.is_symlink():
        fail("ANNOTATION_INVALID", "draft_runs 必须是普通目录")
    rows = []
    seen: set[str] = set()
    for path in sorted(run_root.glob("*.json")):
        _ordinary_file(path, location="draft_run")
        row = validate_draft_run_row(read_json(path), location=f"draft_run[{path.name}]")
        if row["draft_run_id"] in seen or path.name != f"{row['draft_run_id']}.json":
            fail("ANNOTATION_INVALID", "draft_run 文件名或 ID 重复")
        seen.add(row["draft_run_id"])
        rows.append(row)
    return tuple(rows)


def register_draft_run(
    project_root: Path | str,
    *,
    draft_run: Mapping[str, Any],
    freeze_prompt: bool,
) -> dict[str, Any]:
    """登记一次本地生成身份；remaining 首次运行同时冻结 prompt。"""

    root = _ordinary_root(project_root, location="project_root")
    project, _ = load_annotation_project(root)
    run = validate_draft_run_row(draft_run, location="draft_run")
    run_path = root / "draft_runs" / f"{run['draft_run_id']}.json"
    if run_path.is_symlink():
        fail("OUTPUT_EXISTS", f"draft run 不得是链接：{run_path}")
    if freeze_prompt:
        prompt_sha = sha256_file(root / "prompt.txt")
        if run["prompt_sha256"] != prompt_sha:
            fail("ANNOTATION_INVALID", "remaining run prompt 与现场 prompt 不一致")
        if project["frozen_prompt_sha256"] not in {None, prompt_sha}:
            fail("ANNOTATION_INVALID", "project 已冻结为另一 prompt")
        config_sha = run["config_semantic_sha256"]
        if project["frozen_draft_config_sha256"] not in {None, config_sha}:
            fail("ANNOTATION_INVALID", "project 已冻结为另一 draft config")
        project["frozen_prompt_sha256"] = prompt_sha
        project["frozen_draft_config_sha256"] = config_sha
        atomic_write_json(root / "project.json", project)
    if run_path.exists():
        _ordinary_file(run_path, location="draft_run")
        existing = validate_draft_run_row(read_json(run_path), location="draft_run")
        if existing != run:
            fail("OUTPUT_EXISTS", f"draft run ID 已绑定不同 provenance：{run_path}")
        reused = True
    else:
        atomic_write_json(run_path, run)
        reused = False
    # 工作根保留当前 run 快照便于人工检查；完整 provenance 仍在 draft_runs/ 中。
    atomic_write_json(root / "draft_run.json", run)
    return {
        "ok": True,
        "draft_run_id": run["draft_run_id"],
        "reused": reused,
        "formal_acceptance": False,
    }


def append_model_draft(
    project_root: Path | str,
    *,
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    """逐条原子保存生成结果，避免长批次中断时丢失已完成草稿。"""

    root = _ordinary_root(project_root, location="project_root")
    _, assignments = load_annotation_project(root)
    assignment_by_id = {row["record_id"]: row for row in assignments}
    record_id = draft.get("record_id")
    assignment = assignment_by_id.get(record_id)
    existing = load_model_drafts(root, assignments=assignments)
    if assignment is None or record_id in existing:
        fail("ANNOTATION_INVALID", f"draft record 未知或已生成：{record_id}")
    row = validate_model_draft_row(
        draft,
        assignment=assignment,
        location=f"draft[{record_id}]",
    )
    run_path = root / "draft_runs" / f"{row['draft_run_id']}.json"
    _ordinary_file(run_path, location="draft_run")
    run = validate_draft_run_row(read_json(run_path), location="draft_run")
    if run["draft_run_id"] != row["draft_run_id"]:
        fail("ANNOTATION_INVALID", "draft 未绑定现场 run")
    existing[str(record_id)] = row
    ordered = [existing[item["record_id"]] for item in assignments if item["record_id"] in existing]
    atomic_write_jsonl(root / "drafts.jsonl", ordered)
    status = _write_status_after_drafts(root, assignments, existing)
    return {"ok": True, "record_id": record_id, "status": status, "formal_acceptance": False}


def write_draft_results(
    project_root: Path | str,
    *,
    draft_run: Mapping[str, Any],
    new_drafts: Iterable[Mapping[str, Any]],
    freeze_prompt: bool,
) -> dict[str, Any]:
    """测试和小批量调用的便捷入口；生产生成仍逐条落盘。"""

    root = _ordinary_root(project_root, location="project_root")
    register_draft_run(
        project_root,
        draft_run=draft_run,
        freeze_prompt=freeze_prompt,
    )
    _, assignments = load_annotation_project(root)
    assignment_by_id = {row["record_id"]: row for row in assignments}
    existing = load_model_drafts(root, assignments=assignments)
    added = 0
    run_id = draft_run["draft_run_id"]
    for value in new_drafts:
        record_id = value.get("record_id")
        assignment = assignment_by_id.get(record_id)
        if assignment is None or record_id in existing:
            fail("ANNOTATION_INVALID", f"draft record 未知或已生成：{record_id}")
        row = validate_model_draft_row(
            value,
            assignment=assignment,
            location=f"draft[{record_id}]",
        )
        if row["draft_run_id"] != run_id:
            fail("ANNOTATION_INVALID", "draft 未绑定当前 run")
        existing[str(record_id)] = row
        added += 1
    ordered = [existing[item["record_id"]] for item in assignments if item["record_id"] in existing]
    atomic_write_jsonl(root / "drafts.jsonl", ordered)
    status = _write_status_after_drafts(root, assignments, existing)
    return {"ok": True, "added": added, "status": status, "formal_acceptance": False}


def default_region_description(target_status: str) -> dict[str, Any]:
    """模型草稿失败时复用严格合同空模板；只有专家核验后才可发布。"""

    if target_status not in {"target_present", "no_target"}:
        fail("ANNOTATION_INVALID", f"target_status 非法：{target_status}")
    return region_output_template(target_status)


def build_annotation_draft_messages(
    record: Mapping[str, Any],
    *,
    asset_root: Path,
    prompt_text: str,
) -> list[dict[str, Any]]:
    """在正式 v2 message 后追加冻结标注指南，供生成和 validator 共同重算。"""

    if not isinstance(prompt_text, str) or not prompt_text.strip():
        fail("ANNOTATION_INVALID", "annotation draft prompt 不能为空")
    messages = build_mask_grounded_region_messages(record, asset_root=asset_root)
    messages[0]["content"].append({"type": "text", "text": prompt_text.strip()})
    return messages


def _validate_draft_message_identities(
    context: AnnotationAssetContext,
    drafts: Sequence[Mapping[str, Any]],
    run_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    records = {row["record_id"]: row for row in context.records}
    for draft in drafts:
        run = run_by_id[draft["draft_run_id"]]
        record = records.get(draft["record_id"])
        if record is None:
            fail("ANNOTATION_INVALID", "model draft 引用了未知 Region record")
        messages = build_annotation_draft_messages(
            record,
            asset_root=context.root,
            prompt_text=run["prompt_text"],
        )
        if draft["messages_sha256"] != sha256_text(canonical_json(messages)):
            fail("ANNOTATION_INVALID", f"model draft message identity 漂移：{draft['record_id']}")


def validate_verified_annotation_row(
    value: Any,
    *,
    assignment: Mapping[str, Any],
    draft: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, VERIFIED_FIELDS, location)
    if row["schema_version"] != VERIFIED_ANNOTATION_SCHEMA:
        fail("ANNOTATION_INVALID", f"{location} schema 非法")
    if (
        row["record_id"] != assignment["record_id"]
        or row["asset_identity_sha256"] != assignment["asset_identity_sha256"]
        or row["draft_id"] != draft["draft_id"]
    ):
        fail("ANNOTATION_INVALID", f"{location} record/asset/draft identity 不匹配")
    if row["annotator"] != SINGLE_EXPERT_ANNOTATOR:
        fail(
            "ANNOTATION_INVALID",
            f"{location}.annotator 必须严格等于 {SINGLE_EXPERT_ANNOTATOR!r}",
        )
    if row["verification_status"] != VERIFICATION_STATUS:
        fail("ANNOTATION_INVALID", f"{location} 尚未专家核验")
    parsed = parse_region_model_output(row["description"])
    if parsed.target_status.value != assignment["target_status"]:
        fail("ANNOTATION_INVALID", f"{location} target_status 与程序事实不一致")
    quality = assess_region_draft_quality(parsed)
    if quality.status is RegionDraftQualityStatus.LOW_INFORMATION:
        fail(
            "ANNOTATION_LOW_INFORMATION",
            f"{location} 专家最终答案仍是低信息描述",
            details=quality.to_dict(),
        )
    row["description"] = parsed.to_dict()
    return row


def load_verified_work(
    project_root: Path | str,
    *,
    assignments: Sequence[Mapping[str, Any]] | None = None,
    drafts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    root = _ordinary_root(project_root, location="project_root")
    if assignments is None:
        _, loaded = load_annotation_project(root)
        assignments = loaded
    if drafts is None:
        drafts = load_model_drafts(root, assignments=assignments)
    assignment_by_id = {row["record_id"]: row for row in assignments}
    verified: dict[str, dict[str, Any]] = {}
    verified_root = root / "verified"
    if not verified_root.is_dir() or verified_root.is_symlink():
        fail("ANNOTATION_INVALID", "verified 必须是普通目录")
    for path in sorted(verified_root.glob("*.json")):
        _ordinary_file(path, location="verified")
        value = _mapping(read_json(path), f"verified[{path.name}]")
        record_id = value.get("record_id")
        assignment = assignment_by_id.get(record_id)
        draft = drafts.get(str(record_id))
        if assignment is None or draft is None or record_id in verified:
            fail("ANNOTATION_INVALID", f"verified record 未知、无 draft 或重复：{record_id}")
        if path.name != _safe_record_filename(str(record_id)):
            fail("ANNOTATION_INVALID", "verified 文件名与 record_id 不一致")
        verified[str(record_id)] = validate_verified_annotation_row(
            value,
            assignment=assignment,
            draft=draft,
            location=f"verified[{record_id}]",
        )
    return verified


def save_annotation_work(
    *,
    project_root: Path | str,
    record_id: str,
    editor_text: str,
) -> dict[str, Any]:
    """保存尚未核验的编辑文本；允许 JSON 暂时不完整。"""

    root = _ordinary_root(project_root, location="project_root")
    _, assignments = load_annotation_project(root)
    if record_id not in {row["record_id"] for row in assignments}:
        fail("ANNOTATION_INVALID", f"未知 record_id：{record_id}")
    if not isinstance(editor_text, str):
        fail("ANNOTATION_INVALID", "editor_text 必须是字符串")
    row = {
        "schema_version": ANNOTATION_WORK_SCHEMA,
        "record_id": record_id,
        "annotator": SINGLE_EXPERT_ANNOTATOR,
        "editor_text": editor_text,
    }
    work_path = root / "work" / _safe_record_filename(record_id)
    if work_path.exists() or work_path.is_symlink():
        _ordinary_file(work_path, location="work")
    atomic_write_json(work_path, row)
    return {"ok": True, "record_id": record_id, "verified": False}


def verify_annotation_work(
    *,
    project_root: Path | str,
    record_id: str,
    editor_text: str,
) -> dict[str, Any]:
    """严格解析并原子保存单专家最终答案。"""

    root = _ordinary_root(project_root, location="project_root")
    project, assignments = load_annotation_project(root)
    assignment_by_id = {row["record_id"]: row for row in assignments}
    assignment = assignment_by_id.get(record_id)
    if assignment is None:
        fail("ANNOTATION_INVALID", f"未知 record_id：{record_id}")
    context = load_annotation_asset(project["asset_root"], verify_asset_identity=False)
    source_record = next(
        (row for row in context.records if row["record_id"] == record_id),
        None,
    )
    if (
        source_record is None
        or region_asset_identity(context.root, source_record["assets"])
        != assignment["asset_identity_sha256"]
    ):
        fail("ANNOTATION_INVALID", "核验前 asset identity 已漂移")
    drafts = load_model_drafts(root, assignments=assignments)
    draft = drafts.get(record_id)
    if draft is None:
        fail("ANNOTATION_INVALID", "未生成模型草稿；不得直接发布核验答案")
    parsed = parse_region_model_output(editor_text)
    row = {
        "schema_version": VERIFIED_ANNOTATION_SCHEMA,
        "record_id": record_id,
        "asset_identity_sha256": assignment["asset_identity_sha256"],
        "draft_id": draft["draft_id"],
        "annotator": SINGLE_EXPERT_ANNOTATOR,
        "verification_status": VERIFICATION_STATUS,
        "description": parsed.to_dict(),
    }
    validated = validate_verified_annotation_row(
        row,
        assignment=assignment,
        draft=draft,
        location="verified",
    )
    verified_path = root / "verified" / _safe_record_filename(record_id)
    was_verified = verified_path.exists() or verified_path.is_symlink()
    if was_verified:
        _ordinary_file(verified_path, location="verified")
    work_path = root / "work" / _safe_record_filename(record_id)
    if work_path.exists() or work_path.is_symlink():
        _ordinary_file(work_path, location="work")
    atomic_write_json(verified_path, validated)
    work_path.unlink(missing_ok=True)
    previous = _status_snapshot(root, assignments)
    calibration = assignment["partition"] == "calibration"
    status = _status_payload(
        assignments,
        drafts,
        verified_count=int(previous["verified"]) + (0 if was_verified else 1),
        calibration_verified=(
            int(previous["calibration_verified"])
            + (1 if calibration and not was_verified else 0)
        ),
    )
    atomic_write_json(root / "status.json", status)
    return {"ok": True, "record_id": record_id, "verified": True, "status": status}


def annotation_work_item(project_root: Path | str, ordinal: int) -> dict[str, Any]:
    """为本地 Gradio 返回一条只读资产和可编辑 JSON。"""

    root = _ordinary_root(project_root, location="project_root")
    project, assignments = load_annotation_project(root)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not assignments:
        fail("ANNOTATION_INVALID", "ordinal 非法")
    index = ordinal % len(assignments)
    assignment = assignments[index]
    # 导航不重复 hash 全部资产，但会重算当前记录，避免专家在漂移图像上继续标注。
    context = load_annotation_asset(
        project["asset_root"],
        verify_asset_identity=False,
    )
    records = {row["record_id"]: row for row in context.records}
    record = records[assignment["record_id"]]
    assets = _mapping(record["assets"], "record.assets")
    if region_asset_identity(context.root, assets) != assignment["asset_identity_sha256"]:
        fail("ANNOTATION_INVALID", "当前工作项 asset identity 漂移")

    def asset(role: str) -> str | None:
        relative = assets.get(role)
        if relative is None:
            return None
        path = safe_join(context.root, str(relative), location=f"assets.{role}")
        _ordinary_file(path, location=f"assets.{role}")
        return str(path)

    drafts = load_model_drafts(root, assignments=assignments)
    draft = drafts.get(assignment["record_id"])
    verified_path = root / "verified" / _safe_record_filename(assignment["record_id"])
    verified_row = None
    if verified_path.exists() or verified_path.is_symlink():
        _ordinary_file(verified_path, location="verified")
        if draft is None:
            fail("ANNOTATION_INVALID", "verified record 缺少对应 model draft")
        verified_row = validate_verified_annotation_row(
            read_json(verified_path),
            assignment=assignment,
            draft=draft,
            location=f"verified[{assignment['record_id']}]",
        )
    work_path = root / "work" / _safe_record_filename(assignment["record_id"])
    # all 视图允许专家重新打开已核验记录。若其保存了新修改，工作快照必须优先于
    # 旧 verified 答案显示；再次核验后会原子覆盖 verified 并删除该快照。
    if work_path.exists():
        _ordinary_file(work_path, location="work")
        work = _mapping(read_json(work_path), "work")
        _exact(work, WORK_FIELDS, "work")
        if (
            work["schema_version"] != ANNOTATION_WORK_SCHEMA
            or work["record_id"] != assignment["record_id"]
            or work["annotator"] != SINGLE_EXPERT_ANNOTATOR
            or not isinstance(work["editor_text"], str)
        ):
            fail("ANNOTATION_INVALID", "work snapshot identity 非法")
        editor_text = work["editor_text"]
    elif verified_row is not None:
        editor_text = json.dumps(verified_row["description"], ensure_ascii=False, indent=2)
    elif draft is not None and draft["parse_status"] == "valid":
        editor_text = json.dumps(draft["description"], ensure_ascii=False, indent=2)
    else:
        editor_text = json.dumps(
            default_region_description(assignment["target_status"]),
            ensure_ascii=False,
            indent=2,
        )
    draft_quality = (
        assess_region_draft_quality(draft["description"]).to_dict()
        if draft is not None and draft["parse_status"] == "valid"
        else None
    )
    crop_warning: dict[str, Any] | None = None
    mask_facts = record["program_facts"].get("mask")
    window = (
        mask_facts.get("crop_window_xyxy_pixel_half_open")
        if isinstance(mask_facts, Mapping)
        else None
    )
    if (
        isinstance(window, list)
        and len(window) == 4
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in window
        )
    ):
        width = window[2] - window[0]
        height = window[3] - window[1]
        if width > 0 and height > 0:
            reasons = []
            if min(width, height) < 16:
                reasons.append("min_side_lt_16")
            if max(width, height) / min(width, height) > 5.0:
                reasons.append("aspect_ratio_gt_5")
            if reasons:
                crop_warning = {
                    "width_pixels": width,
                    "height_pixels": height,
                    "reasons": reasons,
                    "message": (
                        "局部裁剪较窄或长宽比极端；必须结合完整 RGB 与 binary mask 判断，"
                        "不要把放大后的 crop 细节或边缘过度解释为目标事实。"
                    ),
                }
    status = _status_snapshot(root, assignments)
    return {
        "ordinal": index,
        "total": len(assignments),
        "record_id": assignment["record_id"],
        "source": assignment["source"],
        "split": assignment["split"],
        "target_status": assignment["target_status"],
        "partition": assignment["partition"],
        "optical_full": asset("optical_full"),
        "binary_mask": asset("binary_mask"),
        "context_crop": asset("context_crop"),
        "audit_overlay": asset("audit_overlay"),
        "program_facts": record["program_facts"],
        "model_draft": draft,
        "draft_quality": draft_quality,
        "crop_warning": crop_warning,
        "editor_text": editor_text,
        "verified": verified_row is not None,
        "status": status,
    }
