"""Stage 4 多成员 Region collection 的模型辅助监督与动态训练消息。

本模块把“模型生成”与“专家核验”明确区分。未经专家查看的合格草稿可以作为
``model_generated_unreviewed`` 监督，但绝不会被写成 ``expert_verified`` 或 Gold。
collection 中的每条记录始终解析回原成员根，以保证 full/mask/crop 和旧草稿的
message identity 不因新增轻量索引而改变。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from oa_groundrag.phase3.common import (
    atomic_write_json,
    canonical_json,
    read_json,
    read_jsonl,
    safe_join,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.errors import Phase4Error
from oa_groundrag.phase4.messages import build_mask_grounded_region_messages
from oa_groundrag.phase4.outputs import (
    REGION_OUTPUT_SCHEMA_VERSION,
    RegionDraftQualityStatus,
    assess_region_draft_quality,
    parse_region_model_output,
)

from .contracts import fail
from .region_pipeline import ledger_rows, region_asset_identity
from .single_expert import (
    DRAFT_MODEL_REPOSITORY,
    DRAFT_MODEL_REVISION,
    MODEL_DRAFT_FAILURE_SCHEMA,
    MODEL_DRAFT_RUN_SCHEMA,
    MODEL_DRAFT_SCHEMA,
    VERIFIED_ANNOTATION_SCHEMA,
    _exact,
    _mapping,
    _ordinary_file,
    _ordinary_root,
    _safe_record_filename,
    _sha256,
    _text,
    _validate_draft_message_identities,
    build_annotation_draft_messages,
    load_annotation_asset,
    load_annotation_project,
    load_draft_runs,
    load_model_drafts,
    load_verified_work,
    validate_draft_run_row,
    validate_model_draft_row,
    validate_verified_annotation_row,
)
from .single_expert_drafting import LocalQwenDraftRuntime, load_local_draft_config


EXPECTED_COLLECTION_COUNT = 8450
EXPECTED_LEGACY_DRAFT_COUNT = 20
EXPECTED_LEGACY_VERIFIED_COUNT = 5
EXPECTED_LEGACY_FILE_COUNT = 13
MODEL_ASSISTED_PROJECT_SCHEMA = "oa_groundrag.mask_grounded_region.model_assisted_project.v2"
MODEL_ASSISTED_ASSIGNMENT_SCHEMA = "oa_groundrag.mask_grounded_region.model_assisted_assignment.v2"
IMPORT_PROVENANCE_SCHEMA = "oa_groundrag.mask_grounded_region.legacy_import_provenance.v2"
SUPERVISION_RECORD_SCHEMA = "oa_groundrag.mask_grounded_region.train_supervision_record.v2"
SUPERVISION_PACKAGE_SCHEMA = "oa_groundrag.mask_grounded_region.train_supervision_package.v2"
EXCLUSION_SCHEMA = "oa_groundrag.mask_grounded_region.train_supervision_exclusion.v2"
TRAINING_MESSAGE_SCHEMA = "oa_groundrag.mask_grounded_region.training_message.v2"
TRAINING_MANIFEST_SCHEMA = "oa_groundrag.mask_grounded_region.training_messages.v2"
REFERENCE_AUTHORITY = "mixed_model_and_single_expert"
EXPERT_AUTHORITY = "expert_verified"
MODEL_AUTHORITY = "model_generated_unreviewed"

PROJECT_FIELDS = (
    "schema_version", "project_id", "collection_root", "collection_manifest_sha256",
    "split", "record_count", "ordered_record_ids_sha256", "prompt_sha256",
    "config_semantic_sha256", "legacy_import_sha256", "formal_acceptance",
)
ASSIGNMENT_FIELDS = (
    "schema_version", "ordinal", "record_id", "parent_id", "source", "split",
    "target_status", "member", "member_root", "member_manifest_sha256",
    "asset_identity_sha256",
)
SUPERVISION_FIELDS = (
    "schema_version", "record_id", "parent_id", "source", "split", "draft_id",
    "asset_identity_sha256", "supervision_authority", "quality_status",
    "description", "description_identity_sha256", "expert_annotation_identity_sha256",
)
EXCLUSION_FIELDS = (
    "schema_version", "record_id", "draft_id", "reason_code", "quality_status",
    "details",
)
PACKAGE_MANIFEST_FIELDS = (
    "schema_version", "package_id", "source_collection_root",
    "source_collection_manifest_sha256", "project_identity_sha256", "split",
    "record_count", "eligible_count", "excluded_count", "authority_counts",
    "exclusion_counts", "ordered_record_ids_sha256", "eligible_record_ids_sha256",
    "reference_authority", "expert_consensus", "supervision", "exclusions",
    "model_drafts", "draft_runs", "expert_annotations", "import_provenance",
    "ledger", "training_eligible", "gold", "thresholds_frozen",
    "formal_acceptance", "scientific_acceptance", "sealed_test_evaluated",
)
TRAINING_ROW_FIELDS = (
    "schema_version", "record_id", "parent_id", "source", "logical_role",
    "task_family", "messages", "supervision_identity_sha256",
    "asset_identity_sha256", "supervision_authority",
)
TRAINING_MANIFEST_FIELDS = (
    "schema_version", "source_collection_root", "source_collection_manifest_sha256",
    "supervision_package_root", "supervision_package_manifest_sha256", "split",
    "record_count", "ordered_record_ids_sha256", "messages",
    "assistant_target_schema", "ledger", "training_eligible", "reference_authority",
    "gold", "formal_acceptance", "scientific_acceptance",
)


@dataclass(frozen=True)
class ModelAssistedCollectionEntry:
    """一个 collection 成员记录及其真实资产根。"""

    ordinal: int
    member: str
    member_root: Path
    member_manifest_sha256: str
    record: Mapping[str, Any]
    queue: Mapping[str, Any]


@dataclass(frozen=True)
class ModelAssistedCollectionContext:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    entries: tuple[ModelAssistedCollectionEntry, ...]


@dataclass(frozen=True)
class ModelAssistedProjectContext:
    root: Path
    project: Mapping[str, Any]
    collection: ModelAssistedCollectionContext
    assignments: tuple[Mapping[str, Any], ...]
    entries_by_id: Mapping[str, ModelAssistedCollectionEntry]
    drafts: Mapping[str, Mapping[str, Any]]
    draft_runs: tuple[Mapping[str, Any], ...]
    verified: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class SupervisionDecision:
    eligible: bool
    authority: str | None
    reason_code: str | None
    quality_status: str | None
    description: Mapping[str, Any] | None
    details: Mapping[str, Any]


@dataclass(frozen=True)
class ModelAssistedSupervisionPackage:
    root: Path
    manifest: Mapping[str, Any]
    supervision: tuple[Mapping[str, Any], ...]
    exclusions: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ModelAssistedTrainingArtifact:
    root: Path
    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]


def _load_collection_context(
    value: Path | str,
    *,
    verify_members: bool = True,
) -> ModelAssistedCollectionContext:
    """经 expanded-region validator 装载 collection，不猜成员路径或资产语义。"""

    from .expanded_region import load_expanded_collection_context

    expanded = load_expanded_collection_context(value, verify_members=verify_members)
    entries: list[ModelAssistedCollectionEntry] = []
    seen: set[str] = set()
    for expected_ordinal, item in enumerate(expanded.entries):
        index = dict(item.index)
        record = dict(item.record)
        queue = dict(item.queue)
        record_id = _text(record.get("record_id"), f"collection[{expected_ordinal}].record_id")
        if (
            index.get("ordinal") != expected_ordinal
            or index.get("record_id") != record_id
            or record_id in seen
            or record.get("split") != "train"
            or queue.get("record_id") != record_id
            or queue.get("split") != "train"
        ):
            fail("SPLIT_FORBIDDEN", "expanded collection 必须是唯一、有序的 train-only records")
        if record.get("target_status") not in {"target_present", "no_target"}:
            fail("ANNOTATION_INVALID", f"collection target_status 非法：{record_id}")
        member_root = Path(os.path.abspath(Path(item.member_root)))
        actual_asset = region_asset_identity(member_root, _mapping(record.get("assets"), "record.assets"))
        if (
            index.get("asset_identity_sha256") != actual_asset
            or queue.get("asset_identity_sha256") != actual_asset
        ):
            fail("ANNOTATION_INVALID", f"collection asset identity 漂移：{record_id}")
        seen.add(record_id)
        entries.append(ModelAssistedCollectionEntry(
            ordinal=expected_ordinal,
            member=_text(index.get("member"), f"collection[{expected_ordinal}].member"),
            member_root=member_root,
            member_manifest_sha256=_sha256(
                item.member_manifest_sha256,
                f"collection[{expected_ordinal}].member_manifest_sha256",
            ),
            record=record,
            queue=queue,
        ))
    if len(entries) != EXPECTED_COLLECTION_COUNT:
        fail("ANNOTATION_INVALID", f"model-assisted collection 必须精确为 {EXPECTED_COLLECTION_COUNT} 条")
    source_counts: dict[str, int] = {}
    for entry in entries:
        source = _text(entry.record.get("source"), "record.source")
        source_counts[source] = source_counts.get(source, 0) + 1
    if len(source_counts) != 5 or set(source_counts.values()) != {1690}:
        fail("ANNOTATION_INVALID", "model-assisted collection 必须五来源各 1690 条")
    return ModelAssistedCollectionContext(
        root=Path(expanded.root),
        manifest=dict(expanded.manifest),
        manifest_sha256=str(expanded.manifest_sha256),
        entries=tuple(entries),
    )


def _assignment(entry: ModelAssistedCollectionEntry) -> dict[str, Any]:
    record = entry.record
    return {
        "schema_version": MODEL_ASSISTED_ASSIGNMENT_SCHEMA,
        "ordinal": entry.ordinal,
        "record_id": record["record_id"],
        "parent_id": record["parent_id"],
        "source": record["source"],
        "split": "train",
        "target_status": record["target_status"],
        "member": entry.member,
        "member_root": str(entry.member_root),
        "member_manifest_sha256": entry.member_manifest_sha256,
        "asset_identity_sha256": entry.queue["asset_identity_sha256"],
    }


def _validate_assignment(
    value: Any,
    *,
    entry: ModelAssistedCollectionEntry,
    location: str,
) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, ASSIGNMENT_FIELDS, location)
    if row != _assignment(entry):
        fail("ANNOTATION_INVALID", f"{location} 与 collection member identity 不一致")
    return row


def _directory_file_contract(root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        _ordinary_file(path, location="legacy provenance file")
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _legacy_provenance(
    legacy_root: Path,
    *,
    draft_count: int,
    verified_count: int,
) -> dict[str, Any]:
    # 旧工作根仍可能继续被人工界面修改，因此导入时冻结全部普通文件，而不仅是
    # drafts/verified。当前现场包括 workflow_state 与 draft_run 快照，共 13 个文件。
    files = sorted(path for path in legacy_root.rglob("*") if path.is_file() or path.is_symlink())
    rows = _directory_file_contract(legacy_root, files)
    return {
        "schema_version": IMPORT_PROVENANCE_SCHEMA,
        "source_project_root": str(legacy_root),
        "files": rows,
        "files_root_sha256": sha256_text(canonical_json(rows)),
        "imported_draft_count": draft_count,
        "imported_verified_count": verified_count,
        "formal_acceptance": False,
    }


def _validate_legacy_provenance(value: Any) -> dict[str, Any]:
    row = _mapping(value, "legacy import provenance")
    _exact(
        row,
        (
            "schema_version", "source_project_root", "files", "files_root_sha256",
            "imported_draft_count", "imported_verified_count", "formal_acceptance",
        ),
        "legacy import provenance",
    )
    if row["schema_version"] != IMPORT_PROVENANCE_SCHEMA or row["formal_acceptance"] is not False:
        fail("ANNOTATION_INVALID", "legacy import provenance schema 非法")
    source = _ordinary_root(row["source_project_root"], location="legacy source project")
    files = row["files"]
    if not isinstance(files, list) or not files:
        fail("ANNOTATION_INVALID", "legacy provenance files 不能为空")
    normalized = []
    declared_paths: set[str] = set()
    for index, value in enumerate(files):
        contract = _mapping(value, f"legacy.files[{index}]")
        _exact(contract, ("path", "size_bytes", "sha256"), f"legacy.files[{index}]")
        relative = _text(contract["path"], "legacy.path")
        if relative in declared_paths:
            fail("ANNOTATION_INVALID", "legacy provenance path 重复")
        declared_paths.add(relative)
        path = safe_join(source, relative, location="legacy.path")
        _ordinary_file(path, location="legacy provenance file")
        if contract["size_bytes"] != path.stat().st_size or contract["sha256"] != sha256_file(path):
            fail("ANNOTATION_INVALID", f"legacy source 在导入后漂移：{path}")
        normalized.append(contract)
    actual_paths = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != declared_paths:
        fail(
            "ANNOTATION_INVALID",
            "legacy source 文件集合在导入后漂移",
            details={
                "missing": sorted(declared_paths - actual_paths),
                "unknown": sorted(actual_paths - declared_paths),
            },
        )
    if row["files_root_sha256"] != sha256_text(canonical_json(normalized)):
        fail("ANNOTATION_INVALID", "legacy provenance root identity 漂移")
    for key in ("imported_draft_count", "imported_verified_count"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
            fail("ANNOTATION_INVALID", f"legacy.{key} 非法")
    return row


def _legacy_import_rows(
    provenance: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """从已冻结的 13 文件现场重载导入行，防止新工作根替换旧 draft/verified。"""

    source = Path(str(provenance["source_project_root"]))
    _, assignments = load_annotation_project(source)
    drafts = load_model_drafts(source, assignments=assignments)
    verified = load_verified_work(source, assignments=assignments, drafts=drafts)
    if (
        len(drafts) != provenance["imported_draft_count"]
        or len(verified) != provenance["imported_verified_count"]
    ):
        fail("ANNOTATION_INVALID", "legacy provenance count 与冻结来源内容不一致")
    return drafts, verified


def create_model_assisted_project(
    *,
    collection_root: Path | str,
    legacy_project_root: Path | str,
    config_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """创建 8450 条动态工作根，并只读导入旧 500 项目已有的草稿与核验。"""

    collection = _load_collection_context(collection_root)
    legacy_root = _ordinary_root(legacy_project_root, location="legacy_project_root")
    legacy_project, legacy_assignments = load_annotation_project(legacy_root)
    if legacy_project["split"] != "train" or len(legacy_assignments) != 500:
        fail("SPLIT_FORBIDDEN", "只允许导入冻结的 500 条 train 单专家项目")
    legacy_drafts = load_model_drafts(legacy_root, assignments=legacy_assignments)
    legacy_runs = load_draft_runs(legacy_root)
    legacy_verified = load_verified_work(
        legacy_root,
        assignments=legacy_assignments,
        drafts=legacy_drafts,
    )
    if (
        len(legacy_drafts) != EXPECTED_LEGACY_DRAFT_COUNT
        or len(legacy_verified) != EXPECTED_LEGACY_VERIFIED_COUNT
    ):
        fail(
            "ANNOTATION_INVALID",
            "legacy v1 import 必须绑定当前冻结的 20 drafts / 5 verified 现场",
        )
    legacy_asset = load_annotation_asset(legacy_project["asset_root"])
    _validate_draft_message_identities(
        legacy_asset,
        list(legacy_drafts.values()),
        {row["draft_run_id"]: row for row in legacy_runs},
    )
    config = load_local_draft_config(config_path)
    prompt_path = legacy_root / "prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_file(prompt_path)
    if not legacy_runs or any(
        run["prompt_sha256"] != prompt_sha
        or run["prompt_text"] != prompt_text
        or run["config_semantic_sha256"] != config.semantic_sha256
        for run in legacy_runs
    ):
        fail("ANNOTATION_INVALID", "扩展工作流必须冻结当前旧项目已认可的 prompt/config")
    entries_by_id = {str(item.record["record_id"]): item for item in collection.entries}
    legacy_assignment_by_id = {str(row["record_id"]): row for row in legacy_assignments}
    for record_id, draft in legacy_drafts.items():
        entry = entries_by_id.get(record_id)
        old = legacy_assignment_by_id[record_id]
        if (
            entry is None
            or entry.member_root != legacy_asset.root
            or old["source"] != entry.record["source"]
            or old["target_status"] != entry.record["target_status"]
            or old["asset_identity_sha256"] != entry.queue["asset_identity_sha256"]
            or draft["asset_identity_sha256"] != entry.queue["asset_identity_sha256"]
        ):
            fail("ANNOTATION_INVALID", f"legacy draft 未无损映射到 collection base：{record_id}")
    assignments = [_assignment(item) for item in collection.entries]
    ordered_ids = [row["record_id"] for row in assignments]
    provenance = _legacy_provenance(
        legacy_root,
        draft_count=len(legacy_drafts),
        verified_count=len(legacy_verified),
    )
    if len(provenance["files"]) != EXPECTED_LEGACY_FILE_COUNT:
        fail("ANNOTATION_INVALID", "legacy v1 import 必须冻结当前现场全部 13 个文件")
    project_id = "model_assisted_project_" + sha256_text(canonical_json({
        "collection_manifest_sha256": collection.manifest_sha256,
        "prompt_sha256": prompt_sha,
        "config_semantic_sha256": config.semantic_sha256,
        "ordered_record_ids": ordered_ids,
    }))[:24]
    project = {
        "schema_version": MODEL_ASSISTED_PROJECT_SCHEMA,
        "project_id": project_id,
        "collection_root": str(collection.root),
        "collection_manifest_sha256": collection.manifest_sha256,
        "split": "train",
        "record_count": len(assignments),
        "ordered_record_ids_sha256": sha256_text(canonical_json(ordered_ids)),
        "prompt_sha256": prompt_sha,
        "config_semantic_sha256": config.semantic_sha256,
        "legacy_import_sha256": sha256_text(canonical_json(provenance)),
        "formal_acceptance": False,
    }
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_json("project.json", project)
        writer.write_jsonl("assignments.jsonl", assignments)
        writer.write_json("draft_config.json", config.semantic_dict())
        writer.write_json("import_provenance.json", provenance)
        writer.path("prompt.txt").write_text(prompt_text, encoding="utf-8")
        for name in ("drafts", "draft_runs", "verified"):
            writer.path(name).mkdir(parents=True, exist_ok=False)
        for draft in legacy_drafts.values():
            writer.write_json(f"drafts/{_safe_record_filename(str(draft['record_id']))}", draft)
        for run in legacy_runs:
            writer.write_json(f"draft_runs/{run['draft_run_id']}.json", run)
        for verified in legacy_verified.values():
            writer.write_json(f"verified/{_safe_record_filename(str(verified['record_id']))}", verified)
        root = writer.publish()
    status = model_assisted_project_status(root)
    return {
        "ok": True,
        "root": str(root),
        "project_id": project_id,
        "record_count": len(assignments),
        "imported_drafts": len(legacy_drafts),
        "imported_verified": len(legacy_verified),
        "pending": status["pending"],
        "formal_acceptance": False,
    }


def _load_project_drafts(
    root: Path,
    assignments: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    assignment_by_id = {str(row["record_id"]): row for row in assignments}
    drafts: dict[str, dict[str, Any]] = {}
    draft_ids: set[str] = set()
    draft_root = root / "drafts"
    if not draft_root.is_dir() or draft_root.is_symlink():
        fail("ANNOTATION_INVALID", "model-assisted drafts 必须是普通目录")
    for path in sorted(draft_root.glob("*.json")):
        _ordinary_file(path, location="model-assisted draft")
        raw = _mapping(read_json(path), f"draft[{path.name}]")
        record_id = str(raw.get("record_id", ""))
        assignment = assignment_by_id.get(record_id)
        if (
            assignment is None
            or record_id in drafts
            or path.name != _safe_record_filename(record_id)
        ):
            fail("ANNOTATION_INVALID", f"draft 文件名、record 或唯一性非法：{path}")
        row = validate_model_draft_row(raw, assignment=assignment, location=f"draft[{record_id}]")
        if row["draft_id"] in draft_ids:
            fail("ANNOTATION_INVALID", "model-assisted draft_id 重复")
        draft_ids.add(str(row["draft_id"]))
        drafts[record_id] = row
    return drafts


def _load_project_runs(root: Path) -> tuple[dict[str, Any], ...]:
    run_root = root / "draft_runs"
    if not run_root.is_dir() or run_root.is_symlink():
        fail("ANNOTATION_INVALID", "model-assisted draft_runs 必须是普通目录")
    rows = []
    seen: set[str] = set()
    for path in sorted(run_root.glob("*.json")):
        _ordinary_file(path, location="model-assisted draft run")
        row = validate_draft_run_row(read_json(path), location=f"draft_run[{path.name}]")
        run_id = str(row["draft_run_id"])
        if run_id in seen or path.name != f"{run_id}.json":
            fail("ANNOTATION_INVALID", "draft run ID 或文件名重复")
        seen.add(run_id)
        rows.append(row)
    return tuple(rows)


def _load_project_verified(
    root: Path,
    assignments: Sequence[Mapping[str, Any]],
    drafts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    verified_root = root / "verified"
    if not verified_root.is_dir() or verified_root.is_symlink():
        fail("ANNOTATION_INVALID", "model-assisted verified 必须是普通目录")
    assignment_by_id = {str(row["record_id"]): row for row in assignments}
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(verified_root.glob("*.json")):
        _ordinary_file(path, location="model-assisted verified")
        raw = _mapping(read_json(path), f"verified[{path.name}]")
        record_id = str(raw.get("record_id", ""))
        assignment = assignment_by_id.get(record_id)
        draft = drafts.get(record_id)
        if (
            assignment is None
            or draft is None
            or record_id in rows
            or path.name != _safe_record_filename(record_id)
        ):
            fail("ANNOTATION_INVALID", f"verified 文件名、record 或 draft 绑定非法：{path}")
        rows[record_id] = validate_verified_annotation_row(
            raw,
            assignment=assignment,
            draft=draft,
            location=f"verified[{record_id}]",
        )
    return rows


def _validate_project_message_identities(
    context: ModelAssistedProjectContext,
) -> None:
    run_by_id = {str(row["draft_run_id"]): row for row in context.draft_runs}
    if len(run_by_id) != len(context.draft_runs):
        fail("ANNOTATION_INVALID", "model-assisted draft run ID 重复")
    for draft in context.drafts.values():
        run = run_by_id.get(str(draft["draft_run_id"]))
        entry = context.entries_by_id[str(draft["record_id"])]
        if run is None or draft["record_id"] not in run["record_ids"]:
            fail("ANNOTATION_INVALID", "model-assisted draft 缺少 run provenance")
        messages = build_annotation_draft_messages(
            entry.record,
            asset_root=entry.member_root,
            prompt_text=run["prompt_text"],
        )
        if draft["messages_sha256"] != sha256_text(canonical_json(messages)):
            fail("ANNOTATION_INVALID", f"draft message identity 漂移：{draft['record_id']}")


def load_model_assisted_project(
    project_root: Path | str,
) -> ModelAssistedProjectContext:
    """严格重载 mutable 工作根；成员资产、prompt、config 和旧来源均重新绑定。"""

    root = _ordinary_root(project_root, location="model_assisted_project_root")
    for name in (
        "project.json", "assignments.jsonl", "prompt.txt", "draft_config.json",
        "import_provenance.json",
    ):
        _ordinary_file(root / name, location=f"project.{name}")
    project = _mapping(read_json(root / "project.json"), "model-assisted project")
    _exact(project, PROJECT_FIELDS, "model-assisted project")
    if (
        project["schema_version"] != MODEL_ASSISTED_PROJECT_SCHEMA
        or project["split"] != "train"
        or project["record_count"] != EXPECTED_COLLECTION_COUNT
        or project["formal_acceptance"] is not False
    ):
        fail("SPLIT_FORBIDDEN", "model-assisted project 必须为 8450 条 train-only v2")
    collection = _load_collection_context(project["collection_root"])
    if (
        project["collection_root"] != str(collection.root)
        or project["collection_manifest_sha256"] != collection.manifest_sha256
    ):
        fail("ANNOTATION_INVALID", "project collection identity 漂移")
    prompt = (root / "prompt.txt").read_text(encoding="utf-8")
    if not prompt.strip() or project["prompt_sha256"] != sha256_file(root / "prompt.txt"):
        fail("ANNOTATION_INVALID", "project prompt identity 漂移")
    config = _mapping(read_json(root / "draft_config.json"), "project draft config")
    if project["config_semantic_sha256"] != sha256_text(canonical_json(config)):
        fail("ANNOTATION_INVALID", "project config semantic identity 漂移")
    provenance = _validate_legacy_provenance(read_json(root / "import_provenance.json"))
    if project["legacy_import_sha256"] != sha256_text(canonical_json(provenance)):
        fail("ANNOTATION_INVALID", "project legacy import identity 漂移")
    values = read_jsonl(root / "assignments.jsonl")
    if len(values) != len(collection.entries):
        fail("ANNOTATION_INVALID", "project assignments 数量与 collection 不一致")
    assignments = tuple(
        _validate_assignment(value, entry=entry, location=f"assignments[{index}]")
        for index, (value, entry) in enumerate(zip(values, collection.entries, strict=True))
    )
    ordered_ids = [row["record_id"] for row in assignments]
    if project["ordered_record_ids_sha256"] != sha256_text(canonical_json(ordered_ids)):
        fail("ANNOTATION_INVALID", "project ordered assignment identity 漂移")
    drafts = _load_project_drafts(root, assignments)
    runs = _load_project_runs(root)
    verified = _load_project_verified(root, assignments, drafts)
    imported_drafts, imported_verified = _legacy_import_rows(provenance)
    if any(drafts.get(record_id) != row for record_id, row in imported_drafts.items()) or any(
        verified.get(record_id) != row for record_id, row in imported_verified.items()
    ):
        fail("ANNOTATION_INVALID", "model-assisted work 未无损保留 legacy draft/verified")
    context = ModelAssistedProjectContext(
        root=root,
        project=project,
        collection=collection,
        assignments=assignments,
        entries_by_id={str(entry.record["record_id"]): entry for entry in collection.entries},
        drafts=drafts,
        draft_runs=runs,
        verified=verified,
    )
    _validate_project_message_identities(context)
    if len(imported_drafts) > len(drafts) or len(imported_verified) > len(verified):
        fail("ANNOTATION_INVALID", "legacy import count 超出当前工作内容")
    return context


def model_assisted_project_status(project_root: Path | str) -> dict[str, Any]:
    """返回可重算状态；不信任独立、可漂移的计数快照。"""

    context = load_model_assisted_project(project_root)
    valid = sum(row["parse_status"] == "valid" for row in context.drafts.values())
    return {
        "schema_version": "oa_groundrag.mask_grounded_region.model_assisted_status.v2",
        "total": len(context.assignments),
        "drafted": len(context.drafts),
        "valid_drafts": valid,
        "invalid_drafts": len(context.drafts) - valid,
        "verified": len(context.verified),
        "pending": len(context.assignments) - len(context.drafts),
        "complete": len(context.drafts) == len(context.assignments),
        "formal_acceptance": False,
    }


def generate_model_assisted_drafts(
    *,
    project_root: Path | str,
    config_path: Path | str,
    limit: int | None = None,
    runtime: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """单次加载本地模型、batch=1 逐条生成，并以独立原子文件支持断点续跑。"""

    context = load_model_assisted_project(project_root)
    config = load_local_draft_config(config_path)
    if config.semantic_sha256 != context.project["config_semantic_sha256"]:
        fail("ANNOTATION_INVALID", "model-assisted generation config 与冻结身份不一致")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        fail("ANNOTATION_INVALID", "limit 必须是正整数或 null")
    candidates = [
        row for row in context.assignments if row["record_id"] not in context.drafts
    ]
    if limit is not None:
        candidates = candidates[:limit]

    def progress(event: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({"event": event, **details})

    if not candidates:
        progress("collection_already_drafted", generated=0)
        return {
            "ok": True,
            "generated": 0,
            "invalid": 0,
            "remaining": 0,
            "quality_counts": {},
            "formal_acceptance": False,
        }
    progress("model_loading", pending=len(candidates))
    active_runtime = runtime or LocalQwenDraftRuntime(config)
    if (
        not isinstance(getattr(active_runtime, "model_identity", None), Mapping)
        or not isinstance(getattr(active_runtime, "processor_identity", None), Mapping)
        or not callable(getattr(active_runtime, "generate", None))
    ):
        fail("ANNOTATION_INVALID", "local model-assisted runtime identity/生成接口非法")
    progress("model_ready", pending=len(candidates))
    prompt_text = (context.root / "prompt.txt").read_text(encoding="utf-8")
    record_ids = [str(row["record_id"]) for row in candidates]
    run_id = "draft_run_" + sha256_text(canonical_json({
        "collection_manifest_sha256": context.collection.manifest_sha256,
        "config_semantic_sha256": config.semantic_sha256,
        "prompt_sha256": context.project["prompt_sha256"],
        "record_ids": record_ids,
    }))[:24]
    generation = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": config.seed,
        "single_attempt": True,
    }
    run = {
        "schema_version": MODEL_DRAFT_RUN_SCHEMA,
        "draft_run_id": run_id,
        "config_sha256": config.config_file_sha256,
        "config_semantic_sha256": config.semantic_sha256,
        "config": config.semantic_dict(),
        "model_repository": DRAFT_MODEL_REPOSITORY,
        "model_revision": DRAFT_MODEL_REVISION,
        "model_identity": dict(active_runtime.model_identity),
        "processor_identity": dict(active_runtime.processor_identity),
        "prompt_text": prompt_text,
        "prompt_sha256": context.project["prompt_sha256"],
        "generation": generation,
        "partition": "all",
        "record_ids": record_ids,
        "record_ids_sha256": sha256_text(canonical_json(record_ids)),
        "formal_acceptance": False,
    }
    validate_draft_run_row(run, location="model_assisted.draft_run")
    run_path = context.root / "draft_runs" / f"{run_id}.json"
    if run_path.exists() or run_path.is_symlink():
        _ordinary_file(run_path, location="model-assisted draft run")
        if read_json(run_path) != run:
            fail("OUTPUT_EXISTS", "draft run ID 已绑定不同 provenance")
    else:
        atomic_write_json(run_path, run)
    generated = 0
    invalid = 0
    quality_counts: dict[str, int] = {}
    for current, assignment in enumerate(candidates, start=1):
        record_id = str(assignment["record_id"])
        entry = context.entries_by_id[record_id]
        messages = build_annotation_draft_messages(
            entry.record,
            asset_root=entry.member_root,
            prompt_text=prompt_text,
        )
        raw = active_runtime.generate(messages)
        parse_status = "valid"
        description: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None
        quality_status: str | None = None
        try:
            parsed = parse_region_model_output(raw)
            if parsed.target_status.value != assignment["target_status"]:
                fail("ANNOTATION_INVALID", "模型 draft target_status 与程序事实不一致")
            description = parsed.to_dict()
            quality_status = assess_region_draft_quality(parsed).status.value
            quality_counts[quality_status] = quality_counts.get(quality_status, 0) + 1
        except Phase4Error as error:
            parse_status = "invalid"
            invalid += 1
            code = getattr(error.code, "value", str(error.code))
            failure = {
                "schema_version": MODEL_DRAFT_FAILURE_SCHEMA,
                "code": code,
                "message": str(error),
                "details": dict(error.details),
            }
        except Exception as error:
            parse_status = "invalid"
            invalid += 1
            failure = {
                "schema_version": MODEL_DRAFT_FAILURE_SCHEMA,
                "code": "ANNOTATION_INVALID",
                "message": str(error),
                "details": {},
            }
        draft_id = "draft_" + sha256_text(canonical_json({
            "draft_run_id": run_id,
            "record_id": record_id,
            "asset_identity_sha256": assignment["asset_identity_sha256"],
        }))[:24]
        draft = {
            "schema_version": MODEL_DRAFT_SCHEMA,
            "draft_id": draft_id,
            "draft_run_id": run_id,
            "record_id": record_id,
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "messages_sha256": sha256_text(canonical_json(messages)),
            "raw_output": raw,
            "parse_status": parse_status,
            "description": description,
            "failure": failure,
        }
        validate_model_draft_row(draft, assignment=assignment, location=f"draft[{record_id}]")
        path = context.root / "drafts" / _safe_record_filename(record_id)
        if path.exists() or path.is_symlink():
            fail("OUTPUT_EXISTS", f"并发生成发现已存在 draft：{record_id}")
        atomic_write_json(path, draft)
        generated += 1
        progress(
            "draft_persisted",
            record_id=record_id,
            current=current,
            total=len(candidates),
            parse_status=parse_status,
            quality_status=quality_status,
        )
    # 当前命令独占写入且每条落盘前拒绝覆盖，剩余数可由已验证起始快照确定性计算；
    # 不在 8430 条生成结束后立即再次遍历和哈希全部 collection 资产。
    remaining = len(context.assignments) - len(context.drafts) - generated
    result = {
        "ok": True,
        "draft_run_id": run_id,
        "generated": generated,
        "invalid": invalid,
        "remaining": remaining,
        "quality_counts": dict(sorted(quality_counts.items())),
        "formal_acceptance": False,
    }
    progress("generation_complete", **result)
    return result


def assess_supervision_eligibility(
    *,
    draft: Mapping[str, Any],
    assignment: Mapping[str, Any],
    verified: Mapping[str, Any] | None = None,
) -> SupervisionDecision:
    """按冻结规则选择专家答案或未复核模型草稿，不进行任何宽松修复。"""

    validated_draft = validate_model_draft_row(
        draft,
        assignment=assignment,
        location=f"draft[{assignment['record_id']}]",
    )
    if verified is not None:
        annotation = validate_verified_annotation_row(
            verified,
            assignment=assignment,
            draft=validated_draft,
            location=f"verified[{assignment['record_id']}]",
        )
        parsed = parse_region_model_output(annotation["description"])
        quality = assess_region_draft_quality(parsed)
        return SupervisionDecision(
            eligible=True,
            authority=EXPERT_AUTHORITY,
            reason_code=None,
            quality_status=quality.status.value,
            description=parsed.to_dict(),
            details={},
        )
    if validated_draft["parse_status"] != "valid":
        failure = _mapping(validated_draft["failure"], "draft.failure")
        return SupervisionDecision(
            eligible=False,
            authority=None,
            reason_code="parse_invalid",
            quality_status=None,
            description=None,
            details={"failure_code": failure["code"]},
        )
    parsed = parse_region_model_output(validated_draft["description"])
    if parsed.target_status.value != assignment["target_status"]:
        fail("ANNOTATION_INVALID", "draft target_status 与 assignment 不一致")
    quality = assess_region_draft_quality(parsed)
    if quality.status in {
        RegionDraftQualityStatus.INFORMATIVE,
        RegionDraftQualityStatus.LIMITED_BUT_SPECIFIC,
        RegionDraftQualityStatus.NOT_APPLICABLE_NO_TARGET,
    }:
        return SupervisionDecision(
            eligible=True,
            authority=MODEL_AUTHORITY,
            reason_code=None,
            quality_status=quality.status.value,
            description=parsed.to_dict(),
            details=quality.to_dict(),
        )
    metrics = quality.metrics
    if metrics.get("template_match") is True:
        reason = "template_copy"
    elif metrics.get("summary_is_default") is True:
        reason = "default_summary_low_information"
    else:
        concrete = sum(int(metrics.get(key, 0)) for key in (
            "target_observation_count", "environment_item_count",
            "contrast_observation_count", "specific_limitation_count",
        ))
        reason = None if concrete > 0 else "generic_low_information"
    if reason is not None:
        return SupervisionDecision(
            eligible=False,
            authority=None,
            reason_code=reason,
            quality_status=quality.status.value,
            description=None,
            details=quality.to_dict(),
        )
    return SupervisionDecision(
        eligible=True,
        authority=MODEL_AUTHORITY,
        reason_code=None,
        quality_status=quality.status.value,
        description=parsed.to_dict(),
        details=quality.to_dict(),
    )


def _file_contract(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_file(path)}


def _ledger_contract(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "path": path.name,
        "entry_count": len(rows),
        "file_sha256": sha256_file(path),
        "root_sha256": sha256_text(canonical_json(rows)),
    }


def export_model_assisted_supervision(
    *,
    project_root: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """将完整 8450 条生成现场发布为 expert/model 混合监督与显式排除清单。"""

    context = load_model_assisted_project(project_root)
    if len(context.drafts) != len(context.assignments):
        fail(
            "ANNOTATION_INVALID",
            "必须先为 collection 每条记录保存一次 draft，才能发布监督 package",
            details={"records": len(context.assignments), "drafts": len(context.drafts)},
        )
    supervision: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    authority_counts = {EXPERT_AUTHORITY: 0, MODEL_AUTHORITY: 0}
    exclusion_counts: dict[str, int] = {}
    for assignment in context.assignments:
        record_id = str(assignment["record_id"])
        draft = context.drafts[record_id]
        verified = context.verified.get(record_id)
        decision = assess_supervision_eligibility(
            draft=draft,
            assignment=assignment,
            verified=verified,
        )
        if decision.eligible:
            assert decision.description is not None
            assert decision.authority is not None
            description = dict(decision.description)
            authority_counts[decision.authority] += 1
            supervision.append({
                "schema_version": SUPERVISION_RECORD_SCHEMA,
                "record_id": record_id,
                "parent_id": assignment["parent_id"],
                "source": assignment["source"],
                "split": "train",
                "draft_id": draft["draft_id"],
                "asset_identity_sha256": assignment["asset_identity_sha256"],
                "supervision_authority": decision.authority,
                "quality_status": decision.quality_status,
                "description": description,
                "description_identity_sha256": sha256_text(canonical_json(description)),
                "expert_annotation_identity_sha256": (
                    sha256_text(canonical_json(verified)) if verified is not None else None
                ),
            })
        else:
            assert decision.reason_code is not None
            exclusion_counts[decision.reason_code] = exclusion_counts.get(decision.reason_code, 0) + 1
            exclusions.append({
                "schema_version": EXCLUSION_SCHEMA,
                "record_id": record_id,
                "draft_id": draft["draft_id"],
                "reason_code": decision.reason_code,
                "quality_status": decision.quality_status,
                "details": dict(decision.details),
            })
    ordered_ids = [str(row["record_id"]) for row in context.assignments]
    eligible_ids = [str(row["record_id"]) for row in supervision]
    draft_rows = [context.drafts[record_id] for record_id in ordered_ids]
    verified_rows = [
        context.verified[record_id] for record_id in ordered_ids if record_id in context.verified
    ]
    provenance = _validate_legacy_provenance(
        read_json(context.root / "import_provenance.json")
    )
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("supervision.jsonl", supervision)
        writer.write_jsonl("exclusions.jsonl", exclusions)
        writer.write_jsonl("model_drafts.jsonl", draft_rows)
        writer.write_jsonl("draft_runs.jsonl", context.draft_runs)
        writer.write_jsonl("expert_annotations.jsonl", verified_rows)
        writer.write_json("import_provenance.json", provenance)
        payloads = (
            "supervision.jsonl", "exclusions.jsonl", "model_drafts.jsonl",
            "draft_runs.jsonl", "expert_annotations.jsonl", "import_provenance.json",
        )
        ledger = ledger_rows(writer.staging, payloads)
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        contracts = {name: _file_contract(writer.path(name)) for name in payloads}
        contract_by_role = {
            "supervision": contracts["supervision.jsonl"],
            "exclusions": contracts["exclusions.jsonl"],
            "model_drafts": contracts["model_drafts.jsonl"],
            "draft_runs": contracts["draft_runs.jsonl"],
            "expert_annotations": contracts["expert_annotations.jsonl"],
            "import_provenance": contracts["import_provenance.json"],
        }
        package_id = sha256_text(canonical_json({
            "source_collection_manifest_sha256": context.collection.manifest_sha256,
            "project_identity_sha256": sha256_file(context.root / "project.json"),
            "payloads": {
                key: contract["sha256"] for key, contract in contract_by_role.items()
            },
        }))
        manifest = {
            "schema_version": SUPERVISION_PACKAGE_SCHEMA,
            "package_id": package_id,
            "source_collection_root": str(context.collection.root),
            "source_collection_manifest_sha256": context.collection.manifest_sha256,
            "project_identity_sha256": sha256_file(context.root / "project.json"),
            "split": "train",
            "record_count": len(context.assignments),
            "eligible_count": len(supervision),
            "excluded_count": len(exclusions),
            "authority_counts": dict(sorted(authority_counts.items())),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "ordered_record_ids_sha256": sha256_text(canonical_json(ordered_ids)),
            "eligible_record_ids_sha256": sha256_text(canonical_json(eligible_ids)),
            "reference_authority": REFERENCE_AUTHORITY,
            "expert_consensus": False,
            "supervision": contract_by_role["supervision"],
            "exclusions": contract_by_role["exclusions"],
            "model_drafts": contract_by_role["model_drafts"],
            "draft_runs": contract_by_role["draft_runs"],
            "expert_annotations": contract_by_role["expert_annotations"],
            "import_provenance": contract_by_role["import_provenance"],
            "ledger": _ledger_contract(writer.path("SHA256SUMS.jsonl"), ledger),
            "training_eligible": bool(supervision),
            "gold": False,
            "thresholds_frozen": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    validated = validate_model_assisted_supervision(
        collection_root=context.collection.root,
        package_root=root,
    )
    return {
        "ok": True,
        "root": str(root),
        "package_id": validated.manifest["package_id"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "eligible_count": len(validated.supervision),
        "excluded_count": len(validated.exclusions),
        "authority_counts": validated.manifest["authority_counts"],
        "exclusion_counts": validated.manifest["exclusion_counts"],
        "formal_acceptance": False,
    }


def _validate_ledger(root: Path, manifest: Mapping[str, Any], expected: set[str]) -> None:
    contract = _mapping(manifest["ledger"], "manifest.ledger")
    _exact(contract, ("path", "entry_count", "file_sha256", "root_sha256"), "manifest.ledger")
    path = safe_join(root, _text(contract["path"], "ledger.path"), location="ledger.path")
    _ordinary_file(path, location="package ledger")
    rows = read_jsonl(path)
    if (
        {row.get("path") for row in rows} != expected
        or contract["entry_count"] != len(rows)
        or contract["file_sha256"] != sha256_file(path)
        or contract["root_sha256"] != sha256_text(canonical_json(rows))
    ):
        fail("LEDGER_INVALID", "model-assisted ledger identity 非法")
    for row in rows:
        child = safe_join(root, str(row.get("path")), location="ledger.row.path")
        _ordinary_file(child, location="package payload")
        if row.get("size_bytes") != child.stat().st_size or row.get("sha256") != sha256_file(child):
            fail("LEDGER_INVALID", f"model-assisted payload 篡改：{row.get('path')}")


def _manifest_payload(
    root: Path,
    manifest: Mapping[str, Any],
    key: str,
) -> Path:
    contract = _mapping(manifest[key], f"manifest.{key}")
    _exact(contract, ("path", "sha256"), f"manifest.{key}")
    path = safe_join(root, _text(contract["path"], f"manifest.{key}.path"), location=f"manifest.{key}.path")
    _ordinary_file(path, location=f"package.{key}")
    if contract["sha256"] != sha256_file(path):
        fail("ANNOTATION_INVALID", f"package {key} SHA 漂移")
    return path


def _package_assignments(
    collection: ModelAssistedCollectionContext,
) -> tuple[dict[str, Any], ...]:
    return tuple(_assignment(entry) for entry in collection.entries)


def validate_model_assisted_supervision(
    *,
    collection_root: Path | str,
    package_root: Path | str,
) -> ModelAssistedSupervisionPackage:
    """严格重算混合监督选择；通过不代表 Gold、共识或科学验收。"""

    collection = _load_collection_context(collection_root)
    root = _ordinary_root(package_root, location="model_assisted_package_root")
    manifest_path = root / "manifest.json"
    _ordinary_file(manifest_path, location="package manifest")
    manifest = _mapping(read_json(manifest_path), "package manifest")
    _exact(manifest, PACKAGE_MANIFEST_FIELDS, "package manifest")
    if (
        manifest["schema_version"] != SUPERVISION_PACKAGE_SCHEMA
        or manifest["source_collection_root"] != str(collection.root)
        or manifest["source_collection_manifest_sha256"] != collection.manifest_sha256
        or manifest["split"] != "train"
        or manifest["record_count"] != EXPECTED_COLLECTION_COUNT
        or manifest["reference_authority"] != REFERENCE_AUTHORITY
        or manifest["expert_consensus"] is not False
        or manifest["gold"] is not False
        or manifest["thresholds_frozen"] is not False
        or manifest["formal_acceptance"] is not False
        or manifest["scientific_acceptance"] is not False
        or manifest["sealed_test_evaluated"] is not False
    ):
        fail("FORMAL_EVALUATION_FORBIDDEN", "model-assisted package 身份或科学边界非法")
    keys = (
        "supervision", "exclusions", "model_drafts", "draft_runs",
        "expert_annotations", "import_provenance",
    )
    paths = {key: _manifest_payload(root, manifest, key) for key in keys}
    _validate_ledger(root, manifest, {path.name for path in paths.values()})
    provenance = _validate_legacy_provenance(read_json(paths["import_provenance"]))
    assignments = _package_assignments(collection)
    assignment_by_id = {str(row["record_id"]): row for row in assignments}
    draft_values = read_jsonl(paths["model_drafts"])
    if len(draft_values) != len(assignments):
        fail("ANNOTATION_INVALID", "package 必须保存全部 8450 个 draft")
    drafts: dict[str, dict[str, Any]] = {}
    draft_ids: set[str] = set()
    for index, (assignment, raw) in enumerate(zip(assignments, draft_values, strict=True)):
        if raw.get("record_id") != assignment["record_id"]:
            fail("ANNOTATION_INVALID", "package drafts 顺序必须与 collection 一致")
        row = validate_model_draft_row(raw, assignment=assignment, location=f"drafts[{index}]")
        if row["draft_id"] in draft_ids:
            fail("ANNOTATION_INVALID", "package draft_id 重复")
        draft_ids.add(str(row["draft_id"]))
        drafts[str(row["record_id"])] = row
    run_rows = tuple(
        validate_draft_run_row(value, location=f"draft_runs[{index}]")
        for index, value in enumerate(read_jsonl(paths["draft_runs"]))
    )
    run_by_id = {str(row["draft_run_id"]): row for row in run_rows}
    if len(run_by_id) != len(run_rows):
        fail("ANNOTATION_INVALID", "package draft run 重复")
    for draft in drafts.values():
        run = run_by_id.get(str(draft["draft_run_id"]))
        entry = collection.entries[int(assignment_by_id[str(draft["record_id"])]["ordinal"])]
        if run is None or draft["record_id"] not in run["record_ids"]:
            fail("ANNOTATION_INVALID", "package draft 缺少 run provenance")
        messages = build_annotation_draft_messages(
            entry.record,
            asset_root=entry.member_root,
            prompt_text=run["prompt_text"],
        )
        if draft["messages_sha256"] != sha256_text(canonical_json(messages)):
            fail("ANNOTATION_INVALID", "package draft message identity 漂移")
    expert_values = read_jsonl(paths["expert_annotations"])
    experts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(expert_values):
        record_id = str(raw.get("record_id", ""))
        assignment = assignment_by_id.get(record_id)
        draft = drafts.get(record_id)
        if assignment is None or draft is None or record_id in experts:
            fail("ANNOTATION_INVALID", f"expert_annotations[{index}] record 非法")
        experts[record_id] = validate_verified_annotation_row(
            raw,
            assignment=assignment,
            draft=draft,
            location=f"expert_annotations[{index}]",
        )
    imported_drafts, imported_verified = _legacy_import_rows(provenance)
    if any(drafts.get(record_id) != row for record_id, row in imported_drafts.items()) or any(
        experts.get(record_id) != row for record_id, row in imported_verified.items()
    ):
        fail("ANNOTATION_INVALID", "package 丢失或替换 legacy draft/verified")
    supervision_values = read_jsonl(paths["supervision"])
    exclusion_values = read_jsonl(paths["exclusions"])
    supervisors = {str(row.get("record_id")): row for row in supervision_values}
    excluded = {str(row.get("record_id")): row for row in exclusion_values}
    if (
        len(supervisors) != len(supervision_values)
        or len(excluded) != len(exclusion_values)
        or set(supervisors) & set(excluded)
        or set(supervisors) | set(excluded) != set(assignment_by_id)
    ):
        fail("ANNOTATION_INVALID", "supervision/exclusion 必须互斥且完整覆盖 collection")
    expected_supervision: list[dict[str, Any]] = []
    expected_exclusions: list[dict[str, Any]] = []
    authority_counts = {EXPERT_AUTHORITY: 0, MODEL_AUTHORITY: 0}
    exclusion_counts: dict[str, int] = {}
    for assignment in assignments:
        record_id = str(assignment["record_id"])
        draft = drafts[record_id]
        verified = experts.get(record_id)
        decision = assess_supervision_eligibility(
            draft=draft,
            assignment=assignment,
            verified=verified,
        )
        if decision.eligible:
            assert decision.description is not None and decision.authority is not None
            description = dict(decision.description)
            expected = {
                "schema_version": SUPERVISION_RECORD_SCHEMA,
                "record_id": record_id,
                "parent_id": assignment["parent_id"],
                "source": assignment["source"],
                "split": "train",
                "draft_id": draft["draft_id"],
                "asset_identity_sha256": assignment["asset_identity_sha256"],
                "supervision_authority": decision.authority,
                "quality_status": decision.quality_status,
                "description": description,
                "description_identity_sha256": sha256_text(canonical_json(description)),
                "expert_annotation_identity_sha256": (
                    sha256_text(canonical_json(verified)) if verified is not None else None
                ),
            }
            _exact(_mapping(supervisors[record_id], "supervision"), SUPERVISION_FIELDS, "supervision")
            if supervisors[record_id] != expected:
                fail("ANNOTATION_INVALID", f"supervision 决策漂移：{record_id}")
            authority_counts[decision.authority] += 1
            expected_supervision.append(expected)
        else:
            assert decision.reason_code is not None
            expected = {
                "schema_version": EXCLUSION_SCHEMA,
                "record_id": record_id,
                "draft_id": draft["draft_id"],
                "reason_code": decision.reason_code,
                "quality_status": decision.quality_status,
                "details": dict(decision.details),
            }
            _exact(_mapping(excluded[record_id], "exclusion"), EXCLUSION_FIELDS, "exclusion")
            if excluded[record_id] != expected:
                fail("ANNOTATION_INVALID", f"exclusion 决策漂移：{record_id}")
            exclusion_counts[decision.reason_code] = exclusion_counts.get(decision.reason_code, 0) + 1
            expected_exclusions.append(expected)
    eligible_ids = [row["record_id"] for row in expected_supervision]
    ordered_ids = [row["record_id"] for row in assignments]
    expected_package_id = sha256_text(canonical_json({
        "source_collection_manifest_sha256": collection.manifest_sha256,
        "project_identity_sha256": manifest["project_identity_sha256"],
        "payloads": {key: manifest[key]["sha256"] for key in keys},
    }))
    if (
        manifest["package_id"] != expected_package_id
        or manifest["eligible_count"] != len(expected_supervision)
        or manifest["excluded_count"] != len(expected_exclusions)
        or manifest["eligible_count"] + manifest["excluded_count"] != EXPECTED_COLLECTION_COUNT
        or manifest["authority_counts"] != dict(sorted(authority_counts.items()))
        or manifest["exclusion_counts"] != dict(sorted(exclusion_counts.items()))
        or manifest["ordered_record_ids_sha256"] != sha256_text(canonical_json(ordered_ids))
        or manifest["eligible_record_ids_sha256"] != sha256_text(canonical_json(eligible_ids))
        or manifest["training_eligible"] is not bool(expected_supervision)
    ):
        fail("ANNOTATION_INVALID", "model-assisted package manifest 统计或 identity 漂移")
    return ModelAssistedSupervisionPackage(
        root=root,
        manifest=manifest,
        supervision=tuple(expected_supervision),
        exclusions=tuple(expected_exclusions),
    )


def export_model_assisted_training_messages(
    *,
    collection_root: Path | str,
    package_root: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """只为实际 eligible 的 train 监督生成 assistant-target messages。"""

    collection = _load_collection_context(collection_root)
    package = validate_model_assisted_supervision(
        collection_root=collection.root,
        package_root=package_root,
    )
    if package.manifest["training_eligible"] is not True or not package.supervision:
        fail("SPLIT_FORBIDDEN", "空或非 train model-assisted package 不得导出训练消息")
    entries = {str(entry.record["record_id"]): entry for entry in collection.entries}
    rows = []
    for supervision in package.supervision:
        record_id = str(supervision["record_id"])
        entry = entries[record_id]
        if entry.record["split"] != "train":
            fail("SPLIT_FORBIDDEN", "val/test supervision 不得进入 training messages")
        actual_asset = region_asset_identity(entry.member_root, entry.record["assets"])
        if actual_asset != supervision["asset_identity_sha256"]:
            fail("ANNOTATION_INVALID", f"training export asset identity 漂移：{record_id}")
        parsed = parse_region_model_output(supervision["description"])
        assistant_target = canonical_json(parsed.to_dict())
        messages = build_mask_grounded_region_messages(
            entry.record,
            asset_root=entry.member_root,
            assistant_target=assistant_target,
        )
        rows.append({
            "schema_version": TRAINING_MESSAGE_SCHEMA,
            "record_id": record_id,
            "parent_id": entry.record["parent_id"],
            "source": entry.record["source"],
            "logical_role": "train",
            "task_family": "mask_grounded_region_description",
            "messages": messages,
            "supervision_identity_sha256": sha256_text(canonical_json(supervision)),
            "asset_identity_sha256": actual_asset,
            "supervision_authority": supervision["supervision_authority"],
        })
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("messages.jsonl", rows)
        ledger = ledger_rows(writer.staging, ("messages.jsonl",))
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        manifest = {
            "schema_version": TRAINING_MANIFEST_SCHEMA,
            "source_collection_root": str(collection.root),
            "source_collection_manifest_sha256": collection.manifest_sha256,
            "supervision_package_root": str(package.root),
            "supervision_package_manifest_sha256": sha256_file(package.root / "manifest.json"),
            "split": "train",
            "record_count": len(rows),
            "ordered_record_ids_sha256": sha256_text(canonical_json([
                row["record_id"] for row in rows
            ])),
            "messages": _file_contract(writer.path("messages.jsonl")),
            "assistant_target_schema": REGION_OUTPUT_SCHEMA_VERSION,
            "ledger": _ledger_contract(writer.path("SHA256SUMS.jsonl"), ledger),
            "training_eligible": True,
            "reference_authority": REFERENCE_AUTHORITY,
            "gold": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    artifact = load_model_assisted_training_messages(root)
    return {
        "ok": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "record_count": len(artifact.rows),
        "training_eligible": True,
        "reference_authority": REFERENCE_AUTHORITY,
        "formal_acceptance": False,
    }


def load_model_assisted_training_messages(
    training_root: Path | str,
) -> ModelAssistedTrainingArtifact:
    """训练前严格重载动态条数的 v2 messages 与其混合监督来源。"""

    root = _ordinary_root(training_root, location="model_assisted_training_root")
    manifest_path = root / "manifest.json"
    _ordinary_file(manifest_path, location="training manifest")
    manifest = _mapping(read_json(manifest_path), "training manifest")
    _exact(manifest, TRAINING_MANIFEST_FIELDS, "training manifest")
    if (
        manifest["schema_version"] != TRAINING_MANIFEST_SCHEMA
        or manifest["split"] != "train"
        or isinstance(manifest["record_count"], bool)
        or not isinstance(manifest["record_count"], int)
        or manifest["record_count"] <= 0
        or manifest["assistant_target_schema"] != REGION_OUTPUT_SCHEMA_VERSION
        or manifest["training_eligible"] is not True
        or manifest["reference_authority"] != REFERENCE_AUTHORITY
        or manifest["gold"] is not False
        or manifest["formal_acceptance"] is not False
        or manifest["scientific_acceptance"] is not False
    ):
        fail("SPLIT_FORBIDDEN", "training manifest 不是 train-only mixed v2 artifact")
    collection = _load_collection_context(manifest["source_collection_root"])
    if manifest["source_collection_manifest_sha256"] != collection.manifest_sha256:
        fail("ANNOTATION_INVALID", "training collection identity 漂移")
    package_root = _ordinary_root(
        manifest["supervision_package_root"],
        location="training supervision package",
    )
    if sha256_file(package_root / "manifest.json") != manifest["supervision_package_manifest_sha256"]:
        fail("ANNOTATION_INVALID", "training supervision package manifest 漂移")
    package = validate_model_assisted_supervision(
        collection_root=collection.root,
        package_root=package_root,
    )
    if manifest["record_count"] != len(package.supervision):
        fail("ANNOTATION_INVALID", "training messages 数量必须等于 package eligible 数量")
    messages_path = _manifest_payload(root, manifest, "messages")
    _validate_ledger(root, manifest, {messages_path.name})
    values = read_jsonl(messages_path)
    if len(values) != manifest["record_count"]:
        fail("ANNOTATION_INVALID", "training messages record_count 漂移")
    entries = {str(entry.record["record_id"]): entry for entry in collection.entries}
    supervision_by_id = {str(row["record_id"]): row for row in package.supervision}
    rows = []
    seen: set[str] = set()
    for index, (supervision, value) in enumerate(zip(package.supervision, values, strict=True)):
        row = _mapping(value, f"messages[{index}]")
        _exact(row, TRAINING_ROW_FIELDS, f"messages[{index}]")
        record_id = _text(row["record_id"], f"messages[{index}].record_id")
        if record_id != supervision["record_id"] or record_id in seen:
            fail("ANNOTATION_INVALID", "training messages 顺序、record 或唯一性漂移")
        seen.add(record_id)
        entry = entries.get(record_id)
        if entry is None or record_id not in supervision_by_id or entry.record["split"] != "train":
            fail("SPLIT_FORBIDDEN", "training message 引用未知或非 train record")
        parsed = parse_region_model_output(supervision["description"])
        expected_messages = build_mask_grounded_region_messages(
            entry.record,
            asset_root=entry.member_root,
            assistant_target=canonical_json(parsed.to_dict()),
        )
        actual_asset = region_asset_identity(entry.member_root, entry.record["assets"])
        expected = {
            "schema_version": TRAINING_MESSAGE_SCHEMA,
            "record_id": record_id,
            "parent_id": entry.record["parent_id"],
            "source": entry.record["source"],
            "logical_role": "train",
            "task_family": "mask_grounded_region_description",
            "messages": expected_messages,
            "supervision_identity_sha256": sha256_text(canonical_json(supervision)),
            "asset_identity_sha256": actual_asset,
            "supervision_authority": supervision["supervision_authority"],
        }
        if row != expected:
            fail("ANNOTATION_INVALID", f"training message 监督或 identity 漂移：{record_id}")
        rows.append(row)
    if manifest["ordered_record_ids_sha256"] != sha256_text(canonical_json([
        row["record_id"] for row in rows
    ])):
        fail("ANNOTATION_INVALID", "training ordered IDs identity 漂移")
    return ModelAssistedTrainingArtifact(root=root, manifest=manifest, rows=tuple(rows))


class ModelAssistedTrainingMessageDataset:
    """把 v2 混合监督暴露给现有 DescriptionCollator，不修改其 loss 语义。"""

    def __init__(self, training_root: Path | str) -> None:
        artifact = load_model_assisted_training_messages(training_root)
        self.root = artifact.root
        self.manifest = artifact.manifest
        self.records = artifact.rows

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            fail("ANNOTATION_INVALID", "training message epoch 必须是非负整数")

    def __getitem__(self, index: int) -> Any:
        from oa_groundrag.phase4.contracts import MaskMode
        from oa_groundrag.phase4.data import DescriptionSample

        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        row = self.records[index]
        messages = tuple(row["messages"])
        assistant = messages[-1]["content"][0]["text"]
        return DescriptionSample(
            record_id=str(row["record_id"]),
            parent_id=str(row["parent_id"]),
            logical_role="train",
            task_family="mask_grounded_region_description",
            messages=messages,
            reference_responses=(str(assistant),),
            mask_mode=MaskMode.GT_MASK,
            evidence_ids=(str(row["asset_identity_sha256"]),),
            provenance={
                "training_manifest_sha256": sha256_file(self.root / "manifest.json"),
                "supervision_package_manifest_sha256": self.manifest[
                    "supervision_package_manifest_sha256"
                ],
                "supervision_authority": row["supervision_authority"],
                "reference_authority": REFERENCE_AUTHORITY,
                "gold": False,
                "formal_acceptance": False,
            },
            counterfactual=None,
        )
