"""Stage 4B 本地 Silver 的身份预检、可恢复生成、过滤与验证。"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from oa_groundrag.phase3.common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    first_symlink_component,
    read_json,
    read_jsonl,
    safe_join,
    sha256_file,
    sha256_text,
    stable_hash,
)
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.checkpoint import CheckpointManager
from oa_groundrag.phase4.gate_b_acceptance import verify_gate_b_acceptance
from oa_groundrag.phase4.gate_b_selection import load_gate_b_selection
from oa_groundrag.phase4.model import Qwen3VLModelAdapter
from oa_groundrag.phase4.processing import Qwen3VLProcessorAdapter
from oa_groundrag.phase4.trainer import set_global_seed, training_layout_identity
from oa_groundrag.phase4.evidence import render_context_crop, render_mask_overlay

from .contracts import (
    SILVER_PROMPT_VERSION,
    SUPPORTED_MODALITIES,
    LandslideEvidenceError,
    SilverConfig,
    derive_claims,
    fail,
    load_silver_config,
    unavailable_auxiliary_facts,
    validate_record,
)
from .silver import (
    evidence_identity,
    make_silver_candidate,
    rule_violations,
    validate_silver_candidate,
)
from .validation import validate_corpus


GENERATION_SCHEMA = "oa_groundrag.landslide_evidence.silver_generation.v1"
ATTEMPT_SCHEMA = "oa_groundrag.landslide_evidence.silver_attempt.v1"
FILTER_SCHEMA = "oa_groundrag.landslide_evidence.silver_filter.v1"
REVIEW_SCHEMA = "oa_groundrag.landslide_evidence.silver_review_queue.v1"
REVIEW_ITEM_SCHEMA = "oa_groundrag.landslide_evidence.silver_review_item.v1"
RUN_STATE_SCHEMA = "oa_groundrag.landslide_evidence.silver_run_state.v1"
CORE_FIELDS = (
    "boundary_clarity",
    "surface_or_vegetation_disturbance",
    "terrain_support",
    "evidence_sufficiency",
)
PROMPT_TEMPLATE = (
    "你是遥感影像证据观察助手。只描述提供图像中与给定 mask 相关的可见证据，"
    "target-present 只能使用‘疑似、可能、候选、尚不能确认’等谨慎表述，不得确定该区域为滑坡；"
    "no-target 不得作确定或谨慎的滑坡存在声明。不得推断发生时间、触发原因、速度、稳定性或风险等级。"
    "严格服从 program facts、allowed/forbidden/unavailable claims。"
    "只返回一个 JSON 对象，不要 Markdown、代码围栏或额外文字。"
)
PROMPT_TEMPLATE_SHA256 = sha256_text(PROMPT_TEMPLATE)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SilverContext:
    config: SilverConfig
    corpus_manifest: Mapping[str, Any]
    corpus_records: tuple[Mapping[str, Any], ...]
    corpus_hashes: Mapping[str, str]
    repository: Mapping[str, Any]
    implementation: Mapping[str, str]
    gate_b: Mapping[str, Any]
    gate_context: Any


def _regular_file(path: Path, *, location: str) -> Path:
    path = Path(path)
    linked = first_symlink_component(path)
    if linked is not None or not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        fail("IDENTITY_MISMATCH", f"{location} 必须是普通单链接文件", details={"path": str(path), "linked": None if linked is None else str(linked)})
    return path


def _regular_directory(path: Path, *, location: str) -> Path:
    path = Path(path)
    linked = first_symlink_component(path)
    if linked is not None or not path.is_dir() or path.is_symlink():
        fail("IDENTITY_MISMATCH", f"{location} 必须是普通目录", details={"path": str(path), "linked": None if linked is None else str(linked)})
    return path


def _exact_regular_files(root: Path, expected: set[str], *, code: str) -> None:
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            fail(code, "artifact 含 symlink", details={"path": relative})
        if path.is_dir():
            continue
        _regular_file(path, location=f"artifact.{relative}")
        actual.add(relative)
    if actual != expected:
        fail(code, "artifact 文件集合不匹配", details={
            "missing": sorted(expected - actual), "unexpected": sorted(actual - expected),
        })


def _repository_identity(*, require_clean: bool) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        fail("REPOSITORY_INVALID", "无法读取 Git repository identity", details={"error": str(error)})
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        fail("REPOSITORY_INVALID", "Git HEAD 非法")
    clean = not status
    if require_clean and not clean:
        fail("DIRTY_WORKTREE", "正式 Silver preflight 要求 clean worktree", details={"paths": status[:50], "count": len(status)})
    return {"head": head, "clean": clean}


def _implementation_identity(config: SilverConfig) -> dict[str, str]:
    paths = {
        "contracts.py": Path(__file__).with_name("contracts.py"),
        "silver.py": Path(__file__).with_name("silver.py"),
        "silver_runtime.py": Path(__file__),
        "validation.py": Path(__file__).with_name("validation.py"),
        "run_landslide_evidence.py": REPO_ROOT / "scripts/stage4_landslide_evidence/run_landslide_evidence.py",
        "silver_config.yaml": config.config_path,
    }
    return {name: sha256_file(_regular_file(path, location=f"implementation.{name}")) for name, path in sorted(paths.items())}


def _corpus_context(
    config: SilverConfig, *, verify_source: bool,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, str]]:
    validation = validate_corpus(config.corpus_root, verify_source=verify_source)
    if validation.get("valid") is not True or (verify_source and validation.get("source_verified") is not True):
        fail("CORPUS_INVALID", "Stage 4A Corpus validator 未通过")
    manifest_path = _regular_file(config.corpus_root / "manifest.json", location="corpus.manifest")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        fail("CORPUS_INVALID", "Corpus manifest 必须是对象")
    actual = {
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_id": manifest.get("corpus_id"),
        "records_sha256": manifest.get("records", {}).get("sha256"),
        "ledger_file_sha256": manifest.get("ledger", {}).get("file_sha256"),
        "ledger_root_sha256": manifest.get("ledger", {}).get("root_sha256"),
    }
    if actual != dict(config.corpus_expected):
        fail("CORPUS_IDENTITY_MISMATCH", "Corpus identity 与 Silver config 不一致", details={"expected": dict(config.corpus_expected), "actual": actual})
    records = tuple(validate_record(row, location=f"records[{index}]") for index, row in enumerate(read_jsonl(config.corpus_root / "records.jsonl")))
    if len(records) != 500 or len({row["sample_id"] for row in records}) != 500:
        fail("CORPUS_INVALID", "Silver v1 必须绑定 500 条唯一 Auto Pilot records")
    ledger = read_jsonl(config.corpus_root / "SHA256SUMS.jsonl")
    hashes = {str(row["path"]): str(row["sha256"]) for row in ledger}
    return manifest, records, hashes


def _teacher_context(config: SilverConfig) -> tuple[dict[str, Any], Any]:
    frozen = _regular_file(config.gate_b_frozen_protocol, location="teacher.frozen_protocol")
    if sha256_file(frozen) != config.expected_frozen_protocol_sha256:
        fail("TEACHER_IDENTITY_MISMATCH", "Gate B frozen protocol SHA 不匹配")
    verification = verify_gate_b_acceptance(
        config.gate_b_protocol_config,
        config.gate_b_selection,
        training_root=config.training_root,
        base_run=config.base_run,
        adapter_run=config.adapter_run,
        evaluation_root=config.evaluation_root,
        expected_protocol_file_sha256=config.expected_frozen_protocol_sha256,
        expected_report_sha256=config.expected_gate_b_report_sha256,
    )
    if not verification.formal_acceptance or verification.status != "accepted":
        fail("TEACHER_NOT_ACCEPTED", "RS-General Adapter Gate B 未接受")
    context = load_gate_b_selection(config.gate_b_protocol_config, config.gate_b_selection)
    frozen_payload = read_json(frozen)
    if context.frozen_protocol != frozen_payload:
        fail("TEACHER_IDENTITY_MISMATCH", "Gate B selection 未绑定指定 frozen protocol")
    static = frozen_payload["static_protocol"]
    training = frozen_payload["training_run"]
    best = training["best_checkpoint"]
    actual_artifacts = {
        "expected_training_report_sha256": training["training_report"]["sha256"],
        "expected_best_pointer_sha256": best["pointer_sha256"],
        "expected_checkpoint_manifest_sha256": best["manifest_sha256"],
        "expected_adapter_sha256": best["adapter_sha256"],
        "expected_model_identity_sha256": sha256_text(canonical_json(static["model_identity"])),
        "expected_processor_identity_sha256": sha256_text(canonical_json(static["processor_identity"])),
    }
    if actual_artifacts != dict(config.expected_teacher_artifacts):
        fail("TEACHER_IDENTITY_MISMATCH", "Gate B teacher artifact 锚点与 Silver config 不一致", details={
            "expected": dict(config.expected_teacher_artifacts), "actual": actual_artifacts,
        })
    return verification.to_dict(), context


def _preflight_context(
    config_path: Path | str, *, require_clean: bool = True, verify_source: bool = True,
) -> SilverContext:
    config = load_silver_config(config_path)
    if config.prompt_template_sha256 != PROMPT_TEMPLATE_SHA256:
        fail("PROMPT_IDENTITY_MISMATCH", "Silver prompt template SHA 与 config 不一致")
    runtime = _runtime_identity()
    actual_runtime = {name: runtime.get(name) for name in config.runtime_expected}
    if actual_runtime != dict(config.runtime_expected):
        fail("RUNTIME_IDENTITY_MISMATCH", "Silver runtime 与 config 不一致", details={
            "expected": dict(config.runtime_expected), "actual": actual_runtime,
        })
    repository = _repository_identity(require_clean=require_clean and config.require_clean_worktree)
    manifest, records, hashes = _corpus_context(config, verify_source=verify_source)
    gate_b, gate_context = _teacher_context(config)
    implementation = _implementation_identity(config)
    linked = first_symlink_component(config.output_root)
    if linked is not None:
        fail("OUTPUT_LINK", "Silver output_root 含 symlink", details={"path": str(linked)})
    return SilverContext(config, manifest, records, hashes, repository, implementation, gate_b, gate_context)


def preflight_silver(config_path: Path | str) -> dict[str, Any]:
    context = _preflight_context(config_path)
    plan = _attempt_plan(context, limit=None)
    input_budget = _validate_input_budget(context, plan)
    variant_counts = Counter(str(item["variant"]) for item in plan)
    record_count = len({str(item["record_id"]) for item in plan if item["variant"] == "regular"})
    generation = context.config.output_root / "generation"
    work = context.config.output_root / ".generation.work"
    return {
        "ok": True,
        "formal_run_ready": True,
        "config_identity": {"file_sha256": context.config.config_file_sha256, "semantic_sha256": context.config.semantic_sha256},
        "corpus_identity": dict(context.config.corpus_expected),
        "teacher_identity": dict(context.gate_b),
        "repository": dict(context.repository),
        "implementation": dict(context.implementation),
        "generation_plan": {
            "record_count": record_count,
            "regular_attempt_count": variant_counts["regular"],
            "counterfactual_attempt_count": variant_counts["wrong_mask"] + variant_counts["modality_removal"],
            "attempt_count": len(plan),
            "attempt_plan_sha256": sha256_text(canonical_json(plan)),
        },
        "input_budget": input_budget,
        "output_state": "completed" if generation.exists() else "partial" if work.exists() else "not_started",
    }


def _round_robin_limit(records: Sequence[Mapping[str, Any]], limit: int | None) -> list[Mapping[str, Any]]:
    if limit is None:
        return list(records)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(records):
        fail("ARGUMENT_INVALID", f"--limit 必须位于 [1,{len(records)}]")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source"])].append(record)
    output: list[Mapping[str, Any]] = []
    while len(output) < limit:
        progressed = False
        for source in sorted(groups):
            if groups[source] and len(output) < limit:
                output.append(groups[source].pop(0))
                progressed = True
        if not progressed:
            break
    return output


def _balanced_hash_select(
    records: Sequence[Mapping[str, Any]], *, count: int, seed: int, label: str,
) -> list[Mapping[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source"])].append(record)
    for source in groups:
        groups[source].sort(key=lambda row: stable_hash(seed, label, source, row["sample_id"]))
    output: list[Mapping[str, Any]] = []
    while len(output) < count:
        progressed = False
        for source in sorted(groups):
            if groups[source] and len(output) < count:
                output.append(groups[source].pop(0))
                progressed = True
        if not progressed:
            break
    if len(output) != count:
        fail("SELECTION_CAPACITY", f"{label} 只能选择 {len(output)}/{count}")
    return output


def _image_size(root: Path, record: Mapping[str, Any], role: str) -> tuple[int, int]:
    relative = record["assets"][role]
    path = safe_join(root, relative, location=f"assets.{role}")
    with Image.open(_regular_file(path, location=f"assets.{role}")) as image:
        return image.size


def _is_optical_only_counterfactual_record(record: Mapping[str, Any]) -> bool:
    assets = record.get("assets")
    facts = record.get("program_facts")
    if not isinstance(assets, dict) or not isinstance(facts, dict):
        return False
    modalities = facts.get("modalities")
    return (
        record.get("target_state") == "target_present"
        and assets.get("auxiliaries") == {}
        and record.get("available_modalities") == ["optical"]
        and set(record.get("unavailable_modalities", ())) == set(SUPPORTED_MODALITIES)
        and isinstance(modalities, dict)
        and set(modalities) == set(SUPPORTED_MODALITIES)
        and all(modalities[name] == unavailable_auxiliary_facts() for name in SUPPORTED_MODALITIES)
    )


def _require_optical_only_counterfactual_record(
    record: Mapping[str, Any], *, location: str,
) -> None:
    if not _is_optical_only_counterfactual_record(record):
        fail(
            "COUNTERFACTUAL_INVALID",
            f"{location} 必须是无辅助资产且 dem/slope/insar_velocity 全部 unavailable 的 target-present 纯光学样本",
            details={"sample_id": record.get("sample_id"), "source": record.get("source")},
        )


def _wrong_mask_donor(
    context: SilverContext, record: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    _require_optical_only_counterfactual_record(record, location="wrong-mask 原样本")
    size = _image_size(context.config.corpus_root, record, "mask")
    ranked = sorted(candidates, key=lambda row: stable_hash(context.config.seed, "wrong_mask_donor", record["sample_id"], row["sample_id"]))
    for donor in ranked:
        if (
            donor["sample_id"] == record["sample_id"]
            or donor["source"] != record["source"]
            or not _is_optical_only_counterfactual_record(donor)
        ):
            continue
        if (
            _image_size(context.config.corpus_root, donor, "mask") == size
            and context.corpus_hashes[donor["assets"]["mask"]] != context.corpus_hashes[record["assets"]["mask"]]
        ):
            _require_optical_only_counterfactual_record(donor, location="wrong-mask donor")
            return donor
    fail("SELECTION_CAPACITY", f"{record['sample_id']} 缺少同源同尺寸 wrong-mask donor")


def _attempt_seed(config: SilverConfig, sample_id: str, candidate_index: int) -> int:
    return int(stable_hash(config.seed, "silver_candidate", sample_id, candidate_index)[:8], 16) % (2**31)


def _attempt_plan(context: SilverContext, *, limit: int | None) -> list[dict[str, Any]]:
    selected = _round_robin_limit(context.corpus_records, limit)
    plan: list[dict[str, Any]] = []
    for ordinal, record in enumerate(selected):
        for candidate_index in range(context.config.candidates_per_record):
            body = {
                "record_ordinal": ordinal,
                "record_id": record["record_id"],
                "sample_id": record["sample_id"],
                "candidate_index": candidate_index,
                "variant": "regular",
                "seed": _attempt_seed(context.config, record["sample_id"], candidate_index),
                "donor_record_id": None,
                "removed_modalities": [],
            }
            plan.append({**body, "attempt_id": sha256_text(canonical_json(body))})
    audit_total = context.config.counterfactual_sample_count if limit is None else min(context.config.counterfactual_sample_count, len(selected) // 2)
    wrong_count = audit_total // 2
    removal_count = audit_total - wrong_count
    targets = [row for row in selected if row["target_state"] == "target_present"]
    wrong_pool = [row for row in targets if _is_optical_only_counterfactual_record(row)]
    wrong_records = _balanced_hash_select(wrong_pool, count=wrong_count, seed=context.config.seed, label="wrong_mask")
    removal_pool = [row for row in targets if row["assets"]["auxiliaries"]]
    removal_records = _balanced_hash_select(removal_pool, count=removal_count, seed=context.config.seed, label="modality_removal")
    if limit is None:
        wrong_sources = Counter(str(row["source"]) for row in wrong_records)
        removal_sources = Counter(str(row["source"]) for row in removal_records)
        if len(wrong_sources) != 3 or max(wrong_sources.values()) - min(wrong_sources.values()) > 1:
            fail("SELECTION_CAPACITY", "正式 wrong-mask 未在三个纯光学来源间确定性平衡")
        if len(removal_sources) != 2 or max(removal_sources.values()) - min(removal_sources.values()) > 1:
            fail("SELECTION_CAPACITY", "正式 modality-removal 未在两个辅助模态来源间确定性平衡")
        if set(wrong_sources).intersection(removal_sources):
            fail("SELECTION_CAPACITY", "两类反事实来源合同发生重叠")
    ordinal_by_id = {row["record_id"]: index for index, row in enumerate(selected)}
    for record in wrong_records:
        donor = _wrong_mask_donor(context, record, context.corpus_records)
        body = {
            "record_ordinal": ordinal_by_id[record["record_id"]], "record_id": record["record_id"],
            "sample_id": record["sample_id"], "candidate_index": 0, "variant": "wrong_mask",
            "seed": _attempt_seed(context.config, record["sample_id"], 0),
            "donor_record_id": donor["record_id"], "removed_modalities": [],
        }
        plan.append({**body, "attempt_id": sha256_text(canonical_json(body))})
    for record in removal_records:
        removed = sorted(record["assets"]["auxiliaries"])
        body = {
            "record_ordinal": ordinal_by_id[record["record_id"]], "record_id": record["record_id"],
            "sample_id": record["sample_id"], "candidate_index": 0, "variant": "modality_removal",
            "seed": _attempt_seed(context.config, record["sample_id"], 0),
            "donor_record_id": None, "removed_modalities": removed,
        }
        plan.append({**body, "attempt_id": sha256_text(canonical_json(body))})
    return plan


def _variant_record(
    record: Mapping[str, Any], *, variant: str, donor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(record))
    if variant == "regular":
        return output
    if variant == "wrong_mask":
        if donor is None:
            fail("COUNTERFACTUAL_INVALID", "wrong_mask 缺少 donor")
        _require_optical_only_counterfactual_record(record, location="wrong-mask 原样本")
        _require_optical_only_counterfactual_record(donor, location="wrong-mask donor")
        output["program_facts"]["mask"] = copy.deepcopy(donor["program_facts"]["mask"])
    else:
        output["assets"]["auxiliaries"] = {}
        output["available_modalities"] = ["optical"]
        output["unavailable_modalities"] = list(SUPPORTED_MODALITIES)
        output["program_facts"]["modalities"] = {
            name: unavailable_auxiliary_facts() for name in SUPPORTED_MODALITIES
        }
    allowed, forbidden, unavailable = derive_claims(
        target_state=str(output["target_state"]),
        modality_facts=output["program_facts"]["modalities"],
    )
    output["allowed_claims"] = allowed
    output["forbidden_claims"] = forbidden
    output["unavailable_claims"] = unavailable
    return output


def _atomic_png(image: Image.Image, path: Path) -> None:
    if path.exists():
        with Image.open(_regular_file(path, location="counterfactual derived asset")) as actual:
            expected = np.asarray(image)
            observed = np.asarray(actual.convert(image.mode))
        if expected.shape != observed.shape or not np.array_equal(expected, observed):
            fail("COUNTERFACTUAL_INVALID", "既有 counterfactual derived asset 与确定性重算不一致")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _request_for_attempt(
    context: SilverContext,
    plan: Mapping[str, Any],
    *,
    work_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = {row["record_id"]: row for row in context.corpus_records}
    record = records[plan["record_id"]]
    donor = None if plan["donor_record_id"] is None else records[plan["donor_record_id"]]
    variant_record = _variant_record(record, variant=plan["variant"], donor=donor)
    corpus_root = context.config.corpus_root
    assets = record["assets"]
    roles: list[tuple[str, Path]] = [
        ("optical_full", safe_join(corpus_root, assets["optical"], location="assets.optical")),
    ]
    if plan["variant"] == "wrong_mask":
        assert donor is not None
        _require_optical_only_counterfactual_record(record, location="wrong-mask 原样本")
        _require_optical_only_counterfactual_record(donor, location="wrong-mask donor")
        optical_path = roles[0][1]
        donor_mask_path = safe_join(corpus_root, donor["assets"]["mask"], location="donor.mask")
        with Image.open(_regular_file(optical_path, location="counterfactual.optical")) as source:
            optical = source.convert("RGB")
        with Image.open(_regular_file(donor_mask_path, location="counterfactual.mask")) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
        if mask.shape != (optical.height, optical.width):
            fail("COUNTERFACTUAL_INVALID", "wrong-mask donor 与 optical shape 不一致")
        derived = work_root / "counterfactual_assets" / str(plan["attempt_id"])
        overlay_path, crop_path = derived / "overlay.png", derived / "crop.png"
        _atomic_png(render_mask_overlay(optical, mask, alpha=float(context.corpus_manifest["rendering"]["overlay_alpha"])), overlay_path)
        crop, _ = render_context_crop(
            optical, mask, margin_ratio=float(context.corpus_manifest["rendering"]["context_margin_ratio"]),
        )
        _atomic_png(crop, crop_path)
        roles.extend((("mask_overlay_wrong_region", overlay_path), ("context_crop_wrong_region", crop_path)))
        identity = {
            "optical_sha256": context.corpus_hashes[assets["optical"]],
            "mask_sha256": context.corpus_hashes[donor["assets"]["mask"]],
            "overlay_sha256": sha256_file(overlay_path),
            "crop_sha256": sha256_file(crop_path),
            "auxiliaries": {},
        }
    else:
        if assets["overlay"] is not None:
            roles.append(("mask_overlay", safe_join(corpus_root, assets["overlay"], location="assets.overlay")))
        if assets["crop"] is not None:
            roles.append(("context_crop", safe_join(corpus_root, assets["crop"], location="assets.crop")))
        if plan["variant"] == "regular":
            for name, relative in sorted(assets["auxiliaries"].items()):
                roles.append((f"auxiliary:{name}", safe_join(corpus_root, relative, location=f"assets.auxiliaries.{name}")))
            identity = evidence_identity(record, context.corpus_hashes)
        else:
            regular_identity = evidence_identity(record, context.corpus_hashes)
            identity = {**regular_identity, "auxiliaries": {}}
    for role, path in roles:
        _regular_file(path, location=f"request.{role}")
    contract = {
        "language": "zh-CN",
        "required_fields": {
            "optical_observation": "non-empty Chinese string",
            "boundary_clarity": ["clear", "partially_clear", "unclear", "not_applicable"],
            "surface_or_vegetation_disturbance": ["observed", "not_observed", "uncertain", "not_applicable"],
            "terrain_support": ["supports", "does_not_support", "inconclusive", "unavailable"],
            "possible_confusers": "array of non-empty Chinese strings",
            "evidence_sufficiency": ["sufficient", "limited", "insufficient"],
            "short_summary": "non-empty Chinese string",
        },
        "target_state": variant_record["target_state"],
        "program_facts": variant_record["program_facts"],
        "allowed_claims": variant_record["allowed_claims"],
        "forbidden_claims": variant_record["forbidden_claims"],
        "unavailable_claims": variant_record["unavailable_claims"],
    }
    content: list[dict[str, Any]] = []
    for role, path in roles:
        content.extend((
            {"type": "text", "text": f"Evidence role: {role}."},
            {"type": "image", "image": str(path.resolve())},
        ))
    target_rule = (
        "程序标记为 no-target；只能说明未提供目标区域，不能声称整幅影像不存在滑坡。"
        if variant_record["target_state"] == "no_target"
        else "只围绕 mask 标出的候选区域观察，不要把该区域确认为滑坡。"
    )
    content.append({
        "type": "text",
        "text": f"{PROMPT_TEMPLATE}\n{target_rule}\nContract: {canonical_json(contract)}",
    })
    messages = [{"role": "user", "content": content}]
    prompt_identity = {
        "version": SILVER_PROMPT_VERSION,
        "template_sha256": PROMPT_TEMPLATE_SHA256,
        "request_sha256": sha256_text(canonical_json({
            "roles": [role for role, _ in roles],
            "evidence_identity": identity,
            "target_rule": target_rule,
            "contract": contract,
        })),
    }
    return {"messages": messages, "record": variant_record}, identity, prompt_identity


def _strict_model_json(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip() or raw.strip().startswith("```"):
        fail("MODEL_OUTPUT_INVALID", "模型必须返回无代码围栏的非空 JSON 对象")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate key: {key}")
            output[key] = value
        return output
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite: {token}")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        fail("MODEL_OUTPUT_INVALID", "模型输出不是严格 JSON", details={"error": str(error)})
    if not isinstance(value, dict):
        fail("MODEL_OUTPUT_INVALID", "模型输出顶层必须是对象")
    return value


class LocalAdapterSilverProvider:
    def __init__(self, context: SilverContext) -> None:
        import torch

        if not torch.cuda.is_available():
            fail("CUDA_REQUIRED", "正式 Silver generation 要求可用 CUDA")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda")
        gate = context.gate_context
        frozen = gate.frozen_protocol
        source = gate.protocol_source
        config = source.adapter_config
        self.processor = _silver_processor(context)
        processor_identity = self.processor.identity()
        self.model = Qwen3VLModelAdapter.load(
            config.model, config.adaptation, device=self.device, gradient_checkpointing=False,
        )
        model_identity = self.model.identity.to_dict()
        if model_identity != frozen["static_protocol"]["model_identity"]:
            fail("TEACHER_IDENTITY_MISMATCH", "Silver model identity 与 Gate B 不一致")
        best = frozen["training_run"]["best_checkpoint"]
        checkpoint = context.config.training_root / best["relative_path"]
        payload = CheckpointManager().load(
            checkpoint,
            expected_config_semantic_sha256=config.semantic_sha256,
            expected_benchmark_identity=gate.access.identity.training_identity_dict(),
            expected_validation_selection_identity=None,
            expected_model_identity=model_identity,
            expected_processor_identity=processor_identity,
            expected_training_layout=training_layout_identity(config),
            expected_trainable_names=self.model.trainable_names,
        )
        self.model.load_trainable_state_dict(payload.trainable_state)
        self.identity = {
            "model": model_identity,
            "processor": processor_identity,
            "checkpoint": dict(best),
            "device": torch.cuda.get_device_name(self.device),
        }

    def generate(self, messages: Sequence[Mapping[str, Any]], *, seed: int, config: SilverConfig) -> tuple[str, int, int]:
        import torch

        set_global_seed(seed)
        encoded = self.processor.encode_inference(messages)
        batch = {
            key: value.to(device=self.device, non_blocking=False)
            if isinstance(value, torch.Tensor) else value
            for key, value in encoded.tensors.items()
        }
        values = self.model.generate_text(
            batch,
            processor=self.processor.processor,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        if len(values) != 1 or not values[0].strip():
            fail("MODEL_OUTPUT_INVALID", "单样本 Silver generation 必须返回一个非空文本")
        return values[0], encoded.input_token_count, encoded.image_count


def _runtime_identity() -> dict[str, Any]:
    values = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "peft", "qwen-vl-utils"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def _silver_processor(context: SilverContext) -> Qwen3VLProcessorAdapter:
    gate = context.gate_context
    frozen = gate.frozen_protocol
    config = gate.protocol_source.adapter_config
    processor = Qwen3VLProcessorAdapter(
        processor_path=config.model.processor_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        min_pixels=config.limits.min_pixels,
        max_pixels=config.limits.max_pixels,
        max_images=config.limits.max_images,
        max_input_tokens=context.config.max_input_tokens,
    )
    if processor.identity() != frozen["static_protocol"]["processor_identity"]:
        fail("TEACHER_IDENTITY_MISMATCH", "Silver processor identity 与 Gate B 不一致")
    return processor


def _validate_input_budget(
    context: SilverContext, plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """使用真实本地 processor 对每个唯一请求做 CPU 编码并覆盖全部 attempt。"""

    processor = _silver_processor(context)
    observed: dict[str, tuple[int, int, str]] = {}
    max_tokens = -1
    max_images = -1
    worst_attempt: Mapping[str, Any] | None = None
    covered = 0
    with tempfile.TemporaryDirectory(prefix="oa_groundrag_stage4b_budget_") as temporary:
        work_root = Path(temporary)
        for attempt in plan:
            request, _, prompt_identity = _request_for_attempt(context, attempt, work_root=work_root)
            request_sha = str(prompt_identity["request_sha256"])
            message_sha = sha256_text(canonical_json(request["messages"]))
            if request_sha not in observed:
                encoded = processor.encode_inference(request["messages"])
                if (
                    encoded.input_token_count > context.config.max_input_tokens
                    or encoded.image_count > processor.max_images
                ):
                    fail(
                        "INPUT_BUDGET_EXCEEDED",
                        "Silver request 超过冻结 token/image 预算",
                        details={
                            "attempt_id": attempt["attempt_id"],
                            "sample_id": attempt["sample_id"],
                            "input_tokens": encoded.input_token_count,
                            "max_input_tokens": context.config.max_input_tokens,
                            "image_count": encoded.image_count,
                            "max_images": processor.max_images,
                        },
                    )
                observed[request_sha] = (
                    encoded.input_token_count, encoded.image_count, message_sha,
                )
            input_tokens, image_count, expected_message_sha = observed[request_sha]
            if expected_message_sha != message_sha:
                fail("INPUT_BUDGET_INVALID", "相同 request SHA 对应不同消息内容")
            covered += 1
            if (input_tokens, image_count) > (max_tokens, max_images):
                max_tokens, max_images, worst_attempt = input_tokens, image_count, attempt
    if covered != len(plan) or worst_attempt is None:
        fail("INPUT_BUDGET_INVALID", "输入预算未覆盖全部 generation attempts")
    return {
        "configured_max_input_tokens": context.config.max_input_tokens,
        "max_observed_input_tokens": max_tokens,
        "configured_max_images": processor.max_images,
        "max_observed_images": max_images,
        "worst_attempt_id": worst_attempt["attempt_id"],
        "worst_sample_id": worst_attempt["sample_id"],
        "attempt_count": len(plan),
        "covered_attempt_count": covered,
        "unique_request_count": len(observed),
    }


def _protocol_body(
    context: SilverContext,
    plan: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    repository: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_ids = list(dict.fromkeys(str(item["sample_id"]) for item in plan if item["variant"] == "regular"))
    body = {
        "config": {
            "schema_version": context.config.semantic_dict()["schema_version"],
            "file_sha256": context.config.config_file_sha256,
            "semantic_sha256": context.config.semantic_sha256,
        },
        "corpus": dict(context.config.corpus_expected),
        "teacher": dict(context.gate_b),
        "repository": dict(context.repository if repository is None else repository),
        "implementation": dict(context.implementation),
        "prompt": {"version": SILVER_PROMPT_VERSION, "template_sha256": PROMPT_TEMPLATE_SHA256},
        "runtime_expected": dict(context.config.runtime_expected),
        "generation": {
            "seed": context.config.seed,
            "candidates_per_record": context.config.candidates_per_record,
            "batch_size": context.config.batch_size,
            "max_input_tokens": context.config.max_input_tokens,
            "max_new_tokens": context.config.max_new_tokens,
            "do_sample": context.config.do_sample,
            "temperature": context.config.temperature,
            "top_p": context.config.top_p,
        },
        "formal_run": limit is None,
        "limit": limit,
        "selected_sample_count": len(selected_ids),
        "selected_sample_ids_sha256": sha256_text(canonical_json(selected_ids)),
        "attempt_count": len(plan),
        "attempt_plan_sha256": sha256_text(canonical_json(list(plan))),
    }
    return {**body, "protocol_sha256": sha256_text(canonical_json(body))}


def _attempt_filename(index: int, attempt: Mapping[str, Any]) -> str:
    return f"attempts/{index:05d}_{attempt['attempt_id']}.json"


def _validate_attempt_row(row: Any, attempt: Mapping[str, Any], *, protocol_sha256: str) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != {
        "schema_version", "protocol_sha256", "attempt", "status", "input",
        "raw_output", "raw_output_sha256", "candidate", "failure", "input_token_count", "image_count",
    }:
        fail("SILVER_RUN_INVALID", "attempt artifact 字段不匹配")
    if row["schema_version"] != ATTEMPT_SCHEMA or row["protocol_sha256"] != protocol_sha256 or row["attempt"] != dict(attempt):
        fail("SILVER_RUN_INVALID", "attempt artifact identity 不匹配")
    if row["status"] not in {"completed", "model_output_invalid"}:
        fail("SILVER_RUN_INVALID", "attempt status 非法")
    if not isinstance(row["raw_output"], str) or not row["raw_output"]:
        fail("SILVER_RUN_INVALID", "attempt 必须保留非空 raw output")
    if (
        not isinstance(row["input"], dict)
        or isinstance(row["input_token_count"], bool)
        or not isinstance(row["input_token_count"], int)
        or row["input_token_count"] < 0
        or isinstance(row["image_count"], bool)
        or not isinstance(row["image_count"], int)
        or row["image_count"] < 1
    ):
        fail("SILVER_RUN_INVALID", "attempt input/token/image metadata 非法")
    if row["status"] == "completed":
        if row["failure"] is not None:
            fail("SILVER_RUN_INVALID", "completed attempt payload 非法")
        row["candidate"] = validate_silver_candidate(row["candidate"])
    else:
        if (
            row["candidate"] is not None
            or not isinstance(row["failure"], dict)
            or set(row["failure"]) != {"code", "message", "details"}
            or not isinstance(row["failure"]["code"], str)
            or not isinstance(row["failure"]["message"], str)
            or not isinstance(row["failure"]["details"], dict)
        ):
            fail("SILVER_RUN_INVALID", "invalid attempt payload 非法")
    raw = row["raw_output"]
    expected_raw = None if raw is None else sha256_text(raw)
    if row["raw_output_sha256"] != expected_raw:
        fail("SILVER_RUN_INVALID", "attempt raw output SHA 不匹配")
    return row


def _validate_attempt_binding(
    context: SilverContext,
    row: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    protocol_sha256: str,
    work_root: Path,
) -> dict[str, Any]:
    normalized = _validate_attempt_row(dict(row), attempt, protocol_sha256=protocol_sha256)
    request, identity, prompt_identity = _request_for_attempt(
        context, attempt, work_root=work_root,
    )
    expected_input = {
        "evidence_identity": identity,
        "prompt_identity": prompt_identity,
        "program_facts_sha256": sha256_text(canonical_json(request["record"]["program_facts"])),
    }
    if normalized["input"] != expected_input:
        fail("SILVER_RUN_INVALID", "attempt input identity 不是确定性重算结果")
    max_images = context.gate_context.protocol_source.adapter_config.limits.max_images
    if (
        normalized["input_token_count"] > context.config.max_input_tokens
        or normalized["image_count"] > max_images
    ):
        fail("SILVER_RUN_INVALID", "attempt 记录的输入超过冻结 token/image 预算")
    if normalized["status"] == "completed":
        expected_candidate = make_silver_candidate(
            request["record"],
            _strict_model_json(normalized["raw_output"]),
            attempt_id=str(attempt["attempt_id"]),
            candidate_index=int(attempt["candidate_index"]),
            variant=str(attempt["variant"]),
            seed=int(attempt["seed"]),
            identity=identity,
            prompt_identity=prompt_identity,
            raw_output_sha256=str(normalized["raw_output_sha256"]),
        )
        if normalized["candidate"] != expected_candidate:
            fail("SILVER_RUN_INVALID", "attempt candidate 不是 raw output 和 request 的严格解析结果")
    return normalized


def _output_base(config: SilverConfig, override: Path | None, *, limit: int | None) -> Path:
    if limit is None:
        if override is not None:
            fail("ARGUMENT_INVALID", "正式生成禁止 --output-root override")
        return config.output_root
    if override is None:
        fail("ARGUMENT_REQUIRED", "有限 smoke 必须显式指定 /tmp --output-root")
    path = Path(os.path.abspath(override))
    if path == Path("/tmp"):
        fail("ARGUMENT_INVALID", "有限 smoke output_root 不能是 /tmp 根目录")
    try:
        path.relative_to(Path("/tmp"))
    except ValueError as error:
        fail("ARGUMENT_INVALID", "有限 smoke output_root 必须位于 /tmp")
        raise AssertionError("unreachable") from error
    return path


def _ledger_rows(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail("SILVER_RUN_INVALID", "generation work payload 含 symlink", details={
                "path": path.relative_to(root).as_posix(),
            })
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"generation_manifest.json", "SHA256SUMS.jsonl"}:
            continue
        _regular_file(path, location=f"generation.{relative}")
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows, sha256_text(canonical_json(rows))


def _expected_generation_payload_paths(plan: Sequence[Mapping[str, Any]]) -> set[str]:
    paths = {"run_state.json", "candidates.jsonl", "failures.jsonl"}
    for index, attempt in enumerate(plan):
        paths.add(_attempt_filename(index, attempt))
        if attempt["variant"] == "wrong_mask":
            prefix = f"counterfactual_assets/{attempt['attempt_id']}"
            paths.update((f"{prefix}/overlay.png", f"{prefix}/crop.png"))
    return paths


def generate_silver(
    config_path: Path | str,
    *,
    limit: int | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    context = _preflight_context(config_path)
    plan = _attempt_plan(context, limit=limit)
    protocol = _protocol_body(context, plan, limit=limit)
    input_budget = _validate_input_budget(context, plan)
    base = _output_base(context.config, output_root, limit=limit)
    linked = first_symlink_component(base)
    if linked is not None:
        fail("OUTPUT_LINK", "Silver generation output 含 symlink", details={"path": str(linked)})
    generation_root = base / "generation"
    work_root = base / ".generation.work"
    if generation_root.exists() or generation_root.is_symlink():
        fail("OUTPUT_EXISTS", "Silver generation 已完成，拒绝覆盖", details={"path": str(generation_root)})
    if base.exists() and (base.is_symlink() or not base.is_dir()):
        fail("OUTPUT_EXISTS", "Silver output_root 已存在但不是普通目录", details={"path": str(base)})
    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "in_progress",
        "protocol": protocol,
        "plan": plan,
    }
    resumed_completed_state: dict[str, Any] | None = None
    if work_root.exists() or work_root.is_symlink():
        _regular_directory(work_root, location="generation work root")
        _regular_directory(work_root / "attempts", location="generation attempts root")
        current = read_json(_regular_file(work_root / "run_state.json", location="generation run_state"))
        if current == state:
            pass
        elif (
            isinstance(current, dict)
            and set(current) == {
                "schema_version", "status", "protocol", "plan", "teacher_runtime", "runtime",
            }
            and current.get("schema_version") == RUN_STATE_SCHEMA
            and current.get("status") == "completed"
            and current.get("protocol") == protocol
            and current.get("plan") == plan
            and isinstance(current.get("teacher_runtime"), dict)
            and isinstance(current.get("runtime"), dict)
        ):
            resumed_completed_state = current
        else:
            fail("RESUME_IDENTITY_MISMATCH", "partial generation 与当前 protocol/HEAD/plan 不一致")
        if {path.name for path in base.iterdir()} != {work_root.name}:
            fail("RESUME_IDENTITY_MISMATCH", "partial generation output_root 含未知 sibling")
    else:
        if base.exists() and any(base.iterdir()):
            fail("OUTPUT_EXISTS", "Silver output_root 已含非恢复资产", details={"path": str(base)})
        base.mkdir(parents=True, exist_ok=True)
        work_root.mkdir()
        (work_root / "attempts").mkdir()
        atomic_write_json(work_root / "run_state.json", state)
    expected_files = {_attempt_filename(index, attempt) for index, attempt in enumerate(plan)}
    actual_files = {
        path.relative_to(work_root).as_posix()
        for path in (work_root / "attempts").glob("*.json")
    }
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        fail("RESUME_IDENTITY_MISMATCH", "partial generation 含未知 attempt", details={"paths": unexpected})
    if resumed_completed_state is not None and actual_files != expected_files:
        fail("RESUME_IDENTITY_MISMATCH", "completed-finalization 缺少 attempt artifact", details={
            "missing": sorted(expected_files - actual_files),
        })
    completed: dict[str, dict[str, Any]] = {}
    for index, attempt in enumerate(plan):
        relative = _attempt_filename(index, attempt)
        path = work_root / relative
        if path.exists():
            completed[attempt["attempt_id"]] = _validate_attempt_binding(
                context,
                read_json(_regular_file(path, location=relative)), attempt,
                protocol_sha256=protocol["protocol_sha256"],
                work_root=work_root,
            )
    provider = LocalAdapterSilverProvider(context)
    if resumed_completed_state is not None and (
        resumed_completed_state["teacher_runtime"] != provider.identity
        or resumed_completed_state["runtime"] != _runtime_identity()
    ):
        fail("RESUME_IDENTITY_MISMATCH", "completed-finalization 的 teacher/runtime identity 已变化")
    for index, attempt in enumerate(plan):
        if attempt["attempt_id"] in completed:
            continue
        request, identity, prompt_identity = _request_for_attempt(context, attempt, work_root=work_root)
        raw, input_tokens, image_count = provider.generate(
            request["messages"], seed=int(attempt["seed"]), config=context.config,
        )
        raw_sha = sha256_text(raw)
        candidate = None
        failure = None
        status = "completed"
        try:
            observation = _strict_model_json(raw)
            candidate = make_silver_candidate(
                request["record"], observation,
                attempt_id=str(attempt["attempt_id"]),
                candidate_index=int(attempt["candidate_index"]),
                variant=str(attempt["variant"]),
                seed=int(attempt["seed"]),
                identity=identity,
                prompt_identity=prompt_identity,
                raw_output_sha256=raw_sha,
            )
            validate_silver_candidate(candidate)
        except LandslideEvidenceError as error:
            status = "model_output_invalid"
            failure = {"code": error.code, "message": str(error), "details": error.details}
        artifact = {
            "schema_version": ATTEMPT_SCHEMA,
            "protocol_sha256": protocol["protocol_sha256"],
            "attempt": dict(attempt),
            "status": status,
            "input": {
                "evidence_identity": identity,
                "prompt_identity": prompt_identity,
                "program_facts_sha256": sha256_text(canonical_json(request["record"]["program_facts"])),
            },
            "raw_output": raw,
            "raw_output_sha256": raw_sha,
            "candidate": candidate,
            "failure": failure,
            "input_token_count": input_tokens,
            "image_count": image_count,
        }
        relative = _attempt_filename(index, attempt)
        atomic_write_json(work_root / relative, artifact)
        completed[attempt["attempt_id"]] = _validate_attempt_row(
            artifact, attempt, protocol_sha256=protocol["protocol_sha256"],
        )
    ordered = [completed[item["attempt_id"]] for item in plan]
    candidates = [row["candidate"] for row in ordered if row["status"] == "completed"]
    failures = [
        {
            "attempt_id": row["attempt"]["attempt_id"],
            "record_id": row["attempt"]["record_id"],
            "sample_id": row["attempt"]["sample_id"],
            "variant": row["attempt"]["variant"],
            "candidate_index": row["attempt"]["candidate_index"],
            "raw_output_sha256": row["raw_output_sha256"],
            "failure": row["failure"],
        }
        for row in ordered if row["status"] != "completed"
    ]
    atomic_write_jsonl(work_root / "candidates.jsonl", candidates)
    atomic_write_jsonl(work_root / "failures.jsonl", failures)
    completed_state = {**state, "status": "completed", "teacher_runtime": provider.identity, "runtime": _runtime_identity()}
    atomic_write_json(work_root / "run_state.json", completed_state)
    rows, ledger_root = _ledger_rows(work_root)
    actual_payloads = {row["path"] for row in rows}
    expected_payloads = _expected_generation_payload_paths(plan)
    if actual_payloads != expected_payloads:
        fail("SILVER_RUN_INVALID", "generation work payload 文件集非法", details={"missing": sorted(expected_payloads - actual_payloads), "unexpected": sorted(actual_payloads - expected_payloads)})
    atomic_write_jsonl(work_root / "SHA256SUMS.jsonl", rows)
    manifest = {
        "schema_version": GENERATION_SCHEMA,
        "status": "completed",
        "formal_run": limit is None,
        "protocol": protocol,
        "teacher_runtime": provider.identity,
        "runtime": completed_state["runtime"],
        "selection": {
            "record_count": protocol["selected_sample_count"],
            "ordered_sample_ids_sha256": protocol["selected_sample_ids_sha256"],
            "attempt_count": len(plan),
            "attempt_plan_sha256": protocol["attempt_plan_sha256"],
            "variant_counts": dict(sorted(Counter(item["variant"] for item in plan).items())),
        },
        "candidates": {
            "path": "candidates.jsonl", "count": len(candidates),
            "sha256": sha256_file(work_root / "candidates.jsonl"),
        },
        "failures": {
            "path": "failures.jsonl", "count": len(failures),
            "sha256": sha256_file(work_root / "failures.jsonl"),
        },
        "ledger": {
            "path": "SHA256SUMS.jsonl", "entry_count": len(rows),
            "file_sha256": sha256_file(work_root / "SHA256SUMS.jsonl"),
            "root_sha256": ledger_root,
        },
        "all_attempts_accounted": len(ordered) == len(plan),
        "expert_review_completed": False,
        "oa_grounded_eval": False,
        "gate_c_evaluated": False,
    }
    atomic_write_json(work_root / "generation_manifest.json", manifest)
    work_root.replace(generation_root)
    return {
        "root": str(generation_root),
        "formal_run": limit is None,
        "record_count": protocol["selected_sample_count"],
        "attempt_count": len(plan),
        "candidate_count": len(candidates),
        "failure_count": len(failures),
        "input_budget": input_budget,
        "manifest_sha256": sha256_file(generation_root / "generation_manifest.json"),
    }


def _validate_ledger(
    root: Path, manifest: Mapping[str, Any], *, expected_paths: set[str],
) -> None:
    ledger_contract = manifest.get("ledger")
    if not isinstance(ledger_contract, dict) or set(ledger_contract) != {"path", "entry_count", "file_sha256", "root_sha256"}:
        fail("SILVER_RUN_INVALID", "generation ledger contract 非法")
    ledger_path = _regular_file(root / str(ledger_contract["path"]), location="generation ledger")
    if sha256_file(ledger_path) != ledger_contract["file_sha256"]:
        fail("SILVER_RUN_INVALID", "generation ledger file SHA 不匹配")
    rows = read_jsonl(ledger_path)
    if len(rows) != ledger_contract["entry_count"] or sha256_text(canonical_json(rows)) != ledger_contract["root_sha256"]:
        fail("SILVER_RUN_INVALID", "generation ledger root/count 不匹配")
    ledger_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            fail("SILVER_RUN_INVALID", f"generation ledger[{index}] 字段非法")
        path = safe_join(root, row["path"], location=f"ledger[{index}].path")
        _regular_file(path, location=f"ledger[{index}]")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            fail("SILVER_RUN_INVALID", f"generation ledger[{index}] bytes 不一致")
        ledger_paths.add(str(row["path"]))
    all_paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            fail("SILVER_RUN_INVALID", "generation payload 含 symlink", details={"path": relative})
        if path.is_dir():
            continue
        _regular_file(path, location=f"generation.{relative}")
        all_paths.add(relative)
    actual_paths = all_paths - {"generation_manifest.json", "SHA256SUMS.jsonl"}
    if actual_paths != ledger_paths or ledger_paths != expected_paths:
        fail("SILVER_RUN_INVALID", "generation 实际文件集、ledger 与 plan 不一致", details={
            "missing_from_disk": sorted(ledger_paths - actual_paths),
            "unexpected_on_disk": sorted(actual_paths - ledger_paths),
            "missing_from_ledger": sorted(expected_paths - ledger_paths),
            "unexpected_in_ledger": sorted(ledger_paths - expected_paths),
        })


def _validate_generation(
    context: SilverContext, *, output_base: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = context.config.output_root if output_base is None else Path(output_base)
    root = _regular_directory(base / "generation", location="Silver generation root")
    manifest = read_json(_regular_file(root / "generation_manifest.json", location="generation manifest"))
    required = {
        "schema_version", "status", "formal_run", "protocol", "teacher_runtime", "runtime",
        "selection", "candidates", "failures", "ledger", "all_attempts_accounted",
        "expert_review_completed", "oa_grounded_eval", "gate_c_evaluated",
    }
    if not isinstance(manifest, dict) or set(manifest) != required or manifest.get("schema_version") != GENERATION_SCHEMA or manifest.get("status") != "completed":
        fail("SILVER_RUN_INVALID", "generation manifest 合同非法")
    stored_repository = manifest["protocol"].get("repository")
    if (
        not isinstance(stored_repository, dict)
        or set(stored_repository) != {"head", "clean"}
        or stored_repository.get("clean") is not True
        or not isinstance(stored_repository.get("head"), str)
    ):
        fail("SILVER_RUN_INVALID", "generation repository identity 非法")
    try:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{stored_repository['head']}^{{commit}}"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail("SILVER_RUN_INVALID", "generation 绑定的 Git HEAD 不存在", details={"error": str(error)})
    limit = manifest["protocol"].get("limit")
    plan = _attempt_plan(context, limit=limit)
    expected_protocol = _protocol_body(context, plan, limit=limit, repository=stored_repository)
    if manifest["protocol"] != expected_protocol or manifest["all_attempts_accounted"] is not True:
        fail("SILVER_RUN_INVALID", "generation protocol/plan 与当前身份不一致")
    expected_selection = {
        "record_count": expected_protocol["selected_sample_count"],
        "ordered_sample_ids_sha256": expected_protocol["selected_sample_ids_sha256"],
        "attempt_count": len(plan),
        "attempt_plan_sha256": expected_protocol["attempt_plan_sha256"],
        "variant_counts": dict(sorted(Counter(item["variant"] for item in plan).items())),
    }
    if manifest["formal_run"] is not (limit is None) or manifest["selection"] != expected_selection:
        fail("SILVER_RUN_INVALID", "generation formal/selection 声明不是 protocol 的重算结果")
    if any(manifest[field] is not False for field in ("expert_review_completed", "oa_grounded_eval", "gate_c_evaluated")):
        fail("SILVER_RUN_INVALID", "generation 不得提前声明 review/OA/Gate C")
    _validate_ledger(root, manifest, expected_paths=_expected_generation_payload_paths(plan))
    candidates_path = _regular_file(root / "candidates.jsonl", location="generation candidates")
    failures_path = _regular_file(root / "failures.jsonl", location="generation failures")
    candidates = [validate_silver_candidate(row) for row in read_jsonl(candidates_path)]
    failures = read_jsonl(failures_path)
    for name, path, rows in (("candidates", candidates_path, candidates), ("failures", failures_path, failures)):
        contract = manifest[name]
        if contract != {"path": path.name, "count": len(rows), "sha256": sha256_file(path)}:
            fail("SILVER_RUN_INVALID", f"generation {name} contract 不一致")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        fail("SILVER_RUN_INVALID", "generation candidate_id 重复")
    state = read_json(_regular_file(root / "run_state.json", location="generation run_state"))
    if not isinstance(state, dict) or state.get("schema_version") != RUN_STATE_SCHEMA or state.get("status") != "completed" or state.get("protocol") != expected_protocol or state.get("plan") != plan:
        fail("SILVER_RUN_INVALID", "completed run_state 与 protocol/plan 不一致")
    frozen = context.gate_context.frozen_protocol
    teacher_runtime = manifest["teacher_runtime"]
    expected_teacher_runtime = {
        "model": frozen["static_protocol"]["model_identity"],
        "processor": frozen["static_protocol"]["processor_identity"],
        "checkpoint": frozen["training_run"]["best_checkpoint"],
        "device": teacher_runtime.get("device") if isinstance(teacher_runtime, dict) else None,
    }
    if (
        not isinstance(expected_teacher_runtime["device"], str)
        or not expected_teacher_runtime["device"]
        or teacher_runtime != expected_teacher_runtime
        or state.get("teacher_runtime") != expected_teacher_runtime
    ):
        fail("SILVER_RUN_INVALID", "generation teacher runtime 与冻结模型/checkpoint 不一致")
    runtime = manifest["runtime"]
    runtime_fields = {"python", "platform", "torch", "transformers", "peft", "qwen-vl-utils"}
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_fields
        or not isinstance(runtime.get("platform"), str)
        or not runtime["platform"]
        or any(runtime.get(name) != value for name, value in context.config.runtime_expected.items())
        or state.get("runtime") != runtime
    ):
        fail("SILVER_RUN_INVALID", "generation runtime 与冻结 config/run_state 不一致")
    candidate_by_attempt = {row["attempt_id"]: row for row in candidates}
    failure_by_attempt = {
        str(row.get("attempt_id")): row for row in failures if isinstance(row, dict)
    }
    expected_candidates: list[dict[str, Any]] = []
    expected_failures: list[dict[str, Any]] = []
    for index, attempt in enumerate(plan):
        row = _validate_attempt_binding(
            context,
            read_json(_regular_file(root / _attempt_filename(index, attempt), location=f"attempt[{index}]")),
            attempt,
            protocol_sha256=expected_protocol["protocol_sha256"],
            work_root=root,
        )
        if row["status"] == "completed" and candidate_by_attempt.get(attempt["attempt_id"]) != row["candidate"]:
            fail("SILVER_RUN_INVALID", f"attempt[{index}] candidate 未进入汇总")
        if row["status"] == "completed":
            expected_candidates.append(row["candidate"])
        else:
            expected_failure = {
                "attempt_id": attempt["attempt_id"],
                "record_id": attempt["record_id"],
                "sample_id": attempt["sample_id"],
                "variant": attempt["variant"],
                "candidate_index": attempt["candidate_index"],
                "raw_output_sha256": row["raw_output_sha256"],
                "failure": row["failure"],
            }
            expected_failures.append(expected_failure)
            if failure_by_attempt.get(attempt["attempt_id"]) != expected_failure:
                fail("SILVER_RUN_INVALID", f"attempt[{index}] failure 未严格进入汇总")
    if candidates != expected_candidates or failures != expected_failures:
        fail("SILVER_RUN_INVALID", "candidate/failure 汇总含缺失、额外或顺序漂移")
    return manifest, candidates, failures, plan


def _filter_rows(
    context: SilverContext,
    candidates: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = {row["record_id"]: row for row in context.corpus_records}
    candidates_by_attempt = {row["attempt_id"]: row for row in candidates}
    failure_ids = {str(row.get("attempt_id")) for row in failures}
    plan_by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ordered_record_ids: list[str] = []
    for attempt in plan:
        plan_by_record[attempt["record_id"]].append(attempt)
        if attempt["variant"] == "regular" and attempt["record_id"] not in ordered_record_ids:
            ordered_record_ids.append(attempt["record_id"])
    output: list[dict[str, Any]] = []
    for record_id in ordered_record_ids:
        record = records[record_id]
        attempts = plan_by_record[record_id]
        regular_attempts = sorted(
            (item for item in attempts if item["variant"] == "regular"),
            key=lambda item: item["candidate_index"],
        )
        regular = [candidates_by_attempt.get(item["attempt_id"]) for item in regular_attempts]
        candidate_violations: dict[str, list[str]] = {}
        record_violations: set[str] = set()
        for attempt, candidate in zip(regular_attempts, regular, strict=True):
            if candidate is None:
                code = "CANDIDATE_GENERATION_FAILED" if attempt["attempt_id"] in failure_ids else "CANDIDATE_MISSING"
                record_violations.add(code)
                continue
            violations = sorted(rule_violations(candidate, record))
            candidate_violations[candidate["candidate_id"]] = violations
            record_violations.update(violations)
        core_consistent = False
        if all(candidate is not None for candidate in regular):
            assert regular[0] is not None and regular[1] is not None
            core_consistent = all(
                regular[0]["silver_observation"][field] == regular[1]["silver_observation"][field]
                for field in CORE_FIELDS
            )
        if not core_consistent:
            record_violations.add("CANDIDATE_CORE_DISAGREEMENT")
        counterfactual: list[dict[str, Any]] = []
        for attempt in (item for item in attempts if item["variant"] != "regular"):
            candidate = candidates_by_attempt.get(attempt["attempt_id"])
            violations: set[str] = set()
            if candidate is None:
                violations.add("COUNTERFACTUAL_GENERATION_FAILED" if attempt["attempt_id"] in failure_ids else "COUNTERFACTUAL_MISSING")
            else:
                donor = None if attempt["donor_record_id"] is None else records[attempt["donor_record_id"]]
                variant_record = _variant_record(record, variant=attempt["variant"], donor=donor)
                violations.update(rule_violations(candidate, variant_record))
                if regular[0] is not None and canonical_json(candidate["silver_observation"]) == canonical_json(regular[0]["silver_observation"]):
                    violations.add("MASK_INSENSITIVE_TEXT" if attempt["variant"] == "wrong_mask" else "MODALITY_REMOVAL_INSENSITIVE_TEXT")
                if attempt["variant"] == "modality_removal" and candidate["silver_observation"]["terrain_support"] == "supports":
                    violations.add("REMOVED_MODALITY_SUPPORT")
            record_violations.update(violations)
            counterfactual.append({
                "variant": attempt["variant"],
                "attempt_id": attempt["attempt_id"],
                "candidate_id": None if candidate is None else candidate["candidate_id"],
                "violations": sorted(violations),
            })
        accepted = not record_violations
        canonical_candidate = regular[0] if accepted else None
        confuser_count = max(
            (len(item["silver_observation"]["possible_confusers"]) for item in regular if item is not None),
            default=0,
        )
        output.append({
            "record_id": record_id,
            "sample_id": record["sample_id"],
            "source": record["source"],
            "target_state": record["target_state"],
            "status": "auto_filter_pass" if accepted else "needs_review_or_reject",
            "accepted": accepted,
            "candidate_ids": [None if item is None else item["candidate_id"] for item in regular],
            "candidate_violations": candidate_violations,
            "core_consistent": core_consistent,
            "confuser_count": confuser_count,
            "counterfactual": counterfactual,
            "violations": sorted(record_violations),
            "canonical_candidate": canonical_candidate,
        })
    return output


def _artifact_contract(path: Path, *, count: int) -> dict[str, Any]:
    return {"path": path.name, "count": count, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def filter_silver_run(config_path: Path | str, *, output_root: Path | None = None) -> dict[str, Any]:
    context = _preflight_context(config_path, verify_source=False)
    base = context.config.output_root if output_root is None else Path(os.path.abspath(output_root))
    if output_root is not None:
        if base == Path("/tmp"):
            fail("ARGUMENT_INVALID", "Silver smoke filter output_root 不能是 /tmp 根目录")
        try:
            base.relative_to(Path("/tmp"))
        except ValueError:
            fail("ARGUMENT_INVALID", "Silver smoke filter output_root 必须位于 /tmp")
    generation_manifest, candidates, failures, plan = _validate_generation(context, output_base=base)
    results = _filter_rows(context, candidates, failures, plan)
    accepted = [row["canonical_candidate"] for row in results if row["accepted"]]
    target = base / "filter"
    with AtomicArtifactDirectory(target) as writer:
        writer.write_jsonl("record_results.jsonl", results)
        writer.write_jsonl("accepted_candidates.jsonl", accepted)
        results_path = writer.path("record_results.jsonl")
        accepted_path = writer.path("accepted_candidates.jsonl")
        manifest = {
            "schema_version": FILTER_SCHEMA,
            "status": "completed",
            "formal_run": generation_manifest["formal_run"],
            "generation_manifest_sha256": sha256_file(base / "generation/generation_manifest.json"),
            "generation_protocol_sha256": generation_manifest["protocol"]["protocol_sha256"],
            "record_count": len(results),
            "passed_count": sum(row["accepted"] for row in results),
            "rejected_or_review_count": sum(not row["accepted"] for row in results),
            "violation_counts": dict(sorted(Counter(code for row in results for code in row["violations"]).items())),
            "record_results": _artifact_contract(results_path, count=len(results)),
            "accepted_candidates": _artifact_contract(accepted_path, count=len(accepted)),
            "automatic_filter_completed": True,
            "silver_accepted": False,
            "expert_review_completed": False,
            "oa_grounded_eval": False,
            "gate_c_evaluated": False,
        }
        writer.write_json("filter_manifest.json", manifest)
        published = writer.publish()
    return {
        "root": str(published), "record_count": len(results),
        "passed_count": manifest["passed_count"],
        "rejected_or_review_count": manifest["rejected_or_review_count"],
        "manifest_sha256": sha256_file(published / "filter_manifest.json"),
    }


def _validate_filter(
    context: SilverContext, *, output_base: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = context.config.output_root if output_base is None else Path(output_base)
    generation_manifest, candidates, failures, plan = _validate_generation(context, output_base=base)
    root = _regular_directory(base / "filter", location="Silver filter root")
    _exact_regular_files(
        root,
        {"filter_manifest.json", "record_results.jsonl", "accepted_candidates.jsonl"},
        code="SILVER_FILTER_INVALID",
    )
    manifest_path = _regular_file(root / "filter_manifest.json", location="filter manifest")
    manifest = read_json(manifest_path)
    required = {
        "schema_version", "status", "formal_run", "generation_manifest_sha256",
        "generation_protocol_sha256", "record_count", "passed_count", "rejected_or_review_count",
        "violation_counts", "record_results", "accepted_candidates", "automatic_filter_completed",
        "silver_accepted", "expert_review_completed", "oa_grounded_eval", "gate_c_evaluated",
    }
    if not isinstance(manifest, dict) or set(manifest) != required or manifest.get("schema_version") != FILTER_SCHEMA or manifest.get("status") != "completed":
        fail("SILVER_FILTER_INVALID", "filter manifest 合同非法")
    if (
        manifest["formal_run"] is not generation_manifest["formal_run"]
        or manifest["generation_manifest_sha256"] != sha256_file(base / "generation/generation_manifest.json")
        or manifest["generation_protocol_sha256"] != generation_manifest["protocol"]["protocol_sha256"]
        or manifest["automatic_filter_completed"] is not True
        or any(manifest[field] is not False for field in ("silver_accepted", "expert_review_completed", "oa_grounded_eval", "gate_c_evaluated"))
    ):
        fail("SILVER_FILTER_INVALID", "filter 身份或状态提前扩张")
    results_path = _regular_file(root / "record_results.jsonl", location="filter results")
    accepted_path = _regular_file(root / "accepted_candidates.jsonl", location="filter accepted")
    results = read_jsonl(results_path)
    accepted = [validate_silver_candidate(row) for row in read_jsonl(accepted_path)]
    expected_results = _filter_rows(context, candidates, failures, plan)
    expected_accepted = [row["canonical_candidate"] for row in expected_results if row["accepted"]]
    if results != expected_results or accepted != expected_accepted:
        fail("SILVER_FILTER_INVALID", "filter 结果不是当前规则的确定性输出")
    expected = {
        "record_count": len(results),
        "passed_count": sum(row["accepted"] for row in results),
        "rejected_or_review_count": sum(not row["accepted"] for row in results),
        "violation_counts": dict(sorted(Counter(code for row in results for code in row["violations"]).items())),
        "record_results": _artifact_contract(results_path, count=len(results)),
        "accepted_candidates": _artifact_contract(accepted_path, count=len(accepted)),
    }
    if any(manifest[name] != value for name, value in expected.items()):
        fail("SILVER_FILTER_INVALID", "filter manifest 统计或 artifact identity 不一致")
    return manifest, results, accepted, candidates


def _review_priority(row: Mapping[str, Any], *, seed: int) -> tuple[Any, ...]:
    observation = row["canonical_candidate"]
    sufficiency = None if observation is None else observation["silver_observation"]["evidence_sufficiency"]
    sufficiency_rank = {"insufficient": 0, "limited": 1, "sufficient": 2, None: 0}[sufficiency]
    return (
        int(row["accepted"]),
        -len(row["violations"]),
        -int(row.get("confuser_count", 0)),
        sufficiency_rank,
        stable_hash(seed, "review", row["source"], row["target_state"], row["sample_id"]),
    )


def _select_review_rows(results: Sequence[Mapping[str, Any]], *, count: int, seed: int) -> list[Mapping[str, Any]]:
    sources = sorted({str(row["source"]) for row in results})
    base_quota, remainder = divmod(count, len(sources))
    selected: list[Mapping[str, Any]] = []
    for source_index, source in enumerate(sources):
        quota = base_quota + int(source_index < remainder)
        group = [row for row in results if row["source"] == source]
        no_target = sorted((row for row in group if row["target_state"] == "no_target"), key=lambda row: _review_priority(row, seed=seed))
        target = sorted((row for row in group if row["target_state"] == "target_present"), key=lambda row: _review_priority(row, seed=seed))
        no_target_quota = min(len(no_target), round(quota * len(no_target) / len(group)))
        chosen = no_target[:no_target_quota] + target[: quota - no_target_quota]
        if len(chosen) < quota:
            used = {row["record_id"] for row in chosen}
            remainder_rows = sorted((row for row in group if row["record_id"] not in used), key=lambda row: _review_priority(row, seed=seed))
            chosen.extend(remainder_rows[: quota - len(chosen)])
        selected.extend(chosen)
    selected.sort(key=lambda row: _review_priority(row, seed=seed))
    if len(selected) != count or len({row["record_id"] for row in selected}) != count:
        fail("REVIEW_SELECTION_INVALID", f"review queue 未精确选择 {count} 条")
    return selected


def _review_queue(
    context: SilverContext,
    results: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    selected = _select_review_rows(results, count=count, seed=context.config.review_seed)
    records = {row["record_id"]: row for row in context.corpus_records}
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    queue: list[dict[str, Any]] = []
    for ordinal, result in enumerate(selected):
        record = records[result["record_id"]]
        candidate_rows = [candidates_by_id[candidate_id] for candidate_id in result["candidate_ids"] if candidate_id is not None]
        identity_body = {"record_id": result["record_id"], "candidate_ids": result["candidate_ids"], "filter_violations": result["violations"]}
        queue.append({
            "schema_version": REVIEW_ITEM_SCHEMA,
            "ordinal": ordinal,
            "review_id": sha256_text(canonical_json(identity_body)),
            "record_id": result["record_id"],
            "sample_id": result["sample_id"],
            "source": result["source"],
            "target_state": result["target_state"],
            "assets": copy.deepcopy(record["assets"]),
            "program_facts": copy.deepcopy(record["program_facts"]),
            "filter_result": copy.deepcopy(result),
            "candidates": copy.deepcopy(candidate_rows),
            "human_review": {
                "status": "pending", "reviewer_id": None, "decision": None,
                "selected_candidate_id": None, "corrected_observation": None, "notes": None,
                "oa_grounded_eval_candidate": False, "case_kb_eligible": False,
                "adapter_training_eligible": False,
            },
        })
    return queue


def prepare_review_queue_run(
    config_path: Path | str,
    *,
    review_count: int,
) -> dict[str, Any]:
    context = _preflight_context(config_path, verify_source=False)
    if review_count != context.config.review_count:
        fail("ARGUMENT_INVALID", f"正式 review-count 必须是 {context.config.review_count}")
    filter_manifest, results, _, candidates = _validate_filter(context)
    if filter_manifest["formal_run"] is not True:
        fail("REVIEW_SELECTION_INVALID", "正式 review queue 只能来自 full formal generation")
    queue = _review_queue(context, results, candidates, count=review_count)
    target = context.config.output_root / "review"
    with AtomicArtifactDirectory(target) as writer:
        writer.write_jsonl("review_queue.jsonl", queue)
        queue_path = writer.path("review_queue.jsonl")
        manifest = {
            "schema_version": REVIEW_SCHEMA,
            "status": "prepared_not_reviewed",
            "filter_manifest_sha256": sha256_file(context.config.output_root / "filter/filter_manifest.json"),
            "selection_algorithm": context.config.review_algorithm,
            "seed": context.config.review_seed,
            "record_count": len(queue),
            "source_counts": dict(sorted(Counter(row["source"] for row in queue).items())),
            "target_state_counts": dict(sorted(Counter(row["target_state"] for row in queue).items())),
            "queue": _artifact_contract(queue_path, count=len(queue)),
            "expert_review_completed": False,
            "silver_accepted": False,
            "oa_grounded_eval": False,
        }
        writer.write_json("review_manifest.json", manifest)
        published = writer.publish()
    return {"root": str(published), "record_count": len(queue), "manifest_sha256": sha256_file(published / "review_manifest.json")}


def _validate_review(context: SilverContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filter_manifest, results, _, candidates = _validate_filter(context)
    root = _regular_directory(context.config.output_root / "review", location="Silver review root")
    _exact_regular_files(
        root, {"review_manifest.json", "review_queue.jsonl"}, code="REVIEW_QUEUE_INVALID",
    )
    manifest_path = _regular_file(root / "review_manifest.json", location="review manifest")
    manifest = read_json(manifest_path)
    required = {
        "schema_version", "status", "filter_manifest_sha256", "selection_algorithm", "seed",
        "record_count", "source_counts", "target_state_counts", "queue",
        "expert_review_completed", "silver_accepted", "oa_grounded_eval",
    }
    if not isinstance(manifest, dict) or set(manifest) != required or manifest.get("schema_version") != REVIEW_SCHEMA or manifest.get("status") != "prepared_not_reviewed":
        fail("REVIEW_QUEUE_INVALID", "review manifest 合同非法")
    if (
        filter_manifest["formal_run"] is not True
        or manifest["filter_manifest_sha256"] != sha256_file(context.config.output_root / "filter/filter_manifest.json")
        or manifest["selection_algorithm"] != context.config.review_algorithm
        or manifest["seed"] != context.config.review_seed
        or any(manifest[field] is not False for field in ("expert_review_completed", "silver_accepted", "oa_grounded_eval"))
    ):
        fail("REVIEW_QUEUE_INVALID", "review identity 或状态提前扩张")
    queue_path = _regular_file(root / "review_queue.jsonl", location="review queue")
    queue = read_jsonl(queue_path)
    expected_queue = _review_queue(context, results, candidates, count=context.config.review_count)
    if queue != expected_queue:
        fail("REVIEW_QUEUE_INVALID", "review queue 不是确定性构建结果")
    for ordinal, row in enumerate(queue):
        if (
            set(row) != {"schema_version", "ordinal", "review_id", "record_id", "sample_id", "source", "target_state", "assets", "program_facts", "filter_result", "candidates", "human_review"}
            or row["schema_version"] != REVIEW_ITEM_SCHEMA or row["ordinal"] != ordinal
            or not isinstance(row["human_review"], dict) or row["human_review"].get("status") != "pending"
        ):
            fail("REVIEW_QUEUE_INVALID", f"review_queue[{ordinal}] 合同非法")
    expected_manifest_values = {
        "record_count": len(queue),
        "source_counts": dict(sorted(Counter(row["source"] for row in queue).items())),
        "target_state_counts": dict(sorted(Counter(row["target_state"] for row in queue).items())),
        "queue": _artifact_contract(queue_path, count=len(queue)),
    }
    if any(manifest[name] != value for name, value in expected_manifest_values.items()):
        fail("REVIEW_QUEUE_INVALID", "review manifest 统计或 artifact identity 不一致")
    candidate_ids = {row["candidate_id"] for row in candidates}
    if any(candidate["candidate_id"] not in candidate_ids for row in queue for candidate in row["candidates"]):
        fail("REVIEW_QUEUE_INVALID", "review queue 引用未知 candidate")
    return manifest, queue


def validate_silver_outputs(config_path: Path | str) -> dict[str, Any]:
    context = _preflight_context(config_path, verify_source=False)
    generation, candidates, failures, plan = _validate_generation(context)
    result: dict[str, Any] = {
        "valid": True,
        "formal_run": generation["formal_run"],
        "generation": {
            "attempt_count": len(plan), "candidate_count": len(candidates),
            "failure_count": len(failures),
            "manifest_sha256": sha256_file(context.config.output_root / "generation/generation_manifest.json"),
        },
        "filter": None,
        "review": None,
        "expert_review_completed": False,
        "silver_accepted": False,
        "oa_grounded_eval": False,
        "gate_c_evaluated": False,
    }
    if (context.config.output_root / "filter").exists():
        manifest, rows, accepted, _ = _validate_filter(context)
        result["filter"] = {
            "record_count": len(rows), "passed_count": len(accepted),
            "manifest_sha256": sha256_file(context.config.output_root / "filter/filter_manifest.json"),
            "automatic_filter_completed": manifest["automatic_filter_completed"],
        }
    if (context.config.output_root / "review").exists():
        manifest, queue = _validate_review(context)
        result["review"] = {
            "record_count": len(queue),
            "status": manifest["status"],
            "manifest_sha256": sha256_file(context.config.output_root / "review/review_manifest.json"),
        }
    return result
