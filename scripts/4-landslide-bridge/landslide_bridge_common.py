#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Landslide Bridge M2 公共协议、路径、几何、证据和审核统计工具。

运行方式：内部公共模块，不作为独立入口运行。
写入行为：仅由 4-1 到 4-6 显式调用时写入派生 benchmark。
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml


SCHEMA_VERSION = "qpsalm_landslide_region_description_v1"
BUILDER_VERSION = "landslide_bridge_m2_v7_expert_review_replay_bound"
EVALUATION_GATE_PROTOCOL = "landslide_bridge_evaluation_gate_v2"
EXPERT_ARTIFACT_BINDING_PROTOCOL = (
    "landslide_bridge_expert_artifact_binding_v1_review_sources_and_outputs"
)
EXPERT_REVIEW_REPLAY_PROTOCOL = (
    "landslide_bridge_expert_review_replay_v1_exact_semantic_projection"
)
REVIEW_DECISIONS = {"accept", "revise", "reject"}
EVALUATION_GATE_THRESHOLDS = (
    "no_target_rejection",
    "unsupported_claim_rate",
    "unavailable_unsupported_claim_rate",
    "unsupported_claim_rate_noninferiority",
    "expert_fact_score",
    "target_status_macro_f1",
    "present_recall",
    "absent_recall",
    "false_description_rate",
    "false_rejection_rate",
)
EVALUATION_GATE_COUNTERFACTUAL_MODES = (
    "shuffled_mask",
    "region_swap",
    "cross_parent_region_swap",
    "cross_parent_modality_swap",
    "modality_removal",
)
EVALUATION_GATE_SCIENTIFIC_PROTOCOLS = {
    "erfs_rubric": "qpsalm_erfs_eight_family_parent_macro_v1",
    "claim_inventory": "qpsalm_structured_claim_inventory_v1",
    "retrieval_scorer": "qpsalm_same_image_region_retrieval_v2_parent_ranked",
    "region_protocol_reporting": "separate_assisted_vision_only_v1",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DATASETS_ROOT = Path(os.environ.get("PAPER7_DATASETS_ROOT") or WORKSPACE_ROOT / "datasets").resolve(strict=False)
BENCHMARK_ROOT = Path(os.environ.get("PAPER7_BENCHMARK_ROOT") or WORKSPACE_ROOT / "benchmark").resolve(strict=False)

REGION_FIELD_VALUES = {
    "location": {
        "upper_left", "upper_center", "upper_right", "center_left", "center",
        "center_right", "lower_left", "lower_center", "lower_right", "distributed",
        "unknown", "unavailable",
    },
    "size_class": {"tiny", "small", "medium", "large", "extensive", "unknown", "unavailable"},
    "shape": {"compact", "elongated", "branching", "fragmented", "irregular", "unknown", "unavailable"},
    "elongation": {"low", "moderate", "high", "unknown", "unavailable"},
    "compactness": {"compact", "moderate", "dispersed", "unknown", "unavailable"},
    "fragmentation": {
        "single", "few_components", "many_components", "highly_fragmented",
        "unknown", "unavailable",
    },
}
EVIDENCE_SUPPORT_FIELDS = {"terrain_support", "sar_support", "deformation_support"}
EVIDENCE_SUPPORT_VALUES = {
    "supports", "does_not_support", "insufficient_evidence", "unknown", "unavailable",
}
EVIDENCE_SUFFICIENCY_VALUES = {"sufficient", "partial", "insufficient", "unavailable"}
ABSENT_EVIDENCE_SUPPORT_VALUES = {"insufficient_evidence", "unavailable"}
ABSENT_EVIDENCE_SUFFICIENCY_VALUES = {"insufficient", "unavailable"}


def evaluation_gate_scientific_template() -> dict[str, Any]:
    """Return the human-completed scientific part of the frozen Pilot gate."""
    return {
        **EVALUATION_GATE_SCIENTIFIC_PROTOCOLS,
        "bootstrap": {
            "unit": "parent",
            "confidence": 0.95,
            "samples": 10000,
            "seed": 42,
        },
        # Pilot 完成后由人工填写；prepare 阶段不得猜测正式反事实覆盖门槛。
        "counterfactual_minimum_effective_parents": {
            mode: None for mode in EVALUATION_GATE_COUNTERFACTUAL_MODES
        },
    }


def validate_frozen_evaluation_gate_science(gate: Any) -> list[str]:
    """Validate frozen thresholds/statistics without inferring Pilot outcomes."""
    errors: list[str] = []
    if not isinstance(gate, dict):
        return ["evaluation gate 必须是 JSON object"]
    if gate.get("protocol") != EVALUATION_GATE_PROTOCOL:
        errors.append(f"evaluation gate protocol 不是 {EVALUATION_GATE_PROTOCOL}")
    if gate.get("frozen") is not True or gate.get("status") != "frozen_after_pilot":
        errors.append("evaluation gate 必须由用户显式冻结为 frozen_after_pilot")
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("evaluation gate thresholds 必须是 object")
    else:
        for key in EVALUATION_GATE_THRESHOLDS:
            value = thresholds.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(f"evaluation gate 阈值必须位于 [0,1]: {key}={value!r}")
    scientific = gate.get("scientific_protocol")
    if not isinstance(scientific, dict):
        errors.append("evaluation gate 缺少 scientific_protocol")
        return errors
    for key, expected in EVALUATION_GATE_SCIENTIFIC_PROTOCOLS.items():
        if scientific.get(key) != expected:
            errors.append(
                f"evaluation gate scientific_protocol.{key} 非法: "
                f"expected={expected!r} observed={scientific.get(key)!r}"
            )
    bootstrap = scientific.get("bootstrap")
    if not isinstance(bootstrap, dict):
        errors.append("evaluation gate scientific_protocol.bootstrap 必须是 object")
    else:
        if bootstrap.get("unit") != "parent":
            errors.append("evaluation gate bootstrap.unit 必须为 parent")
        if bootstrap.get("samples") != 10000:
            errors.append("evaluation gate bootstrap.samples 必须为 10000")
        confidence = bootstrap.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isclose(float(confidence), 0.95, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            errors.append("evaluation gate bootstrap.confidence 必须为 0.95")
        seed = bootstrap.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            errors.append("evaluation gate bootstrap.seed 必须为非负整数")
    minimums = scientific.get("counterfactual_minimum_effective_parents")
    if not isinstance(minimums, dict):
        errors.append(
            "evaluation gate scientific_protocol.counterfactual_minimum_effective_parents "
            "必须是 object"
        )
    else:
        if set(minimums) != set(EVALUATION_GATE_COUNTERFACTUAL_MODES):
            errors.append(
                "evaluation gate counterfactual modes 必须精确匹配正式五种模式"
            )
        for mode in EVALUATION_GATE_COUNTERFACTUAL_MODES:
            value = minimums.get(mode)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(
                    "evaluation gate counterfactual minimum 必须为正整数: "
                    f"{mode}={value!r}"
                )
    return errors


def validate_bridge_structured_target(
    target: Any,
    *,
    expected_target_status: str | None = None,
) -> list[str]:
    """Validate the reviewable subset of qpsalm_description_output_v1."""
    errors: list[str] = []
    if not isinstance(target, dict):
        return ["structured target 必须是 JSON object"]
    status = target.get("target_status")
    if status not in {"present", "absent", "uncertain"}:
        errors.append("target_status 非法或缺失")
    if expected_target_status is not None and status != expected_target_status:
        errors.append(
            f"target_status 不得改变 GT 状态: expected={expected_target_status} actual={status}"
        )
    region = target.get("region")
    if not isinstance(region, dict):
        errors.append("region 必须是 object")
    else:
        for field, allowed in REGION_FIELD_VALUES.items():
            if region.get(field) not in allowed:
                errors.append(f"region.{field} 非法或缺失")
    evidence = target.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence 必须是 object")
    else:
        for field in ("surface_observation", "surrounding_context"):
            if not isinstance(evidence.get(field), str) or not evidence[field].strip():
                errors.append(f"evidence.{field} 必须是非空字符串")
        for field in EVIDENCE_SUPPORT_FIELDS:
            if evidence.get(field) not in EVIDENCE_SUPPORT_VALUES:
                errors.append(f"evidence.{field} 非法或缺失")
        if evidence.get("evidence_sufficiency") not in EVIDENCE_SUFFICIENCY_VALUES:
            errors.append("evidence.evidence_sufficiency 非法或缺失")
    if status == "absent" and isinstance(region, dict) and isinstance(evidence, dict):
        # no-target 没有可定位区域；审核修订不得把场景先验写成区域事实。
        for field in REGION_FIELD_VALUES:
            if region.get(field) != "unavailable":
                errors.append(f"absent target 要求 region.{field}=unavailable")
        for field in EVIDENCE_SUPPORT_FIELDS:
            if evidence.get(field) not in ABSENT_EVIDENCE_SUPPORT_VALUES:
                errors.append(
                    f"absent target 的 evidence.{field} 只能是 "
                    "insufficient_evidence/unavailable"
                )
        if evidence.get("evidence_sufficiency") not in ABSENT_EVIDENCE_SUFFICIENCY_VALUES:
            errors.append(
                "absent target 的 evidence.evidence_sufficiency 只能是 "
                "insufficient/unavailable"
            )
    return errors


def flatten_bridge_structured_target(target: dict[str, Any]) -> dict[str, str]:
    """Return ontology fields used for per-field reviewer agreement."""
    region = target.get("region") or {}
    evidence = target.get("evidence") or {}
    result = {"target_status": str(target.get("target_status") or "<missing>")}
    result.update({f"region.{field}": str(region.get(field) or "<missing>") for field in REGION_FIELD_VALUES})
    result.update({
        f"evidence.{field}": str(evidence.get(field) or "<missing>")
        for field in (
            *sorted(EVIDENCE_SUPPORT_FIELDS),
            "evidence_sufficiency", "surface_observation", "surrounding_context",
        )
    })
    return result


def resolve_project_path(ref: str | Path) -> Path:
    path = Path(ref).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if path.parts and path.parts[0] == "benchmark":
        return BENCHMARK_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    if path.parts and path.parts[0] == "datasets":
        return DATASETS_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def to_project_ref(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute():
        return source.as_posix()
    source = source.resolve(strict=False)
    for logical, root in (("benchmark", BENCHMARK_ROOT), ("datasets", DATASETS_ROOT)):
        try:
            return (Path(logical) / source.relative_to(root)).as_posix()
        except ValueError:
            pass
    try:
        return source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return source.as_posix()


def source_benchmark_dir(mode: str, value: str | Path | None = None) -> Path:
    return resolve_project_path(value) if value else BENCHMARK_ROOT / f"multisource_landslide_v2_{mode}"


def bridge_dir(mode: str, value: str | Path | None = None) -> Path:
    return resolve_project_path(value) if value else BENCHMARK_ROOT / f"landslide_region_description_v1_{mode}"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path or "configs/landslide_bridge_v1.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("version") != "landslide_bridge_v1":
        raise ValueError(f"Bridge config 版本不正确: {config_path}")
    return payload


def bridge_parent_from_landslide_v2(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one Landslide V2 parent row into the Bridge parent contract.

    Landslide V2 parent indexes use ``sample_id`` as their primary key, while
    referring rows use ``parent_sample_id``. Bridge records consistently expose
    the latter name. Keeping this conversion at the boundary prevents derived
    referring records from being mistaken for parent records.
    """
    if not isinstance(row, dict):
        raise TypeError(f"Landslide V2 parent 必须是 object，实际为 {type(row).__name__}")
    if row.get("schema_version") != "multisource_landslide_schema_v2":
        raise ValueError(
            "Bridge 仅接受 multisource_landslide_schema_v2 parent，"
            f"实际为 {row.get('schema_version')!r}"
        )
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("Landslide V2 parent 缺少非空 sample_id")
    existing_parent_id = str(row.get("parent_sample_id") or "").strip()
    if existing_parent_id and existing_parent_id != sample_id:
        raise ValueError(
            "Landslide V2 parent 的 sample_id 与 parent_sample_id 冲突: "
            f"sample_id={sample_id!r} parent_sample_id={existing_parent_id!r}"
        )
    if row.get("source_level") != "patch" or row.get("supervision") != "mask":
        raise ValueError(
            f"Bridge parent 必须是 patch/mask 记录: {sample_id} "
            f"source_level={row.get('source_level')!r} supervision={row.get('supervision')!r}"
        )
    required = {
        "split": row.get("split"),
        "dataset_name": row.get("dataset_name"),
        "mask": row.get("mask"),
        "modalities": row.get("modalities"),
        "spatial": row.get("spatial"),
    }
    missing = [name for name, value in required.items() if value in (None, "", {})]
    if missing:
        raise ValueError(f"Landslide V2 parent {sample_id} 缺少 Bridge 必需字段: {missing}")
    if not isinstance(row["mask"], dict) or not row["mask"].get("path"):
        raise ValueError(f"Landslide V2 parent {sample_id} 缺少已物化 mask.path")
    if not isinstance(row["modalities"], dict):
        raise ValueError(f"Landslide V2 parent {sample_id} 的 modalities 必须是 object")

    parent = deepcopy(row)
    parent["parent_sample_id"] = sample_id
    return parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: 非法 JSONL: {exc}") from exc
    return rows


def _parse_review_json_field(
    value: Any,
    field: str,
    path: Path,
    row_number: int,
) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{row_number}: {field} 不是合法 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}:{row_number}: {field} 必须是 JSON object")
    return parsed


def read_review_file(
    path_ref: str | Path,
    expected_reviewer: str | None = None,
) -> list[dict[str, Any]]:
    """Read and normalize one immutable reviewer or arbitration source."""
    path = resolve_project_path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"审核文件不存在: {path_ref} -> {path}")
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    first_row = 2 if path.suffix.casefold() == ".csv" else 1
    for row_number, source in enumerate(rows, start=first_row):
        row = dict(source)
        decision = str(row.get("decision") or "").strip().casefold()
        if decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"{path}:{row_number}: decision 必须为 accept/revise/reject，"
                f"当前={decision!r}"
            )
        reviewer_id = str(row.get("reviewer_id") or "").strip()
        if not reviewer_id:
            raise ValueError(f"{path}:{row_number}: reviewer_id 不能为空")
        if expected_reviewer and reviewer_id != expected_reviewer:
            raise ValueError(
                f"{path}:{row_number}: reviewer_id 应为 {expected_reviewer}"
            )
        row["reviewer_id"] = reviewer_id
        row["decision"] = decision
        row["corrected_structured_targets"] = _parse_review_json_field(
            row.get("corrected_structured_targets"),
            "corrected_structured_targets",
            path,
            row_number,
        )
        row["revised_summary"] = str(row.get("revised_summary") or "").strip()
        if decision == "revise" and (
            row["corrected_structured_targets"] is None
            or not row["revised_summary"]
        ):
            raise ValueError(
                f"{path}:{row_number}: revise 必须填写 "
                "corrected_structured_targets 和 revised_summary"
            )
        normalized.append(row)
    return normalized


def unique_review_rows(
    rows: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Index review rows while rejecting missing or duplicated item IDs."""
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("review_item_id") or "").strip()
        if not item_id:
            raise ValueError(f"{label}: review_item_id 不能为空")
        if item_id in result:
            raise ValueError(f"{label}: review_item_id 重复: {item_id}")
        result[item_id] = row
    return result


def review_revisions_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Require exact structured and summary agreement for double revisions."""
    return (
        left["corrected_structured_targets"]
        == right["corrected_structured_targets"]
        and left["revised_summary"] == right["revised_summary"]
    )


def disputed_review_item_ids(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> set[str]:
    """Return decision or exact-revision disagreements requiring arbitration."""
    if set(left) != set(right):
        raise ValueError("两份 reviewer item 集合不一致")
    return {
        item_id
        for item_id in left
        if not (
            left[item_id]["decision"] == right[item_id]["decision"]
            and (
                left[item_id]["decision"] != "revise"
                or review_revisions_match(left[item_id], right[item_id])
            )
        )
    }


def validate_arbitration_usage(
    arbitration: dict[str, dict[str, Any]],
    *,
    selected_ids: set[str],
    disputed_ids: set[str],
    reviewer_ids: set[str],
) -> None:
    """Reject unknown, non-disputed, or non-independent arbitration records."""
    unexpected = set(arbitration) - selected_ids
    if unexpected:
        raise ValueError(
            f"arbitration 包含未知 review item: {sorted(unexpected)[:3]}"
        )
    arbitration_reviewer_ids = {
        str(row["reviewer_id"]) for row in arbitration.values()
    }
    overlapping = reviewer_ids & arbitration_reviewer_ids
    if overlapping:
        raise ValueError(
            "仲裁者必须独立于 reviewer_1/reviewer_2: "
            f"{sorted(overlapping)}"
        )
    unused = set(arbitration) - disputed_ids
    if unused:
        raise ValueError(
            "arbitration 只能覆盖双审分歧项，存在未使用记录: "
            f"{sorted(unused)[:3]}"
        )


def _resolved_expert_record(
    record: dict[str, Any],
    decision: str,
    response: dict[str, Any],
    status: str,
    reviewer_responses: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if decision == "reject":
        return None
    if decision == "accept":
        structured = deepcopy(record["candidate"]["structured_output"])
        summary = str(record["candidate"]["summary"])
    else:
        structured = deepcopy(response["corrected_structured_targets"])
        summary = str(response["revised_summary"])
    target_errors = validate_bridge_structured_target(
        structured,
        expected_target_status=str(record["target_status"]),
    )
    if target_errors:
        raise ValueError(
            f"{record['bridge_record_id']}: expert structured target 非法: "
            f"{target_errors}"
        )
    if not summary.strip():
        raise ValueError(f"{record['bridge_record_id']}: expert summary 不能为空")
    if not isinstance(record.get("provenance"), dict):
        raise ValueError(f"{record['bridge_record_id']}: candidate 缺少 provenance")
    result = deepcopy(record)
    result["expert_target"] = {
        "structured_output": structured,
        "summary": summary,
        "language": "en",
        "source": "double_reviewed_pilot",
    }
    result["review"] = {
        "status": status,
        "final_decision": decision,
        "reviewer_responses": deepcopy(list(reviewer_responses)),
    }
    result["provenance"]["expert_review_merger"] = BUILDER_VERSION
    return result


def replay_expert_review_merge(
    *,
    candidates: Sequence[dict[str, Any]],
    selection: Sequence[dict[str, Any]],
    reviewer_1: dict[str, dict[str, Any]],
    reviewer_2: dict[str, dict[str, Any]],
    arbitration: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Deterministically replay raw reviews into expert and pending projections.

    This function is shared by the publisher and the independent validator so
    frozen expert truth is derived from the bound human sources, not merely
    accepted because all files carry self-consistent hashes.
    """
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidates:
        record_id = str(row.get("bridge_record_id") or "").strip()
        if not record_id:
            raise ValueError("candidate bridge_record_id 不能为空")
        if record_id in candidate_by_id:
            raise ValueError(f"candidate bridge_record_id 重复: {record_id}")
        candidate_by_id[record_id] = row
    selection_by_item = unique_review_rows(selection, "review_selection")
    selected_ids = set(selection_by_item)
    if set(reviewer_1) != selected_ids or set(reviewer_2) != selected_ids:
        raise ValueError(
            "两份 reviewer 文件必须恰好覆盖 review_selection；"
            f"selection={len(selected_ids)} reviewer_1={len(reviewer_1)} "
            f"reviewer_2={len(reviewer_2)}"
        )
    reviewer_ids = {
        str(row["reviewer_id"])
        for rows in (reviewer_1.values(), reviewer_2.values())
        for row in rows
    }
    disputed_ids = disputed_review_item_ids(reviewer_1, reviewer_2)
    validate_arbitration_usage(
        arbitration,
        selected_ids=selected_ids,
        disputed_ids=disputed_ids,
        reviewer_ids=reviewer_ids,
    )

    expert: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    final_decisions: Counter[str] = Counter()
    for item_id in sorted(selected_ids):
        item = selection_by_item[item_id]
        record_id = str(item.get("bridge_record_id") or "").strip()
        if record_id not in candidate_by_id:
            raise ValueError(
                f"review_selection 引用未知 candidate: {item_id} -> {record_id!r}"
            )
        record = candidate_by_id[record_id]
        first, second = reviewer_1[item_id], reviewer_2[item_id]
        responses = [first, second]
        same = first["decision"] == second["decision"]
        if same and first["decision"] == "revise":
            same = review_revisions_match(first, second)
        if same:
            decision, response = first["decision"], first
            status = {
                "accept": "accepted",
                "revise": "revised",
                "reject": "rejected",
            }[decision]
        elif item_id in arbitration:
            response = arbitration[item_id]
            decision = response["decision"]
            responses.append(response)
            status = "arbitrated"
        else:
            pending.append({
                **deepcopy(item),
                "status": "needs_arbitration",
                "reviewer_responses": deepcopy(responses),
            })
            continue
        final_decisions[decision] += 1
        resolved = _resolved_expert_record(
            record,
            decision,
            response,
            status,
            responses,
        )
        if resolved is not None:
            expert.append(resolved)

    expert.sort(
        key=lambda row: (
            row["split"],
            row["parent_sample_id"],
            row["bridge_record_id"],
        )
    )
    pending.sort(
        key=lambda row: (
            row["split"],
            row["parent_sample_id"],
            row["review_item_id"],
        )
    )
    return {
        "protocol": EXPERT_REVIEW_REPLAY_PROTOCOL,
        "expert": expert,
        "pending": pending,
        "final_decisions": dict(sorted(final_decisions.items())),
        "disputed_review_item_ids": sorted(disputed_ids),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
    )
    return len(values)


def ensure_writable(path: Path, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"输出已存在，请使用 --overwrite: {path}")


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", prefix.casefold()).strip("_")
    return f"{safe}_{stable_hash(*parts)[:length]}"


def safe_slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not result:
        raise ValueError(f"无法生成 slug: {value!r}")
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_artifact_binding(
    path: str | Path,
    *,
    records: int | None = None,
) -> dict[str, Any]:
    """Bind one immutable review source or expert output to its live bytes."""
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact binding 文件不存在: {path} -> {resolved}")
    result: dict[str, Any] = {
        "path": to_project_ref(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if records is not None:
        if isinstance(records, bool) or not isinstance(records, int) or records < 0:
            raise ValueError(f"artifact binding records 非法: {records!r}")
        result["records"] = records
    return result


def validate_file_artifact_binding(
    binding: Any,
    *,
    label: str,
    expected_path: str | Path | None = None,
    expected_records: int | None = None,
) -> list[str]:
    """Replay a recorded file binding without trusting a cached validation report."""
    errors: list[str] = []
    if not isinstance(binding, dict):
        return [f"{label} artifact binding 必须是 object"]
    path_ref = binding.get("path")
    if not isinstance(path_ref, str) or not path_ref.strip():
        return [f"{label} artifact binding 缺少 path"]
    path = resolve_project_path(path_ref)
    if expected_path is not None:
        expected = resolve_project_path(expected_path)
        if path != expected:
            errors.append(
                f"{label} artifact path 不匹配: expected={expected} observed={path}"
            )
    if not path.is_file():
        errors.append(f"{label} artifact 文件不存在: {path_ref} -> {path}")
        return errors
    expected_sha = binding.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        errors.append(f"{label} artifact sha256 非法")
    elif sha256_file(path) != expected_sha:
        errors.append(f"{label} artifact hash 漂移: {path_ref}")
    expected_bytes = binding.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        errors.append(f"{label} artifact bytes 非法")
    elif path.stat().st_size != expected_bytes:
        errors.append(f"{label} artifact bytes 漂移: {path_ref}")
    if expected_records is not None:
        observed_records = binding.get("records")
        if observed_records != expected_records:
            errors.append(
                f"{label} artifact records 不匹配: "
                f"expected={expected_records} observed={observed_records!r}"
            )
    return errors


def load_array(path_ref: str) -> np.ndarray:
    path = resolve_project_path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"数组不存在: {path_ref} -> {path}")
    array = np.load(path)
    return np.asarray(array)


def binary_mask(path_ref: str) -> np.ndarray:
    array = load_array(path_ref)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"mask 必须是 HxW 或 1xHxW: {path_ref} shape={array.shape}")
    return (np.nan_to_num(array) > 0).astype(np.uint8)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def modality_family_combo(parent: dict[str, Any]) -> str:
    families = sorted({
        str(item.get("family"))
        for item in parent.get("modalities", {}).values()
        if item.get("available", True)
    })
    return "+".join(families) if families else "none"


def valid_canvas(parent: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    spec = parent.get("spatial", {}).get("valid_pixel_mask", {})
    path = spec.get("path") if isinstance(spec, dict) else None
    if not path:
        return np.ones(shape, dtype=np.uint8)
    valid = binary_mask(str(path))
    if valid.shape != shape:
        raise ValueError(f"parent valid canvas shape 不一致: {parent['parent_sample_id']}")
    return valid


def connected_components(mask: np.ndarray, valid: np.ndarray, min_pixels: int, min_fraction: float) -> list[np.ndarray]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label((mask > 0) & (valid > 0), structure=structure)
    threshold = max(int(min_pixels), int(round(float(valid.sum()) * float(min_fraction))))
    components: list[tuple[int, np.ndarray]] = []
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        area = int(component.sum())
        if area >= threshold:
            components.append((area, component.astype(np.uint8)))
    components.sort(key=lambda item: (-item[0], sha256_bytes(item[1].tobytes())))
    return [item[1] for item in components]


def size_class(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    if area_ratio < 0.2:
        return "large"
    return "extensive"


def area_bin(area_ratio: float) -> str:
    if area_ratio <= 0:
        return "absent"
    return size_class(area_ratio)


def geometry_from_mask(mask: np.ndarray | None, valid: np.ndarray) -> dict[str, Any]:
    valid_area = int((valid > 0).sum())
    if mask is None or not bool((mask > 0).any()):
        return {
            "area_pixels": 0,
            "valid_area_pixels": valid_area,
            "valid_area_ratio": 0.0,
            "bbox_xyxy_pixel_half_open": None,
            "centroid_xy_normalized": None,
            "location": "unavailable",
            "size_class": "unavailable",
            "shape": "unavailable",
            "elongation": "unavailable",
            "elongation_ratio": None,
            "compactness": "unavailable",
            "compactness_value": None,
            "fragmentation": "unavailable",
            "component_count": 0,
            "orientation_degrees": None,
        }
    binary = (mask > 0) & (valid > 0)
    ys, xs = np.where(binary)
    area = int(xs.size)
    height, width = binary.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    cx, cy = float(xs.mean()), float(ys.mean())
    col = min(2, int(3 * cx / max(width, 1)))
    row = min(2, int(3 * cy / max(height, 1)))
    positions = (
        ("upper_left", "upper_center", "upper_right"),
        ("center_left", "center", "center_right"),
        ("lower_left", "lower_center", "lower_right"),
    )
    coordinates = np.column_stack((xs - cx, ys - cy)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False) if area > 2 else np.eye(2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1.0e-6)
    major_index = int(np.argmax(eigenvalues))
    major, minor = float(eigenvalues[major_index]), float(eigenvalues[1 - major_index])
    elongation_ratio = float(math.sqrt(major / minor))
    vector = eigenvectors[:, major_index]
    orientation = float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))
    boundary = binary & ~ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))
    perimeter = max(float(boundary.sum()), 1.0)
    compactness_value = float(4.0 * math.pi * area / (perimeter * perimeter))
    component_count = int(ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))[1])
    elongation = "low" if elongation_ratio < 1.5 else "moderate" if elongation_ratio < 3.0 else "high"
    compactness = "compact" if compactness_value >= 0.5 else "moderate" if compactness_value >= 0.2 else "dispersed"
    fragmentation = (
        "single" if component_count == 1 else
        "few_components" if component_count <= 3 else
        "many_components" if component_count <= 8 else
        "highly_fragmented"
    )
    shape = (
        "fragmented" if component_count >= 3 else
        "elongated" if elongation_ratio >= 3.0 else
        "compact" if compactness_value >= 0.5 else
        "irregular"
    )
    return {
        "area_pixels": area,
        "valid_area_pixels": valid_area,
        "valid_area_ratio": float(area / max(valid_area, 1)),
        "bbox_xyxy_pixel_half_open": [x1, y1, x2, y2],
        "centroid_xy_normalized": [cx / max(width - 1, 1), cy / max(height - 1, 1)],
        "location": positions[row][col],
        "size_class": size_class(area / max(valid_area, 1)),
        "shape": shape,
        "elongation": elongation,
        "elongation_ratio": elongation_ratio,
        "compactness": compactness,
        "compactness_value": compactness_value,
        "fragmentation": fragmentation,
        "component_count": component_count,
        "orientation_degrees": orientation,
    }


def context_ring(mask: np.ndarray, valid: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    area = max(int((mask > 0).sum()), 1)
    equivalent_radius = math.sqrt(area / math.pi)
    evidence = config["evidence"]
    radius = int(round(equivalent_radius * float(evidence["context_ring_fraction_of_equivalent_radius"])))
    radius = max(int(evidence["context_ring_min_pixels"]), min(int(evidence["context_ring_max_pixels"]), radius))
    dilated = ndimage.binary_dilation(mask > 0, iterations=radius)
    return (dilated & ~(mask > 0) & (valid > 0)).astype(np.uint8)


def mask_digest(mask: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())


def parent_index_ref(source_dir: Path, split: str) -> str:
    return to_project_ref(source_dir / f"indexes/{split}.jsonl")


def stratified_select(rows: Sequence[dict[str, Any]], limit: int, fields: Sequence[str], seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda row: stable_hash(seed, row["parent_sample_id"]))
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "unknown")) for field in fields)
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_hash(seed, row["parent_sample_id"]))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def krippendorff_alpha_nominal(ratings: Sequence[Sequence[str | None]]) -> float | None:
    pairs_total = 0
    disagreements = 0
    counts: defaultdict[str, int] = defaultdict(int)
    total_ratings = 0
    for item in ratings:
        observed = [value for value in item if value is not None]
        for value in observed:
            counts[value] += 1
            total_ratings += 1
        for left_index in range(len(observed)):
            for right_index in range(left_index + 1, len(observed)):
                pairs_total += 1
                disagreements += observed[left_index] != observed[right_index]
    if pairs_total == 0 or total_ratings < 2:
        return None
    observed_disagreement = disagreements / pairs_total
    expected_disagreement = 1.0 - sum((count / total_ratings) ** 2 for count in counts.values())
    return float(1.0 - observed_disagreement / expected_disagreement) if expected_disagreement > 0 else 1.0


def levenshtein_distance(left: str, right: str) -> int:
    """Deterministic character edit distance for expert summary revisions."""
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + int(left_char != right_char),
            ))
        previous = current
    return previous[-1]


def expert_modification_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure final expert edits relative to the frozen rule candidate."""
    distances = []
    normalized_distances = []
    changed_fields = total_fields = 0
    by_decision: Counter[str] = Counter()
    for row in rows:
        candidate = row.get("candidate") or {}
        expert_target = row.get("expert_target") or {}
        candidate_summary = str(candidate.get("summary") or "")
        expert_summary = str(expert_target.get("summary") or "")
        distance = levenshtein_distance(candidate_summary, expert_summary)
        distances.append(distance)
        normalized_distances.append(
            distance / max(len(candidate_summary), len(expert_summary), 1)
        )
        candidate_fields = flatten_bridge_structured_target(
            candidate.get("structured_output") or {}
        )
        expert_fields = flatten_bridge_structured_target(
            expert_target.get("structured_output") or {}
        )
        fields = sorted(set(candidate_fields) | set(expert_fields))
        changed_fields += sum(
            candidate_fields.get(field) != expert_fields.get(field)
            for field in fields
        )
        total_fields += len(fields)
        by_decision[
            str((row.get("review") or {}).get("final_decision") or "unknown")
        ] += 1
    return {
        "num_expert_records": len(rows),
        "summary_mean_edit_distance_characters": (
            sum(distances) / len(distances) if distances else None
        ),
        "summary_mean_normalized_edit_distance": (
            sum(normalized_distances) / len(normalized_distances)
            if normalized_distances else None
        ),
        "structured_claim_fields_changed": changed_fields,
        "structured_claim_fields_total": total_fields,
        "factual_claim_modification_rate": (
            changed_fields / total_fields if total_fields else None
        ),
        "expert_records_by_final_decision": dict(sorted(by_decision.items())),
    }


def expert_review_report_statistics(
    *,
    candidates: Sequence[dict[str, Any]],
    selection: Sequence[dict[str, Any]],
    reviewer_1: dict[str, dict[str, Any]],
    reviewer_2: dict[str, dict[str, Any]],
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Recompute every scientific review statistic from immutable raw sources."""
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidates:
        record_id = str(row.get("bridge_record_id") or "").strip()
        if not record_id or record_id in candidate_by_id:
            raise ValueError(
                f"candidate bridge_record_id 缺失或重复: {record_id!r}"
            )
        candidate_by_id[record_id] = row
    selection_by_item = unique_review_rows(selection, "review_selection")
    selected_ids = set(selection_by_item)
    if set(reviewer_1) != selected_ids or set(reviewer_2) != selected_ids:
        raise ValueError("review statistics 输入未精确覆盖 review_selection")

    left_decisions = [
        reviewer_1[item_id]["decision"] for item_id in sorted(selected_ids)
    ]
    right_decisions = [
        reviewer_2[item_id]["decision"] for item_id in sorted(selected_ids)
    ]
    field_ratings: dict[str, list[list[str]]] = {}
    for item_id in sorted(selected_ids):
        item = selection_by_item[item_id]
        record_id = str(item.get("bridge_record_id") or "")
        if record_id not in candidate_by_id:
            raise ValueError(
                f"review statistics selection 引用未知 candidate: {record_id!r}"
            )
        record = candidate_by_id[record_id]
        reviewer_fields = []
        for response in (reviewer_1[item_id], reviewer_2[item_id]):
            if response["decision"] == "reject":
                reviewer_fields.append({"review_decision": "reject"})
                continue
            structured = (
                record["candidate"]["structured_output"]
                if response["decision"] == "accept"
                else response["corrected_structured_targets"]
            )
            target_errors = validate_bridge_structured_target(
                structured,
                expected_target_status=str(record["target_status"]),
            )
            if target_errors:
                raise ValueError(
                    f"{item_id}/{response['reviewer_id']}: structured target 非法: "
                    f"{target_errors}"
                )
            reviewer_fields.append(flatten_bridge_structured_target(structured))
        field_names = sorted(set(reviewer_fields[0]) | set(reviewer_fields[1]))
        for field in field_names:
            field_ratings.setdefault(field, []).append([
                reviewer_fields[0].get(field, "<rejected>"),
                reviewer_fields[1].get(field, "<rejected>"),
            ])
    field_agreement = {
        field: {
            "cohen_kappa": cohen_kappa(
                [pair[0] for pair in ratings],
                [pair[1] for pair in ratings],
            ),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(ratings),
            "exact_agreement": (
                sum(pair[0] == pair[1] for pair in ratings)
                / max(len(ratings), 1)
            ),
            "num_items": len(ratings),
        }
        for field, ratings in sorted(field_ratings.items())
    }
    disputed_field_counts = {
        field: sum(pair[0] != pair[1] for pair in ratings)
        for field, ratings in sorted(field_ratings.items())
    }
    disagreement_region_sources: Counter[str] = Counter()
    disagreement_modality_combos: Counter[str] = Counter()
    disagreement_evidence_levels: Counter[str] = Counter()
    for item_id in sorted(selected_ids):
        first, second = reviewer_1[item_id], reviewer_2[item_id]
        agreed = first["decision"] == second["decision"]
        if agreed and first["decision"] == "revise":
            agreed = review_revisions_match(first, second)
        if agreed:
            continue
        record = candidate_by_id[
            str(selection_by_item[item_id]["bridge_record_id"])
        ]
        disagreement_region_sources[
            str(record.get("region_source") or "unknown")
        ] += 1
        disagreement_modality_combos[
            str(record.get("modality_family_combo") or "unknown")
        ] += 1
        evidence = record.get("modality_evidence") or {}
        values = evidence.values() if isinstance(evidence, dict) else evidence
        levels = {
            str(value.get("evidence_level") or "unknown")
            for value in values
            if isinstance(value, dict)
        }
        for level in levels or {"unknown"}:
            disagreement_evidence_levels[level] += 1

    final_decisions = Counter(replay.get("final_decisions") or {})
    resolved_count = sum(final_decisions.values())
    expert = replay.get("expert") or []
    pending = replay.get("pending") or []
    return {
        "review_items": len(selected_ids),
        "expert_records": len(expert),
        "pending_arbitration": len(pending),
        "final_decisions": dict(sorted(final_decisions.items())),
        "cohen_kappa_decision": cohen_kappa(left_decisions, right_decisions),
        "krippendorff_alpha_decision": krippendorff_alpha_nominal([
            [reviewer_1[item_id]["decision"], reviewer_2[item_id]["decision"]]
            for item_id in sorted(selected_ids)
        ]),
        "initial_decision_agreement_rate": (
            sum(
                reviewer_1[item_id]["decision"]
                == reviewer_2[item_id]["decision"]
                for item_id in selected_ids
            )
            / max(len(selected_ids), 1)
        ),
        "field_agreement": field_agreement,
        "disputed_field_counts": disputed_field_counts,
        "disagreement_distribution": {
            "region_source": dict(sorted(disagreement_region_sources.items())),
            "modality_family_combo": dict(
                sorted(disagreement_modality_combos.items())
            ),
            "evidence_level": dict(sorted(disagreement_evidence_levels.items())),
        },
        "reviewer_decision_counts": {
            "reviewer_1": dict(sorted(Counter(left_decisions).items())),
            "reviewer_2": dict(sorted(Counter(right_decisions).items())),
        },
        "modification_rate": final_decisions["revise"] / max(resolved_count, 1),
        "acceptance_rate": final_decisions["accept"] / max(resolved_count, 1),
        "rejection_rate": final_decisions["reject"] / max(resolved_count, 1),
        "accepted_or_revised_rate": len(expert) / max(len(selected_ids), 1),
        "expert_modification_statistics": expert_modification_statistics(expert),
    }


def preview_image(path_ref: str, size: int = 512) -> Image.Image:
    path = resolve_project_path(path_ref)
    with Image.open(path) as image:
        image.load()
        result = image.convert("RGB")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    return result
