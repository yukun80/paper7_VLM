#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-aware datasets for global, region-alignment and Landslide Bridge description."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import MODALITY_FAMILY_IDS

from .backbone import transform_region_mask_to_cache
from .json_protocol import strict_json_loads
from .output_protocol import parse_description_output
from .vision_cache import DescriptionVisionFeatureBank, description_cache_key


DescriptionStage = Literal[
    "overfit", "mmrs_caption", "rsicap_caption", "dior_alignment",
    "bridge_auto", "bridge_expert", "predicted_mask",
]
BRIDGE_EXPERT_STATUS = "expert_pilot_frozen"
BRIDGE_GATE_PROTOCOL = "landslide_bridge_evaluation_gate_v2"
BRIDGE_BUILDER_VERSION = "landslide_bridge_m2_v7_expert_review_replay_bound"
BRIDGE_EXPERT_ARTIFACT_PROTOCOL = (
    "landslide_bridge_expert_artifact_binding_v1_review_sources_and_outputs"
)
BRIDGE_EXPERT_REPLAY_PROTOCOL = (
    "landslide_bridge_expert_review_replay_v1_exact_semantic_projection"
)
BRIDGE_ENGINEERING_AUDIT_PROTOCOL = (
    "landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound"
)
DESCRIPTION_BUILDER_VERSION = "description_benchmark_m1_v4_answer_trace"
DESCRIPTION_ENGINEERING_AUDIT_PROTOCOL = (
    "qpsalm_description_engineering_audit_v1_cache_partition_bound"
)
REGION_TRAINING_DATA_PROTOCOL = (
    "qpsalm_region_training_data_binding_v2_cache_candidate_bound"
)
REGION_INPUT_SOURCE_PROTOCOL = (
    "qpsalm_description_region_input_source_v1_cache_projection_bound"
)
FROZEN_GATE_THRESHOLD_KEYS = (
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
FROZEN_GATE_COUNTERFACTUAL_MODES = (
    "shuffled_mask",
    "region_swap",
    "cross_parent_region_swap",
    "cross_parent_modality_swap",
    "modality_removal",
)
FROZEN_GATE_SCIENTIFIC_PROTOCOLS = {
    "erfs_rubric": "qpsalm_erfs_eight_family_parent_macro_v1",
    "claim_inventory": "qpsalm_structured_claim_inventory_v1",
    "retrieval_scorer": "qpsalm_same_image_region_retrieval_v2_parent_ranked",
    "region_protocol_reporting": "separate_assisted_vision_only_v1",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = strict_json_loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"frozen Bridge JSONL 非法: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"frozen Bridge JSONL 记录必须是 object: {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _bound_record_count(path: Path) -> int:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    return len(_read_jsonl_rows(path))


def _revalidate_file_artifact(
    binding: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_records: int | None = None,
) -> Path:
    if not isinstance(binding, dict):
        raise RuntimeError(f"frozen Bridge {label} artifact binding 必须是 object")
    path_ref = binding.get("path")
    if not isinstance(path_ref, str) or not path_ref.strip():
        raise RuntimeError(f"frozen Bridge {label} artifact binding 缺少 path")
    path = resolve_project_path(path_ref)
    assert path is not None
    if expected_path is not None and path != expected_path.resolve(strict=False):
        raise RuntimeError(
            f"frozen Bridge {label} artifact path 不匹配: "
            f"expected={expected_path} observed={path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"frozen Bridge {label} artifact 缺失: {path_ref} -> {path}")
    observed_sha = _sha256_file(path)
    if binding.get("sha256") != observed_sha:
        raise RuntimeError(f"frozen Bridge {label} artifact hash 漂移: {path_ref}")
    observed_bytes = path.stat().st_size
    if binding.get("bytes") != observed_bytes:
        raise RuntimeError(f"frozen Bridge {label} artifact bytes 漂移: {path_ref}")
    if expected_records is not None and binding.get("records") != expected_records:
        raise RuntimeError(
            f"frozen Bridge {label} artifact records 不匹配: "
            f"expected={expected_records} observed={binding.get('records')!r}"
        )
    return path


def _revalidate_expert_artifacts(
    bridge_dir: Path,
    report: dict[str, Any],
    gate_path: Path,
) -> dict[str, Any]:
    """Replay review sources, split projections and validation-level binding."""
    validation_binding = report.get("expert_artifact_binding")
    if not isinstance(validation_binding, dict):
        raise RuntimeError("frozen Bridge validation report 缺少 expert_artifact_binding")
    if (
        validation_binding.get("protocol") != BRIDGE_EXPERT_ARTIFACT_PROTOCOL
        or validation_binding.get("builder_version") != BRIDGE_BUILDER_VERSION
    ):
        raise RuntimeError("frozen Bridge validation-level expert artifact binding 过期")
    review_report_path = bridge_dir / "reports/expert_review_report.json"
    _revalidate_file_artifact(
        validation_binding.get("review_report"),
        label="expert_review_report",
        expected_path=review_report_path,
    )
    review_report = strict_json_loads(
        review_report_path.read_text(encoding="utf-8")
    )
    if (
        review_report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or review_report.get("status") != "complete"
        or review_report.get("frozen_evaluation_gate") is not True
        or (review_report.get("errors") or [])
    ):
        raise RuntimeError("frozen Bridge expert_review_report 不是当前完整 v7 merge report")
    merge_binding = review_report.get("expert_artifact_binding")
    if validation_binding.get("merge_artifacts") != merge_binding:
        raise RuntimeError(
            "frozen Bridge validation report 与 expert_review_report artifact binding 不一致"
        )
    if not isinstance(merge_binding, dict) or (
        merge_binding.get("protocol") != BRIDGE_EXPERT_ARTIFACT_PROTOCOL
        or merge_binding.get("builder_version") != BRIDGE_BUILDER_VERSION
    ):
        raise RuntimeError("frozen Bridge merge-level expert artifact binding 过期")
    semantic_replay = validation_binding.get("semantic_replay")
    if not isinstance(semantic_replay, dict) or (
        semantic_replay.get("protocol") != BRIDGE_EXPERT_REPLAY_PROTOCOL
    ):
        raise RuntimeError("frozen Bridge validation report 缺少精确 expert 语义重放")
    sources = merge_binding.get("sources")
    outputs = merge_binding.get("outputs")
    expected_source_keys = {
        "reviewer_1", "reviewer_2", "arbitration", "evaluation_gate_source",
    }
    expected_output_paths = {
        "expert_all": bridge_dir / "indexes/expert_all.jsonl",
        "expert_train": bridge_dir / "indexes/expert_train.jsonl",
        "expert_val": bridge_dir / "indexes/expert_val.jsonl",
        "expert_test": bridge_dir / "indexes/expert_test.jsonl",
        "pending_arbitration": bridge_dir / "indexes/pending_arbitration.jsonl",
        "evaluation_gate": gate_path,
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_keys:
        raise RuntimeError("frozen Bridge expert artifact sources 集合不完整")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_output_paths):
        raise RuntimeError("frozen Bridge expert artifact outputs 集合不完整")
    review_items = int(review_report.get("review_items", -1))
    review_selection_path = bridge_dir / "manifests/review_selection.jsonl"
    selection_count = len(_read_jsonl_rows(review_selection_path))
    if review_items != selection_count:
        raise RuntimeError(
            "frozen Bridge expert review 未精确覆盖完整 review_selection: "
            f"expected={selection_count} observed={review_items}"
        )
    final_decisions = review_report.get("final_decisions")
    if not isinstance(final_decisions, dict):
        raise RuntimeError("frozen Bridge expert_review_report 缺少 final_decisions")
    try:
        final_count = sum(int(value) for value in final_decisions.values())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frozen Bridge final_decisions 计数非法") from exc
    try:
        pending_count = int(review_report.get("pending_arbitration", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frozen Bridge pending_arbitration 计数非法") from exc
    if final_count != review_items or pending_count != 0:
        raise RuntimeError(
            "frozen Bridge final_decisions/pending 未精确覆盖 review_selection"
        )
    candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
    candidate_count = len(_read_jsonl_rows(candidate_path))
    _revalidate_file_artifact(
        semantic_replay.get("candidate_index"),
        label="semantic_replay.candidate_index",
        expected_path=candidate_path,
        expected_records=candidate_count,
    )
    _revalidate_file_artifact(
        semantic_replay.get("review_selection"),
        label="semantic_replay.review_selection",
        expected_path=review_selection_path,
        expected_records=selection_count,
    )
    if (
        semantic_replay.get("review_items") != review_items
        or semantic_replay.get("pending_arbitration") != pending_count
        or semantic_replay.get("final_decisions") != final_decisions
        or semantic_replay.get("review_report_statistics_verified") is not True
    ):
        raise RuntimeError(
            "frozen Bridge semantic replay 计数/决策/审核统计与 "
            "expert_review_report 不一致"
        )
    source_paths: dict[str, str | None] = {}
    for name in ("reviewer_1", "reviewer_2"):
        source_path = _revalidate_file_artifact(sources[name], label=name)
        source_count = _bound_record_count(source_path)
        _revalidate_file_artifact(
            sources[name], label=name, expected_records=source_count
        )
        if source_count != review_items:
            raise RuntimeError(
                f"frozen Bridge {name} 未精确覆盖 review_items: "
                f"expected={review_items} observed={source_count}"
            )
        source_paths[name] = str(source_path)
    arbitration_binding = sources["arbitration"]
    if arbitration_binding is None:
        source_paths["arbitration"] = None
    else:
        arbitration_path = _revalidate_file_artifact(
            arbitration_binding, label="arbitration"
        )
        arbitration_count = _bound_record_count(arbitration_path)
        _revalidate_file_artifact(
            arbitration_binding,
            label="arbitration",
            expected_records=arbitration_count,
        )
        source_paths["arbitration"] = str(arbitration_path)
    gate_source_path = _revalidate_file_artifact(
        sources["evaluation_gate_source"], label="evaluation_gate_source"
    )
    source_paths["evaluation_gate_source"] = str(gate_source_path)
    source_gate = strict_json_loads(gate_source_path.read_text(encoding="utf-8"))
    expected_published_gate = dict(source_gate)
    expected_published_gate["source_file"] = sources[
        "evaluation_gate_source"
    ]["path"]
    published_gate = strict_json_loads(gate_path.read_text(encoding="utf-8"))
    if published_gate != expected_published_gate:
        raise RuntimeError(
            "frozen Bridge published gate 不是 frozen gate source 的精确带来源副本"
        )

    expert_all_path = expected_output_paths["expert_all"]
    expert_rows = _read_jsonl_rows(expert_all_path)
    split_rows = {
        split: _read_jsonl_rows(expected_output_paths[f"expert_{split}"])
        for split in ("train", "val", "test")
    }
    pending_rows = _read_jsonl_rows(expected_output_paths["pending_arbitration"])
    if pending_rows:
        raise RuntimeError("frozen Bridge 仍含 pending arbitration")
    if int(review_report.get("expert_records", -1)) != len(expert_rows):
        raise RuntimeError("frozen Bridge expert_review_report 记录数与 expert_all 不一致")
    if semantic_replay.get("expert_records") != len(expert_rows):
        raise RuntimeError("frozen Bridge semantic replay 记录数与 expert_all 不一致")
    if sum(int(final_decisions.get(key, 0)) for key in ("accept", "revise")) != len(
        expert_rows
    ):
        raise RuntimeError("frozen Bridge accept/revise 计数与 expert_all 不一致")
    ids = [str(row.get("bridge_record_id") or "") for row in expert_rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("frozen Bridge expert_all bridge_record_id 缺失或重复")
    for split, rows in split_rows.items():
        expected_rows = [row for row in expert_rows if row.get("split") == split]
        if not rows or rows != expected_rows:
            raise RuntimeError(
                f"frozen Bridge expert_{split} 不是 expert_all 的非空精确 split 投影"
            )
        _validate_expert_rows(rows, stage="bridge_expert", split=split)
    output_counts = {
        "expert_all": len(expert_rows),
        "expert_train": len(split_rows["train"]),
        "expert_val": len(split_rows["val"]),
        "expert_test": len(split_rows["test"]),
        "pending_arbitration": 0,
    }
    for name, path in expected_output_paths.items():
        _revalidate_file_artifact(
            outputs[name],
            label=name,
            expected_path=path,
            expected_records=output_counts.get(name),
        )
    return {
        "protocol": BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
        "semantic_replay_protocol": BRIDGE_EXPERT_REPLAY_PROTOCOL,
        "review_report": str(review_report_path),
        "review_report_sha256": _sha256_file(review_report_path),
        "source_paths": source_paths,
        "expert_index_sha256": {
            split: _sha256_file(expected_output_paths[f"expert_{split}"])
            for split in ("train", "val", "test")
        },
    }


def _validate_frozen_scientific_gate(gate: dict[str, Any]) -> None:
    """Require every Pilot-frozen threshold used by M4/M6 formal gates."""
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("frozen Bridge gate 缺少 thresholds object")
    invalid_thresholds = []
    for key in FROZEN_GATE_THRESHOLD_KEYS:
        value = thresholds.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            invalid_thresholds.append(f"{key}={value!r}")
    if invalid_thresholds:
        raise RuntimeError(
            "frozen Bridge gate 的科学阈值未完整冻结: " + ", ".join(invalid_thresholds)
        )
    scientific = gate.get("scientific_protocol")
    if not isinstance(scientific, dict):
        raise RuntimeError("frozen Bridge gate 缺少 scientific_protocol")
    for key, expected in FROZEN_GATE_SCIENTIFIC_PROTOCOLS.items():
        if scientific.get(key) != expected:
            raise RuntimeError(
                f"frozen Bridge gate scientific_protocol.{key} 非法: "
                f"expected={expected!r} observed={scientific.get(key)!r}"
            )
    bootstrap = scientific.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise RuntimeError("frozen Bridge gate 缺少 bootstrap protocol")
    seed = bootstrap.get("seed")
    confidence = bootstrap.get("confidence")
    if (
        bootstrap.get("unit") != "parent"
        or bootstrap.get("samples") != 10000
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isclose(float(confidence), 0.95, abs_tol=1.0e-12)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise RuntimeError("frozen Bridge gate bootstrap 必须为 parent/10000/0.95/非负 seed")
    minimums = scientific.get("counterfactual_minimum_effective_parents")
    if not isinstance(minimums, dict) or set(minimums) != set(
        FROZEN_GATE_COUNTERFACTUAL_MODES
    ):
        raise RuntimeError("frozen Bridge gate 反事实模式集合不完整")
    invalid_minimums = {
        mode: minimums.get(mode)
        for mode in FROZEN_GATE_COUNTERFACTUAL_MODES
        if (
            isinstance(minimums.get(mode), bool)
            or not isinstance(minimums.get(mode), int)
            or int(minimums.get(mode)) <= 0
        )
    }
    if invalid_minimums:
        raise RuntimeError(f"frozen Bridge gate 反事实 parent 门槛非法: {invalid_minimums}")


def require_frozen_expert_bridge(bridge_dir: Path) -> dict[str, Any]:
    """Reject stale expert indexes unless the current Pilot gate is frozen."""
    report_path = bridge_dir / "reports/validation_report.json"
    gate_path = bridge_dir / "manifests/evaluation_gate_manifest.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Bridge 缺少 validation report: {report_path}")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or
        report.get("status") != BRIDGE_EXPERT_STATUS
        or report.get("require_expert_complete") is not True
        or (report.get("errors") or [])
    ):
        raise RuntimeError(
            "D3b/D4/M7 expert 数据要求 Bridge status=expert_pilot_frozen、"
            "require_expert_complete=true 且 errors=[]；"
            f"当前 status={report.get('status')!r}"
        )
    if not gate_path.is_file():
        raise FileNotFoundError(f"frozen Bridge 缺少 evaluation gate: {gate_path}")
    gate = strict_json_loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("protocol") != BRIDGE_GATE_PROTOCOL
        or gate.get("builder_version") != BRIDGE_BUILDER_VERSION
        or gate.get("frozen") is not True
        or gate.get("status") != "frozen_after_pilot"
    ):
        raise RuntimeError("Bridge evaluation gate 不是当前 builder 的人工冻结 v2 Pilot gate")
    _validate_frozen_scientific_gate(gate)
    binding_paths = {
        "pilot_parent_manifest_sha256": bridge_dir / "manifests/pilot_parent_manifest.jsonl",
        "review_selection_sha256": bridge_dir / "manifests/review_selection.jsonl",
        "candidate_index_sha256": bridge_dir / "indexes/candidate_all.jsonl",
    }
    missing = [str(path) for path in binding_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Bridge evaluation gate 绑定文件缺失: {missing}")
    expected = {name: _sha256_file(path) for name, path in binding_paths.items()}
    if gate.get("bindings") != expected:
        raise RuntimeError("Bridge evaluation gate 与当前 Pilot/selection/candidate hash 不一致")
    expert_artifact_audit = _revalidate_expert_artifacts(
        bridge_dir.resolve(strict=False), report, gate_path.resolve(strict=False)
    )
    return {
        "status": BRIDGE_EXPERT_STATUS,
        "validation_report": str(report_path),
        "validation_report_sha256": _sha256_file(report_path),
        "evaluation_gate": str(gate_path),
        "evaluation_gate_sha256": _sha256_file(gate_path),
        "candidate_index": str(
            binding_paths["candidate_index_sha256"].resolve(strict=False)
        ),
        "candidate_index_sha256": expected["candidate_index_sha256"],
        "expert_artifact_audit": expert_artifact_audit,
    }


def require_engineering_bridge(
    bridge_dir: Path,
    vision_bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Revalidate Bridge rows and the cache input used by region stages."""

    report_path = bridge_dir / "reports/validation_report.json"
    candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
    auto_path = bridge_dir / "indexes/auto_train.jsonl"
    for path in (report_path, candidate_path, auto_path):
        if not path.is_file():
            raise FileNotFoundError(f"engineering Bridge 缺少 artifact: {path}")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or report.get("status") not in {
            "awaiting_expert_review", "expert_pilot_frozen",
        }
        or report.get("pilot_protocol_complete") is not True
        or (report.get("errors") or [])
    ):
        raise RuntimeError(
            "D-1/D3a 要求当前 M2 Bridge engineering-valid 且 Pilot 完整；"
            f"当前 status={report.get('status')!r}"
        )
    candidates = _read_jsonl_rows(candidate_path)
    auto_rows = _read_jsonl_rows(auto_path)
    candidate_ids = [str(row.get("bridge_record_id") or "") for row in candidates]
    auto_ids = [str(row.get("bridge_record_id") or "") for row in auto_rows]
    if (
        any(not value for value in candidate_ids + auto_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or len(auto_ids) != len(set(auto_ids))
    ):
        raise RuntimeError("engineering Bridge candidate/auto ID 缺失或重复")
    candidate_by_id = {
        str(row["bridge_record_id"]): row for row in candidates
    }
    expected_train = {
        record_id for record_id, row in candidate_by_id.items()
        if str(row.get("split") or "") == "train"
    }
    if set(auto_ids) != expected_train:
        raise RuntimeError(
            "engineering Bridge auto_train 不是 candidate train 的精确 ID 投影"
        )
    for row in auto_rows:
        record_id = str(row["bridge_record_id"])
        if row != candidate_by_id[record_id]:
            raise RuntimeError(
                f"engineering Bridge auto_train row 已偏离 candidate: {record_id}"
            )
    invalid_authority = [
        record_id for record_id, row in candidate_by_id.items()
        if not isinstance(row.get("candidate"), dict)
        or row["candidate"].get("protocol")
        != "landslide_bridge_rule_candidate_v1"
        or row["candidate"].get("is_expert_truth") is not False
        or "expert_target" in row
    ]
    if invalid_authority:
        raise RuntimeError(
            "engineering Bridge candidate authority 非法，禁止冒充 expert truth: "
            f"{invalid_authority[:10]}"
        )
    cache_inputs = dict(vision_bank.manifest.get("input_fingerprints") or {})
    multisource_parent = dict(cache_inputs.get("multisource_parent") or {})
    cache_root = resolve_project_path(
        str(multisource_parent.get("benchmark") or "")
    )
    candidate_resolved = candidate_path.resolve(strict=False)
    if (
        cache_root is None
        or cache_root.resolve(strict=False) != bridge_dir.resolve(strict=False)
        or multisource_parent.get("index") != "indexes/candidate_all.jsonl"
        or int(multisource_parent.get("size", -1))
        != candidate_resolved.stat().st_size
        or multisource_parent.get("sha256") != _sha256_file(candidate_resolved)
        or multisource_parent.get("validation_report")
        != "reports/validation_report.json"
        or int(multisource_parent.get("validation_report_size", -1))
        != report_path.stat().st_size
        or multisource_parent.get("validation_report_sha256")
        != _sha256_file(report_path)
        or multisource_parent.get("validation_builder_version")
        != BRIDGE_BUILDER_VERSION
        or multisource_parent.get("validation_status")
        != str(report.get("status") or "")
    ):
        raise RuntimeError(
            "Bridge live candidate index 与 Description Vision Cache binding 不一致"
        )
    observed_by_source = dict(sorted(Counter(
        str(row.get("region_source") or "") for row in candidates
    ).items()))
    observed_parents = len({
        str(row.get("parent_sample_id") or "") for row in candidates
    })
    if (
        int(report.get("records", -1)) != len(candidates)
        or int(report.get("parents", -1)) != observed_parents
        or report.get("records_by_region_source") != observed_by_source
    ):
        raise RuntimeError(
            "engineering Bridge validation summary 与 live candidate population 不一致"
        )
    population_payload = [
        {
            "bridge_record_id": str(row["bridge_record_id"]),
            "parent_sample_id": str(row.get("parent_sample_id") or ""),
            "split": str(row.get("split") or ""),
            "region_source": str(row.get("region_source") or ""),
            "candidate_is_expert_truth": row["candidate"]["is_expert_truth"],
        }
        for row in sorted(candidates, key=lambda value: str(value["bridge_record_id"]))
    ]
    population_sha256 = hashlib.sha256(json.dumps(
        population_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return {
        "protocol": BRIDGE_ENGINEERING_AUDIT_PROTOCOL,
        "status": str(report["status"]),
        "builder_version": BRIDGE_BUILDER_VERSION,
        "expert_truth_used": False,
        "validation_report": str(report_path.resolve(strict=False)),
        "validation_report_sha256": _sha256_file(report_path),
        "cache_input_fingerprint": multisource_parent,
        "candidate_index": str(candidate_path.resolve(strict=False)),
        "candidate_index_sha256": _sha256_file(candidate_path),
        "auto_train_index": str(auto_path.resolve(strict=False)),
        "auto_train_index_sha256": _sha256_file(auto_path),
        "candidate_records": len(candidates),
        "auto_train_records": len(auto_rows),
        "population_sha256": population_sha256,
    }


def require_engineering_description(
    description_dir: Path,
    vision_bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Bind live M1.1 partitions to the all-index used by Description Cache v1."""

    report_path = description_dir / "reports/validation_report.json"
    index_paths = {
        name: description_dir / f"indexes/{name}.jsonl"
        for name in ("all", "train", "dev", "test", "train_eligible")
    }
    missing = [str(path) for path in (report_path, *index_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"engineering Description 缺少 artifact: {missing}"
        )
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("builder_version") != DESCRIPTION_BUILDER_VERSION
        or (report.get("errors") or [])
        or int(
            report.get(
                "verified_perceptual_duplicate_cross_split_groups", -1
            )
        ) != 0
    ):
        raise RuntimeError(
            "D-1/D0-D2 要求 engineering-valid Description M1.1 v4，"
            "且 verified cross-split cluster 必须为零"
        )
    cache_inputs = dict(vision_bank.manifest.get("input_fingerprints") or {})
    single_image = dict(cache_inputs.get("single_image") or {})
    cache_root = resolve_project_path(str(single_image.get("benchmark") or ""))
    all_path = index_paths["all"].resolve(strict=False)
    if (
        cache_root is None
        or cache_root.resolve(strict=False) != description_dir.resolve(strict=False)
        or single_image.get("index") != "indexes/all.jsonl"
        or int(single_image.get("size", -1)) != all_path.stat().st_size
        or single_image.get("sha256") != _sha256_file(all_path)
        or single_image.get("validation_report")
        != "reports/validation_report.json"
        or int(single_image.get("validation_report_size", -1))
        != report_path.stat().st_size
        or single_image.get("validation_report_sha256")
        != _sha256_file(report_path)
        or single_image.get("validation_builder_version")
        != DESCRIPTION_BUILDER_VERSION
        or single_image.get("validation_status") != "engineering-valid"
    ):
        raise RuntimeError(
            "Description M1.1 live all index 与 Description Vision Cache binding 不一致"
        )
    rows_by_name = {
        name: _read_jsonl_rows(path) for name, path in index_paths.items()
    }
    all_rows = rows_by_name["all"]
    all_ids = [str(row.get("sample_id") or "") for row in all_rows]
    if any(not value for value in all_ids) or len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Description M1.1 all index sample_id 缺失或重复")
    all_by_id = {
        str(row["sample_id"]): row for row in all_rows
    }
    partition_ids: list[str] = []
    for split in ("train", "dev", "test"):
        split_rows = rows_by_name[split]
        split_ids = [str(row.get("sample_id") or "") for row in split_rows]
        if (
            any(str(row.get("split") or "") != split for row in split_rows)
            or any(
                sample_id not in all_by_id
                or row != all_by_id[sample_id]
                for sample_id, row in zip(split_ids, split_rows, strict=True)
            )
        ):
            raise RuntimeError(
                f"Description M1.1 {split} 不是 all index 的精确投影"
            )
        partition_ids.extend(split_ids)
    if Counter(partition_ids) != Counter(all_ids):
        raise RuntimeError("Description M1.1 train/dev/test 未精确分区 all index")
    train_by_id = {
        str(row["sample_id"]): row for row in rows_by_name["train"]
    }
    expected_eligible: dict[str, dict[str, Any]] = {}
    for sample_id, source_row in train_by_id.items():
        positive_answers = [
            answer for answer in source_row.get("answers", [])
            if float(answer.get("caption_quality_weight", 0.0)) > 0.0
        ]
        if positive_answers:
            expected = dict(source_row)
            expected["answers"] = positive_answers
            expected_eligible[sample_id] = expected
    observed_eligible = {
        str(row.get("sample_id") or ""): row
        for row in rows_by_name["train_eligible"]
    }
    if (
        len(observed_eligible) != len(rows_by_name["train_eligible"])
        or observed_eligible != expected_eligible
    ):
        raise RuntimeError(
            "Description M1.1 train_eligible 不是正权重 train 的精确投影"
        )
    observed_parents = len({
        str(row.get("parent_sample_id") or "") for row in all_rows
    })
    if (
        int(report.get("num_records", -1)) != len(all_rows)
        or int(report.get("deep_checked_records", -1)) != len(all_rows)
        or int(report.get("num_parents", -1)) != observed_parents
        or int(report.get("decoded_unique_images", -1)) != observed_parents
        or int(report.get("materialized_files", -1)) != observed_parents
        or int(report.get("train_eligible_records", -1))
        != len(observed_eligible)
    ):
        raise RuntimeError(
            "Description M1.1 validation summary 与 live index population 不一致"
        )
    index_bindings = {
        name: {
            "path": str(path.resolve(strict=False)),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
            "records": len(rows_by_name[name]),
        }
        for name, path in index_paths.items()
    }
    return {
        "protocol": DESCRIPTION_ENGINEERING_AUDIT_PROTOCOL,
        "builder_version": DESCRIPTION_BUILDER_VERSION,
        "validation_report": str(report_path.resolve(strict=False)),
        "validation_report_sha256": _sha256_file(report_path),
        "cache_input_fingerprint": single_image,
        "indexes": index_bindings,
        "num_records": len(all_rows),
        "num_parents": observed_parents,
        "verified_perceptual_duplicate_cross_split_groups": 0,
    }


def load_frozen_scientific_gate(bridge_dir: Path) -> dict[str, Any]:
    """Load the current bound Pilot gate for formal M4/M6 statistics."""
    audit = require_frozen_expert_bridge(bridge_dir)
    gate_path = Path(audit["evaluation_gate"])
    gate = strict_json_loads(gate_path.read_text(encoding="utf-8"))
    return {
        "audit": audit,
        "thresholds": dict(gate["thresholds"]),
        "scientific_protocol": dict(gate["scientific_protocol"]),
    }


def validate_predicted_index(
    index_path: Path, *, split: str, expert_gate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Bind fixed/OOF masks to the report that published their exact index."""
    if split == "train":
        # 延迟导入避免 data -> predicted_regions -> data 的模块级环依赖。
        from .predicted_regions import revalidate_oof_merged_index

        replay = revalidate_oof_merged_index(
            index_path,
            expected_expert_gate_audit=expert_gate_audit,
        )
        return {**replay, "split": "train"}
    # val/test 同样逐行重放 checkpoint、专家源记录与每个 mask，而非只看顶层 report。
    from .predicted_regions import revalidate_fixed_predicted_index

    return revalidate_fixed_predicted_index(
        index_path,
        split=split,
        expected_expert_gate_audit=expert_gate_audit,
    )


def revalidate_predicted_index_audit(
    audit: Any,
    *,
    expected_split: str,
    expert_gate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a saved predicted-index audit from its bound live artifacts."""
    if (
        not isinstance(audit, dict)
        or str(audit.get("split") or "") != expected_split
        or not str(audit.get("index") or "").strip()
    ):
        raise ValueError(
            f"predicted index audit 缺少 split={expected_split} 的可重放 index"
        )
    index_path = resolve_project_path(str(audit["index"])) or Path(str(audit["index"]))
    current = validate_predicted_index(
        index_path,
        split=expected_split,
        expert_gate_audit=expert_gate_audit,
    )
    if current != audit:
        raise ValueError("predicted index audit 与当前深度重放结果不一致")
    return current


def _validate_expert_rows(
    rows: list[dict[str, Any]], *, stage: str, split: str,
) -> None:
    if not rows:
        raise ValueError(f"expert stage={stage} split={split} 不能为空")
    for row in rows:
        sample_id = str(row.get("bridge_record_id") or row.get("sample_id") or "unknown")
        target = row.get("expert_target")
        if not isinstance(target, dict) or not isinstance(
            target.get("structured_output"), dict
        ) or not str(target.get("summary") or "").strip():
            raise ValueError(f"expert row 缺少人工审核 target: {sample_id}")
        review_status = str((row.get("review") or {}).get("status") or "")
        if review_status not in {"accepted", "revised", "arbitrated"}:
            raise ValueError(f"expert row review status 非法: {sample_id}={review_status!r}")
        predicted = (
            str(row.get("region_source") or "") == "predicted_proposal"
            or str(row.get("schema_version") or "").startswith("qpsalm_predicted_region")
        )
        provenance = row.get("prediction_provenance")
        if (
            stage == "predicted_mask"
            and split == "train"
            and predicted
            and (
                not isinstance(provenance, dict)
                or provenance.get("out_of_fold_verified") is not True
            )
        ):
            raise ValueError(f"D4 train predicted row 未通过 OOF 审计: {sample_id}")


def description_row_sample_id(row: dict[str, Any]) -> str:
    """Return the stable task identity used by evaluation metadata."""
    value = row.get("sample_id") or row.get("bridge_record_id")
    if not value:
        raise ValueError("description row 缺少 sample_id/bridge_record_id")
    return str(value)


def same_parent_region_swap_candidates(
    rows: list[dict[str, Any]],
    sample_id: str,
    *,
    catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic real-region candidates from the same parent.

    The alternate catalog may contain unreviewed Bridge region geometry, but
    its text target is never consumed.  Null/no-target rows are not regions and
    therefore cannot be used as a region-swap shortcut.
    """
    by_id = {description_row_sample_id(row): row for row in rows}
    current = by_id.get(str(sample_id))
    if current is None:
        return []
    parent = str(current.get("parent_sample_id") or "")
    current_region_id = str(current.get("region_id") or "")
    current_mask = (current.get("region_mask") or {}).get("path")
    candidates = []
    for row in catalog if catalog is not None else rows:
        candidate_id = description_row_sample_id(row)
        if candidate_id == str(sample_id):
            continue
        if str(row.get("parent_sample_id") or "") != parent:
            continue
        geometry = row.get("region_geometry") or {}
        has_box = geometry.get("type") == "box"
        has_mask = bool((row.get("region_mask") or {}).get("path"))
        if not (has_box or has_mask):
            continue
        if str(row.get("target_status") or "present") != "present":
            continue
        candidate_region_id = str(row.get("region_id") or "")
        candidate_mask = (row.get("region_mask") or {}).get("path")
        if (
            candidate_region_id
            and candidate_region_id == current_region_id
            and candidate_mask == current_mask
        ):
            continue
        candidates.append(row)
    return sorted(
        candidates,
        key=lambda row: (
            str(row.get("region_source") or "") == str(current.get("region_source") or ""),
            str(row.get("region_id") or ""),
            description_row_sample_id(row),
        ),
    )


def cross_parent_region_swap_candidates(
    rows: list[dict[str, Any]],
    sample_id: str,
    *,
    catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic real-region donors from different parents."""
    by_id = {description_row_sample_id(row): row for row in rows}
    current = by_id.get(str(sample_id))
    if current is None:
        return []
    parent = str(current.get("parent_sample_id") or "")
    candidates = []
    for row in catalog if catalog is not None else rows:
        candidate_id = description_row_sample_id(row)
        donor_parent = str(row.get("parent_sample_id") or "")
        if not donor_parent or donor_parent == parent:
            continue
        geometry = row.get("region_geometry") or {}
        has_box = geometry.get("type") == "box"
        has_mask = bool((row.get("region_mask") or {}).get("path"))
        if not (has_box or has_mask):
            continue
        if str(row.get("target_status") or "present") != "present":
            continue
        rank = hashlib.sha256(
            f"cross-parent-region:{sample_id}:{candidate_id}".encode("utf-8")
        ).hexdigest()
        candidates.append((rank, candidate_id, row))
    return [row for _rank, _candidate_id, row in sorted(candidates)]


def end_to_end_region_support(row: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a Bridge row has an identifiable segmentation target.

    Global masks always map to the global segmentation instruction. Referring
    masks and pseudo components are valid only when inventory deduplication
    attached at least one referring-target alias. A pseudo component without
    such an alias has no language target for the segmentation model and must
    not silently fall back to whole-image segmentation.
    """
    source = str(row.get("region_source") or "unknown")
    if source == "gt_global_mask":
        return True, "global_instruction"
    aliases = [
        value for value in (row.get("source_region_aliases") or [])
        if isinstance(value, dict) and value.get("sample_id")
    ]
    if source == "gt_referring_mask":
        return (bool(aliases), "referring_alias" if aliases else "missing_referring_alias")
    if source == "pseudo_instance_component":
        return (
            bool(aliases),
            "component_with_referring_alias" if aliases else "component_without_language_target",
        )
    if source == "no_target":
        # Empty parents can map to an empty global instruction even when there
        # is no explicit no-target referring alias. The resolver verifies this.
        return True, "no_target_alias_or_empty_global"
    return False, f"unsupported_region_source:{source}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(strict_json_loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 非法 JSONL") from exc
    return rows


def _stable_weighted_index(
    seed: int,
    epoch: int,
    sample_id: str,
    weights: list[float],
) -> int:
    """Draw one deterministic epoch-specific answer from positive quality weights."""
    if not weights or any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("caption answer weights 必须是非空有限非负数列")
    total = sum(weights)
    if total <= 0:
        raise ValueError("caption answer weights 总和必须大于 0")
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}:weighted".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64) * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    return len(weights) - 1


def _caption_source_weights(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    rsicap_mmrs_fraction: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Keep all parents while equalizing sources inside the D0/D1 protocol mix."""
    if stage not in {"mmrs_caption", "rsicap_caption"} or not rows:
        return {}, {"protocol": "not_applicable"}
    counts = Counter(str(row.get("source_dataset") or "unknown") for row in rows)
    source_group = {
        source: ("mmrs" if source.startswith("MMRS-") else "rsicap")
        for source in counts
    }
    sources_by_group: dict[str, list[str]] = {}
    for source, group in source_group.items():
        sources_by_group.setdefault(group, []).append(source)
    if stage == "mmrs_caption":
        group_mass = {"mmrs": 1.0}
    else:
        available_groups = set(sources_by_group)
        requested = {
            "rsicap": 1.0 - float(rsicap_mmrs_fraction),
            "mmrs": float(rsicap_mmrs_fraction),
        }
        normalizer = sum(requested[group] for group in available_groups)
        group_mass = {
            group: requested[group] / max(normalizer, 1.0e-12)
            for group in available_groups
        }
    raw_source_mass = {
        source: group_mass[group] / len(sources_by_group[group])
        for group, sources in sources_by_group.items()
        for source in sources
    }
    # Sum of per-row weights equals number of parents, so optimizer loss scale
    # remains comparable to stages that use unit weights.
    row_scale = float(len(rows))
    by_sample = {
        description_row_sample_id(row): (
            raw_source_mass[str(row.get("source_dataset") or "unknown")]
            / counts[str(row.get("source_dataset") or "unknown")]
            * row_scale
        )
        for row in rows
    }
    return by_sample, {
        "protocol": "qpsalm_caption_parent_epoch_source_weighting_v1",
        "stage": stage,
        "num_parents": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "source_total_mass": {
            source: raw_source_mass[source]
            for source in sorted(raw_source_mass)
        },
        "group_total_mass": dict(sorted(group_mass.items())),
        "row_weight_mean": sum(by_sample.values()) / len(by_sample),
    }


def _stable_subset(rows: list[dict[str, Any]], count: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{namespace}:{row.get('sample_id') or row.get('bridge_record_id')}".encode()
        ).hexdigest(),
    )
    return ranked[:max(0, int(count))]


D_MINUS_ONE_CATEGORIES = ("global", "box", "mask", "null")


def select_d_minus_one_mixture(
    description_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the protocol-required deterministic global/box/mask/null mix.

    Bridge bbox metadata describes a mask and is not a box-conditioned training
    example. D-1 therefore takes real full-image/box tasks from Description M1.1
    and mask/null tasks from Bridge rule candidates. The latter remain explicitly
    non-expert engineering supervision.
    """
    requested = int(count)
    if not 32 <= requested <= 64:
        raise ValueError(
            "D-1 overfit 要求 32-64 条样本；"
            f"当前 requested={requested}"
        )
    pools: dict[str, list[dict[str, Any]]] = {
        "global": [
            row for row in description_rows
            if row.get("task_family") == "global_caption"
            and (row.get("region_geometry") or {}).get("type") == "full_image"
        ],
        "box": [
            row for row in description_rows
            if row.get("task_family") == "region_referring_expression"
            and (row.get("region_geometry") or {}).get("type") == "box"
        ],
        "mask": [
            row for row in bridge_rows
            if str(row.get("split") or "") == "train"
            and str(row.get("region_source") or "") != "no_target"
            and isinstance(row.get("region_mask"), dict)
            and bool((row.get("region_mask") or {}).get("path"))
        ],
        "null": [
            row for row in bridge_rows
            if str(row.get("split") or "") == "train"
            and (
                str(row.get("region_source") or "") == "no_target"
                or str(row.get("target_status") or "") == "absent"
            )
            and not row.get("region_mask")
        ],
    }
    missing = [name for name in D_MINUS_ONE_CATEGORIES if not pools[name]]
    if missing:
        raise RuntimeError(f"D-1 四路混合缺少真实类别: {missing}")

    base, remainder = divmod(requested, len(D_MINUS_ONE_CATEGORIES))
    quotas = {
        name: base + int(index < remainder)
        for index, name in enumerate(D_MINUS_ONE_CATEGORIES)
    }
    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    for name in D_MINUS_ONE_CATEGORIES:
        if len(pools[name]) < quotas[name]:
            raise RuntimeError(
                f"D-1 category={name} 样本不足: "
                f"required={quotas[name]} available={len(pools[name])}"
            )
        selected = _stable_subset(
            pools[name], quotas[name], seed, f"d_minus_one:{name}"
        )
        selected_by_category[name] = [
            {
                **row,
                "_d_minus_one_category": name,
                "_d_minus_one_item_kind": (
                    "description" if name in {"global", "box"} else "bridge"
                ),
                "_d_minus_one_target_authority": (
                    "description_benchmark_answer"
                    if name in {"global", "box"}
                    else "deterministic_rule_candidate_not_expert"
                ),
            }
            for row in selected
        ]

    # Round-robin order guarantees a bounded generation smoke observes all four
    # categories even when max_generate_samples is smaller than the population.
    mixed = [
        selected_by_category[name][offset]
        for offset in range(max(quotas.values()))
        for name in D_MINUS_ONE_CATEGORIES
        if offset < len(selected_by_category[name])
    ]

    native_sizes = set()
    for row in mixed:
        if row["_d_minus_one_item_kind"] == "description":
            visual = row.get("visual_ref") or {}
            size = (visual.get("height"), visual.get("width"))
        else:
            original = (row.get("visual_ref") or {}).get("original_size") or []
            size = tuple(original[:2]) if len(original) >= 2 else (None, None)
        if all(isinstance(value, int) and value > 0 for value in size):
            native_sizes.add(tuple(int(value) for value in size))
    return mixed, {
        "protocol": "qpsalm_d_minus_one_stratified_mixture_v1",
        "requested_samples": requested,
        "selected_samples": len(mixed),
        "sampling_seed": int(seed),
        "category_order": list(D_MINUS_ONE_CATEGORIES),
        "category_counts": {
            name: len(selected_by_category[name]) for name in D_MINUS_ONE_CATEGORIES
        },
        "category_available": {
            name: len(pools[name]) for name in D_MINUS_ONE_CATEGORIES
        },
        "native_source_sizes": [list(value) for value in sorted(native_sizes)],
        "num_native_source_sizes": len(native_sizes),
        "bridge_target_authority": "deterministic_rule_candidate_not_expert",
        "expert_truth_used": False,
    }


def _append_fraction(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    fraction: float,
    *,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    """Keep every primary row and add a deterministic secondary fraction."""
    if not primary or fraction <= 0 or not secondary:
        return list(primary)
    requested = round(len(primary) * float(fraction) / max(1.0 - float(fraction), 1.0e-8))
    return list(primary) + _stable_subset(secondary, requested, seed, namespace)


def _structured_text(record: dict[str, Any], *, expert: bool) -> str:
    if expert:
        target = record.get("expert_target") or {}
        structured = dict(target.get("structured_output") or {})
        summary = str(target.get("summary") or "")
    else:
        candidate = record.get("candidate") or {}
        structured = dict(candidate.get("structured_output") or {})
        summary = str(candidate.get("summary") or "")
    if not summary.strip():
        raise ValueError(
            "Bridge structured target 缺少非空 summary: "
            f"{description_row_sample_id(record)}"
        )
    output = {
        "schema_version": "qpsalm_description_output_v1",
        "target_status": structured.get("target_status", record.get("target_status", "uncertain")),
        "region": structured.get("region") or {},
        "evidence": structured.get("evidence") or {},
        "summary": summary,
    }
    text = json.dumps(
        output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    parsed = parse_description_output(text)
    if not parsed.schema_valid:
        raise ValueError(
            "Bridge structured target 不符合 qpsalm_description_output_v1: "
            f"sample={description_row_sample_id(record)} errors={parsed.parse_errors}"
        )
    return text


def _has_unavailable_modality(row: dict[str, Any]) -> bool:
    evidence = row.get("modality_evidence") or {}
    if isinstance(evidence, dict):
        values = evidence.values()
    elif isinstance(evidence, (list, tuple)):
        values = evidence
    else:
        return False
    for value in values:
        if not isinstance(value, dict):
            continue
        level = str(value.get("evidence_level") or "")
        status = str(value.get("status") or value.get("availability") or "")
        if level.startswith("C_") or status in {"unavailable", "insufficient"}:
            return True
    return False


def bridge_region_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Build target identity metadata without loading cache features or masks."""
    review_responses = ((row.get("review") or {}).get("reviewer_responses") or [])
    panel_paths = sorted({
        str(value.get("panel_path"))
        for value in review_responses
        if isinstance(value, dict) and value.get("panel_path")
    })
    preview_paths = (row.get("visual_ref") or {}).get("preview_paths") or {}
    return {
        "sample_id": str(row["bridge_record_id"]),
        "parent_sample_id": str(row["parent_sample_id"]),
        "task_family": str(row["task_family"]),
        "target_status": str(row.get("target_status") or "uncertain"),
        "source_dataset": str(row.get("dataset_name") or "unknown"),
        "region_pair_id": None,
        "region_id": str(row.get("region_id") or "unknown"),
        "region_source": str(row.get("region_source") or "unknown"),
        "source_region_aliases": [
            dict(value) for value in (row.get("source_region_aliases") or [])
            if isinstance(value, dict)
        ],
        "region_mask_path": (row.get("region_mask") or {}).get("path"),
        "expert_review_panel_path": panel_paths[0] if panel_paths else None,
        "visual_preview_path": preview_paths.get("visual"),
        "multimodal_preview_path": preview_paths.get("modalities"),
        "has_unavailable_modality": _has_unavailable_modality(row),
    }


def filter_evaluation_source(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    split: str,
    source_dataset: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Freeze a source-specific independent test population before limiting."""
    if source_dataset is None:
        return rows, None
    if stage != "rsicap_caption" or split != "test" or source_dataset != "RSIEval":
        raise ValueError(
            "source-specific description evaluation 只允许 "
            "stage=rsicap_caption split=test source=RSIEval"
        )
    selected = [
        row for row in rows
        if str(row.get("source_dataset") or "") == source_dataset
        and str(row.get("task_family") or "") == "global_caption"
    ]
    if not selected:
        raise ValueError("RSIEval source filter 产生空 evaluation population")
    return selected, {
        "protocol": "qpsalm_description_evaluation_source_filter_v1",
        "stage": stage,
        "split": split,
        "source_dataset": source_dataset,
        "rows_before_filter": len(rows),
        "rows_after_filter": len(selected),
    }


def evaluation_region_source_population_sha256(
    rows: list[dict[str, Any]],
) -> str:
    """Hash the exact region identity population selected before eval limiting."""
    identities = sorted(
        (
            str(row.get("sample_id") or description_row_sample_id(row)),
            str(row.get("parent_sample_id") or ""),
            str(row.get("region_id") or ""),
            str(row.get("region_source") or ""),
        )
        for row in rows
    )
    return hashlib.sha256(
        json.dumps(
            identities,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def filter_evaluation_region_source(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    split: str,
    training: bool,
    evaluation_mode: str,
    region_source: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Freeze the GT-global-mask oracle population before sample limiting."""
    if region_source is None:
        return rows, None
    if (
        training
        or stage != "bridge_expert"
        or split not in {"val", "test"}
        or evaluation_mode not in {"gt_mask", "end_to_end"}
        or region_source != "gt_global_mask"
    ):
        raise ValueError(
            "region-source filter 只允许 frozen bridge_expert val/test 的 "
            "GT-mask/end-to-end gt_global_mask"
        )
    selected = [
        row for row in rows
        if str(row.get("region_source") or "") == region_source
    ]
    if not selected:
        raise ValueError("region-source filter 产生空 evaluation population")
    sample_ids = [description_row_sample_id(row) for row in selected]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("region-source filter 后 sample_id 必须唯一")
    return selected, {
        "protocol": "qpsalm_description_region_source_filter_v1",
        "stage": stage,
        "split": split,
        "evaluation_mode": evaluation_mode,
        "region_source": region_source,
        "rows_before_filter": len(rows),
        "rows_after_filter": len(selected),
        "excluded_rows": len(rows) - len(selected),
        "population_sha256": evaluation_region_source_population_sha256(selected),
    }


class DescriptionTaskDataset(Dataset):
    """One task family per dataset instance; joint training uses separate DataLoaders."""

    def __init__(
        self,
        *,
        stage: DescriptionStage,
        split: str,
        vision_bank: DescriptionVisionFeatureBank,
        description_benchmark: str | Path,
        bridge_benchmark: str | Path,
        predicted_index: str | Path | None = None,
        seed: int = 42,
        max_samples: int = 0,
        training: bool = False,
        evaluation_mode: str = "gt_mask",
        evaluation_source_dataset: str | None = None,
        evaluation_region_source: str | None = None,
        rsicap_mmrs_fraction: float = 0.30,
        predicted_mask_fraction: float = 0.25,
        d4_curriculum_sampling_seed: int = 42,
    ) -> None:
        self.stage = stage
        self.split = split
        self.vision_bank = vision_bank
        self.seed = int(seed)
        self.d4_curriculum_sampling_seed = int(d4_curriculum_sampling_seed)
        self.epoch = 0
        self.training = bool(training)
        self.evaluation_mode = str(evaluation_mode)
        self.source_filter_audit: dict[str, Any] | None = None
        self.region_source_filter_audit: dict[str, Any] | None = None
        self.end_to_end_exclusion_counts: Counter[str] = Counter()
        self.end_to_end_source_count = 0
        self.end_to_end_eligible_count = 0
        self.predicted_index_audit: dict[str, Any] | None = None
        self.d_minus_one_sampling_audit: dict[str, Any] | None = None
        self.bridge_engineering_audit: dict[str, Any] | None = None
        self.description_engineering_audit: dict[str, Any] | None = None
        self._verified_mask_hashes: dict[tuple[str, str], str] = {}
        description_dir = resolve_project_path(description_benchmark)
        bridge_dir = resolve_project_path(bridge_benchmark)
        if description_dir is None or bridge_dir is None:
            raise ValueError("description/bridge benchmark 路径不能为空")
        if stage in {
            "bridge_auto", "bridge_expert", "predicted_mask", "overfit",
        }:
            self.bridge_engineering_audit = require_engineering_bridge(
                bridge_dir, vision_bank
            )
        self.expert_gate_audit = (
            require_frozen_expert_bridge(bridge_dir)
            if stage in {"bridge_expert", "predicted_mask"} else None
        )
        if stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}:
            self.description_engineering_audit = (
                require_engineering_description(description_dir, vision_bank)
            )
            index_name = "train_eligible.jsonl" if split == "train" else f"{split}.jsonl"
            rows = _read_jsonl(description_dir / f"indexes/{index_name}")
            if stage == "mmrs_caption":
                rows = [
                    row for row in rows
                    if row["task_family"] == "global_caption" and str(row["source_dataset"]).startswith("MMRS-")
                ]
            elif stage == "rsicap_caption":
                rsicap_rows = [
                    row for row in rows
                    if row["task_family"] == "global_caption" and row["source_dataset"] in {"RSICap", "RSIEval"}
                ]
                if self.training and split == "train":
                    mmrs_rows = [
                        row for row in _read_jsonl(description_dir / "indexes/train_eligible.jsonl")
                        if row["task_family"] == "global_caption"
                        and str(row["source_dataset"]).startswith("MMRS-")
                    ]
                    rows = _append_fraction(
                        rsicap_rows, mmrs_rows, rsicap_mmrs_fraction,
                        seed=self.seed, namespace="d1_rsicap_mmrs",
                    )
                else:
                    rows = rsicap_rows
            else:
                rows = [row for row in rows if row["task_family"] == "region_referring_expression"]
        elif stage == "bridge_auto":
            rows = _read_jsonl(bridge_dir / "indexes/auto_train.jsonl")
            if split != "train":
                rows = []
        elif stage == "overfit":
            self.description_engineering_audit = (
                require_engineering_description(description_dir, vision_bank)
            )
            description_index = description_dir / "indexes/train_eligible.jsonl"
            bridge_index = bridge_dir / "indexes/candidate_all.jsonl"
            description_report_path = description_dir / "reports/validation_report.json"
            bridge_report_path = bridge_dir / "reports/validation_report.json"
            for report_path in (description_report_path, bridge_report_path):
                if not report_path.is_file():
                    raise FileNotFoundError(f"D-1 缺少 benchmark validation report: {report_path}")
            description_report = strict_json_loads(
                description_report_path.read_text(encoding="utf-8")
            )
            bridge_report = strict_json_loads(
                bridge_report_path.read_text(encoding="utf-8")
            )
            if (
                description_report.get("builder_version")
                != DESCRIPTION_BUILDER_VERSION
                or description_report.get("errors")
            ):
                raise RuntimeError(
                    "D-1 要求 engineering-valid Description M1.1 v4 benchmark"
                )
            if (
                bridge_report.get("builder_version") != BRIDGE_BUILDER_VERSION
                or bridge_report.get("status") not in {
                    "awaiting_expert_review", "expert_pilot_frozen",
                }
                or bridge_report.get("errors")
            ):
                raise RuntimeError(
                    "D-1 要求当前 M2 v7 Bridge prepare/frozen artifact；"
                    "awaiting_expert_review 可用于 candidate 工程过拟合，"
                    "但旧 builder 或有错误的 artifact 不可使用"
                )
            description_rows = _read_jsonl(description_index)
            bridge_rows = _read_jsonl(bridge_index)
            requested = min(64, int(max_samples)) if max_samples > 0 else 64
            rows, self.d_minus_one_sampling_audit = select_d_minus_one_mixture(
                description_rows,
                bridge_rows,
                count=requested,
                seed=self.seed,
            )
            self.d_minus_one_sampling_audit.update({
                "bridge_engineering_audit": self.bridge_engineering_audit,
                "description_builder_version": description_report.get(
                    "builder_version"
                ),
                "description_index": str(description_index),
                "description_index_sha256": _sha256_file(description_index),
                "description_validation_report": str(description_report_path),
                "description_validation_report_sha256": _sha256_file(
                    description_report_path
                ),
                "bridge_builder_version": bridge_report.get("builder_version"),
                "bridge_status": bridge_report.get("status"),
                "bridge_index": str(bridge_index),
                "bridge_index_sha256": _sha256_file(bridge_index),
                "bridge_validation_report": str(bridge_report_path),
                "bridge_validation_report_sha256": _sha256_file(
                    bridge_report_path
                ),
            })
        elif stage == "bridge_expert":
            path = bridge_dir / f"indexes/expert_{split}.jsonl"
            rows = _read_jsonl(path)
        elif stage == "predicted_mask":
            if predicted_index is None:
                raise ValueError("predicted_mask stage 需要独立离线 --predicted-index")
            path = resolve_project_path(predicted_index)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"predicted index 不存在: {predicted_index}")
            self.predicted_index_audit = validate_predicted_index(
                path,
                split=split,
                expert_gate_audit=dict(self.expert_gate_audit or {}),
            )
            rows = _read_jsonl(path)
            rows = [row for row in rows if row.get("split") == split]
            if self.training and split == "train":
                expert_path = bridge_dir / "indexes/expert_train.jsonl"
                if not expert_path.is_file():
                    raise FileNotFoundError(
                        "D4 GT/predicted curriculum 需要已冻结 indexes/expert_train.jsonl"
                    )
                expert_rows = _read_jsonl(expert_path)
                requested_predicted = round(
                    len(expert_rows) * float(predicted_mask_fraction)
                    / max(1.0 - float(predicted_mask_fraction), 1.0e-8)
                )
                if requested_predicted > len(rows):
                    raise RuntimeError(
                        "D4 predicted index 不足以实现预注册 curriculum tier: "
                        f"requested={requested_predicted} available={len(rows)} "
                        f"fraction={predicted_mask_fraction}"
                    )
                rows = expert_rows + _stable_subset(
                    rows,
                    requested_predicted,
                    self.d4_curriculum_sampling_seed,
                    "d4_predicted_masks",
                )
        else:
            raise ValueError(f"未知 description stage={stage!r}")
        rows, self.source_filter_audit = filter_evaluation_source(
            rows,
            stage=stage,
            split=split,
            source_dataset=evaluation_source_dataset,
        )
        rows, self.region_source_filter_audit = filter_evaluation_region_source(
            rows,
            stage=stage,
            split=split,
            training=self.training,
            evaluation_mode=self.evaluation_mode,
            region_source=evaluation_region_source,
        )
        if self.evaluation_mode == "end_to_end":
            if stage != "bridge_expert":
                raise ValueError("end_to_end evaluation 只支持 bridge_expert stage")
            self.end_to_end_source_count = len(rows)
            supported_rows = []
            for row in rows:
                supported, reason = end_to_end_region_support(row)
                if supported:
                    supported_rows.append(row)
                else:
                    self.end_to_end_exclusion_counts[reason] += 1
            rows = supported_rows
            self.end_to_end_eligible_count = len(rows)
        if stage in {"bridge_expert", "predicted_mask"}:
            _validate_expert_rows(rows, stage=stage, split=split)
        if stage != "overfit":
            rows.sort(key=lambda row: str(
                row.get("sample_id") or row.get("bridge_record_id") or ""
            ))
        if max_samples > 0 and stage != "overfit":
            rows = _stable_subset(rows, max_samples, self.seed, f"{stage}:{split}:limit")
            rows.sort(key=lambda row: str(
                row.get("sample_id") or row.get("bridge_record_id") or ""
            ))
        self.rows = rows
        self._rows_by_sample_id: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            sample_id = description_row_sample_id(row)
            if sample_id in self._rows_by_sample_id:
                raise ValueError(f"description dataset sample_id 重复: {sample_id}")
            self._rows_by_sample_id[sample_id] = row
        self._region_swap_catalog = self.rows
        if stage in {"bridge_auto", "bridge_expert", "predicted_mask", "overfit"}:
            candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
            if candidate_path.is_file():
                self._region_swap_catalog = [
                    row for row in _read_jsonl(candidate_path)
                    if str(row.get("split") or "") == self.split
                ]
        self._request_family_cache: dict[tuple[str, str], set[str]] = {}
        (
            self._caption_source_weight_by_sample,
            self.caption_sampling_audit,
        ) = _caption_source_weights(
            self.rows,
            stage=self.stage,
            rsicap_mmrs_fraction=rsicap_mmrs_fraction,
        )
        predicted_rows = [
            row for row in self.rows
            if (
                str(row.get("region_source") or "") == "predicted_proposal"
                or str(row.get("schema_version") or "").startswith(
                    "qpsalm_predicted_region"
                )
            )
        ]
        self.curriculum_audit = (
            {
                "protocol": "qpsalm_d4_predicted_mask_curriculum_v1",
                "requested_predicted_fraction": float(predicted_mask_fraction),
                "selection_seed": self.d4_curriculum_sampling_seed,
                "num_total": len(self.rows),
                "num_predicted": len(predicted_rows),
                "num_gt": len(self.rows) - len(predicted_rows),
                "realized_predicted_fraction": (
                    len(predicted_rows) / len(self.rows) if self.rows else 0.0
                ),
                "training_mix": bool(self.training and self.split == "train"),
            }
            if self.stage == "predicted_mask" else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def same_parent_region_swap(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Load and identify another real region from the same image."""
        candidates = same_parent_region_swap_candidates(
            self.rows,
            sample_id,
            catalog=self._region_swap_catalog,
        )
        current = reference_mask.detach().cpu()
        for row in candidates:
            alternate = self._counterfactual_region_mask(row).detach().cpu()
            if alternate.shape != current.shape:
                continue
            if not torch.equal(alternate, current):
                return alternate, {
                    "protocol": "qpsalm_same_parent_region_swap_v1",
                    "parent_sample_id": str(row["parent_sample_id"]),
                    "alternate_sample_id": description_row_sample_id(row),
                    "alternate_region_id": str(row.get("region_id") or "unknown"),
                    "alternate_region_source": str(
                        row.get("region_source") or "region_geometry"
                    ),
                    "alternate_mask_path": (row.get("region_mask") or {}).get("path"),
                }
        return None

    def cross_parent_region_swap(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Load one real region from a deterministic, different parent."""
        current_row = self._rows_by_sample_id.get(str(sample_id))
        if current_row is None:
            return None
        target_parent = str(current_row.get("parent_sample_id") or "")
        candidates = cross_parent_region_swap_candidates(
            self.rows,
            sample_id,
            catalog=self._region_swap_catalog,
        )
        current = reference_mask.detach().cpu()
        for row in candidates:
            donor_parent = str(row.get("parent_sample_id") or "")
            if not donor_parent or donor_parent == target_parent:
                continue
            alternate = self._counterfactual_region_mask(row).detach().cpu()
            if alternate.shape != current.shape or torch.equal(alternate, current):
                continue
            return alternate, {
                "protocol": "qpsalm_cross_parent_region_swap_v1",
                "target_parent_sample_id": target_parent,
                "donor_parent_sample_id": donor_parent,
                "donor_sample_id": description_row_sample_id(row),
                "donor_region_id": str(row.get("region_id") or "unknown"),
                "donor_region_source": str(
                    row.get("region_source") or "region_geometry"
                ),
                "donor_mask_path": (row.get("region_mask") or {}).get("path"),
            }
        return None

    def same_parent_region_swap_mask(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compatibility convenience returning only the resolved mask."""
        resolved = self.same_parent_region_swap(sample_id, reference_mask)
        return resolved[0] if resolved is not None else None

    def _counterfactual_region_mask(self, row: dict[str, Any]) -> torch.Tensor:
        """Load donor geometry without consuming or requiring its text target."""
        if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}:
            return self._description_item(row)["region_mask"]
        return self._bridge_region_mask(row)

    def _request_for_row(self, row: dict[str, Any]) -> tuple[str, str]:
        component = (
            "single_image"
            if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
            or row.get("_d_minus_one_item_kind") == "description"
            else "multisource_parent"
        )
        return component, str(row["parent_sample_id"])

    def _request_families(self, request: tuple[str, str]) -> set[str]:
        cached = self._request_family_cache.get(request)
        if cached is not None:
            return cached
        record = self.vision_bank.record(*request)
        families = set()
        for view in record["views"]:
            source = [str(value) for value in view.get("source_families") or []]
            family = (
                source[0]
                if source and len(set(source)) == 1 and source[0] in MODALITY_FAMILY_IDS
                else "unknown"
            )
            families.add(family)
        self._request_family_cache[request] = families
        return families

    def cross_parent_modality_swap_request(
        self,
        sample_id: str,
    ) -> tuple[tuple[str, str], dict[str, Any]] | None:
        """Select a deterministic donor parent sharing at least one view family."""
        current = self._rows_by_sample_id.get(str(sample_id))
        if current is None:
            return None
        current_parent = str(current["parent_sample_id"])
        current_request = self._request_for_row(current)
        current_families = self._request_families(current_request)
        seen_parents = {current_parent}
        for row in sorted(self.rows, key=description_row_sample_id):
            donor_parent = str(row.get("parent_sample_id") or "")
            if not donor_parent or donor_parent in seen_parents:
                continue
            seen_parents.add(donor_parent)
            donor_request = self._request_for_row(row)
            common = sorted(current_families & self._request_families(donor_request))
            if not common:
                continue
            return donor_request, {
                "protocol": "qpsalm_cross_parent_modality_donor_v1",
                "target_parent_sample_id": current_parent,
                "donor_parent_sample_id": donor_parent,
                "common_modality_families": common,
            }
        return None

    def _description_item(self, row: dict[str, Any]) -> dict[str, Any]:
        answers = [
            answer for answer in row.get("answers", [])
            if self.split != "train" or float(answer.get("caption_quality_weight", 1.0)) > 0
        ]
        if not answers:
            raise ValueError(f"description record 没有可训练 answer: {row['sample_id']}")
        answer = answers[_stable_weighted_index(
            self.seed,
            self.epoch,
            str(row["sample_id"]),
            [float(value.get("caption_quality_weight", 1.0)) for value in answers],
        )]
        visual = row["visual_ref"]
        width, height = int(visual["width"]), int(visual["height"])
        geometry = row["region_geometry"]
        source_mask = torch.zeros((1, height, width), dtype=torch.float32)
        if geometry["type"] == "full_image":
            source_mask.fill_(1.0)
        elif geometry["type"] == "box":
            x1, y1, x2, y2 = [int(value) for value in geometry["bbox_xyxy_pixel_half_open"]]
            source_mask[:, y1:y2, x1:x2] = 1.0
        elif geometry["type"] not in {"null"}:
            raise ValueError(f"M1 record region type 暂不支持: {geometry['type']}")
        cache = self.vision_bank.record("single_image", str(row["parent_sample_id"]))
        region = transform_region_mask_to_cache(source_mask, cache["views"][0]["render_transform"])
        return {
            "request": ("single_image", str(row["parent_sample_id"])),
            "region_mask": region,
            "instruction": str(row["instruction"]),
            "target_text": str(answer["text"]),
            "reference_texts": [str(value["text"]) for value in answers],
            "structured_output": False,
            "weight": (
                float(answer.get("caption_quality_weight", 1.0))
                * float(self._caption_source_weight_by_sample.get(
                    str(row["sample_id"]), 1.0
                ))
            ),
            "sample_id": str(row["sample_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "task_family": str(row["task_family"]),
            "target_status": str(row.get("target_status") or "present"),
            "source_dataset": str(row.get("source_dataset") or "unknown"),
            "visual_image_path": str(visual.get("path") or ""),
            "region_pair_id": row.get("region_pair_id"),
        }

    def _bridge_region_mask_and_binding(
        self, row: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Materialize Bridge geometry and bind its deterministic cache projection."""
        cache = self.vision_bank.record("multisource_parent", str(row["parent_sample_id"]))
        transform = dict(cache["views"][0]["render_transform"])
        source_binding: dict[str, Any]
        if row.get("region_mask"):
            mask_ref = row["region_mask"]
            path = resolve_project_path(mask_ref["path"])
            if path is None or not path.is_file():
                raise FileNotFoundError(f"region mask 不存在: {mask_ref.get('path')}")
            expected_hash = str(mask_ref.get("sha256") or "")
            observed_hash = ""
            if self.stage == "predicted_mask" and len(expected_hash) != 64:
                raise ValueError("predicted-mask row 缺少 region_mask.sha256")
            if expected_hash:
                key = (str(path.resolve(strict=False)), expected_hash)
                observed_hash = self._verified_mask_hashes.get(key)
                if observed_hash is None:
                    observed_hash = _sha256_file(path)
                    self._verified_mask_hashes[key] = observed_hash
                if observed_hash != expected_hash:
                    raise ValueError(f"region mask hash 不一致: {path}")
            values = np.load(path, allow_pickle=False)
            expected_shape = tuple(int(value) for value in (mask_ref.get("shape") or []))
            if expected_shape and tuple(values.shape) != expected_shape:
                raise ValueError(
                    f"region mask shape 不一致: observed={tuple(values.shape)} "
                    f"expected={expected_shape}"
                )
            if self.stage == "predicted_mask" and (
                values.ndim != 2 or not np.isin(values, (0, 1)).all()
            ):
                raise ValueError("predicted region mask 必须是二维 binary array")
            if values.ndim == 2:
                values = values[None]
            source_mask = torch.from_numpy((values > 0).astype(np.float32))
            source_binding = {
                "kind": "binary_npy",
                "path": str(mask_ref["path"]),
                "file_sha256": observed_hash or _sha256_file(path),
                "bytes": int(path.stat().st_size),
                "shape": list(source_mask.shape[-2:]),
                "positive_pixels": int(source_mask.sum().item()),
            }
        else:
            source_h = int(transform["source_h"])
            source_w = int(transform["source_w"])
            source_mask = torch.zeros((1, source_h, source_w), dtype=torch.float32)
            source_binding = {
                "kind": "null",
                "path": None,
                "file_sha256": None,
                "bytes": 0,
                "shape": [source_h, source_w],
                "positive_pixels": 0,
            }
        projected = transform_region_mask_to_cache(source_mask, transform)
        lookup_key = str(cache.get("lookup_key") or description_cache_key(
            "multisource_parent", str(row["parent_sample_id"])
        ))
        cache_fingerprint = str(cache.get("cache_fingerprint") or hashlib.sha256(
            json.dumps(
                {"lookup_key": lookup_key, "render_transform": transform},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest())
        return projected, {
            "protocol": REGION_INPUT_SOURCE_PROTOCOL,
            "sample_id": str(row["bridge_record_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "region_id": str(row.get("region_id") or "unknown"),
            "region_source": str(row.get("region_source") or "unknown"),
            "cache_lookup_key": lookup_key,
            "cache_fingerprint": cache_fingerprint,
            "render_transform": transform,
            "source_mask": source_binding,
        }

    def _bridge_region_mask(self, row: dict[str, Any]) -> torch.Tensor:
        """Materialize only Bridge geometry; counterfactual donors need no text."""
        return self._bridge_region_mask_and_binding(row)[0]

    def _bridge_item(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.stage in {"bridge_expert", "predicted_mask"} and not isinstance(
            row.get("expert_target"), dict
        ):
            raise ValueError(
                "expert/predicted-mask row 缺少 expert_target；禁止回退到 candidate"
            )
        region, region_source_binding = self._bridge_region_mask_and_binding(row)
        # Predicted-mask rows inherit the reviewed target from their source row.
        # Falling back to the deterministic candidate here would make fixed-mask
        # evaluation measure a different target than the GT-mask expert run.
        expert = self.stage in {"bridge_expert", "predicted_mask"}
        return {
            "request": ("multisource_parent", str(row["parent_sample_id"])),
            "region_mask": region,
            "instruction": str(row["instruction"]),
            "target_text": _structured_text(row, expert=expert),
            "reference_texts": [_structured_text(row, expert=expert)],
            "structured_output": True,
            "weight": 1.0,
            **bridge_region_metadata(row),
            "region_input_source_binding": region_source_binding,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.stage == "overfit":
            item = (
                self._description_item(row)
                if row.get("_d_minus_one_item_kind") == "description"
                else self._bridge_item(row)
            )
            item["d_minus_one_category"] = str(row["_d_minus_one_category"])
            item["d_minus_one_target_authority"] = str(
                row["_d_minus_one_target_authority"]
            )
            return item
        return (
            self._description_item(row)
            if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
            else self._bridge_item(row)
        )


def collate_description(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("description batch 不能为空")
    shapes = {tuple(item["region_mask"].shape) for item in items}
    if len(shapes) != 1:
        raise ValueError(f"description region canvas 必须一致: {sorted(shapes)}")
    return {
        "requests": [item["request"] for item in items],
        "region_masks": torch.stack([item["region_mask"] for item in items]),
        "instructions": [item["instruction"] for item in items],
        "target_texts": [item["target_text"] for item in items],
        "reference_texts": [item["reference_texts"] for item in items],
        "structured_outputs": [bool(item["structured_output"]) for item in items],
        "weights": torch.tensor([float(item["weight"]) for item in items], dtype=torch.float32),
        "metadata": [{
            key: item[key]
            for key in (
                "sample_id", "parent_sample_id", "task_family", "target_status",
                "source_dataset", "region_pair_id", "region_id", "region_source",
                "source_region_aliases", "region_mask_path", "expert_review_panel_path",
                "visual_preview_path", "multimodal_preview_path", "visual_image_path",
                "has_unavailable_modality",
                "region_input_source_binding",
                "d_minus_one_category", "d_minus_one_target_authority",
            )
            if key in item
        } for item in items],
    }
