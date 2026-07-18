#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6 三模式正式评价与 D4 最终课程的统一验收门禁。"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
)
from ..training.checkpoint import (
    description_protocol_assets_spec,
    validate_description_stage_lineage,
)
from .comparison import (
    absolute_candidate_gate,
)
from .counterfactual_gate import counterfactual_gate
from .formal_inputs import (
    formal_seed_binding,
    load_evaluation_rows,
    validate_evaluation_checkpoint_provenance,
    validate_expert_binding,
)
from .d4_curriculum import (
    validate_d4_final_acceptance_for_m7,
)
from ..protocols.io import sha256_file as _sha256_file, strict_json_loads
from ..data.records import evaluation_region_source_population_sha256
from ..data.engineering_contracts import revalidate_predicted_index_audit
from ..data.expert_contracts import load_frozen_scientific_gate
from .d_minus_one import revalidate_saved_d_minus_one_acceptance
from .contracts import (
    DESCRIPTION_EVALUATION_PROTOCOL,
    EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
)
from .artifacts import revalidate_evaluation_mask_artifact
from .targets import END_TO_END_TARGET_PROTOCOL, EndToEndTargetResolver
from .expert_factuality import EXPERT_FACTUALITY_PROTOCOL
from .cycle_localization import (
    CYCLE_LOCALIZATION_PROTOCOL,
    CYCLE_PROMPT_PROTOCOL,
)
from ..data.source_binding import (
    revalidate_segmentation_instruction_source_binding,
)


M6_ACCEPTANCE_GATE_PROTOCOL = (
    "qpsalm_m6_acceptance_v10_strict_json_finite"
)
M6_ACCEPTANCE_AUDIT_PROTOCOL = (
    "qpsalm_m6_acceptance_audit_v10_strict_json_finite"
)


def _resolved_file(value: Any, *, label: str) -> Path:
    path = resolve_project_path(str(value or ""))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"M6 {label} 不存在: {value}")
    return path.resolve(strict=False)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"M6 {label} 不是合法 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"M6 {label} 顶层必须为 object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            strict_json_loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"M6 {label} 不是合法 JSONL: {path}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"M6 {label} 每行必须为 object: {path}")
    return values


def _require_mapping_match(
    expected: dict[str, Any], observed: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(observed, dict):
        raise ValueError(f"M6 {label} 缺少 segmentation target mapping")
    mismatched = [
        key for key, value in expected.items() if observed.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"M6 {label} 与 segmentation instruction source 重放不一致: "
            f"{mismatched}"
        )
    return observed


def _revalidate_end_to_end_target_audit(
    *,
    report: dict[str, Any],
    generation_rows: dict[str, dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay every online Bridge-to-segmentation mapping from its source index."""
    coverage = dict(report.get("end_to_end_coverage") or {})
    if coverage.get("protocol") != END_TO_END_TARGET_PROTOCOL:
        raise ValueError("M6 end-to-end target protocol 不兼容")
    source_binding = coverage.get("segmentation_source_binding")
    source_rows = revalidate_segmentation_instruction_source_binding(
        source_binding
    )
    resolver = EndToEndTargetResolver(source_rows)
    by_sample: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        sample_id = str(row.get("bridge_sample_id") or "")
        if not sample_id or sample_id in by_sample:
            raise ValueError("M6 end-to-end target audit bridge_sample_id 非空且唯一")
        by_sample[sample_id] = row
    if set(by_sample) != set(generation_rows):
        raise ValueError("M6 end-to-end target audit 与 generation population 不一致")

    mapping_counts: Counter[str] = Counter()
    segmentation_samples: set[str] = set()
    threshold = float(coverage.get("mask_threshold", -1.0))
    for sample_id, generation in generation_rows.items():
        standalone = by_sample[sample_id]
        nested = generation.get("end_to_end_segmentation_target")
        if standalone != nested:
            raise ValueError(
                "M6 end-to-end standalone audit 与 raw generation 内嵌映射不一致"
            )
        if standalone.get("region_input_mask_artifact") != generation.get(
            "region_input_mask_artifact"
        ):
            raise ValueError(
                "M6 end-to-end target audit 与实际 description region mask 不一致"
            )
        if standalone.get("region_input_source_binding") != generation.get(
            "region_input_source_binding"
        ):
            raise ValueError(
                "M6 end-to-end target audit 与 source-space online mask 不一致"
            )
        expected = resolver.resolve(generation)
        _require_mapping_match(expected, standalone, label=f"end-to-end:{sample_id}")
        if (
            str(standalone.get("bridge_region_source") or "")
            != "gt_global_mask"
            or str(standalone.get("mapping_kind") or "")
            != "global_instruction"
            or not isinstance(standalone.get("segmentation_resize_transform"), dict)
            or not isinstance(standalone.get("description_render_transform"), dict)
            or not math.isclose(
                float(standalone.get("mask_threshold", -2.0)),
                threshold,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("M6 end-to-end target audit 几何或阈值绑定非法")
        mapping_counts[str(standalone["mapping_kind"])] += 1
        segmentation_samples.add(str(standalone["segmentation_sample_id"]))

    expected_count = len(generation_rows)
    if (
        int(coverage.get("source_bridge_rows", -1)) != expected_count
        or int(coverage.get("eligible_bridge_rows_before_limit", -1))
        != expected_count
        or int(coverage.get("evaluated_rows", -1)) != expected_count
        or dict(coverage.get("excluded_by_reason") or {})
        or dict(coverage.get("mapping_counts") or {})
        != dict(sorted(mapping_counts.items()))
        or int(coverage.get("unique_segmentation_inferences", -1))
        != len(segmentation_samples)
    ):
        raise ValueError("M6 end-to-end coverage 无法由逐条 target audit 重算")
    return {
        "protocol": END_TO_END_TARGET_PROTOCOL,
        "segmentation_source_binding": source_binding,
        "num_rows": expected_count,
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "unique_segmentation_inferences": len(segmentation_samples),
        "mask_threshold": threshold,
    }


def _revalidate_cycle_localization(
    *,
    evaluation_root: Path,
    report: dict[str, Any],
    generation_rows: dict[str, dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute full-population cycle identity and pixel metrics from raw rows."""
    summary = dict(report.get("cycle_localization") or {})
    if summary.get("protocol") != CYCLE_LOCALIZATION_PROTOCOL:
        raise ValueError("M6 cycle localization protocol 不兼容")
    source_binding = summary.get("segmentation_source_binding")
    source_rows = revalidate_segmentation_instruction_source_binding(
        source_binding
    )
    resolver = EndToEndTargetResolver(source_rows)
    by_sample: dict[str, dict[str, Any]] = {}
    for row in cycle_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in by_sample:
            raise ValueError("M6 cycle localization sample_id 非空且唯一")
        by_sample[sample_id] = row
    if set(by_sample) != set(generation_rows):
        raise ValueError("M6 cycle localization 未覆盖完整 GT generation population")

    grouped: dict[str, list[float]] = defaultdict(list)
    threshold: float | None = None
    for sample_id, generation in generation_rows.items():
        row = by_sample[sample_id]
        if str(row.get("parent_sample_id") or "") != str(
            generation.get("parent_sample_id") or ""
        ):
            raise ValueError("M6 cycle localization parent identity 不一致")
        cycle_audit = row.get("cycle_audit")
        if not isinstance(cycle_audit, dict) or (
            cycle_audit.get("protocol") != CYCLE_PROMPT_PROTOCOL
        ):
            raise ValueError("M6 cycle localization 缺少 raw-text prompt audit")
        expected = resolver.resolve(generation)
        _require_mapping_match(
            expected,
            cycle_audit.get("target_mapping"),
            label=f"cycle:{sample_id}",
        )
        raw = str(generation.get("raw_generation") or "")
        if (
            cycle_audit.get("generated_text_sha256")
            != hashlib.sha256(raw.encode("utf-8")).hexdigest()
            or int(cycle_audit.get("generated_text_characters", -1)) != len(raw)
        ):
            raise ValueError("M6 cycle localization raw generation binding 已漂移")
        current_threshold = float(cycle_audit.get("mask_threshold", -1.0))
        if threshold is None:
            threshold = current_threshold
        elif not math.isclose(
            current_threshold, threshold, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("M6 cycle localization mask threshold 不一致")

        _prediction_binding, prediction = revalidate_evaluation_mask_artifact(
            evaluation_root,
            row.get("prediction_mask_artifact"),
            expected_role="cycle_prediction",
            expected_sample_id=sample_id,
        )
        _target_binding, target = revalidate_evaluation_mask_artifact(
            evaluation_root,
            row.get("target_mask_artifact"),
            expected_role="cycle_target",
            expected_sample_id=sample_id,
        )
        if prediction.shape != target.shape:
            raise ValueError("M6 cycle prediction/target mask artifact shape 不一致")
        predicted_bool = prediction.astype(bool, copy=False)
        target_bool = target.astype(bool, copy=False)
        intersection = int(np.logical_and(predicted_bool, target_bool).sum())
        union = int(np.logical_or(predicted_bool, target_bool).sum())
        target_pixels = int(target_bool.sum())
        predicted_pixels = int(predicted_bool.sum())
        if (
            int(row.get("intersection_pixels", -1)) != intersection
            or int(row.get("union_pixels", -1)) != union
            or int(row.get("target_pixels", -1)) != target_pixels
            or int(row.get("predicted_pixels", -1)) != predicted_pixels
        ):
            raise ValueError("M6 cycle localization 像素计数无法从 mask artifacts 重算")
        expected_iou = intersection / union if union else 1.0
        if (
            not math.isclose(
                float(row.get("region_iou", -1.0)),
                expected_iou,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or bool(row.get("target_empty")) != (target_pixels == 0)
            or bool(row.get("prediction_empty")) != (predicted_pixels == 0)
            or bool(row.get("empty_target_correct"))
            != (target_pixels == 0 and predicted_pixels == 0)
        ):
            raise ValueError("M6 cycle localization 派生指标无法从像素计数重算")
        grouped[str(row["parent_sample_id"])].append(expected_iou)

    parent_values = [
        sum(values) / len(values) for _parent, values in sorted(grouped.items())
    ]
    parent_macro = sum(parent_values) / len(parent_values) if parent_values else None
    expected_count = len(generation_rows)
    if (
        summary.get("coverage_complete") is not True
        or int(summary.get("source_bridge_rows", -1)) != expected_count
        or int(summary.get("eligible_bridge_rows", -1)) != expected_count
        or int(summary.get("target_evaluations", -1)) != expected_count
        or int(summary.get("evaluated_samples", -1)) != expected_count
        or int(summary.get("evaluated_parents", -1)) != len(parent_values)
        or int(summary.get("requested", -1)) != 0
        or not math.isclose(
            float(summary.get("parent_macro_region_iou", -1.0)),
            float(parent_macro if parent_macro is not None else -2.0),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("M6 cycle localization summary 无法由完整 population 重算")
    return {
        "protocol": CYCLE_LOCALIZATION_PROTOCOL,
        "segmentation_source_binding": source_binding,
        "num_rows": expected_count,
        "num_parents": len(parent_values),
        "parent_macro_region_iou": parent_macro,
        "mask_threshold": threshold,
    }


def _evaluation_artifact(
    *,
    evaluation_dir: str | Path,
    expert_report: str | Path,
    expected_stage: str,
    expected_checkpoint_stage: str,
    expected_mode: str,
    expected_seed: int,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    root = resolve_project_path(evaluation_dir) or Path(evaluation_dir)
    rows, report = load_evaluation_rows(root, require_complete_generation=True)
    if (
        report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL
        or report.get("stage") != expected_stage
        or report.get("split") != "val"
        or report.get("evaluation_mode") != expected_mode
        or report.get("region_protocol") != "vision_only"
        or report.get("expert_gate_audit") != frozen["audit"]
        or not rows
    ):
        raise ValueError(
            f"M6 {expected_mode} 必须是 frozen expert val 的完整 Vision-only evaluation"
        )
    checkpoint_payload_provenance = validate_evaluation_checkpoint_provenance(
        root, report
    )
    checkpoint_binding = dict(report.get("checkpoint_binding") or {})
    if (
        checkpoint_binding.get("protocol")
        != EVALUATION_CHECKPOINT_BINDING_PROTOCOL
        or checkpoint_binding.get("checkpoint_stage")
        != expected_checkpoint_stage
        or checkpoint_binding.get("seed_match") is not True
    ):
        raise ValueError(f"M6 {expected_mode} checkpoint binding 非法")
    region_filter = report.get("region_source_filter_audit")
    if expected_mode in {"gt_mask", "end_to_end"}:
        if (
            not isinstance(region_filter, dict)
            or region_filter.get("protocol")
            != "qpsalm_description_region_source_filter_v1"
            or region_filter.get("region_source") != "gt_global_mask"
            or int(region_filter.get("rows_after_filter", -1)) != len(rows)
            or any(
                str(row.get("region_source") or "") != "gt_global_mask"
                for row in rows.values()
            )
            or region_filter.get("population_sha256")
            != evaluation_region_source_population_sha256(list(rows.values()))
        ):
            raise ValueError(
                f"M6 {expected_mode} 缺少 gt_global_mask population filter"
            )
    elif region_filter is not None:
        raise ValueError("M6 fixed_prediction 不应伪装成 GT region-source filter")
    predicted_artifact_provenance = None
    if expected_mode == "fixed_prediction":
        predicted_artifact_provenance = revalidate_predicted_index_audit(
            report.get("predicted_index_audit"),
            expected_split="val",
            expert_gate_audit=frozen["audit"],
        )
        saved_migration = dict(
            (checkpoint_payload_provenance.get("checkpoint_payload_provenance") or {})
            .get("checkpoint_metadata", {})
            .get("segmentation_migration", {})
        )
        if (
            predicted_artifact_provenance.get(
                "segmentation_checkpoint_sha256"
            )
            != saved_migration.get("source_sha256")
        ):
            raise ValueError(
                "M6 fixed prediction masks 与 description checkpoint 的 segmentation source 不一致"
            )
    elif report.get("predicted_index_audit") is not None:
        raise ValueError(
            f"M6 {expected_mode} 不应携带 fixed predicted-index audit"
        )
    seed_binding = formal_seed_binding(
        report, expected_seed=expected_seed, label=f"m6:{expected_mode}:{root}"
    )
    checkpoint = _resolved_file(report.get("checkpoint"), label="checkpoint")
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 != str(report.get("checkpoint_sha256") or ""):
        raise ValueError(f"M6 {expected_mode} checkpoint SHA-256 已漂移")
    checkpoint_metadata = dict(report.get("checkpoint_metadata") or {})
    metadata = dict(checkpoint_metadata.get("metadata") or {})
    checkpoint_config = require_serialized_segdesc_config(
        metadata.get("config"), label=f"M6 {expected_mode} checkpoint config"
    )
    checkpoint_description_benchmark = serialized_segdesc_config_value(
        checkpoint_config, "description_benchmark"
    )
    if not isinstance(checkpoint_description_benchmark, str) or not (
        checkpoint_description_benchmark.strip()
    ):
        raise ValueError(
            f"M6 {expected_mode} checkpoint 缺少 description_benchmark binding"
        )
    if str(metadata.get("stage") or "") != expected_checkpoint_stage:
        raise ValueError(f"M6 {expected_mode} checkpoint metadata stage 不一致")
    d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
        metadata.get("d_minus_one_acceptance"),
        expected_description_benchmark=checkpoint_description_benchmark,
    )
    stage_lineage = validate_description_stage_lineage(
        metadata.get("stage_lineage"),
        expected_target_stage=expected_checkpoint_stage,
    )
    protocol_assets = checkpoint_metadata.get("description_protocol_assets")
    if protocol_assets != description_protocol_assets_spec():
        raise ValueError(
            f"M6 {expected_mode} checkpoint ontology/schema binding 已漂移"
        )

    expert_path = _resolved_file(expert_report, label=f"{expected_mode} expert report")
    expert = _read_json(expert_path, label=f"{expected_mode} expert report")
    if expert.get("protocol") != EXPERT_FACTUALITY_PROTOCOL:
        raise ValueError(f"M6 {expected_mode} expert report protocol 不兼容")
    validate_expert_binding(
        expert,
        root,
        expert_report_path=expert_path,
        label=f"M6 {expected_mode}",
    )
    parents = {str(row["parent_sample_id"]) for row in rows.values()}
    if set(expert.get("per_parent_scores") or {}) != parents:
        raise ValueError(f"M6 {expected_mode} expert/report parent population 不一致")
    absolute_gate = absolute_candidate_gate(
        report, expert, frozen["thresholds"]
    )
    counterfactual_path = root / "counterfactual_generations.jsonl"
    counterfactual_rows = _read_jsonl(
        counterfactual_path, label=f"{expected_mode} counterfactual generations"
    )
    counterfactual_audit = counterfactual_gate(
        report,
        counterfactual_rows,
        frozen["scientific_protocol"],
        rows,
    )
    return {
        "evaluation_mode": expected_mode,
        "evaluation_dir": str(root.resolve(strict=False)),
        "evaluation_report": str((root / "eval_report.json").resolve(strict=False)),
        "evaluation_report_sha256": _sha256_file(root / "eval_report.json"),
        "raw_generations": str((root / "raw_generations.jsonl").resolve(strict=False)),
        "raw_generations_sha256": _sha256_file(root / "raw_generations.jsonl"),
        "expert_report": str(expert_path),
        "expert_report_sha256": _sha256_file(expert_path),
        "counterfactual_generations": str(
            counterfactual_path.resolve(strict=False)
        ),
        "counterfactual_generations_sha256": _sha256_file(counterfactual_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_seed_binding": seed_binding,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_payload_provenance": checkpoint_payload_provenance,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
        "description_protocol_assets": protocol_assets,
        "sample_ids": sorted(rows),
        "parent_ids": sorted(parents),
        "num_samples": len(rows),
        "region_source_filter_audit": region_filter,
        "predicted_artifact_provenance": predicted_artifact_provenance,
        "absolute_factuality_gate": absolute_gate,
        "counterfactual_gate": counterfactual_audit,
        "evaluation_mask_artifacts": report.get("evaluation_mask_artifacts"),
        "passed": bool(
            absolute_gate.get("passed") and counterfactual_audit.get("passed")
        ),
    }


def build_m6_acceptance_gate(
    *,
    gt_evaluation_dir: str | Path,
    gt_expert_report: str | Path,
    fixed_evaluation_dir: str | Path,
    fixed_expert_report: str | Path,
    end_to_end_evaluation_dir: str | Path,
    end_to_end_expert_report: str | Path,
    bridge_benchmark: str | Path,
    d4_final_gate: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Bind GT oracle, fixed prediction and online end-to-end M6 evidence."""
    bridge_root = resolve_project_path(bridge_benchmark) or Path(bridge_benchmark)
    frozen = load_frozen_scientific_gate(bridge_root)
    gt = _evaluation_artifact(
        evaluation_dir=gt_evaluation_dir,
        expert_report=gt_expert_report,
        expected_stage="bridge_expert",
        expected_checkpoint_stage="bridge_expert",
        expected_mode="gt_mask",
        expected_seed=int(seed),
        frozen=frozen,
    )
    fixed = _evaluation_artifact(
        evaluation_dir=fixed_evaluation_dir,
        expert_report=fixed_expert_report,
        expected_stage="predicted_mask",
        expected_checkpoint_stage="predicted_mask",
        expected_mode="fixed_prediction",
        expected_seed=int(seed),
        frozen=frozen,
    )
    end_to_end = _evaluation_artifact(
        evaluation_dir=end_to_end_evaluation_dir,
        expert_report=end_to_end_expert_report,
        expected_stage="bridge_expert",
        expected_checkpoint_stage="predicted_mask",
        expected_mode="end_to_end",
        expected_seed=int(seed),
        frozen=frozen,
    )
    if gt["parent_ids"] != fixed["parent_ids"]:
        raise ValueError("M6 GT-mask 与 fixed-prediction expert parent population 不一致")
    if end_to_end["parent_ids"] != fixed["parent_ids"]:
        raise ValueError("M6 end-to-end 与 fixed-prediction expert parent population 不一致")
    if any(
        value["num_samples"] != len(value["parent_ids"])
        for value in (gt, fixed, end_to_end)
    ):
        raise ValueError("M6 三模式要求每个 gt_global_mask parent 恰好一个 evaluation row")
    if fixed["checkpoint_sha256"] != end_to_end["checkpoint_sha256"]:
        raise ValueError("M6 fixed/end-to-end 未使用同一 D4 final checkpoint")
    if not (
        gt["d_minus_one_acceptance"]
        == fixed["d_minus_one_acceptance"]
        == end_to_end["d_minus_one_acceptance"]
    ):
        raise ValueError("M6 三模式未继承同一个 D-1 acceptance")
    if not (
        gt["description_protocol_assets"]
        == fixed["description_protocol_assets"]
        == end_to_end["description_protocol_assets"]
    ):
        raise ValueError("M6 三模式 ontology/schema assets 不一致")

    fixed_metadata = dict(fixed["checkpoint_metadata"].get("metadata") or {})
    train_region_audit = fixed_metadata.get("region_data_audit")
    fixed_report = _read_json(
        Path(fixed["evaluation_report"]), label="fixed evaluation report"
    )
    predicted_val_audit = fixed_report.get("predicted_index_audit")
    if not isinstance(train_region_audit, dict) or not isinstance(
        predicted_val_audit, dict
    ):
        raise ValueError("M6 fixed evaluation 缺少 D4 train/fixed-val audit")
    d4_acceptance = validate_d4_final_acceptance_for_m7(
        d4_final_gate,
        seed=int(seed),
        initialize_from=fixed["checkpoint"],
        expert_gate_audit=frozen["audit"],
        train_region_data_audit=train_region_audit,
        val_predicted_index_audit=predicted_val_audit,
    )
    if (
        d4_acceptance.get("m4_suite_acceptance", {}).get(
            "candidate_checkpoint_sha256"
        )
        != gt["checkpoint_sha256"]
    ):
        raise ValueError("M6 GT-mask checkpoint 不是 M4 suite 接受的 D3b full-MGRR")

    cycle_path = Path(gt["evaluation_dir"]) / "cycle_localization.jsonl"
    cycle_rows = _read_jsonl(cycle_path, label="cycle localization")
    gt_report = _read_json(
        Path(gt["evaluation_report"]), label="GT evaluation report"
    )
    gt_generation_rows, _ = load_evaluation_rows(
        gt["evaluation_dir"], require_complete_generation=True
    )
    cycle_validation = _revalidate_cycle_localization(
        evaluation_root=Path(gt["evaluation_dir"]),
        report=gt_report,
        generation_rows=gt_generation_rows,
        cycle_rows=cycle_rows,
    )
    end_audit_path = (
        Path(end_to_end["evaluation_dir"]) / "end_to_end_target_audit.jsonl"
    )
    end_audit_rows = _read_jsonl(end_audit_path, label="end-to-end target audit")
    end_report = _read_json(
        Path(end_to_end["evaluation_report"]), label="end-to-end evaluation report"
    )
    end_generation_rows, _ = load_evaluation_rows(
        end_to_end["evaluation_dir"], require_complete_generation=True
    )
    end_to_end_validation = _revalidate_end_to_end_target_audit(
        report=end_report,
        generation_rows=end_generation_rows,
        audit_rows=end_audit_rows,
    )
    if (
        cycle_validation["segmentation_source_binding"]
        != end_to_end_validation["segmentation_source_binding"]
    ):
        raise ValueError("M6 GT cycle 与 end-to-end 未复用同一 segmentation source")
    checks = {
        "gt_factuality_gate_passed": gt["absolute_factuality_gate"]["passed"],
        "fixed_factuality_gate_passed": fixed[
            "absolute_factuality_gate"
        ]["passed"],
        "end_to_end_factuality_gate_passed": end_to_end[
            "absolute_factuality_gate"
        ]["passed"],
        "gt_counterfactual_gate_passed": gt["counterfactual_gate"]["passed"],
        "fixed_counterfactual_gate_passed": fixed["counterfactual_gate"]["passed"],
        "end_to_end_counterfactual_gate_passed": end_to_end[
            "counterfactual_gate"
        ]["passed"],
        "gt_fixed_parent_population_identical": gt["parent_ids"] == fixed["parent_ids"],
        "end_to_end_parent_population_identical": (
            end_to_end["parent_ids"] == fixed["parent_ids"]
        ),
        "one_row_per_parent_in_all_modes": all(
            value["num_samples"] == len(value["parent_ids"])
            for value in (gt, fixed, end_to_end)
        ),
        "fixed_and_end_to_end_checkpoint_identical": (
            fixed["checkpoint_sha256"] == end_to_end["checkpoint_sha256"]
        ),
        "d4_final_acceptance_passed": d4_acceptance.get("passed") is True,
        "cycle_localization_complete": cycle_validation["num_rows"] > 0,
        "end_to_end_target_audit_complete": (
            end_to_end_validation["num_rows"] > 0
        ),
        "cycle_and_end_to_end_segmentation_source_identical": True,
    }
    errors = [name for name, passed in checks.items() if not passed]
    inputs = {
        "gt_evaluation_dir": str(Path(gt["evaluation_dir"])),
        "gt_expert_report": str(Path(gt["expert_report"])),
        "fixed_evaluation_dir": str(Path(fixed["evaluation_dir"])),
        "fixed_expert_report": str(Path(fixed["expert_report"])),
        "end_to_end_evaluation_dir": str(Path(end_to_end["evaluation_dir"])),
        "end_to_end_expert_report": str(Path(end_to_end["expert_report"])),
        "bridge_benchmark": str(bridge_root.resolve(strict=False)),
        "d4_final_gate": str(_resolved_file(d4_final_gate, label="D4 final gate")),
        "seed": int(seed),
    }
    return {
        "protocol": M6_ACCEPTANCE_GATE_PROTOCOL,
        "status": "engineering-valid" if not errors else "engineering-invalid",
        "passed": not errors,
        "seed": int(seed),
        "inputs": inputs,
        "checks": checks,
        "errors": errors,
        "frozen_gate_audit": frozen["audit"],
        "d_minus_one_acceptance": fixed["d_minus_one_acceptance"],
        "d4_final_acceptance": d4_acceptance,
        "gt_mask": gt,
        "fixed_prediction": fixed,
        "end_to_end": end_to_end,
        "cycle_localization": {
            "path": str(cycle_path.resolve(strict=False)),
            "sha256": _sha256_file(cycle_path),
            "validation": cycle_validation,
        },
        "end_to_end_target_audit": {
            "path": str(end_audit_path.resolve(strict=False)),
            "sha256": _sha256_file(end_audit_path),
            "validation": end_to_end_validation,
        },
    }


def validate_m6_acceptance_gate(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Reproduce one M6 artifact without conflating replay with acceptance.

    A fully bound ``passed=false`` report is a valid scientific outcome.  It is
    publishable for audit, but :func:`validate_m6_acceptance_for_m7` separately
    enforces the positive gate before returning an initialization audit.
    """
    gate_path = _resolved_file(path, label="acceptance gate")
    gate = _read_json(gate_path, label="acceptance gate")
    if gate.get("protocol") != M6_ACCEPTANCE_GATE_PROTOCOL:
        raise ValueError("M6 acceptance gate protocol 不兼容")
    inputs = dict(gate.get("inputs") or {})
    rebuilt = build_m6_acceptance_gate(
        gt_evaluation_dir=inputs.get("gt_evaluation_dir") or "",
        gt_expert_report=inputs.get("gt_expert_report") or "",
        fixed_evaluation_dir=inputs.get("fixed_evaluation_dir") or "",
        fixed_expert_report=inputs.get("fixed_expert_report") or "",
        end_to_end_evaluation_dir=inputs.get("end_to_end_evaluation_dir") or "",
        end_to_end_expert_report=inputs.get("end_to_end_expert_report") or "",
        bridge_benchmark=inputs.get("bridge_benchmark") or "",
        d4_final_gate=inputs.get("d4_final_gate") or "",
        seed=int(inputs.get("seed", -1)),
    )
    if rebuilt != gate:
        raise ValueError("M6 acceptance gate 与绑定源重新计算结果不一致")
    return gate_path, gate


def validate_m6_acceptance_for_m7(
    path: str | Path,
    *,
    seed: int,
    initialize_from: str | Path,
    train_region_data_audit: dict[str, Any],
) -> dict[str, Any]:
    gate_path, gate = validate_m6_acceptance_gate(path)
    if gate.get("passed") is not True or gate.get("errors"):
        raise ValueError("M6 acceptance gate 尚未通过，不能授权 M7")
    if int(gate.get("seed", -1)) != int(seed):
        raise ValueError("M6 acceptance seed 与 M7 run 不一致")
    checkpoint = _resolved_file(initialize_from, label="M7 initialize-from checkpoint")
    fixed = dict(gate.get("fixed_prediction") or {})
    if (
        str(checkpoint) != fixed.get("checkpoint")
        or _sha256_file(checkpoint) != fixed.get("checkpoint_sha256")
    ):
        raise ValueError("M7 initialize-from 不是 M6 三模式验收的 D4 checkpoint")
    d4 = dict(gate.get("d4_final_acceptance") or {})
    if d4.get("source_train_region_data_audit") != train_region_data_audit:
        raise ValueError("M7 region train data 与 M6 acceptance 不一致")

    # M7 三种子聚合只比较科学人口，不比较每个 seed 各自不同的 checkpoint。
    # 因此把三种评价模式的 parent population 显式压缩进可重算 audit。
    population_audit = {
        mode: {
            "num_parents": len(gate[mode]["parent_ids"]),
            "parent_ids_sha256": hashlib.sha256(
                "\n".join(gate[mode]["parent_ids"]).encode("utf-8")
            ).hexdigest(),
        }
        for mode in ("gt_mask", "fixed_prediction", "end_to_end")
    }
    return {
        "protocol": M6_ACCEPTANCE_AUDIT_PROTOCOL,
        "passed": True,
        "gate": str(gate_path),
        "gate_sha256": _sha256_file(gate_path),
        "seed": int(seed),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "d_minus_one_acceptance": gate["d_minus_one_acceptance"],
        "d4_final_acceptance": d4,
        "frozen_gate_audit": gate["frozen_gate_audit"],
        "evaluation_parent_populations": population_audit,
        "segmentation_instruction_source_binding": gate[
            "end_to_end_target_audit"
        ]["validation"]["segmentation_source_binding"],
        "gt_mask_checkpoint_sha256": gate["gt_mask"]["checkpoint_sha256"],
        "fixed_prediction_checkpoint_sha256": fixed["checkpoint_sha256"],
        "end_to_end_checkpoint_sha256": gate["end_to_end"]["checkpoint_sha256"],
    }


def revalidate_saved_m6_acceptance(
    saved: Any,
    *,
    seed: int,
    train_region_data_audit: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(saved, dict) or saved.get("passed") is not True:
        raise ValueError("joint checkpoint 缺少 M6 acceptance audit")
    rebuilt = validate_m6_acceptance_for_m7(
        saved.get("gate") or "",
        seed=int(seed),
        initialize_from=saved.get("source_checkpoint") or "",
        train_region_data_audit=train_region_data_audit,
    )
    if rebuilt != saved:
        raise ValueError("joint checkpoint 保存的 M6 acceptance audit 已漂移")
    return rebuilt
