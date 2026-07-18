#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep validation for one M7 full-val retention gate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qpsalm_seg.engine.evaluator import (
    SEGMENTATION_EVAL_MANIFEST_PROTOCOL,
    SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL,
    validate_segmentation_prediction_population,
)
from qpsalm_seg.engine.threshold import normalize_thresholds
from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
    serialized_segdesc_config_without,
)
from ..data.vision_cache import revalidate_description_cache_artifact
from ..training.checkpoint import (
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
    validate_description_stage_lineage,
)
from ..training.joint_lifecycle import (
    revalidate_joint_initialization_audit,
    validate_joint_checkpoint_execution,
)
from ..training.run_artifacts import validate_checkpoint_run_completion
from ..protocols.io import canonical_sha256
from ..protocols.versions import JOINT_TRAINING_COMPLETION_PROTOCOL
from .d4_curriculum import (
    D4_FINAL_FRACTION,
    revalidate_saved_d4_final_acceptance,
)
from .d_minus_one import revalidate_saved_d_minus_one_acceptance
from .m6_acceptance import revalidate_saved_m6_acceptance
from .retention_contracts import (
    M7_RUN_LOCAL_CONFIG_FIELDS,
    RETENTION_EVAL_BINDING_PROTOCOL,
    RETENTION_GATE_PROTOCOL,
    bound_file,
    finite_float,
    integer,
    json_object,
    sha256_file,
    validate_eval_report_matches_gate,
    validate_population,
)
from .retention_inputs import (
    build_baseline_checkpoint_replay_audit,
    segmentation_metric_input_population,
)


def validate_baseline_binding(gate: dict[str, Any]) -> dict[str, Any]:
    binding = gate.get("baseline_binding")
    if not isinstance(binding, dict) or binding.get("valid") is not True:
        raise ValueError("retention gate 缺少有效 baseline_binding")
    if str(binding.get("split") or "") != "val":
        raise ValueError("retention baseline 必须绑定 val split")
    report_path, report_sha256 = bound_file(
        binding.get("eval_report"),
        binding.get("eval_report_sha256"),
        label="baseline eval report",
    )
    manifest_path, manifest_sha256 = bound_file(
        binding.get("eval_manifest"),
        binding.get("eval_manifest_sha256"),
        label="baseline eval manifest",
    )
    checkpoint_path, checkpoint_sha256 = bound_file(
        binding.get("checkpoint"),
        binding.get("checkpoint_sha256"),
        label="baseline checkpoint",
    )
    report = json_object(report_path, label="baseline eval report")
    prediction_population = validate_segmentation_prediction_population(
        report.get("prediction_population")
    )
    manifest = json_object(manifest_path, label="baseline eval manifest")
    validate_eval_report_matches_gate(report, gate, prefix="baseline")
    resolved = dict(manifest.get("resolved_config") or {})
    report_binding = manifest.get("eval_report_binding")
    if not isinstance(report_binding, dict):
        raise ValueError("baseline eval manifest 缺少 report 字节绑定")
    report_threshold = finite_float(
        report.get("threshold"), label="baseline report threshold"
    )
    resolved_threshold = finite_float(
        resolved.get("eval_threshold"), label="baseline resolved threshold"
    )
    binding_threshold = finite_float(
        report_binding.get("eval_threshold"),
        label="baseline manifest report-binding threshold",
    )
    resolved_sweep = normalize_thresholds(resolved.get("threshold_sweep"))
    binding_sweep = normalize_thresholds(
        report_binding.get("threshold_sweep")
    )
    sweep_report = report.get("threshold_sweep") or {}
    observed_sweep_keys = set(
        dict(sweep_report.get("overall_by_threshold") or {})
    ) if isinstance(sweep_report, dict) else set()
    expected_sweep_keys = {f"{value:.2f}" for value in resolved_sweep}
    manifest_checkpoint = resolve_project_path(str(manifest.get("checkpoint") or ""))
    manifest_report_path = resolve_project_path(str(report_binding.get("path") or ""))
    if (
        manifest.get("protocol") != SEGMENTATION_EVAL_MANIFEST_PROTOCOL
        or manifest.get("created_by") != "qpsalm-eval"
        or report_binding.get("protocol")
        != SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL
        or manifest_report_path is None
        or manifest_report_path.resolve(strict=False) != report_path
        or str(report_binding.get("sha256") or "") != report_sha256
        or str(report_binding.get("prediction_population_sha256") or "")
        != str(prediction_population.get("sha256") or "")
        or not (0.0 <= report_threshold <= 1.0)
        or abs(report_threshold - resolved_threshold) > 1.0e-12
        or abs(report_threshold - binding_threshold) > 1.0e-12
        or abs(
            report_threshold
            - finite_float(
                binding.get("eval_threshold"),
                label="retention baseline binding threshold",
            )
        ) > 1.0e-12
        or abs(
            report_threshold
            - finite_float(
                prediction_population.get("threshold"),
                label="baseline prediction population threshold",
            )
        ) > 1.0e-12
        or not isinstance(resolved.get("threshold_sweep"), list)
        or not isinstance(report_binding.get("threshold_sweep"), list)
        or not isinstance(report.get("threshold_sweep"), dict)
        or binding_sweep != resolved_sweep
        or binding.get("threshold_sweep") != resolved_sweep
        or observed_sweep_keys != expected_sweep_keys
        or integer(
            report_binding.get("bytes", -1), label="baseline report bytes"
        ) != int(report_path.stat().st_size)
        or str(manifest.get("split") or "") != "val"
        or manifest_checkpoint is None
        or manifest_checkpoint.resolve(strict=False) != checkpoint_path
        or str(manifest.get("checkpoint_sha256") or "") != checkpoint_sha256
        or integer(manifest.get("checkpoint_step", -1), label="baseline manifest step")
        != integer(binding.get("checkpoint_step", -2), label="baseline binding step")
        or integer(report.get("checkpoint_step", -3), label="baseline report step")
        != integer(binding.get("checkpoint_step", -2), label="baseline binding step")
        or str(resolved.get("instruction_ablation") or "normal") != "normal"
        or str(resolved.get("visual_ablation") or "normal") != "normal"
    ):
        raise ValueError("baseline eval manifest/report/checkpoint 绑定不一致")
    return {
        "eval_report": str(report_path),
        "eval_report_sha256": report_sha256,
        "eval_report_bytes": int(report_path.stat().st_size),
        "eval_report_manifest_binding": report_binding,
        "prediction_population": {
            key: value
            for key, value in prediction_population.items()
            if key != "rows"
        },
        "eval_manifest": str(manifest_path),
        "eval_manifest_sha256": manifest_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": integer(
            binding.get("checkpoint_step"), label="baseline checkpoint step"
        ),
        "eval_threshold": report_threshold,
        "threshold_sweep": resolved_sweep,
    }


def validate_m7_retention_gate(
    gate_path: str | Path,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Deep-validate one formal M7 full-val retention artifact."""
    path = resolve_project_path(gate_path) or Path(gate_path)
    if not path.is_file():
        raise ValueError(f"M7 retention gate 不存在: {gate_path}")
    gate = json_object(path, label="M7 retention gate")
    if gate.get("protocol") != RETENTION_GATE_PROTOCOL:
        raise ValueError(
            f"M7 retention gate protocol 不兼容: {gate.get('protocol')!r}; "
            f"expected={RETENTION_GATE_PROTOCOL!r}"
        )
    metadata = dict((gate.get("checkpoint_metadata") or {}).get("metadata") or {})
    checkpoint_config = require_serialized_segdesc_config(
        metadata.get("config"), label="M7 retention gate checkpoint config"
    )
    seed_values = {
        "CLI seed": expected_seed,
        "gate expected_seed": gate.get("expected_seed"),
        "joint checkpoint seed": gate.get("joint_checkpoint_seed"),
        "checkpoint config seed": serialized_segdesc_config_value(
            checkpoint_config, "seed"
        ),
    }
    try:
        normalized_seeds = {
            label: integer(value, label=label) for label, value in seed_values.items()
        }
    except ValueError as exc:
        raise ValueError(f"M7 retention seed binding 不完整: {seed_values}") from exc
    if len(set(normalized_seeds.values())) != 1 or gate.get("seed_match") is not True:
        raise ValueError(f"M7 retention seed binding 不一致: {seed_values}")
    try:
        joint_execution_audit = validate_joint_checkpoint_execution(
            metadata,
            checkpoint_step=integer(
                gate.get("checkpoint_step"), label="joint checkpoint step"
            ),
        )
    except RuntimeError as exc:
        raise ValueError("M7 retention joint execution contract 非法") from exc
    if (
        gate.get("joint_execution_contract_valid") is not True
        or gate.get("joint_execution_audit") != joint_execution_audit
    ):
        raise ValueError("M7 retention gate 未绑定可重算的 joint execution contract")
    protocol_assets = (gate.get("checkpoint_metadata") or {}).get(
        "description_protocol_assets"
    )
    if (
        protocol_assets != description_protocol_assets_spec()
        or gate.get("description_protocol_assets_current") is not True
    ):
        raise ValueError("M7 retention joint checkpoint ontology/schema binding 已漂移")
    checkpoint_path, checkpoint_sha256 = bound_file(
        gate.get("checkpoint"),
        gate.get("joint_checkpoint_sha256"),
        label="joint checkpoint",
    )
    try:
        checkpoint_provenance = inspect_segdesc_checkpoint(checkpoint_path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError("M7 retention joint checkpoint payload 无法重放") from exc
    if (
        gate.get("checkpoint_metadata")
        != checkpoint_provenance["checkpoint_metadata"]
        or integer(gate.get("checkpoint_step"), label="joint checkpoint step")
        != checkpoint_provenance["checkpoint_step"]
        or checkpoint_sha256 != checkpoint_provenance["checkpoint_sha256"]
    ):
        raise ValueError("M7 retention joint checkpoint step/metadata 与 payload 不一致")
    try:
        joint_run_completion = validate_checkpoint_run_completion(
            checkpoint_path,
            expected_completion_protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
            expected_stage="joint",
            expected_role="validation_best",
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError("M7 retention joint training run 未成功完成") from exc
    if gate.get("joint_run_completion_audit") != joint_run_completion:
        raise ValueError("M7 retention joint run completion binding 已漂移")
    checkpoint_architecture = dict(
        checkpoint_provenance["checkpoint_metadata"].get(
            "description_architecture_spec"
        ) or {}
    )
    try:
        description_cache_artifact_provenance = (
            revalidate_description_cache_artifact(
                checkpoint_architecture.get(
                    "description_cache_artifact_binding"
                )
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "M7 retention joint checkpoint Description Vision Cache artifact 无法重放"
        ) from exc
    if (
        str(metadata.get("stage") or "") != "joint"
        or metadata.get("checkpoint_role") != "validation_best"
        or gate.get("joint_checkpoint") is not True
    ):
        raise ValueError("M7 retention 只接受 joint validation_best checkpoint")
    if (
        serialized_segdesc_config_value(
            checkpoint_config, "joint_region_stage"
        ) != "predicted_mask"
        or not math.isclose(
            finite_float(
                serialized_segdesc_config_value(
                    checkpoint_config, "predicted_mask_fraction"
                ),
                label="joint predicted_mask_fraction",
            ),
            D4_FINAL_FRACTION,
            abs_tol=1.0e-12,
        )
        or gate.get("predicted_mask_main_route") is not True
    ):
        raise ValueError("M7 retention 只接受 D4 75% predicted-mask 主路线")
    region_data_audit = metadata.get("region_data_audit")
    if not isinstance(region_data_audit, dict):
        raise ValueError("joint checkpoint 缺少 region_data_audit")
    d4_final_acceptance = revalidate_saved_d4_final_acceptance(
        metadata.get("d4_final_acceptance"),
        seed=expected_seed,
        train_region_data_audit=region_data_audit,
    )
    if (
        gate.get("d4_final_acceptance_audit") != d4_final_acceptance
        or gate.get("d4_final_acceptance_valid") is not True
    ):
        raise ValueError("M7 retention gate 未绑定 joint checkpoint 的 D4 final acceptance")
    m6_acceptance = revalidate_saved_m6_acceptance(
        metadata.get("m6_acceptance"),
        seed=expected_seed,
        train_region_data_audit=region_data_audit,
    )
    if (
        gate.get("m6_acceptance_audit") != m6_acceptance
        or gate.get("m6_acceptance_valid") is not True
        or m6_acceptance.get("d4_final_acceptance") != d4_final_acceptance
    ):
        raise ValueError("M7 retention gate 未绑定 joint checkpoint 的完整 M6 acceptance")
    try:
        joint_initialization_audit = revalidate_joint_initialization_audit(
            metadata.get("joint_initialization_audit"),
            expected_seed=expected_seed,
            region_stage="predicted_mask",
            region_data_audit=region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=dict(
                checkpoint_provenance["checkpoint_metadata"].get(
                    "segmentation_migration"
                ) or {}
            ),
            require_m6_binding=True,
        )
    except RuntimeError as exc:
        raise ValueError(
            "M7 retention joint initialization source 无法重放"
        ) from exc
    if (
        gate.get("joint_initialization_audit")
        != joint_initialization_audit
        or gate.get("joint_initialization_valid") is not True
        or metadata.get("segmentation_migration_lineage")
        != joint_initialization_audit.get("segmentation_migration_lineage")
    ):
        raise ValueError("M7 retention gate 未绑定可重放的 joint initialization source")
    checkpoint_description_benchmark = serialized_segdesc_config_value(
        checkpoint_config, "description_benchmark"
    )
    if not isinstance(checkpoint_description_benchmark, str) or not (
        checkpoint_description_benchmark.strip()
    ):
        raise ValueError("M7 joint checkpoint 缺少 description_benchmark binding")
    d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
        metadata.get("d_minus_one_acceptance"),
        expected_description_benchmark=checkpoint_description_benchmark,
    )
    if (
        gate.get("d_minus_one_acceptance_audit") != d_minus_one_acceptance
        or gate.get("d_minus_one_acceptance_valid") is not True
        or m6_acceptance.get("d_minus_one_acceptance")
        != d_minus_one_acceptance
    ):
        raise ValueError("M7 retention gate 未绑定 joint checkpoint 的 D-1 acceptance")
    stage_lineage = validate_description_stage_lineage(
        metadata.get("stage_lineage"),
        expected_target_stage="predicted_mask",
    )
    if (
        gate.get("stage_lineage_audit") != stage_lineage
        or gate.get("stage_lineage_valid") is not True
    ):
        raise ValueError("M7 retention gate 未绑定完整 D0-D4 stage lineage")

    binding = gate.get("joint_eval_binding")
    if not isinstance(binding, dict) or binding.get("protocol") != RETENTION_EVAL_BINDING_PROTOCOL:
        raise ValueError("M7 retention gate 缺少 joint eval report binding")
    report_path, report_sha256 = bound_file(
        binding.get("eval_report"),
        binding.get("eval_report_sha256"),
        label="joint eval report",
    )
    binding_checkpoint, binding_checkpoint_sha256 = bound_file(
        binding.get("checkpoint"),
        binding.get("checkpoint_sha256"),
        label="joint eval checkpoint",
    )
    if (
        binding_checkpoint != checkpoint_path
        or binding_checkpoint_sha256 != checkpoint_sha256
        or str(binding.get("split") or "") != "val"
        or str(binding.get("sample_population_sha256") or "")
        != str((gate.get("joint_sample_population") or {}).get("sha256") or "")
    ):
        raise ValueError("joint eval binding 与 retention gate 不一致")
    joint_report = json_object(report_path, label="joint eval report")
    validate_eval_report_matches_gate(joint_report, gate, prefix="joint")

    baseline_binding = validate_baseline_binding(gate)
    migration = gate.get("segmentation_migration")
    checkpoint_migration = (gate.get("checkpoint_metadata") or {}).get(
        "segmentation_migration"
    )
    if not isinstance(migration, dict) or migration != checkpoint_migration:
        raise ValueError("joint checkpoint segmentation migration audit 不一致")
    if (
        str(migration.get("source_sha256") or "")
        != baseline_binding["checkpoint_sha256"]
        or gate.get("baseline_source_checkpoint_match") is not True
    ):
        raise ValueError("joint checkpoint 未继承当前 retention baseline")
    replay_audit = gate.get("baseline_checkpoint_replay_audit")
    if not isinstance(replay_audit, dict):
        raise ValueError("M7 retention gate 缺少 baseline checkpoint replay audit")
    replay_report_path = (replay_audit.get("replay_report") or {}).get("path")
    replay_path, _replay_sha256 = bound_file(
        replay_report_path,
        (replay_audit.get("replay_report") or {}).get("sha256"),
        label="baseline replay report",
    )
    frozen_report = json_object(
        Path(baseline_binding["eval_report"]), label="baseline eval report"
    )
    replay_report = json_object(replay_path, label="baseline replay report")
    rebuilt_replay_audit = build_baseline_checkpoint_replay_audit(
        frozen_report,
        replay_report,
        baseline_binding=baseline_binding,
        segmentation_migration=migration,
        replay_report_path=replay_path,
    )
    if (
        replay_audit != rebuilt_replay_audit
        or gate.get("baseline_checkpoint_replayed") is not True
    ):
        raise ValueError("M7 retention baseline checkpoint replay audit 已漂移")

    baseline_count = integer(gate.get("baseline_num_samples"), label="baseline_num_samples")
    joint_count = integer(gate.get("joint_num_samples"), label="joint_num_samples")
    if baseline_count <= 0 or joint_count != baseline_count:
        raise ValueError("M7 retention full-val 样本数不一致")
    baseline_population = gate.get("baseline_sample_population")
    joint_population = gate.get("joint_sample_population")
    if not isinstance(baseline_population, dict) or not isinstance(joint_population, dict):
        raise ValueError("M7 retention 缺少 sample population")
    baseline_population_audit = validate_population(
        baseline_population, expected_count=baseline_count, label="baseline"
    )
    joint_population_audit = validate_population(
        joint_population, expected_count=joint_count, label="joint"
    )
    if baseline_population != joint_population:
        raise ValueError("M7 retention baseline/joint sample population 不一致")
    baseline_metric_inputs = segmentation_metric_input_population(
        frozen_report.get("prediction_population")
    )
    joint_metric_inputs = segmentation_metric_input_population(
        joint_report.get("prediction_population")
    )
    if (
        baseline_metric_inputs != joint_metric_inputs
        or gate.get("baseline_metric_input_population")
        != baseline_metric_inputs
        or gate.get("joint_metric_input_population") != joint_metric_inputs
        or gate.get("same_metric_input_population") is not True
    ):
        raise ValueError(
            "M7 retention baseline/joint target 或 valid-mask population 不一致"
        )

    baseline_threshold = finite_float(gate.get("baseline_threshold"), label="baseline_threshold")
    joint_threshold = finite_float(gate.get("joint_threshold"), label="joint_threshold")
    baseline_dice = finite_float(gate.get("baseline_positive_dice"), label="baseline_positive_dice")
    joint_dice = finite_float(gate.get("joint_positive_dice"), label="joint_positive_dice")
    maximum_drop = finite_float(gate.get("maximum_allowed_drop"), label="maximum_allowed_drop")
    observed_drop = finite_float(gate.get("absolute_drop"), label="absolute_drop")
    if not (0.0 <= baseline_dice <= 1.0 and 0.0 <= joint_dice <= 1.0):
        raise ValueError("M7 retention positive Dice 超出 [0, 1]")
    if maximum_drop < 0.0 or abs(baseline_threshold - joint_threshold) > 1.0e-12:
        raise ValueError("M7 retention threshold/drop 配置非法")
    recomputed_drop = baseline_dice - joint_dice
    recomputed_preliminary = recomputed_drop <= maximum_drop
    if (
        abs(observed_drop - recomputed_drop) > 1.0e-12
        or gate.get("preliminary_passed") is not recomputed_preliminary
    ):
        raise ValueError("M7 retention Dice drop 计算与 gate 不一致")
    required_true = (
        "full_split_requested",
        "same_population_size",
        "population_protocol_match",
        "population_schema_valid",
        "population_identity_valid",
        "population_counts_valid",
        "same_sample_population",
        "same_metric_input_population",
        "same_threshold",
        "scientific_gate_eligible",
        "formal_report_binding_complete",
        "predicted_mask_main_route",
        "d4_final_acceptance_valid",
        "d_minus_one_acceptance_valid",
        "stage_lineage_valid",
        "m6_acceptance_valid",
        "joint_initialization_valid",
        "description_protocol_assets_current",
        "baseline_checkpoint_replayed",
    )
    missing_true = [name for name in required_true if gate.get(name) is not True]
    if str(gate.get("split") or "") != "val" or missing_true:
        raise ValueError(f"M7 retention 不是可发布的 full-val gate: {missing_true}")
    if gate.get("baseline_comparison_mode") != "frozen_full_report":
        raise ValueError("M7 正式 retention 必须使用 frozen full baseline report")
    recomputed_passed = bool(recomputed_preliminary)
    if gate.get("passed") is not recomputed_passed:
        raise ValueError("M7 retention passed 与绑定证据不一致")

    # 除 seed 外，三条链必须使用完全相同的联合训练配置。
    scientific_config = serialized_segdesc_config_without(
        checkpoint_config,
        M7_RUN_LOCAL_CONFIG_FIELDS,
        label="M7 retention gate checkpoint config",
    )
    return {
        "seed": int(expected_seed),
        "gate": str(path.resolve(strict=False)),
        "gate_sha256": sha256_file(path),
        "joint_eval_report": str(report_path),
        "joint_eval_report_sha256": report_sha256,
        "joint_checkpoint": str(checkpoint_path),
        "joint_checkpoint_sha256": checkpoint_sha256,
        "joint_checkpoint_step": integer(
            gate.get("checkpoint_step"), label="joint checkpoint step"
        ),
        "joint_checkpoint_payload_provenance": checkpoint_provenance,
        "description_cache_artifact_provenance": (
            description_cache_artifact_provenance
        ),
        "joint_scientific_config_sha256": canonical_sha256(scientific_config),
        "joint_execution_audit": joint_execution_audit,
        "joint_training_population_binding": joint_execution_audit[
            "training_population_binding"
        ],
        "m6_acceptance_audit": m6_acceptance,
        "joint_initialization_audit": joint_initialization_audit,
        "d4_final_acceptance_audit": d4_final_acceptance,
        "d_minus_one_acceptance_audit": d_minus_one_acceptance,
        "stage_lineage_audit": stage_lineage,
        "description_protocol_assets": protocol_assets,
        "baseline_binding": baseline_binding,
        "baseline_checkpoint_replay_audit": rebuilt_replay_audit,
        "baseline_population": baseline_population_audit,
        "joint_population": joint_population_audit,
        "metric_input_population": baseline_metric_inputs,
        "baseline_positive_dice": baseline_dice,
        "joint_positive_dice": joint_dice,
        "absolute_drop": recomputed_drop,
        "maximum_allowed_drop": maximum_drop,
        "passed": recomputed_passed,
    }
