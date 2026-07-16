#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行 M7 full-val segmentation retention 门禁。

用途：现场重放原 segmentation baseline，再在相同完整 val 上评估 joint checkpoint，验证
positive Dice 下降不超过 0.01。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_segdesc_retention --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --seed 42 --checkpoint
outputs/qpsalm_description/JOINT/checkpoint_best.pt --baseline-eval-report
outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json
--device cuda --output-dir outputs/qpsalm_description/JOINT/retention_full_val
主要输出：baseline_segmentation_replay.json、joint_segmentation_eval.json 和 retention_gate.json。
写入行为：只写 --output-dir；不以 monitor 指标替代 full-val 门禁。
注意：--max-samples 仅供 smoke；非零时只报告 preliminary_passed，正式 passed 恒为 false。
所属流程：M7 最终验收。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import traceback

from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.description.checkpoint import (
    DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
    validate_segmentation_migration_lineage,
    validate_description_stage_lineage,
)
from qpsalm_seg.description.d4_curriculum import (
    D4_FINAL_FRACTION,
    revalidate_saved_d4_final_acceptance,
)
from qpsalm_seg.description.d_minus_one import (
    D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
    revalidate_saved_d_minus_one_acceptance,
)
from qpsalm_seg.description.m6_acceptance import (
    M6_ACCEPTANCE_AUDIT_PROTOCOL,
    revalidate_saved_m6_acceptance,
)
from qpsalm_seg.description.joint_trainer import (
    JOINT_INITIALIZATION_PROTOCOL,
    revalidate_joint_initialization_audit,
    validate_joint_checkpoint_execution,
)
from qpsalm_seg.description.json_protocol import strict_json_loads
from qpsalm_seg.description.run_artifacts import (
    JOINT_TRAINING_COMPLETION_PROTOCOL,
    validate_checkpoint_run_completion,
)
from qpsalm_seg.description.retention import (
    BASELINE_CHECKPOINT_REPLAY_PROTOCOL,
    RETENTION_GATE_PROTOCOL,
    bind_joint_evaluation_report,
    build_baseline_checkpoint_replay_audit,
    segmentation_metric_input_population,
    validate_m7_retention_gate,
)
from qpsalm_seg.description.vision_cache import (
    revalidate_description_cache_artifact,
)
from qpsalm_seg.engine.evaluator import (
    SAMPLE_IDENTITY_FIELDS,
    SAMPLE_IDENTITY_PROTOCOL,
    SEGMENTATION_EVAL_MANIFEST_PROTOCOL,
    SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL,
    validate_segmentation_prediction_population,
)
from qpsalm_seg.engine.threshold import normalize_thresholds
from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-val segmentation retention.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-eval-report", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _positive_dice(report: dict) -> float:
    return float((((report.get("metrics") or {}).get("positive_only") or {}).get("dice")) or 0.0)


def _num_samples(report: dict) -> int:
    return int((report.get("coverage") or {}).get("num_samples") or 0)


def _sample_population(report: dict) -> dict:
    value = (report.get("coverage") or {}).get("sample_population") or {}
    return value if isinstance(value, dict) else {}


def baseline_eval_binding(report_path: Path, report: dict, *, split: str) -> dict:
    """Bind the baseline report to the exact segmentation checkpoint that produced it."""
    manifest_path = report_path.parent / "eval_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"segmentation baseline 缺少 eval_manifest.json: {manifest_path}")
    manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    report_resolved = report_path.resolve(strict=False)
    report_from_file = strict_json_loads(
        report_resolved.read_text(encoding="utf-8")
    )
    if not isinstance(report_from_file, dict) or report_from_file != report:
        raise RuntimeError("segmentation baseline 内存 report 与磁盘文件不一致")
    prediction_population = validate_segmentation_prediction_population(
        report_from_file.get("prediction_population")
    )
    report_sha256 = _sha256_file(report_resolved)
    report_bytes = int(report_resolved.stat().st_size)
    report_binding = dict(manifest.get("eval_report_binding") or {})
    manifest_report_path = resolve_project_path(
        str(report_binding.get("path") or "")
    )
    checkpoint_path = resolve_project_path(str(manifest.get("checkpoint") or ""))
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError("segmentation baseline manifest 的 checkpoint 不存在")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    resolved = dict(manifest.get("resolved_config") or {})
    try:
        report_threshold = float(report.get("threshold"))
        resolved_threshold = float(resolved.get("eval_threshold"))
        prediction_threshold = float(prediction_population.get("threshold"))
        binding_threshold = float(report_binding.get("eval_threshold"))
        resolved_sweep = normalize_thresholds(resolved.get("threshold_sweep"))
        binding_sweep = normalize_thresholds(report_binding.get("threshold_sweep"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("segmentation baseline threshold 配置非法") from exc
    sweep_report = report.get("threshold_sweep") or {}
    observed_sweep_keys = set(
        dict(sweep_report.get("overall_by_threshold") or {})
    ) if isinstance(sweep_report, dict) else set()
    expected_sweep_keys = {f"{value:.2f}" for value in resolved_sweep}
    errors = []
    if manifest.get("protocol") != SEGMENTATION_EVAL_MANIFEST_PROTOCOL:
        errors.append("manifest_protocol")
    if manifest.get("created_by") != "qpsalm-eval":
        errors.append("created_by")
    if report_binding.get("protocol") != SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL:
        errors.append("eval_report_binding_protocol")
    if (
        manifest_report_path is None
        or manifest_report_path.resolve(strict=False) != report_resolved
    ):
        errors.append("eval_report_path")
    if report_binding.get("sha256") != report_sha256:
        errors.append("eval_report_sha256")
    if (
        report_binding.get("prediction_population_sha256")
        != prediction_population.get("sha256")
    ):
        errors.append("prediction_population_sha256")
    if not (
        0.0 <= report_threshold <= 1.0
        and abs(report_threshold - resolved_threshold) <= 1.0e-12
        and abs(report_threshold - prediction_threshold) <= 1.0e-12
        and abs(report_threshold - binding_threshold) <= 1.0e-12
    ):
        errors.append("eval_threshold")
    if (
        not isinstance(resolved.get("threshold_sweep"), list)
        or not isinstance(report_binding.get("threshold_sweep"), list)
        or not isinstance(report.get("threshold_sweep"), dict)
        or binding_sweep != resolved_sweep
        or observed_sweep_keys != expected_sweep_keys
    ):
        errors.append("threshold_sweep")
    if int(report_binding.get("bytes", -1)) != report_bytes:
        errors.append("eval_report_bytes")
    if str(manifest.get("split") or "") != split:
        errors.append("split")
    if int(manifest.get("checkpoint_step", -1)) != int(report.get("checkpoint_step", -2)):
        errors.append("checkpoint_step")
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        errors.append("checkpoint_sha256")
    if str(resolved.get("instruction_ablation") or "normal") != "normal":
        errors.append("instruction_ablation")
    if str(resolved.get("visual_ablation") or "normal") != "normal":
        errors.append("visual_ablation")
    if errors:
        raise RuntimeError(f"segmentation baseline eval binding 非法: {errors}")
    return {
        "valid": True,
        "eval_report": str(report_path),
        "eval_report_sha256": report_sha256,
        "eval_report_bytes": report_bytes,
        "eval_report_manifest_binding": report_binding,
        "prediction_population": {
            key: value
            for key, value in prediction_population.items()
            if key != "rows"
        },
        "eval_manifest": str(manifest_path),
        "eval_manifest_sha256": _sha256_file(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(manifest["checkpoint_step"]),
        "split": split,
        "preset": manifest.get("preset"),
        "eval_threshold": report_threshold,
        "threshold_sweep": resolved_sweep,
    }


def build_retention_gate(
    baseline: dict,
    report: dict,
    *,
    split: str,
    max_samples: int,
    checkpoint: str,
    checkpoint_step: int,
    checkpoint_metadata: dict,
    maximum_allowed_drop: float,
    baseline_binding: dict,
    expected_seed: int | None = None,
    d4_final_acceptance_audit: dict | None = None,
    m6_acceptance_audit: dict | None = None,
    joint_initialization_audit: dict | None = None,
    d_minus_one_acceptance_audit: dict | None = None,
    stage_lineage_audit: dict | None = None,
    baseline_checkpoint_replay_audit: dict | None = None,
) -> dict:
    """Build the formal gate without accepting count-only population matching."""
    baseline_dice = _positive_dice(baseline)
    current_dice = _positive_dice(report)
    drop = baseline_dice - current_dice
    baseline_n = _num_samples(baseline)
    current_n = _num_samples(report)
    baseline_threshold = float(baseline.get("threshold", 0.5))
    current_threshold = float(report.get("threshold", 0.5))
    baseline_population = _sample_population(baseline)
    current_population = _sample_population(report)
    baseline_prediction_population = validate_segmentation_prediction_population(
        baseline.get("prediction_population")
    )
    joint_prediction_population = validate_segmentation_prediction_population(
        report.get("prediction_population")
    )
    baseline_metric_input_population = segmentation_metric_input_population(
        baseline_prediction_population
    )
    joint_metric_input_population = segmentation_metric_input_population(
        joint_prediction_population
    )
    same_metric_input_population = bool(
        baseline_metric_input_population == joint_metric_input_population
    )
    baseline_population_hash = str(baseline_population.get("sha256") or "")
    current_population_hash = str(current_population.get("sha256") or "")
    population_schema_valid = bool(
        baseline_population.get("protocol") == SAMPLE_IDENTITY_PROTOCOL
        and current_population.get("protocol") == SAMPLE_IDENTITY_PROTOCOL
        and tuple(baseline_population.get("fields") or ()) == tuple(SAMPLE_IDENTITY_FIELDS)
        and tuple(current_population.get("fields") or ()) == tuple(SAMPLE_IDENTITY_FIELDS)
    )
    population_protocol_match = population_schema_valid
    population_identity_valid = bool(
        baseline_population.get("complete")
        and baseline_population.get("unique")
        and current_population.get("complete")
        and current_population.get("unique")
    )
    population_counts_valid = bool(
        int(baseline_population.get("num_records", -1)) == baseline_n
        and int(baseline_population.get("num_unique_sample_ids", -1)) == baseline_n
        and int(current_population.get("num_records", -1)) == current_n
        and int(current_population.get("num_unique_sample_ids", -1)) == current_n
    )
    same_sample_population = bool(
        population_protocol_match
        and population_identity_valid
        and population_counts_valid
        and baseline_population_hash
        and baseline_population_hash == current_population_hash
    )
    full_split = int(max_samples) == 0
    baseline_comparison_mode = (
        "frozen_full_report" if full_split else "live_limited_replay"
    )
    same_population_size = baseline_n > 0 and current_n == baseline_n
    same_threshold = abs(current_threshold - baseline_threshold) <= 1.0e-12
    checkpoint_stage = str((checkpoint_metadata.get("metadata") or {}).get("stage") or "")
    checkpoint_config = dict(
        (checkpoint_metadata.get("metadata") or {}).get("config") or {}
    )
    try:
        checkpoint_predicted_fraction = float(
            checkpoint_config.get("predicted_mask_fraction")
        )
    except (TypeError, ValueError):
        checkpoint_predicted_fraction = -1.0
    checkpoint_seed = checkpoint_config.get("seed")
    seed_match = bool(
        expected_seed is None
        or (
            checkpoint_seed is not None
            and int(checkpoint_seed) == int(expected_seed)
        )
    )
    joint_checkpoint = checkpoint_stage == "joint"
    try:
        joint_execution_audit = validate_joint_checkpoint_execution(
            dict(checkpoint_metadata.get("metadata") or {}),
            checkpoint_step=int(checkpoint_step),
        )
        joint_execution_contract_valid = True
    except (RuntimeError, TypeError, ValueError) as exc:
        joint_execution_audit = {
            "protocol": "qpsalm_segdesc_joint_execution_audit_v1",
            "passed": False,
            "error": str(exc),
        }
        joint_execution_contract_valid = False
    predicted_mask_main_route = bool(
        checkpoint_config.get("joint_region_stage") == "predicted_mask"
        and abs(checkpoint_predicted_fraction - D4_FINAL_FRACTION)
        <= 1.0e-12
    )
    try:
        accepted_fraction = float(
            (d4_final_acceptance_audit or {}).get("current_fraction")
        )
    except (TypeError, ValueError):
        accepted_fraction = -1.0
    d4_final_acceptance_valid = bool(
        isinstance(d4_final_acceptance_audit, dict)
        and d4_final_acceptance_audit.get("passed") is True
        and abs(accepted_fraction - D4_FINAL_FRACTION)
        <= 1.0e-12
    )
    m6_acceptance_valid = bool(
        isinstance(m6_acceptance_audit, dict)
        and m6_acceptance_audit.get("protocol")
        == M6_ACCEPTANCE_AUDIT_PROTOCOL
        and m6_acceptance_audit.get("passed") is True
        and m6_acceptance_audit.get("d4_final_acceptance")
        == d4_final_acceptance_audit
    )
    joint_initialization_valid = bool(
        isinstance(joint_initialization_audit, dict)
        and joint_initialization_audit.get("protocol")
        == JOINT_INITIALIZATION_PROTOCOL
        and joint_initialization_audit.get("passed") is True
        and joint_initialization_audit.get("formal_m6_bound") is True
        and isinstance(
            joint_initialization_audit.get("segmentation_migration_lineage"),
            dict,
        )
        and joint_initialization_audit["segmentation_migration_lineage"].get(
            "passed"
        ) is True
        and (checkpoint_metadata.get("metadata") or {}).get(
            "joint_initialization_audit"
        ) == joint_initialization_audit
        and (checkpoint_metadata.get("metadata") or {}).get(
            "segmentation_migration_lineage"
        ) == joint_initialization_audit.get("segmentation_migration_lineage")
    )
    d_minus_one_acceptance_valid = bool(
        isinstance(d_minus_one_acceptance_audit, dict)
        and d_minus_one_acceptance_audit.get("protocol")
        == D_MINUS_ONE_ACCEPTANCE_PROTOCOL
        and d_minus_one_acceptance_audit.get("passed") is True
    )
    stage_lineage_valid = bool(
        isinstance(stage_lineage_audit, dict)
        and stage_lineage_audit.get("protocol")
        == DESCRIPTION_STAGE_LINEAGE_PROTOCOL
        and stage_lineage_audit.get("target_stage") == "predicted_mask"
    )
    segmentation_migration = dict(
        checkpoint_metadata.get("segmentation_migration") or {}
    )
    description_protocol_assets_current = bool(
        checkpoint_metadata.get("description_protocol_assets")
        == description_protocol_assets_spec()
    )
    baseline_source_checkpoint_match = bool(
        str(baseline_binding.get("checkpoint_sha256") or "")
        and str(baseline_binding.get("checkpoint_sha256"))
        == str(segmentation_migration.get("source_sha256") or "")
    )
    baseline_checkpoint_replayed = bool(
        isinstance(baseline_checkpoint_replay_audit, dict)
        and baseline_checkpoint_replay_audit.get("protocol")
        == BASELINE_CHECKPOINT_REPLAY_PROTOCOL
        and baseline_checkpoint_replay_audit.get("passed") is True
        and baseline_checkpoint_replay_audit.get("checkpoint_sha256")
        == baseline_binding.get("checkpoint_sha256")
    )
    preliminary_passed = drop <= float(maximum_allowed_drop)
    scientific_gate_eligible = (
        split == "val"
        and full_split
        and same_population_size
        and same_sample_population
        and same_metric_input_population
        and same_threshold
        and joint_checkpoint
        and joint_execution_contract_valid
        and baseline_binding.get("valid") is True
        and baseline_source_checkpoint_match
        and baseline_checkpoint_replayed
        and seed_match
        and predicted_mask_main_route
        and d4_final_acceptance_valid
        and d_minus_one_acceptance_valid
        and stage_lineage_valid
        and m6_acceptance_valid
        and joint_initialization_valid
        and description_protocol_assets_current
    )
    return {
        "protocol": RETENTION_GATE_PROTOCOL,
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "checkpoint_metadata": checkpoint_metadata,
        "baseline_binding": baseline_binding,
        "split": split,
        "baseline_num_samples": baseline_n,
        "joint_num_samples": current_n,
        "full_split_requested": full_split,
        "baseline_comparison_mode": baseline_comparison_mode,
        "same_population_size": same_population_size,
        "baseline_sample_population": baseline_population,
        "joint_sample_population": current_population,
        "baseline_prediction_population_sha256": (
            baseline_prediction_population["sha256"]
        ),
        "joint_prediction_population_sha256": (
            joint_prediction_population["sha256"]
        ),
        "baseline_metric_input_population": (
            baseline_metric_input_population
        ),
        "joint_metric_input_population": joint_metric_input_population,
        "same_metric_input_population": same_metric_input_population,
        "population_protocol_match": population_protocol_match,
        "population_schema_valid": population_schema_valid,
        "population_identity_valid": population_identity_valid,
        "population_counts_valid": population_counts_valid,
        "same_sample_population": same_sample_population,
        "baseline_threshold": baseline_threshold,
        "joint_threshold": current_threshold,
        "same_threshold": same_threshold,
        "joint_checkpoint": joint_checkpoint,
        "joint_execution_audit": joint_execution_audit,
        "joint_execution_contract_valid": joint_execution_contract_valid,
        "predicted_mask_main_route": predicted_mask_main_route,
        "d4_final_acceptance_audit": d4_final_acceptance_audit,
        "d4_final_acceptance_valid": d4_final_acceptance_valid,
        "m6_acceptance_audit": m6_acceptance_audit,
        "m6_acceptance_valid": m6_acceptance_valid,
        "joint_initialization_audit": joint_initialization_audit,
        "joint_initialization_valid": joint_initialization_valid,
        "description_protocol_assets_current": description_protocol_assets_current,
        "d_minus_one_acceptance_audit": d_minus_one_acceptance_audit,
        "d_minus_one_acceptance_valid": d_minus_one_acceptance_valid,
        "stage_lineage_audit": stage_lineage_audit,
        "stage_lineage_valid": stage_lineage_valid,
        "segmentation_migration": segmentation_migration,
        "baseline_source_checkpoint_match": baseline_source_checkpoint_match,
        "baseline_checkpoint_replay_audit": (
            baseline_checkpoint_replay_audit
        ),
        "baseline_checkpoint_replayed": baseline_checkpoint_replayed,
        "expected_seed": int(expected_seed) if expected_seed is not None else None,
        "joint_checkpoint_seed": (
            int(checkpoint_seed) if checkpoint_seed is not None else None
        ),
        "seed_match": seed_match,
        "baseline_positive_dice": baseline_dice,
        "joint_positive_dice": current_dice,
        "absolute_drop": drop,
        "maximum_allowed_drop": float(maximum_allowed_drop),
        "preliminary_passed": preliminary_passed,
        "scientific_gate_eligible": scientific_gate_eligible,
        # 正式 passed 还必须绑定原始 joint eval report；纯内存 gate 不能发布。
        "formal_report_binding_complete": False,
        "passed": False,
    }


def main() -> None:
    args = parse_args()
    if int(args.max_samples) < 0:
        raise SystemExit("--max-samples 必须大于等于 0")
    if int(args.max_samples) == 0 and args.split != "val":
        raise SystemExit("M7 正式 retention 只接受 --split val")
    config = load_segdesc_config(args.config, {"seed": args.seed})
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    baseline_path = resolve_project_path(args.baseline_eval_report) or Path(args.baseline_eval_report)
    checkpoint_path = resolve_project_path(args.checkpoint) or Path(args.checkpoint)
    if output.exists() and not output.is_dir():
        raise SystemExit(f"retention output-dir 不是目录: {output}")
    protected = {
        "config": args.config,
        "baseline-eval-report": baseline_path,
        "checkpoint": checkpoint_path,
        "segmentation-config": config.segmentation_config,
        "segmentation-checkpoint": config.segmentation_checkpoint,
        "segmentation-vision-cache": config.segmentation_vision_cache,
        "description-vision-cache": config.description_vision_cache,
        "description-benchmark": config.description_benchmark,
        "bridge-benchmark": config.bridge_benchmark,
        "predicted-index": config.predicted_index,
        "predicted-val-index": config.predicted_val_index,
        "d4-final-acceptance-gate": config.d4_final_acceptance_gate,
        "m6-acceptance-gate": config.m6_acceptance_gate,
    }
    try:
        validate_output_replacement_safety(output, protected)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        not args.overwrite_output
        and output.is_dir()
        and any(output.iterdir())
    ):
        raise SystemExit(
            "retention output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    from qpsalm_seg.description.checkpoint import load_segdesc_checkpoint
    from qpsalm_seg.description.common import description_device, write_json
    from qpsalm_seg.description.runtime import build_segdesc_model
    from qpsalm_seg.engine.common import build_eval_loader
    from qpsalm_seg.engine.evaluator import evaluate

    try:
        baseline = strict_json_loads(baseline_path.read_text(encoding="utf-8"))
        baseline_binding = baseline_eval_binding(baseline_path, baseline, split=args.split)
        device = description_device(args.device)
        model, migration = build_segdesc_model(config, device)
        checkpoint_preflight = inspect_segdesc_checkpoint(checkpoint_path)
        preflight_metadata = checkpoint_preflight["checkpoint_metadata"]
        preflight_payload = dict(preflight_metadata.get("metadata") or {})
        if (
            preflight_payload.get("stage") != "joint"
            or preflight_payload.get("checkpoint_role") != "validation_best"
        ):
            raise RuntimeError(
                "正式 M7 retention 只接受 joint validation_best checkpoint"
            )
        joint_run_completion = validate_checkpoint_run_completion(
            checkpoint_path,
            expected_completion_protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
            expected_stage="joint",
            expected_role="validation_best",
        )
        preflight_architecture = dict(
            preflight_metadata.get("description_architecture_spec") or {}
        )
        # 正式 full-val 前重放只读 cache 工件，避免昂贵评估结束后才发现 shard 漂移。
        revalidate_description_cache_artifact(
            preflight_architecture.get("description_cache_artifact_binding")
        )
        runtime_migration_lineage = validate_segmentation_migration_lineage(
            migration,
            preflight_metadata,
        )
        if (
            preflight_payload.get("segmentation_migration_lineage")
            != runtime_migration_lineage
        ):
            raise RuntimeError(
                "M7 joint checkpoint segmentation lineage 与当前 runtime 不一致"
            )
        validate_joint_checkpoint_execution(
            preflight_payload,
            checkpoint_step=int(checkpoint_preflight["checkpoint_step"]),
        )
        checkpoint_payload = preflight_payload
        region_data_audit = checkpoint_payload.get("region_data_audit")
        if not isinstance(region_data_audit, dict):
            raise RuntimeError("M7 joint checkpoint 缺少 region_data_audit")
        d4_final_acceptance = revalidate_saved_d4_final_acceptance(
            checkpoint_payload.get("d4_final_acceptance"),
            seed=config.seed,
            train_region_data_audit=region_data_audit,
        )
        m6_acceptance = revalidate_saved_m6_acceptance(
            checkpoint_payload.get("m6_acceptance"),
            seed=config.seed,
            train_region_data_audit=region_data_audit,
        )
        if m6_acceptance.get("d4_final_acceptance") != d4_final_acceptance:
            raise RuntimeError("M7 checkpoint 的 M6 与 D4 final acceptance 不一致")
        joint_initialization_audit = revalidate_joint_initialization_audit(
            checkpoint_payload.get("joint_initialization_audit"),
            expected_seed=config.seed,
            region_stage="predicted_mask",
            region_data_audit=region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=dict(
                preflight_metadata.get("segmentation_migration") or {}
            ),
            require_m6_binding=True,
        )
        d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
            checkpoint_payload.get("d_minus_one_acceptance")
        )
        stage_lineage = validate_description_stage_lineage(
            checkpoint_payload.get("stage_lineage"),
            expected_target_stage="predicted_mask",
        )
        segmentation_config = replace(
            model.segmentation.config,
            max_val_samples=args.max_samples or None,
        )
        replay_threshold = float(baseline_binding["eval_threshold"])
        replay_threshold_sweep = list(baseline_binding["threshold_sweep"])
        baseline_loader = build_eval_loader(segmentation_config, args.split)
        with model.controller.adapter_scope("default"):
            baseline_replay = evaluate(
                model.segmentation,
                baseline_loader,
                device,
                threshold=replay_threshold,
                threshold_sweep=replay_threshold_sweep,
            )
        baseline_replay["checkpoint_step"] = int(
            baseline_binding["checkpoint_step"]
        )
        baseline_replay_path = output / "baseline_segmentation_replay.json"
        write_json(baseline_replay_path, baseline_replay)
        formal_run = int(args.max_samples) == 0
        if formal_run:
            baseline_for_gate = baseline
            baseline_checkpoint_replay_audit = (
                build_baseline_checkpoint_replay_audit(
                    baseline,
                    baseline_replay,
                    baseline_binding=baseline_binding,
                    segmentation_migration=migration,
                    replay_report_path=baseline_replay_path,
                )
            )
        else:
            # smoke 必须比较同一有限总体上的现场 baseline/joint，不能混用 full-val 指标。
            baseline_for_gate = baseline_replay
            baseline_checkpoint_replay_audit = None
        step, metadata = load_segdesc_checkpoint(args.checkpoint, model)
        if (
            int(step) != int(checkpoint_preflight["checkpoint_step"])
            or metadata != preflight_metadata
            or _sha256_file(checkpoint_path)
            != checkpoint_preflight["checkpoint_sha256"]
        ):
            raise RuntimeError(
                "M7 joint checkpoint 在 preflight 与模型加载之间发生漂移"
            )
        loader = build_eval_loader(segmentation_config, args.split)
        with model.controller.adapter_scope("default"):
            report = evaluate(
                model.segmentation,
                loader,
                device,
                threshold=replay_threshold,
                threshold_sweep=replay_threshold_sweep,
            )
        gate = build_retention_gate(
            baseline_for_gate,
            report,
            split=args.split,
            max_samples=args.max_samples,
            checkpoint=args.checkpoint,
            checkpoint_step=step,
            checkpoint_metadata=metadata,
            maximum_allowed_drop=config.segmentation_retention_max_drop,
            baseline_binding=baseline_binding,
            expected_seed=config.seed,
            d4_final_acceptance_audit=d4_final_acceptance,
            m6_acceptance_audit=m6_acceptance,
            joint_initialization_audit=joint_initialization_audit,
            d_minus_one_acceptance_audit=d_minus_one_acceptance,
            stage_lineage_audit=stage_lineage,
            baseline_checkpoint_replay_audit=(
                baseline_checkpoint_replay_audit
            ),
        )
        gate["joint_checkpoint_sha256"] = _sha256_file(checkpoint_path)
        gate["joint_run_completion_audit"] = joint_run_completion
        joint_report_path = output / "joint_segmentation_eval.json"
        write_json(joint_report_path, report)
        gate = bind_joint_evaluation_report(
            gate,
            eval_report_path=joint_report_path,
            checkpoint_path=checkpoint_path,
        )
        gate_path = output / "retention_gate.json"
        if formal_run:
            # 先验证候选文件，再原子改名；失败目录不得残留貌似正式的 gate。
            candidate_gate_path = output / "retention_gate.candidate.json"
            write_json(candidate_gate_path, gate)
            validate_m7_retention_gate(
                candidate_gate_path,
                expected_seed=config.seed,
            )
            candidate_gate_path.replace(gate_path)
        else:
            write_json(gate_path, gate)
    except BaseException as exc:
        (output / "retention_gate.json").unlink(missing_ok=True)
        (output / "retention_gate.candidate.json").unlink(missing_ok=True)
        write_json(output / "failure_report.json", {
            "protocol": "qpsalm_segdesc_retention_failure_v2_no_partial_gate",
            "retention_gate_published": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    print(json.dumps(gate, ensure_ascii=False, allow_nan=False))
    accepted = (
        bool(gate["preliminary_passed"])
        if int(args.max_samples) > 0 else bool(gate["passed"])
    )
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
