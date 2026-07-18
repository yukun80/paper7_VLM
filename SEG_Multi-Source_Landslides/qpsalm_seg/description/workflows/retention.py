#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute one M7 full-val retention replay and publish its strict gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import traceback

from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)

from ..protocols.config import load_segdesc_config
from ..data.vision_cache import revalidate_description_cache_artifact
from ..evaluation.d4_curriculum import revalidate_saved_d4_final_acceptance
from ..evaluation.d_minus_one import revalidate_saved_d_minus_one_acceptance
from ..evaluation.m6_acceptance import revalidate_saved_m6_acceptance
from ..evaluation.retention import (
    bind_joint_evaluation_report,
    build_baseline_checkpoint_replay_audit,
    validate_m7_retention_gate,
)
from ..evaluation.retention_build import (
    baseline_eval_binding,
    build_retention_gate,
)
from ..protocols.io import sha256_file, strict_json_loads
from ..training.checkpoint import (
    inspect_segdesc_checkpoint,
    validate_description_stage_lineage,
    validate_segmentation_migration_lineage,
)
from ..training.joint_lifecycle import (
    revalidate_joint_initialization_audit,
    validate_joint_checkpoint_execution,
)
from ..training.run_artifacts import validate_checkpoint_run_completion
from ..protocols.versions import JOINT_TRAINING_COMPLETION_PROTOCOL


def run_retention_workflow(
    *,
    config_path: str,
    seed: int | None,
    checkpoint_ref: str,
    baseline_eval_report: str,
    split: str,
    device_name: str,
    max_samples: int,
    output_dir: str,
    overwrite_output: bool = False,
) -> dict:
    if int(max_samples) < 0:
        raise SystemExit("--max-samples 必须大于等于 0")
    if int(max_samples) == 0 and split != "val":
        raise SystemExit("M7 正式 retention 只接受 --split val")
    config = load_segdesc_config(config_path, {"seed": seed})
    output = resolve_project_path(output_dir) or Path(output_dir)
    baseline_path = (
        resolve_project_path(baseline_eval_report) or Path(baseline_eval_report)
    )
    checkpoint_path = resolve_project_path(checkpoint_ref) or Path(checkpoint_ref)
    if output.exists() and not output.is_dir():
        raise SystemExit(f"retention output-dir 不是目录: {output}")
    protected = {
        "config": config_path,
        "baseline-eval-report": baseline_path,
        "checkpoint": checkpoint_path,
        "segmentation-config": config.model.segmentation_config,
        "segmentation-checkpoint": config.model.segmentation_checkpoint,
        "segmentation-vision-cache": config.model.segmentation_vision_cache,
        "description-vision-cache": config.model.description_vision_cache,
        "description-benchmark": config.data.description_benchmark,
        "bridge-benchmark": config.data.bridge_benchmark,
        "predicted-index": config.data.predicted_index,
        "predicted-val-index": config.data.predicted_val_index,
        "d4-final-acceptance-gate": config.training.d4_final_acceptance_gate,
        "m6-acceptance-gate": config.training.m6_acceptance_gate,
    }
    try:
        validate_output_replacement_safety(output, protected)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        not overwrite_output
        and output.is_dir()
        and any(output.iterdir())
    ):
        raise SystemExit(
            "retention output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    from qpsalm_seg.description.training.checkpoint import load_segdesc_checkpoint
    from qpsalm_seg.description.data.loaders import description_device
    from qpsalm_seg.description.protocols.io import atomic_write_json as write_json
    from qpsalm_seg.description.training.runtime import build_segdesc_model
    from qpsalm_seg.engine.common import build_eval_loader
    from qpsalm_seg.engine.evaluator import evaluate

    try:
        baseline = strict_json_loads(baseline_path.read_text(encoding="utf-8"))
        baseline_binding = baseline_eval_binding(baseline_path, baseline, split=split)
        device = description_device(device_name)
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
            seed=config.training.seed,
            train_region_data_audit=region_data_audit,
        )
        m6_acceptance = revalidate_saved_m6_acceptance(
            checkpoint_payload.get("m6_acceptance"),
            seed=config.training.seed,
            train_region_data_audit=region_data_audit,
        )
        if m6_acceptance.get("d4_final_acceptance") != d4_final_acceptance:
            raise RuntimeError("M7 checkpoint 的 M6 与 D4 final acceptance 不一致")
        joint_initialization_audit = revalidate_joint_initialization_audit(
            checkpoint_payload.get("joint_initialization_audit"),
            expected_seed=config.training.seed,
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
            max_val_samples=max_samples or None,
        )
        replay_threshold = float(baseline_binding["eval_threshold"])
        replay_threshold_sweep = list(baseline_binding["threshold_sweep"])
        baseline_loader = build_eval_loader(segmentation_config, split)
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
        formal_run = int(max_samples) == 0
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
        step, metadata = load_segdesc_checkpoint(checkpoint_ref, model)
        if (
            int(step) != int(checkpoint_preflight["checkpoint_step"])
            or metadata != preflight_metadata
            or sha256_file(checkpoint_path)
            != checkpoint_preflight["checkpoint_sha256"]
        ):
            raise RuntimeError(
                "M7 joint checkpoint 在 preflight 与模型加载之间发生漂移"
            )
        loader = build_eval_loader(segmentation_config, split)
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
            split=split,
            max_samples=max_samples,
            checkpoint=checkpoint_ref,
            checkpoint_step=step,
            checkpoint_metadata=metadata,
            maximum_allowed_drop=config.joint.segmentation_retention_max_drop,
            baseline_binding=baseline_binding,
            expected_seed=config.training.seed,
            d4_final_acceptance_audit=d4_final_acceptance,
            m6_acceptance_audit=m6_acceptance,
            joint_initialization_audit=joint_initialization_audit,
            d_minus_one_acceptance_audit=d_minus_one_acceptance,
            stage_lineage_audit=stage_lineage,
            baseline_checkpoint_replay_audit=(
                baseline_checkpoint_replay_audit
            ),
        )
        gate["joint_checkpoint_sha256"] = sha256_file(checkpoint_path)
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
                expected_seed=config.training.seed,
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
    return gate
