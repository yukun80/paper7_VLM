#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 task-isolated joint-training workflow.

用途：统一 M7 输出所有权、run lifecycle、terminal checkpoint 与失败终态。
推荐调用：由 ``qpsalm-segdesc train joint`` 薄入口传入 config v2。
输入：SegDescConfig、accepted M6 initialization、device 和输出控制参数。
输出：原子发布 joint training_report.json 或 failure_report.json。
写入行为：只写 config.training.output_dir，不修改基准、cache 或源 checkpoint。
工作流阶段：M7 engineering orchestration；科学验收仍依赖 retention gate。
"""

from __future__ import annotations

import shutil
from pathlib import Path
import traceback
from typing import Any

from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)

from ..protocols.config import SegDescConfig
from ..protocols.io import atomic_write_json


class JointTrainingLaunchError(ValueError):
    """The requested M7 initialization or output ownership is unsafe."""


def _prepare_joint_output(
    config: SegDescConfig,
    *,
    config_reference: str,
    resume: str | None,
    initialize_from: str | None,
    overwrite_output: bool,
) -> Path:
    if resume and overwrite_output:
        raise JointTrainingLaunchError(
            "--resume 不能与 --overwrite-output 同时使用"
        )
    if resume and initialize_from:
        raise JointTrainingLaunchError(
            "--resume 不能与 --initialize-from 同时使用"
        )
    if not resume and not initialize_from:
        raise JointTrainingLaunchError(
            "新 M7 run 必须提供 --initialize-from；续训请使用 --resume"
        )
    output = (
        resolve_project_path(config.training.output_dir)
        or Path(config.training.output_dir)
    )
    if output.exists() and not output.is_dir():
        raise JointTrainingLaunchError(f"M7 output-dir 不是目录: {output}")
    try:
        validate_output_replacement_safety(output, {
            "config": config_reference,
            "segmentation-config": config.model.segmentation_config,
            "segmentation-checkpoint": config.model.segmentation_checkpoint,
            "segmentation-vision-cache": config.model.segmentation_vision_cache,
            "description-vision-cache": config.model.description_vision_cache,
            "description-benchmark": config.data.description_benchmark,
            "bridge-benchmark": config.data.bridge_benchmark,
            "initialize-from": initialize_from,
            "predicted-index": config.data.predicted_index,
            "predicted-val-index": config.data.predicted_val_index,
            "d4-final-acceptance-gate": (
                config.training.d4_final_acceptance_gate
            ),
            "m6-acceptance-gate": config.training.m6_acceptance_gate,
        })
    except ValueError as exc:
        raise JointTrainingLaunchError(str(exc)) from exc
    if resume:
        source = resolve_project_path(resume) or Path(resume)
        if (
            not output.is_dir()
            or source.resolve(strict=False).parent
            != output.resolve(strict=False)
        ):
            raise JointTrainingLaunchError(
                "--resume 必须使用同一非空 output-dir 内的 checkpoint，"
                "以保留 baseline、progress 和历史"
            )
    elif output.is_dir() and any(output.iterdir()) and not overwrite_output:
        raise JointTrainingLaunchError(
            "新 M7 run 的 output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_joint_training(
    config: SegDescConfig,
    *,
    config_reference: str,
    device_name: str,
    resume: str | None,
    initialize_from: str | None,
    overwrite_output: bool = False,
) -> dict[str, Any]:
    """Run one M7 training attempt and publish its completion contract."""
    output = _prepare_joint_output(
        config,
        config_reference=config_reference,
        resume=resume,
        initialize_from=initialize_from,
        overwrite_output=overwrite_output,
    )
    from ..training.checkpoint import inspect_segdesc_checkpoint
    from ..training.joint_contracts import JOINT_PROGRESS_PROTOCOL
    from ..training.joint_trainer import train_joint_segdesc
    from ..protocols.versions import JOINT_TRAINING_COMPLETION_PROTOCOL
    from ..training.run_artifacts import (
        build_training_completion_report,
        prepare_training_attempt,
        validate_terminal_checkpoint_provenance,
    )

    attempt_audit = prepare_training_attempt(output, resume=bool(resume))
    try:
        report = train_joint_segdesc(
            config,
            device_name=device_name,
            resume=resume,
            initialize_from=initialize_from,
        )
        optional_artifacts: dict[str, Any] = {
            "joint_gradient_gate": output / "joint_gradient_gate.json",
            "joint_validation_latest": output / "joint_validation_latest.json",
        }
        if report.get("checkpoint_best"):
            optional_artifacts.update({
                "checkpoint_best": report["checkpoint_best"],
                "validation_best": output / "joint_validation_best.json",
            })
        terminal = validate_terminal_checkpoint_provenance(
            inspect_segdesc_checkpoint(report["checkpoint_last"]),
            checkpoint=report["checkpoint_last"],
            expected_step=int(report["steps"]),
            expected_stage="joint",
            progress_key="joint_progress",
            expected_progress_protocol=JOINT_PROGRESS_PROTOCOL,
            progress_artifact=output / "joint_coverage_latest.json",
            progress_artifact_name="joint_coverage_latest",
            history_artifact=output / "joint_history.jsonl",
            history_artifact_name="joint_history",
        )
        completion = build_training_completion_report(
            protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
            report={
                **report,
                "attempt_audit": attempt_audit,
                "terminal_checkpoint_audit": terminal,
            },
            required_artifacts={
                "checkpoint_last": report["checkpoint_last"],
                "joint_coverage_latest": output / "joint_coverage_latest.json",
                "joint_history": output / "joint_history.jsonl",
                "joint_manifest": output / "joint_manifest.json",
                "segmentation_monitor_baseline": (
                    output / "segmentation_monitor_baseline.json"
                ),
            },
            optional_artifacts=optional_artifacts,
        )
        atomic_write_json(output / "training_report.json", completion)
        return report
    except BaseException as exc:
        atomic_write_json(output / "failure_report.json", {
            "protocol": "qpsalm_segdesc_joint_training_failure_v2_attempt_bound",
            "region_stage": config.joint.joint_region_stage,
            "attempt_audit": attempt_audit,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
