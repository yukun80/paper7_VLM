#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D-1/D0-D4 description training workflow.

用途：统一输出所有权、preflight、训练 completion 与失败终态。
推荐调用：由 ``qpsalm-segdesc train`` 薄入口传入已解析的 config v2。
输入：SegDescConfig、设备、resume/initialize 与输出控制参数。
输出：preflight report 或训练 report，并原子发布 run lifecycle artifacts。
写入行为：只写 config.training.output_dir，不修改 benchmark/cache/checkpoint 输入。
工作流阶段：M5/M6 D-1 与 D0-D4。
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


class DescriptionLaunchError(ValueError):
    """The requested run ownership or launch mode is unsafe."""


def _validate_launch_mode(
    *,
    resume: str | None,
    initialize_from: str | None,
    overwrite_output: bool,
    preflight_only: bool,
    formal_output_dir: str | None,
    d0_preflight_report: str | None,
) -> None:
    if preflight_only and (resume or initialize_from):
        raise DescriptionLaunchError(
            "--preflight-only 禁止 --resume/--initialize-from"
        )
    if preflight_only and not formal_output_dir:
        raise DescriptionLaunchError("--preflight-only 必须提供 --formal-output-dir")
    if preflight_only and d0_preflight_report:
        raise DescriptionLaunchError(
            "--preflight-only 不接受 --d0-preflight-report"
        )
    if not preflight_only and formal_output_dir:
        raise DescriptionLaunchError(
            "--formal-output-dir 只允许与 --preflight-only 一起使用"
        )
    if resume and overwrite_output:
        raise DescriptionLaunchError("--resume 不能与 --overwrite-output 同时使用")
    if resume and initialize_from:
        raise DescriptionLaunchError("--resume 不能与 --initialize-from 同时使用")


def _prepare_output(
    config: SegDescConfig,
    *,
    config_reference: str,
    resume: str | None,
    initialize_from: str | None,
    overwrite_output: bool,
    d0_preflight_report: str | None,
) -> Path:
    output = (
        resolve_project_path(config.training.output_dir)
        or Path(config.training.output_dir)
    )
    output_resolved = output.resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise DescriptionLaunchError(
            f"description training output-dir 不是目录: {output}"
        )
    try:
        validate_output_replacement_safety(output, {
            "config": config_reference,
            "segmentation-config": config.model.segmentation_config,
            "segmentation-checkpoint": config.model.segmentation_checkpoint,
            "segmentation-vision-cache": config.model.segmentation_vision_cache,
            "description-vision-cache": config.model.description_vision_cache,
            "description-benchmark": config.data.description_benchmark,
            "bridge-benchmark": config.data.bridge_benchmark,
            "unified-benchmark": config.data.unified_benchmark,
            "artifact-readiness-report": (
                config.data.artifact_readiness_report
            ),
            "initialize-from": initialize_from,
            "predicted-index": config.data.predicted_index,
            "predicted-val-index": config.data.predicted_val_index,
            "d-minus-one-gate": config.training.d_minus_one_gate,
            "d4-curriculum-gate": config.training.d4_curriculum_gate,
            "d0-preflight-report": d0_preflight_report,
        })
    except ValueError as exc:
        raise DescriptionLaunchError(str(exc)) from exc
    if resume:
        source = resolve_project_path(resume) or Path(resume)
        if not output.is_dir() or source.resolve(strict=False).parent != output_resolved:
            raise DescriptionLaunchError(
                "--resume 必须使用同一非空 output-dir 内的 checkpoint，"
                "以保留历史、数据审计和失败记录"
            )
    elif output.is_dir() and any(output.iterdir()) and not overwrite_output:
        raise DescriptionLaunchError(
            "新 run 的 output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_description_training(
    config: SegDescConfig,
    *,
    config_reference: str,
    device_name: str,
    resume: str | None = None,
    initialize_from: str | None = None,
    overwrite_output: bool = False,
    preflight_only: bool = False,
    formal_output_dir: str | None = None,
    d0_preflight_report: str | None = None,
) -> dict[str, Any]:
    """Run one construction-only preflight or one complete description run."""
    _validate_launch_mode(
        resume=resume,
        initialize_from=initialize_from,
        overwrite_output=overwrite_output,
        preflight_only=preflight_only,
        formal_output_dir=formal_output_dir,
        d0_preflight_report=d0_preflight_report,
    )
    if (
        not preflight_only
        and config.training.stage == "mmrs_caption"
        and (overwrite_output or initialize_from)
    ):
        forbidden = (
            "--overwrite-output" if overwrite_output else "--initialize-from"
        )
        raise DescriptionLaunchError(
            "正式 D0 必须逐字消费 preflight 发布的安全启动语义；"
            f"禁止追加 {forbidden}"
        )
    d0_preflight_acceptance: dict[str, Any] | None = None
    artifact_readiness_acceptance: dict[str, Any] | None = None
    if config.training.stage == "overfit":
        if not config.data.artifact_readiness_report:
            raise DescriptionLaunchError(
                "D-1 overfit 必须提供 --artifact-readiness-report"
            )
        from ..data.artifact_readiness import (
            validate_artifact_readiness_report,
        )

        try:
            artifact_readiness_acceptance = (
                validate_artifact_readiness_report(
                    config.data.artifact_readiness_report,
                    expected_description_benchmark=(
                        config.data.description_benchmark
                    ),
                    expected_bridge_benchmark=config.data.bridge_benchmark,
                    expected_unified_benchmark=config.data.unified_benchmark,
                    expected_description_cache=(
                        config.model.description_vision_cache
                    ),
                )
            )
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise DescriptionLaunchError(str(exc)) from exc
    if not preflight_only and config.training.stage == "mmrs_caption":
        if not d0_preflight_report:
            raise DescriptionLaunchError(
                "正式 D0 必须提供 --d0-preflight-report"
            )
        from .d0_acceptance import validate_d0_preflight_for_launch

        try:
            d0_preflight_acceptance = validate_d0_preflight_for_launch(
                config,
                config_reference=config_reference,
                report_reference=d0_preflight_report,
                device_name=device_name,
            )
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise DescriptionLaunchError(str(exc)) from exc
    elif d0_preflight_report:
        raise DescriptionLaunchError(
            "--d0-preflight-report 只允许正式 D0 stage=mmrs_caption"
        )
    output = _prepare_output(
        config,
        config_reference=config_reference,
        resume=resume,
        initialize_from=initialize_from,
        overwrite_output=overwrite_output,
        d0_preflight_report=d0_preflight_report,
    )
    if preflight_only:
        from .d0_preflight import run_d0_preflight

        return run_d0_preflight(
            config,
            device_name=device_name,
            output_dir=output,
            formal_output_dir=str(formal_output_dir),
        )

    from ..training.checkpoint import inspect_segdesc_checkpoint
    from ..protocols.versions import DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
    from ..training.run_artifacts import (
        build_training_completion_report,
        prepare_training_attempt,
        validate_terminal_checkpoint_provenance,
    )
    from ..training.streams import DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
    from ..training.trainer import train_description

    attempt_audit = prepare_training_attempt(output, resume=bool(resume))
    if d0_preflight_acceptance is not None:
        atomic_write_json(
            output / "d0_preflight_acceptance.json",
            d0_preflight_acceptance,
        )
    try:
        report = train_description(
            config,
            device_name=device_name,
            resume=resume,
            initialize_from=initialize_from,
            artifact_readiness_acceptance=(
                artifact_readiness_acceptance
            ),
            d0_preflight_acceptance=d0_preflight_acceptance,
        )
        optional_artifacts: dict[str, Any] = {
            "validation_best": output / "validation_best.json",
            "d_minus_one_overfit_validation": (
                output / "d_minus_one_overfit_validation.json"
            ),
        }
        if report.get("checkpoint_best"):
            optional_artifacts["checkpoint_best"] = report["checkpoint_best"]
        terminal = validate_terminal_checkpoint_provenance(
            inspect_segdesc_checkpoint(report["checkpoint_last"]),
            checkpoint=report["checkpoint_last"],
            expected_step=int(report["steps"]),
            expected_stage=config.training.stage,
            progress_key="training_progress",
            expected_progress_protocol=DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
            progress_artifact=output / "training_progress_latest.json",
            progress_artifact_name="training_progress_latest",
            history_artifact=output / "train_history.jsonl",
            history_artifact_name="train_history",
        )
        required_artifacts: dict[str, Any] = {
            "checkpoint_last": report["checkpoint_last"],
            "dataset_summary": output / "dataset_summary.json",
            "resolved_config": output / "resolved_config.json",
            "train_history": output / "train_history.jsonl",
            "training_progress_latest": (
                output / "training_progress_latest.json"
            ),
            "trainable_parameter_manifest": (
                output / "trainable_parameter_manifest.json"
            ),
        }
        if d0_preflight_acceptance is not None:
            required_artifacts["d0_preflight_acceptance"] = (
                output / "d0_preflight_acceptance.json"
            )
        completion = build_training_completion_report(
            protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
            report={
                **report,
                "attempt_audit": attempt_audit,
                "terminal_checkpoint_audit": terminal,
            },
            required_artifacts=required_artifacts,
            optional_artifacts=optional_artifacts,
        )
        atomic_write_json(output / "training_report.json", completion)
        return report
    except BaseException as exc:
        atomic_write_json(output / "failure_report.json", {
            "protocol": "qpsalm_description_training_failure_v2_attempt_bound",
            "stage": config.training.stage,
            "attempt_audit": attempt_audit,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
