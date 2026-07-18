#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6 description evaluation workflow.

用途：统一 GT-mask、fixed-prediction 与 end-to-end 评价的输出所有权和发布。
推荐调用：由 ``qpsalm-segdesc evaluate`` 薄入口传入 config v2。
输入：SegDescConfig、checkpoint、split、device 与反事实开关。
输出：原子发布 eval_report.json；失败时只发布 failure_report.json。
写入行为：只写 config.training.output_dir，不修改 benchmark/cache/checkpoint。
工作流阶段：M6 engineering/scientific evaluation orchestration。
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
from ..protocols.io import atomic_write_json, sha256_file


class DescriptionEvaluationLaunchError(ValueError):
    """The requested evaluation output ownership is unsafe."""


def _prepare_evaluation_output(
    config: SegDescConfig,
    *,
    config_reference: str,
    checkpoint: str,
    overwrite_output: bool,
) -> tuple[Path, Path]:
    output = (
        resolve_project_path(config.training.output_dir)
        or Path(config.training.output_dir)
    )
    checkpoint_path = resolve_project_path(checkpoint) or Path(checkpoint)
    if output.exists() and not output.is_dir():
        raise DescriptionEvaluationLaunchError(
            f"description eval output-dir 不是目录: {output}"
        )
    try:
        validate_output_replacement_safety(output, {
            "config": config_reference,
            "checkpoint": checkpoint_path,
            "segmentation-config": config.model.segmentation_config,
            "segmentation-checkpoint": config.model.segmentation_checkpoint,
            "segmentation-vision-cache": config.model.segmentation_vision_cache,
            "description-vision-cache": config.model.description_vision_cache,
            "description-benchmark": config.data.description_benchmark,
            "bridge-benchmark": config.data.bridge_benchmark,
            "predicted-index": config.data.predicted_index,
        })
    except ValueError as exc:
        raise DescriptionEvaluationLaunchError(str(exc)) from exc
    if output.is_dir() and any(output.iterdir()) and not overwrite_output:
        raise DescriptionEvaluationLaunchError(
            "description eval output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output, checkpoint_path


def run_description_evaluation(
    config: SegDescConfig,
    *,
    config_reference: str,
    checkpoint: str,
    split: str,
    device_name: str,
    run_counterfactuals: bool = True,
    overwrite_output: bool = False,
) -> dict[str, Any]:
    """Run and publish one source-bound description evaluation."""
    output, checkpoint_path = _prepare_evaluation_output(
        config,
        config_reference=config_reference,
        checkpoint=checkpoint,
        overwrite_output=overwrite_output,
    )
    from ..data.loaders import (
        build_description_dataset,
        build_description_loader,
        description_device,
        set_description_seed,
    )
    from ..evaluation.publication import (
        build_evaluation_publication_audit,
        validate_evaluation_checkpoint_binding,
    )
    from ..evaluation.runner import evaluate_description
    from ..training.checkpoint import load_segdesc_checkpoint
    from ..training.runtime import build_segdesc_model

    try:
        set_description_seed(config.training.seed)
        device = description_device(device_name)
        model, runtime_migration = build_segdesc_model(config, device)
        step, checkpoint_report = load_segdesc_checkpoint(
            checkpoint_path, model
        )
        dataset = build_description_dataset(
            config,
            model.description_backbone.bank,
            split=split,
            training=False,
        )
        checkpoint_binding = validate_evaluation_checkpoint_binding(
            config,
            checkpoint_report,
            runtime_migration,
            getattr(dataset, "predicted_index_audit", None),
            checkpoint=checkpoint_path,
        )
        loader = build_description_loader(dataset, config, training=False)
        report = evaluate_description(
            model,
            loader,
            config,
            device,
            split=split,
            output_dir=output,
            run_counterfactuals=run_counterfactuals,
            publish_report=False,
        )
        report.update({
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_step": step,
            "checkpoint_metadata": checkpoint_report,
            "checkpoint_binding": checkpoint_binding,
        })
        report["publication_audit"] = build_evaluation_publication_audit(
            output, report
        )
        atomic_write_json(output / "eval_report.json", report)
        return report
    except BaseException as exc:
        # 正式路径只允许一个终态报告，避免 partial report 被误当作验收结果。
        (output / "eval_report.json").unlink(missing_ok=True)
        (output / "eval_report.json.tmp").unlink(missing_ok=True)
        atomic_write_json(output / "failure_report.json", {
            "protocol": "qpsalm_description_evaluation_failure_v2_no_partial_report",
            "stage": config.training.stage,
            "evaluation_mode": config.evaluation.evaluation_mode,
            "eval_report_published": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
