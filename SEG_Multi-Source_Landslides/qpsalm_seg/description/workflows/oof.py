#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D4 out-of-fold prediction workflows.

用途：集中 OOF fold 构建、预测区域导出与 held-out prediction 合并。
推荐调用：由 build_oof_folds/export_predicted_regions/merge_oof_predictions 薄入口调用。
输入：冻结 Bridge expert index、segmentation index/checkpoint/cache 与 fold manifest。
输出：fold indexes、source-bound predicted masks/index 或 merged OOF index。
写入行为：只写显式输出，不修改 benchmark、checkpoint、cache 或 datasets。
工作流阶段：M6 D4 artifact orchestration。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)
from qpsalm_seg.presets import apply_preset

from ..data.oof import build_oof_fold_indexes
from ..data.predicted_regions import (
    export_predicted_regions,
    merge_oof_predictions,
)


class OOFLaunchError(ValueError):
    """The requested D4 output ownership is unsafe."""


def run_oof_fold_build(
    *,
    segmentation_index: str,
    bridge_index: str,
    num_folds: int,
    seed: int,
    output_dir: str,
    overwrite_output: bool,
) -> dict[str, Any]:
    output = resolve_project_path(output_dir) or Path(output_dir)
    try:
        validate_output_replacement_safety(output, {
            "segmentation-index": segmentation_index,
            "bridge-index": bridge_index,
        })
    except ValueError as exc:
        raise OOFLaunchError(str(exc)) from exc
    if output.exists() and not output.is_dir():
        raise OOFLaunchError(f"OOF output-dir 不是目录: {output}")
    if output.exists():
        if not overwrite_output:
            raise OOFLaunchError(
                f"output 已存在，使用 --overwrite-output: {output}"
            )
        shutil.rmtree(output)
    return build_oof_fold_indexes(
        segmentation_index=segmentation_index,
        bridge_index=bridge_index,
        output_dir=output,
        num_folds=num_folds,
        seed=seed,
    )


def run_oof_merge(
    *, fold_manifest: str, input_indexes: list[str], output: str
) -> dict[str, Any]:
    return merge_oof_predictions(
        fold_manifest=fold_manifest,
        input_indexes=input_indexes,
        output=output,
    )


def run_predicted_region_export(
    *,
    segmentation_config: str,
    preset: str | None,
    checkpoint: str,
    source_index: str,
    split: str,
    vision_feature_cache: str | None,
    train_index: str | None,
    val_index: str | None,
    prediction_index: str | None,
    fold_manifest: str | None,
    checkpoint_fold: str | None,
    threshold: float,
    max_parents: int,
    device_name: str,
    output_dir: str,
    overwrite_output: bool,
) -> dict[str, Any]:
    config = apply_preset(load_config(segmentation_config), preset)
    config = apply_config_overrides(config, {
        "vision_feature_cache": vision_feature_cache,
        "train_index": train_index,
        "val_index": val_index,
        "modality_dropout": 0.0,
        "train_hflip_prob": 0.0,
        "train_vflip_prob": 0.0,
    })
    output = resolve_project_path(output_dir) or Path(output_dir)
    try:
        validate_output_replacement_safety(output, {
            "checkpoint": checkpoint,
            "source-index": source_index,
            "vision-feature-cache": vision_feature_cache,
            "train-index": train_index,
            "val-index": val_index,
            "prediction-index": prediction_index,
            "fold-manifest": fold_manifest,
        })
    except ValueError as exc:
        raise OOFLaunchError(str(exc)) from exc
    if output.exists() and not output.is_dir():
        raise OOFLaunchError(
            f"predicted-region output-dir 不是目录: {output}"
        )
    if output.is_dir() and any(output.iterdir()) and not overwrite_output:
        raise OOFLaunchError(
            "predicted-region output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if overwrite_output and output.exists():
        shutil.rmtree(output)
    return export_predicted_regions(
        segmentation_config=config,
        checkpoint=checkpoint,
        source_index=source_index,
        split=split,
        output_dir=output,
        device=torch.device(device_name),
        threshold=threshold,
        fold_manifest=fold_manifest,
        checkpoint_fold=checkpoint_fold,
        prediction_index=prediction_index,
        max_parents=max_parents,
    )
