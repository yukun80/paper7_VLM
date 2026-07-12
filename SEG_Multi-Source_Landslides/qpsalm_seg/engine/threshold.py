#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold sweep and original-canvas metric restoration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qpsalm_seg.metrics import MetricAccumulator, batch_binary_metrics
from qpsalm_seg.visualize import restore_mask_to_original


def normalize_thresholds(values) -> list[float]:
    return sorted({round(float(value), 4) for value in (values or []) if 0 <= float(value) <= 1})


def compute_threshold_sweep_report(accumulators: dict[float, MetricAccumulator]) -> dict[str, Any]:
    by_threshold, groups_by_threshold = {}, {}
    best_dice = best_iou = None
    best_dice_groups: dict[str, dict[str, Any]] = {}
    best_iou_groups: dict[str, dict[str, Any]] = {}
    for threshold, accumulator in sorted(accumulators.items()):
        groups = accumulator.compute()
        overall = groups.get("overall")
        if not overall:
            continue
        key = f"{threshold:.2f}"
        by_threshold[key], groups_by_threshold[key] = overall, groups
        candidate = {"threshold": threshold, **overall}
        if best_dice is None or candidate.get("dice", -1) > best_dice.get("dice", -1):
            best_dice = candidate
        if best_iou is None or candidate.get("iou", -1) > best_iou.get("iou", -1):
            best_iou = candidate
        for group, values in groups.items():
            row = {"threshold": threshold, **values}
            if group not in best_dice_groups or row.get("dice", -1) > best_dice_groups[group].get("dice", -1):
                best_dice_groups[group] = row
            if group not in best_iou_groups or row.get("iou", -1) > best_iou_groups[group].get("iou", -1):
                best_iou_groups[group] = row
    return {
        "overall_by_threshold": by_threshold,
        "groups_by_threshold": groups_by_threshold,
        "best_by_dice": best_dice,
        "best_by_iou": best_iou,
        "best_by_dice_per_group": dict(sorted(best_dice_groups.items())),
        "best_by_iou_per_group": dict(sorted(best_iou_groups.items())),
    }


def restored_original_space_metrics(logits, target, metadata, threshold, valid_mask=None):
    probabilities = torch.sigmoid(logits.detach().float().cpu())
    targets = target.detach().float().cpu()
    valids = valid_mask.detach().float().cpu() if valid_mask is not None else None
    records = []
    for index, meta in enumerate(metadata):
        prediction = (probabilities[index, 0].numpy() >= float(threshold)).astype(np.uint8)
        truth = (targets[index, 0].numpy() >= 0.5).astype(np.uint8)
        restored_prediction = restore_mask_to_original(prediction, meta.get("resize_transform"))
        restored_truth = restore_mask_to_original(truth, meta.get("resize_transform"))
        restored_valid = (
            restore_mask_to_original(
                (valids[index, 0].numpy() >= 0.5).astype(np.uint8), meta.get("resize_transform")
            )
            if valids is not None else None
        )
        if (
            restored_prediction is None
            or restored_truth is None
            or (valids is not None and restored_valid is None)
        ):
            sample_id = meta.get("sample_id", f"batch_index={index}")
            raise ValueError(
                f"cannot restore original-space metrics for sample={sample_id}; "
                f"invalid resize_transform={meta.get('resize_transform')!r}"
            )
        pred_tensor = torch.from_numpy(restored_prediction).float()[None, None]
        truth_tensor = torch.from_numpy(restored_truth).float()[None, None]
        valid_tensor = (
            torch.from_numpy(restored_valid).float()[None, None]
            if restored_valid is not None else None
        )
        pred_logits = torch.where(pred_tensor > 0.5, pred_tensor.new_tensor(20), pred_tensor.new_tensor(-20))
        records.extend(batch_binary_metrics(pred_logits, truth_tensor, threshold=0.5, valid_mask=valid_tensor))
    return records


def canvas_original_metric_delta(canvas, original):
    canvas_overall, original_overall = canvas.get("overall") or {}, original.get("overall") or {}
    return {
        key: float(canvas_overall.get(key, 0)) - float(original_overall.get(key, 0))
        for key in ("dice", "iou", "precision", "recall")
        if key in canvas_overall and key in original_overall
    }
