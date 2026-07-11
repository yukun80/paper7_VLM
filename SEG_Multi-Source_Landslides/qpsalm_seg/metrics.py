#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Valid-region segmentation metrics and grouped aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def batch_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """逐样本计算指标；padding 像素永远不进入混淆矩阵。"""
    pred = torch.sigmoid(logits) >= float(threshold)
    target_bool = target >= 0.5
    valid = torch.ones_like(target_bool) if valid_mask is None else valid_mask >= 0.5
    if valid.shape[-2:] != target_bool.shape[-2:]:
        valid = torch.nn.functional.interpolate(valid.float(), size=target_bool.shape[-2:], mode="nearest") >= 0.5

    out: list[dict[str, float]] = []
    for p, t, v in zip(pred, target_bool, valid):
        p = p[v]
        t = t[v]
        tp = float((p & t).sum().item())
        fp = float((p & ~t).sum().item())
        fn = float((~p & t).sum().item())
        target_positive = bool(t.any().item())
        pred_positive = bool(p.any().item())
        if not target_positive:
            correct = float(not pred_positive)
            score = 1.0 if correct else 0.0
            out.append(
                {
                    "dice": score,
                    "iou": score,
                    "precision": score,
                    "recall": score,
                    "is_positive": 0.0,
                    "negative_accuracy": correct,
                    "empty_false_positive_rate": fp / max(float(v.sum().item()), 1.0),
                }
            )
            continue
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        out.append(
            {
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "is_positive": 1.0,
            }
        )
    return out


class MetricAccumulator:
    """按 overall、正负样本和数据语义分组聚合指标。"""

    def __init__(self) -> None:
        self.groups: dict[str, list[dict[str, float]]] = defaultdict(list)

    def update(self, metrics: list[dict[str, float]], metadata: list[dict[str, Any]]) -> None:
        for item, meta in zip(metrics, metadata):
            self.groups["overall"].append(item)
            polarity = "positive_only" if item.get("is_positive", 0.0) > 0.5 else "negative_only"
            self.groups[polarity].append(item)
            for group in (
                f"raw_combo={meta.get('raw_combo', 'unknown')}",
                f"canonical_combo={meta.get('canonical_combo', 'unknown')}",
                f"sensor_combo={meta.get('sensor_combo', 'unknown')}",
                f"normalization_combo={meta.get('normalization_combo', 'unknown')}",
                f"dataset={meta.get('dataset_name', 'unknown')}",
                f"gsd_token={meta.get('gsd_token', 'unknown')}",
                f"target_area_px_bin={meta.get('target_area_px_bin', 'unknown')}",
                f"target_area_fraction_bin={meta.get('target_area_fraction_bin', 'unknown')}",
                f"ground_area_m2_bin={meta.get('ground_area_m2_bin', 'unknown')}",
            ):
                self.groups[group].append(item)
                self.groups[f"{group}/{polarity}"].append(item)

    def compute(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for group, values in sorted(self.groups.items()):
            if not values:
                continue
            keys = sorted({key for item in values for key in item})
            summary: dict[str, float] = {"n": float(len(values))}
            for key in keys:
                observed = [float(item[key]) for item in values if key in item]
                if observed:
                    summary[key] = sum(observed) / len(observed)
            out[group] = summary
        return out
