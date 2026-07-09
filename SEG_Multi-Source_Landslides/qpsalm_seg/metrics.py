#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分割指标与分组聚合。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def batch_binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> list[dict[str, float]]:
    """逐样本计算 Dice/IoU/Precision/Recall。"""
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = (target >= 0.5).float()
    out: list[dict[str, float]] = []
    for p, t in zip(pred, target):
        p = p.reshape(-1)
        t = t.reshape(-1)
        tp = float((p * t).sum().item())
        fp = float((p * (1.0 - t)).sum().item())
        fn = float(((1.0 - p) * t).sum().item())
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            precision = recall = dice = iou = 1.0
        else:
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
            iou = tp / (tp + fp + fn + 1e-6)
        out.append({"dice": dice, "iou": iou, "precision": precision, "recall": recall})
    return out


class MetricAccumulator:
    """按 overall/dataset/modality/sensor/normalization 分组聚合指标。"""

    def __init__(self) -> None:
        self.groups: dict[str, list[dict[str, float]]] = defaultdict(list)

    def update(self, metrics: list[dict[str, float]], metadata: list[dict[str, Any]]) -> None:
        for item, meta in zip(metrics, metadata):
            self.groups["overall"].append(item)
            self.groups[f"raw_combo={meta.get('raw_combo', 'unknown')}"].append(item)
            self.groups[f"canonical_combo={meta.get('canonical_combo', 'unknown')}"].append(item)
            self.groups[f"sensor_combo={meta.get('sensor_combo', 'unknown')}"].append(item)
            self.groups[f"normalization_combo={meta.get('normalization_combo', 'unknown')}"].append(item)
            self.groups[f"dataset={meta.get('dataset_name', 'unknown')}"].append(item)

    def compute(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for group, values in sorted(self.groups.items()):
            if not values:
                continue
            keys = values[0].keys()
            out[group] = {
                key: sum(item[key] for item in values) / len(values)
                for key in keys
            }
            out[group]["n"] = float(len(values))
        return out
