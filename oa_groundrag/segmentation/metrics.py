"""OA-AuxSeg 二值分割与 no-target 指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _safe_ratio(numerator: float, denominator: float, *, empty: float = 1.0) -> float:
    return numerator / denominator if denominator > 0 else empty


@dataclass
class SegmentationMetrics:
    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    positive_tp: int = 0
    positive_fp: int = 0
    positive_fn: int = 0
    empty_count: int = 0
    empty_false_positive: int = 0
    empty_probability_sum: float = 0.0
    sample_count: int = 0

    @torch.no_grad()
    def update(self, probability: Tensor, target: Tensor) -> None:
        if probability.shape != target.shape:
            raise ValueError("metric probability 与 target shape 不一致")
        prediction = probability >= self.threshold
        truth = target >= 0.5
        self.tp += int((prediction & truth).sum().item())
        self.fp += int((prediction & ~truth).sum().item())
        self.fn += int((~prediction & truth).sum().item())
        for index in range(probability.shape[0]):
            sample_truth = truth[index]
            sample_prediction = prediction[index]
            if bool(sample_truth.any()):
                self.positive_tp += int(
                    (sample_prediction & sample_truth).sum().item()
                )
                self.positive_fp += int(
                    (sample_prediction & ~sample_truth).sum().item()
                )
                self.positive_fn += int(
                    (~sample_prediction & sample_truth).sum().item()
                )
            else:
                self.empty_count += 1
                self.empty_false_positive += int(bool(sample_prediction.any()))
                self.empty_probability_sum += float(
                    probability[index].mean().item()
                )
        self.sample_count += int(probability.shape[0])

    def compute(self) -> dict[str, Any]:
        iou = _safe_ratio(self.tp, self.tp + self.fp + self.fn)
        dice = _safe_ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        precision = _safe_ratio(self.tp, self.tp + self.fp)
        recall = _safe_ratio(self.tp, self.tp + self.fn)
        positive_iou = _safe_ratio(
            self.positive_tp,
            self.positive_tp + self.positive_fp + self.positive_fn,
        )
        positive_dice = _safe_ratio(
            2 * self.positive_tp,
            2 * self.positive_tp + self.positive_fp + self.positive_fn,
        )
        return {
            "sample_count": self.sample_count,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": dice,
            "positive_only_iou": positive_iou,
            "positive_only_dice": positive_dice,
            "no_target_count": self.empty_count,
            "no_target_false_positive_rate": _safe_ratio(
                self.empty_false_positive, self.empty_count, empty=0.0
            ),
            "empty_mean_foreground_probability": _safe_ratio(
                self.empty_probability_sum, self.empty_count, empty=0.0
            ),
        }
