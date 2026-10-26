"""OA-AuxSeg 当前损失合同：BCE 与 Dice。"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def bce_dice_loss(
    logits: Tensor, targets: Tensor, *, smooth: float = 1.0
) -> tuple[Tensor, dict[str, Tensor]]:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape={tuple(logits.shape)} 与 target={tuple(targets.shape)} 不符"
        )
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("二值分割 logits 必须是 B1HW")
    targets = targets.to(dtype=logits.dtype)
    bce = functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    reduce_dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    dice = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
    total = bce + dice
    return total, {"bce": bce.detach(), "dice": dice.detach()}
