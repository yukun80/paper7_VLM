"""VLM backend 共用的训练输出与 assistant-only loss。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from oa_groundrag.vlm.errors import ModelError, ReasonCode


@dataclass(frozen=True)
class ForwardResult:
    loss: Tensor
    logits: Tensor | None


def assistant_sample_mean_causal_loss(
    logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """对 batch 内每个样本的 assistant token loss 等权平均。"""

    if (
        logits.ndim != 3
        or labels.ndim != 2
        or logits.shape[:2] != labels.shape
        or logits.shape[-1] <= 0
    ):
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "logits/labels shape 不符合 causal loss 合同",
        )
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    supervised = shifted_labels.ne(-100)
    counts = supervised.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "batch 中存在没有 assistant 监督 token 的样本",
        )
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1).to(logits.device),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    sample_losses = (
        (token_losses * supervised.to(token_losses.dtype)).sum(dim=1)
        / counts.to(token_losses.dtype)
    )
    loss = sample_losses.mean()
    if not bool(torch.isfinite(loss)):
        raise ModelError(
            ReasonCode.NONFINITE_NUMBER,
            "逐样本 assistant-only loss 非有限",
        )
    return loss
