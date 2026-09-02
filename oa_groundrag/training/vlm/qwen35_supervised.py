"""Qwen3.5 训练期仅监督位置 logits；不进入 backend、Gate B 或 runtime。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oa_groundrag.vlm.backends.modeling import ForwardResult
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.vlm.processing import model_tensor_batch


LOSS_PROJECTION_ID = "qwen3_5_supervised_positions.v1"
CROSS_ENTROPY_DTYPE = "float32"


def causal_supervised_positions(labels: Tensor) -> tuple[Tensor, Tensor]:
    """返回 causal shift 后标签与 batch=1 的有监督 logit 位置。"""

    if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape[1] < 2:
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "Qwen3.5 监督位置 loss 只接受 batch=1 且序列长度 >= 2",
        )
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    positions = torch.nonzero(
        shifted_labels[0].ne(-100),
        as_tuple=False,
    ).flatten()
    if positions.numel() == 0:
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "Qwen3.5 batch 没有 causal-shift 后 assistant 监督位置",
        )
    return shifted_labels, positions


def supervised_position_causal_loss(
    logits: Tensor,
    shifted_labels: Tensor,
    positions: Tensor,
) -> Tensor:
    """对一维选中位置的 logits 计算原 assistant-token mean FP32 CE。"""

    valid = (
        logits.ndim == 3
        and logits.shape[0] == 1
        and logits.shape[-1] > 0
        and shifted_labels.ndim == 2
        and shifted_labels.shape[0] == 1
        and positions.ndim == 1
        and positions.dtype == torch.long
        and positions.numel() > 0
        and logits.shape[1] == positions.numel()
    )
    if not valid:
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "Qwen3.5 selected logits/labels/positions shape 不符合合同",
        )
    if (
        int(positions[0]) < 0
        or int(positions[-1]) >= shifted_labels.shape[1]
        or not bool(torch.all(positions[1:] > positions[:-1]))
    ):
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "Qwen3.5 logits_to_keep 位置必须严格递增且位于序列内",
        )
    targets = shifted_labels[0].index_select(
        0,
        positions.to(shifted_labels.device),
    )
    if bool(targets.eq(-100).any()):
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "Qwen3.5 选中位置包含 ignore label",
        )
    loss = F.cross_entropy(
        logits[0].float(),
        targets.to(logits.device),
        reduction="mean",
    )
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise ModelError(
            ReasonCode.NONFINITE_NUMBER,
            "Qwen3.5 监督位置 FP32 loss 非有限或 dtype 漂移",
        )
    return loss


class Qwen35SupervisedPositionAdapter:
    """训练专属 decorator；除 forward loss 外显式委托原 adapter。"""

    def __init__(self, base: Any) -> None:
        identity = getattr(base, "identity", None)
        identity_row = (
            identity.to_dict() if callable(getattr(identity, "to_dict", None)) else {}
        )
        if (
            identity_row.get("backend") != "qwen3_5"
            or identity_row.get("model_type") != "qwen3_5"
            or not isinstance(getattr(base, "model", None), nn.Module)
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "监督位置训练包装器只接受 Qwen3.5 adapter",
            )
        self._base = base

    @property
    def model(self) -> nn.Module:
        return self._base.model

    @property
    def identity(self) -> Any:
        return self._base.identity

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return self._base.trainable_names

    @property
    def trainable_parameter_count(self) -> int:
        return self._base.trainable_parameter_count

    def train(self) -> None:
        self._base.train()

    def eval(self) -> None:
        self._base.eval()

    def forward(self, batch: Mapping[str, Any]) -> ForwardResult:
        tensor_batch = model_tensor_batch(batch)
        labels = tensor_batch.pop("labels", None)
        if not isinstance(labels, Tensor):
            raise ModelError(
                ReasonCode.LOSS_MASK_INVALID,
                "Qwen3.5 监督位置训练 forward 缺少 labels",
            )
        shifted_labels, positions = causal_supervised_positions(labels)
        output = self.model(
            **tensor_batch,
            use_cache=False,
            logits_to_keep=positions,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise ModelError(
                ReasonCode.LOSS_MASK_INVALID,
                "Qwen3.5 监督位置 forward 未返回 logits",
            )
        return ForwardResult(
            loss=supervised_position_causal_loss(
                logits,
                shifted_labels,
                positions,
            ),
            logits=logits,
        )

    def generate_text(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._base.generate_text(*args, **kwargs)

    def trainable_state_dict(self) -> Mapping[str, Tensor]:
        return self._base.trainable_state_dict()

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._base.load_trainable_state_dict(state)


def wrap_qwen35_for_supervised_position_training(
    adapter: Any,
) -> Qwen35SupervisedPositionAdapter:
    return Qwen35SupervisedPositionAdapter(adapter)

