#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多尺度特征融合与可选 box prior refinement。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import ConvBlock


class MultiScaleFeatureFusion(nn.Module):
    """将多源 adapter 输出整理为 mask feature 与 transformer memory feature。

    优先使用 per-modality feature stack：每个模态先形成 high/mid/low 金字塔，
    再按 condition-aware gate 在各尺度融合。这样 DEM/InSAR/SAR 证据不会在进入
    多尺度空间恢复前被过早平均掉。若旧调用只传 fused_modalities，仍走兼容路径。
    """

    def __init__(self, decoder_dim: int, use_box_prior: bool = False) -> None:
        super().__init__()
        d = int(decoder_dim)
        self.stem = ConvBlock(d, d)
        self.fuse_high = ConvBlock(d, d)
        self.box_prior_adapter = ConvBlock(1, d) if use_box_prior else None
        self.down1 = nn.Sequential(nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1), nn.GELU(), ConvBlock(d, d))
        self.down2 = nn.Sequential(nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1), nn.GELU(), ConvBlock(d, d))
        self.mid_lateral = nn.Conv2d(d, d, kernel_size=1)
        self.low_lateral = nn.Conv2d(d, d, kernel_size=1)
        self.fpn_mask_fuse = ConvBlock(d, d)
        self.fpn_memory_fuse = ConvBlock(d, d)
        self.scale_gate_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d * 3),
                    nn.Linear(d * 3, d),
                    nn.GELU(),
                    nn.Linear(d, 1),
                )
                for name in ("high", "mid", "low")
            }
        )

    def _single_pyramid(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high = self.fuse_high(self.stem(feature))
        mid = self.down1(high)
        low = self.down2(mid)
        return high, mid, low

    @staticmethod
    def _weighted_sum(features: torch.Tensor | list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(features, dim=1) if isinstance(features, list) else features
        view_shape = (weights.shape[0], weights.shape[1]) + (1,) * (stacked.ndim - 2)
        return (stacked * weights.view(view_shape)).sum(dim=1)

    def _scale_gate_weights(
        self,
        scale_name: str,
        scale_features: torch.Tensor,
        base_gate_weights: torch.Tensor,
        condition_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """在 base modality gate 上加入尺度特征条件化偏移。"""
        if condition_embedding is None:
            return base_gate_weights
        pooled = scale_features.mean(dim=(3, 4))
        condition = condition_embedding.unsqueeze(1).expand_as(pooled)
        context = torch.cat([pooled, condition, pooled * condition], dim=-1)
        offsets = self.scale_gate_heads[scale_name](context).squeeze(-1)
        active = base_gate_weights > 0
        logits = torch.log(base_gate_weights.clamp_min(1.0e-6)) + offsets
        logits = logits.masked_fill(~active, -1.0e4)
        weights = torch.softmax(logits, dim=1) * active.to(logits.dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def _fuse_modality_pyramids(
        self,
        modality_features: torch.Tensor,
        gate_weights: torch.Tensor,
        condition_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        highs: list[torch.Tensor] = []
        mids: list[torch.Tensor] = []
        lows: list[torch.Tensor] = []
        for idx in range(modality_features.shape[1]):
            high_i, mid_i, low_i = self._single_pyramid(modality_features[:, idx])
            highs.append(high_i)
            mids.append(mid_i)
            lows.append(low_i)
        high_stack = torch.stack(highs, dim=1)
        mid_stack = torch.stack(mids, dim=1)
        low_stack = torch.stack(lows, dim=1)
        high_gate = self._scale_gate_weights("high", high_stack, gate_weights, condition_embedding)
        mid_gate = self._scale_gate_weights("mid", mid_stack, gate_weights, condition_embedding)
        low_gate = self._scale_gate_weights("low", low_stack, gate_weights, condition_embedding)
        high = self._weighted_sum(high_stack, high_gate)
        mid = self._weighted_sum(mid_stack, mid_gate)
        low = self._weighted_sum(low_stack, low_gate)
        return high, mid, low, {
            "scale_gate_high": high_gate,
            "scale_gate_mid": mid_gate,
            "scale_gate_low": low_gate,
        }

    def forward(
        self,
        fused_modalities: torch.Tensor,
        bbox_prior: torch.Tensor | None = None,
        modality_features: torch.Tensor | None = None,
        gate_weights: torch.Tensor | None = None,
        condition_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scale_gates: dict[str, torch.Tensor] = {}
        if modality_features is not None and gate_weights is not None:
            high, mid, low, scale_gates = self._fuse_modality_pyramids(
                modality_features,
                gate_weights,
                condition_embedding=condition_embedding,
            )
        else:
            high, mid, low = self._single_pyramid(fused_modalities)
        if self.box_prior_adapter is not None:
            if bbox_prior is None:
                bbox_prior = torch.zeros(
                    (high.shape[0], 1, high.shape[-2], high.shape[-1]),
                    dtype=high.dtype,
                    device=high.device,
                )
            high = high + self.box_prior_adapter(bbox_prior)

        mid_up = F.interpolate(self.mid_lateral(mid), size=high.shape[-2:], mode="bilinear", align_corners=False)
        low_up = F.interpolate(self.low_lateral(low), size=high.shape[-2:], mode="bilinear", align_corners=False)
        mask_features = self.fpn_mask_fuse(high + mid_up + low_up)
        high_down = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        mid_down = F.interpolate(mid, size=low.shape[-2:], mode="bilinear", align_corners=False)
        memory_features = self.fpn_memory_fuse(low + high_down + mid_down)
        return {
            "mask_features": mask_features,
            "memory_features": memory_features,
            "high_features": high,
            "mid_features": mid,
            "low_features": low,
            **scale_gates,
        }
