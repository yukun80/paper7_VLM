#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多源遥感模态 adapter 与 condition-aware gating。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.data import CANONICAL_CHANNELS, CANONICAL_MODALITIES

from .common import ConvBlock, _valid_group_count


MODALITY_ADAPTER_SETTINGS = {
    "hr_optical": {"use_gradients": False, "dilation": 1, "hidden_ratio": 0.50},
    "s2": {"use_gradients": False, "dilation": 2, "hidden_ratio": 0.75},
    "s1": {"use_gradients": True, "dilation": 3, "hidden_ratio": 0.50},
    "dem": {"use_gradients": True, "dilation": 2, "hidden_ratio": 0.50},
    "insar": {"use_gradients": True, "dilation": 2, "hidden_ratio": 0.50},
}


class ChannelAttention(nn.Module):
    """轻量通道注意力，帮助 S2/SAR 等多通道模态学习有效波段或极化组合。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, int(channels) * 2)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialContextBlock(nn.Module):
    """depthwise dilated context，用很小代价扩大 SAR/DEM/InSAR 的地学上下文感受野。"""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = _valid_group_count(channels)
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=int(dilation),
                dilation=int(dilation),
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RemoteSensingModalityAdapter(nn.Module):
    """面向单一遥感模态的轻量 adapter，保留物理模态差异而不是统一裸卷积。"""

    def __init__(
        self,
        name: str,
        in_channels: int,
        decoder_dim: int,
        use_gradients: bool = False,
        dilation: int = 1,
        hidden_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.name = name
        self.use_gradients = bool(use_gradients)
        norm_groups = _valid_group_count(in_channels, preferred=min(4, max(1, in_channels)))
        hidden = max(16, int(round(float(decoder_dim) * float(hidden_ratio))))
        hidden = min(int(decoder_dim), hidden)
        stem_channels = int(in_channels) * (3 if self.use_gradients else 1)
        self.input_norm = nn.GroupNorm(norm_groups, in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(stem_channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.local_encoder = ConvBlock(hidden, decoder_dim)
        self.context = SpatialContextBlock(decoder_dim, dilation=max(1, int(dilation)))

    @staticmethod
    def _gradient_features(x: torch.Tensor) -> torch.Tensor:
        dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
        return torch.cat([x, dx, dy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(self.input_norm(x))
        if self.use_gradients:
            x = self._gradient_features(x)
        return self.context(self.local_encoder(self.stem(x)))


class MultiSourceAdapterBank(nn.Module):
    """每类遥感模态独立编码，再按可用性和条件语义做可学习融合。"""

    def __init__(self, decoder_dim: int, modality_dropout: float) -> None:
        super().__init__()
        self.modality_dropout = float(modality_dropout)
        self.adapters = nn.ModuleDict(
            {
                name: RemoteSensingModalityAdapter(
                    name=name,
                    in_channels=CANONICAL_CHANNELS[name],
                    decoder_dim=decoder_dim,
                    **MODALITY_ADAPTER_SETTINGS[name],
                )
                for name in CANONICAL_MODALITIES
            }
        )
        self.modality_embeddings = nn.Parameter(torch.zeros(len(CANONICAL_MODALITIES), decoder_dim))
        self.task_condition_gate = nn.Sequential(
            nn.LayerNorm(decoder_dim * 4),
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 1),
        )
        nn.init.normal_(self.modality_embeddings, std=0.02)

    def _drop_availability(self, availability: torch.Tensor) -> torch.Tensor:
        """训练时随机丢弃可用模态，同时保证每个样本至少保留一个可用模态。"""
        if not self.training or self.modality_dropout <= 0:
            return availability
        keep = torch.ones_like(availability)
        random_drop = torch.rand_like(availability) < self.modality_dropout
        keep[random_drop] = 0.0
        keep = keep * availability
        empty = keep.sum(dim=1) <= 0
        if empty.any():
            for row_idx in torch.nonzero(empty, as_tuple=False).flatten():
                available_idx = torch.nonzero(availability[row_idx] > 0, as_tuple=False).flatten()
                if available_idx.numel() > 0:
                    keep[row_idx, available_idx[0]] = 1.0
        return keep

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        availability: torch.Tensor,
        condition_embedding: torch.Tensor,
        proposal_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = []
        active = self._drop_availability(availability)
        for idx, name in enumerate(CANONICAL_MODALITIES):
            feat = self.adapters[name](modalities[name])
            feat = feat + self.modality_embeddings[idx].view(1, -1, 1, 1)
            encoded.append(feat)
        stacked = torch.stack(encoded, dim=1)

        if proposal_context is None:
            proposal_context = torch.zeros_like(condition_embedding)
        pooled_features = stacked.mean(dim=(3, 4))
        condition = condition_embedding.unsqueeze(1).expand_as(pooled_features)
        proposal = proposal_context.unsqueeze(1).expand_as(pooled_features)
        gate_context = torch.cat(
            [pooled_features, condition, proposal, pooled_features * condition],
            dim=-1,
        )
        logits = self.task_condition_gate(gate_context).squeeze(-1)
        logits = logits.masked_fill(active <= 0, -1.0e4)
        gate_weights = torch.softmax(logits, dim=1) * active
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        fused = (stacked * gate_weights[:, :, None, None, None]).sum(dim=1)
        feature_norms = stacked.detach().float().flatten(2).pow(2).mean(dim=2).sqrt()
        feature_norms = feature_norms * active.detach().float()
        gate_feature_norms = pooled_features.detach().float().pow(2).mean(dim=2).sqrt()
        gate_feature_norms = gate_feature_norms * active.detach().float()
        return {
            "fused": fused,
            "stacked_features": stacked,
            "modality_feature_norms": feature_norms,
            "modality_gate_feature_norms": gate_feature_norms,
            "modality_gate_weights": gate_weights,
            "modality_active_mask": active,
            "modality_gate_logits": logits,
        }
