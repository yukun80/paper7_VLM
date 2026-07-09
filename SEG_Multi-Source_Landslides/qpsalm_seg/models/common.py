#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型通用积木。"""

from __future__ import annotations

import torch
from torch import nn


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    """为 GroupNorm 选择能整除通道数的 group 数，小 batch 训练比 BatchNorm 稳定。"""
    groups = min(int(preferred), int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return max(1, groups)


class ConvBlock(nn.Module):
    """小型 Conv-GN-GELU block，用于轻量遥感特征编码。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _valid_group_count(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """两层 MLP。"""

    def __init__(self, dim: int, hidden_dim: int | None = None, out_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        out = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
