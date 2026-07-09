#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口：模型实现已拆分到 qpsalm_seg.models。"""

from __future__ import annotations

from .models import (
    ConvBlock,
    MultiSourceAdapterBank,
    MultiSourceQwenPSALMSeg,
    MultiScaleFeatureFusion,
    PSALMConditionAwareMaskDecoder,
)

LightweightMaskDecoder = PSALMConditionAwareMaskDecoder

__all__ = [
    "ConvBlock",
    "MultiSourceAdapterBank",
    "MultiScaleFeatureFusion",
    "PSALMConditionAwareMaskDecoder",
    "LightweightMaskDecoder",
    "MultiSourceQwenPSALMSeg",
]
