#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for main SANE/QMEF/PMRD experiment presets."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import QPSalmConfig


PRESETS: dict[str, dict[str, Any]] = {
    "dev_smoke": {
        "decoder_dim": 64,
        "num_heads": 4,
        "num_decoder_layers": 1,
        "size_buckets": [],
        "target_size": 64,
        "max_native_size": 64,
        "num_mask_tokens": 4,
        "modality_dropout": 0.0,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.5,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.2,
        "missing_modality_consistency_weight": 0.0,
    },
    "sane_baseline": {
        "decoder_dim": 256,
        "num_heads": 8,
        "num_decoder_layers": 2,
        "size_buckets": [64, 128, 256, 384],
        "max_native_size": 384,
        "num_mask_tokens": 1,
        "modality_dropout": 0.0,
        "use_query_spatial_attention": False,
        "use_mask_refinement": False,
        "proposal_set_loss_weight": 0.0,
        "coarse_proposal_loss_weight": 0.0,
        "semantic_verifier_loss_weight": 0.0,
        "missing_modality_consistency_weight": 0.0,
    },
    "sane_qmef": {
        "decoder_dim": 256,
        "num_heads": 8,
        "num_decoder_layers": 2,
        "size_buckets": [64, 128, 256, 384],
        "max_native_size": 384,
        "num_mask_tokens": 1,
        "modality_dropout": 0.2,
        "use_query_spatial_attention": True,
        "use_mask_refinement": False,
        "proposal_set_loss_weight": 0.0,
        "coarse_proposal_loss_weight": 0.0,
        "semantic_verifier_loss_weight": 0.1,
        "missing_modality_consistency_weight": 0.05,
    },
    "sane_qmef_pmrd": {
        "decoder_dim": 256,
        "num_heads": 8,
        "num_decoder_layers": 2,
        "size_buckets": [64, 128, 256, 384],
        "max_native_size": 384,
        "num_mask_tokens": 16,
        "modality_dropout": 0.2,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.75,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.25,
        "missing_modality_consistency_weight": 0.05,
    },
    "full_multiview": {
        "decoder_dim": 256,
        "num_heads": 8,
        "num_decoder_layers": 2,
        "size_buckets": [64, 128, 256, 384],
        "max_native_size": 384,
        "num_mask_tokens": 16,
        "modality_dropout": 0.2,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.75,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.25,
        "missing_modality_consistency_weight": 0.1,
    },
}


PRESET_CHOICES = tuple(PRESETS)


def apply_preset(config: QPSalmConfig, name: str | None) -> QPSalmConfig:
    preset = str(name or config.preset or "sane_qmef_pmrd")
    if preset not in PRESETS:
        raise ValueError(f"未知 preset={preset!r}; 可选: {', '.join(PRESET_CHOICES)}")
    return replace(config, preset=preset, **PRESETS[preset])
