#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for benchmark-v2 algorithm presets."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import QPSalmConfig


_RAW_COMMON: dict[str, Any] = {
    "controller": "text_probe",
    "decoder_dim": 256,
    "num_heads": 8,
    "num_decoder_layers": 2,
    "size_buckets": [64, 128, 256, 384],
    "max_native_size": 384,
    "qwen_view_pooling": "tokens",
    "use_pretrained_sane": False,
}


PRESETS: dict[str, dict[str, Any]] = {
    "raw_sane_baseline": {
        **_RAW_COMMON,
        "num_mask_tokens": 1,
        "modality_dropout": 0.0,
        "use_qmef": False,
        "use_query_spatial_attention": False,
        "use_mask_refinement": False,
        "proposal_set_loss_weight": 0.0,
        "coarse_proposal_loss_weight": 0.0,
        "semantic_verifier_loss_weight": 0.0,
        "missing_modality_consistency_weight": 0.0,
    },
    "raw_sane_qmef": {
        **_RAW_COMMON,
        "num_mask_tokens": 1,
        "modality_dropout": 0.2,
        "use_qmef": True,
        "use_query_spatial_attention": True,
        "use_mask_refinement": False,
        "proposal_set_loss_weight": 0.0,
        "coarse_proposal_loss_weight": 0.0,
        "semantic_verifier_loss_weight": 0.1,
        "missing_modality_consistency_weight": 0.05,
    },
    "raw_sane_qmef_pmrd": {
        **_RAW_COMMON,
        "num_mask_tokens": 16,
        "modality_dropout": 0.2,
        "use_qmef": True,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.75,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.25,
        "missing_modality_consistency_weight": 0.05,
    },
    "pretrained_sane_qmef_pmrd": {
        **_RAW_COMMON,
        "num_mask_tokens": 16,
        "modality_dropout": 0.2,
        "use_pretrained_sane": True,
        "use_qmef": True,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.75,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.25,
        "missing_modality_consistency_weight": 0.1,
    },
    "qwen_psalm_full": {
        **_RAW_COMMON,
        "controller": "qwen_mask_query",
        "qwen_gradient_checkpointing": "disabled",
        "num_mask_tokens": 16,
        "modality_dropout": 0.2,
        "use_pretrained_sane": True,
        "use_qmef": True,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.75,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.25,
        "missing_modality_consistency_weight": 0.1,
        "size_buckets": [64, 128, 256],
        "max_native_size": 256,
    },
    "qwen_mask_query_frozen": {
        **_RAW_COMMON,
        "controller": "qwen_mask_query",
        "qwen_lora_trainable": False,
        "qwen_gradient_checkpointing": "disabled",
        "num_mask_tokens": 16,
        "modality_dropout": 0.2,
        "use_pretrained_sane": True,
        "use_qmef": True,
        "use_query_spatial_attention": True,
        "use_mask_refinement": True,
        "proposal_set_loss_weight": 0.75,
        "coarse_proposal_loss_weight": 0.25,
        "semantic_verifier_loss_weight": 0.25,
        "missing_modality_consistency_weight": 0.1,
        "size_buckets": [64, 128, 256],
        "max_native_size": 256,
    },
}


PRESET_CHOICES = tuple(PRESETS)


def apply_preset(config: QPSalmConfig, name: str | None) -> QPSalmConfig:
    preset = str(name or config.preset or "raw_sane_qmef_pmrd")
    if preset not in PRESETS:
        raise ValueError(f"未知 preset={preset!r}; 可选: {', '.join(PRESET_CHOICES)}")
    switching = name is not None and preset != str(config.preset)
    defaults = QPSalmConfig()
    resolved = {}
    for key, value in PRESETS[preset].items():
        current = getattr(config, key)
        resolved[key] = value if switching or current == getattr(defaults, key) else current
    return replace(config, preset=preset, **resolved)
