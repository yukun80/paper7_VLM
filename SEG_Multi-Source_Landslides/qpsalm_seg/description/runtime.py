#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model construction and named optimizer groups for description stages."""

from __future__ import annotations

import math
from typing import Any

import torch

from qpsalm_seg.config import load_config
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.presets import apply_preset

from .backbone import DescriptionCacheBackboneEncoder
from .checkpoint import migrate_segmentation_checkpoint
from .config import SegDescConfig
from .model import DESCRIPTION_ADAPTER_NAME, SegmentationGroundedDescriptionModel
from .vision_cache import DescriptionVisionFeatureBank


def build_segdesc_model(
    config: SegDescConfig,
    device: torch.device,
) -> tuple[SegmentationGroundedDescriptionModel, dict[str, Any]]:
    segmentation_config = apply_preset(
        load_config(
            config.segmentation_config,
            overrides={"vision_feature_cache": config.segmentation_vision_cache},
        ),
        config.segmentation_preset,
    )
    segmentation = MultiSourceQwenPSALMSeg(segmentation_config, device)
    migration = migrate_segmentation_checkpoint(config.segmentation_checkpoint, segmentation)
    bank = DescriptionVisionFeatureBank(config.description_vision_cache)
    backbone = DescriptionCacheBackboneEncoder(bank, int(segmentation_config.decoder_dim)).to(device)
    model = SegmentationGroundedDescriptionModel(
        segmentation,
        description_backbone=backbone,
        region_encoder=config.region_encoder,
    )
    for module in (
        model.mgrr,
        model.region_to_hidden,
        model.description_view_to_hidden,
        model.alignment_text_projection,
    ):
        module.to(device)
    model.region_type.data = model.region_type.data.to(device)
    model.instruction_type.data = model.instruction_type.data.to(device)
    model.visual_type.data = model.visual_type.data.to(device)
    model.alignment_temperature.data = model.alignment_temperature.data.to(device)
    return model, migration


def description_parameter_groups(
    model: SegmentationGroundedDescriptionModel,
    config: SegDescConfig,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: dict[str, list[torch.nn.Parameter]] = {
        "desc_adapter": [], "region_modules_decay": [], "region_modules_no_decay": [],
    }
    region_prefixes = (
        "description_backbone.", "mgrr.", "region_to_hidden.",
        "description_view_to_hidden.", "alignment_text_projection.",
    )
    direct_names = {"region_type", "instruction_type", "visual_type", "alignment_temperature"}
    for name, parameter in model.named_parameters():
        if f".{DESCRIPTION_ADAPTER_NAME}." in name and "lora_" in name:
            selected["desc_adapter"].append(parameter)
        elif name in direct_names or name.startswith(region_prefixes):
            no_decay = (
                name.endswith(".bias") or "norm" in name.casefold() or name in direct_names
            )
            selected["region_modules_no_decay" if no_decay else "region_modules_decay"].append(parameter)
    if not selected["desc_adapter"]:
        raise RuntimeError("optimizer 未找到 desc_adapter LoRA 参数")
    if not selected["region_modules_decay"] and not selected["region_modules_no_decay"]:
        raise RuntimeError("optimizer 未找到 MGRR/description projection 参数")
    for values in selected.values():
        for parameter in values:
            parameter.requires_grad_(True)
    return [
        {
            "name": "desc_adapter",
            "params": selected["desc_adapter"],
            "lr": config.learning_rate * config.desc_adapter_lr_scale,
            "weight_decay": 0.0,
        },
        {
            "name": "region_modules_decay",
            "params": selected["region_modules_decay"],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "region_modules_no_decay",
            "params": selected["region_modules_no_decay"],
            "lr": config.learning_rate,
            "weight_decay": 0.0,
        },
    ]


def build_description_optimizer(
    model: SegmentationGroundedDescriptionModel,
    config: SegDescConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    groups = description_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1.0e-8)

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
