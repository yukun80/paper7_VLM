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

from ..modeling.backbone import DescriptionCacheBackboneEncoder
from .checkpoint import migrate_segmentation_checkpoint
from ..protocols.config import SegDescConfig
from ..modeling.model import DESCRIPTION_ADAPTER_NAME, SegmentationGroundedDescriptionModel
from ..data.vision_cache import DescriptionVisionFeatureBank
from ..protocols.stages import get_stage_spec


def build_segdesc_model(
    config: SegDescConfig,
    device: torch.device,
) -> tuple[SegmentationGroundedDescriptionModel, dict[str, Any]]:
    segmentation_config = apply_preset(
        load_config(
            config.model.segmentation_config,
            overrides={"vision_feature_cache": config.model.segmentation_vision_cache},
        ),
        config.model.segmentation_preset,
    )
    segmentation = MultiSourceQwenPSALMSeg(segmentation_config, device)
    migration = migrate_segmentation_checkpoint(config.model.segmentation_checkpoint, segmentation)
    bank = DescriptionVisionFeatureBank(config.model.description_vision_cache)
    backbone = DescriptionCacheBackboneEncoder(bank, int(segmentation_config.decoder_dim)).to(device)
    model = SegmentationGroundedDescriptionModel(
        segmentation,
        description_backbone=backbone,
        region_encoder=config.model.region_encoder,
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
        "desc_adapter": [], "description_modules_decay": [], "description_modules_no_decay": [],
    }
    stage_spec = get_stage_spec(config.training.stage)
    trainable_prefixes = stage_spec.trainable_prefixes
    direct_names = set(stage_spec.trainable_direct_parameters)
    train_desc_adapter = stage_spec.trains_desc_adapter
    for name, parameter in model.named_parameters():
        if (
            train_desc_adapter
            and f".{DESCRIPTION_ADAPTER_NAME}." in name
            and "lora_" in name
        ):
            selected["desc_adapter"].append(parameter)
        elif name in direct_names or name.startswith(trainable_prefixes):
            no_decay = (
                name.endswith(".bias") or "norm" in name.casefold() or name in direct_names
            )
            selected[
                "description_modules_no_decay" if no_decay else "description_modules_decay"
            ].append(parameter)
    if train_desc_adapter and not selected["desc_adapter"]:
        raise RuntimeError("optimizer 未找到 desc_adapter LoRA 参数")
    if not selected["description_modules_decay"] and not selected["description_modules_no_decay"]:
        raise RuntimeError(f"optimizer 未找到 stage={config.training.stage} 对应的 description 参数")
    for values in selected.values():
        for parameter in values:
            parameter.requires_grad_(True)
    groups = [
        {
            "name": "description_modules_decay",
            "params": selected["description_modules_decay"],
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        },
        {
            "name": "description_modules_no_decay",
            "params": selected["description_modules_no_decay"],
            "lr": config.training.learning_rate,
            "weight_decay": 0.0,
        },
    ]
    if selected["desc_adapter"]:
        groups.insert(0, {
            "name": "desc_adapter",
            "params": selected["desc_adapter"],
            "lr": config.training.learning_rate * config.training.desc_adapter_lr_scale,
            "weight_decay": 0.0,
        })
    return groups


def description_trainable_parameter_manifest(
    model: SegmentationGroundedDescriptionModel,
    parameter_groups: list[dict[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    observed: set[int] = set()
    for group in parameter_groups:
        parameters = list(group["params"])
        names = []
        numel = 0
        for parameter in parameters:
            parameter_id = id(parameter)
            name = names_by_id.get(parameter_id)
            if name is None:
                raise RuntimeError("optimizer parameter 不属于 description model")
            if parameter_id in observed:
                raise RuntimeError(f"optimizer parameter 重复分组: {name}")
            observed.add(parameter_id)
            names.append(name)
            numel += int(parameter.numel())
        groups.append({
            "name": str(group.get("name")),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "num_parameters": len(parameters),
            "numel": numel,
            "parameter_names": sorted(names),
        })
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if observed != expected:
        raise RuntimeError("optimizer parameter groups 与 requires_grad 集合不一致")
    return {
        "protocol": "qpsalm_description_trainable_parameters_v1",
        "stage": str(stage),
        "groups": groups,
        "total_parameters": sum(group["num_parameters"] for group in groups),
        "total_numel": sum(group["numel"] for group in groups),
    }


def build_description_optimizer(
    model: SegmentationGroundedDescriptionModel,
    config: SegDescConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    groups = description_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1.0e-8)

    def schedule(step: int) -> float:
        if step < config.training.warmup_steps:
            return float(step + 1) / max(config.training.warmup_steps, 1)
        progress = (
            step - config.training.warmup_steps
        ) / max(
            config.training.max_steps - config.training.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def description_optimizer_audit(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    """Describe the construction-time optimizer contract before any step."""

    return {
        "type": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
        "groups": [str(group.get("name")) for group in optimizer.param_groups],
        "state_entries": len(optimizer.state),
    }
