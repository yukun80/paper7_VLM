#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimizer parameter groups and staged QLoRA activation."""

from __future__ import annotations

from typing import Any

import torch

from qpsalm_seg.config import QPSalmConfig


CONTROLLER_EMBEDDING_NAMES = (
    "controller.text_type",
    "controller.view_description_type",
    "controller.view_attention_query",
    "controller.evidence_anchors",
    "controller.anchor_availability",
    "controller.mask_embeddings",
    "controller.visual_family_embedding",
)
CONTROLLER_PROJECTION_NAMES = (
    "controller.view_to_hidden",
    "controller.output_projection",
)


def _no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return parameter.ndim <= 1 or name.endswith(".bias") or "norm" in lowered


def parameter_role(name: str) -> str:
    if name.startswith("controller.model.") and "lora_" in name:
        return "qwen_lora"
    if name.startswith(CONTROLLER_EMBEDDING_NAMES):
        return "controller_embedding"
    if name.startswith(CONTROLLER_PROJECTION_NAMES):
        return "controller_projection"
    return "dense"


def build_optimizer(
    model: torch.nn.Module,
    config: QPSalmConfig,
) -> tuple[torch.optim.AdamW, list[torch.nn.Parameter]]:
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    all_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        all_parameters.append(parameter)
        role = parameter_role(name)
        grouped.setdefault((role, _no_decay(name, parameter)), []).append(parameter)

    role_scale = {
        "qwen_lora": float(config.qwen_lora_lr_scale),
        "controller_embedding": float(config.controller_lr_scale),
        "controller_projection": float(config.controller_lr_scale),
        "dense": 1.0,
    }
    parameter_groups: list[dict[str, Any]] = []
    for (role, no_decay), parameters in sorted(grouped.items()):
        parameter_groups.append({
            "params": parameters,
            "lr": float(config.lr) * role_scale[role],
            "weight_decay": 0.0 if no_decay or role in {"qwen_lora", "controller_embedding"} else float(config.weight_decay),
            "group_role": role,
            "lr_scale": role_scale[role],
        })
    optimizer = torch.optim.AdamW(parameter_groups, lr=float(config.lr))
    return optimizer, all_parameters


def qwen_training_stage(config: QPSalmConfig, step: int) -> str:
    if config.controller != "qwen_mask_query":
        return "dense"
    if not config.qwen_lora_trainable:
        return "qwen_frozen"
    if int(step) < max(0, int(config.qwen_lora_start_step)):
        return "decoder_warmup"
    return "qlora_active"


def apply_optimizer_schedule(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: QPSalmConfig,
    *,
    step: int,
    lr_multiplier: float,
) -> str:
    stage = qwen_training_stage(config, step)
    lora_enabled = stage == "qlora_active"
    for name, parameter in model.named_parameters():
        if name.startswith("controller.model.") and "lora_" in name:
            parameter.requires_grad_(lora_enabled)
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        role = str(group.get("group_role", "dense"))
        group["lr"] = (
            float(config.lr) * float(lr_multiplier) * scale
            if role != "qwen_lora" or lora_enabled
            else 0.0
        )
    return stage


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "role": str(group.get("group_role", "dense")),
            "lr_scale": float(group.get("lr_scale", 1.0)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "num_tensors": len(group["params"]),
            "num_parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]


def qwen_lora_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Return the LoRA parameters that belong to the online Qwen controller."""
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("controller.model.") and "lora_" in name
    }


def snapshot_qwen_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Capture a small, device-local snapshot for a one-step update check."""
    return {
        name: parameter.detach().clone()
        for name, parameter in qwen_lora_parameters(model).items()
        if parameter.requires_grad
    }


def qwen_lora_gradient_summary(model: torch.nn.Module) -> dict[str, float | int | bool]:
    parameters = qwen_lora_parameters(model)
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters.values()
        if parameter.requires_grad and parameter.grad is not None
    ]
    norms = [gradient.norm() for gradient in gradients]
    return {
        "num_parameters": len(parameters),
        "num_with_grad": len(gradients),
        "num_nonzero": sum(bool(torch.count_nonzero(gradient)) for gradient in gradients),
        "norm_sum": float(torch.stack(norms).sum().detach().cpu()) if norms else 0.0,
        "all_finite": all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
    }


def qwen_lora_update_summary(
    model: torch.nn.Module,
    before: dict[str, torch.Tensor],
) -> dict[str, float | int | bool]:
    current = qwen_lora_parameters(model)
    deltas = [
        (current[name].detach() - value).float()
        for name, value in before.items()
        if name in current
    ]
    norms = [delta.norm() for delta in deltas]
    return {
        "num_parameters": len(before),
        "num_changed": sum(bool(torch.count_nonzero(delta)) for delta in deltas),
        "norm_sum": float(torch.stack(norms).sum().detach().cpu()) if norms else 0.0,
        "all_finite": all(bool(torch.isfinite(delta).all()) for delta in deltas),
    }
