#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact v2 checkpoint I/O that excludes frozen Qwen base weights."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import torch

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.controllers import CONTROLLER_SEQUENCE_PROTOCOL
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from .optimizer import qwen_training_stage


CHECKPOINT_FORMAT = "qpsalm_sane_qmef_pmrd_v5"
ARCHITECTURE_FIELDS = (
    "preset", "controller", "decoder_dim", "num_heads", "num_decoder_layers",
    "num_mask_tokens", "use_pretrained_sane", "use_qmef",
    "use_query_spatial_attention", "use_mask_refinement", "deformable_points",
    "qwen_4bit", "qwen_lora_rank", "qwen_lora_alpha",
    "qwen_lora_dropout", "qwen_lora_last_n_layers", "qwen_lora_trainable",
    "qwen_view_tokens_per_view",
    "qwen_max_text_tokens", "qwen_view_pooling", "qwen_attn_implementation",
)
RUNTIME_FIELDS = (
    "batch_size", "grad_accum_steps", "query_chunk_size",
    "qwen_gradient_checkpointing", "amp_dtype", "qwen_lora_start_step",
    "qwen_lora_lr_scale", "controller_lr_scale",
)
TRAINING_SCHEDULE_FIELDS = (
    "qwen_lora_start_step", "qwen_lora_lr_scale", "controller_lr_scale",
)


def architecture_spec(config: QPSalmConfig) -> dict[str, Any]:
    return {
        **{name: getattr(config, name) for name in ARCHITECTURE_FIELDS},
        "controller_sequence_protocol": CONTROLLER_SEQUENCE_PROTOCOL,
    }


def evidence_protocol(model: MultiSourceQwenPSALMSeg) -> dict[str, Any] | None:
    bank = getattr(model, "vision_bank", None)
    if bank is None:
        return None
    fields = (
        "format", "renderer_version", "model_revision", "processor_revision",
        "prompt_version", "pooling_method", "layers", "spatial_channels",
        "token_dim", "spatial_sizes", "render_size", "view_tokens_per_view", "subset_policy",
        "input_protocol",
    )
    return {name: bank.manifest.get(name) for name in fields}


def validate_checkpoint_training_schedule(
    checkpoint: dict[str, Any],
    config: QPSalmConfig,
) -> None:
    if config.controller != "qwen_mask_query":
        return
    observed_runtime = checkpoint.get("runtime_spec") or {}
    mismatched_schedule = {
        name: {"checkpoint": observed_runtime.get(name), "current": getattr(config, name)}
        for name in TRAINING_SCHEDULE_FIELDS
        if observed_runtime.get(name) != getattr(config, name)
    }
    if mismatched_schedule:
        raise RuntimeError(
            f"checkpoint QLoRA training schedule 不一致: {mismatched_schedule}"
        )
    checkpoint_step = int(checkpoint.get("step", 0))
    expected_stage = qwen_training_stage(config, checkpoint_step)
    observed_stage = checkpoint.get("resume_training_stage")
    if observed_stage != expected_stage:
        raise RuntimeError(
            "checkpoint QLoRA training stage 不一致: "
            f"checkpoint={observed_stage!r} expected={expected_stage!r} step={checkpoint_step}"
        )


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: Path,
    model: MultiSourceQwenPSALMSeg,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: QPSalmConfig,
    update_last: bool = True,
    include_optimizer: bool = True,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    excluded_prefixes = ("controller.model.",)
    state = {
        key: value
        for key, value in model.state_dict().items()
        if key in trainable or ".lora_" in key or not key.startswith(excluded_prefixes)
    }
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "model_state": state,
        "excluded_frozen_prefixes": list(excluded_prefixes),
        "trainable_parameter_names": sorted(trainable),
        "architecture_spec": architecture_spec(config),
        "runtime_spec": {name: getattr(config, name) for name in RUNTIME_FIELDS},
        # ``step`` is the next optimizer step after restoring this payload.
        "resume_training_stage": qwen_training_stage(config, step),
        "evidence_protocol": evidence_protocol(model),
        "config": dict(config.__dict__),
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
        if scaler is not None and scaler.is_enabled():
            payload["grad_scaler_state"] = scaler.state_dict()
    _atomic_save(payload, path)
    last = path.parent / "checkpoint_last.pt"
    if update_last and last != path:
        _atomic_save(payload, last)


def _step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("checkpoint_step_"))
    except ValueError:
        return -1


def prune_step_checkpoints(out_dir: Path, keep_recent: int) -> list[str]:
    if keep_recent < 0:
        return []
    paths = sorted((path for path in out_dir.glob("checkpoint_step_*.pt") if _step(path) >= 0), key=_step)
    removed = []
    for path in paths[:max(0, len(paths) - int(keep_recent))]:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def load_checkpoint(
    path: str | Path,
    model: MultiSourceQwenPSALMSeg,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 checkpoint={checkpoint.get('format')!r}; expected={CHECKPOINT_FORMAT}. v2 不兼容旧权重。"
        )
    expected_architecture = architecture_spec(model.config)
    observed_architecture = checkpoint.get("architecture_spec")
    if observed_architecture != expected_architecture:
        raise RuntimeError(
            "checkpoint architecture spec 不一致: "
            f"checkpoint={observed_architecture} current={expected_architecture}"
        )
    expected_evidence = evidence_protocol(model)
    observed_evidence = checkpoint.get("evidence_protocol")
    if observed_evidence != expected_evidence:
        raise RuntimeError(
            "checkpoint evidence protocol 不一致: "
            f"checkpoint={observed_evidence} current={expected_evidence}"
        )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    excluded = tuple(str(value) for value in checkpoint.get("excluded_frozen_prefixes") or [])
    illegal_missing = [key for key in incompatible.missing_keys if not key.startswith(excluded)]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint 架构不一致: missing={illegal_missing[:8]} unexpected={incompatible.unexpected_keys[:8]}"
        )
    if optimizer is not None and "optimizer_state" in checkpoint:
        validate_checkpoint_training_schedule(checkpoint, model.config)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and scaler.is_enabled() and "grad_scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["grad_scaler_state"])
    return int(checkpoint.get("step", 0))
