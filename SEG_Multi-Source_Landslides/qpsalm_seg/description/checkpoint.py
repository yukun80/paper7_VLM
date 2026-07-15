#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit segmentation checkpoint migration and unified segdesc checkpoint I/O."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Any

import torch

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.paths import resolve_project_path

from .model import (
    DESCRIPTION_ADAPTER_NAME,
    DESCRIPTION_SEQUENCE_PROTOCOL,
    SegmentationGroundedDescriptionModel,
)


SEGDESC_CHECKPOINT_FORMAT = "qpsalm_segdesc_v1"
SEGMENTATION_STATE_PREFIXES = ("controller.", "sane.", "qmef.", "pmrd.")
FROZEN_QWEN_PREFIX = "segmentation.controller.model."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_segmentation_checkpoint(
    checkpoint_path: str | Path,
    segmentation: MultiSourceQwenPSALMSeg,
) -> dict[str, Any]:
    """Load qpsalm_sane_qmef_pmrd_v5 after an explicit state-key whitelist audit."""
    path = resolve_project_path(checkpoint_path) or Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"只允许迁移 {CHECKPOINT_FORMAT}，当前为 {payload.get('format')!r}"
        )
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("segmentation checkpoint 缺少 model_state")
    illegal = [key for key in state if not key.startswith(SEGMENTATION_STATE_PREFIXES)]
    if illegal:
        raise RuntimeError(f"segmentation checkpoint 包含白名单外参数: {illegal[:8]}")
    step = load_checkpoint(path, segmentation)
    return {
        "source_path": str(path.resolve(strict=False)),
        "source_sha256": _sha256_file(path),
        "source_format": CHECKPOINT_FORMAT,
        "source_step": step,
        "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
    }


def _required_state(model: SegmentationGroundedDescriptionModel) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if ".lora_" in key or not key.startswith(FROZEN_QWEN_PREFIX)
    }


def _adapter_names(model: SegmentationGroundedDescriptionModel) -> tuple[str, ...]:
    configs = getattr(model.controller.model, "peft_config", None)
    if not isinstance(configs, dict):
        raise RuntimeError("统一 segdesc checkpoint 需要 PEFT model")
    names = tuple(sorted(str(value) for value in configs))
    expected = ("default", DESCRIPTION_ADAPTER_NAME)
    if names != tuple(sorted(expected)):
        raise RuntimeError(f"Adapter 名称必须严格为 {expected}，当前为 {names}")
    return names


def _description_architecture_spec(
    model: SegmentationGroundedDescriptionModel,
) -> dict[str, Any]:
    return {
        "region_encoder": model.region_encoder_name,
        "decoder_dim": int(model.segmentation.config.decoder_dim),
        "description_sequence_protocol": DESCRIPTION_SEQUENCE_PROTOCOL,
    }


def save_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    step: int,
    segmentation_migration: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    state = _required_state(model)
    payload: dict[str, Any] = {
        "format": SEGDESC_CHECKPOINT_FORMAT,
        "step": int(step),
        "model_state": state,
        "required_state_keys": sorted(state),
        "frozen_qwen_prefix": FROZEN_QWEN_PREFIX,
        "adapter_names": list(_adapter_names(model)),
        "description_sequence_protocol": DESCRIPTION_SEQUENCE_PROTOCOL,
        "description_architecture_spec": _description_architecture_spec(model),
        "segmentation_migration": dict(segmentation_migration),
        "segmentation_architecture_spec": dict(model.segmentation.config.__dict__),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["grad_scaler_state"] = scaler.state_dict()
    _atomic_save(payload, Path(path))


def load_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, dict[str, Any]]:
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 segdesc checkpoint={payload.get('format')!r}; expected={SEGDESC_CHECKPOINT_FORMAT}"
        )
    if payload.get("description_sequence_protocol") != DESCRIPTION_SEQUENCE_PROTOCOL:
        raise RuntimeError("segdesc description sequence protocol 不一致")
    if payload.get("description_architecture_spec") != _description_architecture_spec(model):
        raise RuntimeError(
            "segdesc description architecture 不一致；resume 必须保持 region encoder 完全相同"
        )
    if tuple(sorted(payload.get("adapter_names") or [])) != _adapter_names(model):
        raise RuntimeError("segdesc adapter names 不一致")
    expected_state = _required_state(model)
    observed_keys = set((payload.get("model_state") or {}).keys())
    if observed_keys != set(expected_state):
        missing = sorted(set(expected_state) - observed_keys)
        unexpected = sorted(observed_keys - set(expected_state))
        raise RuntimeError(
            f"segdesc checkpoint state 不一致: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    illegal_missing = [
        key for key in incompatible.missing_keys
        if not (key.startswith(FROZEN_QWEN_PREFIX) and ".lora_" not in key)
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"segdesc checkpoint 加载不完整: missing={illegal_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and scaler.is_enabled() and "grad_scaler_state" in payload:
        scaler.load_state_dict(payload["grad_scaler_state"])
    return int(payload.get("step", 0)), {
        "segmentation_migration": dict(payload.get("segmentation_migration") or {}),
        "metadata": dict(payload.get("metadata") or {}),
    }


def initialize_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
) -> tuple[int, dict[str, Any]]:
    """Load a new D-stage, allowing only an explicit region-encoder replacement."""
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 segdesc checkpoint={payload.get('format')!r}; expected={SEGDESC_CHECKPOINT_FORMAT}"
        )
    if payload.get("description_sequence_protocol") != DESCRIPTION_SEQUENCE_PROTOCOL:
        raise RuntimeError("segdesc description sequence protocol 不一致")
    if tuple(sorted(payload.get("adapter_names") or [])) != _adapter_names(model):
        raise RuntimeError("segdesc adapter names 不一致")
    source_spec = dict(payload.get("description_architecture_spec") or {})
    target_spec = _description_architecture_spec(model)
    non_region_source = {key: value for key, value in source_spec.items() if key != "region_encoder"}
    non_region_target = {key: value for key, value in target_spec.items() if key != "region_encoder"}
    if non_region_source != non_region_target:
        raise RuntimeError(
            "segdesc initialize architecture 不一致: "
            f"source={non_region_source} target={non_region_target}"
        )
    source_state = payload.get("model_state") or {}
    target_state = _required_state(model)
    region_changed = source_spec.get("region_encoder") != target_spec["region_encoder"]
    loaded = {}
    skipped_source = []
    for key, value in source_state.items():
        if region_changed and key.startswith("mgrr."):
            skipped_source.append(key)
            continue
        target = target_state.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            raise RuntimeError(f"segdesc initialize 非 region 参数不匹配: {key}")
        loaded[key] = value
    expected_loaded = {
        key for key in target_state
        if not (region_changed and key.startswith("mgrr."))
    }
    if set(loaded) != expected_loaded:
        missing = sorted(expected_loaded - set(loaded))
        unexpected = sorted(set(loaded) - expected_loaded)
        raise RuntimeError(
            f"segdesc initialize state 不完整: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(loaded, strict=False)
    allowed_missing = {
        key for key in target_state if region_changed and key.startswith("mgrr.")
    } | {
        key for key in model.state_dict()
        if key.startswith(FROZEN_QWEN_PREFIX) and ".lora_" not in key
    }
    illegal_missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"segdesc initialize 加载不完整: missing={illegal_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return int(payload.get("step", 0)), {
        "segmentation_migration": dict(payload.get("segmentation_migration") or {}),
        "metadata": dict(payload.get("metadata") or {}),
        "initialization": {
            "source_checkpoint": str(resolved),
            "source_region_encoder": source_spec.get("region_encoder"),
            "target_region_encoder": target_spec["region_encoder"],
            "region_encoder_reinitialized": region_changed,
            "skipped_source_region_keys": sorted(skipped_source),
            "initialized_target_region_keys": sorted(allowed_missing & set(target_state)),
        },
    }
