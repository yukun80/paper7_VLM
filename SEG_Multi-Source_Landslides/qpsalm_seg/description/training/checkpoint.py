#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit segmentation checkpoint migration and unified segdesc checkpoint I/O."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import struct
import time
from typing import Any, Callable

import numpy as np
import torch

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
)
from ..modeling.model import (
    DESCRIPTION_ADAPTER_NAME,
    SegmentationGroundedDescriptionModel,
)
from ..modeling.mgrr import MGRR_PROTOCOL, MGRR_ROI_GRID_SIZES
from ..protocols.versions import (
    DESCRIPTION_SEQUENCE_PROTOCOL,
    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
    STRICT_RELOAD_PROBE_PROTOCOL,
    STRUCTURED_GENERATION_PROTOCOL,
)
from ..protocols.io import (
    sha256_file,
    tensor_raw_bytes as _tensor_raw_bytes,
    tensor_sha256 as _tensor_sha256,
)
from .checkpoint_contracts import (
    DESCRIPTION_CHECKPOINT_ROLES,
    DESCRIPTION_PROTOCOL_ASSETS,
    DESCRIPTION_STAGE_CHECKPOINT_ROLE,
    DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
    DESCRIPTION_STAGE_PREDECESSOR,
    D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
    FROZEN_QWEN_PREFIX,
    REGION_ARCHITECTURE_FIELDS,
    SEGDESC_CHECKPOINT_FORMAT,
    SEGDESC_CHECKPOINT_PROVENANCE_PROTOCOL,
    SEGMENTATION_ARCHITECTURE_FIELDS,
    SEGMENTATION_MIGRATION_LINEAGE_PROTOCOL,
    SEGMENTATION_STATE_PREFIXES,
    TRAINING_RNG_STATE_PROTOCOL,
    build_description_stage_lineage,
    checkpoint_metadata_report,
    checkpoint_state,
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
    read_segdesc_checkpoint_step,
    validate_description_stage_lineage,
    validate_resume_run_config,
    validate_segmentation_migration_lineage,
)


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
        "source_sha256": sha256_file(path),
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



def _restore_resume_state(
    payload: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
) -> None:
    """Resume is strict; cross-stage weight initialization uses another API."""
    requested = (
        ("optimizer_state", optimizer, optimizer is not None),
        ("scheduler_state", scheduler, scheduler is not None),
        (
            "grad_scaler_state",
            scaler,
            scaler is not None and scaler.is_enabled(),
        ),
    )
    for key, target, required in requested:
        if not required:
            continue
        if key not in payload:
            raise RuntimeError(f"resume checkpoint 缺少 {key}")
        target.load_state_dict(payload[key])
    if any(required for _key, _target, required in requested):
        restore_training_rng_state(payload.get("training_rng_state"))


def capture_training_rng_state() -> dict[str, Any]:
    """Snapshot every process-level RNG used by resumable SegDesc training."""
    cuda_available = bool(torch.cuda.is_available())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else []
    return {
        "protocol": TRAINING_RNG_STATE_PROTOCOL,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "cuda_available": cuda_available,
        "cuda_device_count": len(cuda_states),
    }


def restore_training_rng_state(value: Any) -> None:
    """Strictly restore a current RNG snapshot; old resume checkpoints are rejected."""
    if not isinstance(value, dict) or value.get("protocol") != TRAINING_RNG_STATE_PROTOCOL:
        raise RuntimeError(
            "resume checkpoint 缺少当前 training RNG state；旧实验 checkpoint "
            "不能用于同一 run 精确续训"
        )
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if any(name not in value for name in required):
        raise RuntimeError("resume checkpoint training RNG state 字段不完整")
    saved_cuda = bool(value.get("cuda_available"))
    current_cuda = bool(torch.cuda.is_available())
    cuda_states = list(value.get("torch_cuda") or [])
    if saved_cuda != current_cuda:
        raise RuntimeError(
            "resume checkpoint CUDA RNG 环境不一致: "
            f"checkpoint={saved_cuda} current={current_cuda}"
        )
    if saved_cuda and (
        int(value.get("cuda_device_count", -1)) != torch.cuda.device_count()
        or len(cuda_states) != torch.cuda.device_count()
    ):
        raise RuntimeError("resume checkpoint CUDA device/RNG state 数量不一致")
    try:
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch_cpu"])
        if saved_cuda:
            torch.cuda.set_rng_state_all(cuda_states)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("resume checkpoint training RNG state 无法恢复") from exc


def _description_architecture_spec(
    model: SegmentationGroundedDescriptionModel,
) -> dict[str, Any]:
    bank = getattr(getattr(model, "description_backbone", None), "bank", None)
    manifest = dict(getattr(bank, "manifest", {}) or {})
    artifact_binding = (
        bank.artifact_binding()
        if bank is not None and callable(getattr(bank, "artifact_binding", None))
        else None
    )
    return {
        "region_encoder": model.region_encoder_name,
        "mgrr_protocol": (
            MGRR_PROTOCOL if isinstance(model.mgrr, torch.nn.Module)
            and hasattr(model.mgrr, "roi_queries") else None
        ),
        "mgrr_max_components": getattr(model.mgrr, "max_components", None),
        "mgrr_component_coverage": getattr(model.mgrr, "component_coverage", None),
        "mgrr_roi_grid_sizes": (
            [list(value) for value in MGRR_ROI_GRID_SIZES]
            if hasattr(model.mgrr, "roi_queries") else None
        ),
        "mgrr_roi_query_count": (
            int(model.mgrr.roi_queries.shape[0])
            if hasattr(model.mgrr, "roi_queries") else None
        ),
        "decoder_dim": int(model.segmentation.config.decoder_dim),
        "description_sequence_protocol": DESCRIPTION_SEQUENCE_PROTOCOL,
        "structured_generation_protocol": STRUCTURED_GENERATION_PROTOCOL,
        "description_cache_protocol": manifest.get("format"),
        "description_cache_renderer_version": manifest.get("renderer_version"),
        "description_cache_model_revision": manifest.get("model_revision"),
        "description_cache_processor_revision": manifest.get("processor_revision"),
        "description_cache_spatial_channels": manifest.get("spatial_channels"),
        "description_cache_layers": manifest.get("layers"),
        "description_cache_artifact_binding": artifact_binding,
    }


def _segmentation_architecture_spec(
    model: SegmentationGroundedDescriptionModel,
) -> dict[str, Any]:
    config = model.segmentation.config
    return {
        name: getattr(config, name, None)
        for name in SEGMENTATION_ARCHITECTURE_FIELDS
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
        "description_protocol_assets": description_protocol_assets_spec(),
        "description_architecture_spec": _description_architecture_spec(model),
        "segmentation_migration": dict(segmentation_migration),
        "segmentation_architecture_spec": _segmentation_architecture_spec(model),
        "training_rng_state": capture_training_rng_state(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["grad_scaler_state"] = scaler.state_dict()
    checkpoint_metadata_report(payload)
    _atomic_save(payload, Path(path))


def load_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    expected_stage: str | None = None,
) -> tuple[int, dict[str, Any]]:
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 segdesc checkpoint={payload.get('format')!r}; expected={SEGDESC_CHECKPOINT_FORMAT}"
        )
    if payload.get("description_sequence_protocol") != DESCRIPTION_SEQUENCE_PROTOCOL:
        raise RuntimeError("segdesc description sequence protocol 不一致")
    if payload.get("description_protocol_assets") != description_protocol_assets_spec():
        raise RuntimeError("segdesc ontology/schema protocol assets 不一致")
    checkpoint_metadata = checkpoint_metadata_report(payload)
    source_stage = str((payload.get("metadata") or {}).get("stage") or "")
    if expected_stage is not None and source_stage != expected_stage:
        raise RuntimeError(
            f"resume checkpoint stage 不一致: expected={expected_stage!r} "
            f"observed={source_stage!r}"
        )
    if payload.get("description_architecture_spec") != _description_architecture_spec(model):
        raise RuntimeError(
            "segdesc description architecture 不一致；resume 必须保持 region encoder 完全相同"
        )
    if payload.get("segmentation_architecture_spec") != _segmentation_architecture_spec(model):
        raise RuntimeError(
            "segdesc segmentation architecture 不一致；Qwen/SANE/QMEF/PMRD 参数语义不可迁移"
        )
    if tuple(sorted(payload.get("adapter_names") or [])) != _adapter_names(model):
        raise RuntimeError("segdesc adapter names 不一致")
    expected_state = _required_state(model)
    state = checkpoint_state(payload)
    observed_keys = set(state)
    if observed_keys != set(expected_state):
        missing = sorted(set(expected_state) - observed_keys)
        unexpected = sorted(observed_keys - set(expected_state))
        raise RuntimeError(
            f"segdesc checkpoint state 不一致: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    illegal_missing = [
        key for key in incompatible.missing_keys
        if not (key.startswith(FROZEN_QWEN_PREFIX) and ".lora_" not in key)
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"segdesc checkpoint 加载不完整: missing={illegal_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    _restore_resume_state(
        payload, optimizer=optimizer, scheduler=scheduler, scaler=scaler
    )
    return int(payload.get("step", 0)), checkpoint_metadata


def _state_sha256(value: Any) -> str:
    """Hash optimizer/scheduler/RNG state without pickle or device metadata."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"none;")
        elif isinstance(item, bool):
            digest.update(b"bool:1;" if item else b"bool:0;")
        elif isinstance(item, int):
            digest.update(f"int:{item};".encode("ascii"))
        elif isinstance(item, float):
            digest.update(b"float:")
            digest.update(struct.pack(">d", item))
            digest.update(b";")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(f"str:{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
            digest.update(b";")
        elif isinstance(item, bytes):
            digest.update(f"bytes:{len(item)}:".encode("ascii"))
            digest.update(item)
            digest.update(b";")
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor:")
            update(str(tensor.dtype))
            update(list(tensor.shape))
            raw = _tensor_raw_bytes(tensor)
            digest.update(f"raw:{len(raw)}:".encode("ascii"))
            digest.update(raw)
            digest.update(b";")
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray:")
            update(str(array.dtype))
            update(list(array.shape))
            raw = array.tobytes()
            digest.update(f"raw:{len(raw)}:".encode("ascii"))
            digest.update(raw)
            digest.update(b";")
        elif isinstance(item, np.generic):
            digest.update(b"numpy-scalar:")
            update(str(item.dtype))
            update(item.item())
        elif isinstance(item, dict):
            digest.update(f"dict:{len(item)}:".encode("ascii"))
            ordered = sorted(
                item.items(),
                key=lambda pair: (
                    type(pair[0]).__module__,
                    type(pair[0]).__qualname__,
                    repr(pair[0]),
                ),
            )
            for key, nested in ordered:
                update(key)
                update(nested)
            digest.update(b";")
        elif isinstance(item, (list, tuple)):
            digest.update(
                f"{type(item).__name__}:{len(item)}:".encode("ascii")
            )
            for nested in item:
                update(nested)
            digest.update(b";")
        else:
            raise TypeError(
                "strict reload state 包含不可哈希类型: "
                f"{type(item).__module__}.{type(item).__qualname__}"
            )

    update(value)
    return digest.hexdigest()


def _corrupt_optimizer_state(optimizer: torch.optim.Optimizer) -> str:
    """Change one live optimizer field so reload restoration is observable."""

    for parameter_index, parameter in enumerate(
        value for group in optimizer.param_groups for value in group["params"]
    ):
        state = optimizer.state.get(parameter) or {}
        for name in sorted(state, key=str):
            value = state[name]
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                with torch.no_grad():
                    value.reshape(-1)[0].add_(1)
                return f"state[{parameter_index}].{name}"
    if not optimizer.param_groups:
        raise RuntimeError("reload probe optimizer 没有 parameter group")
    group = optimizer.param_groups[0]
    learning_rate = float(group.get("lr", 0.0))
    group["lr"] = learning_rate + max(abs(learning_rate), 1.0e-6)
    return "param_groups[0].lr"


def _corrupt_scheduler_state(scheduler: Any) -> str:
    """Change one scheduler cursor using its public state_dict contract."""

    state = dict(scheduler.state_dict())
    for name in ("last_epoch", "_step_count"):
        value = state.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            state[name] = value + 1
            scheduler.load_state_dict(state)
            return name
    raise RuntimeError("reload probe scheduler 缺少可扰动的 step cursor")


def _corrupt_scaler_state(scaler: torch.amp.GradScaler) -> str:
    """Change enabled GradScaler state through its public state_dict API."""

    state = dict(scaler.state_dict())
    scale = state.get("scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise RuntimeError("reload probe GradScaler 缺少 scale")
    state["scale"] = float(scale) * 2.0
    scaler.load_state_dict(state)
    return "scale"


def _advance_all_training_rngs() -> None:
    """Move every RNG represented by capture_training_rng_state()."""

    random.random()
    np.random.random()
    torch.rand((), device="cpu")
    for device_index in range(torch.cuda.device_count()):
        torch.rand((), device=torch.device("cuda", device_index))


def verify_segdesc_checkpoint_reload(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    expected_stage: str,
) -> tuple[int, dict[str, Any]]:
    """Prove strict restoration of model, optimizer, scheduler, scaler and RNG.

    Merely loading a just-saved checkpoint into the unchanged live model cannot
    demonstrate restoration.  This probe first verifies a description-adapter
    sentinel against the checkpoint, mutates it, then exercises the normal strict
    model and resume-state loader and verifies exact byte restoration.
    """
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "reload probe 仅支持当前 segdesc checkpoint format: "
            f"{payload.get('format')!r}"
        )
    checkpoint_metadata_report(payload)
    state = checkpoint_state(payload)
    named_parameters = dict(model.named_parameters())
    candidates = [
        name
        for name in sorted(state)
        if (
            f".{DESCRIPTION_ADAPTER_NAME}." in name
            and "lora_" in name
            and name in named_parameters
            and named_parameters[name].is_floating_point()
            and named_parameters[name].numel() > 0
        )
    ]
    if not candidates:
        raise RuntimeError("reload probe 未找到可变的 desc_adapter LoRA 哨兵参数")
    for key in ("optimizer_state", "scheduler_state", "training_rng_state"):
        if key not in payload:
            raise RuntimeError(f"reload probe checkpoint 缺少 {key}")
    scaler_requested = bool(scaler is not None and scaler.is_enabled())
    if scaler_requested and "grad_scaler_state" not in payload:
        raise RuntimeError("reload probe checkpoint 缺少 grad_scaler_state")

    expected_state_hashes = {
        "optimizer": _state_sha256(payload["optimizer_state"]),
        "scheduler": _state_sha256(payload["scheduler_state"]),
        "rng": _state_sha256(payload["training_rng_state"]),
        "scaler": (
            _state_sha256(payload["grad_scaler_state"])
            if scaler_requested else None
        ),
    }
    live_state_hashes_before = {
        "optimizer": _state_sha256(optimizer.state_dict()),
        "scheduler": _state_sha256(scheduler.state_dict()),
        "rng": _state_sha256(capture_training_rng_state()),
        "scaler": (
            _state_sha256(scaler.state_dict()) if scaler_requested else None
        ),
    }
    if live_state_hashes_before != expected_state_hashes:
        mismatched = sorted(
            name for name in expected_state_hashes
            if live_state_hashes_before[name] != expected_state_hashes[name]
        )
        raise RuntimeError(
            "reload probe 开始前 live training state 已与刚保存 checkpoint "
            f"不一致: {mismatched}"
        )

    sentinel_name = candidates[0]
    sentinel = named_parameters[sentinel_name]
    checkpoint_sha256 = sha256_file(resolved)
    expected_sha256 = _tensor_sha256(state[sentinel_name])
    before_sha256 = _tensor_sha256(sentinel)
    if before_sha256 != expected_sha256:
        raise RuntimeError(
            "reload probe 开始前 live model 已与刚保存 checkpoint 不一致: "
            f"parameter={sentinel_name}"
        )
    with torch.no_grad():
        sentinel.reshape(-1)[0].add_(1.0)
    corrupted_sha256 = _tensor_sha256(sentinel)
    if corrupted_sha256 == expected_sha256:
        raise RuntimeError("reload probe 未能改变 desc_adapter 哨兵参数")
    corrupted_fields = {
        "optimizer": _corrupt_optimizer_state(optimizer),
        "scheduler": _corrupt_scheduler_state(scheduler),
        "rng": "python+numpy+torch_cpu+torch_cuda",
        "scaler": (
            _corrupt_scaler_state(scaler) if scaler_requested else None
        ),
    }
    _advance_all_training_rngs()
    corrupted_state_hashes = {
        "optimizer": _state_sha256(optimizer.state_dict()),
        "scheduler": _state_sha256(scheduler.state_dict()),
        "rng": _state_sha256(capture_training_rng_state()),
        "scaler": (
            _state_sha256(scaler.state_dict()) if scaler_requested else None
        ),
    }
    uncorrupted = sorted(
        name for name in expected_state_hashes
        if expected_state_hashes[name] is not None
        and corrupted_state_hashes[name] == expected_state_hashes[name]
    )
    if uncorrupted:
        raise RuntimeError(
            f"reload probe 未能扰动 training state: {uncorrupted}"
        )
    step, metadata = load_segdesc_checkpoint(
        resolved,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_stage=expected_stage,
    )
    restored_sha256 = _tensor_sha256(named_parameters[sentinel_name])
    restored_state_hashes = {
        "optimizer": _state_sha256(optimizer.state_dict()),
        "scheduler": _state_sha256(scheduler.state_dict()),
        "rng": _state_sha256(capture_training_rng_state()),
        "scaler": (
            _state_sha256(scaler.state_dict()) if scaler_requested else None
        ),
    }
    state_restored = {
        name: restored_state_hashes[name] == expected_state_hashes[name]
        for name in expected_state_hashes
    }
    if restored_sha256 != expected_sha256 or not all(state_restored.values()):
        raise RuntimeError(
            "strict segdesc checkpoint reload 未完整恢复状态: "
            f"model={restored_sha256 == expected_sha256} state={state_restored}"
        )
    return step, {
        "protocol": STRICT_RELOAD_PROBE_PROTOCOL,
        "passed": True,
        "checkpoint": str(resolved.resolve(strict=False)),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(step),
        "expected_stage": str(expected_stage),
        "sentinel_parameter": sentinel_name,
        "before_sha256": before_sha256,
        "corrupted_sha256": corrupted_sha256,
        "restored_sha256": restored_sha256,
        "optimizer_state_restored": state_restored["optimizer"],
        "scheduler_state_restored": state_restored["scheduler"],
        "rng_state_restored": state_restored["rng"],
        "grad_scaler_state_requested": scaler_requested,
        "grad_scaler_state_restored": state_restored["scaler"],
        "state_probe": {
            "expected_sha256": expected_state_hashes,
            "before_sha256": live_state_hashes_before,
            "corrupted_sha256": corrupted_state_hashes,
            "restored_sha256": restored_state_hashes,
            "corrupted_fields": corrupted_fields,
        },
        "segmentation_migration": dict(
            metadata.get("segmentation_migration") or {}
        ),
    }


def initialize_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    target_stage: str | None = None,
    expected_seed: int | None = None,
    allow_same_stage_curriculum: bool = False,
    require_run_completion: bool = False,
    run_completion_validator: Callable[..., dict[str, Any]] | None = None,
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
    if payload.get("description_protocol_assets") != description_protocol_assets_spec():
        raise RuntimeError("segdesc initialize ontology/schema protocol assets 不一致")
    checkpoint_metadata = checkpoint_metadata_report(payload)
    source_metadata = dict(payload.get("metadata") or {})
    source_stage = str(source_metadata.get("stage") or "")
    source_config = require_serialized_segdesc_config(
        source_metadata.get("config"), label="initialize source checkpoint config"
    )
    source_seed = serialized_segdesc_config_value(source_config, "seed")
    source_run_completion: dict[str, Any] | None = None
    if expected_seed is not None and (
        source_seed is None or int(source_seed) != int(expected_seed)
    ):
        raise RuntimeError(
            "跨 stage initialize seed lineage 不一致: "
            f"expected={int(expected_seed)} observed={source_seed!r}"
        )
    if target_stage is not None:
        expected_source = DESCRIPTION_STAGE_PREDECESSOR.get(target_stage)
        if expected_source is None:
            raise RuntimeError(
                f"stage={target_stage!r} 必须从 segmentation checkpoint 新建，"
                "不能使用 segdesc --initialize-from"
            )
        same_stage_curriculum = bool(
            allow_same_stage_curriculum
            and target_stage == "predicted_mask"
            and source_stage == "predicted_mask"
        )
        if source_stage != expected_source and not same_stage_curriculum:
            raise RuntimeError(
                "跨 stage initialize 顺序非法: "
                f"target={target_stage!r} expected_source={expected_source!r} "
                f"observed_source={source_stage!r}"
            )
        expected_role = DESCRIPTION_STAGE_CHECKPOINT_ROLE.get(source_stage)
        if source_metadata.get("checkpoint_role") != expected_role:
            # 角色来自 checkpoint 本体而非文件名；随后再绑定同目录成功 completion。
            # D3a 无人工 validation，只有它使用 terminal_last；其他阶段使用 best。
            raise RuntimeError(
                "跨 stage initialize checkpoint role 非法: "
                f"source_stage={source_stage!r} expected={expected_role!r} "
                f"observed={source_metadata.get('checkpoint_role')!r}"
            )
        if run_completion_validator is None:
            raise RuntimeError(
                "跨 stage initialize 必须显式提供 run_completion_validator"
            )
        try:
            source_run_completion = run_completion_validator(
                resolved,
                expected_completion_protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
                expected_stage=source_stage,
                expected_role=expected_role,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "跨 stage initialize source run 尚未成功完成或 completion 已漂移"
            ) from exc
    if require_run_completion and source_run_completion is None:
        expected_role = DESCRIPTION_STAGE_CHECKPOINT_ROLE.get(source_stage)
        if expected_role is None:
            raise RuntimeError(
                f"initialize source stage 没有完成角色定义: {source_stage!r}"
            )
        if run_completion_validator is None:
            raise RuntimeError(
                "initialize source completion 验证必须显式提供 run_completion_validator"
            )
        try:
            source_run_completion = run_completion_validator(
                resolved,
                expected_completion_protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
                expected_stage=source_stage,
                expected_role=expected_role,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "initialize source run 尚未成功完成或 completion 已漂移"
            ) from exc
    if tuple(sorted(payload.get("adapter_names") or [])) != _adapter_names(model):
        raise RuntimeError("segdesc adapter names 不一致")
    if payload.get("segmentation_architecture_spec") != _segmentation_architecture_spec(model):
        raise RuntimeError("segdesc initialize segmentation architecture 不一致")
    source_spec = dict(payload.get("description_architecture_spec") or {})
    target_spec = _description_architecture_spec(model)
    non_region_source = {
        key: value for key, value in source_spec.items()
        if key not in REGION_ARCHITECTURE_FIELDS
    }
    non_region_target = {
        key: value for key, value in target_spec.items()
        if key not in REGION_ARCHITECTURE_FIELDS
    }
    if non_region_source != non_region_target:
        raise RuntimeError(
            "segdesc initialize architecture 不一致: "
            f"source={non_region_source} target={non_region_target}"
        )
    source_state = checkpoint_state(payload)
    target_state = _required_state(model)
    region_changed = source_spec.get("region_encoder") != target_spec["region_encoder"]
    if not region_changed:
        source_region_spec = {
            key: source_spec.get(key) for key in REGION_ARCHITECTURE_FIELDS
        }
        target_region_spec = {
            key: target_spec.get(key) for key in REGION_ARCHITECTURE_FIELDS
        }
        if source_region_spec != target_region_spec:
            raise RuntimeError(
                "segdesc initialize MGRR protocol 不一致；同 region encoder 不允许静默迁移: "
                f"source={source_region_spec} target={target_region_spec}"
            )
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
        **checkpoint_metadata,
        "initialization": {
            "source_checkpoint": str(resolved),
            "source_checkpoint_sha256": sha256_file(resolved),
            "source_stage": source_stage or None,
            "target_stage": target_stage,
            "source_seed": int(source_seed) if source_seed is not None else None,
            "target_seed": int(expected_seed) if expected_seed is not None else None,
            "seed_match": expected_seed is None or int(source_seed) == int(expected_seed),
            "same_stage_curriculum": bool(
                allow_same_stage_curriculum
                and target_stage == "predicted_mask"
                and source_stage == "predicted_mask"
            ),
            "source_region_encoder": source_spec.get("region_encoder"),
            "target_region_encoder": target_spec["region_encoder"],
            "region_encoder_reinitialized": region_changed,
            "skipped_source_region_keys": sorted(skipped_source),
            "initialized_target_region_keys": sorted(allowed_missing & set(target_state)),
            "source_run_completion": source_run_completion,
        },
    }
