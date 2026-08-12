"""OA-AuxSeg 当前 schema 的原子 checkpoint 保存与严格恢复。"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts import (
    BENCHMARK_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    ModelRegistry,
    OAAuxSegConfig,
    PHASE2_EXCLUDED_SOURCES,
    PHASE2_INCLUDED_SOURCES,
    RuntimeConfig,
)
from .fusion import FUSION_CONTRACT
from .model import (
    ARCHITECTURE_NAME,
    AUXILIARY_STEM_CONTRACT,
    BACKBONE_CONTRACT_NAME,
    BACKBONE_STAGE_CHANNELS,
    BACKBONE_STAGE_DEPTHS,
    DECODER_CONTRACT,
    OPTICAL_STEM_CONTRACT,
    OAAuxSegModel,
    optical_rgb_indices,
)
from .validity import VALIDITY_CONTRACT


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": str(name),
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["name"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint 含 CUDA RNG，但当前 CUDA 不可用")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG 数量与当前设备数不一致")
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validated_benchmark_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint Benchmark 合同必须是对象")
    expected_fields = {
        "schema_version",
        "index_sha256",
        "manifest_sha256",
        "source_selection",
        "included_sources",
        "excluded_sources",
    }
    if set(value) != expected_fields:
        raise ValueError("checkpoint Benchmark 合同字段不匹配")
    if value["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("checkpoint Benchmark schema 不受支持")
    included = value["included_sources"]
    excluded = value["excluded_sources"]
    if not (
        isinstance(included, list)
        and included
        and all(isinstance(source, str) for source in included)
        and isinstance(excluded, list)
        and all(isinstance(source, str) for source in excluded)
    ):
        raise ValueError("checkpoint Benchmark source selection 非法")
    if set(included) & set(excluded):
        raise ValueError("checkpoint Benchmark sources 不能重叠")
    if tuple(included) != PHASE2_INCLUDED_SOURCES:
        raise ValueError("checkpoint Benchmark included_sources 不匹配")
    if tuple(excluded) != PHASE2_EXCLUDED_SOURCES:
        raise ValueError("checkpoint Benchmark excluded_sources 不匹配")
    selection = str(value["source_selection"])
    if selection != ("subset" if excluded else "all"):
        raise ValueError("checkpoint Benchmark source_selection 非法")
    for name in ("index_sha256", "manifest_sha256"):
        digest = value[name]
        if not (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"checkpoint Benchmark {name} 非法")
    return {
        "schema_version": str(value["schema_version"]),
        "index_sha256": str(value["index_sha256"]),
        "manifest_sha256": str(value["manifest_sha256"]),
        "source_selection": selection,
        "included_sources": list(included),
        "excluded_sources": list(excluded),
    }


def build_checkpoint(
    *,
    model: OAAuxSegModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    benchmark_contract: Mapping[str, Any],
    subset_sampler_state: Mapping[str, Any] | None,
    training_batcher_state: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    if step < 0:
        raise ValueError("checkpoint step 不能小于 0")
    contract = model.model_contract()
    if not contract["backbone_sha256"]:
        raise ValueError("正式 checkpoint 必须记录本地 backbone SHA-256")
    runtime_config = training_state.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise ValueError("checkpoint training_state 缺少 runtime_config")
    RuntimeConfig.from_dict(runtime_config)
    frozen_benchmark_contract = _validated_benchmark_contract(
        benchmark_contract
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_contract": contract,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_state": {
            "enabled": bool(scaler.is_enabled()),
            "state": scaler.state_dict(),
        },
        "step": int(step),
        "benchmark_contract": frozen_benchmark_contract,
        "rng_state": capture_rng_state(),
        "subset_sampler_state": (
            dict(subset_sampler_state) if subset_sampler_state is not None else None
        ),
        "training_batcher_state": dict(training_batcher_state),
        "training_state": dict(training_state),
    }


def save_training_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(path, build_checkpoint(**kwargs))


def read_checkpoint(
    path: Path,
    *,
    expected_benchmark_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint 顶层必须是对象")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"只支持 {CHECKPOINT_SCHEMA_VERSION}，实际为 "
            f"{payload.get('schema_version')!r}"
        )
    required = {
        "schema_version",
        "model_contract",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "amp_state",
        "step",
        "benchmark_contract",
        "rng_state",
        "subset_sampler_state",
        "training_batcher_state",
        "training_state",
    }
    unknown = set(payload) - required
    missing = required - set(payload)
    if unknown or missing:
        raise ValueError(
            f"checkpoint 字段不匹配，missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    recorded_benchmark_contract = _validated_benchmark_contract(
        payload["benchmark_contract"]
    )
    payload["benchmark_contract"] = recorded_benchmark_contract
    if expected_benchmark_contract is not None:
        expected = _validated_benchmark_contract(
            expected_benchmark_contract
        )
        if recorded_benchmark_contract != expected:
            raise ValueError("checkpoint 与当前 Benchmark 合同不一致")
    training_state = payload["training_state"]
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint training_state 必须是对象")
    runtime_config = training_state.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise ValueError("checkpoint training_state 缺少 runtime_config")
    RuntimeConfig.from_dict(runtime_config)
    if not isinstance(payload["training_batcher_state"], dict):
        raise ValueError("checkpoint training_batcher_state 必须是对象")
    return payload


def model_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> OAAuxSegModel:
    contract = payload["model_contract"]
    allowed = {
        "config",
        "registry",
        "architecture",
        "backbone",
        "backbone_stage_channels",
        "backbone_stage_depths",
        "backbone_sha256",
        "optical_stem_contract",
        "auxiliary_stem_contract",
        "optical_rgb_mapping",
        "fusion_contract",
        "decoder_contract",
        "validity_contract",
        "modality_weight_order",
        "modality_weight_map_strides",
        "region_feature_dim",
    }
    if set(contract) != allowed:
        raise ValueError("checkpoint model_contract 字段不匹配")
    if contract["architecture"] != ARCHITECTURE_NAME:
        raise ValueError("checkpoint architecture 合同不受支持")
    if contract["backbone"] != BACKBONE_CONTRACT_NAME:
        raise ValueError("checkpoint backbone 合同不受支持")
    if contract["backbone_stage_channels"] != list(BACKBONE_STAGE_CHANNELS):
        raise ValueError("checkpoint backbone stage channels 不匹配")
    if contract["backbone_stage_depths"] != list(BACKBONE_STAGE_DEPTHS):
        raise ValueError("checkpoint backbone stage depths 不匹配")
    if contract["optical_stem_contract"] != OPTICAL_STEM_CONTRACT:
        raise ValueError("checkpoint optical stem 合同不受支持")
    if contract["auxiliary_stem_contract"] != AUXILIARY_STEM_CONTRACT:
        raise ValueError("checkpoint auxiliary stem 合同不受支持")
    if contract["fusion_contract"] != FUSION_CONTRACT:
        raise ValueError("checkpoint fusion 合同不受支持")
    if contract["decoder_contract"] != DECODER_CONTRACT:
        raise ValueError("checkpoint decoder 合同不受支持")
    if contract["validity_contract"] != VALIDITY_CONTRACT:
        raise ValueError("checkpoint validity 合同不受支持")
    config = OAAuxSegConfig.from_dict(contract["config"])
    registry = ModelRegistry.from_dict(contract["registry"])
    expected_weight_order = list(registry.modality_weight_order)
    if contract["modality_weight_order"] != expected_weight_order:
        raise ValueError("checkpoint modality weight order 与 registry 不一致")
    if contract["modality_weight_map_strides"] != [4, 8, 16, 32]:
        raise ValueError("checkpoint modality weight map strides 不匹配")
    expected_mapping = [
        {
            "channel_names": list(signature),
            "rgb_indices": list(optical_rgb_indices(signature)),
            "extra_indices": [
                index
                for index in range(len(signature))
                if index not in set(optical_rgb_indices(signature))
            ],
        }
        for signature in registry.optical_signatures
    ]
    if contract["optical_rgb_mapping"] != expected_mapping:
        raise ValueError("checkpoint optical RGB mapping 不匹配")
    expected_feature_dim = registry.region_feature_dim(
        config.region_projection_dim
    )
    if contract["region_feature_dim"] != expected_feature_dim:
        raise ValueError("checkpoint region feature 维度与 registry 不一致")
    model = OAAuxSegModel(
        config,
        registry,
        backbone_weights=None,
        gradient_checkpointing=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.backbone_sha256 = str(contract["backbone_sha256"])
    return model.to(device)


def restore_training_state(
    *,
    payload: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> None:
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    amp_state = payload["amp_state"]
    if bool(amp_state["enabled"]) != bool(scaler.is_enabled()):
        raise ValueError("checkpoint AMP 配置与当前运行不一致")
    scaler.load_state_dict(amp_state["state"])
    restore_rng_state(payload["rng_state"])


def optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def module_parameter_snapshot(
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def maximum_parameter_change(
    module: nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> float:
    changes = []
    for name, parameter in module.named_parameters():
        if name in snapshot:
            changes.append(
                float(
                    (parameter.detach().cpu() - snapshot[name])
                    .abs()
                    .max()
                    .item()
                )
            )
    return max(changes, default=0.0)
