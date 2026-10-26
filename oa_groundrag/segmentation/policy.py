"""OA-AuxSeg 训练/推理共享的模态策略与只读校验。"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import OAAuxSegBatch, RuntimeConfig, SUPPORTED_BACKBONE
from .data import (
    AuxiliarySubsetSampler,
    available_auxiliaries_by_sample,
    filter_auxiliaries,
    registry_from_benchmark,
)
from .model import OAAuxSegModel


def autocast_context(config: RuntimeConfig, device: torch.device):
    if device.type == "cuda" and config.use_bf16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("配置要求 bf16，但当前 GPU 不支持")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def prepare_policy_batch(
    batch: OAAuxSegBatch,
    *,
    variant: str,
    subset_sampler: AuxiliarySubsetSampler | None,
) -> tuple[OAAuxSegBatch, list[tuple[str, ...]], list[tuple[str, ...]]]:
    available = available_auxiliaries_by_sample(batch)
    if variant == "optical_only":
        active = [tuple() for _ in available]
    elif variant == "proposed_dropout" and subset_sampler is not None:
        active = subset_sampler.sample(available)
    else:
        active = available
    return filter_auxiliaries(batch, active), available, active


def validate_inference_config(
    payload: Mapping[str, Any],
    model: OAAuxSegModel,
    config: RuntimeConfig,
) -> None:
    stored = RuntimeConfig.from_dict(payload["training_state"]["runtime_config"])
    if model.variant != config.variant or stored.variant != config.variant:
        raise ValueError("运行配置 variant 与 checkpoint 不一致")
    if config.backbone != SUPPORTED_BACKBONE or stored.backbone != config.backbone:
        raise ValueError("运行配置 backbone 与 checkpoint 不一致")
    if stored.normalization != config.normalization:
        raise ValueError("运行配置 normalization 与 checkpoint 不一致")
    if (
        model.config.region_threshold != config.region_threshold
        or model.config.min_region_area != config.min_region_area
    ):
        raise ValueError("运行配置区域提取合同与 checkpoint 不一致")


def validate_benchmark_registry(
    model: OAAuxSegModel,
    benchmark_root: Path,
) -> None:
    if model.registry != registry_from_benchmark(benchmark_root):
        raise ValueError("checkpoint registry 与当前 Benchmark index 不一致")
