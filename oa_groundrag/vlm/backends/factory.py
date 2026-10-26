"""从类型化配置构造唯一所选 VLM backend。"""

from __future__ import annotations

from typing import Any

from oa_groundrag.vlm.config import ModelSection, VLMConfig

from .contracts import VLMBackendSpec, VLMModelAdapter, VLMProcessorAdapter
from .registry import DEFAULT_VLM_BACKEND_REGISTRY, VLMBackendRegistry


def _backend_name(value: str | ModelSection | VLMConfig | Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, ModelSection):
        return value.backend
    model = getattr(value, "model", None)
    if isinstance(model, ModelSection):
        return model.backend
    raise TypeError("backend resolver 只接受 name、ModelSection 或 VLMConfig")


def resolve_vlm_backend(
    value: str | ModelSection | VLMConfig | Any,
    *,
    registry: VLMBackendRegistry = DEFAULT_VLM_BACKEND_REGISTRY,
) -> VLMBackendSpec:
    return registry.resolve(_backend_name(value))


def build_processor_adapter(
    config: VLMConfig | Any,
    *,
    registry: VLMBackendRegistry = DEFAULT_VLM_BACKEND_REGISTRY,
) -> VLMProcessorAdapter:
    return resolve_vlm_backend(config, registry=registry).processor_builder(config)


def build_model_adapter(
    config: VLMConfig | Any,
    *,
    device: Any,
    gradient_checkpointing: bool,
    registry: VLMBackendRegistry = DEFAULT_VLM_BACKEND_REGISTRY,
) -> VLMModelAdapter:
    return resolve_vlm_backend(config, registry=registry).model_builder(
        config,
        device,
        gradient_checkpointing,
    )
