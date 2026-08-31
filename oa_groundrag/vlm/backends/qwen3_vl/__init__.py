"""Qwen3-VL backend 规格；M2 将具体实现迁入本包。"""

from __future__ import annotations

from typing import Any

from oa_groundrag.vlm.backends.contracts import VLMBackendSpec
from oa_groundrag.vlm.config import VLMConfig


def _build_processor(config: VLMConfig) -> Any:
    from .processing import Qwen3VLProcessorAdapter

    return Qwen3VLProcessorAdapter(
        processor_path=config.model.processor_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        min_pixels=config.limits.min_pixels,
        max_pixels=config.limits.max_pixels,
        max_images=config.limits.max_images,
        max_input_tokens=config.limits.max_input_tokens,
    )


def _build_model(
    config: VLMConfig,
    device: Any,
    gradient_checkpointing: bool,
) -> Any:
    from .model import Qwen3VLModelAdapter

    return Qwen3VLModelAdapter.load(
        config.model,
        config.adaptation,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
    )


BACKEND_SPEC = VLMBackendSpec(
    name="qwen3_vl",
    model_types=("qwen3_vl",),
    architectures=("Qwen3VLForConditionalGeneration",),
    processor_classes=("Qwen3VLProcessor",),
    supports_thinking=False,
    processor_builder=_build_processor,
    model_builder=_build_model,
)
