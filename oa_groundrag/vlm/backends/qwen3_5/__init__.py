"""Qwen3.5 backend 规格；processor/model 分别由 M3/M4 实现。"""

from __future__ import annotations

from typing import Any

from oa_groundrag.vlm.backends.contracts import VLMBackendSpec
from oa_groundrag.vlm.backends.assets import verify_model_asset_ledger
from oa_groundrag.vlm.config import VLMConfig
from oa_groundrag.vlm.errors import ModelError, ReasonCode


def _build_processor(config: VLMConfig) -> Any:
    from .processing import Qwen3_5ProcessorAdapter

    model = config.model
    if (
        model.asset_ledger_path is None
        or model.hub_repo_id is None
        or model.hub_revision is None
    ):
        raise ModelError(
            ReasonCode.ASSET_LEDGER_INVALID,
            "qwen3_5 config 必须绑定 Hub revision 与 asset ledger",
        )
    assets = verify_model_asset_ledger(
        model.asset_ledger_path,
        expected_backend="qwen3_5",
        expected_repo_id=model.hub_repo_id,
        expected_revision=model.hub_revision,
        expected_model_root=model.path,
    )
    return Qwen3_5ProcessorAdapter(
        processor_path=model.processor_path,
        local_files_only=model.local_files_only,
        trust_remote_code=model.trust_remote_code,
        min_pixels=config.limits.min_pixels,
        max_pixels=config.limits.max_pixels,
        max_images=config.limits.max_images,
        max_input_tokens=config.limits.max_input_tokens,
        enable_thinking=False,
        asset_identity=assets.identity_dict(),
    )


def _build_model(
    config: VLMConfig,
    device: Any,
    gradient_checkpointing: bool,
) -> Any:
    from .model import Qwen35ModelAdapter

    model = config.model
    if (
        model.asset_ledger_path is None
        or model.hub_repo_id is None
        or model.hub_revision is None
    ):
        raise ModelError(
            ReasonCode.ASSET_LEDGER_INVALID,
            "qwen3_5 config 必须绑定 Hub revision 与 asset ledger",
        )
    assets = verify_model_asset_ledger(
        model.asset_ledger_path,
        expected_backend="qwen3_5",
        expected_repo_id=model.hub_repo_id,
        expected_revision=model.hub_revision,
        expected_model_root=model.path,
    )
    return Qwen35ModelAdapter.load(
        model,
        config.adaptation,
        assets=assets,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
    )


BACKEND_SPEC = VLMBackendSpec(
    name="qwen3_5",
    model_types=("qwen3_5",),
    architectures=("Qwen3_5ForConditionalGeneration",),
    processor_classes=("Qwen3VLProcessor",),
    supports_thinking=True,
    processor_builder=_build_processor,
    model_builder=_build_model,
)
