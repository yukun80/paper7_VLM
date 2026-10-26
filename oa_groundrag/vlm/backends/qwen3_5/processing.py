"""Qwen3.5-4B Jinja/thinking processor 后端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.vlm.errors import ProcessingError, ReasonCode

from ...processing import EncodedSample, _LocalMultimodalProcessorAdapter


QWEN3_5_PROCESSOR_IDENTITY_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


class Qwen3_5ProcessorAdapter(_LocalMultimodalProcessorAdapter):
    """显式控制 Qwen3.5 thinking，训练时强制 non-thinking。"""

    def __init__(
        self,
        *,
        processor_path: Path,
        local_files_only: bool,
        trust_remote_code: bool,
        min_pixels: int,
        max_pixels: int,
        max_images: int,
        max_input_tokens: int,
        enable_thinking: bool,
        asset_identity: Mapping[str, Any],
    ) -> None:
        if not isinstance(enable_thinking, bool):
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "enable_thinking 必须是 bool",
            )
        required_identity = {
            "backend",
            "hub_repo_id",
            "hub_revision",
            "model_root",
            "asset_ledger_path",
            "asset_ledger_file_sha256",
            "asset_ledger_payload_sha256",
            "asset_total_size_bytes",
            "asset_file_count",
            "weight_shards",
        }
        if (
            not isinstance(asset_identity, Mapping)
            or set(asset_identity) != required_identity
            or asset_identity.get("backend") != "qwen3_5"
            or asset_identity.get("hub_repo_id") != "Qwen/Qwen3.5-4B"
        ):
            raise ProcessingError(
                ReasonCode.ASSET_LEDGER_INVALID,
                "Qwen3.5 processor 缺少已验证 asset identity",
            )
        self.thinking_enabled = enable_thinking
        self._asset_identity = dict(asset_identity)
        super().__init__(
            processor_path=processor_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_images=max_images,
            max_input_tokens=max_input_tokens,
            identity_file_names=QWEN3_5_PROCESSOR_IDENTITY_FILES,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        self._worker_clone_kwargs = {
            "processor_path": self.processor_path,
            "local_files_only": True,
            "trust_remote_code": False,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_images": max_images,
            "max_input_tokens": max_input_tokens,
            "enable_thinking": enable_thinking,
            "asset_identity": self._asset_identity,
        }

    def identity(self) -> dict[str, Any]:
        value = super().identity()
        value.update(
            {
                "backend": "qwen3_5",
                "chat_template_format": "jinja",
                "thinking_supported": True,
                "asset_identity": dict(self._asset_identity),
            }
        )
        return value

    def with_thinking(self, enabled: bool) -> "Qwen3_5ProcessorAdapter":
        return Qwen3_5ProcessorAdapter(
            **{
                **self._worker_clone_kwargs,
                "enable_thinking": enabled,
            }
        )

    def encode_training(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        if self.thinking_enabled:
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5 训练必须使用 non-thinking template",
            )
        return super().encode_training(messages)

