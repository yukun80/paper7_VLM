"""Qwen3-VL processor 封装与无硬编码 token 的 assistant-only loss mask。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import first_symlink_component

from .data import DescriptionSample
from .errors import ProcessingError, ReasonCode
from oa_groundrag.grounding.messages import strip_assistant_message


def count_message_images(messages: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "message content 必须是列表",
            )
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in {
                "text",
                "image",
            }:
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    "message content 只允许 text/image 对象",
                )
            count += int(item["type"] == "image")
    return count


def assistant_only_labels(
    full_input_ids: Tensor,
    prompt_input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """由 chat template 的真实前缀确定 loss 区间，不猜 assistant token id。"""

    if full_input_ids.ndim != 1 or prompt_input_ids.ndim != 1:
        raise ProcessingError(
            ReasonCode.LOSS_MASK_INVALID,
            "input ids 必须是一维",
        )
    prompt_length = int(prompt_input_ids.numel())
    if prompt_length >= int(full_input_ids.numel()):
        raise ProcessingError(
            ReasonCode.LOSS_MASK_INVALID,
            "assistant target 为空或 prompt 不短于完整序列",
        )
    if not torch.equal(full_input_ids[:prompt_length], prompt_input_ids):
        raise ProcessingError(
            ReasonCode.LOSS_MASK_INVALID,
            "prompt ids 不是完整训练序列的严格前缀",
        )
    labels = full_input_ids.clone()
    labels[:prompt_length] = -100
    if attention_mask is not None:
        if attention_mask.shape != full_input_ids.shape:
            raise ProcessingError(
                ReasonCode.LOSS_MASK_INVALID,
                "attention_mask 与 input_ids shape 不一致",
            )
        labels[attention_mask == 0] = -100
    if not bool((labels != -100).any()):
        raise ProcessingError(
            ReasonCode.LOSS_MASK_INVALID,
            "assistant-only labels 没有监督 token",
        )
    return labels


@dataclass(frozen=True)
class EncodedSample:
    tensors: Mapping[str, Tensor]
    labels: Tensor | None
    input_token_count: int
    image_count: int


def single_inference_tensor_batch(encoded: EncodedSample) -> dict[str, Tensor]:
    """将单个 EncodedSample 变成 Qwen generate batch，不改视觉拼接维。"""

    if encoded.labels is not None:
        raise ProcessingError(
            ReasonCode.LOSS_MASK_INVALID,
            "single inference batch 不接受 labels",
        )
    token_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
    output: dict[str, Tensor] = {}
    for key, value in encoded.tensors.items():
        if key in token_keys:
            if value.ndim != 1:
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    f"single inference {key} 必须是一维",
                )
            output[key] = value.unsqueeze(0)
        elif key == "pixel_values":
            if value.ndim != 2:
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    "single inference pixel_values 必须保持二维拼接",
                )
            output[key] = value
        elif key == "image_grid_thw":
            if value.ndim != 2 or value.shape[1] != 3:
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    "single inference image_grid_thw 必须是 [images,3]",
                )
            output[key] = value
        else:
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                f"single inference 出现未知 tensor：{key}",
            )
    if "input_ids" not in output or "attention_mask" not in output:
        raise ProcessingError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "single inference 缺少 input_ids/attention_mask",
        )
    visual = {"pixel_values", "image_grid_thw"} & set(output)
    if visual and visual != {"pixel_values", "image_grid_thw"}:
        raise ProcessingError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "single inference 视觉 tensors 必须成对出现",
        )
    return output


class ProcessorAdapter(Protocol):
    pad_token_id: int

    def clone_for_worker(self) -> "ProcessorAdapter":
        ...

    def encode_training(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        ...

    def encode_inference(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        ...

    def encode_text_inference(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        ...


class Qwen3VLProcessorAdapter:
    """本地 AutoProcessor + qwen-vl-utils 的薄适配层。"""

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
    ) -> None:
        if not local_files_only or trust_remote_code:
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "processor 必须 local_files_only 且 trust_remote_code=false",
            )
        path = Path(processor_path)
        linked = first_symlink_component(path)
        if linked is not None:
            raise ProcessingError(
                ReasonCode.OUTPUT_LINK,
                f"processor_path 含链接组件：{linked}",
            )
        if path.is_symlink() or not path.is_dir():
            raise ProcessingError(
                ReasonCode.ASSET_MISSING,
                f"processor_path 必须是本地普通目录：{path}",
            )
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "processor pixel limits 非法",
            )
        if max_images <= 0 or max_input_tokens <= 0:
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "processor image/token limits 非法",
            )
        try:
            from transformers import AutoProcessor
        except ImportError as error:
            raise ProcessingError(
                ReasonCode.ASSET_MISSING,
                "缺少 transformers，无法构造 Qwen processor",
            ) from error
        self.processor = AutoProcessor.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        token_id = self.processor.tokenizer.pad_token_id
        if token_id is None:
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen tokenizer 缺少 pad_token_id",
            )
        self.pad_token_id = int(token_id)
        self.max_images = max_images
        self.max_input_tokens = max_input_tokens
        self.processor_path = path.resolve()
        self._worker_clone_kwargs = {
            "processor_path": self.processor_path,
            "local_files_only": True,
            "trust_remote_code": False,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_images": max_images,
            "max_input_tokens": max_input_tokens,
        }

    def clone_for_worker(self) -> "Qwen3VLProcessorAdapter":
        """为一个 CPU 预取 worker 创建独立 processor，避免共享可变状态。"""

        return Qwen3VLProcessorAdapter(**self._worker_clone_kwargs)

    def identity(self) -> dict[str, Any]:
        files: dict[str, str] = {}
        for name in (
            "preprocessor_config.json",
            "tokenizer_config.json",
            "chat_template.json",
        ):
            path = self.processor_path / name
            if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
                raise ProcessingError(
                    ReasonCode.ASSET_MISSING,
                    f"processor identity 缺少普通单链接文件：{name}",
                )
            files[name] = sha256_file(path)
        return {
            "processor_path": str(self.processor_path),
            "files": files,
            "processor_class": type(self.processor).__name__,
            "tokenizer_class": type(self.processor.tokenizer).__name__,
            "pad_token_id": self.pad_token_id,
        }

    def encode_training(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        message_list = [dict(message) for message in messages]
        if (
            len(message_list) != 2
            or message_list[0].get("role") != "user"
            or message_list[1].get("role") != "assistant"
        ):
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "训练消息必须是单轮 user/assistant",
            )
        image_count = self._validate_images(message_list)
        prompt_messages = strip_assistant_message(message_list)
        full = self._encode(
            message_list,
            add_generation_prompt=False,
        )
        prompt = self._encode(
            prompt_messages,
            add_generation_prompt=True,
        )
        full_ids = full["input_ids"]
        prompt_ids = prompt["input_ids"]
        labels = assistant_only_labels(
            full_ids,
            prompt_ids,
            attention_mask=full["attention_mask"],
        )
        return EncodedSample(
            tensors=full,
            labels=labels,
            input_token_count=int(full_ids.numel()),
            image_count=image_count,
        )

    def encode_inference(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        message_list = strip_assistant_message(
            [dict(message) for message in messages]
        )
        image_count = self._validate_images(message_list)
        tensors = self._encode(
            message_list,
            add_generation_prompt=True,
        )
        return EncodedSample(
            tensors=tensors,
            labels=None,
            input_token_count=int(tensors["input_ids"].numel()),
            image_count=image_count,
        )

    def encode_text_inference(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> EncodedSample:
        """Stage 6 文本 Pass-2 专用入口；不改变既有视觉接口的至少一图合同。"""

        raw_messages = [dict(message) for message in messages]
        continue_final_message = (
            len(raw_messages) == 2
            and raw_messages[0].get("role") == "user"
            and raw_messages[1].get("role") == "assistant"
        )
        if continue_final_message:
            assistant_content = raw_messages[1].get("content")
            if (
                not isinstance(assistant_content, list)
                or len(assistant_content) != 1
                or assistant_content[0].get("type") != "text"
                or not isinstance(assistant_content[0].get("text"), str)
                or not assistant_content[0]["text"]
            ):
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    "text-only assistant prefill 必须是单一非空文本块",
                )
            message_list = raw_messages
        else:
            message_list = strip_assistant_message(raw_messages)
        if count_message_images(message_list) != 0:
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "text-only inference 不接受图像",
            )
        text = self.processor.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )
        encoded = self.processor(
            text=[text],
            padding=False,
            return_tensors="pt",
        )
        allowed = {"input_ids", "attention_mask", "mm_token_type_ids"}
        required = {"input_ids", "attention_mask"}
        unknown = set(encoded) - allowed
        if unknown or required - set(encoded):
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen text-only processor tensor keys 不匹配",
                details={
                    "unknown": sorted(unknown),
                    "missing": sorted(required - set(encoded)),
                },
            )
        output: dict[str, Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != 1:
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    f"text-only processor {key} shape/type 非法",
                )
            output[key] = value[0].contiguous()
        token_count = int(output["input_ids"].numel())
        if token_count > self.max_input_tokens:
            raise ProcessingError(
                ReasonCode.TOKEN_LIMIT_EXCEEDED,
                f"text-only input tokens={token_count} 超过上限 {self.max_input_tokens}",
            )
        return EncodedSample(
            tensors=output,
            labels=None,
            input_token_count=token_count,
            image_count=0,
        )

    def _validate_images(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        count = count_message_images(messages)
        if count > self.max_images:
            raise ProcessingError(
                ReasonCode.IMAGE_LIMIT_EXCEEDED,
                f"message images={count} 超过 max_images={self.max_images}",
            )
        if count == 0:
            raise ProcessingError(
                ReasonCode.ASSET_MISSING,
                "VLM message 至少需要一张图像",
            )
        return count

    def _encode(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> dict[str, Tensor]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise ProcessingError(
                ReasonCode.ASSET_MISSING,
                "缺少 qwen-vl-utils",
            ) from error
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if video_inputs:
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "Shared VLM 不接受 video 输入",
            )
        encoded = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=False,
            return_tensors="pt",
        )
        allowed = {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        }
        unknown = set(encoded) - allowed
        required = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        }
        if unknown or required - set(encoded):
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen processor tensor keys 与当前适配器不一致",
                details={
                    "unknown": sorted(unknown),
                    "missing": sorted(required - set(encoded)),
                },
            )
        output: dict[str, Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, Tensor):
                raise ProcessingError(
                    ReasonCode.TYPE_MISMATCH,
                    f"processor 输出 {key} 不是 Tensor",
                )
            if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
                if value.ndim != 2 or value.shape[0] != 1:
                    raise ProcessingError(
                        ReasonCode.TYPE_MISMATCH,
                        f"processor {key} shape 非法",
                    )
                output[key] = value[0].contiguous()
            else:
                output[key] = value.contiguous()
        token_count = int(output["input_ids"].numel())
        if token_count > self.max_input_tokens:
            raise ProcessingError(
                ReasonCode.TOKEN_LIMIT_EXCEEDED,
                f"input tokens={token_count} 超过上限 {self.max_input_tokens}",
            )
        return output


class DescriptionCollator:
    """对已编码样本右侧 padding，并拼接 Qwen 视觉网格。"""

    def __init__(
        self,
        processor: ProcessorAdapter,
        *,
        training: bool,
    ) -> None:
        self.processor = processor
        self.training = training

    def clone_for_worker(self) -> "DescriptionCollator":
        clone = getattr(self.processor, "clone_for_worker", None)
        if not callable(clone):
            raise ProcessingError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "异步输入要求 processor 实现 clone_for_worker()",
            )
        return DescriptionCollator(clone(), training=self.training)

    def __call__(
        self,
        samples: Sequence[DescriptionSample],
    ) -> dict[str, Any]:
        if not samples:
            raise ProcessingError(
                ReasonCode.TYPE_MISMATCH,
                "不能 collate 空 batch",
            )
        encoded = [
            (
                self.processor.encode_training(sample.messages)
                if self.training
                else self.processor.encode_inference(sample.messages)
            )
            for sample in samples
        ]
        max_length = max(item.input_token_count for item in encoded)
        batch_size = len(encoded)
        input_ids = torch.full(
            (batch_size, max_length),
            self.processor.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )
        mm_token_type_ids = (
            torch.zeros((batch_size, max_length), dtype=torch.long)
            if any("mm_token_type_ids" in item.tensors for item in encoded)
            else None
        )
        labels = (
            torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long,
            )
            if self.training
            else None
        )
        for index, item in enumerate(encoded):
            length = item.input_token_count
            input_ids[index, :length] = item.tensors["input_ids"]
            attention_mask[index, :length] = item.tensors["attention_mask"]
            if mm_token_type_ids is not None:
                value = item.tensors.get("mm_token_type_ids")
                if value is None:
                    raise ProcessingError(
                        ReasonCode.MODEL_IDENTITY_MISMATCH,
                        "batch 中 mm_token_type_ids 键不一致",
                    )
                mm_token_type_ids[index, :length] = value
            if labels is not None:
                if item.labels is None:
                    raise ProcessingError(
                        ReasonCode.LOSS_MASK_INVALID,
                        "训练 batch 缺少 labels",
                    )
                labels[index, :length] = item.labels
        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.cat(
                [item.tensors["pixel_values"] for item in encoded],
                dim=0,
            ),
            "image_grid_thw": torch.cat(
                [item.tensors["image_grid_thw"] for item in encoded],
                dim=0,
            ),
            "record_ids": [sample.record_id for sample in samples],
            "parent_ids": [sample.parent_id for sample in samples],
            "task_families": [sample.task_family for sample in samples],
            "mask_modes": [sample.mask_mode.value for sample in samples],
            "reference_responses": [
                sample.reference_responses for sample in samples
            ],
            "evidence_ids": [sample.evidence_ids for sample in samples],
            "provenance": [sample.provenance for sample in samples],
            "counterfactual": [sample.counterfactual for sample in samples],
            "input_token_counts": [
                item.input_token_count for item in encoded
            ],
            "image_counts": [item.image_count for item in encoded],
        }
        if mm_token_type_ids is not None:
            batch["mm_token_type_ids"] = mm_token_type_ids
        if labels is not None:
            batch["labels"] = labels
        return batch


def model_tensor_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    allowed = {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
        "labels",
    }
    return {
        key: value
        for key, value in batch.items()
        if key in allowed and isinstance(value, Tensor)
    }
