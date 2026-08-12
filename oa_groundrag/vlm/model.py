"""Qwen3-VL 模型封装与唯一 prompt-only/LoRA 适配主线。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import first_symlink_component

from .config import AdaptationSection, ModelSection
from .errors import ModelError, ReasonCode
from .processing import model_tensor_batch


EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS = 2_127_532_032
EXPECTED_ATTENTION_LORA_R8_PARAMETERS = 3_211_264


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key: {key}")
        output[key] = value
    return output


def _safetensors_parameter_count(header: bytes) -> int:
    value = json.loads(header, object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict):
        raise ValueError("safetensors header must be an object")
    total = 0
    for name, contract in value.items():
        if name == "__metadata__":
            continue
        shape = contract.get("shape") if isinstance(contract, dict) else None
        if (
            not isinstance(shape, list)
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            )
        ):
            raise ValueError(f"invalid tensor shape: {name}")
        total += math.prod(shape)
    return total


@dataclass(frozen=True)
class ModelIdentity:
    model_path: str
    model_type: str
    config_sha256: str
    weights_metadata_sha256: str
    processor_config_sha256: str
    base_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "config_sha256": self.config_sha256,
            "weights_metadata_sha256": self.weights_metadata_sha256,
            "processor_config_sha256": self.processor_config_sha256,
            "base_parameter_count": self.base_parameter_count,
        }


@dataclass(frozen=True)
class ForwardResult:
    loss: Tensor
    logits: Tensor | None


def assistant_sample_mean_causal_loss(
    logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """逐样本等权的 assistant-only causal loss。"""

    if (
        logits.ndim != 3
        or labels.ndim != 2
        or logits.shape[:2] != labels.shape
        or logits.shape[-1] <= 0
    ):
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "logits/labels shape 不符合 causal loss 合同",
        )
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    supervised = shifted_labels.ne(-100)
    counts = supervised.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "batch 中存在没有 assistant 监督 token 的样本",
        )
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1).to(logits.device),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    sample_losses = (
        (token_losses * supervised.to(token_losses.dtype)).sum(dim=1)
        / counts.to(token_losses.dtype)
    )
    loss = sample_losses.mean()
    if not bool(torch.isfinite(loss)):
        raise ModelError(
            ReasonCode.NONFINITE_NUMBER,
            "逐样本 assistant-only loss 非有限",
        )
    return loss


def local_model_identity(
    path: Path,
    *,
    base_parameter_count: int,
    model_type: str,
) -> ModelIdentity:
    root = Path(path)
    linked = first_symlink_component(root)
    if linked is not None:
        raise ModelError(
            ReasonCode.OUTPUT_LINK,
            f"本地模型路径含链接组件：{linked}",
        )
    if not root.is_dir():
        raise ModelError(
            ReasonCode.ASSET_MISSING,
            f"本地模型目录不存在：{root}",
        )
    required = ("config.json", "preprocessor_config.json")
    values: dict[str, str] = {}
    for name in required:
        child = root / name
        if (
            not child.is_file()
            or child.is_symlink()
            or child.stat().st_nlink != 1
        ):
            raise ModelError(
                ReasonCode.ASSET_MISSING,
                f"本地模型缺少普通单链接文件：{child}",
            )
        values[name] = sha256_file(child)
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if index.is_file() and not index.is_symlink() and index.stat().st_nlink == 1:
        weights_metadata_sha256 = sha256_file(index)
    elif (
        single.is_file()
        and not single.is_symlink()
        and single.stat().st_nlink == 1
    ):
        try:
            with single.open("rb") as handle:
                header_length_bytes = handle.read(8)
                if len(header_length_bytes) != 8:
                    raise ValueError("short safetensors header")
                header_length = int.from_bytes(
                    header_length_bytes,
                    byteorder="little",
                    signed=False,
                )
                if header_length <= 0 or header_length > 64 * 1024 * 1024:
                    raise ValueError("invalid safetensors header length")
                header = handle.read(header_length)
                if len(header) != header_length:
                    raise ValueError("truncated safetensors header")
                metadata_parameter_count = _safetensors_parameter_count(header)
        except (OSError, ValueError) as error:
            raise ModelError(
                ReasonCode.ASSET_MISSING,
                "无法读取本地 safetensors metadata",
                details={"error": str(error)},
            ) from error
        digest = hashlib.sha256()
        digest.update(str(single.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(header_length_bytes)
        digest.update(header)
        weights_metadata_sha256 = digest.hexdigest()
        if metadata_parameter_count != base_parameter_count:
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "safetensors metadata 参数量与模型 identity 不一致",
                details={
                    "expected": base_parameter_count,
                    "actual": metadata_parameter_count,
                },
            )
    else:
        raise ModelError(
            ReasonCode.ASSET_MISSING,
            "本地模型缺少 model.safetensors 或 index",
        )
    return ModelIdentity(
        model_path=str(root.resolve()),
        model_type=model_type,
        config_sha256=values["config.json"],
        weights_metadata_sha256=weights_metadata_sha256,
        processor_config_sha256=values["preprocessor_config.json"],
        base_parameter_count=base_parameter_count,
    )


class Qwen3VLModelAdapter:
    """负责加载、冻结、LoRA、逐样本 loss 和生成。"""

    def __init__(
        self,
        model: nn.Module,
        *,
        identity: ModelIdentity,
        adaptation: AdaptationSection,
    ) -> None:
        self.model = model
        self.identity = identity
        self.adaptation = adaptation
        self._trainable_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self.trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if any(
            any(token in name.lower() for token in ("visual", "vision", "merger"))
            for name in self._trainable_names
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "vision/merger 出现可训练参数，违反主线冻结合同",
            )
        if adaptation.strategy == "prompt_only":
            if self.trainable_parameter_count != 0:
                raise ModelError(
                    ReasonCode.MODEL_IDENTITY_MISMATCH,
                    "prompt-only baseline 必须有 0 个训练参数",
                )
        elif (
            identity.model_type == "qwen3_vl"
            and identity.base_parameter_count
            == EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS
            and self.trainable_parameter_count
            != EXPECTED_ATTENTION_LORA_R8_PARAMETERS
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3-VL-2B attention LoRA 参数量不符合锁定设计",
                details={
                    "expected": EXPECTED_ATTENTION_LORA_R8_PARAMETERS,
                    "actual": self.trainable_parameter_count,
                },
            )

    @classmethod
    def load(
        cls,
        model_config: ModelSection,
        adaptation: AdaptationSection,
        *,
        device: torch.device,
        gradient_checkpointing: bool,
    ) -> "Qwen3VLModelAdapter":
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as error:
            raise ModelError(
                ReasonCode.ASSET_MISSING,
                "缺少 transformers",
            ) from error
        dtype = {"bfloat16": torch.bfloat16}.get(model_config.dtype)
        if dtype is None:
            raise ModelError(
                ReasonCode.INVALID_ENUM,
                f"不支持 dtype={model_config.dtype}",
            )
        model = AutoModelForImageTextToText.from_pretrained(
            model_config.path,
            local_files_only=model_config.local_files_only,
            trust_remote_code=model_config.trust_remote_code,
            dtype=dtype,
            attn_implementation=model_config.attn_implementation,
        )
        base_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        model_type = str(getattr(model.config, "model_type", "unknown"))
        identity = local_model_identity(
            model_config.path,
            base_parameter_count=base_parameter_count,
            model_type=model_type,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if gradient_checkpointing:
            for module in model.modules():
                module_config = getattr(module, "config", None)
                if hasattr(module_config, "use_cache"):
                    module_config.use_cache = False
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        if adaptation.strategy == "lora":
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as error:
                raise ModelError(
                    ReasonCode.ASSET_MISSING,
                    "LoRA 配置需要本地 peft",
                ) from error
            lora_config = LoraConfig(
                r=adaptation.rank,
                lora_alpha=adaptation.alpha,
                lora_dropout=adaptation.dropout,
                target_modules=list(adaptation.target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
        elif adaptation.strategy != "prompt_only":
            raise ModelError(
                ReasonCode.INVALID_ENUM,
                f"不支持 adaptation={adaptation.strategy}",
            )
        model.to(device)
        return cls(model, identity=identity, adaptation=adaptation)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return self._trainable_names

    def train(self) -> None:
        self.model.train()
        self._force_frozen_vision_eval()

    def eval(self) -> None:
        self.model.eval()

    def _force_frozen_vision_eval(self) -> None:
        for name, module in self.model.named_modules():
            if any(
                token in name.lower() for token in ("visual", "vision", "merger")
            ):
                module.eval()

    def forward(self, batch: Mapping[str, Any]) -> ForwardResult:
        tensor_batch = model_tensor_batch(batch)
        labels = tensor_batch.pop("labels", None)
        if not isinstance(labels, Tensor):
            raise ModelError(
                ReasonCode.LOSS_MASK_INVALID,
                "Qwen 训练 forward 缺少 labels",
            )
        output = self.model(
            **tensor_batch,
            use_cache=False,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise ModelError(
                ReasonCode.LOSS_MASK_INVALID,
                "Qwen forward 未返回 logits",
            )
        return ForwardResult(
            loss=assistant_sample_mean_causal_loss(logits, labels),
            logits=logits,
        )

    @torch.no_grad()
    def generate_text(
        self,
        batch: Mapping[str, Any],
        *,
        processor: Any,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        logits_processor: Any | None = None,
    ) -> list[str]:
        self.eval()
        tensors = model_tensor_batch(batch)
        tensors.pop("labels", None)
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_arguments.update(
                temperature=temperature,
                top_p=top_p,
            )
        if logits_processor is not None:
            generation_arguments["logits_processor"] = logits_processor
        generated = self.model.generate(**tensors, **generation_arguments)
        input_length = tensors["input_ids"].shape[1]
        continuation = generated[:, input_length:]
        values = processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [str(value).strip() for value in values]

    def trainable_state_dict(self) -> dict[str, Tensor]:
        state = self.model.state_dict()
        missing = set(self._trainable_names) - set(state)
        if missing:
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                f"trainable parameters 不在 state_dict：{sorted(missing)}",
            )
        return {
            name: state[name].detach().cpu().contiguous()
            for name in self._trainable_names
        }

    def load_trainable_state_dict(
        self,
        state: Mapping[str, Tensor],
    ) -> None:
        if set(state) != set(self._trainable_names):
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "adapter state 参数名与当前模型不一致",
                details={
                    "missing": sorted(set(self._trainable_names) - set(state)),
                    "unexpected": sorted(set(state) - set(self._trainable_names)),
                },
            )
        current = self.model.state_dict()
        for name, value in state.items():
            if current[name].shape != value.shape:
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    f"{name}: adapter tensor shape 不一致",
                )
        incompatible = self.model.load_state_dict(dict(state), strict=False)
        unexpected = set(incompatible.unexpected_keys)
        missing_trainable = set(self._trainable_names) & set(
            incompatible.missing_keys
        )
        if unexpected or missing_trainable:
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "adapter state 加载不完整",
                details={
                    "unexpected": sorted(unexpected),
                    "missing_trainable": sorted(missing_trainable),
                },
            )
