"""Qwen3.5-4B 固定 full-attention LoRA 模型后端。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.vlm.errors import ModelError, ReasonCode

from ..assets import VerifiedModelAssets
from ..modeling import ForwardResult, assistant_sample_mean_causal_loss
from ...config import AdaptationSection, ModelSection
from ...processing import model_tensor_batch


EXPECTED_QWEN35_4B_CHECKPOINT_PARAMETERS = 4_659_865_088
EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS = 4_539_265_536
EXPECTED_QWEN35_UNUSED_MTP_PARAMETERS = 120_599_552
QWEN35_FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
QWEN35_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPECTED_QWEN35_FULL_ATTENTION_LORA_R8_PARAMETERS = 1_572_864
EXPECTED_QWEN35_LORA_TENSORS = 64

QWEN35_UNUSED_MTP_TENSORS = frozenset(
    {
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
)

_LORA_NAME = re.compile(
    r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])\.default\.weight$"
)


@dataclass(frozen=True)
class Qwen35ModelIdentity:
    backend: str
    model_path: str
    hub_repo_id: str
    hub_revision: str
    asset_ledger_path: str
    asset_ledger_file_sha256: str
    asset_ledger_payload_sha256: str
    model_type: str
    architecture: str
    config_sha256: str
    weights_index_sha256: str
    checkpoint_parameter_count: int
    unused_mtp_parameter_count: int
    base_parameter_count: int
    full_attention_layers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "hub_repo_id": self.hub_repo_id,
            "hub_revision": self.hub_revision,
            "asset_ledger_path": self.asset_ledger_path,
            "asset_ledger_file_sha256": self.asset_ledger_file_sha256,
            "asset_ledger_payload_sha256": self.asset_ledger_payload_sha256,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "config_sha256": self.config_sha256,
            "weights_index_sha256": self.weights_index_sha256,
            "checkpoint_parameter_count": self.checkpoint_parameter_count,
            "unused_mtp_parameter_count": self.unused_mtp_parameter_count,
            "base_parameter_count": self.base_parameter_count,
            "full_attention_layers": list(self.full_attention_layers),
        }


def _official_config_contract(root: Path) -> None:
    try:
        value = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "无法读取 Qwen3.5 config.json",
            details={"error": str(error)},
        ) from error
    text = value.get("text_config") if isinstance(value, dict) else None
    vision = value.get("vision_config") if isinstance(value, dict) else None
    expected_layer_types = tuple(
        "full_attention" if index in QWEN35_FULL_ATTENTION_LAYERS
        else "linear_attention"
        for index in range(32)
    )
    valid = (
        isinstance(value, dict)
        and value.get("model_type") == "qwen3_5"
        and value.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
        and isinstance(text, dict)
        and text.get("model_type") == "qwen3_5_text"
        and text.get("num_hidden_layers") == 32
        and text.get("full_attention_interval") == 4
        and tuple(text.get("layer_types", ())) == expected_layer_types
        and text.get("hidden_size") == 2560
        and text.get("num_attention_heads") == 16
        and text.get("num_key_value_heads") == 4
        and text.get("head_dim") == 256
        and text.get("dtype") == "bfloat16"
        and isinstance(vision, dict)
        and vision.get("model_type") == "qwen3_5"
        and vision.get("depth") == 24
        and vision.get("out_hidden_size") == 2560
    )
    if not valid:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5-4B 混合拓扑与冻结设计不一致",
        )


def _identity(
    model_config: ModelSection,
    assets: VerifiedModelAssets,
    *,
    base_parameter_count: int,
) -> Qwen35ModelIdentity:
    return Qwen35ModelIdentity(
        backend="qwen3_5",
        model_path=str(assets.model_root),
        hub_repo_id=assets.repo_id,
        hub_revision=assets.revision,
        asset_ledger_path=str(assets.ledger_path),
        asset_ledger_file_sha256=assets.ledger_file_sha256,
        asset_ledger_payload_sha256=assets.ledger_payload_sha256,
        model_type="qwen3_5",
        architecture="Qwen3_5ForConditionalGeneration",
        config_sha256=sha256_file(model_config.path / "config.json"),
        weights_index_sha256=sha256_file(
            model_config.path / "model.safetensors.index.json"
        ),
        checkpoint_parameter_count=EXPECTED_QWEN35_4B_CHECKPOINT_PARAMETERS,
        unused_mtp_parameter_count=EXPECTED_QWEN35_UNUSED_MTP_PARAMETERS,
        base_parameter_count=base_parameter_count,
        full_attention_layers=QWEN35_FULL_ATTENTION_LAYERS,
    )


def local_model_identity(
    model_config: ModelSection,
    assets: VerifiedModelAssets,
) -> Qwen35ModelIdentity:
    """不实例化权重地复算固定 Qwen3.5 模型身份。"""

    if (
        assets.backend != "qwen3_5"
        or assets.repo_id != model_config.hub_repo_id
        or assets.revision != model_config.hub_revision
        or assets.model_root.resolve() != model_config.path.resolve()
        or (
            model_config.asset_ledger_path is None
            or assets.ledger_path.resolve()
            != model_config.asset_ledger_path.resolve()
        )
    ):
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5 config 与已验证模型资产身份不一致",
        )
    _official_config_contract(model_config.path)
    return _identity(
        model_config,
        assets,
        base_parameter_count=EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS,
    )


def _audit_checkpoint_runtime_mapping(model: nn.Module, root: Path) -> None:
    """锁定官方 checkpoint 的 MTP 额外权重与 AutoModel 绑权视图。"""

    try:
        from safetensors import safe_open

        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict):
            raise TypeError("weight_map must be an object")
        disk_names = set(weight_map)
        runtime_names = set(model.state_dict())
        missing = disk_names - runtime_names
        extra = runtime_names - disk_names
        checkpoint_count = 0
        missing_count = 0
        for shard in sorted(set(weight_map.values())):
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    count = math.prod(handle.get_slice(name).get_shape())
                    checkpoint_count += count
                    if name in missing:
                        missing_count += count
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "无法审计 Qwen3.5 checkpoint/runtime tensor 映射",
            details={"error": str(error)},
        ) from error
    if (
        missing != QWEN35_UNUSED_MTP_TENSORS
        or extra != {"lm_head.weight"}
        or checkpoint_count != EXPECTED_QWEN35_4B_CHECKPOINT_PARAMETERS
        or missing_count != EXPECTED_QWEN35_UNUSED_MTP_PARAMETERS
    ):
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5 checkpoint/runtime tensor 映射不符合官方固定 revision",
            details={
                "missing": sorted(missing),
                "extra": sorted(extra),
                "checkpoint_parameter_count": checkpoint_count,
                "unused_mtp_parameter_count": missing_count,
            },
        )


def _locked_adaptation(adaptation: AdaptationSection) -> None:
    if not adaptation.freeze_vision or not adaptation.freeze_merger:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5 必须冻结 vision 与 merger",
        )
    if adaptation.strategy == "prompt_only":
        valid = (
            not adaptation.target_modules
            and adaptation.rank == 0
            and adaptation.alpha == 0
            and adaptation.dropout == 0
        )
    elif adaptation.strategy == "lora":
        valid = (
            adaptation.target_modules == QWEN35_LORA_TARGET_MODULES
            and adaptation.rank == 8
            and adaptation.alpha == 16
            and adaptation.dropout == 0.05
        )
    else:
        raise ModelError(
            ReasonCode.INVALID_ENUM,
            f"不支持 adaptation={adaptation.strategy}",
        )
    if not valid:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5 adaptation 参数与冻结设计不一致",
        )


class Qwen35ModelAdapter:
    """严格锁定 Qwen3.5-4B 的模型身份、冻结范围和 LoRA 拓扑。"""

    def __init__(
        self,
        model: nn.Module,
        *,
        identity: Qwen35ModelIdentity,
        adaptation: AdaptationSection,
    ) -> None:
        _locked_adaptation(adaptation)
        if (
            identity.backend != "qwen3_5"
            or identity.model_type != "qwen3_5"
            or identity.architecture != "Qwen3_5ForConditionalGeneration"
            or identity.full_attention_layers != QWEN35_FULL_ATTENTION_LAYERS
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5 adapter 拒绝跨家族模型身份",
            )
        self.model = model
        self.identity = identity
        self.adaptation = adaptation
        self._trainable_names = tuple(
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        self._validate_trainable_topology()

    @classmethod
    def load(
        cls,
        model_config: ModelSection,
        adaptation: AdaptationSection,
        *,
        assets: VerifiedModelAssets,
        device: torch.device,
        gradient_checkpointing: bool,
    ) -> "Qwen35ModelAdapter":
        _locked_adaptation(adaptation)
        if (
            model_config.backend != "qwen3_5"
            or model_config.path.resolve() != assets.model_root
            or model_config.hub_repo_id != assets.repo_id
            or model_config.hub_revision != assets.revision
            or model_config.dtype != "bfloat16"
            or model_config.attn_implementation != "sdpa"
            or not model_config.local_files_only
            or model_config.trust_remote_code
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5 model config 与已验证资产/运行合同不一致",
            )
        _official_config_contract(assets.model_root)
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as error:
            raise ModelError(
                ReasonCode.ASSET_MISSING,
                "当前 transformers 缺少 AutoModelForMultimodalLM",
            ) from error
        load_arguments: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            load_arguments["device_map"] = {"": device}
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_config.path,
                **load_arguments,
            )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5-4B 权重加载失败",
                details={"error": str(error)},
            ) from error
        model_type = str(getattr(model.config, "model_type", "unknown"))
        architectures = tuple(getattr(model.config, "architectures", ()) or ())
        _audit_checkpoint_runtime_mapping(model, assets.model_root)
        base_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if (
            model_type != "qwen3_5"
            or architectures != ("Qwen3_5ForConditionalGeneration",)
            or base_parameter_count
            != EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "真实 Qwen3.5 模型身份或参数量不符",
                details={
                    "model_type": model_type,
                    "architectures": list(architectures),
                    "expected_parameters": (
                        EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS
                    ),
                    "actual_parameters": base_parameter_count,
                },
            )
        identity = _identity(
            model_config,
            assets,
            base_parameter_count=base_parameter_count,
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
                    "Qwen3.5 LoRA 需要本地 peft",
                ) from error
            model = get_peft_model(
                model,
                LoraConfig(
                    r=adaptation.rank,
                    lora_alpha=adaptation.alpha,
                    lora_dropout=adaptation.dropout,
                    target_modules=list(adaptation.target_modules),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        model.to(device)
        return cls(model, identity=identity, adaptation=adaptation)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return self._trainable_names

    def _validate_trainable_topology(self) -> None:
        if any(
            any(token in name.lower() for token in ("visual", "vision", "merger"))
            for name in self._trainable_names
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "vision/merger 出现可训练参数",
            )
        if self.adaptation.strategy == "prompt_only":
            if self._trainable_names or self.trainable_parameter_count != 0:
                raise ModelError(
                    ReasonCode.MODEL_IDENTITY_MISMATCH,
                    "Qwen3.5 prompt-only 必须有 0 个训练参数",
                )
            return
        observed: set[tuple[int, str, str]] = set()
        malformed: list[str] = []
        for name in self._trainable_names:
            matched = _LORA_NAME.search(name)
            if matched is None:
                malformed.append(name)
                continue
            observed.add((int(matched.group(1)), matched.group(2), matched.group(3)))
        expected = {
            (layer, module, branch)
            for layer in QWEN35_FULL_ATTENTION_LAYERS
            for module in QWEN35_LORA_TARGET_MODULES
            for branch in ("A", "B")
        }
        if (
            malformed
            or observed != expected
            or len(self._trainable_names) != EXPECTED_QWEN35_LORA_TENSORS
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5 LoRA 未精确落在 8 个 full-attention 层",
                details={
                    "malformed": malformed,
                    "missing": sorted(expected - observed),
                    "unexpected": sorted(observed - expected),
                },
            )
        if (
            self.identity.base_parameter_count
            == EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS
            and self.trainable_parameter_count
            != EXPECTED_QWEN35_FULL_ATTENTION_LORA_R8_PARAMETERS
        ):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5 full-attention LoRA 参数量不符合锁定设计",
                details={
                    "expected": EXPECTED_QWEN35_FULL_ATTENTION_LORA_R8_PARAMETERS,
                    "actual": self.trainable_parameter_count,
                },
            )

    def train(self) -> None:
        self.model.train()
        self._force_frozen_vision_eval()

    def eval(self) -> None:
        self.model.eval()

    def _force_frozen_vision_eval(self) -> None:
        for name, module in self.model.named_modules():
            if any(token in name.lower() for token in ("visual", "vision", "merger")):
                module.eval()

    def forward(self, batch: Mapping[str, Any]) -> ForwardResult:
        tensor_batch = model_tensor_batch(batch)
        labels = tensor_batch.pop("labels", None)
        if not isinstance(labels, Tensor):
            raise ModelError(
                ReasonCode.LOSS_MASK_INVALID,
                "Qwen3.5 训练 forward 缺少 labels",
            )
        output = self.model(**tensor_batch, use_cache=False)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise ModelError(
                ReasonCode.LOSS_MASK_INVALID,
                "Qwen3.5 forward 未返回 logits",
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
        arguments: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            arguments.update(temperature=temperature, top_p=top_p)
        if logits_processor is not None:
            arguments["logits_processor"] = logits_processor
        generated = self.model.generate(**tensors, **arguments)
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
        if set(self._trainable_names) - set(state):
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "Qwen3.5 trainable 参数不在 state_dict",
            )
        return {
            name: state[name].detach().cpu().contiguous()
            for name in self._trainable_names
        }

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != set(self._trainable_names):
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "Qwen3.5 Adapter state 与当前家族/拓扑不兼容",
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
                    f"{name}: Qwen3.5 Adapter tensor shape 不一致",
                )
        incompatible = self.model.load_state_dict(dict(state), strict=False)
        if incompatible.unexpected_keys or (
            set(self._trainable_names) & set(incompatible.missing_keys)
        ):
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "Qwen3.5 Adapter state 加载不完整",
            )
