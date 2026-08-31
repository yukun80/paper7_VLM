"""已发布 Grounded Adapter 的只读推理配置与加载器。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    portable_relative_path,
    read_json,
    require_exact_keys,
    require_mapping,
    resolve_config_path,
)

from .checkpoint import CheckpointManager
from .backends import (
    VLMModelAdapter,
    VLMProcessorAdapter,
    build_model_adapter,
    build_processor_adapter,
)
from .config import VLMConfig, _load_yaml, load_config
from .errors import ConfigError, ContractError, ReasonCode


GROUNDED_RUNTIME_CONFIG_SCHEMA = "oa_groundrag.grounded_runtime.config.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GroundedRuntimeBinding(Protocol):
    config_path: Path
    best_pointer_sha256: str
    checkpoint_manifest_sha256: str
    adapter_sha256: str
    workflow_state_sha256: str


@dataclass(frozen=True)
class GroundedRuntimeConfig:
    schema_version: str
    base: VLMConfig
    workflow_root: Path
    published_training_config_semantic_sha256: str
    max_new_tokens: int
    config_path: Path
    semantic_sha256: str

    @property
    def training_root(self) -> Path:
        return self.workflow_root / "training"

    @property
    def model(self):
        return self.base.model

    @property
    def adaptation(self):
        return self.base.adaptation

    @property
    def limits(self):
        return replace(
            self.base.limits,
            max_input_tokens=4096,
            max_new_tokens=self.max_new_tokens,
        )

    @property
    def generation(self):
        return replace(self.base.generation, max_new_tokens=self.max_new_tokens)


@dataclass(frozen=True)
class GroundedRuntimeReference:
    config: GroundedRuntimeConfig
    pointer: Mapping[str, Any]
    checkpoint: Path


@dataclass(frozen=True)
class GroundedRuntimeBundle:
    config: GroundedRuntimeConfig
    processor: VLMProcessorAdapter
    model: VLMModelAdapter
    checkpoint: Path
    identity: Mapping[str, Any]


def _sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是小写 SHA-256")
    return value


def _regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> None:
    if (
        first_symlink_component(path) is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        raise ContractError(ReasonCode.OUTPUT_LINK, f"{label} 必须是普通单链接文件：{path}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ContractError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            f"{label} SHA-256 漂移：{path}",
        )


def load_grounded_runtime_config(path: Path | str) -> GroundedRuntimeConfig:
    """读取不含 Benchmark、compact 或 Eval-dev 绑定的推理配置。"""

    config_path = Path(os.path.abspath(Path(path)))
    if (
        first_symlink_component(config_path) is not None
        or not config_path.is_file()
        or config_path.is_symlink()
        or config_path.stat().st_nlink != 1
    ):
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"Grounded runtime 配置必须是普通文件：{config_path}")
    row = _load_yaml(config_path)
    require_exact_keys(
        row,
        required=(
            "schema_version",
            "base_config",
            "workflow_root",
            "published_training_config_semantic_sha256",
            "generation",
        ),
        location="$",
    )
    if row["schema_version"] != GROUNDED_RUNTIME_CONFIG_SCHEMA:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 {GROUNDED_RUNTIME_CONFIG_SCHEMA}",
        )
    base_path = resolve_config_path(
        config_path.parent,
        row["base_config"],
        location="$.base_config",
    )
    workflow_root = resolve_config_path(
        config_path.parent,
        row["workflow_root"],
        location="$.workflow_root",
    )
    generation = require_mapping(row["generation"], location="$.generation")
    require_exact_keys(generation, required=("max_new_tokens",), location="$.generation")
    max_new_tokens = generation["max_new_tokens"]
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens != 768:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "$.generation.max_new_tokens 必须保持发布值 768",
        )
    base = load_config(base_path)
    if (
        base.adaptation.strategy != "lora"
        or not base.adaptation.freeze_vision
        or not base.adaptation.freeze_merger
    ):
        raise ConfigError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Grounded runtime 只接受冻结 vision/merger 的 LoRA 语义主体",
        )
    training_sha = _sha256(
        row["published_training_config_semantic_sha256"],
        location="$.published_training_config_semantic_sha256",
    )
    semantic_payload = {
        "schema_version": GROUNDED_RUNTIME_CONFIG_SCHEMA,
        "base_config_semantic_sha256": base.semantic_sha256,
        "workflow_root": str(workflow_root),
        "published_training_config_semantic_sha256": training_sha,
        "generation": {"max_new_tokens": max_new_tokens},
    }
    return GroundedRuntimeConfig(
        schema_version=GROUNDED_RUNTIME_CONFIG_SCHEMA,
        base=base,
        workflow_root=workflow_root,
        published_training_config_semantic_sha256=training_sha,
        max_new_tokens=max_new_tokens,
        config_path=config_path,
        semantic_sha256=sha256_text(canonical_json(semantic_payload)),
    )


def resolve_grounded_runtime_checkpoint(
    config: GroundedRuntimeConfig,
    binding: GroundedRuntimeBinding,
) -> GroundedRuntimeReference:
    """按发布 SHA 验证 workflow、best pointer、manifest 与 Adapter。"""

    state_path = config.workflow_root / "workflow_state.json"
    _regular_file(
        state_path,
        label="Grounded workflow state",
        expected_sha256=binding.workflow_state_sha256,
    )
    state = read_json(state_path)
    if (
        state.get("stage") != "complete"
        or state.get("config_semantic_sha256")
        != config.published_training_config_semantic_sha256
        or state.get("sealed_test_evaluated") is not False
        or state.get("formal_acceptance") is not False
        or state.get("scientific_acceptance") is not False
    ):
        raise ContractError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Grounded workflow state 与发布训练身份或科学边界不兼容",
        )
    pointer_path = config.training_root / "best_checkpoint.json"
    _regular_file(
        pointer_path,
        label="Grounded best pointer",
        expected_sha256=binding.best_pointer_sha256,
    )
    pointer = read_json(pointer_path)
    expected_pointer = {
        "schema_version",
        "selection_metric",
        "step",
        "macro_task_loss",
        "overall_loss",
        "checkpoint",
        "formal_acceptance",
    }
    if (
        not isinstance(pointer, dict)
        or set(pointer) != expected_pointer
        or pointer.get("selection_metric") != "region_monitor_loss"
        or pointer.get("formal_acceptance") is not False
        or isinstance(pointer.get("step"), bool)
        or not isinstance(pointer.get("step"), int)
        or pointer["step"] <= 0
    ):
        raise ContractError(ReasonCode.CHECKPOINT_CORRUPT, "Grounded best pointer 合同非法")
    relative = portable_relative_path(
        str(pointer["checkpoint"]),
        location="best_checkpoint.checkpoint",
    )
    checkpoint = config.training_root.joinpath(*relative.parts)
    try:
        checkpoint.resolve().relative_to(config.training_root.resolve())
    except ValueError as error:
        raise ContractError(ReasonCode.PATH_ESCAPE, "Grounded best checkpoint 路径逃逸") from error
    if (
        first_symlink_component(checkpoint) is not None
        or checkpoint.is_symlink()
        or not checkpoint.is_dir()
    ):
        raise ContractError(ReasonCode.OUTPUT_LINK, "Grounded best checkpoint 不是普通目录")
    manifest_path = checkpoint / "manifest.json"
    adapter_path = checkpoint / "adapter" / "adapter_model.safetensors"
    _regular_file(
        manifest_path,
        label="Grounded checkpoint manifest",
        expected_sha256=binding.checkpoint_manifest_sha256,
    )
    _regular_file(
        adapter_path,
        label="Grounded Adapter",
        expected_sha256=binding.adapter_sha256,
    )
    manifest = read_json(manifest_path)
    if (
        manifest.get("cursor", {}).get("global_step") != pointer["step"]
        or manifest.get("config_semantic_sha256")
        != config.published_training_config_semantic_sha256
    ):
        raise ContractError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "Grounded pointer、checkpoint step 或训练配置身份不一致",
        )
    return GroundedRuntimeReference(config=config, pointer=pointer, checkpoint=checkpoint)


def load_grounded_runtime_generator(
    config: GroundedRuntimeConfig,
    binding: GroundedRuntimeBinding,
    *,
    device: Any,
) -> GroundedRuntimeBundle:
    """只加载已发布 Grounded LoRA；不读取任何训练或评价数据。"""

    reference = resolve_grounded_runtime_checkpoint(config, binding)
    processor_config = replace(config.base, limits=config.limits)
    processor = build_processor_adapter(processor_config)
    model = build_model_adapter(
        config.base,
        device=device,
        gradient_checkpointing=False,
    )
    manifest, trainable = CheckpointManager().load_trainable_for_inference(
        reference.checkpoint,
        expected_manifest_sha256=binding.checkpoint_manifest_sha256,
        expected_adapter_sha256=binding.adapter_sha256,
        expected_config_semantic_sha256=(
            config.published_training_config_semantic_sha256
        ),
        expected_model_identity=model.identity.to_dict(),
        expected_processor_identity=processor.identity(),
        expected_trainable_names=model.trainable_names,
        expected_trainable_parameter_count=model.trainable_parameter_count,
    )
    model.load_trainable_state_dict(trainable)
    identity = {
        "grounded_runtime_config_semantic_sha256": config.semantic_sha256,
        "published_training_config_semantic_sha256": (
            config.published_training_config_semantic_sha256
        ),
        "best_pointer_sha256": binding.best_pointer_sha256,
        "best_step": reference.pointer["step"],
        "checkpoint": str(reference.checkpoint.resolve()),
        "checkpoint_manifest_sha256": binding.checkpoint_manifest_sha256,
        "adapter_sha256": binding.adapter_sha256,
        "model_identity": model.identity.to_dict(),
        "processor_identity": processor.identity(),
        "trainable_parameter_count": manifest["trainable_parameter_count"],
        "loaded_components": ["trainable_lora_state"],
        "excluded_components": [
            "benchmark_payload",
            "compact_training",
            "evaluation_selection",
            "optimizer",
            "scheduler",
            "rng",
            "sampler",
        ],
    }
    return GroundedRuntimeBundle(
        config=config,
        processor=processor,
        model=model,
        checkpoint=reference.checkpoint,
        identity=identity,
    )
