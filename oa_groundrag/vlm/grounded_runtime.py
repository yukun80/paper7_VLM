"""已发布 Grounded Adapter 的只读推理配置与加载器。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import (
    PortablePathError,
    bundle_relative_reference,
    first_symlink_component,
    resolve_config_reference,
    resolve_bundle_reference,
    validate_bundle_root,
)
from oa_groundrag.data.rs_general.io import (
    portable_relative_path,
    require_bool,
    read_json,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
)

from .checkpoint import CheckpointManager
from .backends import (
    VLMModelAdapter,
    VLMProcessorAdapter,
    build_model_adapter,
    build_processor_adapter,
)
from .config import AdaptationSection, GenerationSection, ModelSection, _load_yaml
from .errors import ConfigError, ContractError, ReasonCode


GROUNDED_RUNTIME_CONFIG_SCHEMA = "oa_groundrag.grounded_runtime.config.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GroundedRuntimeBinding(Protocol):
    config_path: Path
    best_pointer_sha256: str
    checkpoint_manifest_sha256: str
    adapter_sha256: str
    workflow_state_sha256: str


@dataclass(frozen=True)
class GroundedRuntimeLimits:
    max_images: int
    max_input_tokens: int
    min_pixels: int
    max_pixels: int
    max_new_tokens: int


@dataclass(frozen=True)
class GroundedRuntimeConfig:
    schema_version: str
    bundle_root: Path
    workflow_root: Path
    model: ModelSection
    adaptation: AdaptationSection
    limits: GroundedRuntimeLimits
    generation: GenerationSection
    published_training_config_semantic_sha256: str
    config_path: Path
    semantic_sha256: str

    @property
    def training_root(self) -> Path:
        return self.workflow_root / "training"


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


def _portable_runtime_identity(
    identity: Mapping[str, Any],
    *,
    bundle_root: Path,
    path_key: str,
) -> dict[str, Any]:
    """将现场模型身份中的路径转换为可迁移的包根相对引用。"""

    value = identity.get(path_key)
    if not isinstance(value, str):
        raise ContractError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            f"模型身份缺少字符串字段 {path_key}",
        )
    try:
        reference = bundle_relative_reference(bundle_root, Path(value))
    except PortablePathError as error:
        raise ContractError(
            ReasonCode.PATH_ESCAPE,
            f"模型身份不在可迁移包内：{value}",
        ) from error
    return {**dict(identity), path_key: reference}


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
            "bundle_root",
            "workflow_root",
            "published_training_config_semantic_sha256",
            "model",
            "adaptation",
            "limits",
            "generation",
        ),
        location="$",
    )
    if row["schema_version"] != GROUNDED_RUNTIME_CONFIG_SCHEMA:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 {GROUNDED_RUNTIME_CONFIG_SCHEMA}",
        )
    try:
        bundle_root = validate_bundle_root(
            resolve_config_reference(
                config_path,
                require_string(row["bundle_root"], location="$.bundle_root"),
            )
        )
        workflow_root = resolve_bundle_reference(
            bundle_root,
            require_string(row["workflow_root"], location="$.workflow_root"),
            expected="directory",
        )
    except PortablePathError as error:
        raise ConfigError(
            ReasonCode.PATH_ESCAPE,
            f"Grounded runtime bundle 路径非法：{error}",
        ) from error
    model_row = require_mapping(row["model"], location="$.model")
    require_exact_keys(
        model_row,
        required=(
            "backend",
            "path",
            "processor_path",
            "local_files_only",
            "dtype",
            "attn_implementation",
            "trust_remote_code",
        ),
        location="$.model",
    )
    try:
        model_path = resolve_bundle_reference(
            bundle_root,
            require_string(model_row["path"], location="$.model.path"),
            expected="directory",
        )
        processor_path = resolve_bundle_reference(
            bundle_root,
            require_string(
                model_row["processor_path"],
                location="$.model.processor_path",
            ),
            expected="directory",
        )
    except PortablePathError as error:
        raise ConfigError(
            ReasonCode.PATH_ESCAPE,
            f"Grounded runtime model 路径非法：{error}",
        ) from error
    model = ModelSection(
        backend=require_string(model_row["backend"], location="$.model.backend"),
        path=model_path,
        processor_path=processor_path,
        local_files_only=require_bool(
            model_row["local_files_only"], location="$.model.local_files_only"
        ),
        dtype=require_string(model_row["dtype"], location="$.model.dtype"),
        attn_implementation=require_string(
            model_row["attn_implementation"],
            location="$.model.attn_implementation",
        ),
        trust_remote_code=require_bool(
            model_row["trust_remote_code"], location="$.model.trust_remote_code"
        ),
    )
    if (
        model.backend != "qwen3_vl"
        or not model.local_files_only
        or model.dtype != "bfloat16"
        or model.attn_implementation != "sdpa"
        or model.trust_remote_code
        or model.path != model.processor_path
    ):
        raise ConfigError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Grounded runtime 仅接受已发布的本地 Qwen3-VL-2B 模型合同",
        )
    adaptation_row = require_mapping(row["adaptation"], location="$.adaptation")
    require_exact_keys(
        adaptation_row,
        required=(
            "strategy",
            "target_modules",
            "rank",
            "alpha",
            "dropout",
            "freeze_vision",
            "freeze_merger",
        ),
        location="$.adaptation",
    )
    target_modules = adaptation_row["target_modules"]
    if not isinstance(target_modules, list) or not all(
        isinstance(value, str) and value for value in target_modules
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "$.adaptation.target_modules 必须是非空字符串列表",
        )
    dropout = adaptation_row["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "$.adaptation.dropout 必须是数值")
    adaptation = AdaptationSection(
        strategy=require_string(
            adaptation_row["strategy"], location="$.adaptation.strategy"
        ),
        target_modules=tuple(target_modules),
        rank=require_int(adaptation_row["rank"], location="$.adaptation.rank", minimum=1),
        alpha=require_int(
            adaptation_row["alpha"], location="$.adaptation.alpha", minimum=1
        ),
        dropout=float(dropout),
        freeze_vision=require_bool(
            adaptation_row["freeze_vision"], location="$.adaptation.freeze_vision"
        ),
        freeze_merger=require_bool(
            adaptation_row["freeze_merger"], location="$.adaptation.freeze_merger"
        ),
    )
    if (
        adaptation.strategy != "lora"
        or adaptation.target_modules != ("q_proj", "k_proj", "v_proj", "o_proj")
        or adaptation.rank != 8
        or adaptation.alpha != 16
        or adaptation.dropout != 0.05
        or not adaptation.freeze_vision
        or not adaptation.freeze_merger
    ):
        raise ConfigError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Grounded runtime LoRA 合同与已发布 Adapter 不兼容",
        )
    limits_row = require_mapping(row["limits"], location="$.limits")
    limit_fields = (
        "max_images",
        "max_input_tokens",
        "min_pixels",
        "max_pixels",
        "max_new_tokens",
    )
    require_exact_keys(limits_row, required=limit_fields, location="$.limits")
    limits = GroundedRuntimeLimits(
        **{
            name: require_int(limits_row[name], location=f"$.limits.{name}", minimum=1)
            for name in limit_fields
        }
    )
    if limits != GroundedRuntimeLimits(5, 4096, 12544, 200704, 768):
        raise ConfigError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Grounded runtime 输入与生成上限必须保持发布值",
        )
    generation = require_mapping(row["generation"], location="$.generation")
    require_exact_keys(
        generation,
        required=("max_new_tokens", "do_sample", "temperature", "top_p"),
        location="$.generation",
    )
    temperature = generation["temperature"]
    top_p = generation["top_p"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (temperature, top_p)
    ):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "Grounded generation 数值非法")
    generation_section = GenerationSection(
        max_new_tokens=require_int(
            generation["max_new_tokens"],
            location="$.generation.max_new_tokens",
            minimum=1,
        ),
        do_sample=require_bool(
            generation["do_sample"], location="$.generation.do_sample"
        ),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    if generation_section != GenerationSection(768, False, 0.0, 1.0):
        raise ConfigError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Grounded generation 必须保持发布值",
        )
    training_sha = _sha256(
        row["published_training_config_semantic_sha256"],
        location="$.published_training_config_semantic_sha256",
    )
    semantic_payload = {
        "schema_version": GROUNDED_RUNTIME_CONFIG_SCHEMA,
        "workflow_root": require_string(
            row["workflow_root"], location="$.workflow_root"
        ),
        "model": {
            **model_row,
            "path": require_string(model_row["path"], location="$.model.path"),
            "processor_path": require_string(
                model_row["processor_path"], location="$.model.processor_path"
            ),
        },
        "adaptation": adaptation_row,
        "limits": limits_row,
        "published_training_config_semantic_sha256": training_sha,
        "generation": generation,
    }
    return GroundedRuntimeConfig(
        schema_version=GROUNDED_RUNTIME_CONFIG_SCHEMA,
        bundle_root=bundle_root,
        workflow_root=workflow_root,
        model=model,
        adaptation=adaptation,
        limits=limits,
        generation=generation_section,
        published_training_config_semantic_sha256=training_sha,
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
    processor = build_processor_adapter(config)
    model = build_model_adapter(
        config,
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
        expected_model_identity=_portable_runtime_identity(
            model.identity.to_dict(),
            bundle_root=config.bundle_root,
            path_key="model_path",
        ),
        expected_processor_identity=_portable_runtime_identity(
            processor.identity(),
            bundle_root=config.bundle_root,
            path_key="processor_path",
        ),
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
