"""严格 YAML 配置；算法只消费类型化合同。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component

from oa_groundrag.grounding.contracts import (
    CONFIG_SCHEMA_VERSION,
    DataMode,
    MaskMode,
)
from .errors import ConfigError, ReasonCode
CONFIG_SCHEMA_VERSION_V3 = "rs_vlm.config.v3"
SUPPORTED_VLM_BACKENDS = frozenset({"qwen3_vl", "qwen3_5"})


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML key 必须可哈希",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"重复 YAML key：{key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RunSection:
    name: str
    seed: int
    mode: DataMode
    mask_mode: MaskMode
    output_root: Path
    resume_checkpoint: Path | None


@dataclass(frozen=True)
class DataSection:
    benchmark_root: Path
    roles: tuple[str, ...]
    task_families: tuple[str, ...]
    expected_manifest_schema: str
    expected_canonical_schema: str
    expected_manifest_sha256: str
    expected_validation_sha256: str
    expected_build_id: str
    expected_payload_sha256: str
    expected_hash_manifest_sha256: str
    parent_balanced: bool
    source_weights: Mapping[str, float]
    task_weights: Mapping[str, float]


@dataclass(frozen=True)
class LimitsSection:
    max_parents: int
    max_records: int
    max_assets: int
    max_asset_bytes: int
    max_copied_asset_bytes: int
    max_images: int
    max_input_tokens: int
    max_total_tokens: int
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    max_probe_shards: int
    max_records_per_shard: int
    max_probe_records: int


@dataclass(frozen=True)
class ModelSection:
    path: Path
    processor_path: Path
    local_files_only: bool
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    backend: str = "qwen3_vl"
    hub_repo_id: str | None = None
    hub_revision: str | None = None
    asset_ledger_path: Path | None = None


@dataclass(frozen=True)
class AdaptationSection:
    strategy: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    freeze_vision: bool
    freeze_merger: bool


@dataclass(frozen=True)
class TrainingSection:
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    scheduler: str
    max_grad_norm: float
    max_steps: int
    epochs: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    checkpoint_interval: int
    log_interval: int
    validation_interval: int
    validation_max_parents: int


@dataclass(frozen=True)
class GenerationSection:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


@dataclass(frozen=True)
class VLMConfig:
    schema_version: str
    run: RunSection
    data: DataSection
    limits: LimitsSection
    model: ModelSection
    adaptation: AdaptationSection
    training: TrainingSection
    generation: GenerationSection
    config_path: Path
    semantic_sha256: str

    def semantic_dict(self) -> dict[str, Any]:
        """排除输出位置与 resume 指针的可恢复语义快照。"""

        value = {
            "schema_version": self.schema_version,
            "run": {
                "seed": self.run.seed,
                "mode": self.run.mode.value,
                "mask_mode": self.run.mask_mode.value,
            },
            "data": {
                "benchmark_root": str(self.data.benchmark_root),
                "roles": list(self.data.roles),
                "task_families": list(self.data.task_families),
                "expected_manifest_schema": self.data.expected_manifest_schema,
                "expected_canonical_schema": self.data.expected_canonical_schema,
                "expected_manifest_sha256": self.data.expected_manifest_sha256,
                "expected_validation_sha256": (
                    self.data.expected_validation_sha256
                ),
                "expected_build_id": self.data.expected_build_id,
                "expected_payload_sha256": self.data.expected_payload_sha256,
                "expected_hash_manifest_sha256": (
                    self.data.expected_hash_manifest_sha256
                ),
                "parent_balanced": self.data.parent_balanced,
                "source_weights": dict(sorted(self.data.source_weights.items())),
                "task_weights": dict(sorted(self.data.task_weights.items())),
            },
            "limits": {
                name: getattr(self.limits, name)
                for name in self.limits.__dataclass_fields__
            },
            "model": {
                "path": str(self.model.path),
                "processor_path": str(self.model.processor_path),
                "local_files_only": self.model.local_files_only,
                "dtype": self.model.dtype,
                "attn_implementation": self.model.attn_implementation,
                "trust_remote_code": self.model.trust_remote_code,
            },
            "adaptation": {
                "strategy": self.adaptation.strategy,
                "target_modules": list(self.adaptation.target_modules),
                "rank": self.adaptation.rank,
                "alpha": self.adaptation.alpha,
                "dropout": self.adaptation.dropout,
                "freeze_vision": self.adaptation.freeze_vision,
                "freeze_merger": self.adaptation.freeze_merger,
            },
            "training": {
                name: getattr(self.training, name)
                for name in self.training.__dataclass_fields__
            },
            "generation": {
                name: getattr(self.generation, name)
                for name in self.generation.__dataclass_fields__
            },
        }

        if self.schema_version == CONFIG_SCHEMA_VERSION_V3:
            value["model"].update(
                {
                    "backend": self.model.backend,
                    "hub_repo_id": self.model.hub_repo_id,
                    "hub_revision": self.model.hub_revision,
                    "asset_ledger_path": str(self.model.asset_ledger_path),
                }
            )
        return value

    def snapshot_dict(self) -> dict[str, Any]:
        value = self.semantic_dict()
        value["run"] = {
            "name": self.run.name,
            **value["run"],
            "output_root": str(self.run.output_root),
            "resume_checkpoint": (
                None
                if self.run.resume_checkpoint is None
                else str(self.run.resume_checkpoint)
            ),
        }
        value["semantic_sha256"] = self.semantic_sha256
        return value


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是字符串键对象",
        )
    return dict(value)


def _keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    location: str,
) -> None:
    expected = set(required)
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ConfigError(
            ReasonCode.UNKNOWN_FIELD,
            f"{location}: 未知字段 {unknown}",
        )
    if missing:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 缺少字段 {missing}",
        )


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是非空字符串",
        )
    return value.strip()


def _bool(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是 bool",
        )
    return value


def _int(
    value: Any,
    *,
    location: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是 >= {minimum} 的整数",
        )
    return value


def _float(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是数值",
        )
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(
            ReasonCode.NONFINITE_NUMBER,
            f"{location}: 必须是有限数",
        )
    if minimum is not None and result < minimum:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须 >= {minimum}",
        )
    if maximum is not None and (
        result >= maximum if maximum_exclusive else result > maximum
    ):
        operator = "<" if maximum_exclusive else "<="
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须 {operator} {maximum}",
        )
    return result


def _path(
    value: Any,
    *,
    base: Path,
    location: str,
    nullable: bool = False,
) -> Path | None:
    if nullable and value is None:
        return None
    text = _string(value, location=location)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _string_list(
    value: Any,
    *,
    location: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是字符串列表",
        )
    output = tuple(item.strip() for item in value)
    if not allow_empty and not output:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 不能为空",
        )
    if len(output) != len(set(output)):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 不允许重复",
        )
    return output


def _nullable_sha256(value: Any, *, location: str) -> str | None:
    if value is None:
        return None
    result = _string(value, location=location)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是小写 SHA-256 或 null",
        )
    return result


def _sha256(value: Any, *, location: str) -> str:
    result = _nullable_sha256(value, location=location)
    if result is None:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是小写 SHA-256",
        )
    return result


def _commit_revision(value: Any, *, location: str) -> str:
    result = _string(value, location=location)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是 40 位小写 commit SHA",
        )
    return result


def _weights(value: Any, *, location: str) -> dict[str, float]:
    row = _mapping(value, location=location)
    output: dict[str, float] = {}
    for key, child in row.items():
        if not key.strip():
            raise ConfigError(
                ReasonCode.TYPE_MISMATCH,
                f"{location}: weight key 不能为空",
            )
        output[key] = _float(
            child,
            location=f"{location}.{key}",
            minimum=0.0,
        )
        if output[key] <= 0:
            raise ConfigError(
                ReasonCode.TYPE_MISMATCH,
                f"{location}.{key}: weight 必须 > 0",
            )
    return output


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise ConfigError(
            ReasonCode.OUTPUT_LINK,
            f"配置必须是普通单链接文件：{path}",
        )
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"无法严格读取 YAML：{path}",
            details={"error": str(error)},
        ) from error
    return _mapping(value, location="$")


def load_config(path: Path | str) -> VLMConfig:
    config_path = Path(os.path.abspath(Path(path)))
    linked = first_symlink_component(config_path)
    if linked is not None:
        raise ConfigError(
            ReasonCode.OUTPUT_LINK,
            f"配置路径含链接组件：{linked}",
        )
    base = config_path.parent
    row = _load_yaml(config_path)
    section_names = (
        "schema_version",
        "run",
        "data",
        "limits",
        "model",
        "adaptation",
        "training",
        "generation",
    )
    _keys(row, required=section_names, location="$")
    schema_version = _string(
        row["schema_version"],
        location="$.schema_version",
    )
    if schema_version not in {
        CONFIG_SCHEMA_VERSION,
        CONFIG_SCHEMA_VERSION_V3,
    }:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            "仅支持 schema_version="
            f"{CONFIG_SCHEMA_VERSION}/{CONFIG_SCHEMA_VERSION_V3}",
        )

    run_row = _mapping(row["run"], location="$.run")
    _keys(
        run_row,
        required=(
            "name",
            "seed",
            "mode",
            "mask_mode",
            "output_root",
            "resume_checkpoint",
        ),
        location="$.run",
    )
    try:
        mode = DataMode(run_row["mode"])
        mask_mode = MaskMode(run_row["mask_mode"])
    except (TypeError, ValueError) as error:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            "$.run mode/mask_mode 非法",
        ) from error
    run = RunSection(
        name=_string(run_row["name"], location="$.run.name"),
        seed=_int(run_row["seed"], location="$.run.seed"),
        mode=mode,
        mask_mode=mask_mode,
        output_root=_path(
            run_row["output_root"],
            base=base,
            location="$.run.output_root",
        ),
        resume_checkpoint=_path(
            run_row["resume_checkpoint"],
            base=base,
            location="$.run.resume_checkpoint",
            nullable=True,
        ),
    )

    data_row = _mapping(row["data"], location="$.data")
    _keys(
        data_row,
        required=(
            "benchmark_root",
            "roles",
            "task_families",
            "expected_manifest_schema",
            "expected_canonical_schema",
            "expected_manifest_sha256",
            "expected_validation_sha256",
            "expected_build_id",
            "expected_payload_sha256",
            "expected_hash_manifest_sha256",
            "parent_balanced",
            "source_weights",
            "task_weights",
        ),
        location="$.data",
    )
    data = DataSection(
        benchmark_root=_path(
            data_row["benchmark_root"],
            base=base,
            location="$.data.benchmark_root",
        ),
        roles=_string_list(data_row["roles"], location="$.data.roles"),
        task_families=_string_list(
            data_row["task_families"],
            location="$.data.task_families",
        ),
        expected_manifest_schema=_string(
            data_row["expected_manifest_schema"],
            location="$.data.expected_manifest_schema",
        ),
        expected_canonical_schema=_string(
            data_row["expected_canonical_schema"],
            location="$.data.expected_canonical_schema",
        ),
        expected_manifest_sha256=_sha256(
            data_row["expected_manifest_sha256"],
            location="$.data.expected_manifest_sha256",
        ),
        expected_validation_sha256=_sha256(
            data_row["expected_validation_sha256"],
            location="$.data.expected_validation_sha256",
        ),
        expected_build_id=_string(
            data_row["expected_build_id"],
            location="$.data.expected_build_id",
        ),
        expected_payload_sha256=_sha256(
            data_row["expected_payload_sha256"],
            location="$.data.expected_payload_sha256",
        ),
        expected_hash_manifest_sha256=_sha256(
            data_row["expected_hash_manifest_sha256"],
            location="$.data.expected_hash_manifest_sha256",
        ),
        parent_balanced=_bool(
            data_row["parent_balanced"],
            location="$.data.parent_balanced",
        ),
        source_weights=_weights(
            data_row["source_weights"],
            location="$.data.source_weights",
        ),
        task_weights=_weights(
            data_row["task_weights"],
            location="$.data.task_weights",
        ),
    )

    limits_row = _mapping(row["limits"], location="$.limits")
    limit_names = (
        "max_parents",
        "max_records",
        "max_assets",
        "max_asset_bytes",
        "max_copied_asset_bytes",
        "max_images",
        "max_input_tokens",
        "max_total_tokens",
        "min_pixels",
        "max_pixels",
        "max_new_tokens",
        "max_probe_shards",
        "max_records_per_shard",
        "max_probe_records",
    )
    _keys(limits_row, required=limit_names, location="$.limits")
    limits_values = {
        name: _int(
            limits_row[name],
            location=f"$.limits.{name}",
            minimum=0 if name == "max_copied_asset_bytes" else 1,
        )
        for name in limit_names
    }
    limits = LimitsSection(**limits_values)
    if limits.min_pixels > limits.max_pixels:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "min_pixels 不能大于 max_pixels",
        )
    model_row = _mapping(row["model"], location="$.model")
    model_names = (
        "path",
        "processor_path",
        "local_files_only",
        "dtype",
        "attn_implementation",
        "trust_remote_code",
    )
    if schema_version == CONFIG_SCHEMA_VERSION_V3:
        model_names = (
            *model_names,
            "backend",
            "hub_repo_id",
            "hub_revision",
            "asset_ledger_path",
        )
    _keys(
        model_row,
        required=model_names,
        location="$.model",
    )
    backend = (
        "qwen3_vl"
        if schema_version == CONFIG_SCHEMA_VERSION
        else _string(model_row["backend"], location="$.model.backend")
    )
    if backend not in SUPPORTED_VLM_BACKENDS:
        raise ConfigError(
            ReasonCode.BACKEND_UNKNOWN,
            f"$.model.backend: 未知 backend {backend!r}",
            details={"available": sorted(SUPPORTED_VLM_BACKENDS)},
        )
    hub_repo_id = None
    hub_revision = None
    asset_ledger_path = None
    if schema_version == CONFIG_SCHEMA_VERSION_V3:
        hub_repo_id = _string(
            model_row["hub_repo_id"],
            location="$.model.hub_repo_id",
        )
        hub_revision = _commit_revision(
            model_row["hub_revision"],
            location="$.model.hub_revision",
        )
        asset_ledger_path = _path(
            model_row["asset_ledger_path"],
            base=base,
            location="$.model.asset_ledger_path",
        )
        if backend == "qwen3_5" and hub_repo_id != "Qwen/Qwen3.5-4B":
            raise ConfigError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "qwen3_5 backend 固定官方 Qwen/Qwen3.5-4B",
            )
    model = ModelSection(
        path=_path(model_row["path"], base=base, location="$.model.path"),
        processor_path=_path(
            model_row["processor_path"],
            base=base,
            location="$.model.processor_path",
        ),
        local_files_only=_bool(
            model_row["local_files_only"],
            location="$.model.local_files_only",
        ),
        dtype=_string(model_row["dtype"], location="$.model.dtype"),
        attn_implementation=_string(
            model_row["attn_implementation"],
            location="$.model.attn_implementation",
        ),
        trust_remote_code=_bool(
            model_row["trust_remote_code"],
            location="$.model.trust_remote_code",
        ),
        backend=backend,
        hub_repo_id=hub_repo_id,
        hub_revision=hub_revision,
        asset_ledger_path=asset_ledger_path,
    )
    if (
        not model.local_files_only
        or model.dtype != "bfloat16"
        or model.attn_implementation != "sdpa"
        or model.trust_remote_code
    ):
        raise ConfigError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "本阶段固定 local_files_only=true、bfloat16、sdpa、"
            "trust_remote_code=false",
        )
    expected_max_new_tokens = 384 if backend == "qwen3_vl" else 768
    if (
        limits.max_images != 5
        or limits.max_input_tokens != 2048
        or limits.min_pixels != 28 * 28 * 16
        or limits.max_pixels != 28 * 28 * 256
        or limits.max_new_tokens > expected_max_new_tokens
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{backend} 主线固定 max_images=5、input_tokens=2048、"
            "pixels=28x28x[16,256]、"
            f"max_new_tokens<={expected_max_new_tokens}",
        )

    adaptation_row = _mapping(row["adaptation"], location="$.adaptation")
    _keys(
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
    adaptation = AdaptationSection(
        strategy=_string(
            adaptation_row["strategy"],
            location="$.adaptation.strategy",
        ),
        target_modules=_string_list(
            adaptation_row["target_modules"],
            location="$.adaptation.target_modules",
            allow_empty=True,
        ),
        rank=_int(
            adaptation_row["rank"],
            location="$.adaptation.rank",
        ),
        alpha=_int(
            adaptation_row["alpha"],
            location="$.adaptation.alpha",
        ),
        dropout=_float(
            adaptation_row["dropout"],
            location="$.adaptation.dropout",
            minimum=0.0,
            maximum=1.0,
            maximum_exclusive=True,
        ),
        freeze_vision=_bool(
            adaptation_row["freeze_vision"],
            location="$.adaptation.freeze_vision",
        ),
        freeze_merger=_bool(
            adaptation_row["freeze_merger"],
            location="$.adaptation.freeze_merger",
        ),
    )
    if adaptation.strategy not in {"prompt_only", "lora"}:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            "adaptation.strategy 只支持 prompt_only/lora",
        )
    if not adaptation.freeze_vision or not adaptation.freeze_merger:
        raise ConfigError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Shared VLM 主线固定冻结 vision 与 merger",
        )
    if adaptation.strategy == "prompt_only":
        if (
            adaptation.target_modules
            or adaptation.rank != 0
            or adaptation.alpha != 0
            or adaptation.dropout != 0
        ):
            raise ConfigError(
                ReasonCode.TYPE_MISMATCH,
                "prompt_only 要求空 target_modules 且 rank/alpha/dropout=0",
            )
    else:
        if (
            adaptation.target_modules
            != ("q_proj", "k_proj", "v_proj", "o_proj")
            or adaptation.rank != 8
            or adaptation.alpha != 16
            or adaptation.dropout != 0.05
        ):
            raise ConfigError(
                ReasonCode.TYPE_MISMATCH,
                "当前唯一 LoRA 主线固定 q/k/v/o、r=8、alpha=16、dropout=0.05",
            )

    training_row = _mapping(row["training"], location="$.training")
    training_names = (
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "scheduler",
        "max_grad_norm",
        "max_steps",
        "epochs",
        "num_workers",
        "prefetch_factor",
        "pin_memory",
        "checkpoint_interval",
        "log_interval",
        "validation_interval",
        "validation_max_parents",
    )
    _keys(training_row, required=training_names, location="$.training")
    training = TrainingSection(
        batch_size=_int(
            training_row["batch_size"],
            location="$.training.batch_size",
            minimum=1,
        ),
        gradient_accumulation_steps=_int(
            training_row["gradient_accumulation_steps"],
            location="$.training.gradient_accumulation_steps",
            minimum=1,
        ),
        gradient_checkpointing=_bool(
            training_row["gradient_checkpointing"],
            location="$.training.gradient_checkpointing",
        ),
        optimizer=_string(
            training_row["optimizer"],
            location="$.training.optimizer",
        ),
        learning_rate=_float(
            training_row["learning_rate"],
            location="$.training.learning_rate",
            minimum=0.0,
        ),
        weight_decay=_float(
            training_row["weight_decay"],
            location="$.training.weight_decay",
            minimum=0.0,
        ),
        warmup_ratio=_float(
            training_row["warmup_ratio"],
            location="$.training.warmup_ratio",
            minimum=0.0,
            maximum=1.0,
            maximum_exclusive=True,
        ),
        scheduler=_string(
            training_row["scheduler"],
            location="$.training.scheduler",
        ),
        max_grad_norm=_float(
            training_row["max_grad_norm"],
            location="$.training.max_grad_norm",
            minimum=0.0,
        ),
        max_steps=_int(
            training_row["max_steps"],
            location="$.training.max_steps",
            minimum=1,
        ),
        epochs=_int(
            training_row["epochs"],
            location="$.training.epochs",
            minimum=1,
        ),
        num_workers=_int(
            training_row["num_workers"],
            location="$.training.num_workers",
            minimum=0,
        ),
        prefetch_factor=_int(
            training_row["prefetch_factor"],
            location="$.training.prefetch_factor",
            minimum=0,
        ),
        pin_memory=_bool(
            training_row["pin_memory"],
            location="$.training.pin_memory",
        ),
        checkpoint_interval=_int(
            training_row["checkpoint_interval"],
            location="$.training.checkpoint_interval",
            minimum=1,
        ),
        log_interval=_int(
            training_row["log_interval"],
            location="$.training.log_interval",
            minimum=1,
        ),
        validation_interval=_int(
            training_row["validation_interval"],
            location="$.training.validation_interval",
            minimum=1,
        ),
        validation_max_parents=_int(
            training_row["validation_max_parents"],
            location="$.training.validation_max_parents",
            minimum=1,
        ),
    )
    if (
        (
            training.batch_size,
            training.gradient_accumulation_steps,
        )
        not in {(1, 16), (4, 4)}
        or not training.gradient_checkpointing
        or training.optimizer != "adamw"
        or training.learning_rate != 2e-4
        or training.warmup_ratio != 0.03
        or training.scheduler != "cosine"
        or training.max_grad_norm != 1.0
        or training.num_workers > 4
        or (
            training.num_workers == 0
            and (
                training.prefetch_factor != 0
                or training.pin_memory
            )
        )
        or (
            training.num_workers > 0
            and (
                not 1 <= training.prefetch_factor <= 4
                or not training.pin_memory
            )
        )
        or training.validation_max_parents > 128
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "24 GB 主线只接受 batch/accum=1/16 或 4/4，"
            "固定 effective batch=16、checkpointing、AdamW、"
            "lr=2e-4、warmup=0.03、cosine、clip=1；同步输入要求"
            "workers/prefetch/pin=0/0/false，有序预取要求 workers=1..4、"
            "prefetch=1..4、pin=true，"
            "且 validation_max_parents<=128",
        )
    if backend == "qwen3_5" and (
        training.batch_size,
        training.gradient_accumulation_steps,
    ) != (1, 16):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "Qwen3.5-4B 24 GB 主线固定 batch_size=1、"
            "gradient_accumulation_steps=16",
        )

    generation_row = _mapping(row["generation"], location="$.generation")
    _keys(
        generation_row,
        required=("max_new_tokens", "do_sample", "temperature", "top_p"),
        location="$.generation",
    )
    generation = GenerationSection(
        max_new_tokens=_int(
            generation_row["max_new_tokens"],
            location="$.generation.max_new_tokens",
            minimum=1,
        ),
        do_sample=_bool(
            generation_row["do_sample"],
            location="$.generation.do_sample",
        ),
        temperature=_float(
            generation_row["temperature"],
            location="$.generation.temperature",
            minimum=0.0,
        ),
        top_p=_float(
            generation_row["top_p"],
            location="$.generation.top_p",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    if generation.max_new_tokens > limits.max_new_tokens:
        raise ConfigError(
            ReasonCode.TOKEN_LIMIT_EXCEEDED,
            "generation.max_new_tokens 超过 limits.max_new_tokens",
        )
    if generation.do_sample:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "主线要求确定性 do_sample=false",
        )

    if mode is not DataMode.EXTERNAL_GENERIC:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"config {schema_version} 只支持 external_generic",
        )
    if mask_mode is not MaskMode.EXTERNAL_GENERIC:
        raise ConfigError(
            ReasonCode.EXTERNAL_MASK_FORBIDDEN,
            f"config {schema_version} 只支持 external_generic mask_mode",
        )
    if set(data.roles) - {"external_train", "external_val"}:
        raise ConfigError(
            ReasonCode.ROLE_FORBIDDEN,
            "External mode roles 只能是 external_train/external_val",
        )

    provisional = VLMConfig(
        schema_version=schema_version,
        run=run,
        data=data,
        limits=limits,
        model=model,
        adaptation=adaptation,
        training=training,
        generation=generation,
        config_path=config_path,
        semantic_sha256="",
    )
    semantic_sha256 = sha256_text(canonical_json(provisional.semantic_dict()))
    return VLMConfig(
        **{
            field: getattr(provisional, field)
            for field in provisional.__dataclass_fields__
            if field != "semantic_sha256"
        },
        semantic_sha256=semantic_sha256,
    )


def apply_runtime_overrides(
    config: VLMConfig,
    *,
    output_root: Path | None = None,
    resume_checkpoint: Path | None = None,
    log_interval: int | None = None,
) -> VLMConfig:
    """应用显式 CLI 覆盖并重新计算严格配置语义摘要。"""

    run = config.run
    if output_root is not None:
        run = replace(
            run,
            output_root=Path(os.path.abspath(output_root)),
        )
    if resume_checkpoint is not None:
        run = replace(
            run,
            resume_checkpoint=Path(os.path.abspath(resume_checkpoint)),
        )
    training = config.training
    if log_interval is not None:
        training = replace(
            training,
            log_interval=_int(
                log_interval,
                location="--log-interval",
                minimum=1,
            ),
        )
    provisional = replace(
        config,
        run=run,
        training=training,
        semantic_sha256="",
    )
    return replace(
        provisional,
        semantic_sha256=sha256_text(
            canonical_json(provisional.semantic_dict())
        ),
    )
