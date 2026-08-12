"""严格 YAML 构建与 Qwen 导出配置。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .io import (
    canonical_json,
    reject_nonfinite,
    require_bool,
    require_enum,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    resolve_config_path,
    sha256_text,
)
from .contracts import (
    BUILD_CONFIG_VERSION,
    DESCRIPTION_MULTITASK_PROFILE,
    EXTERNAL_SPLIT_VERSION,
    EXPORT_CONFIG_VERSION,
    LogicalRole,
    QWEN_TEMPLATE_VERSION,
    RS_GENERALDESC_ROLES,
    RS_GENERALDESC_TASK_FAMILIES,
    TaskFamily,
)
from .errors import ConfigError, ReasonCode


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    value: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in value
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML mapping key 必须可哈希",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"重复 YAML key：{key!r}",
                key_node.start_mark,
            )
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class SourceConfig:
    enabled: bool
    root: Path


@dataclass(frozen=True)
class ExternalSplitConfig:
    version: str
    validation_percent: int


@dataclass(frozen=True)
class TextPolicyConfig:
    version: str
    forbidden_patterns: tuple[str, ...]
    reasoning_forbidden_patterns: tuple[str, ...]


@dataclass(frozen=True)
class AssetPolicyConfig:
    image_format: str
    image_quality: int
    jpeg_subsampling: int
    target_size: tuple[int, int] | None


@dataclass(frozen=True)
class ShardingConfig:
    records_per_shard: int


@dataclass(frozen=True)
class LimitsConfig:
    max_parents: int | None
    max_records: int | None
    max_assets: int | None
    max_copied_bytes: int | None
    max_validation_candidates_per_task: int
    per_task: Mapping[str, int]


@dataclass(frozen=True)
class BuildConfig:
    schema_version: str
    profile: str
    seed: int
    output_root: Path
    sources: Mapping[str, SourceConfig]
    external_split: ExternalSplitConfig
    text_policy: TextPolicyConfig
    asset_policy: AssetPolicyConfig
    sharding: ShardingConfig
    workers: int
    limits: LimitsConfig
    deletion_allowlist: bool
    config_path: Path
    semantic_hash: str

    def sanitized(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "seed": self.seed,
            "sources": {
                name: {"enabled": value.enabled}
                for name, value in sorted(self.sources.items())
            },
            "external_split": {
                "version": self.external_split.version,
                "validation_percent": self.external_split.validation_percent,
            },
            "text_policy": {
                "version": self.text_policy.version,
                "forbidden_patterns": list(self.text_policy.forbidden_patterns),
                "reasoning_forbidden_patterns": list(
                    self.text_policy.reasoning_forbidden_patterns
                ),
            },
            "asset_policy": {
                "image_format": self.asset_policy.image_format,
                "image_quality": self.asset_policy.image_quality,
                "jpeg_subsampling": self.asset_policy.jpeg_subsampling,
                "target_size": (
                    list(self.asset_policy.target_size)
                    if self.asset_policy.target_size is not None
                    else None
                ),
            },
            "sharding": {
                "records_per_shard": self.sharding.records_per_shard,
            },
            "workers": self.workers,
            "limits": {
                "max_parents": self.limits.max_parents,
                "max_records": self.limits.max_records,
                "max_assets": self.limits.max_assets,
                "max_copied_bytes": self.limits.max_copied_bytes,
                "max_validation_candidates_per_task": (
                    self.limits.max_validation_candidates_per_task
                ),
                "per_task": dict(sorted(self.limits.per_task.items())),
            },
            "deletion_allowlist": self.deletion_allowlist,
        }


@dataclass(frozen=True)
class ExportConfig:
    schema_version: str
    profile: str
    benchmark_root: Path
    output_root: Path
    seed: int
    purpose: str
    roles: tuple[str, ...]
    task_families: tuple[str, ...]
    template_version: str
    config_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(
            ReasonCode.SCHEMA_MISMATCH,
            f"无法严格读取 YAML：{path}",
            details={"error": str(error)},
        ) from error
    reject_nonfinite(value, location=str(path))
    return require_mapping(value, location="$")


def _optional_limit(value: Any, *, location: str) -> int | None:
    if value is None:
        return None
    return require_int(value, location=location, minimum=1)


def _parse_source(
    value: Any,
    *,
    base: Path,
    location: str,
) -> SourceConfig:
    row = require_mapping(value, location=location)
    require_exact_keys(row, required=("enabled", "root"), location=location)
    return SourceConfig(
        enabled=require_bool(row["enabled"], location=f"{location}.enabled"),
        root=resolve_config_path(base, row["root"], location=f"{location}.root"),
    )


def load_build_config(path: Path | str) -> BuildConfig:
    config_path = Path(path).resolve()
    base = config_path.parent
    row = _load_yaml(config_path)
    required = (
        "schema_version",
        "profile",
        "seed",
        "output_root",
        "sources",
        "external_split",
        "text_policy",
        "asset_policy",
        "sharding",
        "workers",
        "limits",
        "deletion_allowlist",
    )
    require_exact_keys(row, required=required, location="$")
    schema_version = require_string(row["schema_version"], location="$.schema_version")
    if schema_version != BUILD_CONFIG_VERSION:
        raise ConfigError(
            ReasonCode.SCHEMA_MISMATCH,
            f"仅支持 {BUILD_CONFIG_VERSION}",
        )
    profile = require_enum(
        row["profile"], choices=("smoke", "full"), location="$.profile"
    )
    sources_row = require_mapping(row["sources"], location="$.sources")
    require_exact_keys(
        sources_row,
        required=("rsgpt", "mmrs1m", "disasterm3"),
        location="$.sources",
    )
    sources = {
        name: _parse_source(
            sources_row[name], base=base, location=f"$.sources.{name}"
        )
        for name in ("rsgpt", "mmrs1m", "disasterm3")
    }

    split_row = require_mapping(
        row["external_split"], location="$.external_split"
    )
    require_exact_keys(
        split_row,
        required=("version", "validation_percent"),
        location="$.external_split",
    )
    split_version = require_string(
        split_row["version"], location="$.external_split.version"
    )
    if split_version != EXTERNAL_SPLIT_VERSION:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 external_split.version={EXTERNAL_SPLIT_VERSION}",
        )
    validation_percent = require_int(
        split_row["validation_percent"],
        location="$.external_split.validation_percent",
        minimum=0,
    )
    if validation_percent > 50:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "$.external_split.validation_percent 必须在 0..50",
        )
    external_split = ExternalSplitConfig(
        version=split_version,
        validation_percent=validation_percent,
    )

    if not any(source.enabled for source in sources.values()):
        raise ConfigError(
            ReasonCode.SCHEMA_MISMATCH,
            "至少启用一个 external source",
        )

    text_row = require_mapping(row["text_policy"], location="$.text_policy")
    require_exact_keys(
        text_row,
        required=(
            "version",
            "forbidden_patterns",
            "reasoning_forbidden_patterns",
        ),
        location="$.text_policy",
    )
    for name in ("forbidden_patterns", "reasoning_forbidden_patterns"):
        if (
            not isinstance(text_row[name], list)
            or not text_row[name]
            or not all(
            isinstance(item, str) and item for item in text_row[name]
            )
        ):
            raise ConfigError(
                ReasonCode.TYPE_MISMATCH,
                f"$.text_policy.{name}: 必须是非空字符串列表",
            )
        for pattern in text_row[name]:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ConfigError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"$.text_policy.{name}: 非法正则 {pattern!r}",
                    details={"error": str(error)},
                ) from error
    text_policy = TextPolicyConfig(
        version=require_string(text_row["version"], location="$.text_policy.version"),
        forbidden_patterns=tuple(text_row["forbidden_patterns"]),
        reasoning_forbidden_patterns=tuple(
            text_row["reasoning_forbidden_patterns"]
        ),
    )

    asset_row = require_mapping(row["asset_policy"], location="$.asset_policy")
    require_exact_keys(
        asset_row,
        required=(
            "image_format",
            "image_quality",
            "jpeg_subsampling",
            "target_size",
        ),
        location="$.asset_policy",
    )
    image_format = require_enum(
        asset_row["image_format"],
        choices=("jpeg", "png"),
        location="$.asset_policy.image_format",
    )
    target_raw = asset_row["target_size"]
    if target_raw is None:
        target_size = None
    elif (
        isinstance(target_raw, list)
        and len(target_raw) == 2
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in target_raw
        )
    ):
        target_size = (int(target_raw[0]), int(target_raw[1]))
    else:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "$.asset_policy.target_size 必须为 null 或 [width,height]",
        )
    asset_policy = AssetPolicyConfig(
        image_format=image_format,
        image_quality=require_int(
            asset_row["image_quality"],
            location="$.asset_policy.image_quality",
            minimum=1,
        ),
        jpeg_subsampling=require_int(
            asset_row["jpeg_subsampling"],
            location="$.asset_policy.jpeg_subsampling",
            minimum=0,
        ),
        target_size=target_size,
    )
    if asset_policy.image_quality > 100 or asset_policy.jpeg_subsampling > 2:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "JPEG quality 必须 <=100，subsampling 必须 <=2",
        )

    shard_row = require_mapping(row["sharding"], location="$.sharding")
    require_exact_keys(
        shard_row, required=("records_per_shard",), location="$.sharding"
    )
    sharding = ShardingConfig(
        records_per_shard=require_int(
            shard_row["records_per_shard"],
            location="$.sharding.records_per_shard",
            minimum=1,
        )
    )
    limits_row = require_mapping(row["limits"], location="$.limits")
    require_exact_keys(
        limits_row,
        required=(
            "max_parents",
            "max_records",
            "max_assets",
            "max_copied_bytes",
            "max_validation_candidates_per_task",
            "per_task",
        ),
        location="$.limits",
    )
    per_task_raw = require_mapping(limits_row["per_task"], location="$.limits.per_task")
    per_task = {
        require_string(key, location="$.limits.per_task.key"): require_int(
            value, location=f"$.limits.per_task.{key}", minimum=1
        )
        for key, value in per_task_raw.items()
    }
    limits = LimitsConfig(
        max_parents=_optional_limit(
            limits_row["max_parents"], location="$.limits.max_parents"
        ),
        max_records=_optional_limit(
            limits_row["max_records"], location="$.limits.max_records"
        ),
        max_assets=_optional_limit(
            limits_row["max_assets"], location="$.limits.max_assets"
        ),
        max_copied_bytes=_optional_limit(
            limits_row["max_copied_bytes"], location="$.limits.max_copied_bytes"
        ),
        max_validation_candidates_per_task=require_int(
            limits_row["max_validation_candidates_per_task"],
            location="$.limits.max_validation_candidates_per_task",
            minimum=1,
        ),
        per_task=per_task,
    )
    output_root = resolve_config_path(base, row["output_root"], location="$.output_root")
    config = BuildConfig(
        schema_version=schema_version,
        profile=profile,
        seed=require_int(row["seed"], location="$.seed", minimum=0),
        output_root=output_root,
        sources=sources,
        external_split=external_split,
        text_policy=text_policy,
        asset_policy=asset_policy,
        sharding=sharding,
        workers=require_int(row["workers"], location="$.workers", minimum=1),
        limits=limits,
        deletion_allowlist=require_bool(
            row["deletion_allowlist"], location="$.deletion_allowlist"
        ),
        config_path=config_path,
        semantic_hash="",
    )
    semantic_hash = sha256_text(canonical_json(config.sanitized()))
    return BuildConfig(**{**config.__dict__, "semantic_hash": semantic_hash})


def load_export_config(path: Path | str) -> ExportConfig:
    config_path = Path(path).resolve()
    base = config_path.parent
    row = _load_yaml(config_path)
    require_exact_keys(
        row,
        required=(
            "schema_version",
            "profile",
            "benchmark_root",
            "output_root",
            "seed",
            "purpose",
            "roles",
            "task_families",
            "template_version",
        ),
        location="$",
    )
    schema_version = require_string(row["schema_version"], location="$.schema_version")
    if schema_version != EXPORT_CONFIG_VERSION:
        raise ConfigError(
            ReasonCode.SCHEMA_MISMATCH,
            f"仅支持 {EXPORT_CONFIG_VERSION}",
        )
    roles = row["roles"]
    tasks = row["task_families"]
    if not isinstance(roles, list) or not all(isinstance(item, str) for item in roles):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "$.roles 必须是字符串列表")
    if not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH, "$.task_families 必须是字符串列表"
        )
    if not roles or len(set(roles)) != len(roles):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "$.roles 必须是非空且不重复的逻辑角色列表",
        )
    if not tasks or len(set(tasks)) != len(tasks):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "$.task_families 必须显式列出至少一个且不允许重复",
        )
    try:
        parsed_roles = {LogicalRole(role) for role in roles}
        parsed_tasks = {TaskFamily(task) for task in tasks}
    except ValueError as error:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            "export roles/task_families 含非法枚举",
        ) from error
    if not parsed_roles <= RS_GENERALDESC_ROLES:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            "RS-GeneralDesc export 只接受 external_train/external_val",
        )
    if not parsed_tasks <= RS_GENERALDESC_TASK_FAMILIES:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            "RS-GeneralDesc export 只接受七类 External 任务",
        )
    template = require_string(row["template_version"], location="$.template_version")
    if template != QWEN_TEMPLATE_VERSION:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 template_version={QWEN_TEMPLATE_VERSION}",
        )
    profile = require_string(row["profile"], location="$.profile")
    if profile != DESCRIPTION_MULTITASK_PROFILE:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 profile={DESCRIPTION_MULTITASK_PROFILE}",
        )
    return ExportConfig(
        schema_version=schema_version,
        profile=profile,
        benchmark_root=resolve_config_path(
            base, row["benchmark_root"], location="$.benchmark_root"
        ),
        output_root=resolve_config_path(
            base, row["output_root"], location="$.output_root"
        ),
        seed=require_int(row["seed"], location="$.seed", minimum=0),
        purpose=require_enum(
            row["purpose"],
            choices=("training", "validation"),
            location="$.purpose",
        ),
        roles=tuple(roles),
        task_families=tuple(tasks),
        template_version=template,
        config_path=config_path,
    )
