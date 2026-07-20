"""Pydantic configuration contract for the P1 raw-source audit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from sami_gsd.contracts.canonical import LicenseRecord, StrictModel, validate_portable_path


class RootSpec(StrictModel):
    """Portable runtime-root policy with an optional environment override."""

    env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]
    relative_to: Literal["repository_root", "repository_parent"]
    default: str

    _default_is_portable = field_validator("default")(validate_portable_path)


class AuditSettings(StrictModel):
    """Deterministic and read-only scanner behavior."""

    hash_algorithm: Literal["sha256"]
    include_hidden: bool
    follow_symlinks: Literal[False]


class SourceConfig(StrictModel):
    """One raw source plus its fail-closed license snapshot."""

    source_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    display_name: Annotated[str, Field(min_length=1)]
    local_path: str
    enabled: bool
    allowed_task_roles: tuple[Literal["inventory", "t1", "t2", "t3", "t4", "language_global", "language_region"], ...]
    license: LicenseRecord

    _local_path_is_portable = field_validator("local_path")(validate_portable_path)

    @model_validator(mode="after")
    def source_key_matches_license(self) -> Self:
        """Prevent registry rows from being attached to the wrong source."""

        if self.source_key != self.license.source_key:
            raise ValueError("source_key must match license.source_key")
        if self.license.allowed_for_training and not self.allowed_task_roles:
            raise ValueError("training-eligible source requires at least one allowed task role")
        return self


class BenchmarkAuditConfig(StrictModel):
    """Strict configuration for ``sami-gsd data audit``."""

    schema_version: Literal["sami_benchmark_audit_config_v3"]
    benchmark_name: Literal["SAMI Landslide Grounded Benchmark v3"]
    mode: Literal["small", "full"]
    seed: int
    benchmark_relative_path: str
    datasets_root: RootSpec
    benchmark_root: RootSpec
    audit: AuditSettings
    sources: tuple[SourceConfig, ...]

    _benchmark_path_is_portable = field_validator("benchmark_relative_path")(validate_portable_path)

    @model_validator(mode="after")
    def source_keys_are_unique(self) -> Self:
        """Reject ambiguous source or local-root bindings."""

        keys = [source.source_key for source in self.sources]
        paths = [source.local_path for source in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("source_key values must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("source local_path values must be unique")
        expected_suffix = f"sami_landslide_v3/{self.mode}"
        if self.benchmark_relative_path != expected_suffix:
            raise ValueError(f"benchmark_relative_path must be {expected_suffix!r}")
        return self


def load_audit_config(path: Path) -> BenchmarkAuditConfig:
    """Read and validate one YAML audit config.

    Args:
        path: UTF-8 YAML path.

    Returns:
        A frozen validated config.

    Raises:
        FileNotFoundError: The config does not exist.
        ValueError: YAML is empty or not a mapping.
        yaml.YAMLError: YAML syntax is invalid.
        pydantic.ValidationError: The mapping violates the contract.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit config must be a non-empty YAML mapping")
    return BenchmarkAuditConfig.model_validate(payload)


def resolve_root(spec: RootSpec, *, repository_root: Path, override: Path | None = None) -> Path:
    """Resolve a runtime root without storing its machine path in artifacts."""

    if override is not None:
        return override.expanduser().resolve()
    environment_value = os.environ.get(spec.env)
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    base = repository_root if spec.relative_to == "repository_root" else repository_root.parent
    return (base / spec.default).resolve()
