"""Pydantic configuration contract for the P1 raw-source audit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from sami_gsd.contracts.canonical import StrictModel, validate_portable_path


LanguageComponentName = Literal[
    "rsicd",
    "ucm",
    "sydney",
    "nwpu",
    "rsitmd",
    "dior_rsvg",
    "rsicap",
    "rsieval",
]
LanguageTaskRole = Literal["language_global", "language_region"]


class SourceProvenance(StrictModel):
    """Minimal scientific source/citation metadata with no runtime approval semantics."""

    source_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_:-]*$")]
    source_name: Annotated[str, Field(min_length=1)]
    source_root: str
    source_document: str | None
    citation_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_:-]*$")]
    upstream_url: str | None
    provenance_notes: Annotated[str, Field(min_length=1)]

    @field_validator("source_root")
    @classmethod
    def source_root_is_dataset_logical_path(cls, value: str) -> str:
        """Keep registry roots portable and rooted in the logical datasets namespace."""

        value = validate_portable_path(value)
        if not value.startswith("datasets/") or value == "datasets/":
            raise ValueError("source_root must be below datasets/")
        return value

    @field_validator("source_document")
    @classmethod
    def source_document_is_portable(cls, value: str | None) -> str | None:
        """Validate an optional local provenance document without interpreting its terms."""

        return None if value is None else validate_portable_path(value)

    @field_validator("upstream_url")
    @classmethod
    def upstream_url_is_https(cls, value: str | None) -> str | None:
        """Accept only an explicit HTTPS upstream citation URL."""

        if value is not None and not value.startswith("https://"):
            raise ValueError("upstream_url must use HTTPS")
        return value

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


class SplitSettings(StrictModel):
    """Frozen deterministic parent-level split proportions."""

    train: Annotated[float, Field(gt=0.0, lt=1.0)]
    val: Annotated[float, Field(gt=0.0, lt=1.0)]
    test: Annotated[float, Field(gt=0.0, lt=1.0)]

    @model_validator(mode="after")
    def proportions_sum_to_one(self) -> Self:
        """Reject implicit normalization of split ratios."""

        if abs(self.train + self.val + self.test - 1.0) > 1e-12:
            raise ValueError("split proportions must sum exactly to one within 1e-12")
        return self


class DuplicateSettings(StrictModel):
    """Frozen exact/perceptual duplicate verification policy."""

    dhash_candidate_max_distance: Annotated[int, Field(ge=0, le=64)]
    verified_rgb64_mae_threshold: Annotated[float, Field(ge=0.0)]
    normalized_rgb_hw: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]

    @field_validator("normalized_rgb_hw")
    @classmethod
    def rgb64_is_frozen(cls, value: tuple[int, int]) -> tuple[int, int]:
        """The verified duplicate protocol always uses RGB 64x64."""

        if value != (64, 64):
            raise ValueError("normalized_rgb_hw must be exactly (64, 64)")
        return value


class MaterializationSettings(StrictModel):
    """Deterministic reference-canvas and output policy."""

    canvas_hw: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    resize_policy: Literal["fit_inside_then_symmetric_zero_pad"]
    image_interpolation: Literal["bilinear_half_pixel_center"]
    mask_valid_interpolation: Literal["nearest"]
    image_dtype: Literal["float32"]
    mask_valid_dtype: Literal["uint8"]


class DescriptionSubsetSettings(StrictModel):
    """Frozen language-source selection and exclusions."""

    mmrs_caption_sources: tuple[
        Literal["rsicd", "ucm", "sydney", "nwpu", "rsitmd"], ...
    ]
    include_dior_rsvg_short_phrase_only: Literal[True]
    include_rsicap: Literal[True]
    rsieval_policy: Literal["permanent_test_only"]
    excluded_mmrs_tasks: tuple[
        Literal["total", "classification", "detection", "vqa", "infrared", "unrelated_sar"], ...
    ]

    @model_validator(mode="after")
    def exact_frozen_selection(self) -> Self:
        """Keep the P1 subset equal to the governing scientific protocol."""

        expected_sources = ("rsicd", "ucm", "sydney", "nwpu", "rsitmd")
        expected_excluded = ("total", "classification", "detection", "vqa", "infrared", "unrelated_sar")
        if self.mmrs_caption_sources != expected_sources:
            raise ValueError("MMRS caption sources must match the frozen ordered selection")
        if self.excluded_mmrs_tasks != expected_excluded:
            raise ValueError("MMRS exclusions must match the frozen ordered selection")
        return self


class BuildSettings(StrictModel):
    """Complete P1 Small/Full construction policy."""

    materialization: MaterializationSettings
    split: SplitSettings
    duplicates: DuplicateSettings
    description_subset: DescriptionSubsetSettings
    small_max_parents_per_source: Annotated[int, Field(gt=0)]


class LanguageComponentConfig(StrictModel):
    """One scientifically selected language component inside a shared raw root."""

    component: LanguageComponentName
    component_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*:[a-z0-9][a-z0-9_-]*$")]
    task_roles: tuple[LanguageTaskRole, ...]
    split_policy: Literal["train_candidate", "permanent_test_only"]
    provenance: SourceProvenance

    @model_validator(mode="after")
    def role_and_test_policy_are_frozen(self) -> Self:
        """Bind each component to its sole scientific role and split policy."""

        expected_role = "language_region" if self.component == "dior_rsvg" else "language_global"
        if self.task_roles != (expected_role,):
            raise ValueError(f"{self.component} must declare exactly {expected_role}")
        expected_split = "permanent_test_only" if self.component == "rsieval" else "train_candidate"
        if self.split_policy != expected_split:
            raise ValueError(f"{self.component} must use split_policy={expected_split}")
        if self.provenance.source_key != self.component_key:
            raise ValueError("component provenance source_key must equal component_key")
        return self


class SourceConfig(StrictModel):
    """One local raw source plus technical roles and minimal provenance."""

    source_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    enabled: bool
    task_roles: tuple[Literal["inventory", "t1", "t2", "t3", "t4", "language_global", "language_region"], ...]
    provenance: SourceProvenance
    language_components: tuple[LanguageComponentConfig, ...] = ()

    @model_validator(mode="after")
    def source_and_component_provenance_is_closed(self) -> Self:
        """Bind source identity, frozen component coverage and scientific roles."""

        if self.source_key != self.provenance.source_key:
            raise ValueError("source_key must match provenance.source_key")
        expected_components: dict[str, tuple[str, ...]] = {
            "mmrs_1m": ("rsicd", "ucm", "sydney", "nwpu", "rsitmd", "dior_rsvg"),
            "rsgpt": ("rsicap", "rsieval"),
        }
        expected = expected_components.get(self.source_key, ())
        actual = tuple(component.component for component in self.language_components)
        if actual != expected:
            raise ValueError(f"{self.source_key} language_components must be exactly {expected}")
        if expected:
            if self.task_roles != ("inventory",):
                raise ValueError("aggregate language containers expose only the inventory role")
            for component in self.language_components:
                if component.component_key != f"{self.source_key}:{component.component}":
                    raise ValueError("language component_key must bind source_key and component")
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
    build: BuildSettings
    sources: tuple[SourceConfig, ...]

    _benchmark_path_is_portable = field_validator("benchmark_relative_path")(validate_portable_path)

    @model_validator(mode="after")
    def source_keys_are_unique(self) -> Self:
        """Reject ambiguous source or local-root bindings."""

        keys = [source.source_key for source in self.sources]
        paths = [source.provenance.source_root for source in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("source_key values must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("source provenance roots must be unique")
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
