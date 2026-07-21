"""Strict configuration contracts for the P2 SAMI model skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from sami_gsd.contracts.canonical import Sha256, StrictModel, validate_portable_path


class QwenModelSettings(StrictModel):
    """Frozen official Qwen3-VL loading policy for the greenfield wrapper."""

    backend: Literal["qwen3_vl_official"]
    family: Literal["Qwen3-VL-2B"]
    model_path: str
    local_files_only: Literal[True]
    trust_remote_code: Literal[False]
    dtype: Literal["bfloat16", "float32"]
    attention_implementation: Literal["sdpa", "eager"]
    frozen_vision_tower: Literal[True]

    _model_path_is_portable = field_validator("model_path")(validate_portable_path)


class PixelBudgetProfile(StrictModel):
    """One pre-registered native multi-image pixel/view budget."""

    profile: Literal["S", "M"]
    reference_max_pixels: Annotated[int, Field(gt=0)]
    support_max_pixels: Annotated[int, Field(gt=0)]
    max_views: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def values_match_frozen_profile(self) -> Self:
        """Reject an unregistered profile hidden behind the S/M name."""

        expected = {
            "S": (512 * 512, 384 * 384, 4),
            "M": (768 * 768, 448 * 448, 6),
        }[self.profile]
        actual = (self.reference_max_pixels, self.support_max_pixels, self.max_views)
        if actual != expected:
            raise ValueError(f"profile {self.profile} must use the frozen budget {expected}")
        return self


class ProcessorSettings(StrictModel):
    """Official processor and deterministic task-neutral prompt policy."""

    use_official_chat_template: Literal[True]
    processor_min_pixels: Annotated[int, Field(gt=0)]
    patch_size: Literal[16]
    spatial_merge_size: Literal[2]
    prompt_version: Literal["sami_sensor_cards_v1_task_neutral"]
    system_prompt: Annotated[str, Field(min_length=1)]
    user_instruction: Annotated[str, Field(min_length=1)]


class CacheEquivalenceSettings(StrictModel):
    """Numerical thresholds for transparent memoization."""

    cosine_similarity_min: Annotated[float, Field(ge=0.0, le=1.0)]
    fp32_max_abs: Annotated[float, Field(gt=0.0)]
    bf16_max_abs: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def thresholds_match_protocol(self) -> Self:
        """Keep P2 cache acceptance equal to the governing task specification."""

        if self.cosine_similarity_min != 0.9999:
            raise ValueError("cosine_similarity_min must be exactly 0.9999")
        if self.fp32_max_abs != 1e-4 or self.bf16_max_abs != 5e-3:
            raise ValueError("cache max-absolute thresholds must be FP32=1e-4 and BF16=5e-3")
        return self


class CacheSettings(StrictModel):
    """Write-once P2 cache policy; old cache formats are never accepted."""

    schema_version: Literal["sami_qwen_backbone_cache_v1"]
    enabled_by_default: bool
    relative_root: str
    equivalence: CacheEquivalenceSettings

    _root_is_portable = field_validator("relative_root")(validate_portable_path)


class BenchmarkBinding(StrictModel):
    """Accepted Canonical Benchmark v3 identity consumed by P2 smoke."""

    schema_version: Literal["sami_benchmark_manifest_v1"]
    mode: Literal["small"]
    relative_path: Literal["sami_landslide_v3/small"]
    aggregate_sha256: Sha256
    validation_aggregate_sha256: Sha256


class ModelSmokeSettings(StrictModel):
    """Bounded one-forward reporting policy."""

    profile_s_peak_limit_gib: Annotated[float, Field(gt=0.0)]
    output_schema_version: Literal["sami_p2_model_smoke_report_v1"]

    @field_validator("profile_s_peak_limit_gib")
    @classmethod
    def memory_limit_is_frozen(cls, value: float) -> float:
        """Reserve roughly 2 GiB on the target 24 GiB device."""

        if value != 22.0:
            raise ValueError("Profile S peak limit must be exactly 22.0 GiB")
        return value


class SamiModelConfig(StrictModel):
    """Single YAML source of truth for the P2 minimum model skeleton."""

    schema_version: Literal["sami_model_config_v1"]
    model: QwenModelSettings
    processor: ProcessorSettings
    active_profile: Literal["S", "M"]
    pixel_budgets: tuple[PixelBudgetProfile, PixelBudgetProfile]
    cache: CacheSettings
    benchmark: BenchmarkBinding
    smoke: ModelSmokeSettings

    @model_validator(mode="after")
    def profiles_are_complete_and_ordered(self) -> Self:
        """Require exactly one S then one M profile and a declared active profile."""

        if tuple(profile.profile for profile in self.pixel_budgets) != ("S", "M"):
            raise ValueError("pixel_budgets must contain exactly ordered profiles S then M")
        return self

    def active_pixel_budget(self) -> PixelBudgetProfile:
        """Return the selected registered pixel budget."""

        return next(profile for profile in self.pixel_budgets if profile.profile == self.active_profile)


def load_model_config(path: Path) -> SamiModelConfig:
    """Read one strict UTF-8 P2 model YAML without resolving runtime paths."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model config must be a non-empty YAML mapping")
    return SamiModelConfig.model_validate(payload)


__all__ = [
    "BenchmarkBinding",
    "CacheEquivalenceSettings",
    "CacheSettings",
    "ModelSmokeSettings",
    "PixelBudgetProfile",
    "ProcessorSettings",
    "QwenModelSettings",
    "SamiModelConfig",
    "load_model_config",
]
