"""Typed reference-canvas decisions for Canonical Benchmark v3.

The contracts are file-I/O free.  They make reference selection evidence and
T1--T4 spatial eligibility serializable instead of leaving them as implicit
dataset-reader behavior.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from sami_gsd.contracts.canonical import Fraction, PositiveInt, Sha256, StrictModel


class ReferenceCanvasCandidate(StrictModel):
    """One source grid considered during deterministic reference selection."""

    modality_id: Annotated[str, Field(min_length=1)]
    original_hw: tuple[PositiveInt, PositiveInt]
    mask_grid: Literal["native", "registered", "none"]
    annotation_origin: Literal["official", "human"] | None
    valid_coverage: Fraction
    native_gsd_m: Annotated[float, Field(gt=0.0)] | None
    single_image_language: bool
    coordinate_inverse_available: bool

    @model_validator(mode="after")
    def mask_metadata_is_complete(self) -> Self:
        """Keep spatial supervision distinct from a language-only image."""

        if (self.mask_grid == "none") != (self.annotation_origin is None):
            raise ValueError("mask_grid and annotation_origin must either both describe a mask or both be absent")
        if self.single_image_language and self.mask_grid != "none":
            raise ValueError("single-image language candidates must not claim a spatial mask grid")
        return self


class ReferenceCanvasDecision(StrictModel):
    """Auditable result of applying the frozen reference-canvas priorities."""

    reference_modality_id: Annotated[str, Field(min_length=1)]
    original_hw: tuple[PositiveInt, PositiveInt]
    selection_rule: Literal[
        "authoritative_native_mask",
        "registered_mask_complete_finest_gsd",
        "single_image_original",
    ]
    considered_modality_ids: tuple[str, ...]
    candidate_set_sha256: Sha256
    inverse_transform_available: bool
    spatial_tasks_eligible: bool
    spatial_exclusion_reason: Literal["coordinate_inverse_unavailable"] | None

    @model_validator(mode="after")
    def eligibility_matches_inverse(self) -> Self:
        """T1--T4 eligibility requires an explicit coordinate inverse."""

        if self.reference_modality_id not in self.considered_modality_ids:
            raise ValueError("reference modality must be present in considered_modality_ids")
        if tuple(sorted(self.considered_modality_ids)) != self.considered_modality_ids:
            raise ValueError("considered_modality_ids must use stable lexical order")
        if len(set(self.considered_modality_ids)) != len(self.considered_modality_ids):
            raise ValueError("considered_modality_ids must be unique")
        if self.spatial_tasks_eligible != self.inverse_transform_available:
            raise ValueError("spatial_tasks_eligible must equal coordinate inverse availability")
        expected_reason = None if self.spatial_tasks_eligible else "coordinate_inverse_unavailable"
        if self.spatial_exclusion_reason != expected_reason:
            raise ValueError("spatial_exclusion_reason must describe inverse unavailability exactly")
        return self


__all__ = ["ReferenceCanvasCandidate", "ReferenceCanvasDecision"]
