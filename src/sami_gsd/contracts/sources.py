"""Strict P1.3 contracts for raw-source evidence and canonical candidates.

These audit records are deliberately not Canonical Parent v3 records. They
capture enough source-side evidence to decide whether a source can later be
materialized without turning audit projections into runtime training rows.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from sami_gsd.contracts.canonical import Sha256, StrictModel, validate_portable_path
from sami_gsd.contracts.spatial import ReferenceCanvasDecision


class RawAssetEvidence(StrictModel):
    """One immutable source asset or one indexed virtual-array asset."""

    asset_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")]
    role: Literal["reference_image", "support_image", "mask", "annotation_index"]
    logical_path: str
    container: Literal["png", "jpeg", "tiff", "npy", "hdf5", "netcdf", "json", "jsonl"]
    internal_key: str | int | None
    sample_index: Annotated[int, Field(ge=0)] | None
    byte_size: Annotated[int, Field(ge=0)]
    sha256: Sha256
    array_shape: tuple[Annotated[int, Field(gt=0)], ...] | None
    native_hw: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]] | None
    channel_count: Annotated[int, Field(gt=0)] | None
    dtype: str | None
    band_names: tuple[str, ...]
    metadata_evidence: tuple[str, ...]
    crs: str | None = None
    geotransform: tuple[float, float, float, float, float, float] | None = None
    nodata_values: tuple[float | int | None, ...] = ()
    metadata_attributes: tuple[tuple[str, str], ...] = ()

    _logical_path_is_portable = field_validator("logical_path")(validate_portable_path)

    @model_validator(mode="after")
    def bind_spatial_and_index_metadata(self) -> Self:
        """Reject underspecified image/mask assets and malformed virtual rows."""

        if self.role in {"reference_image", "support_image", "mask"} and self.native_hw is None:
            raise ValueError("spatial asset evidence requires native_hw")
        if self.container == "npy":
            if self.array_shape is None or self.sample_index is None or self.dtype is None:
                raise ValueError("NPY virtual assets require shape, dtype and sample_index")
            if self.sample_index >= self.array_shape[0]:
                raise ValueError("NPY sample_index is outside the first array dimension")
        elif self.container in {"hdf5", "netcdf"}:
            if self.array_shape is None or self.dtype is None or self.internal_key is None:
                raise ValueError("HDF5/NetCDF assets require shape, dtype and internal_key")
            if self.sample_index is not None and self.sample_index >= self.array_shape[0]:
                raise ValueError("container sample_index is outside the first array dimension")
        elif self.sample_index is not None:
            raise ValueError("sample_index is reserved for virtual array assets")
        if self.role == "annotation_index" and self.native_hw is not None:
            raise ValueError("annotation indexes must not claim a raster grid")
        if self.channel_count is not None and self.band_names and self.channel_count != len(self.band_names):
            raise ValueError("channel_count must match the explicit band_names length")
        if self.nodata_values and self.channel_count is not None and len(self.nodata_values) != self.channel_count:
            raise ValueError("nodata_values must be empty or match channel_count")
        if tuple(sorted(self.metadata_attributes)) != self.metadata_attributes:
            raise ValueError("metadata_attributes must be sorted for deterministic hashing")
        return self


class RawSourceRecord(StrictModel):
    """A source-side record frozen to audit-only use."""

    schema_version: Literal["sami_raw_source_record_v1"]
    source_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    record_id: Annotated[str, Field(min_length=1)]
    source_group_id: Annotated[str, Field(min_length=1)]
    source_declared_split: str | None
    split: Literal["audit"]
    record_type: Literal["spatial_mask", "derived_spatial_mask", "global_language", "region_language"]
    assets: tuple[RawAssetEvidence, ...]
    grouping_evidence: tuple[str, ...]
    ambiguity_flags: tuple[str, ...]
    asset_set_sha256: Sha256

    @model_validator(mode="after")
    def bind_assets(self) -> Self:
        """Enforce a non-empty unique source asset set."""

        asset_ids = [asset.asset_id for asset in self.assets]
        if not asset_ids or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("raw source record requires non-empty unique asset_id values")
        return self


class ModalityCandidate(StrictModel):
    """Source metadata projected toward, but not promoted into, ModalityRecord."""

    modality_id: Annotated[str, Field(min_length=1)]
    family: Literal["optical", "multispectral", "sar", "dem", "slope", "insar", "deformation", "other"]
    sensor: Annotated[str, Field(min_length=1)]
    product_type: Annotated[str, Field(min_length=1)]
    band_names: tuple[str, ...]
    native_asset_id: Annotated[str, Field(min_length=1)]
    native_hw: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    native_gsd_m: Annotated[float, Field(gt=0.0)] | None
    units: str | None
    signed: bool | None
    sign_convention: str | None
    alignment_status: Literal["reference", "aligned", "global_only", "unresolved"]
    alignment_evidence: tuple[str, ...]
    valid_status: Literal["unresolved", "source_nodata_only", "explicit_valid_mask"]


class AnnotationCandidate(StrictModel):
    """Unmaterialized mask or language-region evidence for one candidate."""

    global_mask_asset_id: str | None
    target_status: Literal["unknown"]
    normalized_box_xyxy: tuple[float, float, float, float] | None
    phrase: str | None
    annotation_origin: Literal["official", "source_caption", "source_expression", "derived"]

    @model_validator(mode="after")
    def validate_optional_region_pair(self) -> Self:
        """Require a strict normalized box whenever a region phrase is present."""

        if (self.normalized_box_xyxy is None) != (self.phrase is None):
            raise ValueError("normalized region box and phrase must be present together")
        if self.normalized_box_xyxy is not None:
            x0, y0, x1, y1 = self.normalized_box_xyxy
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError("normalized_box_xyxy must satisfy 0<=min<max<=1")
        return self


class CanonicalParentCandidate(StrictModel):
    """Audit-only projection that may later materialize into CanonicalParentV3."""

    schema_version: Literal["sami_canonical_parent_candidate_v1"]
    parent_id: Annotated[str, Field(min_length=1)]
    source_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    record_id: Annotated[str, Field(min_length=1)]
    source_group_id: Annotated[str, Field(min_length=1)]
    split: Literal["audit"]
    task_roles: tuple[Literal["t1", "t2", "t3", "t4", "language_global", "language_region"], ...]
    reference_decision: ReferenceCanvasDecision
    modalities: tuple[ModalityCandidate, ...]
    annotations: AnnotationCandidate
    raw_asset_set_sha256: Sha256
    materialization_status: Literal["ready", "blocked"]
    blockers: tuple[str, ...]

    @model_validator(mode="after")
    def bind_candidate_identity_and_reference(self) -> Self:
        """Keep the projection self-consistent without silently promoting it."""

        modality_ids = [modality.modality_id for modality in self.modalities]
        if not modality_ids or len(modality_ids) != len(set(modality_ids)):
            raise ValueError("candidate requires non-empty unique modalities")
        references = [modality for modality in self.modalities if modality.alignment_status == "reference"]
        if len(references) != 1 or references[0].modality_id != self.reference_decision.reference_modality_id:
            raise ValueError("candidate must bind exactly one modality to the reference decision")
        expected = "blocked" if self.blockers else "ready"
        if self.materialization_status != expected:
            raise ValueError("materialization_status must be derived from technical blockers")
        return self


class SourceSampleProjection(StrictModel):
    """One strict raw record and its deterministic canonical projection."""

    schema_version: Literal["sami_source_sample_projection_v1"]
    raw_record: RawSourceRecord
    canonical_candidate: CanonicalParentCandidate
    projection_sha256: Sha256

    @model_validator(mode="after")
    def raw_and_candidate_match(self) -> Self:
        """Bind source identity and asset fingerprints across both layers."""

        raw = self.raw_record
        candidate = self.canonical_candidate
        if (raw.source_key, raw.record_id, raw.source_group_id) != (
            candidate.source_key,
            candidate.record_id,
            candidate.source_group_id,
        ):
            raise ValueError("raw and canonical candidate identities do not match")
        if raw.asset_set_sha256 != candidate.raw_asset_set_sha256:
            raise ValueError("raw and canonical candidate asset fingerprints do not match")
        return self


__all__ = [
    "AnnotationCandidate",
    "CanonicalParentCandidate",
    "ModalityCandidate",
    "RawAssetEvidence",
    "RawSourceRecord",
    "SourceSampleProjection",
]
