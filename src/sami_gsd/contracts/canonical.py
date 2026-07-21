"""Strict Canonical Benchmark v3 record contracts.

These models contain no file I/O. Paths are portable logical references and all
spatial boxes use the frozen reference-pixel half-open convention.
"""

from __future__ import annotations

from datetime import date
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
HalfOpenBox = tuple[int, int, int, int]
JsonScalar = str | int | float | bool | None


class StrictModel(BaseModel):
    """Base for immutable, finite and extra-field-forbidding contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def validate_portable_path(value: str) -> str:
    """Return a portable logical path or raise ``ValueError``.

    The contract never embeds machine-absolute paths and never permits parent
    traversal. Runtime root resolution happens outside public records.
    """

    if not value or "\\" in value:
        raise ValueError("path must be a non-empty POSIX logical path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path must be relative and must not contain traversal")
    return value


def validate_half_open_box(value: HalfOpenBox) -> HalfOpenBox:
    """Validate ``[x0, y0, x1, y1)`` ordering without reading image data."""

    x0, y0, x1, y1 = value
    if min(value) < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("half-open box requires non-negative x0< x1 and y0< y1")
    return value


class ArtifactRef(StrictModel):
    """Portable reference to an immutable benchmark asset."""

    path: str
    sha256: Sha256

    _path_is_portable = field_validator("path")(validate_portable_path)


class TransformStep(StrictModel):
    """One recorded spatial transform between two ``(height, width)`` grids.

    ``invertible`` refers to coordinates inside the retained valid-content
    footprint.  Crop and pad are therefore coordinate-invertible for retained
    pixels even though discarded/padded raster content cannot be reconstructed.
    """

    operation: Literal["identity", "crop", "resize", "pad", "affine", "reproject"]
    input_hw: tuple[PositiveInt, PositiveInt]
    output_hw: tuple[PositiveInt, PositiveInt]
    interpolation: Literal["image_bilinear_mask_valid_nearest", "not_applicable"]
    invertible: bool
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_p1_spatial_operation(self) -> Self:
        """Validate the exact auditable contract for P1 crop/resize/pad steps."""

        input_h, input_w = self.input_hw
        parameters = self.parameters
        if self.operation == "identity":
            if self.input_hw != self.output_hw or parameters:
                raise ValueError("identity requires equal grids and empty parameters")
            if self.interpolation != "not_applicable" or not self.invertible:
                raise ValueError("identity must be invertible and use no interpolation")
        elif self.operation == "crop":
            expected = {"top", "left", "height", "width"}
            if set(parameters) != expected:
                raise ValueError("crop parameters must be exactly top, left, height and width")
            top = _required_non_negative_int(parameters, "top")
            left = _required_non_negative_int(parameters, "left")
            height = _required_positive_int(parameters, "height")
            width = _required_positive_int(parameters, "width")
            if (height, width) != self.output_hw:
                raise ValueError("crop height/width must equal output_hw")
            if top + height > input_h or left + width > input_w:
                raise ValueError("crop exceeds input grid")
            if self.interpolation != "not_applicable" or not self.invertible:
                raise ValueError("crop coordinates must be invertible and use no interpolation")
        elif self.operation == "resize":
            expected_parameters = {
                "coordinate_mapping": "pixel_edges",
                "raster_sampling": "half_pixel_centers",
                "raster_border_mode": "clamp",
            }
            if parameters != expected_parameters:
                raise ValueError("resize must record pixel-edge coordinates and half-pixel-center sampling")
            if self.input_hw == self.output_hw:
                raise ValueError("same-size resize must be recorded as identity")
            if self.interpolation != "image_bilinear_mask_valid_nearest" or not self.invertible:
                raise ValueError("resize must use the frozen image/mask/valid interpolation policy")
        elif self.operation == "pad":
            expected = {"top", "bottom", "left", "right", "image_fill", "mask_fill", "valid_fill"}
            if set(parameters) != expected:
                raise ValueError("pad parameters must record offsets and image/mask/valid fill values")
            top = _required_non_negative_int(parameters, "top")
            bottom = _required_non_negative_int(parameters, "bottom")
            left = _required_non_negative_int(parameters, "left")
            right = _required_non_negative_int(parameters, "right")
            if not any((top, bottom, left, right)):
                raise ValueError("zero-width pad must be recorded as identity")
            if (input_h + top + bottom, input_w + left + right) != self.output_hw:
                raise ValueError("pad offsets do not produce output_hw")
            for name in ("image_fill", "mask_fill", "valid_fill"):
                if parameters[name] not in (0, 0.0, False):
                    raise ValueError(f"{name} must be zero so padding cannot become evidence")
            if self.interpolation != "not_applicable" or not self.invertible:
                raise ValueError("pad coordinates must be invertible on valid content and use no interpolation")
        return self


def _required_non_negative_int(parameters: dict[str, JsonScalar], name: str) -> int:
    """Read a non-negative integer transform parameter without bool coercion."""

    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_positive_int(parameters: dict[str, JsonScalar], name: str) -> int:
    """Read a positive integer transform parameter without bool coercion."""

    value = _required_non_negative_int(parameters, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_transform_sequence(steps: tuple[TransformStep, ...]) -> tuple[TransformStep, ...]:
    """Reject empty or spatially discontinuous transform sequences."""

    if not steps:
        raise ValueError("transform chain must contain at least one recorded step")
    for previous, following in zip(steps, steps[1:], strict=False):
        if previous.output_hw != following.input_hw:
            raise ValueError("transform chain contains a discontinuous grid transition")
    return steps


class SourceIdentity(StrictModel):
    """Source-side grouping identifiers used before split assignment."""

    dataset: Annotated[str, Field(min_length=1)]
    record_id: Annotated[str, Field(min_length=1)]
    scene_id: str | None
    event_id: str | None
    region_id: str | None
    source_group_id: Annotated[str, Field(min_length=1)]


class ReferenceCanvas(StrictModel):
    """Frozen spatial canvas shared by masks, boxes and aligned modalities."""

    reference_modality_id: Annotated[str, Field(min_length=1)]
    coordinate_space: Literal["reference_pixel_half_open"]
    original_hw: tuple[PositiveInt, PositiveInt]
    canvas_hw: tuple[PositiveInt, PositiveInt]
    valid_mask_path: str
    transform_chain: tuple[TransformStep, ...]
    inverse_transform_available: bool
    crs: str | None
    geotransform: tuple[float, float, float, float, float, float] | None

    _path_is_portable = field_validator("valid_mask_path")(validate_portable_path)

    @model_validator(mode="after")
    def inverse_flag_matches_chain(self) -> Self:
        """Bind endpoints, continuity and inverse availability to the chain."""

        steps = validate_transform_sequence(self.transform_chain)
        if steps[0].input_hw != self.original_hw or steps[-1].output_hw != self.canvas_hw:
            raise ValueError("reference transform chain endpoints must match original_hw and canvas_hw")
        derived_inverse = all(step.invertible for step in steps)
        if self.inverse_transform_available != derived_inverse:
            raise ValueError("inverse_transform_available must equal the chain's coordinate inverse availability")
        return self


class BandMetadata(StrictModel):
    """Machine-readable metadata for one band or polarization."""

    name: Annotated[str, Field(min_length=1)]
    wavelength_nm: Annotated[float, Field(gt=0.0)] | None
    polarization: str | None
    units: str | None


class NormalizationRecord(StrictModel):
    """Recorded rendering normalization; never a geoscientific fact."""

    method: Literal[
        "none",
        "min_max",
        "z_score",
        "percentile_clip",
        "log_scale",
        "source_defined",
        "unknown",
    ]
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)
    statistics_source: str | None


class QualityRecord(StrictModel):
    """Source quality state and explicit machine-readable flags."""

    status: Literal["unknown", "usable", "degraded", "rejected"]
    flags: tuple[str, ...] = ()
    notes: str | None


class RenderPolicy(StrictModel):
    """Deterministic view-rendering policy for an individual modality."""

    mode: Literal["rgb", "band_composite", "grayscale", "pseudocolor", "normalized_scalar"]
    channels: tuple[str, ...]
    clip_percentiles: tuple[float, float] | None

    @field_validator("clip_percentiles")
    @classmethod
    def ordered_percentiles(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Require an ordered percentile interval inside ``[0, 100]``."""

        if value is not None and not (0.0 <= value[0] < value[1] <= 100.0):
            raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100")
        return value


class ModalityRecord(StrictModel):
    """One physical observation and its source/reference spatial mappings."""

    modality_id: Annotated[str, Field(min_length=1)]
    family: Literal["optical", "multispectral", "sar", "dem", "slope", "insar", "deformation", "other"]
    sensor: Annotated[str, Field(min_length=1)]
    product_type: Annotated[str, Field(min_length=1)]
    band_names: tuple[str, ...]
    band_metadata: tuple[BandMetadata, ...]
    orbit: str | None
    acquisition_time: str | None
    time_range: tuple[str, str] | None
    native_gsd_m: Annotated[float, Field(gt=0.0)] | None
    aligned_gsd_m: Annotated[float, Field(gt=0.0)] | None
    units: str | None
    signed: bool | None
    sign_convention: str | None
    normalization: NormalizationRecord
    quality: QualityRecord
    availability_status: Literal[
        "missing",
        "present_zero_valid",
        "present_partial_valid",
        "present_valid",
    ]
    valid_coverage: Fraction
    native_asset_path: str | None
    aligned_asset_path: str | None
    valid_mask_path: str | None
    source_to_reference_transform: tuple[TransformStep, ...] | None
    reference_to_source_transform: tuple[TransformStep, ...] | None
    alignment_status: Literal["reference", "aligned", "global_only", "unavailable"]
    render_policy: RenderPolicy
    hashes: dict[str, Sha256]

    @field_validator("native_asset_path", "aligned_asset_path", "valid_mask_path")
    @classmethod
    def optional_path_is_portable(cls, value: str | None) -> str | None:
        """Validate present paths while retaining explicit null states."""

        return None if value is None else validate_portable_path(value)

    @model_validator(mode="after")
    def availability_matches_assets(self) -> Self:
        """Keep missing, zero-valid, partial-valid and valid states distinct."""

        status = self.availability_status
        if status == "missing":
            if self.valid_coverage != 0.0 or any(
                path is not None for path in (self.native_asset_path, self.aligned_asset_path, self.valid_mask_path)
            ):
                raise ValueError("missing modality must have zero coverage and no asset paths")
            if self.alignment_status != "unavailable" or any(
                chain is not None
                for chain in (self.source_to_reference_transform, self.reference_to_source_transform)
            ):
                raise ValueError("missing modality must be spatially unavailable with no transform")
            return self
        if self.native_asset_path is None or self.valid_mask_path is None:
            raise ValueError("present modality requires native asset and valid-mask paths")
        if status == "present_zero_valid" and self.valid_coverage != 0.0:
            raise ValueError("present_zero_valid requires zero valid coverage")
        if status == "present_partial_valid" and not 0.0 < self.valid_coverage < 1.0:
            raise ValueError("present_partial_valid requires coverage strictly between zero and one")
        if status == "present_valid" and self.valid_coverage != 1.0:
            raise ValueError("present_valid requires full valid coverage")
        transforms = (self.source_to_reference_transform, self.reference_to_source_transform)
        if self.alignment_status in {"reference", "aligned"}:
            if any(chain is None for chain in transforms):
                raise ValueError("reference/aligned modality requires transforms in both directions")
            for chain in transforms:
                assert chain is not None
                validate_transform_sequence(chain)
        elif any(chain is not None for chain in transforms):
            raise ValueError("global_only/unavailable modality must not expose pixel-level transforms")
        return self


class ReferringRegion(StrictModel):
    """One referring expression bound to a semantic region mask."""

    region_id: Annotated[str, Field(min_length=1)]
    expression: Annotated[str, Field(min_length=1)]
    mask_ref: ArtifactRef
    bbox_half_open: HalfOpenBox
    annotation_origin: Literal["official", "human"]

    _box_is_half_open = field_validator("bbox_half_open")(validate_half_open_box)


class AnnotationRecord(StrictModel):
    """Parent-level semantic and referring supervision references."""

    global_landslide_mask: ArtifactRef | None
    global_mask_origin: Literal["official", "human", "pseudo"] | None
    global_target_status: Literal["positive", "no_target", "unknown"]
    referring_regions: tuple[ReferringRegion, ...]
    no_target_eligibility: bool
    region_fact_refs: tuple[str, ...]

    @model_validator(mode="after")
    def mask_and_origin_match(self) -> Self:
        """A present global mask must retain its source annotation class."""

        if (self.global_landslide_mask is None) != (self.global_mask_origin is None):
            raise ValueError("global mask and global_mask_origin must be present together")
        return self


class ProvenanceRecord(StrictModel):
    """Immutable source identity and derivation summary."""

    source_registry_key: Annotated[str, Field(min_length=1)]
    source_paths: tuple[str, ...]
    source_record_sha256: Sha256
    scanner_version: Annotated[str, Field(min_length=1)]
    derivation_steps: tuple[str, ...]

    @field_validator("source_paths")
    @classmethod
    def source_paths_are_portable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Validate every original logical source path."""

        return tuple(validate_portable_path(path) for path in value)


class LicenseRecord(StrictModel):
    """Source-license snapshot copied into a canonical parent record."""

    source_key: Annotated[str, Field(min_length=1)]
    license_status: Literal["verified", "restricted", "unknown"]
    license_name: Annotated[str, Field(min_length=1)]
    license_url_or_document: str | None
    allowed_for_training: bool
    allowed_for_evaluation: bool
    allowed_for_redistribution: bool
    academic_only: bool
    attribution: Annotated[str, Field(min_length=1)]
    reviewed_by: str | None
    review_date: date | None

    @field_validator("license_url_or_document")
    @classmethod
    def license_evidence_is_portable_or_https(cls, value: str | None) -> str | None:
        """Accept a portable local document or an explicit HTTPS evidence URL."""

        if value is None:
            return None
        if value.startswith("https://"):
            return value
        if "://" in value:
            raise ValueError("license evidence URL must use HTTPS")
        return validate_portable_path(value)

    @model_validator(mode="after")
    def unknown_or_unreviewed_never_trains(self) -> Self:
        """Fail closed for unknown or unreviewed training sources."""

        unknown = self.license_status == "unknown" or self.license_name.lower() == "unknown"
        if self.allowed_for_training and (unknown or not self.reviewed_by or self.review_date is None):
            raise ValueError("training eligibility requires a reviewed, non-unknown license")
        return self


class HashRecord(StrictModel):
    """Source and materialized-asset hashes excluding the enclosing record."""

    source_record_sha256: Sha256
    assets: dict[str, Sha256]


class CanonicalParentV3(StrictModel):
    """Canonical parent record on one reference canvas."""

    schema_version: Literal["sami_canonical_parent_v3"]
    parent_id: Annotated[str, Field(min_length=1)]
    source: SourceIdentity
    split: Literal["train", "val", "test", "audit"]
    reference_canvas: ReferenceCanvas
    modalities: tuple[ModalityRecord, ...]
    annotations: AnnotationRecord
    provenance: ProvenanceRecord
    license: LicenseRecord
    hashes: HashRecord
    annotation_status: Literal["unlabeled", "auto", "silver", "gold"]

    @model_validator(mode="after")
    def modality_ids_are_unique_and_reference_exists(self) -> Self:
        """Bind the reference canvas to exactly one declared reference view."""

        modality_ids = [modality.modality_id for modality in self.modalities]
        if len(modality_ids) != len(set(modality_ids)):
            raise ValueError("modality_id values must be unique within a parent")
        if self.reference_canvas.reference_modality_id not in modality_ids:
            raise ValueError("reference_modality_id must name a declared modality")
        reference_modalities = [modality for modality in self.modalities if modality.alignment_status == "reference"]
        if len(reference_modalities) != 1:
            raise ValueError("canonical parent must declare exactly one reference-aligned modality")
        if reference_modalities[0].modality_id != self.reference_canvas.reference_modality_id:
            raise ValueError("reference_modality_id must name the reference-aligned modality")
        return self


class RegionGeometry(StrictModel):
    """Optional task-region geometry on the frozen reference canvas."""

    coordinate_space: Literal["reference_pixel_half_open"]
    region_id: str | None
    bbox_half_open: HalfOpenBox | None

    @field_validator("bbox_half_open")
    @classmethod
    def optional_box_is_half_open(cls, value: HalfOpenBox | None) -> HalfOpenBox | None:
        """Validate a present region box."""

        return None if value is None else validate_half_open_box(value)


class TaskViewV3(StrictModel):
    """A task view derived only after its parent split is frozen."""

    task_id: Annotated[str, Field(min_length=1)]
    parent_id: Annotated[str, Field(min_length=1)]
    task_type: Literal["t1_global", "t2_referring", "t3_gt_region", "t4_predicted_region"]
    instruction: Annotated[str, Field(min_length=1)]
    target_status: Literal["positive", "no_target", "unknown"]
    region_geometry: RegionGeometry | None
    target_mask_ref: ArtifactRef | None
    target_box_ref: HalfOpenBox | None
    answer_ref: ArtifactRef | None
    annotation_origin: Literal[
        "official",
        "human",
        "expert",
        "pseudo",
        "oof_prediction",
        "online_prediction",
        "derived_no_target",
    ]
    weight: Annotated[float, Field(ge=0.0)]

    @field_validator("target_box_ref")
    @classmethod
    def optional_target_box_is_half_open(cls, value: HalfOpenBox | None) -> HalfOpenBox | None:
        """Validate a present target box."""

        return None if value is None else validate_half_open_box(value)

    @model_validator(mode="after")
    def task_geometry_matches_type(self) -> Self:
        """Prevent region tasks from silently losing their spatial binding."""

        if self.task_type in {"t2_referring", "t3_gt_region", "t4_predicted_region"} and self.region_geometry is None:
            raise ValueError("region/referring task requires region_geometry")
        if self.target_status == "positive" and self.target_mask_ref is None:
            raise ValueError("positive task requires target_mask_ref")
        return self
