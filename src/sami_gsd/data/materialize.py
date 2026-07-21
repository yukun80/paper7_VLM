"""Canonical Benchmark v3 preprocessing and atomic asset materialization.

This module is the array-level P1 boundary. Raw-source adapters must decode an
explicit HWC array and binary valid/mask rasters before calling it; the
materializer never guesses layout, nodata, units, sign, grouping or license.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sami_gsd.contracts.canonical import (
    AnnotationRecord,
    ArtifactRef,
    BandMetadata,
    CanonicalParentV3,
    HashRecord,
    LicenseRecord,
    ModalityRecord,
    NormalizationRecord,
    ProvenanceRecord,
    QualityRecord,
    ReferenceCanvas,
    ReferringRegion,
    RenderPolicy,
    SourceIdentity,
    TransformStep,
)
from sami_gsd.contracts.language import DescriptionSourceRecord
from sami_gsd.data.adapters.formats import read_image_header
from sami_gsd.data.transforms import (
    build_transform_chain,
    crop_step,
    identity_step,
    pad_step,
    quantize_covering_box,
    resize_step,
)
from sami_gsd.utilities.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_bytes


MATERIALIZER_VERSION = "sami_canonical_materializer_v3_component_license_bound"


class MaterializationError(ValueError):
    """Raised when a source array cannot be materialized without guessing."""


@dataclass(frozen=True)
class SourceModalityInput:
    """One explicit decoded source modality on the reference-aligned grid.

    ``array`` must be finite HWC after invalid pixels are masked by ``valid``.
    Both objects are NumPy-compatible CPU arrays; layout is never inferred.
    """

    modality_id: str
    family: Literal["optical", "multispectral", "sar", "dem", "slope", "insar", "deformation", "other"]
    sensor: str
    product_type: str
    band_names: tuple[str, ...]
    array: Any
    valid: Any
    source_logical_path: str
    source_sha256: str
    units: str | None = None
    signed: bool | None = None
    sign_convention: str | None = None
    native_gsd_m: float | None = None
    orbit: str | None = None
    acquisition_time: str | None = None
    crs: str | None = None
    geotransform: tuple[float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class SourceReferringInput:
    """One official/human referring mask on the reference source grid."""

    region_id: str
    expression: str
    mask: Any
    annotation_origin: Literal["official", "human"]


@dataclass(frozen=True)
class SpatialParentInput:
    """Fully resolved source record accepted for deterministic materialization."""

    parent_id: str
    source: SourceIdentity
    reference_modality_id: str
    modalities: tuple[SourceModalityInput, ...]
    global_mask: Any
    global_mask_origin: Literal["official", "human", "pseudo"]
    referring_regions: tuple[SourceReferringInput, ...]
    license: LicenseRecord
    source_record_sha256: str
    annotation_status: Literal["auto", "silver", "gold"]


@dataclass(frozen=True)
class MaterializedParent:
    """Validated audit-split parent plus materialization statistics."""

    parent: CanonicalParentV3
    valid_pixel_count: int
    excluded_pixel_count: int
    positive_valid_pixel_count: int


@dataclass(frozen=True)
class LanguageParentInput:
    """Exact-image language records resolved to one canonical visual parent."""

    parent_id: str
    records: tuple[DescriptionSourceRecord, ...]
    raw_image_path: Path


@dataclass(frozen=True)
class MaterializedLanguageParent:
    """One unlabeled canonical image parent shared by language targets."""

    parent: CanonicalParentV3
    source_record_ids: tuple[str, ...]
    valid_pixel_count: int
    excluded_pixel_count: int


def _numpy() -> Any:
    """Load the declared data dependency with an actionable failure."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - minimal installs
        raise MaterializationError("materialization requires the sami-groundsegdesc[data] extra") from error
    return np


def build_fit_pad_transform(
    input_hw: tuple[int, int],
    canvas_hw: tuple[int, int],
) -> tuple[TransformStep, ...]:
    """Fit the full source grid inside a fixed canvas and symmetrically zero-pad."""

    input_h, input_w = input_hw
    canvas_h, canvas_w = canvas_hw
    if min(input_h, input_w, canvas_h, canvas_w) <= 0:
        raise MaterializationError("input and canvas grids must be positive")
    scale = min(canvas_h / input_h, canvas_w / input_w)
    resized_h = min(canvas_h, max(1, int(math.floor(input_h * scale + 0.5))))
    resized_w = min(canvas_w, max(1, int(math.floor(input_w * scale + 0.5))))
    steps: list[TransformStep] = []
    current_hw = input_hw
    if (resized_h, resized_w) != current_hw:
        steps.append(resize_step(current_hw, (resized_h, resized_w)))
        current_hw = (resized_h, resized_w)
    top = (canvas_h - resized_h) // 2
    bottom = canvas_h - resized_h - top
    left = (canvas_w - resized_w) // 2
    right = canvas_w - resized_w - left
    if any((top, bottom, left, right)):
        steps.append(pad_step(current_hw, top=top, bottom=bottom, left=left, right=right))
    if not steps:
        steps.append(identity_step(input_hw))
    return build_transform_chain(steps)


def invert_fit_pad_transform(steps: tuple[TransformStep, ...]) -> tuple[TransformStep, ...]:
    """Build the explicit coordinate inverse for identity/resize/pad chains."""

    inverse: list[TransformStep] = []
    for step in reversed(steps):
        if step.operation == "identity":
            inverse.append(identity_step(step.output_hw))
        elif step.operation == "resize":
            inverse.append(resize_step(step.output_hw, step.input_hw))
        elif step.operation == "pad":
            inverse.append(
                crop_step(
                    step.output_hw,
                    top=int(step.parameters["top"]),
                    left=int(step.parameters["left"]),
                    height=step.input_hw[0],
                    width=step.input_hw[1],
                )
            )
        else:
            raise MaterializationError(f"no materializer inverse for operation: {step.operation}")
    return build_transform_chain(inverse)


def transform_geotransform(
    geotransform: tuple[float, float, float, float, float, float] | None,
    steps: tuple[TransformStep, ...],
) -> tuple[float, float, float, float, float, float] | None:
    """Project one GDAL affine tuple onto the fit/pad reference canvas."""

    if geotransform is None:
        return None
    origin_x, pixel_x, row_x, origin_y, pixel_y, row_y = geotransform
    source_x_at_output_zero = 0.0
    source_y_at_output_zero = 0.0
    scale_x = 1.0
    scale_y = 1.0
    for step in steps:
        if step.operation == "resize":
            scale_x *= step.input_hw[1] / step.output_hw[1]
            scale_y *= step.input_hw[0] / step.output_hw[0]
        elif step.operation == "pad":
            source_x_at_output_zero -= float(step.parameters["left"]) * scale_x
            source_y_at_output_zero -= float(step.parameters["top"]) * scale_y
        elif step.operation != "identity":
            raise MaterializationError(f"geotransform projection does not support: {step.operation}")
    return (
        origin_x + pixel_x * source_x_at_output_zero + row_x * source_y_at_output_zero,
        pixel_x * scale_x,
        row_x * scale_y,
        origin_y + pixel_y * source_x_at_output_zero + row_y * source_y_at_output_zero,
        pixel_y * scale_x,
        row_y * scale_y,
    )


def _validate_arrays(array: Any, valid: Any, *, modality_id: str) -> tuple[Any, Any]:
    """Normalize explicit HWC/binary arrays and reject valid non-finite values."""

    np = _numpy()
    image = np.asarray(array)
    valid_array = np.asarray(valid)
    if image.ndim != 3:
        raise MaterializationError(f"{modality_id} array must be explicit HWC")
    if valid_array.ndim != 2 or valid_array.shape != image.shape[:2]:
        raise MaterializationError(f"{modality_id} valid mask must be HW and match the image")
    if not np.issubdtype(image.dtype, np.number):
        raise MaterializationError(f"{modality_id} array dtype must be numeric")
    if not np.all((valid_array == 0) | (valid_array == 1)):
        raise MaterializationError(f"{modality_id} valid mask must be binary")
    valid_bool = valid_array.astype(bool, copy=False)
    if np.any(~np.isfinite(image[valid_bool])):
        raise MaterializationError(f"{modality_id} has non-finite values inside its valid domain")
    normalized = image.astype("<f4", copy=True)
    normalized[~valid_bool] = 0.0
    return normalized, valid_bool.astype("u1", copy=False)


def _resize_image(image: Any, output_hw: tuple[int, int]) -> Any:
    """Vectorized bilinear half-pixel-center resize for explicit HWC arrays."""

    np = _numpy()
    input_h, input_w, _ = image.shape
    output_h, output_w = output_hw
    source_y = np.clip((np.arange(output_h) + 0.5) * input_h / output_h - 0.5, 0.0, input_h - 1)
    source_x = np.clip((np.arange(output_w) + 0.5) * input_w / output_w - 0.5, 0.0, input_w - 1)
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(input_h - 1, y0 + 1)
    x1 = np.minimum(input_w - 1, x0 + 1)
    wy = (source_y - y0).astype("<f4")[:, None, None]
    wx = (source_x - x0).astype("<f4")[None, :, None]
    top = (1.0 - wx) * image[y0[:, None], x0[None, :]] + wx * image[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx) * image[y1[:, None], x0[None, :]] + wx * image[y1[:, None], x1[None, :]]
    return ((1.0 - wy) * top + wy * bottom).astype("<f4", copy=False)


def _resize_binary(binary: Any, output_hw: tuple[int, int]) -> Any:
    """Vectorized nearest resize using the frozen integer center mapping."""

    np = _numpy()
    input_h, input_w = binary.shape
    output_h, output_w = output_hw
    y = np.minimum(input_h - 1, ((2 * np.arange(output_h) + 1) * input_h) // (2 * output_h))
    x = np.minimum(input_w - 1, ((2 * np.arange(output_w) + 1) * input_w) // (2 * output_w))
    return binary[y[:, None], x[None, :]].astype("u1", copy=False)


def _apply_image(image: Any, steps: tuple[TransformStep, ...]) -> Any:
    """Apply the supported transform chain to one HWC image."""

    np = _numpy()
    result = image
    for step in steps:
        if tuple(result.shape[:2]) != step.input_hw:
            raise MaterializationError("image shape disagrees with transform chain")
        if step.operation == "resize":
            result = _resize_image(result, step.output_hw)
        elif step.operation == "pad":
            result = np.pad(
                result,
                (
                    (int(step.parameters["top"]), int(step.parameters["bottom"])),
                    (int(step.parameters["left"]), int(step.parameters["right"])),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0.0,
            )
        elif step.operation != "identity":
            raise MaterializationError(f"unsupported materialization operation: {step.operation}")
    return result.astype("<f4", copy=False)


def _apply_binary(binary: Any, steps: tuple[TransformStep, ...]) -> Any:
    """Apply nearest resize and zero padding to one binary HW raster."""

    np = _numpy()
    result = binary.astype("u1", copy=False)
    for step in steps:
        if tuple(result.shape) != step.input_hw:
            raise MaterializationError("binary raster shape disagrees with transform chain")
        if step.operation == "resize":
            result = _resize_binary(result, step.output_hw)
        elif step.operation == "pad":
            result = np.pad(
                result,
                (
                    (int(step.parameters["top"]), int(step.parameters["bottom"])),
                    (int(step.parameters["left"]), int(step.parameters["right"])),
                ),
                mode="constant",
                constant_values=0,
            )
        elif step.operation != "identity":
            raise MaterializationError(f"unsupported materialization operation: {step.operation}")
    if not np.all((result == 0) | (result == 1)):
        raise MaterializationError("binary transform produced a non-binary raster")
    return result.astype("u1", copy=False)


def _npy_bytes(array: Any) -> bytes:
    """Serialize a standardized array without pickle or non-deterministic metadata."""

    np = _numpy()
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_array(root: Path, logical_path: str, array: Any) -> ArtifactRef:
    """Atomically publish one deterministic NPY artifact and return its binding."""

    content = _npy_bytes(array)
    atomic_write_bytes(root / logical_path, content)
    return ArtifactRef(path=logical_path, sha256=sha256_bytes(content))


def _decode_language_rgb(path: Path, *, expected_hw: tuple[int, int]) -> tuple[Any, Any, str, bytes]:
    """Decode one signature-checked PNG/JPEG without applying hidden geometry."""

    np = _numpy()
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - minimal installs
        raise MaterializationError("language materialization requires the sami-groundsegdesc[data] extra") from error
    header = read_image_header(path)
    raw_bytes = path.read_bytes()
    suffix = "png" if header.container == "png" else "jpg"
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image.load()
        if (image.height, image.width) != expected_hw or expected_hw != (header.height, header.width):
            raise MaterializationError("language image grid differs from its frozen source record")
        has_transparency = "A" in image.getbands() or "transparency" in image.info
        if has_transparency:
            rgba = np.asarray(image.convert("RGBA"), dtype="u1")
            rgb = rgba[..., :3]
            valid = (rgba[..., 3] > 0).astype("u1")
        else:
            rgb = np.asarray(image.convert("RGB"), dtype="u1")
            valid = np.ones(expected_hw, dtype="u1")
    if not valid.any():
        raise MaterializationError("language reference image has zero valid coverage")
    return rgb, valid, suffix, raw_bytes


def materialize_language_parent(
    source: LanguageParentInput,
    *,
    benchmark_root: Path,
    canvas_hw: tuple[int, int],
) -> MaterializedLanguageParent:
    """Materialize one exact-image language parent without fabricating masks.

    Caption and region-phrase targets are published separately after the
    duplicate-connected parent split is frozen. This parent therefore remains
    spatially unlabeled and cannot create T1/T2 supervision by itself.
    """

    records = tuple(sorted(source.records, key=lambda item: item.record_id))
    if not records or len({record.record_id for record in records}) != len(records):
        raise MaterializationError("language parent requires non-empty unique source records")
    first = records[0]
    if any(record.source_key != first.source_key for record in records):
        raise MaterializationError("one language parent cannot combine source-license keys")
    if any(record.image.sha256 != first.image.sha256 or record.image.native_hw != first.image.native_hw for record in records):
        raise MaterializationError("language parent records must bind one exact source image")
    if any(record.license != first.license for record in records):
        raise MaterializationError("language parent records carry conflicting license snapshots")
    if any(record.training_eligible for record in records) and not first.license.allowed_for_training:
        raise MaterializationError("training language parent requires an approved training license")
    if any(record.split_policy == "permanent_test_only" for record in records) and not first.license.allowed_for_evaluation:
        raise MaterializationError("permanent-test language parent requires approved evaluation use")

    rgb, native_valid, suffix, raw_bytes = _decode_language_rgb(
        source.raw_image_path,
        expected_hw=first.image.native_hw,
    )
    if sha256_bytes(raw_bytes) != first.image.sha256:
        raise MaterializationError("language image bytes changed after subset selection")
    native, valid = _validate_arrays(rgb, native_valid, modality_id="reference_image")
    original_hw = tuple(int(value) for value in native.shape[:2])
    chain = build_fit_pad_transform(original_hw, canvas_hw)
    inverse_chain = invert_fit_pad_transform(chain)
    aligned = _apply_image(native, chain)
    aligned_valid = _apply_binary(valid, chain)
    aligned[aligned_valid == 0] = 0.0
    parent_directory = f"assets/{source.parent_id}"
    native_path = f"{parent_directory}/reference_image.native.{suffix}"
    atomic_write_bytes(benchmark_root / native_path, raw_bytes)
    native_ref = ArtifactRef(path=native_path, sha256=first.image.sha256)
    aligned_ref = _write_array(
        benchmark_root,
        f"{parent_directory}/reference_image.aligned.npy",
        aligned,
    )
    valid_ref = _write_array(
        benchmark_root,
        f"{parent_directory}/reference_image.valid.npy",
        aligned_valid,
    )
    valid_count = int(aligned_valid.sum())
    total_count = int(aligned_valid.size)
    coverage = float(valid_count / total_count)
    availability = "present_valid" if coverage == 1.0 else "present_partial_valid"
    source_record_sha256 = sha256_bytes(
        canonical_json_bytes([record.model_dump(mode="json") for record in records])
    )
    source_paths = tuple(
        sorted(
            {record.image.logical_path for record in records}
            | {answer.index_logical_path for record in records for answer in record.answers}
        )
    )
    modality = ModalityRecord(
        modality_id="reference_image",
        family="optical",
        sensor="source-rendered-optical",
        product_type="rendered-remote-sensing-rgb",
        band_names=("R", "G", "B"),
        band_metadata=tuple(
            BandMetadata(name=name, wavelength_nm=None, polarization=None, units=None)
            for name in ("R", "G", "B")
        ),
        orbit=None,
        acquisition_time=None,
        time_range=None,
        native_gsd_m=None,
        aligned_gsd_m=None,
        units=None,
        signed=False,
        sign_convention=None,
        normalization=NormalizationRecord(method="none", parameters={}, statistics_source=None),
        quality=QualityRecord(
            status="usable",
            flags=() if coverage == 1.0 else ("transparent_pixels_excluded",),
            notes=None,
        ),
        availability_status=availability,
        valid_coverage=coverage,
        native_asset_path=native_ref.path,
        aligned_asset_path=aligned_ref.path,
        valid_mask_path=valid_ref.path,
        source_to_reference_transform=chain,
        reference_to_source_transform=inverse_chain,
        alignment_status="reference",
        render_policy=RenderPolicy(mode="rgb", channels=("R", "G", "B"), clip_percentiles=None),
        hashes={
            "source": first.image.sha256,
            "native": native_ref.sha256,
            "aligned": aligned_ref.sha256,
            "valid": valid_ref.sha256,
        },
    )
    parent = CanonicalParentV3(
        schema_version="sami_canonical_parent_v3",
        parent_id=source.parent_id,
        source=SourceIdentity(
            dataset=first.source_key,
            record_id=f"language-image/{first.image.sha256}",
            scene_id=None,
            event_id=None,
            region_id=None,
            source_group_id=f"language/{first.source_key}/{first.image.sha256}",
        ),
        split="audit",
        reference_canvas=ReferenceCanvas(
            reference_modality_id="reference_image",
            coordinate_space="reference_pixel_half_open",
            original_hw=original_hw,
            canvas_hw=canvas_hw,
            valid_mask_path=valid_ref.path,
            transform_chain=chain,
            inverse_transform_available=True,
            crs=None,
            geotransform=None,
        ),
        modalities=(modality,),
        annotations=AnnotationRecord(
            global_landslide_mask=None,
            global_mask_origin=None,
            global_target_status="unknown",
            referring_regions=(),
            no_target_eligibility=False,
            region_fact_refs=(),
        ),
        provenance=ProvenanceRecord(
            source_registry_key=first.component_license_key,
            source_paths=source_paths,
            source_record_sha256=source_record_sha256,
            scanner_version=MATERIALIZER_VERSION,
            derivation_steps=(
                "decode_selected_png_or_jpeg_without_geometry_inference",
                "convert_source_render_to_rgb",
                "fit_inside_reference_canvas",
                "bilinear_image_nearest_valid",
                "zero_padding_excluded",
            ),
        ),
        license=first.license,
        hashes=HashRecord(
            source_record_sha256=source_record_sha256,
            assets={
                "reference_image.native": native_ref.sha256,
                "reference_image.aligned": aligned_ref.sha256,
                "reference_image.valid": valid_ref.sha256,
            },
        ),
        annotation_status="unlabeled",
    )
    canonical_json_bytes(parent.model_dump(mode="json"))
    return MaterializedLanguageParent(
        parent=parent,
        source_record_ids=tuple(record.record_id for record in records),
        valid_pixel_count=valid_count,
        excluded_pixel_count=total_count - valid_count,
    )


def _bbox_from_mask(mask: Any) -> tuple[int, int, int, int]:
    """Return the tight half-open box around a non-empty binary mask."""

    np = _numpy()
    y, x = np.nonzero(mask)
    if not len(x):
        raise MaterializationError("a referring mask cannot be empty after valid-mask application")
    return (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)


def _render_policy(modality: SourceModalityInput) -> RenderPolicy:
    """Choose a deterministic machine rendering policy without making physical claims."""

    channels = modality.band_names[:3] if len(modality.band_names) >= 3 else modality.band_names
    if modality.family == "optical" and len(channels) == 3:
        mode = "rgb"
    elif len(channels) >= 3:
        mode = "band_composite"
    elif modality.family in {"dem", "slope", "insar", "deformation"}:
        mode = "normalized_scalar"
    else:
        mode = "grayscale"
    return RenderPolicy(mode=mode, channels=channels, clip_percentiles=None)


def materialize_spatial_parent(
    source: SpatialParentInput,
    *,
    benchmark_root: Path,
    canvas_hw: tuple[int, int],
) -> MaterializedParent:
    """Materialize one resolved spatial parent under an audit split.

    Split assignment and task expansion happen only after duplicate clustering.
    This function therefore emits ``split='audit'`` and never writes an index.
    """

    np = _numpy()
    if not source.license.allowed_for_training:
        raise MaterializationError("canonical materialization requires an approved training license")
    if source.license.license_status == "unknown":
        raise MaterializationError("unknown-license source cannot be materialized")
    if not source.modalities:
        raise MaterializationError("spatial parent requires at least one modality")
    modality_ids = tuple(modality.modality_id for modality in source.modalities)
    if len(set(modality_ids)) != len(modality_ids) or source.reference_modality_id not in modality_ids:
        raise MaterializationError("reference modality must uniquely identify one source modality")

    reference_input = next(item for item in source.modalities if item.modality_id == source.reference_modality_id)
    reference_image, reference_valid = _validate_arrays(
        reference_input.array,
        reference_input.valid,
        modality_id=reference_input.modality_id,
    )
    original_hw = tuple(int(value) for value in reference_image.shape[:2])
    chain = build_fit_pad_transform(original_hw, canvas_hw)
    inverse_chain = invert_fit_pad_transform(chain)
    parent_directory = f"assets/{source.parent_id}"

    global_mask = np.asarray(source.global_mask)
    if global_mask.ndim != 2 or tuple(global_mask.shape) != original_hw:
        raise MaterializationError("global mask must be HW on the reference source grid")
    if not np.all((global_mask == 0) | (global_mask == 1)):
        raise MaterializationError("global mask must be binary")
    aligned_reference_valid = _apply_binary(reference_valid, chain)
    aligned_global_mask = _apply_binary(global_mask.astype("u1", copy=False), chain)
    effective_global_mask = aligned_global_mask & aligned_reference_valid
    global_mask_ref = _write_array(
        benchmark_root,
        f"{parent_directory}/global_mask.npy",
        effective_global_mask,
    )

    modality_records: list[ModalityRecord] = []
    asset_hashes: dict[str, str] = {"global_mask": global_mask_ref.sha256}
    reference_valid_ref: ArtifactRef | None = None
    for modality in source.modalities:
        native, valid = _validate_arrays(modality.array, modality.valid, modality_id=modality.modality_id)
        if tuple(native.shape[:2]) != original_hw:
            raise MaterializationError(
                f"{modality.modality_id} is not on the resolved reference grid; reproject before materialization"
            )
        if modality.crs != reference_input.crs:
            raise MaterializationError(f"{modality.modality_id} CRS differs from the resolved reference grid")
        if (modality.geotransform is None) != (reference_input.geotransform is None):
            raise MaterializationError(f"{modality.modality_id} geotransform availability differs from reference")
        if modality.geotransform is not None and reference_input.geotransform is not None:
            if not all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(modality.geotransform, reference_input.geotransform, strict=True)
            ):
                raise MaterializationError(f"{modality.modality_id} geotransform differs from reference")
        if native.shape[2] != len(modality.band_names):
            raise MaterializationError(f"{modality.modality_id} band names do not match HWC channels")
        aligned = _apply_image(native, chain)
        aligned_valid = _apply_binary(valid, chain) & aligned_reference_valid
        aligned[aligned_valid == 0] = 0.0
        native_ref = _write_array(
            benchmark_root,
            f"{parent_directory}/{modality.modality_id}.native.npy",
            native,
        )
        aligned_ref = _write_array(
            benchmark_root,
            f"{parent_directory}/{modality.modality_id}.aligned.npy",
            aligned,
        )
        valid_ref = _write_array(
            benchmark_root,
            f"{parent_directory}/{modality.modality_id}.valid.npy",
            aligned_valid,
        )
        if modality.modality_id == source.reference_modality_id:
            reference_valid_ref = valid_ref
        coverage = float(aligned_valid.sum() / aligned_valid.size)
        if coverage == 0.0:
            availability = "present_zero_valid"
        elif coverage == 1.0:
            availability = "present_valid"
        else:
            availability = "present_partial_valid"
        aligned_gsd = None
        if modality.native_gsd_m is not None:
            resize_scale = next(
                (step.output_hw[0] / step.input_hw[0] for step in chain if step.operation == "resize"),
                1.0,
            )
            aligned_gsd = modality.native_gsd_m / resize_scale
        modality_records.append(
            ModalityRecord(
                modality_id=modality.modality_id,
                family=modality.family,
                sensor=modality.sensor,
                product_type=modality.product_type,
                band_names=modality.band_names,
                band_metadata=tuple(
                    BandMetadata(name=name, wavelength_nm=None, polarization=None, units=modality.units)
                    for name in modality.band_names
                ),
                orbit=modality.orbit,
                acquisition_time=modality.acquisition_time,
                time_range=None,
                native_gsd_m=modality.native_gsd_m,
                aligned_gsd_m=aligned_gsd,
                units=modality.units,
                signed=modality.signed,
                sign_convention=modality.sign_convention,
                normalization=NormalizationRecord(method="none", parameters={}, statistics_source=None),
                quality=QualityRecord(status="usable", flags=(), notes=None),
                availability_status=availability,
                valid_coverage=coverage,
                native_asset_path=native_ref.path,
                aligned_asset_path=aligned_ref.path,
                valid_mask_path=valid_ref.path,
                source_to_reference_transform=chain,
                reference_to_source_transform=inverse_chain,
                alignment_status="reference" if modality.modality_id == source.reference_modality_id else "aligned",
                render_policy=_render_policy(modality),
                hashes={
                    "source": modality.source_sha256,
                    "native": native_ref.sha256,
                    "aligned": aligned_ref.sha256,
                    "valid": valid_ref.sha256,
                },
            )
        )
        asset_hashes[f"{modality.modality_id}.native"] = native_ref.sha256
        asset_hashes[f"{modality.modality_id}.aligned"] = aligned_ref.sha256
        asset_hashes[f"{modality.modality_id}.valid"] = valid_ref.sha256

    assert reference_valid_ref is not None
    referring_records: list[ReferringRegion] = []
    for region in sorted(source.referring_regions, key=lambda item: item.region_id):
        region_mask = np.asarray(region.mask)
        if region_mask.ndim != 2 or tuple(region_mask.shape) != original_hw:
            raise MaterializationError(f"referring mask {region.region_id} is not on the reference source grid")
        if not np.all((region_mask == 0) | (region_mask == 1)):
            raise MaterializationError(f"referring mask {region.region_id} must be binary")
        aligned_region = _apply_binary(region_mask.astype("u1", copy=False), chain) & aligned_reference_valid
        region_ref = _write_array(
            benchmark_root,
            f"{parent_directory}/regions/{region.region_id}.npy",
            aligned_region,
        )
        referring_records.append(
            ReferringRegion(
                region_id=region.region_id,
                expression=region.expression,
                mask_ref=region_ref,
                bbox_half_open=_bbox_from_mask(aligned_region),
                annotation_origin=region.annotation_origin,
            )
        )
        asset_hashes[f"region.{region.region_id}"] = region_ref.sha256

    positive_count = int(effective_global_mask.sum())
    valid_count = int(aligned_reference_valid.sum())
    total_count = int(aligned_reference_valid.size)
    target_status = "positive" if positive_count else "no_target"
    source_paths = tuple(sorted(modality.source_logical_path for modality in source.modalities))
    parent = CanonicalParentV3(
        schema_version="sami_canonical_parent_v3",
        parent_id=source.parent_id,
        source=source.source,
        split="audit",
        reference_canvas=ReferenceCanvas(
            reference_modality_id=source.reference_modality_id,
            coordinate_space="reference_pixel_half_open",
            original_hw=original_hw,
            canvas_hw=canvas_hw,
            valid_mask_path=reference_valid_ref.path,
            transform_chain=chain,
            inverse_transform_available=True,
            crs=reference_input.crs,
            geotransform=transform_geotransform(reference_input.geotransform, chain),
        ),
        modalities=tuple(modality_records),
        annotations=AnnotationRecord(
            global_landslide_mask=global_mask_ref,
            global_mask_origin=source.global_mask_origin,
            global_target_status=target_status,
            referring_regions=tuple(referring_records),
            no_target_eligibility=target_status == "no_target" and valid_count > 0,
            region_fact_refs=(),
        ),
        provenance=ProvenanceRecord(
            source_registry_key=source.license.source_key,
            source_paths=source_paths,
            source_record_sha256=source.source_record_sha256,
            scanner_version=MATERIALIZER_VERSION,
            derivation_steps=(
                "decode_explicit_hwc_and_binary_valid",
                "fit_inside_reference_canvas",
                "bilinear_image_nearest_mask_valid",
                "zero_padding_excluded",
            ),
        ),
        license=source.license,
        hashes=HashRecord(source_record_sha256=source.source_record_sha256, assets=dict(sorted(asset_hashes.items()))),
        annotation_status=source.annotation_status,
    )
    canonical_json_bytes(parent.model_dump(mode="json"))
    return MaterializedParent(
        parent=parent,
        valid_pixel_count=valid_count,
        excluded_pixel_count=total_count - valid_count,
        positive_valid_pixel_count=positive_count,
    )


__all__ = [
    "LanguageParentInput",
    "MATERIALIZER_VERSION",
    "MaterializationError",
    "MaterializedLanguageParent",
    "MaterializedParent",
    "SourceModalityInput",
    "SourceReferringInput",
    "SpatialParentInput",
    "build_fit_pad_transform",
    "invert_fit_pad_transform",
    "materialize_language_parent",
    "materialize_spatial_parent",
    "transform_geotransform",
]
