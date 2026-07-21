"""Audit-only adapters for live sources whose sampled structure is unambiguous."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from sami_gsd.contracts.config import SourceConfig
from sami_gsd.contracts.sources import (
    AnnotationCandidate,
    CanonicalParentCandidate,
    ModalityCandidate,
    RawAssetEvidence,
    RawSourceRecord,
    SourceSampleProjection,
)
from sami_gsd.contracts.spatial import ReferenceCanvasCandidate
from sami_gsd.contracts.spatial import ReferenceCanvasDecision
from sami_gsd.data.adapters.base import AdapterDescriptor, SourceAdapterError
from sami_gsd.data.adapters.formats import (
    read_first_json_array_item,
    read_geotiff_header,
    read_hdf5_dataset_header,
    read_image_header,
    read_netcdf_variable_header,
    read_npy_header,
)
from sami_gsd.data.reference_canvas import select_reference_canvas
from sami_gsd.data.source_loaders.sen12 import Sen12LoadingError, load_sen12_parents
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


ADAPTER_VERSION = "sami_source_adapter_p1_v2_spatial_metadata"


def _logical(source: SourceConfig, relative: Path | str) -> str:
    """Build one portable dataset-rooted path from a source-relative path."""

    relative_text = Path(relative).as_posix()
    return f"{source.provenance.source_root}/{relative_text}"


def _image_asset(
    source_root: Path,
    source: SourceConfig,
    relative: Path,
    *,
    asset_id: str,
    role: Literal["reference_image", "mask"],
) -> RawAssetEvidence:
    """Bind a PNG/JPEG asset by header, size and full sample SHA-256."""

    path = source_root / relative
    if not path.is_file():
        raise SourceAdapterError(f"required source asset is missing: {relative.as_posix()}")
    header = read_image_header(path)
    band_names = ("R", "G", "B") if header.channels == 3 else tuple(
        f"channel_{index + 1}" for index in range(header.channels)
    )
    return RawAssetEvidence(
        asset_id=asset_id,
        role=role,
        logical_path=_logical(source, relative),
        container=header.container,
        internal_key=None,
        sample_index=None,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        array_shape=(header.height, header.width, header.channels),
        native_hw=(header.height, header.width),
        channel_count=header.channels,
        dtype=header.dtype,
        band_names=band_names,
        metadata_evidence=("container_signature", "container_header"),
    )


def _geotiff_asset(
    source_root: Path,
    source: SourceConfig,
    relative: Path,
    *,
    asset_id: str,
    role: Literal["reference_image", "support_image", "mask"],
    band_names: tuple[str, ...] | None = None,
) -> RawAssetEvidence:
    """Bind one GeoTIFF by immutable bytes and header-only spatial metadata."""

    path = source_root / relative
    if not path.is_file():
        raise SourceAdapterError(f"required source asset is missing: {relative.as_posix()}")
    header = read_geotiff_header(path)
    names = band_names or tuple(f"source_band_{index + 1}" for index in range(header.channel_count))
    if len(names) != header.channel_count:
        raise SourceAdapterError("explicit GeoTIFF band names do not match the raster band count")
    dtype = header.dtypes[0] if len(set(header.dtypes)) == 1 else ",".join(header.dtypes)
    return RawAssetEvidence(
        asset_id=asset_id,
        role=role,
        logical_path=_logical(source, relative),
        container="tiff",
        internal_key=None,
        sample_index=None,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        array_shape=(header.height, header.width, header.channel_count),
        native_hw=(header.height, header.width),
        channel_count=header.channel_count,
        dtype=dtype,
        band_names=names,
        metadata_evidence=("geotiff_header", "crs", "geotransform", "nodata_values"),
        crs=header.crs,
        geotransform=header.geotransform,
        nodata_values=header.nodata_values,
    )


def _mixed_image_asset(
    source_root: Path,
    source: SourceConfig,
    relative: Path,
    *,
    asset_id: str,
    role: Literal["reference_image", "mask"],
) -> RawAssetEvidence:
    """Read signature-disguised PNG/JPEG first, then a genuine GeoTIFF."""

    try:
        return _image_asset(source_root, source, relative, asset_id=asset_id, role=role)
    except SourceAdapterError as error:
        if "unsupported image container" not in str(error):
            raise
    names = ("mask",) if role == "mask" else None
    return _geotiff_asset(
        source_root,
        source,
        relative,
        asset_id=asset_id,
        role=role,
        band_names=names,
    )


def _hdf5_asset(
    source_root: Path,
    source: SourceConfig,
    relative: Path,
    *,
    internal_key: str,
    asset_id: str,
    role: Literal["reference_image", "mask"],
    band_names: tuple[str, ...],
) -> RawAssetEvidence:
    """Bind an HDF5 dataset without reading its pixel payload."""

    path = source_root / relative
    if not path.is_file():
        raise SourceAdapterError(f"required source asset is missing: {relative.as_posix()}")
    header = read_hdf5_dataset_header(path, internal_key=internal_key)
    if len(header.shape) == 2:
        native_hw = (header.shape[0], header.shape[1])
        channels = 1
    elif len(header.shape) == 3:
        native_hw = (header.shape[0], header.shape[1])
        channels = header.shape[2]
    else:
        raise SourceAdapterError("HDF5 image/mask dataset must be HW or HWC")
    if len(band_names) != channels:
        raise SourceAdapterError("HDF5 band names do not match dataset channels")
    return RawAssetEvidence(
        asset_id=asset_id,
        role=role,
        logical_path=_logical(source, relative),
        container="hdf5",
        internal_key=internal_key,
        sample_index=None,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        array_shape=header.shape,
        native_hw=native_hw,
        channel_count=channels,
        dtype=header.dtype,
        band_names=band_names,
        metadata_evidence=("hdf5_dataset_header", "hdf5_dataset_attributes"),
        metadata_attributes=header.attributes,
    )


def _netcdf_asset(
    source_root: Path,
    source: SourceConfig,
    relative: Path,
    *,
    internal_key: str,
    asset_id: str,
    role: Literal["reference_image", "support_image", "mask"],
    band_name: str,
) -> RawAssetEvidence:
    """Bind one source NetCDF variable after the formal loader resolves its record."""

    path = source_root / relative
    header = read_netcdf_variable_header(path, internal_key=internal_key)
    if len(header.shape) != 3:
        raise SourceAdapterError("Sen12 variables must use a three-dimensional time/x/y grid")
    return RawAssetEvidence(
        asset_id=asset_id,
        role=role,
        logical_path=_logical(source, relative),
        container="netcdf",
        internal_key=internal_key,
        sample_index=None,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        array_shape=header.shape,
        native_hw=(header.shape[2], header.shape[1]),
        channel_count=1,
        dtype=header.dtype,
        band_names=(band_name,),
        metadata_evidence=("netcdf_variable_header", "formal_single_time_loader"),
        metadata_attributes=header.attributes,
    )


def _index_asset(
    source_root: Path,
    source: SourceConfig,
    relative: Path,
    *,
    asset_id: str,
    container: Literal["json", "jsonl"],
) -> RawAssetEvidence:
    """Bind the exact sampled annotation/index source."""

    path = source_root / relative
    return RawAssetEvidence(
        asset_id=asset_id,
        role="annotation_index",
        logical_path=_logical(source, relative),
        container=container,
        internal_key=None,
        sample_index=None,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        array_shape=None,
        native_hw=None,
        channel_count=None,
        dtype=None,
        band_names=(),
        metadata_evidence=("bounded_first_record_decode",),
    )


def _asset_fingerprint(assets: tuple[RawAssetEvidence, ...]) -> str:
    """Hash a stable asset projection including full sampled-file hashes."""

    return sha256_bytes(canonical_json_bytes([asset.model_dump(mode="json") for asset in assets]))


def _reference_decision(
    modality_id: str,
    native_hw: tuple[int, int],
    *,
    record_type: str,
    native_gsd_m: float | None = None,
) -> ReferenceCanvasDecision:
    """Apply P1.2 selection to a single, source-confirmed reference grid."""

    is_language = record_type in {"global_language", "region_language"}
    return select_reference_canvas(
        (
            ReferenceCanvasCandidate(
                modality_id=modality_id,
                original_hw=native_hw,
                mask_grid="none" if is_language else "native",
                annotation_origin=None if is_language else "official",
                valid_coverage=1.0,
                native_gsd_m=native_gsd_m,
                single_image_language=is_language,
                coordinate_inverse_available=True,
            ),
        )
    )


def _projection(
    *,
    source: SourceConfig,
    record_id: str,
    source_group_id: str,
    source_declared_split: str | None,
    record_type: Literal["spatial_mask", "derived_spatial_mask", "global_language", "region_language"],
    assets: tuple[RawAssetEvidence, ...],
    modality: ModalityCandidate,
    annotation: AnnotationCandidate,
    task_roles: tuple[Literal["t1", "t2", "t3", "t4", "language_global", "language_region"], ...],
    grouping_evidence: tuple[str, ...],
    ambiguity_flags: tuple[str, ...],
    blockers: tuple[str, ...],
    support_modalities: tuple[ModalityCandidate, ...] = (),
) -> SourceSampleProjection:
    """Construct and cross-bind both strict layers of one projection."""

    fingerprint = _asset_fingerprint(assets)
    raw = RawSourceRecord(
        schema_version="sami_raw_source_record_v1",
        source_key=source.source_key,
        record_id=record_id,
        source_group_id=source_group_id,
        source_declared_split=source_declared_split,
        split="audit",
        record_type=record_type,
        assets=assets,
        grouping_evidence=grouping_evidence,
        ambiguity_flags=ambiguity_flags,
        asset_set_sha256=fingerprint,
    )
    decision = _reference_decision(
        modality.modality_id,
        modality.native_hw,
        record_type=record_type,
        native_gsd_m=modality.native_gsd_m,
    )
    candidate = CanonicalParentCandidate(
        schema_version="sami_canonical_parent_candidate_v1",
        parent_id=f"{source.source_key}-{sha256_bytes(record_id.encode('utf-8'))[:20]}",
        source_key=source.source_key,
        record_id=record_id,
        source_group_id=source_group_id,
        split="audit",
        task_roles=task_roles,
        reference_decision=decision,
        modalities=(modality, *support_modalities),
        annotations=annotation,
        raw_asset_set_sha256=fingerprint,
        materialization_status="blocked" if blockers else "ready",
        blockers=tuple(sorted(set(blockers))),
    )
    digest_payload = {
        "raw_record": raw.model_dump(mode="json"),
        "canonical_candidate": candidate.model_dump(mode="json"),
    }
    return SourceSampleProjection(
        schema_version="sami_source_sample_projection_v1",
        raw_record=raw,
        canonical_candidate=candidate,
        projection_sha256=sha256_bytes(canonical_json_bytes(digest_payload)),
    )


def _optical_modality(asset: RawAssetEvidence, *, product_type: str = "source_image") -> ModalityCandidate:
    """Project only header-confirmed optical fields; unknown metadata stays null."""

    if asset.native_hw is None:
        raise SourceAdapterError("optical spatial asset is missing native_hw")
    return ModalityCandidate(
        modality_id="reference_optical",
        family="optical",
        sensor="source_unspecified_optical",
        product_type=product_type,
        band_names=asset.band_names,
        native_asset_id=asset.asset_id,
        native_hw=asset.native_hw,
        native_gsd_m=None,
        units=None,
        signed=None,
        sign_convention=None,
        alignment_status="reference",
        alignment_evidence=("same_native_grid_as_reference",),
        valid_status="unresolved",
    )


class Sen12Adapter:
    """Audit the same event-balanced annotated records used by the formal loader."""

    descriptor = AdapterDescriptor(
        source_key="sen12_landslides",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("spatial_mask",),
        blockers=(),
    )

    def extract_samples(
        self,
        source_root: Path,
        source_config: SourceConfig,
        *,
        limit: int,
    ) -> tuple[SourceSampleProjection, ...]:
        """Decode bounded formal parents and project their exact NetCDF evidence."""

        try:
            parents = load_sen12_parents(source_config, source_root=source_root, limit=limit)
        except Sen12LoadingError as error:
            raise SourceAdapterError(str(error)) from error
        projections: list[SourceSampleProjection] = []
        for parent in parents:
            event, sample = parent.source.record_id.split("/", maxsplit=1)
            s2_relative = Path("s2") / f"{event}_s2_{sample}.nc"
            asc_relative = Path("s1asc") / f"{event}_s1asc_{sample}.nc"
            dsc_relative = Path("s1dsc") / f"{event}_s1dsc_{sample}.nc"
            assets = (
                _netcdf_asset(
                    source_root,
                    source_config,
                    s2_relative,
                    internal_key="B02",
                    asset_id="s2_reference",
                    role="reference_image",
                    band_name="B02",
                ),
                _netcdf_asset(
                    source_root,
                    source_config,
                    asc_relative,
                    internal_key="VV",
                    asset_id="s1_ascending",
                    role="support_image",
                    band_name="VV",
                ),
                _netcdf_asset(
                    source_root,
                    source_config,
                    dsc_relative,
                    internal_key="VV",
                    asset_id="s1_descending",
                    role="support_image",
                    band_name="VV",
                ),
                _netcdf_asset(
                    source_root,
                    source_config,
                    s2_relative,
                    internal_key="DEM",
                    asset_id="dem",
                    role="support_image",
                    band_name="DEM",
                ),
                _netcdf_asset(
                    source_root,
                    source_config,
                    s2_relative,
                    internal_key="MASK",
                    asset_id="mask",
                    role="mask",
                    band_name="mask",
                ),
            )

            candidates: list[ModalityCandidate] = []
            asset_ids = ("s2_reference", "s1_ascending", "s1_descending", "dem")
            for index, (modality, asset_id) in enumerate(zip(parent.modalities, asset_ids, strict=True)):
                candidates.append(
                    ModalityCandidate(
                        modality_id=modality.modality_id,
                        family=modality.family,
                        sensor=modality.sensor,
                        product_type=modality.product_type,
                        band_names=modality.band_names,
                        native_asset_id=asset_id,
                        native_hw=tuple(int(value) for value in modality.array.shape[:2]),
                        native_gsd_m=modality.native_gsd_m,
                        units=modality.units,
                        signed=modality.signed,
                        sign_convention=modality.sign_convention,
                        alignment_status="reference" if index == 0 else "aligned",
                        alignment_evidence=("formal_loader_exact_shared_grid",),
                        valid_status="explicit_valid_mask",
                    )
                )
            projections.append(
                _projection(
                    source=source_config,
                    record_id=parent.source.record_id,
                    source_group_id=parent.source.source_group_id,
                    source_declared_split=None,
                    record_type="spatial_mask",
                    assets=assets,
                    modality=candidates[0],
                    support_modalities=tuple(candidates[1:]),
                    annotation=AnnotationCandidate(
                        global_mask_asset_id="mask",
                        target_status="unknown",
                        normalized_box_xyxy=None,
                        phrase=None,
                        annotation_origin="official",
                    ),
                    task_roles=("t1",),
                    grouping_evidence=("event_token", "sample_number", "paired_s2_s1asc_s1dsc"),
                    ambiguity_flags=(),
                    blockers=(),
                )
            )
        return tuple(projections)


class GDCLDAdapter:
    """Sample same-name GDCLD train patch/image-mask pairs."""

    descriptor = AdapterDescriptor(
        source_key="gdcld",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("spatial_mask",),
        blockers=(
            "scene_test_image_mask_grid_mismatch",
            "source_grouping_policy_unresolved",
        ),
    )

    def extract_samples(self, source_root: Path, source_config: SourceConfig, *, limit: int) -> tuple[SourceSampleProjection, ...]:
        """Extract bounded mixed-container patch pairs; mismatched test scenes stay blocked."""

        names = sorted(path.name for path in (source_root / "train_data").iterdir() if path.is_file())
        projections: list[SourceSampleProjection] = []
        for name in names:
            if len(projections) >= limit:
                break
            try:
                image = _mixed_image_asset(
                    source_root,
                    source_config,
                    Path("train_data") / name,
                    asset_id="reference",
                    role="reference_image",
                )
                mask = _mixed_image_asset(
                    source_root,
                    source_config,
                    Path("train_label") / name,
                    asset_id="mask",
                    role="mask",
                )
            except SourceAdapterError:
                raise
            if image.native_hw != mask.native_hw:
                raise SourceAdapterError(f"GDCLD patch image/mask shape conflict: {name}")
            projections.append(
                _projection(
                    source=source_config,
                    record_id=f"train/{name}",
                    source_group_id=f"gdcld/patch/{Path(name).stem}",
                    source_declared_split="train",
                    record_type="spatial_mask",
                    assets=(image, mask),
                    modality=_optical_modality(image, product_type="rgb_patch"),
                    annotation=AnnotationCandidate(
                        global_mask_asset_id="mask",
                        target_status="unknown",
                        normalized_box_xyxy=None,
                        phrase=None,
                        annotation_origin="official",
                    ),
                    task_roles=("t1",),
                    grouping_evidence=("same_basename_image_mask_pair",),
                    ambiguity_flags=("original_scene_identity_unresolved",),
                    blockers=("original_scene_identity_unresolved",),
                )
            )
        return tuple(projections)


class Landslide4SenseAdapter:
    """Audit same-suffix HDF5 image/mask pairs with explicit metadata uncertainty."""

    descriptor = AdapterDescriptor(
        source_key="landslide4sense",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("spatial_mask",),
        blockers=("band_semantics_unresolved", "original_scene_grouping_unresolved"),
    )

    def extract_samples(
        self,
        source_root: Path,
        source_config: SourceConfig,
        *,
        limit: int,
    ) -> tuple[SourceSampleProjection, ...]:
        """Read stable numeric pairs and leave unknown band/group semantics blocked."""

        image_root = source_root / "TrainData/img"
        image_paths = sorted(image_root.glob("image_*.h5"), key=lambda path: path.name)
        projections: list[SourceSampleProjection] = []
        for image_path in image_paths[:limit]:
            suffix = image_path.stem.removeprefix("image_")
            mask_path = source_root / "TrainData/mask" / f"mask_{suffix}.h5"
            image_relative = image_path.relative_to(source_root)
            mask_relative = mask_path.relative_to(source_root)
            image_header = read_hdf5_dataset_header(image_path, internal_key="img")
            if len(image_header.shape) != 3:
                raise SourceAdapterError("Landslide4Sense image must use HWC layout")
            image = _hdf5_asset(
                source_root,
                source_config,
                image_relative,
                internal_key="img",
                asset_id="reference",
                role="reference_image",
                band_names=tuple(f"source_band_{index + 1}" for index in range(image_header.shape[2])),
            )
            mask = _hdf5_asset(
                source_root,
                source_config,
                mask_relative,
                internal_key="mask",
                asset_id="mask",
                role="mask",
                band_names=("mask",),
            )
            if image.native_hw != mask.native_hw:
                raise SourceAdapterError(f"Landslide4Sense image/mask shape conflict: {suffix}")
            modality = ModalityCandidate(
                modality_id="reference_multispectral",
                family="multispectral",
                sensor="source_unspecified_multispectral",
                product_type="fourteen_channel_patch",
                band_names=image.band_names,
                native_asset_id="reference",
                native_hw=image.native_hw,
                native_gsd_m=None,
                units=None,
                signed=None,
                sign_convention=None,
                alignment_status="reference",
                alignment_evidence=("same_hdf5_patch_grid_as_mask",),
                valid_status="unresolved",
            )
            projections.append(
                _projection(
                    source=source_config,
                    record_id=f"TrainData/{suffix}",
                    source_group_id=f"landslide4sense/patch/{suffix}",
                    source_declared_split="train",
                    record_type="spatial_mask",
                    assets=(image, mask),
                    modality=modality,
                    annotation=AnnotationCandidate(
                        global_mask_asset_id="mask",
                        target_status="unknown",
                        normalized_box_xyxy=None,
                        phrase=None,
                        annotation_origin="official",
                    ),
                    task_roles=("t1",),
                    grouping_evidence=("same_numeric_suffix_hdf5_pair",),
                    ambiguity_flags=("band_semantics_unresolved", "original_scene_grouping_unresolved"),
                    blockers=("band_semantics_unresolved", "original_scene_grouping_unresolved"),
                )
            )
        return tuple(projections)


def _aligned_grid_matches(reference: RawAssetEvidence, support: RawAssetEvidence) -> bool:
    """Require exact grid, CRS and transform equality for pixel-level support."""

    transforms_match = (
        reference.geotransform is not None
        and support.geotransform is not None
        and all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(reference.geotransform, support.geotransform, strict=True)
        )
    )
    return (
        reference.native_hw == support.native_hw
        and reference.crs == support.crs
        and transforms_match
    )


class MultimodalLandslideAdapter:
    """Audit co-registered RGB, DEM, InSAR-like and label GeoTIFF records."""

    descriptor = AdapterDescriptor(
        source_key="multimodal_landslide",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("spatial_mask",),
        blockers=("insar_units_and_sign_convention_unresolved",),
    )

    def extract_samples(
        self,
        source_root: Path,
        source_config: SourceConfig,
        *,
        limit: int,
    ) -> tuple[SourceSampleProjection, ...]:
        """Read explicit split lists and exact same-stem grids without quantitative claims."""

        inner = source_root / "multimodal-landslide-dataset"
        if not inner.is_dir():
            raise SourceAdapterError("multimodal dataset inner root is missing")
        entries: list[tuple[str, str]] = []
        for split_name in ("train", "val"):
            split_path = inner / f"{split_name}.txt"
            if not split_path.is_file():
                continue
            for line in split_path.read_text(encoding="utf-8").splitlines():
                stem = Path(line.strip()).stem
                if stem:
                    entries.append((split_name, stem))
        entries = sorted(set(entries), key=lambda item: (item[0], item[1]))
        projections: list[SourceSampleProjection] = []
        for split_name, stem in entries[:limit]:
            prefix = Path("multimodal-landslide-dataset")
            rgb = _geotiff_asset(
                source_root,
                source_config,
                prefix / "rgb" / f"{stem}.tif",
                asset_id="reference_rgb",
                role="reference_image",
                band_names=("R", "G", "B"),
            )
            dem = _geotiff_asset(
                source_root,
                source_config,
                prefix / "dem" / f"{stem}.tif",
                asset_id="support_dem",
                role="support_image",
                band_names=("DEM",),
            )
            insar = _geotiff_asset(
                source_root,
                source_config,
                prefix / "insar_vel" / f"{stem}.tif",
                asset_id="support_insar",
                role="support_image",
                band_names=("source_insar_like_value",),
            )
            mask = _geotiff_asset(
                source_root,
                source_config,
                prefix / "label" / f"{stem}.tif",
                asset_id="mask",
                role="mask",
                band_names=("mask",),
            )
            if not all(_aligned_grid_matches(rgb, asset) for asset in (dem, insar, mask)):
                raise SourceAdapterError(f"multimodal source grids are not exactly co-registered: {stem}")
            reference = ModalityCandidate(
                modality_id="reference_rgb",
                family="optical",
                sensor="source_unspecified_optical",
                product_type="rgb_raster",
                band_names=rgb.band_names,
                native_asset_id="reference_rgb",
                native_hw=rgb.native_hw,
                native_gsd_m=None,
                units=None,
                signed=None,
                sign_convention=None,
                alignment_status="reference",
                alignment_evidence=("exact_crs_transform_and_grid_match",),
                valid_status="source_nodata_only",
            )
            supports = (
                ModalityCandidate(
                    modality_id="support_dem",
                    family="dem",
                    sensor="source_unspecified_dem",
                    product_type="digital_elevation_raster",
                    band_names=dem.band_names,
                    native_asset_id="support_dem",
                    native_hw=dem.native_hw,
                    native_gsd_m=None,
                    units=None,
                    signed=None,
                    sign_convention=None,
                    alignment_status="aligned",
                    alignment_evidence=("exact_crs_transform_and_grid_match",),
                    valid_status="source_nodata_only",
                ),
                ModalityCandidate(
                    modality_id="support_insar",
                    family="insar",
                    sensor="source_unspecified_insar",
                    product_type="insar_like_raster",
                    band_names=insar.band_names,
                    native_asset_id="support_insar",
                    native_hw=insar.native_hw,
                    native_gsd_m=None,
                    units=None,
                    signed=None,
                    sign_convention=None,
                    alignment_status="aligned",
                    alignment_evidence=("exact_crs_transform_and_grid_match",),
                    valid_status="source_nodata_only",
                ),
            )
            prefix_group = stem.split("_", maxsplit=1)[0]
            projections.append(
                _projection(
                    source=source_config,
                    record_id=f"{split_name}/{stem}",
                    source_group_id=f"multimodal/{prefix_group}/{stem}",
                    source_declared_split=split_name,
                    record_type="spatial_mask",
                    assets=(rgb, dem, insar, mask),
                    modality=reference,
                    support_modalities=supports,
                    annotation=AnnotationCandidate(
                        global_mask_asset_id="mask",
                        target_status="unknown",
                        normalized_box_xyxy=None,
                        phrase=None,
                        annotation_origin="official",
                    ),
                    task_roles=("t1",),
                    grouping_evidence=("explicit_split_list", "same_stem_multimodal_record", "region_prefix"),
                    ambiguity_flags=("insar_units_and_sign_convention_unresolved",),
                    blockers=("insar_units_and_sign_convention_unresolved",),
                )
            )
        return tuple(projections)


class LMHLDAdapter:
    """Sample one virtual NPY row from deterministic region/split arrays."""

    descriptor = AdapterDescriptor(
        source_key="lmhld",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("spatial_mask",),
        blockers=("original_scene_grouping_unresolved", "band_semantics_unresolved"),
    )

    def extract_samples(self, source_root: Path, source_config: SourceConfig, *, limit: int) -> tuple[SourceSampleProjection, ...]:
        """Read NPY headers only and bind virtual row zero for bounded evidence."""

        regions = sorted(
            path for path in (source_root / "LMHLD_dataset_different_patch_sizes").iterdir() if path.is_dir()
        )
        projections: list[SourceSampleProjection] = []
        for region in regions:
            if len(projections) >= limit:
                break
            image_path = region / "train_images.npy"
            mask_path = region / "train_labels.npy"
            if not image_path.is_file() or not mask_path.is_file():
                continue
            image_header = read_npy_header(image_path)
            mask_header = read_npy_header(mask_path)
            if len(image_header.shape) != 4 or len(mask_header.shape) != 4:
                raise SourceAdapterError("LMHLD arrays must be explicit NCHW tensors")
            if image_header.shape[0] != mask_header.shape[0] or image_header.shape[-2:] != mask_header.shape[-2:]:
                raise SourceAdapterError("LMHLD image/mask NPY shapes conflict")
            relative_image = image_path.relative_to(source_root)
            relative_mask = mask_path.relative_to(source_root)
            image = RawAssetEvidence(
                asset_id="reference",
                role="reference_image",
                logical_path=_logical(source_config, relative_image),
                container="npy",
                internal_key=None,
                sample_index=0,
                byte_size=image_path.stat().st_size,
                sha256=sha256_file(image_path),
                array_shape=image_header.shape,
                native_hw=image_header.shape[-2:],
                channel_count=image_header.shape[1],
                dtype=image_header.dtype,
                band_names=tuple(f"source_band_{index + 1}" for index in range(image_header.shape[1])),
                metadata_evidence=("npy_header", "virtual_first_axis_sample"),
            )
            mask = RawAssetEvidence(
                asset_id="mask",
                role="mask",
                logical_path=_logical(source_config, relative_mask),
                container="npy",
                internal_key=None,
                sample_index=0,
                byte_size=mask_path.stat().st_size,
                sha256=sha256_file(mask_path),
                array_shape=mask_header.shape,
                native_hw=mask_header.shape[-2:],
                channel_count=mask_header.shape[1],
                dtype=mask_header.dtype,
                band_names=("mask",),
                metadata_evidence=("npy_header", "virtual_first_axis_sample"),
            )
            modality = ModalityCandidate(
                modality_id="reference_multiband",
                family="optical",
                sensor="source_unspecified_multiband_optical",
                product_type="multiband_patch",
                band_names=image.band_names,
                native_asset_id="reference",
                native_hw=image.native_hw,
                native_gsd_m=None,
                units=None,
                signed=None,
                sign_convention=None,
                alignment_status="reference",
                alignment_evidence=("same_array_sample_grid_as_mask",),
                valid_status="unresolved",
            )
            projections.append(
                _projection(
                    source=source_config,
                    record_id=f"different_patch_sizes/{region.name}/train/0",
                    source_group_id=f"lmhld/{region.name}/virtual_row_0",
                    source_declared_split="train",
                    record_type="spatial_mask",
                    assets=(image, mask),
                    modality=modality,
                    annotation=AnnotationCandidate(
                        global_mask_asset_id="mask",
                        target_status="unknown",
                        normalized_box_xyxy=None,
                        phrase=None,
                        annotation_origin="official",
                    ),
                    task_roles=("t1",),
                    grouping_evidence=("same_region_split_first_axis_index",),
                    ambiguity_flags=("original_scene_identity_unresolved", "band_semantics_unresolved"),
                    blockers=("original_scene_identity_unresolved", "band_semantics_unresolved"),
                )
            )
        return tuple(projections)


def _extract_message_image(row: dict[str, Any]) -> str:
    """Read exactly one Qwen-style image reference from a JSONL row."""

    images = [
        item.get("image")
        for message in row.get("messages", [])
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "image"
    ]
    if len(images) != 1 or not isinstance(images[0], str):
        raise SourceAdapterError("LandslideBench row must contain exactly one image reference")
    return images[0]


class LandslideBenchAdapter:
    """Audit derived Qwen JSONL/image/mask pairs without trusting their prose."""

    descriptor = AdapterDescriptor(
        source_key="landslidebench_agent",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("derived_spatial_mask",),
        blockers=("derived_source_provenance_unresolved", "multi_zoom_grouping_unresolved"),
    )

    def extract_samples(self, source_root: Path, source_config: SourceConfig, *, limit: int) -> tuple[SourceSampleProjection, ...]:
        """Read bounded JSONL rows and same-basename masks; never import legacy readers."""

        index_relative = Path("qwen3vl_landslide_train.jsonl")
        index_path = source_root / index_relative
        projections: list[SourceSampleProjection] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(projections) >= limit:
                    break
                row = json.loads(line)
                image_relative = Path(_extract_message_image(row))
                image = _image_asset(
                    source_root, source_config, image_relative, asset_id="reference", role="reference_image"
                )
                mask = _image_asset(
                    source_root,
                    source_config,
                    Path("mask") / image_relative.name,
                    asset_id="mask",
                    role="mask",
                )
                if image.native_hw != mask.native_hw:
                    raise SourceAdapterError("LandslideBench image/mask shape conflict")
                index = _index_asset(
                    source_root, source_config, index_relative, asset_id="annotation", container="jsonl"
                )
                stem_without_level = re.sub(r"_Level_\d+$", "", image_relative.stem)
                projections.append(
                    _projection(
                        source=source_config,
                        record_id=f"train/line_{line_number}",
                        source_group_id=f"landslidebench/{stem_without_level}",
                        source_declared_split="train",
                        record_type="derived_spatial_mask",
                        assets=(image, mask, index),
                        modality=_optical_modality(image, product_type="derived_web_image"),
                        annotation=AnnotationCandidate(
                            global_mask_asset_id="mask",
                            target_status="unknown",
                            normalized_box_xyxy=None,
                            phrase=None,
                            annotation_origin="derived",
                        ),
                        task_roles=("t1",),
                        grouping_evidence=("jsonl_image_reference", "same_basename_mask", "zoom_suffix_removed"),
                        ambiguity_flags=("derived_source_provenance_unresolved", "multi_zoom_grouping_unverified"),
                        blockers=("derived_source_provenance_unresolved", "multi_zoom_grouping_unverified"),
                    )
                )
        return tuple(projections)


def _mmrs_image_relative(value: str) -> Path:
    """Map the observed MMRS JSON ``data/...`` prefix to its local source root."""

    path = Path(value)
    if not path.parts or path.parts[0] != "data" or len(path.parts) < 3:
        raise SourceAdapterError("MMRS image path does not use the observed data/... layout")
    return Path(*path.parts[1:])


class MMRSAdapter:
    """Audit only the five frozen caption subsets plus DIOR-RSVG short phrases."""

    descriptor = AdapterDescriptor(
        source_key="mmrs_1m",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("global_language", "region_language"),
        blockers=(),
    )
    _caption_indexes = (
        Path("json/caption/caption_nwpu.json"),
        Path("json/caption/caption_rsicd.json"),
        Path("json/caption/caption_rsitmd.json"),
        Path("json/caption/caption_syndney.json"),
        Path("json/caption/caption_ucm.json"),
    )

    def extract_samples(self, source_root: Path, source_config: SourceConfig, *, limit: int) -> tuple[SourceSampleProjection, ...]:
        """Decode only the first record of each explicitly selected source file."""

        definitions: list[tuple[Path, Literal["global_language", "region_language"]]] = [
            *((path, "global_language") for path in self._caption_indexes),
            (Path("json/RSVG/rsvg_trainval.json"), "region_language"),
        ]
        projections: list[SourceSampleProjection] = []
        for index_relative, record_type in definitions[:limit]:
            row = read_first_json_array_item(source_root / index_relative)
            if not isinstance(row, dict) or not isinstance(row.get("image"), str):
                raise SourceAdapterError(f"MMRS first record is missing image: {index_relative.as_posix()}")
            image_relative = _mmrs_image_relative(row["image"])
            image = _image_asset(
                source_root, source_config, image_relative, asset_id="reference", role="reference_image"
            )
            annotation_index = _index_asset(
                source_root, source_config, index_relative, asset_id="annotation", container="json"
            )
            box = None
            phrase = None
            if record_type == "region_language":
                conversations = row.get("conversations")
                if not isinstance(conversations, list) or len(conversations) < 2:
                    raise SourceAdapterError("DIOR-RSVG first record lacks a prompt/phrase pair")
                prompt = conversations[0].get("value", "")
                match = re.search(r":\s*(\[[^\]]+\])\s*$", prompt)
                phrase_value = conversations[1].get("value")
                if match is None or not isinstance(phrase_value, str):
                    raise SourceAdapterError("DIOR-RSVG region box or phrase is unresolved")
                parsed_box = json.loads(match.group(1))
                if not isinstance(parsed_box, list) or len(parsed_box) != 4:
                    raise SourceAdapterError("DIOR-RSVG normalized box must contain four values")
                box = tuple(float(value) for value in parsed_box)
                phrase = phrase_value
            component = index_relative.stem.removeprefix("caption_")
            record_id = f"{component}/{image_relative.as_posix()}"
            projections.append(
                _projection(
                    source=source_config,
                    record_id=record_id,
                    source_group_id=f"mmrs/{image_relative.as_posix()}",
                    source_declared_split=None,
                    record_type=record_type,
                    assets=(image, annotation_index),
                    modality=_optical_modality(image),
                    annotation=AnnotationCandidate(
                        global_mask_asset_id=None,
                        target_status="unknown",
                        normalized_box_xyxy=box,
                        phrase=phrase,
                        annotation_origin="source_expression" if record_type == "region_language" else "source_caption",
                    ),
                    task_roles=("language_region",) if record_type == "region_language" else ("language_global",),
                    grouping_evidence=("exact_source_image_path",),
                    ambiguity_flags=(),
                    blockers=(),
                )
            )
        return tuple(projections)


class RSGPTAdapter:
    """Audit one RSICap and one permanent-test RSIEval record."""

    descriptor = AdapterDescriptor(
        source_key="rsgpt",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("global_language",),
        blockers=(),
    )

    def extract_samples(self, source_root: Path, source_config: SourceConfig, *, limit: int) -> tuple[SourceSampleProjection, ...]:
        """Decode bounded first annotations while retaining RSIEval test-only evidence."""

        definitions = (
            (Path("dataset/RSICap/captions.json"), "RSICap", "train_candidate"),
            (Path("dataset/RSIEval/annotations.json"), "RSIEval", "test_only"),
        )
        projections: list[SourceSampleProjection] = []
        for index_relative, component, declared_split in definitions[:limit]:
            row = read_first_json_array_item(source_root / index_relative, array_key="annotations")
            if not isinstance(row, dict) or not isinstance(row.get("filename"), str):
                raise SourceAdapterError(f"{component} first annotation lacks filename")
            image_relative = Path("dataset") / component / "images" / row["filename"]
            image = _image_asset(
                source_root, source_config, image_relative, asset_id="reference", role="reference_image"
            )
            annotation_index = _index_asset(
                source_root, source_config, index_relative, asset_id="annotation", container="json"
            )
            projections.append(
                _projection(
                    source=source_config,
                    record_id=f"{component}/{row['filename']}",
                    source_group_id=f"rsgpt/{component}/{row['filename']}",
                    source_declared_split=declared_split,
                    record_type="global_language",
                    assets=(image, annotation_index),
                    modality=_optical_modality(image),
                    annotation=AnnotationCandidate(
                        global_mask_asset_id=None,
                        target_status="unknown",
                        normalized_box_xyxy=None,
                        phrase=None,
                        annotation_origin="source_caption",
                    ),
                    task_roles=("language_global",),
                    grouping_evidence=("exact_filename_annotation_binding",),
                    ambiguity_flags=(),
                    blockers=(),
                )
            )
        return tuple(projections)


__all__ = [
    "GDCLDAdapter",
    "LMHLDAdapter",
    "Landslide4SenseAdapter",
    "LandslideBenchAdapter",
    "MMRSAdapter",
    "MultimodalLandslideAdapter",
    "RSGPTAdapter",
]
