"""Audit-only adapters for live sources whose sampled structure is unambiguous."""

from __future__ import annotations

import json
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
from sami_gsd.data.adapters.formats import read_first_json_array_item, read_image_header, read_npy_header
from sami_gsd.data.reference_canvas import select_reference_canvas
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


ADAPTER_VERSION = "sami_source_adapter_p1_3_v1"


def _logical(source: SourceConfig, relative: Path | str) -> str:
    """Build one portable dataset-rooted path from a source-relative path."""

    relative_text = Path(relative).as_posix()
    return f"datasets/{source.local_path}/{relative_text}"


def _license_blockers(source: SourceConfig) -> tuple[str, ...]:
    """Expose the exact fail-closed reason without changing the license record."""

    blockers = ["license_not_approved_for_training"]
    if source.license.license_status == "unknown":
        blockers.append("license_status_unknown")
    elif source.license.license_status == "restricted":
        blockers.append("restricted_license_requires_human_approval")
    return tuple(blockers)


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
        training_eligible=False,
        record_type=record_type,
        assets=assets,
        grouping_evidence=grouping_evidence,
        ambiguity_flags=ambiguity_flags,
        license=source.license,
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
        training_eligible=False,
        task_roles=task_roles,
        reference_decision=decision,
        modalities=(modality,),
        annotations=annotation,
        license=source.license,
        raw_asset_set_sha256=fingerprint,
        materialization_status="audit_only",
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


class GDCLDAdapter:
    """Sample same-name GDCLD train patch/image-mask pairs."""

    descriptor = AdapterDescriptor(
        source_key="gdcld",
        adapter_version=ADAPTER_VERSION,
        implementation_status="implemented",
        supported_record_types=("spatial_mask",),
        blockers=(
            "mixed_png_tiff_patch_container_partial_implementation",
            "scene_test_image_mask_grid_mismatch",
            "source_grouping_policy_unresolved",
        ),
    )

    def extract_samples(self, source_root: Path, source_config: SourceConfig, *, limit: int) -> tuple[SourceSampleProjection, ...]:
        """Extract bounded train-patch pairs; mismatched large test scenes stay blocked."""

        names = sorted(path.name for path in (source_root / "train_data").iterdir() if path.is_file())
        projections: list[SourceSampleProjection] = []
        for name in names:
            if len(projections) >= limit:
                break
            try:
                image = _image_asset(
                    source_root,
                    source_config,
                    Path("train_data") / name,
                    asset_id="reference",
                    role="reference_image",
                )
                mask = _image_asset(
                    source_root,
                    source_config,
                    Path("train_label") / name,
                    asset_id="mask",
                    role="mask",
                )
            except SourceAdapterError as error:
                if "unsupported image container" in str(error):
                    continue
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
                    blockers=_license_blockers(source_config)
                    + (
                        "mixed_png_tiff_patch_container_partial_implementation",
                        "original_scene_identity_unresolved",
                    ),
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
                    blockers=_license_blockers(source_config)
                    + ("original_scene_identity_unresolved", "band_semantics_unresolved"),
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
                        blockers=_license_blockers(source_config)
                        + ("derived_source_provenance_unresolved", "multi_zoom_grouping_unverified"),
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
        blockers=("component_licenses_unresolved",),
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
            blockers = _license_blockers(source_config) + ("component_license_unresolved",)
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
                    ambiguity_flags=("component_license_unresolved",),
                    blockers=blockers,
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
        blockers=("academic_only_use_requires_human_approval",),
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
            blockers = _license_blockers(source_config)
            if component == "RSIEval":
                blockers += ("permanent_test_only",)
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
                    blockers=blockers,
                )
            )
        return tuple(projections)


__all__ = ["GDCLDAdapter", "LMHLDAdapter", "LandslideBenchAdapter", "MMRSAdapter", "RSGPTAdapter"]
