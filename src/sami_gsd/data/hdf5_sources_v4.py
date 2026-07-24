"""Strict HDF5 source reader for P1 Benchmark v4.

This module projects only the fields registered by the v4 config.  Source
attributes, machine paths, dialogue text, and visualization metadata never
enter canonical identity.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sami_gsd.contracts.benchmark_v4 import (
    DuplicateComponentV1,
    GroupIdentityV1,
    Hdf5ArrayRefV1,
    Hdf5SourceRecordV1,
    ScientificProvenanceV1,
    ValidityDescriptorV1,
    seal_source_record,
    validate_benchmark_logical_path,
    validate_dataset_logical_path,
)
from sami_gsd.utilities.artifacts import reject_non_finite, sha256_file


class Hdf5SourceV4Error(ValueError):
    """A source row or HDF5 asset violates the frozen P1 contract."""


def _as_mapping(value: Any, *, location: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif callable(getattr(value, "model_dump", None)):
        payload = value.model_dump(mode="json")
    elif callable(getattr(value, "to_mapping", None)):
        payload = value.to_mapping()
    else:
        raise TypeError(f"{location} must expose a JSON mapping")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{location} projection must be a mapping")
    result = dict(payload)
    reject_non_finite(result)
    return result


def _strict_json_object(line: str, *, location: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        payload = json.loads(line, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise Hdf5SourceV4Error(f"invalid JSON at {location}: {error}") from error
    if not isinstance(payload, dict):
        raise Hdf5SourceV4Error(f"JSON row must be an object at {location}")
    return payload


def resolve_dataset_logical_path(logical_path: str, datasets_root: Path) -> Path:
    """Resolve one ``datasets/...`` path while enforcing root containment."""

    validate_dataset_logical_path(logical_path)
    relative = PurePosixPath(logical_path).relative_to("datasets")
    root = datasets_root.resolve()
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Hdf5SourceV4Error(
            f"logical source path escapes datasets root: {logical_path}"
        ) from error
    return resolved


def resolve_benchmark_logical_path(
    logical_path: str,
    benchmark_root: Path,
) -> Path:
    """Resolve one materialized Benchmark path without a datasets fallback."""

    validate_benchmark_logical_path(logical_path, require_hdf5=True)
    relative = PurePosixPath(logical_path).relative_to("benchmark")
    root = benchmark_root.resolve()
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Hdf5SourceV4Error(
            f"logical Benchmark path escapes benchmark root: {logical_path}"
        ) from error
    return resolved


def _join_hdf5_logical_path(hdf5_base: str, row_value: Any) -> str:
    if not isinstance(row_value, str) or not row_value:
        raise Hdf5SourceV4Error("HDF5 row path must be a non-empty string")
    if "\\" in row_value or row_value.startswith(("/", "file://")):
        raise Hdf5SourceV4Error("absolute or non-POSIX HDF5 row path is forbidden")
    if row_value.startswith("datasets/"):
        logical = row_value
    else:
        relative = PurePosixPath(row_value)
        if any(part in {"", ".", ".."} for part in row_value.split("/")):
            raise Hdf5SourceV4Error("HDF5 row path contains traversal")
        logical = str(PurePosixPath(hdf5_base) / relative)
    validate_dataset_logical_path(logical, require_hdf5=True)
    base = PurePosixPath(hdf5_base)
    try:
        PurePosixPath(logical).relative_to(base)
    except ValueError as error:
        raise Hdf5SourceV4Error(
            f"HDF5 path is outside configured source base: {logical}"
        ) from error
    return logical


def _benchmark_asset_logical_path(
    source: Mapping[str, Any],
    source_logical_path: str,
) -> str:
    """Map a source HDF5 into the self-contained Benchmark asset hierarchy."""

    source_root = PurePosixPath(str(source["source_root"]))
    source_path = PurePosixPath(source_logical_path)
    try:
        relative = source_path.relative_to(source_root)
    except ValueError as error:
        raise Hdf5SourceV4Error(
            f"source HDF5 is outside source_root: {source_logical_path}"
        ) from error
    logical = str(
        PurePosixPath("benchmark/sami_landslide_hdf5_v4/small/assets")
        / str(source["source_key"])
        / relative
    )
    validate_benchmark_logical_path(logical, require_hdf5=True)
    return logical


@dataclass(frozen=True, slots=True)
class SourceIndexSpec:
    """Normalized index projection used by both builder and validator."""

    source_key: str
    index_logical_path: str
    source_declared_split: str | None
    canonical_split: str


@dataclass(slots=True)
class SourceObservation:
    """One canonical source record plus arrays needed for streaming statistics."""

    source_record: Hdf5SourceRecordV1
    image_values: Any
    mask_values: Any
    channel_valid: Any
    pixel_valid: Any
    target_valid: Any
    image_source_path: Path
    mask_source_path: Path

    def to_source_record_payload(self) -> dict[str, Any]:
        return self.source_record.model_dump(mode="json")


def parse_source_index_spec(
    source_config: Any,
    index_config: Any,
) -> SourceIndexSpec:
    source = _as_mapping(source_config, location="source config")
    index = _as_mapping(index_config, location="source index config")
    required = {"logical_path", "source_declared_split", "canonical_split"}
    if set(index) != required:
        raise Hdf5SourceV4Error(
            "source index keys differ: "
            f"missing={sorted(required - set(index))}, "
            f"extra={sorted(set(index) - required)}"
        )
    validate_dataset_logical_path(str(index["logical_path"]))
    declared = index["source_declared_split"]
    canonical = index["canonical_split"]
    if declared is not None and declared != canonical:
        raise Hdf5SourceV4Error("native source split cannot be rewritten")
    if source["split_assurance"] == "train_only" and canonical != "train":
        raise Hdf5SourceV4Error("train-only source cannot emit val/test")
    return SourceIndexSpec(
        source_key=str(source["source_key"]),
        index_logical_path=str(index["logical_path"]),
        source_declared_split=declared,
        canonical_split=str(canonical),
    )


def iter_source_index(
    source_config: Any,
    datasets_root: Path,
) -> Iterator[tuple[SourceIndexSpec, int, dict[str, Any]]]:
    """Stream authoritative JSONL rows without retaining source-only fields."""

    source = _as_mapping(source_config, location="source config")
    if source["ingestion_status"] == "not_ready":
        if source.get("indexes"):
            raise Hdf5SourceV4Error("not-ready source must have no indexes")
        return
    indexes = source.get("indexes")
    if not isinstance(indexes, list):
        raise Hdf5SourceV4Error("ready source indexes must be an array")
    for raw_index in indexes:
        spec = parse_source_index_spec(source, raw_index)
        index_path = resolve_dataset_logical_path(
            spec.index_logical_path,
            datasets_root,
        )
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Hdf5SourceV4Error(
                        f"blank JSONL row at {spec.index_logical_path}:{line_number}"
                    )
                yield (
                    spec,
                    line_number,
                    _strict_json_object(
                        line,
                        location=f"{spec.index_logical_path}:{line_number}",
                    ),
                )


def _read_validity_dataset(
    image_handle: Any,
    mask_handle: Any,
    *,
    owner: str,
    key: str | None,
    role: str,
) -> Any | None:
    if key is None:
        if owner != "absent":
            raise Hdf5SourceV4Error(f"{role} owner/key availability disagrees")
        return None
    handles = {
        "image_hdf5": image_handle,
        "mask_hdf5": mask_handle,
    }
    if owner not in handles:
        raise Hdf5SourceV4Error(f"{role} has unsupported owner: {owner}")
    handle = handles[owner]
    if key not in handle:
        raise Hdf5SourceV4Error(f"{role} dataset is missing: {owner}:{key}")
    return handle[key][...]


def _binary_bool_array(values: Any, *, shape: tuple[int, ...], role: str) -> Any:
    import numpy as np

    array = np.asarray(values)
    if array.shape != shape:
        raise Hdf5SourceV4Error(
            f"{role} shape mismatch: expected {shape}, observed {array.shape}"
        )
    if not bool(np.isin(array, (0, 1, False, True)).all()):
        raise Hdf5SourceV4Error(f"{role} values must be binary")
    return array.astype(bool, copy=False)


def _group_payload(
    source: Mapping[str, Any],
    row: Mapping[str, Any],
) -> GroupIdentityV1:
    field = source.get("group_field")
    if field is None:
        return GroupIdentityV1(
            group_id=None,
            group_kind="unknown",
            evidence=tuple(source.get("group_evidence") or ()),
            completeness="unavailable",
        )
    raw_value = row.get(str(field))
    if raw_value is None or str(raw_value).strip() == "":
        raise Hdf5SourceV4Error(
            f"configured group field is absent or empty: {source['source_key']}:{field}"
        )
    return GroupIdentityV1(
        group_id=f"{source['source_key']}:{raw_value}",
        group_kind=source["group_kind"],
        evidence=tuple(source.get("group_evidence") or ()),
        completeness=source["group_completeness"],
    )


def _duplicate_payload(
    source: Mapping[str, Any],
    row: Mapping[str, Any],
) -> DuplicateComponentV1:
    field = source.get("duplicate_component_field")
    if field is None:
        return DuplicateComponentV1(
            component_id=None,
            evidence_level="unavailable",
        )
    raw_value = row.get(str(field))
    if raw_value is None or str(raw_value).strip() == "":
        return DuplicateComponentV1(
            component_id=None,
            evidence_level="unavailable",
        )
    return DuplicateComponentV1(
        component_id=f"{source['source_key']}:{raw_value}",
        evidence_level=source["duplicate_evidence_level"],
    )


def observe_source_row(
    source_config: Any,
    spec: SourceIndexSpec,
    row: Mapping[str, Any],
    *,
    datasets_root: Path,
) -> SourceObservation:
    """Reopen one HDF5 pair and create its strict canonical observation."""

    import h5py
    import numpy as np

    source = _as_mapping(source_config, location="source config")
    sample_field = str(source["sample_id_field"])
    raw_sample_id = row.get(sample_field)
    if not isinstance(raw_sample_id, (str, int)) or isinstance(raw_sample_id, bool):
        raise Hdf5SourceV4Error(f"invalid source sample id in field {sample_field!r}")
    source_sample_id = str(raw_sample_id)
    if not source_sample_id or any(ord(character) < 32 for character in source_sample_id):
        raise Hdf5SourceV4Error("source sample id is empty or contains a control character")

    row_split_field = source.get("row_split_field")
    if row_split_field is not None:
        observed_split = row.get(str(row_split_field))
        if observed_split != spec.source_declared_split:
            raise Hdf5SourceV4Error(
                f"source row split drift for {source['source_key']}:{source_sample_id}"
            )
    elif spec.source_declared_split is not None:
        raise Hdf5SourceV4Error("declared split requires a row_split_field")

    image_logical = _join_hdf5_logical_path(
        str(source["hdf5_base"]),
        row.get(str(source["image_path_field"])),
    )
    mask_logical = _join_hdf5_logical_path(
        str(source["hdf5_base"]),
        row.get(str(source["mask_path_field"])),
    )
    image_path = resolve_dataset_logical_path(image_logical, datasets_root)
    mask_path = resolve_dataset_logical_path(mask_logical, datasets_root)
    if not image_path.is_file() or not mask_path.is_file():
        missing = image_path if not image_path.is_file() else mask_path
        raise FileNotFoundError(missing)

    image_sha256 = sha256_file(image_path)
    mask_sha256 = sha256_file(mask_path)
    validity = ValidityDescriptorV1.from_mapping(source["validity"])
    channels = tuple(source["channels"])

    with h5py.File(image_path, "r") as image_handle, h5py.File(mask_path, "r") as mask_handle:
        if "/image" not in image_handle or "/mask" not in mask_handle:
            raise Hdf5SourceV4Error("required /image or /mask dataset is missing")
        image_dataset = image_handle["/image"]
        mask_dataset = mask_handle["/mask"]
        if image_dataset.ndim != 3 or mask_dataset.ndim != 2:
            raise Hdf5SourceV4Error("image/mask layout must be CHW/HW")
        if image_dataset.shape[1:] != mask_dataset.shape:
            raise Hdf5SourceV4Error("image and mask spatial shapes differ")
        if image_dataset.shape[0] != len(channels):
            raise Hdf5SourceV4Error("image channel count differs from channel contract")
        if np.dtype(image_dataset.dtype) != np.dtype("float32"):
            raise Hdf5SourceV4Error(
                f"/image dtype must be float32, observed {image_dataset.dtype}"
            )
        if np.dtype(mask_dataset.dtype) != np.dtype("uint8"):
            raise Hdf5SourceV4Error(
                f"/mask dtype must be uint8, observed {mask_dataset.dtype}"
            )

        image_values = image_dataset[...]
        mask_values = mask_dataset[...]
        channel_count, height, width = image_values.shape
        channel_raw = _read_validity_dataset(
            image_handle,
            mask_handle,
            owner=validity.channel_valid_owner,
            key=validity.channel_valid_key,
            role="channel_valid",
        )
        channel_valid = (
            np.ones((channel_count,), dtype=bool)
            if channel_raw is None
            else _binary_bool_array(
                channel_raw,
                shape=(channel_count,),
                role="channel_valid",
            )
        )
        valid_mask_raw = _read_validity_dataset(
            image_handle,
            mask_handle,
            owner=validity.valid_mask_owner,
            key=validity.valid_mask_key,
            role="valid_mask",
        )
        target_valid = (
            np.ones((height, width), dtype=bool)
            if valid_mask_raw is None
            else _binary_bool_array(
                valid_mask_raw,
                shape=(height, width),
                role="valid_mask",
            )
        )
        pixel_raw = _read_validity_dataset(
            image_handle,
            mask_handle,
            owner=validity.pixel_valid_owner,
            key=validity.pixel_valid_key,
            role="pixel_valid",
        )
        if validity.input_pixel_valid_derivation == "read_per_channel_pixel_valid":
            if pixel_raw is None:
                raise Hdf5SourceV4Error("per-channel validity dataset is required")
            pixel_valid = _binary_bool_array(
                pixel_raw,
                shape=(channel_count, height, width),
                role="pixel_valid",
            )
        elif (
            validity.input_pixel_valid_derivation
            == "broadcast_label_valid_to_present_channels"
        ):
            pixel_valid = np.broadcast_to(
                target_valid[None, :, :],
                (channel_count, height, width),
            ).copy()
        elif (
            validity.input_pixel_valid_derivation
            == "broadcast_present_channels_over_full_grid"
        ):
            pixel_valid = np.ones((channel_count, height, width), dtype=bool)
        else:
            raise Hdf5SourceV4Error("unsupported input validity derivation")
        pixel_valid &= channel_valid[:, None, None]

    if not bool(np.isin(mask_values, (0, 1)).all()):
        raise Hdf5SourceV4Error("/mask must contain only binary values")
    if not bool(np.isfinite(image_values[pixel_valid]).all()):
        raise Hdf5SourceV4Error("valid image pixels contain NaN or Inf")
    if int(np.count_nonzero(target_valid)) == 0:
        raise Hdf5SourceV4Error("target-valid population is empty")

    image_ref = Hdf5ArrayRefV1(
        source_logical_path=image_logical,
        benchmark_logical_path=_benchmark_asset_logical_path(
            source,
            image_logical,
        ),
        sha256=image_sha256,
        size_bytes=image_path.stat().st_size,
        dataset_key="/image",
        shape=tuple(int(size) for size in image_values.shape),
        dtype=str(image_values.dtype),
        layout="CHW",
        value_semantics=None,
    )
    mask_ref = Hdf5ArrayRefV1(
        source_logical_path=mask_logical,
        benchmark_logical_path=_benchmark_asset_logical_path(
            source,
            mask_logical,
        ),
        sha256=mask_sha256,
        size_bytes=mask_path.stat().st_size,
        dataset_key="/mask",
        shape=tuple(int(size) for size in mask_values.shape),
        dtype=str(mask_values.dtype),
        layout="HW",
        value_semantics={"0": "background", "1": "landslide"},
    )
    provenance = ScientificProvenanceV1.from_mapping(source["provenance"])
    source_record = seal_source_record(
        {
            "schema_version": "sami_hdf5_source_record_v1",
            "source_key": source["source_key"],
            "source_sample_id": source_sample_id,
            "ingestion_status": "ready",
            "source_declared_split": spec.source_declared_split,
            "canonical_split": spec.canonical_split,
            "split_assurance": source["split_assurance"],
            "evaluation_eligibility": source["evaluation_eligibility"],
            "group": _group_payload(source, row).model_dump(mode="json"),
            "duplicate_component": _duplicate_payload(
                source,
                row,
            ).model_dump(mode="json"),
            "image": image_ref.model_dump(mode="json"),
            "mask": mask_ref.model_dump(mode="json"),
            "channels": list(channels),
            "validity": validity.model_dump(mode="json"),
            "provenance": provenance.model_dump(mode="json"),
        }
    )
    return SourceObservation(
        source_record=source_record,
        image_values=image_values,
        mask_values=mask_values,
        channel_valid=channel_valid,
        pixel_valid=pixel_valid,
        target_valid=target_valid,
        image_source_path=image_path,
        mask_source_path=mask_path,
    )


def iter_source_observations(
    source_config: Any,
    datasets_root: Path,
) -> Iterator[SourceObservation]:
    """Yield one fully replayed observation at a time for build or validation."""

    source = _as_mapping(source_config, location="source config")
    if source["ingestion_status"] == "not_ready":
        if source.get("indexes"):
            raise Hdf5SourceV4Error("not-ready source must not declare indexes")
        return
    channel_schema = source.get("channel_schema")
    channel_schema_sha256 = source.get("channel_schema_sha256")
    if not isinstance(channel_schema, str) or not isinstance(
        channel_schema_sha256,
        str,
    ):
        raise Hdf5SourceV4Error("ready source lacks channel-schema binding")
    channel_schema_path = resolve_dataset_logical_path(
        channel_schema,
        datasets_root,
    )
    if sha256_file(channel_schema_path) != channel_schema_sha256:
        raise Hdf5SourceV4Error(
            f"channel schema hash drift: {source['source_key']}"
        )
    for spec, _line_number, row in iter_source_index(source, datasets_root):
        yield observe_source_row(
            source,
            spec,
            row,
            datasets_root=datasets_root,
        )


def iter_source_asset_paths(
    source_config: Any,
    datasets_root: Path,
) -> Iterator[tuple[str, str, Path]]:
    """Yield the exact image/mask files selected for Benchmark materialization."""

    source = _as_mapping(source_config, location="source config")
    if source["ingestion_status"] == "not_ready":
        return
    for _spec, _line_number, row in iter_source_index(source, datasets_root):
        for role, field_name in (
            ("image", str(source["image_path_field"])),
            ("mask", str(source["mask_path_field"])),
        ):
            logical = _join_hdf5_logical_path(
                str(source["hdf5_base"]),
                row.get(field_name),
            )
            physical = resolve_dataset_logical_path(logical, datasets_root)
            if not physical.is_file():
                raise FileNotFoundError(physical)
            yield role, logical, physical


__all__ = [
    "Hdf5SourceV4Error",
    "SourceIndexSpec",
    "SourceObservation",
    "iter_source_index",
    "iter_source_asset_paths",
    "iter_source_observations",
    "observe_source_row",
    "parse_source_index_spec",
    "resolve_dataset_logical_path",
    "resolve_benchmark_logical_path",
]
