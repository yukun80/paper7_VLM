"""P1 HDF5-first Canonical Benchmark v4 strict data contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping, Self, TypeAlias


IngestionStatus: TypeAlias = Literal["ready", "not_ready"]
CanonicalSplit: TypeAlias = Literal["train", "val", "test"]
SplitAssurance: TypeAlias = Literal[
    "verified_group_isolated",
    "source_declared_unverified",
    "train_only",
]
EvaluationEligibility: TypeAlias = Literal["strict", "exploratory", "train_only"]
GroupKind: TypeAlias = Literal["scene", "event", "location", "source_sample", "unknown"]
GroupCompleteness: TypeAlias = Literal["verified", "partial", "unavailable"]
ModalityFamily: TypeAlias = Literal[
    "optical",
    "multispectral",
    "dem",
    "slope",
    "sar",
    "insar",
    "other",
]
NormalizationKind: TypeAlias = Literal[
    "none",
    "divide_255",
    "zscore_valid_pixels",
    "source_preprocessed",
]
ValiditySource: TypeAlias = Literal[
    "channel_valid",
    "pixel_valid",
    "valid_mask",
    "implicit_present",
]
ValidityOwner: TypeAlias = Literal["image_hdf5", "mask_hdf5", "absent"]
PixelValidityDerivation: TypeAlias = Literal[
    "read_per_channel_pixel_valid",
    "broadcast_label_valid_to_present_channels",
    "broadcast_present_channels_over_full_grid",
]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

SHA256_PATTERN_LENGTH = 64
_CANONICAL_SPLITS = frozenset({"train", "val", "test"})
_INGESTION_STATUSES = frozenset({"ready", "not_ready"})
_MODALITY_FAMILIES = frozenset(
    {"optical", "multispectral", "dem", "slope", "sar", "insar", "other"}
)
_NORMALIZATION_KINDS = frozenset(
    {"none", "divide_255", "zscore_valid_pixels", "source_preprocessed"}
)
_VALIDITY_SOURCES = frozenset(
    {"channel_valid", "pixel_valid", "valid_mask", "implicit_present"}
)
_GROUP_KINDS = frozenset({"scene", "event", "location", "source_sample", "unknown"})
_GROUP_COMPLETENESS = frozenset({"verified", "partial", "unavailable"})
_VALIDITY_OWNERS = frozenset({"image_hdf5", "mask_hdf5", "absent"})
_PIXEL_VALIDITY_DERIVATIONS = frozenset(
    {
        "read_per_channel_pixel_valid",
        "broadcast_label_valid_to_present_channels",
        "broadcast_present_channels_over_full_grid",
    }
)
_ASSURANCE_ELIGIBILITY = {
    "verified_group_isolated": "strict",
    "source_declared_unverified": "exploratory",
    "train_only": "train_only",
}
_NON_SPECTRAL_MODALITIES = frozenset({"dem", "slope", "sar", "insar"})


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    actual = set(payload)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{location} keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def validate_sha256(value: str, *, field_name: str = "sha256") -> str:
    """Validate a lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != SHA256_PATTERN_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return value


def validate_portable_path(
    value: str,
    *,
    field_name: str,
    required_prefix: str | None = None,
) -> str:
    """Validate a relative POSIX path without machine-specific state."""

    _require_non_empty(value, field_name=field_name)
    if "\\" in value:
        raise ValueError(f"{field_name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field_name} must be relative without traversal")
    if required_prefix is not None and not value.startswith(required_prefix):
        raise ValueError(f"{field_name} must start with {required_prefix!r}")
    return value


def validate_dataset_logical_path(
    value: str,
    *,
    field_name: str = "logical_path",
    require_hdf5: bool = False,
) -> str:
    """Validate a canonical ``datasets/...`` reference."""

    validate_portable_path(value, field_name=field_name, required_prefix="datasets/")
    if value == "datasets/":
        raise ValueError(f"{field_name} must identify an asset below datasets/")
    if require_hdf5 and PurePosixPath(value).suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"{field_name} must identify an HDF5 file")
    return value


def validate_benchmark_logical_path(
    value: str,
    *,
    field_name: str = "benchmark_logical_path",
    require_hdf5: bool = False,
) -> str:
    """Validate a portable path inside the frozen Benchmark v4 package."""

    prefix = "benchmark/sami_landslide_hdf5_v4/small/"
    validate_portable_path(value, field_name=field_name, required_prefix=prefix)
    if value == prefix:
        raise ValueError(f"{field_name} must identify an asset below Benchmark v4")
    if require_hdf5 and PurePosixPath(value).suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"{field_name} must identify an HDF5 file")
    return value


def _json_ready(value: Any) -> JsonValue:
    if is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract payload contains NaN or Inf")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported contract value type: {type(value).__name__}")


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one finite JSON mapping using the repository canonical JSON convention."""

    rendered = json.dumps(
        _json_ready(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(f"{rendered}\n".encode("utf-8")).hexdigest()


class JsonDataclass:
    """Small dataclass compatibility surface used by builders and validators."""

    def model_dump(self, *, mode: Literal["json"] = "json") -> dict[str, JsonValue]:
        if mode != "json":
            raise ValueError("only model_dump(mode='json') is supported")
        result = _json_ready(self)
        if not isinstance(result, dict):
            raise TypeError("contract serialization did not produce a mapping")
        return result


@dataclass(frozen=True, slots=True)
class ChannelDescriptorV1(JsonDataclass):
    """Scientific identity and known/unknown state for one scalar channel."""

    index: int
    channel_key: str
    display_name: str
    modality_family: ModalityFamily
    physical_unit: str | None
    wavelength_nm: float | None
    wavelength_known: bool
    gsd_m: float | None
    gsd_known: bool
    normalization: NormalizationKind
    validity_source: ValiditySource
    schema_version: Literal["sami_channel_descriptor_v1"] = "sami_channel_descriptor_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "sami_channel_descriptor_v1":
            raise ValueError("invalid ChannelDescriptorV1 schema_version")
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("channel index must be a non-negative integer")
        _require_non_empty(self.channel_key, field_name="channel_key")
        _require_non_empty(self.display_name, field_name="display_name")
        if self.modality_family not in _MODALITY_FAMILIES:
            raise ValueError("modality_family is invalid")
        if self.normalization not in _NORMALIZATION_KINDS:
            raise ValueError("normalization is invalid")
        if self.validity_source not in _VALIDITY_SOURCES:
            raise ValueError("validity_source is invalid")
        if self.physical_unit is not None:
            _require_non_empty(self.physical_unit, field_name="physical_unit")
        if self.wavelength_known != (self.wavelength_nm is not None):
            raise ValueError("wavelength_known must exactly match wavelength_nm availability")
        if self.wavelength_nm is not None and (
            not math.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0.0
        ):
            raise ValueError("known wavelength_nm must be finite and positive")
        if self.gsd_known != (self.gsd_m is not None):
            raise ValueError("gsd_known must exactly match gsd_m availability")
        if self.gsd_m is not None and (
            not math.isfinite(self.gsd_m) or self.gsd_m <= 0.0
        ):
            raise ValueError("known gsd_m must be finite and positive")
        if self.modality_family in _NON_SPECTRAL_MODALITIES and (
            self.wavelength_known or self.wavelength_nm is not None
        ):
            raise ValueError("non-spectral modalities cannot declare wavelength")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, location="ChannelDescriptorV1")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class GroupIdentityV1(JsonDataclass):
    """Best available grouping evidence without overstating completeness."""

    group_id: str | None
    group_kind: GroupKind
    evidence: tuple[str, ...]
    completeness: GroupCompleteness

    def __post_init__(self) -> None:
        if self.group_kind not in _GROUP_KINDS:
            raise ValueError("group_kind is invalid")
        if self.completeness not in _GROUP_COMPLETENESS:
            raise ValueError("group completeness is invalid")
        if self.group_id is not None:
            _require_non_empty(self.group_id, field_name="group_id")
        for item in self.evidence:
            _require_non_empty(item, field_name="group.evidence[]")
        if self.completeness == "verified" and self.group_id is None:
            raise ValueError("verified group evidence requires group_id")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {"group_id", "group_kind", "evidence", "completeness"},
            location="GroupIdentityV1",
        )
        return cls(
            group_id=payload["group_id"],
            group_kind=payload["group_kind"],
            evidence=tuple(payload["evidence"]),
            completeness=payload["completeness"],
        )


@dataclass(frozen=True, slots=True)
class DuplicateComponentV1(JsonDataclass):
    """Known duplicate-component evidence, including explicit unavailability."""

    component_id: str | None
    evidence_level: Literal["verified", "candidate", "unavailable"]

    def __post_init__(self) -> None:
        if self.evidence_level not in {"verified", "candidate", "unavailable"}:
            raise ValueError("duplicate evidence_level is invalid")
        if self.component_id is not None:
            _require_non_empty(self.component_id, field_name="component_id")
        if self.evidence_level == "verified" and self.component_id is None:
            raise ValueError("verified duplicate component requires component_id")
        if self.evidence_level == "unavailable" and self.component_id is not None:
            raise ValueError("unavailable duplicate evidence requires component_id=null")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {"component_id", "evidence_level"},
            location="DuplicateComponentV1",
        )
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class Hdf5ArrayRefV1(JsonDataclass):
    """Content-bound source/copy pair for one dataset inside one HDF5 file."""

    source_logical_path: str
    benchmark_logical_path: str
    sha256: str
    size_bytes: int
    dataset_key: str
    shape: tuple[int, ...]
    dtype: str
    layout: Literal["CHW", "HW"]
    value_semantics: dict[str, str] | None

    def __post_init__(self) -> None:
        validate_dataset_logical_path(
            self.source_logical_path,
            field_name="source_logical_path",
            require_hdf5=True,
        )
        validate_benchmark_logical_path(
            self.benchmark_logical_path,
            require_hdf5=True,
        )
        expected_prefix = "benchmark/sami_landslide_hdf5_v4/small/assets/"
        if not self.benchmark_logical_path.startswith(expected_prefix):
            raise ValueError("benchmark HDF5 copy must be stored below assets/")
        validate_sha256(self.sha256)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("HDF5 size_bytes must be a positive integer")
        if not self.dataset_key.startswith("/") or "//" in self.dataset_key:
            raise ValueError("dataset_key must be an absolute HDF5 key")
        if any(part in {"", ".", ".."} for part in self.dataset_key[1:].split("/")):
            raise ValueError("dataset_key contains an invalid segment")
        if self.layout not in {"CHW", "HW"}:
            raise ValueError("HDF5 array layout must be CHW or HW")
        expected_rank = 3 if self.layout == "CHW" else 2
        if len(self.shape) != expected_rank or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.shape
        ):
            raise ValueError(f"{self.layout} shape must contain {expected_rank} positive integers")
        _require_non_empty(self.dtype, field_name="dtype")
        if self.layout == "CHW" and self.value_semantics is not None:
            raise ValueError("image HDF5 references cannot declare mask value semantics")
        if self.layout == "HW" and self.value_semantics != {
            "0": "background",
            "1": "landslide",
        }:
            raise ValueError("mask value_semantics must be the frozen binary mapping")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {
                "source_logical_path",
                "benchmark_logical_path",
                "sha256",
                "size_bytes",
                "dataset_key",
                "shape",
                "dtype",
                "layout",
                "value_semantics",
            },
            location="Hdf5ArrayRefV1",
        )
        return cls(
            source_logical_path=payload["source_logical_path"],
            benchmark_logical_path=payload["benchmark_logical_path"],
            sha256=payload["sha256"],
            size_bytes=payload["size_bytes"],
            dataset_key=payload["dataset_key"],
            shape=tuple(payload["shape"]),
            dtype=payload["dtype"],
            layout=payload["layout"],
            value_semantics=(
                None
                if payload["value_semantics"] is None
                else dict(payload["value_semantics"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidityDescriptorV1(JsonDataclass):
    """Exact storage ownership and derivation of label/input validity."""

    valid_mask_key: str | None
    valid_mask_owner: ValidityOwner
    pixel_valid_key: str | None
    pixel_valid_owner: ValidityOwner
    channel_valid_key: str | None
    channel_valid_owner: ValidityOwner
    label_valid_semantics: str
    input_pixel_valid_derivation: PixelValidityDerivation
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        for key_name, owner_name in (
            ("valid_mask_key", "valid_mask_owner"),
            ("pixel_valid_key", "pixel_valid_owner"),
            ("channel_valid_key", "channel_valid_owner"),
        ):
            key = getattr(self, key_name)
            owner = getattr(self, owner_name)
            if owner not in _VALIDITY_OWNERS:
                raise ValueError(f"{owner_name} is invalid")
            if (key is None) != (owner == "absent"):
                raise ValueError(f"{key_name} and {owner_name} availability disagree")
            if key is not None and not key.startswith("/"):
                raise ValueError(f"{key_name} must be an absolute HDF5 dataset key")
        _require_non_empty(
            self.label_valid_semantics,
            field_name="label_valid_semantics",
        )
        for note in self.notes:
            _require_non_empty(note, field_name="validity.notes[]")
        if self.input_pixel_valid_derivation not in _PIXEL_VALIDITY_DERIVATIONS:
            raise ValueError("input_pixel_valid_derivation is invalid")
        if (
            self.input_pixel_valid_derivation == "read_per_channel_pixel_valid"
            and self.pixel_valid_key is None
        ):
            raise ValueError("per-channel validity derivation requires pixel_valid_key")
        if (
            self.input_pixel_valid_derivation
            == "broadcast_label_valid_to_present_channels"
            and self.valid_mask_key is None
        ):
            raise ValueError("label-valid broadcast requires valid_mask_key")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, location="ValidityDescriptorV1")
        return cls(
            **{
                **dict(payload),
                "notes": tuple(payload["notes"]),
            }
        )


@dataclass(frozen=True, slots=True)
class ScientificProvenanceV1(JsonDataclass):
    """Non-gating scientific provenance permitted in P1 records."""

    source_name: str
    source_document: str | None
    citation_key: str
    upstream_url: str | None
    provenance_notes: str

    def __post_init__(self) -> None:
        _require_non_empty(self.source_name, field_name="source_name")
        _require_non_empty(self.citation_key, field_name="citation_key")
        _require_non_empty(self.provenance_notes, field_name="provenance_notes")
        if self.source_document is not None:
            validate_dataset_logical_path(
                self.source_document,
                field_name="source_document",
            )
        if self.upstream_url is not None and not self.upstream_url.startswith("https://"):
            raise ValueError("upstream_url must use HTTPS")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, location="ScientificProvenanceV1")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class RegisteredRGBViewV1(JsonDataclass):
    """Metadata-only RGB view tied to train-only normalization."""

    source_indices: tuple[int, int, int]
    channel_keys: tuple[str, str, str]
    normalization_binding: str
    mapping_evidence: str
    schema_version: Literal["sami_registered_rgb_view_v1"] = (
        "sami_registered_rgb_view_v1"
    )
    view_id: Literal["registered_rgb"] = "registered_rgb"
    role: Literal["rgb"] = "rgb"

    def __post_init__(self) -> None:
        if self.schema_version != "sami_registered_rgb_view_v1":
            raise ValueError("invalid RegisteredRGBViewV1 schema_version")
        if self.view_id != "registered_rgb" or self.role != "rgb":
            raise ValueError("registered RGB view identity is invalid")
        if len(set(self.source_indices)) != 3 or min(self.source_indices) < 0:
            raise ValueError("registered RGB indices must be three unique non-negative values")
        if len(set(self.channel_keys)) != 3:
            raise ValueError("registered RGB channel keys must be unique")
        for key in self.channel_keys:
            _require_non_empty(key, field_name="registered_rgb.channel_keys[]")
        validate_sha256(
            self.normalization_binding,
            field_name="normalization_binding",
        )
        _require_non_empty(self.mapping_evidence, field_name="mapping_evidence")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, location="RegisteredRGBViewV1")
        return cls(
            source_indices=tuple(payload["source_indices"]),
            channel_keys=tuple(payload["channel_keys"]),
            normalization_binding=payload["normalization_binding"],
            mapping_evidence=payload["mapping_evidence"],
            schema_version=payload["schema_version"],
            view_id=payload["view_id"],
            role=payload["role"],
        )


@dataclass(frozen=True, slots=True)
class Hdf5SourceRecordV1(JsonDataclass):
    """One strict image/mask HDF5 source binding."""

    source_key: str
    source_sample_id: str
    ingestion_status: IngestionStatus
    source_declared_split: CanonicalSplit | None
    canonical_split: CanonicalSplit
    split_assurance: SplitAssurance
    evaluation_eligibility: EvaluationEligibility
    group: GroupIdentityV1
    duplicate_component: DuplicateComponentV1
    image: Hdf5ArrayRefV1
    mask: Hdf5ArrayRefV1
    channels: tuple[ChannelDescriptorV1, ...]
    validity: ValidityDescriptorV1
    provenance: ScientificProvenanceV1
    record_sha256: str
    schema_version: Literal["sami_hdf5_source_record_v1"] = (
        "sami_hdf5_source_record_v1"
    )

    def __post_init__(self) -> None:
        if self.schema_version != "sami_hdf5_source_record_v1":
            raise ValueError("invalid Hdf5SourceRecordV1 schema_version")
        _require_non_empty(self.source_key, field_name="source_key")
        _require_non_empty(self.source_sample_id, field_name="source_sample_id")
        if self.ingestion_status not in _INGESTION_STATUSES:
            raise ValueError("ingestion_status is invalid")
        if self.ingestion_status != "ready":
            raise ValueError("not-ready sources cannot produce source records")
        if self.source_declared_split is not None and (
            self.source_declared_split not in _CANONICAL_SPLITS
        ):
            raise ValueError("source_declared_split is invalid")
        if (
            self.source_declared_split is not None
            and self.source_declared_split != self.canonical_split
        ):
            raise ValueError("source-declared split cannot be rewritten")
        _validate_assurance(
            canonical_split=self.canonical_split,
            split_assurance=self.split_assurance,
            evaluation_eligibility=self.evaluation_eligibility,
        )
        if self.image.layout != "CHW" or self.mask.layout != "HW":
            raise ValueError("source image/mask layouts must be CHW/HW")
        if self.image.dataset_key != "/image" or self.mask.dataset_key != "/mask":
            raise ValueError("source image/mask dataset keys must be /image and /mask")
        if self.image.shape[1:] != self.mask.shape:
            raise ValueError("image and mask spatial shapes must match")
        if len(self.channels) != self.image.shape[0]:
            raise ValueError("channel descriptor count must equal image channel count")
        if tuple(channel.index for channel in self.channels) != tuple(
            range(len(self.channels))
        ):
            raise ValueError("channel indices must be contiguous and ordered")
        validate_sha256(self.record_sha256, field_name="record_sha256")
        if self.record_sha256 != _sealed_digest(self.model_dump(mode="json")):
            raise ValueError("source record_sha256 does not match canonical payload")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, location="Hdf5SourceRecordV1")
        return cls(
            source_key=payload["source_key"],
            source_sample_id=payload["source_sample_id"],
            ingestion_status=payload["ingestion_status"],
            source_declared_split=payload["source_declared_split"],
            canonical_split=payload["canonical_split"],
            split_assurance=payload["split_assurance"],
            evaluation_eligibility=payload["evaluation_eligibility"],
            group=GroupIdentityV1.from_mapping(payload["group"]),
            duplicate_component=DuplicateComponentV1.from_mapping(
                payload["duplicate_component"]
            ),
            image=Hdf5ArrayRefV1.from_mapping(payload["image"]),
            mask=Hdf5ArrayRefV1.from_mapping(payload["mask"]),
            channels=tuple(
                ChannelDescriptorV1.from_mapping(item) for item in payload["channels"]
            ),
            validity=ValidityDescriptorV1.from_mapping(payload["validity"]),
            provenance=ScientificProvenanceV1.from_mapping(payload["provenance"]),
            record_sha256=payload["record_sha256"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class CanonicalParentV4(JsonDataclass):
    """Canonical sample projection bound to one HDF5 source record."""

    parent_id: str
    source_key: str
    source_sample_id: str
    canonical_split: CanonicalSplit
    split_assurance: SplitAssurance
    evaluation_eligibility: EvaluationEligibility
    channels: tuple[ChannelDescriptorV1, ...]
    image_ref: Hdf5ArrayRefV1
    mask_ref: Hdf5ArrayRefV1
    validity_ref: ValidityDescriptorV1
    registered_views: tuple[RegisteredRGBViewV1, ...]
    group: GroupIdentityV1
    source_record_sha256: str
    record_sha256: str
    schema_version: Literal["sami_canonical_parent_v4"] = "sami_canonical_parent_v4"

    def __post_init__(self) -> None:
        if self.schema_version != "sami_canonical_parent_v4":
            raise ValueError("invalid CanonicalParentV4 schema_version")
        _require_non_empty(self.parent_id, field_name="parent_id")
        _require_non_empty(self.source_key, field_name="source_key")
        _require_non_empty(self.source_sample_id, field_name="source_sample_id")
        _validate_assurance(
            canonical_split=self.canonical_split,
            split_assurance=self.split_assurance,
            evaluation_eligibility=self.evaluation_eligibility,
        )
        validate_sha256(
            self.source_record_sha256,
            field_name="source_record_sha256",
        )
        validate_sha256(self.record_sha256, field_name="record_sha256")
        if self.record_sha256 != _sealed_digest(self.model_dump(mode="json")):
            raise ValueError("canonical parent record_sha256 does not match payload")
        for view in self.registered_views:
            if max(view.source_indices) >= len(self.channels):
                raise ValueError("registered RGB source index exceeds channel count")
            selected = tuple(
                self.channels[index].channel_key for index in view.source_indices
            )
            if selected != view.channel_keys:
                raise ValueError("registered RGB keys do not match selected channels")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        _require_exact_keys(payload, expected, location="CanonicalParentV4")
        return cls(
            parent_id=payload["parent_id"],
            source_key=payload["source_key"],
            source_sample_id=payload["source_sample_id"],
            canonical_split=payload["canonical_split"],
            split_assurance=payload["split_assurance"],
            evaluation_eligibility=payload["evaluation_eligibility"],
            channels=tuple(
                ChannelDescriptorV1.from_mapping(item) for item in payload["channels"]
            ),
            image_ref=Hdf5ArrayRefV1.from_mapping(payload["image_ref"]),
            mask_ref=Hdf5ArrayRefV1.from_mapping(payload["mask_ref"]),
            validity_ref=ValidityDescriptorV1.from_mapping(payload["validity_ref"]),
            registered_views=tuple(
                RegisteredRGBViewV1.from_mapping(item)
                for item in payload["registered_views"]
            ),
            group=GroupIdentityV1.from_mapping(payload["group"]),
            source_record_sha256=payload["source_record_sha256"],
            record_sha256=payload["record_sha256"],
            schema_version=payload["schema_version"],
        )


def _validate_assurance(
    *,
    canonical_split: str,
    split_assurance: str,
    evaluation_eligibility: str,
) -> None:
    if canonical_split not in _CANONICAL_SPLITS:
        raise ValueError("canonical_split is invalid")
    expected = _ASSURANCE_ELIGIBILITY.get(split_assurance)
    if expected is None or evaluation_eligibility != expected:
        raise ValueError("split assurance and evaluation eligibility disagree")
    if split_assurance == "train_only" and canonical_split != "train":
        raise ValueError("train_only records must use canonical_split=train")


def _sealed_digest(payload: Mapping[str, Any]) -> str:
    core = dict(payload)
    core.pop("record_sha256", None)
    return canonical_payload_sha256(core)


def seal_source_record(payload: Mapping[str, Any]) -> Hdf5SourceRecordV1:
    """Add and validate ``record_sha256`` for one unsealed source-record payload."""

    core = dict(payload)
    core.setdefault("schema_version", "sami_hdf5_source_record_v1")
    if "record_sha256" in core:
        raise ValueError("seal_source_record expects an unsealed payload")
    core["record_sha256"] = canonical_payload_sha256(core)
    return Hdf5SourceRecordV1.from_mapping(core)


def seal_canonical_parent(payload: Mapping[str, Any]) -> CanonicalParentV4:
    """Add and validate ``record_sha256`` for one unsealed canonical parent."""

    core = dict(payload)
    core.setdefault("schema_version", "sami_canonical_parent_v4")
    if "record_sha256" in core:
        raise ValueError("seal_canonical_parent expects an unsealed payload")
    core["record_sha256"] = canonical_payload_sha256(core)
    return CanonicalParentV4.from_mapping(core)


@dataclass(frozen=True, slots=True)
class NormalizationChannelStatisticV1(JsonDataclass):
    """Train-valid scalar statistics for one source/channel identity."""

    source_key: str
    channel_key: str
    channel_index: int
    valid_pixel_count: int
    mean: float
    std: float

    def __post_init__(self) -> None:
        _require_non_empty(self.source_key, field_name="source_key")
        _require_non_empty(self.channel_key, field_name="channel_key")
        if self.channel_index < 0 or self.valid_pixel_count <= 0:
            raise ValueError("normalization counts and indices must be positive/valid")
        if not all(math.isfinite(value) for value in (self.mean, self.std)):
            raise ValueError("normalization statistics must be finite")
        if self.std < 0.0:
            raise ValueError("normalization std cannot be negative")


@dataclass(frozen=True, slots=True)
class NormalizationManifestV1(JsonDataclass):
    """Normalization artifact bound only to canonical-train valid pixels."""

    population_split: Literal["train"]
    source_index_sha256: str
    population_sha256: str
    statistics: tuple[NormalizationChannelStatisticV1, ...]
    aggregate_sha256: str
    schema_version: Literal["sami_normalization_manifest_v1"] = (
        "sami_normalization_manifest_v1"
    )
    protocol: Literal["zscore_canonical_train_valid_pixels_v1"] = (
        "zscore_canonical_train_valid_pixels_v1"
    )

    def __post_init__(self) -> None:
        if self.schema_version != "sami_normalization_manifest_v1":
            raise ValueError("invalid normalization schema_version")
        if self.protocol != "zscore_canonical_train_valid_pixels_v1":
            raise ValueError("invalid normalization protocol")
        if self.population_split != "train" or not self.statistics:
            raise ValueError("normalization requires non-empty train statistics")
        for name in (
            "source_index_sha256",
            "population_sha256",
            "aggregate_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        statistic_payloads = [
            statistic.model_dump(mode="json") for statistic in self.statistics
        ]
        if canonical_payload_sha256(statistic_payloads) != self.aggregate_sha256:
            raise ValueError("normalization aggregate_sha256 mismatch")


@dataclass(frozen=True, slots=True)
class BenchmarkManifestV4(JsonDataclass):
    """Top-level immutable binding for Benchmark v4 artifacts."""

    benchmark_name: str
    mode: Literal["small"]
    benchmark_relative_path: str
    config_sha256: str
    source_contract_sha256: str
    source_inventory_sha256: str
    source_registry_sha256: str
    normalization_binding_sha256: str
    channel_catalog_sha256: str
    materialization_index_sha256: str
    artifact_bindings: dict[str, str]
    materialized_asset_count: int
    materialized_size_bytes: int
    source_record_count: int
    parent_count: int
    split_counts: dict[str, int]
    assurance_counts: dict[str, int]
    eligibility_counts: dict[str, int]
    strict_generalization_status: Literal["unavailable", "available"]
    aggregate_sha256: str
    schema_version: Literal["sami_benchmark_manifest_v4"] = (
        "sami_benchmark_manifest_v4"
    )
    protocol: Literal["sami_hdf5_materialized_copy_v1"] = (
        "sami_hdf5_materialized_copy_v1"
    )

    def __post_init__(self) -> None:
        if self.schema_version != "sami_benchmark_manifest_v4":
            raise ValueError("invalid benchmark manifest schema_version")
        if self.protocol != "sami_hdf5_materialized_copy_v1":
            raise ValueError("invalid benchmark manifest protocol")
        if self.mode != "small":
            raise ValueError("Benchmark v4 mode must be small")
        validate_portable_path(
            self.benchmark_relative_path,
            field_name="benchmark_relative_path",
        )
        for name in (
            "config_sha256",
            "source_contract_sha256",
            "source_inventory_sha256",
            "source_registry_sha256",
            "normalization_binding_sha256",
            "channel_catalog_sha256",
            "materialization_index_sha256",
            "aggregate_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        _validate_count_mapping(self.split_counts, allowed=_CANONICAL_SPLITS)
        _validate_count_mapping(
            self.assurance_counts,
            allowed=frozenset(_ASSURANCE_ELIGIBILITY),
        )
        _validate_count_mapping(
            self.eligibility_counts,
            allowed=frozenset(_ASSURANCE_ELIGIBILITY.values()),
        )
        if min(
            self.source_record_count,
            self.parent_count,
            self.materialized_asset_count,
            self.materialized_size_bytes,
        ) < 0:
            raise ValueError("record counts cannot be negative")
        if self.materialized_asset_count == 0 or self.materialized_size_bytes == 0:
            raise ValueError("materialized Benchmark must contain copied HDF5 bytes")
        if self.source_record_count != self.parent_count:
            raise ValueError("source and parent counts must match")
        if sum(self.split_counts.values()) != self.parent_count:
            raise ValueError("split counts must cover every parent exactly once")
        strict_count = self.eligibility_counts.get("strict", 0)
        expected_status = "unavailable" if strict_count == 0 else "available"
        if self.strict_generalization_status != expected_status:
            raise ValueError("strict generalization status disagrees with population")
        for path, digest in self.artifact_bindings.items():
            validate_portable_path(path, field_name="artifact_bindings path")
            validate_sha256(digest, field_name=f"artifact_bindings[{path}]")
        if canonical_payload_sha256(
            dict(sorted(self.artifact_bindings.items()))
        ) != self.aggregate_sha256:
            raise ValueError("manifest artifact aggregate_sha256 mismatch")


@dataclass(frozen=True, slots=True)
class BenchmarkStatisticsV4(JsonDataclass):
    """Disjoint P1 population statistics with no strict-cohort overclaim."""

    source_counts: dict[str, dict[str, int]]
    source_record_count: int
    parent_count: int
    positive_count: int
    no_target_count: int
    split_counts: dict[str, int]
    assurance_counts: dict[str, int]
    eligibility_counts: dict[str, int]
    strict_population: int
    strict_generalization_status: Literal["unavailable", "available"]
    normalization_binding_sha256: str
    aggregate_sha256: str
    schema_version: Literal["sami_benchmark_statistics_v4"] = (
        "sami_benchmark_statistics_v4"
    )

    def __post_init__(self) -> None:
        if self.schema_version != "sami_benchmark_statistics_v4":
            raise ValueError("invalid statistics schema_version")
        counts = (
            self.source_record_count,
            self.parent_count,
            self.positive_count,
            self.no_target_count,
            self.strict_population,
        )
        if any(value < 0 for value in counts):
            raise ValueError("statistics counts cannot be negative")
        _validate_count_mapping(self.split_counts, allowed=_CANONICAL_SPLITS)
        _validate_count_mapping(
            self.assurance_counts,
            allowed=frozenset(_ASSURANCE_ELIGIBILITY),
        )
        _validate_count_mapping(
            self.eligibility_counts,
            allowed=frozenset(_ASSURANCE_ELIGIBILITY.values()),
        )
        validate_sha256(
            self.normalization_binding_sha256,
            field_name="normalization_binding_sha256",
        )
        validate_sha256(self.aggregate_sha256, field_name="aggregate_sha256")
        if self.source_record_count != self.parent_count:
            raise ValueError("statistics source and parent counts must match")
        if self.positive_count + self.no_target_count != self.parent_count:
            raise ValueError("positive/no-target counts must partition parents")
        if sum(self.split_counts.values()) != self.parent_count:
            raise ValueError("statistics split counts must partition parents")
        if self.strict_population != self.eligibility_counts.get("strict", 0):
            raise ValueError("strict_population disagrees with eligibility counts")
        if self.strict_population == 0 and self.strict_generalization_status != "unavailable":
            raise ValueError("zero strict population requires unavailable status")
        core = self.model_dump(mode="json")
        core.pop("aggregate_sha256")
        core.pop("schema_version")
        if canonical_payload_sha256(core) != self.aggregate_sha256:
            raise ValueError("statistics aggregate_sha256 mismatch")


@dataclass(frozen=True, slots=True)
class BenchmarkValidationReportV4(JsonDataclass):
    """Independent-validator result; only ``errors=[]`` can support P1 acceptance."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    benchmark_manifest_sha256: str
    artifact_count: int
    materialized_asset_count: int
    materialized_size_bytes: int
    source_record_count: int
    parent_count: int
    split_counts: dict[str, int]
    assurance_counts: dict[str, int]
    eligibility_counts: dict[str, int]
    normalization_binding_sha256: str
    aggregate_sha256: str
    schema_version: Literal["sami_benchmark_validation_report_v4"] = (
        "sami_benchmark_validation_report_v4"
    )
    protocol: Literal["sami_hdf5_materialized_copy_validator_v1"] = (
        "sami_hdf5_materialized_copy_validator_v1"
    )

    def __post_init__(self) -> None:
        if self.schema_version != "sami_benchmark_validation_report_v4":
            raise ValueError("invalid validation report schema_version")
        if self.protocol != "sami_hdf5_materialized_copy_validator_v1":
            raise ValueError("invalid validation report protocol")
        for message in (*self.errors, *self.warnings):
            _require_non_empty(message, field_name="validation message")
        for name in (
            "benchmark_manifest_sha256",
            "normalization_binding_sha256",
            "aggregate_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if min(
            self.artifact_count,
            self.materialized_asset_count,
            self.materialized_size_bytes,
            self.source_record_count,
            self.parent_count,
        ) < 0:
            raise ValueError("validation counts cannot be negative")
        _validate_count_mapping(self.split_counts, allowed=_CANONICAL_SPLITS)
        _validate_count_mapping(
            self.assurance_counts,
            allowed=frozenset(_ASSURANCE_ELIGIBILITY),
        )
        _validate_count_mapping(
            self.eligibility_counts,
            allowed=frozenset(_ASSURANCE_ELIGIBILITY.values()),
        )
        core = self.model_dump(mode="json")
        core.pop("aggregate_sha256")
        if canonical_payload_sha256(core) != self.aggregate_sha256:
            raise ValueError("validation report aggregate_sha256 mismatch")


def _validate_count_mapping(
    values: Mapping[str, int],
    *,
    allowed: frozenset[str],
) -> None:
    if not set(values).issubset(allowed):
        raise ValueError(f"count mapping contains unknown keys: {sorted(set(values) - allowed)}")
    if any(isinstance(value, bool) or value < 0 for value in values.values()):
        raise ValueError("count mapping values must be non-negative integers")


__all__ = [
    "BenchmarkManifestV4",
    "BenchmarkStatisticsV4",
    "BenchmarkValidationReportV4",
    "CanonicalParentV4",
    "ChannelDescriptorV1",
    "DuplicateComponentV1",
    "GroupIdentityV1",
    "Hdf5ArrayRefV1",
    "Hdf5SourceRecordV1",
    "NormalizationChannelStatisticV1",
    "NormalizationManifestV1",
    "RegisteredRGBViewV1",
    "ScientificProvenanceV1",
    "ValidityDescriptorV1",
    "canonical_payload_sha256",
    "seal_canonical_parent",
    "seal_source_record",
    "validate_benchmark_logical_path",
    "validate_dataset_logical_path",
    "validate_portable_path",
    "validate_sha256",
]
