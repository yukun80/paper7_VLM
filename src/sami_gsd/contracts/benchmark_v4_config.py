"""Strict configuration loader for P1 HDF5 Benchmark v4.

The YAML stores only execution choices and exact source-index bindings.  The
audited channel descriptors are reopened from the bound inventory so a copied
configuration cannot silently drift from the P0R evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from sami_gsd.contracts.benchmark_v4 import (
    ChannelDescriptorV1,
    ScientificProvenanceV1,
    ValidityDescriptorV1,
    validate_dataset_logical_path,
    validate_portable_path,
    validate_sha256,
)


_EXPECTED_SOURCE_KEYS = frozenset(
    {
        "gdcld",
        "landslide4sense",
        "landslidebench_agent",
        "lmhld",
        "multimodal_landslide",
        "sen12_landslides",
    }
)
_SPLITS = frozenset({"train", "val", "test"})
_ASSURANCE_ELIGIBILITY = {
    "source_declared_unverified": "exploratory",
    "train_only": "train_only",
}
_EXPECTED_INDEX_BINDINGS = {
    "gdcld": (
        (
            "datasets/GDCLD/jsonl/sample_index_train.jsonl",
            "train",
            "train",
        ),
        (
            "datasets/GDCLD/jsonl/sample_index_val.jsonl",
            "val",
            "val",
        ),
        (
            "datasets/GDCLD/jsonl/sample_index_test.jsonl",
            "test",
            "test",
        ),
    ),
    "landslide4sense": (
        (
            "datasets/landslide4sense/hdf5/conversion_manifest.jsonl",
            None,
            "train",
        ),
    ),
    "landslidebench_agent": (
        (
            "datasets/LandslideBench_agent/jsonl/sample_index_train.jsonl",
            "train",
            "train",
        ),
        (
            "datasets/LandslideBench_agent/jsonl/sample_index_val.jsonl",
            "val",
            "val",
        ),
        (
            "datasets/LandslideBench_agent/jsonl/sample_index_test.jsonl",
            "test",
            "test",
        ),
    ),
    "lmhld": (
        (
            "datasets/LMHLD/jsonl/sample_index_train.jsonl",
            "train",
            "train",
        ),
        (
            "datasets/LMHLD/jsonl/sample_index_val.jsonl",
            "val",
            "val",
        ),
        (
            "datasets/LMHLD/jsonl/sample_index_test.jsonl",
            "test",
            "test",
        ),
    ),
    "multimodal_landslide": (
        (
            "datasets/multimodal-landslide-dataset/jsonl/sample_index_train.jsonl",
            "train",
            "train",
        ),
        (
            "datasets/multimodal-landslide-dataset/jsonl/sample_index_val.jsonl",
            "val",
            "val",
        ),
    ),
    "sen12_landslides": (),
}


def _strict_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing or extra:
        raise ValueError(
            f"{location} keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _repo_path(repository_root: Path, logical_path: str) -> Path:
    validate_portable_path(logical_path, field_name="repository binding")
    resolved = (repository_root / logical_path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"repository binding escapes root: {logical_path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


@dataclass(frozen=True, slots=True)
class SourceIndexV1:
    """One authoritative source index and its non-random split projection."""

    logical_path: str
    source_declared_split: Literal["train", "val", "test"] | None
    canonical_split: Literal["train", "val", "test"]

    def __post_init__(self) -> None:
        validate_dataset_logical_path(self.logical_path)
        if self.canonical_split not in _SPLITS:
            raise ValueError("canonical_split is invalid")
        if self.source_declared_split is not None:
            if self.source_declared_split not in _SPLITS:
                raise ValueError("source_declared_split is invalid")
            if self.source_declared_split != self.canonical_split:
                raise ValueError("native source split cannot be rewritten")

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        if mode != "json":
            raise ValueError("only JSON projection is supported")
        return {
            "logical_path": self.logical_path,
            "source_declared_split": self.source_declared_split,
            "canonical_split": self.canonical_split,
        }


@dataclass(frozen=True, slots=True)
class Hdf5SourceConfigV1:
    """Runtime-free source reader configuration embedded in the v4 registry."""

    source_key: str
    ingestion_status: Literal["ready", "not_ready"]
    source_root: str
    hdf5_base: str
    indexes: tuple[SourceIndexV1, ...]
    sample_id_field: str
    image_path_field: str
    mask_path_field: str
    row_split_field: str | None
    split_assurance: Literal["source_declared_unverified", "train_only"] | None
    evaluation_eligibility: Literal["exploratory", "train_only"] | None
    group_field: str | None
    group_kind: Literal["scene", "event", "location", "source_sample", "unknown"]
    group_completeness: Literal["partial", "unavailable"]
    group_evidence: tuple[str, ...]
    duplicate_component_field: str | None
    duplicate_evidence_level: Literal["candidate", "unavailable"]
    channel_schema: str | None
    channel_schema_sha256: str | None
    channels: tuple[ChannelDescriptorV1, ...]
    validity: ValidityDescriptorV1 | None
    registered_rgb: dict[str, Any] | None
    provenance: ScientificProvenanceV1
    risks: tuple[str, ...]
    known_location_cross_split_conflict_count: int | None
    expected_pair_count: int
    expected_positive_count: int
    expected_no_target_count: int
    expected_source_split_counts: dict[str, int]

    def __post_init__(self) -> None:
        validate_dataset_logical_path(self.source_root)
        validate_dataset_logical_path(self.hdf5_base)
        expected_counts = (
            self.expected_pair_count,
            self.expected_positive_count,
            self.expected_no_target_count,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in expected_counts
        ):
            raise ValueError("inventory population counts must be non-negative integers")
        if (
            self.expected_positive_count + self.expected_no_target_count
            != self.expected_pair_count
        ):
            raise ValueError("inventory positive/no-target counts must partition pairs")
        if not set(self.expected_source_split_counts).issubset(_SPLITS):
            raise ValueError("inventory split counts contain an unknown split")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.expected_source_split_counts.values()
        ):
            raise ValueError("inventory split counts must be non-negative integers")
        if sum(self.expected_source_split_counts.values()) != self.expected_pair_count:
            raise ValueError("inventory split counts must partition pairs")
        conflict_count = self.known_location_cross_split_conflict_count
        if conflict_count is not None and (
            isinstance(conflict_count, bool)
            or not isinstance(conflict_count, int)
            or conflict_count < 0
        ):
            raise ValueError(
                "known_location_cross_split_conflict_count must be "
                "a non-negative integer or null"
            )
        if self.ingestion_status == "ready":
            if self.expected_pair_count <= 0:
                raise ValueError("ready source must have a positive expected population")
            if (
                self.split_assurance not in _ASSURANCE_ELIGIBILITY
                or self.evaluation_eligibility
                != _ASSURANCE_ELIGIBILITY[self.split_assurance]
            ):
                raise ValueError("split assurance and evaluation eligibility disagree")
            if not self.indexes or not self.channels or self.validity is None:
                raise ValueError("ready sources require indexes, channels, and validity")
            if self.split_assurance == "train_only" and any(
                item.canonical_split != "train" for item in self.indexes
            ):
                raise ValueError("train-only source cannot produce val/test rows")
            if self.channel_schema is None or self.channel_schema_sha256 is None:
                raise ValueError("ready source requires channel-schema binding")
        else:
            if self.expected_pair_count != 0:
                raise ValueError("not-ready source must have zero expected population")
            if (
                self.split_assurance is not None
                or self.evaluation_eligibility is not None
            ):
                raise ValueError(
                    "not-ready source must not declare split assurance or "
                    "evaluation eligibility"
                )
            if self.indexes or self.channels or self.validity is not None:
                raise ValueError("not-ready source cannot declare consumable records")
            if self.registered_rgb is not None:
                raise ValueError("not-ready source cannot register a model view")
        if self.channel_schema is not None:
            validate_dataset_logical_path(self.channel_schema)
        if self.channel_schema_sha256 is not None:
            validate_sha256(
                self.channel_schema_sha256,
                field_name="channel_schema_sha256",
            )

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        if mode != "json":
            raise ValueError("only JSON projection is supported")
        return {
            "source_key": self.source_key,
            "ingestion_status": self.ingestion_status,
            "source_root": self.source_root,
            "hdf5_base": self.hdf5_base,
            "indexes": [item.model_dump(mode="json") for item in self.indexes],
            "sample_id_field": self.sample_id_field,
            "image_path_field": self.image_path_field,
            "mask_path_field": self.mask_path_field,
            "row_split_field": self.row_split_field,
            "split_assurance": self.split_assurance,
            "evaluation_eligibility": self.evaluation_eligibility,
            "group_field": self.group_field,
            "group_kind": self.group_kind,
            "group_completeness": self.group_completeness,
            "group_evidence": list(self.group_evidence),
            "duplicate_component_field": self.duplicate_component_field,
            "duplicate_evidence_level": self.duplicate_evidence_level,
            "channel_schema": self.channel_schema,
            "channel_schema_sha256": self.channel_schema_sha256,
            "channels": [
                channel.model_dump(mode="json") for channel in self.channels
            ],
            "validity": (
                None if self.validity is None else self.validity.model_dump(mode="json")
            ),
            "registered_rgb": self.registered_rgb,
            "provenance": self.provenance.model_dump(mode="json"),
            "risks": list(self.risks),
            "known_location_cross_split_conflict_count": (
                self.known_location_cross_split_conflict_count
            ),
            "expected_pair_count": self.expected_pair_count,
            "expected_positive_count": self.expected_positive_count,
            "expected_no_target_count": self.expected_no_target_count,
            "expected_source_split_counts": dict(
                sorted(self.expected_source_split_counts.items())
            ),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkV4Config:
    """Fully expanded, finite P1 build configuration."""

    benchmark_name: str
    mode: Literal["small"]
    benchmark_relative_path: str
    source_contract_path: str
    source_inventory_path: str
    sources: tuple[Hdf5SourceConfigV1, ...]
    schema_version: Literal["sami_benchmark_v4_config_v1"] = (
        "sami_benchmark_v4_config_v1"
    )
    protocol: Literal["sami_hdf5_materialized_copy_v1"] = (
        "sami_hdf5_materialized_copy_v1"
    )

    def __post_init__(self) -> None:
        if self.benchmark_relative_path != "sami_landslide_hdf5_v4/small":
            raise ValueError("Benchmark v4 Small output path is frozen")
        validate_portable_path(
            self.benchmark_relative_path,
            field_name="benchmark_relative_path",
        )
        validate_portable_path(
            self.source_contract_path,
            field_name="source_contract_path",
        )
        validate_portable_path(
            self.source_inventory_path,
            field_name="source_inventory_path",
        )
        keys = tuple(source.source_key for source in self.sources)
        if keys != tuple(sorted(keys)) or set(keys) != _EXPECTED_SOURCE_KEYS:
            raise ValueError("source configs must contain the six frozen keys in sorted order")
        ready = [source for source in self.sources if source.ingestion_status == "ready"]
        if len(ready) != 5:
            raise ValueError("P1 requires exactly five ready sources")
        strict = [
            source
            for source in ready
            if source.evaluation_eligibility == "strict"
        ]
        if strict:
            raise ValueError("P1 baseline has no strict cohort")

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        if mode != "json":
            raise ValueError("only JSON projection is supported")
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "benchmark_name": self.benchmark_name,
            "mode": self.mode,
            "benchmark_relative_path": self.benchmark_relative_path,
            "source_contract_path": self.source_contract_path,
            "source_inventory_path": self.source_inventory_path,
            "sources": [source.model_dump(mode="json") for source in self.sources],
        }

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _channels_from_inventory(
    source_inventory: Mapping[str, Any],
    *,
    validity_source: str,
) -> tuple[ChannelDescriptorV1, ...]:
    raw_channels = source_inventory.get("canonical_channels")
    display_names = (source_inventory.get("image_dataset") or {}).get("channels")
    if not isinstance(raw_channels, list) or not isinstance(display_names, list):
        raise ValueError("ready inventory source lacks channel/display-name evidence")
    if len(raw_channels) != len(display_names):
        raise ValueError("inventory channel/display-name lengths disagree")
    channels: list[ChannelDescriptorV1] = []
    for expected_index, (raw, display_name) in enumerate(
        zip(raw_channels, display_names, strict=True)
    ):
        if not isinstance(raw, Mapping) or raw.get("index") != expected_index:
            raise ValueError("inventory channel indices must be contiguous and ordered")
        payload = {
            "schema_version": "sami_channel_descriptor_v1",
            "index": raw["index"],
            "channel_key": raw["channel_key"],
            "display_name": str(display_name),
            "modality_family": raw["modality_family"],
            "physical_unit": raw["physical_unit"],
            "wavelength_nm": raw["wavelength_nm"],
            "wavelength_known": raw["wavelength_known"],
            "gsd_m": raw["gsd_m"],
            "gsd_known": raw["gsd_known"],
            "normalization": raw["normalization"],
            "validity_source": validity_source,
        }
        channels.append(ChannelDescriptorV1.from_mapping(payload))
    return tuple(channels)


def _parse_source(
    raw: Mapping[str, Any],
    *,
    inventory_by_key: Mapping[str, Mapping[str, Any]],
) -> Hdf5SourceConfigV1:
    expected = {
        "source_key",
        "ingestion_status",
        "source_root",
        "hdf5_base",
        "indexes",
        "sample_id_field",
        "image_path_field",
        "mask_path_field",
        "row_split_field",
        "split_assurance",
        "evaluation_eligibility",
        "group_field",
        "group_kind",
        "group_completeness",
        "group_evidence",
        "duplicate_component_field",
        "duplicate_evidence_level",
        "validity",
        "provenance",
        "risks",
        "known_location_cross_split_conflict_count",
    }
    _strict_keys(raw, expected, location=f"source[{raw.get('source_key', '?')}]")
    source_key = str(raw["source_key"])
    if source_key not in inventory_by_key:
        raise ValueError(f"source missing from bound inventory: {source_key}")
    inventory = inventory_by_key[source_key]
    if raw["ingestion_status"] != inventory["ingestion_status"]:
        raise ValueError(f"ingestion status drifts from inventory: {source_key}")
    if raw["source_root"] != inventory["source_root"]:
        raise ValueError(f"source root drifts from inventory: {source_key}")
    if raw["split_assurance"] != inventory["split_assurance"]:
        raise ValueError(f"split assurance drifts from inventory: {source_key}")
    if raw["evaluation_eligibility"] != inventory["evaluation_eligibility"]:
        raise ValueError(f"evaluation eligibility drifts from inventory: {source_key}")
    expected_conflict_count = inventory.get(
        "location_cross_split_conflict_count"
    )
    if (
        raw["known_location_cross_split_conflict_count"]
        != expected_conflict_count
    ):
        raise ValueError(
            "known location conflict count drifts from inventory: "
            f"{source_key}"
        )
    inventory_risks = inventory.get("risks")
    if (
        not isinstance(raw["risks"], list)
        or not isinstance(inventory_risks, list)
        or raw["risks"] != inventory_risks
    ):
        raise ValueError(f"risk list drifts from inventory: {source_key}")
    expected_pair_count = inventory.get("pair_count")
    expected_positive_count = inventory.get("positive_count")
    expected_no_target_count = inventory.get("no_target_count")
    raw_split_counts = inventory.get("source_split_counts")
    if raw_split_counts is None:
        expected_source_split_counts = (
            {"train": expected_pair_count}
            if raw["ingestion_status"] == "ready"
            else {}
        )
    elif isinstance(raw_split_counts, Mapping):
        expected_source_split_counts = {
            str(split): count for split, count in raw_split_counts.items()
        }
    else:
        raise ValueError(f"inventory source split counts are invalid: {source_key}")

    raw_indexes = raw["indexes"]
    if not isinstance(raw_indexes, list):
        raise ValueError(f"source indexes must be an array: {source_key}")
    parsed_indexes: list[SourceIndexV1] = []
    for item in raw_indexes:
        if not isinstance(item, Mapping):
            raise ValueError(f"source index must be a mapping: {source_key}")
        _strict_keys(
            item,
            {"logical_path", "source_declared_split", "canonical_split"},
            location=f"source[{source_key}].indexes[]",
        )
        parsed_indexes.append(
            SourceIndexV1(
                logical_path=item["logical_path"],
                source_declared_split=item["source_declared_split"],
                canonical_split=item["canonical_split"],
            )
        )
    indexes = tuple(parsed_indexes)
    observed_bindings = tuple(
        (
            item.logical_path,
            item.source_declared_split,
            item.canonical_split,
        )
        for item in indexes
    )
    if observed_bindings != _EXPECTED_INDEX_BINDINGS[source_key]:
        raise ValueError(
            f"source index selection/order drifts from frozen P1 binding: {source_key}"
        )
    inventory_indexes = set(inventory.get("authoritative_indexes") or ())
    if any(index.logical_path not in inventory_indexes for index in indexes):
        raise ValueError(f"config uses a non-authoritative index: {source_key}")

    raw_validity = raw["validity"]
    if raw["ingestion_status"] == "ready":
        if not isinstance(raw_validity, Mapping):
            raise ValueError("ready source validity must be a mapping")
        validity = ValidityDescriptorV1.from_mapping(raw_validity)
        validity_source = (
            "pixel_valid"
            if validity.pixel_valid_key is not None
            else "channel_valid"
            if validity.channel_valid_key is not None
            else "valid_mask"
            if validity.valid_mask_key is not None
            else "implicit_present"
        )
        channels = _channels_from_inventory(
            inventory,
            validity_source=validity_source,
        )
        registered_raw = inventory.get("registered_rgb")
        if not isinstance(registered_raw, Mapping):
            raise ValueError(f"ready source lacks registered RGB evidence: {source_key}")
        registered_rgb = {
            "source_indices": list(registered_raw["source_indices"]),
            "channel_keys": list(registered_raw["channel_keys"]),
            "mapping_evidence": registered_raw["mapping_evidence"],
        }
        channel_schema = inventory["channel_schema"]
        channel_schema_sha256 = inventory["channel_schema_sha256"]
    else:
        if raw_validity is not None:
            raise ValueError("not-ready source validity must be null")
        validity = None
        channels = ()
        registered_rgb = None
        channel_schema = None
        channel_schema_sha256 = None

    provenance = ScientificProvenanceV1.from_mapping(raw["provenance"])
    return Hdf5SourceConfigV1(
        source_key=source_key,
        ingestion_status=raw["ingestion_status"],
        source_root=raw["source_root"],
        hdf5_base=raw["hdf5_base"],
        indexes=indexes,
        sample_id_field=raw["sample_id_field"],
        image_path_field=raw["image_path_field"],
        mask_path_field=raw["mask_path_field"],
        row_split_field=raw["row_split_field"],
        split_assurance=raw["split_assurance"],
        evaluation_eligibility=raw["evaluation_eligibility"],
        group_field=raw["group_field"],
        group_kind=raw["group_kind"],
        group_completeness=raw["group_completeness"],
        group_evidence=tuple(raw["group_evidence"]),
        duplicate_component_field=raw["duplicate_component_field"],
        duplicate_evidence_level=raw["duplicate_evidence_level"],
        channel_schema=channel_schema,
        channel_schema_sha256=channel_schema_sha256,
        channels=channels,
        validity=validity,
        registered_rgb=registered_rgb,
        provenance=provenance,
        risks=tuple(raw["risks"]),
        known_location_cross_split_conflict_count=raw[
            "known_location_cross_split_conflict_count"
        ],
        expected_pair_count=expected_pair_count,
        expected_positive_count=expected_positive_count,
        expected_no_target_count=expected_no_target_count,
        expected_source_split_counts=expected_source_split_counts,
    )


def load_benchmark_v4_config(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> BenchmarkV4Config:
    """Load YAML, reopen its inventory binding, and reject every unknown field."""

    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark v4 config root must be a mapping")
    expected = {
        "schema_version",
        "protocol",
        "benchmark_name",
        "mode",
        "benchmark_relative_path",
        "source_contract_path",
        "source_inventory_path",
        "sources",
    }
    _strict_keys(payload, expected, location="BenchmarkV4Config")
    if payload["schema_version"] != "sami_benchmark_v4_config_v1":
        raise ValueError("unsupported Benchmark v4 config schema")
    if payload["protocol"] != "sami_hdf5_materialized_copy_v1":
        raise ValueError("unsupported Benchmark v4 protocol")
    if repository_root is None:
        repository_root = path.parent.parent
    repository_root = repository_root.resolve()
    _repo_path(repository_root, str(payload["source_contract_path"]))
    inventory_path = _repo_path(
        repository_root,
        str(payload["source_inventory_path"]),
    )
    inventory_payload = _strict_json(inventory_path)
    inventory_sources = inventory_payload.get("sources")
    if not isinstance(inventory_sources, list):
        raise ValueError("source inventory lacks sources array")
    inventory_by_key = {
        str(source["source_key"]): source
        for source in inventory_sources
        if isinstance(source, Mapping)
    }
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError("config sources must be an array")
    sources = tuple(
        _parse_source(source, inventory_by_key=inventory_by_key)
        for source in raw_sources
    )
    return BenchmarkV4Config(
        benchmark_name=payload["benchmark_name"],
        mode=payload["mode"],
        benchmark_relative_path=payload["benchmark_relative_path"],
        source_contract_path=payload["source_contract_path"],
        source_inventory_path=payload["source_inventory_path"],
        sources=sources,
        schema_version=payload["schema_version"],
        protocol=payload["protocol"],
    )


__all__ = [
    "BenchmarkV4Config",
    "Hdf5SourceConfigV1",
    "SourceIndexV1",
    "load_benchmark_v4_config",
]
