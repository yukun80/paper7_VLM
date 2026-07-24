"""阶段 1B 五个只读 HDF5 数据源的显式读取合同。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from benchmark_common import SOURCE_ORDER, SPLIT_ORDER, read_json, read_jsonl, stable_rank


@dataclass(frozen=True)
class SourceContract:
    source: str
    root_name: str
    optical_indices: tuple[int, ...]
    auxiliaries: Mapping[str, tuple[int, ...]]
    group_status: str


@dataclass
class SourceSample:
    sample_id: str
    source: str
    source_sample_key: str
    split: str
    split_source: str
    image_path: Path
    mask_path: Path
    optical_indices: tuple[int, ...]
    optical_channel_names: tuple[str, ...]
    auxiliary_indices: dict[str, tuple[int, ...]]
    auxiliary_channel_names: dict[str, tuple[str, ...]]
    positive: bool
    source_group_id: str | None
    group_status: str
    source_schema_version: str
    provenance: dict[str, Any]
    source_record_sha256: str | None


@dataclass
class LoadedSourceSample:
    image: np.ndarray
    pixel_valid: np.ndarray
    channel_valid: np.ndarray
    mask: np.ndarray
    label_valid: np.ndarray | None


CONTRACTS = {
    "gdcld": SourceContract(
        "gdcld", "GDCLD", (0, 1, 2), {}, "known"
    ),
    "lmhld": SourceContract(
        "lmhld", "LMHLD", (0, 1, 2, 3), {}, "unknown"
    ),
    "landslidebench_agent": SourceContract(
        "landslidebench_agent",
        "LandslideBench_agent",
        (0, 1, 2),
        {},
        "known",
    ),
    "landslide4sense": SourceContract(
        "landslide4sense",
        "landslide4sense",
        tuple(range(12)),
        {"slope": (12,), "dem": (13,)},
        "unknown",
    ),
    "multimodal_landslide": SourceContract(
        "multimodal_landslide",
        "multimodal-landslide-dataset",
        (0, 1, 2),
        {"dem": (3,), "insar_velocity": (4,)},
        "sample_only",
    ),
}


def _channel_contract(
    datasets_root: Path, contract: SourceContract
) -> tuple[str, list[str]]:
    schema_path = datasets_root / contract.root_name / "hdf5/channel_schema.json"
    schema = read_json(schema_path)
    channels = schema.get("channels")
    if not isinstance(channels, list):
        raise ValueError(f"{schema_path}: 缺少 channels")
    ordered = sorted(channels, key=lambda row: int(row["index"]))
    expected = list(range(len(ordered)))
    actual = [int(row["index"]) for row in ordered]
    if actual != expected:
        raise ValueError(f"{schema_path}: channel index 必须连续，实际 {actual}")
    return str(schema["schema_version"]), [str(row["name"]) for row in ordered]


def _provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    suffixes = (
        "_sha256",
        "_path",
        "_image",
        "_mask",
        "_index",
        "_window",
    )
    exact = {
        "source_image",
        "source_mask",
        "source_key",
        "subset",
        "region",
        "scene",
        "geotransform",
        "crs_wkt",
    }
    return {
        key: value
        for key, value in row.items()
        if key in exact or key.endswith(suffixes)
    }


def _resolve(root: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _base_sample(
    *,
    row: Mapping[str, Any],
    contract: SourceContract,
    root: Path,
    schema_version: str,
    channel_names: Sequence[str],
    split: str,
    split_source: str,
    group_id: str | None,
    group_status: str | None = None,
) -> SourceSample:
    sample_key = str(row["sample_key"])
    optical_names = tuple(channel_names[index] for index in contract.optical_indices)
    auxiliary_names = {
        name: tuple(channel_names[index] for index in indices)
        for name, indices in contract.auxiliaries.items()
    }
    positive_count = row.get(
        "positive_pixel_count", row.get("mask_positive_pixel_count", 0)
    )
    positive = bool(row.get("has_landslide", int(positive_count) > 0))
    return SourceSample(
        sample_id=f"{contract.source}::{sample_key}",
        source=contract.source,
        source_sample_key=sample_key,
        split=split,
        split_source=split_source,
        image_path=_resolve(root, str(row["image_hdf5"])),
        mask_path=_resolve(root, str(row["mask_hdf5"])),
        optical_indices=contract.optical_indices,
        optical_channel_names=optical_names,
        auxiliary_indices=dict(contract.auxiliaries),
        auxiliary_channel_names=auxiliary_names,
        positive=positive,
        source_group_id=group_id,
        group_status=group_status or contract.group_status,
        source_schema_version=schema_version,
        provenance=_provenance(row),
        source_record_sha256=(
            str(row["record_sha256"]) if row.get("record_sha256") else None
        ),
    )


def _discover_indexed_source(
    datasets_root: Path, contract: SourceContract
) -> list[SourceSample]:
    root = datasets_root / contract.root_name
    schema_version, channel_names = _channel_contract(datasets_root, contract)
    paths = sorted((root / "jsonl").glob("sample_index_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"{root}/jsonl: 缺少 sample_index_*.jsonl")
    samples: list[SourceSample] = []
    for path in paths:
        for row in read_jsonl(path):
            split = str(row.get("split", ""))
            if split not in SPLIT_ORDER:
                raise ValueError(f"{path}: 非法 split={split!r}")
            if contract.source == "gdcld":
                if row.get("source_origin") == "original":
                    group_id = f"gdcld_original::{row['sample_key']}"
                else:
                    group_id = (
                        f"gdcld::{row.get('source_origin','')}::"
                        f"{row.get('region','')}::{row.get('scene','')}"
                    )
            elif contract.source == "landslidebench_agent":
                group_id = str(row["location_key"])
            elif contract.source == "multimodal_landslide":
                group_id = str(row.get("source_key", row["sample_key"]))
            else:
                group_id = None
            samples.append(
                _base_sample(
                    row=row,
                    contract=contract,
                    root=root,
                    schema_version=schema_version,
                    channel_names=channel_names,
                    split=split,
                    split_source="source_existing",
                    group_id=group_id,
                )
            )
    return samples


def _largest_remainder_counts(total: int) -> dict[str, int]:
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    raw = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())
    priority = sorted(
        SPLIT_ORDER,
        key=lambda split: (-(raw[split] - counts[split]), SPLIT_ORDER.index(split)),
    )
    for split in priority[:remaining]:
        counts[split] += 1
    return counts


def _discover_landslide4sense(
    datasets_root: Path, split_seed: int
) -> list[SourceSample]:
    contract = CONTRACTS["landslide4sense"]
    dataset_root = datasets_root / contract.root_name
    root = dataset_root / "hdf5"
    schema_version, channel_names = _channel_contract(datasets_root, contract)
    manifest_path = root / "conversion_manifest.jsonl"
    rows = read_jsonl(manifest_path)
    split_by_key: dict[str, str] = {}
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subset.setdefault(str(row["subset"]), []).append(row)
    for subset, subset_rows in sorted(by_subset.items()):
        ordered = sorted(
            subset_rows,
            key=lambda row: (
                stable_rank(split_seed, contract.source, subset, row["sample_key"]),
                str(row["sample_key"]),
            ),
        )
        counts = _largest_remainder_counts(len(ordered))
        cursor = 0
        for split in SPLIT_ORDER:
            for row in ordered[cursor : cursor + counts[split]]:
                split_by_key[str(row["sample_key"])] = split
            cursor += counts[split]
    samples: list[SourceSample] = []
    for row in rows:
        samples.append(
            _base_sample(
                row=row,
                contract=contract,
                root=root,
                schema_version=schema_version,
                channel_names=channel_names,
                split=split_by_key[str(row["sample_key"])],
                split_source="benchmark_deterministic_v1",
                group_id=None,
                group_status="unknown",
            )
        )
    return samples


def discover_all_sources(
    datasets_root: Path, split_seed: int
) -> tuple[list[SourceSample], dict[str, Any]]:
    datasets_root = datasets_root.resolve()
    samples: list[SourceSample] = []
    source_counts: dict[str, int] = {}
    for source in SOURCE_ORDER:
        if source == "landslide4sense":
            discovered = _discover_landslide4sense(datasets_root, split_seed)
        else:
            discovered = _discover_indexed_source(datasets_root, CONTRACTS[source])
        source_counts[source] = len(discovered)
        samples.extend(discovered)
    missing = [
        str(path)
        for sample in samples
        for path in (sample.image_path, sample.mask_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"发现 {len(missing)} 个缺失 HDF5，首个为 {missing[0]}"
        )
    duplicate_ids = [
        sample_id
        for sample_id, count in Counter(
            sample.sample_id for sample in samples
        ).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"sample_id 重复，首个为 {duplicate_ids[0]}")
    leakage_groups: dict[str, set[str]] = {}
    for sample in samples:
        if sample.source != "landslidebench_agent" or sample.source_group_id is None:
            continue
        leakage_groups.setdefault(sample.source_group_id, set()).add(sample.split)
    approved = sorted(
        group for group, splits in leakage_groups.items() if len(splits) > 1
    )
    metadata = {
        "source_counts": source_counts,
        "approved_group_split_exceptions": {
            "landslidebench_agent": approved
        },
    }
    return samples, metadata


def load_source_sample(sample: SourceSample) -> LoadedSourceSample:
    with h5py.File(sample.image_path, "r") as handle:
        if "image" not in handle or "channel_valid" not in handle:
            raise ValueError(f"{sample.image_path}: 必须包含 /image 和 /channel_valid")
        image = np.asarray(handle["image"][...], dtype=np.float32)
        channel_valid = np.asarray(handle["channel_valid"][...], dtype=np.uint8)
        if "pixel_valid" in handle:
            pixel_valid = np.asarray(handle["pixel_valid"][...], dtype=np.uint8)
        else:
            pixel_valid = np.ones(image.shape, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"{sample.image_path}: /image 必须是 CHW")
    if channel_valid.shape != (image.shape[0],):
        raise ValueError(f"{sample.image_path}: /channel_valid shape 错误")
    if pixel_valid.shape != image.shape:
        raise ValueError(f"{sample.image_path}: /pixel_valid shape 错误")
    with h5py.File(sample.mask_path, "r") as handle:
        if "mask" not in handle:
            raise ValueError(f"{sample.mask_path}: 缺少 /mask")
        mask = np.asarray(handle["mask"][...])
        label_valid = (
            np.asarray(handle["valid_mask"][...], dtype=np.uint8)
            if "valid_mask" in handle
            else None
        )
    expected_channels = max(
        (*sample.optical_indices, *(i for v in sample.auxiliary_indices.values() for i in v))
    ) + 1
    if image.shape[0] != expected_channels:
        raise ValueError(
            f"{sample.image_path}: 通道数 {image.shape[0]}，预期 {expected_channels}"
        )
    if mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"{sample.sample_id}: image HW={image.shape[-2:]} 与 mask HW={mask.shape[-2:]} 不一致"
        )
    finite = np.isfinite(image)
    pixel_valid = (
        pixel_valid.astype(bool)
        & finite
        & channel_valid[:, None, None].astype(bool)
    ).astype(np.uint8)
    if sample.source == "gdcld" and label_valid is not None:
        pixel_valid &= label_valid[None, ...].astype(np.uint8)
    return LoadedSourceSample(
        image=image,
        pixel_valid=pixel_valid,
        channel_valid=channel_valid,
        mask=mask,
        label_valid=label_valid,
    )
