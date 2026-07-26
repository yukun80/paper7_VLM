"""阶段 1B 六个只读 HDF5 数据源的显式读取合同。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from benchmark_common import (
    SOURCE_ORDER,
    SPLIT_ORDER,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_bytes,
    stable_rank,
)


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
    "sen12landslides": SourceContract(
        "sen12landslides",
        "Sen12Landslides",
        tuple(range(10)),
        {
            "sar_ascending": (10, 11),
            "sar_descending": (12, 13),
            "dem": (14,),
        },
        "known",
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
        "pre_hdf5",
        "post_hdf5",
        "mask_hdf5",
        "available_modalities",
        "placeholder_channels_post",
        "source_channel_names",
        "status",
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
    auxiliary_indices: Mapping[str, tuple[int, ...]] | None = None,
    auxiliary_channel_names: Mapping[str, tuple[str, ...]] | None = None,
    positive: bool | None = None,
) -> SourceSample:
    sample_key = str(row["sample_key"])
    selected_auxiliaries = dict(
        contract.auxiliaries if auxiliary_indices is None else auxiliary_indices
    )
    optical_names = tuple(channel_names[index] for index in contract.optical_indices)
    auxiliary_names = (
        {
            name: tuple(channel_names[index] for index in indices)
            for name, indices in selected_auxiliaries.items()
        }
        if auxiliary_channel_names is None
        else dict(auxiliary_channel_names)
    )
    if set(auxiliary_names) != set(selected_auxiliaries):
        raise ValueError(f"{sample_key}: 辅助模态通道名与索引合同不一致")
    positive_count = row.get(
        "positive_pixel_count", row.get("mask_positive_pixel_count", 0)
    )
    is_positive = (
        bool(row.get("has_landslide", int(positive_count) > 0))
        if positive is None
        else positive
    )
    source_record_sha256 = row.get("record_sha256")
    if not source_record_sha256:
        source_record_sha256 = sha256_bytes(
            canonical_json(dict(row)).encode("utf-8")
        )
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
        auxiliary_indices=selected_auxiliaries,
        auxiliary_channel_names=auxiliary_names,
        positive=is_positive,
        source_group_id=group_id,
        group_status=group_status or contract.group_status,
        source_schema_version=schema_version,
        provenance=_provenance(row),
        source_record_sha256=str(source_record_sha256),
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


def _sen12_region(sample_key: str) -> str:
    region, separator, identifier = sample_key.rpartition("_")
    if not separator or not region or not identifier.isdigit():
        raise ValueError(f"Sen12 sample_key 无法解析 region：{sample_key!r}")
    return region


def _balanced_group_splits(
    group_sizes: Mapping[str, int], split_seed: int, source: str
) -> dict[str, str]:
    total = sum(group_sizes.values())
    targets = _largest_remainder_counts(total)
    assigned = {split: 0 for split in SPLIT_ORDER}
    group_to_split: dict[str, str] = {}
    ordered_groups = sorted(
        group_sizes,
        key=lambda group: (
            -group_sizes[group],
            stable_rank(split_seed, source, group),
            group,
        ),
    )
    for group in ordered_groups:
        size = group_sizes[group]

        def score(split: str) -> tuple[float, int]:
            projected = dict(assigned)
            projected[split] += size
            squared_error = sum(
                ((projected[name] - targets[name]) ** 2) / max(targets[name], 1)
                for name in SPLIT_ORDER
            )
            return squared_error, SPLIT_ORDER.index(split)

        selected = min(SPLIT_ORDER, key=score)
        group_to_split[group] = selected
        assigned[selected] += size
    if total >= len(SPLIT_ORDER) and any(assigned[split] == 0 for split in SPLIT_ORDER):
        raise ValueError(f"{source}: group-aware split 未覆盖全部 split：{assigned}")
    return group_to_split


def _discover_sen12landslides(
    datasets_root: Path, split_seed: int
) -> list[SourceSample]:
    contract = CONTRACTS["sen12landslides"]
    dataset_root = datasets_root / contract.root_name
    root = dataset_root / "hdf5"
    schema_version, channel_names = _channel_contract(datasets_root, contract)
    manifest_path = root / "conversion_manifest.jsonl"
    rows = read_jsonl(manifest_path)
    converted = [row for row in rows if row.get("status") == "converted"]
    if len(converted) != len(rows):
        raise ValueError(
            f"{manifest_path}: manifest 应只包含 converted，实际 "
            f"{len(converted)}/{len(rows)}"
        )
    group_sizes = Counter(_sen12_region(str(row["sample_key"])) for row in rows)
    group_splits = _balanced_group_splits(
        group_sizes, split_seed, contract.source
    )
    samples: list[SourceSample] = []
    for row in rows:
        sample_key = str(row["sample_key"])
        region = _sen12_region(sample_key)
        placeholders = {
            int(index) for index in row.get("placeholder_channels_post", [])
        }
        if placeholders & set(contract.optical_indices):
            raise ValueError(f"{sample_key}: post 光学通道不得缺失")
        selected_auxiliaries: dict[str, tuple[int, ...]] = {}
        for name, indices in contract.auxiliaries.items():
            missing = placeholders & set(indices)
            if missing and missing != set(indices):
                raise ValueError(
                    f"{sample_key}: {name} 只缺部分通道，无法形成模态合同"
                )
            if not missing:
                selected_auxiliaries[name] = indices
        if "dem" not in selected_auxiliaries:
            raise ValueError(f"{sample_key}: DEM 不得缺失")
        mask_path = _resolve(root, str(row["mask_hdf5"]))
        with h5py.File(mask_path, "r") as handle:
            if "positive_pixel_count" in handle.attrs:
                positive_count = int(handle.attrs["positive_pixel_count"])
            else:
                positive_count = int(
                    np.count_nonzero(np.asarray(handle["mask"][...]))
                )
        source_row = dict(row)
        source_row["image_hdf5"] = row["post_hdf5"]
        source_row["positive_pixel_count"] = positive_count
        source_row["source_channel_names"] = list(channel_names)
        selected_channel_names = {
            name: tuple(channel_names[index] for index in indices)
            for name, indices in selected_auxiliaries.items()
        }
        selected_channel_names["dem"] = ("DEM",)
        samples.append(
            _base_sample(
                row=source_row,
                contract=contract,
                root=root,
                schema_version=schema_version,
                channel_names=channel_names,
                split=group_splits[region],
                split_source="benchmark_deterministic_region_v1",
                group_id=f"sen12landslides::{region}",
                group_status="known",
                auxiliary_indices=selected_auxiliaries,
                auxiliary_channel_names=selected_channel_names,
                positive=positive_count > 0,
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
        elif source == "sen12landslides":
            discovered = _discover_sen12landslides(datasets_root, split_seed)
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
        channel_valid_raw = np.asarray(handle["channel_valid"][...])
        if "pixel_valid" in handle:
            pixel_valid_raw = np.asarray(handle["pixel_valid"][...])
        else:
            pixel_valid_raw = np.ones(image.shape, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"{sample.image_path}: /image 必须是 CHW")
    if channel_valid_raw.shape != (image.shape[0],):
        raise ValueError(f"{sample.image_path}: /channel_valid shape 错误")
    if not np.isin(channel_valid_raw, (0, 1)).all():
        raise ValueError(f"{sample.image_path}: /channel_valid 只能包含 0/1")
    if pixel_valid_raw.shape != image.shape:
        raise ValueError(f"{sample.image_path}: /pixel_valid shape 错误")
    if not np.isin(pixel_valid_raw, (0, 1)).all():
        raise ValueError(f"{sample.image_path}: /pixel_valid 只能包含 0/1")
    channel_valid = channel_valid_raw.astype(np.uint8)
    pixel_valid = pixel_valid_raw.astype(np.uint8)
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
    if sample.source == "sen12landslides":
        expected_valid = set(sample.optical_indices)
        for indices in sample.auxiliary_indices.values():
            expected_valid.update(indices)
        actual_valid = set(np.flatnonzero(channel_valid == 1).tolist())
        if actual_valid != expected_valid:
            raise ValueError(
                f"{sample.sample_id}: manifest 模态与 post /channel_valid 不一致"
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
