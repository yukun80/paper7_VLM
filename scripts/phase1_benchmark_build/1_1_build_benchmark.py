#!/usr/bin/env python3
"""阶段 1B：构建 small/full 统一 Benchmark。

命令：python 1_1_build_benchmark.py --mode small --patch-size 224
输入：只读 ../datasets 下五个 HDF5 数据源及其 JSONL 索引。
输出：../benchmark/oa_auxseg_hdf5_v1/{small,full}。
写入：拒绝覆盖，先写同级临时目录，完成后原子发布。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from benchmark_common import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_SEED,
    SCHEMA_VERSION,
    SOURCE_ORDER,
    SPLIT_ORDER,
    RunningChannelStats,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    iter_chunks,
    resize_binary_mask,
    resize_continuous_with_validity,
    sha256_bytes,
    sha256_file,
    stable_rank,
)
from benchmark_sources import (
    SourceSample,
    discover_all_sources,
    load_source_sample,
)


def _source_sort_key(sample: SourceSample) -> tuple[int, int, str]:
    return (
        SOURCE_ORDER.index(sample.source),
        SPLIT_ORDER.index(sample.split),
        sample.sample_id,
    )


def select_small(
    samples: Sequence[SourceSample], *, per_source: int, seed: int
) -> list[SourceSample]:
    if per_source <= 0:
        raise ValueError("--small-per-source 必须大于 0")
    selected: list[SourceSample] = []
    for source in SOURCE_ORDER:
        source_samples = [sample for sample in samples if sample.source == source]
        buckets: dict[tuple[str, bool], list[SourceSample]] = defaultdict(list)
        for sample in source_samples:
            buckets[(sample.split, sample.positive)].append(sample)
        for key in buckets:
            buckets[key].sort(
                key=lambda sample: (
                    stable_rank(seed, source, key[0], int(key[1]), sample.sample_id),
                    sample.sample_id,
                )
            )
        picked: list[SourceSample] = []
        keys = sorted(
            buckets,
            key=lambda item: (SPLIT_ORDER.index(item[0]), 0 if item[1] else 1),
        )
        positions = {key: 0 for key in keys}
        while len(picked) < min(per_source, len(source_samples)):
            progressed = False
            for key in keys:
                position = positions[key]
                if position >= len(buckets[key]):
                    continue
                picked.append(buckets[key][position])
                positions[key] += 1
                progressed = True
                if len(picked) == min(per_source, len(source_samples)):
                    break
            if not progressed:
                break
        selected.extend(picked)
    return sorted(selected, key=_source_sort_key)


def estimate_logical_bytes(
    samples: Sequence[SourceSample], patch_size: int
) -> dict[str, Any]:
    by_source: dict[str, int] = Counter()
    for sample in samples:
        channels = len(sample.optical_indices) + sum(
            len(indices) for indices in sample.auxiliary_indices.values()
        )
        # float32 values + uint8 pixel validity per channel + uint8 mask.
        by_source[sample.source] += patch_size * patch_size * (channels * 5 + 1)
    total = sum(by_source.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "patch_size": [patch_size, patch_size],
        "sample_count": len(samples),
        "logical_bytes": total,
        "logical_gib": total / (1024**3),
        "by_source": {
            source: {
                "logical_bytes": by_source[source],
                "logical_gib": by_source[source] / (1024**3),
            }
            for source in SOURCE_ORDER
        },
        "note": "未压缩逻辑上界；不含 JSONL/manifest，且不存储 mask validity。",
    }


def _logicalize(value: Any, datasets_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _logicalize(child, datasets_root)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_logicalize(child, datasets_root) for child in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return str(path.resolve().relative_to(datasets_root.resolve()))
            except ValueError:
                return value
    return value


def _create_array(
    handle: h5py.File | h5py.Group,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    chunks: tuple[int, ...],
) -> h5py.Dataset:
    return handle.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )


def _prepare_sample(
    sample: SourceSample, patch_size: int
) -> dict[str, Any]:
    loaded = load_source_sample(sample)
    original_size = [int(loaded.image.shape[1]), int(loaded.image.shape[2])]
    optical, optical_valid, optical_channel_valid = (
        resize_continuous_with_validity(
            loaded.image[list(sample.optical_indices)],
            loaded.pixel_valid[list(sample.optical_indices)],
            loaded.channel_valid[list(sample.optical_indices)],
            patch_size,
        )
    )
    auxiliaries: dict[str, dict[str, np.ndarray]] = {}
    for name, indices in sorted(sample.auxiliary_indices.items()):
        values, valid, channel_valid = resize_continuous_with_validity(
            loaded.image[list(indices)],
            loaded.pixel_valid[list(indices)],
            loaded.channel_valid[list(indices)],
            patch_size,
        )
        auxiliaries[name] = {
            "values": values,
            "pixel_valid": valid,
            "channel_valid": channel_valid,
        }
    mask = resize_binary_mask(loaded.mask, loaded.label_valid, patch_size)
    return {
        "original_size": original_size,
        "optical": optical,
        "optical_pixel_valid": optical_valid,
        "optical_channel_valid": optical_channel_valid,
        "auxiliaries": auxiliaries,
        "mask": mask,
    }


def _update_stats(
    running: dict[str, dict[str, RunningChannelStats]],
    names: dict[str, dict[str, tuple[str, ...]]],
    sample: SourceSample,
    prepared: dict[str, Any],
) -> None:
    if sample.split != "train":
        return
    source_stats = running.setdefault(sample.source, {})
    source_names = names.setdefault(sample.source, {})
    if "optical" not in source_stats:
        source_stats["optical"] = RunningChannelStats.create(
            len(sample.optical_channel_names)
        )
        source_names["optical"] = sample.optical_channel_names
    source_stats["optical"].update(
        prepared["optical"], prepared["optical_pixel_valid"]
    )
    for modality, data in prepared["auxiliaries"].items():
        channel_names = sample.auxiliary_channel_names[modality]
        if modality not in source_stats:
            source_stats[modality] = RunningChannelStats.create(len(channel_names))
            source_names[modality] = channel_names
        source_stats[modality].update(data["values"], data["pixel_valid"])


def _write_shard(
    *,
    staging: Path,
    shard_relative: Path,
    samples: Sequence[SourceSample],
    patch_size: int,
    datasets_root: Path,
    running_stats: dict[str, dict[str, RunningChannelStats]],
    stats_names: dict[str, dict[str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    if not samples:
        return []
    first = samples[0]
    shard_path = staging / shard_relative
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name(f".{shard_path.name}.{uuid.uuid4().hex}.tmp")
    count = len(samples)
    optical_channels = len(first.optical_indices)
    rows: list[dict[str, Any]] = []
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["source"] = first.source
            handle.attrs["split"] = first.split
            handle.attrs["patch_size"] = patch_size
            handle.attrs["optical_channel_names_json"] = json.dumps(
                first.optical_channel_names, ensure_ascii=False
            )
            optical_ds = _create_array(
                handle,
                "optical",
                (count, optical_channels, patch_size, patch_size),
                "float32",
                (1, 1, patch_size, patch_size),
            )
            optical_valid_ds = _create_array(
                handle,
                "optical_pixel_valid",
                (count, optical_channels, patch_size, patch_size),
                "uint8",
                (1, 1, patch_size, patch_size),
            )
            optical_channel_valid_ds = _create_array(
                handle,
                "optical_channel_valid",
                (count, optical_channels),
                "uint8",
                (min(count, 256), optical_channels),
            )
            mask_ds = _create_array(
                handle,
                "mask",
                (count, 1, patch_size, patch_size),
                "uint8",
                (1, 1, patch_size, patch_size),
            )
            auxiliary_datasets: dict[str, dict[str, h5py.Dataset]] = {}
            auxiliary_root = handle.create_group("auxiliary")
            for name, indices in sorted(first.auxiliary_indices.items()):
                group = auxiliary_root.create_group(name)
                channel_count = len(indices)
                group.attrs["channel_names_json"] = json.dumps(
                    first.auxiliary_channel_names[name], ensure_ascii=False
                )
                auxiliary_datasets[name] = {
                    "values": _create_array(
                        group,
                        "values",
                        (count, channel_count, patch_size, patch_size),
                        "float32",
                        (1, 1, patch_size, patch_size),
                    ),
                    "pixel_valid": _create_array(
                        group,
                        "pixel_valid",
                        (count, channel_count, patch_size, patch_size),
                        "uint8",
                        (1, 1, patch_size, patch_size),
                    ),
                    "channel_valid": _create_array(
                        group,
                        "channel_valid",
                        (count, channel_count),
                        "uint8",
                        (min(count, 256), channel_count),
                    ),
                }
            for row_index, sample in enumerate(samples):
                prepared = _prepare_sample(sample, patch_size)
                optical_ds[row_index] = prepared["optical"]
                optical_valid_ds[row_index] = prepared["optical_pixel_valid"]
                optical_channel_valid_ds[row_index] = prepared[
                    "optical_channel_valid"
                ]
                mask_ds[row_index] = prepared["mask"]
                for name, data in prepared["auxiliaries"].items():
                    for key, values in data.items():
                        auxiliary_datasets[name][key][row_index] = values
                _update_stats(
                    running_stats, stats_names, sample, prepared
                )
                foreground_ratio = float(prepared["mask"].mean(dtype=np.float64))
                row: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "source_sample_key": sample.source_sample_key,
                    "split": sample.split,
                    "split_source": sample.split_source,
                    "storage": {
                        "shard": shard_relative.as_posix(),
                        "row": row_index,
                    },
                    "optical": {
                        "channel_names": list(sample.optical_channel_names),
                        "shape": [
                            len(sample.optical_channel_names),
                            patch_size,
                            patch_size,
                        ],
                        "pixel_validity": True,
                        "channel_validity": True,
                    },
                    "auxiliaries": {
                        name: {
                            "present": True,
                            "channel_names": list(
                                sample.auxiliary_channel_names[name]
                            ),
                            "shape": [
                                len(sample.auxiliary_channel_names[name]),
                                patch_size,
                                patch_size,
                            ],
                            "pixel_validity": True,
                            "channel_validity": True,
                        }
                        for name in sorted(sample.auxiliary_indices)
                    },
                    "mask": {
                        "shape": [1, patch_size, patch_size],
                        "values": [0, 1],
                    },
                    "foreground_ratio": foreground_ratio,
                    "resize": {
                        "original_size": prepared["original_size"],
                        "target_size": [patch_size, patch_size],
                        "scale_y": patch_size / prepared["original_size"][0],
                        "scale_x": patch_size / prepared["original_size"][1],
                        "continuous_interpolation": "bilinear",
                        "discrete_interpolation": "nearest",
                    },
                    "source_group_id": sample.source_group_id,
                    "group_status": sample.group_status,
                    "source_schema_version": sample.source_schema_version,
                    "source_record_sha256": sample.source_record_sha256,
                    "provenance": _logicalize(
                        sample.provenance, datasets_root
                    ),
                }
                row["record_sha256"] = sha256_bytes(
                    canonical_json(row).encode("utf-8")
                )
                rows.append(row)
        temporary.replace(shard_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return rows


def _statistics_json(
    running: dict[str, dict[str, RunningChannelStats]],
    names: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "train_split_valid_pixels_after_resize",
        "sources": {
            source: {
                modality: running[source][modality].to_rows(
                    names[source][modality]
                )
                for modality in sorted(running[source])
            }
            for source in SOURCE_ORDER
            if source in running
        },
    }


def build_benchmark(
    *,
    datasets_root: Path,
    output_base: Path,
    mode: str,
    patch_size: int,
    small_per_source: int,
    seed: int,
    split_seed: int,
    shard_target_mib: int,
) -> Path:
    all_samples, discovery = discover_all_sources(datasets_root, split_seed)
    samples = (
        select_small(all_samples, per_source=small_per_source, seed=seed)
        if mode == "small"
        else sorted(all_samples, key=_source_sort_key)
    )
    target = output_base / mode
    if target.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{target}")
    output_base.mkdir(parents=True, exist_ok=True)
    staging = output_base / f".{mode}.tmp.{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"临时输出意外存在：{staging}")
    staging.mkdir(parents=False)
    config = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "datasets_root": "../datasets",
        "output": target.name,
        "patch_size": patch_size,
        "small_per_source": small_per_source,
        "seed": seed,
        "split_seed": split_seed,
        "landslide4sense_split_ratios": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        "shard_target_mib": shard_target_mib,
        "mask_validity_stored": False,
    }
    records: list[dict[str, Any]] = []
    running_stats: dict[str, dict[str, RunningChannelStats]] = {}
    stats_names: dict[str, dict[str, tuple[str, ...]]] = {}
    try:
        atomic_write_json(staging / "build_config.json", config)
        grouped: dict[tuple[str, str], list[SourceSample]] = defaultdict(list)
        for sample in samples:
            grouped[(sample.source, sample.split)].append(sample)
        for source in SOURCE_ORDER:
            for split in SPLIT_ORDER:
                group_samples = sorted(
                    grouped.get((source, split), []), key=lambda item: item.sample_id
                )
                if not group_samples:
                    continue
                first = group_samples[0]
                total_channels = len(first.optical_indices) + sum(
                    len(indices) for indices in first.auxiliary_indices.values()
                )
                logical_per_sample = (
                    patch_size * patch_size * (total_channels * 5 + 1)
                )
                samples_per_shard = max(
                    1,
                    int(shard_target_mib * 1024 * 1024 // logical_per_sample),
                )
                for shard_index, shard_samples in enumerate(
                    iter_chunks(group_samples, samples_per_shard)
                ):
                    relative = (
                        Path("data")
                        / source
                        / split
                        / f"shard-{shard_index:05d}.h5"
                    )
                    records.extend(
                        _write_shard(
                            staging=staging,
                            shard_relative=relative,
                            samples=shard_samples,
                            patch_size=patch_size,
                            datasets_root=datasets_root,
                            running_stats=running_stats,
                            stats_names=stats_names,
                        )
                    )
        records.sort(key=lambda row: (
            SOURCE_ORDER.index(row["source"]),
            SPLIT_ORDER.index(row["split"]),
            row["sample_id"],
        ))
        atomic_write_jsonl(staging / "index.jsonl", records)
        atomic_write_json(
            staging / "source_statistics.json",
            _statistics_json(running_stats, stats_names),
        )
        source_counts = Counter(row["source"] for row in records)
        split_counts = Counter(row["split"] for row in records)
        source_split_counts: dict[str, dict[str, int]] = {}
        for source in SOURCE_ORDER:
            source_split_counts[source] = {
                split: sum(
                    1
                    for row in records
                    if row["source"] == source and row["split"] == split
                )
                for split in SPLIT_ORDER
            }
        data_files = sorted((staging / "data").rglob("*.h5"))
        hashed_files = [
            staging / "build_config.json",
            staging / "index.jsonl",
            staging / "source_statistics.json",
            *data_files,
        ]
        hashes = [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in hashed_files
        ]
        atomic_write_jsonl(staging / "SHA256SUMS.jsonl", hashes)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "sample_count": len(records),
            "source_counts": {
                source: source_counts[source] for source in SOURCE_ORDER
            },
            "split_counts": {
                split: split_counts[split] for split in SPLIT_ORDER
            },
            "source_split_counts": source_split_counts,
            "full_candidate_source_counts": discovery["source_counts"],
            "approved_group_split_exceptions": discovery[
                "approved_group_split_exceptions"
            ],
            "warnings": [
                {
                    "code": "approved_cross_split_source_groups",
                    "source": "landslidebench_agent",
                    "count": len(
                        discovery["approved_group_split_exceptions"][
                            "landslidebench_agent"
                        ]
                    ),
                    "message": "项目负责人要求保留全部源 split；这些 location_key 作为已知例外。",
                }
            ],
            "index_sha256": sha256_file(staging / "index.jsonl"),
            "files": hashes,
            "content_sha256": sha256_bytes(
                canonical_json(hashes).encode("utf-8")
            ),
            "contract": {
                "optical_layout": "CHW",
                "mask_layout": "1HW",
                "batch_mask_layout": "B1HW",
                "mask_validity_stored": False,
                "missing_auxiliary_representation": "absent",
            },
        }
        atomic_write_json(staging / "manifest.json", manifest)
        staging.replace(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument(
        "--datasets-root", type=Path, default=repo_root.parent / "datasets"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root.parent / "benchmark/oa_auxseg_hdf5_v1",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--small-per-source", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shard-target-mib", type=int, default=512)
    parser.add_argument("--estimate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.patch_size <= 0 or args.shard_target_mib <= 0:
        raise ValueError("--patch-size 和 --shard-target-mib 必须大于 0")
    samples, discovery = discover_all_sources(args.datasets_root, args.split_seed)
    selected = (
        select_small(samples, per_source=args.small_per_source, seed=args.seed)
        if args.mode == "small"
        else sorted(samples, key=_source_sort_key)
    )
    if args.estimate_only:
        report = estimate_logical_bytes(selected, args.patch_size)
        report["source_counts"] = dict(Counter(s.source for s in selected))
        report["split_counts"] = dict(Counter(s.split for s in selected))
        report["approved_group_split_exception_count"] = len(
            discovery["approved_group_split_exceptions"]["landslidebench_agent"]
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    target = build_benchmark(
        datasets_root=args.datasets_root,
        output_base=args.output_root,
        mode=args.mode,
        patch_size=args.patch_size,
        small_per_source=args.small_per_source,
        seed=args.seed,
        split_seed=args.split_seed,
        shard_target_mib=args.shard_target_mib,
    )
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"构建完成：{target}\n"
        f"样本数：{manifest['sample_count']}\n"
        f"index_sha256：{manifest['index_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
