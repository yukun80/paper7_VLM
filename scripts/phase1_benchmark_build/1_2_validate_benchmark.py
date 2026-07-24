#!/usr/bin/env python3
"""阶段 1B：独立验证 Benchmark 结构、索引、shape、hash 与数值。

命令：python 1_2_validate_benchmark.py --benchmark-root .../small --deep
输入：已构建的 Benchmark 目录。
输出：stdout JSON 验证报告；不修改 Benchmark。
写入：无。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from benchmark_common import (
    SCHEMA_VERSION,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)


def _error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def _warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def _validate_binary(
    report: dict[str, Any], name: str, values: np.ndarray
) -> bool:
    unique = np.unique(values)
    if not set(unique.tolist()).issubset({0, 1}):
        _error(report, f"{name}: 非法二值取值 {unique[:20].tolist()}")
        return False
    return True


def validate_benchmark(root: Path, *, deep: bool = False) -> dict[str, Any]:
    root = root.resolve()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_root": str(root),
        "deep": deep,
        "errors": [],
        "warnings": [],
        "sample_count": 0,
        "source_counts": {},
        "split_counts": {},
        "checked_shards": 0,
        "checked_samples": 0,
    }
    required = (
        "manifest.json",
        "build_config.json",
        "index.jsonl",
        "source_statistics.json",
        "SHA256SUMS.jsonl",
    )
    for name in required:
        if not (root / name).is_file():
            _error(report, f"缺少必需文件：{name}")
    if report["errors"]:
        return report
    try:
        manifest = read_json(root / "manifest.json")
        config = read_json(root / "build_config.json")
        rows = read_jsonl(root / "index.jsonl")
        hashes = read_jsonl(root / "SHA256SUMS.jsonl")
    except Exception as error:
        _error(report, f"读取 manifest/config/index/hash 失败：{error}")
        return report
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _error(report, f"manifest schema_version={manifest.get('schema_version')!r}")
    if config.get("schema_version") != SCHEMA_VERSION:
        _error(report, f"config schema_version={config.get('schema_version')!r}")
    if manifest.get("contract", {}).get("mask_validity_stored") is not False:
        _error(report, "manifest 必须声明 mask_validity_stored=false")
    index_hash = sha256_file(root / "index.jsonl")
    if manifest.get("index_sha256") != index_hash:
        _error(report, "index.jsonl SHA-256 与 manifest 不一致")
    hash_by_path: dict[str, dict[str, Any]] = {}
    for item in hashes:
        path = str(item.get("path", ""))
        if not path or path in hash_by_path:
            _error(report, f"SHA256SUMS 路径缺失或重复：{path!r}")
            continue
        hash_by_path[path] = item
        file_path = root / path
        if not file_path.is_file():
            _error(report, f"hash 清单文件缺失：{path}")
            continue
        if file_path.stat().st_size != int(item["size_bytes"]):
            _error(report, f"文件大小不符：{path}")
        if sha256_file(file_path) != item["sha256"]:
            _error(report, f"文件 SHA-256 不符：{path}")
    if manifest.get("content_sha256") != sha256_bytes(
        canonical_json(hashes).encode("utf-8")
    ):
        _error(report, "content_sha256 与 SHA256SUMS 内容不一致")
    if int(manifest.get("sample_count", -1)) != len(rows):
        _error(report, "manifest sample_count 与 index 行数不一致")
    seen_ids: set[str] = set()
    shard_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            _error(report, f"index:{line_number}: sample_id 缺失")
            continue
        if sample_id in seen_ids:
            _error(report, f"index:{line_number}: sample_id 重复 {sample_id}")
        seen_ids.add(sample_id)
        if row.get("schema_version") != SCHEMA_VERSION:
            _error(report, f"{sample_id}: schema_version 错误")
        expected_hash = row.get("record_sha256")
        hash_payload = dict(row)
        hash_payload.pop("record_sha256", None)
        actual_hash = sha256_bytes(canonical_json(hash_payload).encode("utf-8"))
        if expected_hash != actual_hash:
            _error(report, f"{sample_id}: record_sha256 错误")
        storage = row.get("storage", {})
        shard = storage.get("shard")
        row_index = storage.get("row")
        if not isinstance(shard, str) or not isinstance(row_index, int):
            _error(report, f"{sample_id}: storage.shard/row 非法")
            continue
        if not (root / shard).is_file():
            _error(report, f"{sample_id}: shard 不存在 {shard}")
        shard_rows[shard].append(row)
        if row.get("group_status") == "known" and row.get("source_group_id"):
            group_splits[
                (str(row["source"]), str(row["source_group_id"]))
            ].add(str(row["split"]))
    approved = {
        (source, group)
        for source, groups in manifest.get(
            "approved_group_split_exceptions", {}
        ).items()
        for group in groups
    }
    for key, splits in sorted(group_splits.items()):
        if len(splits) <= 1:
            continue
        if key in approved:
            _warning(
                report,
                f"已批准跨 split group：{key[0]}::{key[1]} -> {sorted(splits)}",
            )
        else:
            _error(
                report,
                f"未批准跨 split group：{key[0]}::{key[1]} -> {sorted(splits)}",
            )
    report["sample_count"] = len(rows)
    report["source_counts"] = dict(Counter(row.get("source") for row in rows))
    report["split_counts"] = dict(Counter(row.get("split") for row in rows))
    if not deep:
        report["status"] = "pass" if not report["errors"] else "fail"
        return report
    patch_size = int(config["patch_size"])
    for shard, indexed_rows in sorted(shard_rows.items()):
        shard_path = root / shard
        if not shard_path.is_file():
            continue
        try:
            with h5py.File(shard_path, "r") as handle:
                names: list[str] = []
                handle.visit(names.append)
                forbidden = [name for name in names if "mask_valid" in name.lower()]
                if forbidden:
                    _error(report, f"{shard}: 禁止存在 mask validity：{forbidden}")
                required_datasets = (
                    "optical",
                    "optical_pixel_valid",
                    "optical_channel_valid",
                    "mask",
                )
                for name in required_datasets:
                    if name not in handle:
                        _error(report, f"{shard}: 缺少 /{name}")
                if any(name not in handle for name in required_datasets):
                    continue
                sample_capacity = int(handle["mask"].shape[0])
                for row in indexed_rows:
                    sample_id = row["sample_id"]
                    row_index = int(row["storage"]["row"])
                    if row_index < 0 or row_index >= sample_capacity:
                        _error(report, f"{sample_id}: shard row 越界")
                        continue
                    optical = np.asarray(handle["optical"][row_index])
                    optical_valid = np.asarray(
                        handle["optical_pixel_valid"][row_index]
                    )
                    optical_channel_valid = np.asarray(
                        handle["optical_channel_valid"][row_index]
                    )
                    mask = np.asarray(handle["mask"][row_index])
                    expected_optical_shape = tuple(row["optical"]["shape"])
                    if optical.shape != expected_optical_shape:
                        _error(
                            report,
                            f"{sample_id}: optical shape={optical.shape}，"
                            f"索引为 {expected_optical_shape}",
                        )
                    if optical_valid.shape != optical.shape:
                        _error(report, f"{sample_id}: optical pixel validity shape 错误")
                    if optical_channel_valid.shape != (optical.shape[0],):
                        _error(report, f"{sample_id}: optical channel validity shape 错误")
                    if mask.shape != (1, patch_size, patch_size):
                        _error(report, f"{sample_id}: mask shape={mask.shape}")
                    if not np.isfinite(optical).all():
                        _error(report, f"{sample_id}: optical 包含 NaN/Inf")
                    if _validate_binary(
                        report, f"{sample_id} optical pixel validity", optical_valid
                    ):
                        if np.any(optical[optical_valid == 0] != 0):
                            _error(report, f"{sample_id}: optical 无效像素未清零")
                    _validate_binary(
                        report,
                        f"{sample_id} optical channel validity",
                        optical_channel_valid,
                    )
                    if _validate_binary(report, f"{sample_id} mask", mask):
                        ratio = float(mask.mean(dtype=np.float64))
                        if abs(ratio - float(row["foreground_ratio"])) > 1e-12:
                            _error(report, f"{sample_id}: foreground_ratio 不一致")
                    for modality, contract in row["auxiliaries"].items():
                        prefix = f"auxiliary/{modality}"
                        if prefix not in handle:
                            _error(report, f"{sample_id}: 缺少辅助模态 {modality}")
                            continue
                        group = handle[prefix]
                        for key in ("values", "pixel_valid", "channel_valid"):
                            if key not in group:
                                _error(
                                    report,
                                    f"{sample_id}: {modality} 缺少 {key}",
                                )
                        if any(
                            key not in group
                            for key in ("values", "pixel_valid", "channel_valid")
                        ):
                            continue
                        values = np.asarray(group["values"][row_index])
                        valid = np.asarray(group["pixel_valid"][row_index])
                        channel_valid = np.asarray(
                            group["channel_valid"][row_index]
                        )
                        if values.shape != tuple(contract["shape"]):
                            _error(
                                report,
                                f"{sample_id}: {modality} shape={values.shape}",
                            )
                        if valid.shape != values.shape:
                            _error(
                                report,
                                f"{sample_id}: {modality} pixel validity shape 错误",
                            )
                        if channel_valid.shape != (values.shape[0],):
                            _error(
                                report,
                                f"{sample_id}: {modality} channel validity shape 错误",
                            )
                        if not np.isfinite(values).all():
                            _error(
                                report, f"{sample_id}: {modality} 包含 NaN/Inf"
                            )
                        if _validate_binary(
                            report, f"{sample_id} {modality} pixel validity", valid
                        ) and np.any(values[valid == 0] != 0):
                            _error(
                                report, f"{sample_id}: {modality} 无效像素未清零"
                            )
                        _validate_binary(
                            report,
                            f"{sample_id} {modality} channel validity",
                            channel_valid,
                        )
                    report["checked_samples"] += 1
        except OSError as error:
            _error(report, f"{shard}: HDF5 无法打开：{error}")
        report["checked_shards"] += 1
    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--deep", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_benchmark(args.benchmark_root, deep=args.deep)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
