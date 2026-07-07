#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-5：生成最终 train/val/test/unlabeled split 文件。

脚本作用：整理统一索引的最终 split 文件，并写入后续 dataloader 可用的
dataset-balanced、modality-combo-balanced 采样权重。
主要输入：benchmark/multisource_landslide_v1_<mode>/indexes/all.jsonl。
主要输出：indexes/train.jsonl、val.jsonl、test.jsonl、unlabeled.jsonl、
reports/split_report.json。
是否改写原始数据：不会改写 datasets/；只重写 benchmark/ 下的最终 split 索引。
典型用法：python scripts/1-benchmark/1-5_build_splits.py --benchmark-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geohazard_benchmark_common import (
    DEFAULT_BENCHMARK_ROOT,
    modality_combo,
    read_jsonl,
    split_index_paths,
    to_repo_rel,
    write_json,
    write_split_indexes,
)


def hash_value(key: str) -> float:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def maybe_derive_val(samples: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    if ratio <= 0:
        return samples
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.get("supervision") == "mask":
            by_dataset[str(sample.get("dataset_name"))].add(str(sample.get("split")))
    train_only = {dataset for dataset, splits in by_dataset.items() if splits == {"train"}}
    out: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        dataset = str(item.get("dataset_name"))
        if dataset in train_only and item.get("split") == "train" and item.get("supervision") == "mask":
            key = f"{item.get('dataset_name')}/{item.get('source_key')}"
            if hash_value(key) < ratio:
                item["split"] = "val"
                item["split_source"] = f"derived_from_train_ratio_{ratio}"
                flags = set(item.get("quality_flags") or [])
                flags.add("derived_val_from_train")
                item["quality_flags"] = sorted(flags)
        out.append(item)
    return out


def add_sampling_weights(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 dataset 和模态组合写入简单均衡权重，供 dataloader 后续选择使用。"""
    dataset_counts = Counter(sample.get("dataset_name", "unknown") for sample in samples if sample.get("split") == "train")
    combo_counts = Counter(modality_combo(sample) for sample in samples if sample.get("split") == "train")
    out: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        dataset = item.get("dataset_name", "unknown")
        combo = modality_combo(item)
        if item.get("split") == "train":
            dataset_weight = 1.0 / max(dataset_counts[dataset], 1)
            combo_weight = 1.0 / max(combo_counts[combo], 1)
        else:
            dataset_weight = 0.0
            combo_weight = 0.0
        item["sampling"] = {
            "dataset_balanced_weight": dataset_weight,
            "modality_combo_balanced_weight": combo_weight,
            "modality_combo": combo,
        }
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理最终 split JSONL，并写入训练采样权重字段。")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT, help="后缀式 small 或 full benchmark 输出目录。")
    parser.add_argument("--derive-val-ratio", type=float, default=0.0, help="对只有 train 的数据集派生 val 的比例；默认不派生。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(split_index_paths(args.benchmark_dir)["all"])
    samples = maybe_derive_val(samples, args.derive_val_ratio)
    samples = add_sampling_weights(samples)
    write_split_indexes(args.benchmark_dir, samples)
    split_counts = Counter(sample.get("split", "unknown") for sample in samples)
    report = {
        "说明": "最终 split 已写回 indexes/*.jsonl；默认保留官方 split。",
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "derive_val_ratio": args.derive_val_ratio,
        "num_by_split": dict(sorted(split_counts.items())),
    }
    write_json(args.benchmark_dir / "reports" / "split_report.json", report)
    print(f"最终 split 已生成: {to_repo_rel(args.benchmark_dir / 'indexes')}")


if __name__ == "__main__":
    main()
