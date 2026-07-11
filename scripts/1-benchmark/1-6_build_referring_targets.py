#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-6：基于已物化 benchmark 构建结构化指代目标。

用途：在 1-4 物化和 1-5 split 后，基于 benchmark 内部 mask/mask.npy
生成方位、尺度、形态、数量四类 referring targets。这里不生成训练文本，
只生成 expression-level target mask、grounding 和类别/子类结构化字段。
主要输入：benchmark/multisource_landslide_v1_<mode>/indexes/all.jsonl、
data/**/mask/mask.npy 和已有 preview/visual.png。
主要输出：data/**/referring/**/mask.npy、preview/referring.png、
indexes/referring_target_*.jsonl 和 sample_meta.json 中的 referring_targets。
写入行为：不会读取或改写 datasets/，也不会重写已有模态 .npy；会写 referring 派生产物。
所属流程：benchmark 构建 1-6；当前主模型不直接使用 referring 数据，但 benchmark 流程保留该产物。
推荐运行命令：
  python scripts/1-benchmark/1-6_build_referring_targets.py \
    --benchmark-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

import geohazard_referring_common as referring
from geohazard_benchmark_common import (
    DEFAULT_BENCHMARK_ROOT,
    ensure_dir,
    final_index_paths,
    flatten_referring_target_samples,
    modality_combo,
    project_path_arg,
    read_jsonl,
    referring_target_index_paths,
    resolve_repo_path,
    to_repo_rel,
    write_json,
    write_referring_target_split_indexes,
)


def sample_dir_for(sample: dict[str, Any], benchmark_dir: Path) -> Path:
    """优先使用 provenance.materialized_dir，缺失时按标准布局推导样本目录。"""
    materialized = (sample.get("provenance") or {}).get("materialized_dir")
    if materialized:
        path = resolve_repo_path(materialized)
        if path is not None:
            return path
    return benchmark_dir / "data" / str(sample["split"]) / str(sample["dataset_name"]) / str(sample["sample_id"])


def load_mask(sample: dict[str, Any]) -> Any | None:
    """只从 benchmark 内最终 mask.npy 读取二值 mask，不回读 datasets/。"""
    mask_info = sample.get("mask") or {}
    if mask_info.get("format") != "npy":
        return None
    path = resolve_repo_path(mask_info.get("path"))
    if path is None or not path.exists():
        return None
    arr = np.load(path)
    return (np.asarray(arr) > 0).astype(np.uint8)


def load_visual_preview(sample: dict[str, Any], sample_dir: Path, no_preview: bool) -> Any | None:
    """读取已有 preview/visual.png 作为 referring preview 底图。"""
    if no_preview:
        return None
    preview_paths = ((sample.get("preview") or {}).get("paths") or {})
    path = resolve_repo_path(preview_paths.get("visual"))
    if path is None:
        path = sample_dir / "preview" / "visual.png"
    if not path.exists():
        return None
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    return referring.to_chw(arr)


def read_sample_meta(sample_dir: Path) -> dict[str, Any]:
    """读取样本级元数据；缺失或损坏时返回空 dict。"""
    meta_path = sample_dir / "sample_meta.json"
    if not meta_path.exists():
        return {}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def merge_referring_target_config(benchmark_dir: Path) -> None:
    """preprocess_config.yaml 缺少 referring_target 时补写当前规则配置。"""
    config_path = benchmark_dir / "preprocess_config.yaml"
    current: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current = loaded
    if current.get("referring_target"):
        return
    current["referring_target"] = referring.build_referring_config()
    ensure_dir(config_path.parent)
    config_path.write_text(yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_sample_targets(
    sample: dict[str, Any],
    benchmark_dir: Path,
    *,
    overwrite: bool,
    no_preview: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """为单个样本生成 referring targets；返回带临时 targets 的样本副本和状态。"""
    item = dict(sample)
    sample_dir = sample_dir_for(item, benchmark_dir)
    meta = read_sample_meta(sample_dir)
    existing_targets = meta.get("referring_targets") if isinstance(meta.get("referring_targets"), list) else []
    status = {
        "sample_id": item.get("sample_id"),
        "dataset_name": item.get("dataset_name"),
        "split": item.get("split"),
        "action": "skipped",
        "num_targets": len(existing_targets),
        "reason": "",
        "quality_flags": [],
    }

    if item.get("supervision") != "mask":
        status["reason"] = "not_supervised_mask"
        return item, status
    final_mask = item.get("mask")
    if not isinstance(final_mask, dict):
        status["reason"] = "missing_mask_field"
        return item, status
    if final_mask.get("empty_mask") is True:
        status["reason"] = "empty_mask"
        return item, status
    if existing_targets and not overwrite:
        item["referring_targets"] = existing_targets
        status["action"] = "kept_existing"
        status["reason"] = "already_has_referring_targets"
        return item, status

    mask = load_mask(item)
    if mask is None:
        status["reason"] = "mask_npy_missing_or_unsupported"
        return item, status

    visual = load_visual_preview(item, sample_dir, no_preview)
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="referring_target_dry_run_") as tmp_dir:
            targets, referring_preview_path, errors, flags = referring.build_referring_targets(
                item,
                Path(tmp_dir),
                visual,
                mask,
                final_mask,
                enable_preview=not no_preview,
            )
    else:
        targets, referring_preview_path, errors, flags = referring.build_referring_targets(
            item,
            sample_dir,
            visual,
            mask,
            final_mask,
            enable_preview=not no_preview,
        )

    status["num_targets"] = len(targets)
    status["quality_flags"] = flags
    if errors:
        status["preview_errors"] = errors
    if not targets:
        status["reason"] = "no_referring_target_generated"
        return item, status

    status["action"] = "would_build" if dry_run else "built"
    item["referring_targets"] = targets
    if dry_run:
        return item, status

    meta_out = dict(meta or item)
    meta_out.update({
        "sample_id": item.get("sample_id"),
        "dataset_name": item.get("dataset_name"),
        "split": item.get("split"),
        "referring_targets": targets,
    })

    preview = dict(meta_out.get("preview") or item.get("preview") or {})
    preview_paths = dict(preview.get("paths") or {})
    if referring_preview_path:
        preview_paths["referring"] = referring_preview_path
    preview["paths"] = preview_paths
    preview_errors = list(preview.get("errors") or [])
    preview_errors.extend(errors)
    preview["errors"] = preview_errors
    meta_out["preview"] = preview

    quality_flags = set(meta_out.get("quality_flags") or item.get("quality_flags") or [])
    quality_flags.update(flags)
    quality_flags.add("referring_targets_generated_from_materialized_mask")
    if errors:
        quality_flags.add("referring_preview_failed")
    meta_out["quality_flags"] = sorted(quality_flags)
    write_json(sample_dir / "sample_meta.json", meta_out)
    return item, status


def add_referring_target_sampling_weights(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给 target-level 样本增加 dataset/modality/category/subtype 均衡权重。"""
    train_rows = [sample for sample in samples if sample.get("split") == "train"]
    dataset_counts = Counter(sample.get("dataset_name", "unknown") for sample in train_rows)
    combo_counts = Counter(modality_combo(sample) for sample in train_rows)
    category_counts = Counter(sample.get("category", "unknown") for sample in train_rows)
    subtype_counts = Counter(f"{sample.get('category', 'unknown')}:{sample.get('subtype', 'unknown')}" for sample in train_rows)
    out: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        combo = modality_combo(item)
        category = item.get("category", "unknown")
        subtype = item.get("subtype", "unknown")
        subtype_key = f"{category}:{subtype}"
        if item.get("split") == "train":
            item["sampling"] = {
                "dataset_balanced_weight": 1.0 / max(dataset_counts[item.get("dataset_name", "unknown")], 1),
                "modality_combo_balanced_weight": 1.0 / max(combo_counts[combo], 1),
                "referring_category_balanced_weight": 1.0 / max(category_counts[category], 1),
                "referring_subtype_balanced_weight": 1.0 / max(subtype_counts[subtype_key], 1),
                "modality_combo": combo,
                "referring_category": category,
                "referring_subtype": subtype,
            }
        else:
            item["sampling"] = {
                "dataset_balanced_weight": 0.0,
                "modality_combo_balanced_weight": 0.0,
                "referring_category_balanced_weight": 0.0,
                "referring_subtype_balanced_weight": 0.0,
                "modality_combo": combo,
                "referring_category": category,
                "referring_subtype": subtype,
            }
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于已物化 benchmark 构建结构化 referring target。")
    parser.add_argument("--benchmark-dir", type=project_path_arg, default=DEFAULT_BENCHMARK_ROOT, help="目标 benchmark 目录。")
    parser.add_argument("--overwrite", action="store_true", help="已有 referring_targets 时重新生成。")
    parser.add_argument("--dry-run", action="store_true", help="只统计可生成数量，不写文件。")
    parser.add_argument("--max-samples", type=int, default=None, help="调试用：最多处理多少个样本。")
    parser.add_argument("--no-preview", action="store_true", help="只生成 target mask，不生成 preview/referring.png。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_path = final_index_paths(args.benchmark_dir)["all"]
    samples = read_jsonl(all_path)
    if not samples:
        raise SystemExit(f"未找到最终索引或索引为空: {all_path}")

    if not args.dry_run:
        merge_referring_target_config(args.benchmark_dir)

    processed: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    selected = samples[: args.max_samples] if args.max_samples else samples
    progress = tqdm(selected, desc="构建指代目标", unit="sample")
    for sample in progress:
        item, status = build_sample_targets(
            sample,
            args.benchmark_dir,
            overwrite=args.overwrite,
            no_preview=args.no_preview,
            dry_run=args.dry_run,
        )
        processed.append(item)
        statuses.append(status)
        progress.set_postfix({
            "生成": sum(1 for row in statuses if row["action"] in {"built", "would_build"}),
            "跳过": sum(1 for row in statuses if row["action"] == "skipped"),
        })

    referring_target_samples = add_referring_target_sampling_weights(flatten_referring_target_samples(processed))
    action_counts = Counter(str(row["action"]) for row in statuses)
    reason_counts = Counter(str(row["reason"]) for row in statuses if row.get("reason"))
    category_counts = Counter(str(row.get("category", "unknown")) for row in referring_target_samples)
    report = {
        "说明": "指代目标构建只读取 benchmark 内部 all.jsonl 和 mask.npy，不回读 datasets/ 原始数据；训练文本由 2-instruction 生成。",
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "no_preview": args.no_preview,
        "max_samples": args.max_samples,
        "num_input_samples": len(samples),
        "num_processed_samples": len(selected),
        "num_referring_target_samples": len(referring_target_samples),
        "action_counts": dict(sorted(action_counts.items())),
        "skip_reason_counts": dict(sorted(reason_counts.items())),
        "referring_target_category_counts": dict(sorted(category_counts.items())),
        "referring_target_index": to_repo_rel(referring_target_index_paths(args.benchmark_dir)["all"]),
        "status_examples": statuses[:50],
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    write_referring_target_split_indexes(args.benchmark_dir, referring_target_samples)
    write_json(args.benchmark_dir / "reports" / "referring_target_build_report.json", report)
    print(
        "指代目标构建完成: "
        f"父样本 {len(selected)} 条，referring target {len(referring_target_samples)} 条 -> "
        f"{to_repo_rel(referring_target_index_paths(args.benchmark_dir)['all'])}"
    )


if __name__ == "__main__":
    main()
