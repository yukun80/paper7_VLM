#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-6：基于已物化 benchmark 构建指代表达监督。

脚本作用：在 1-4 物化和 1-5 split 后，基于 benchmark 内部 mask/mask.npy
生成方位、尺度、形态、数量四类 referring expressions。
主要输入：benchmark/multisource_landslide_v1_<mode>/indexes/all.jsonl、
data/**/mask/mask.npy 和已有 preview/visual.png。
主要输出：data/**/referring/**/mask.npy、preview/referring.png、
indexes/referring_*.jsonl、更新后的 indexes/all.jsonl 和 sample_meta.json。
是否改写原始数据：不会读取或改写 datasets/，也不会重写已有模态 .npy。
典型用法：
  python scripts/1-benchmark/1-6_build_referring_expressions.py \
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
    flatten_referring_samples,
    modality_combo,
    read_jsonl,
    referring_index_paths,
    resolve_repo_path,
    to_repo_rel,
    write_json,
    write_referring_split_indexes,
    write_split_indexes,
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


def merge_referring_config(benchmark_dir: Path) -> None:
    """preprocess_config.yaml 缺少 referring_expression 时补写当前规则配置。"""
    config_path = benchmark_dir / "preprocess_config.yaml"
    current: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current = loaded
    if current.get("referring_expression"):
        return
    current["referring_expression"] = referring.build_referring_config()
    ensure_dir(config_path.parent)
    config_path.write_text(yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding="utf-8")


def expression_categories(sample: dict[str, Any]) -> list[str]:
    return [str(expr.get("category", "unknown")) for expr in sample.get("referring_expressions") or []]


def build_sample_referring(sample: dict[str, Any], benchmark_dir: Path, *, overwrite: bool, no_preview: bool, dry_run: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """为单个样本补生成 referring expressions。"""
    item = dict(sample)
    status = {
        "sample_id": item.get("sample_id"),
        "dataset_name": item.get("dataset_name"),
        "split": item.get("split"),
        "action": "skipped",
        "num_expressions": len(item.get("referring_expressions") or []),
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
    if item.get("referring_expressions") and not overwrite:
        status["action"] = "kept_existing"
        status["reason"] = "already_has_referring_expressions"
        return item, status

    sample_dir = sample_dir_for(item, benchmark_dir)
    mask = load_mask(item)
    if mask is None:
        status["reason"] = "mask_npy_missing_or_unsupported"
        return item, status

    visual = load_visual_preview(item, sample_dir, no_preview)
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="referring_build_dry_run_") as tmp_dir:
            expressions, referring_preview_path, errors, flags = referring.build_referring_expressions(
                item,
                Path(tmp_dir),
                visual,
                mask,
                final_mask,
                enable_preview=not no_preview,
            )
    else:
        expressions, referring_preview_path, errors, flags = referring.build_referring_expressions(
            item,
            sample_dir,
            visual,
            mask,
            final_mask,
            enable_preview=not no_preview,
        )
    status["num_expressions"] = len(expressions)
    status["quality_flags"] = flags
    if errors:
        status["preview_errors"] = errors
    if not expressions:
        status["reason"] = "no_referring_expression_generated"
        return item, status

    status["action"] = "built"
    status["reason"] = ""
    if dry_run:
        status["action"] = "would_build"
        return item, status

    item["referring_expressions"] = expressions
    preview = dict(item.get("preview") or {})
    preview_paths = dict(preview.get("paths") or {})
    if referring_preview_path:
        preview_paths["referring"] = referring_preview_path
    preview["paths"] = preview_paths
    preview_errors = list(preview.get("errors") or [])
    preview_errors.extend(errors)
    preview["errors"] = preview_errors
    item["preview"] = preview

    quality_flags = set(item.get("quality_flags") or [])
    quality_flags.update(flags)
    quality_flags.add("referring_generated_from_materialized_mask")
    if errors:
        quality_flags.add("referring_preview_failed")
    item["quality_flags"] = sorted(quality_flags)

    write_json(sample_dir / "sample_meta.json", item)
    return item, status


def add_referring_sampling_weights(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给 expression-level 样本增加 dataset/modality/category/subtype 均衡权重。"""
    train_rows = [sample for sample in samples if sample.get("split") == "train"]
    dataset_counts = Counter(sample.get("dataset_name", "unknown") for sample in train_rows)
    combo_counts = Counter(modality_combo(sample) for sample in train_rows)
    category_counts = Counter((sample.get("referring_expression") or {}).get("category", "unknown") for sample in train_rows)
    subtype_counts = Counter(
        f"{(sample.get('referring_expression') or {}).get('category', 'unknown')}:{(sample.get('referring_expression') or {}).get('subtype', 'unknown')}"
        for sample in train_rows
    )
    out: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        expression = item.get("referring_expression") or {}
        combo = modality_combo(item)
        category = expression.get("category", "unknown")
        subtype = expression.get("subtype", "unknown")
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
    parser = argparse.ArgumentParser(description="基于已物化 benchmark 构建指代表达监督。")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT, help="目标 benchmark 目录。")
    parser.add_argument("--overwrite", action="store_true", help="已有 referring_expressions 时重新生成。")
    parser.add_argument("--dry-run", action="store_true", help="只统计可生成数量，不写文件。")
    parser.add_argument("--max-samples", type=int, default=None, help="调试用：最多处理多少个样本。")
    parser.add_argument("--no-preview", action="store_true", help="只生成表达和 mask，不生成 preview/referring.png。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_path = final_index_paths(args.benchmark_dir)["all"]
    samples = read_jsonl(all_path)
    if not samples:
        raise SystemExit(f"未找到最终索引或索引为空: {all_path}")

    if not args.dry_run:
        merge_referring_config(args.benchmark_dir)

    processed: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    selected = samples[: args.max_samples] if args.max_samples else samples
    untouched = samples[len(selected):] if args.max_samples else []
    progress = tqdm(selected, desc="构建指代表达", unit="sample")
    for sample in progress:
        item, status = build_sample_referring(
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

    final_samples = processed + untouched
    referring_samples = add_referring_sampling_weights(flatten_referring_samples(final_samples))

    action_counts = Counter(str(row["action"]) for row in statuses)
    reason_counts = Counter(str(row["reason"]) for row in statuses if row.get("reason"))
    candidate_referring_samples = sum(int(row.get("num_expressions") or 0) for row in statuses if row["action"] in {"built", "would_build", "kept_existing"})
    category_counts = Counter()
    for sample in final_samples:
        category_counts.update(expression_categories(sample))
    report = {
        "说明": "指代表达构建只读取 benchmark 内部 all.jsonl 和 mask.npy，不回读 datasets/ 原始数据。",
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "no_preview": args.no_preview,
        "max_samples": args.max_samples,
        "num_input_samples": len(samples),
        "num_processed_samples": len(selected),
        "num_referring_samples": len(referring_samples),
        "num_candidate_referring_samples_in_processed": candidate_referring_samples,
        "action_counts": dict(sorted(action_counts.items())),
        "skip_reason_counts": dict(sorted(reason_counts.items())),
        "referring_category_counts": dict(sorted(category_counts.items())),
        "referring_index": to_repo_rel(referring_index_paths(args.benchmark_dir)["all"]),
        "status_examples": statuses[:50],
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    write_split_indexes(args.benchmark_dir, final_samples)
    write_referring_split_indexes(args.benchmark_dir, referring_samples)
    write_json(args.benchmark_dir / "reports" / "referring_build_report.json", report)
    print(
        "指代表达构建完成: "
        f"父样本 {len(final_samples)} 条，referring 样本 {len(referring_samples)} 条 -> "
        f"{to_repo_rel(referring_index_paths(args.benchmark_dir)['all'])}"
    )


if __name__ == "__main__":
    main()
