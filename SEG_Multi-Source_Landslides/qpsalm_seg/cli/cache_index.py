#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 QPSALM 核心 instruction 索引缓存。

脚本作用：从 benchmark 的 instruction JSONL 中过滤 Phase 1 核心模板，生成更小的
训练/验证 JSONL，减少真实训练启动时的大索引扫描开销。
主要输入：indexes/instruction_train.jsonl、indexes/instruction_val.jsonl。
主要输出：outputs/qpsalm_index_cache/qpsalm_core_{train,val}.jsonl 与 summary.json。
是否改写原始数据：不会。
典型用法：python -m qpsalm_seg.cli.cache_index --config SEG_Multi-Source_Landslides/configs/qpsalm_tiny_text_probe.yaml。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qpsalm_seg.cli.inspect_data import load_light_config
from qpsalm_seg.indexing import (
    canonical_modality_combo,
    iter_jsonl,
    normalization_combo,
    raw_modality_combo,
    resolve_repo_path,
    sensor_combo,
    should_skip_row,
)


def interleave_by_canonical_combo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按规范模态组合 round-robin 写出，避免 val 前几个 batch 只覆盖一个组合。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        combo = canonical_modality_combo(row)
        if combo not in buckets:
            buckets[combo] = []
            order.append(combo)
        buckets[combo].append(row)
    interleaved: list[dict[str, Any]] = []
    max_len = max((len(items) for items in buckets.values()), default=0)
    for idx in range(max_len):
        for combo in order:
            items = buckets[combo]
            if idx < len(items):
                interleaved.append(items[idx])
    return interleaved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache core QPSALM instruction rows into small JSONL files.")
    parser.add_argument("--config", default="SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_core.yaml")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/qpsalm_index_cache")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional source scan cap for quick tests.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional kept-row cap per split.")
    parser.add_argument(
        "--strategy",
        choices=["first", "balanced-canonical"],
        default="first",
        help="first keeps source order; balanced-canonical keeps up to N samples per canonical combo.",
    )
    parser.add_argument("--samples-per-combo", type=int, default=8)
    return parser.parse_args()


def cache_split(
    split: str,
    benchmark_dir: Path,
    index_rel: str,
    core_templates: list[str],
    output_dir: Path,
    max_rows: int | None,
    max_samples: int | None,
    strategy: str,
    samples_per_combo: int,
) -> dict[str, Any]:
    index_path = resolve_repo_path(benchmark_dir / index_rel)
    if index_path is None or not index_path.exists():
        raise FileNotFoundError(index_path)
    out_path = output_dir / f"qpsalm_core_{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = Counter()
    templates = Counter()
    raw_combos = Counter()
    canonical_combos = Counter()
    sensor_combos = Counter()
    normalization_combos = Counter()
    rows_seen = 0
    rows_kept = 0
    selected: list[dict[str, Any]] = []
    selected_by_combo: Counter[str] = Counter()
    for row in iter_jsonl(index_path, max_rows=max_rows):
        rows_seen += 1
        reason = should_skip_row(row, core_templates)
        if reason is not None:
            skipped[reason] += 1
            continue
        combo = canonical_modality_combo(row)
        if strategy == "balanced-canonical" and selected_by_combo[combo] >= samples_per_combo:
            continue
        selected.append(row)
        selected_by_combo[combo] += 1
        rows_kept += 1
        templates[str(row.get("template_id") or row.get("task_template_id") or "unknown")] += 1
        raw_combos[raw_modality_combo(row)] += 1
        canonical_combos[combo] += 1
        sensor_combos[sensor_combo(row)] += 1
        normalization_combos[normalization_combo(row)] += 1
        if max_samples is not None and rows_kept >= max_samples:
            break
    output_rows = interleave_by_canonical_combo(selected) if strategy == "balanced-canonical" else selected
    with out_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "split": split,
        "source_index": str(index_path),
        "cache_index": str(out_path),
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "strategy": strategy,
        "output_order": "round_robin_canonical_combo" if strategy == "balanced-canonical" else "source_order",
        "samples_per_combo": samples_per_combo if strategy == "balanced-canonical" else None,
        "skipped": dict(sorted(skipped.items())),
        "templates": dict(templates.most_common()),
        "raw_combos": dict(raw_combos.most_common()),
        "canonical_combos": dict(canonical_combos.most_common()),
        "sensor_combos": dict(sensor_combos.most_common()),
        "normalization_combos": dict(normalization_combos.most_common()),
    }


def main() -> None:
    args = parse_args()
    config = load_light_config(args.config)
    benchmark_dir = Path(args.benchmark_dir or str(config["benchmark_dir"]))
    output_dir = resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise FileNotFoundError(args.output_dir)
    core_templates = list(config["core_templates"])
    splits = ["train", "val"] if args.split == "both" else [args.split]
    reports = []
    for split in splits:
        index_rel = str(config["train_index"] if split == "train" else config["val_index"])
        reports.append(
            cache_split(
                split=split,
                benchmark_dir=benchmark_dir,
                index_rel=index_rel,
                core_templates=core_templates,
                output_dir=output_dir,
                max_rows=args.max_rows,
                max_samples=args.max_samples,
                strategy=args.strategy,
                samples_per_combo=args.samples_per_combo,
            )
        )
    summary = {
        "output_dir": str(output_dir),
        "reports": reports,
        "train_index_override": str(output_dir / "qpsalm_core_train.jsonl"),
        "val_index_override": str(output_dir / "qpsalm_core_val.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
