#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 Multi-Source Qwen-PSALM-Seg 训练索引。

用途：统计核心模板样本、模态组合、shape、GSD 和被跳过的索引行。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.inspect_data --config
SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --split train --limit 16
主要输入：benchmark 的 indexes/instruction_{train,val,test}.jsonl。
主要输出：终端文本或 JSON。
写入行为：只读检查，不写 benchmark 或 outputs。
所属流程：数据构建后的质量检查，也可用于训练前核对模态分布。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from qpsalm_seg.config import load_config
from qpsalm_seg.indexing import iter_jsonl, stats_to_text, summarize_rows
from qpsalm_seg.paths import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect QPSALM instruction dataset fields.")
    parser.add_argument("--config", default="SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=16, help="Limit displayed entries per counter.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally scan only the first N rows.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmark_ref = args.benchmark_dir or str(config.benchmark_dir)
    benchmark_dir = resolve_repo_path(benchmark_ref)
    if benchmark_dir is None:
        raise FileNotFoundError(benchmark_ref)
    if args.split == "train":
        index_rel = str(config.train_index)
    elif args.split == "val":
        index_rel = str(config.val_index)
    else:
        index_rel = str(config.test_index)
    index_path = resolve_repo_path(benchmark_dir / index_rel)
    if index_path is None:
        raise FileNotFoundError(benchmark_dir / index_rel)
    stats = summarize_rows(iter_jsonl(index_path, max_rows=args.max_rows), config.task_families)
    if args.json:
        print(json.dumps(dataclasses.asdict(stats), ensure_ascii=False, indent=2))
    else:
        print(f"index={index_path}")
        print(stats_to_text(stats, limit=args.limit))


if __name__ == "__main__":
    main()
