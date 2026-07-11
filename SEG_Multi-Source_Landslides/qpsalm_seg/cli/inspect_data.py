#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 Multi-Source Qwen-PSALM-Seg 训练索引。

用途：统计核心模板样本、模态组合、shape、GSD 和被跳过的索引行。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.inspect_data --config
SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml --split train --limit 16
主要输入：benchmark 的 indexes/instruction_{train,val,test}.jsonl。
主要输出：终端文本或 JSON。
写入行为：只读检查，不写 benchmark 或 outputs。
所属流程：数据构建后的质量检查，也可用于训练前核对模态分布。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from qpsalm_seg.indexing import iter_jsonl, stats_to_text, summarize_rows
from qpsalm_seg.paths import resolve_repo_path


DEFAULT_CORE_TEMPLATES = [
    "generic_landslide_v1",
    "negative_aware_landslide_v1",
    "multisource_landslide_v1",
    "terrain_evidence_landslide_v1",
    "sar_terrain_landslide_v1",
    "insar_evidence_landslide_v1",
]


def load_light_config(path: str | None) -> dict[str, object]:
    """用 stdlib 读取本项目简单 YAML 配置，避免 inspect 依赖 PyYAML/torch。"""
    data: dict[str, object] = {
        "benchmark_dir": "benchmark/multisource_landslide_v1_small",
        "train_index": "indexes/instruction_train.jsonl",
        "val_index": "indexes/instruction_val.jsonl",
        "test_index": "indexes/instruction_test.jsonl",
        "core_templates": list(DEFAULT_CORE_TEMPLATES),
    }
    if not path:
        return data
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(path)
    lines = config_path.read_text(encoding="utf-8").splitlines()
    current_list: str | None = None
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if current_list and stripped.startswith("- "):
            item = stripped[2:].strip().strip("'\"")
            data.setdefault(current_list, [])
            assert isinstance(data[current_list], list)
            data[current_list].append(item)
            continue
        current_list = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "core_templates":
            data[key] = []
            current_list = key
        elif key in {"benchmark_dir", "train_index", "val_index", "test_index"}:
            data[key] = value.strip("'\"")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect QPSALM instruction dataset fields.")
    parser.add_argument("--config", default="SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_core.yaml")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=16, help="Limit displayed entries per counter.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally scan only the first N rows.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_light_config(args.config)
    benchmark_ref = args.benchmark_dir or str(config["benchmark_dir"])
    benchmark_dir = resolve_repo_path(benchmark_ref)
    if benchmark_dir is None:
        raise FileNotFoundError(benchmark_ref)
    if args.split == "train":
        index_rel = str(config["train_index"])
    elif args.split == "val":
        index_rel = str(config["val_index"])
    else:
        index_rel = str(config["test_index"])
    core_templates = list(config["core_templates"])
    index_path = resolve_repo_path(benchmark_dir / index_rel)
    if index_path is None:
        raise FileNotFoundError(benchmark_dir / index_rel)
    stats = summarize_rows(iter_jsonl(index_path, max_rows=args.max_rows), core_templates)
    if args.json:
        print(json.dumps(dataclasses.asdict(stats), ensure_ascii=False, indent=2))
    else:
        print(f"index={index_path}")
        print(stats_to_text(stats, limit=args.limit))


if __name__ == "__main__":
    main()
