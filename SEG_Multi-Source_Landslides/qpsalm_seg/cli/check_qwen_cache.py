#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 Qwen condition cache 是否覆盖当前训练/验证文本。

脚本作用：不启动训练、不加载 Qwen 模型，只读取 JSONL index 和
condition_embedding_cache，验证 PSALM 双文本 schema 的 embedding 覆盖情况。
主要输入：YAML 配置、train/val index、condition embedding cache。
主要输出：coverage JSON；默认 coverage 不完整时以非零状态退出。
是否改写原始数据：不会；仅在 --output 指定时写报告 JSON。
典型用法：python -m qpsalm_seg.cli.check_qwen_cache --config ... --condition-embedding-cache ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qpsalm_seg.config import load_config
from qpsalm_seg.data import resolve_repo_path
from qpsalm_seg.qwen_cache import verify_qwen_cache_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check QPSALM qwen_cache text coverage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--condition-embedding-cache", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--preview-limit", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Print missing text report but exit 0. Useful for diagnostics.",
    )
    return parser.parse_args()


def write_json(path_ref: str | Path, payload: dict[str, Any]) -> Path:
    path = resolve_repo_path(path_ref)
    if path is None:
        raise FileNotFoundError(path_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config,
        overrides={
            "train_index": args.train_index,
            "val_index": args.val_index,
            "condition_embedding_cache": args.condition_embedding_cache,
            "controller": "qwen_cache",
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
    )
    report = verify_qwen_cache_coverage(
        config,
        splits=tuple(args.splits),
        preview_limit=int(args.preview_limit),
    )
    if args.output:
        report["output"] = str(write_json(args.output, report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ok") and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
