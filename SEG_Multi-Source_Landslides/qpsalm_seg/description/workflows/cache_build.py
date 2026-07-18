#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 cache build/verify thin workflow.

用途：将统一命令转发到 task-neutral Description Vision Cache v1 builder。
推荐调用：``qpsalm-segdesc cache build|verify``。
输入：Description/Bridge benchmark、可选 segmentation cache v3 和 Qwen 配置。
输出：cache manifest、shards 与 validation_report.json。
写入行为：仅由 data.cache_builder 管理显式 output-dir。
工作流阶段：M3 artifact orchestration。
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from ..data.cache_builder import build_or_verify_cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify task-neutral Description Vision Cache v1"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--description-benchmark", required=True)
    parser.add_argument("--bridge-benchmark")
    parser.add_argument("--segmentation-vision-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=["qwen", "hash-smoke"], default="qwen"
    )
    parser.add_argument("--components", default="single_image,multisource_parent")
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--layers", default="5,11,17,23")
    parser.add_argument("--spatial-sizes", default="16,8,6,4")
    parser.add_argument("--view-tokens", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    report = build_or_verify_cache(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report.get("errors"):
        raise SystemExit(1)
