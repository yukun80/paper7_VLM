#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only M2/M3 artifact readiness aggregation before D-1.

用途：统一重验 Description v4、Bridge v7、Unified v3 和 M3 v3 cache。
推荐调用：``qpsalm-segdesc validate artifacts``。
输入：四个已发布 artifact 目录及 mode。
输出：原子写入 artifact_readiness_report.json；不把 auto candidate 当 expert truth。
写入行为：只写显式 ``--output``，其余输入全部只读。
工作流阶段：D-1 前的 M2/M3 工程门禁。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qpsalm_seg.paths import resolve_project_path

from ..data.artifact_readiness import (
    ARTIFACT_READINESS_PROTOCOL,
    build_artifact_readiness_report,
)
from ..protocols.io import atomic_write_json


def run_artifact_readiness(
    *,
    mode: str,
    description_benchmark: str | Path,
    bridge_benchmark: str | Path,
    unified_benchmark: str | Path,
    description_cache: str | Path,
    output: str | Path,
) -> dict[str, object]:
    report = build_artifact_readiness_report(
        mode=mode,
        description_benchmark=description_benchmark,
        bridge_benchmark=bridge_benchmark,
        unified_benchmark=unified_benchmark,
        description_cache=description_cache,
    )
    destination = resolve_project_path(output) or Path(output)
    atomic_write_json(destination, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Description/Bridge/Unified/M3 artifacts for D-1"
    )
    parser.add_argument("--mode", choices=["small", "full"], required=True)
    parser.add_argument("--description-benchmark", required=True)
    parser.add_argument("--bridge-benchmark", required=True)
    parser.add_argument("--unified-benchmark", required=True)
    parser.add_argument("--description-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_artifact_readiness(
        mode=args.mode,
        description_benchmark=args.description_benchmark,
        bridge_benchmark=args.bridge_benchmark,
        unified_benchmark=args.unified_benchmark,
        description_cache=args.description_cache,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report.get("ready") is not True:
        raise SystemExit(1)
