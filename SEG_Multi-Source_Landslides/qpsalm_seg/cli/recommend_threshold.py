#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 QPSALM 报告中推荐二值化阈值。

用途：读取 threshold sweep，推荐 overall 与分组阈值并生成可复制 eval 命令。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.recommend_threshold --run outputs/RUN --output outputs/RUN/threshold_recommendations.json
主要输入：QPSALM run 目录或 JSON 报告。
主要输出：阈值推荐 JSON，可选写入 threshold_recommendations.json。
写入行为：只在指定 --output 时写报告。
所属流程：评估完成后的 threshold calibration；不重新训练模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.paths import resolve_repo_path
from qpsalm_seg.thresholding import recommend_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend QPSALM mask threshold from threshold_sweep.")
    parser.add_argument("--run", required=True, help="Run directory, eval directory, run_summary.json, or eval_report.json.")
    parser.add_argument("--block", default="auto", help="auto, eval, validation_best, validation, or report.")
    parser.add_argument(
        "--group-prefix",
        action="append",
        default=None,
        help="Group prefix to include, e.g. family_combo=. Can be repeated. Default: family_combo=.",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefixes = tuple(args.group_prefix or ["family_combo="])
    report = recommend_thresholds(
        args.run,
        block_name=args.block,
        group_prefixes=prefixes,
        limit=max(0, int(args.limit)),
        eval_device=args.eval_device,
    )
    if args.output:
        output = resolve_repo_path(args.output) or Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["output"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
