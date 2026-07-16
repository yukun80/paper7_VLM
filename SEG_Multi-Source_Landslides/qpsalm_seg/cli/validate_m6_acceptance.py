#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发布 M6 GT/fixed/end-to-end 三模式统一验收门禁。

用途：深度绑定 D-1、D0-D4 lineage、M4/D4、三种 expert val 模式、cycle 与在线 target audit。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.validate_m6_acceptance --gt-eval-dir <dir> --gt-expert-report <json>
--fixed-eval-dir <dir> --fixed-expert-report <json> --end-to-end-eval-dir <dir>
--end-to-end-expert-report <json> --bridge-benchmark <dir> --d4-final-gate <json>
--seed 42 --output <json>
输入：完整 frozen expert val 的三模式评价、三份盲审事实性报告和 D4 final gate。
输出：先写隐藏候选并从三模式绑定源完整重建，再原子发布 M6 acceptance gate；
科学阈值失败时保留可重放的 `passed=false` 报告，但不能授权 M7。
写入行为：只写 --output，不运行模型、benchmark、CUDA 或训练。
所属流程：M6 最终工程/科学输入绑定；M7 只能从 passed gate 继续。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.common import write_json
from qpsalm_seg.description.m6_acceptance import (
    build_m6_acceptance_gate,
    validate_m6_acceptance_gate,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate complete M6 evidence.")
    parser.add_argument("--gt-eval-dir", required=True)
    parser.add_argument("--gt-expert-report", required=True)
    parser.add_argument("--fixed-eval-dir", required=True)
    parser.add_argument("--fixed-expert-report", required=True)
    parser.add_argument("--end-to-end-eval-dir", required=True)
    parser.add_argument("--end-to-end-expert-report", required=True)
    parser.add_argument("--bridge-benchmark", required=True)
    parser.add_argument("--d4-final-gate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_m6_acceptance_gate(
        gt_evaluation_dir=args.gt_eval_dir,
        gt_expert_report=args.gt_expert_report,
        fixed_evaluation_dir=args.fixed_eval_dir,
        fixed_expert_report=args.fixed_expert_report,
        end_to_end_evaluation_dir=args.end_to_end_eval_dir,
        end_to_end_expert_report=args.end_to_end_expert_report,
        bridge_benchmark=args.bridge_benchmark,
        d4_final_gate=args.d4_final_gate,
        seed=args.seed,
    )
    output = resolve_project_path(args.output) or Path(args.output)
    candidate = output.with_name(f".{output.name}.candidate")
    try:
        write_json(candidate, report)
        validate_m6_acceptance_gate(candidate)
        candidate.replace(output)
    finally:
        candidate.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
