#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 D4 fixed expert-val curriculum/最终 M7 准入 gate。

用途：用冻结 Pilot 阈值审计当前 D3b/D4 checkpoint 的完整 Vision-only val generation 与
双人 ERFS 报告；只允许 D3b→25%→50%→75% 相邻升档，或为 75% checkpoint 发布 M7 gate。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.validate_d4_curriculum --eval-dir outputs/RUN/eval_fixed_val
--expert-report outputs/RUN/eval_fixed_val/expert_factuality_report.json
--bridge-benchmark benchmark/landslide_region_description_v1_small --current-fraction 0.25
--next-fraction 0.50 --seed 42 --output outputs/RUN/d4_to_50_gate.json
输入：当前协议完整 val evaluation、对应双人专家事实性报告和 frozen Bridge Pilot。
输出：先写隐藏候选并从其绑定源完整重建，完全一致后原子发布 transition 或
`--final-m7` acceptance gate；指标未达标时返回非零并保留可审计 gate。
写入行为：只写 --output，不修改 generation、review、checkpoint、benchmark 或 datasets。
所属流程：M6 D4 课程升档与 M7 初始化前置验收。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.common import write_json
from qpsalm_seg.description.d4_curriculum import (
    build_d4_curriculum_gate,
    validate_d4_curriculum_gate,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one D4 curriculum tier.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--expert-report", required=True)
    parser.add_argument("--bridge-benchmark", required=True)
    parser.add_argument(
        "--m4-suite-gate",
        default=None,
        help="仅 0->0.25 必需：五 baseline M4 suite gate",
    )
    parser.add_argument(
        "--current-fraction", type=float, choices=[0.0, 0.25, 0.50, 0.75], required=True
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--next-fraction", type=float, choices=[0.25, 0.50, 0.75]
    )
    destination.add_argument("--final-m7", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_d4_curriculum_gate(
        evaluation_dir=args.eval_dir,
        expert_report=args.expert_report,
        bridge_benchmark=args.bridge_benchmark,
        current_fraction=args.current_fraction,
        next_fraction=None if args.final_m7 else args.next_fraction,
        seed=args.seed,
        m4_suite_gate=args.m4_suite_gate,
    )
    output = resolve_project_path(args.output) or Path(args.output)
    candidate = output.with_name(f".{output.name}.candidate")
    try:
        write_json(candidate, report)
        validate_d4_curriculum_gate(candidate)
        candidate.replace(output)
    finally:
        candidate.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
