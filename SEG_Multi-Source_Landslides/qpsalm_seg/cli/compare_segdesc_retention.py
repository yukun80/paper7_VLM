#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚合 M7 Small 三种子 full-val segmentation retention 门禁。

用途：深度校验三份 retention_gate.json 的 seed、checkpoint、原始评估报告、
full-val population 和共同 segmentation baseline，并要求三条独立 seed 链全部通过。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.compare_segdesc_retention --retention-gate
outputs/qpsalm_description/m7_joint_seed42/retention_full_val/retention_gate.json
--seed 42 --retention-gate
outputs/qpsalm_description/m7_joint_seed123/retention_full_val/retention_gate.json
--seed 123 --retention-gate
outputs/qpsalm_description/m7_joint_seed3407/retention_full_val/retention_gate.json
--seed 3407 --output outputs/qpsalm_description/m7_retention_seed_gate.json
输入：当前协议生成的三份正式 full-val retention gate 及对应训练 seed。
输出：原子写入一个三种子聚合审计 JSON；未通过时保留报告并返回非零。
写入行为：只写 --output，不修改 checkpoint、评估目录、benchmark 或 datasets。
所属流程：M7 Small 最终工程/科学验收；不能替代 M2 专家冻结或 M6 专家评价。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.common import write_json
from qpsalm_seg.description.retention import (
    aggregate_m7_retention_seed_gates,
    validate_m7_retention_seed_gate,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate three independent M7 full-val retention gates."
    )
    parser.add_argument("--retention-gate", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate_m7_retention_seed_gates(
        args.retention_gate,
        seeds=args.seed,
    )
    output = resolve_project_path(args.output) or Path(args.output)
    candidate = output.with_name(f".{output.name}.candidate")
    try:
        write_json(candidate, report)
        validate_m7_retention_seed_gate(candidate)
        candidate.replace(output)
    finally:
        candidate.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
