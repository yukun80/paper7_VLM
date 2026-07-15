#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较三 seed 的区域描述消融。

用途：对齐固定 test parent，执行逐样本 paired bootstrap，并联合检查 retrieval 与 UFCR。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.compare_description_runs --baseline outputs/crop_s42/eval --candidate
outputs/mgrr_s42/eval --seed 42 --baseline outputs/crop_s123/eval --candidate
outputs/mgrr_s123/eval --seed 123 --baseline outputs/crop_s3407/eval --candidate
outputs/mgrr_s3407/eval --seed 3407 --baseline-retrieval outputs/crop_s42/dior_eval
--candidate-retrieval outputs/mgrr_s42/dior_eval --baseline-retrieval outputs/crop_s123/dior_eval
--candidate-retrieval outputs/mgrr_s123/dior_eval --baseline-retrieval outputs/crop_s3407/dior_eval
--candidate-retrieval outputs/mgrr_s3407/dior_eval --baseline-expert
outputs/crop_s42/expert_factuality_report.json --candidate-expert
outputs/mgrr_s42/expert_factuality_report.json --baseline-expert
outputs/crop_s123/expert_factuality_report.json --candidate-expert
outputs/mgrr_s123/expert_factuality_report.json --baseline-expert
outputs/crop_s3407/expert_factuality_report.json --candidate-expert
outputs/mgrr_s3407/expert_factuality_report.json --output outputs/mgrr_seed_gate.json
主要输入：相同样本的 eval_report.json 与 raw_generations.jsonl。
主要输出：paired CI、same-image R@1、claim-level UFCR 与 2/3 seed 门禁。
写入行为：只写 --output，不修改任何运行目录。
所属流程：M4/M6 模块准入。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.common import write_json
from qpsalm_seg.description.comparison import compare_description_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paired description runs.")
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--baseline-retrieval", action="append", required=True)
    parser.add_argument("--candidate-retrieval", action="append", required=True)
    parser.add_argument("--baseline-expert", action="append", required=True)
    parser.add_argument("--candidate-expert", action="append", required=True)
    parser.add_argument("--unsupported-noninferiority", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_description_seeds(
        args.baseline,
        args.candidate,
        seeds=args.seed,
        unsupported_noninferiority=args.unsupported_noninferiority,
        baseline_retrieval_dirs=args.baseline_retrieval,
        candidate_retrieval_dirs=args.candidate_retrieval,
        baseline_expert_reports=args.baseline_expert,
        candidate_expert_reports=args.candidate_expert,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed_fraction_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
