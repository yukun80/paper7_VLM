#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成或汇总区域描述专家事实性评分。

用途：为冻结 raw generations 建立双人审核模板，或计算 parent-level ERFS。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.score_expert_factuality --eval-dir outputs/RUN/eval_gt --write-template
--output outputs/RUN/eval_gt/expert_review_template.jsonl
审核后命令：同一入口提供两次 --review，并将 --output 指向 expert_factuality_report.json。
主要输入：raw_generations.jsonl 和人工填写的 review JSONL。
主要输出：审核模板或 qpsalm_expert_region_factuality_v2_source_revalidated 报告。
写入行为：只写 --output；不会修改模型输出或审核文件。
所属流程：M6 正式 ERFS 主要终点评价。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.workflows.review import (
    ReviewLaunchError,
    run_expert_factuality_review,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score expert region factuality")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--review", action="append", default=[])
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = run_expert_factuality_review(
            eval_dir=args.eval_dir,
            reviews=args.review,
            minimum_reviewers=args.minimum_reviewers,
            seed=args.seed,
            write_template=args.write_template,
            output=args.output,
            overwrite_output=args.overwrite_output,
        )
    except ReviewLaunchError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.write_template:
        report = {
            "report": args.output,
            "num_parents": report["num_parents"],
            "erfs": report["expert_region_factuality_score"],
        }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
