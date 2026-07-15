#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成或汇总区域描述专家事实性评分。

用途：为冻结 raw generations 建立双人审核模板，或计算 parent-level ERFS。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.score_expert_factuality --eval-dir outputs/RUN/eval_gt --write-template
--output outputs/RUN/eval_gt/expert_review_template.jsonl
审核后命令：同一入口提供两次 --review，并将 --output 指向 expert_factuality_report.json。
主要输入：raw_generations.jsonl 和人工填写的 review JSONL。
主要输出：审核模板或 qpsalm_expert_region_factuality_v1 报告。
写入行为：只写 --output；不会修改模型输出或审核文件。
所属流程：M6 正式 ERFS 主要终点评价。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.common import write_json
from qpsalm_seg.description.expert_factuality import (
    aggregate_expert_factuality,
    build_expert_review_template,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score expert region factuality")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--review", action="append", default=[])
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output) or Path(args.output)
    if args.write_template:
        if args.review:
            raise ValueError("--write-template 与 --review 不能同时使用")
        rows = build_expert_review_template(args.eval_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(json.dumps({"template": str(output), "num_samples": len(rows)}, ensure_ascii=False))
        return
    if len(args.review) < args.minimum_reviewers:
        raise ValueError("正式 ERFS 至少提供 minimum-reviewers 份独立审核文件")
    report = aggregate_expert_factuality(
        args.eval_dir,
        args.review,
        seed=args.seed,
        minimum_reviewers=args.minimum_reviewers,
    )
    write_json(output, report)
    print(json.dumps({
        "report": str(output),
        "num_parents": report["num_parents"],
        "erfs": report["expert_region_factuality_score"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

