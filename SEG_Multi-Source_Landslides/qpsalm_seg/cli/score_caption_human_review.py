#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成或汇总冻结 RSIEval caption 的双人人工审核。

用途：建立不含参考答案的事实性/详细度/可读性审核模板，或汇总两份独立审核。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.score_caption_human_review --eval-dir <RSIEval eval> --write-template
--output <eval>/caption_human_review_template.jsonl
输入：完整冻结 RSIEval raw generations；汇总时另需至少两份人工填写的 JSONL。
输出：绑定 generation、图像和审核文件哈希的模板或 parent-macro JSON 报告。
写入行为：只原子写入 --output；不改 benchmark、generation、图像或审核输入。
所属流程：M6 RSIEval 人工次要评价；禁止用于 test checkpoint 选择。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.caption_human_review import (
    aggregate_caption_human_reviews,
    build_caption_human_review_template,
    write_caption_review_jsonl,
)
from qpsalm_seg.description.common import write_json
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or aggregate blind RSIEval caption human reviews."
    )
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--review", action="append", default=[])
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output) or Path(args.output)
    if output.exists() and not args.overwrite_output:
        raise SystemExit("caption human review output 已存在；请改路径或显式覆盖")
    if args.write_template:
        if args.review:
            raise ValueError("--write-template 与 --review 不能同时使用")
        rows = build_caption_human_review_template(args.eval_dir)
        write_caption_review_jsonl(output, rows)
        print(json.dumps({
            "template": str(output),
            "num_samples": len(rows),
            "reference_target_hidden": True,
        }, ensure_ascii=False))
        return
    if len(args.review) < max(2, int(args.minimum_reviewers)):
        raise ValueError("正式 caption human review 至少提供两份独立审核文件")
    report = aggregate_caption_human_reviews(
        args.eval_dir,
        args.review,
        seed=args.seed,
        minimum_reviewers=args.minimum_reviewers,
    )
    write_json(output, report)
    print(json.dumps({
        "report": str(output),
        "num_parents": report["num_parents"],
        "metrics": report["metrics"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
