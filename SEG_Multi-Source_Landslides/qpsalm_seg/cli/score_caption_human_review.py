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

from qpsalm_seg.description.workflows.review import (
    ReviewLaunchError,
    run_caption_human_review,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = run_caption_human_review(
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
            "metrics": report["metrics"],
        }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
