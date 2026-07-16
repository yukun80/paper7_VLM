#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为冻结 RSIEval generation 计算正式 caption 指标。

用途：用 pycocoevalcap 与显式本地 BERTScore 模型计算 BLEU/METEOR/ROUGE/CIDEr/SPICE/BERTScore。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.score_caption_metrics --eval-dir <RSIEval test eval> --bertscore-model <local model>
--bertscore-num-layers <encoder layer> --output <eval>/caption_metrics.json --device cuda
输入：完整 rsicap_caption/test eval 输出、本地 BERTScore encoder。
输出：绑定 eval/generation/model hash 的原子 JSON 报告。
写入行为：只写 --output；不修改 generation、benchmark 或 checkpoint。
所属流程：M6 RSIEval 次要语言评价；不得替代区域 grounding 指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from qpsalm_seg.description.caption_metrics import (
    score_caption_metrics,
    write_caption_metric_report,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score official RSIEval caption metrics.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--bertscore-model", required=True)
    parser.add_argument(
        "--bertscore-num-layers",
        required=True,
        type=int,
        help="BERTScore 所用本地 encoder 的显式输出层；不按模型名称猜测",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output) or Path(args.output)
    if output.exists() and not args.overwrite_output:
        raise SystemExit("caption metric output 已存在；请改路径或使用 --overwrite-output")
    try:
        report = score_caption_metrics(
            args.eval_dir,
            bertscore_model=args.bertscore_model,
            bertscore_num_layers=args.bertscore_num_layers,
            bertscore_batch_size=args.bertscore_batch_size,
            device=args.device,
            seed=args.seed,
        )
    except BaseException as exc:
        failure = {
            "protocol": "qpsalm_rsieval_caption_metrics_failure_v1",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path = output.with_name(output.stem + ".failure.json")
        write_caption_metric_report(failure_path, failure)
        raise
    write_caption_metric_report(output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
