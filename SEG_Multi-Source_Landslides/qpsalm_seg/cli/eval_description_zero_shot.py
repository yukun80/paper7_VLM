#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行 Qwen3-VL 原生全图描述 zero-shot 基线。

用途：不加载 MGRR/desc_adapter，直接评价 Qwen3-VL 的 M1 single-image caption 能力。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_description_zero_shot --model models_zoo/Qwen3-VL-2B-Instruct
--benchmark benchmark/qpsalm_description_v2_small --split dev --device cuda
--output-dir outputs/qpsalm_description/zero_shot_dev
主要输出：raw_generations.jsonl、caption token F1 和 bootstrap CI。
写入行为：只写 --output-dir；不加载或修改 segmentation checkpoint。
所属流程：M6 D-1；本入口不声称具备区域 grounded description 能力。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate native Qwen zero-shot global caption.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    from qpsalm_seg.description.zero_shot import evaluate_zero_shot_global_caption

    report = evaluate_zero_shot_global_caption(
        model_path=args.model,
        benchmark=args.benchmark,
        split=args.split,
        output_dir=output,
        device=torch.device(args.device),
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        load_4bit=not args.no_4bit,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
