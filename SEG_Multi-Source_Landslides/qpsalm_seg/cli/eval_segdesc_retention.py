#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行 M7 full-val segmentation retention 门禁。

用途：在相同完整 val 上重放 segmentation baseline 并评估 joint checkpoint。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_segdesc_retention --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --seed 42 --checkpoint
outputs/qpsalm_description/JOINT/checkpoint_best.pt --baseline-eval-report
outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json
--device cuda --output-dir outputs/qpsalm_description/JOINT/retention_full_val
主要输出：baseline_segmentation_replay.json、joint_segmentation_eval.json 和 retention_gate.json。
写入行为：只写 --output-dir；不以 monitor 指标代替 full-val 门禁。
注意：--max-samples 仅供 smoke；非零时正式 passed 恒为 false。
所属流程：M7 最终验收。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.workflows.retention import run_retention_workflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full-val segmentation retention."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-eval-report", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gate = run_retention_workflow(
        config_path=args.config,
        seed=args.seed,
        checkpoint_ref=args.checkpoint,
        baseline_eval_report=args.baseline_eval_report,
        split=args.split,
        device_name=args.device,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        overwrite_output=args.overwrite_output,
    )
    print(json.dumps(gate, ensure_ascii=False, allow_nan=False))
    accepted = (
        bool(gate["preliminary_passed"])
        if int(args.max_samples) > 0 else bool(gate["passed"])
    )
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
