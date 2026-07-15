#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行 M7 full-val segmentation retention 门禁。

用途：在与基线相同的完整 val 上评估 joint checkpoint，验证 positive Dice 下降不超过 0.01。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_segdesc_retention --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --checkpoint
outputs/qpsalm_description/JOINT/checkpoint_best.pt --baseline-eval-report
outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json
--device cuda --output-dir outputs/qpsalm_description/JOINT/retention_full_val
主要输出：joint_segmentation_eval.json 和 retention_gate.json。
写入行为：只写 --output-dir；不以 monitor 指标替代 full-val 门禁。
所属流程：M7 最终验收。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-val segmentation retention.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-eval-report", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _positive_dice(report: dict) -> float:
    return float((((report.get("metrics") or {}).get("positive_only") or {}).get("dice")) or 0.0)


def main() -> None:
    args = parse_args()
    config = load_segdesc_config(args.config)
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = resolve_project_path(args.baseline_eval_report) or Path(args.baseline_eval_report)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    from qpsalm_seg.description.checkpoint import load_segdesc_checkpoint
    from qpsalm_seg.description.common import description_device, write_json
    from qpsalm_seg.description.runtime import build_segdesc_model
    from qpsalm_seg.engine.common import build_eval_loader
    from qpsalm_seg.engine.evaluator import evaluate

    device = description_device(args.device)
    model, _migration = build_segdesc_model(config, device)
    step, metadata = load_segdesc_checkpoint(args.checkpoint, model)
    segmentation_config = replace(
        model.segmentation.config,
        max_val_samples=args.max_samples or None,
    )
    loader = build_eval_loader(segmentation_config, args.split)
    with model.controller.adapter_scope("default"):
        report = evaluate(
            model.segmentation,
            loader,
            device,
            threshold=segmentation_config.eval_threshold,
        )
    baseline_dice = _positive_dice(baseline)
    current_dice = _positive_dice(report)
    drop = baseline_dice - current_dice
    gate = {
        "protocol": "qpsalm_segdesc_retention_v1",
        "checkpoint": args.checkpoint,
        "checkpoint_step": step,
        "checkpoint_metadata": metadata,
        "split": args.split,
        "num_samples": (report.get("coverage") or {}).get("num_samples"),
        "baseline_positive_dice": baseline_dice,
        "joint_positive_dice": current_dice,
        "absolute_drop": drop,
        "maximum_allowed_drop": config.segmentation_retention_max_drop,
        "passed": drop <= config.segmentation_retention_max_drop,
    }
    write_json(output / "joint_segmentation_eval.json", report)
    write_json(output / "retention_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()
