#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估 segmentation-grounded description checkpoint。

用途：分别执行 GT-mask oracle、固定预测 mask 或端到端分割后描述评价。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_description --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --stage bridge_expert
--checkpoint outputs/qpsalm_description/RUN/checkpoint_best.pt --split val
--evaluation-mode gt_mask --device cuda --output-dir outputs/qpsalm_description/RUN/eval_gt
主要输出：eval_report.json、raw_generations.jsonl 和 counterfactual_generations.jsonl。
写入行为：只写 --output-dir，不修复或覆盖原始模型输出。
所属流程：M6 描述评价；主结构指标只使用未修复 raw JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from qpsalm_seg.description.config import (
    DESCRIPTION_EVAL_MODES,
    DESCRIPTION_STAGES,
    load_segdesc_config,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation-grounded descriptions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=DESCRIPTION_STAGES, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "val", "test"], default="val")
    parser.add_argument("--evaluation-mode", choices=DESCRIPTION_EVAL_MODES, default=None)
    parser.add_argument("--region-protocol", choices=["vision_only", "assisted"], default=None)
    parser.add_argument(
        "--region-encoder",
        choices=[
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        ],
        default=None,
    )
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument("--segmentation-mask-threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-generate-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--counterfactual-samples", type=int, default=None)
    parser.add_argument("--no-counterfactuals", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_segdesc_config(args.config, {
        "stage": args.stage,
        "evaluation_mode": args.evaluation_mode,
        "region_protocol": args.region_protocol,
        "region_encoder": args.region_encoder,
        "predicted_index": args.predicted_index,
        "segmentation_mask_threshold": args.segmentation_mask_threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_val_samples": args.max_val_samples,
        "max_generate_samples": args.max_generate_samples,
        "max_new_tokens": args.max_new_tokens,
        "counterfactual_samples": args.counterfactual_samples,
        "output_dir": args.output_dir,
    })
    output = resolve_project_path(config.output_dir) or Path(config.output_dir)
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    from qpsalm_seg.description.checkpoint import load_segdesc_checkpoint
    from qpsalm_seg.description.common import (
        build_description_dataset,
        build_description_loader,
        description_device,
        set_description_seed,
        write_json,
    )
    from qpsalm_seg.description.evaluator import evaluate_description
    from qpsalm_seg.description.runtime import build_segdesc_model

    set_description_seed(config.seed)
    device = description_device(args.device)
    model, migration = build_segdesc_model(config, device)
    step, metadata = load_segdesc_checkpoint(args.checkpoint, model)
    dataset = build_description_dataset(
        config, model.description_backbone.bank, split=args.split, training=False
    )
    loader = build_description_loader(dataset, config, training=False)
    report = evaluate_description(
        model,
        loader,
        config,
        device,
        split=args.split,
        output_dir=output,
        run_counterfactuals=not args.no_counterfactuals,
    )
    report.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "checkpoint_metadata": metadata,
        "segmentation_migration": migration,
    })
    write_json(output / "eval_report.json", report)
    print(json.dumps({
        "eval_report": str(output / "eval_report.json"),
        "checkpoint_step": step,
        "stage": config.stage,
        "split": args.split,
        "mode": config.evaluation_mode,
        "num_samples": report["num_samples"],
        "generation_metrics": report["generation_metrics"],
        "same_image_retrieval": report["same_image_retrieval"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
