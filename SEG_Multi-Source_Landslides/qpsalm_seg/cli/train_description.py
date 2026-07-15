#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 segmentation-grounded description 模型。

用途：运行 D-1、D0-D4 的独立描述训练，保存 qpsalm_segdesc_v1 best/last 权重。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.train_description --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --stage overfit
--device cuda --max-steps 100 --output-dir outputs/qpsalm_description/overfit --overwrite-output
主要输入：已验证的 M1/M2 benchmark、description vision cache v1 和分割 checkpoint。
主要输出：checkpoint_best.pt、checkpoint_last.pt、validation 与 raw generation 报告。
写入行为：只写 --output-dir；不会改写 benchmark、cache 或分割 checkpoint。
所属流程：M6 描述训练；所有命令由用户手动运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from qpsalm_seg.description.config import DESCRIPTION_STAGES, load_segdesc_config
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation-grounded region description.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=DESCRIPTION_STAGES, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--region-protocol", choices=["vision_only", "assisted"], default=None)
    parser.add_argument(
        "--region-encoder",
        choices=[
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        ],
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-generate-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--initialize-from", default=None, help="Load model weights only for a new D-stage.")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_segdesc_config(args.config, {
        "stage": args.stage,
        "region_protocol": args.region_protocol,
        "region_encoder": args.region_encoder,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_generate_samples": args.max_generate_samples,
        "learning_rate": args.learning_rate,
        "amp_dtype": args.amp_dtype,
        "val_interval": args.val_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "predicted_index": args.predicted_index,
        "output_dir": args.output_dir,
    })
    output = resolve_project_path(config.output_dir) or Path(config.output_dir)
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    from qpsalm_seg.description.trainer import train_description

    report = train_description(
        config,
        device_name=args.device,
        resume=args.resume,
        initialize_from=args.initialize_from,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
