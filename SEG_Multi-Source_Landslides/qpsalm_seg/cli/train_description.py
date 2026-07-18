#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 segmentation-grounded description 模型。

用途：运行 D-1、D0-D4 的独立描述训练，保存 qpsalm_segdesc_v1 best/last 权重。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.train_description --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --stage overfit
--seed 42 --device cuda --max-steps 100
--batch-size 2 --max-train-samples 64
--output-dir outputs/qpsalm_description/overfit_seed42 --overwrite-output
主要输入：已验证的 M1/M2 benchmark、description vision cache v1 和分割 checkpoint。
主要输出：checkpoint_best.pt、checkpoint_last.pt、validation 与 raw generation 报告。
写入行为：只写 --output-dir；不会改写 benchmark、cache 或分割 checkpoint。
所属流程：M6 描述训练；所有命令由用户手动运行。
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from qpsalm_seg.description.protocols.config import (
    DESCRIPTION_STAGES,
    load_segdesc_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation-grounded region description.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=DESCRIPTION_STAGES, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument(
        "--predicted-val-index",
        default=None,
        help="D4 固定 val prediction index；不得与 OOF train index 混用",
    )
    parser.add_argument(
        "--d4-curriculum-gate",
        default=None,
        help="前一档 fixed expert-val 通过后发布的相邻升档 gate",
    )
    parser.add_argument(
        "--d-minus-one-gate",
        default=None,
        help="D0 必需的当前 D-1 统一工程门禁",
    )
    parser.add_argument(
        "--artifact-readiness-report",
        default=None,
        help="D-1 overfit 必需的当前 Bridge v7/Unified v3/M3 v3 readiness",
    )
    parser.add_argument(
        "--predicted-mask-fraction",
        type=float,
        choices=[0.25, 0.50, 0.75],
        default=None,
        help="D4 预注册 predicted-mask curriculum tier",
    )
    parser.add_argument(
        "--d4-curriculum-sampling-seed",
        type=int,
        default=None,
        help="跨模型 seed 固定 D4 predicted-row population 的独立非负 seed",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--initialize-from", default=None, help="Load model weights only for a new D-stage.")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="构建并审计 D0 model/data/collator/optimizer，但不执行 optimizer step",
    )
    parser.add_argument(
        "--formal-output-dir",
        default=None,
        help="D0 preflight 要绑定的唯一正式训练输出目录",
    )
    parser.add_argument(
        "--d0-preflight-report",
        default=None,
        help="正式 D0 必需的 ready preflight_report.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_segdesc_config(args.config, {
        "stage": args.stage,
        "seed": args.seed,
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
        "predicted_val_index": args.predicted_val_index,
        "d_minus_one_gate": args.d_minus_one_gate,
        "artifact_readiness_report": args.artifact_readiness_report,
        "d4_curriculum_gate": args.d4_curriculum_gate,
        "predicted_mask_fraction": args.predicted_mask_fraction,
        "d4_curriculum_sampling_seed": args.d4_curriculum_sampling_seed,
        "output_dir": args.output_dir,
    })
    from qpsalm_seg.description.workflows.train import (
        DescriptionLaunchError,
        run_description_training,
    )

    try:
        report = run_description_training(
            config,
            config_reference=args.config,
            device_name=args.device,
            resume=args.resume,
            initialize_from=args.initialize_from,
            overwrite_output=args.overwrite_output,
            preflight_only=args.preflight_only,
            formal_output_dir=args.formal_output_dir,
            d0_preflight_report=args.d0_preflight_report,
        )
    except DescriptionLaunchError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False))
    if args.preflight_only and report.get("ready") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
