#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交替训练 segmentation 与 description 双 Adapter。

用途：运行 M7 三 DataLoader 交替训练并执行 monitor segmentation retention gate。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.train_segdesc_joint --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --seed 42 --device cuda
--initialize-from outputs/qpsalm_description/M6/checkpoint_best.pt
--region-stage predicted_mask --predicted-mask-fraction 0.75
--d4-final-acceptance-gate outputs/qpsalm_description/M6/d4_final_m7_gate.json
--m6-acceptance-gate outputs/qpsalm_description/M6/m6_acceptance_gate.json
--output-dir outputs/qpsalm_description/joint_seed42 --overwrite-output
主要输入：冻结的 M1/M2 benchmark、description cache、原分割 checkpoint。
主要输出：qpsalm_segdesc_v1 best/last、三任务历史和 monitor retention 报告。
写入行为：只写 --output-dir；不会构造混合 collate 或改写源 benchmark。
所属流程：M7；需先完成专家 Bridge 与 M6 Small 门禁。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.protocols.config import load_segdesc_config
from qpsalm_seg.description.workflows.joint_train import (
    JointTrainingLaunchError,
    run_joint_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint segmentation-description alternating trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--segmentation-batch-size", type=int, default=None)
    parser.add_argument("--description-batch-size", type=int, default=None)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="每个 optimizer step 对同一任务累积的 microbatch 数。",
    )
    parser.add_argument(
        "--train-shared-segmentation-dense",
        action="store_true",
        default=None,
        help="Ablation only: also update SANE/QMEF/PMRD and controller projections.",
    )
    parser.add_argument("--region-stage", choices=["bridge_auto", "bridge_expert", "predicted_mask"], default=None)
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument(
        "--predicted-val-index",
        default=None,
        help="M7 predicted-mask region loader 的固定 val prediction index",
    )
    parser.add_argument(
        "--d4-final-acceptance-gate",
        default=None,
        help="75% predicted tier 自身通过 fixed expert-val 后发布的 M7 gate",
    )
    parser.add_argument(
        "--m6-acceptance-gate",
        default=None,
        help="GT/fixed/end-to-end、cycle 与 D4 final 共同通过的 M6 gate",
    )
    parser.add_argument(
        "--predicted-mask-fraction",
        type=float,
        choices=[0.75],
        default=None,
        help="M7 predicted-mask 主路线固定沿用已验收的 D4 75% tier",
    )
    parser.add_argument(
        "--d4-curriculum-sampling-seed",
        type=int,
        default=None,
        help="必须与 D4 三档相同的固定 predicted-row selection seed",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--initialize-from", required=False, help="Required for a new M7 run: load the accepted M6 model weights.")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_segdesc_config(args.config, {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "val_interval": args.val_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "joint_segmentation_batch_size": args.segmentation_batch_size,
        "joint_description_batch_size": args.description_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "joint_train_shared_segmentation_dense": args.train_shared_segmentation_dense,
        "joint_region_stage": args.region_stage,
        "predicted_index": args.predicted_index,
        "predicted_val_index": args.predicted_val_index,
        "d4_final_acceptance_gate": args.d4_final_acceptance_gate,
        "m6_acceptance_gate": args.m6_acceptance_gate,
        "predicted_mask_fraction": args.predicted_mask_fraction,
        "d4_curriculum_sampling_seed": args.d4_curriculum_sampling_seed,
        "output_dir": args.output_dir,
    })
    try:
        report = run_joint_training(
            config,
            config_reference=args.config,
            device_name=args.device,
            resume=args.resume,
            initialize_from=args.initialize_from,
            overwrite_output=args.overwrite_output,
        )
    except JointTrainingLaunchError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
