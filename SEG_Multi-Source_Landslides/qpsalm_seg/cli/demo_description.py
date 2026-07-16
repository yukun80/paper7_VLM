#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动 benchmark 区域描述交互界面。

用途：手动选择 M1/M2 样本、GT/fixed/end-to-end region、full/zero 反事实和指令，展示
实际输入 mask、raw generation、解析结果与 mask provenance。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.demo_description --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --stage bridge_expert
--checkpoint outputs/qpsalm_description/RUN/checkpoint_best.pt --split val --device cuda
主要输出：本地 Gradio 页面，默认 http://127.0.0.1:7861。
写入行为：不写 benchmark 或 checkpoint；并发固定为 1。
所属流程：M6 质检与汇报展示。
"""

from __future__ import annotations

import argparse

from qpsalm_seg.description.config import DESCRIPTION_STAGES, load_segdesc_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch grounded description Gradio demo.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=DESCRIPTION_STAGES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation-mode",
        choices=["gt_mask", "fixed_prediction", "end_to_end"],
        default=None,
    )
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument(
        "--region-source", choices=["gt_global_mask"], default=None,
        help="可选：与正式 M6 global-only population 使用同一 region filter",
    )
    parser.add_argument("--segmentation-mask-threshold", type=float, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--region-protocol", choices=["vision_only", "assisted"], default=None)
    parser.add_argument(
        "--region-encoder",
        choices=[
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        ],
        default=None,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_segdesc_config(args.config, {
        "stage": args.stage,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "predicted_index": args.predicted_index,
        "evaluation_region_source": args.region_source,
        "segmentation_mask_threshold": args.segmentation_mask_threshold,
        "max_val_samples": args.max_val_samples,
        "region_protocol": args.region_protocol,
        "region_encoder": args.region_encoder,
    })
    from qpsalm_seg.description.demo import DescriptionDemoSession, build_demo

    session = DescriptionDemoSession(config, args.checkpoint, args.split, args.device)
    app = build_demo(session)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
