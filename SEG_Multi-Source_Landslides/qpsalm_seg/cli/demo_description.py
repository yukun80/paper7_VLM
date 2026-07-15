#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动 benchmark 区域描述交互界面。

用途：手动选择 M1/M2 样本、region/full/zero mask 和指令，展示 raw generation 与解析结果。
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_segdesc_config(args.config, {"stage": args.stage})
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
