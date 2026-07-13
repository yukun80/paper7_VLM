#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动 Benchmark 交互式分割页面。

用途：在浏览器中选择 val/test 样本、活动模态和自定义指令，调用 checkpoint 分割。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.demo
--config CONFIG --preset qwen_psalm_full --checkpoint CHECKPOINT
--vision-feature-cache CACHE --split val --device cuda
主要输入：benchmark-v2、Qwen vision cache v3 和 v5 checkpoint。
主要输出：本机 Gradio 页面；不会改写 benchmark 或 checkpoint。
"""

from __future__ import annotations

import argparse

from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.demo_app import build_demo
from qpsalm_seg.inference import InferenceSession
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the QPSALM benchmark inference demo.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-feature-cache", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_preset(load_config(args.config), args.preset)
    config = apply_config_overrides(config, {
        "benchmark_dir": args.benchmark_dir,
        "vision_feature_cache": args.vision_feature_cache,
        "modality_dropout": 0.0,
        "num_workers": 0,
    })
    session = InferenceSession(
        config,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    demo = build_demo(session)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
