#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6 interactive demo workflow.

用途：构建 source-bound demo session 并以单并发启动 Gradio。
推荐调用：由 ``qpsalm_seg.cli.demo_description`` 薄入口调用。
输入：config v2、checkpoint、split、device 和监听地址。
输出：本地 Gradio 页面。
写入行为：不修改 benchmark、cache 或 checkpoint。
工作流阶段：M6 manual inspection orchestration。
"""

from __future__ import annotations

from ..protocols.config import SegDescConfig


def run_description_demo(
    config: SegDescConfig,
    *,
    checkpoint: str,
    split: str,
    device_name: str,
    host: str,
    port: int,
) -> None:
    from ..evaluation.demo import DescriptionDemoSession, build_demo

    session = DescriptionDemoSession(config, checkpoint, split, device_name)
    app = build_demo(session)
    app.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=False,
    )
