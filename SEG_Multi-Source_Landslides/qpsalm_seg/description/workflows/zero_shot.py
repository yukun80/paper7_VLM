#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D-1 native-Qwen zero-shot workflow.

用途：统一 materialized-image zero-shot baseline 的输出所有权与失败终态。
推荐调用：由 ``qpsalm-segdesc evaluate zero-shot`` 薄入口调用。
输入：Qwen model、Description benchmark、split、device 和样本预算。
输出：eval_report/raw_generations；失败时只发布 failure_report.json。
写入行为：只写 output_dir，不加载 segmentation checkpoint 或 MGRR。
工作流阶段：M5/M6 D-1 baseline orchestration。
"""

from __future__ import annotations

import shutil
from pathlib import Path
import traceback
from typing import Any

import torch

from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)

from ..protocols.io import atomic_write_json


class ZeroShotLaunchError(ValueError):
    """The requested D-1 zero-shot output ownership is unsafe."""


def run_zero_shot_evaluation(
    *,
    model_path: str,
    benchmark: str,
    split: str,
    output_dir: str,
    device_name: str,
    max_samples: int,
    max_new_tokens: int,
    seed: int,
    load_4bit: bool,
    overwrite_output: bool,
) -> dict[str, Any]:
    if int(max_samples) != 64:
        raise ZeroShotLaunchError(
            "当前 D-1 zero-shot 协议固定要求 --max-samples 64"
        )
    output = resolve_project_path(output_dir) or Path(output_dir)
    try:
        validate_output_replacement_safety(output, {
            "model": model_path,
            "benchmark": benchmark,
        })
    except ValueError as exc:
        raise ZeroShotLaunchError(str(exc)) from exc
    if output.exists() and not output.is_dir():
        raise ZeroShotLaunchError(f"zero-shot output-dir 不是目录: {output}")
    if output.is_dir() and any(output.iterdir()) and not overwrite_output:
        raise ZeroShotLaunchError(
            "zero-shot output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    from ..evaluation.zero_shot import evaluate_zero_shot_global_caption

    try:
        return evaluate_zero_shot_global_caption(
            model_path=model_path,
            benchmark=benchmark,
            split=split,
            output_dir=output,
            device=torch.device(device_name),
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            seed=seed,
            load_4bit=load_4bit,
        )
    except BaseException as exc:
        # 成功报告与 raw generation 是一个发布单元；失败时不得留下
        # 可被误读为完整 baseline 的半成品。
        for name in ("eval_report.json", "raw_generations.jsonl"):
            (output / name).unlink(missing_ok=True)
        atomic_write_json(output / "failure_report.json", {
            "protocol": "qpsalm_qwen_zero_shot_failure_v2_no_partial_report",
            "eval_report_published": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
