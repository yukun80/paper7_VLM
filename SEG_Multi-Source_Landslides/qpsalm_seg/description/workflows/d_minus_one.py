#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D-1 zero-shot/overfit command defaults and final gate publication.

用途：集中 D-1 的固定工程预算与 zero-shot/overfit 双 run 验收编排。
推荐调用：``qpsalm-segdesc evaluate zero-shot``、``train d-minus-one``，最后
``validate d-minus-one``。
输入：CLI token 或两个既有 run 目录。
输出：补全后的 CLI token 或原子发布的当前 D-1 v13 gate。
写入行为：验收时只写显式 gate 路径，不运行模型或训练。
工作流阶段：M5/M6 D-1。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from qpsalm_seg.paths import resolve_project_path

from ..evaluation.d_minus_one import (
    validate_d_minus_one_gate,
    validate_d_minus_one_runs,
    write_d_minus_one_gate,
)


D_MINUS_ONE_TRAIN_FIXED_OPTIONS = {
    "--stage": "overfit",
    "--max-steps": "100",
    "--max-train-samples": "64",
}
D_MINUS_ONE_ZERO_SHOT_FIXED_OPTIONS = {"--max-samples": "64"}


def apply_fixed_options(
    arguments: Sequence[str],
    fixed: dict[str, str],
) -> list[str]:
    """Apply a fixed workflow budget and reject conflicting CLI values."""
    tokens = list(arguments)
    for option, expected in fixed.items():
        observed: list[str] = []
        for index, token in enumerate(tokens):
            if token == option:
                if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                    raise ValueError(f"{option} 缺少值")
                observed.append(tokens[index + 1])
            elif token.startswith(f"{option}="):
                observed.append(token.split("=", 1)[1])
        if any(value != expected for value in observed):
            raise ValueError(
                f"D-1 固定预算禁止覆盖 {option}: "
                f"expected={expected!r} observed={observed!r}"
            )
        if not observed:
            tokens.extend((option, expected))
    return tokens


def d_minus_one_train_arguments(arguments: Sequence[str]) -> list[str]:
    tokens = apply_fixed_options(arguments, D_MINUS_ONE_TRAIN_FIXED_OPTIONS)
    observed_batch_sizes: list[int] = []
    for index, token in enumerate(tokens):
        if token == "--batch-size":
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError("--batch-size 缺少值")
            observed_batch_sizes.append(int(tokens[index + 1]))
        elif token.startswith("--batch-size="):
            observed_batch_sizes.append(int(token.split("=", 1)[1]))
    if not observed_batch_sizes:
        tokens.extend(("--batch-size", "2"))
    elif len(set(observed_batch_sizes)) != 1:
        raise ValueError(
            "D-1 --batch-size 出现冲突值: "
            f"{observed_batch_sizes!r}"
        )
    elif observed_batch_sizes[0] < 2:
        raise ValueError("D-1 24 GiB 工程门禁要求 --batch-size >= 2")
    return tokens


def d_minus_one_zero_shot_arguments(arguments: Sequence[str]) -> list[str]:
    return apply_fixed_options(arguments, D_MINUS_ONE_ZERO_SHOT_FIXED_OPTIONS)


def validate_and_publish_d_minus_one(
    zero_shot_dir: str | Path,
    overfit_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Deep-validate both runs and atomically publish only a self-valid gate."""
    report = validate_d_minus_one_runs(zero_shot_dir, overfit_dir)
    target = resolve_project_path(output) or Path(output)
    if report["errors"]:
        write_d_minus_one_gate(target, report)
        return report
    candidate = target.with_name(f".{target.name}.candidate")
    try:
        write_d_minus_one_gate(candidate, report)
        validate_d_minus_one_gate(candidate)
        candidate.replace(target)
    finally:
        candidate.unlink(missing_ok=True)
    return report
