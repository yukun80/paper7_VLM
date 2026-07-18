#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified thin entrypoint for M3-M7 SegDesc workflows.

用途：统一 cache、train、evaluate 和 validation 命令；本文件不包含算法实现。
推荐命令：``qpsalm-segdesc --help`` 或 ``python -m qpsalm_seg.cli.segdesc``。
输入/输出/写入行为：完全由被选中的 workflow 定义。
工作流阶段：M3-M7 command boundary。
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence


Command = Callable[[Sequence[str] | None], None]


def _help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qpsalm-segdesc",
        description="Unified SegDesc M3-M7 command line",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("cache", "train", "evaluate", "validate"),
    )
    parser.add_argument("action", nargs="?")
    return parser


def _resolve(tokens: list[str]) -> tuple[Command, list[str]]:
    if not tokens or tokens[0] in {"-h", "--help"}:
        _help_parser().print_help()
        raise SystemExit(0)
    command = tokens.pop(0)
    if command == "cache":
        if not tokens:
            raise SystemExit("cache 需要 action: build|migrate|verify")
        action = tokens.pop(0)
        if action == "migrate":
            from qpsalm_seg.description.workflows.cache_migration import main

            return main, tokens
        if action in {"build", "verify"}:
            from qpsalm_seg.description.workflows.cache_build import main

            if action == "verify" and "--verify-only" not in tokens:
                tokens.append("--verify-only")
            return main, tokens
        raise SystemExit(f"未知 cache action={action!r}; 允许 build|migrate|verify")
    if command == "train":
        if tokens and tokens[0] == "joint":
            tokens.pop(0)
            from qpsalm_seg.cli.train_segdesc_joint import main

            return main, tokens
        if tokens and tokens[0] == "d-minus-one":
            tokens.pop(0)
            from qpsalm_seg.description.workflows.d_minus_one import (
                d_minus_one_train_arguments,
            )

            tokens = d_minus_one_train_arguments(tokens)
        from qpsalm_seg.cli.train_description import main

        return main, tokens
    if command == "evaluate":
        if tokens and tokens[0] == "zero-shot":
            tokens.pop(0)
            from qpsalm_seg.description.workflows.d_minus_one import (
                d_minus_one_zero_shot_arguments,
            )

            tokens = d_minus_one_zero_shot_arguments(tokens)
            from qpsalm_seg.cli.eval_description_zero_shot import main

            return main, tokens
        from qpsalm_seg.cli.eval_description import main

        return main, tokens
    if command == "validate":
        if not tokens:
            raise SystemExit(
                "validate 需要 action: artifacts|d-minus-one|m4|d4|m6|retention"
            )
        action = tokens.pop(0)
        if action == "artifacts":
            from qpsalm_seg.description.workflows.artifact_readiness import main
        elif action == "d-minus-one":
            from qpsalm_seg.cli.validate_d_minus_one import main
        elif action == "m4":
            from qpsalm_seg.cli.validate_m4_region_encoder_suite import main
        elif action == "d4":
            from qpsalm_seg.cli.validate_d4_curriculum import main
        elif action == "m6":
            from qpsalm_seg.cli.validate_m6_acceptance import main
        elif action == "retention":
            from qpsalm_seg.cli.eval_segdesc_retention import main
        else:
            raise SystemExit(
                f"未知 validate action={action!r}; "
                "允许 artifacts|d-minus-one|m4|d4|m6|retention"
            )
        return main, tokens
    raise SystemExit(f"未知 command={command!r}; 允许 cache|train|evaluate|validate")


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        command, arguments = _resolve(tokens)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    command(arguments)


if __name__ == "__main__":
    main()
