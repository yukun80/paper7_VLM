#!/usr/bin/env python3
"""用途：运行 Mask-Grounded Region Adapter train-only 可恢复闭环。

命令：run-stage5-workflow --config <Stage5 YAML>。
输入：6,974 条 compact train、RS-General Benchmark/Adapter。
输出：warm-start checkpoint、train/monitor 与 RS-General retention 报告。
写入：只写配置指定的全新 train-only 根；合法阶段复用，非法已有根拒绝覆盖。
阶段：Mask-Grounded training curriculum；不是正式评价或科学验收。
运行：由配置显式选择本地 VLM backend 与 GPU；不访问 OA-GroundedEval、RAG 或 test。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.training.grounding.resource_profile import (
    profile_stage5_resource_schedule,
)
from oa_groundrag.training.grounding.loss_parity import (
    run_qwen35_loss_parity,
)
from oa_groundrag.training.grounding.workflow import run_stage5_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask-Grounded Adapter train-only 入口")
    commands = parser.add_subparsers(dest="command", required=True)
    workflow = commands.add_parser(
        "run-stage5-workflow",
        help="按 preflight→retention probes→全新训练→retention 顺序运行",
    )
    workflow.add_argument("--config", type=Path, required=True)
    workflow.add_argument(
        "--stop-after-steps",
        type=int,
        choices=(1, 20, 100),
        help="仅用于 v4 按顺序执行 1/20/100-step bounded CUDA resource gate",
    )
    profile = commands.add_parser(
        "profile-stage5-resources",
        help="CPU 回放 v4 两 epoch/16,000 条序列并冻结最坏样本",
    )
    profile.add_argument("--config", type=Path, required=True)
    parity = commands.add_parser(
        "verify-qwen35-loss-parity",
        help="真实 CUDA 比较 full/projected loss 与 64 个 LoRA gradients",
    )
    parity.add_argument("--config", type=Path, required=True)
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream, flush=True)


def entrypoint(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "profile-stage5-resources":
            result = profile_stage5_resource_schedule(
                arguments.config,
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        elif arguments.command == "verify-qwen35-loss-parity":
            result = run_qwen35_loss_parity(arguments.config)
        elif arguments.command == "run-stage5-workflow":
            result = run_stage5_workflow(
                arguments.config,
                stop_after_steps=arguments.stop_after_steps,
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        else:
            raise AssertionError("unreachable")
        _print(result)
        return 0
    except (LandslideEvidenceError, RSGeneralDescError, VLMError) as error:
        code = getattr(error.code, "value", error.code)
        _print(
            {
                "ok": False,
                "code": code,
                "message": str(error),
                "details": dict(error.details),
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
