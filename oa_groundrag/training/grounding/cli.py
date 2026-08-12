#!/usr/bin/env python3
"""用途：运行 Stage 5 Mask-Grounded Region Adapter 的完整可恢复闭环。

命令：run-stage5-workflow --config <Stage5 YAML>。
输入：6,974 条 compact train、RS-General Benchmark/Adapter、340 条 OA-GroundedEval-dev。
输出：Base/RS-General/Region 三组 GT-mask 报告、warm-start checkpoint 与 retention 报告。
写入：只写配置指定的新 Stage 5 根；合法阶段复用，部分或非法已有根拒绝覆盖。
阶段：Stage 5 Mask-Grounded Baseline；不是 Gate F、正式验收或 sealed-test 评价。
运行：使用本地 Qwen3-VL-2B 与 GPU；不访问 API、RAG、predicted mask、val replay 或 test。
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
from oa_groundrag.training.grounding.workflow import run_stage5_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5 Mask-Grounded Adapter 固定入口")
    commands = parser.add_subparsers(dest="command", required=True)
    workflow = commands.add_parser(
        "run-stage5-workflow",
        help="按 preflight→Base→RS-General→train/resume→Region→retention 顺序运行",
    )
    workflow.add_argument("--config", type=Path, required=True)
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream, flush=True)


def entrypoint(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command != "run-stage5-workflow":
            raise AssertionError("unreachable")
        result = run_stage5_workflow(
            arguments.config,
            progress_callback=lambda value: _print(value, stream=sys.stderr),
        )
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
