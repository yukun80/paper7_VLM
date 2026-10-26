#!/usr/bin/env python3
"""用途：运行 OA-GroundRAG Instruction-Routed Unified Inference。

命令：oa-groundrag --config <YAML> --request <JSON> --output-root <fresh-root>。
输入：显式 UnifiedTask、已有 OA-AuxSeg/Stage 5/Text Bank 合同及本地 train/val 或用户资产。
输出：原子发布 request/response 或 failure、manifest、SHA ledger 与按需 sidecar。
写入：仅写调用方给出的全新 output-root；--dry-run 完全不写入。
阶段：OA-GroundRAG v3 P0 统一推理；不训练、不评价、不访问 sealed test。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.runtime.config import build_unified_runtime, load_unified_config
from oa_groundrag.runtime.contracts import (
    UnifiedInferenceError,
    UnifiedRequest,
    reject_test_or_sealed_path,
)
from oa_groundrag.runtime.router import CapabilityRouter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OA-GroundRAG v3 P0 显式任务统一推理",
    )
    parser.add_argument("--config", type=Path, required=True, help="Unified v2 严格 YAML")
    parser.add_argument("--request", type=Path, required=True, help="UnifiedRequest JSON")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-root", type=Path, help="必须不存在的新输出根")
    destination.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置/请求并打印 ExecutionPlan；不构造 provider 或写文件",
    )
    return parser


def _print(value: Any, *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        file=stream,
        flush=True,
    )


def _execute(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reject_test_or_sealed_path(args.request, label="request JSON")
    config = load_unified_config(args.config)
    request = UnifiedRequest.from_dict(read_json(args.request))
    request.validate_paths()
    if args.dry_run:
        plan = CapabilityRouter().route(request)
        _print({
            "ok": True,
            "dry_run": True,
            "request_id": request.request_id,
            "task": request.task.value,
            "execution_plan": plan.to_dict(),
            "providers_constructed": False,
            "artifacts_read": False,
            "output_written": False,
        })
        return 0
    assert args.output_root is not None
    output_root = reject_test_or_sealed_path(args.output_root, label="output_root")
    runtime = build_unified_runtime(config)
    response = runtime.infer(request, output_root=output_root)
    _print(response.to_dict())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _execute(argv)
    except UnifiedInferenceError as error:
        _print({"ok": False, **error.to_dict()}, stream=sys.stderr)
        return 2
    except (VLMError, RSGeneralDescError) as error:
        _print({
            "ok": False,
            "reason_code": getattr(getattr(error, "code", None), "value", "RUNTIME_ERROR"),
            "message": str(error),
            "details": dict(getattr(error, "details", {})),
            "cause_type": type(error).__name__,
        }, stream=sys.stderr)
        return 2
    except Exception as error:
        _print({
            "ok": False,
            "reason_code": "RUNTIME_ERROR",
            "message": str(error),
            "cause_type": type(error).__name__,
        }, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
