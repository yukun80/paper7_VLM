#!/usr/bin/env python3
"""用途：构建、验证 Stage 4 Corpus，并运行 Stage 4B 本地 Silver 闭环。

命令：python scripts/stage4_landslide_evidence/run_landslide_evidence.py --help
输入：冻结的 OA-AuxSeg train Corpus、Stage 4A/4B 严格配置和已接受 RS-General Adapter。
输出：新 Corpus 根或可恢复 Silver generation/filter/review 根；validate 只读。
写入：新输出使用原子目录且拒绝覆盖；不修改 Benchmark、模型和既有 outputs。
阶段：Stage 4A Auto Pilot 与 Stage 4B 本地 Silver；GPU generation 由负责人启动。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.phase3.errors import RSGeneralDescError
from oa_groundrag.phase4.errors import Phase4Error
from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.pipeline import build_auto
from oa_groundrag.landslide_evidence.silver_runtime import (
    filter_silver_run,
    generate_silver,
    preflight_silver,
    prepare_review_queue_run,
    validate_silver_outputs,
)
from oa_groundrag.landslide_evidence.validation import validate_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 Landslide Evidence Corpus 与本地 Silver")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-auto", help="构建 train-only deterministic Auto Pilot").add_argument("--config", type=Path, required=True)
    commands.add_parser("validate", help="只读验证正式 Corpus").add_argument("--root", type=Path, required=True)
    silver = commands.add_parser("silver", help="Stage 4B 本地 Silver 严格闭环")
    silver.add_argument(
        "--action",
        choices=("preflight", "generate", "filter", "prepare-review", "validate"),
        required=True,
    )
    silver.add_argument("--config", type=Path, required=True)
    silver.add_argument("--limit", type=int)
    silver.add_argument("--output-root", type=Path)
    silver.add_argument("--review-count", type=int)
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


def _silver(args: argparse.Namespace) -> int:
    if args.action == "preflight":
        if args.limit is not None or args.output_root is not None or args.review_count is not None:
            raise LandslideEvidenceError("ARGUMENT_INVALID", "preflight 不接受 limit/output/review 参数")
        _print(preflight_silver(args.config))
        return 0
    if args.action == "generate":
        if args.review_count is not None:
            raise LandslideEvidenceError("ARGUMENT_INVALID", "generate 不接受 --review-count")
        _print(generate_silver(args.config, limit=args.limit, output_root=args.output_root))
        return 0
    if args.action == "filter":
        if args.limit is not None or args.review_count is not None:
            raise LandslideEvidenceError("ARGUMENT_INVALID", "filter 不接受 limit/review 参数")
        _print(filter_silver_run(args.config, output_root=args.output_root))
        return 0
    if args.action == "prepare-review":
        if args.limit is not None or args.output_root is not None or args.review_count is None:
            raise LandslideEvidenceError("ARGUMENT_REQUIRED", "prepare-review 只接受并要求 --review-count")
        _print(prepare_review_queue_run(args.config, review_count=args.review_count))
        return 0
    if args.limit is not None or args.output_root is not None or args.review_count is not None:
        raise LandslideEvidenceError("ARGUMENT_INVALID", "validate 不接受 limit/output/review 参数")
    _print(validate_silver_outputs(args.config))
    return 0


def entrypoint(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-auto":
            result = build_auto(args.config)
            _print({
                "ok": True, "root": str(result.root), "manifest_sha256": result.manifest_sha256,
                "selected_ids_sha256": result.selected_ids_sha256, "record_count": result.record_count,
                "asset_count": result.asset_count, "asset_bytes": result.asset_bytes,
            })
            return 0
        if args.command == "validate":
            _print(validate_corpus(args.root))
            return 0
        return _silver(args)
    except (LandslideEvidenceError, Phase4Error, RSGeneralDescError) as error:
        _print({"ok": False, "code": error.code, "message": str(error),
                "details": error.details}, stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
