#!/usr/bin/env python3
"""用途：构建并验证 Stage 4A Landslide Evidence Corpus。

命令：python scripts/stage4_landslide_evidence/run_landslide_evidence.py --help
输入：冻结的 OA-AuxSeg train Benchmark 与 Stage 4A 严格配置。
输出：新 Corpus 根；validate 只读。
写入：新输出使用原子目录且拒绝覆盖；不修改 Benchmark、模型和既有 outputs。
阶段：Stage 4A deterministic Auto Pilot。
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
from oa_groundrag.landslide_evidence.validation import validate_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4A Landslide Evidence Corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-auto", help="构建 train-only deterministic Auto Pilot").add_argument("--config", type=Path, required=True)
    commands.add_parser("validate", help="只读验证正式 Corpus").add_argument("--root", type=Path, required=True)
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


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
        raise AssertionError("unreachable")
    except (LandslideEvidenceError, Phase4Error, RSGeneralDescError) as error:
        _print({"ok": False, "code": error.code, "message": str(error),
                "details": error.details}, stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
