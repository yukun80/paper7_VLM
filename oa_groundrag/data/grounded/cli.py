#!/usr/bin/env python3
"""用途：构建、验证和评价 Grounded Corpus 与 OA-GroundedEval 资产。

命令：python scripts/data/grounded_corpus.py --help
输入：冻结 OA-AuxSeg Benchmark 的人工 GT mask、v2 配置及可选人工 annotation/prediction。
输出：全新 Region Corpus、OA-GroundedEval-dev、message、annotation 或开发评价根。
写入：所有输出原子发布并拒绝覆盖；不修改 Benchmark、checkpoint、Gate B 或既有 Stage 4A。
阶段：Stage 4 Mask-Grounded Region Corpus 与 OA-GroundedEval-dev。
GPU：全部命令 CPU-only；render-messages 不调用 processor/model，evaluate-dev 不生成 prediction。
split：Corpus 只读 train shard；Eval-dev 只读 val shard；任何 test 配置或 test shard 均拒绝。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.data.grounded.annotation.queue import (
    export_annotation_queue,
    validate_annotations,
)
from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.grounded.grounded_eval import build_eval_dev
from oa_groundrag.data.grounded.workflow import build_auto, build_region_corpus
from oa_groundrag.data.grounded.region_validation import (
    validate_eval_dev,
    validate_region_corpus,
)
from oa_groundrag.data.grounded.validation import validate_corpus
from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.evaluation.grounding.observations import evaluate_dev
from oa_groundrag.grounding.messages import render_mask_grounded_region_messages


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded Corpus 数据与评价入口")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("build-pilot", help="构建 train-only deterministic pilot").add_argument(
        "--config", type=Path, required=True
    )
    commands.add_parser("validate-pilot", help="严格验证 pilot Corpus").add_argument(
        "--root", type=Path, required=True
    )
    commands.add_parser("build-region-corpus", help="构建 train-only Region Corpus").add_argument(
        "--config", type=Path, required=True
    )
    commands.add_parser("validate-region-corpus", help="严格验证 Region Corpus").add_argument(
        "--root", type=Path, required=True
    )
    build_eval = commands.add_parser("build-eval-dev", help="构建 val-only OA-GroundedEval-dev")
    build_eval.add_argument("--config", type=Path, required=True)
    validate_eval = commands.add_parser("validate-eval-dev", help="严格验证 OA-GroundedEval-dev")
    validate_eval.add_argument("--root", type=Path, required=True)
    validate_eval.add_argument("--train-corpus-root", type=Path, required=True)
    export = commands.add_parser("export-annotation-queue", help="导出无答案人工标注队列")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    annotations = commands.add_parser("validate-annotations", help="导入并验证人工 annotation")
    annotations.add_argument("--asset-root", type=Path, required=True)
    annotations.add_argument("--annotations", type=Path, required=True)
    annotations.add_argument("--output-root", type=Path, required=True)
    render = commands.add_parser("render-messages", help="渲染 v2 messages，不调用模型")
    render.add_argument("--root", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument(
        "--representation-mode",
        choices=(
            "full_only", "crop_only", "full_plus_mask", "full_plus_mask_plus_crop",
            "overlay_audit_baseline",
        ),
    )
    render.add_argument("--allow-audit-only", action="store_true")
    evaluate = commands.add_parser("evaluate-dev", help="评价已有 prediction；不运行生成")
    evaluate.add_argument("--eval-root", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--annotations-root", type=Path)
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


def _build_result(value: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "root": str(value.root),
        "manifest_sha256": value.manifest_sha256,
        "records_sha256": value.records_sha256,
        "ledger_sha256": value.ledger_sha256,
        "record_count": value.record_count,
        "asset_count": value.asset_count,
        "asset_bytes": value.asset_bytes,
    }


def entrypoint(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-pilot":
            result = build_auto(args.config)
            _print({
                "ok": True,
                "root": str(result.root),
                "manifest_sha256": result.manifest_sha256,
                "selected_ids_sha256": result.selected_ids_sha256,
                "record_count": result.record_count,
                "asset_count": result.asset_count,
                "asset_bytes": result.asset_bytes,
            })
        elif args.command == "validate-pilot":
            _print(validate_corpus(args.root, verify_source=True))
        elif args.command == "build-region-corpus":
            _print(_build_result(build_region_corpus(args.config)))
        elif args.command == "validate-region-corpus":
            _print(validate_region_corpus(args.root, verify_source=True))
        elif args.command == "build-eval-dev":
            _print(_build_result(build_eval_dev(args.config)))
        elif args.command == "validate-eval-dev":
            _print(validate_eval_dev(
                args.root,
                train_corpus_root=args.train_corpus_root,
                verify_source=True,
            ))
        elif args.command == "export-annotation-queue":
            _print(export_annotation_queue(args.root, args.output_root))
        elif args.command == "validate-annotations":
            _print(validate_annotations(args.asset_root, args.annotations, args.output_root))
        elif args.command == "render-messages":
            _print(render_mask_grounded_region_messages(
                asset_root=args.root,
                output_root=args.output_root,
                representation_mode=args.representation_mode,
                allow_audit_only=args.allow_audit_only,
            ))
        elif args.command == "evaluate-dev":
            _print(evaluate_dev(
                eval_root=args.eval_root,
                predictions_path=args.predictions,
                output_root=args.output_root,
                annotations_root=args.annotations_root,
            ))
        else:
            raise AssertionError("unreachable")
        return 0
    except (LandslideEvidenceError, VLMError, RSGeneralDescError) as error:
        _print(
            {
                "ok": False,
                "code": error.code,
                "message": str(error),
                "details": error.details,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
