#!/usr/bin/env python3
"""用途：构建并验证 OA-GroundRAG Evidence-Constrained Text RAG。

命令：python scripts/infer/text_rag.py --help
输入：Stage 6 YAML、只读 PDF、Stage 5 Pass-1/dev 与本地模型权重。
输出：新 Bank、dev retrieval 或 bounded paired GPU artifact；全部拒绝覆盖。
写入：仅写配置指定的新 Stage 6 输出根；不修改 PDF、Stage 5 或 sealed test。
阶段：Stage 6 文本专业知识 RAG；不训练、不运行正式科学 Gate。
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

from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.retrieval.bank import build_bank, validate_bank
from oa_groundrag.retrieval.contracts import load_stage6_config
from oa_groundrag.retrieval.workflow import (
    generate_paired,
    preflight,
    prepare_dev_selection,
    retrieve_dev,
    validate_retrieval,
    validate_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OA-GroundRAG Stage 6 Text RAG")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="显式 Stage 6 构建/开发配置；仓库不再提供已退役 Eval-dev 默认值",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="只读核验来源、dev、Stage 5 best 与 BGE 权重")
    commands.add_parser("build-bank", help="审计 PDF/OCR 并原子构建 Bank/index")
    validate_bank_parser = commands.add_parser("validate-bank", help="重算 Bank 身份")
    validate_bank_parser.add_argument("--root", type=Path)
    commands.add_parser("prepare-dev", help="只读重算冻结 dev selection")
    commands.add_parser("retrieve-dev", help="构建 160 queries 与 80 balanced packets")
    validate_retrieval_parser = commands.add_parser("validate-retrieval", help="重算 retrieval/rank 身份")
    validate_retrieval_parser.add_argument("--root", type=Path)
    generation = commands.add_parser("generate-paired", help="运行 source-balanced 真实 GPU paired Pass-2")
    generation.add_argument("--limit", type=int, required=True)
    generation.add_argument("--output-root", type=Path)
    validate_run_parser = commands.add_parser("validate-run", help="只读重算 paired run")
    validate_run_parser.add_argument("--root", type=Path)
    return parser


def _progress(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_stage6_config(args.config)
    if args.command == "preflight":
        result = preflight(config.config_path)
    elif args.command == "build-bank":
        result = build_bank(config, progress=_progress)
    elif args.command == "validate-bank":
        result = validate_bank(args.root or config.bank_root, config=config, verify_sources=True)
    elif args.command == "prepare-dev":
        selection = prepare_dev_selection(config)
        result = {
            "ok": True,
            "selection_id": selection["selection_id"],
            "record_count": selection["record_count"],
            "source_counts": selection["source_counts"],
            "sealed_test_accessed": False,
        }
    elif args.command == "retrieve-dev":
        result = retrieve_dev(config.config_path)
    elif args.command == "validate-retrieval":
        result = validate_retrieval(args.root or config.retrieval_root, config=config)
    elif args.command == "generate-paired":
        result = generate_paired(
            config.config_path,
            limit=args.limit,
            output_root=args.output_root,
        )
    elif args.command == "validate-run":
        result = validate_run(args.root or config.generation_root, config=config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def entrypoint() -> int:
    try:
        return main()
    except (VLMError, RSGeneralDescError) as error:
        details = getattr(error, "details", {})
        code = getattr(getattr(error, "code", None), "value", "STAGE6_ERROR")
        print(
            json.dumps(
                {"ok": False, "reason_code": code, "message": str(error), "details": details},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
