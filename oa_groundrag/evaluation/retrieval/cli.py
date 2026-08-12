#!/usr/bin/env python3
"""用途：运行 Stage 6 Gate D automatic-only 开发评价。

命令：python scripts/evaluate/gate_d.py --help
输入：严格 Gate D 配置、已发布 Stage 6 retrieval 与 5-pair engineering smoke。
输出：冻结 protocol、25-pair GPU run、automatic-only evaluation；均拒绝覆盖。
写入：只写配置声明的新 Stage 6 输出根，不修改知识源、Stage 5 或 sealed test。
阶段：Gate D development-only；不生成专家结论、不训练、不声称科学通过。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.evaluation.retrieval.gate_d import (
    evaluate_gate_d,
    generate_gate_d_pairs,
    prepare_gate_d_protocol,
    validate_gate_d,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 6 Gate D automatic-only development")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/retrieval/gate_d_dev_v1.yaml",
        help="Gate D 严格配置",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="冻结排除 smoke 后的 25-record protocol 与 prompt audit")
    commands.add_parser("generate", help="在 CUDA 上运行 25 个 no-RAG/text-RAG pairs")
    commands.add_parser("evaluate", help="发布不含质量胜负结论的 automatic-only report")
    commands.add_parser("validate", help="只读重算 protocol、run 与 evaluation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_gate_d_protocol(args.config)
    elif args.command == "generate":
        result = generate_gate_d_pairs(args.config)
    elif args.command == "evaluate":
        result = evaluate_gate_d(args.config)
    elif args.command == "validate":
        result = validate_gate_d(args.config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def entrypoint() -> int:
    try:
        return main()
    except (VLMError, RSGeneralDescError) as error:
        code = getattr(getattr(error, "code", None), "value", "GATE_D_DEV_ERROR")
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason_code": code,
                    "message": str(error),
                    "details": getattr(error, "details", {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
