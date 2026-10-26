#!/usr/bin/env python3
"""用途：创建或只读验证固定 Hub revision 的 VLM 模型资产 ledger。

命令：python scripts/data/vlm_model_assets.py {create,verify} --help
输入：本地模型根、backend、Hub repo/commit 与 ledger 路径。
输出：验证后的 identity JSON。
写入：create 只创建全新 ledger，拒绝覆盖；verify 完全只读。所属能力：vlm/data。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from oa_groundrag.vlm.backends.assets import (
    create_model_asset_ledger,
    verify_model_asset_ledger,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM 固定模型资产 ledger")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--model-root", type=Path, required=True)
    create.add_argument("--ledger", type=Path, required=True)
    create.add_argument("--backend", required=True)
    create.add_argument("--repo-id", required=True)
    create.add_argument("--revision", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--ledger", type=Path, required=True)
    verify.add_argument("--model-root", type=Path)
    verify.add_argument("--backend")
    verify.add_argument("--repo-id")
    verify.add_argument("--revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        result = create_model_asset_ledger(
            model_root=arguments.model_root,
            ledger_path=arguments.ledger,
            backend=arguments.backend,
            repo_id=arguments.repo_id,
            revision=arguments.revision,
        )
    else:
        result = verify_model_asset_ledger(
            arguments.ledger,
            expected_backend=arguments.backend,
            expected_repo_id=arguments.repo_id,
            expected_revision=arguments.revision,
            expected_model_root=arguments.model_root,
        )
    print(json.dumps(result.identity_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
