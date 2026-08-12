#!/usr/bin/env python3
"""用途：启动统一推理 CLI；输入/输出/写入边界见 ``oa-groundrag --help``。

命令：python scripts/infer/oa_groundrag.py --help
输入：Unified v2 配置与显式 UnifiedRequest JSON。
输出：原子发布的 response/failure、manifest、ledger 与 sidecar。
写入：仅在显式提供全新 output-root 时写入；dry-run 不写入。
所属能力：Instruction-Routed Unified Inference。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.runtime.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
