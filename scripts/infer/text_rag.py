#!/usr/bin/env python3
"""用途：运行 Text Evidence Bank 检索与 Evidence-Constrained Pass-2。

命令：python scripts/infer/text_rag.py --help
输入：retrieval 配置、已有 Bank、query 或 Pass-1 observation/facts。
输出：retrieval packet、验证报告或 Pass-2 prediction。
写入：仅各子命令显式的新输出根；不修改 Bank、VLM observation 或 facts。
所属能力：Knowledge Augmentation。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.retrieval.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
