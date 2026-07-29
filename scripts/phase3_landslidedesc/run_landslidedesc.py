#!/usr/bin/env python3
"""Phase 2 三库描述与视觉理解多任务 Benchmark 统一入口。

用途：调用库层 audit、build、validate 或 export。
输入：严格 YAML 配置，或已构建的 canonical 根。
输出：JSON 报告、自包含 canonical Benchmark，或 task-aware Qwen messages。
写入：audit/validate 默认只读；build/export 拒绝覆盖并经 staging 原子发布。
所属阶段：算法 Phase 2；仓库实现路径 phase3。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.phase3.cli import entrypoint


if __name__ == "__main__":
    entrypoint()
