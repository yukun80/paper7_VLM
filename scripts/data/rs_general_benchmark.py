#!/usr/bin/env python3
"""RS-GeneralDesc 三库描述与视觉理解多任务 Benchmark 统一入口。

命令：python scripts/data/rs_general_benchmark.py --help
用途：调用库层 audit、build、repackage、validate 或 export。
输入：严格 YAML 配置，或已构建的 canonical 根。
输出：JSON 报告、自包含 canonical Benchmark，或 task-aware Qwen messages。
写入：audit/validate 不修改 Benchmark；报告、build、repackage 和 export 均拒绝覆盖并原子发布。
所属能力：RS-GeneralDesc 数据生产；curriculum provenance 为 Stage 2。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.data.rs_general.cli import entrypoint


if __name__ == "__main__":
    entrypoint()
