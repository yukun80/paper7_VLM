#!/usr/bin/env python3
"""用途：运行算法 Phase 3（仓库 phase4）的 preflight/train/infer/evaluate/smoke。

命令：python
  scripts/phase4_mask_grounded_description/run_mask_grounded_description.py --help
输入：严格 phase4 YAML、已发布 canonical Benchmark、可选显式 checkpoint/predictions。
输出：仅写命令指定的新输出根；preflight 只打印 JSON，不写数据。
写入：拒绝链接、路径逃逸和覆盖已有输出；不修改 Benchmark/source/model。
阶段：算法 Phase 3 / 仓库 phase4，Mask-Grounded VLM Description。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.phase4.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
