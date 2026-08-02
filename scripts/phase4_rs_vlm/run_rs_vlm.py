#!/usr/bin/env python3
"""用途：运行算法 Phase 3（仓库 phase4）的训练工具与独立 Gate B。

命令：python
  scripts/phase4_rs_vlm/run_rs_vlm.py --help
输入：严格 RS-VLM 配置或 Gate B 协议、canonical Benchmark、冻结 selection。
输出：仅写命令指定的新输出根；preflight 只打印 JSON，Gate B 拒绝覆盖。
写入：拒绝链接、路径逃逸和覆盖已有输出；不修改 Benchmark/source/model。
阶段：算法 Phase 3 / 仓库 phase4，RS-VLM；mask 核心仅供后续阶段接入。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.phase4.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
