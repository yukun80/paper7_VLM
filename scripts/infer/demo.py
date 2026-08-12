#!/usr/bin/env python3
"""用途：启动只读 OA-GroundRAG Unified Demo Workbench。

命令：python scripts/infer/demo.py --config configs/runtime/demo_v1.yaml --port 7860
输入：Demo v1 配置及其绑定的只读 Benchmark、Frozen Eval、模型与 Text Bank。
输出：127.0.0.1 上的 Benchmark Browser、Gallery、Task Runner 与 Result Viewer。
写入：仅写独立 Demo root；不修改 annotation、Benchmark、checkpoint 或正式评价资产。
所属能力：Instruction-Routed Unified Inference / Research Demo。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.runtime.demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
