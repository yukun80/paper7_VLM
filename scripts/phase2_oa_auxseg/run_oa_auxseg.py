#!/usr/bin/env python3
"""阶段 2 OA-AuxSeg 统一入口。

用途：运行 train/evaluate/infer/smoke/overfit。
命令：python scripts/phase2_oa_auxseg/run_oa_auxseg.py <子命令> --config <JSON>。
输入：只读 oa_auxseg_hdf5_v1 Benchmark、本地 backbone 权重或当前 checkpoint。
输出：训练 checkpoint/报告，或原子 JSONL 与 NPZ 推理结果。
写入：只写配置指定的仓库输出目录，默认拒绝覆盖推理结果。
阶段：OA-GroundRAG Phase 2，不含 Region Grounding、VLM 或 RAG。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.phase2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
