#!/usr/bin/env python3
"""阶段 2 OA-AuxSeg 统一入口。

用途：运行 train/evaluate/infer/smoke/overfit。
命令：python scripts/phase2_oa_auxseg/run_oa_auxseg.py <子命令> --config <JSON>。
输入：只读 oa_auxseg_hdf5_v1 Benchmark、本地 backbone 权重或当前 checkpoint。
输出：训练 checkpoint/报告，或原子 JSONL 与 NPZ 推理结果。
写入：只写配置指定的仓库输出目录，默认拒绝覆盖推理结果。
阶段：OA-GroundRAG Phase 2，不含 Region Grounding、VLM 或 RAG。


python scripts/phase2_oa_auxseg/run_oa_auxseg.py smoke \
  --config configs/phase2_oa_auxseg/small_smoke.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py overfit \
  --config configs/phase2_oa_auxseg/small_overfit.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_proposed_dropout.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_proposed_dropout_balanced.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_optical_only.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/full_proposed_dropout.json


smoke	六种 variant 的工程冒烟测试	每种 1 step	使用真实可用模态
overfit	验证完整模型是否有足够拟合能力	最多 1000	使用全部可用辅助模态，关闭采样/dropout
train proposed_dropout	验证任意辅助模态训练能力	300	uniform 定长 batch + 0.2 modality dropout
train balanced	与 uniform 做同 seed 对照	300	4 positive + 4 empty，不使用 source
train optical_only	训练纯光学消融基线	100	完全不使用辅助模态
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.phase2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
