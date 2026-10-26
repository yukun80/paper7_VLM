#!/usr/bin/env python3
"""Spatial Perception / Stage 1 OA-AuxSeg 统一入口。

用途：运行 train/evaluate/infer/smoke/overfit/finalize。
命令：python scripts/train/oa_auxseg.py <子命令> --config <JSON>。
输入：只读 oa_auxseg_hdf5_v1 Benchmark、本地 backbone 权重或当前 checkpoint。
输出：训练 checkpoint/报告、人工定版报告，或原子 JSONL 与 NPZ 推理结果。
写入：只写配置指定的仓库输出目录，finalize 和推理结果默认拒绝覆盖。
阶段：OA-GroundRAG 新路线 Stage 1；不含 Region Grounding、VLM 或 RAG。


python scripts/train/oa_auxseg.py smoke \
  --config configs/segmentation/small_smoke.json

python scripts/train/oa_auxseg.py overfit \
  --config configs/segmentation/small_overfit.json

python scripts/train/oa_auxseg.py train \
  --config configs/segmentation/small_proposed_dropout.json

python scripts/train/oa_auxseg.py train \
  --config configs/segmentation/small_proposed_dropout_balanced.json

python scripts/train/oa_auxseg.py train \
  --config configs/segmentation/small_optical_only.json

python scripts/train/oa_auxseg.py train \
  --config configs/segmentation/full_proposed_dropout.json

python scripts/train/oa_auxseg.py finalize \
  --config configs/segmentation/full_proposed_dropout_b16_nockpt_e100.json \
  --checkpoint outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt \
  --termination-reason project_owner_manual_stop


smoke	六种 variant 的工程冒烟测试	每种 1 step	使用真实可用模态
overfit	验证完整模型是否有足够拟合能力	最多 1000	使用全部可用辅助模态，关闭采样/dropout
train proposed_dropout	验证任意辅助模态训练能力	300	uniform 定长 batch + 0.2 modality dropout
train balanced	与 uniform 做同 seed 对照	300	4 positive + 4 empty，不使用 source
train optical_only	训练纯光学消融基线	100	完全不使用辅助模态
finalize	冻结人工停止训练的 best 权重	0 step	只运行 train/val 前向评价并原子写报告
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.training.segmentation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
