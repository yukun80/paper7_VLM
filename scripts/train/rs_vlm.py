#!/usr/bin/env python3
"""用途：运行 Shared RS-Geohazard MLLM 训练工具与独立 Gate B。

命令：python
  scripts/train/rs_vlm.py --help
输入：严格 RS-VLM 配置或 Gate B 协议、prediction、canonical Benchmark、冻结 selection。
输出：仅写命令指定的新输出根；preflight/媒体查询只打印，Gate B 拒绝覆盖。
写入：拒绝链接、路径逃逸和覆盖已有输出；不修改 Benchmark/source/model。
curriculum：Stage 3 RS-VLM；mask 核心仅供后续 grounded 阶段接入。

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/train/rs_vlm.py gate-b-locate-media \
  --predictions outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1/adapter/predictions.jsonl \
  --line-number 1
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.vlm.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
