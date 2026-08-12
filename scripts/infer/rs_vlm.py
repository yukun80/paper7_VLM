#!/usr/bin/env python3
"""用途：运行 Shared RS-Geohazard MLLM 推理。

命令：python scripts/infer/rs_vlm.py --help
输入：VLM 配置、checkpoint 与 canonical multimodal messages。
输出：全新根内的 prediction JSONL 与 manifest。
写入：只写显式的新 prediction 根；不训练。
所属能力：Grounded Multimodal Understanding。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.vlm.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint(["infer", *sys.argv[1:]]))
