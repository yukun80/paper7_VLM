#!/usr/bin/env python3
"""用途：运行 OA-AuxSeg 只读推理；写入全新 JSONL/NPZ 输出根。

命令：python scripts/infer/oa_auxseg.py --help
输入：OA-AuxSeg 配置、checkpoint 与只读数据根。
输出：全新根内的 predictions.jsonl 与 predictions.npz。
写入：仅写显式的新推理根；不训练。
所属能力：Spatial Perception。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.training.segmentation.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["infer", *sys.argv[1:]]))
