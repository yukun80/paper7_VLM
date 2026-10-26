#!/usr/bin/env python3
"""用途：严格重载并只读评价 OA-AuxSeg checkpoint。

命令：python scripts/evaluate/oa_auxseg.py --help
输入：OA-AuxSeg 配置、checkpoint 与只读 train/val 数据。
输出：全新评价报告。
写入：仅写显式的新评价输出；不训练、不访问 sealed test。
所属能力：Spatial Perception evaluation。
"""

import sys

from oa_groundrag.training.segmentation.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
