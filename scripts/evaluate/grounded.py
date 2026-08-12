#!/usr/bin/env python3
"""用途：评价已有 grounded predictions；不运行模型生成。

命令：python scripts/evaluate/grounded.py --help
输入：OA-GroundedEval-dev 与已有 grounded prediction。
输出：全新评价报告与 manifest。
写入：只写显式的新评价根；不训练、不访问 sealed test。
所属能力：Grounded Evidence evaluation。
"""

import sys

from oa_groundrag.data.grounded.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint(["evaluate-dev", *sys.argv[1:]]))
