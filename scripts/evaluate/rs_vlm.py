#!/usr/bin/env python3
"""用途：评价已有 Shared VLM predictions；不运行生成。

命令：python scripts/evaluate/rs_vlm.py --help
输入：VLM 配置与已有 prediction JSONL。
输出：全新自动评价报告。
写入：仅写显式的新评价根；不训练。
所属能力：Grounded Multimodal Understanding evaluation。
"""

import sys

from oa_groundrag.vlm.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint(["evaluate", *sys.argv[1:]]))
