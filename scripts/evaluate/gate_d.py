#!/usr/bin/env python3
"""用途：运行 Gate D automatic-only 开发评价。

命令：python scripts/evaluate/gate_d.py --help
输入：Gate D 配置、冻结 Pass-2 run 与既有协议资产。
输出：全新 protocol/run/evaluation 根或只读验证结果。
写入：依子命令写入全新 protocol/run/evaluation 根；validate 只读。
所属能力：Knowledge Augmentation evaluation。
"""

from oa_groundrag.evaluation.retrieval.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
