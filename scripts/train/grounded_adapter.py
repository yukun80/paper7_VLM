#!/usr/bin/env python3
"""用途：运行 Mask-Grounded Adapter 训练 curriculum。

命令：python scripts/train/grounded_adapter.py --help
输入：Mask-Grounded curriculum 配置及已冻结 warm-start/监督资产。
输出：adapter checkpoint、workflow state 与 train/val 报告。
写入：只写配置声明的新训练根；既有根拒绝覆盖。
所属能力：Grounded Multimodal Understanding training。
"""

from oa_groundrag.training.grounding.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
