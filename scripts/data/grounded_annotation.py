#!/usr/bin/env python3
"""用途：运行 Grounded 单专家 annotation 数据流程。

命令：python scripts/data/grounded_annotation.py --help
输入：冻结 train/val Region 资产、prompt 与本地 VLM 配置。
输出：annotation 工作根、已核验 package 或训练 messages。
写入：只写显式的新工作/发布根；不访问 sealed test。
所属能力：Grounded Multimodal Understanding 数据生产。
"""

from oa_groundrag.data.grounded.annotation.cli import (
    TRAIN_WORKFLOW_PATHS,
    _parser,
    entrypoint,
)


if __name__ == "__main__":
    raise SystemExit(entrypoint())
