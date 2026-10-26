#!/usr/bin/env python3
"""用途：准备 Grounded 扩展监督和 compact 训练资产。

命令：python scripts/data/grounded_supervision.py --help
输入：冻结 Region Corpus、单专家资产、prompt 与本地 VLM 配置。
输出：扩展 Region、监督 package 与 compact training messages。
写入：只写固定的新数据根；不训练、不访问 sealed test。
所属能力：Grounded Multimodal Understanding 数据生产。
"""

from oa_groundrag.data.grounded.supervision.cli import (
    COMPACT_TRAINING_ROOT,
    MODEL_ASSISTED_WORKFLOW_PATHS,
    _parser,
    entrypoint,
)


if __name__ == "__main__":
    raise SystemExit(entrypoint())
