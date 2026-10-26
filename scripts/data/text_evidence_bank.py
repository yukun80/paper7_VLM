#!/usr/bin/env python3
"""用途：构建或验证 Text Evidence Bank 与检索索引。

命令：python scripts/data/text_evidence_bank.py --help
输入：strict retrieval 配置、知识文档或已有 observation/query。
输出：Text Evidence Bank、retrieval packet 或 Pass-2 prediction。
写入：构建命令拒绝覆盖；验证命令只读。
所属能力：Knowledge Augmentation 数据生产。
"""

from oa_groundrag.retrieval.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
