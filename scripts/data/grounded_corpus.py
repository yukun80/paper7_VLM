#!/usr/bin/env python3
"""用途：构建、验证和评价 Grounded Corpus 数据资产。

命令：python scripts/data/grounded_corpus.py --help
输入：严格 Grounded 配置、既有 Benchmark 或已有 prediction。
输出：版本化 Corpus、Eval-dev、annotation queue 或评价报告。
写入：构建命令拒绝覆盖并原子发布；验证命令只读。
所属能力：Grounded Multimodal Understanding 数据生产。
"""

from oa_groundrag.data.grounded.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
