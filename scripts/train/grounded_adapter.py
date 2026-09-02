#!/usr/bin/env python3
"""用途：运行 Mask-Grounded Adapter 训练 curriculum。

命令：先运行 `profile-stage5-resources --config <v4 YAML>`，再运行
`run-stage5-workflow --config <v4 YAML> --stop-after-steps {1,20,100}`。
输入：Mask-Grounded curriculum 配置及已冻结 warm-start/监督资产。
输出：adapter checkpoint、workflow state 与 train/val 报告。
写入：只写配置声明的新训练根；既有根拒绝覆盖。
所属能力：Grounded Multimodal Understanding training。
smoke：同一 profile 独立根严格按 1→20→100 恢复；正式训练使用另一新根。
"""

from oa_groundrag.training.grounding.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
