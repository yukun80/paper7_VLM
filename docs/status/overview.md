# 当前游标

[返回本节目录](README.md) · [文档总目录](../README.md)

本隔离分支当前执行 Geo-GroundRAG v4 实验重构；本次实施与运行结果见
[共享分割状态](shared-segmentation.md)。下表保留 v3.2 历史基线，不代表本次新模型验收。

| 字段 | 当前值 |
| --- | --- |
| 更新时间 | `2026-10-22`（交付工具人工填写项扩充完成；研究实验结果未改变） |
| program | `OA_GROUNDRAG_V3` |
| 当前工程状态 | `2B_PORTABLE_RUNTIME_REPUBLISHED / QWEN3_5_4B_M7_GRADIENT_EQUIVALENCE_GATE_FAILED` |
| 当前授权任务 | 独立 benchmark_delivery 工具及人工填写项扩充、真实小示例与文档交付已完成；未授权新训练或正式评价 |
| 下一任务 | 见[下一步与授权](next.md)；交付工具完成后不自动启动原有后续实验 |
| 文档整理基线 | `main@766aceb82029812051c7fbdb057cfe5df2d3ccad`；开始时工作树干净 |
| 发行接口 | `oa-groundrag==0.2.0`；`oa-groundrag = oa_groundrag.runtime.cli:main` |
| 总体科学状态 | P0 与能力驱动重构仅为工程完成；Frozen Eval-dev 及其下游开发评价产物已退役，不升级 Gate A/C/D 或系统科学验收 |
| Benchmark test / sealed test | 未评价 / 未访问 |

## 如何继续阅读

- 已有能力与可用范围：[能力表](capabilities.md)。
- 4B 各步骤的实施记录：[模型进展](model-progress.md)。
- 模型、数据和知识库的定位依据：[成果清单](assets.md)与[发布基线](release.md)。
- 退役的开发评价链：[退役记录](retired.md)，不能再当成当前可用结果。
- 正式验收范围：[验收边界](acceptance.md)，运行成功不能替代科学评价。
- 旧运行证据与此次检查：[历史验证](validation.md)、[文档检查](documentation-check.md)。
- 后续运行与研究所需授权：[下一步](next.md)。

本次独立工具交付见[交付状态](benchmark-delivery.md)。原有研究状态没有推进；恢复研究工作时仍须明确新的任务范围。
