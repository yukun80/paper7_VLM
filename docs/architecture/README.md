# 算法与系统设计

[文档总目录](../README.md)

本目录保留 v3.2 分章设计，并新增负责人批准的隔离 v4 共享分割实验；两者的产物与验收分别记录。

建议先读总体设计，再看各能力的现有实现；训练与评价章节保留批准条件，不表示已经完成。状态仅以[进度目录](../status/README.md)为准。

[冻结原文与版本关系](../archive/README.md)用于追溯；科学约束或阈值发生冲突时不得自行选择或放宽，应先确认新的设计授权。

- [0. 设计目标与不变量](overview.md)
- [1. 模型家族与独立产物链](model-families.md)
- [2. 后端架构与公共合同](backends.md)
- [3. 配置与身份](identity.md)
- [4. Qwen3.5 processor 与 reasoning 合同](processing.md)
- [5. Qwen3.5 LoRA 与单卡资源合同](resources.md)
- [6. 两段训练 curriculum](training.md)
- [7. 评价与发布状态机](evaluation.md)
- [8. Unified Runtime](runtime.md)
- [9. 串行实施与停止门](sequence.md)
- [10. 范围外事项](scope.md)
- [11. 参考](references.md)
- [空间分割：现有实现](segmentation.md)
- [区域观察：现有实现](grounding.md)
- [知识检索与解释：现有实现](retrieval.md)
- [任务组织：现有实现](orchestration.md)

- [数据与训练资料的职责](data.md)
- [后续设计与当前实现的关系](future.md)

- [Geo-GroundRAG v4：共享分割实验](shared-segmentation.md)
