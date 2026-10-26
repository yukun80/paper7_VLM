# 8. 科研结果应该怎样表述

[返回本节目录](README.md) · [文档总目录](../../README.md)

<a id="guide-science"></a>

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

请区分三层：

1. **工程可运行**：代码、合同、路由、测试和受限 smoke 能执行。
2. **自动开发评价**：在 train/val 范围用自动指标观察行为，仍可能没有 Gold 或专家共识。
3. **科学验收**：预注册指标和阈值、冻结协议、独立评价及首次 test 规则共同满足。

因此：

- checkpoint 可加载，不等于模型科学有效。
- 开发集指标改善，不等于正式 Gate 通过。
- 自动引用合法，不等于专业解释正确。
- candidate mask 不等于“确认滑坡”。
- 没有 retrieval Gold 时，不能声称 Recall@K、MRR 或 nDCG 已证明检索有效。
- test/sealed test 不能用于反推阈值、Prompt 或 parser。

当前具体状态不要从本文记忆，请始终查看 [当前进度](<../../status/README.md>)。
