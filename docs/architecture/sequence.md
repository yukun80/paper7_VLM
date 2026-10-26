# 9. 串行实施与停止门

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

**设计与现状差异：** 原设计中的 resume 步骤属于冻结记录；当前训练入口只允许从 step 0 写入全新输出目录，不支持恢复训练。4B 后续资源检查仍未启动，不能直接执行旧步骤；继续前须重新确认方案。见[使用说明](../usage/commands.md)与[下一步](../status/next.md)。

实施固定为 M0–M10：设计与基线、公共 backend、2B 迁移、4B processor/资产、4B 模型与
资源门、RS-General 训练、Gate B、Grounded 训练、candidate runtime、100 条验证、默认发布。

每个模块必须先通过本模块测试和所有已完成模块回归，才能进入下一模块。M7 v3.2 的
内部串行顺序为 R0 保护/画像、R1 等价 loss 与合同、R2 allocator 资源门、R3 正式训练/
retention；R3 完成后停在 M8 前。任何以下条件
触发 fail-closed 停止并在 [当前进度](../status/README.md) 留证：

- protected asset 身份漂移或输出根已存在；
- schema/backend/template/ledger/topology 不一致；
- CUDA OOM、峰值超过可用单卡容量或真实训练 smoke 失败；
- Gate B 任一预注册判据失败；
- Grounded 任一零容忍指标失败；
- 需要 QLoRA、缩减输入、可选 kernel 或 hybrid LoRA 才能继续。

停止后不得自行变更实验设计；任何替代方案必须重新 grill 并形成新的负责人授权。
