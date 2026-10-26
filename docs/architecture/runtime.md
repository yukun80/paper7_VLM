# 8. Unified Runtime

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

本章的 4B 运行配置名是设计目标，不是现有发布入口；可用配置见[配置说明](../usage/configuration.md)。

`UnifiedRequest`、`UnifiedResponse` 与 `UnifiedTask` wire schema 保持 v1。模型只在进程/会话
启动时通过显式 YAML 选择：

```text
configs/runtime/inference_v2.yaml   # 冻结 2B
configs/runtime/demo_v1.yaml        # 冻结 2B
configs/runtime/inference_qwen3_5_4b_v1.yaml
configs/runtime/demo_qwen3_5_4b_v1.yaml
```

不得新增 per-request 模型字段，不同时驻留两个模型。lazy provider 只加载所选 backend，
release 后释放该模型。推理不得读取 training/evaluation payload、compact supervision、
HDF5 或 sealed/test 路径。

六任务保持：`VLM_ONLY`、`SEGMENT_ONLY`、`REGION_UNDERSTANDING`、
`SEGMENT_AND_UNDERSTAND`、`KNOWLEDGE_QA`、`REGION_INTERPRETATION`。RAG 只用于后两项中
既有明确需要知识增强的路径，不改变 Router 的确定性计划。
