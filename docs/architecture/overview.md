# 0. 设计目标与不变量

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

OA-GroundRAG 保持单一主链：

```text
OA-AuxSeg
→ Shared RS-Geohazard MLLM
→ Grounded Multimodal Understanding
→ Evidence-Constrained Text RAG
→ Unified Instruction-Routed Runtime
```

本版本只扩展 Shared MLLM 的基础模型选择：保留 Qwen3-VL-2B 的全部冻结资产与
运行入口，新增 Qwen3.5-4B 的独立训练、评价和运行链。它不是第二条算法主线，
也不改变 OA-AuxSeg、Evidence Builder、Text Evidence Bank 或 Router 的科学职责。

以下不变量继续成立：

1. 光学影像是空间边界基准；SAR、InSAR、DEM、多光谱等只能作为显式可选证据。
2. 分割只产生 probability、mask、no-target、candidate region 与程序化区域事实。
3. MLLM 不生成或修改 mask，不把候选区域升级为确认滑坡。
4. Text RAG 只消费用户问题、Programmatic Facts 与 Pass-1 structured observation；
   检索证据不能伪装成当前图像事实。
5. `UnifiedTask` 必须由调用方显式提供，Router 不让 LLM 猜任务。
6. 训练 curriculum 与运行时解耦；运行时只加载所选家族的最终 Grounded Adapter。
7. train/val 可用于工程与预注册 Gate；test/sealed 在首次正式评价前保持盲态。
8. `max_steps` 是预算上限，不是模型有效性、Gate 通过或科学验收的充分条件。
