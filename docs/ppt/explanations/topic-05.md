# 5. 六类任务不是六套模型

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

| <code>UnifiedTask</code> | 必要输入 | Spatial | Grounding | Shared MLLM | Text RAG | 主要输出 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| <code>VLM_ONLY</code> | 图像 + instruction | 否 | 否 | 是 | 否 | 普通视觉文本 |
| <code>SEGMENT_ONLY</code> | spatial input | 是 | 否 | 否 | 否 | 概率图、mask、候选区域 |
| <code>REGION_UNDERSTANDING</code> | 单图 + user mask | 否 | 是 | 是 | 否 | 区域视觉观察 |
| <code>SEGMENT_AND_UNDERSTAND</code> | spatial input + instruction | 是 | 是 | 是 | 否 | 全局 mask 对应的区域观察 |
| <code>KNOWLEDGE_QA</code> | 纯文本问题 | 否 | 否 | 是 | 是 | 有文本证据约束的知识回答 |
| <code>REGION_INTERPRETATION</code> | 图像/空间输入 + 指定区域来源 + 问题 | 视来源而定 | 是 | 是 | 是 | 区域观察与专业解释 |

注意三点：

- 表里的“是”表示调用某项能力，不表示把所有模型同时放在 GPU。
- <code>REGION_INTERPRETATION</code> 使用 user mask 时不调用 OA-AuxSeg；使用
  OA-AuxSeg candidate 时才调用。
- Router 是确定性控制逻辑，不是论文里的新神经网络，也不做语义自动分类。

真实映射见 [router.py](<../../../oa_groundrag/runtime/router.py>) 和
[contracts.py](<../../../oa_groundrag/runtime/contracts.py>)。
