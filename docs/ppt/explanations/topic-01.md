# 1. 用四个角色建立直觉

[返回本节目录](README.md) · [文档总目录](../../README.md)

<a id="guide-basics"></a>

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

想象有一支遥感判读小组：

| 类比角色 | 它回答什么 | 项目能力 | 稳定代码位置 |
| --- | --- | --- | --- |
| 定位员 | 候选区域在哪里？ | OA-AuxSeg / Spatial Perception | [segmentation](<../../../oa_groundrag/segmentation>) |
| 观察员 | 指定区域在图像中呈现什么？ | Shared MLLM + Grounding | [vlm](<../../../oa_groundrag/vlm>)、[grounding](<../../../oa_groundrag/grounding>) |
| 知识查证员 | 这些观察有哪些专业解释、混淆因素和限制？ | Evidence-Constrained Text RAG | [retrieval](<../../../oa_groundrag/retrieval>) |
| 调度员 | 这次任务需要谁参加、按什么顺序？ | Deterministic Runtime Router | [runtime](<../../../oa_groundrag/runtime>) |

四个角色组成的主链是：

~~~text
在哪里？       → OA-AuxSeg
看到了什么？   → Grounded MLLM
如何谨慎解释？ → Evidence-Constrained RAG
何时调用？     → Deterministic Runtime Router
~~~

这里最重要的词是“**按需**”。普通遥感图像描述不需要先做分割；纯分割不需要加载大模型；
专业知识问答不需要图像；只有候选区域解释才需要把区域观察与知识增强接起来。
