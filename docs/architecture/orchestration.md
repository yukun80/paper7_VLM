# 任务组织

[返回本节目录](README.md) · [文档总目录](../README.md)

调用方明确选择六类任务，程序据此决定是否调用分割、视觉理解和检索；模型按需加载。普通区域解释与演示页面的候选选择流程有所不同。

请求、执行和结果说明见[统一推理](../usage/inference.md)，交互说明见[演示页面](../usage/demo.md)。

实现入口：[oa_groundrag/runtime/router.py](../../oa_groundrag/runtime/router.py)。相关使用入口见[使用说明](../usage/README.md)。
