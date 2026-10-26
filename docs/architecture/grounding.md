# 区域观察

[返回本节目录](README.md) · [文档总目录](../README.md)

将光学原图、二值区域图和上下文裁剪交给视觉语言模型；区域面积、位置等由程序计算。检索知识不能改写这些观察和事实。

区域选择、消息构造与输出检查分别位于 grounding 下，运行时加载入口见 [grounded_runtime.py](../../oa_groundrag/vlm/grounded_runtime.py)。

实现入口：[oa_groundrag/grounding/evidence.py](../../oa_groundrag/grounding/evidence.py)。相关使用入口见[使用说明](../usage/README.md)。
