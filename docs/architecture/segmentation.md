# 空间分割

[返回本节目录](README.md) · [文档总目录](../README.md)

以光学影像作为空间基准，辅助信息按可用性参与融合。输出概率图、二值区域、无目标状态与候选区域，不直接作滑坡确认。

辅助模态的有效范围与当前接入情况以[验收边界](../status/acceptance.md)为准。

实现入口：[oa_groundrag/segmentation/model.py](../../oa_groundrag/segmentation/model.py)。相关使用入口见[使用说明](../usage/README.md)。
