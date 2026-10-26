# 第06页｜以光学影像为基准的多模态融合

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.1 多模态融合模型架构设计
- 研究状态：OA-AuxSeg 代码、Benchmark 与定版权重已构建；Gate A 未执行

### 页面标题与核心结论

- 页面标题：**以光学影像确定位置，其他数据仅在有效时参与分析**
- 核心结论：光学或多光谱影像负责确定位置和边界，DEM、坡度和编码 InSAR 仅在数据有效时参与融合。

### 上屏文字

1. 光学或多光谱影像负责确定目标位置，并作为输出位置和边界的基准
2. DEM、坡度和编码 InSAR 通过有效性检查后参与分析
3. SAR 目前尚未接入 OA-AuxSeg 的实际输入范围

### 主视觉布局

保留一条从左到右贯通的蓝色光学处理流程；三类辅助数据从下方经过有效性检查后接入三个简化的多尺度
融合节点；右侧只展示概率图、mask、无目标状态和候选区域四类空间输出。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for OA-AuxSeg, DEM, InSAR, SAR, and mask. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Use one continuous blue optical processing path from left to right: 光学或多光谱输入, 光学特征提取网络, 多尺度质量检查与融合, 空间解码器. Below it, draw exactly three optional inputs labeled DEM, 坡度, and 编码 InSAR. Each optional input must pass through 有效性与覆盖度检查 before joining the fusion area. On the right show exactly four outputs: 概率图, 二值 mask, 无目标状态, 候选区域. Add a small Chinese note 当前未接入 SAR. Use minimal feature icons and large labels. Do not add formulas, detailed network layers, formal acceptance claims, or unspecified sensors.
```

### 准确标签清单

- 光学／多光谱输入
- DEM｜坡度｜编码 InSAR
- 有效性与覆盖度检查
- 光学特征提取网络
- 多尺度检查与融合
- 概率图｜二值 mask｜无目标状态｜候选区域

### 本地参考图或真实素材

- [参考图](<../figure/Figure 3 — Optical-Anchored Spatial Perception.png>)

### 内部事实边界（不进入 PPT）

- OA-AuxSeg 当前辅助输入配置只接受 `dem / insar_velocity / slope`。
- 编码 InSAR 的单位和正负方向未确认时，只能作为图像输入。
- 53,645 条 Benchmark 和定版权重是现有工程成果，不等于 Gate A 通过。

---
