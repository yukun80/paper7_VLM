# 第02页｜传统单任务模型的三类瓶颈

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：研究背景
- 研究状态：背景页，不作成果声明

### 页面标题与核心结论

- 页面标题：**传统单任务模型难以同时利用多源数据并支持风险判断**
- 核心结论：现有方法往往分别处理不同数据和任务，难以从多源影像直接得到可核查的候选区域说明。

### 上屏文字

1. 光学、地形和 InSAR 数据的分辨率与含义不同，难以直接联合使用
2. 裸岩、开挖边坡和阴影外观相近，容易造成候选区域误判
3. 仅给出类别或 mask，仍不能说明候选区域为何值得关注

### 主视觉布局

左侧放置三个相互分离的问题区，分别表示数据差异、易混淆场景和结果说明不足；右侧用一个
铁路业务需求框接收三条箭头，框内只写多源信息理解、候选区域定位和结果说明。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for technical terms explicitly listed for this page: none. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Show exactly three separated problem areas on the left labeled 多源数据难以联合, 相似地物容易误判, and 结果缺少原因说明. Use small icons for optical imagery, terrain, deformation, bare rock, excavation slope, shadow, category, and mask. On the right, draw one blue panel labeled 铁路多源遥感分析, containing exactly three Chinese items: 多源信息理解, 候选区域定位, 结果说明. Connect each problem area to the right panel with one arrow. Keep the graphic minimal and low density. Do not add neural network layers, equations, performance numbers, generated disaster imagery, or completed capability claims.
```

### 准确标签清单

- 多源数据难以联合
- 相似地物容易误判
- 结果缺少原因说明
- 铁路多源遥感分析
- 多源信息理解｜候选区域定位｜结果说明

---
