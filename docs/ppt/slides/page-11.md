# 第11页｜第二章：任务三

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：任务三总览
- 研究状态：基础样本库已构建；指令规范与模板已构建；领域数据集和微调为后续工作

### 页面标题与核心结论

- 页面标题：**任务三已形成基础样本库，专业指令数据和领域训练仍在推进**
- 核心结论：基础样本库已经建立，指令数据正在建设，后续根据模型验证结果继续补充样本和修订指令。

### 上屏文字

1. T3.1 多源样本库已构建，铁路场景样本仍需要继续补充
2. T3.2 指令格式和模板已经确定，Geo-Instruct 数据集正在建设
3. T3.3 领域微调将在指令数据完成质量检查后开展

### 主视觉布局

中央放置三个由左至右的模块：样本库、Geo-Instruct、领域迭代训练；第一个为蓝色实框，
第二个分为蓝色的指令格式和模板已构建、橙色的数据集建设中，第三个为紫色虚线。底部用一条返回箭头
表示验证问题用于补充样本和修订指令。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for T3.1, T3.2, T3.3, and Geo-Instruct. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly three connected areas from left to right: T3.1 典型地质要素解译样本库 已构建; T3.2 with 指令格式和模板 已构建 and Geo-Instruct 数据集 建设中; T3.3 领域微调与迭代训练 下一阶段. Draw one return arrow from 模型验证发现的问题 to the first two areas and label it 补充样本并修订指令. Use sparse dataset, instruction, and training icons. Do not show a completed Geo-Instruct dataset, a sample count, completed domain training, railway field validation, or additional data products.
```

### 准确标签清单

- 第二章｜任务三
- T3.1 典型地质要素解译样本库｜已构建
- 指令格式和模板｜已构建
- T3.2 Geo-Instruct 数据集｜建设中
- T3.3 领域微调与迭代训练｜下一阶段
- 模型验证发现的问题
- 补充样本并修订指令

### 本地参考图或真实素材

- [参考图](<../figure/Figure 2 — Dataset Construction.png>)，仅参考数据关系版式。

### 内部事实边界（不进入 PPT）

- 现有数据集不等同于已完成的 Geo-Instruct 正式数据集。
- 当前主体数据来自开源遥感与滑坡数据，不得写成铁路自采样本库。
- 领域微调尚未完成，验证问题用于后续改进表示方法设计。

---
