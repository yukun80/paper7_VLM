# 第12页｜三套可追溯数据集

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库；T3.2 领域专用指令数据集构建基础
- 研究状态：三套基础数据集已构建，统计单位彼此独立

### 页面标题与核心结论

- 页面标题：**三套数据分别服务于分割、通用理解和区域描述**
- 核心结论：三套数据分别支持像素分割、通用遥感理解和候选区域描述，统计单位不同，不能直接相加。

### 上屏文字

1. OA-AuxSeg Benchmark 包含 53,645 Samples，用于训练和检查像素分割
2. RS-GeneralDesc Benchmark 包含 274,693 Records，用于训练七类通用遥感理解
3. Compact Grounded Supervision 包含 6,974 条区域监督消息，用于候选区域描述

### 主视觉布局

横向并列三个等宽数据集卡片，每张卡片包含一张抽象缩略图、一个大号数字、独立的统计单位和
对应能力。卡片之间用细虚线表示后续配合，但不绘制相加符号或总量。底部标注统计单位不同和不能直接相加。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for OA-AuxSeg Benchmark, RS-GeneralDesc Benchmark, Compact Grounded Supervision, Samples, Records, and mask. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Arrange exactly three equal data cards from left to right. The first card is OA-AuxSeg Benchmark with 53,645 Samples and the Chinese purpose 像素分割训练. The second is RS-GeneralDesc Benchmark with 274,693 Records and the Chinese purpose 七类通用遥感理解. The third is Compact Grounded Supervision with 6,974 条区域监督消息 and the Chinese purpose 候选区域描述. Add the footer 统计单位不同 不能直接相加. Use identical card structures and low information density. Do not add a total, Gold labels, expert consensus, completed Geo-Instruct claims, or any new number.
```

### 准确标签清单

- OA-AuxSeg Benchmark｜53,645 Samples
- RS-GeneralDesc Benchmark｜274,693 Records
- Compact Grounded Supervision｜6,974 条区域监督消息
- 像素分割训练｜七类通用遥感理解｜候选区域描述
- 统计单位不同｜不能直接相加

### 本地参考图或真实素材

- [参考图](<../figure/Figure 2 — Dataset Construction.png>)
- 实际数据身份与统计以 [当前进度](../../status/README.md) 中绑定的 manifest、ledger 和 schema 为准。

### 内部事实边界（不进入 PPT）

- 53,645 的单位是 Samples，274,693 的单位是 Records，6,974 的单位是 Structured Region Messages。
- 6,974 条 compact supervision 不是人工 Gold，也不是专家共识标签。
- 本页不把三套数据称为已完成的铁路专属数据集。

---
