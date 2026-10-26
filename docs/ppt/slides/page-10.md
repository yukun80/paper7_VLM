# 第10页｜当前风险分析信息与后续集成

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.3 风险评估方法构建与集成
- 研究状态：空间证据前端已构建；区域观察待优化验证；正式风险集成为下一阶段

### 页面标题与核心结论

- 页面标题：**当前系统提供候选区域及可核查信息，正式风险等级留待下一阶段**
- 核心结论：现阶段先输出候选区域的位置、面积和形状，后续加入线路距离、暴露对象和脆弱性后再评估风险等级。

### 上屏文字

1. 当前可提供 mask、候选区域以及自动计算的位置、面积和形状信息
2. 候选区域的 4B 结构化描述仍需完成区域训练和结果验证
3. 加入线路距离、暴露对象和脆弱性后，再开展正式风险等级评估

### 主视觉布局

上方为当前处理步骤：光学影像、OA-AuxSeg、mask 与候选区域、自动计算信息，使用蓝色实线；中部的
结构化区域观察使用橙色框。下方为后续风险评估步骤：线路距离、暴露对象、脆弱性和时序信息汇入风险
评估模块，全部使用紫色虚线；最右端只显示基于可核查信息生成的风险说明，不展示示例等级。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for OA-AuxSeg and mask. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw two horizontal processing rows. The upper row is labeled 当前可提供的信息 and uses blue arrows through exactly four areas: 光学影像, OA-AuxSeg, mask 与候选区域, 自动计算的位置面积形状. Connect it to one orange area labeled 结构化区域观察 待优化验证. The lower row is labeled 下一阶段风险评估 and uses purple dashed arrows from 线路距离, 暴露对象, 脆弱性, and 时序信息 to 风险评估方法, then to 基于可核查信息生成的风险说明. Add one Chinese note 当前不输出正式风险等级. Keep the graphic sparse. Do not generate a risk map, numeric risk score, railway threat conclusion, field result, or additional evidence source.
```

### 准确标签清单

- 当前可提供的信息
- 光学影像｜OA-AuxSeg｜mask 与候选区域｜位置面积形状
- 结构化区域观察｜待优化验证
- 下一阶段风险评估
- 线路距离｜暴露对象｜脆弱性｜时序证据
- 风险评估方法构建与集成｜基于可核查信息生成的风险说明
- 当前不输出正式风险等级

### 本地参考图或真实素材

- [参考图](<../figure/Figure 1 — Overall Framework.png>)，仅参考当前空间证据链。
- 本页风险集成部分为项目任务映射图，不使用虚构铁路数据。

### 内部事实边界（不进入 PPT）

- 当前运行时不得直接输出正式 `risk_level`、稳定性或实际威胁结论。
- 线路距离、暴露度和脆弱性尚未形成当前系统的可验收集成模块。
- 风险说明在本页是下一阶段交付方向，不是现有模型输出样例。

---
