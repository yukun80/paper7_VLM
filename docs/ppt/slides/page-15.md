# 第15页｜Geo-Instruct 指令生成与质量检查

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T3.2 领域专用指令数据集构建
- 研究状态：指令规范与模板原型已构建；Geo-Instruct 正式数据集建设中

### 页面标题与核心结论

- 页面标题：**指令生成只回答样本能够证明的内容，并保留人工复核**
- 核心结论：每条指令都对应具体影像、mask 和位置数据，自动生成的回答经过三级检查后才能进入数据集。

### 上屏文字

1. 光学影像、mask 以及位置和面积信息共同限定模型可以回答的范围
2. 指令按照判断、类别、依据、图像质量、位置和建议组织回答
3. 回答只有经过规则检查、多源一致性检查和人工抽查后才能进入正式数据集

### 主视觉布局

中央采用由左至右的三级质量检查：样本包进入指令模板，生成结构化回答草案，再依次通过三项检查，
末端输出带 source ID、template ID 和 QC 状态的 Geo-Instruct Record。前半段为蓝色实线，正式
数据集输出框使用橙色边框并标注建设中。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for Geo-Instruct, Record, source ID, template ID, QC, and mask. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly four areas from left to right: 同源样本包, 指令模板 已构建, 结构化回答草案, Geo-Instruct Record 建设中. The first area contains 光学影像, mask, 辅助信息, 位置面积形状. The template area shows six Chinese fields connected by arrows: 判断, 类别, 依据, 图像质量, 位置, 建议. Between the draft and final record, draw exactly three checks labeled 规则检查, 多源一致性检查, 人工抽查. The final record contains source ID, template ID, and QC 状态. Keep the graphic sparse. Do not add a dataset count, completed status, expert consensus, automatic acceptance, or invented instruction examples.
```

### 准确标签清单

- 同源样本包｜光学影像／mask／辅助信息／位置面积形状
- 指令模板｜已构建
- 判断｜类别｜依据｜图像质量｜位置｜建议
- 规则检查｜多源一致性检查｜人工抽查
- Geo-Instruct Record｜建设中
- source ID｜template ID｜QC 状态

### 本地参考图或真实素材

- [参考图](<../figure/Figure 2 — Dataset Construction.png>)，仅参考样本到监督记录的流程表达。
- 当前 instruction schema、template 与 compact supervision 数据仅作为格式和质量检查设计依据。

### 内部事实边界（不进入 PPT）

- 当前可陈述的成果是指令规范与模板已构建，不是 Geo-Instruct 数据集已完成。
- 不提供未经正式清点和固定的数据版本所确认的 Geo-Instruct 样本规模。
- 自动生成文本必须保留来源记录和 QC 状态，不能默认视为人工 Gold。

---
