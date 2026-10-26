# 第07页｜Mask-Grounded 区域理解

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.1 多模态融合模型架构设计
- 研究状态：Qwen3-VL-2B 区域理解工程原型已构建；Qwen3.5-4B 区域适配待优化验证

### 页面标题与核心结论

- 页面标题：**全幅影像、区域 mask 和局部影像共同限定候选区域的观察范围**
- 核心结论：全幅环境、区域位置、局部细节和自动计算信息共同送入模型，使区域观察按照固定字段组织并可复核。

### 上屏文字

1. 全幅光学影像提供候选区域的整体环境和周边地物信息
2. 二值 mask、候选区局部影像以及位置、面积和形状信息共同明确观察范围
3. 输出按八类字段记录可见现象、可能误判因素和信息局限，不直接给出风险等级

### 主视觉布局

左侧为浅绿色区域证据区，集中展示全幅光学影像、二值 mask、候选区局部影像和自动计算信息。
中央并列 Qwen3-VL-2B 与 Qwen3.5-4B 两张模型卡，共用右侧的字段检查模块和 2×4 区域观察网格。
2B 使用蓝色实线并标为已构建，4B 使用橙色虚线并标为待优化验证；本页不再放置训练流程。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background, closely following the academic-paper composition and restrained visual language of the supplied Figure 4. Use thin charcoal outlines, rounded rectangular panels, pale green evidence areas, pale blue current-capability cards, pale orange pending-capability cards, compact abstract icons, orthogonal arrows, consistent spacing, and generous whitespace. Use Figure 4 only as a reference for layout and hierarchy. Do not copy its English labels, embedded remote-sensing thumbnails, lower training curriculum, or omitted system components. Use a flat vector style without photorealism, decorative scenery, heavy shadows, three-dimensional effects, or dense model-layer diagrams.

All visible explanatory text, process labels, status labels, arrow captions, legends, and panel headings must be in Simplified Chinese. English is allowed only for these technical terms: Qwen3-VL-2B, Qwen3.5-4B, Mask-Grounded, mask, BBox. Do not render any English layout instruction in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of inserting English or garbled characters. Place the exact title Mask-Grounded 区域理解方法 at the top center.

Organize the figure as one left-to-right runtime process with four coordinated regions: 区域证据, 双模型区域理解, 字段检查与格式解析, 结构化区域观察. Do not draw a training section at the bottom. Keep the main flow centered vertically and preserve the visual rhythm of Figure 4.

Design the left evidence region as one large pale-green rounded panel labeled 区域证据. Do not display separate user-mask, global-mask, or candidate-selection branches. Inside the panel, arrange four evidence cards in a clean two-by-two layout. Label the cards 全幅光学影像, 二值 mask, 候选区局部影像, and 自动计算信息. Use abstract line icons only: a grid-and-landform symbol for the full image, a black-and-white region symbol for the mask, a magnified rectangular crop symbol for the local image, and a document symbol for calculated information. Do not invent satellite imagery, railway-site photographs, segmentation results, or model answers.

Inside 自动计算信息, show exactly six short entries: BBox, 质心, 面积比例, 连通区域, 形状, 裁剪范围. Present these items as read-only supporting information. Use solid dark arrows from all four evidence cards toward the model area. Make it visually clear that the formal visual inputs are the full optical image, the binary mask, and the local context crop, while the calculated spatial information is separate read-only text support. Do not add DEM, slope, InSAR, SAR, or other auxiliary modalities to the model input.

In the central model region, create one shared container labeled Mask-Grounded 区域理解 and place two equal parallel model cards inside it. Stack the cards vertically so they receive the same evidence without duplicating the evidence panel. The upper card has a pale-blue fill, a solid blue outline, and the labels Qwen3-VL-2B and 区域理解工程原型, with a blue status badge 已构建. Connect it to the next module with a solid dark arrow. The lower card has a pale-orange fill, an orange dashed outline, and the labels Qwen3.5-4B and 区域适配, with an orange status badge 待优化验证. Connect it to the next module only with an orange dashed arrow. Do not imply that the 4B branch has completed formal training, produced validated output, or replaced the 2B engineering prototype.

Place a compact rounded module labeled 字段检查与格式解析 immediately to the right of the two model cards. Both model branches point to this shared output contract, but preserve the solid line for the 2B branch and the orange dashed line for the 4B branch. This module represents schema checking and field organization, not an additional reasoning stage.

On the far right, draw one large pale-orange rounded panel labeled 结构化区域观察. Inside it, create a precise two-by-four grid with exactly eight Chinese fields: 目标外观, 目标形态, 周边环境, 区域与背景差异, 可能误判因素, 信息充分程度, 当前局限, 简短总结. These are field names only. Do not generate example answers, descriptions of a specific landslide, confidence values, risk scores, or threat conclusions. Add the exact footer label 仅记录视觉观察，不作专业诊断或风险定级.

Use solid dark arrows for existing evidence flow and the existing 2B engineering path. Use orange dashed outlines and arrows only for the 4B pending-validation path. Keep each visible label short and presentation-readable. Do not include knowledge retrieval, a knowledge base, citations, a second reasoning stage, training data, training stages, performance metrics, scientific-acceptance claims, or any unspecified model capability.
```

### 准确标签清单

- 区域证据｜全幅光学影像｜二值 mask｜候选区局部影像｜自动计算信息
- BBox｜质心｜面积比例｜连通区域｜形状｜裁剪范围
- Qwen3-VL-2B｜区域理解工程原型｜已构建
- Qwen3.5-4B｜Mask-Grounded 区域适配｜待优化验证
- 字段检查与格式解析
- 目标外观｜目标形态｜周边环境｜区域与背景差异
- 可能误判因素｜信息充分程度｜当前局限｜简短总结
- 仅记录视觉观察，不作专业诊断或风险定级

### 本地参考图或真实素材

- [参考图](<../figure/Figure 4 — Mask-Grounded Region Understanding.png>)

### 内部事实边界（不进入 PPT）

- 现有区域 Adapter 属于 2B 工程原型，`formal_acceptance=false / scientific_acceptance=false`。
- 4B Mask-Grounded 未通过内部梯度一致性检查，未进入 allocator 资源检查或正式训练。
- 右侧八类字段是统一输出 schema，不表示已经取得可发布的 4B 区域观察结果。
- 当前输出规则不允许直接断言 `risk_level`、稳定性或实际威胁。

---
