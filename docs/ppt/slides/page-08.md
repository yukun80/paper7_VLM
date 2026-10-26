# 第08页｜分阶段训练安排

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.2 多阶段演进训练策略；衔接 T3.3 领域微调与迭代训练
- 研究状态：2B 区域 Adapter 已构建；4B 通用适配与独立验证已完成，区域适配待优化验证；Geo-Instruct 属于下一阶段

### 页面标题与核心结论

- 页面标题：**两种模型采用相同训练顺序，当前差异集中在区域适配阶段**
- 核心结论：基础模型先完成通用遥感适配和独立验证，再使用区域指令继续训练；2B 已形成区域原型，4B 区域训练仍需解决计算一致性问题。

### 上屏文字

1. RS-GeneralDesc 274,693 Records 支持通用适配，区域训练使用 6,974 条混合指令
2. 两种模型均已完成 1000 步通用适配及 256 条样本的独立对比验证
3. 2B 区域 Adapter 已构建，4B 区域训练暂缓，Geo-Instruct 属于下一阶段

### 主视觉布局

中央为一条从左到右的公共训练路线：基础模型、RS-General 通用适配、独立对比验证检查点、
Mask-Grounded 区域适配和 Geo-Instruct 领域训练。通用适配与区域适配卡片分别列出 2B、4B
两行状态；下方用两张数据卡区分通用训练数据和区域训练指令。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background, using the academic-paper style of the lower training-curriculum area in the supplied Figure 4. Use thin charcoal outlines, rounded rectangular stage cards, compact two-row status blocks, clean horizontal arrows, restrained pale-blue, pale-orange, and pale-purple colors, consistent margins, and balanced whitespace. Use Figure 4 only as a reference for visual hierarchy. Do not copy its English labels, image thumbnails, runtime evidence section, or original model-status claims. Use a flat vector style without photorealism, training curves, heavy shadows, three-dimensional effects, or dense network-layer diagrams.

All visible explanatory text, process labels, status labels, arrow captions, legends, and panel headings must be in Simplified Chinese. English is allowed only for these technical terms: Qwen3-VL-2B, Qwen3.5-4B, RS-General, RS-GeneralDesc, Mask-Grounded, Geo-Instruct, Records, Adapter. Do not render any English layout instruction in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of inserting English or garbled characters. Place the exact title 双模型分阶段训练路线 at the top center.

Draw one shared horizontal training route from left to right with four large stage cards and one smaller checkpoint between the second and third stages. The route is 基础模型, RS-General 通用适配, 独立对比验证, Mask-Grounded 区域适配, Geo-Instruct 领域训练. The 独立对比验证 element must be a compact checkpoint symbol, not a large training-stage card. Use clear arrows and keep all stages on one horizontal axis.

Design the first large stage card as 基础模型. Inside it, place two equal rows labeled Qwen3-VL-2B and Qwen3.5-4B. Use a neutral pale-blue container and do not compare model quality or size.

Design the second large stage card as RS-General 通用适配. Inside it, place two aligned blue rows. The first row reads Qwen3-VL-2B 已完成 1000 步. The second row reads Qwen3.5-4B 已完成 1000 步. Below this stage, attach one compact blue data card labeled RS-GeneralDesc｜274,693 Records. Connect the data card upward to the common adaptation stage. Do not imply that the number of steps alone proves a performance improvement.

Place a small diamond or hexagonal checkpoint after the common adaptation stage. Label it 独立对比验证｜256 条. Inside or directly below it, show two short blue labels: Qwen3-VL-2B 已完成 and Qwen3.5-4B 已完成. This checkpoint verifies each model family within its own comparison scope. Do not draw it as a scientific superiority comparison between 2B and 4B.

Between the checkpoint and the regional-adaptation stage, add the exact Chinese arrow caption 沿用同模型家族的通用适配权重. Do not use the English phrase warm start in the visible image.

Design the third large stage card as Mask-Grounded 区域适配. Split it into two aligned rows. The upper row uses a pale-blue fill and solid outline, labeled Qwen3-VL-2B｜区域 Adapter 已构建. The lower row uses a pale-orange fill and orange dashed outline, labeled Qwen3.5-4B｜待优化验证｜正式训练尚未启动. Use a solid blue connector to the 2B row and an orange dashed connector to the 4B row. Do not display the internal failure code or any Gate name in the image.

Below the regional-adaptation stage, attach one compact green data card labeled Mask-Grounded｜6,974 条混合指令. Inside the same card, add exactly two smaller labels: 90% 区域监督 and 10% 通用回放. Keep this card visually separate from RS-GeneralDesc because the units and training purposes are different. Do not add the two quantities together.

Design the final large stage card as Geo-Instruct 领域训练. Use a pale-purple fill, a purple dashed outline, and the status label 下一阶段. Connect to it with a purple dashed arrow. Do not imply that the dataset, training, evaluation, or results already exist.

Use blue solid borders and arrows for completed or constructed stages, orange dashed borders and arrows for the 4B regional stage that requires further validation, and purple dashed borders and arrows only for the future Geo-Instruct stage. Add a small bottom legend with exactly three Chinese labels: 已完成或已构建, 待优化验证, 下一阶段. Keep the information density moderate and every visible label short enough for a presentation slide.

Do not include internal Gate names, failure codes, performance gains, accuracy values, training curves, completed 4B regional training, a completed Geo-Instruct dataset, cross-model superiority claims, knowledge retrieval, risk analysis, or any additional training stage.
```

### 准确标签清单

- 双模型分阶段训练路线
- 基础模型｜Qwen3-VL-2B｜Qwen3.5-4B
- RS-General 通用适配｜已完成 1000 步
- RS-GeneralDesc｜274,693 Records
- 独立对比验证｜256 条｜两种模型均已完成
- 沿用同模型家族的通用适配权重
- Mask-Grounded 区域适配
- Qwen3-VL-2B｜区域 Adapter 已构建
- Qwen3.5-4B｜待优化验证｜正式训练尚未启动
- Mask-Grounded｜6,974 条混合指令｜90% 区域监督｜10% 通用回放
- Geo-Instruct 领域训练｜下一阶段

### 本地参考图或真实素材

- [参考图](<../figure/Figure 4 — Mask-Grounded Region Understanding.png>)：仅参考底部训练流程的版式与科研绘图语言。
- [当前进度](../../status/README.md)：2B、4B 通用适配、独立验证与区域适配状态的实时证据。
- 本页为新绘训练路线，不绑定训练曲线截图。

### 内部事实边界（不进入 PPT）

- 两个模型的 `1,000 steps` 是已完成训练运行事实，不单独等于科学性能提升。
- 两个模型的独立验证仅适用于各自冻结的同家族比较范围，不支持跨模型优越性结论。
- 2B Region Adapter 已有 step-900 资产，但 `formal_acceptance=false / scientific_acceptance=false`。
- 4B Grounded 在 `V4_GRADIENT_EQUIVALENCE_GATE_FAILED` 后串行停止，未进入资源检查和正式训练。
- Geo-Instruct 数据集和领域训练均不得写成已经完成。

---
