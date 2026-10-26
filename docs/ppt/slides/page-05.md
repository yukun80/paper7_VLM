# 第05页｜Qwen3.5-4B 总体模型框架

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.1、T2.2、T2.3
- 研究状态：总体架构已构建；4B 区域理解方法待优化验证

### 页面标题与核心结论

- 页面标题：**专用分割模型确定区域位置，Qwen3.5-4B 负责理解和描述**
- 核心结论：OA-AuxSeg 负责生成位置和边界信息，Qwen3.5-4B 负责通用遥感理解，区域理解方法连接两类结果。

### 上屏文字

1. OA-AuxSeg 从多源数据中给出概率图、mask、无目标状态和候选区域
2. Qwen3.5-4B 已完成通用遥感适配和 256 条样本的独立对比验证
3. 利用候选区域进行精细描述的方法已设计，4B 区域训练尚未完成

### 主视觉布局

将整体框架压缩为四个横向模块：多源空间输入、OA-AuxSeg、区域证据接口、Qwen3.5-4B 与结果输出。
通用遥感理解直接使用光学图像；区域理解使用 OA-AuxSeg 给出的候选区域信息，并以橙色边框标为待优化验证。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background, with the visual refinement and structural hierarchy of a high-quality academic paper framework figure. Use the supplied Figure 1 only as a reference for composition and graphic language: thin charcoal outlines, rounded rectangular panels, nested capability bands, restrained pastel colors, compact abstract icons, clean orthogonal arrows, consistent internal margins, and balanced whitespace. Do not copy its English labels, embedded remote-sensing thumbnails, or omitted system components. Use a flat vector style without photorealism, decorative scenery, heavy shadows, three-dimensional effects, or dense neural-network layers.

All visible explanatory text, process labels, status labels, arrow captions, legends, and panel headings must be in Simplified Chinese. English is allowed only for these technical terms: Qwen3.5-4B, OA-AuxSeg, Mask-Grounded, DEM, InSAR, mask, VLM. Do not render any English layout instruction in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of inserting English or garbled characters. Place the exact Chinese title 多源遥感大模型总体框架 at the top center in a restrained academic heading style.

Organize the figure as four coordinated primary regions rather than four isolated boxes. Preserve the visual rhythm of Figure 1 while redesigning the spacing after removing its lower knowledge-related area. The four regions are 多源空间输入, OA-AuxSeg 空间定位, Mask-Grounded 区域理解, and Qwen3.5-4B 与结果输出. Arrange 多源空间输入 on the left, place the blue OA-AuxSeg capability band in the upper center, place the green Mask-Grounded capability band directly below it, and place the orange Qwen3.5-4B model column with three aligned output cards on the right. Keep both central capability bands equal in visual importance and do not enlarge either band to occupy the removed lower area. Rebalance panel widths, vertical alignment, arrow lengths, and whitespace so the complete framework remains centered.

In 多源空间输入, use abstract line icons instead of fabricated image thumbnails. Show one primary input card labeled 光学或多光谱影像. Below it, group three smaller auxiliary cards under 可用辅助数据: DEM, 坡度, 编码 InSAR. Add the exact note 仅用于空间分析 beside the auxiliary group. Represent optical data with a simple grid-and-landform icon, DEM with contour lines, slope with a directional gradient icon, encoded InSAR with abstract phase-wave lines, and mask with a simple black-and-white region symbol. These must be schematic icons, not invented satellite scenes, railway photographs, terrain products, or model results.

Embed a narrow task-selection card between the input region and the two central capability bands, visually subordinate to the four primary regions and not presented as a fifth region. Label it 任务选择 and include four compact options: 通用理解, 分割, 区域理解, 分割与理解. Use thin gray dashed control arrows from this card to the relevant capability band, while all actual data movement uses solid dark arrows. Do not add other task types.

Design the upper-center light-blue band as OA-AuxSeg 空间定位. Inside it, show 光学或多光谱影像 as the spatial reference and 可用辅助数据 entering a compact module labeled 有效模态融合, followed by 多尺度分割. Connect these modules from left to right with solid arrows. At the right edge of the blue band, use four vertically aligned result cards labeled 概率图, 二值 mask, 无目标状态, 候选区域. Use only abstract mini-icons. Draw a direct solid arrow from these results to the output card 空间结果. Make it visually explicit that DEM, 坡度, and 编码 InSAR participate only in this OA-AuxSeg spatial-analysis band.

Design the lower-center light-green band as Mask-Grounded 区域理解. Feed it from the OA-AuxSeg candidate-region result with a solid downward connector. Inside the band, place three equal evidence cards labeled 全幅光学影像, 二值 mask, 候选区局部影像. Connect them to a compact read-only information card labeled 位置、面积、形状. Then connect the complete regional evidence group to the regional-understanding entry of Qwen3.5-4B. Add a small orange-outlined status badge labeled 待优化验证. Do not route DEM, 坡度, or 编码 InSAR into this band or into Qwen3.5-4B.

Design the right-side model column in pale orange with the main heading Qwen3.5-4B. Inside it, create two clearly separated capability cards. The upper card is labeled 通用遥感理解 with a blue status badge 已完成适配; connect the original 光学或多光谱影像 directly to this card using a solid arrow. The lower card is labeled Mask-Grounded 区域理解 with an orange-outlined status badge 待优化验证; connect it only to 全幅光学影像, 二值 mask, 候选区局部影像, and 位置、面积、形状 from the green band. Do not imply that the pending regional capability has completed training or produced validated results.

Place exactly three output cards at the far right, vertically aligned and visually consistent with Figure 1: 空间结果, 通用描述与问答, 结构化区域观察. The 空间结果 card is supplied by OA-AuxSeg and may contain small schematic symbols for 二值 mask, 无目标状态, and 候选区域. The 通用描述与问答 card is supplied by the completed general-understanding capability. The 结构化区域观察 card is connected to the pending regional-understanding capability and must carry a small orange label 待优化验证. Do not show a fourth output card.

Use a restrained academic palette: pale blue for OA-AuxSeg, pale green for Mask-Grounded, pale orange for Qwen3.5-4B, dark gray for existing solid data arrows, light gray for dashed task-selection arrows, and orange only for pending-validation borders and badges. Do not use purple. Keep text large enough for a presentation slide, limit each card to one short label, avoid paragraphs inside the figure, and maintain low-to-medium information density.

Do not include knowledge retrieval, a knowledge base, a text evidence bank, a second reasoning stage, citation generation, risk grades, threat conclusions, performance numbers, SAR as a current input, additional modalities, invented railway-site evidence, or any unspecified model capability. Do not add model-layer names, Transformer blocks, loss functions, training curves, or evaluation metrics.
```

### 准确标签清单

- 多源空间输入
- OA-AuxSeg
- 区域证据接口
- Qwen3.5-4B
- 通用遥感理解｜已完成适配
- Mask-Grounded 区域理解｜待优化验证
- 空间结果｜通用描述／问答｜结构化区域观察

### 本地参考图或真实素材

- [参考图](<../figure/Figure 1 — Overall Framework.png>)

### 内部事实边界（不进入 PPT）

- 当前 P0 的正式区域输入仍为 Full Optical + Binary Mask + Context Crop。
- 辅助预览不进入当前 4B formal grounded input。
- 图中的 4B 区域理解部分表示方法设计位置，不表示已经形成可发布 Adapter。

---
