# 多源遥感大模型与地质要素智能解译：20 页研究进展 PPT 图文设计与 Imagegen 提示词

## 使用说明

本文件用于制作面向项目甲方的 20 页研究进展汇报。页面内容以
`docs/实施方案v4.docx` 的任务二、任务三为项目口径，以 `REBUILD_PROGRESS.md`
的实时状态为成果依据，以当前真实模型、数据集和运行结果为证据。

- 画布统一为 `16:9 landscape scientific infographic, pure white background`。
- 每页只有一个视觉中心，上屏文字不超过 3 条，不使用大段正文或密集网络层名称。
- 蓝色实线表示已完成或已构建能力；橙色边框表示待优化验证；紫色虚线表示下一阶段。
- 状态标签统一使用：`已完成 / 已构建 / 待优化验证 / 下一阶段`。
- Imagegen 不得生成虚构遥感影像、实验数字、铁路现场成果或模型输出。
- 对真实影像和模型结果，只允许重新编排版式，不得改变像素、mask 或输出文字。
- 页面不显示论文引用；本文件中的素材路径和事实边界仅供制图与内部审核。
- Qwen3.5-4B 的 RS-General 适配与独立对比验证已完成；Mask-Grounded 的训练计算一致性
  检查未达到要求，未形成可发布的 4B Grounded Adapter。
- 后续知识增强路线只在第 20 页作为探索性展望出现，不进入当前成果链。

## 统一视觉与文字规则

1. 页面标题置于顶部左侧，采用结论式中文标题。
2. 项目编号和研究状态放在右上角小型标签内，例如 `T2.1｜已构建`。
3. 图中只保留必要的短标签；复杂解释放入本文件的内部事实边界，不进入 PPT。
4. Imagegen 若不能稳定生成准确中文，先生成无文字框架，再按照准确标签清单人工覆盖。
5. 所有候选区保持候选语义，不使用已确认滑坡、实际威胁等越界措辞。

---

## 第01页｜封面：多源遥感大模型与地质要素智能解译

### 项目对应与研究状态

- 项目对应：任务二、任务三总体汇报
- 研究状态：研究进展汇报

### 页面标题与核心结论

- 页面标题：**多源遥感大模型与地质要素智能解译研究进展**
- 核心结论：任务二负责模型方法研究，任务三负责样本和指令数据建设，共同支持铁路地质遥感智能解译。

### 上屏文字

1. 任务二：面向铁路的多源要素遥感大模型构建
2. 任务三：典型地质遥感要素样本库构建与智能识别
3. 研究进展汇报｜汇报人、单位、日期待填

---

## 第02页｜传统单任务模型的三类瓶颈

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

## 第03页｜任务二与任务三的六项子任务关系

### 项目对应与研究状态

- 项目对应：T2.1 至 T2.3、T3.1 至 T3.3
- 研究状态：总体任务关系已明确

### 页面标题与核心结论

- 页面标题：**任务二负责模型能力，任务三负责数据准备和持续改进**
- 核心结论：样本库和指令数据支持模型训练，验证中发现的问题再用于补充样本和修订指令。

### 上屏文字

1. T3.1 提供多源样本，T3.2 正在按统一格式建设指令和回答内容
2. T2.1 和 T2.2 负责多源信息处理、通用理解和区域理解训练
3. 模型验证发现的问题进入 T3.3，用于下一轮样本和训练改进

### 主视觉布局

采用上下双层任务关系图：上层为任务二三个模块，下层为任务三三个模块。任务三前两个模块向任务二的
架构和训练模块提供数据；任务二的验证问题通过返回箭头进入任务三的持续改进。任务编号完整可见。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for the task identifiers T2.1, T2.2, T2.3, T3.1, T3.2, and T3.3. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly six rounded task modules in two rows. The upper row is labeled 任务二 面向铁路的多源要素遥感大模型构建 and contains T2.1 多模态融合模型架构设计, T2.2 多阶段演进训练策略, and T2.3 风险评估方法构建与集成. The lower row is labeled 任务三 典型地质遥感要素样本库构建与智能识别 and contains T3.1 典型地质要素解译样本库, T3.2 领域专用指令数据集, and T3.3 领域微调与迭代训练. Draw arrows from T3.1 and T3.2 toward model design and training. Draw one return arrow from 模型验证发现的问题 toward T3.3 and label it 补充样本并修订指令. Keep the layout sparse. Do not add algorithm layers, experiment numbers, extra tasks, or invented completion status.
```

### 准确标签清单

- T2.1 多模态融合模型架构设计
- T2.2 多阶段演进训练策略
- T2.3 风险评估方法构建与集成
- T3.1 典型地质要素解译样本库
- T3.2 领域专用指令数据集
- T3.3 领域微调与迭代训练
- 模型验证发现的问题
- 补充样本并修订指令


---

## 第04页｜第一章：任务二

### 项目对应与研究状态

- 项目对应：任务二总览
- 研究状态：已构建／已完成／待优化验证／下一阶段

### 页面标题与核心结论

- 页面标题：**任务二已完成多源处理基础和 4B 通用遥感适配**
- 核心结论：多模态架构和 4B 通用适配已有工程基础，4B 区域能力与铁路风险集成尚未完成。

### 上屏文字

1. T2.1 已构建多模态融合框架，为光学和辅助数据的联合处理提供基础
2. T2.2 已完成 4B 通用遥感适配，区域适配仍需优化和验证
3. T2.3 铁路风险评估方法将在下一阶段完成构建与集成

### 主视觉布局

中央使用三阶台阶，左低右高，分别对应 T2.1、T2.2、T2.3。每级台阶右上角放状态角标；
T2.2 内部再分成蓝色通用适配已完成和橙色区域适配待优化验证两部分。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for T2.1, T2.2, T2.3, and 4B. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Use one three-step ascending staircase as the visual center. The first blue step is T2.1 多模态融合模型架构设计 with 已构建. The second step is T2.2 多阶段演进训练策略 and contains two areas: 4B 通用遥感适配 已完成 and 4B 区域适配 待优化验证. The third step has a purple dashed border and reads T2.3 风险评估方法构建与集成 下一阶段. Keep large Chinese labels and generous white space. Do not add performance charts, completed regional training, railway risk results, or unspecified modules.
```

### 准确标签清单

- 第一章｜任务二
- 多源处理｜4B 通用理解｜风险评估
- T2.1｜已构建
- 4B 通用遥感适配｜已完成
- 4B 区域适配｜待优化验证
- T2.3｜下一阶段

### 本地参考图或真实素材

- 无；章节页采用状态台阶，不复用模型图。

### 内部事实边界（不进入 PPT）

- 4B RS-General Adapter 和独立 Gate B 已完成。
- 4B Mask-Grounded 当前因梯度一致性未达到内部要求而停止，不能使用即将完成等表述。

---

## 第05页｜Qwen3.5-4B 总体模型框架

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 1 — Overall Framework.png`

### 内部事实边界（不进入 PPT）

- 当前 P0 的正式区域输入仍为 Full Optical + Binary Mask + Context Crop。
- 辅助预览不进入当前 4B formal grounded input。
- 图中的 4B 区域理解部分表示方法设计位置，不表示已经形成可发布 Adapter。

---

## 第06页｜以光学影像为基准的多模态融合

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 3 — Optical-Anchored Spatial Perception.png`

### 内部事实边界（不进入 PPT）

- OA-AuxSeg 当前辅助输入配置只接受 `dem / insar_velocity / slope`。
- 编码 InSAR 的单位和正负方向未确认时，只能作为图像输入。
- 53,645 条 Benchmark 和定版权重是现有工程成果，不等于 Gate A 通过。

---

## 第07页｜Mask-Grounded 区域理解

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 4 — Mask-Grounded Region Understanding.png`

### 内部事实边界（不进入 PPT）

- 现有区域 Adapter 属于 2B 工程原型，`formal_acceptance=false / scientific_acceptance=false`。
- 4B Mask-Grounded 未通过内部梯度一致性检查，未进入 allocator 资源检查或正式训练。
- 右侧八类字段是统一输出 schema，不表示已经取得可发布的 4B 区域观察结果。
- 当前输出规则不允许直接断言 `risk_level`、稳定性或实际威胁。

---

## 第08页｜分阶段训练安排

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 4 — Mask-Grounded Region Understanding.png`：仅参考底部训练流程的版式与科研绘图语言。
- `REBUILD_PROGRESS.md`：2B、4B 通用适配、独立验证与区域适配状态的实时证据。
- 本页为新绘训练路线，不绑定训练曲线截图。

### 内部事实边界（不进入 PPT）

- 两个模型的 `1,000 steps` 是已完成训练运行事实，不单独等于科学性能提升。
- 两个模型的独立验证仅适用于各自冻结的同家族比较范围，不支持跨模型优越性结论。
- 2B Region Adapter 已有 step-900 资产，但 `formal_acceptance=false / scientific_acceptance=false`。
- 4B Grounded 在 `V4_GRADIENT_EQUIVALENCE_GATE_FAILED` 后串行停止，未进入资源检查和正式训练。
- Geo-Instruct 数据集和领域训练均不得写成已经完成。

---


## 第10页｜当前风险分析信息与后续集成

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 1 — Overall Framework.png`，仅参考当前空间证据链。
- 本页风险集成部分为项目任务映射图，不使用虚构铁路数据。

### 内部事实边界（不进入 PPT）

- 当前运行时不得直接输出正式 `risk_level`、稳定性或实际威胁结论。
- 线路距离、暴露度和脆弱性尚未形成当前系统的可验收集成模块。
- 风险说明在本页是下一阶段交付方向，不是现有模型输出样例。

---

## 第11页｜第二章：任务三

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 2 — Dataset Construction.png`，仅参考数据关系版式。

### 内部事实边界（不进入 PPT）

- 现有数据集不等同于已完成的 Geo-Instruct 正式数据集。
- 当前主体数据来自开源遥感与滑坡数据，不得写成铁路自采样本库。
- 领域微调尚未完成，验证问题用于后续改进表示方法设计。

---

## 第12页｜三套可追溯数据集

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 2 — Dataset Construction.png`
- 实际数据身份与统计以 `REBUILD_PROGRESS.md` 中绑定的 manifest、ledger 和 schema 为准。

### 内部事实边界（不进入 PPT）

- 53,645 的单位是 Samples，274,693 的单位是 Records，6,974 的单位是 Structured Region Messages。
- 6,974 条 compact supervision 不是人工 Gold，也不是专家共识标签。
- 本页不把三套数据称为已完成的铁路专属数据集。

---

## 第13页｜样本库加工步骤

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库构建
- 研究状态：基础加工步骤与版本化数据已构建；铁路场景补充为后续工作

### 页面标题与核心结论

- 页面标题：**多源样本经过检查、配准和分组后形成可训练数据**
- 核心结论：每条样本先检查文件和模态配对，再统一空间位置并按场景划分训练集、验证集和测试集。

### 上屏文字

1. 首先检查开源滑坡数据是否完整、可读，并确认不同模态正确配对
2. 以光学影像为位置基准，对齐 DEM、编码 InSAR 和 mask
3. 按场景划分 train、val 和 test，并用 HDF5、manifest、ledger 记录数据版本

### 主视觉布局

使用一条由左至右的五阶段加工流程：多源数据源、质量清洗、以光学影像为基准进行空间对齐、分组划分、版本化数据输出。
在第四阶段画三条互不交叉的分组线表示 train、val、test；末端输出 HDF5、manifest、ledger、schema 四张卡。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for DEM, InSAR, mask, train, val, test, HDF5, manifest, ledger, schema, and SHA-256. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly five processing steps from left to right: 开源多模态滑坡数据, 文件和配对检查, 以光学影像为基准进行空间对齐, 按场景分组, 版本化数据输出. The first step shows 光学, DEM, 编码 InSAR, and mask. The second shows 可读性, 尺寸一致性, and 模态配对. The fourth branches into train, val, and test. The final step shows HDF5, manifest, ledger, schema, and SHA-256. Keep all action labels in Chinese and the layout sparse. Do not generate remote sensing images, railway field samples, sample totals, extra modalities, or expert annotation claims.
```

### 准确标签清单

- 开源多模态滑坡数据
- 格式检查与质量清洗
- 以光学影像为基准进行空间对齐
- 按场景分组｜train｜val｜test
- HDF5｜manifest｜ledger｜schema｜SHA-256
- 光学｜DEM｜编码 InSAR｜mask

### 本地参考图或真实素材

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 2 — Dataset Construction.png`，参考加工步骤版式。
- `/home/yukun80/codes/datasets/multimodal-landslide-dataset/`，只读数据源；本页不复制素材。

### 内部事实边界（不进入 PPT）

- 当前样本主体来自开源滑坡数据，不能写成铁路沿线实地采集。
- HDF5 格式统一不代表模态语义、数值范围或有效覆盖天然一致。
- train／val／test 的正式身份以相应 manifest 和 ledger 为准，不从文件名推断。

---

## 第14页｜样本加工与质量控制

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库构建
- 研究状态：基础数据检查已构建；四层专业标签体系建设中

### 页面标题与核心结论

- 页面标题：**先排除损坏和错配样本，再补充影响识别难度的专业标签**
- 核心结论：基础检查保证文件和模态配对正确，后续标签说明样本类别、易混淆程度、图像质量和关键地貌特征。

### 上屏文字

1. 已完成文件损坏、格式转换和多源数据配对检查，排除不可用样本
2. 已检查同一场景是否同时出现在 train、val 和 test，避免评价结果失真
3. 正在补充类别、易混淆样本、图像质量和关键地貌特征标签

### 主视觉布局

采用上下两层结构。上层为蓝色基础数据检查流程，依次放四类检查；下层为橙色
四层专业标签，用四块叠层卡片展示标签维度。两层之间使用向下箭头，表示只有通过基础检查的
样本才进入领域标注。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for train, val, test, manifest, ledger, schema, and SHA-256. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Use a two-level vertical structure. The upper blue area is labeled 基础数据检查 已构建 and contains exactly four checks: 文件可用性, 格式转换, 模态配对, 场景重复. Add manifest, ledger, schema, and SHA-256 below it. Use one downward arrow to an orange four-layer stack labeled 专业标签 建设中. The layers are 类别标签, 易混淆样本, 图像质量, 关键地貌特征. Add one Chinese note 场景重复检查覆盖 train val test. Keep the layout sparse. Do not add quality scores, sample counts, expert consensus, annotation examples, or railway imagery.
```

### 准确标签清单

- 基础数据检查｜已构建
- 文件可用性｜格式转换｜模态配对｜场景重复
- manifest｜ledger｜schema｜SHA-256
- 专业标签｜建设中
- 类别标签｜易混淆样本｜图像质量｜关键地貌特征

### 本地参考图或真实素材

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 2 — Dataset Construction.png`，仅参考数据治理语言。
- 实际质量检查要求以仓库中的版本化 manifest、ledger 与 schema 为准。

### 内部事实边界（不进入 PPT）

- 四层统一标签体系尚在建设中，不能与基础工程检查一起标成已完成。
- 不对当前样本做未有证据支持的专家一致性或质量百分比声明。
- 本页不改变、不重建任何现有 Benchmark 数据。

---

## 第15页｜Geo-Instruct 指令生成与质量检查

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

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 2 — Dataset Construction.png`，仅参考样本到监督记录的流程表达。
- 当前 instruction schema、template 与 compact supervision 数据仅作为格式和质量检查设计依据。

### 内部事实边界（不进入 PPT）

- 当前可陈述的成果是指令规范与模板已构建，不是 Geo-Instruct 数据集已完成。
- 不提供未经正式清点和固定的数据版本所确认的 Geo-Instruct 样本规模。
- 自动生成文本必须保留来源记录和 QC 状态，不能默认视为人工 Gold。

---

## 第16页｜领域微调与错误案例改进

### 项目对应与研究状态

- 项目对应：T3.3 领域微调与迭代训练；衔接 T2.2 多阶段演进训练策略
- 研究状态：4B 通用适配已完成；区域适配待优化验证；领域微调为下一阶段

### 页面标题与核心结论

- 页面标题：**通用适配完成后，区域训练暂缓，错误案例用于后续改进**
- 核心结论：4B 通用遥感训练已经完成，区域训练因计算一致性问题暂停，后续将根据错误案例补充样本和修订指令。

### 上屏文字

1. Qwen3.5-4B 通用遥感适配和 256 条样本的独立对比验证已完成
2. 区域适配的训练计算一致性未达到要求，正式训练尚未启动
3. 后续将针对误报、漏报和依据错位补充样本并修订指令

### 主视觉布局

上半部为四阶段训练时间轴：基础模型、通用遥感 SFT、Mask-Grounded SFT、Geo-Instruct 领域 SFT；
第二阶段为蓝色完成态，第三阶段在入口处设置橙色暂停标记，第四阶段为紫色虚线。下半部将三类错误
连接到补充样本、修订指令和下一轮训练三个步骤。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for Qwen3.5-4B, RS-General, SFT, Mask-Grounded, and Geo-Instruct. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Across the upper half draw exactly four stages: Qwen3.5-4B 基础模型, RS-General 通用遥感 SFT 已完成, Mask-Grounded 区域 SFT 待优化验证, Geo-Instruct 领域 SFT 下一阶段. Add a blue status card labeled 独立对比验证 已完成 after the second stage. Add an orange pause marker labeled 训练计算一致性检查未达到要求 before formal regional training. In the lower half show 误报, 漏报, and 依据错位 leading to 补充样本, 修订指令, and 下一轮训练. Keep the layout sparse. Do not show completed regional training, completed domain training, performance gains, automatic online learning, or additional stages.
```

### 准确标签清单

- Qwen3.5-4B 基础模型
- RS-General 通用遥感 SFT｜已完成
- 独立对比验证｜已完成
- Mask-Grounded 区域 SFT｜待优化验证
- 训练计算一致性检查未达到要求
- Geo-Instruct 领域 SFT｜下一阶段
- 误报｜漏报｜依据错位
- 补充样本｜修订指令｜下一轮训练

### 本地参考图或真实素材

- `REBUILD_PROGRESS.md`：4B 通用适配、独立对比验证和 Mask-Grounded 训练计算一致性检查的实时状态。
- 本页为新绘训练与错误案例改进图，不使用虚构训练曲线。

### 内部事实边界（不进入 PPT）

- 当前只确认 4B RS-General Adapter 完成并通过固定范围内的独立 Gate B。
- 4B Mask-Grounded 未形成可发布 Adapter；Geo-Instruct 领域 SFT 尚未实施。
- 使用错误案例改进样本和指令是方法设计，不代表已完成线上自动学习系统。

---

## 第17页｜单条多源样本真实截图

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库构建
- 研究状态：真实同源样本已构建；本页仅展示数据组织

### 页面标题与核心结论

- 页面标题：**同一编号关联光学、DEM、编码 InSAR 和像素标注**
- 核心结论：同一样本编号对应四类空间位置一致的数据，光学影像作为位置基准，其他数据提供地形、形变编码和像素标签。

### 上屏文字

1. 同一样本编号对应光学、DEM、编码 InSAR 和 mask 四类数据
2. 光学影像作为位置基准，四张截图保持相同空间范围和原始比例
3. 编码 InSAR 只作图像输入，不解释具体位移数值和正负方向

### 主视觉布局

采用 2×2 大图矩阵，按左上光学、右上 DEM、左下编码 InSAR、右下 mask 排列；中央交叉位置
放置同一样本编号。四图使用相同尺寸和边界，不裁剪、不重着色；页脚标注真实同源样本和仅重新编排版式。

### 真实截图制作建议

1. 直接读取下列四个真实文件，使用程序或图像查看器分别导出截图，不调用生成式制图工具。
2. 在 PPT 中建立四个等大的图片框，按光学、DEM、编码 InSAR、mask 的顺序完整嵌入，不裁剪、不拉伸、不重着色。
3. 四图上方分别添加短标签，中央标注样本编号 `multimodal_landslide_loess_000002`，页脚标注真实同源样本和仅重新编排版式。
4. 若程序显示带色标的编码 InSAR，只保留原始色标和原始文字，不补写单位、位移方向或定量解释。

### 准确标签清单

- multimodal_landslide_loess_000002
- 光学｜DEM｜编码 InSAR｜mask
- 同一样本编号
- 光学影像作为位置基准
- 真实同源样本｜仅重新编排版式

### 本地参考图或真实素材

- 光学：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/rgb/train/multimodal_landslide_loess_000002.png`
- DEM：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/dem/train/multimodal_landslide_loess_000002.png`
- 编码 InSAR：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/insar_vel/train/multimodal_landslide_loess_000002.png`
- mask：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/mask/train/multimodal_landslide_loess_000002.png`

### 内部事实边界（不进入 PPT）

- 四张图必须原样使用，不允许生成式工具重绘或补全像素。
- 编码 InSAR 仅是已有编码图；在单位和符号未核验前不得解释为定量位移。
- 该样本来自开源 Multimodal Landslide 数据，不是铁路项目现场样本。

---

## 第18页｜多任务能力与案例矩阵

### 项目对应与研究状态

- 项目对应：T2.1 多模态融合模型架构设计；T2.3 风险评估集成；T3.3 领域微调
- 研究状态：通用描述与 OA-AuxSeg 案例为真实输出；区域与风险两格为流程示意

### 页面标题与核心结论

- 页面标题：**两类真实输出可以复核，区域理解和风险分析仍是方法示意**
- 核心结论：铁路场景描述和分割结果使用真实输出，区域理解与风险分析只展示处理步骤，不作为 4B 实验结果。

### 上屏文字

1. 真实输出包括 Qwen3.5-4B 铁路场景描述和 OA-AuxSeg 分割结果
2. Mask-Grounded 区域理解和风险分析当前只展示处理步骤
3. 两类方法示意均不代表 4B 区域训练或风险评估结果

### 主视觉布局

采用 2×2 等宽案例矩阵。左上放外部验证铁路场景影像与真实 4B prediction；右上放同一次
OA-AuxSeg Demo 的 optical、global mask、overlay。左下使用 PPT 形状绘制区域理解的输入和输出步骤，右下使用
PPT 形状绘制风险分析的信息和说明步骤；下排统一使用橙色边框并标注方法示意和非 4B 实验结果。

### 真实截图制作建议

1. 左上直接使用下列外部验证影像，并从对应 `predictions.jsonl` 核对 record ID、模型角色和原始 prediction；英文输出由制作者原样粘贴，不交给生成式工具改写。
2. 右上直接使用同一 Demo run 下的 `optical_full.png`、`global_mask.png` 和 `predicted_overlay.png`，保持原始像素、颜色和比例。
3. 左下使用 PPT 基本形状绘制全幅光学影像、mask、候选区局部影像进入结构化区域观察的步骤，并标注方法示意和非 4B 实验结果。
4. 右下使用 PPT 基本形状绘制候选区域信息、线路暴露信息进入风险说明的步骤，并标注方法示意和非 4B 实验结果；没有真实结果时不得添加模拟描述或风险等级。

### 准确标签清单

- 通用描述｜真实 4B prediction
- `There are many buildings on both sides of the railway .`
- 像素分割｜真实 Demo 输出
- 区域理解｜流程示意，非 4B 实验结果
- 风险分析｜流程示意，非 4B 实验结果

### 本地参考图或真实素材

- 4B 铁路影像：`/home/yukun80/codes/benchmark/rs_generaldesc_v1/assets/image/90/90dcb4c0f5e9bf41cbb9587d74f67094f9db0ee9be15b8451519156621de6d5a.jpg`
- 4B prediction：`/home/yukun80/codes/paper7_VLM/outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen35_4b_r851bf6e8_v1/adapter/predictions.jsonl`（对应 record 的 prediction）
- Demo 光学：`/home/yukun80/codes/paper7_VLM/outputs/demo/unified_workbench_v1/runs/demo_20260828T144934105536Z_b43e5bc9089f48ae8d6fd73f58e60b02/inputs/optical_full.png`
- Demo mask：`/home/yukun80/codes/paper7_VLM/outputs/demo/unified_workbench_v1/runs/demo_20260828T144934105536Z_b43e5bc9089f48ae8d6fd73f58e60b02/viewer_assets/segment_only/global_mask.png`
- Demo overlay：`/home/yukun80/codes/paper7_VLM/outputs/demo/unified_workbench_v1/runs/demo_20260828T144934105536Z_b43e5bc9089f48ae8d6fd73f58e60b02/viewer_assets/segment_only/predicted_overlay.png`

### 内部事实边界（不进入 PPT）

- 左上是 RS-GeneralDesc 外部验证样例，不是本项目铁路现场验证影像。
- 上排只证明对应单条输出可追溯，不外推整体业务精度。
- 下排不得放置伪造的 4B 区域描述、风险等级或实际威胁结论。

---

## 第19页｜当前进展状态看板

### 项目对应与研究状态

- 项目对应：任务二、任务三阶段成果总览
- 研究状态：按 2026-09-02 `REBUILD_PROGRESS.md` 快照

### 页面标题与核心结论

- 页面标题：**4B 通用遥感适配已完成独立评测，区域训练因一致性问题暂缓**
- 核心结论：当前已经形成可复核的数据集、分割能力和通用理解能力，区域训练、领域训练和风险评估仍需继续完成。

### 上屏文字

1. 已完成 4B 通用遥感训练 1000 步，并完成 256 条样本的独立对比评测
2. 已构建 OA-AuxSeg、三套数据集、区域证据接口和指令模板
3. 下一阶段将推进区域训练、Geo-Instruct、铁路样本补充和风险评估

### 主视觉布局

画面中央为四列等宽看板：已完成、已构建、待优化验证、下一阶段。每列最多四张短卡片；第三列最上方
使用橙色醒目标注 4B 区域训练因计算一致性问题暂缓，第四列使用紫色虚线。页脚给出状态日期。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for Qwen3.5-4B, OA-AuxSeg, Geo-Instruct, and 4B. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly four equal columns labeled 已完成, 已构建, 待优化验证, and 下一阶段. The first column contains 4B 通用遥感训练 1000 步 and 独立对比评测 256 条. The second contains OA-AuxSeg, 三套数据集, 区域证据接口, and 指令模板. The orange third column contains 4B 区域训练因计算一致性问题暂缓 and 正式训练尚未启动. The purple dashed fourth column contains Geo-Instruct 数据集, 铁路场景样本补充, and 风险评估. Add the footer 状态截至 2026-09-02. Keep the layout sparse. Do not add progress percentages, completed regional training, completed domain training, railway field results, or additional milestones.
```

### 准确标签清单

- 已完成｜已构建｜待优化验证｜下一阶段
- 4B 通用遥感训练｜1000 步
- 独立对比评测｜256 条
- OA-AuxSeg｜三套数据集
- 区域证据接口｜指令模板
- 4B 区域训练因计算一致性问题暂缓
- Geo-Instruct 数据集｜铁路场景样本补充｜风险评估集成
- 状态截至 2026-09-02

### 本地参考图或真实素材

- `REBUILD_PROGRESS.md`，唯一实时进展、运行结果、产物身份与验收证据依据。
- 本页为新绘状态看板，不使用模拟进度条。

### 内部事实边界（不进入 PPT）

- 4B 通用训练完成与独立 Gate B 通过，不代表全部项目 Gate 已通过。
- OA-AuxSeg 的 Gate A、后续 Gate C 和 Gate D 均仍未执行或未验收。
- 下一阶段列是计划，不是已启动或已验收成果。

---

## 第20页｜下一阶段路线与项目验收目标

### 项目对应与研究状态

- 项目对应：T2.2、T2.3、T3.1、T3.2、T3.3 后续工作与终期验收
- 研究状态：下一阶段；专业知识增强／RAG 为探索性展望

### 页面标题与核心结论

- 页面标题：**下一阶段先恢复 4B 区域训练，再完成专业数据、领域微调和风险评估**
- 核心结论：后续依次处理训练计算一致性问题、建设专业数据、开展领域微调和风险评估，RAG 只作为探索方向。

### 上屏文字

1. 查明并修正 4B 区域训练的计算一致性问题，完成小规模和中等规模验证
2. 建设 Geo-Instruct 和铁路场景样本，开展领域微调与风险评估
3. 项目验收目标：mIoU≥75%｜分类准确率≥90%｜风险描述 ROUGE-L≥0.4

### 主视觉布局

中央为三阶段右向路线：第一阶段恢复 4B 区域训练，第二阶段建设领域数据并开展微调和风险评估，第三阶段
探索专业知识增强 RAG。第一阶段用橙色转蓝色表示从问题修正进入验证，第二阶段用紫色虚线，最右端 RAG 使用更淡的
紫色虚线并标注探索性展望。底部单独放三个浅灰目标卡，统一标注项目验收目标，不做完成度展示。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for 4B, Geo-Instruct, RAG, mIoU, and ROUGE-L. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly three future work areas from left to right. The first is 4B 区域训练恢复与验证 and contains 修正计算一致性问题, 小规模验证, 中等规模验证, 正式训练复核. The second is 专业数据与风险评估 and contains Geo-Instruct 与铁路样本, 领域微调, 风险评估. The third is a light purple dashed area labeled 专业知识增强 RAG and 探索性展望. Use Figure 5 only as a layout reference and do not copy its English labels or current system meaning. Along the bottom place exactly three gray target cards under 项目验收目标: 分割 mIoU≥75%, 要素分类准确率≥90%, 风险描述 ROUGE-L≥0.4. Keep the layout sparse. Do not add achieved metrics, completed RAG, railway field results, or unspecified modules.
```

### 准确标签清单

- 4B 区域训练恢复与验证
- 修正计算一致性问题｜小规模验证｜中等规模验证｜正式训练复核
- 专业数据与风险评估
- Geo-Instruct 与铁路样本｜领域微调｜风险评估集成
- 专业知识增强 RAG｜探索性展望
- 项目验收目标
- 分割 mIoU≥75%
- 要素分类准确率≥90%
- 风险描述 ROUGE-L≥0.4

### 本地参考图或真实素材

- `/home/yukun80/codes/paper7_VLM/docs/PPT/figure/Figure 5 — Evidence-Constrained Geohazard RAG.png`，仅作最右端展望模块的版式参考。
- `docs/实施方案v4.docx`，三个终期指标的项目口径来源。
- `REBUILD_PROGRESS.md`，确定下一阶段起点和当前未完成边界。

### 内部事实边界（不进入 PPT）

- 专业知识增强／RAG 尚不是当前已完成系统组件，也没有可归因的科学增益。
- 三个数值只表示项目验收目标，不表示当前模型已经达到。
- 4B 区域训练、Geo-Instruct 正式数据集、领域微调和铁路风险集成均需后续验证。

---
