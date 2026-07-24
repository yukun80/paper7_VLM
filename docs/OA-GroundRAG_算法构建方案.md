# OA-GroundRAG 算法构建方案

> **路线全称：** Optical-Anchored Arbitrary-Auxiliary Segmentation, Region Grounding and Retrieval-Augmented Understanding for Landslides  
> **中文名称：** 光学锚定任意辅助模态滑坡分割、区域指代与检索增强理解  
> **建议保存路径：** `docs/OA-GroundRAG_算法构建方案.md`  
> **文档用途：** 作为 Codex Agent 从代码清理、Benchmark 构建、分割模型、区域指代、VLM 描述到 RAG 集成的唯一实施说明。  
> **硬件边界：** 单张约 24 GB GPU；正式长训练由人工启动，Codex 只实现程序、运行短测试并给出正式命令。

## 1. 路线概述

本研究不再构建一个同时承担任意通道输入、异尺寸融合、像素级分割、指令理解和长文本生成的单体模型，而是将问题拆成四个依次实现、可以独立训练和评价的阶段：

```text
Phase 1：OA-AuxSeg
输入：光学影像 + 任意可用辅助模态
输出：global mask + candidate regions + region features

Phase 2：Region Grounding Adapter
输入：candidate regions + text instruction
输出：selected mask / no-target

Phase 3：VLM Description
输入：selected mask + multimodal evidence
输出：区域结构化描述、自然语言摘要和问答结果

Phase 4：RAG
输入：evidence + retrieved knowledge
输出：带可追溯知识依据的专业回答
```

四个阶段通过清晰接口连接，不进行一次性端到端联合训练。

总体数据流为：

```text
本地 HDF5 滑坡数据
        ↓
统一 Benchmark：固定 patch 大小、统一重采样、统一数据合同
        ↓
OA-AuxSeg 多源分割
        ↓
全局 mask、候选区域、区域特征
        ↓
Region Grounding Adapter
        ↓
用户指令对应的 selected mask
        ↓
Mask-Grounded Multimodal Evidence
        ↓
Qwen3-VL 区域描述与问答
        ↓
RAG 检索专业知识和相似案例
        ↓
证据受限的最终回答
```

## 2. 固定研究边界

### 2.1 必要模态

每个正式分割样本必须包含光学影像。光学影像负责：

- 定义参考画布；
- 提供滑坡边界和纹理；
- 输出最终 mask；
- 提供候选区域的主要视觉特征。

### 2.2 任意辅助模态

可选辅助模态包括但不限于：

- 多波段光学或 Sentinel-2；
- SAR；
- InSAR；
- DEM；
- slope。

“任意辅助模态”表示已注册辅助模态集合中的任意可用子集。光学永不缺失，辅助模态允许全部缺失。

### 2.3 明确排除

不实施：

- 无光学输入的主线分割；
- 灾前灾后变化检测；
- 时间差分和目标跟踪；
- Any2Seg 式 VLM 蒸馏；
- 分割、指代、描述和 RAG 的联合反向传播；
- 任意未知传感器自动接入；
- 旧模型、旧 Benchmark 和旧 checkpoint 兼容层；
- 根据单时相影像自由推断触发因素、发生时间、运动速度、未来失稳概率和风险等级。

## 3. 实施总原则

1. `../datasets` 中的数据视为已经统一为 HDF5 格式的原始研究数据，但其内部字段、通道数、样本大小、数值范围和 mask 编码仍需由 Codex 实际审计。
2. Benchmark 构建时通过运行参数指定固定目标 patch 大小，例如 224×224。不同数据集的原始 patch 统一上采样或下采样到该尺寸后再混合训练。
3. 不在方案中预设 HDF5 字段名称、归一化参数和数据源配置。Codex 必须先读取真实数据结构，再生成对应实现参数。
4. 不在项目开始时固定最终目录结构。先按功能模块和依赖顺序实现，代码结构在接口稳定后再整理。
5. `../external` 中的 CMNeXt、Grasp-Any-Region、PSALM 和 Qwen-VL-Series-Finetun 仅作为只读参考。新代码不得直接 import 整个外部工程。
6. 每一阶段必须先完成最小闭环、短测试和独立验收，再进入下一阶段。
7. Codex 不自动运行正式长训练。遇到正式训练节点时给出可复制命令、输入、输出和验收标准。
8. 只保留一个简洁进度文件 `REBUILD_PROGRESS.md`，不生成大量 ADR、handoff、gate 和中间治理文档。

## 4. 阶段 0：理解任务与清理当前代码库

### 4.1 目标

移除与 OA-GroundRAG 无关的旧活动实现，建立干净的重新实现起点。

### 4.2 Codex 首先需要完成的审计

Codex 应读取：

- 本文档；
- 根目录 `README.md`；
- 根目录 `AGENTS.md`；
- 当前代码树；
- `../external`；
- `../datasets` 的目录概况。

只需回答：

- 当前有哪些旧模型、旧数据管线、旧 Trainer、旧 CLI 和旧测试；
- 哪些内容与新路线直接冲突；
- 哪些通用工具仍可保留；
- 当前工作区是否存在未提交人工修改。

### 4.3 删除范围

应删除的旧活动实现包括：

- 旧的多源分割模型；
- 旧的分割—描述统一模型；
- 旧 Benchmark 和 instruction/Bridge/SegDesc 构建脚本；
- 旧视觉 cache 和文本 cache 协议；
- 旧 Trainer、evaluator 和推理入口；
- 旧配置；
- 旧模型测试；
- 旧算法说明和失效命令。

### 4.4 保留范围

必须保留：

- `../datasets`；
- `../external`；
- 仍需使用的模型权重；
- 本文档；
- Git 历史；
- 与具体旧模型无关且经审查可复用的通用工具。

### 4.5 清理红线

- 不建立 `legacy/` 目录；
- 不编写旧类名 alias；
- 不编写旧配置转换器；
- 不编写旧 checkpoint 兼容加载器；
- 不让新代码 import 旧包；
- 如果工作区存在不属于本任务的未提交修改，停止并报告。

### 4.6 阶段 0 验收

- 当前活动代码中不再存在旧主线入口；
- README 不再展示旧运行命令；
- `REBUILD_PROGRESS.md` 已建立；
- 新方案文档成为后续实现依据；
- 暂未开始 Benchmark 和模型实现。

## 5. 阶段 1A：真实 HDF5 数据审计

### 5.1 目标

在编写 Benchmark builder 前，先确定所有真实数据源的结构和可用性。

### 5.2 审计内容

Codex 必须只读遍历 `../datasets` 中候选 HDF5 文件，统计：

- 文件数量和数据源；
- group 和 dataset 层级；
- 每个 dataset 的 shape、dtype 和属性；
- 样本数量；
- 光学通道数；
- 辅助模态类型和通道数；
- 原始 patch H×W 分布；
- mask 字段和取值范围；
- 空 mask 数量和前景比例；
- NaN、Inf、nodata、全零、常量通道；
- 光学、辅助模态和 mask 是否逐样本对应；
- 同一 parent 内不同模态是否表示相同地理范围；
- 是否存在显式 valid mask；
- 是否存在 source scene、event、region 或 parent 分组信息。

### 5.3 审计产物

只生成一份简洁的数据审计报告，记录真实观察结果和待人工确认项。不要生成许可证报告，也不要研究数据授权。

### 5.4 停止条件

遇到下列情况才停止：

- 无法判断哪个字段是光学；
- 无法判断哪个字段是 mask；
- 模态与 mask 无法配对；
- 同一记录中的模态是否同空间范围无法确定；
- mask 编码无法解释。

普通字段差异由 source adapter 解决，不应成为停止原因。

## 6. 阶段 1B：统一 Benchmark 构建

### 6.1 Benchmark 的定位

HDF5 数据是统一格式的原始数据；Benchmark 是为训练重新组织的固定尺寸样本集合。

Benchmark builder 必须接收一个目标 patch 大小参数：

```text
--patch-size N
```

N 可以为 224、256 或后续实验尺寸，代码中不得写死。

### 6.2 样本统一流程

```text
读取一个 HDF5 原始样本
        ↓
识别光学、mask 和可用辅助模态
        ↓
验证同一空间范围和有效区域
        ↓
根据目标 patch 大小统一上采样或下采样
        ↓
统一输出形状
        ↓
写入 Benchmark 或可随机读取的索引
```

### 6.3 Resize 策略

Codex 根据实际尺寸分布决定是否采用直接 resize 或保持比例后 padding，但必须遵守：

- 连续影像使用双线性或双三次插值；
- mask 和 valid mask 使用最近邻；
- resize 后 mask 重新二值化；
- nodata 不得通过插值污染有效区域；
- DEM、InSAR 等只改变空间采样，不改变单位；
- 保存原始尺寸、目标尺寸和空间变换信息；
- 同一样本所有 `aligned_dense` 模态最终得到相同 H×W。

### 6.4 大样本与已有 patch

如果 HDF5 记录本身已经是独立 patch，则直接统一 resize。

如果某些记录明显大于普通 patch，且直接缩小会导致滑坡消失，则 Codex 应先实现窗口切分，再将窗口统一到目标 patch 大小。

最终决定必须根据实际数据审计，不预先假设所有记录都是整图或切片。

### 6.5 数据划分

split 必须按原始 parent、scene、event 或 region group 进行，不能在统一 resize 后随机拆分。

同一 parent 的不同模态、不同 resize 版本、不同窗口和不同任务视图必须属于同一个 split。

### 6.6 Benchmark 样本合同

每个样本至少向 DataLoader 提供：

- optical；
- binary mask；
- auxiliary modality mapping；
- modality availability；
- 每个模态 valid mask；
- source ID；
- parent/group ID；
- 原始尺寸；
- 目标 patch 大小；
- resize 或窗口变换；
- 前景比例；
- split。

保存格式由 Codex 根据数据量和 I/O 性能决定，不在本文档预先固定。

### 6.7 Benchmark 验收

- 所有技术可用的数据源被扫描；
- 所有输出样本具有相同目标 patch H×W；
- 不同通道数通过数据合同保留；
- mask 与各模态空间一致；
- optical-only 和多辅助模态样本均可形成 batch；
- 同一 parent 不跨 split；
- 两次构建产生相同样本数量和索引摘要；
- DataLoader 可完成短批量迭代；
- 输出中没有机器绑定绝对路径。

## 7. Phase 1：OA-AuxSeg

### 7.1 任务定义

输入：

```text
optical + arbitrary available auxiliary modalities
```

输出：

```text
global mask
candidate regions
region features
no-target score
diagnostic modality weights
```

其中 `candidate regions` 是从最终语义 mask 中提取的候选滑坡区域，不宣称是人工实例。

### 7.2 实现顺序

OA-AuxSeg 必须分五步实现。

#### 步骤 1：光学分割基线

先实现纯光学二值滑坡分割：

```text
optical
→ hierarchical encoder
→ lightweight decoder
→ mask logits
```

要求：

- 只选择一个成熟 backbone 主线；
- 使用有效区域 BCE 和 Dice；
- 实现 checkpoint、评价和推理；
- 不加入辅助模态、质量选择、文本或 VLM。

验收：

- 32–64 样本可以过拟合；
- checkpoint reload 后输出一致；
- no-target 样本能输出近空 mask；
- IoU、Dice、Precision、Recall 和 F1 可计算；
- 短训练能在单卡运行。

#### 步骤 2：辅助模态输入适配

为实际审计确认存在的每种辅助模态建立轻量输入 adapter。

原则：

- adapter 解决不同通道数和基础数值统计；
- 后续辅助 encoder 尽可能共享；
- 不为每个模态复制完整大型 backbone；
- 不建立庞大的 sensor/band/orbit embedding；
- 缺失模态不使用固定全零张量冒充有效输入；
- 无效区域在第一层前被屏蔽。

#### 步骤 3：CMNeXt 式任意辅助模态注入

以光学特征为主，辅助模态提供增量信息：

```text
optical feature as query
auxiliary features as evidence
→ auxiliary aggregation
→ residual injection
→ enhanced optical feature
```

要求：

- 光学浅层高分辨率特征保持独立；
- 辅助信息在中高层注入；
- 注入为残差形式；
- 残差强度近零初始化；
- 支持 0、1 或多个辅助模态；
- 对辅助模态顺序保持不变性；
- 注入模块可完全关闭用于消融。

首版只实现一种最小注入算子，不同时维护多套复杂实现。

#### 步骤 4：简化 MAGIC Quality Selection

在辅助模态聚合前计算一次质量分数。

质量信息可以来源于：

- 辅助特征统计；
- valid coverage；
- HDF5 中真实存在的质量字段；
- resize 比例；
- 与光学特征的相容程度。

具体字段由数据审计决定。

要求：

- 对当前可用辅助模态进行 permutation-invariant 评分；
- 加入 null auxiliary 状态；
- 零覆盖或严重异常模态可被抑制；
- 全辅助缺失时退化为 optical-only；
- 不实现复杂离散 top-k；
- 质量权重不作为地学证据；
- 不在后续模块重复 reliability。

#### 步骤 5：完整训练和模态鲁棒性

训练时随机选择辅助模态子集：

```text
active_aux ⊆ available_aux
```

光学永远存在。

必须覆盖：

- optical-only；
- optical + 单一辅助模态；
- optical + 多辅助模态；
- optical + all available。

主 loss 首版保持 BCE + Dice。其他 loss 只有在明确问题出现后再增加。

### 7.3 OA-AuxSeg 的区域输出

最终语义 mask 在阈值化后进行确定性区域提取：

```text
semantic mask
→ connected-region extraction
→ small-region filtering
→ candidate regions
```

每个候选区域至少包含：

- region ID；
- binary mask；
- bbox；
- centroid；
- area；
- confidence；
- masked optical feature；
- masked fused feature；
- geometry feature；
- active modality summary。

`region feature` 由分割模型中稳定的空间特征进行 mask pooling 获得，不单独训练大型实例 proposal head。

如果相邻滑坡在语义 mask 中被合并，该限制必须记录；首版不通过复杂实例分割强行解决。

### 7.4 OA-AuxSeg 核心对照

至少实现：

1. optical-only；
2. optical + direct input concatenation；
3. optical + auxiliary mean fusion；
4. optical + CMNeXt-style injection；
5. injection + quality selection；
6. injection + quality selection + modality dropout。

### 7.5 Phase 1 验收

- Benchmark 和训练闭环可运行；
- optical-only 基线稳定；
- 任意实际辅助模态子集可 forward；
- 模态顺序不改变模态身份；
- 全辅助缺失退化为 optical-only；
- 可导出 global mask；
- 可导出 candidate regions；
- 可导出 region features；
- 可导出 no-target 分数；
- 正式训练命令准备完成。

## 8. Phase 2：Region Grounding Adapter

### 8.1 任务定义

输入：

```text
candidate regions + text instruction
```

输出：

```text
selected region mask
或 no-target
```

这一阶段实现语言驱动的目标选择，不重新训练像素级分割器。

### 8.2 数据构建

Region Grounding 数据必须在 OA-AuxSeg 稳定后构建。

训练样本包括：

- 图像和多模态输入；
- candidate region list；
- 文本指令；
- target region index 或 no-target；
- 对应 target mask。

数据来源按难度递进：

1. 确定性几何指令；
2. 视觉属性指令；
3. 简单上下文关系指令；
4. no-target 指令；
5. 困难负样本。

不得用未审核的自由生成长文本作为正式测试真值。

### 8.3 Adapter 结构

首版采用小型 region-text matching 结构：

```text
text instruction
→ frozen Qwen text/VLM representation
→ text projection

candidate region features
→ region projection

text-region interaction
→ candidate scores + no-target score
```

可以使用 cosine similarity + MLP、小型 cross-attention 或少层 Transformer scorer。不得一开始把 Qwen3-VL 全模型和 OA-AuxSeg 联合训练。

### 8.4 训练顺序

- G0：规则选择基线；
- G1：冻结分割器，训练 Grounding Adapter；
- G2：固定预测候选区域训练；
- G3：必要时对 Qwen 增加少量 LoRA。

### 8.5 Grounding 指标

- region selection accuracy；
- no-target accuracy；
- selected-mask IoU；
- selected-mask Dice；
- top-k region recall；
- text paraphrase consistency；
- distractor rejection；
- GT candidate 和 predicted candidate 分层结果。

### 8.6 Phase 2 验收

- 能从多个候选区域中选择用户指定区域；
- 无目标时能输出 no-target；
- 文本同义改写保持稳定；
- selected mask 与候选 region 一致；
- grounding 错误和分割错误可以分别统计；
- 不修改 OA-AuxSeg 的像素预测权重。

## 9. Phase 3：VLM Description

### 9.1 任务定义

输入：

```text
selected mask + multimodal evidence + user question
```

输出：

```text
structured region facts
natural-language description
question answer
```

### 9.2 Mask-Grounded Evidence Builder

每个 selected mask 转换为 VLM 输入证据，包括：

- 光学全图；
- 光学 mask overlay；
- 保留上下文的光学 region crop；
- 可对齐辅助模态的全图和区域图；
- 确定性几何事实；
- 模态可用性；
- valid coverage；
- 单位和 sign convention；
- 禁止推断列表。

以下字段由程序计算，VLM 不得修改：

- bbox；
- centroid；
- area；
- area ratio；
- image location；
- elongation；
- compactness；
- fragmentation；
- active modality list。

### 9.3 VLM 输入方式

第一版先使用 Qwen3-VL 原生多图输入：

```text
full optical
+ mask overlay
+ region crop
+ optional auxiliary views
+ deterministic facts
+ user question
```

GAR 式 RoI feature replay 作为后续增强项。只有原生多图在区域对应和细节理解上明显不足时，才实现标准 RoIAlign region replay，不直接复制整个 GAR 工程。

### 9.4 描述任务

首版支持：

- 描述 selected region；
- 说明区域位置和形态；
- 描述光学可见扰动；
- 说明地形、SAR 或 InSAR 是否提供支持；
- 列出可能混淆对象；
- 判断证据是否充分；
- 回答与该区域有关的有限问题。

### 9.5 描述约束

- 缺失模态对应字段输出 unavailable；
- 覆盖不足时输出 insufficient evidence；
- 无单位或 sign convention 时禁止定量物理结论；
- 单时相数据禁止推断发生时间；
- 无现场资料时禁止输出确定风险等级；
- 不把质量权重或 attention 当作专业证据。

### 9.6 训练顺序

- D0：Prompt-only baseline；
- D1：GT-mask description；
- D2：fixed-predicted-mask description；
- D3：必要时训练独立 Description LoRA。

Description LoRA 不与 OA-AuxSeg 联合训练。

### 9.7 评价

- structured field accuracy；
- target-status accuracy；
- modality attribution accuracy；
- unsupported claim rate；
- evidence sufficiency accuracy；
- expert factuality；
- mask-region consistency；
- no-target response correctness。

必须加入 mask swap、wrong-region mask、empty mask、modality removal 和 cross-parent region swap。

### 9.8 Phase 3 验收

- 能针对 selected mask 描述正确区域；
- mask 改变后描述相应改变；
- 移除某模态后不再生成该模态支持结论；
- 几何事实不被 VLM 改写；
- GT-mask 和 predicted-mask 结果分开报告；
- Qwen3-VL 训练不是分割模型的前置条件。

## 10. Phase 4：RAG

### 10.1 任务定义

输入：

```text
Mask-Grounded Evidence + user question + external knowledge
```

输出：

```text
knowledge-grounded description and answer
```

RAG 不参与像素分割，也不控制 OA-AuxSeg decoder。

### 10.2 知识类型

知识库至少分为：

1. 专业文本知识；
2. 专家审核滑坡案例；
3. 困难负样本和混淆案例；
4. SAR、InSAR、DEM 等模态解释规则。

### 10.3 检索设计边界

不使用 OpenCLIP，不照搬 Geo-MMRAG 的统一图文向量空间。

建议采用分索引检索：

- 专业文本使用适合中英文技术文档的文本 embedding，并结合关键词检索；
- 光学案例使用自监督视觉特征或 OA-AuxSeg 光学区域特征；
- SAR、InSAR、DEM 案例使用 OA-AuxSeg 对应辅助 encoder 的同模态区域特征；
- 各路检索结果在后期进行排序融合。

具体 embedding 模型由 Phase 4 开始时根据资源、语言和检索实验单独确定，不在当前阶段写死。

### 10.4 RAG 输入接口

Phase 3 的 prompt builder 必须预留 `retrieved evidence cards`。每条证据至少含：

- knowledge ID；
- 内容；
- 来源；
- 适用模态；
- 支持的 claim；
- 禁止的 claim；
- 相关性分数。

### 10.5 训练与评价

RAG 首先采用无需训练的检索增强生成。

核心对照：

1. no RAG；
2. text-only RAG；
3. text + optical case retrieval；
4. text + multimodal case retrieval；
5. full retrieval + evidence constraint。

指标：

- Recall@K；
- nDCG 或 MRR；
- evidence citation precision；
- expert relevance；
- unsupported claim rate；
- irrelevant knowledge robustness；
- confounder retrieval accuracy。

### 10.6 Phase 4 验收

- 知识库可构建、保存和重载；
- 检索结果可重复；
- 不同模态只进入正确的案例索引；
- 回答能返回知识来源；
- 无关知识不会明显改变正确结论；
- RAG 失败不影响分割和 Grounding 的独立运行。

## 11. 端到端推理

最终统一流程为：

```text
用户指令
    ↓
判断是否需要分割、区域选择、描述或知识问答
    ↓
OA-AuxSeg 输出 mask、regions、region features
    ↓
Region Grounding Adapter 选择 selected mask
    ↓
Evidence Builder 生成多模态区域证据
    ↓
可选 RAG 检索
    ↓
Qwen3-VL 生成最终回答
```

必须支持四种运行模式：

1. segmentation only；
2. segmentation + grounding；
3. segmentation + grounding + description；
4. segmentation + grounding + description + RAG。

## 12. 训练阶段总结

| 阶段 | 训练对象 | 冻结对象 | 主要监督 |
|---|---|---|---|
| OA-0 | 光学分割模型 | 无 | optical + mask |
| OA-1 | 辅助 adapter、辅助 encoder、注入模块 | 可保留光学预训练权重 | multimodal image + mask |
| OA-2 | quality selector | 已稳定分割主干可部分冻结 | mask + modality dropout |
| G-1 | Region Grounding Adapter | OA-AuxSeg、Qwen 主体 | text + candidate region target |
| G-2 | 可选 Qwen LoRA | OA-AuxSeg | text + predicted candidates |
| D-0 | 不训练 | OA-AuxSeg、Qwen | prompt-only |
| D-1 | 可选 Description LoRA | OA-AuxSeg、Grounding | mask-grounded expert description |
| R-0 | 不训练或仅训练 reranker | 分割与描述模型 | retrieval relevance |

不进行四阶段联合训练。

## 13. 最小论文实验矩阵

### 13.1 分割实验

- optical-only；
- direct concat；
- mean auxiliary fusion；
- CMNeXt-style injection；
- injection + quality selection；
- proposed + modality dropout。

### 13.2 Grounding 实验

- geometry rule；
- text-region similarity；
- trained Grounding Adapter；
- GT candidates；
- predicted candidates；
- no-target；
- paraphrase；
- distractor regions。

### 13.3 描述实验

- full image only；
- crop only；
- full + overlay + crop；
- multimodal evidence；
- optional RoI replay；
- GT mask；
- fixed predicted mask；
- end-to-end selected mask。

### 13.4 RAG 实验

- no RAG；
- text RAG；
- optical cases；
- multimodal cases；
- full evidence-constrained RAG。

## 14. 进度记录

只使用根目录 `REBUILD_PROGRESS.md`，内容保持简短：

- 当前阶段；
- 当前实现目标；
- 已完成内容；
- 主要新增、修改和删除文件；
- 已运行测试和结果；
- 当前阻塞；
- 下一条命令。

普通子步骤不生成独立 handoff，不生成大量 ADR、gate、license 或审计文档。

## 15. Codex 工作规则

Codex 每次开始工作时读取：

1. 本文档；
2. `REBUILD_PROGRESS.md`；
3. 根 README；
4. 根 AGENTS。

然后从当前阶段继续。

Codex 可在一个阶段内连续实现多个子步骤，不因普通子步骤结束而暂停。

仅在以下情况停止：

- HDF5 字段或通道含义无法确定；
- 光学、mask 或辅助模态无法配对；
- 模态空间关系无法确定；
- 需要覆盖原始数据；
- 需要人工标注或地学判断；
- 需要正式长训练；
- 测试失败且无法根据实际错误定位。

遇到正式训练节点时必须给出：

- 执行目录；
- 环境激活方式；
- 完整命令；
- 输入 Benchmark；
- patch 大小；
- checkpoint；
- 输出目录；
- 预期报告；
- 验收标准；
- 需要用户返回的日志。

## 16. 最终完成定义

项目完成必须同时满足：

1. 旧活动代码已从当前主线清理。
2. 已审计真实 HDF5 数据结构。
3. Benchmark 可通过参数统一不同原始 patch 到固定尺寸。
4. 多个数据源可混合形成训练 batch。
5. OA-AuxSeg 可在 optical-only 和任意辅助模态子集下运行。
6. 模型输出 global mask、candidate regions、region features 和 no-target。
7. Region Grounding Adapter 可根据文本选择目标 region 或 no-target。
8. Qwen3-VL 可基于 selected mask 和多模态证据生成描述和回答。
9. mask、region、modality 和文本错误可以分别评价。
10. RAG 可以独立启用和关闭。
11. 分割、Grounding、Description 和 RAG 均有独立评价入口。
12. 正式长训练由人工启动，Codex 已提供完整命令。
13. README 只保留 OA-GroundRAG 当前有效命令。
14. `REBUILD_PROGRESS.md` 记录全部阶段完成。
