# OA-GroundRAG 算法构建方案

> **路线全称：** Optical-Anchored Arbitrary-Auxiliary Segmentation and Mask-Grounded Retrieval-Augmented Understanding for Landslides
>
> **中文名称：** 光学锚定任意辅助模态滑坡分割与掩膜驱动检索增强理解
>
> **保存路径：** `docs/OA-GroundRAG_算法构建方案.md`
>
> **文档用途：** 作为 Codex Agent 从旧代码清理、HDF5 Benchmark 构建、OA-AuxSeg 分割、VLM 区域描述到 RAG 集成的唯一详细实施说明。
>
> **硬件边界：** 单张约 24 GB GPU；正式长训练由人工启动，Codex 只实现程序、运行短测试并给出正式命令。

## 1. 路线调整

本研究不再将 Region Grounding Adapter 作为独立核心阶段，也不以严格指代分割为当前论文主任务。新的主线由三个可独立训练和评价的阶段组成：

```text
Phase 1：OA-AuxSeg
输入：光学影像 + 任意可用辅助模态
输出：global mask + candidate regions + no-target + optional region features

Phase 2：Mask-Grounded VLM Description
输入：global mask 或明确指定的 candidate-region mask
     + multimodal evidence + question
输出：结构化事实、区域描述和问答

Phase 3：RAG
输入：Mask-Grounded Evidence + question + retrieved knowledge
输出：有来源、受证据约束的专业回答
```

三个阶段通过清晰接口连接，不进行一次性联合反向传播。总体数据流为：

```text
本地 HDF5 数据
        ↓
固定 patch 大小的统一 Benchmark
        ↓
OA-AuxSeg
        ↓
global mask / candidate regions / no-target / optional region features
        ↓
确定目标 mask：global / all regions / ID / 坐标 / 规则 / 编号 overlay
        ↓
Mask-Grounded Multimodal Evidence Builder
        ↓
Qwen3-VL 区域描述与问答
        ↓
可选 RAG 检索专业知识和相似案例
        ↓
证据受限且带来源的最终回答
```

路线名称中的“Ground”表示描述与回答必须绑定于明确的分割 mask，而不是必须训练文本指代分割网络。

## 2. 为什么取消独立 Grounding Adapter

严格指代分割需要额外构建 `image + text query + target mask` 数据，并处理多候选区域、文本关系、no-target、预测候选误差和专门评价。这会显著增加数据审核、训练和实验成本，却不是当前研究要回答的首要问题。

当前核心研究问题是：

1. 光学锚定条件下，任意辅助模态是否能稳定改善滑坡分割；
2. VLM 是否能在明确 mask 和多模态证据约束下生成可靠描述；
3. RAG 是否能引入可追溯专业知识，同时减少无依据结论。

首版区域选择直接采用确定性或可核验方式：

- 描述 global mask；
- 逐一描述全部 candidate regions；
- 按 `region_id` 指定区域；
- 按 bbox 或点击坐标指定区域；
- 按面积排序选择；
- 按九宫格位置规则选择；
- 给候选区域编号后，让 Qwen3-VL 在 overlay 中返回 `region_id`。

因此，Phase 1 完成后直接进入 Mask-Grounded VLM Description。只有三阶段主线完成，且真实应用证明自然语言区域选择是必要能力时，才考虑非阻塞的轻量 region scorer。

## 3. 固定研究边界

### 3.1 必要模态

每个正式分割样本必须包含光学影像。光学影像负责：

- 定义参考画布；
- 提供滑坡边界和纹理；
- 输出最终 mask；
- 提供候选区域的主要视觉证据。

### 3.2 任意辅助模态

可选辅助模态包括但不限于：

- 多波段光学或 Sentinel-2；
- SAR；
- InSAR；
- DEM；
- slope；
- 其他已明确物理意义和空间对应关系的数据。

“任意辅助模态”表示已注册辅助模态集合中的任意可用子集。光学永不缺失，辅助模态允许全部缺失。

### 3.3 当前不实施

- 无光学输入的主线分割；
- 灾前灾后变化检测；
- 时间差分和目标跟踪；
- Any2Seg 式 VLM 蒸馏；
- 分割、描述和 RAG 联合反向传播；
- 独立 Region Grounding Adapter；
- 大规模指代分割数据构建；
- 任意未知传感器自动接入；
- 旧模型、旧 Benchmark 和旧 checkpoint 兼容层；
- 根据单时相影像自由推断触发因素、发生时间、运动速度、未来失稳概率或确定风险等级。

## 4. 实施原则

1. `../datasets` 中的数据视为已统一为 HDF5 格式的原始研究数据，但其内部字段、通道数、尺寸、数值范围和 mask 编码必须由 Codex 实际审计。
2. Benchmark 构建时通过 `--patch-size` 指定统一目标尺寸；不同数据集的原始 patch 上采样或下采样后再混合训练。
3. 不预设 HDF5 字段名称、通道顺序、科学含义和归一化参数，所有 source 参数来自真实审计。
4. 不在项目开始时固定最终代码目录；先按依赖顺序实现，接口稳定后再整理。
5. `../external` 中 CMNeXt、Grasp-Any-Region、PSALM 和 Qwen-VL-Series-Finetun 只读参考，新代码不得直接依赖完整外部工程。
6. 每阶段先完成最小闭环、短测试和独立验收，再进入下一阶段。
7. Codex 不自动运行正式长训练，只给出可复制命令、输入、输出和验收标准。
8. 只保留 `REBUILD_PROGRESS.md` 作为活动进度文件，不生成大量 ADR、handoff、重复过程报告或许可证文档。
9. OA-AuxSeg、Description 和 RAG 保持可单独运行；下游失败不得改变上游输出。
10. 所有专业结论必须能追溯到输入影像、确定性事实或检索知识，模型内部 quality weight 和 attention 不作为地学证据。

## 5. Stage 0：清理旧代码

### 5.1 目标

移除与 OA-GroundRAG 三阶段主线无关的旧活动实现，建立干净的重新实现起点。

### 5.2 开始前审计

Codex 首先读取：

- 本文档；
- 根目录 `README.md`；
- 根目录 `AGENTS.md`；
- `REBUILD_PROGRESS.md`；
- 当前代码树；
- `../external`；
- `../datasets` 的目录概况。

审计只需确定：

- 当前有哪些旧模型、旧数据管线、旧 Trainer、旧 CLI 和旧测试；
- 哪些内容与新路线直接冲突；
- 哪些通用工具仍可保留；
- 当前工作区是否存在未提交人工修改。

### 5.3 删除范围

原则上删除：

- 旧多源分割模型；
- 旧分割—描述统一模型；
- 旧指代分割、Bridge 和 SegDesc；
- 旧 Benchmark builder；
- 旧 instruction、视觉 cache 和文本 cache 协议；
- 旧 Trainer、evaluator 和推理入口；
- 旧配置和旧模型测试；
- 旧算法说明和失效命令。

### 5.4 保留范围

必须保留：

- `../datasets`；
- `../external`；
- 仍需使用的模型权重；
- 本文档；
- Git 历史；
- `参考文献/` 和 `docs/archive/`；
- 经审查与旧模型无关的通用工具。

### 5.5 清理红线

- 不建立 `legacy/`；
- 不保留旧类名 alias；
- 不写旧配置转换器；
- 不兼容旧 checkpoint；
- 新代码不得 import 旧包；
- 不覆盖用户已有修改和正式训练产物；
- 工作区存在不属于本任务的未提交修改时，保留并报告；只有无法安全绕开时才停止。

### 5.6 Stage 0 验收

- 当前活动代码中不再存在旧主线入口；
- README 不再展示旧运行命令；
- `REBUILD_PROGRESS.md` 已建立；
- 本文档成为后续实现依据；
- 暂未开始 Benchmark 或模型实现。

## 6. Stage 1：HDF5 审计与统一 Benchmark

### 6.1 真实数据审计

在编写 Benchmark builder 前，Codex 必须只读遍历 `../datasets` 中候选 HDF5 文件，统计：

- HDF5 文件数量和数据源；
- group 和 dataset 层级；
- 每个 dataset 的 shape、dtype 和 attrs；
- 各数据源样本数量；
- 光学和辅助模态通道数；
- 原始 patch 的 H×W 分布；
- mask 字段、值域、空 mask 数量和前景比例；
- NaN、Inf、nodata、全零和常量通道；
- 光学、辅助模态和 mask 是否逐样本对应；
- 同一 parent 内不同模态是否表示相同地理范围；
- 是否存在显式 valid mask；
- 是否存在 source、scene、event、region 或 parent 分组信息。

审计只生成一份简洁报告，记录真实观察和待人工确认项，不研究数据授权，也不生成许可证报告。

只有出现以下情况才停止：

- 无法判断哪个字段是光学；
- 无法判断哪个字段是 mask；
- 模态与 mask 无法配对；
- 同一记录中模态是否覆盖相同空间无法确定；
- mask 编码无法解释。

普通字段差异由 source adapter 解决，不作为停止原因。

### 6.2 Benchmark 定位

HDF5 是统一容器格式的原始研究数据；Benchmark 是为训练重新组织的固定尺寸样本集合。Benchmark builder 必须接收：

```text
--patch-size N
```

`N` 可为 224、256 或后续实验尺寸，代码中不得写死。

### 6.3 样本统一流程

```text
读取 HDF5 原始样本
        ↓
识别 optical / mask / available auxiliaries
        ↓
验证空间对应和 valid area
        ↓
根据目标 patch size 上采样、下采样或窗口切分
        ↓
统一输出形状和数据合同
        ↓
写入 Benchmark 或可随机读取的索引
```

### 6.4 Resize 规则

Codex 根据实际尺寸分布决定直接 resize 或保持比例后 padding，但必须遵守：

- 连续影像使用 bilinear 或 bicubic；
- mask 和 valid mask 使用 nearest；
- mask resize 后重新二值化；
- nodata 不得通过插值污染有效区域；
- DEM、InSAR 等 resize 只改变空间采样，不改变物理单位；
- 保存原始尺寸、目标尺寸和空间变换；
- 同一样本所有 `aligned_dense` 模态最终具有相同 H×W。

### 6.5 大样本与已有 patch

若 HDF5 记录本身是独立 patch，直接统一 resize。

若记录明显大于普通 patch，且直接缩小会使小滑坡消失，则先窗口切分，再统一到目标 patch 大小。具体策略必须根据真实审计决定，不预设所有记录都是整图或切片。

### 6.6 数据划分

split 必须按原始 parent、scene、event 或 region group 进行，不能在统一 resize 后随机拆分。

同一 parent 的不同模态、不同 resize 版本、不同窗口和不同任务视图必须属于同一 split。

### 6.7 Benchmark 样本合同

每个样本至少向 DataLoader 提供：

- optical；
- binary mask；
- auxiliary modality mapping；
- modality availability；
- 每个模态的 valid mask；
- source ID；
- parent/group ID；
- 原始尺寸；
- 目标 patch 大小；
- resize 或窗口变换；
- 前景比例；
- split。

保存格式由 Codex 根据数据规模和 I/O 性能决定，不在本文档预先固定。

### 6.8 Benchmark 验收

- 所有技术可用的数据源被扫描；
- 所有输出样本具有相同目标 patch H×W；
- 不同通道数通过数据合同保留；
- mask 与各模态空间一致；
- optical-only 和多辅助模态样本均可组成 batch；
- 同一 parent 不跨 split；
- 两次构建产生相同样本数量和索引摘要；
- DataLoader 可完成短批量迭代；
- 输出不包含机器绑定绝对路径。

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
no-target score
diagnostic modality weights
optional region features
```

`candidate regions` 从最终语义 mask 中确定性提取，不宣称为人工实例。`region features` 是可选只读输出，不是 Phase 2 的前置条件。

### 7.2 Step 1：光学分割基线

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

### 7.3 Step 2：辅助模态输入适配

为真实审计确认存在的每种辅助模态建立轻量 input adapter。

原则：

- adapter 解决不同通道数和基础数值统计；
- 后续辅助 encoder 尽可能共享；
- 不为每个模态复制完整大型 backbone；
- 不建立庞大的 sensor、band 或 orbit embedding；
- 缺失模态不使用固定全零张量冒充有效输入；
- 无效区域在第一层前被屏蔽。

### 7.4 Step 3：CMNeXt 式任意辅助模态注入

以光学特征为主，辅助模态提供增量证据：

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
- 注入采用残差形式；
- 残差强度近零初始化；
- 支持 0、1 或多个辅助模态；
- 对辅助模态顺序保持不变性；
- 注入模块可完全关闭用于消融。

首版只实现一种最小注入算子，不同时维护多套复杂实现。

### 7.5 Step 4：简化 MAGIC Quality Selection

在辅助模态聚合前计算一次质量分数。质量信息可来自：

- 辅助特征统计；
- valid coverage；
- HDF5 中真实存在的质量字段；
- resize 比例；
- 与光学特征的相容程度。

具体字段由真实数据审计决定。

要求：

- 对当前可用辅助模态进行 permutation-invariant 评分；
- 加入 null auxiliary 状态；
- 零覆盖或严重异常模态可被抑制；
- 全辅助缺失时退化为 optical-only；
- 不实现复杂离散 top-k；
- 质量权重不作为地学证据；
- 不在后续模块重复 reliability。

### 7.6 Step 5：完整训练与模态鲁棒性

训练时随机选择辅助模态子集：

```text
active_aux ⊆ available_aux
```

光学永远存在。训练必须覆盖：

- optical-only；
- optical + 单一辅助模态；
- optical + 多辅助模态；
- optical + all available。

主 loss 首版保持 BCE + Dice。其他 loss 只有在明确问题出现后才增加。

### 7.7 候选区域与只读特征接口

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
- active modality summary。

如需下游分析或可选扩展，可额外只读导出：

- masked optical feature；
- masked fused feature；
- geometry feature。

region feature 由分割模型中稳定的空间特征进行 mask pooling 获得，不单独训练大型实例 proposal head。导出接口不得改变默认 forward、分割数值、loss、checkpoint schema 或训练逻辑。

如果相邻滑坡在语义 mask 中被合并，该限制必须记录；首版不通过复杂实例分割强行解决。

### 7.8 核心对照

至少实现：

1. optical-only；
2. optical + direct input concatenation；
3. optical + auxiliary mean fusion；
4. optical + CMNeXt-style injection；
5. injection + quality selection；
6. injection + quality selection + modality dropout。

### 7.9 Phase 1 验收

- Benchmark 和训练闭环可运行；
- optical-only 基线稳定；
- 任意真实辅助模态子集可 forward；
- 模态顺序不改变模态身份；
- 全辅助缺失退化为 optical-only；
- 可导出 global mask；
- 可导出 candidate regions；
- 可导出 no-target 分数；
- region features 在需要时可通过稳定只读接口导出；
- 正式训练命令准备完成。

## 8. Phase 2：Mask-Grounded VLM Description

### 8.1 任务定义

输入：

```text
global mask 或明确指定的 candidate-region mask
+ multimodal evidence
+ user question
```

输出：

```text
structured region facts
natural-language description
question answer
```

Phase 2 不训练独立 Region Grounding Adapter。进入 Evidence Builder 的目标 mask 必须是 global mask、已有 candidate-region mask 或空 mask，不新增像素级 decoder。

### 8.2 区域选择

首版支持：

- 使用 global mask；
- 逐一处理全部 candidate regions；
- 按 `region_id` 指定；
- 按 bbox 或点击坐标匹配已有候选；
- 按面积排序选择；
- 按九宫格位置规则选择；
- 将候选编号绘制在 overlay 上，由 Qwen3-VL 返回 `region_id`。

所有选择方式最终只能返回已有候选 mask 或空 mask。区域选择错误单独记录，但不作为主论文的指代分割任务，也不阻塞 Description。

### 8.3 Mask-Grounded Evidence Builder

目标 mask 转换为 VLM 输入证据，包括：

- 光学全图；
- 光学 mask overlay；
- 保留上下文的光学 region crop；
- 可对齐辅助模态的全图和区域图；
- 确定性几何事实；
- 模态可用性；
- valid coverage；
- 单位和 sign convention；
- 禁止推断列表。

以下字段由程序确定性计算，VLM 不得改写：

- bbox；
- centroid；
- area；
- area ratio；
- image location；
- elongation；
- compactness；
- fragmentation；
- active modality list。

空 mask 和 no-target 必须作为显式状态进入 Evidence Builder，不得伪造区域 crop 或几何事实。

### 8.4 VLM 输入方式

第一版使用 Qwen3-VL 原生多图输入：

```text
full optical
+ mask overlay
+ region crop
+ optional auxiliary views
+ deterministic facts
+ user question
+ optional retrieved evidence cards
```

GAR 式 RoI feature replay 只作为后续增强。只有原生多图不能稳定关注目标区域或局部细节时，才实现标准 RoIAlign 的轻量 region replay，不复制整个 GAR 工程。其目的仅是同时保持全局上下文与区域细节。

### 8.5 描述任务与证据边界

首版支持：

- 描述 global mask 或单个 candidate region；
- 说明区域位置、大小和形态；
- 描述光学可见扰动；
- 说明地形、SAR 或 InSAR 是否提供支持；
- 列出可能混淆对象；
- 判断证据是否充分；
- 回答与目标区域有关的有限问题。

严格约束：

- 缺失模态对应字段输出 `unavailable`；
- 覆盖不足时输出 `insufficient evidence`；
- 无单位或 sign convention 时禁止定量物理结论；
- 单时相数据禁止推断发生时间；
- 无现场资料时禁止输出确定风险等级；
- quality weight 和 attention 不作为专业证据；
- deterministic facts 不得由 VLM 重算或改写。

### 8.6 训练顺序

- D0：冻结 Qwen3-VL 的 prompt-only baseline；
- D1：GT-mask description；
- D2：fixed predicted-mask description；
- D3：只有在 prompt-only 明确不足时训练独立 Description LoRA。

Description LoRA 不与 OA-AuxSeg 联合训练，也不反向修改分割输出。

### 8.7 评价与对照

指标：

- structured field accuracy；
- target-status accuracy；
- modality attribution accuracy；
- unsupported claim rate；
- evidence sufficiency accuracy；
- expert factuality；
- mask-region consistency；
- no-target response correctness。

反事实与鲁棒性检查：

- mask swap；
- wrong-region mask；
- empty mask；
- modality removal；
- cross-parent region swap。

核心输入对照：

- full image only；
- crop only；
- full + overlay + crop；
- multimodal evidence；
- optional RoI replay；
- GT mask；
- fixed predicted mask；
- end-to-end predicted mask。

GT-mask、fixed predicted-mask 和 end-to-end predicted-mask 必须分开报告，避免把分割误差混入 Description 能力结论。

### 8.8 Phase 2 验收

- 能描述 global mask 和单个 candidate region；
- mask 改变后描述相应改变；
- 移除某模态后不再生成该模态支持结论；
- deterministic facts 不被 VLM 改写；
- 空 mask 和 no-target 得到正确响应；
- GT-mask 与 predicted-mask 结果分开报告；
- Qwen3-VL 训练不是分割模型前置条件；
- 不依赖指代分割数据；
- 不依赖独立 Grounding Adapter。

## 9. Phase 3：RAG

### 9.1 任务定义

输入：

```text
Mask-Grounded Evidence + user question + retrieved knowledge
```

输出：

```text
knowledge-grounded description and answer with sources
```

RAG 不参与像素分割，不控制 OA-AuxSeg decoder，也不负责生成或选择 mask。

### 9.2 知识类型

知识库至少区分：

1. 专业文本知识；
2. 专家审核滑坡案例；
3. 困难负样本和混淆案例；
4. SAR、InSAR、DEM 等模态解释规则。

### 9.3 检索设计边界

不使用 OpenCLIP，不照搬统一图文向量空间。建议采用分索引检索：

- 专业文本使用适合中英文技术文档的文本 embedding，并结合关键词检索；
- 光学案例使用自监督视觉特征或 OA-AuxSeg 光学区域特征；
- SAR、InSAR、DEM 案例使用 OA-AuxSeg 对应辅助 encoder 的同模态区域特征；
- 各路检索结果在后期进行排序融合。

具体 embedding 模型在 Phase 3 开始时根据资源、语言和检索实验单独评估，不在当前阶段写死。

### 9.4 RAG 输入接口

Phase 2 的 prompt builder 预留 `retrieved evidence cards`。每条证据至少包含：

- knowledge ID；
- 内容；
- 来源；
- 适用模态；
- 支持的 claim；
- 禁止的 claim；
- 相关性分数。

检索证据只能补充或解释当前 mask-grounded evidence，不能覆盖确定性几何事实和模态可用性。

### 9.5 训练与评价

RAG 首先采用无需训练的检索增强生成；只有检索排序明显不足时才考虑轻量 reranker。

核心对照：

1. no RAG；
2. text-only RAG；
3. text + optical case retrieval；
4. text + multimodal case retrieval；
5. full retrieval + evidence constraints。

指标：

- Recall@K；
- nDCG 或 MRR；
- evidence citation precision；
- expert relevance；
- unsupported claim rate；
- irrelevant knowledge robustness；
- confounder retrieval accuracy。

### 9.6 Phase 3 验收

- 知识库可构建、保存和重载；
- 检索结果可重复；
- 不同模态只进入正确的案例索引；
- 回答返回知识来源；
- 无关知识不明显改变正确结论；
- 检索证据不覆盖确定性输入事实；
- RAG 失败不影响分割和 Description 的独立运行。

## 10. 可选扩展：轻量区域指令选择

该扩展不是主线阶段，不是论文必需实验，也不是 Description 或 RAG 的验收条件。仅在以下条件同时具备时考虑：

- 单图多滑坡且用户必须通过自然语言选择；
- 确定性 ID、坐标、规则和 Qwen 编号 overlay 仍不能满足需求；
- 已有足够且经过审核的文本—区域配对数据。

可选结构：

```text
candidate region features + text embedding
→ lightweight scorer
→ selected region / no-target
```

实现时必须：

- 冻结 OA-AuxSeg；
- 只使用已有候选区域；
- 不新增或重训像素 decoder；
- 不改变默认分割 forward、loss 和 checkpoint；
- 不升级为大规模指代分割模型；
- 不与 Description 或 RAG 联合反向传播；
- 不阻塞三阶段主线。

若无需自然语言区域选择，region features 继续只作为可选只读接口，不为该扩展预先构建正式 cache。

## 11. Codex 实施顺序

Codex 按以下依赖顺序实施：

1. 清理旧活动实现；
2. 审计真实 HDF5；
3. 构建固定 patch Benchmark；
4. 实现 optical-only baseline；
5. 实现辅助 adapters 和共享 encoder；
6. 实现 CMNeXt 式注入；
7. 实现简化 MAGIC quality selection；
8. 冻结并验收 OA-AuxSeg，导出 mask、regions 和 no-target；
9. 实现 global、ID、坐标、规则和编号 overlay 等区域选择；
10. 实现 Mask-Grounded Evidence Builder；
11. 实现 Qwen3-VL Description；
12. 在分割和描述稳定后实现 RAG；
13. 仅在主线完成且确有必要时实现轻量区域指令选择。

每阶段内部连续执行，不因普通子步骤完成而暂停。

训练边界如下：

| 阶段 | 训练对象 | 冻结对象 | 主要监督 |
|---|---|---|---|
| OA-0 | 光学分割模型 | 无 | optical + mask |
| OA-1 | 辅助 adapter、辅助 encoder、注入模块 | 可保留光学预训练权重 | multimodal image + mask |
| OA-2 | quality selector | 已稳定分割主干可部分冻结 | mask + modality dropout |
| D-0 | 不训练 | OA-AuxSeg、Qwen3-VL | prompt-only |
| D-1 | 可选 Description LoRA | OA-AuxSeg | mask-grounded expert description |
| R-0 | 不训练或仅训练 reranker | 分割与描述模型 | retrieval relevance |

不得将以上阶段合并为联合训练。

## 12. 进度与停止规则

只使用根目录 `REBUILD_PROGRESS.md`，保持简洁并记录：

- 当前阶段；
- 当前实现目标；
- 已完成内容；
- 主要新增、修改和删除文件；
- 已运行测试及结果；
- 当前阻塞；
- 下一条命令。

普通子步骤不生成独立 handoff、ADR、许可证或重复审计文档。

Codex 每次开始工作时读取：

1. 本文档；
2. `REBUILD_PROGRESS.md`；
3. 根目录 `README.md`；
4. 根目录 `AGENTS.md`。

Codex 仅在以下情况停止：

- HDF5 字段或通道含义无法确定；
- optical、mask 或 auxiliary 无法配对；
- 模态空间关系无法确认；
- 需要覆盖原始数据或正式产物；
- 需要人工标注或地学判断；
- 需要正式长训练；
- 测试失败且无法从实际错误定位。

正式训练节点必须给出：

- 执行目录；
- 环境激活方式；
- 完整命令；
- 输入 Benchmark；
- patch size；
- checkpoint；
- 输出目录；
- 预期报告；
- 验收标准；
- 需要用户返回的日志。

## 13. 最终完成定义

主线完成必须同时满足：

1. 旧活动代码已从当前主线清理；
2. 已审计真实 HDF5 数据结构；
3. Benchmark 可通过参数统一不同原始 patch 到固定尺寸；
4. 多个数据源可混合组成训练 batch；
5. OA-AuxSeg 可在 optical-only 和任意辅助模态子集下运行；
6. OA-AuxSeg 输出 global mask、candidate regions 和 no-target，并可选只读导出 region features；
7. 区域可通过 global、all regions、ID、坐标、规则或 Qwen 编号 overlay 明确指定；
8. Evidence Builder 可根据 global mask、candidate-region mask 或空 mask 构建多模态证据；
9. Qwen3-VL 可基于 mask 和证据生成结构化描述与问答；
10. GT-mask、fixed predicted-mask 和 end-to-end predicted-mask 可分别评价；
11. 模态缺失或证据不足时不生成对应专业结论；
12. RAG 可独立启用和关闭，且回答返回知识来源；
13. 主线不依赖独立 Region Grounding Adapter；
14. 可选轻量区域指令选择不阻塞主线；
15. 正式长训练由人工执行，Codex 提供完整命令；
16. README 只保留 OA-GroundRAG 当前有效命令；
17. `REBUILD_PROGRESS.md` 记录三阶段主线完成。
