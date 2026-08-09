# OA-GroundRAG 算法完整构建方案（重构版）

> **路线全称：** Optical-Anchored Arbitrary-Auxiliary Segmentation and Mask-Grounded Retrieval-Augmented Understanding for Landslides
> **中文名称：** 光学锚定任意辅助模态滑坡分割与掩膜驱动检索增强理解
> **版本：** 2.0
> **日期：** 2026-08-01
> **文档定位：** 本文件已替代旧版 OA-GroundRAG 方案，是后续 Codex Agent 继续开发、实验和整合的唯一算法实施依据。
> **硬件边界：** 单张约 24 GB GPU；正式长训练由项目负责人启动，Coding Agent 只运行短测试、工程门和有界 smoke。
> **当前状态：** Stage 3 RS-General Adapter Gate B completed and accepted / Stage 4 Landslide Evidence Corpus and OA-GroundedEval pending; Gate A, ablations, sealed test and formal fixed masks remain deferred

---

## 1. 研究定位与本轮核心决策

本研究不再以“构建一个端到端统一训练、同时完成多源分割、指代分割、滑坡描述和知识问答的单体大模型”为目标。

新的主线采用模块化多任务框架：

```text
多源遥感图像 + 用户指令
            ↓
任务控制与输入检查
            ↓
OA-AuxSeg 专用多源滑坡分割
            ↓
global mask / candidate regions / no-target
            ↓
Mask-Grounded Evidence Builder
            ↓
通用遥感 VLM 进行视觉观察
            ↓
滑坡知识与案例 RAG 检索
            ↓
VLM 生成证据受限的描述、问答和报告
```

本轮冻结以下决定：

1. **光学影像是滑坡分割必要主模态。** SAR、InSAR、DEM、坡度和多波段影像是任意可选辅助模态。
2. **OA-AuxSeg 与 VLM 保持独立。** 分割模型负责像素空间感知；VLM 负责语言理解和区域视觉理解；RAG 负责专业知识解释。
3. **不将严格指代分割设为核心任务。** 不默认构建大规模“自然语言指令—滑坡 mask”训练集，也不训练独立 Region Grounding Adapter。
4. **通用遥感描述微调与滑坡专业知识分开。** 通用遥感数据用于训练 RS-General Description Adapter；滑坡规范、环境差异、传感器解释和典型案例主要进入 RAG。
5. **不默认构建大规模真实滑坡 mask-grounded 训练集。** 先使用通用遥感 VLM、mask evidence 和 RAG 完成零样本或少样本验证。只有 mask 对应或证据约束能力不足时，才启用小规模 Landslide-Evidence Adapter。
6. **必须保留真实滑坡 mask-grounded 评价集。** 不训练不等于不评价。必须用人工审核的 OA-GroundedEval 验证模型是否真正关注 mask、正确使用辅助模态并拒绝不支持结论。
7. **RAG 不直接替分割结果寻找理由。** 分割输出始终称为“候选滑坡区域”，最终系统需要同时报告支持证据、反对证据、混淆对象和证据限制。

---

## 2. 论文问题与核心贡献

### 2.1 核心科学问题

给定一幅必要光学影像和任意可用的辅助遥感观测，能否：

1. 通过光学主分支、任意辅助模态注入和质量选择稳定分割疑似滑坡；
2. 将分割 mask 转换为可审计的多模态区域证据；
3. 利用通用遥感 VLM 识别当前区域的可见特征；
4. 根据环境、传感器、数据质量和相似案例检索滑坡专业知识；
5. 生成只陈述当前输入和检索证据所支持事实的描述与报告？

### 2.2 推荐论文贡献

#### 贡献一：光学锚定任意辅助模态滑坡分割

构建 OA-AuxSeg，在光学必要条件下支持 DEM、坡度、InSAR、SAR 和多光谱等任意辅助模态子集，并处理缺失模态、有效覆盖和质量差异。

#### 贡献二：Mask-Grounded Multimodal Evidence Interface

建立从专业分割模型到 VLM 的标准接口，将 mask、区域视觉输入、确定性几何事实、辅助模态统计、可用性和禁止结论组织为可审计 Evidence Packet。

#### 贡献三：环境与传感器条件化的滑坡知识—案例 RAG

将滑坡规范、专业论文、传感器解释规则、正案例和困难负样本组织为分层知识库，根据当前环境、模态和候选区域进行检索，减少大规模滑坡领域微调需求。

---

## 3. 整体算法框架

### 3.1 统一推理流程

```text
用户指令
    ↓
Task Controller
    ├── 仅分割
    ├── 分割并描述
    ├── 评价候选区域
    ├── 解释多模态证据
    ├── 生成报告
    └── 纯知识问答
    ↓
输入检查与模态注册
    ↓
OA-AuxSeg
    ↓
global mask / candidate regions / no-target
    ↓
目标区域确定
    ├── global mask
    ├── 全部 candidate regions
    ├── region_id
    ├── bbox / 点击坐标
    ├── 面积或位置规则
    └── 编号 overlay 上由 VLM 返回 region_id
    ↓
Mask-Grounded Evidence Builder
    ↓
第一遍 VLM：视觉观察
    ↓
RAG Query Builder
    ↓
文本规则检索 + 同域案例检索 + 困难负样本检索
    ↓
第二遍 VLM：专业解释、问答和报告
```

### 3.2 模块职责边界

| 模块 | 唯一职责 | 不承担的职责 |
|---|---|---|
| Task Controller | 解析用户任务并编排模块 | 不输出像素 mask |
| OA-AuxSeg | 滑坡语义分割和候选区域提取 | 不解释成因，不访问知识库 |
| Evidence Builder | 计算确定性事实并组织区域证据 | 不自由生成专业结论 |
| RS-General VLM | 通用遥感场景和区域视觉理解 | 不替代专业分割模型 |
| Landslide RAG | 检索专业知识、案例和限制条件 | 不生成 mask，不改写确定性事实 |
| Final Generator | 综合视觉证据和检索知识生成回答 | 不把候选 mask 当作已确认滑坡 |

---

## 4. 当前代码资产与新方案的对应关系

### 4.1 `oa_groundrag/phase2`

继续作为 **OA-AuxSeg** 主实现。

保留：

- 光学主分支；
- 多通道输入；
- 辅助模态 registry；
- CMNeXt 式任意辅助模态注入；
- 简化 MAGIC quality selection；
- modality dropout；
- mask、no-target、candidate regions 和 region features；
- 训练、评价、checkpoint 和推理。

需要继续验证：

- optical-only 基线；
- direct concat；
- mean fusion；
- CMNeXt injection；
- quality selection；
- proposed dropout；
- 不同模态组合与低质量模态子集；
- full 正式训练和多随机种子。

### 4.2 `oa_groundrag/phase3`

原生产品定义为：

> **RS-GeneralDesc Benchmark**

只负责整合通用遥感描述和问答数据，用于训练与监控 RS-General Description Adapter。

该产品不包含 OA 真实 mask-grounded 训练记录。Stage 2 原生合同为：

- manifest：`rs_generaldesc.manifest.v1`；
- canonical：`rs_generaldesc.canonical.v1`；
- scope：`rs_generaldesc_external_train_val`；
- full deep validation 完成后 eligible 且 blockers 为空；
- Dataset/exporter 只接受 `external_train/external_val` 和七类 RS-GeneralDesc task；
- builder 配置没有 mask、Gold/Silver 或混合数据字段；validator 不接受其他 schema。

未来 `repackage` 路由锁定前代四项 identity，严格解析 hash ledger；record/metadata 在
读取前核验，asset 在同一次流式复制中计算 size/SHA，manifest layout、asset inventory、
ledger 和实际文件集合必须一致，全部 entry 验证完成后才允许写等价结论。随后按内容
fingerprint 重建 parent/record/provenance ID 和所有派生 SHA，在 sibling staging 中执行
full deep validation 后原子发布。历史 native 迁移报告由加固前实现生成，前代 root 已
删除，不能由当前修复追溯性地升级为严格逐文件证明；这也不构成 native payload 损坏
证据。额外 scope report 与旧 API/schema 兼容逻辑均不存在。

native manifest 只接受 RS-GeneralDesc 通用遥感训练/监控范围，不接受 OA-Grounded
数据，也不评价 Gate B。`external_val` 在 Stage 2 始终是训练监控用途。

### 4.3 `oa_groundrag/phase4`

作为 **RS-VLM** 实现基础，并保留后续 Mask-Grounded Baseline 核心。

保留：

- RegionSelector；
- EvidenceBuilder；
- Qwen3-VL processor/model；
- prompt-only；
- LoRA；
- checkpoint/resume；
- 推理和评价；
- GT、fixed predicted 和 end-to-end mask 隔离；
- mask swap、modality removal 等反事实接口。

已完成：

- RS-General Adapter 训练、best checkpoint 闭环和独立 Gate B；

后续需要新增：

- 可选 Landslide-Evidence Adapter 的路由；
- 两遍式 VLM 推理；
- RAG Evidence Provider；
- retrieved evidence cards；
- 最终带引用输出；
- wrong-mask + RAG 防合理化评价。

### 4.4 `RAG_tmp`

`RAG_tmp` 作为工程原型保留，但不直接作为最终算法实现。

可复用：

- PDF/OCR；
- 最小知识单元；
- SQLite FTS5；
- BGE-M3；
- Qdrant Embedded；
- RRF；
- BGE reranker；
- authority boost；
- 文件、页码和章节引用；
- 证据不足和越界拒答。

需要调整：

- 从独立问答应用改为 Evidence Retrieval Provider；
- 最终生成不再由其中的 Ollama Qwen3-8B负责；
- 返回结构化 Evidence Cards；
- 增加案例索引和模态索引；
- 接收 EvidenceBuilder 生成的结构化查询；
- 与 `paper7_VLM` 建立稳定接口。

---

## 5. Benchmark 与数据资产重新划分

新方案将数据拆成五类独立资产。

### 5.1 OA-AuxSeg Benchmark

用于：

- optical-only 分割；
- 任意辅助模态分割；
- no-target；
- 模态缺失；
- 质量选择；
- candidate region 和 region feature 导出。

保持当前 HDF5 和固定 patch 构建逻辑。

### 5.2 RS-GeneralDesc Benchmark

只整合通用遥感视觉语言数据：

- RSGPT / RSICap / RSIEval 中适合的 Caption 和 QA；
- MMRS-1M 中 Caption、VQA 和 bbox→短语；
- DisasterM3 中受图像支持的 scene、count、relation 和 visible report。

职责：

- 学习遥感图像语言；
- 学习场景和地物；
- 学习数量、颜色、位置和空间关系；
- 学习遥感问答和报告表达；
- 为 RS-General Adapter 提供训练与监控。

不承担：

- 滑坡专业知识；
- 滑坡区域真值；
- 滑坡正式评价；
- 真实 OA mask-grounded 训练。

native manifest 必须同时绑定 build ID、payload SHA-256、semantic config SHA-256 和
hash-manifest SHA-256。RS-VLM preflight 直接核验这些 identity、canonical schema、
eligible/空 blockers 与 saved clean deep validation，不使用旁路报告。

### 5.3 Landslide Evidence Corpus

用于补充当前 Benchmark 缺少的滑坡文本语料。

它不是一个大规模自由报告数据集，而是以**结构化事实为中心**的滑坡区域证据语料。

数据来源：

- OA-AuxSeg train split 中的 GT mask；
- 必要时加入高质量 OOF predicted mask；
- 只使用 train parent；
- val/test parent、同事件、同区域近重复不得进入。

内容分三层。

#### A. Programmatic Facts

由程序直接生成：

- target status；
- bbox；
- centroid；
- area 和 area ratio；
- image location；
- component count；
- elongation；
- compactness；
- valid coverage；
- modality availability；
- DEM 高程统计；
- slope 统计；
- InSAR mask 内外统计；
- SAR mask 内外统计；
- no-target；
- 数值单位和 sign convention；
- 允许和禁止的 claim。

这部分可以覆盖全部可用 train 样本。

#### B. Teacher Silver Observations

使用本地 VLM 或外部多模态 API 自动生成候选视觉观察。

输入必须包括：

- 光学全图；
- mask overlay；
- context crop；
- 可用辅助模态视图；
- Programmatic Facts；
- 模态可用性；
- 禁止结论。

模型只生成：

- 光学可见特征；
- 地形视觉观察；
- InSAR/SAR 定性空间观察；
- 可能混淆对象；
- 证据充分性；
- 简短摘要。

禁止教师自由生成：

- 发生时间；
- 触发原因；
- 精确运动速度；
- 稳定性等级；
- 风险等级；
- 无输入模态支持的结论。

#### C. Expert Gold

专家审核少量代表样本，重点审核：

- mask 对应；
- 光学可见滑坡特征；
- 困难负样本；
- 地形支持；
- 形变支持；
- SAR 支持；
- 证据充分性；
- 混淆对象；
- 不允许生成的结论；
- 一段简短摘要。

Expert Gold 主要用于 OA-GroundedEval，也可少量用于可选 adapter 训练。

### 5.4 Landslide Mask-Case Knowledge Base

保存代表性滑坡正案例、边界案例和困难负样本。

每个案例至少包括：

- 原始光学图像；
- GT mask；
- mask overlay；
- context crop；
- 可用辅助模态；
- Programmatic Facts；
- 专家观察；
- 支持证据；
- 反对证据；
- 混淆对象；
- 适用环境；
- 传感器和产品信息；
- 证据限制；
- 来源和 split。

知识库案例只能从 train split 选取。

### 5.5 OA-GroundedEval

小规模正式滑坡区域理解评价集，不用于通用遥感训练。

覆盖：

- 正确滑坡 mask；
- no-target；
- 单滑坡与多滑坡；
- 小目标与大目标；
- 清晰边界与模糊边界；
- optical-only；
- optical + DEM；
- optical + InSAR；
- optical + SAR；
- 多辅助模态；
- 低质量和低覆盖；
- 采石场；
- 道路切坡；
- 裸岩；
- 河滩；
- 云影、山影和噪声；
- 错误预测 mask。

评价内容：

- mask-region consistency；
- modality attribution；
- unsupported claim；
- confounder recognition；
- evidence sufficiency；
- citation correctness；
- wrong-mask resistance。

---

## 6. 滑坡文本语料自动生成方案

### 6.1 总体流程

```text
GT mask / OOF predicted mask
        ↓
Evidence Renderer
        ↓
Programmatic Facts
        ↓
教师 VLM 生成候选观察
        ↓
规则检查
        ↓
第二次独立验证
        ↓
过滤为 Silver
        ↓
专家抽查或审核
        ↓
Gold / Case Knowledge / Optional Training
```

### 6.2 样本选择策略

不需要对所有分割样本调用外部 API。

按以下维度分层抽样：

- source；
- 模态组合；
- optical channel signature；
- 环境类型；
- foreground ratio；
- component count；
- mask shape；
- optical quality；
- auxiliary valid coverage；
- positive / no-target；
- 正例 / 困难负样本；
- 不同数据源和区域。

建议先构建数百条 pilot，再扩展到数千条 Silver，而不是一开始生成全量长报告。

### 6.3 外部 API 生成要求

如使用外部 API：

1. 只发送项目允许外发的图像和字段；
2. 每个请求保存模型名称、版本、prompt hash、输入 asset hash 和时间；
3. 强制结构化输出；
4. temperature 保持低值；
5. 同一样本至少生成两个独立候选；
6. API 输出不能直接成为 test 真值；
7. 输出缓存，失败可恢复；
8. 禁止上传 val/test 或保密数据；
9. 不能把 API 的专业推断直接当作事实；
10. API 仅生成候选文本，不修改 mask。

如不使用外部 API，可使用本地 Qwen3-VL 或其他强 VLM 完成同样流程。

### 6.4 自动规则过滤

Silver 至少通过以下规则：

- 所有几何和数值字段与程序事实一致；
- 缺失模态不得出现对应证据；
- 低覆盖模态不能输出强支持；
- 单时相不得出现发生时间；
- 没有单位不得出现定量速度；
- no-target 不得生成滑坡区域描述；
- 被禁止的 claim 不得出现；
- 输出必须区分视觉观察与专业解释；
- 同一样本两次生成的核心字段必须一致；
- mask swap 后描述应发生合理变化；
- modality removal 后相关字段必须删除；
- wrong-region 样本不得被强制解释为滑坡；
- 近重复文本需要去重。

### 6.5 专家审核方式

专家不需要为每个样本撰写长报告。

建议审核结构化字段：

- target status；
- optical evidence；
- terrain evidence；
- deformation evidence；
- SAR evidence；
- confounders；
- evidence sufficiency；
- forbidden claims；
- short summary。

专家审核成本优先投向：

- OA-GroundedEval；
- 困难负样本；
- 不同环境类型；
- 多源证据冲突；
- API 低一致性样本。

---

## 7. OA-AuxSeg 模型设计与 Adapter 优化

### 7.1 总体结构

```text
Optical Input
    ↓
Optical Main Encoder
    ↓
Multi-scale Optical Features
                  ↑
Auxiliary Inputs
    ↓
Modality-specific Input Adapters
    ↓
Shared / Partially Shared Auxiliary Encoder
    ↓
Quality Selection
    ↓
CMNeXt-style Residual Injection
    ↓
Segmentation Decoder
    ↓
Mask / Regions / No-target
```

### 7.2 光学 Adapter

保留当前“共享 RGB stem + extra-band residual”的设计思想：

- 所有光学签名共享官方 RGB 主分支；
- 非 RGB 波段通过签名专属浅层 residual 进入；
- residual 零初始化；
- 不改变纯 RGB 官方 stem 的初始输出；
- 不把 source ID 作为模型输入；
- 只使用通道名和真实数据统计确定归一化。

优化建议：

1. 先冻结官方 RGB stem，完成稳定训练后再逐步解冻；
2. 对额外波段设置独立学习率；
3. 使用 channel-valid mask；
4. 不为每个光学通道签名复制完整 backbone；
5. 对 3/4/12 通道分别报告消融。

### 7.3 辅助模态 Adapter

每种辅助模态建立轻量 input adapter：

- DEM；
- slope；
- InSAR；
- SAR；
- 未来真实存在的其他辅助模态。

Adapter 只负责：

- 通道映射；
- 基础归一化；
- valid mask；
- 物理值范围稳定；
- 映射到统一特征维度。

不负责：

- 专业解释；
- 可靠性重复判断；
- 报告生成。

### 7.4 质量选择优化

质量选择只出现一次。

输入可包括：

- feature statistics；
- valid coverage；
- optical overlap；
- resize ratio；
- 可用的质量字段；
- sensor/product card。

加入 null auxiliary，使模型可拒绝无效辅助信息。

禁止：

- 在 decoder 再增加第二套 reliability；
- 把 quality weight 写入最终地学证据；
- 通过 source ID 学习数据集偏差。

### 7.5 注入优化

CMNeXt 式注入保持：

- 光学作为主分支；
- 辅助模态在中高层提供增量；
- 残差注入；
- 近零初始化；
- optical-only 硬旁路；
- 模态顺序不变性；
- 注入可关闭。

首版避免同时维护多种复杂注入结构。

### 7.6 Region Features

当前 region feature 可作为：

- 区域描述的诊断输入；
- 案例检索 baseline；
- 可选 region selector；
- error analysis。

但分割特征未必天然适合相似案例检索。

建议后续增加一个独立、轻量的 retrieval projection：

```text
frozen region feature
→ small projection head
→ retrieval embedding
```

projection 只在案例相关性数据上训练，不反向改变分割模型。

---

## 8. VLM 与 Adapter 设计

### 8.1 基础模型

继续采用 Qwen3-VL-2B 作为主模型，视觉 encoder 和 merger 默认冻结。

### 8.2 RS-General Adapter

训练数据：

- RS-GeneralDesc Benchmark。

目标能力：

- 通用遥感图像描述；
- 场景理解；
- 地物识别；
- 数量和空间关系；
- 遥感问答；
- 报告表达风格。

沿用当前 LoRA 设计：

- attention `q/k/v/o_proj`；
- 小参数量；
- 视觉塔冻结；
- 单卡训练。

### 8.3 Landslide-Evidence Adapter

默认不训练。

仅在以下 gate 失败时启用：

- 模型忽略 mask；
- 经常描述 mask 外内容；
- no-target 处理失败；
- modality removal 后仍输出对应结论；
- 结构化证据字段不稳定；
- RAG 后仍无法控制不支持结论。

推荐实现方式：

1. 以 RS-General Adapter 权重为初始化；
2. 单独保存为 Landslide-Evidence Adapter；
3. 使用 Landslide Evidence Corpus 中的 Gold、Auto 和高质量 Silver；
4. 加入一定比例 RS-General replay；
5. 使用更低学习率；
6. 设置通用遥感 retention gate；
7. 不与 OA-AuxSeg 联合训练。

推理时：

- 通用遥感任务使用 RS-General Adapter；
- 滑坡 mask-grounded 任务使用 Landslide-Evidence Adapter；
- 两者分别保存和评价，不强求运行时叠加。

### 8.4 两遍式 VLM 推理

#### Pass 1：Visual Observation

输入：

- 全图；
- mask overlay；
- region crop；
- 可用辅助模态视图；
- 程序事实；
- 用户问题。

输出只包括：

- 可见区域特征；
- 多模态空间一致性；
- 证据缺失；
- 不作专业结论。

#### Pass 2：Knowledge-Grounded Interpretation

输入：

- Pass 1 视觉观察；
- Programmatic Facts；
- retrieved evidence cards；
- 用户问题；
- 禁止结论。

输出：

- 支持证据；
- 反对证据；
- 混淆对象；
- 证据充分性；
- 建议补充数据；
- 最终摘要；
- 引用来源。

---

## 9. Mask-Grounded Evidence Builder

### 9.1 视觉证据

生成：

- optical full image；
- mask overlay；
- region crop；
- optional mask-only view；
- DEM/slope view；
- InSAR view；
- SAR view；
- 其他 aligned auxiliary views。

### 9.2 确定性事实

程序计算：

- bbox；
- centroid；
- area；
- area ratio；
- image location；
- component count；
- elongation；
- compactness；
- fragmentation；
- optical valid coverage；
- auxiliary coverage；
- modality availability；
- units；
- sign convention。

### 9.3 辅助模态事实

#### DEM / slope

- 高程均值、范围和分位数；
- 平均坡度；
- 高坡度比例；
- mask 内外差异；
- 有效覆盖率。

#### InSAR

- 产品类型；
- 单位；
- LOS sign convention；
- mask 内均值、中位数和分位数；
- mask 内外差异；
- 有效覆盖；
- 可用质量信息；
- 是否存在连续异常的程序指标。

#### SAR

在数据已定标时：

- 后向散射统计；
- mask 内外差异；
- 纹理统计；
- 有效覆盖。

未定标时只提供归一化视觉观察，不输出严格物理结论。

### 9.4 证据边界

Evidence Builder 必须输出：

- available claims；
- forbidden claims；
- missing evidence；
- required verification。

---

## 10. Evidence-Constrained Text RAG

### 10.1 RAG 的定位

OA-GroundRAG 中的 RAG 不负责重新识别图像，也不负责从知识库中“寻找理由”证明候选区域一定是滑坡。

当前主线将能力明确拆分为：

```text
OA-AuxSeg
→ Where：候选区域在哪里

Mask-Grounded Region Adapter
→ What：mask 指定区域看到了什么

Evidence-Constrained Text RAG
→ How to interpret：这些视觉观察在专业上可能意味着什么、还可能是什么、当前不能判断什么
```

因此 Stage 6 的核心问题是：

> 给定 Mask-Grounded Region Adapter 已经得到的区域视觉观察，能否从滑坡、遥感和传感器专业资料中检索与当前证据直接相关的解释、混淆因素和证据限制，使 VLM 从“区域视觉描述”进一步提升为“有来源、有反例、有边界的专业解释”？

RAG 负责：

- 为当前视觉观察提供专业解释依据；
- 主动补充可能的混淆对象和反例；
- 提供光学、SAR、InSAR、DEM 等证据的适用条件和解释限制；
- 在证据不足时给出需要补充的验证信息；
- 为专业解释提供可追溯来源和页码。

RAG 不负责：

- 生成或修改 mask；
- 改写 Programmatic Facts；
- 改写 Pass-1 已得到的视觉观察；
- 将候选区域自动升级为确认滑坡；
- 根据通用知识推断发生时间、确定触发原因、稳定性或正式风险等级；
- 在 Stage 6 首版中检索视觉案例、训练 learnable retriever 或建立知识图谱。

---

### 10.2 Landslide Text Evidence Bank

Stage 6 只建立一个物理知识库：

> **Landslide Text Evidence Bank**

首版语料来自项目已有的地质灾害规范、滑坡经典分类文献、官方手册、遥感识别规程以及 SAR/InSAR 专业资料。当前 `docs/RAG_knowledge/` 中已有资料可作为 v1 初始语料，不要求在 Stage 6 开发前继续大规模扩充来源。

首版资料主要覆盖：

- 滑坡类型、运动方式与专业术语；
- 地质灾害遥感识别与调查规则；
- 光学影像中的滑坡可见特征与判读边界；
- InSAR LOS、相干性、解缠、大气和几何限制；
- 地质灾害调查、核查和评价中的证据要求；
- 可能导致错误解释的常见限制和混淆因素。

每个源文件必须保存：

- `source_id`；
- 原始文件名；
- 文件 SHA-256；
- 标题；
- `source_kind`；
- `authority_class`；
- `source_status`；
- 标准编号或年份（若明确存在）；
- PDF 页数；
- 解析覆盖率；
- OCR/解析质量状态。

`source_status`、`authority_class` 和标准适用状态必须来自显式 source registry，不允许仅根据文件名自动猜测“现行、失效、草案或正式”等法律/规范状态。

Stage 6 不自动下载新资料，不修改原始 PDF，不因为部分扫描页无法解析而伪造文本。无法可靠提取的页面必须标记为 `ocr_required` 或 `unavailable`，不得进入正式检索单元。

---

### 10.3 Text Evidence Unit

Stage 6 不把固定长度的 PDF chunk 直接作为最终检索证据，而是构建较短、自包含、尽量保持“条件—解释—限制”完整性的 **Text Evidence Unit**。

建议 schema：

```text
unit_id
source_id
source_sha256
pdf_page
printed_page
section
knowledge_type
content
modality
conditions
tags
authority_class
source_status
extraction_method
content_sha256
```

首版只保留三个核心 `knowledge_type`：

1. `interpretation`
   - 解释某类视觉或遥感现象在滑坡识别中可能意味着什么；
   - 强调“可作为证据”而不是“看到即确认”。

2. `confounder`
   - 记录与滑坡外观相似的道路切坡、采石场、裸岩、河滩、冲沟、施工扰动等；
   - 记录这些对象为什么容易混淆，以及可用于区分的条件。

3. `limitation`
   - 记录单时相光学、SAR、InSAR、DEM 等证据的使用边界、误差来源和不可支持的推断；
   - 可同时包含“建议补充哪些数据或核查手段”的内容，不再建立独立 verification index。

专业术语、传感器名称、地貌和滑坡部位等不建立独立物理索引，只作为 `tags` 和查询扩展词使用。

禁止把全局生成边界重复写入每个知识单元。诸如“不得给出正式风险等级”“不得由单时相确定发生时间”等通用限制由 Final Generation Policy 统一约束；只有当某条资料本身讨论特定限制时，才作为 `limitation` 单元入库。

---

### 10.4 语料切分原则

Stage 6 首版优先采用：

```text
PDF
→ page
→ section / clause / paragraph
→ self-contained Text Evidence Unit
```

切分遵循：

- 标准优先按条款、章节或完整规则切分；
- 论文、手册和培训资料优先按自然段或小节切分；
- 不为了固定 token 数强行拆开同一条规则中的适用条件和限制；
- 过长单元只在自然段边界继续拆分；
- 保存 PDF 页码和 section；
- 不使用大窗口 overlap 制造大量近重复 chunk；
- exact duplicate 和内容 hash 重复项只保留可追溯关系，不重复进入检索候选。

首版不要求使用 LLM 自动重写全部知识单元，也不要求人工为所有 PDF 构造复杂 Evidence Card。

---

### 10.5 Task-based RAG Switch

Stage 6 不训练额外的 Retrieval Reflection 模型，而由现有 Task Controller 确定是否调用 RAG。

默认规则：

```text
Segmentation
Scene Description
Mask Description
→ RAG OFF

Candidate Evaluation
Multimodal Evidence QA
Report Generation
Knowledge QA
→ RAG ON
```

其中 `Mask Description` 的目标只是客观描述 mask 区域和周围环境，不应被专业知识提前污染。

---

### 10.6 Evidence-conditioned Query Builder

Stage 6 不直接把用户问题或整段 Pass-1 输出作为一个大查询，而只构造两个确定性 retrieval intent。

输入：

```text
用户问题
+ target status
+ Programmatic Facts（只读）
+ Pass-1 Target Appearance
+ Pass-1 Target Morphology
+ Pass-1 Surrounding Environment
+ Pass-1 Region–Context Contrast
+ Pass-1 Possible Confusers
+ Pass-1 Evidence Sufficiency
+ Available Modalities
```

#### Query A：Interpretation Query

主要使用：

- 用户问题；
- Target Appearance；
- Target Morphology；
- Region–Context Contrast；
- Available Modalities。

用于检索 `interpretation` 单元。

#### Query B：Counter / Limitation Query

主要使用：

- Possible Confusers；
- Surrounding Environment；
- Evidence Sufficiency；
- Available Modalities；
- 用户问题中的限制性意图。

用于检索 `confounder` 与 `limitation` 单元。

Query Builder 不允许因为候选区域来自滑坡分割模型而自动加入“典型滑坡”“确认滑坡”等正向结论词。

---

### 10.7 混合文本检索

Stage 6 首版只实现一种稳定、低复杂度的 Hybrid Retrieval：

```text
source-status / modality metadata filter
        ↓
SQLite FTS5 lexical retrieval
+
BGE-M3 dense retrieval
        ↓
Reciprocal Rank Fusion
        ↓
knowledge-type quota
        ↓
Balanced Evidence Packet
```

首版不要求：

- Qdrant；
- cross-encoder reranker；
- knowledge graph；
- agentic retrieval；
- learnable retriever；
- multimodal image-text joint retriever；
- 复杂的 authority boost 公式。

知识库规模较小时，dense embedding 可直接保存为本地向量矩阵并使用余弦相似度检索。`authority_class` 用于过滤、报告和同分情况下的稳定排序，不需要引入复杂可学习权重。

正式 Stage 6 配置使用 FTS5 + BGE-M3 + RRF。若本地 BGE-M3 暂不可用，可以保留 lexical-only 工程 fallback，但不得把 lexical-only 结果宣称为正式 Hybrid RAG 结果。

---

### 10.8 Balanced Evidence Packet

Stage 6 不把相似度最高的 Top-K 文本全部发送给 VLM，而是按知识作用限制证据数量。

首版建议最多返回 6 条：

```text
2 interpretation
2 confounder
2 limitation
```

当某一类证据不足时可以少于 6 条，不应使用其他类型无条件填满配额。

每条 Evidence Item 只需包含：

```text
evidence_id
unit_id
knowledge_type
content
modality
conditions
source_id
source_title
pdf_page
section
authority_class
source_status
lexical_rank
dense_rank
rrf_score
```

这组结果统一形成：

> **Balanced Evidence Packet**

RAG 模块只返回 Evidence Packet，不直接生成最终用户答案。

---

### 10.9 两遍式推理

Stage 6 使用严格解耦的两遍式流程。

#### Pass 1：视觉观察

```text
Original Full RGB
+ Binary Mask
+ Clean Context Crop
        ↓
Mask-Grounded Region Adapter
        ↓
Structured Visual Observation
```

Pass 1 不访问知识库。

#### Retrieval：文本证据检索

```text
Structured Visual Observation
+ User Question
        ↓
Two-query Builder
        ↓
Hybrid Retrieval
        ↓
Balanced Evidence Packet
```

#### Pass 2：专业解释

```text
User Question
+ target status / Programmatic Facts
+ Pass-1 Visual Observation
+ Balanced Evidence Packet
+ Generation Policy
        ↓
Qwen3-VL-2B + 当前 Mask-Grounded Region Adapter
        ↓
Evidence-Constrained Interpretation
```

Pass 2 首版只使用文本输入，不再次发送图像。这样可以：

- 避免重复视觉 token；
- 明确视觉事实来自 Pass 1；
- 防止 RAG 阶段重新看图并改写 Pass-1 观察；
- 使 no-RAG 与 text-RAG 使用同一生成器进行公平对比。

若未来证明 text-only Pass 2 明显不足，再单独讨论 Final Generator，不作为 Stage 6 v1 的必需组件。

---

### 10.10 Pass-2 输出合同

Stage 6 首版输出保持简洁：

```text
supporting_interpretation
confounders
limitations
recommended_verification
summary
```

其中前三类专业 claim 需要绑定 Evidence ID。

示意：

```json
{
  "supporting_interpretation": [
    {"text": "...", "evidence_ids": ["E01"]}
  ],
  "confounders": [
    {"text": "...", "evidence_ids": ["E03"]}
  ],
  "limitations": [
    {"text": "...", "evidence_ids": ["E05"]}
  ],
  "recommended_verification": [
    {"text": "...", "evidence_ids": ["E05"]}
  ],
  "summary": "..."
}
```

约束：

- RAG 不复制或重写 mask geometry；
- RAG 不改变 target status 的程序事实；
- RAG 不把 knowledge unit 内容声明成当前图像已经观察到的事实；
- Evidence ID 必须真实存在于当前 Balanced Evidence Packet；
- `text_rag` 模式下，专业解释、混淆因素和限制结论必须有 Evidence ID；
- `no_rag` 对照允许 `evidence_ids=[]`，但使用相同 Pass-2 输出结构和同一生成器；
- 最终仍称“候选区域”，不因知识检索自动升级为确认滑坡。

---

### 10.11 Stage 6 首版不做的内容

为保证主算法框架尽快闭环，以下内容明确延后：

- 正案例视觉检索；
- 困难负样本图像检索；
- DINO/CLIP/Region Feature 相似案例索引；
- SAR/InSAR/DEM 分模态视觉 embedding；
- case-only / hybrid case RAG；
- multimodal retriever 训练；
- 知识图谱；
- 多 Agent 检索；
- RAG 专用 LoRA；
- sealed-test 正式 RAG 评价。

这些内容只有在 Stage 6 文本 RAG 已证明有效后才考虑扩展。

---

## 11. 统一多任务接口

参考 M3D 的“统一交互、专业模块执行”思想，支持以下任务。

### 11.1 Segmentation

```text
识别并分割图中的疑似滑坡。
```

输出：mask、candidate regions、no-target、confidence。

### 11.2 Scene Description

```text
描述当前遥感场景。
```

使用 RS-General Adapter，不调用分割。

### 11.3 Mask Description

```text
描述候选区域 2。
```

调用 OA-AuxSeg 输出和 Evidence Builder。

### 11.4 Candidate Evaluation

```text
该候选区域是否具有滑坡遥感特征？
```

输出：支持、反对、混淆对象、证据充分性。

### 11.5 Multimodal Evidence QA

```text
InSAR 和 DEM 是否支持该候选区域？
```

使用程序事实、视觉观察和 RAG。

### 11.6 Report Generation

生成结构化区域报告，但明确：

- 候选区域不是确认结论；
- 不能替代现场调查；
- 不给出正式风险等级。

### 11.7 Knowledge QA

纯文本问题直接调用 RAG，不必运行 OA-AuxSeg。

### 11.8 RAG Routing

统一接口不对所有任务强制调用知识库。

```text
Segmentation               → OA-AuxSeg only
Scene Description          → RS-General / Region VLM, RAG OFF
Mask Description           → Mask-Grounded Region Adapter, RAG OFF
Candidate Evaluation       → Pass 1 + Text RAG + Pass 2
Multimodal Evidence QA     → Programmatic Facts + Pass 1 + Text RAG + Pass 2
Report Generation          → Pass 1 + Text RAG + Pass 2
Knowledge QA               → Text RAG + text-only generator
```

该路由替代额外的 learned retrieval-decision model，减少参数和训练工作量。
---

## 12. 训练与实施阶段

### Stage 0：冻结现有资产

- 核对当前 phase2/phase3/phase4 状态；
- 不重写已通过的核心接口；
- 更新 README、AGENTS 和本方案；
- 把旧方案移入 archive 或删除活动引用。

阶段依赖包含两条支线：OA-AuxSeg 工程定版 → 未来 Gate A → formal fixed masks；
RS-GeneralDesc Stage 2 → Adapter 重训 → Gate B。两条支线在 Mask-Grounded 阶段汇合。
Gate A 延后不等于 Gate A 通过，也不阻止独立的 Stage 2/3 支线准备；依赖 formal masks
的下游工作仍必须等待 Gate A。

### Stage 1：OA-AuxSeg 工程定版（Gate A 待执行）

- 冻结项目负责人确认的 proposed `checkpoint_best.pt`；
- 完成训练报告、严格重载以及 train/val 工程评价；
- 保留 optical-only、不同融合方式、模态组合和 no-target 的代码能力；
- 在完整框架搭建后再执行分割消融、多随机种子和 Gate A；
- Gate A 通过后才运行 sealed test 并导出固定预测 mask。

当前 `rs_vlm.config.v2` 仅开放 `external_generic`，用于 Stage 3
RS-General Base/Adapter。配置固定实际 manifest/validation SHA，并绑定 native
canonical/build/payload/hash identity；preflight 按 ledger 验证被读取的 metadata 和 shard
layout，Dataset 在消费前验证 shard/asset bytes。配置不含旁路作用域报告、旧治理字段、
未使用的 External evidence/evaluation 段或 v1 alias。GT/fixed/end-to-end mask、
RegionSelector、EvidenceBuilder、mask-grounded messages、反事实 evaluator 和 AuxSeg
inference 核心继续保留；临时 Mask-Grounded Dataset 合同已撤回，Stage 4/5 冻结新数据
schema 前没有可运行的 mask-grounded data mode。

训练配置中的 `max_steps` 是计划预算上限，不是必须跑满的权重有效性或 Gate A
判据。项目负责人可以依据训练轨迹主动停止优化，并按训练开始前已实现的 checkpoint
选择规则定版既有 best checkpoint。人工停止必须在报告中记录为
`project_owner_manual_stop`，不能伪装为跑满计划或自动 early stopping，也不自动要求
恢复训练。

当前 batch-16 proposed 训练已由项目负责人宣布结束，不再续训。既有
`checkpoint_best.pt` 作为当前最终权重，`checkpoint_last.pt` 和其后的未 checkpoint
日志只保留为轨迹证据；不得复制、重命名或改写 checkpoint。权重工程定版与 Gate A
是两个不同结论：前者不因消融和 Gate 尚未执行而失效，后者也不能由 checkpoint
存在或当前 validation 指标替代。

### Stage 2：RS-GeneralDesc Benchmark 验收

已完成：

- 确定性重发布到 `/home/yukun80/codes/benchmark/rs_generaldesc_v1`；
- 274,693 records、104,954 parents；历史报告记录 train/val 成员和 asset 等价，但旧
  repackage 未逐项验证前代 ledger 的全部 record/metadata，不能追溯性地称为严格证明；
- native manifest/canonical 为 `rs_generaldesc.*.v1`，eligible 且 blockers 为空；
- 新树单次 full deep validation 为 0 error / 0 warning；
- build/payload/hash-manifest identity 分别为 `build_3ebc09a4daad10e121fc14c2727d9896e10371a95bbaf6b780d15aa42eaf3c03`、
  `549281f296b357bce256e6af71cec7412fe17e36052d6a8674f4876ae2d06e0b`、
  `55ac26d9771ce8385318fbd23a10b999afb754ac195e823be659f4e49b0a7090`；
- phase4 三份活动配置直接绑定 native identity；
- Stage 3 prompt-only Base 使用 `rs_generaldesc_prompt_only_qwen3vl_2b.yaml`；Stage 2
  没有预先指定 Gate 集，Stage 3 从 `external_val` 另行冻结与训练期 monitoring parents
  零交集的 Gate B 集合。

Stage 2 的当前验收依据是 native manifest、固定 build/payload/hash identity、eligible、
空 blockers 与 saved clean deep validation。没有证据表明当前 native payload 损坏；
本次身份加固未修改或重扫真实 Benchmark。

前代 Benchmark 和 Adapter outputs 已在全部验收通过后删除，不保留备份、链接、alias
或兼容包装。Stage 2 结论本身不等于 OA-Grounded acceptance，也不等于 RS-General
Adapter Gate B 通过；后者已由随后独立执行的 Stage 3 protocol 给出接受结论。

### Stage 3：RS-General Adapter

已完成：

- prompt-only Base 与 native identity 上的 RS-General LoRA 配置均已固定；
- 24 GB 训练布局使用 physical batch 1、gradient accumulation 16，保持 effective
  batch 16；native 训练完成到 step 1000，`best_checkpoint.json` 按预注册的
  `macro_task_loss` 选择 step 1000，值为 `0.8427927826882716`；
- training report 保留 `formal_acceptance=false`，因为它只表达训练闭环；
- 独立 `rs_vlm.gate_b_protocol.v1` 冻结了与训练 monitoring parents 零交集的 256-parent
  Base-vs-Adapter 固定生成集合；
- Base/Adapter 各完成 256 predictions、0 failures，10,000 次 paired bootstrap 与六项
  判据全部通过；Gate B report 以 `formal_acceptance=true` 接受 Adapter。

checkpoint 存在或 teacher-forced validation loss 更低仍不能单独构成 acceptance；
Stage 3 结论只来自固定 Gate B report，且不扩张到 OA-Grounded、mask-grounded、Gate A
或 sealed test。

### Stage 4：Mask-Grounded Landslide Region Corpus 与 OA-GroundedEval

#### 4.1 阶段定位

Stage 4 不再依赖 OA-AuxSeg 的预测结果，也不评价分割模型是否正确。该阶段统一使用已有滑坡数据中的人工标注二值 mask，将 mask 作为唯一的目标区域定位信息，研究通用遥感 VLM 是否能够：

1. 准确关注 mask 所指定的滑坡区域；
2. 描述 mask 内部可见的光学特征；
3. 结合未被修改的原始遥感影像，描述滑坡区域周围的地形、地表覆盖和邻近地物环境；
4. 比较 mask 区域与周围环境之间的颜色、纹理、植被和形态差异；
5. 在证据不足时避免生成发生时间、触发原因、稳定性、风险等级等不受图像支持的专业结论。

Stage 4 的核心任务由“为分割结果生成专业报告”简化为：

> 给定一幅原始遥感影像和一个人工标注的滑坡 mask，生成同时覆盖目标区域内部特征、周围环境特征及区域—环境关系的 mask-grounded 遥感描述。

Stage 4 只建立区域描述数据和评价协议，不调用 OA-AuxSeg，不使用 predicted mask，不生成正式 fixed predicted masks，也不访问滑坡 RAG。预测 mask、RAG 和端到端流程分别在后续 Stage 5、Stage 6 和 Stage 7 中处理。

---

#### 4.2 输入数据

每条 Stage 4 样本至少包含：

- 原始光学遥感影像；
- 与光学影像严格同尺寸、同空间范围的二值滑坡 mask；
- 数据源、split 和样本身份；
- 原始数据中已经明确提供的传感器、分辨率和有效区信息；
- 可选的人工类别或场景元数据。

首版 Stage 4 以光学影像和二值 mask 为核心输入。DEM、坡度、InSAR 和 SAR 不作为 Stage 4 首版描述的必要输入，避免不同数据源的物理单位、符号、有效覆盖和配准差异增加标注与评价复杂度。

后续若需要研究多源区域证据，可在保持 Stage 4 光学基线不变的前提下，建立独立的辅助模态扩展版本，不得直接修改已冻结的首版数据和评价协议。

---

#### 4.3 区域视觉输入构建

为了避免彩色 overlay 改变光学影像的颜色、纹理和对比度，正式发送给 VLM 的区域视觉输入由以下三部分组成：

##### A. Original Full Image

完整、未修改的原始光学影像，用于保留：

- 整体地形和场景背景；
- 滑坡所在坡面或沟谷环境；
- 周围植被、道路、河流、建筑和裸地分布；
- mask 区域与整个场景之间的空间关系。

##### B. Binary Mask Prompt

与原始影像同尺寸的单通道二值 mask：

- mask 内部像素为目标区域；
- mask 外部像素为背景区域；
- mask 仅提供空间关注位置；
- mask 不修改原始 RGB 像素；
- mask 不携带颜色、类别或置信度信息。

二值 mask 应作为独立图像、独立空间提示或后续 mask pooling 的输入，不与原始光学影像进行颜色混合。

##### C. Clean Context Crop

根据 mask 外接矩形向外扩展固定比例，直接从原始光学影像中裁剪得到无标记的上下文图像。该图像同时保留：

- mask 区域内部的局部视觉细节；
- mask 邻近区域的地表环境；
- 滑坡区域与周围背景之间的过渡关系。

首版建议使用固定的 context margin ratio，并在数据发布前冻结。不同样本不得根据人工主观判断使用不同裁剪尺度。

彩色 mask overlay 仅作为专家审核、数据检查和论文可视化资产，不作为 VLM 正式描述输入，也不参与区域颜色、纹理和植被特征评价。

---

#### 4.4 Mask-Grounded Region Record

每个目标样本构建一条独立的 Mask-Grounded Region Record。记录内容分为确定性字段和人工描述字段。

##### 4.4.1 确定性字段

由程序根据二值 mask 直接计算：

- target status；
- mask bbox；
- mask centroid；
- mask area 和 area ratio；
- mask 在图像中的相对位置；
- 连通区域数量；
- context crop window；
- 原始影像尺寸；
- 数据源和 split；
- 图像、mask 和 crop 的资产身份。

这些字段只用于区域定位、数据检查和评价，不由 VLM 重新计算或改写。

##### 4.4.2 区域描述字段

人工描述或评价字段统一组织为以下六类：

1. **Target Appearance**
   - mask 区域的主要色调；
   - 表面纹理和粗糙程度；
   - 植被覆盖或裸露特征；
   - 内部均一性或异质性；
   - 可见边界是否清晰。

2. **Target Morphology**
   - 区域整体形态；
   - 是否呈长条状、块状、舌状或不规则形态；
   - 是否存在多个分离部分；
   - 区域延伸方向的定性描述。

3. **Surrounding Environment**
   - 周围主要地表覆盖类型；
   - 周围植被、裸地、道路、河流、建筑或农田情况；
   - 目标所在的坡面、沟谷、坡脚或其他可见地形环境；
   - 周围是否存在明显的人类活动扰动。

4. **Region–Context Contrast**
   - mask 内外色调差异；
   - mask 内外纹理差异；
   - mask 内外植被覆盖差异；
   - 目标边界与背景之间的过渡特征；
   - 目标区域与周围地物的邻接关系。

5. **Possible Confusers**
   - 裸岩；
   - 采石场；
   - 道路切坡；
   - 冲沟；
   - 河滩；
   - 农田翻耕；
   - 施工扰动；
   - 阴影或低质量区域；
   - 其他可能与滑坡外观相似的地表区域。

6. **Evidence Sufficiency and Summary**
   - 当前图像是否足以支持清晰区域描述；
   - 哪些特征清晰可见；
   - 哪些特征因分辨率、遮挡、阴影或图像质量无法判断；
   - 一段简短、客观的综合描述。

---

#### 4.5 描述边界

Stage 4 的描述对象是“人工 mask 指定的滑坡区域及其周围遥感环境”，不是重新判断该区域是否一定为滑坡。

允许描述：

- 图像中直接可见的颜色、纹理、形态和植被特征；
- mask 区域与周围环境的视觉差异；
- 区域所处的相对位置；
- 周围地表覆盖和邻近地物；
- 可能存在的视觉混淆对象；
- 图像证据的充分性和局限性。

禁止描述：

- 滑坡发生的具体时间；
- 降雨、地震或工程活动等确定触发原因；
- 精确位移速度；
- 运动方向和变形阶段；
- 稳定性等级；
- 危险性或风险等级；
- 灾害规模等级；
- 对人员、道路或建筑的实际威胁；
- 现场调查才能确认的地质结构和物质组成。

所有输出均应使用“mask 指定区域”“目标区域”或“标注滑坡区域”等表述，不将视觉描述扩张为新的地质灾害确认结论。

---

#### 4.6 数据资产划分

Stage 4 形成两个相互独立的资产。

##### A. Mask-Grounded Landslide Region Corpus

用于开发和可选训练，只使用 OA-AuxSeg Benchmark 的 train split。

Corpus 包括：

- 原始光学影像；
- 二值 GT mask；
- clean context crop；
- 可选的人工审核 overlay；
- 程序确定性区域事实；
- 结构化区域描述；
- 数据来源和资产身份；
- 描述审核状态。

Corpus 首版不要求对全部 train 样本生成长文本。建议采用分层抽样方式，覆盖：

- 不同数据源；
- 不同 mask 面积；
- 单连通和多连通区域；
- 不同区域位置；
- 清晰和模糊边界；
- 不同周围环境；
- 典型困难负背景。

Teacher Silver 不作为 Stage 4 必须完成的资产。只有 Stage 5 表明通用遥感 VLM 无法稳定理解 mask 或区域上下文时，才对 train split 中的部分样本生成候选 Silver，并经过规则过滤和专家抽查后用于可选 Adapter 训练。

##### B. OA-GroundedEval

OA-GroundedEval 用于评价 VLM 的 mask-grounded 区域理解能力，不用于训练。

OA-GroundedEval 使用与训练 Corpus 不重叠的 val/test 样本，并至少覆盖：

- 小、中、大面积滑坡区域；
- 单连通和多连通区域；
- 不同图像位置；
- 清晰和模糊区域边界；
- 不同植被覆盖和地表环境；
- 裸岩、道路切坡、采石场和施工扰动等混淆背景；
- 低对比度、阴影和低质量图像；
- no-target 或 empty-mask 样本。

开发阶段只使用 val 构建 OA-GroundedEval-dev。正式 test 必须保持封存，在描述协议、模型输入形式、评价字段和阈值冻结后才允许使用。

---

#### 4.7 反事实评价

为了验证 VLM 是否真正使用 mask，而不是只进行整图描述，OA-GroundedEval 增加少量由现有 GT mask 确定性生成的反事实样本。

包括：

1. **Empty Mask**
   - 将二值 mask 置空；
   - 模型应报告无指定目标区域；
   - 不得继续描述原滑坡区域。

2. **Mask Shift**
   - 将 mask 平移到同一影像中的非目标区域；
   - 描述应随关注区域变化；
   - 不得复用原目标区域描述。

3. **Mask Swap**
   - 在同一影像存在多个区域时交换 mask；
   - 描述应对应新的 mask 区域。

4. **Context Removal**
   - 只提供 mask 区域，不提供完整影像或上下文裁剪；
   - 用于评价周围环境信息对描述质量的影响。

这些反事实 mask 只用于评价，不作为滑坡训练真值，也不替代原始 GT mask。

---

#### 4.8 评价任务与指标

OA-GroundedEval 主要评价以下能力：

1. **Mask–Region Consistency**
   - 描述是否对应 mask 内部区域；
   - 是否错误描述 mask 外部的其他显著地物。

2. **Target Appearance Accuracy**
   - 色调、纹理、植被和边界描述是否正确。

3. **Surrounding Environment Accuracy**
   - 周围地表覆盖和邻近地物描述是否正确。

4. **Region–Context Relation Accuracy**
   - 是否准确描述目标与周围环境的差异和空间关系。

5. **Confuser Recognition**
   - 是否能够指出可能的视觉混淆对象；
   - 是否避免把所有裸露区域都解释为确定滑坡证据。

6. **Unsupported Claim Rate**
   - 是否生成触发原因、时间、风险、稳定性和精确运动信息等不受支持结论。

7. **Evidence Sufficiency Accuracy**
   - 是否能够在低质量或证据不足时正确降低结论强度。

8. **Counterfactual Sensitivity**
   - mask 变化后描述是否发生合理变化；
   - empty mask 时是否拒绝生成目标区域描述。

评价以专家结构化字段判断为主，传统 BLEU、ROUGE 和 BERTScore 只作为文本相似性的补充指标，不作为主要科学结论。

---

#### 4.9 人工标注与质量控制

专家不需要为每条样本撰写长篇滑坡报告，只需完成统一结构化字段和简短摘要。

推荐审核流程：

1. 第一名标注者完成区域和环境描述；
2. 第二名标注者检查 mask 对应、事实正确性和禁止结论；
3. 对 target appearance、surrounding environment、possible confusers 和 evidence sufficiency 的分歧进行仲裁；
4. 冻结最终 Gold 描述和评价字段；
5. 保存标注者、审核者、版本和时间信息。

质量控制必须包括：

- train/val/test 隔离；
- sample ID 和资产身份绑定；
- 图像、mask 和 crop 尺寸一致；
- 二值 mask 严格为 0/1 或 0/255；
- context crop 可由 mask 确定性重建；
- overlay 不进入正式 VLM 输入；
- 不从文件名推断地理位置或物理意义；
- 不修改原始光学影像像素；
- 不允许 test 样本进入 prompt 调优、Teacher Silver 或 Adapter 训练。

---

#### 4.10 Stage 4 实施顺序

Stage 4 按以下顺序执行：

1. 冻结现有 train-only deterministic Auto Corpus；
2. 增加独立二值 mask 资产；
3. 将红色 overlay 降级为人工审核资产；
4. 构建 original full image + binary mask + clean context crop 的正式输入协议；
5. 建立永久 Stage 4 数据与消息合同测试；
6. 从 val split 构建小规模 OA-GroundedEval-dev；
7. 完成专家结构化标注和审核；
8. 在 Base Qwen3-VL 与 RS-General Adapter 上运行 GT-mask 区域描述基线；
9. 比较 full image、crop、overlay、binary mask 和 full + mask + crop 等输入方式；
10. 根据评价结果决定是否需要 Teacher Silver 或 Landslide-Evidence Adapter；
11. 冻结正式 OA-GroundedEval 协议；
12. 在后续 Gate 冻结后才使用 sealed test。

---

#### 4.11 Stage 4 完成标准

Stage 4 只有同时满足以下条件才视为完成：

- 已建立不修改原始 RGB 的 mask-grounded 输入协议；
- 已完成 train-only Region Corpus；
- 已完成 OA-GroundedEval-dev；
- 已完成专家结构化标注与审核；
- 已实现 correct mask、empty mask 和 mask shift 等反事实评价；
- Base 与 RS-General Adapter 的 GT-mask baseline 已运行；
- 已报告 mask-region consistency、区域特征、周围环境、unsupported claim 和反事实指标；
- 已明确判断是否需要 Teacher Silver；
- val/test 与训练数据保持严格隔离；
- 正式 test 仍保持封存；
- 没有使用 OA-AuxSeg 预测 mask 或未通过 Gate A 的分割结果。

Stage 4 的输出是一个独立的 mask-grounded 区域描述数据和评价基础，不代表 OA-AuxSeg 已通过科学验证，也不代表完整 OA-GroundRAG、RAG 或端到端系统已经完成。

### Stage 5：Mask-Grounded Baseline

依次比较：

1. base Qwen3-VL；
2. RS-General Adapter；
3. full image；
4. crop；
5. overlay + crop；
6. multimodal evidence；
7. GT mask；
8. fixed predicted mask；
9. wrong mask。

### Stage 6：Evidence-Constrained Text RAG

Stage 6 在现有 Mask-Grounded Region Adapter 之后构建文本专业知识增强闭环。

实施顺序：

1. 审计 `docs/RAG_knowledge/` 中现有 PDF，并建立显式 Source Registry；
2. 完成 PDF 文本解析和质量报告，不可靠扫描页标记为 `ocr_required`；
3. 按 section / clause / paragraph 构建自包含 Text Evidence Unit；
4. 将知识单元归为 `interpretation / confounder / limitation` 三类；
5. 构建 SQLite FTS5 索引和本地 BGE-M3 dense embedding；
6. 实现 lexical + dense + RRF Hybrid Retrieval；
7. 实现两个确定性 query：Interpretation 与 Counter/Limitation；
8. 实现最多 2+2+2 的 Balanced Evidence Packet；
9. 实现 text-only Pass-2 message 与严格输出合同；
10. 在 OA-GroundedEval-dev 中现有可用 baseline records 上完成 bounded no-RAG vs text-RAG 开发对照；
11. 保存 retrieval provenance、Evidence ID、source/page/section 和生成身份；
12. Stage 6 不访问 sealed test，不修改 Stage 5 checkpoint，也不因为 64 条 no-target 失败阻塞文本 RAG 主链开发。

Stage 6 的核心对照必须使用相同 Pass-1、相同 Pass-2 generator 和相同生成参数：

```text
Pass-1 Observation + Pass-2 without retrieved evidence
vs
Pass-1 Observation + Balanced Evidence Packet + Pass-2
```

这样可以把性能差异尽量归因于外部知识，而不是第二遍生成本身。

Stage 6 工程完成标准：

- Text Evidence Bank v1 可构建、可验证、可追溯；
- 三类知识单元可检索；
- FTS5 + BGE-M3 + RRF 路径可运行；
- 两个 query builder 可确定性重算；
- Balanced Evidence Packet 可稳定生成；
- Pass-2 text-only 路径与当前 Region Adapter 打通；
- no-RAG / text-RAG 使用同一输出 schema；
- Evidence ID 与 source/page/section 可以严格验证；
- 知识库失败时不会修改分割或 Pass-1 结果；
- 当前只完成 development protocol，不声称 Gate D、sealed-test 或最终科学验收通过。

---

### Stage 7：可选 Case RAG 扩展

Stage 7 不作为 Stage 6 完成前提。

只有 Stage 6 表明文本 RAG 对专业解释有稳定增益后，才考虑：

- train-only 正案例；
- 困难负样本案例；
- optical region embedding；
- same-modality case retrieval；
- text-only vs case-only vs text+case。

首版主论文若文本 Evidence RAG 已能完整支撑核心结论，可以将 Stage 7 作为扩展实验而非强制主线模块。

### Stage 8：可选 Landslide-Evidence Adapter

只有 Gate 失败时执行。

训练：

- Gold；
- Auto；
- 过滤 Silver；
- external replay；
- retention gate。

### Stage 9：统一推理与报告

- Task Controller；
- 两遍式生成；
- 引用；
- failure artifact；
- end-to-end demo；
- 完整评价。

---

## 13. 决策 Gate

### Gate A：OA-AuxSeg 是否可靠

要求：

- 分割性能达到设定门槛；
- no-target FPR 可接受；
- auxiliary 模态不会系统性降低性能；
- checkpoint 可严格恢复。

跑满计划 `max_steps` 不是 Gate A 的独立要求。当前 best checkpoint 的负责人定版不
表示 Gate A 已执行或通过；Gate A 必须在首次正式 test 前只使用 train/val 预注册，
不得读取 test 或根据当前 validation 快照反推门槛。

### Gate B：通用遥感微调是否有效

Gate B 使用独立协议
`configs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b.yaml`，不是
`rs_vlm.config.v2` 训练配置。协议在首次正式生成前预注册，并绑定 Benchmark 的
manifest/validation/build/payload/hash-ledger、Base/Adapter 配置文件与 semantic SHA、
completed training report、training monitoring selection、validation results、最终
best pointer、step-1000 checkpoint manifest、Adapter 权重、模型/processor 和实现文件
SHA。冻结产物使用：

- `rs_vlm.gate_b_protocol.v1`；
- `rs_vlm.gate_b_selection.v1`；
- `rs_vlm.gate_b_generation.v1`；
- `rs_vlm.gate_b_report.v1`。

固定 Gate B selection 只扫描 manifest 指定的 `external_val` record metadata，不读取
图像，不使用 reference、loss 或模型输出排序。seed 为 `20260802`，排除训练期 128 个
monitoring parents，选择 256 个不同 parent、每个 parent 一条 record。七类任务固定顺序
为 `bbox_region_caption / global_caption / object_count / scene_understanding /
spatial_relation / visible_change_report / visual_qa`。先覆盖全部非空 source-task cell，
再将任务配额尽量均分；稀缺任务不足时按固定任务顺序循环回填，每个 task 内对 source
做确定性 water-fill。每一步通过二分匹配保证剩余 cell 覆盖、task quota 与 parent 唯一性
仍可完成。无法精确得到 256 条、三源七任务、全部可用 cell 或 monitoring 零交集时拒绝
发布。

Base 与 Adapter 分两个进程顺序加载同一个 Qwen3-VL-2B；Adapter 只能从 training root 的
`best_checkpoint.json` 解析最终 step-1000 LoRA。两侧使用相同 processor、
`qwen3vl_messages.v2` renderer、图像/token 限制和 frozen selection 顺序。生成固定为
greedy decoding：`do_sample=false`、`max_new_tokens=384`、temperature `0`、top-p `1`；
实际 Transformers 调用不传无效 sampling kwargs。shard 在解析前、asset 在渲染/解码前
通过共享 hash ledger 核验。任一侧不是恰好 256 predictions、出现 failure、身份或文件
SHA 不匹配，运行均为 `invalid`，不算 Adapter 科学失败。

评价先对每条 Base/Adapter prediction 严格配对。开放生成任务
`bbox_region_caption / global_caption / visible_change_report` 的 primary 为最佳 reference
token-F1 与最佳 reference ROUGE-L-F1 的均值；短答案任务
`visual_qa / object_count / scene_understanding / spatial_relation` 的 primary 为
normalized exact match 与最佳 reference token-F1 的均值。Unicode lowercase `\w+`
分词，token-F1 使用多重集交集，ROUGE-L-F1 使用 token LCS。先在 task 内求均值，再对
七类 task 等权；source macro 先算实际存在的 source-task cell，再在 source 内等权。

置信区间使用 parent-level、task-stratified paired bootstrap：每个 task 内有放回抽样，
七类 task 等权，NumPy `PCG64(20260802)`，10,000 次，线性 2.5%/97.5% percentile。
只有以下六项同时满足才接受 Adapter：

1. Adapter−Base primary task-macro 的 95% CI 下界严格大于 0；
2. 至少四类 task 的 primary delta 严格大于 0；
3. 任一 task primary delta 不低于 `-0.02`；
4. 任一 source macro delta 不低于 `-0.02`；
5. 三类开放任务 macro ROUGE-L delta 不低于 0；
6. 四类短答案 macro normalized exact-match delta 不低于 0。

科学通过、科学未通过和基础设施/合同无效分别使用退出码 `0/1/2`。冻结后不得根据生成
结果修改 selection、指标或门槛；若实现错误，保留原运行并标为 invalid，升级协议版本
且使用全新输出根。只有 `gate-b-evaluate` 已原子写出 `status=completed` 报告后的 exit 1
才表示科学未通过；未捕获的程序错误即使由 Python 返回 1，也不得在缺少 completed report
时解释为科学结论。明确匹配的 CUDA 基础设施 RuntimeError 在生成 CLI 边界归类为
structured invalid / exit 2，普通 RuntimeError 继续抛出。

2026-08-02 正式 v1 运行已经完成并接受 Adapter：Base/Adapter 均为 256 predictions、
0 failures，输入顺序、69,798 tokens 和 338 images 一致；primary task-macro 从
`0.2197314184` 提升到 `0.4559849197`，delta 为 `0.2362535013`，10,000 次 bootstrap
95% CI 为 `[0.2061703267, 0.2670255801]`，七类 task delta 均为正，六项判据全部
PASS。正式 report SHA-256 为
`b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165`；完整 artifact
SHA 锚点见 `REBUILD_PROGRESS.md`。只读 verifier 已重新推导 selection、paired scores、
metrics、bootstrap 和 report，确认发布 selection 与预注册算法逐项一致、monitoring
parent 交集为 0。

该结果只证明 RS-GeneralDesc native v1 固定 lexical protocol 下的 Base-vs-Adapter
相对提升。v1 selection loader 本身不重新推导算法唯一输出，模型完整权重、tokenizer、
generation config、传递实现依赖和运行环境也未形成完整 ledger；当前没有实际漂移证据。
monitoring/Gate 共享一个 asset SHA，涉及两条 Gate spatial records；删除这两条的非门控
事后敏感性仍六项 PASS，CI 下界为 `0.2038789283`。Gate 内 spatial records 存在重复
asset，task/source 配额是 capacity-constrained，lexical metric 同时测量内容、答案风格
与词面重合，v1 也未保存真实 finish reason。这些限制不事后改变 v1 判据或 PASS；未来
重跑须升级 v2，预注册完整 model/tokenizer/runtime ledger、asset-component selection、
task-aware metric、grouped bootstrap 和 finish reason，并使用全新输出根。

### Gate C：模型是否真正关注 mask

比较：

- full image；
- crop；
- overlay + crop；
- mask swap；
- wrong-region；
- empty mask。

若失败，先优化 Evidence Representation，而不是直接加入 RAG。

### Gate D：文本 RAG 是否提供真实增益

Stage 6 首版 Gate D 只比较：

```text
no RAG
vs
Evidence-Constrained Text RAG
```

不把 case RAG、hybrid case RAG 作为 Stage 6 Gate D 的必需条件。

Gate D 重点回答：

1. RAG 是否使专业解释更有依据，而不是只增加语言长度；
2. 是否能够稳定检索到 interpretation、confounder 和 limitation 三类证据；
3. citation 是否真实指向当前 Evidence Packet 和原始 PDF 页码；
4. RAG 是否没有增加 forbidden / unsupported claim；
5. 在证据不足时，RAG 是否能够保留或增强 limitation，而不是强行确认候选区域。

在缺少人工 Gold 的当前阶段，Gate D 先作为 development gate，只完成自动合同、检索质量和 bounded qualitative evaluation，不冻结最终科学阈值。

正式 Gate D 及 sealed-test 阈值在完整算法框架完成、人工评价协议补齐后再冻结。

### Gate E：是否需要 Landslide-Evidence Adapter

只有以下情况之一持续存在才训练：

- mask 对应不足；
- 证据字段格式不稳定；
- no-target 错误；
- modality removal 不敏感；
- RAG 无法控制幻觉。

### Gate F：专业适配是否损伤通用遥感能力

比较 RS-General Adapter 与 Landslide-Evidence Adapter 在固定通用遥感验证集上的性能。

---

## 14. 评价体系

### 14.1 分割评价

- IoU；
- Dice；
- Precision；
- Recall；
- F1；
- positive-only Dice；
- no-target FPR；
- 模态组合；
- 低质量模态；
- 显存和速度。

### 14.2 区域理解评价

- target-status accuracy；
- mask-region consistency；
- structured field accuracy；
- modality attribution；
- evidence sufficiency；
- confounder recognition；
- unsupported claim rate；
- expert factuality；
- no-target correctness。

## 14.3 RAG 评价

Stage 6 v1 优先评价少量真正与当前设计相关的指标。

### 检索层

- Evidence Unit identity validity；
- source/page/section traceability；
- lexical/dense/RRF 可重算性；
- knowledge-type coverage；
- duplicate evidence rate；
- modality applicability；
- citation validity。

在后续建立人工 retrieval gold 后，再增加：

- Recall@K；
- MRR；
- nDCG；
- expert relevance。

### 生成层

- output schema validity；
- claim–evidence binding validity；
- forbidden claim rate；
- citation precision；
- evidence-type utilization；
- limitation preservation；
- no-RAG vs text-RAG 专业解释差异。

当前没有人工 dev Gold 时，不使用 BLEU/ROUGE 作为 RAG 的主要结论。

### 14.4 端到端评价

分开报告：

```text
GT mask + no RAG
GT mask + RAG
fixed predicted mask + no RAG
fixed predicted mask + RAG
end-to-end mask + RAG
wrong mask + RAG
```

---

## 15. RAG 关键消融

Stage 6 首版只保留：

```text
no RAG
vs
text RAG
```

可选的低成本检索消融：

```text
FTS5 only
vs
BGE-M3 only
vs
FTS5 + BGE-M3 + RRF
```

可选的证据平衡消融：

```text
Top-K similarity only
vs
Balanced 2 interpretation + 2 confounder + 2 limitation
```

暂不要求：

- case-only；
- hard-negative case retrieval；
- text + case hybrid；
- reranker 消融；
- knowledge graph；
- multi-agent retrieval。

---

## 16. 防止 RAG 合理化候选区域

最终提示不得写：

```text
这是一个滑坡，请解释其特征。
```

应使用候选区域表述：

```text
该区域由当前空间提示或分割模块指定为候选区域。
请只依据 Pass-1 视觉观察和给定的专业证据，说明哪些现象可能支持滑坡解释，
哪些混淆对象也可能产生相似表现，以及当前证据不能支持哪些结论。
```

Stage 6 通过两项简单机制降低确认偏差：

1. Query Builder 同时生成 Interpretation Query 与 Counter/Limitation Query；
2. Balanced Evidence Packet 固定保留 confounder 和 limitation 配额，不允许全部 Top-K 都是正向滑坡知识。

Stage 6 首轮只需在现有可用开发样本上验证该机制可运行。wrong-mask、shifted-mask、empty-mask 的完整 anti-rationalization 实验可在整体框架打通后统一补做，不阻塞当前 Stage 6 工程开发。

---

## 17. Codex 实施规则

Codex 每次读取：

1. 本文档；
2. `REBUILD_PROGRESS.md`；
3. 根 README；
4. 根 AGENTS。

进展记录只使用 `REBUILD_PROGRESS.md`，记录：

- 当前 Stage；
- 当前目标；
- 已完成项；
- 主要文件；
- 测试；
- 阻塞；
- 下一条命令。

不生成大量：

- handoff；
- ADR；
- license gate；
- 冗余审计报告。

Codex 不因普通子任务结束而暂停。仅在以下情况停止：

- 数据字段或模态含义不明确；
- 需要人工 Gold 审核；
- 需要调用外部付费 API 且未授权；
- 需要覆盖原始数据或 accepted artifacts；
- 需要正式长训练；
- 需要地学专家判断；
- 测试失败且无法从真实错误定位。

---

## 18. 最终完成定义

OA-GroundRAG 主线完成的判断标准不是“所有代码均已实现”或“所有模型均已训练”，而是空间感知、区域视觉理解、专业知识检索和证据约束生成已经形成相互独立、可追溯且能够端到端组合的完整闭环。

工程完成、开发评价、科学 Gate 和 sealed-test 正式评价必须严格区分。某一模块代码存在、训练完成或开发集结果可用，不自动表示对应科学 Gate 已通过。

主线最终完成至少满足以下条件。

1. **OA-AuxSeg 完成稳定的多源滑坡空间感知能力。**

   OA-AuxSeg 保持光学影像为必要主模态，并支持 DEM、slope、InSAR、SAR、多光谱等真实存在的任意辅助模态组合。

   系统能够稳定输出：

   - global mask；
   - candidate regions；
   - no-target；
   - segmentation confidence；
   - 必要的 region information。

   当前负责人定版 checkpoint 可以作为工程主权重，但正式使用 predicted mask 进入最终评价前仍需要完成 Gate A。Gate A 至少验证分割性能、no-target FPR、不同辅助模态组合和 checkpoint 可恢复性。

   跑满计划 `max_steps` 不是独立完成条件。

2. **RS-GeneralDesc Benchmark 与 RS-General Adapter 完成并保持可追溯。**

   RS-GeneralDesc Benchmark 只承担通用遥感视觉语言能力训练，不混入 OA-GroundedEval、test 或滑坡专业知识库内容。

   RS-General Adapter 需要：

   - 完成可恢复 LoRA 训练；
   - 保持模型、processor、Benchmark 和 checkpoint 身份可追溯；
   - 通过独立 Gate B；
   - 证明相对于 Base Qwen3-VL 在通用遥感任务上具有稳定增益。

   Gate B 的结论只代表通用遥感能力适配成功，不自动扩张到 mask-grounded、RAG 或完整系统。

3. **Mask-Grounded Landslide Region Corpus 建立稳定的区域视觉监督。**

   Corpus 只使用允许进入训练的 train split，并至少包含：

   - Original Full RGB；
   - Binary Mask；
   - Clean Context Crop；
   - Programmatic Facts；
   - structured region description；
   - source / parent / split；
   - asset identity；
   - supervision provenance。

   数据可以来自人工核验或经过严格规则过滤的模型辅助监督，但必须明确标记 supervision authority，不得把未审核模型输出伪装成 Gold。

4. **Mask-Grounded Region Adapter 能够完成 Pass-1 区域视觉观察。**

   Region Adapter 在 RS-General Adapter 基础上进行第二阶段轻量适配，用于学习：

   - mask 指定区域的视觉关注；
   - Target Appearance；
   - Target Morphology；
   - Surrounding Environment；
   - Region–Context Contrast；
   - Possible Confusers；
   - Evidence Sufficiency。

   Pass-1 的职责仅为回答：

   > 当前 mask 指定区域及其周围环境“看起来是什么”。

   Pass-1 不访问专业知识库，不生成确定触发原因、发生时间、稳定性、正式风险等级等图像无法直接支持的专业结论。

5. **OA-GroundedEval 能够独立评价 mask-grounded 区域理解。**

   OA-GroundedEval 与训练数据保持 parent/split 隔离，并至少支持：

   - correct mask；
   - no-target；
   - empty mask；
   - shifted mask；
   - context removal；
   - 必要时的 mask swap。

   开发阶段允许先使用 automatic-contract evaluation 验证主链。完整科学评价阶段再补充必要的人工 reference、专家事实性判断和最终阈值冻结。

   当前存在的局部 no-target 或结构化输出问题可以作为后续完整性优化处理，但在正式 Gate C 或 sealed-test 之前必须明确解决或纳入失败统计，不得静默修复。

6. **Mask-Grounded Evidence Builder 能够提供不可被语言模型改写的确定性证据。**

   Evidence Builder 至少组织：

   - target status；
   - bbox；
   - centroid；
   - mask area / area ratio；
   - component count；
   - crop window；
   - image size；
   - modality availability；
   - asset identity；
   - 已知物理量、单位和有效覆盖；
   - missing evidence。

   Programmatic Facts 与视觉生成结果必须分离。

   VLM 和 RAG 都不能修改这些程序确定性事实。

7. **Landslide Text Evidence Bank v1 完成并具有完整来源追溯能力。**

   Stage 6 建立一个统一 Text Evidence Bank，而不是多个复杂物理知识库。

   首版知识只划分为三类：

   - `interpretation`：解释某类遥感视觉现象可能支持什么专业判断；
   - `confounder`：描述道路切坡、采石场、裸岩、河滩、施工扰动等混淆对象；
   - `limitation`：描述不同遥感证据的适用边界、误差来源及进一步核查要求。

   每个 Text Evidence Unit 至少绑定：

   - unit ID；
   - source ID；
   - source SHA-256；
   - PDF page；
   - section / clause；
   - knowledge type；
   - content；
   - modality；
   - conditions；
   - tags；
   - authority class；
   - source status；
   - content hash。

   原始 PDF、解析文本、知识单元和索引之间必须能够追溯。

8. **知识切分保持专业规则的“条件—解释—限制”完整性。**

   Stage 6 不以固定 token chunk 作为唯一切分依据。

   标准优先按条款或章节切分，论文、手册和培训资料优先按自然段或小节切分。

   不允许为了满足固定长度而把：

   > 适用条件 → 可支持解释 → 使用限制

   拆成互相失去上下文的独立证据。

9. **Task Controller 能够确定性控制是否启用 RAG。**

   不额外训练 Retrieval Reflection 模型。

   默认路由为：

   ```text
   Segmentation
   Scene Description
   Mask Description
   → RAG OFF

   Candidate Evaluation
   Multimodal Evidence QA
   Report Generation
   Knowledge QA
   → RAG ON

---

## 19. 最终研究叙事

本研究不是一个把多个模型简单拼接的系统，也不是一个依赖大规模滑坡文本微调的领域大模型。

核心思想是：

```text
OA-AuxSeg 负责可靠空间感知；
程序负责确定性几何和物理统计；
通用遥感 VLM 负责视觉观察和语言交互；
RAG 负责环境、传感器和滑坡专业知识；
最终 VLM 负责生成受证据约束的报告。
```

建议将最终研究叙事中的 RAG 核心表述统一为：

```text
OA-AuxSeg 负责候选区域的空间感知；
程序负责确定性几何和已知物理统计；
RS-General / Mask-Grounded Region Adapter 负责遥感场景和区域视觉观察；
Evidence-Constrained Text RAG 根据当前视觉观察检索解释、混淆和限制知识；
第二遍生成器在不改变视觉事实的条件下完成有来源约束的专业解释。
```

Stage 6 的核心创新不需要表述为“提出复杂的新型 RAG 网络”，而应强调两个与地质灾害遥感解释直接相关的设计：

1. **Evidence-conditioned Retrieval**：检索条件来自 mask-grounded visual observation，而不是仅来自用户问题或整幅图像；
2. **Balanced Evidence Retrieval**：知识检索同时保留解释、混淆和限制证据，避免只检索正向滑坡知识造成确认偏差。

最终需要证明的是：

> 在不继续增加滑坡领域参数微调、不改变候选 mask 和视觉事实的情况下，专业文本知识能否把区域视觉描述提升为更有依据、更知道反例、更清楚证据边界的滑坡遥感解释。

最终论文应证明：

1. 多源辅助模态如何改善或稳定滑坡分割；
2. mask-grounded evidence 如何让 VLM 聚焦正确区域；
3. 通用遥感微调能否保留广泛地物理解；
4. RAG 是否能在不进行大规模滑坡领域微调的情况下提升专业事实性；
5. 专业案例和困难负样本是否比纯规范文本更有效；
6. 系统是否能够识别证据不足，而不是为任意分割结果生成合理化解释。
