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

## 10. RAG 知识库构建

### 10.1 RAG 的定位

RAG 负责：

- 动态提供滑坡专业知识；
- 解释不同环境下的判别差异；
- 提供传感器和产品使用限制；
- 检索相似正案例；
- 检索困难负样本；
- 生成可追溯引用。

RAG 不负责：

- 生成 mask；
- 修改分割输出；
- 覆盖程序事实；
- 根据通用知识强制确认候选区域为滑坡。

### 10.2 知识库分层

#### 层一：专业规则库

来源：

- 地质灾害规范；
- 滑坡遥感解译教材；
- InSAR、SAR、DEM 技术资料；
- 学术论文；
- 调查报告；
- 传感器产品说明。

每个知识单元需要记录：

- 适用环境；
- 适用模态；
- 可观察证据；
- 使用前提；
- 支持结论；
- 禁止结论；
- 混淆因素；
- 来源和页码。

#### 层二：滑坡正案例库

保存 train split 中专家确认的 mask-description 案例。

#### 层三：困难负样本库

包括：

- 采石场；
- 道路切坡；
- 裸岩坡；
- 河滩；
- 冲沟；
- 施工扰动；
- 云影、山影；
- SAR 叠掩、阴影和 speckle；
- InSAR 低相干、大气和解缠异常。

#### 层四：传感器解释库

专门组织：

- InSAR LOS 限制；
- 升降轨差异；
- 相干性和低相干；
- DEM 与坡度；
- SAR 几何畸变；
- 分辨率和尺度影响；
- 数值单位和符号约定。

### 10.3 文本检索

可继续使用：

- SQLite FTS5；
- BGE-M3；
- Qdrant Embedded；
- RRF；
- BGE reranker；
- authority boost。

检索查询来自：

```text
用户问题
+ candidate status
+ Programmatic Facts
+ Pass 1 visual observations
+ environment metadata
+ available modalities
+ required claim types
```

### 10.4 图像和案例检索

不使用 OpenCLIP 统一图文空间。

建议采用分模态索引：

- 光学区域：OA-AuxSeg optical feature 或 DINOv3；
- DEM/slope：terrain encoder feature；
- InSAR：InSAR auxiliary encoder feature；
- SAR：SAR auxiliary encoder feature。

各模态只在同模态索引内检索，后期融合。

### 10.5 案例筛选

只使用 train split。

必须排除：

- val/test parent；
- 同事件样本；
- 同 source group 泄漏；
- exact duplicate；
- perceptual near duplicate；
- 无专家确认 mask；
- 缺少关键元数据的案例。

### 10.6 Evidence Cards

RAG 返回结构化证据，不直接生成最终回答。

每条 Evidence Card 至少包含：

- knowledge/case ID；
- content；
- source；
- page/section；
- modality；
- environment；
- applicable conditions；
- supported claims；
- forbidden claims；
- confounders；
- retrieval score；
- authority class。

### 10.7 检索融合

建议顺序：

```text
metadata filter
→ text lexical retrieval
→ text dense retrieval
→ same-modality case retrieval
→ reciprocal rank fusion
→ reranking
→ source and evidence-type diversity
→ final evidence cards
```

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

### Stage 4：Landslide Evidence Corpus 与 OA-GroundedEval

- 程序生成 Auto facts；
- 分层选择 API/本地教师样本；
- 生成 Silver；
- 完成规则过滤；
- 完成人工 Gold；
- 冻结正式 val/test。

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

### Stage 6：文本 RAG

- 将 RAG_tmp 改为 Evidence Provider；
- 构建规则知识；
- 接入 EvidenceBuilder 查询；
- 完成 no RAG vs text RAG。

### Stage 7：案例 RAG

- 构建正案例；
- 构建困难负样本；
- 建立分模态索引；
- 完成 text-only、case-only 和 hybrid 对照。

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

### Gate D：RAG 是否提供真实增益

比较：

- no RAG；
- text RAG；
- case RAG；
- hybrid RAG。

必须降低 unsupported claim，并提高专家事实性和引用准确率。

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

### 14.3 RAG 评价

- Recall@K；
- MRR；
- nDCG；
- citation precision；
- expert relevance；
- source diversity；
- confounder retrieval；
- irrelevant knowledge robustness；
- modality match accuracy。

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

## 15. 关键消融

### 分割

- optical-only；
- direct concat；
- mean fusion；
- CMNeXt injection；
- quality selection；
- modality dropout。

### VLM

- Base；
- RS-General Adapter；
- optional Landslide-Evidence Adapter；
- full image；
- crop；
- overlay；
- overlay + crop；
- multimodal evidence。

### RAG

- no RAG；
- text-only；
- positive cases；
- hard-negative cases；
- text + cases；
- 无 metadata filter；
- environment-conditioned filter；
- sensor-conditioned filter。

### 反事实

- mask swap；
- region swap；
- empty mask；
- modality removal；
- wrong sign/unit；
- irrelevant knowledge injection；
- conflicting knowledge；
- wrong predicted mask。

---

## 16. RAG 防止合理化错误分割

最终提示和任务定义不得写：

```text
这是一个滑坡，请解释其特征。
```

应写为：

```text
该区域由滑坡分割模型标记为候选区域。请根据当前视觉证据、
程序事实和检索知识，评估其是否具备滑坡遥感特征，并列出
支持证据、反对证据、可能混淆对象和证据限制。
```

RAG Query Builder 必须同时检索：

- 滑坡支持知识；
- 相似正案例；
- 困难负样本；
- 证据限制；
- 传感器误差来源。

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

主线完成必须满足：

1. OA-AuxSeg 完成负责人定版的工程报告，并在未来完成正式评价和 Gate A；跑满计划
   `max_steps` 不是独立完成条件；
2. RS-GeneralDesc Benchmark 完成 external-only 验收；
3. RS-General Adapter 完成训练并通过通用遥感 gate；
4. Landslide Evidence Corpus 完成 Auto、Silver 和必要 Gold；
5. OA-GroundedEval 完成人工审核并封存 test；
6. Evidence Builder 能构建光学、辅助模态、程序事实和证据约束；
7. Qwen3-VL 能完成 mask-grounded 视觉观察；
8. RAG 能返回文本规则、正案例和困难负样本；
9. RAG 返回结构化 Evidence Cards 和引用；
10. 最终 VLM 输出支持、反对、混淆对象和限制；
11. wrong-mask + RAG 不会被稳定合理化为滑坡；
12. GT、fixed predicted 和 end-to-end 分层评价；
13. 可选 Landslide-Evidence Adapter 只在 Gate 失败时实施；
14. 通用遥感能力 retention 得到验证；
15. 统一 Task Controller 能编排分割、描述、证据问答和报告；
16. RAG 失败不影响分割模型独立运行；
17. README 只保留当前有效命令；
18. `REBUILD_PROGRESS.md` 反映真实科学完成状态，而不是仅反映代码存在。

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

最终论文应证明：

1. 多源辅助模态如何改善或稳定滑坡分割；
2. mask-grounded evidence 如何让 VLM 聚焦正确区域；
3. 通用遥感微调能否保留广泛地物理解；
4. RAG 是否能在不进行大规模滑坡领域微调的情况下提升专业事实性；
5. 专业案例和困难负样本是否比纯规范文本更有效；
6. 系统是否能够识别证据不足，而不是为任意分割结果生成合理化解释。
