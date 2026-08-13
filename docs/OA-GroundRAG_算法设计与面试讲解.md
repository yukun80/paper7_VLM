# OA-GroundRAG 算法设计与面试讲解

> 面向第一次接触遥感、多模态大模型和 RAG 的读者。目标不是背术语，而是能从问题出发，
> 解释这个项目为什么这样设计、每个模块如何协作，以及面试时哪些结论可以说、哪些不能说。

## 0. 先说明这份文档的边界

这是一份**学习与面试讲解**，不是新的算法协议，也不是实时进度报告。

- 冻结算法定义以 [OA-GroundRAG 算法构建方案](OA-GroundRAG_算法构建方案_0811.md) 为准。
- 当前进度、Gate 状态和产物身份只看 [REBUILD_PROGRESS](../REBUILD_PROGRESS.md)。
- 稳定运行入口看 [README](../README.md)，协作规则看 [AGENTS](../AGENTS.md)。
- 本文只解释现有设计，不宣布新的科学结果，不把开发评价写成正式验收。
- 文中的小林面试题链接只用于标明题目来源。回答全部根据本项目重新组织，没有复制原文。

如果只记住一句话，请记住：

> OA-GroundRAG 不是让一个大模型包办所有事情，而是让空间专家、视觉语言模型和知识检索
> 各做自己擅长的事，再由显式任务决定本次需要调用哪些能力。

### 建议阅读路线

- 第一次阅读：[角色直觉与术语](#guide-basics) → [具体案例](#guide-example) →
  [核心模块](#guide-modules) → [科学边界](#guide-science)。
- 准备项目介绍：[Q1–Q7 架构题](#interview-architecture)。
- 准备算法岗位：[Q8–Q15 LLM/VLM 题](#interview-vlm)。
- 准备 RAG 岗位：[Q16–Q26 RAG 题](#interview-rag)。
- 准备系统与科研追问：[Q27–Q30 Runtime/Governance 题](#interview-runtime)。
- 面试前十分钟：[十句复习卡](#review-cards)。

所有链接使用显式英文锚点，即使阅读器不自动生成中文标题锚点也能跳转。

<a id="guide-basics"></a>

## 1. 用四个角色建立直觉

想象有一支遥感判读小组：

| 类比角色 | 它回答什么 | 项目能力 | 稳定代码位置 |
| --- | --- | --- | --- |
| 定位员 | 候选区域在哪里？ | OA-AuxSeg / Spatial Perception | [segmentation](../oa_groundrag/segmentation/) |
| 观察员 | 指定区域在图像中呈现什么？ | Shared MLLM + Grounding | [vlm](../oa_groundrag/vlm/)、[grounding](../oa_groundrag/grounding/) |
| 知识查证员 | 这些观察有哪些专业解释、混淆因素和限制？ | Evidence-Constrained Text RAG | [retrieval](../oa_groundrag/retrieval/) |
| 调度员 | 这次任务需要谁参加、按什么顺序？ | Deterministic Runtime Router | [runtime](../oa_groundrag/runtime/) |

四个角色组成的主链是：

~~~text
在哪里？       → OA-AuxSeg
看到了什么？   → Grounded MLLM
如何谨慎解释？ → Evidence-Constrained RAG
何时调用？     → Deterministic Runtime Router
~~~

这里最重要的词是“**按需**”。普通遥感图像描述不需要先做分割；纯分割不需要加载大模型；
专业知识问答不需要图像；只有候选区域解释才需要把区域观察与知识增强接起来。

## 2. 零基础术语表

| 术语 | 最直白的解释 | 在本项目里的含义 |
| --- | --- | --- |
| 像素 | 图像中的一个小格子 | 分割模型对每个格子预测属于目标的可能性 |
| mask | 与图像同尺寸的黑白区域图 | 白色表示关注区域，黑色表示其他区域；它不是地物真实颜色 |
| 分割 | 给每个像素分类 | OA-AuxSeg 输出候选滑坡区域的概率图和 mask |
| no-target | 当前样本没有可用目标区域 | 不是报错；它是必须被严格表达的一种合法结果 |
| LLM | 主要处理和生成文本的大语言模型 | Shared MLLM 中负责自回归语言生成的主体 |
| VLM / MLLM | 同时处理图像和文字的视觉语言模型/多模态大模型 | 本项目使用 Qwen3-VL-2B-Instruct |
| Grounding | 把语言回答约束到明确的图像区域和证据 | 原图、二值 mask、clean crop、程序事实共同限定“正在说哪里” |
| Programmatic Facts | 程序从 mask 确定计算的事实 | bbox、质心、面积、组件数等；模型只能读取，不能改写 |
| LoRA | 冻结大部分基座参数，只训练小型低秩增量 | 项目只在 attention 的 q/k/v/o 投影上训练适配参数 |
| Tokenizer | 把文本转换成模型能处理的 token ID | VLM 输入编码和 RAG 文本长度检查都依赖 tokenizer |
| Embedding | 把一段文字变成语义向量 | BGE-M3 把问题和证据单元映射到可比较的向量空间 |
| RAG | 先检索外部资料，再基于资料生成 | 给专业解释提供可追溯的文本证据，不改变模型权重 |
| Router | 根据显式任务生成执行计划 | 不看图、不推理、不让 LLM 猜任务，只映射 <code>UnifiedTask</code> |
| Provider | 对某项已有能力的稳定运行封装 | spatial、shared_mllm、evidence、text_rag |
| Gate | 预先定义的科学验收关口 | 工程能跑不等于 Gate 已通过 |
| provenance | 一个结果从何而来、由什么配置和资产产生 | Git、配置、manifest、SHA 和 checkpoint 元数据共同追溯 |

## 3. 只需要理解的三个公式

### 3.1 Attention：当前 token 应该关注哪些信息

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

- <code>Q</code>（Query）表示“我现在想找什么”。
- <code>K</code>（Key）表示“每条已有信息适合被怎样匹配”。
- <code>V</code>（Value）表示“匹配后真正取回什么内容”。
- <code>QKᵀ</code> 得到匹配分数，除以 <code>√d_k</code> 防止维度较大时数值过尖。
- <code>softmax</code> 把分数变成权重，最后对 <code>V</code> 加权汇总。

本项目没有重新发明 Attention；它在 Qwen3-VL 的既有 Attention 投影
<code>q_proj/k_proj/v_proj/o_proj</code> 上加入 LoRA。

### 3.2 LoRA：只学习一个小增量

$$
W^\prime = W + \frac{\alpha}{r}BA
$$

- <code>W</code> 是冻结的基座权重。
- <code>A</code> 和 <code>B</code> 是需要训练的小矩阵。
- <code>r</code> 是低秩维度；项目配置为 8。
- <code>α</code> 控制增量缩放；项目配置为 16。

直觉上，大矩阵 <code>W</code> 不动，只学习“如何稍微调整它”。这使单卡训练可行，也让
基座模型身份与适配器身份可以分别记录。

### 3.3 RRF：不直接比较两种检索分数

$$
\operatorname{RRF}(d)=\sum_i\frac{1}{k+\operatorname{rank}_i(d)}
$$

- <code>d</code> 是一个候选证据单元。
- <code>rank_i(d)</code> 是它在第 <code>i</code> 条检索路径中的名次。
- <code>k</code> 是平滑常数；项目配置为 60。

关键词检索分数和向量相似度不是同一量纲，硬把原始分数相加不可靠。RRF 只融合“名次”，
让一个同时被两路排在前面的证据获得更高融合分。

<a id="guide-example"></a>

## 4. 从一个问题走完整条链路

假设用户提出：

> 请定位影像中的候选区域，并结合可追溯资料说明可能解释、混淆因素和仍需核查的证据。

### 4.1 总体能力图

~~~mermaid
flowchart LR
    U[用户显式选择任务] --> R[Deterministic Router]
    R -->|需要空间定位| S[OA-AuxSeg]
    R -->|需要区域证据| G[Grounded Evidence Interface]
    R -->|需要视觉语言能力| V[Shared RS-Geohazard MLLM]
    R -->|需要专业知识| K[Text Evidence Retrieval]
    S -->|概率图 / mask / 候选区域| G
    G -->|原图 + mask + clean crop + 程序事实| V
    V -->|结构化视觉观察| K
    K -->|平衡 Evidence Packet| V
    V --> O[严格结构化响应]
~~~

不支持 Mermaid 的阅读器，可以按下面七步理解：

1. 用户明确选择任务，而不是让模型猜。
2. Router 只生成执行计划。
3. 如果需要定位，OA-AuxSeg 输出 mask 和候选区域。
4. Grounding 把区域变成原图、二值 mask、无标记 crop 和只读几何事实。
5. Shared MLLM 先描述当前图像直接支持的视觉观察。
6. 只有任务需要专业解释时，Retrieval 才检索知识并构造 Evidence Packet。
7. Shared MLLM 在第二遍生成中引用 packet，输出解释、替代解释、限制和核查建议。

### 4.2 训练与运行是两张不同的图

~~~mermaid
flowchart TB
    subgraph Training_Curriculum[训练 curriculum：能力如何获得]
        B[Qwen3-VL-2B 基座] --> RSG[RS-General Adaptation]
        RSG --> GR[Grounded Region Adaptation]
        Replay[RS-General Replay] --> GR
        GR --> SA[Shared RS-Geohazard Adapter]
    end

    subgraph Runtime_Architecture[运行 architecture：请求需要谁]
        Task[UnifiedTask] --> Router[Capability Router]
        Router --> Spatial[可选 OA-AuxSeg]
        Router --> Shared[一个 Shared MLLM + 最终 Adapter]
        Router --> Rag[可选 Text RAG]
    end
~~~

文字兜底：

1. 训练时先让模型适应通用遥感描述，再让同一适配路径学习 mask-grounded 区域观察。
2. Grounded 训练以 RS-General checkpoint warm start，并混入通用遥感 replay。
3. 运行时不会先加载 RS-General Adapter、再串行加载 Grounded Adapter。
4. 运行时只加载最终 Shared Adapter；RS-General 主要保留为 warm start、基线和 retention 参照。

### 4.3 Region Interpretation 时序

~~~mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant S as Spatial Provider
    participant E as Evidence Provider
    participant M as Shared MLLM
    participant T as Text RAG

    U->>R: REGION_INTERPRETATION + 显式 region_source
    alt OA_AUXSEG_CANDIDATE
        R->>S: infer
        S-->>R: global mask + candidates
        R->>S: release
    else USER_MASK
        Note over R,S: 不调用 OA-AuxSeg
    end
    R->>E: build grounded evidence
    E-->>R: full image + binary mask + crop + facts
    R->>M: Pass-1 visual observation
    M-->>R: structured observation
    R->>M: release
    R->>T: retrieve with two deterministic queries
    T-->>R: balanced Evidence Packet
    R->>T: release
    R->>M: Pass-2 text-only constrained generation
    M-->>R: cited interpretation JSON
    R->>M: release
    R-->>U: UnifiedResponse + audit trace
~~~

文字兜底：

1. 用户 mask 路径跳过分割；OA-AuxSeg candidate 路径才先调用空间 provider。
2. 区域证据构建是确定性程序，不让 LLM 自己画框或计算面积。
3. 第一遍只做视觉观察。
4. 释放 Shared MLLM 后再加载/运行检索所需资源，降低单卡峰值常驻压力。
5. 第二遍是 text-only：消费问题、程序事实、第一遍观察和检索 packet。
6. 输出必须通过 schema、Evidence ID、知识类型和禁止式结论检查。

### 4.4 RAG 的离线与在线

~~~mermaid
flowchart LR
    subgraph Offline[离线：建立 Text Evidence Bank]
        PDF[登记的 PDF] --> Parse[文本解析 / 必要时 OCR]
        Parse --> Unit[按结构切成 Evidence Unit]
        Unit --> Meta[来源 / 页码 / 章节 / 类型 / SHA]
        Meta --> FTS[SQLite FTS5 索引]
        Meta --> Emb[BGE-M3 归一化向量]
    end

    subgraph Online[在线：只在任务需要时检索]
        Q[用户问题 + 可选区域观察] --> Q1[解释查询]
        Q --> Q2[混淆与限制查询]
        Q1 --> Hybrid[FTS5 + Dense]
        Q2 --> Hybrid
        Hybrid --> RRF[RRF 融合]
        RRF --> Packet[三类平衡 Evidence Packet]
        Packet --> Pass2[证据约束生成]
    end
~~~

文字兜底：

1. 离线阶段先固定来源身份，再解析、切分、去重和建立两类索引。
2. 在线阶段不让 LLM自由改写问题，而是从固定字段构造两条查询。
3. 每条查询均走关键词和 dense 两路召回，再用 RRF 融合。
4. packet 按 interpretation、confounder、limitation 分配配额，并避免同页重复占满。
5. 生成结果中的 Evidence ID 必须真实存在且类型匹配。

区域解释会把 Pass-1 observation 和 Programmatic Facts 加入上下文；纯
<code>KNOWLEDGE_QA</code> 没有图像、mask 或 Pass-1，使用空 observation/facts 与
<code>general</code> 模态合同。两者都使用确定性双查询，但不能说知识问答也先“看图一遍”。

## 5. 六类任务不是六套模型

| <code>UnifiedTask</code> | 必要输入 | Spatial | Grounding | Shared MLLM | Text RAG | 主要输出 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| <code>VLM_ONLY</code> | 图像 + instruction | 否 | 否 | 是 | 否 | 普通视觉文本 |
| <code>SEGMENT_ONLY</code> | spatial input | 是 | 否 | 否 | 否 | 概率图、mask、候选区域 |
| <code>REGION_UNDERSTANDING</code> | 单图 + user mask | 否 | 是 | 是 | 否 | 区域视觉观察 |
| <code>SEGMENT_AND_UNDERSTAND</code> | spatial input + instruction | 是 | 是 | 是 | 否 | 全局 mask 对应的区域观察 |
| <code>KNOWLEDGE_QA</code> | 纯文本问题 | 否 | 否 | 是 | 是 | 有文本证据约束的知识回答 |
| <code>REGION_INTERPRETATION</code> | 图像/空间输入 + 指定区域来源 + 问题 | 视来源而定 | 是 | 是 | 是 | 区域观察与专业解释 |

注意三点：

- 表里的“是”表示调用某项能力，不表示把所有模型同时放在 GPU。
- <code>REGION_INTERPRETATION</code> 使用 user mask 时不调用 OA-AuxSeg；使用
  OA-AuxSeg candidate 时才调用。
- Router 是确定性控制逻辑，不是论文里的新神经网络，也不做语义自动分类。

真实映射见 [router.py](../oa_groundrag/runtime/router.py) 和
[contracts.py](../oa_groundrag/runtime/contracts.py)。

<a id="guide-modules"></a>

## 6. 五个核心模块：为什么这样设计，实际上怎样工作

### 6.1 OA-AuxSeg：专业“定位员”

**解决的问题。** 普通 VLM 擅长用语言概括图像，但像素级边界是另一种任务。OA-AuxSeg
专门回答候选区域在哪里，保留独立的分割模型比让语言模型顺便输出 mask 更容易冻结数学、
验证 checkpoint 和单独评价空间能力。

**实际结构。** 当前模型使用 ConvNeXt-Small 光学主干、多阶段辅助模态融合和
SegFormer-style 四尺度 decoder。它输出 mask logits、mask probability、no-target score、
modality weights、候选连通区域和区域特征。详见
[model.py](../oa_groundrag/segmentation/model.py)、
[fusion.py](../oa_groundrag/segmentation/fusion.py) 和
[regions.py](../oa_groundrag/segmentation/regions.py)。

**为什么光学锚定。** 光学影像定义空间边界基准；DEM、InSAR velocity、slope 等是可缺失
的辅助证据。registry 明确记录某个样本有哪些模态，模型通过 availability 与 null auxiliary
处理缺失，而不是从文件名猜通道含义。当前受支持辅助 registry 的实际边界应以
[contracts.py](../oa_groundrag/segmentation/contracts.py) 和实时状态为准，不能把未来计划
中的 SAR 说成已经接入。

**no-target 为什么重要。** 空 mask 可能意味着当前没有候选，而不是系统坏了。后续合同要求
no-target 时不得虚构 bbox、crop、区域数组或目标存在性。严格失败比“为了输出好看而自动补一个
区域”更适合科研审计。

### 6.2 Shared MLLM：共享“观察员”

项目使用本地 Qwen3-VL-2B-Instruct 作为共享语义主体。视觉 encoder/merger 保持冻结，
只在 Attention 的 <code>q_proj/k_proj/v_proj/o_proj</code> 上训练 rank-8 LoRA；配置见
[RS-General LoRA](../configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml)。

这里的“Shared”有两个含义：

1. 普通遥感描述和 mask-grounded 区域观察使用同一个 MLLM 主体。
2. 不为每个 runtime task 串接不同 Adapter；最终 Grounded Adapter 已从 RS-General
   warm start，并通过 replay 维持通用能力。

Grounded curriculum 的 region/replay 比例、warm start 和生成限制见
[Grounded train-only 配置](../configs/vlm/grounded/train_v2.yaml)。
运行 loader 只读取最终 trainable LoRA state，见
[grounded_runtime.py](../oa_groundrag/vlm/grounded_runtime.py)。

### 6.3 Grounding：把“哪里”变成可审计输入

Grounding 不是再训练一个分割器，而是一份明确的证据接口：

- <code>optical_full</code>：完整、无标记的光学图，保留全局场景。
- <code>binary_mask</code>：严格二值位置提示；白色不是地物颜色。
- <code>context_crop</code>：直接从原始 RGB 裁剪的无标记局部图；crop 边缘不是目标边界。
- Programmatic Facts：bbox、质心、面积比例、组件数、紧致度、伸长率等确定性事实。
- Limitations：对齐、覆盖率、单位、符号约定和当前证据缺口。

为什么不是只给 crop？因为 crop 会丢失全局环境，还容易让模型把裁剪边缘误认为目标边界。
为什么不是只给 mask overlay？因为彩色覆盖会污染真实颜色。正式输入使用原图、二值 mask 和
clean crop；overlay 只允许作为明确标记的审计消融。

程序事实由 [evidence.py](../oa_groundrag/grounding/evidence.py) 计算，消息合同由
[messages.py](../oa_groundrag/grounding/messages.py) 构造。模型只能观察视觉内容，不能
重新生成或修改几何。

还要区分空间模型的多源输入与 Shared MLLM 的多源证据：当前 Unified P0 合同中的
<code>auxiliary_views</code> 只允许 region task 携带对齐、覆盖率、单位和符号约定等元数据，
不能据此宣称 Shared MLLM 已实际消费完整 SAR/InSAR/DEM grounded 图像证据。多源 Grounded
Evidence 的实时实施状态与下一任务只看 [REBUILD_PROGRESS](../REBUILD_PROGRESS.md)。

### 6.4 Evidence-Constrained Text RAG：知识“查证员”

Text Evidence Bank 保存的不是任意网页，而是登记来源的专业 PDF 证据单元。每个单元带有
来源、页码、章节、知识类型、适用模态和可重算身份。构建逻辑见
[bank.py](../oa_groundrag/retrieval/bank.py)，来源登记见
[sources_v1.yaml](../configs/retrieval/sources_v1.yaml)。

在线检索的真实组合是：

~~~text
解释查询 + 混淆/限制查询
        ↓
每条查询：FTS5 BM25 + BGE-M3 dense
        ↓
RRF 融合
        ↓
interpretation / confounder / limitation 平衡 packet
        ↓
text-only Pass-2
~~~

当前规模下使用 SQLite FTS5 和本地 NumPy dense matrix 足够透明、可复现，也没有证据表明
必须引入独立向量数据库或 learned reranker。配置中的召回深度、RRF 常数和配额见
[runtime_v1.yaml](../configs/retrieval/runtime_v1.yaml)，实现见
[search.py](../oa_groundrag/retrieval/search.py)。

Pass-2 不是“搜到什么就自由发挥”。解析器检查严格 JSON、Evidence ID、证据类型、禁止重写
几何、禁止把候选确认成滑坡、禁止把文献知识说成当前图像观察。见
[pass2.py](../oa_groundrag/retrieval/pass2.py)。

### 6.5 Unified Runtime：确定性“调度员”

Router 只把显式 <code>UnifiedTask</code> 映射为 <code>ExecutionPlan</code>。它不调用 LLM
猜意图，也不做 ReAct、反思或自主规划。Provider 失败会返回稳定 reason code，不会偷偷换用
另一个能力。

单卡约 24 GB 的关键不是把所有东西永久放进显存，而是：

1. 按任务惰性加载所需 provider。
2. 空间推理完成后按配置释放空间模型。
3. 第一遍视觉观察完成后释放 Shared MLLM，再运行检索。
4. 第二遍需要生成时重新获得 Shared MLLM，并在结束后释放。
5. 记录 provider 调用、释放、身份和显存 trace。

配置见 [inference_v2.yaml](../configs/runtime/inference_v2.yaml)，编排见
[inference.py](../oa_groundrag/runtime/inference.py)。

候选 ID 缺失或不存在时，runtime 可以按明确合同退到 OA-AuxSeg global mask，并在
<code>RegionSelection</code> 和 limitation 中记录原因。这是**特定且可见的候选选择规则**，
不是“所有错误都自动降级”。模型输出非法、身份不匹配、test/sealed 路径或 provider 失败仍然
fail closed。

## 7. 关键设计取舍

| 选择 | 为什么 | 代价或边界 |
| --- | --- | --- |
| 专业分割器与 MLLM 解耦 | 数学、权重、数据和评价可分别冻结 | 需要明确的 Grounding 接口 |
| 显式 task enum | 路由可预测、可测试、可审计 | 用户/上层调用方必须给出任务 |
| RAG 按需启用 | 普通视觉任务不被额外延迟和知识干扰 | 需要准确区分任务合同 |
| 视觉观察与知识解释分两遍 | 防止把文献知识伪装成图像观察 | 推理步骤更多，需管理资源释放 |
| 程序事实只读 | 几何不依赖语言模型猜测 | 模型不能“修正”程序输出 |
| FTS5 + BGE-M3 + RRF | 兼顾精确词与语义，融合透明 | 尚无 learned reranker |
| 三类平衡 packet | 同时给支持、反例和限制 | 可能牺牲单一相关性排序的顶部得分 |
| 严格 JSON 与拒绝静默修复 | 暴露真实模型失败，便于科研复现 | 表面有效率可能低于自动修复方案 |
| lazy provider | 适配单卡资源，避免全模型常驻 | 会有重新加载开销 |
| 不采用 Agent | 当前六任务和依赖关系固定，无需自主规划 | 不能处理开放式自主任务分解 |

当前主线明确不引入 Agent 自主规划、知识图谱、Case RAG、学习型 Router 或 joint
OA-AuxSeg/Qwen/RAG training。未来是否引入必须由新的问题、消费者和科学证据驱动，不能因为
它们流行就加入。

<a id="guide-science"></a>

## 8. 科研结果应该怎样表述

请区分三层：

1. **工程可运行**：代码、合同、路由、测试和受限 smoke 能执行。
2. **自动开发评价**：在 train/val 范围用自动指标观察行为，仍可能没有 Gold 或专家共识。
3. **科学验收**：预注册指标和阈值、冻结协议、独立评价及首次 test 规则共同满足。

因此：

- checkpoint 可加载，不等于模型科学有效。
- 开发集指标改善，不等于正式 Gate 通过。
- 自动引用合法，不等于专业解释正确。
- candidate mask 不等于“确认滑坡”。
- 没有 retrieval Gold 时，不能声称 Recall@K、MRR 或 nDCG 已证明检索有效。
- test/sealed test 不能用于反推阈值、Prompt 或 parser。

当前具体状态不要从本文记忆，请始终查看 [REBUILD_PROGRESS](../REBUILD_PROGRESS.md)。

## 9. 面试题使用方法

下面恰好 30 道题。每题都包含：

- **来源**：小林题库原题或本项目追问；
- **口语回答**：先给出可以直接说的 30–60 秒主答；
- **概念拆解**：解释为什么；
- **项目落地**：说明当前系统如何做，并给出证据路径；
- **常见追问**：准备面试官继续深挖；
- **不要这样回答**：避免夸大或混淆。

口语回答不是让你逐字背诵。更好的方式是先记住每题的“问题—选择—理由—边界”四步，
然后用自己的语言回答。

### 小林题库筛选结果

| 小林专题 | 与项目高度相关的题 | 本文对应 | 只作延伸或不作为项目经验 |
| --- | --- | --- | --- |
| [RAG](https://xiaolinnote.com/ai/rag/rag_info.html) | RAG 流程、RAG vs 微调、Chunking、Embedding、关键词/向量检索、Query Rewrite、多路召回、向量库/图数据库选型、幻觉、评价、更新、落地难点 | Q14、Q16–Q26 | Self-RAG 等高级范式未实现；图数据库只讲选型边界 |
| [大模型工程](https://xiaolinnote.com/ai/llm/llm_info.html) | LLM/VLM 基础、Transformer、Tokenizer、微调、LoRA、解码、Prompt、幻觉、评价、模型选型 | Q8–Q15、Q25、Q29–Q30 | RoPE、GQA、FlashAttention、KV Cache、量化、DPO/GRPO、MoE 和部署框架未做专项实现 |
| [Agent](https://xiaolinnote.com/ai/agent/agent_info.html) | Workflow、Agent、Tools 的区别 | Q27–Q28 | ReAct、Reflection、记忆和 Multi-Agent 不在当前主线 |
| [工具调用与框架](https://xiaolinnote.com/ai/) | 只用于帮助区分内部 provider 与外部工具协议 | 第 14 节边界表 | Function Calling、MCP、Skill、LangChain/LangGraph 均不能包装成项目经验 |

Q1–Q7、Q24、Q28 等“项目追问”不是照抄网站题目，而是面试官在听完通用概念后最可能沿着
当前架构继续追问的设计选择。它们的答案以本地实现为准。

<a id="interview-architecture"></a>

## 10. 架构与多模态主线（Q1–Q7）

### Q1. 请用一分钟介绍 OA-GroundRAG

**来源：** 本项目高频开场题；可结合小林题库“你的项目里有没有用过 LLM，怎么用的”的
[总览](https://xiaolinnote.com/ai/)。

**口语回答：**

> OA-GroundRAG 是一个面向地质灾害遥感的 instruction-routed 多模态框架。它把问题拆成
> 三层：OA-AuxSeg 负责候选区域在哪里，Shared Qwen3-VL 负责该区域在影像中呈现什么，
> Text RAG 负责基于可追溯专业资料给出可能解释、混淆因素和限制。中间通过 Grounded
> Evidence Interface 传递原图、二值 mask、clean crop 和只读几何事实。系统不是固定流水线，
> 而是由六类显式任务确定性选择能力；这样既保留专业分割精度，也控制大模型幻觉和单卡资源。

**概念拆解：**

- “instruction-routed”表示用户先明确任务，Router 再生成执行计划。
- “grounded”表示回答必须指向明确区域、图像证据和程序事实。
- “knowledge-augmented”表示外部知识按需检索，不是把所有资料永久写进参数。

**项目落地：** 空间能力在 [segmentation](../oa_groundrag/segmentation/)，语义主体在
[vlm](../oa_groundrag/vlm/)，区域合同在 [grounding](../oa_groundrag/grounding/)，知识增强在
[retrieval](../oa_groundrag/retrieval/)，统一编排在 [runtime](../oa_groundrag/runtime/)。

**常见追问：**

- 问：核心创新是不是 Router？答：不是。Router 是工程组织机制；科学问题在空间感知、
  grounded evidence 和证据受限知识增强。
- 问：输出是不是确认滑坡？答：不是。系统输出候选区域、可见观察和受约束解释，不能越权为
  最终灾害确认。

**不要这样回答：** “这是一个先分割、再问大模型、最后搜文档的流水线。”这既漏掉按需路由，
也漏掉只读事实、两遍生成和证据合同。

### Q2. 为什么按 Where、What、How 拆分问题

**来源：** 本项目追问。

**口语回答：**

> 因为“在哪里”“看到了什么”“专业上如何解释”是三种不同的不确定性。Where 是像素级空间
> 问题，适合专业分割模型；What 是指定区域的视觉语言理解；How 需要外部专业知识和对混淆
> 因素的约束。如果全部交给一个模型，它很容易把候选定位、视觉事实和专业推断混成一句流畅
> 但无法审计的话。拆开后每层都有独立输入、输出、失败原因和评价方法。

**概念拆解：**

- Where 的基本单位是像素，指标通常关心区域重叠、空目标和边界。
- What 的基本单位是可见观察，必须区分区域内部、周围环境和两者差异。
- How 的基本单位是有来源的专业命题，还要同时考虑反例、限制和核查建议。

**项目落地：** OA-AuxSeg 输出 mask；Grounding 把 mask 转成可审计 evidence；Shared MLLM
输出结构化观察；RAG 只在解释类任务中提供 interpretation/confounder/limitation 证据。

**常见追问：**

- 问：拆分会不会导致错误传播？答：会，所以接口必须显式保存 mask 来源、target status、
  limitation 和 trace，而不是隐藏上游不确定性。
- 问：为什么不让 RAG 直接看图？答：当前 Text RAG 检索的是专业文本知识；图像事实先由
  grounded visual pass 产生，避免文献知识倒灌成“图上看见了什么”。

**不要这样回答：** “分模块只是代码更整齐。”主要动机是科学问题、事实类型和评价合同不同，
代码结构只是结果。

### Q3. 为什么分割专家与 MLLM 解耦，而不是端到端联合训练

**来源：** 本项目追问；可联系小林的
[大模型微调方案](https://xiaolinnote.com/ai/llm/finetuning.html)。

**口语回答：**

> 当前项目保留 OA-AuxSeg 独立，是因为像素定位已有专业网络、冻结权重和独立数据合同，而
> Qwen3-VL 负责语言与多图理解。用 Grounding 接口连接二者，就能分别验证 mask 数学和语言
> 输出，也能让用户 mask 直接进入区域理解。联合训练会增加显存、数据配对和梯度耦合风险，
> 还会改变已有 checkpoint 语义；在没有证据证明联合训练必要之前，不值得扩大问题。

**概念拆解：**

- “解耦”不是模块互不交流，而是通过版本化输入输出合同交流。
- 端到端训练的优点是可能共同优化，代价是每个错误更难归因。
- 独立模型允许只运行分割、只运行 VLM，也允许用户提供外部 mask。

**项目落地：** OA-AuxSeg 的 forward 不依赖 Qwen hidden state；Grounding 消费 mask 与图像
生成 evidence；Shared MLLM 消费 evidence message。runtime 会按任务顺序调用并释放 provider。

**常见追问：**

- 问：是否考虑过 <code>[SEG]</code> token？答：相关工作有这类路线，但当前项目明确不以
  LLM token 驱动 OA-AuxSeg，也不声称比较后全面优于它。
- 问：什么时候才值得联合训练？答：需要明确失败证据表明接口式方案无法学习关键依赖，并有
  足够配对数据、显存和预注册评价；不是因为“端到端听起来更先进”。

**不要这样回答：** “解耦一定比分割大模型更准。”项目选择的是可复现性和现有资产适配，
不是已经完成所有路线的公平比较。

### Q4. OA-AuxSeg 为什么以光学为锚，并如何处理辅助模态缺失和 no-target

**来源：** 本项目追问。

**口语回答：**

> 光学影像提供统一的空间边界基准，因此主干始终从光学特征出发；DEM、InSAR velocity、
> slope 等只作为实际可用的辅助证据。每个样本通过 registry 和 availability 明确声明模态，
> 模型有 null auxiliary 语义并学习辅助权重，不能靠补零后假装模态存在。输出还包含
> no-target score：没有有效目标是合法状态，后续必须返回空区域合同，不能硬造 bbox 或描述。

**概念拆解：**

- “锚定”表示空间坐标和主要边界以光学为准，不表示其他模态没有价值。
- “任意辅助”表示允许支持集合中的模态组合缺失，不表示任何未知传感器都能直接输入。
- no-target 与“低置信但强制选一个区域”不同，它保护负样本和空场景语义。

**项目落地：** 模态 registry 与顺序在
[segmentation/contracts.py](../oa_groundrag/segmentation/contracts.py)，融合在
[segmentation/fusion.py](../oa_groundrag/segmentation/fusion.py)，输出在
[segmentation/model.py](../oa_groundrag/segmentation/model.py)。

**常见追问：**

- 问：缺失模态是不是全部填 0？答：张量实现可能需要占位，但 availability/null auxiliary
  合同必须让模型知道“缺失”与“数值为零”不是一个科学含义。
- 问：SAR 已经用了吗？答：不能这样说。当前实际 registry 边界以代码和实时进度为准，
  未来多源计划不能当成已实现能力。

**不要这样回答：** “OA-AuxSeg 可以自动理解任何遥感通道。”通道、单位、配准和有效性都
必须有显式合同。

### Q5. 什么是 mask-grounded understanding，为什么同时提供原图、mask、crop 和程序事实

**来源：** 本项目追问。

**口语回答：**

> Mask-grounded understanding 是让模型围绕指定像素区域回答，而不是泛泛描述整幅图。
> 原图提供全局场景，二值 mask 指明位置，clean crop 提供局部细节，程序事实提供确定的几何。
> 四者职责不同：如果只给 crop，会丢上下文并把裁剪边缘当边界；只给 overlay 会污染颜色；
> 只给原图又不清楚问的是哪里。程序事实由代码计算，模型只读，避免它用语言猜面积和质心。

**概念拆解：**

- Grounding 的本质是建立“语言命题—区域—证据”的可核对关系。
- mask 是空间提示，不是语义标签，也不等于目标已被科学确认。
- clean crop 不画彩色边框，因此能保留区域内真实视觉值。

**项目落地：** [grounding/evidence.py](../oa_groundrag/grounding/evidence.py) 确定性计算
bbox、centroid、area、components、compactness、elongation 和 crop window；
[grounding/messages.py](../oa_groundrag/grounding/messages.py) 固定多图顺序和只读合同。

**常见追问：**

- 问：为什么还需要 full image？答：局部纹理必须放回坡面、植被、道路和周边环境中比较。
- 问：模型发现程序 bbox 错了怎么办？答：模型不能静默改写；应让上游合同失败或报告限制，
  保持事实来源清晰。

**不要这样回答：** “mask 告诉模型这里就是滑坡。”mask 只告诉模型“关注这里”。

### Q6. 为什么把视觉观察与专业解释分成两遍

**来源：** 本项目追问；关联小林的
[RAG 幻觉治理](https://xiaolinnote.com/ai/rag/17_hallucination.html)。

**口语回答：**

> 第一遍只允许说当前图像直接支持的观察，例如区域外观、形态、周边环境和可见性限制。
> 第二遍才把问题、只读事实、第一遍观察和检索证据交给模型，输出可能解释、混淆因素和核查
> 建议。这样能检查某句话到底来自图像还是文献，防止模型读到“滑坡常见特征”后反过来说
> “当前图像已经观察到该特征”。两遍不是两个模型，而是同一 Shared MLLM 的两种受约束调用。

**概念拆解：**

- Pass-1 是 observation，不做触发原因、稳定性、风险和确认式诊断。
- Retrieval 根据 observation 构造查询，但不能修改 observation。
- Pass-2 是 text-only knowledge-conditioned generation，不再读取图像重新“发现事实”。

**项目落地：** Grounded 输出合同在
[grounding/outputs.py](../oa_groundrag/grounding/outputs.py)，两条确定性查询与 packet 在
[retrieval/search.py](../oa_groundrag/retrieval/search.py)，Pass-2 禁止式检查在
[retrieval/pass2.py](../oa_groundrag/retrieval/pass2.py)。

**常见追问：**

- 问：为什么不一次 Prompt 全做完？答：一次生成很难可靠区分视觉来源和知识来源，也无法在
  检索前获得结构化区域观察。
- 问：两遍能完全消除幻觉吗？答：不能；它提升可检查性，还需要检索评价、引用校验和专家评价。

**不要这样回答：** “Pass-2 会纠正 Pass-1 的事实。”Pass-2 不允许改写程序事实或把知识写成
新的图像观察。

### Q7. 训练 curriculum 与 runtime architecture 有什么区别

**来源：** 本项目追问。

**口语回答：**

> Curriculum 描述能力怎样学出来，runtime 描述一次请求怎样执行。训练时，Qwen3-VL 先做
> RS-General 遥感适配，再从该 checkpoint warm start 做 Grounded Region 训练，并混入
> RS-General replay。运行时不串联两个 Adapter，只加载最终 Shared Adapter，再按任务决定
> 是否调用分割和 RAG。把两者分开，历史 Stage 可以保留在 checkpoint provenance 中，却不会
> 强迫每次请求重放整个训练历史。

**概念拆解：**

- Curriculum 是“先学通用遥感，再学指定区域观察”的学习顺序。
- Runtime 是“这次问题需要 spatial/shared/rag 中哪些能力”的执行图。
- warm start 是权重初始化关系，不是线上 provider 调用关系。

**项目落地：** curriculum 见
[Grounded train-only 配置](../configs/vlm/grounded/train_v2.yaml)；
最终 Adapter 加载见 [grounded_runtime.py](../oa_groundrag/vlm/grounded_runtime.py)；
runtime 绑定见 [inference_v2.yaml](../configs/runtime/inference_v2.yaml)。

**常见追问：**

- 问：replay 的作用是什么？答：Grounded 训练中保留一部分通用遥感样本，降低只学区域格式
  后遗忘通用能力的风险；它不是 retention 已经科学通过的证明。
- 问：为什么配置和输出还含 Stage 名？答：它们是训练与产物 provenance；稳定源码结构已经按
  segmentation/vlm/grounding/retrieval/runtime 组织。

**不要这样回答：** “线上先跑 RS-General Adapter，再跑 Stage 5 Adapter。”当前 loader
加载的是最终 trainable LoRA state。

<a id="interview-vlm"></a>

## 11. LLM、VLM 与适配训练（Q8–Q15）

### Q8. LLM 和 VLM 有什么区别，Qwen3-VL 在本项目中如何消费图像和文本

**来源：** 小林
[什么是大语言模型](https://xiaolinnote.com/ai/llm/what_is_llm.html)与
[Transformer 架构](https://xiaolinnote.com/ai/llm/transformer_architecture.html)的项目化追问。

**口语回答：**

> LLM 的统一接口是把 token 序列自回归地生成后续 token；VLM 在这个接口前增加视觉编码与
> 跨模态对齐，让图像也能变成语言模型可消费的表示。本项目通过 Qwen3-VL processor 按消息
> 顺序处理多幅图和文本，得到 text token、pixel values 和 image grid 信息，再由模型生成文本。
> 普通任务可以只给原图，grounded 任务按固定顺序给原图、二值 mask 和 clean crop。我们没有
> 重写 Qwen 的视觉 tokenizer，而是封装并验证官方 processor 的输入合同。

**概念拆解：**

- LLM 的核心输出仍是“下一个 token 的概率”。
- VLM 的额外难点是把二维图像变成视觉表示，并让文本 token 能引用这些表示。
- “多模态”不等于把数组随便拼在一起；图像数量、顺序、尺寸和 grid 都有处理合同。

**项目落地：** [vlm/processing.py](../oa_groundrag/vlm/processing.py) 统计消息图像、
调用 Qwen vision processor、检查 <code>pixel_values/image_grid_thw</code>，并区分训练与推理；
[grounding/messages.py](../oa_groundrag/grounding/messages.py) 固定 grounded 多图语义。

**常见追问：**

- 问：mask 是一个特殊 token 吗？答：当前不是。它作为严格二值图像输入，同时有文字说明，
  不通过 <code>[SEG]</code> token 驱动分割。
- 问：为什么图像顺序重要？答：每幅图承担不同角色，顺序和角色文字共同告诉模型哪个是全图、
  哪个是位置提示、哪个是局部细节。

**不要这样回答：** “VLM 就是把图片转成文字后再交给 LLM。”视觉表示和文本在模型上下文中
联合处理，不能简化成一个未经说明的外部 caption 步骤。

### Q9. Transformer Attention、Q/K/V 是什么，为什么 LoRA 落在 q/k/v/o 投影

**来源：** 小林
[Transformer 架构](https://xiaolinnote.com/ai/llm/transformer_architecture.html)。

**口语回答：**

> Attention 可以理解为当前表示用 Q 去匹配上下文的 K，再按匹配权重汇总 V；多头机制让不同
> 子空间关注不同关系。q_proj、k_proj、v_proj 负责生成 Q/K/V，o_proj 把多头结果映射回模型
> 隐藏空间。本项目在这些语言 Attention 投影上加 LoRA，因为它们直接影响信息如何选择和组合，
> 用少量参数就能适配遥感描述与区域关注；视觉 encoder 和 merger 则保持冻结。

**概念拆解：**

- Q 不是“问题文本”，而是每个 token 当前用来查询上下文的向量。
- K/V 也来自隐藏表示的线性投影。
- o_proj 是 Attention 汇总后的输出投影，不是额外的分类头。
- LoRA 改变的是这些投影的增量，不改变 Attention 公式本身。

**项目落地：** 目标模块在
[RS-General 配置](../configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml) 中明确为
<code>q_proj/k_proj/v_proj/o_proj</code>；
[vlm/model.py](../oa_groundrag/vlm/model.py) 创建 PEFT <code>LoraConfig</code> 并检查
vision/merger 没有可训练参数。

**常见追问：**

- 问：为什么除以 <code>√d_k</code>？答：维度增加时点积方差会变大，缩放能避免 softmax
  过早饱和。
- 问：是否比较过只调 Q/V？答：当前正式选择是 q/k/v/o；没有完整消融就不能声称它必然优于
  其他 target 组合。

**不要这样回答：** “LoRA 训练了 Attention。”更准确地说，它训练 Attention 线性投影上的
低秩增量，基座投影保持冻结。

### Q10. Tokenizer 在 VLM 输入和 RAG 文本切分中分别起什么作用

**来源：** 小林
[什么是 Tokenizer](https://xiaolinnote.com/ai/llm/tokenizer.html)。

**口语回答：**

> Tokenizer 是文字到整数 token ID 的桥梁，但本项目有两套不同用途的 tokenizer。Qwen
> tokenizer 与 chat template 一起编码用户、图像占位和 assistant 文本，训练时还用于构造只对
> assistant 回答计算损失的 labels；BGE-M3 tokenizer 用来检查 Evidence Unit 是否超过 dense
> 模型长度，并编码文档和查询。两者词表和 token 数可能不同，不能用字符数或其中一套的计数
> 替代另一套。

**概念拆解：**

- 模型不能直接计算字符串，需要 token ID 和 embedding。
- 子词 tokenizer 在“序列不要太长”和“能表达新词”之间折中。
- chat template 还编码消息角色；错误模板会让模型分不清 user 与 assistant。
- 文档 chunk 的字符长度和 token 长度是两个不同约束。

**项目落地：** Qwen 编码和 assistant-only label 在
[vlm/processing.py](../oa_groundrag/vlm/processing.py)；BGE-M3 的
<code>token_count</code>、截断与最大长度在
[retrieval/search.py](../oa_groundrag/retrieval/search.py) 和
[retrieval runtime 配置](../configs/retrieval/runtime_v1.yaml)。

**常见追问：**

- 问：为什么不硬编码 assistant 起止 token？答：不同 tokenizer/chat template 的特殊 token
  可能不同，项目从 processor 结果构造 mask，避免依赖猜测的 token ID。
- 问：为什么 chunk 不能只按 1000 个汉字？答：模型限制按 token 计算，而且中英混合、数字和
  专业符号的 token 比例并不固定。

**不要这样回答：** “Tokenizer 就是中文分词。”它处理的是模型词表中的子词/符号和特殊
token，不等于传统意义的中文词语切分。

### Q11. 常见微调方式有哪些，为什么项目选择 SFT LoRA

**来源：** 小林
[大模型微调方案](https://xiaolinnote.com/ai/llm/finetuning.html)。

**口语回答：**

> 这个问题要分两个维度回答：参数怎么更新，以及训练目标是什么。参数更新可以是全量微调、
> LoRA 或量化基座上的 QLoRA；目标可以是监督微调 SFT，也可以是偏好对齐 DPO、RLHF 等。
> 本项目使用 SFT 加 LoRA：用结构化目标教模型完成遥感描述和 grounded observation，只更新
> q/k/v/o 的低秩参数，冻结视觉 encoder/merger。原因是任务有明确监督格式、单卡资源有限，
> 当前也没有需要 DPO/RLHF 的成对偏好数据和验收目标。

**概念拆解：**

- SFT 回答“用什么目标学”：给定输入，最大化参考 assistant token 的概率。
- LoRA 回答“哪些参数学”：只更新低秩 Adapter。
- 两个术语不互斥，所以“LoRA 和 SFT 二选一”是错误问法。
- QLoRA 还会量化基座；当前项目没有采用，不能把 LoRA 自动说成 QLoRA。

**项目落地：** [vlm/processing.py](../oa_groundrag/vlm/processing.py) 构造
assistant-only labels；[training/vlm/trainer.py](../oa_groundrag/training/vlm/trainer.py)
执行监督训练；适配边界由 [vlm/config.py](../oa_groundrag/vlm/config.py) 强校验。

**常见追问：**

- 问：为什么不全量微调？答：当前 24 GB 级单卡、已有通用基座和任务规模下，LoRA 更符合
  资源与 provenance 要求；这不是说全量微调理论上永远更差。
- 问：为什么不用 DPO/GRPO？答：当前目标是有明确 schema 的监督任务，没有构建相应偏好/
  奖励数据和科学协议。

**不要这样回答：** “LoRA 是一种损失函数”或“项目做了 RLHF”。二者都与真实实现不符。

### Q12. LoRA 的数学原理、资源优势和 checkpoint provenance 是什么

**来源：** 小林
[LoRA 技术](https://xiaolinnote.com/ai/llm/lora.html)。

**口语回答：**

> LoRA 假设任务适配所需的权重变化可以用低秩矩阵表示，把原来的 <code>W</code> 冻结，只学习
> <code>BA</code>，前向等价于 <code>W + α/r·BA</code>。这样训练参数和优化器状态都显著减少，
> 同一基座还能保存不同适配器。本项目更看重的另一点是 provenance：checkpoint 只保存
> trainable LoRA state，同时绑定基座、配置、可训练参数名和数量；加载时严格核对，避免把一个
> Adapter 悄悄装到不兼容模型上。

**概念拆解：**

- 若 <code>W</code> 是 <code>d×k</code>，则可令 <code>B</code> 为 <code>d×r</code>、
  <code>A</code> 为 <code>r×k</code>，且 <code>r</code> 远小于 <code>d,k</code>。
- 资源节省主要发生在需要梯度和优化器状态的参数上。
- Adapter 文件较小不代表推理时不需要基座模型。
- 是否把 LoRA 合并进基座是部署选择；当前 loader 保留独立 trainable state 身份。

**项目落地：** 参数与 PEFT 注入见 [vlm/model.py](../oa_groundrag/vlm/model.py)；
严格保存/加载和 trainable key 核对见
[vlm/checkpoint.py](../oa_groundrag/vlm/checkpoint.py)。

**常见追问：**

- 问：rank 越大越好吗？答：表达能力增加的同时参数和过拟合风险也增加；当前 rank-8 是固定
  工程选择，必须靠消融而非直觉判断最优。
- 问：LoRA 是否没有推理开销？答：取决于是否合并和运行实现，不应绝对化；本项目强调可验证
  加载语义，而没有发布跨后端性能结论。

**不要这样回答：** “LoRA checkpoint 本身就是完整模型。”没有匹配的基座和 processor，它
不能独立复现推理。

### Q13. 为什么先 RS-General adaptation，再 Grounded adaptation，并保留 replay

**来源：** 本项目追问；关联小林
[微调方案](https://xiaolinnote.com/ai/llm/finetuning.html)。

**口语回答：**

> RS-General adaptation 先让基座熟悉通用遥感描述、VQA、空间关系等任务；Grounded
> adaptation 再从该 checkpoint 开始，学习原图、mask、crop 和严格区域输出。这样比直接从
> 通用 VLM 跳到窄域区域合同更平滑。Grounded 训练还按配置混入 10% RS-General replay，
> 用来降低通用能力遗忘风险。要注意 replay 是治理手段，不等于 retention 已经自动得到科学证明。

**概念拆解：**

- warm start 复用上一阶段已学习的遥感语义。
- curriculum 是任务由宽到窄的排序，不是线上串行推理。
- replay 在学习新任务时重放旧任务样本，用于缓解 catastrophic forgetting。
- 是否真正保持能力仍需独立 retention evaluation。

**项目落地：** RS-General task families 在
[RS-General 配置](../configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml)；
Grounded 配置明确 warm start、90% region micro 与 10% replay，见
[mask_grounded train-only 配置](../configs/vlm/grounded/train_v2.yaml)。

**常见追问：**

- 问：为什么不用两个 Adapter 在 runtime 动态切换？答：最终 Grounded Adapter 已在同一
  适配路径中承接通用能力并 replay；当前架构选择一个 Shared MLLM，减少身份和显存复杂度。
- 问：10% 是理论最优吗？答：不是。它是当前配置值，没有充分消融不能上升为通用结论。

**不要这样回答：** “replay 保证模型不会遗忘。”它只降低风险，科学结论要靠评价。

### Q14. RAG 与微调分别解决什么问题，为什么本项目同时需要两者

**来源：** 小林
[RAG 与微调的区别](https://xiaolinnote.com/ai/rag/3_rag_vs_finetune.html)。

**口语回答：**

> 微调主要改变模型“怎样完成任务”，例如怎样读取 mask、多图顺序和输出 schema；RAG 在推理
> 时提供“这次回答可以查阅什么外部知识”，例如解释、混淆因素和限制。前者写进少量 LoRA
> 参数，后者保留在可更新、可引用的 Evidence Bank。项目需要两者，是因为只做 RAG 不能自然
> 教会模型遵守 grounded 输出合同，只做微调又难以让每条专业知识有页码和 Evidence ID。

**概念拆解：**

- 微调改变参数，更新通常要重新训练和验证。
- RAG 不改变生成模型参数，知识可独立更新和追溯。
- RAG 不是保证真实的“外挂数据库”；检索可能错，生成也可能越过证据。
- 两者可以组合，但职责必须清楚。

**项目落地：** LoRA 负责 RS-General 与 Grounded capability；Text Bank 保存外部专业知识；
Pass-2 要求每项回答引用当前 packet 的 Evidence ID。

**常见追问：**

- 问：新规范发布后怎么办？答：先通过 source registry、解析、切分、embedding 和 identity
  流程发布新版本 Bank，而不是立刻重训 MLLM。
- 问：能否只靠 Prompt 不微调？答：可以作为 baseline，但当前已有监督数据和严格多图格式；
  是否足够必须由公平评价决定。

**不要这样回答：** “微调负责知识，RAG 负责格式。”在当前项目中恰好主要相反。

### Q15. 为什么选择 Qwen3-VL-2B，而不是更大模型

**来源：** 小林
[模型选型](https://xiaolinnote.com/ai/llm/model_selection.html)。

**口语回答：**

> 选型先看约束而不是只看参数量。本项目需要本地离线多图处理、可训练 LoRA、明确 processor
> 身份、单张约 24 GB GPU 可运行，并且已经有 Qwen3-VL-2B 的冻结基座和训练资产。2B 规模让
> 空间模型、检索模型与 MLLM 可以通过 lazy loading 在单卡编排。这个选择是当前资源和复现
> 路线下的工程决策，不代表我们做过所有大模型的全面横评，也不代表更大模型不会提升能力。

**概念拆解：**

- 模型选型至少考虑能力、显存、延迟、许可证/本地性、processor、微调生态和已有资产。
- 更大参数通常增加容量，也增加训练、加载和迭代成本。
- 科研项目还要考虑模型 revision 是否能固定和重算身份。

**项目落地：** 模型路径、本地加载、bfloat16 和 SDPA 在
[RS-General 配置](../configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml)；
processor 与模型文件身份分别由 [vlm/processing.py](../oa_groundrag/vlm/processing.py) 和
[vlm/model.py](../oa_groundrag/vlm/model.py) 校验。

**常见追问：**

- 问：为什么不用 API 大模型？答：当前路线要求本地权重、训练和严格资产身份；外部 API 的
  版本漂移与数据边界不符合这套复现链。
- 问：有模型选型 benchmark 吗？答：不能虚构。当前能说明的是约束匹配和既有资产路线，
  不是完整 head-to-head 科学比较。

**不要这样回答：** “2B 的效果已经超过所有大模型。”没有相应比较证据。

<a id="interview-rag"></a>

## 12. Evidence-Constrained RAG（Q16–Q26）

### Q16. 什么是 RAG，本项目的离线建库和在线推理流程是什么

**来源：** 小林
[什么是 RAG 及完整流程](https://xiaolinnote.com/ai/rag/1_whatisrag.html)、
[在线工作流程](https://xiaolinnote.com/ai/rag/10_online_workflow.html)。

**口语回答：**

> RAG 是 Retrieval-Augmented Generation：生成前先从外部知识库检索证据，再把证据放进
> 模型上下文。项目离线阶段登记 PDF 来源和身份，解析或 OCR，按文档结构切成 Evidence Unit，
> 标注页码、章节、知识类型和 SHA，再建立 FTS5 与 BGE-M3 dense 索引。在线阶段从用户问题和
> 可选的 Pass-1 观察构造两条确定性查询，分别走关键词和 dense 检索，用 RRF 融合并按三类
> 知识配额组成 packet，最后让 Shared MLLM 做带 Evidence ID 的 text-only Pass-2。纯知识
> 问答没有 Pass-1，区域解释才携带视觉观察与程序事实。

**概念拆解：**

- Retrieval 负责从大集合中找到较相关的小集合。
- Augmentation 是把检索结果连同来源信息放进 Prompt。
- Generation 仍由 MLLM 完成，因此检索正确不等于生成一定正确。
- 离线建库与在线问答必须分开，不能每个问题都重新解析全部 PDF。

**项目落地：** 离线链路在 [retrieval/bank.py](../oa_groundrag/retrieval/bank.py)，检索和
packet 在 [retrieval/search.py](../oa_groundrag/retrieval/search.py)，生成合同在
[retrieval/pass2.py](../oa_groundrag/retrieval/pass2.py)。

**常见追问：**

- 问：RAG 会训练 Qwen 吗？答：不会；当前 RAG 在 inference time 注入文本证据。
- 问：为什么 Pass-2 是 text-only？答：它只解释已冻结的程序事实和 Pass-1 观察，防止再次看图
  后把检索知识混成新观察。

**不要这样回答：** “RAG 就是从数据库查几条结果返回。”如果没有生成、引用合同和离线索引
流程，只描述了搜索系统的一部分。

### Q17. RAG 能解决和不能解决哪些问题

**来源：** 小林
[RAG 主要解决什么问题](https://xiaolinnote.com/ai/rag/2_rag_problems.html)。

**口语回答：**

> RAG 适合解决模型参数没有覆盖、需要更新或需要引用来源的外部知识问题。本项目用它补充地质
> 灾害解释、混淆因素和证据限制，让回答能回到具体证据单元。但 RAG 不能修复错误 mask，不能
> 把不可见内容变成图像事实，也不能保证专业结论正确；来源错误、切分错误、检索漏召回和生成
> 越界仍然存在。因此它是知识增强和溯源机制，不是“消灭幻觉”的保证书。

**概念拆解：**

- 参数知识更新慢、难引用；外部 Bank 可以版本化更新。
- RAG 改善“没有依据可读”的问题，但不能自动判断依据是否适用于当前场景。
- 视觉事实、程序事实和文献知识是三个不同来源，不能互相覆盖。

**项目落地：** Text RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1
observation，也禁止把 candidate 升级为 confirmed landslide。

**常见追问：**

- 问：给足文档是否就不会幻觉？答：不会。模型可能忽略、误读或错误组合文档，所以还要做
  citation/type/forbidden-claim 校验。
- 问：RAG 能替代专家吗？答：当前只提供有来源的候选解释和限制，科学验收仍需 Gold、专家和
  冻结协议。

**不要这样回答：** “RAG 让模型获得真实世界知识，所以回答都是真的。”检索相关性与命题
真实性不是一回事。

### Q18. 文档如何切分，怎样避免破坏语义及来源身份

**来源：** 小林
[Chunking 策略](https://xiaolinnote.com/ai/rag/4_chunking.html)。

**口语回答：**

> Chunk 太大时主题混杂、向量会平均掉细节，太小时语义和适用条件会被切断。本项目不是简单
> 固定字符滑窗，而是先保留 PDF 页身份，再按 parser profile、段落和条款结构生成 Evidence
> Unit；同时用 BGE tokenizer 限制最大 token，过滤过短或坏字符内容。每个 unit 保存 source、
> page、section、clause、content SHA 和 duplicate 关系，所以检索后还能回到原文位置。

**概念拆解：**

- chunk 是检索和引用的最小文本单元。
- 语义完整性不仅取决于长度，还取决于标题、条款和条件是否一起保留。
- OCR 只是文本获取方式，必须记录 extraction method 和质量限制。
- content hash 用于精确去重；相似语义不应被错误当成字节级重复。

**项目落地：** 提取、OCR、unit 生成、去重和 ledger 在
[retrieval/bank.py](../oa_groundrag/retrieval/bank.py)；最小字符数和最大 token 数在
[retrieval runtime 配置](../configs/retrieval/runtime_v1.yaml)。

**常见追问：**

- 问：为什么不统一使用 500 token + 100 overlap？答：那是通用起点，不是本项目已验证的最优
  方案；法规条款、标准和论文段落的结构不同。
- 问：如何知道切分好不好？答：检查边界完整性、来源可回溯性，并在 retrieval Gold 上评价
  Recall@K/MRR/nDCG，而不是只看 chunk 数量。

**不要这样回答：** “chunk 越小检索越准。”过小会丢条件、主语和上下文。

### Q19. Embedding 是什么，为什么采用 BGE-M3，应该怎样评价

**来源：** 小林
[Embedding 的选择与评价](https://xiaolinnote.com/ai/rag/6_embedding.html)。

**口语回答：**

> Embedding 把文本映射为向量，让语义接近的查询和证据在向量空间更接近。本项目固定本地
> BGE-M3 revision，取 dense CLS 表示并做 L2 归一化；检索时归一化向量点积等价于 cosine
> similarity。选择它是当前多语言专业文本、本地运行和固定模型身份下的工程方案。真正评价
> 不能只引用通用榜单，还要在本项目 query–evidence Gold 上比较 Recall@K、MRR、nDCG、延迟
> 和资源；没有 Gold 时不能宣称选型已科学最优。

**概念拆解：**

- 向量维度不是“知识条数”，而是模型学习出的表示坐标。
- L2 归一化后，向量长度不再影响相似度。
- 通用 embedding benchmark 与地质规范、中文条款、InSAR 术语的分布可能不同。
- embedding revision、tokenizer 和最大长度都是模型身份的一部分。

**项目落地：** [retrieval/search.py](../oa_groundrag/retrieval/search.py) 固定 model identity、
1024 维 dense 表示、归一化与点积；模型路径和 revision 在
[runtime_v1.yaml](../configs/retrieval/runtime_v1.yaml)。

**常见追问：**

- 问：为什么不用 Qwen hidden state 做检索？答：当前独立 embedding 模型便于固定、批量离线
  编码和单独评价；没有证据支持临时复用生成模型 hidden 更好。
- 问：维度越大越好吗？答：不一定；表示能力、存储、检索成本和领域效果要共同评价。

**不要这样回答：** “用了 BGE-M3，所以语义检索肯定准确。”模型名不能替代领域 Gold。

### Q20. 关键词检索与向量检索有什么区别

**来源：** 小林
[向量检索与关键词检索](https://xiaolinnote.com/ai/rag/11_retrieval_types.html)。

**口语回答：**

> 关键词检索擅长精确词、规范编号、术语和数字，但换一种表达可能漏掉；dense 检索擅长语义
> 近义，例如“坡体位移迹象”和“形变证据”，但可能召回措辞相似却条件不适用的段落。本项目
> 同时使用 SQLite FTS5 的 BM25 排名和 BGE-M3 dense 相似度，再用 RRF 融合。这样保留专业
> 术语精确命中，也补上自然语言改写。

**概念拆解：**

- BM25 依据词频、逆文档频率和文档长度等信号排序。
- dense retrieval 比较学习到的语义向量。
- 两路原始分数量纲不同，因此项目融合排名而不是直接相加分数。
- metadata 中的 knowledge type 和 modality 还会限制候选集合。

**项目落地：** FTS5 查询使用 <code>-bm25</code> 排序，dense 路径使用归一化矩阵点积；
实现都在 [retrieval/search.py](../oa_groundrag/retrieval/search.py)。

**常见追问：**

- 问：专业文档是不是只用关键词就够？答：规范编号适合关键词，但用户问题常用非标准表述；
  组合通常更稳，最终仍需领域评价。
- 问：dense 是否理解事实？答：它学习相似性，不保证逻辑、数值或适用条件正确。

**不要这样回答：** “向量检索比关键词更先进，所以完全替代 BM25。”两者错误模式互补。

### Q21. 什么是多路召回和 RRF，本项目如何融合检索结果

**来源：** 小林
[多路召回](https://xiaolinnote.com/ai/rag/13_multi_retrieval.html)。

**口语回答：**

> 多路召回是用不同查询或不同检索器得到多个候选列表，再做融合。本项目有两个维度：查询上
> 分“支持性解释”和“混淆/限制”两条；每条又走 lexical 与 dense 两路。RRF 对每个证据按
> <code>1/(k+rank)</code> 累加，不要求 BM25 与 cosine 分数同尺度。融合后还不是最终 packet，
> 还要按知识类型配额选取，并避免同一来源页重复占满。

**概念拆解：**

- 多路召回目标是扩大互补覆盖，不是简单增加返回条数。
- RRF 看排序名次，对不同检索器的分数校准不敏感。
- <code>k</code> 越大，头部名次差异越平缓；当前配置固定为 60。
- 融合排序与业务配额是两个步骤。

**项目落地：** rank 融合与确定性 tie-break 在
[search.py](../oa_groundrag/retrieval/search.py)，召回深度、RRF 和 quota 在
[runtime_v1.yaml](../configs/retrieval/runtime_v1.yaml)。

**常见追问：**

- 问：RRF 是否学习参数？答：当前不是 learned ranker，它是确定性公式。
- 问：为什么还要 learned reranker？答：它可能提升细粒度相关性，但当前尚未引入；应先用 Gold
  证明现有排序瓶颈，再评估收益、延迟和可复现性。

**不要这样回答：** “用了四路召回。”准确说是两类 query，每类内部有 lexical/dense 两路；
候选可以跨查询汇总。

### Q22. Query Rewrite 有什么作用，为什么这里采用确定性双查询而非 LLM 自由改写

**来源：** 小林
[Query Rewrite](https://xiaolinnote.com/ai/rag/12_query_rewrite.html)。

**口语回答：**

> Query Rewrite 用于把口语、省略或含糊问题改成更适合检索的表达。通用 RAG 常让 LLM 扩写
> 多个 query，但本项目更关心审计：区域解释把用户问题与 Pass-1 的固定字段组合成两条确定性
> 查询。第一条取目标外观、形态和区域对比，找 interpretation；第二条取可能混淆对象、周边
> 环境、证据充分性和限制，找 confounder/limitation。纯知识问答沿用同一双查询结构，但这些
> observation 字段为空。相同输入必然得到相同查询，也不会凭空添加一个图上没观察到的专业
> 术语。

**概念拆解：**

- Rewrite 可能改善召回，也可能改变用户意图或注入幻觉。
- 确定性 query builder 的表达能力较窄，但来源可追溯。
- 两条 query 主动让支持性证据与反证/限制都参与，而不是只找“证明滑坡”的材料。

**项目落地：** <code>build_interpretation_query</code> 与
<code>build_counter_limitation_query</code> 在
[retrieval/search.py](../oa_groundrag/retrieval/search.py)。

**常见追问：**

- 问：以后能否加入 LLM rewrite？答：可以作为有冻结 Prompt、保留原 query、记录改写和 Gold
  消融的候选方案；不能静默替换当前合同。
- 问：知识 QA 没有 Pass-1 怎么办？答：它有独立的 text task/query 合同，不应伪造图像字段。

**不要这样回答：** “项目没有 Query Rewrite。”更准确地说，它没有自由生成式 rewrite，
而是使用确定性结构化 query construction。

### Q23. 为什么当前没有使用独立向量数据库、learned reranker 或知识图谱

**来源：** 小林
[向量数据库选型](https://xiaolinnote.com/ai/rag/rag_info.html)、
[图数据库适用场景](https://xiaolinnote.com/ai/rag/16_graph_db.html)的项目化回答。

**口语回答：**

> 技术选型由规模和查询需求决定。当前 Bank 可以用 SQLite FTS5 加本地归一化 NumPy matrix
> 做确定性全量相似度搜索，部署简单、身份容易冻结，尚没有证据表明 ANN 向量数据库是瓶颈。
> learned reranker 会增加模型、延迟和新的评价身份；知识图谱适合实体关系和多跳查询，而当前
> 主要需求是从专业段落检索解释、混淆和限制。先证明真实瓶颈，再引入对应复杂度，比为“架构
> 完整”堆组件更合理。

**概念拆解：**

- 向量数据库主要解决大规模向量存储、ANN、过滤、更新和服务化，不是 RAG 的必需定义。
- reranker 对召回候选做更昂贵的 query-document 联合相关性评分。
- 知识图谱显式表达实体与关系，适合多跳和关系约束；文本适用条件不一定天然是图结构。

**项目落地：** [retrieval/bank.py](../oa_groundrag/retrieval/bank.py) 发布 SQLite/FTS 与 dense
矩阵；[retrieval/search.py](../oa_groundrag/retrieval/search.py) 进行精确点积和 RRF。

**常见追问：**

- 问：规模增长后怎么演进？答：先量化索引内存、P95 延迟、更新频率和 Recall，再考虑 FAISS/
  Qdrant/Milvus 等；迁移时要保持 unit ID、过滤与结果等价性。
- 问：永远不用知识图谱吗？答：不是。当前没有多跳实体关系的明确消费者和评价，未来出现该
  问题再设计，不提前承诺。

**不要这样回答：** “NumPy 一定比向量数据库快”或“知识图谱没有用。”这里只是当前约束下
没有引入的收益证据。

### Q24. 为什么 Evidence Packet 要平衡 interpretation、confounder 和 limitation

**来源：** 本项目追问；关联小林
[RAG 检索优化题目目录](https://xiaolinnote.com/ai/rag/rag_info.html)。

**口语回答：**

> 如果只按相关性取 Top-K，问题本身带有“滑坡”词时，结果很容易全是支持性材料，模型就会
> 形成确认偏差。本项目把知识分为 interpretation、confounder、limitation，并按配置分别取固定
> 配额；同时不让同一来源同一页重复占多个位置。这样 packet 不只告诉模型“可能为什么”，也
> 强制提供“还可能是什么”和“当前不能判断什么”。

**概念拆解：**

- interpretation：某类观察可能对应的专业解释。
- confounder：外观相似但机制不同的混淆对象。
- limitation：证据适用条件、缺失模态和需要补充的核查。
- 平衡是安全与科学保守性的结构先验，不代表三类同等相关。

**项目落地：** 当前配置为三类各 2 个候选上限；packet 构造检查 unit 去重和 source-page
去重，见 [retrieval/search.py](../oa_groundrag/retrieval/search.py)。

**常见追问：**

- 问：配额不满怎么办？答：packet 如实保留实际选中项，不能用错误类型填充后伪装完整；上层应
  报告证据不足。
- 问：配额 2/2/2 是否最优？答：它是当前固定配置，不是未经 Gold/专家消融即可泛化的结论。

**不要这样回答：** “平衡 packet 保证回答中立。”它降低单边证据风险，不能替代来源质量和
专业审查。

### Q25. 如何防止 RAG 幻觉、伪引用和把知识误写成图像事实

**来源：** 小林
[RAG 幻觉治理](https://xiaolinnote.com/ai/rag/17_hallucination.html)、
[大模型幻觉](https://xiaolinnote.com/ai/llm/hallucination.html)。

**口语回答：**

> 我们把风险分层控制。输入层先把程序事实、Pass-1 观察和 Text Evidence 分开；检索层只允许
> 已发布 Bank unit 进入 packet；生成层用确定性解码、严格 Prompt 和 JSON 约束；输出层逐项
> 检查 Evidence ID 是否存在、知识类型是否匹配，并用禁止式规则拒绝几何改写、确认滑坡、
> 触发原因断言和“文献证明当前图像可见”等泄漏。失败返回明确 reason code，不自动编一份
> 看似正常的答案。

**概念拆解：**

- 检索幻觉：没有找到适用证据或找到错误证据。
- 生成幻觉：证据正确但模型超出、曲解或伪造引用。
- 引用合法性只证明 ID 存在，不自动证明文本蕴含该结论。
- fail closed 让失败可见，但不会神奇提升模型本身能力。

**项目落地：** Evidence ID/type 与禁止式 claim 校验在
[retrieval/pass2.py](../oa_groundrag/retrieval/pass2.py)；provider 错误映射在
[runtime/inference.py](../oa_groundrag/runtime/inference.py)。

**常见追问：**

- 问：正则规则能消除所有幻觉吗？答：不能，它只覆盖预定义高风险模式；仍需 entailment/
  citation correctness 和专家评价。
- 问：为什么不自动修 JSON？答：静默修复会掩盖真实模型输出和改变科学统计；当前选择记录
  INVALID_MODEL_OUTPUT。

**不要这样回答：** “有引用就没有幻觉。”引用可能不支持命题，也可能适用条件不匹配。

### Q26. 如何评价、更新 RAG，当前最困难的科学问题是什么

**来源：** 小林
[RAG 效果量化](https://xiaolinnote.com/ai/rag/18_evaluation.html)、
[知识库更新](https://xiaolinnote.com/ai/rag/19_dynamic_update.html)、
[落地最难之处](https://xiaolinnote.com/ai/rag/20_hardest_parts.html)。

**口语回答：**

> 评价要拆成检索和生成。检索需要 query–relevant-unit Gold，计算 Recall@K、MRR、nDCG，
> 并按知识类型分析；生成需要检查 JSON、引用合法与蕴含、unsupported claim、是否篡改事实，
> 再做专家相关性和有无 RAG 的配对评价。更新时不原位覆盖 Bank，而是登记新来源或 revision，
> 重跑解析、切分、embedding、ledger 和 manifest，发布新身份。当前最难的不是把组件跑通，
> 而是建立无 test leakage 的领域 Gold 与专家科学验收。

**概念拆解：**

- Recall@K：相关证据是否进入前 K。
- MRR：第一个相关证据出现得有多早。
- nDCG：多个分级相关证据的整体排序质量。
- 生成评价还要验证“引用是否真正支持命题”，不能只看字符串 ID。
- 动态更新必须保留旧版本，才能复现旧实验。

**项目落地：** 评价逻辑在 [evaluation/retrieval](../oa_groundrag/evaluation/retrieval/)；
Bank identity、source registry 和发布保护在
[retrieval/bank.py](../oa_groundrag/retrieval/bank.py)。实时哪些指标已有、哪些仍为空，只看
[REBUILD_PROGRESS](../REBUILD_PROGRESS.md)。

**常见追问：**

- 问：没有 Gold 能否看生成答案觉得不错？答：可以做开发观察，不能把主观样例升级为检索
  科学有效性。
- 问：如何避免 test leakage？答：在首次 test 前只用 train/val 构建 Gold、阈值和 Prompt，
  冻结协议后才进行一次正式 test；sealed test 不进入检索 Bank。

**不要这样回答：** “RAG 已经通过自动评价，所以科学有效。”工程合同、自动指标和专家科学
验收是不同层次。

<a id="interview-runtime"></a>

## 13. Runtime 与科研治理（Q27–Q30）

### Q27. Workflow、Agent、Tools 有什么区别，为什么本项目不是 Agent

**来源：** 小林
[Workflow、Agent、Tools 的区别](https://xiaolinnote.com/ai/agent/3_workflow_tools.html)。

**口语回答：**

> Tool 是可被调用的单项能力，例如分割、区域证据构建或检索；Workflow 是开发者预先定义的
> 执行关系；Agent 则让 LLM 根据目标动态决定下一步、选择工具并可能循环反思。OA-GroundRAG
> 当前是确定性 Workflow：任务只有六类，依赖关系和危险边界都已知，显式 Router 比 LLM 自主
> 规划更可复现，也更容易保证不访问 sealed test、不越过 Gate。没有开放式任务分解需求，就
> 不为热点引入 Agent。

**概念拆解：**

- Tool 解决“能做什么”，Workflow 解决“按什么固定逻辑做”，Agent 解决“由模型动态决定怎么做”。
- Router 虽然会“选择能力”，但选择依据是代码中的 task enum，不是 LLM 推理。
- 确定性不表示简单；它仍可有条件分支、错误处理、资源释放和 trace。

**项目落地：** [runtime/router.py](../oa_groundrag/runtime/router.py) 是纯映射且不导入模型；
[runtime/providers.py](../oa_groundrag/runtime/providers.py) 定义能力协议；
[runtime/inference.py](../oa_groundrag/runtime/inference.py) 执行计划。

**常见追问：**

- 问：以后什么情况下需要 Agent？答：当需求变成开放式多步骤研究、工具集合动态变化，并且
  自主规划的收益能被安全与任务完成率评价证明时，再建立新的边界。
- 问：不用 LangChain 是不是重复造轮子？答：当前编排很小且合同严格，直接代码更透明；若框架
  能解决真实的状态图、持久化或可观测性问题，再按消费者决定。

**不要这样回答：** “Router 就是一个 Agent。”是否由 LLM 自主选择和循环，是二者的关键区别。

### Q28. 六类任务如何确定性路由，为什么采用 lazy provider 和显式释放

**来源：** 本项目追问；关联小林
[大模型部署题目目录](https://xiaolinnote.com/ai/llm/llm_info.html)。

**口语回答：**

> 请求必须携带 <code>UnifiedTask</code>，Router 先验证输入合同，再生成四个能力开关和
> region source。VLM_ONLY 只用 Shared MLLM，SEGMENT_ONLY 只用 spatial，用户 mask 的区域
> 理解跳过分割，知识问答跳过图像，区域解释才组合 Grounding 和 RAG。由于 OA-AuxSeg、
> Qwen3-VL 和 BGE-M3 不需要同时常驻，runtime 按配置在阶段间 release、gc 和清 CUDA cache，
> 并记录 trace，适配约 24 GB 单卡。

**概念拆解：**

- lazy provider 表示能力第一次需要时才加载其重资源。
- release 不只是删除一个 Python 变量，还要让 provider 清理模型引用并处理 CUDA cache。
- 确定性 routing 便于为每个任务断言 provider 顺序和禁止输入。
- 资源释放是工程行为，不改变模型数学或输出合同。

**项目落地：** 六类映射见 [runtime/router.py](../oa_groundrag/runtime/router.py)，请求验证和
reason code 见 [runtime/contracts.py](../oa_groundrag/runtime/contracts.py)，释放开关见
[runtime 配置](../configs/runtime/inference_v2.yaml)。

**常见追问：**

- 问：候选 ID 找不到会怎样？答：只有 OA-AuxSeg candidate 选择合同允许记录原因后退到 global
  mask；它会进入 <code>RegionSelection</code> 和 limitation，不是静默行为。
- 问：provider 失败会不会换另一个模型？答：不会。身份、模型输出或 provider 失败返回稳定
  reason code，避免改变实验条件。

**不要这样回答：** “所有失败都会 fail closed、绝不 fallback。”候选选择有一个明确、可审计
的 global-mask fallback；应准确区分它与 provider/身份/输出失败。

### Q29. 解码、Prompt、严格 JSON 和拒绝静默修复如何提高复现性

**来源：** 小林
[解码策略](https://xiaolinnote.com/ai/llm/decoding_strategies.html)、
[Prompt 工程](https://xiaolinnote.com/ai/llm/prompt_engineering.html)。

**口语回答：**

> 科研推理追求相同输入和身份得到可比较输出，所以项目主线固定 <code>do_sample=false</code>、
> temperature 0，而不是用 Top-P/Top-K 增加创意。Prompt 明确图片角色、只读事实、允许字段、
> Evidence ID 和禁止结论；生成还配合严格 JSON/FSM 与 parser。若出现重复 key、非有限数、
> 未知证据、错误枚举或禁止 claim，就记录 INVALID_MODEL_OUTPUT，不静默补字段。这样有效率
> 可能不漂亮，但失败率是真实可复现的。

**概念拆解：**

- greedy/deterministic decoding 每步选择确定输出；sampling 从概率分布抽样。
- Prompt 是软约束，parser/schema 是代码层硬约束，两者不能互相替代。
- JSON FSM 可以限制可生成结构，最终仍要做语义校验。
- 自动修复改变原始模型输出，会污染模型能力统计和错误归因。

**项目落地：** 确定性 generation 在
[RS-General 配置](../configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml)；
grounded Prompt 在 [grounding/messages.py](../oa_groundrag/grounding/messages.py)；
严格解析在 [vlm/outputs.py](../oa_groundrag/vlm/outputs.py) 和
[retrieval/pass2.py](../oa_groundrag/retrieval/pass2.py)。

**常见追问：**

- 问：<code>do_sample=false</code> 是否绝对保证跨 GPU 位级一致？答：不能；底层 kernel、
  dtype、库版本也会影响数值，所以还要冻结环境和身份。
- 问：为什么不让模型输出自然语言再由另一个 LLM 修复？答：那会增加不可审计的第二个生成器，
  并把首个模型失败藏起来；可作为产品层策略，但不能混入当前科学统计。

**不要这样回答：** “Prompt 能保证模型不犯错。”Prompt 只能引导，硬合同和评价仍然必要。

### Q30. 如何设计模型与系统评价，并避免 Gate 越权和 test leakage

**来源：** 小林
[大模型能力评测](https://xiaolinnote.com/ai/llm/evaluation_metrics.html)的项目化回答。

**口语回答：**

> 评价必须跟能力边界对齐，不能只看一个总分。空间模块评价 mask、no-target 和候选区域；
> Shared MLLM 评价通用遥感能力与 grounded observation；RAG 分开评价 retrieval 和
> evidence-constrained generation；runtime 评价六任务路由、失败码、provider 顺序和资源释放。
> 科学 Gate 要在首次 test 前用 train/val 预注册指标、阈值和协议，再冻结实现与资产。test/
> sealed test 不能进入 Corpus、Prompt 调参、阈值或当前 Bank；工程 smoke 也不能被写成科学 PASS。

**概念拆解：**

- unit test 验证代码合同，benchmark 验证任务表现，Gate 验证预注册科学主张。
- development set 用于选择，test 用于最终无偏评价；反复查看 test 会让它变成开发集。
- artifact identity 保证“评价的是哪套权重、数据、配置和代码”可重算。
- 分层评价便于判断失败来自分割、观察、检索、生成还是编排。

**项目落地：** 能力评价分别位于
[evaluation/segmentation.py](../oa_groundrag/evaluation/segmentation.py)、
[evaluation/rs_general](../oa_groundrag/evaluation/rs_general/)、
[evaluation/grounding](../oa_groundrag/evaluation/grounding/) 和
[evaluation/retrieval](../oa_groundrag/evaluation/retrieval/)；runtime 还会在读取前词法拒绝
test/sealed 路径。实时 Gate 结论只看 [REBUILD_PROGRESS](../REBUILD_PROGRESS.md)。

**常见追问：**

- 问：模型能运行且自动指标通过，为什么不能叫科学通过？答：可能缺预注册阈值、独立 Gold、
  专家评价或首次 test；工程成功与科学主张的证据等级不同。
- 问：test 完成后发现问题能否改 Prompt 再测？答：不能把同一个 test 继续当无偏正式集；
  应修复后发布新协议，并使用未泄漏的新冻结测试边界。

**不要这样回答：** “我们有很多 unit tests，所以算法有效。”测试数量证明不了科学效果。

## 14. 相关但不能包装成项目经验的题

下面这些是大模型面试常见基础。它们与底层模型或未来部署相关，但当前仓库没有把它们作为
自研模块或已完成实验。面试时可以解释概念和选型边界，不能说“项目已经做过”。

| 题目 | 应理解到什么程度 | 与当前项目的准确关系 |
| --- | --- | --- |
| RoPE / 位置编码 | 为什么 token 顺序需要位置信息，RoPE 如何编码相对位置 | Qwen3-VL 基座内部能力；本项目没有修改或消融 |
| MHA、GQA、FlashAttention | 计算/显存瓶颈与共享 KV、IO-aware kernel 的基本思想 | 配置使用 PyTorch SDPA；没有自研 Attention kernel 或性能横评 |
| KV Cache / Prompt Caching | 自回归时复用历史 K/V 与跨请求前缀缓存的区别 | 模型库可能在单次 generate 内使用 cache；项目没有发布跨请求缓存系统 |
| INT8/INT4、AWQ/GPTQ | 权重量化的精度、校准、显存和吞吐取舍 | 当前 bfloat16 路线未量化；不能声称做过量化无损实验 |
| DPO / PPO / GRPO / RLHF | SFT 后的偏好或奖励优化目标与数据要求 | 当前只做 SFT LoRA，没有偏好优化或强化学习 |
| MCP / Function Calling / Skills | 模型决定调用、工具协议与操作知识封装的层次 | 当前 provider 是项目内部 Python 协议，不是 MCP server，也不是模型 Function Call |
| LangChain / LangGraph | 通用链式组件和状态图编排解决的问题 | 当前固定 Workflow 直接实现，没有使用这些框架 |
| ReAct / Reflection / Multi-Agent | 自主规划、工具循环、反思与多角色协作 | 明确不在当前主线；不能把确定性 Router 称作 Agent |
| Knowledge Graph | 实体、关系、多跳查询的适用问题 | 当前 Text Bank 是可追溯文本检索，没有知识图谱 |

这些专题可从小林的
[大模型工程目录](https://xiaolinnote.com/ai/llm/llm_info.html)、
[Agent 目录](https://xiaolinnote.com/ai/agent/agent_info.html)和
[工具调用目录](https://xiaolinnote.com/ai/)继续学习。回答时建议采用：

> 我理解该技术解决的问题是……；当前项目由于……没有采用；如果出现……瓶颈，我会先用……
> 指标验证，再决定是否引入。

这比为了显得技术栈多而虚构实践更可靠。

<a id="review-cards"></a>

## 15. 面试前的十句复习卡

1. OA-GroundRAG 的主问题是 Where、What、How，不是让一个模型包办全部任务。
2. OA-AuxSeg 是独立空间专家；mask 是关注区域，不是确认滑坡标签。
3. Grounding 用原图、二值 mask、clean crop 和只读程序事实连接像素与语言。
4. Shared MLLM 是 Qwen3-VL-2B 加最终 Grounded LoRA，不是 runtime 串联两套 Adapter。
5. 项目使用 SFT LoRA：SFT 是训练目标，LoRA 是参数更新方式。
6. Pass-1 只说图像观察，Pass-2 才结合文本知识谨慎解释。
7. Retrieval 是双查询、FTS5 + BGE-M3、RRF 和三类平衡 Evidence Packet。
8. Router 是显式 task 驱动的 Workflow，不是 Agent，也不让 LLM 猜任务。
9. 严格失败、证据 ID 和 artifact identity 是科研复现的一部分，不是多余的工程负担。
10. 工程完成、自动开发评价和科学 Gate 是三个层次；任何时候都不要越级表述。

## 16. 题目来源与项目证据入口

### 面试题来源

- [小林面试笔记：AI 面试题总览](https://xiaolinnote.com/ai/)
- [RAG 面试专题](https://xiaolinnote.com/ai/rag/rag_info.html)
- [大模型工程面试专题](https://xiaolinnote.com/ai/llm/llm_info.html)
- [Agent 面试专题](https://xiaolinnote.com/ai/agent/agent_info.html)

### 项目证据

- 冻结设计：[OA-GroundRAG 算法构建方案](OA-GroundRAG_算法构建方案_0811.md)
- 实时状态：[REBUILD_PROGRESS](../REBUILD_PROGRESS.md)
- 稳定入口：[README](../README.md)
- 能力代码：[segmentation](../oa_groundrag/segmentation/)、
  [vlm](../oa_groundrag/vlm/)、[grounding](../oa_groundrag/grounding/)、
  [retrieval](../oa_groundrag/retrieval/)、[runtime](../oa_groundrag/runtime/)
- 训练与评价：[training](../oa_groundrag/training/)、
  [evaluation](../oa_groundrag/evaluation/)

最后再提醒一次：小林题库帮助我们选择高频问题；本项目为什么这样回答，必须由当前冻结设计、
代码、配置和科学边界共同决定。
