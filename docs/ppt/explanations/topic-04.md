# 4. 从一个问题走完整条链路

[返回本节目录](README.md) · [文档总目录](../../README.md)

<a id="guide-example"></a>

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

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
