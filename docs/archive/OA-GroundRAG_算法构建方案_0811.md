# OA-GroundRAG：Instruction-Routed Grounded Geohazard MLLM

> **路线全称：** Optical-Anchored Grounded and Knowledge-Augmented Multimodal Model for Geohazard Remote Sensing  
> **中文名称：** 光学锚定、区域接地与知识增强的地质灾害遥感多模态模型  
> **版本：** 3.0  
> **设计日期：** 2026-08-11  
> **基础模型：** Qwen3-VL-2B  
> **硬件边界：** 单张约 24 GB GPU  
> **文档定位：** 本文定义 OA-GroundRAG 后续模型架构、能力边界、训练逻辑和推理组织方式；实时训练进度、checkpoint、artifact identity、Gate 与 sealed-test 状态统一以 `REBUILD_PROGRESS.md` 为准，不在本文硬编码。

---

## 0. 当前仓库基线与本次重构原则

当前仓库已经具备四类相对独立且可复用的核心资产：

1. `oa_groundrag/phase2` 中的 OA-AuxSeg，多源滑坡像素定位模型；
2. `oa_groundrag/phase4` 中以 Qwen3-VL-2B 为基础的 RS-General / Mask-Grounded 视觉语言能力；
3. Mask-Grounded Evidence Builder 和 full image + binary mask + context crop 区域输入机制；
4. `oa_groundrag/text_rag` 中的 Evidence-Constrained Text RAG。

实时进度文件显示：OA-AuxSeg 已完成工程定版权重但尚未通过 Gate A；RS-General Adapter 已完成训练和 Gate B；Stage 5 Mask-Grounded Region Adapter 已完成训练和 automatic-only dev 评价；Stage 6 Text RAG 已完成工程主链和 25-pair automatic-only 开发评价，但 Gate C/D、专家科学评价和 sealed test 均未完成。

当前 GitHub 最新可见提交为 `261cd212...`，主要同步 `AGENTS.md` 和 `REBUILD_PROGRESS.md` 的活动路线，没有引入新的模型训练。该提交发生后，`REBUILD_PROGRESS.md` 内记录的 `current_head=d6478025...` 已比真实 Git HEAD 落后一提交，这属于文档身份元数据漂移，不改变 Stage 5/6 的科学状态。

本次重构不推翻上述有效资产，而只改变**最终模型的组织范式**：

旧范式：

```text
Segmentation
→ Region Description
→ RAG
```

新范式：

```text
Multi-source Remote Sensing Inputs
+ User Instruction
+ Optional Mask / Region
        ↓
Instruction / Capability Routing
        ↓
Shared RS-Geohazard MLLM
        ├── Direct Visual Understanding
        ├── Optional OA-AuxSeg Spatial Capability
        ├── Optional Grounded Region Understanding
        └── Optional Evidence-Constrained Text RAG
        ↓
Unified Response
```

核心原则是：

> **训练过程可以分阶段，运行时不应被训练阶段顺序绑架。**

因此：

```text
training stages ≠ runtime stages
```

RS-General → Mask-Grounded 是能力迁移 curriculum，不再被解释成推理时必须依次经过两个 Adapter。

---

# 1. 研究定位

OA-GroundRAG 的最终目标不是构建一个所有请求都经过“分割—区域描述—RAG”的固定流水线，也不是把 OA-AuxSeg、Qwen3-VL 和检索模块强制联合训练成单体网络。

本研究将 OA-GroundRAG 定义为：

> **Instruction-Routed Grounded Geohazard MLLM：以共享遥感地质灾害 MLLM 为语义主体，根据用户 instruction 和是否存在空间提示，选择性调用专业滑坡定位、mask-grounded 区域理解和专业知识检索能力。**

其中：

- Qwen3-VL-2B + Grounded Adapter 是共享视觉—语言语义主体；
- OA-AuxSeg 是按需调用的专业像素空间能力；
- Grounded Multimodal Evidence Interface 是空间模型与 MLLM 之间的标准区域接口；
- Text RAG 是按知识需求调用的外部专业记忆；
- segmentation、region understanding 和 RAG 不要求每次请求全部运行。

这一思路借鉴 M3D-LaMed 的 generalist MLLM 组织方式。M3D 使用共享 3D image encoder、spatial pooling perceiver 和 LLM 支持 report generation、VQA、positioning 和 segmentation；只有 segmentation instruction 需要时，才使用 `[SEG]` hidden embedding 驱动独立 SegVol，而不是将所有任务依次串联。

OA-GroundRAG 借鉴的是这一**任务组织思想**，而不是复制其 3D encoder、SegVol 或 `[SEG]` token 实现。

---

# 2. 核心科学问题

本研究围绕三个互补问题展开。

## 2.1 Where：多源遥感条件下候选滑坡在哪里？

输入可能包括：

```text
Optical / RGB
Sentinel-2 / Multispectral
SAR
InSAR
DEM / Terrain
```

光学影像提供空间边界基准，其余模态作为任意可用辅助证据。

需要解决：

- 不同辅助模态可以存在或缺失；
- 模态具有不同空间覆盖和数据质量；
- 光学边界与辅助模态信息如何协同；
- 输出必须保持可靠像素级 mask 和 no-target 能力。

这一问题由 **OA-AuxSeg Spatial Capability** 负责。

---

## 2.2 What：指定区域实际上呈现什么视觉证据？

区域可以来自：

```text
user mask
GT mask
OA-AuxSeg predicted mask
OA-AuxSeg candidate region
```

模型应回答：

- 区域外观如何；
- 区域形态如何；
- 周围是什么环境；
- 区域与背景有什么差异；
- 当前可见哪些潜在混淆对象；
- 当前视觉证据是否充分。

这一问题由：

```text
Grounded Multimodal Evidence Interface
+
Shared RS-Geohazard MLLM
```

负责。

它不承担滑坡最终专业诊断。

---

## 2.3 How to interpret：这些观察在专业上可能意味着什么？

当任务需要专业解释时，需要进一步回答：

- 当前视觉特征在滑坡判识中可能意味着什么；
- 哪些对象可能产生类似外观；
- 当前传感器和观测条件有什么限制；
- 当前证据不能支持哪些结论；
- 应补充哪些数据或核查手段。

这一问题由：

```text
Evidence-Constrained Text RAG
+
Shared RS-Geohazard MLLM
```

负责。

RAG 永远不能：

- 修改 mask；
- 改写 Programmatic Facts；
- 改写已经获得的视觉观察；
- 把知识库内容描述成“当前影像已经观察到的事实”；
- 因为检索到了滑坡知识而把候选区域升级为确认滑坡。

---

# 3. 最终模型能力集合

最终模型至少支持以下八类用户能力。

| Capability | 示例 | Spatial Expert | Region Interface | RAG |
|---|---|---:|---:|---:|
| Global / Scene Description | “描述这幅遥感影像。” | OFF | OFF | OFF |
| Remote-Sensing VQA | “该区域主要是什么土地覆盖？” | OFF | 可选 | 通常 OFF |
| Landslide Segmentation | “分割图中的候选滑坡区域。” | ON | OFF | OFF |
| User-Mask Region Understanding | “描述这个 mask 区域。” | OFF | ON | OFF |
| Segment-and-Understand | “定位候选滑坡并描述其视觉特征。” | ON | ON | OFF |
| Candidate-Region Interpretation | “该候选区域可能是什么？” | 按 region 来源 | ON | ON |
| Professional QA | “InSAR 为什么不能直接确定三维运动？” | OFF | OFF/可选 | ON |
| Evidence-Constrained Report | “结合当前区域和专业资料给出解释报告。” | 按需 | ON | ON |

这些能力共享同一个语义主体，但不共享相同的执行路径。

---

# 4. 统一模型架构

## 4.1 总体架构

```text
          Multi-Source Remote Sensing Inputs
     Optical / S2 / SAR / InSAR / DEM / ...
                         +
                  User Instruction
                         +
             Optional Mask / Region
                         │
                         ▼
          Instruction / Capability Router
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
        ▼                ▼                 ▼
 Direct Visual      Spatial Capability   Knowledge Capability
 Capability               │                 │
        │              OA-AuxSeg         Text Evidence Bank
        │                 │                 │
        │           mask / regions           │
        │                 │                 │
        └──────────┬──────┘                 │
                   ▼                        │
      Grounded Multimodal Evidence          │
              Interface                    │
                   │                        │
                   ▼                        │
          Shared RS-Geohazard MLLM          │
     Qwen3-VL-2B + Grounded Adapter         │
                   │                        │
                   ├───────────────┬────────┘
                   │               │
             visual response   evidence-conditioned
                                  generation
                                   │
                                   ▼
                         Unified Response
                   text / mask / citations /
                   limitations / region result
```

---

## 4.2 Instruction / Capability Router

Router 的职责只有一个：

> 根据显式任务定义决定本次请求需要调用哪些已有能力。

首版不训练 Router。

推荐固定六个核心 execution task：

```text
VLM_ONLY
SEGMENT_ONLY
REGION_UNDERSTANDING
SEGMENT_AND_UNDERSTAND
KNOWLEDGE_QA
REGION_INTERPRETATION
```

Router 输出的是执行计划，例如：

```text
needs_spatial
needs_shared_mllm
needs_region
needs_rag
region_source
response_type
```

Router 不负责：

- 图像识别；
- mask 预测；
- RAG 检索；
- 自由语言生成；
- 专业判断。

因此 Router 是**系统组织机制而不是论文算法创新点**。

首版优先使用显式 task enum，不让 LLM 自行猜测任务类型。

---

## 4.3 Shared RS-Geohazard MLLM

共享语义主体继续使用：

```text
Qwen3-VL-2B
+
LoRA Adapter
```

当前最合理的候选权重是 Stage 5 Mask-Grounded Region Adapter best checkpoint。

原因是当前 Stage 5 本身已经采用：

```text
RS-General Adapter
        ↓ warm start
Mask-Grounded Region training
+ RS-General replay
        ↓
Mask-Grounded Region Adapter
```

因此新的解释应当是：

```text
RS-General Adaptation
       ↓
Grounded Region Adaptation
       ↓
Shared RS-Geohazard Adapter
```

而不是：

```text
runtime:
RS-General Adapter
→ Region Adapter
```

RS-General Adapter 在新架构中主要保留为：

- training warm-start；
- general RS baseline；
- retention reference。

当前进度记录显示 Stage 5 best 为 step 900，但其 340 条 OA-GroundedEval-dev 中仍有 64 条 no-target strict-output failure，因此该权重目前只能称：

> **Shared RS-Geohazard MLLM candidate**

不能称为最终科学验收模型。

当前 `Qwen3VLModelAdapter` 已采用冻结 vision/merger、只训练 attention LoRA 的轻量方案，对 Qwen3-VL-2B 仅约 3.21M 可训练 LoRA 参数，非常适合单卡约 24 GB 的继续开发。

---

## 4.4 Direct Visual Capability

以下任务应直接进入 Shared MLLM：

```text
scene description
global caption
ordinary remote-sensing VQA
image-level visual reasoning
```

计算路径：

```text
Image(s)
+ Instruction
↓
Shared RS-Geohazard MLLM
↓
Text Response
```

这些任务：

```text
OA-AuxSeg OFF
Region Interface OFF
RAG OFF
```

这一步是区别于旧固定流水线的重要变化。

---

## 4.5 OA-AuxSeg Spatial Capability

OA-AuxSeg 保持独立专业模型。

当前实现基于 ConvNeXt-Small、多阶段辅助模态融合和 SegFormer-style decoder，可输出：

```text
mask_logits
mask_probability
no_target_score
modality_weights
candidate_regions
region_features
```

它仅在需要空间定位时调用：

```text
SEGMENT_ONLY
SEGMENT_AND_UNDERSTAND
需要自动获得候选区域的 REGION_INTERPRETATION
```

当前已经存在：

```text
oa_groundrag.phase2.engine.run_inference()
```

以及：

```text
run_oa_auxseg.py infer
```

因此统一框架必须复用其模型和 checkpoint 读取逻辑，不重新实现 segmentation backbone。

首版明确不采用：

```text
[SEG] token
→ Qwen hidden state
→ OA-AuxSeg
```

也不要求 OA-AuxSeg 与 Qwen 联合反向传播。

原因是当前目标语义固定为滑坡，额外使用 LLM hidden embedding 未证明具有必要性，而且会破坏现有已完成空间模型的独立性。

---

## 4.6 Grounded Multimodal Evidence Interface

这是新统一架构中最重要的跨模型接口。

它负责把：

```text
segmentation output
or user-provided region
```

转换为 Shared MLLM 可以稳定理解的 grounded input。

统一 Region Source：

```text
USER_MASK
GT_MASK            # training/evaluation only
OA_AUXSEG_GLOBAL
OA_AUXSEG_CANDIDATE
```

首版输入继续复用当前已经实现的：

```text
Full Optical Image
+
Binary Mask
+
Clean Context Crop
```

当前 `build_mask_grounded_region_messages()` 已将 binary mask 作为独立图像送入 Qwen3-VL，并明确说明白色只代表 spatial prompt、crop 边缘不是目标边界。

Evidence Builder 同时负责程序计算：

- bbox；
- centroid；
- area；
- connected components；
- compactness；
- elongation；
- context crop window；

这些 Programmatic Facts 始终只读，MLLM 不得重新生成或修改。

现有 generic Evidence Builder 还已经支持 auxiliary views，并保存：

```text
alignment
coverage
unit
sign convention
quantitative_claims_allowed
directional_claims_allowed
```

因此下一步无需重新设计新的 SANE/QMEF/MGRR，而应在该接口上扩展真正的多源 region evidence。

---

## 4.7 Grounded Region Capability

当存在 mask 或 region 时：

```text
Image
+ Mask
+ Context
+ optional auxiliary evidence
↓
Shared RS-Geohazard MLLM
↓
Structured Visual Observation
```

输出主要覆盖：

```text
target appearance
target morphology
surrounding environment
region-context contrast
possible confusers
evidence sufficiency
limitations
summary
```

该能力只是：

> **观察当前输入支持的视觉事实。**

它不调用知识库。

因此：

```text
REGION_UNDERSTANDING
→ RAG OFF
```

旧系统中的 Pass-1 在新架构中重新命名为：

> **Grounded Visual Observation Capability**

“Pass-1”只表示一种复杂推理请求中的第一步，不再是整个模型的固定阶段。

---

## 4.8 Evidence-Constrained Text RAG Capability

现有 Stage 6 保持为外部知识能力，不改变其核心边界。

当前已经存在：

```text
Text Evidence Bank
FTS5
BGE-M3 dense retrieval
RRF
Balanced Evidence Packet
```

并已有显式任务路由：

```text
scene_description                  RAG OFF
mask_grounded_region_description   RAG OFF
candidate_interpretation           RAG ON
professional_qa                    RAG ON
evidence_constrained_report        RAG ON
```

新统一架构只需将该规则提升为整个系统的 routing policy。

RAG 有两种主要入口。

### Region-conditioned RAG

```text
Grounded Visual Observation
↓
Evidence-conditioned Query Builder
↓
Text Evidence Bank
↓
Balanced Evidence
↓
Shared MLLM
```

### Pure Knowledge QA

```text
Professional Question
↓
Text RAG
↓
Shared RS-Geohazard MLLM
↓
Professional Answer
```

纯知识问答不得要求先运行 segmentation。

---

## 4.9 Unified Response

不同 task 可以返回不同内容，但应保持统一顶层语义：

```text
task
text
mask                     optional
candidate_regions         optional
region_observation        optional
citations                 optional
limitations
```

对于 `SEGMENT_ONLY`：

```text
text optional
mask required
```

对于普通视觉任务：

```text
text required
mask absent
citations absent
```

对于专业解释：

```text
text required
region_observation available
citations available
```

Unified Response 是工程接口，不作为论文创新点。

---

# 5. 当前仓库资产在新架构中的定位

| 当前资产 | 旧定位 | 新定位 | 处理 |
|---|---|---|---|
| OA-AuxSeg | 第一阶段分割 | Optional Spatial Expert | 保留 |
| RS-General Adapter | 独立 VLM 阶段 | Shared MLLM warm-start / baseline | 保留但退出 runtime 主链 |
| Mask-Grounded Region Adapter | 第二阶段区域模型 | Shared RS-Geohazard MLLM candidate | 提升为共享语义主体 |
| Evidence Builder | 分割到描述的中间工具 | Grounded Multimodal Evidence Interface | 提升架构地位 |
| Text Evidence Bank | 固定最后一级 RAG | Optional Professional Memory | 保留 |
| Pass-1 | 固定第一遍 | Optional Grounded Visual Observation | 改语义 |
| Pass-2 | 固定第二遍 | Optional Knowledge-Conditioned Generation | 改语义 |

---

# 6. 训练策略：Training Curriculum 与 Runtime 解耦

## 6.1 OA-AuxSeg

OA-AuxSeg 保持独立训练：

```text
OA-AuxSeg Benchmark
↓
Optical + Arbitrary Auxiliary Modalities
↓
Spatial Expert
```

不与 Qwen 联合训练。

---

## 6.2 Shared MLLM 能力迁移

Shared MLLM 的训练仍允许分阶段：

```text
Qwen3-VL-2B
↓
RS-General Adaptation
↓
Mask-Grounded Region Adaptation
↓
Shared RS-Geohazard Adapter
```

该 curriculum 的目的分别是：

### RS-General Adaptation

学习：

- 遥感场景语言；
- 地物词汇；
- remote-sensing VQA；
- spatial / count / caption 表达。

### Grounded Region Adaptation

学习：

- mask spatial prompt；
- full-image context；
- region-specific observation；
- visual evidence limitation；
- region/background distinction。

因此：

```text
RS-General Adapter
→ Mask-Grounded Adapter
```

是**参数能力迁移过程**，不是推理时的模型串联。

---

## 6.3 RAG

Text RAG 首版不训练：

```text
retriever + evidence bank
```

不进行：

```text
backprop through retrieval
joint OA-AuxSeg / Qwen / RAG training
```

---

## 6.4 Optional Unified Instruction Adaptation

只有当 P0/P1 表明 Shared Adapter 无法稳定支持不同 capability 时，才考虑小规模 unified instruction adaptation。

训练数据可以混合：

```text
RS-General
+
Mask-Grounded Region
+
Multi-Source Grounded Region
```

但仍只训练 Qwen LoRA。

首版不增加：

```text
[SEG]
[RAG]
[REGION]
```

等 special token。

---

# 7. Instruction Routing

推荐执行路径如下。

## 7.1 Scene Description

```text
Describe this remote-sensing image.
↓
Shared RS-Geohazard MLLM
↓
Text
```

不运行：

```text
OA-AuxSeg
RAG
```

---

## 7.2 Ordinary Remote-Sensing VQA

```text
Image
+ Question
↓
Shared RS-Geohazard MLLM
↓
Answer
```

---

## 7.3 Landslide Segmentation

```text
Multi-source Inputs
↓
OA-AuxSeg
↓
Mask / Candidate Regions / No-target
```

不运行 MLLM 或 RAG，除非用户同时要求语言解释。

---

## 7.4 User-Mask Region Understanding

```text
Image
+ User Mask
↓
Grounded Evidence Interface
↓
Shared RS-Geohazard MLLM
↓
Visual Observation
```

RAG OFF。

---

## 7.5 Segment-and-Understand

```text
Multi-source Inputs
↓
OA-AuxSeg
↓
Mask / Candidate Region
↓
Grounded Evidence Interface
↓
Shared RS-Geohazard MLLM
↓
Visual Observation
```

默认 RAG OFF。

---

## 7.6 Candidate-Region Interpretation

若已有 mask：

```text
Mask / Region
↓
Grounded Evidence Interface
↓
Shared MLLM visual observation
↓
Text RAG
↓
Shared MLLM professional interpretation
```

若没有 mask、用户要求自动发现候选区域：

```text
OA-AuxSeg
↓
Candidate Region
↓
Grounded Evidence Interface
↓
Shared MLLM
↓
Text RAG
↓
Professional Interpretation
```

---

## 7.7 Professional Knowledge QA

```text
Question
↓
Text RAG
↓
Shared RS-Geohazard MLLM
↓
Professional Answer + Citations
```

完全不需要 OA-AuxSeg。

---

## 7.8 Evidence-Constrained Report

```text
Optional Images / Region
+ Question
        ↓
Optional Grounded Observation
        ↓
Text RAG
        ↓
Shared RS-Geohazard MLLM
        ↓
Evidence-Constrained Report
```

---

# 8. 一级核心模块

论文和架构图中只保留三个主要能力块。

## Module A：Optical-Anchored Spatial Perception

对应：

```text
OA-AuxSeg
```

唯一职责：

> 在光学主导和任意辅助模态条件下稳定输出候选滑坡 mask。

---

## Module B：Grounded Multimodal Understanding

对应：

```text
Grounded Multimodal Evidence Interface
+
Shared RS-Geohazard MLLM
```

唯一职责：

> 将 user / GT / predicted mask 转换为受空间约束的区域视觉理解。

---

## Module C：Evidence-Constrained Knowledge Augmentation

对应：

```text
Text Evidence Bank
+
Evidence-conditioned Retrieval
+
Shared MLLM
```

唯一职责：

> 在知识型 instruction 下补充专业解释、混淆对象与证据限制。

Instruction Router 是 orchestration，不计为论文算法模块。

---

# 9. 推荐论文核心创新

论文贡献控制为三条。

## Contribution 1：Optical-Anchored Arbitrary-Auxiliary Landslide Perception

提出光学锚定的任意辅助模态滑坡定位机制，使模型在 SAR、InSAR、DEM、多光谱等辅助信息任意存在或缺失时仍保持统一的像素定位能力。

主要验证：

```text
optical-only
vs auxiliary modalities
vs missing modalities
```

---

## Contribution 2：Grounded Multimodal Evidence Interface

提出专业 segmentation / user region 到通用 MLLM 的统一空间接口：

```text
mask
+
global context
+
region context
+
sensor availability / limitations
↓
shared MLLM
```

其创新重点不是重新设计 GAR，而是：

> **让专业多源遥感 spatial expert 与共享 MLLM 通过可审计 grounded evidence 对接。**

---

## Contribution 3：Instruction-Triggered Evidence-Constrained Knowledge Augmentation

根据任务和真实视觉观察决定是否检索专业知识：

```text
visual observation
→ evidence-conditioned retrieval
→ interpretation + confounder + limitation
```

而不是：

```text
predicted landslide
→ retrieve supporting landslide knowledge
```

核心目标是避免 knowledge-induced rationalization。

---

# 10. 与现有工作的关系

## 10.1 与 M3D-LaMed

M3D-LaMed 的核心价值在于：

```text
shared image encoder
+
shared MLLM
+
instruction
→ multiple tasks
```

并通过 `[SEG]` embedding 条件调用独立 segmentation module。

OA-GroundRAG 借鉴其：

> **共享语义主体 + 条件执行专业能力**

而不同点是：

- 处理的是多源地球观测数据而非 3D CT；
- spatial expert 面向滑坡多源分割；
- 不要求 LLM hidden state 驱动 spatial expert；
- 增加 mask-grounded evidence interface；
- 增加 evidence-constrained professional retrieval。

---

## 10.2 与 PSALM

PSALM 通过 LMM 更新 mask tokens，再通过 mask generator 完成像素分割，是：

```text
language / task prompt
→ LMM-updated mask queries
→ segmentation
```

OA-GroundRAG 不重新构建这一套 mask-query segmentation，而保留独立 OA-AuxSeg。

研究重点从：

```text
如何让 LMM 自身生成 mask
```

转向：

```text
如何让专业 spatial model
与 shared MLLM
在区域和知识层面形成统一能力
```

---

## 10.3 与 Qwen3-VL-Seg

Qwen3-VL-Seg 将 MLLM grounded box 作为结构先验，通过轻量 decoder 转换为精细 mask。

它主要解决：

```text
language grounding
→ box
→ dense mask
```

OA-GroundRAG 的空间问题是：

```text
optical
+ arbitrary auxiliary sensors
→ landslide mask
```

因此当前没有必要重新加入 box-guided decoder。

---

## 10.4 与 GAR

GAR 的核心贡献是 mask prompt 下同时利用局部细节和全局上下文，通过 RoI-aligned feature replay 提升 region-level MLLM understanding。

OA-GroundRAG 与 GAR 的关系是：

```text
GAR:
mask → contextual region understanding

OA-GroundRAG:
professional multi-source segmentation / user mask
→ grounded multimodal evidence
→ region understanding
→ optional geohazard knowledge augmentation
```

因此不能把 mask-grounded region understanding 本身宣称为首次提出。

---

# 11. 计算资源设计

本研究保持单张约 24 GB GPU 可实现。

核心原则：

### 不同时驻留所有大模型

Router 应采用 lazy capability loading：

```text
VLM_ONLY
→ only Shared MLLM

SEGMENT_ONLY
→ only OA-AuxSeg

KNOWLEDGE_QA
→ Shared MLLM + retrieval

SEGMENT_AND_UNDERSTAND
→ spatial inference
→ region construction
→ Shared MLLM
```

必要时空间模型完成后释放 GPU 内存，再加载 MLLM。

### 不联合训练

不进行：

```text
OA-AuxSeg
↕
Qwen3-VL
↕
Retriever
```

联合反向传播。

### 参数高效适配

继续以 LoRA 为主。

当前 Qwen3-VL-2B 视觉塔保持冻结，符合现有计算预算。

---

# 12. 后续开发路线

历史 Stage 名称继续保留用于：

- checkpoint identity；
- artifact provenance；
- Git 审计；
- 旧 CLI；
- 现有实验记录。

但从本版本开始，不再使用 Stage 1→9 作为**最终论文模型结构**。

---

## P0：Unified Inference Core

目标：

> 将现有能力第一次真正组织成 instruction-routed runtime。

实现：

```text
UnifiedRequest
↓
Capability Router
↓
Execution Plan
↓
Existing Providers
↓
UnifiedResponse
```

支持：

```text
VLM_ONLY
SEGMENT_ONLY
REGION_UNDERSTANDING
SEGMENT_AND_UNDERSTAND
KNOWLEDGE_QA
REGION_INTERPRETATION
```

P0 不训练任何模型。

---

## P1：Multi-Source Grounded Evidence

目前 Stage 5 的正式区域输入主要还是：

```text
optical_full
binary_mask
context_crop
```

而 Evidence Builder 已经具备 auxiliary view 合同。

P1 的核心问题因此是：

> 如何让 Shared RS-Geohazard MLLM 在 mask-grounded 模式下真正利用 SAR、InSAR、DEM 和其他辅助观测？

优先使用 Qwen3-VL 原生多图输入：

```text
Optical
Mask
Optical Context
SAR Evidence
InSAR Evidence
DEM Evidence
+
sensor metadata
```

不增加新的视觉 backbone。

---

## P2：Optional Unified Instruction Adaptation

只有 P1 显示 Shared Adapter 在多 capability / multi-source 情况下能力不足时实施。

继续：

```text
Qwen3-VL LoRA only
```

不重新训练 OA-AuxSeg。

---

## P3：Scientific Evaluation and Final Integration

在统一框架和多源 evidence 路径稳定后，再冻结：

- spatial Gate；
- grounded understanding Gate；
- RAG Gate；
- final runtime protocol。

在此前保持 sealed test 封存。

---

# 13. 明确不进入当前主线的设计

以下内容不作为当前算法路线：

```text
SANE
QMEF
PMRD
MGRR
new proposal decoder
new grounding decoder
[SEG] token-driven OA-AuxSeg
learned task router
large Agent system
knowledge graph
Case RAG
joint end-to-end OA-AuxSeg + Qwen + RAG training
```

除非后续实验明确证明现有能力存在不可由简单接口解决的瓶颈，否则不得重新引入。

---

# 14. 最终论文模型定义

最终 OA-GroundRAG 应被描述为：

> **一个以共享 RS-Geohazard MLLM 为语义主体、以 OA-AuxSeg 为可选专业空间能力、以 Grounded Multimodal Evidence Interface 连接像素预测与区域语言理解，并根据 instruction 选择性调用 Evidence-Constrained Text RAG 的地质灾害遥感多模态框架。**

其关键不在于把所有任务塞入同一个网络，而在于：

```text
One semantic core
+
conditional specialized capabilities
+
one grounded spatial interface
+
optional professional memory
```

使一个系统能够根据用户实际问题在：

```text
看整幅图
定位候选区域
理解指定区域
解释专业证据
回答知识问题
```

之间灵活切换。
