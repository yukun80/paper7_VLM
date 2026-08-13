# OA-GroundRAG

OA-GroundRAG 面向遥感滑坡场景，将空间感知、基于区域的多模态理解、专业知识增强和统一
推理组织为一条稳定能力链：

```text
Spatial Perception
→ Grounded Multimodal Understanding
→ Knowledge Augmentation
→ Unified Inference
```

当前长期工程结构按算法能力和工程职责命名。Stage 仍用于训练 curriculum、checkpoint
provenance、历史报告和正式产物身份，但不再决定长期源码、配置、脚本或测试的一级目录。

## 文档权威

- [`docs/OA-GroundRAG_算法构建方案_0811.md`](docs/OA-GroundRAG_算法构建方案_0811.md)
  是冻结的详细算法设计；其中的状态文字只是冻结快照。
- [`REBUILD_PROGRESS.md`](REBUILD_PROGRESS.md) 是唯一实时进度文件，记录当前授权、运行
  结果、冻结产物身份、验收证据和下一任务。
- [`AGENTS.md`](AGENTS.md) 规定仓库操作、安全边界和文档治理。
- `docs/archive/` 只保存历史资料，不是活动接口或进度依据。

README 只描述长期稳定的项目结构和入口，不固定 checkpoint、正式产物 SHA 或阶段完成
状态。开始任何写入、训练或正式评价前，必须重新读取 `REBUILD_PROGRESS.md`。

## 能力与代码位置

| 能力 | 稳定源码 | 主要职责 |
|---|---|---|
| Spatial Perception / OA-AuxSeg | `oa_groundrag/segmentation/` | 模型数学、稀疏辅助模态合同、checkpoint、区域提取与只读推理 |
| Shared RS-Geohazard MLLM | `oa_groundrag/vlm/` | Qwen processor/model、LoRA loader、生成、输出与 grounded adapter runtime |
| Grounded Evidence Interface | `oa_groundrag/grounding/` | RegionSelector、EvidenceBuilder、消息与严格输出合同 |
| Knowledge Augmentation | `oa_groundrag/retrieval/` | Text Evidence Bank、混合检索、Evidence Packet 与 Pass-2 |
| Unified Inference | `oa_groundrag/runtime/` | `UnifiedTask`、确定性 router、lazy provider 与执行编排 |

训练、数据生产和评价不进入 runtime 主链：

```text
oa_groundrag/
├── artifacts/                 通用身份、原子写入和路径安全
├── data/
│   ├── oa_auxseg/             OA-AuxSeg Benchmark producer/validator
│   ├── rs_general/            RS-GeneralDesc Benchmark producer/consumer
│   └── grounded/              Corpus、annotation、supervision 与 workflow
├── segmentation/              OA-AuxSeg 稳定模型与推理接口
├── vlm/                       Shared MLLM 稳定模型与推理接口
├── grounding/                 Grounded Evidence Interface
├── retrieval/                 Evidence-Constrained Text RAG
├── runtime/                   Instruction-Routed Unified Inference
├── training/
│   ├── segmentation/
│   ├── vlm/
│   └── grounding/
└── evaluation/
    ├── segmentation.py        OA-AuxSeg 评价
    ├── vlm.py                 Shared MLLM 评价
    ├── rs_general/            Gate B
    ├── grounding/
    └── retrieval/             Gate D
```

## 环境与受保护资产

仓库使用 Python 3.11。推荐环境：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
python --version
```

默认目录关系：

```text
/home/yukun80/codes/
├── datasets/    只读原始训练资产
├── benchmark/   版本化 Benchmark
├── external/    只读第三方参考代码
└── paper7_VLM/  本仓库
```

- `../datasets` 和 `../external` 默认只读。
- `../benchmark` 只有在相应数据生产任务获明确授权时才能写入，并拒绝覆盖已有根。
- `models_zoo/`、`outputs/`、`docs/RAG_knowledge/`、`参考文献/` 和 `docs/archive/`
  必须保留。
- 输出根、schema 字符串和 checkpoint metadata 中的历史 phase/stage 名称属于 scientific
  provenance，不随 package 迁移改写。
- `yukun80/RAG_tmp` 只作外部原型参考，不复制进仓库，也不是运行时依赖。

## 配置

配置按使用者任务组织：

```text
configs/
├── segmentation/              OA-AuxSeg 训练、评价和推理
├── data/rs_general/           RS-GeneralDesc 构建、验证和导出
├── vlm/
│   ├── rs_general/            Shared MLLM / RS-General Adapter
│   └── grounded/              Grounded runtime 与 train-only curriculum
├── grounding/                 Corpus、annotation 和监督数据
│   └── prompts/
├── retrieval/                 Text Bank runtime、构建与评价配方
└── runtime/                   Unified Inference
```

统一推理使用 `oa_groundrag.unified_inference.config.v2`。配置显式绑定 `spatial`、
`semantic_core` 和 `retrieval`；Shared MLLM 不再通过 retrieval 配置间接定位，retrieval
provider 也不负责模型 loader。当前稳定 runtime 配置为：

- `configs/vlm/grounded/runtime_v1.yaml`：只验证已发布 workflow state、best
  pointer、checkpoint manifest、Adapter 与当前 model/processor/LoRA topology；推理不读
  compact training 或 OA-GroundedEval payload。
- `configs/retrieval/runtime_v1.yaml`：只绑定 source registry、Text Evidence Bank、BGE-M3
  和 retrieval 参数；运行时不依赖任何 development prediction/evaluation 输出。
- `configs/vlm/grounded/train_v2.yaml`：新的 train-only curriculum，不含 Eval-dev 或基于
  Eval-dev 的 baseline 生成/评价阶段。

已退役 Eval-dev 和 development outputs 的旧默认配置不再随仓库发布。构建与评价
实现仍按职责保留，但若要开启新的 development evaluation，必须在新授权下显式
提供新配置，不会因 CLI 默认值读取已删除路径。

## CLI

安装后的稳定统一入口：

```bash
oa-groundrag --help
```

从源码 checkout 运行时，用户按动作查找脚本：

```text
scripts/
├── data/       Benchmark、Corpus、annotation、supervision、Text Bank
├── train/      OA-AuxSeg、RS-VLM、Mask-Grounded Adapter
├── infer/      OA-AuxSeg、RS-VLM、Text RAG、OA-GroundRAG
└── evaluate/   OA-AuxSeg、RS-VLM、Gate B、Grounded、Gate D
```

所有脚本都是薄入口；算法位于 `oa_groundrag/`。常用帮助命令：

```bash
python scripts/data/oa_auxseg_benchmark.py --help
python scripts/data/rs_general_benchmark.py --help
python scripts/data/grounded_corpus.py --help
python scripts/data/grounded_annotation.py --help
python scripts/data/grounded_supervision.py --help
python scripts/data/text_evidence_bank.py --help

python scripts/train/oa_auxseg.py --help
python scripts/train/rs_vlm.py --help
python scripts/train/grounded_adapter.py --help

python scripts/infer/oa_auxseg.py --help
python scripts/infer/rs_vlm.py --help
python scripts/infer/text_rag.py --help
python scripts/infer/oa_groundrag.py --help
python scripts/infer/demo.py --help

python scripts/evaluate/oa_auxseg.py --help
python scripts/evaluate/rs_vlm.py --help
python scripts/evaluate/gate_b.py --help
python scripts/evaluate/grounded.py --help
python scripts/evaluate/gate_d.py --help
```

### Unified Inference

调用方必须显式提供 `UnifiedTask`，不会让 LLM 猜任务。稳定 Python 流程是：

```text
UnifiedRequest
→ CapabilityRouter
→ ExecutionPlan
→ UnifiedInferenceRuntime
→ UnifiedResponse
```

六个任务保持为 `VLM_ONLY`、`SEGMENT_ONLY`、`REGION_UNDERSTANDING`、
`SEGMENT_AND_UNDERSTAND`、`KNOWLEDGE_QA` 和 `REGION_INTERPRETATION`。Router 只产生
确定性 capability 计划，不导入 torch，也不加载模型或 Bank。

只校验请求和执行计划：

```bash
python scripts/infer/oa_groundrag.py \
  --config configs/runtime/inference_v2.yaml \
  --request request.json \
  --dry-run
```

执行只读推理并写入调用方指定的全新根：

```bash
python scripts/infer/oa_groundrag.py \
  --config configs/runtime/inference_v2.yaml \
  --request request.json \
  --output-root /tmp/oa_groundrag_request_001
```

`UnifiedRequest`、`UnifiedResponse` 和 `UnifiedTask` 的 wire schema 保持 v1。普通 runtime
拒绝 `GT_MASK`、test/sealed 路径和已有输出根；candidate ID 缺失或无匹配时按现有合同回退
到 OA-AuxSeg global mask，不自动选择 Top-1。

### Unified Demo Workbench

只读 Workbench 复用现有 Benchmark reader、六个 `UnifiedTask`、router、runtime 和 lazy
providers，用于浏览 train/val Benchmark、人工维护 qualitative Demo Gallery，并展示空间结果、
grounded observation、Evidence Packet、Pass-2 与执行轨迹。它不修改 Benchmark、checkpoint、
Adapter、Text Bank 或正式 outputs。冻结评价不再是 Workbench 数据模式；评价 reader
与实现仍保留，但已退役 Eval-dev selection/development outputs 的默认配置已删除。

Workbench 的三个逻辑页签固定为 Benchmark Browser、Demo Gallery 和
Task Runner / Result Viewer / Trace；中文界面分别显示为“基准数据浏览”、
“Demo Gallery”和“任务运行 / 结果查看 / 执行轨迹”。

页面顶部提供 `界面语言 / Interface Language`，默认中文，可在 `中文` 与 `English` 间即时
切换三个页签的标题、字段、按钮、说明、状态消息、Dataframe 表头及 candidate/输入预览说明。
语言切换只更新 Gradio 表示层，不重新读取 Benchmark/HDF5，不加载模型、不执行推理或检索，
也不改变当前 sample、Gallery、Demo run、result task 或 candidate snapshot/token。模型 prompt、
用户输入、模型原始输出、Evidence 内容、科学 JSON、schema、枚举、ID、路径和 reason code 始终
保持原值，不做自动翻译。

推荐交互顺序是：

```text
Benchmark Browser 选择样本
→ 查看 Full Optical、逐通道 Optical/Multispectral 与真实 auxiliary preview
→ 运行 SEGMENT_ONLY 或包含 spatial task 的 Suite
→ 查看同一次 OA-AuxSeg 结果的 Candidate Preview
→ 明确选择 candidate，或人工确认 OA-AuxSeg global mask
→ 点击“运行所选 REGION_INTERPRETATION”
→ 查看 Pass-1、retrieval、citations、Pass-2 与 execution trace
```

首次运行默认 Suite 时，`REGION_INTERPRETATION` 会显示 `WAITING_FOR_CANDIDATE`，不会自动
选择 Top-1，也不会把缺失 candidate 的 global fallback 伪装成具体区域解释。切换样本或产生
新的 spatial run 后，旧 candidate 选择立即失效。逐通道与 auxiliary Gallery 只是
`Spatial Expert Input Preview`；当前 P0 MLLM formal grounded input 仍只有 Full Optical、
Binary Mask 和 Context Crop，preview 不会进入 `UnifiedRequest.auxiliary_views`。

纯 `KNOWLEDGE_QA` 与当前 Browser selection 完全解耦：不会读取 Benchmark payload、
构造 spatial input、运行 OA-AuxSeg 或产生 test receipt。问题留空时使用配置中的默认 prompt；
界面顶部 banner 和只读 `Effective prompts` 区会显示本次实际输入及其来源。其他视觉任务则消费
banner 中明确显示的当前样本；Benchmark Browser 初始化时已经定位一条记录，即使用户没有再次
点击样本按钮，也不是“无输入运行”。

安装可选 UI 依赖并从本地回环地址启动：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m pip install -e '.[demo]'

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/infer/demo.py \
  --config configs/runtime/demo_v1.yaml \
  --port 7860
```

默认 `allow_test_demo: false`：test 只显示锁定的 index metadata，不打开 HDF5 payload，
不能推理或加入 Gallery。只有负责人显式修改独立 Demo 授权后才能去盲化；届时每次访问先
发布 test receipt，并永久标记其不再具有 blind/sealed evaluation 属性。Reference/GT mask
始终只作 `Reference / Audit Only`，不会伪装成 `USER_MASK`。

## 稳定 Python 接口

大型根级 facade 不导出训练和评价实现。按职责导入：

```python
from oa_groundrag.segmentation import SpatialInferenceSession
from oa_groundrag.grounding import EvidenceBuilder, RegionSelector
from oa_groundrag.runtime import UnifiedInferenceRuntime, UnifiedRequest, UnifiedTask
```

- `oa_groundrag.segmentation`：OA-AuxSeg model/contracts/inference session。
- `oa_groundrag.vlm`：Qwen model、processor、checkpoint 和 grounded adapter loader。
- `oa_groundrag.grounding`：region、evidence、messages 和 output contracts。
- `oa_groundrag.retrieval`：Text RAG contracts、retriever 与 Pass-2。
- `oa_groundrag.runtime`：统一请求、路由、provider 和 runtime。
- 训练和评价通过 `oa_groundrag.training.*`、`oa_groundrag.evaluation.*` 显式导入。

## 科学合同边界

- OA-AuxSeg 只输出 probability、mask、no-target、候选区域和 modality evidence；模型
  `forward` 不读取 HDF5。
- RS-GeneralDesc 只接受 canonical manifest/payload/ledger/asset identity，不接入 mask 或
  grounded supervision。
- Grounded Evidence 使用原始光学图、严格 binary mask、clean crop、Programmatic Facts 和
  Pass-1 structured observation；empty/no-target 合同不得静默修复。
- Text RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1 观察，不把检索知识说成
  当前图像事实，也不把候选区域升级为确认滑坡。
- Gate B、Gate D、Benchmark、checkpoint、Text Evidence Bank 和正式 outputs 的身份与历史
  实现提交绑定；路径迁移不伪造旧 verifier 继续通过。

## 测试

测试按能力镜像源码职责，并全部带 package marker：

```text
tests/
├── test_architecture.py
├── data/{oa_auxseg,rs_general,grounded}/
├── segmentation/
├── vlm/
├── grounding/
├── retrieval/
├── runtime/
├── training/
├── evaluation/{rs_general,grounding,retrieval}/
└── integration/
```

标准发现命令：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

需要本地 Benchmark、checkpoint 或 CUDA 的测试必须显式 skip 或使用独立 bounded smoke，
不得为追求绿色结果重建正式资产或访问 sealed test。实时测试结果只记录在
`REBUILD_PROGRESS.md`。
