# OA-GroundRAG v3.1：双后端 Shared RS-Geohazard MLLM

> **冻结日期：** 2026-08-29  
> **状态：** 负责人授权的新冻结算法设计  
> **替代关系：** 本文件自发布起替代 `OA-GroundRAG_算法构建方案_0811.md`
> 作为当前唯一详细算法设计；0811 版本永久只读，仅用于历史 provenance。  
> **实时状态：** 实施进度、产物 SHA、checkpoint 与验收结果只记录在
> `REBUILD_PROGRESS.md`，不得从本文件推断。

## 0. 设计目标与不变量

OA-GroundRAG 保持单一主链：

```text
OA-AuxSeg
→ Shared RS-Geohazard MLLM
→ Grounded Multimodal Understanding
→ Evidence-Constrained Text RAG
→ Unified Instruction-Routed Runtime
```

本版本只扩展 Shared MLLM 的基础模型选择：保留 Qwen3-VL-2B 的全部冻结资产与
运行入口，新增 Qwen3.5-4B 的独立训练、评价和运行链。它不是第二条算法主线，
也不改变 OA-AuxSeg、Evidence Builder、Text Evidence Bank 或 Router 的科学职责。

以下不变量继续成立：

1. 光学影像是空间边界基准；SAR、InSAR、DEM、多光谱等只能作为显式可选证据。
2. 分割只产生 probability、mask、no-target、candidate region 与程序化区域事实。
3. MLLM 不生成或修改 mask，不把候选区域升级为确认滑坡。
4. Text RAG 只消费用户问题、Programmatic Facts 与 Pass-1 structured observation；
   检索证据不能伪装成当前图像事实。
5. `UnifiedTask` 必须由调用方显式提供，Router 不让 LLM 猜任务。
6. 训练 curriculum 与运行时解耦；运行时只加载所选家族的最终 Grounded Adapter。
7. train/val 可用于工程与预注册 Gate；test/sealed 在首次正式评价前保持盲态。
8. `max_steps` 是预算上限，不是模型有效性、Gate 通过或科学验收的充分条件。

## 1. 模型家族与独立产物链

### 1.1 保留链：Qwen3-VL-2B

```text
Qwen3-VL-2B-Instruct
→ RS-General Adapter 2B
→ Gate B 2B（冻结）
→ Mask-Grounded Adapter 2B
→ Runtime Profile 2B
```

现有 v2 配置、模型目录、Adapter、checkpoint、Gate B、Benchmark、outputs 和运行时
YAML 的字节身份保持不变。新实现可以改变活动源码的组织，但不得伪造旧 verifier、
重写旧 manifest 或通过兼容 alias 延续旧具体类接口。

### 1.2 新增链：Qwen3.5-4B

```text
Qwen/Qwen3.5-4B（官方后训练多模态模型，固定 revision）
→ RS-General Adapter 4B
→ Gate B 4B（独立协议）
→ Mask-Grounded Adapter 4B
→ Runtime Profile 4B
```

模型必须是官方 `Qwen/Qwen3.5-4B`，不得替换为 `Qwen/Qwen3.5-4B-Base`。
Qwen3.5 使用 `qwen3_5` 混合架构；其默认 thinking 行为必须由任务策略显式控制。

两个家族必须绑定各自的：

- backend name 与 architecture/model type；
- 固定本地模型根、Hub repository 与不可变 revision；
- processor/tokenizer/template 身份；
- 完整模型文件与权重分片 SHA-256 ledger；
- LoRA topology、Adapter、checkpoint 和 best pointer；
- 配置 schema、训练输出、Gate 协议与运行时 profile。

跨家族 Adapter、checkpoint 或 warm-start 必须 fail closed。

## 2. 后端架构与公共合同

生产代码按能力组织在：

```text
oa_groundrag/vlm/
├── backends/
│   ├── contracts.py
│   ├── registry.py
│   ├── qwen3_vl/
│   └── qwen3_5/
├── checkpoint.py
├── inference.py
└── grounded_runtime.py
```

`VLMBackendSpec` 是不可变规格，至少绑定 backend、model type、architecture、
processor type、模型/processor 构造器、身份策略、thinking 能力和 LoRA target 解析。
registry 必须拒绝未知 backend 和重复注册。对外稳定构造入口为：

```python
resolve_vlm_backend()
build_processor_adapter()
build_model_adapter()
```

训练器、checkpoint loader、Grounded provider 与 Unified Runtime 只依赖模型无关协议，
不得导入具体家族实现。`qwen3_vl` 与 `qwen3_5` 可以共享无状态公共 utility，但不得
共享家族常量、topology 假设、template 身份或模型参数计数。

不提供旧具体类的 legacy alias、旧目录包装或 phase/stage 一级源码入口。

## 3. 配置与身份

### 3.1 RS-VLM 配置

新增严格 `rs_vlm.config.v3`：

```yaml
schema: rs_vlm.config.v3
model:
  backend: qwen3_vl | qwen3_5
  model_root: ...
```

v3 除 `backend` 外继续显式绑定 processor、图像/token 限制、dtype、attention、LoRA、
训练预算、数据 manifest/ledger 与输出根。未知键、缺失键、schema/backend 不一致、
模型目录与 backend 不匹配必须拒绝。

既有 `rs_vlm.config.v2` 只为冻结 2B 资产保留读取语义；原 YAML 及语义 SHA 不修改。

### 3.2 模型与 processor ledger

4B 下载必须使用固定完整 Hub commit revision，进入全新版本化目录。发布 ledger 对每个
实际消费文件记录相对路径、字节数、SHA-256 和角色；分片模型必须列出 index 与每个
safetensors shard，不能只哈希 index。ledger 本身使用 canonical JSON 并记录自身 SHA。

processor 身份至少覆盖：

- `config.json`、`generation_config.json`；
- tokenizer config、tokenizer/vocab/merges 与 special tokens（存在时）；
- preprocessor/video preprocessor；
- `chat_template.jinja` 或家族明确允许的模板文件；
- Hub repo 与固定 revision。

路径必须是普通文件且位于绑定根内；symlink、缺失、额外未登记权重分片、SHA 不符均拒绝。

## 4. Qwen3.5 processor 与 reasoning 合同

Qwen3.5 processor 后端必须验证：文本、单图、三图、assistant-only label mask、tensor keys、
离线加载以及 template 参数透传。训练和所有结构化任务强制：

```python
chat_template_kwargs={"enable_thinking": False}
```

运行时策略固定为：

| 任务/路径 | thinking | decoding | max_new_tokens | wire response |
| --- | --- | --- | ---: | --- |
| `VLM_ONLY` | enabled | greedy | 2048 | 仅最终答案 |
| Grounded Pass-1 | disabled | greedy | 768 | 严格 structured output |
| Text-only Pass-2 | disabled | greedy | 768 | 证据受限最终答案 |
| 训练与评价 | disabled | deterministic | 配置固定 | 不消费 reasoning |

`<think>` 或等价 reasoning 内容不得进入 `UnifiedResponse` 公共正文。provider metadata
只记录 `thinking_enabled` 与 reasoning token 数，不记录 reasoning 文本。

## 5. Qwen3.5 LoRA 与单卡资源合同

硬件目标是单张约 24 GB GPU，基础模型 BF16、SDPA、gradient checkpointing，视觉塔与
merger 严格冻结。首版 LoRA 只覆盖 Qwen3.5 8 个 full-attention 层中的：

```text
q_proj, k_proj, v_proj, o_proj
r=8, alpha=16, dropout=0.05, bias=none
```

DeltaNet 的 `in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj` 不进入首版。
实现必须按真实模块拓扑选择目标，不能仅凭叶子名误命中非 full-attention 层；只能有
上述 LoRA 参数和允许的 adapter wrapper state 可训练。

资源门按顺序执行：tiny forward/backward → state round-trip → 跨家族拒绝 → 真实 CUDA
推理 → 1-step → 20-step。记录设备、软件栈、输入边界、峰值 allocated/reserved memory。
任何 OOM、拓扑不符或 processor/kernel 不兼容立即停止；本轮不得自动安装可选 kernel、
量化、缩减输入或扩展成 hybrid LoRA。

## 6. 两段训练 curriculum

### 6.1 RS-General Adaptation

- 数据：同一冻结 RS-GeneralDesc Benchmark，仅 train/val；
- 初始化：Qwen3.5-4B 固定基模 revision；
- template：non-thinking；
- batch：micro batch 1，gradient accumulation 16；
- optimizer/scheduler/LR：沿用冻结 2B 配方，LR `2e-4`；
- 预算：`max_steps=1000`；
- 发布：同模型 val monitor 选择 best，输出 checkpoint manifest、Adapter、best pointer、
  配置/数据/模型/processor/ledger/实现身份。

先通过 bounded smoke 才能进入正式训练。输出根必须全新且拒绝覆盖。

### 6.2 Mask-Grounded Region Adaptation

- 初始化：只允许同一 Qwen3.5 家族的 RS-General best Adapter；
- 输入：原始 full optical、严格 binary mask、clean context crop；
- 数据：冻结 6,974 条 compact supervision，90% Region / 10% RS-General replay；
- optimizer/scheduler/RNG/sampler：fresh state；
- template：non-thinking；
- LR：`5e-5`；
- 预算：`max_steps=1000`；
- 输出：独立 checkpoint、Adapter、best pointer 与完整身份 manifest。

先执行 1-step 与 20-step smoke；任何跨家族 warm-start 必须拒绝。

## 7. 评价与发布状态机

### 7.1 独立 Gate B

4B Gate B 精确复用冻结 2B Gate B 的 256 个 val record ID 与六项预注册判据，但发布新的
protocol ID、schema、implementation identity 与输出根，并绑定 4B backend、revision、
processor、训练 checkpoint 和生成配置。不得读取 test。

4B Gate B 失败即终止后续链。另可对同一 256 条生成 2B/4B paired CI 报告，但必须标记
`report_only=true`；该报告不参与默认升级，也不能支持“4B 优于 2B”的科学结论。

### 7.2 Grounded 绝对验证

在全新、不可覆盖的 Benchmark 根构建 100 条 val-only 评价资产。确定性选择沿用预注册
seed 与规则：5 个来源各 20；每源 16 target、4 no-target；每源 target 的
small/medium/large=`5/6/5`。不得恢复或覆盖已退役 Eval-dev，不得读取 test。

零容忍标准为：

- 100/100 成功生成；
- schema、target status、input identity、no-target empty-region 合同均为 1.0；
- forbidden claim、overlay leakage、unresolved output 均为 0。

### 7.3 发布状态

```text
training_complete
  + Gate B PASS
  + 24GB resource gate PASS
  + six-task runtime smoke PASS
  = candidate

candidate
  + Grounded 100-record zero-tolerance PASS
  = engineering_validated / scientific_acceptance=false
```

默认 runtime/Demo 推荐命令只在 `engineering_validated` 后切换到 4B。升级只依据 4B
自身验收；paired report 不构成门。Gate C、专家评价和 sealed test 仍未完成，因此
`scientific_acceptance` 必须保持 false。

## 8. Unified Runtime

`UnifiedRequest`、`UnifiedResponse` 与 `UnifiedTask` wire schema 保持 v1。模型只在进程/会话
启动时通过显式 YAML 选择：

```text
configs/runtime/inference_v2.yaml   # 冻结 2B
configs/runtime/demo_v1.yaml        # 冻结 2B
configs/runtime/inference_qwen3_5_4b_v1.yaml
configs/runtime/demo_qwen3_5_4b_v1.yaml
```

不得新增 per-request 模型字段，不同时驻留两个模型。lazy provider 只加载所选 backend，
release 后释放该模型。推理不得读取 training/evaluation payload、compact supervision、
HDF5 或 sealed/test 路径。

六任务保持：`VLM_ONLY`、`SEGMENT_ONLY`、`REGION_UNDERSTANDING`、
`SEGMENT_AND_UNDERSTAND`、`KNOWLEDGE_QA`、`REGION_INTERPRETATION`。RAG 只用于后两项中
既有明确需要知识增强的路径，不改变 Router 的确定性计划。

## 9. 串行实施与停止门

实施固定为 M0–M10：设计与基线、公共 backend、2B 迁移、4B processor/资产、4B 模型与
资源门、RS-General 训练、Gate B、Grounded 训练、candidate runtime、100 条验证、默认发布。

每个模块必须先通过本模块测试和所有已完成模块回归，才能进入下一模块。任何以下条件
触发 fail-closed 停止并在 `REBUILD_PROGRESS.md` 留证：

- protected asset 身份漂移或输出根已存在；
- schema/backend/template/ledger/topology 不一致；
- CUDA OOM、峰值超过可用单卡容量或真实训练 smoke 失败；
- Gate B 任一预注册判据失败；
- Grounded 任一零容忍指标失败；
- 需要 QLoRA、缩减输入、可选 kernel 或 hybrid LoRA 才能继续。

停止后不得自行变更实验设计；任何替代方案必须重新 grill 并形成新的负责人授权。

## 10. 范围外事项

本设计不修改 annotation teacher Qwen3-VL-8B、OA-AuxSeg、retrieval 算法、Text Bank、
Gate D、可选 Case RAG、OA-AuxSeg Gate A、Gate C 或 sealed test 合同；不复制第三方参考实现，
不覆盖既有 2B/Benchmark/checkpoint/outputs/models_zoo 资产，不创建 legacy alias，不 commit/push。

## 11. 参考

- Qwen3.5-4B 官方模型卡：<https://huggingface.co/Qwen/Qwen3.5-4B>
- Qwen3.5-4B 官方配置：<https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json>
- Transformers Qwen3.5 文档：<https://huggingface.co/docs/transformers/model_doc/qwen3_5>
- 历史冻结设计：`docs/OA-GroundRAG_算法构建方案_0811.md`

