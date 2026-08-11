# OA-GroundRAG

本项目研究光学锚定、任意辅助模态增强的滑坡分割，以及基于候选区域证据的遥感视觉
观察、专业知识检索和证据受限生成。

## 文档权威

- [`docs/OA-GroundRAG_算法构建方案_0811.md`](docs/OA-GroundRAG_算法构建方案_0811.md) 是冻结的
  详细算法设计。文件中的状态文字属于设计冻结时的快照，不作为实时进度依据。
- [`REBUILD_PROGRESS.md`](REBUILD_PROGRESS.md) 是唯一实施进度文件，保存阶段状态、运行
  结果、冻结产物身份、验收证据和下一任务。
- [`AGENTS.md`](AGENTS.md) 保存仓库操作、安全和文档治理规则。
- `docs/archive/` 只保存历史资料，不是活动设计、接口或进度依据。

README 只描述长期有效的项目结构和运行入口，不复制阶段完成状态、测试结果或正式产物
SHA。开始任何工作前，应先读取 `REBUILD_PROGRESS.md`。

## 路线与工程映射

```text
Stage 0  资产冻结与权威路线切换
Stage 1  OA-AuxSeg 工程定版与 Gate A
Stage 2  RS-GeneralDesc Benchmark
Stage 3  RS-General Adapter 与 Gate B
Stage 4  Landslide Evidence Corpus 与 OA-GroundedEval
Stage 5  Mask-Grounded Baseline
Stage 6  文本 RAG
Stage 7  案例 RAG
Stage 8  可选 Landslide-Evidence Adapter
Stage 9  统一推理与报告
```

工程目录保留历史 phase 名称，不要求随 Stage 重命名：

```text
oa_groundrag/phase2/                    OA-AuxSeg
oa_groundrag/phase3/                    RS-GeneralDesc Benchmark
oa_groundrag/phase4/                    RS-VLM、区域证据与评价核心
oa_groundrag/landslide_evidence/        Landslide Evidence Corpus
oa_groundrag/text_rag/                  Evidence-Constrained Text RAG
oa_groundrag/unified/                   Instruction-Routed Unified Inference
scripts/phase2_oa_auxseg/               OA-AuxSeg 薄 CLI
scripts/phase3_rs_generaldesc/           RS-GeneralDesc 薄 CLI
scripts/phase4_rs_vlm/                   RS-VLM 薄 CLI
scripts/stage4_landslide_evidence/       Evidence Corpus 薄 CLI
scripts/stage6_text_rag/                 Text RAG 薄 CLI
scripts/unified/                         Unified Runtime 薄 CLI
configs/                                人工维护的严格配置
tests/                                  单元与合成回归测试
```

## 环境与外部资产

仓库使用 Python 3.11。推荐环境入口：

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
- `../benchmark` 仅在对应构建任务获得明确授权后写入，并拒绝覆盖已有根。
- `models_zoo/`、`outputs/`、`docs/RAG_knowledge/`、`参考文献/` 和 `docs/archive/`
  默认保留，不得因文档调整而重写。
- `yukun80/RAG_tmp` 只作外部工程原型参考，不复制进仓库，也不作为运行时依赖。

## 稳定运行入口

所有业务逻辑位于共享库，脚本只负责参数解析和错误分类。先使用 `--help` 查看活动接口：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py --help
python scripts/phase3_rs_generaldesc/run_rs_generaldesc.py --help
python scripts/phase4_rs_vlm/run_rs_vlm.py --help
python scripts/stage4_landslide_evidence/run_landslide_evidence.py --help
python scripts/stage4_landslide_evidence/run_mask_grounded_region.py --help
python scripts/stage4_landslide_evidence/run_single_expert_annotation.py --help
python scripts/stage4_landslide_evidence/run_model_assisted_supervision.py --help
python scripts/unified/run_oa_groundrag.py --help
```

### Instruction-Routed Unified Inference

P0 统一入口要求调用方显式提供 `UnifiedTask`，不会让 LLM 猜任务。稳定 Python API 为
`UnifiedRequest → CapabilityRouter → ExecutionPlan → UnifiedInferenceRuntime → UnifiedResponse`；
六个任务是 `VLM_ONLY`、`SEGMENT_ONLY`、`REGION_UNDERSTANDING`、
`SEGMENT_AND_UNDERSTAND`、`KNOWLEDGE_QA` 和 `REGION_INTERPRETATION`。Router 只产生布尔
capability 计划，不导入 torch，也不加载模型或 Bank。

最小知识问答请求示例：

```json
{
  "schema_version": "oa_groundrag.unified_request.v1",
  "request_id": "example-knowledge-qa",
  "task": "KNOWLEDGE_QA",
  "instruction": "Why can InSAR LOS measurements not directly determine full 3-D displacement?",
  "images": [],
  "user_mask": null,
  "spatial_input": null,
  "region_source": "NONE",
  "candidate_region_id": null,
  "auxiliary_views": [],
  "include_audit": true
}
```

只校验请求和确定性计划，不构造 provider 或读取模型、Bank、checkpoint：

```bash
python scripts/unified/run_oa_groundrag.py \
  --config configs/unified/inference_v1.yaml \
  --request request.json \
  --dry-run
```

真实推理只写调用方指定的全新输出根，并按任务惰性加载所需 provider：

```bash
python scripts/unified/run_oa_groundrag.py \
  --config configs/unified/inference_v1.yaml \
  --request request.json \
  --output-root /tmp/oa_groundrag_request_001
```

`REGION_INTERPRETATION + OA_AUXSEG_CANDIDATE` 只按精确 candidate ID 选择。ID 缺失、ID
不存在或 candidates 为空时，响应以 exit code 0 兼容回退到 OA-AuxSeg global mask，并在
非审计字段 `region_selection` 和 `limitations` 中记录原因；不自动选择 Top-1。普通 runtime
拒绝 `GT_MASK`、`test`/`sealed` 路径和已有输出根。Stage 5 best、OA-AuxSeg 与 BGE-M3
采用单重型 provider 常驻策略；RS-General Adapter 只是训练 curriculum，不作为 runtime 的
前置生成阶段。

### OA-AuxSeg

OA-AuxSeg 负责光学主导、辅助模态可选的滑坡分割。配置位于
`configs/phase2_oa_auxseg/`，实现位于 `oa_groundrag/phase2/`。训练、评价、推理和离线
finalization 必须严格绑定 Benchmark、registry、模型和 checkpoint 合同。

训练预算上限不自动构成 Gate A；工程定版权重、正式 test 和 Gate A 是不同状态。具体
checkpoint、报告和评价状态只查 `REBUILD_PROGRESS.md`。

### RS-GeneralDesc Benchmark

RS-GeneralDesc 使用 RSGPT、MMRS-1M 和 DisasterM3 的 External train/val 文本任务。
活动实现只接受原生 canonical/manifest 合同，不提供 OA mask、Gold/Silver、混合构建或
旧 schema alias。

```bash
python scripts/phase3_rs_generaldesc/run_rs_generaldesc.py --help
```

Builder、validator、repackage、Dataset 和 exporter 都必须以 manifest、payload、ledger、
shard 和 asset identity 绑定实际消费文件。正式 Benchmark 身份和验证证据只记录在
`REBUILD_PROGRESS.md`。

### RS-VLM 与 Gate B

RS-VLM 复用 Qwen processor/model、LoRA、checkpoint、推理、结构化 evidence 和反事实
评价核心。External 通用描述与 mask-grounded 数据流必须保持合同隔离。

```bash
python scripts/phase4_rs_vlm/run_rs_vlm.py preflight --help
python scripts/phase4_rs_vlm/run_rs_vlm.py train --help
python scripts/phase4_rs_vlm/run_rs_vlm.py gate-b-verify --help
```

Gate B 的协议、selection、Base/Adapter prediction、报告和 artifact SHA 属于版本化正式
证据，不在 README 固定。读取和复核时使用 `REBUILD_PROGRESS.md` 中登记的对应根和身份。

按 prediction 的一基行号定位持久化 canonical 图片：

```bash
python scripts/phase4_rs_vlm/run_rs_vlm.py gate-b-locate-media \
  --predictions <predictions.jsonl> \
  --line-number <N> \
  --benchmark-root <rs-generaldesc-root>
```

该工具只读验证 prediction、manifest、ledger、record 和 asset identity；它不重新生成
预测，也不能替代 Gate B verifier。

### Landslide Evidence Corpus

Stage 4 Corpus 工具只负责从获准的 OA-AuxSeg train 数据构建和验证确定性区域证据。既有
输出根默认只读；是否允许新构建必须以 `REBUILD_PROGRESS.md` 和负责人授权为准。

```bash
python scripts/stage4_landslide_evidence/run_landslide_evidence.py build-auto \
  --config <stage4a-config.yaml>

python scripts/stage4_landslide_evidence/run_landslide_evidence.py validate \
  --root <corpus-root>
```

Corpus record 可保留面向未来的 Silver/review 字段，但 Auto Corpus 本身不是人工真值、
完整案例知识库、分割 Benchmark 或 OA-GroundedEval。Teacher Silver、专家审核和
OA-GroundedEval 必须作为独立授权任务实施。

新版 Mask-Grounded Region 工具使用独立 schema 和输出根：Corpus 只允许人工 GT mask
与 train shard；OA-GroundedEval-dev 只允许 val shard，并在代码级拒绝 test。正式 VLM
输入将未加标记的 full RGB、独立 PNG-L binary mask 和 clean context crop 分开组织，
彩色 overlay 仅作为 audit-only 资产。

```bash
python scripts/stage4_landslide_evidence/run_mask_grounded_region.py \
  build-region-corpus --config <region-corpus-config.yaml>

python scripts/stage4_landslide_evidence/run_mask_grounded_region.py \
  validate-region-corpus --root <region-corpus-root>

python scripts/stage4_landslide_evidence/run_mask_grounded_region.py \
  build-eval-dev --config <oa-grounded-eval-dev-config.yaml>

python scripts/stage4_landslide_evidence/run_mask_grounded_region.py \
  validate-eval-dev --root <eval-dev-root> --train-corpus-root <region-corpus-root>
```

同一入口还提供 `export-annotation-queue`、`validate-annotations`、`render-messages` 和
`evaluate-dev`。其中 message renderer 不调用模型；开发 evaluator 只消费已有 prediction，
在专家协议和阈值冻结前始终输出 `formal_acceptance=false`。

单专家最小标注入口将本地 Qwen 草稿与专家最终答案分离。train package 只用于
`expert_verified_train_supervision`，val baseline package 只作为
`single_expert_dev_reference`；两者都不是 Gold 或专家共识。只有草稿生成命令使用 GPU，
且配置强制 `local_files_only=true`，不会调用外部 API 或自动下载模型。
草稿生成按记录逐条执行：模型在一次命令中只加载一次，每次只处理一个 sample；`--limit 1`
只限制本次生成数量。人工工作台固定写入 `annotator="expert"`；`pending` 视图只显示当前
partition 中已有草稿且尚未核验的记录，`all` 视图用于重新打开已核验答案。

新版 Region Corpus、Eval-dev、可恢复工作根和最终训练资产统一位于
`../benchmark/oa_grounded_stage4_v1/`。train 的稳定一键入口如下；它首次推进 20 条
calibration 并在核验完成后退出，负责人检查 prompt/config 后再次运行同一命令，才推进
剩余 480 条。达到 500/500 后会自动发布并严格验证 annotation package 与 training messages。

```bash
python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  run-train-workflow
```

该一键命令只接受可选的 `--port`。中断后重复运行会跳过已有草稿和已核验记录；合法的部分
发布会继续完成，非法既有发布根则拒绝覆盖。显存释放和调度由运行负责人在命令外管理。
工作台启动前只在当前 Python 进程中保留并精确补全 `NO_PROXY` / `no_proxy` 的
`127.0.0.1,localhost`，无需改动 shell 或手工关闭全局代理；Gradio 启动失败会关闭已部分
启动的服务并返回结构化错误。

模型消息只提供字段、类型、英文枚举和逐维视觉问题，不向 Qwen 注入完整答案模板。生成后会
独立重算 `informative`、`limited_but_specific`、`low_information` 或
`not_applicable_no_target`；JSON 合法不等于草稿有信息，target 空模板复制会在工作台启动前
被拒绝。人工界面同时提供分组字段表单、始终可见的只读 canonical JSON 和折叠的高级 JSON
文本区；高级 JSON 只有通过严格解析后才能同步回表单。数组按每行一项编辑，no-target 的区域
字段自动锁定，窄幅 crop 只显示警告而不替换原始输入，audit overlay 继续默认折叠。

项目创建时会复制当时的 prompt 供 20 条 calibration 使用。只有 calibration 全部核验完成、
负责人再次运行一键命令确认进入 remaining 阶段时，仓库 prompt 才会原子同步到工作根，并与
generation config 一起冻结身份；冻结后继续运行会拒绝 prompt 或 config 漂移。
以下细粒度命令保留用于只读验证、开发参考和人工恢复：

```bash
python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  create-annotation-project --asset-root <region-or-eval-root> \
  --output-root <fresh-work-root> --intended-use <intended-use> \
  --prompt <prompt.txt>

python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  generate-annotation-drafts --project-root <work-root> \
  --config <local-qwen-config.yaml> --partition <calibration-or-remaining-or-all>

python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  serve-annotation --project-root <work-root> \
  --partition <calibration-or-remaining-or-all> --view <pending-or-all>

python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  export-verified-annotations --project-root <work-root> \
  --output-root <fresh-package-root>

python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  validate-annotations --asset-root <region-or-eval-root> \
  --package-root <package-root>

python scripts/stage4_landslide_evidence/run_single_expert_annotation.py \
  export-training-messages --asset-root <train-region-root> \
  --annotations-root <train-package-root> --output-root <fresh-messages-root>
```

创建 val 项目时使用 `--train-project-root <frozen-train-work-root>`，不再提供 `--prompt`，
以强制复用 train calibration 后冻结的 prompt 和草稿配置。训练消息导出代码级拒绝
val/test package；`MaskGroundedTrainingMessageDataset` 会重新验证 source、annotation、
message 和 ledger identity，再返回现有 `DescriptionCollator` 可消费的监督样本。

大规模模型辅助监督使用独立的 Stage 4 v2 根
`../benchmark/oa_grounded_stage4_v2/`。扩展 Corpus 只从 OA Benchmark 的 train GT 构建；
轻量 collection 通过强身份引用冻结的 500 条 Corpus 与新增成员，不复制旧图像资产。先运行
CPU 准备命令，再由负责人显式启动本地模型逐条补齐草稿：

```bash
python scripts/stage4_landslide_evidence/run_model_assisted_supervision.py \
  prepare-expanded-corpus

python scripts/stage4_landslide_evidence/run_model_assisted_supervision.py \
  run-train-workflow

python scripts/stage4_landslide_evidence/run_model_assisted_supervision.py \
  publish-compact-training

python scripts/stage4_landslide_evidence/run_model_assisted_supervision.py \
  validate-compact-training
```

第二条命令只使用本地冻结 Qwen 配置，不调用外部 API，也不启动人工界面；中断后重跑会跳过
已原子落盘的草稿。已有专家答案优先标记为 `expert_verified`，其他通过严格 parser、禁止结论
和信息量检查的草稿标记为 `model_generated_unreviewed`，不合格项进入 exclusions。因此训练
messages 数量由实际合格记录决定，不等同于 collection 总数。该混合监督不是 Gold、专家共识
或科学验收，所有发布 manifest 均保持 `formal_acceptance=false`。

Stage 5 使用独立 compact 消息作为训练入口，并从冻结的 RS-General Adapter LoRA 权重
warm-start；状态机会依次执行 Base/RS-General GT-mask baseline、Region 训练与自动开发评价，
最后仅报告 RS-General retention 变化：

```bash
python scripts/phase4_rs_vlm/run_mask_grounded_adapter.py \
  run-stage5-workflow \
  --config configs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1.yaml
```

该入口不读取 sealed test，不把 retention 结果解释成 Gate F，也不产生正式科学接受结论。

### Evidence-Constrained Text RAG

Stage 6 只消费用户问题、已发布的 Pass-1 structured visual observation 和程序事实；它不再
输入图像，不生成或修改 mask，也不改写 Pass-1 视觉观察。依赖安装与统一入口如下：

```bash
python -m pip install -e '.[phase4,stage6]'
python scripts/stage6_text_rag/run_text_rag.py --help
python scripts/stage6_text_rag/run_text_rag.py preflight
```

正式配置显式绑定知识来源、固定 revision 的本地 dense 权重、Stage 5 best pointer、
OA-GroundedEval-dev 和 Pass-1 predictions。首次构建必须使用全新输出根；已有正式根只运行
只读 validator，不允许静默覆盖：

```bash
python scripts/stage6_text_rag/run_text_rag.py build-bank
python scripts/stage6_text_rag/run_text_rag.py validate-bank
python scripts/stage6_text_rag/run_text_rag.py prepare-dev
python scripts/stage6_text_rag/run_text_rag.py retrieve-dev
python scripts/stage6_text_rag/run_text_rag.py validate-retrieval
```

paired Pass-2 使用相同 Pass-1、generator、prompt 主体和 greedy decoding；`no_rag` 与
`text_rag` 的唯一主要差异是 evidence packet。该命令需要 CUDA，只运行配置冻结的开发样本：

```bash
python scripts/stage6_text_rag/run_text_rag.py generate-paired --limit 5
python scripts/stage6_text_rag/run_text_rag.py validate-run
```

在既有 80-record retrieval 与工程 smoke 之上，automatic-only Gate D 开发评价使用独立
严格协议。`prepare` 冻结并审计样本与 text-only token，`generate` 需要 CUDA，`evaluate`
只计算可重算的结构、引用和成对文本描述性指标，`validate` 只读复核全部三层产物：

```bash
python scripts/stage6_text_rag/run_gate_d_dev.py \
  --config configs/stage6_text_rag/gate_d_dev_v1.yaml prepare
python scripts/stage6_text_rag/run_gate_d_dev.py \
  --config configs/stage6_text_rag/gate_d_dev_v1.yaml generate
python scripts/stage6_text_rag/run_gate_d_dev.py \
  --config configs/stage6_text_rag/gate_d_dev_v1.yaml evaluate
python scripts/stage6_text_rag/run_gate_d_dev.py \
  --config configs/stage6_text_rag/gate_d_dev_v1.yaml validate
```

Bank/retrieval/paired validator 通过仅表示工程合同成立，不表示 Gate D、专家评价或科学验收
通过；正式产物身份和当前边界只查 `REBUILD_PROGRESS.md`。

## 核心科学边界

- 光学影像是分割主模态和空间边界基准。
- DEM、InSAR、SAR、多光谱等只能作为真实存在且合同明确的可选证据。
- 不从文件名、通道顺序或 HDF5 格式猜测模态、单位、配准或物理语义。
- 分割输出只表达 probability、mask、no-target 和区域信息，不直接生成专业结论。
- RAG 只检索专业规则、案例、反例和限制，不生成 mask，也不改写程序确定性事实。
- 缺失、低覆盖或物理语义未知的辅助模态必须输出 unavailable/unknown，不补造结论。
- val/test、Gate、训练和正式产物之间必须保持预注册、split 隔离和身份可追溯。
- Gate 结论只在其冻结作用域内有效，不自动扩张到其他 Stage 或完整系统验收。

## 测试入口

以下命令描述稳定的回归入口；实际执行结果和未运行项只记录在进度文件：

```bash
python -m unittest discover -s tests/phase1_benchmark_build -v
python -m unittest discover -s tests/phase2_oa_auxseg -v
python -m unittest discover -s tests/phase3_rs_generaldesc -v
python -m unittest discover -s tests/phase4_rs_vlm -v
python -m unittest discover -s tests/stage6_text_rag -v
git diff --check
```

测试、preflight 或 validator 通过不等于科学 Gate 通过。任何 GPU、长训练、正式评价、
Benchmark 写入、外部 API 或 sealed test 操作都需要与当前任务相符的明确授权。
