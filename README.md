# OA-GroundRAG

本项目研究光学锚定、任意辅助模态增强的滑坡分割，以及基于候选区域证据的遥感视觉
观察、专业知识检索和证据受限生成。

## 文档权威

- [`docs/OA-GroundRAG_算法构建方案.md`](docs/OA-GroundRAG_算法构建方案.md) 是冻结的
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
scripts/phase2_oa_auxseg/               OA-AuxSeg 薄 CLI
scripts/phase3_rs_generaldesc/           RS-GeneralDesc 薄 CLI
scripts/phase4_rs_vlm/                   RS-VLM 薄 CLI
scripts/stage4_landslide_evidence/       Evidence Corpus 薄 CLI
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
```

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
git diff --check
```

测试、preflight 或 validator 通过不等于科学 Gate 通过。任何 GPU、长训练、正式评价、
Benchmark 写入、外部 API 或 sealed test 操作都需要与当前任务相符的明确授权。
