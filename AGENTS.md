# Repository Agent Guide

## 1. 当前研究主线

本仓库基于现有 OA-AuxSeg、RS-GeneralDesc 和 Mask-Grounded VLM 资产，按
 OA-GroundRAG v3.1 路线继续开发。权威边界为：

1. 项目负责人当前明确指令；
2. `docs/OA-GroundRAG_算法构建方案_0829.md` 中的冻结算法设计；
3. `REBUILD_PROGRESS.md` 中的实时阶段、产物身份和验收证据；
4. `README.md` 中的稳定入口和使用说明。

`docs/archive/` 仅保存历史资料，不是当前设计、接口或验收依据。
算法方案中的状态文字只是设计冻结时的快照，不得用于判断现场进度。

## 2. 阶段顺序

```text
Stage 0  冻结现有资产并切换权威路线
Stage 1  OA-AuxSeg 工程定版与 Gate A
Stage 2  RS-GeneralDesc Benchmark 验收
Stage 3  RS-General Adapter 评价
Stage 4  Landslide Evidence Corpus 与 OA-GroundedEval
Stage 5  Mask-Grounded Baseline
Stage 6  Evidence-Constrained Text RAG
Stage 7  可选 Case RAG
Stage 8  可选 Landslide-Evidence Adapter
Stage 9  统一推理与报告
```

这些 Stage 包含两条可独立推进的依赖支线：OA-AuxSeg 工程定版 → Gate A →
formal fixed masks；RS-GeneralDesc Stage 2 → Adapter 训练 → Gate B。两条支线在
Mask-Grounded 阶段汇合。任何依赖 Gate 的下游产物都不得越过对应 Gate。
Stage 只用于训练 curriculum、配置/产物 provenance 和历史进度；长期源码、配置、脚本
与测试必须按算法能力和工程职责组织，不得再以 phase/stage 作为一级目录。

## 3. 动态状态与授权边界

AGENTS 不固定任何 Stage 完成状态、运行数字、checkpoint、产物 SHA 或验收结论。
每次任务必须重新读取 `REBUILD_PROGRESS.md`，以其中的当前阶段、写入授权、
冻结身份和下一任务为准。不得从算法方案或 README 推断实时进度。

活动产品统一称为 **RS-GeneralDesc Benchmark**，训练系统统一称为 **RS-VLM**。
`oa_groundrag.data.rs_general` 只处理 External 通用遥感文本数据；mask-grounded 证据、
OA-GroundedEval 和 RAG 使用各自独立合同。任何新产物都必须以版本化 manifest、ledger、
schema 和实际消费文件绑定身份。

进入后续任务前，必须重新核对 `REBUILD_PROGRESS.md` 并取得项目负责人对相应写入、
正式评价或长训练的明确授权。未经新授权禁止：

- 运行 GPU、训练、正式评估或长时间任务；
- 下载数据、模型或依赖；
- 修改 `../datasets`、`../benchmark` 或 `../external`；
- 修改或复制第三方参考实现；
- 修改既有 Benchmark、checkpoint、训练输出或模型权重；
- 未经授权重新实现或运行 Teacher Silver、生成 Gold/OA-GroundedEval；
- 修改、重建或扩展现有 Stage 6、实施 Stage 7、运行正式 Gate D 或访问 sealed test；
- 创建 legacy 目录、兼容包装、alias 或旧接口适配层。

Gate 的科学判据必须只基于 train/val 预注册并在首次正式 test 前冻结，不得读取
test 后反推阈值。计划 `max_steps` 是预算上限，不是权重有效性或 Gate 的独立条件；
负责人定版 checkpoint 也不等于科学 Gate 通过。

## 4. 数据与外部资产

默认根目录：

```text
/home/yukun80/codes/
├── datasets/    只读 HDF5 原始训练资产
├── benchmark/   后续阶段生成的统一 Benchmark
├── external/    第三方算法参考代码
└── paper7_VLM/  当前仓库
```

- `../datasets` 只读；不得覆盖、重命名、移动或删除文件。
- `../benchmark` 的写入必须由后续 Benchmark 阶段明确授权，且不得覆盖已有输出。
- `../external` 只作阅读参考，不得作为运行时依赖或复制进项目代码。
- `models_zoo/` 保存本地模型权重与元数据；未经明确授权不得删除或改写。
- `参考文献/`、`docs/RAG_knowledge/` 和 `docs/archive/` 必须保留。
- GitHub `yukun80/RAG_tmp` 只作为外部工程原型；不得直接导入、复制或作为运行时依赖。
- `outputs/` 中既有产物默认只读；正式发布身份不得因文档改名而重写。

HDF5 格式统一不代表字段、模态、配准、数值范围或科学语义统一。任何读取合同必须来自现场只读审计。

## 5. 新系统边界

- 活动算法框架固定为 OA-AuxSeg → RS-GeneralDesc/RS-VLM → Mask-Grounded Region
  Adapter → Evidence-Constrained Text RAG → 可选 Case RAG；各阶段通过版本化合同和
  可重算身份衔接，不增加并行的替代主链。
- 光学影像是分割主模态和空间边界基准。
- SAR、InSAR、DEM、多光谱等只能作为可选辅助证据。
- 分割模型只输出概率图、mask、no-target 状态和区域信息。
- RS-General VLM 先完成通用遥感视觉理解，再基于 mask、光学区域和可用辅助证据完成
  候选区域观察。
- Stage 6 只消费用户问题、Programmatic Facts 和 Pass-1 structured visual observation，
  从统一文本证据库检索 interpretation、confounder、limitation 三类专业知识，并在
  text-only Pass-2 中完成证据受限解释。
- Landslide RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1 视觉观察，
  不把检索知识描述为当前图像已经观察到的事实，也不得把候选区域升级为确认滑坡。
- Landslide-Evidence Adapter 仅在 Gate E 失败时实施，不作为默认训练阶段。

## 6. 工程规则

- Python 3.11，四空格缩进，公共合同使用类型标注。
- 优先使用 `pathlib`、严格 JSON/JSONL、原子写入和 SHA-256。
- 稳定能力位置为 `segmentation`、`vlm`、`grounding`、`retrieval`、`runtime`；数据生产、
  训练和评价分别进入 `data`、`training`、`evaluation`。
- `configs/` 按能力组织；`scripts/` 按 `data/train/infer/evaluate` 组织；`tests/` 镜像能力
  边界。历史 Stage 可保留在配置名、schema、输出根和 provenance 中。
- 新可执行脚本必须有简短中文头部，说明用途、命令、输入、输出、写入行为和所属能力；
  如 Stage 对 provenance 有意义，可同时说明 Stage。
- 算法不得写在 CLI 中。
- 生产 package 不得反向导入 `scripts`，不得创建 phase/stage compatibility alias。
- 不从文件名猜测通道科学含义。
- 不在模型 `forward` 中读取 HDF5。
- 保留用户已有改动；禁止 `git reset --hard`、`git checkout --` 和广泛清理。
- 未经明确请求不得 commit 或 push。

## 7. 文档职责

- `docs/OA-GroundRAG_算法构建方案_0829.md`：冻结的唯一详细算法设计。不得因实施
  进度、运行结果或产物发布修改；只有负责人明确授权新的设计版本时才能变更。
- `docs/OA-GroundRAG_算法构建方案_0811.md`：历史冻结设计，永久只读，仅用于
  provenance，不再作为当前接口或验收依据。
- `README.md`：稳定项目概览、接口和运行入口；不记录实施进度或正式产物身份。
- `REBUILD_PROGRESS.md`：唯一实时进度、运行结果、产物身份和验收证据文件。
- `docs/archive/`：只读历史资料。

不要新增 ADR、handoff、audit、worklog 或重复运行说明。

## 8. 新会话检查

1. 读取本文件和 `REBUILD_PROGRESS.md`，确认实时阶段、授权和冻结身份。
2. 读取冻结算法方案和 README，获取设计与稳定入口，不从其状态文字推断进度。
3. 运行只读 Git branch、HEAD、status 和 diff 检查。
4. 核对 `../datasets`、`../benchmark`、`../external` 的现场状态。
5. 只完成获得授权的当前任务最小闭环。
6. 将实际进度、验证结果和下一步只更新到 `REBUILD_PROGRESS.md`。
