# Repository Agent Guide

## 1. 当前研究主线

本仓库基于现有 OA-AuxSeg、RS-GeneralDesc 和 Mask-Grounded VLM 资产，按
OA-GroundRAG v2 路线继续开发。当前设计权威为：

1. 项目负责人当前明确指令；
2. `docs/OA-GroundRAG_算法构建方案.md`；
3. `REBUILD_PROGRESS.md`；
4. `README.md`。

`docs/archive/` 仅保存历史资料，不是当前设计、接口或验收依据。

## 2. 阶段顺序

```text
Stage 0  冻结现有资产并切换权威路线
Stage 1  OA-AuxSeg 工程定版（Gate A 待执行）
Stage 2  RS-GeneralDesc Benchmark 验收
Stage 3  RS-General Adapter 评价
Stage 4  Landslide Evidence Corpus 与 OA-GroundedEval
Stage 5  Mask-Grounded Baseline
Stage 6  文本 RAG
Stage 7  案例 RAG
Stage 8  可选 Landslide-Evidence Adapter
Stage 9  统一推理与报告
```

这些 Stage 包含两条可独立推进的依赖支线：OA-AuxSeg 工程定版 → 未来 Gate A →
formal fixed masks；RS-GeneralDesc Stage 2 → Adapter 重训 → Gate B。两条支线在后续
Mask-Grounded 阶段汇合。Gate A 延后不等于通过，也不阻止独立的 Stage 2/3 支线准备；
任何依赖 Gate 的下游产物仍不得越过对应 Gate。工程目录无需随 Stage 重命名。

## 3. 当前边界

当前科学状态是
`Stage 2 RS-GeneralDesc Benchmark native v1 acceptance completed / Stage 3 RS-General Adapter retraining and Base-vs-Adapter Gate B pending; Gate A, ablations, sealed test and formal fixed masks remain deferred`。
Stage 2 以 `/home/yukun80/codes/benchmark/rs_generaldesc_v1`、
`rs_generaldesc.manifest.v1` 和 `rs_generaldesc.canonical.v1` 原生发布
External train/val；manifest 本身 eligible 且 blockers 为空，不再需要额外作用域报告。
该结论不等于 OA-Grounded 数据验收或 Gate B 通过，`external_val` 仍只用于训练监控。
Stage 1 的 batch-16
proposed final 权重仍为 `checkpoint_best.pt` step 206820，不再续训；Gate A、消融、
sealed test 和正式 fixed masks 继续延后。

活动产品统一称为 **RS-GeneralDesc Benchmark**，训练系统统一称为 **RS-VLM**。
phase3 只保留三类 External source、两类 role 与七类文本任务；没有 OA Gold/Silver、
mask target、混合 builder、旧 schema validator 或兼容 alias。phase4 配置为
`rs_vlm.config.v2`，其他产物合同仍为 `rs_vlm.*.v1`；preflight 和 Dataset 按需绑定
native manifest、validation、build、payload、ledger、shard 与 asset identity。
GT/fixed/end-to-end mask、RegionSelector、EvidenceBuilder、mask-grounded messages 和
AuxSeg inference 仅作为 Stage 5 核心保留；临时 Mask-Grounded Dataset 合同未公开，
须等新数据 schema 冻结后才能接入活动运行入口。上一轮候选 Adapter
产物已随身份迁移删除，Stage 3 必须重新训练后再冻结 Gate B。

进入后续任务前，必须重新核对 `REBUILD_PROGRESS.md` 并取得项目负责人对相应写入、
正式评价或长训练的明确授权。未经新授权禁止：

- 运行 GPU、训练、正式评估或长时间任务；
- 下载数据、模型或依赖；
- 修改 `../datasets`、`../benchmark` 或 `../external`；
- 修改或复制第三方参考实现；
- 修改既有 Benchmark、checkpoint、训练输出或模型权重；
- 提前实施 Stage 4 的数据构建或 Stage 6/7 的 RAG；
- 创建 legacy 目录、兼容包装、alias 或旧接口适配层。

Gate A/B 的科学判据必须只基于 train/val 预注册并在首次正式 test 前冻结，不得读取
test 后反推阈值。计划 `max_steps` 是预算上限，不是权重有效性或 Gate A 的独立条件；
负责人定版 final checkpoint 也不等于 Gate A 已通过。

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

- 光学影像是分割主模态和空间边界基准。
- SAR、InSAR、DEM、多光谱等只能作为可选辅助证据。
- 分割模型只输出概率图、mask、no-target 状态和区域信息。
- RS-General VLM 先完成通用遥感视觉理解，再基于 mask、光学区域和可用辅助证据完成
  候选区域观察。
- Landslide RAG 只检索专业规则、案例、反例和限制，不生成 mask、不改写确定性事实，
  也不得把错误候选稳定合理化为滑坡。
- Landslide-Evidence Adapter 仅在 Gate E 失败时实施，不作为默认训练阶段。

旧 SANE、QMEF、PMRD、MGRR、SegDesc、Bridge、proposal、query 和 reliability 路线不得恢复到活动代码。

## 6. 工程规则

- Python 3.11，四空格缩进，公共合同使用类型标注。
- 优先使用 `pathlib`、严格 JSON/JSONL、原子写入和 SHA-256。
- 新可执行脚本必须有简短中文头部，说明用途、命令、输入、输出、写入行为和所属阶段。
- 算法不得写在 CLI 中。
- 不从文件名猜测通道科学含义。
- 不在模型 `forward` 中读取 HDF5。
- 保留用户已有改动；禁止 `git reset --hard`、`git checkout --` 和广泛清理。
- 未经明确请求不得 commit 或 push。

## 7. 文档职责

- `docs/OA-GroundRAG_算法构建方案.md`：唯一详细算法设计。
- `README.md`：当前项目概览和有效运行入口。
- `REBUILD_PROGRESS.md`：唯一活动进度文件。
- `docs/archive/`：只读历史资料。

不要新增 ADR、handoff、audit、worklog 或重复运行说明。

## 8. 新会话检查

1. 读取本文件、新算法方案、README 和 REBUILD_PROGRESS。
2. 运行只读 Git branch、HEAD、status 和 diff 检查。
3. 核对 `../datasets`、`../benchmark`、`../external` 的现场状态。
4. 确认当前阶段和写入授权。
5. 只完成当前阶段的最小闭环。
6. 报告实际修改、检查命令、未运行程序、阻塞和下一步。
