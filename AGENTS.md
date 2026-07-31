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
Stage 1  OA-AuxSeg 正式验收
Stage 2  RS-GeneralDesc Benchmark 验收
Stage 3  RS-General Adapter 评价
Stage 4  Landslide Evidence Corpus 与 OA-GroundedEval
Stage 5  Mask-Grounded Baseline
Stage 6  文本 RAG
Stage 7  案例 RAG
Stage 8  可选 Landslide-Evidence Adapter
Stage 9  统一推理与报告
```

这些 Stage 是科学依赖顺序，不要求重命名已有 `phase2/phase3/phase4` 工程目录。
不得跳过 Gate，不得把代码存在、checkpoint 文件存在或中途指标当作正式验收。

## 3. 当前边界

Stage 0 已完成权威迁移；当前科学状态是
`Stage 1 OA-AuxSeg formal acceptance pending`。进入 Stage 1 或更后阶段前，
必须重新核对 `REBUILD_PROGRESS.md` 并取得项目负责人对相应写入、正式评价或长训练的
明确授权。未经新授权禁止：

- 运行 GPU、训练、正式评估或长时间任务；
- 下载数据、模型或依赖；
- 修改 `../datasets`、`../benchmark` 或 `../external`；
- 修改或复制第三方参考实现；
- 修改既有 Benchmark、checkpoint、训练输出或模型权重；
- 提前实施 Stage 2 的合同迁移、Stage 4 的数据构建或 Stage 6/7 的 RAG；
- 创建 legacy 目录、兼容包装、alias 或旧接口适配层。

Gate A/B 的科学判据必须只基于 train/val 预注册并在首次正式 test 前冻结，不得读取
test 后反推阈值。

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
