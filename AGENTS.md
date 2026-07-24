# Repository Agent Guide

## 1. 当前研究主线

本仓库正在从零重建 OA-AuxSeg + VLM 滑坡遥感研究系统。当前设计权威为：

1. 项目负责人当前明确指令；
2. `docs/光学锚定任意辅助模态滑坡分割与VLM区域理解_算法构建方案.md`；
3. `README.md`；
4. `REBUILD_PROGRESS.md`。

`docs/archive/` 仅保存历史资料，不是当前设计、接口或验收依据。

## 2. 阶段顺序

```text
阶段 0  清理旧实现并建立最小工程基线
阶段 1  只读审计真实 HDF5
阶段 2  构建统一 Benchmark
阶段 3  光学分割 baseline
阶段 4  辅助模态编码
阶段 5  任意辅助模态注入
阶段 6  质量选择
阶段 7  完整分割训练与评价
阶段 8  VLM 指令路由与区域证据
阶段 9  VLM 区域描述与问答
阶段 10 端到端集成
```

不得跳过数据审计直接继承旧字段、通道顺序、归一化参数或 Benchmark 协议。

## 3. 当前边界

阶段 0 只允许清理、文档整理和静态验证。禁止：

- 编写模型、Trainer、Evaluator、Benchmark builder 或 RAG；
- 运行 GPU、训练、正式评估或长时间任务；
- 下载数据、模型或依赖；
- 修改 `../datasets`、`../benchmark` 或 `../external`；
- 修改或复制第三方参考实现；
- 创建 legacy 目录、兼容包装、alias 或旧接口适配层。

进入后续阶段前，先读取新算法方案和 `REBUILD_PROGRESS.md`，并核对项目负责人的当前授权。

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
- `参考文献/` 和 `docs/archive/` 必须保留。

HDF5 格式统一不代表字段、模态、配准、数值范围或科学语义统一。任何读取合同必须来自现场只读审计。

## 5. 新系统边界

- 光学影像是分割主模态和空间边界基准。
- SAR、InSAR、DEM、多光谱等只能作为可选辅助证据。
- 分割模型只输出概率图、mask、no-target 状态和区域信息。
- VLM 在分割稳定后，基于 mask、光学区域和可用辅助证据完成区域理解。
- RAG 只预留接口，不是当前实现重点。

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

- 新算法方案：唯一详细设计。
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
