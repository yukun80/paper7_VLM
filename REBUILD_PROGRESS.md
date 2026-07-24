# REBUILD_PROGRESS

## 当前状态

- program: `OA_AUXSEG_VLM_REBUILD`
- phase: `0`
- phase_name: `REPOSITORY_CLEANUP_AND_MINIMAL_BASELINE`
- phase_status: `complete`
- execution_date: `2026-07-24`
- branch: `main`
- baseline_head: `1c5f1eb269c793026925381c43e7601f4593f1af`
- model_implemented: `false`
- benchmark_built: `false`
- training_run: `false`
- gpu_run: `false`
- download_run: `false`
- commit_performed: `false`
- push_performed: `false`

## 已完成

- 删除旧 `SEG_Multi-Source_Landslides` 运行时及其模型、Trainer、Evaluator、CLI 和测试。
- 删除旧 `scripts`、`configs`、`schemas`、`src/sami_gsd`、`tests` 和环境探针。
- 删除旧活动 Task Spec、Codex Prompt、ADR、audit、handoff、report、research 和 worklog。
- 保留 `docs/archive` 的全部四个历史文件，未改写其内容。
- 删除整个旧 `outputs`，包括 cache、checkpoint、报告和可视化。
- 删除旧目录中的日志、Python cache 和字节码。
- 重写 README 和 AGENTS，建立 OA-AuxSeg + VLM 当前路线及只读资产边界。
- 建立无活动包、无 CLI、无依赖的最小 Python 3.11 `pyproject.toml`。
- 纳入新的中文算法构建方案，并保留项目负责人已有的旧方案删除与 Task Spec 归档调整。

当前 Git diff 包含 352 个本次补丁删除的旧跟踪文件。另有两个清理前已存在的用户删除：
`docs/OA_RAGSEG_IMPLEMENTATION_PLAN.md` 和原 `docs/REFACTOR_TASK_SPEC.md`；后者的新归档副本位于
`docs/archive/REFACTOR_TASK_SPEC.md`。

## 保留资产

- `/home/yukun80/codes/datasets`
- `/home/yukun80/codes/benchmark`
- `/home/yukun80/codes/external`
- `models_zoo`
- `参考文献`
- `docs/archive`
- `LICENSE`
- `.gitignore`
- Git 历史

2026-07-24 只读复计：

| 数据源 | HDF5 文件 | 大小 |
| --- | ---: | ---: |
| GDCLD | 26,894 | 26.211 GiB |
| LMHLD | 56,370 | 3.072 GiB |
| LandslideBench_agent | 4,260 | 1.569 GiB |
| Landslide4Sense | 7,598 | 1.570 GiB |
| multimodal-landslide-dataset | 12,168 | 0.825 GiB |
| 合计 | 107,290 | 33.247 GiB |

Sen12Landslides 当前没有 HDF5。`../benchmark` 顶层为空。

## 实际检查

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 精确旧目标不存在 | 0 | 旧代码、配置、测试、输出和非归档过程文档均已消失 |
| HDF5 文件级只读复计 | 0 | 107,290 个文件，33.247 GiB |
| Benchmark 顶层检查 | 0 | 空 |
| archive SHA-256 清理前后对照 | 0 | 四个文件哈希完全一致 |
| `git diff -- docs/archive` | 0 | 无跟踪内容修改 |
| `git diff -- models_zoo docs/archive` | 0 | 无受保护资产修改 |
| Markdown UTF-8/BOM/fence 检查 | 0 | 4 个活动文档通过 |
| TOML 解析与无 CLI/依赖断言 | 0 | 通过 |
| 旧运行时标识扫描 | 1 | 预期无匹配 |
| `git diff --check` | 0 | 通过 |

旧运行时扫描排除了 archive、模型参考资料、新算法方案以及 README/AGENTS 中的说明性历史文字。

## 未运行

- Benchmark builder 或 validator
- 单元训练、GPU/CUDA、正式训练或评估
- 模型、Trainer、Evaluator 或 RAG
- 数据、模型或依赖下载
- commit 或 push

## 当前阻塞

无阶段 0 阻塞。

## 下一步

进入阶段 1：“多源 HDF5 数据审计与统一 Benchmark 构建”的只读审计部分。

先检查全部实际 HDF5 的 group、dataset、shape、dtype、属性、通道、mask 值域、validity、
异常值和空间对应关系，再冻结新的读取合同、split 原则、目标 patch 参数和重采样规则。
审计完成前不实现 Benchmark builder，不进入模型实现。
