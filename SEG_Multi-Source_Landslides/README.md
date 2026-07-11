# Multi-Source Qwen-PSALM-Seg

面向异构多源遥感滑坡指令分割的单卡研究原型。主模型已重构为 **SANE -> QMEF -> PMRD**，研究范围限定为单时相或同期多源证据分割；Qwen3-VL 作为冻结的 semantic/evidence controller，不生成 bbox，也不承担密集像素编码。

完整运行手册、全部 CLI、benchmark/instruction 构建、训练、val/test 推理和诊断命令统一维护在仓库根目录 [README.md](../README.md)。本文件只说明模型架构、preset 和核心训练约定。

索引中的 `datasets/...` 与 `benchmark/...` 是逻辑路径，默认解析到仓库同级的 `../datasets`、`../benchmark`。可使用 `PAPER7_DATASETS_ROOT`、`PAPER7_BENCHMARK_ROOT` 或 CLI `--benchmark-dir` 覆盖，无需改写 JSONL。

## 架构

- **SANE** (`models/sane.py`)：逐波段共享 stem，注入 modality family、sensor、band、orbit、GSD 与 quality embedding；每个模态独立产生 1/4、1/8、1/16 原生尺度特征，不截断多光谱波段。
- **QMEF** (`models/qmef.py`)：用 GSD-aware `grid_sample` 聚合器对齐模态；仅保留样本级可靠性 prior 和 query-spatial cross-modal attention。
- **PMRD** (`models/pmrd.py`)：PSALM-style mask tokens 生成 proposal set；由统一 semantic-evidence verifier 给 proposal 打 relevance 分，再通过 mask-aware 区域证据进行第二轮细化和 relevance-gated union。
- **Semantic evidence**：task、condition、evidence reasoning 与可选 Qwen 多视图 token 通过 attention 形成统一证据对象，不再使用 condition/evidence/visual 三套独立 scorer。

数据接口由 `schema.py` 中的 `ModalityInstance`、`ModalityBatch`、`MultiScaleFeatures`、`SemanticEvidence`、`ProposalSet` 和 `SegmentationOutput` 定义。resize/pad 产生的 `valid_mask` 会贯穿 loss、proposal matching、metrics 和可视化恢复。

## Preset

算法参数由 `qpsalm_seg/presets.py` 统一管理：

- `sane_baseline`：单 query 的空间编码基线。
- `sane_qmef`：增加多源可靠性和 query-spatial attention。
- `sane_qmef_pmrd`：增加 proposal set、Hungarian/coverage supervision 和两轮细化，默认主线。
- `full_multiview`：在主线基础上接入 Qwen visual cache v2。
- `dev_smoke`：仅用于快速回归，不作为正式实验结果。

## 核心训练约定

Small 主线训练：

```bash
PRESET=sane_qmef_pmrd \
BENCHMARK_SIZE=small \
RUN_NAME=sane_qmef_pmrd_small \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

Full benchmark 将 `BENCHMARK_SIZE` 改为 `full`。Qwen 多视图版本将 `PRESET` 改为 `full_multiview`。脚本只管理 benchmark、preset、run name、device 和运行目录；模型、epoch/batch 与 loss 参数不再散落在 shell 环境变量中。small/full YAML 默认分别按 10/5 epochs 解析 optimizer steps，启动日志会打印 `steps_per_epoch` 和 `estimated_epochs`。默认只维护 `checkpoint_best.pt` 与 `checkpoint_last.pt`，step checkpoint 和逐 step validation report 均需在 Python 配置中显式启用。

Qwen 多视图真实性对照仍由 visual cache CLI 提供：`--shuffle-views-across-samples` 打乱父样本，`--drop-view-pattern` 移除指定视图，`--pooling-method` 选择 token pooling。具体命令见根 README。

## 评估产物

训练脚本会 reload best/last checkpoint 并运行 eval。重点产物包括：

- `validation_best.json` / `eval_report.json`：canvas 与原尺寸 Dice、IoU、Precision、Recall，positive-only、negative accuracy、empty false-positive rate 及模态组合分组。
- `proposal_diagnostics.csv`：Dice-best query、relevance-top query、rank 与 score gap。
- `modality_reliability.csv`：QMEF 样本级可靠性 prior。
- `query_modality_attention.csv`：每个 proposal 的跨模态证据注意力。
- `visualization_manifest.jsonl`：多模态元数据、最终 mask、best proposal 和原尺寸恢复路径。

回归测试、静态检查、独立训练 CLI、val/test 推理、完整多模态 overview、结果比较和诊断命令均见根 README，避免在两个文档中重复维护。
