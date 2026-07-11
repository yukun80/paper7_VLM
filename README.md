# Multi-Source Qwen-PSALM-Seg

本仓库用于构建多源遥感滑坡 instruction segmentation benchmark，并研究可处理任意模态组合、不同空间分辨率和缺失模态的 Multi-Source Qwen-PSALM-Seg。当前主模型采用 **SANE -> QMEF -> PMRD** 架构，Qwen3-VL 作为冻结的 semantic/evidence controller，不生成 bbox，也不承担密集像素编码。

本 README 是仓库唯一的完整运行手册。算法细节见 [SEG_Multi-Source_Landslides/ALGORITHM.md](SEG_Multi-Source_Landslides/ALGORITHM.md)，研究任务说明见 [docs/Task_Introduction.md](docs/Task_Introduction.md)。

## 1. 环境与目录约定

所有命令均从 `paper7_VLM` 仓库根目录执行，并假设已经激活 `qwen3vl` 环境：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
```

当前大体量数据目录位于仓库同级：

```text
/home/yukun80/codes/
├── datasets/                         # 原始数据
├── benchmark/                        # 派生 benchmark
└── paper7_VLM/
    ├── SEG_Multi-Source_Landslides/  # 模型、训练、评估和实验配置
    ├── scripts/                       # benchmark 与 instruction 数据流程
    ├── configs/                       # instruction 模板
    ├── models_zoo/                    # 本地 Qwen3-VL 权重
    ├── outputs/                       # 训练和推理产物
    ├── external/                      # 第三方参考实现
    └── docs/                          # 研究文档
```

JSONL 和 YAML 中继续使用可移植逻辑路径 `datasets/...`、`benchmark/...`。运行时默认映射到仓库同级目录，也可覆盖：

```bash
export PAPER7_DATASETS_ROOT=/path/to/datasets
export PAPER7_BENCHMARK_ROOT=/path/to/benchmark
```

QPSALM CLI 推荐直接使用模块入口，无需安装包：

```bash
export PYTHONPATH=SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}
python -m qpsalm_seg.cli.inspect_data --help
```

也可以执行可编辑安装后使用短命令：

```bash
python -m pip install -e SEG_Multi-Source_Landslides
qpsalm-inspect-data --help
```

## 2. 构建 Benchmark

### 2.1 推荐总控命令

Small benchmark 用于开发和固定 split 实验：

```bash
bash scripts/run_1_build_benchmark.sh small
```

Full benchmark 用于完整训练：

```bash
bash scripts/run_1_build_benchmark.sh full
```

连续构建 small 和 full：

```bash
bash scripts/run_1_build_benchmark.sh both
```

Small 模式默认每个 `dataset_name + split` 最多取 1000 条，可临时降低：

```bash
SMALL_LIMIT=100 bash scripts/run_1_build_benchmark.sh small
```

脚本不会改写 `datasets/`，主要输出为 `benchmark/multisource_landslide_v1_{small,full}` 下的物化数组、统一索引、验证报告和统计报告。

### 2.2 独立阶段程序

通常不需要逐个运行这些程序；它们用于重新执行或调试某个 benchmark 阶段。

| 阶段 | 程序 | 用途 | 主要输出 |
| --- | --- | --- | --- |
| 1-1 | `1-1_scan_sources.py` | 扫描原始数据和格式 | `source_manifest.csv`、`dataset_inventory.json` |
| 1-2 | `1-2_build_index.py` | 构建统一 source JSONL | `indexes/source_*.jsonl` |
| 1-3 | `1-3_validate_index.py` | 验证 source/final/referring 索引 | `reports/validation_report*.json` |
| 1-4 | `1-4_preprocess_samples.py` | 物化模态和 mask 数组 | `data/**`、`indexes/all.jsonl` |
| 1-5 | `1-5_build_splits.py` | 生成最终 split 和采样权重 | `indexes/{train,val,test,unlabeled}.jsonl` |
| 1-6 | `1-6_build_referring_targets.py` | 构建派生 referring targets | `referring_target_*.jsonl`、target masks |
| 1-7 | `1-7_summarize_benchmark.py` | 汇总数据质量和分布 | `statistics.json`、`cleaning_report.md` |

```bash
python scripts/1-benchmark/1-1_scan_sources.py \
  --datasets-root datasets \
  --out-dir benchmark/multisource_landslide_v1_small

python scripts/1-benchmark/1-2_build_index.py \
  --mode small --small-limit 1000 --seed 42 \
  --datasets-root datasets \
  --out-dir benchmark/multisource_landslide_v1_small

python scripts/1-benchmark/1-3_validate_index.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small --stage source

python scripts/1-benchmark/1-4_preprocess_samples.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small --strategy materialize

python scripts/1-benchmark/1-3_validate_index.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small --stage final

python scripts/1-benchmark/1-5_build_splits.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small

python scripts/1-benchmark/1-6_build_referring_targets.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small

python scripts/1-benchmark/1-3_validate_index.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small --stage referring_target

python scripts/1-benchmark/1-7_summarize_benchmark.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small
```

## 3. 构建 Instruction 数据

必须先完成同一规模的 benchmark。

```bash
bash scripts/run_2_build_instruction_dataset.sh small
bash scripts/run_2_build_instruction_dataset.sh full
# 或连续处理两种规模
bash scripts/run_2_build_instruction_dataset.sh both
```

主要输出为 benchmark 目录内的 `indexes/instruction_*.jsonl` 和 `reports/instruction_*.json`。独立阶段命令如下：

```bash
python scripts/2-instruction/2-1_build_instruction_templates.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small \
  --template-config configs/instruction_templates/multisource_landslide_v1.yaml

python scripts/2-instruction/2-2_apply_instruction_templates.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small \
  --template-config configs/instruction_templates/multisource_landslide_v1.yaml

python scripts/2-instruction/2-3_validate_instruction_index.py \
  --benchmark-dir benchmark/multisource_landslide_v1_small \
  --template-config configs/instruction_templates/multisource_landslide_v1.yaml
```

## 4. 数据检查与核心索引缓存

检查 instruction 数据、模态组合、sensor、normalization 和 GSD：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.inspect_data \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --split train --limit 16
```

生成保留全部核心样本、按 canonical combo 交错排序的 train/val 缓存：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_index \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --benchmark-dir benchmark/multisource_landslide_v1_small \
  --output-dir outputs/qpsalm_index_cache_small \
  --split both --strategy round-robin-canonical
```

只缓存完整多模态 test 样本：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_index \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml \
  --benchmark-dir benchmark/multisource_landslide_v1_full \
  --output-dir outputs/qpsalm_eval_indices/full_multimodal_test \
  --split test --strategy round-robin-canonical --require-multimodal
```

## 5. Smoke 回归

Smoke 使用 hash text cache，不加载真实 Qwen 权重，只验证数据、模型、loss、checkpoint reload、评估和可视化闭环：

```bash
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
```

可选覆盖：

```bash
DEVICE=cpu MAX_STEPS=5 OUTPUT_DIR=outputs/qpsalm_refactor_smoke \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
```

## 6. 正式训练

正式入口会依次生成核心索引、Qwen 文本 cache、可选多视图 cache，训练模型，reload checkpoint 评估，并输出 summary 与 diagnose report。

### 6.1 Small 消融顺序

```bash
BENCHMARK_SIZE=small PRESET=sane_baseline RUN_NAME=sane_baseline_small \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh

BENCHMARK_SIZE=small PRESET=sane_qmef RUN_NAME=sane_qmef_small \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh

BENCHMARK_SIZE=small PRESET=sane_qmef_pmrd RUN_NAME=sane_qmef_pmrd_small \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh

BENCHMARK_SIZE=small PRESET=full_multiview RUN_NAME=full_multiview_small \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

Preset 含义：

| Preset | 研究能力 |
| --- | --- |
| `sane_baseline` | Sensor-Aware Native-Scale Encoder 基线 |
| `sane_qmef` | 增加多源可靠性 prior 与 query-spatial attention |
| `sane_qmef_pmrd` | 增加 proposal set、统一 verifier 和两轮 refinement |
| `full_multiview` | 在主线模型中接入 Qwen 多视图视觉证据 cache |
| `dev_smoke` | 仅用于开发回归，不用于正式精度结论 |

### 6.2 Full 训练

只建议将通过 small 固定 split 验证的 preset 用于 full benchmark：

```bash
BENCHMARK_SIZE=full PRESET=sane_qmef_pmrd RUN_NAME=sane_qmef_pmrd_full \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

多视图 full 训练：

```bash
BENCHMARK_SIZE=full PRESET=full_multiview RUN_NAME=full_multiview_full \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

续训已有运行：

```bash
BENCHMARK_SIZE=full PRESET=sane_qmef_pmrd RUN_NAME=sane_qmef_pmrd_full \
RUN_CONTROL=--resume-existing \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

模型、batch、epoch、loss 和 optimizer 参数由 YAML 与 `qpsalm_seg/presets.py` 管理。Shell 只负责 benchmark 规模、preset、run name、device、路径和覆盖/续训控制。

### 6.3 直接调用训练 CLI

需要精细覆盖单次实验参数时可绕过总控 Shell：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --preset sane_qmef_pmrd \
  --benchmark-dir benchmark/multisource_landslide_v1_small \
  --controller qwen_cache \
  --condition-embedding-cache outputs/RUN/condition_cache.pt \
  --train-index outputs/RUN/index_cache/qpsalm_core_train.jsonl \
  --val-index outputs/RUN/index_cache/qpsalm_core_val.jsonl \
  --output-dir outputs/RUN/train \
  --device cuda --skip-torch-preflight --overwrite-output
```

## 7. Qwen 文本与多视图缓存

### 7.1 文本 cache

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_qwen_embeddings \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --train-index outputs/RUN/index_cache/qpsalm_core_train.jsonl \
  --val-index outputs/RUN/index_cache/qpsalm_core_val.jsonl \
  --output outputs/RUN/condition_cache.pt \
  --backend qwen --device cuda --overwrite
```

检查文本覆盖率：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.check_qwen_cache \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --train-index outputs/RUN/index_cache/qpsalm_core_train.jsonl \
  --val-index outputs/RUN/index_cache/qpsalm_core_val.jsonl \
  --condition-embedding-cache outputs/RUN/condition_cache.pt
```

### 7.2 Qwen 多视图视觉 cache

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_qwen_visual_evidence \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --train-index outputs/RUN/index_cache/qpsalm_core_train.jsonl \
  --val-index outputs/RUN/index_cache/qpsalm_core_val.jsonl \
  --output outputs/RUN/multiview_cache_v2.pt \
  --backend qwen --device cuda --pooling-method vision-token --overwrite
```

真实性消融使用相同入口：

```bash
# 跨样本打乱 view
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.cache_qwen_visual_evidence \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --output outputs/qpsalm_multiview_shuffled.pt --backend qwen --device cuda \
  --shuffle-views-across-samples --overwrite

# 移除 SAR view
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.cache_qwen_visual_evidence \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --output outputs/qpsalm_multiview_no_sar.pt --backend qwen --device cuda \
  --drop-view-pattern sar --overwrite
```

## 8. Val/Test 推理与可视化

### 8.1 Val 推理

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml \
  --preset sane_qmef_pmrd \
  --checkpoint outputs/RUN/train/checkpoint_best.pt \
  --split val \
  --val-index outputs/RUN/index_cache/qpsalm_core_val.jsonl \
  --condition-embedding-cache outputs/RUN/condition_cache.pt \
  --output-dir outputs/RUN/eval_val \
  --device cuda --skip-torch-preflight --overwrite-output
```

### 8.2 完整多模态 Test 推理

第一步，生成多模态 test index：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_index \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml \
  --benchmark-dir benchmark/multisource_landslide_v1_full \
  --split test \
  --output-dir outputs/qpsalm_eval_indices/full_multimodal_test \
  --strategy round-robin-canonical --require-multimodal
```

第二步，为 test prompt 建立文本 cache：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_qwen_embeddings \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml \
  --eval-index outputs/qpsalm_eval_indices/full_multimodal_test/qpsalm_core_test.jsonl \
  --eval-split test \
  --output outputs/qpsalm_eval_indices/full_multimodal_test/condition_cache.pt \
  --backend qwen --device cuda --overwrite
```

第三步，对全部样本推理并为每个样本生成一张多模态总览：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml \
  --preset sane_qmef_pmrd \
  --checkpoint outputs/RUN/train/checkpoint_best.pt \
  --split test \
  --test-index outputs/qpsalm_eval_indices/full_multimodal_test/qpsalm_core_test.jsonl \
  --condition-embedding-cache outputs/qpsalm_eval_indices/full_multimodal_test/condition_cache.pt \
  --output-dir outputs/RUN/eval_multimodal_test_full \
  --max-val-samples 0 --visualize-all --export-multimodal-overview \
  --device cuda --skip-torch-preflight --overwrite-output
```

使用 `full_multiview` checkpoint 时，还需为 test index 建立 visual cache，并向 eval 传入 `--visual-evidence-cache`：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_qwen_visual_evidence \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml \
  --eval-index outputs/qpsalm_eval_indices/full_multimodal_test/qpsalm_core_test.jsonl \
  --output outputs/qpsalm_eval_indices/full_multimodal_test/multiview_cache_v2.pt \
  --backend qwen --device cuda --overwrite
```

## 9. 结果汇总、比较与诊断

汇总训练和 eval 产物：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.summarize_run \
  --run-dir outputs/RUN/train --eval-dir outputs/RUN/eval_val
```

比较两个 preset：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.compare_runs \
  --baseline-summary outputs/sane_baseline/train \
  --candidate-summary outputs/sane_qmef_pmrd/train \
  --output outputs/preset_comparison.json
```

诊断低精度、query selection 和模态 attention：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.diagnose_run \
  --run outputs/RUN/train/run_summary.json \
  --output outputs/RUN/train/diagnose_report.json
```

推荐二值化阈值：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.recommend_threshold \
  --run outputs/RUN/eval_val \
  --output outputs/RUN/eval_val/threshold_recommendations.json
```

导出指标、QMEF 和 PMRD 诊断表：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.export_tables \
  --input outputs/RUN/eval_val/eval_report.json \
  --output-dir outputs/RUN/eval_val/tables
```

## 10. 全部 QPSALM CLI 索引

`SEG_Multi-Source_Landslides/pyproject.toml` 注册了以下 13 个命令。表中的模块命令无需安装包即可运行；短命令需要先执行 `python -m pip install -e SEG_Multi-Source_Landslides`。

| 短命令 | Python 模块 | 用途 |
| --- | --- | --- |
| `qpsalm-inspect-data` | `qpsalm_seg.cli.inspect_data` | 检查 instruction 数据和模态分布 |
| `qpsalm-cache-index` | `qpsalm_seg.cli.cache_index` | 缓存核心 train/val/test 索引 |
| `qpsalm-check-env` | `qpsalm_seg.cli.check_env` | 可选环境诊断 |
| `qpsalm-cache-qwen-embeddings` | `qpsalm_seg.cli.cache_qwen_embeddings` | 缓存 Qwen 文本证据 |
| `qpsalm-cache-qwen-visual-evidence` | `qpsalm_seg.cli.cache_qwen_visual_evidence` | 缓存 Qwen 多视图视觉证据 |
| `qpsalm-check-qwen-cache` | `qpsalm_seg.cli.check_qwen_cache` | 检查文本 cache 覆盖率 |
| `qpsalm-train` | `qpsalm_seg.cli.train` | 直接训练 SANE/QMEF/PMRD |
| `qpsalm-eval` | `qpsalm_seg.cli.eval` | val/test 推理、指标和可视化 |
| `qpsalm-summarize-run` | `qpsalm_seg.cli.summarize_run` | 汇总训练与 eval 产物 |
| `qpsalm-compare-runs` | `qpsalm_seg.cli.compare_runs` | 比较两个实验运行 |
| `qpsalm-diagnose-run` | `qpsalm_seg.cli.diagnose_run` | 诊断精度、proposal 和模态证据 |
| `qpsalm-recommend-threshold` | `qpsalm_seg.cli.recommend_threshold` | 推荐 mask 二值化阈值 |
| `qpsalm-export-tables` | `qpsalm_seg.cli.export_tables` | 导出指标和算法诊断 CSV |

统一查看帮助：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.train --help
# 或完成可编辑安装后
qpsalm-train --help
```

## 11. 测试与静态检查

运行全部单元测试：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m unittest discover -s SEG_Multi-Source_Landslides/tests -p 'test_*.py' -v
```

分别运行核心模型和 renderer 测试：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m unittest SEG_Multi-Source_Landslides/tests/test_refactor_core.py -v

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m unittest SEG_Multi-Source_Landslides/tests/test_renderer.py -v
```

Python 语法检查：

```bash
python -B -m py_compile \
  scripts/1-benchmark/*.py \
  scripts/2-instruction/*.py \
  SEG_Multi-Source_Landslides/qpsalm_seg/*.py \
  SEG_Multi-Source_Landslides/qpsalm_seg/models/*.py \
  SEG_Multi-Source_Landslides/qpsalm_seg/cli/*.py
```

Shell 语法检查：

```bash
bash -n scripts/run_1_build_benchmark.sh
bash -n scripts/run_2_build_instruction_dataset.sh
bash -n SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
bash -n SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
```

可选环境诊断，不属于正常训练前置步骤：

```bash
python env_test.py

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.check_env \
  --benchmark-dir benchmark/multisource_landslide_v1_small
```

## 12. 主要输出

正式训练默认只维护 `checkpoint_best.pt` 和 `checkpoint_last.pt`。主要产物包括：

- `validation_best.json` / `eval_report.json`：canvas 与原尺寸指标，包括 overall、positive-only、negative accuracy 和 empty false-positive rate。
- `proposal_diagnostics.csv`：PMRD Dice-best query 与 relevance-top query 的差异。
- `modality_reliability.csv`：QMEF 样本级模态可靠性 prior。
- `query_modality_attention.csv`：每个 proposal 的跨模态证据注意力。
- `visualization_manifest.jsonl`：模态元数据、final mask、best proposal 和原尺寸恢复路径。
- `run_summary.json` / `diagnose_report.json`：实验完整性汇总和低精度诊断。

大型数据、benchmark、模型权重、checkpoint、日志和第三方仓库不应提交到 Git。
