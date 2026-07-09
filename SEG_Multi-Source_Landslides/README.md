# Multi-Source Qwen-PSALM-Seg Prototype

本目录是面向 `benchmark/multisource_landslide_v1_small` 的多源遥感滑坡 instruction segmentation 研究原型。当前主目标是跑通单卡 Qwen-cache 训练、验证指标、checkpoint reload、分组诊断和 mask 可视化闭环。

## 当前能力

- 读取 `instruction_train.jsonl` / `instruction_val.jsonl`，默认过滤 Phase 1 核心模板，暂不训练 referring 样本。
- 将 `optical_rgb`、`optical_multiband`、`multispectral`、`sar_asc`、`sar_dsc`、`dem`、`slope`、`insar_vel` 映射到 `hr_optical/s2/s1/dem/insar` canonical 模态槽位。
- 支持不同 H/W patch 等比例 resize + padding，metadata 和可视化 manifest 会记录 `resize_transform`。
- 输出多个 mask proposals、condition scores、最终 mask、Dice/IoU/Precision/Recall、grouped metrics 和 overlay PNG。
- 支持 Qwen3-VL frozen controller、`qwen_cache` 预计算文本 embedding 路径，以及轻量 `text_probe` 开发回归路径。
- 支持可选 `use_box_prior` 分支，把 bbox prior 作为 Qwen3-VL-Seg 风格 box-guided refinement 注入 decoder 特征。
- 默认启用 hard-combo loss weighting，加强 `s1`、`dem+s2`、`dem+s1+s2` 等上一轮验证中较弱的模态组合。

## 模型结构

`qpsalm_seg/model.py` 只保留兼容 shim，实际模型拆到 `qpsalm_seg/models/`：

- `common.py`: `ConvBlock`、MLP 等基础模块。
- `modality.py`: per-modality adapters、availability/dropout 和 condition-aware modality gating。
- `fusion.py`: 每个模态独立形成 high/mid/low 金字塔，再按 condition-aware gate 在各尺度融合，可选 bbox prior 注入。
- `decoder.py`: PSALM-style learnable mask tokens、proposal decoder、two-tower condition-aware proposal scorer。
- `qpsalm.py`: 总装 controller、modality adapters、fusion、decoder 和 loss。

数据层同时输出 `proposal_context_text`、`condition_prompt_text` 和向后兼容的 `condition_text`。模型分别编码 proposal context 与 condition prompt：前者更新 mask tokens、驱动 proposal generation；后者进入 condition-aware scorer 判断 proposal 是否匹配任务语义。这对应 PSALM 的 proposal generation 与 condition classification 解耦思想。

## 主训练入口

默认使用 `/home/yukun80/miniconda3/envs/qwen3vl/bin/python`，并假设 torch/CUDA 已由用户手动确认。完整 Phase 1 baseline 训练：

```bash
MODE=baseline \
RUN_NAME=qwen_cached_baseline_t256_b4_s64_hardcombo \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

完整脚本默认参数包括：

- `TARGET_SIZE=256`
- `BATCH_SIZE=2`
- `GRAD_ACCUM_STEPS=2`
- `MAX_STEPS=6000`
- `EMBEDDING_BATCH_SIZE=1`
- `SAMPLES_PER_COMBO=64`
- `MAX_VAL_SAMPLES=0`
- `LOG_INTERVAL=100`
- `CANONICAL_COMBO_LOSS_WEIGHTS="s1=2.5,dem+s2=2.5,dem+s1+s2=1.5"`

常用覆盖项：

```bash
MAX_STEPS=1000 BATCH_SIZE=1 GRAD_ACCUM_STEPS=4 TARGET_SIZE=192 \
RUN_CONTROL=--resume-existing \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

如需单独训练 box-prior 分支：

```bash
MODE=box-prior RUN_NAME=qwen_cached_boxprior_t256_b4_s64 \
RUN_CONTROL=--overwrite \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
```

训练结束后脚本会自动运行 `qpsalm-verify-phase1`，检查 checkpoint、eval metrics、PNG/mask exports、visualization manifest、analysis tables 和 runtime metadata。只想训练不验收时可加 `VERIFY_AFTER_RUN=0`。

## 开发回归

保留一个标准小步 smoke，用于修改模型结构后快速验证 train/eval/reload/visualization 是否被破坏：

```bash
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
```

该脚本仍可用 `hash-smoke` + CPU 只验证编排路径：

```bash
CONFIG=SEG_Multi-Source_Landslides/configs/qpsalm_tiny_text_probe.yaml \
OUTPUT_ROOT=/tmp/qpsalm_dev_smoke \
RUN_NAME=dev \
DEVICE=cpu \
CONTROLLER=qwen_cache \
EMBEDDING_BACKEND=hash-smoke \
MAX_STEPS=1 \
MAX_TRAIN_SAMPLES=4 \
MAX_VAL_SAMPLES=4 \
VERIFY_AFTER_RUN=1 \
bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
```

`hash-smoke` 只用于验证索引、cache、模块化模型、训练/eval/reload/可视化编排链路；正式实验应使用 `EMBEDDING_BACKEND=qwen` 和 `DEVICE=cuda`。

## 数据与 Cache

检查数据索引：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m qpsalm_seg.cli.inspect_data \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --limit 16 \
  --max-rows 5000
```

手动生成均衡核心索引：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m qpsalm_seg.cli.cache_index \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --output-dir outputs/qpsalm_index_cache_balanced \
  --strategy balanced-canonical \
  --samples-per-combo 64
```

手动生成 Qwen condition cache：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m qpsalm_seg.cli.cache_qwen_embeddings \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml \
  --train-index outputs/qpsalm_index_cache_balanced/qpsalm_core_train.jsonl \
  --val-index outputs/qpsalm_index_cache_balanced/qpsalm_core_val.jsonl \
  --output outputs/qpsalm_qwen_condition_cache_balanced.pt \
  --device cuda \
  --batch-size 1 \
  --backend qwen \
  --overwrite
```

训练/eval 会检查 cache coverage：按当前 train/val index 派生 `condition_text`、`proposal_context_text` 和 `condition_prompt_text`，确认它们都存在于 `condition_embedding_cache`。

## Eval 与诊断

checkpoint reload eval：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m qpsalm_seg.cli.eval \
  --config outputs/qpsalm_phase1/qwen_cached_baseline_t256_b4_s64_hardcombo/baseline/resolved_config.yaml \
  --checkpoint outputs/qpsalm_phase1/qwen_cached_baseline_t256_b4_s64_hardcombo/baseline/checkpoint_best.pt \
  --device cuda \
  --output-dir outputs/qpsalm_phase1/qwen_cached_baseline_t256_b4_s64_hardcombo/baseline_eval_reload \
  --skip-torch-preflight
```

推荐训练后查看：

- `validation_best.json`: overall、raw combo、canonical combo、condition metrics。
- `analysis_tables/metrics.csv`: 分组 Dice/IoU/Precision/Recall。
- `analysis_tables/modality_gates.csv`: 不同 condition 下 DEM/SAR/InSAR 等证据权重。
- `analysis_tables/proposal_diagnostics.csv`: selected query 与 supervised best query 是否一致。
- `threshold_recommendations.json`: overall 和 per-combo 阈值建议。
- `visualizations/*/visualization_manifest.jsonl`: condition、raw/canonical modality combo、resize transform、mask 路径。

## 代码检查

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -B -m py_compile \
  SEG_Multi-Source_Landslides/qpsalm_seg/*.py \
  SEG_Multi-Source_Landslides/qpsalm_seg/models/*.py \
  SEG_Multi-Source_Landslides/qpsalm_seg/cli/*.py

bash -n SEG_Multi-Source_Landslides/scripts/*.sh
```
