#!/usr/bin/env bash
# 用途：用 hash text cache 验证重构模型的最小训练、评估、checkpoint reload 和汇总闭环。
# 推荐运行命令：bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
# 主要输入：small benchmark instruction 索引和 dev_smoke preset；不加载真实 Qwen 权重。
# 主要输出：outputs/qpsalm_refactor_smoke 下的 cache、checkpoint、报告和 4 组可视化。
# 写入行为：写 OUTPUT_DIR；训练和评估子目录采用 overwrite 模式。
# 所属流程：开发回归入口，不作为正式精度实验结果。
# 可选覆盖：DEVICE、MAX_STEPS、OUTPUT_DIR、CONFIG、PYTHON_BIN。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
DEVICE="${DEVICE:-cpu}"
MAX_STEPS="${MAX_STEPS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qpsalm_refactor_smoke}"
CONFIG="${CONFIG:-SEG_Multi-Source_Landslides/configs/qpsalm_tiny_text_probe.yaml}"
CONDITION_CACHE="${CONDITION_CACHE:-${OUTPUT_DIR}/condition_cache.pt}"
export PYTHONPATH="SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_embeddings \
  --config "${CONFIG}" \
  --output "${CONDITION_CACHE}" \
  --backend hash-smoke \
  --hash-hidden-size 64 \
  --overwrite

"${PYTHON_BIN}" -m qpsalm_seg.cli.train \
  --config "${CONFIG}" \
  --preset dev_smoke \
  --device "${DEVICE}" \
  --controller qwen_cache \
  --condition-embedding-cache "${CONDITION_CACHE}" \
  --max-steps "${MAX_STEPS}" \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --max-val-batches 4 \
  --batch-size 1 \
  --num-workers 0 \
  --val-interval "${MAX_STEPS}" \
  --save-interval "${MAX_STEPS}" \
  --num-visualizations 4 \
  --output-dir "${OUTPUT_DIR}/train" \
  --overwrite-output \
  --skip-torch-preflight

"${PYTHON_BIN}" -m qpsalm_seg.cli.eval \
  --config "${CONFIG}" \
  --preset dev_smoke \
  --checkpoint "${OUTPUT_DIR}/train/checkpoint_last.pt" \
  --device "${DEVICE}" \
  --controller qwen_cache \
  --condition-embedding-cache "${CONDITION_CACHE}" \
  --max-val-samples 4 \
  --max-val-batches 4 \
  --batch-size 1 \
  --num-workers 0 \
  --num-visualizations 4 \
  --output-dir "${OUTPUT_DIR}/eval" \
  --overwrite-output \
  --skip-torch-preflight

"${PYTHON_BIN}" -m qpsalm_seg.cli.summarize_run \
  --run-dir "${OUTPUT_DIR}/train" \
  --eval-dir "${OUTPUT_DIR}/eval" \
  --min-visualizations 4
