#!/usr/bin/env bash
# 用途：使用 text-probe 在 benchmark-v2 small 上执行 5-step CPU/GPU 回归。
# 推荐命令：bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_smoke.sh
# 输入：small v2 instruction train/val；不加载 Qwen 权重或视觉缓存。
# 输出：outputs/qpsalm_v2/smoke 下的 checkpoint、eval report 和可视化。
# 写入：覆盖 smoke 输出目录，不修改 benchmark。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
DEVICE="${DEVICE:-cpu}"
CONFIG="SEG_Multi-Source_Landslides/configs/qpsalm_v2_smoke.yaml"
OUTPUT_DIR="outputs/qpsalm_v2/smoke"
export PYTHONPATH="SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m qpsalm_seg.cli.train \
  --config "${CONFIG}" --preset raw_sane_qmef_pmrd --controller text_probe \
  --device "${DEVICE}" --output-dir "${OUTPUT_DIR}/train" \
  --overwrite-output --skip-torch-preflight

"${PYTHON_BIN}" -m qpsalm_seg.cli.eval \
  --config "${CONFIG}" --preset raw_sane_qmef_pmrd --controller text_probe \
  --checkpoint "${OUTPUT_DIR}/train/checkpoint_last.pt" --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}/eval" --overwrite-output --skip-torch-preflight
