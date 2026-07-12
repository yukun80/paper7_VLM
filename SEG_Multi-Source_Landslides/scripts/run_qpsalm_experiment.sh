#!/usr/bin/env bash
# 用途：准备 Qwen vision cache v3，并运行 benchmark-v2 正式训练与验证。
# 推荐命令：BENCHMARK_SIZE=small PRESET=qwen_psalm_full RUN_NAME=small_qwen \
#   RUN_CONTROL=--overwrite bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
# 输入：multisource_landslide_v2_{small,full}、本地 Qwen3-VL 权重和 Python preset。
# 输出：outputs/qpsalm_v2/<RUN_NAME> 下的 cache、best/last checkpoint、eval 和可视化。
# 写入：只写 outputs；--overwrite 会覆盖同名 cache 和运行目录。
# 前置：完成 benchmark-v2 与 instruction 索引构建。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-small}"
PRESET="${PRESET:-qwen_psalm_full}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-${PRESET}_${BENCHMARK_SIZE}_seed${SEED}}"
RUN_CONTROL="${RUN_CONTROL:-}"

case "${BENCHMARK_SIZE}" in
  small)
    CONFIG="SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml"
    DEFAULT_BENCHMARK_DIR="benchmark/multisource_landslide_v2_small"
    ;;
  full)
    CONFIG="SEG_Multi-Source_Landslides/configs/qpsalm_v2_full.yaml"
    DEFAULT_BENCHMARK_DIR="benchmark/multisource_landslide_v2_full"
    ;;
  *) echo "BENCHMARK_SIZE must be small or full" >&2; exit 2 ;;
esac
BENCHMARK_DIR="${BENCHMARK_DIR:-${DEFAULT_BENCHMARK_DIR}}"
case "${RUN_CONTROL}" in
  ""|--overwrite|--resume-existing) ;;
  *) echo "RUN_CONTROL must be empty, --overwrite, or --resume-existing" >&2; exit 2 ;;
esac

export PYTHONPATH="SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="outputs/qpsalm_v2/${RUN_NAME}"
CACHE_DIR="outputs/qpsalm_v2/cache/${BENCHMARK_SIZE}_${PRESET}_qwen_vision_v3"
EVAL_DIR="${RUN_DIR}/eval_val"
VISION_ARGS=()

echo "benchmark=${BENCHMARK_DIR} preset=${PRESET} seed=${SEED} run=${RUN_DIR}"

if [[ "${PRESET}" == "pretrained_sane_qmef_pmrd" || "${PRESET}" == "qwen_psalm_full" ]]; then
  if [[ "${RUN_CONTROL}" == "--overwrite" || ! -f "${CACHE_DIR}/manifest.json" ]]; then
    "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_vision_features \
      --config "${CONFIG}" --benchmark-dir "${BENCHMARK_DIR}" \
      --preset "${PRESET}" \
      --output-dir "${CACHE_DIR}" --device "${DEVICE}" --backend qwen --overwrite
  else
    "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_vision_features \
      --config "${CONFIG}" --preset "${PRESET}" \
      --output-dir "${CACHE_DIR}" --verify-only
  fi
  VISION_ARGS+=(--vision-feature-cache "${CACHE_DIR}")
fi

TRAIN_CONTROL=()
RESUME_ARGS=()
if [[ "${RUN_CONTROL}" == "--overwrite" ]]; then
  TRAIN_CONTROL+=(--overwrite-output)
elif [[ "${RUN_CONTROL}" == "--resume-existing" && -f "${RUN_DIR}/checkpoint_last.pt" ]]; then
  RESUME_ARGS+=(--resume "${RUN_DIR}/checkpoint_last.pt")
fi

"${PYTHON_BIN}" -m qpsalm_seg.cli.train \
  --config "${CONFIG}" --preset "${PRESET}" --benchmark-dir "${BENCHMARK_DIR}" \
  --device "${DEVICE}" --seed "${SEED}" "${VISION_ARGS[@]}" \
  --output-dir "${RUN_DIR}" --skip-torch-preflight \
  "${TRAIN_CONTROL[@]}" "${RESUME_ARGS[@]}"

CHECKPOINT="${RUN_DIR}/checkpoint_best.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then CHECKPOINT="${RUN_DIR}/checkpoint_last.pt"; fi

EVAL_CONTROL=()
if [[ "${RUN_CONTROL}" == "--overwrite" ]]; then EVAL_CONTROL+=(--overwrite-output); fi
"${PYTHON_BIN}" -m qpsalm_seg.cli.eval \
  --config "${CONFIG}" --preset "${PRESET}" --benchmark-dir "${BENCHMARK_DIR}" \
  --checkpoint "${CHECKPOINT}" --device "${DEVICE}" --seed "${SEED}" \
  "${VISION_ARGS[@]}" --output-dir "${EVAL_DIR}" \
  --export-multimodal-overview --skip-torch-preflight "${EVAL_CONTROL[@]}"
