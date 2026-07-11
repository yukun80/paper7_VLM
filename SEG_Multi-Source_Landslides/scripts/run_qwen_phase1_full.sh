#!/usr/bin/env bash
# 用途：构建核心索引与 Qwen cache，并训练、评估、汇总和诊断 SANE-QMEF-PMRD。
# 推荐运行命令：BENCHMARK_SIZE=small PRESET=sane_qmef_pmrd RUN_NAME=small_main \
#   RUN_CONTROL=--overwrite bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
# Full 示例：BENCHMARK_SIZE=full PRESET=sane_qmef_pmrd RUN_NAME=full_main \
#   RUN_CONTROL=--overwrite bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_full.sh
# 主要输入：instruction train/val 索引、本地 Qwen3-VL 权重、YAML runtime 配置和 Python preset。
# 主要输出：核心索引、Qwen cache、best/last checkpoint、eval report、分析表和可视化。
# 写入行为：写 outputs/qpsalm_refactor；RUN_CONTROL=--overwrite 会覆盖同名运行产物。
# 所属流程：主模型正式实验入口；应先完成 benchmark 与 instruction 数据构建。
# 注意事项：算法与训练规模来自 Python preset + YAML，本脚本只编排数据、设备和运行目录。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-small}"
PRESET="${PRESET:-sane_qmef_pmrd}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-${PRESET}_${BENCHMARK_SIZE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qpsalm_refactor}"
RUN_CONTROL="${RUN_CONTROL:-}"

case "${BENCHMARK_SIZE}" in
  small)
    CONFIG="${CONFIG:-SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml}"
    BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark/multisource_landslide_v1_small}"
    ;;
  full)
    CONFIG="${CONFIG:-SEG_Multi-Source_Landslides/configs/qpsalm_full_qwen_cached_core.yaml}"
    BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark/multisource_landslide_v1_full}"
    ;;
  *)
    echo "Unsupported BENCHMARK_SIZE=${BENCHMARK_SIZE}; expected small or full" >&2
    exit 2
    ;;
esac

case "${RUN_CONTROL}" in
  ""|--overwrite|--resume-existing) ;;
  *) echo "Unsupported RUN_CONTROL=${RUN_CONTROL}" >&2; exit 2 ;;
esac

DEFAULT_BENCHMARK_ROOT="${WORKSPACE_ROOT}/benchmark"
if [[ ! -d "${DEFAULT_BENCHMARK_ROOT}" && -d "${REPO_ROOT}/benchmark" ]]; then
  DEFAULT_BENCHMARK_ROOT="${REPO_ROOT}/benchmark"
fi
export PAPER7_BENCHMARK_ROOT="${PAPER7_BENCHMARK_ROOT:-${DEFAULT_BENCHMARK_ROOT}}"
export PYTHONPATH="SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
INDEX_DIR="${RUN_ROOT}/index_cache"
TRAIN_INDEX="${INDEX_DIR}/qpsalm_core_train.jsonl"
VAL_INDEX="${INDEX_DIR}/qpsalm_core_val.jsonl"
CONDITION_CACHE="${RUN_ROOT}/condition_cache.pt"
VISUAL_CACHE="${RUN_ROOT}/multiview_cache_v2.pt"
TRAIN_DIR="${RUN_ROOT}/${PRESET}"
EVAL_DIR="${RUN_ROOT}/${PRESET}_eval"

echo "benchmark_dir=${BENCHMARK_DIR}"
echo "benchmark_root=${PAPER7_BENCHMARK_ROOT}"
echo "config=${CONFIG} preset=${PRESET} run_root=${RUN_ROOT}"

if [[ "${RUN_CONTROL}" == "" && -f "${TRAIN_DIR}/checkpoint_last.pt" ]]; then
  echo "Run already exists: ${TRAIN_DIR}; set RUN_CONTROL=--overwrite or --resume-existing" >&2
  exit 2
fi

if [[ "${RUN_CONTROL}" == "--overwrite" || ! -f "${TRAIN_INDEX}" || ! -f "${VAL_INDEX}" ]]; then
  "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_index \
    --config "${CONFIG}" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --output-dir "${INDEX_DIR}" \
    --split both \
    --strategy round-robin-canonical
fi

if [[ "${RUN_CONTROL}" == "--overwrite" || ! -f "${CONDITION_CACHE}" ]]; then
  "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_embeddings \
    --config "${CONFIG}" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --train-index "${TRAIN_INDEX}" \
    --val-index "${VAL_INDEX}" \
    --output "${CONDITION_CACHE}" \
    --device "${DEVICE}" \
    --backend qwen \
    --overwrite
fi

VISUAL_ARGS=()
if [[ "${PRESET}" == "full_multiview" ]]; then
  if [[ "${RUN_CONTROL}" == "--overwrite" || ! -f "${VISUAL_CACHE}" ]]; then
    "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_visual_evidence \
      --config "${CONFIG}" \
      --benchmark-dir "${BENCHMARK_DIR}" \
      --train-index "${TRAIN_INDEX}" \
      --val-index "${VAL_INDEX}" \
      --output "${VISUAL_CACHE}" \
      --device "${DEVICE}" \
      --backend qwen \
      --overwrite
  fi
  VISUAL_ARGS+=(--visual-evidence-cache "${VISUAL_CACHE}")
fi

TRAIN_CONTROL=()
RESUME_ARGS=()
if [[ "${RUN_CONTROL}" == "--overwrite" ]]; then
  TRAIN_CONTROL+=(--overwrite-output)
elif [[ "${RUN_CONTROL}" == "--resume-existing" && -f "${TRAIN_DIR}/checkpoint_last.pt" ]]; then
  RESUME_ARGS+=(--resume "${TRAIN_DIR}/checkpoint_last.pt")
fi

"${PYTHON_BIN}" -m qpsalm_seg.cli.train \
  --config "${CONFIG}" \
  --preset "${PRESET}" \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --device "${DEVICE}" \
  --controller qwen_cache \
  --condition-embedding-cache "${CONDITION_CACHE}" \
  --train-index "${TRAIN_INDEX}" \
  --val-index "${VAL_INDEX}" \
  --output-dir "${TRAIN_DIR}" \
  --skip-torch-preflight \
  "${VISUAL_ARGS[@]}" \
  "${TRAIN_CONTROL[@]}" \
  "${RESUME_ARGS[@]}"

CHECKPOINT="${TRAIN_DIR}/checkpoint_best.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${TRAIN_DIR}/checkpoint_last.pt"
fi

EVAL_CONTROL=()
if [[ "${RUN_CONTROL}" == "--overwrite" ]]; then
  EVAL_CONTROL+=(--overwrite-output)
fi

"${PYTHON_BIN}" -m qpsalm_seg.cli.eval \
  --config "${CONFIG}" \
  --preset "${PRESET}" \
  --checkpoint "${CHECKPOINT}" \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --device "${DEVICE}" \
  --controller qwen_cache \
  --condition-embedding-cache "${CONDITION_CACHE}" \
  --val-index "${VAL_INDEX}" \
  --output-dir "${EVAL_DIR}" \
  --skip-torch-preflight \
  "${VISUAL_ARGS[@]}" \
  "${EVAL_CONTROL[@]}"

"${PYTHON_BIN}" -m qpsalm_seg.cli.summarize_run \
  --run-dir "${TRAIN_DIR}" \
  --eval-dir "${EVAL_DIR}"

"${PYTHON_BIN}" -m qpsalm_seg.cli.diagnose_run \
  --run "${TRAIN_DIR}/run_summary.json" \
  --output "${TRAIN_DIR}/diagnose_report.json"
