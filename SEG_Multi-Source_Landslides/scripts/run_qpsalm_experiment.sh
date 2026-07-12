#!/usr/bin/env bash
# 用途：准备 Qwen vision cache v3，并运行 benchmark-v2 正式训练与验证。
# 推荐命令：BENCHMARK_SIZE=small PRESET=qwen_psalm_full RUN_NAME=small_qwen \
#   RUN_CONTROL=--overwrite CACHE_CONTROL=reuse \
#   bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
# 输入：multisource_landslide_v2_{small,full}、本地 Qwen3-VL 权重和 Python preset。
# 输出：outputs/qpsalm_v2/<RUN_NAME> 下的 cache、best/last checkpoint、eval 和可视化。
# 写入：只写 outputs；RUN_CONTROL 仅控制运行目录，CACHE_CONTROL 单独控制 cache。
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
CACHE_CONTROL="${CACHE_CONTROL:-reuse}"
MEMORY_GATE="${MEMORY_GATE:-1}"

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
case "${CACHE_CONTROL}" in
  reuse|verify|overwrite) ;;
  *) echo "CACHE_CONTROL must be reuse, verify, or overwrite" >&2; exit 2 ;;
esac

export PYTHONPATH="SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="outputs/qpsalm_v2/${RUN_NAME}"
CACHE_DIR="outputs/qpsalm_v2/cache/${BENCHMARK_SIZE}_${PRESET}_qwen_vision_v3"
EVAL_DIR="${RUN_DIR}/eval_val"
VISION_ARGS=()

echo "[RUN] benchmark=${BENCHMARK_DIR} preset=${PRESET} seed=${SEED} run=${RUN_DIR}"

if [[ "${PRESET}" == "pretrained_sane_qmef_pmrd" || "${PRESET}" == "qwen_psalm_full" ]]; then
  if [[ "${CACHE_CONTROL}" == "overwrite" || ! -f "${CACHE_DIR}/manifest.json" ]]; then
    if [[ "${CACHE_CONTROL}" == "verify" ]]; then
      echo "[CACHE] missing=${CACHE_DIR}/manifest.json" >&2
      exit 2
    fi
    echo "[CACHE] action=build path=${CACHE_DIR}"
    "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_vision_features \
      --config "${CONFIG}" --benchmark-dir "${BENCHMARK_DIR}" \
      --preset "${PRESET}" \
      --output-dir "${CACHE_DIR}" --device "${DEVICE}" --backend qwen --overwrite
  else
    echo "[CACHE] action=verify path=${CACHE_DIR}"
    "${PYTHON_BIN}" -m qpsalm_seg.cli.cache_qwen_vision_features \
      --config "${CONFIG}" --preset "${PRESET}" \
      --output-dir "${CACHE_DIR}" --verify-only
  fi
  VISION_ARGS+=(--vision-feature-cache "${CACHE_DIR}")
fi

if [[ "${PRESET}" == "qwen_psalm_full" && "${MEMORY_GATE}" == "1" ]]; then
  CONFIG_HASH="$("${PYTHON_BIN}" -c 'import hashlib,sys; h=hashlib.sha256(); [h.update(open(p,"rb").read()) for p in sys.argv[1:]]; print(h.hexdigest()[:12])' \
    "${CONFIG}" \
    SEG_Multi-Source_Landslides/qpsalm_seg/controllers.py \
    SEG_Multi-Source_Landslides/qpsalm_seg/data/dataset.py \
    SEG_Multi-Source_Landslides/qpsalm_seg/data/samplers.py \
    SEG_Multi-Source_Landslides/qpsalm_seg/schema.py \
    SEG_Multi-Source_Landslides/qpsalm_seg/models/qpsalm.py \
    SEG_Multi-Source_Landslides/qpsalm_seg/cli/integration_check.py)"
  GATE_REPORT="outputs/qpsalm_v2/cache/integration_${BENCHMARK_SIZE}_${PRESET}_seed${SEED}_${CONFIG_HASH}.json"
  GATE_PASSED=0
  if [[ -f "${GATE_REPORT}" ]] && "${PYTHON_BIN}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); q=d.get("checks",{}).get("qwen",{}); ok=d.get("acceptance",{}).get("passed") and q.get("protocol_version")=="qwen_representative_batch_v4"; raise SystemExit(0 if ok else 1)' \
    "${GATE_REPORT}"; then
    GATE_PASSED=1
  fi
  if [[ "${CACHE_CONTROL}" == "overwrite" || "${GATE_PASSED}" != "1" ]]; then
    echo "[GATE] action=run limit_gib=22.5 report=${GATE_REPORT}"
    "${PYTHON_BIN}" -m qpsalm_seg.cli.integration_check \
      --config "${CONFIG}" --benchmark-dir "${BENCHMARK_DIR}" --mode qwen \
      --qwen-preset "${PRESET}" \
      --vision-feature-cache "${CACHE_DIR}" --device "${DEVICE}" \
      --max-memory-gib 22.5 --output "${GATE_REPORT}"
  else
    echo "[GATE] action=reuse report=${GATE_REPORT}"
  fi
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
echo "[EVAL] action=full_val checkpoint=${CHECKPOINT} output=${EVAL_DIR}"
"${PYTHON_BIN}" -m qpsalm_seg.cli.eval \
  --config "${CONFIG}" --preset "${PRESET}" --benchmark-dir "${BENCHMARK_DIR}" \
  --checkpoint "${CHECKPOINT}" --device "${DEVICE}" --seed "${SEED}" \
  "${VISION_ARGS[@]}" --output-dir "${EVAL_DIR}" \
  --export-multimodal-overview --skip-torch-preflight "${EVAL_CONTROL[@]}"
