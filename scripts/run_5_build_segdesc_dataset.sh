#!/usr/bin/env bash
# 用途：依次构建、验证并汇总 QPSALM segmentation-description 统一引用索引。
# 推荐命令：RUN_CONTROL=--overwrite PYTHON_BIN=python bash scripts/run_5_build_segdesc_dataset.sh small
# 输入：已验证的 Landslide V2、Description V2 和 Landslide Bridge；不会复制其图像或 mask。
# 输出：../benchmark/multisource_landslide_segdesc_v1_<mode>。
# 注意：Bridge 未冻结时只发布 auto component；残留 expert index/gate 会被记录并忽略。

set -euo pipefail

MODE="${1:-small}"
if [[ "${MODE}" != "small" && "${MODE}" != "full" ]]; then
  echo "Usage: bash scripts/run_5_build_segdesc_dataset.sh [small|full]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_CONTROL="${RUN_CONTROL:-}"
DRY_RUN="${DRY_RUN:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/multisource_landslide_segdesc_v1_${MODE}}"

COMMON=(--mode "${MODE}" --output-dir "${OUTPUT_DIR}" --max-samples "${MAX_SAMPLES}" --seed "${SEED}")
if [[ -n "${RUN_CONTROL}" ]]; then COMMON+=("${RUN_CONTROL}"); fi
if [[ -n "${DRY_RUN}" ]]; then COMMON+=(--dry-run); fi

echo "[SEGDESC] mode=${MODE} output=${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/5-segdesc/5-1_build_unified_index.py "${COMMON[@]}"
if [[ -n "${DRY_RUN}" ]]; then
  echo "[SEGDESC] dry-run completed; validation and summary require published indexes"
  exit 0
fi
"${PYTHON_BIN}" scripts/5-segdesc/5-2_validate_unified_index.py "${COMMON[@]}"
"${PYTHON_BIN}" scripts/5-segdesc/5-3_summarize_unified_index.py "${COMMON[@]}"
