#!/usr/bin/env bash
# 用途：构建 qpsalm_description_v2 的 M0 审计与 M1 自包含图片 benchmark。
# 推荐运行命令：PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python bash scripts/run_3_build_description_benchmark.sh small
# 主要输入：仓库同级 datasets/MMRS-1M、datasets/RSGPT/dataset/RSICap|RSIEval（兼容旧布局）。
# 主要输出：仓库同级 benchmark/qpsalm_description_v2_<mode>/ 的 data、indexes、manifests、reports。
# 写入行为：复制入选图片到 benchmark，不修改原图；RUN_CONTROL=--overwrite 可覆盖派生产物。
# 所属流程：docs/benchmark_GAR.md 的 M0/M1；通过后才进入 Landslide Bridge M2。
# 环境变量：PYTHON_BIN、PAPER7_DATASETS_ROOT、PAPER7_BENCHMARK_ROOT、PAPER7_RSGPT_DATA_ROOT、
# DESCRIPTION_SMALL_MMRS_PARENTS、DESCRIPTION_SMALL_DIOR_PARENTS、DESCRIPTION_COPY_WORKERS、
# DESCRIPTION_PERCEPTUAL_MAE_THRESHOLD、
# MAX_SAMPLES、SEED、RUN_CONTROL。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-small}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SMALL_MMRS_PARENTS="${DESCRIPTION_SMALL_MMRS_PARENTS:-12000}"
SMALL_DIOR_PARENTS="${DESCRIPTION_SMALL_DIOR_PARENTS:-5000}"
RUN_CONTROL="${RUN_CONTROL:-}"
COPY_WORKERS="${DESCRIPTION_COPY_WORKERS:-8}"
PERCEPTUAL_MAE_THRESHOLD="${DESCRIPTION_PERCEPTUAL_MAE_THRESHOLD:-3.0}"

export PAPER7_DATASETS_ROOT="${PAPER7_DATASETS_ROOT:-${WORKSPACE_ROOT}/datasets}"
export PAPER7_BENCHMARK_ROOT="${PAPER7_BENCHMARK_ROOT:-${WORKSPACE_ROOT}/benchmark}"

run_one_mode() {
  local mode="$1"
  local output_dir="${PAPER7_BENCHMARK_ROOT}/qpsalm_description_v2_${mode}"
  local common=(--mode "${mode}" --output-dir "${output_dir}" --seed "${SEED}" --max-samples "${MAX_SAMPLES}")
  local control=()
  if [[ -n "${RUN_CONTROL}" ]]; then
    control+=("${RUN_CONTROL}")
  fi

  echo "[DESC] mode=${mode} output=${output_dir}"
  "${PYTHON_BIN}" scripts/3-description/3-1_scan_description_sources.py "${common[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/3-description/3-2_build_global_caption_index.py "${common[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/3-description/3-3_build_region_alignment_index.py "${common[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/3-description/3-4_deduplicate_and_split.py "${common[@]}" \
    --small-mmrs-parents "${SMALL_MMRS_PARENTS}" --small-dior-parents "${SMALL_DIOR_PARENTS}" \
    --perceptual-mae-threshold "${PERCEPTUAL_MAE_THRESHOLD}" "${control[@]}"
  "${PYTHON_BIN}" scripts/3-description/3-5_materialize_description_images.py "${common[@]}" \
    --workers "${COPY_WORKERS}" "${control[@]}"
  "${PYTHON_BIN}" scripts/3-description/3-6_validate_description_benchmark.py "${common[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/3-description/3-7_summarize_description_benchmark.py "${common[@]}" "${control[@]}"
  echo "[DESC] complete=${output_dir}"
}

case "${MODE}" in
  small|full) run_one_mode "${MODE}" ;;
  both) run_one_mode small; run_one_mode full ;;
  *) echo "错误：MODE 必须是 small、full 或 both，当前为 ${MODE}" >&2; exit 2 ;;
esac
