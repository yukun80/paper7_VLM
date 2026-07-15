#!/usr/bin/env bash
# 用途：构建 Landslide Bridge M2 自动事实、候选描述、双人审核包并在人工审核后冻结专家 Pilot。
# 推荐准备命令：BRIDGE_STAGE=prepare BRIDGE_PILOT_PARENTS=300 RUN_CONTROL=--overwrite PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python bash scripts/run_4_build_landslide_bridge.sh small
# 推荐合并命令：BRIDGE_STAGE=merge REVIEWER_1=<完成的reviewer_1.jsonl> REVIEWER_2=<完成的reviewer_2.jsonl> ARBITRATION_FILE=<仲裁.jsonl> EVALUATION_GATE=<冻结gate.json> RUN_CONTROL=--overwrite bash scripts/run_4_build_landslide_bridge.sh small
# 主要输入：benchmark/multisource_landslide_v2_<mode>；merge 另需人工审核文件。
# 主要输出：benchmark/landslide_region_description_v1_<mode> 的 indexes、masks、review package 和 reports。
# 写入行为：只写 Bridge benchmark；不修改原始 datasets、Landslide V2 或 checkpoint。
# 所属流程：docs/benchmark_GAR.md M2；prepare 不会生成专家标签，merge 不允许静默跳过仲裁或 gate。
# 环境变量：BRIDGE_STAGE、BRIDGE_PILOT_PARENTS、PYTHON_BIN、PAPER7_BENCHMARK_ROOT、
# SOURCE_BENCHMARK、BRIDGE_OUTPUT_DIR、REVIEWER_1、REVIEWER_2、ARBITRATION_FILE、
# EVALUATION_GATE、MAX_SAMPLES、SEED、RUN_CONTROL、REQUIRE_EXPERT_COMPLETE。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-small}"
BRIDGE_STAGE="${BRIDGE_STAGE:-prepare}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PILOT_PARENTS="${BRIDGE_PILOT_PARENTS:-300}"
RUN_CONTROL="${RUN_CONTROL:-}"

export PAPER7_BENCHMARK_ROOT="${PAPER7_BENCHMARK_ROOT:-${WORKSPACE_ROOT}/benchmark}"
SOURCE_BENCHMARK="${SOURCE_BENCHMARK:-${PAPER7_BENCHMARK_ROOT}/multisource_landslide_v2_${MODE}}"
BRIDGE_OUTPUT_DIR="${BRIDGE_OUTPUT_DIR:-${PAPER7_BENCHMARK_ROOT}/landslide_region_description_v1_${MODE}}"

control=()
if [[ -n "${RUN_CONTROL}" ]]; then
  control+=("${RUN_CONTROL}")
fi
common=(--mode "${MODE}" --output-dir "${BRIDGE_OUTPUT_DIR}" --config configs/landslide_bridge_v1.yaml --seed "${SEED}")

run_prepare() {
  echo "[BRIDGE] stage=prepare mode=${MODE} source=${SOURCE_BENCHMARK} output=${BRIDGE_OUTPUT_DIR}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-1_inventory_regions.py \
    "${common[@]}" --source-benchmark "${SOURCE_BENCHMARK}" --pilot-parents "${PILOT_PARENTS}" \
    --max-samples "${MAX_SAMPLES}" "${control[@]}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-2_extract_region_facts.py \
    "${common[@]}" --source-benchmark "${SOURCE_BENCHMARK}" "${control[@]}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-3_build_candidate_descriptions.py \
    "${common[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-4_build_review_package.py \
    "${common[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-6_validate_landslide_bridge.py \
    "${common[@]}" "${control[@]}"
  echo "[BRIDGE] prepare_complete=${BRIDGE_OUTPUT_DIR} status_report=${BRIDGE_OUTPUT_DIR}/reports/validation_report.json"
}

run_merge() {
  if [[ -z "${REVIEWER_1:-}" || -z "${REVIEWER_2:-}" || -z "${EVALUATION_GATE:-}" ]]; then
    echo "错误：merge 必须设置 REVIEWER_1、REVIEWER_2 和人工冻结的 EVALUATION_GATE。" >&2
    exit 2
  fi
  merge_args=(--reviewer-1 "${REVIEWER_1}" --reviewer-2 "${REVIEWER_2}" --evaluation-gate "${EVALUATION_GATE}")
  if [[ -n "${ARBITRATION_FILE:-}" ]]; then
    merge_args+=(--arbitration-file "${ARBITRATION_FILE}")
  fi
  echo "[BRIDGE] stage=merge mode=${MODE} output=${BRIDGE_OUTPUT_DIR}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-5_merge_expert_reviews.py \
    "${common[@]}" "${merge_args[@]}" "${control[@]}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-6_validate_landslide_bridge.py \
    "${common[@]}" --require-expert-complete "${control[@]}"
  echo "[BRIDGE] merge_complete=${BRIDGE_OUTPUT_DIR} status=expert_pilot_frozen"
}

run_validate() {
  validate_args=()
  if [[ "${REQUIRE_EXPERT_COMPLETE:-0}" == "1" ]]; then
    validate_args+=(--require-expert-complete)
  fi
  echo "[BRIDGE] stage=validate mode=${MODE} output=${BRIDGE_OUTPUT_DIR}"
  "${PYTHON_BIN}" scripts/4-landslide-bridge/4-6_validate_landslide_bridge.py \
    "${common[@]}" "${validate_args[@]}" "${control[@]}"
}

case "${MODE}" in
  small|full) ;;
  *) echo "错误：MODE 必须为 small 或 full，当前为 ${MODE}" >&2; exit 2 ;;
esac

case "${BRIDGE_STAGE}" in
  prepare) run_prepare ;;
  merge) run_merge ;;
  validate) run_validate ;;
  *) echo "错误：BRIDGE_STAGE 必须为 prepare、merge 或 validate，当前为 ${BRIDGE_STAGE}" >&2; exit 2 ;;
esac
