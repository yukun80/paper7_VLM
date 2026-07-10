#!/usr/bin/env bash
# 用途：按统一配置运行 proposal verifier 四组消融实验。
# 运行：BENCHMARK_SIZE=full VISUAL_BACKEND=qwen bash SEG_Multi-Source_Landslides/scripts/run_qwen_verifier_ablation.sh
# 说明：condition/evidence-text 分支不生成 visual cache；visual 分支默认使用 Qwen visual evidence cache。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARK_SIZE="${BENCHMARK_SIZE:-small}"
STAGES="${STAGES:-condition_only evidence_text visual_evidence evidence_visual}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen_verifier_${BENCHMARK_SIZE}}"
VISUAL_BACKEND="${VISUAL_BACKEND:-qwen}"
MODE="${MODE:-baseline}"

for STAGE in ${STAGES}; do
  case "${STAGE}" in
    condition_only|evidence_text)
      STAGE_VISUAL_BACKEND="off"
      ;;
    visual_evidence|evidence_visual)
      STAGE_VISUAL_BACKEND="${VISUAL_BACKEND}"
      ;;
    *)
      echo "Unsupported verifier stage: ${STAGE}" >&2
      echo "Expected one of: condition_only evidence_text visual_evidence evidence_visual" >&2
      exit 2
      ;;
  esac

  echo "==> verifier_stage=${STAGE} visual_backend=${STAGE_VISUAL_BACKEND}"
  BENCHMARK_SIZE="${BENCHMARK_SIZE}" \
  VERIFIER_STAGE="${STAGE}" \
  VISUAL_EVIDENCE_BACKEND="${STAGE_VISUAL_BACKEND}" \
  MODE="${MODE}" \
  RUN_NAME="${RUN_NAME_PREFIX}_${STAGE}" \
  bash "${SCRIPT_DIR}/run_qwen_phase1_full.sh"
done
