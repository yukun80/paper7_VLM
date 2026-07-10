#!/usr/bin/env bash
# 用途：在单卡 CUDA 上运行真实 Qwen embedding cache + QPSALM 小步训练闭环。
# 运行：bash SEG_Multi-Source_Landslides/scripts/run_qwen_phase1_smoke.sh
# 说明：默认跑 baseline/evidence 5 step；box-prior 仅作为 legacy ablation，需显式 MODE=box-prior 或 MODE=both。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
CONFIG="${CONFIG:-SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qpsalm_phase1_qwen_smoke}"
RUN_NAME="${RUN_NAME:-qwen_smoke}"
MODE="${MODE:-baseline}"
DEVICE="${DEVICE:-cuda}"
CONTROLLER="${CONTROLLER:-qwen_cache}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-qwen}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-}"
ALLOW_QWEN_CPU="${ALLOW_QWEN_CPU:-0}"
MAX_STEPS="${MAX_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-8}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
SAMPLES_PER_COMBO="${SAMPLES_PER_COMBO:-2}"
NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
EVAL_BEST_THRESHOLD="${EVAL_BEST_THRESHOLD:-1}"
FOREGROUND_BCE_POS_WEIGHT="${FOREGROUND_BCE_POS_WEIGHT:-2.0}"
MASK_TVERSKY_WEIGHT="${MASK_TVERSKY_WEIGHT:-0.5}"
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.3}"
TVERSKY_BETA="${TVERSKY_BETA:-0.7}"
EMPTY_MASK_SUPPRESSION_WEIGHT="${EMPTY_MASK_SUPPRESSION_WEIGHT:-0.0}"
EMPTY_PROPOSAL_SUPPRESSION_WEIGHT="${EMPTY_PROPOSAL_SUPPRESSION_WEIGHT:-0.0}"
PROPOSAL_POSITIVE_WEIGHT="${PROPOSAL_POSITIVE_WEIGHT:-1.0}"
CONDITION_POSITIVE_WEIGHT="${CONDITION_POSITIVE_WEIGHT:-1.0}"
QUERY_DIVERSITY_LOSS_WEIGHT="${QUERY_DIVERSITY_LOSS_WEIGHT:-0.0}"
SELECTION_RANKING_LOSS_WEIGHT="${SELECTION_RANKING_LOSS_WEIGHT:-0.2}"
PROPOSAL_MASK_DIVERSITY_LOSS_WEIGHT="${PROPOSAL_MASK_DIVERSITY_LOSS_WEIGHT:-0.0}"
GATE_ENTROPY_LOSS_WEIGHT="${GATE_ENTROPY_LOSS_WEIGHT:-0.0}"
PROPOSAL_SOFT_TARGET_TOPK="${PROPOSAL_SOFT_TARGET_TOPK:-1}"
PROPOSAL_SOFT_TARGET_TEMPERATURE="${PROPOSAL_SOFT_TARGET_TEMPERATURE:-0.10}"
QUERY_USAGE_BALANCE_LOSS_WEIGHT="${QUERY_USAGE_BALANCE_LOSS_WEIGHT:-0.0}"
TRAIN_HFLIP_PROB="${TRAIN_HFLIP_PROB:-0.0}"
TRAIN_VFLIP_PROB="${TRAIN_VFLIP_PROB:-0.0}"
SELECTION_PROPOSAL_WEIGHT="${SELECTION_PROPOSAL_WEIGHT:-1.0}"
SELECTION_CONDITION_WEIGHT="${SELECTION_CONDITION_WEIGHT:-1.0}"
SELECTION_TEMPERATURE="${SELECTION_TEMPERATURE:-1.0}"
FINAL_FOREGROUND_GATE_WEIGHT="${FINAL_FOREGROUND_GATE_WEIGHT:-0.0}"
CANONICAL_COMBO_LOSS_WEIGHTS="${CANONICAL_COMBO_LOSS_WEIGHTS:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-1}"
RUN_CONTROL="${RUN_CONTROL:---overwrite}"
QPSALM_EXTRA_ARGS="${QPSALM_EXTRA_ARGS:-}"
VERIFY_AFTER_RUN="${VERIFY_AFTER_RUN:-1}"

export PYTHONPATH="SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

OPTIONAL_ARGS=()
if [[ -n "${EMBEDDING_DEVICE}" ]]; then
  OPTIONAL_ARGS+=(--embedding-device "${EMBEDDING_DEVICE}")
fi
if [[ "${ALLOW_QWEN_CPU}" == "1" ]]; then
  OPTIONAL_ARGS+=(--allow-qwen-cpu)
fi
if [[ "${EVAL_BEST_THRESHOLD}" == "1" ]]; then
  OPTIONAL_ARGS+=(--eval-best-threshold)
fi

"${PYTHON_BIN}" -m qpsalm_seg.cli.run_phase1 \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --mode "${MODE}" \
  --device "${DEVICE}" \
  --controller "${CONTROLLER}" \
  --embedding-backend "${EMBEDDING_BACKEND}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --index-strategy balanced-canonical \
  --samples-per-combo "${SAMPLES_PER_COMBO}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-val-samples "${MAX_VAL_SAMPLES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --num-visualizations "${NUM_VISUALIZATIONS}" \
  --eval-threshold "${EVAL_THRESHOLD}" \
  --foreground-bce-pos-weight "${FOREGROUND_BCE_POS_WEIGHT}" \
  --mask-tversky-weight "${MASK_TVERSKY_WEIGHT}" \
  --tversky-alpha "${TVERSKY_ALPHA}" \
  --tversky-beta "${TVERSKY_BETA}" \
  --empty-mask-suppression-weight "${EMPTY_MASK_SUPPRESSION_WEIGHT}" \
  --empty-proposal-suppression-weight "${EMPTY_PROPOSAL_SUPPRESSION_WEIGHT}" \
  --proposal-positive-weight "${PROPOSAL_POSITIVE_WEIGHT}" \
  --condition-positive-weight "${CONDITION_POSITIVE_WEIGHT}" \
  --query-diversity-loss-weight "${QUERY_DIVERSITY_LOSS_WEIGHT}" \
  --selection-ranking-loss-weight "${SELECTION_RANKING_LOSS_WEIGHT}" \
  --proposal-mask-diversity-loss-weight "${PROPOSAL_MASK_DIVERSITY_LOSS_WEIGHT}" \
  --gate-entropy-loss-weight "${GATE_ENTROPY_LOSS_WEIGHT}" \
  --proposal-soft-target-topk "${PROPOSAL_SOFT_TARGET_TOPK}" \
  --proposal-soft-target-temperature "${PROPOSAL_SOFT_TARGET_TEMPERATURE}" \
  --query-usage-balance-loss-weight "${QUERY_USAGE_BALANCE_LOSS_WEIGHT}" \
  --train-hflip-prob "${TRAIN_HFLIP_PROB}" \
  --train-vflip-prob "${TRAIN_VFLIP_PROB}" \
  --selection-proposal-weight "${SELECTION_PROPOSAL_WEIGHT}" \
  --selection-condition-weight "${SELECTION_CONDITION_WEIGHT}" \
  --selection-temperature "${SELECTION_TEMPERATURE}" \
  --final-foreground-gate-weight "${FINAL_FOREGROUND_GATE_WEIGHT}" \
  --canonical-combo-loss-weights "${CANONICAL_COMBO_LOSS_WEIGHTS}" \
  --num-workers "${NUM_WORKERS}" \
  --min-visualizations "${NUM_VISUALIZATIONS}" \
  "${OPTIONAL_ARGS[@]}" \
  ${RUN_CONTROL} \
  ${QPSALM_EXTRA_ARGS}

if [[ "${VERIFY_AFTER_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" -m qpsalm_seg.cli.verify_phase1 \
    --run-root "${OUTPUT_ROOT}/${RUN_NAME}" \
    --require-mode "${MODE}" \
    --require-embedding-backend "${EMBEDDING_BACKEND}" \
    --require-device "${DEVICE}" \
    --min-visualizations "${NUM_VISUALIZATIONS}"
fi
