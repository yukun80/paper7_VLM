#!/usr/bin/env bash
set -euo pipefail

TRAIN_REPO="${TRAIN_REPO:-external/Qwen-VL-Series-Finetune}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATA_PATH="${DATA_PATH:-benchmark/geohazard_halluground_v2_full/llava_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3vl2b_smoke_lora}"

EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_LORA_MODULES="${NUM_LORA_MODULES:--1}"
BITS="${BITS:-16}"

IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-100352}"   # 128 * 28 * 28
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-200704}"   # 256 * 28 * 28
IMAGE_RESIZED_WIDTH="${IMAGE_RESIZED_WIDTH:-}"
IMAGE_RESIZED_HEIGHT="${IMAGE_RESIZED_HEIGHT:-}"

USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-False}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"
BF16="${BF16:-True}"
FP16="${FP16:-False}"
TF32="${TF32:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
LAZY_PREPROCESS="${LAZY_PREPROCESS:-True}"
REPORT_TO="${REPORT_TO:-tensorboard}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"

PROJECT_ROOT="$(pwd -P)"

abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$PROJECT_ROOT/$value"
  fi
}

TRAIN_REPO_ABS="$(abs_path "$TRAIN_REPO")"
DATA_PATH_ABS="$(abs_path "$DATA_PATH")"
IMAGE_FOLDER_ABS="$(abs_path "$IMAGE_FOLDER")"
OUTPUT_DIR_ABS="$(abs_path "$OUTPUT_DIR")"

if [[ "$DEEPSPEED_CONFIG" = /* ]]; then
  DEEPSPEED_CONFIG_ABS="$DEEPSPEED_CONFIG"
else
  DEEPSPEED_CONFIG_ABS="$TRAIN_REPO_ABS/$DEEPSPEED_CONFIG"
fi

if [[ ! -d "$TRAIN_REPO_ABS" ]]; then
  echo "Missing training repo: $TRAIN_REPO_ABS" >&2
  echo "Clone it first:" >&2
  echo "  git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git external/Qwen-VL-Series-Finetune" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_REPO_ABS/src/train/train_sft.py" ]]; then
  echo "Missing training entrypoint: $TRAIN_REPO_ABS/src/train/train_sft.py" >&2
  exit 2
fi

if [[ ! -f "$DATA_PATH_ABS" ]]; then
  echo "Missing training data: $DATA_PATH_ABS" >&2
  echo "Run scripts/2-1_export_qwen_splits.py and scripts/2-2_convert_qwen_to_llava.py first." >&2
  exit 2
fi

if [[ ! -d "$IMAGE_FOLDER_ABS" ]]; then
  echo "Missing image folder: $IMAGE_FOLDER_ABS" >&2
  exit 2
fi

if [[ ! -f "$DEEPSPEED_CONFIG_ABS" ]]; then
  echo "Missing Deepspeed config: $DEEPSPEED_CONFIG_ABS" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR_ABS"

cd "$TRAIN_REPO_ABS"
export PYTHONPATH="src:${PYTHONPATH:-}"

image_resize_args=()
if [[ -n "$IMAGE_RESIZED_WIDTH" ]]; then
  image_resize_args+=(--image_resized_width "$IMAGE_RESIZED_WIDTH")
fi
if [[ -n "$IMAGE_RESIZED_HEIGHT" ]]; then
  image_resize_args+=(--image_resized_height "$IMAGE_RESIZED_HEIGHT")
fi

deepspeed src/train/train_sft.py \
  --deepspeed "$DEEPSPEED_CONFIG_ABS" \
  --model_id "$MODEL_ID" \
  --data_path "$DATA_PATH_ABS" \
  --image_folder "$IMAGE_FOLDER_ABS" \
  --output_dir "$OUTPUT_DIR_ABS" \
  --remove_unused_columns False \
  --lora_enable True \
  --vision_lora False \
  --use_dora False \
  --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --num_lora_modules "$NUM_LORA_MODULES" \
  --bits "$BITS" \
  --freeze_llm True \
  --freeze_vision_tower True \
  --freeze_merger True \
  --bf16 "$BF16" \
  --fp16 "$FP16" \
  --tf32 "$TF32" \
  --disable_flash_attn2 "$DISABLE_FLASH_ATTN2" \
  --use_liger_kernel "$USE_LIGER_KERNEL" \
  --num_train_epochs "$EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --lr_scheduler_type "$LR_SCHEDULER" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --image_min_pixels "$IMAGE_MIN_PIXELS" \
  --image_max_pixels "$IMAGE_MAX_PIXELS" \
  "${image_resize_args[@]}" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --report_to "$REPORT_TO" \
  --lazy_preprocess "$LAZY_PREPROCESS" \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
