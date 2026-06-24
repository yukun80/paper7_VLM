#!/usr/bin/env bash
set -euo pipefail

TRAIN_REPO="${TRAIN_REPO:-external/Qwen-VL-Series-Finetune}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATA_PATH="${DATA_PATH:-benchmark/geohazard_halluground_v2_full/llava_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3vl2b_geohazard_lora}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-262144}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-524288}"
BITS="${BITS:-16}"
USE_LIGER="${USE_LIGER:-False}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"
REPORT_TO="${REPORT_TO:-tensorboard}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"
PROJECT_ROOT="$(pwd -P)"
if [[ "$TRAIN_REPO" = /* ]]; then
  TRAIN_REPO_ABS="$TRAIN_REPO"
else
  TRAIN_REPO_ABS="$PROJECT_ROOT/$TRAIN_REPO"
fi
if [[ "$DATA_PATH" = /* ]]; then
  DATA_PATH_ABS="$DATA_PATH"
else
  DATA_PATH_ABS="$PROJECT_ROOT/$DATA_PATH"
fi
if [[ "$IMAGE_FOLDER" = /* ]]; then
  IMAGE_FOLDER_ABS="$IMAGE_FOLDER"
else
  IMAGE_FOLDER_ABS="$PROJECT_ROOT/$IMAGE_FOLDER"
fi
if [[ "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_DIR_ABS="$OUTPUT_DIR"
else
  OUTPUT_DIR_ABS="$PROJECT_ROOT/$OUTPUT_DIR"
fi

if [[ ! -d "$TRAIN_REPO_ABS" ]]; then
  echo "Missing training repo: $TRAIN_REPO" >&2
  echo "Clone it first, for example:" >&2
  echo "  git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git $TRAIN_REPO" >&2
  exit 2
fi

if [[ ! -f "$DATA_PATH_ABS" ]]; then
  echo "Missing training data: $DATA_PATH" >&2
  echo "Run scripts/2-1_export_qwen_splits.py and scripts/2-2_convert_qwen_to_llava.py first." >&2
  exit 2
fi

if [[ -f "$TRAIN_REPO_ABS/scripts/finetune_lora.sh" ]]; then
  cd "$TRAIN_REPO_ABS"
  bash scripts/finetune_lora.sh \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH_ABS" \
    --image_folder "$IMAGE_FOLDER_ABS" \
    --output_dir "$OUTPUT_DIR_ABS" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --lora_enable True \
    --vision_lora False \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bits "$BITS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --image_resized_width "$IMAGE_SIZE" \
    --image_resized_height "$IMAGE_SIZE" \
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --use_liger "$USE_LIGER" \
    --disable_flash_attn2 "$DISABLE_FLASH_ATTN2" \
    --report_to "$REPORT_TO" \
    --deepspeed "$DEEPSPEED_CONFIG"
else
  echo "Could not find $TRAIN_REPO/scripts/finetune_lora.sh." >&2
  echo "The upstream training repo layout may have changed; update TRAIN_REPO or this wrapper." >&2
  exit 2
fi
