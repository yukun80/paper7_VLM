#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-benchmark/geohazard_halluground_v2_full}"
SEN12_ROOT="${SEN12_ROOT:-datasets/Sen12Landslides}"
GDCLD_ROOT="${GDCLD_ROOT:-datasets/GDCLD/extracted}"
GDCLD_TILE_SIZE="${GDCLD_TILE_SIZE:-512}"
GDCLD_STRIDE="${GDCLD_STRIDE:-512}"
GDCLD_NEGATIVE_RATIO="${GDCLD_NEGATIVE_RATIO:-1.0}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-100}"
INCLUDE_FUTURE_WORK="${INCLUDE_FUTURE_WORK:-0}"

if [[ ! -d "$SEN12_ROOT/s2" || ! -d "$SEN12_ROOT/s1asc" || ! -d "$SEN12_ROOT/s1dsc" ]]; then
  echo "Missing Sen12Landslides subdirs under $SEN12_ROOT" >&2
  exit 2
fi

for rel in train_data train_label val_data val_label test_data test_label; do
  if [[ ! -e "$GDCLD_ROOT/$rel" ]]; then
    echo "Missing GDCLD extracted path: $GDCLD_ROOT/$rel" >&2
    echo "Extract datasets/GDCLD/train_data.zip, val_data.zip, and test_data.zip into $GDCLD_ROOT first." >&2
    exit 2
  fi
done

python scripts/1-1_scan_sources.py \
  --sen12-root "$SEN12_ROOT" \
  --gdcld-root "$GDCLD_ROOT" \
  --out-dir "$OUT_DIR" \
  --clean

python scripts/1-2_prepare_sen12_views.py \
  --out-dir "$OUT_DIR" \
  --version v2

gdcld_args=(
  --out-dir "$OUT_DIR"
  --gdcld-tile-size "$GDCLD_TILE_SIZE"
  --gdcld-stride "$GDCLD_STRIDE"
  --gdcld-negative-ratio "$GDCLD_NEGATIVE_RATIO"
)
if [[ "$INCLUDE_FUTURE_WORK" == "1" || "$INCLUDE_FUTURE_WORK" == "true" ]]; then
  gdcld_args+=(--include-gdcld-future-work)
fi

python scripts/1-3_prepare_gdcld_tiles.py "${gdcld_args[@]}"

python scripts/1-4_merge_annotations.py \
  --out-dir "$OUT_DIR"

python scripts/1-5_export_training_files.py \
  --out-dir "$OUT_DIR" \
  --version v2

python scripts/1-6_validate_and_summarize.py \
  --out-dir "$OUT_DIR" \
  --audit-samples "$AUDIT_SAMPLES"
