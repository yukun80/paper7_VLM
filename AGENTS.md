# Repository Guidelines

## Project Structure & Module Organization

This repository builds the GeoHazard-HalluGround V2 benchmark and prepares Qwen3-VL-2B LoRA fine-tuning data. Core code lives in `scripts/`: numbered `1-*` scripts build the benchmark, `2-*` scripts export fine-tuning splits, and `train_qwen3vl2b_lora.sh` wraps Qwen-VL-Series-Finetune training. Shared helpers are in `scripts/geohazard_common.py`.

Design notes live in `docs/`. Reference material is kept in `参考文献/` and `knowledge/`. Large local artifacts such as `datasets/`, `benchmark/`, `models/`, `outputs/`, and `external/` should generally stay uncommitted.

## Build, Test, and Development Commands

Build the V2 benchmark:

```bash
bash scripts/run_geohazard_v2_pipeline.sh
```

Export full Qwen/LLaVA training files:

```bash
python scripts/2-1_export_qwen_splits.py --out-dir benchmark/geohazard_halluground_v2_full
python scripts/2-2_convert_qwen_to_llava.py --out-dir benchmark/geohazard_halluground_v2_full --check-images
```

Run LoRA training:

```bash
bash scripts/train_qwen3vl2b_lora.sh
```

Dry-run zero-shot evaluation:

```bash
python scripts/eval_qwen3vl2b_zero_shot.py --input benchmark/geohazard_halluground_v2_full/llava_test.json --dry-run
```

Lightweight syntax checks:

```bash
bash -n scripts/*.sh
python -m py_compile scripts/*.py
```

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, and type hints where they clarify data contracts. Prefer `pathlib`, `argparse`, and structured JSON/JSONL reads and writes over ad hoc string parsing. Keep CLI defaults aligned with `benchmark/geohazard_halluground_v2_full`. Preserve the existing numbered script order and use descriptive lowercase names for generated artifacts.

## Testing Guidelines

There is no formal test suite yet. Use the validation and smoke workflow: `validation_report.json` must have an empty `errors` list, conversion should pass with `--check-images`, and LoRA smoke training should complete before full training. For training changes, report the command used, sample count, output directory, and whether `DISABLE_FLASH_ATTN2` was enabled.

## Commit & Pull Request Guidelines

Existing commits use concise Chinese action/result messages. Follow that style or use a short imperative English summary. Pull requests should describe changed scripts, dataset assumptions, commands run, output paths, and GPU/environment changes. Do not commit large datasets, model weights, checkpoints, logs, or cloned external repositories.
