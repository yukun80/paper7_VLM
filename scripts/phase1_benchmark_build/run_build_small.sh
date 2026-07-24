#!/usr/bin/env bash
# 阶段 1B small 入口。
# 用途：构建、验证、汇总并 smoke 统一 Benchmark。
# 命令：bash scripts/phase1_benchmark_build/run_build_small.sh
# 输入：只读 ../datasets；输出：../benchmark/oa_auxseg_hdf5_v1/small。
# 写入：目标存在即停止，不覆盖。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${REPO_ROOT}/../datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/../benchmark/oa_auxseg_hdf5_v1}"
PATCH_SIZE="${PATCH_SIZE:-224}"
SMALL_PER_SOURCE="${SMALL_PER_SOURCE:-32}"
SEED="${SEED:-20260724}"
SPLIT_SEED="${SPLIT_SEED:-20260724}"
SHARD_TARGET_MIB="${SHARD_TARGET_MIB:-512}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_1_build_benchmark.py" \
  --mode small \
  --datasets-root "${DATASETS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --patch-size "${PATCH_SIZE}" \
  --small-per-source "${SMALL_PER_SOURCE}" \
  --seed "${SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --shard-target-mib "${SHARD_TARGET_MIB}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_2_validate_benchmark.py" \
  --benchmark-root "${OUTPUT_ROOT}/small" \
  --deep

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_3_summarize_benchmark.py" \
  --benchmark-root "${OUTPUT_ROOT}/small"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_4_smoke_dataloader.py" \
  --benchmark-root "${OUTPUT_ROOT}/small"
