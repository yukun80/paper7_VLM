#!/usr/bin/env bash
# 阶段 1B full 入口（仅由项目负责人运行）。
# 用途：构建、验证、汇总并 smoke 全量统一 Benchmark。
# 命令：bash scripts/phase1_benchmark_build/run_build_full.sh
# 输入：只读 ../datasets；输出：../benchmark/oa_auxseg_hdf5_v1/full。
# 写入：目标存在即停止，不覆盖；可能需要较长时间和较大磁盘空间。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yukun80/miniconda3/envs/qwen3vl/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${REPO_ROOT}/../datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/../benchmark/oa_auxseg_hdf5_v1}"
PATCH_SIZE="${PATCH_SIZE:-224}"
SEED="${SEED:-20260724}"
SPLIT_SEED="${SPLIT_SEED:-20260724}"
SHARD_TARGET_MIB="${SHARD_TARGET_MIB:-512}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_1_build_benchmark.py" \
  --mode full \
  --datasets-root "${DATASETS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --patch-size "${PATCH_SIZE}" \
  --seed "${SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --shard-target-mib "${SHARD_TARGET_MIB}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_2_validate_benchmark.py" \
  --benchmark-root "${OUTPUT_ROOT}/full" \
  --deep

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_3_summarize_benchmark.py" \
  --benchmark-root "${OUTPUT_ROOT}/full"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_4_smoke_dataloader.py" \
  --benchmark-root "${OUTPUT_ROOT}/full"
