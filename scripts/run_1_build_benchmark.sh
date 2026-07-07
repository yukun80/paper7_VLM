#!/usr/bin/env bash
# 批运行脚本：构建多源滑坡 benchmark。
#
# 脚本作用：按固定顺序运行 1-1 到 1-7，完成数据清单、统一索引、
# source 校验、真实物化预处理、final 校验、split、指代表达构建、
# referring 校验和统计报告构建。
# 主要输入：datasets/ 原始数据目录，以及 MODE=small/full/both。
# 主要输出：benchmark/multisource_landslide_v1_<mode>/ 下的自包含 .npy 数据、索引和报告。
# 是否改写原始数据：不会改写 datasets/，只写 benchmark/ 派生产物。
# 特别说明：1-6 不读取 datasets/，只基于已物化的 benchmark/data/**/mask.npy 生成指代表达。
# 环境变量覆盖：SMALL_LIMIT 默认 1000，可用 SMALL_LIMIT=100 临时降低；
# DATASETS_ROOT、BENCHMARK_PREFIX、SEED、PYTHON_BIN、USE_EXTENDED_POOL 也可覆盖；
# 默认 PYTHON_BIN=python，建议先 conda activate qwen3vl。
#
# 用法：
#   bash scripts/run_1_build_benchmark.sh small
#   bash scripts/run_1_build_benchmark.sh full
#   bash scripts/run_1_build_benchmark.sh both
#   SMALL_LIMIT=100 bash scripts/run_1_build_benchmark.sh small

set -euo pipefail

MODE="${1:-small}"
SMALL_LIMIT="${SMALL_LIMIT:-1000}"
SEED="${SEED:-42}"
DATASETS_ROOT="${DATASETS_ROOT:-datasets}"
BENCHMARK_PREFIX="${BENCHMARK_PREFIX:-benchmark/multisource_landslide_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_one_mode() {
  local mode="$1"
  local out_dir="${BENCHMARK_PREFIX}_${mode}"
  local extended_args=()

  # full 模式默认不启用 extended_pool；需要时可设置 USE_EXTENDED_POOL=1。
  if [[ "${mode}" == "full" && "${USE_EXTENDED_POOL:-0}" == "1" ]]; then
    extended_args+=(--use-extended-pool)
  fi

  echo "==> [1-1] 扫描原始数据目录: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-1_scan_sources.py \
    --datasets-root "${DATASETS_ROOT}" \
    --out-dir "${out_dir}"

  echo "==> [1-2] 构建源 JSONL 索引: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-2_build_index.py \
    --mode "${mode}" \
    --small-limit "${SMALL_LIMIT}" \
    --seed "${SEED}" \
    --datasets-root "${DATASETS_ROOT}" \
    --out-dir "${out_dir}" \
    "${extended_args[@]}"

  echo "==> [1-3/source] 验证源索引质量: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-3_validate_index.py \
    --benchmark-dir "${out_dir}" \
    --stage source

  echo "==> [1-4] 物化预处理数据并生成最终训练索引: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-4_preprocess_samples.py \
    --benchmark-dir "${out_dir}" \
    --strategy materialize

  echo "==> [1-3/final] 验证最终自包含索引质量: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-3_validate_index.py \
    --benchmark-dir "${out_dir}" \
    --stage final

  echo "==> [1-5] 生成最终 split 与采样权重: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-5_build_splits.py \
    --benchmark-dir "${out_dir}"

  echo "==> [1-6] 基于已物化数据构建指代表达: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-6_build_referring_expressions.py \
    --benchmark-dir "${out_dir}"

  echo "==> [1-3/referring] 验证指代表达训练索引质量: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-3_validate_index.py \
    --benchmark-dir "${out_dir}" \
    --stage referring

  echo "==> [1-7] 汇总统计与清洗报告: ${mode}"
  "${PYTHON_BIN}" scripts/1-benchmark/1-7_summarize_benchmark.py \
    --benchmark-dir "${out_dir}"

  echo "==> 完成 ${mode} benchmark: ${out_dir}"
}

case "${MODE}" in
  small)
    run_one_mode small
    ;;
  full)
    run_one_mode full
    ;;
  both)
    run_one_mode small
    run_one_mode full
    ;;
  *)
    echo "错误：MODE 必须是 small、full 或 both，当前为 ${MODE}" >&2
    exit 2
    ;;
esac
