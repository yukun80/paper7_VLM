#!/usr/bin/env bash
# 批运行脚本：构建多源滑坡 instruction segmentation 训练索引。
#
# 脚本作用：按固定顺序运行 2-1 到 2-3，完成任务指令模板校验、
# instruction_*.jsonl 生成和 instruction 索引验证。
# 主要输入：benchmark/multisource_landslide_v1_<mode>/indexes/all.jsonl、
# indexes/referring_target_all.jsonl，以及 configs/instruction_templates/*.yaml。
# 主要输出：benchmark/multisource_landslide_v1_<mode>/indexes/instruction_*.jsonl
# 和 reports/instruction_*.json。
# 是否改写原始数据：不会改写 datasets/；不会覆盖 1-benchmark 生成的 all/referring_target 索引。
# 环境变量覆盖：BENCHMARK_PREFIX、TEMPLATE_CONFIG、PYTHON_BIN 可覆盖；
# 默认 PYTHON_BIN=python，建议先 conda activate qwen3vl。
#
# 用法：
#   bash scripts/run_2_build_instruction_dataset.sh small
#   bash scripts/run_2_build_instruction_dataset.sh full
#   bash scripts/run_2_build_instruction_dataset.sh both

set -euo pipefail

MODE="${1:-small}"
BENCHMARK_PREFIX="${BENCHMARK_PREFIX:-benchmark/multisource_landslide_v1}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/instruction_templates/multisource_landslide_v1.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_one_mode() {
  local mode="$1"
  local out_dir="${BENCHMARK_PREFIX}_${mode}"

  echo "==> [2-1] 校验任务指令模板: ${mode}"
  "${PYTHON_BIN}" scripts/2-instruction/2-1_build_instruction_templates.py \
    --benchmark-dir "${out_dir}" \
    --template-config "${TEMPLATE_CONFIG}"

  echo "==> [2-2] 应用任务指令模板生成 instruction 索引: ${mode}"
  "${PYTHON_BIN}" scripts/2-instruction/2-2_apply_instruction_templates.py \
    --benchmark-dir "${out_dir}" \
    --template-config "${TEMPLATE_CONFIG}"

  echo "==> [2-3] 验证 instruction 索引质量: ${mode}"
  "${PYTHON_BIN}" scripts/2-instruction/2-3_validate_instruction_index.py \
    --benchmark-dir "${out_dir}" \
    --template-config "${TEMPLATE_CONFIG}"

  echo "==> 完成 ${mode} instruction dataset: ${out_dir}"
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
