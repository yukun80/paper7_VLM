#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 2-1：生成或校验任务指令模板配置。

用途：读取 configs/instruction_templates/multisource_landslide_v1.yaml，
检查模板字段、模板 ID、任务类别覆盖情况，并输出模板校验报告。
主要输入：任务指令模板 YAML。
主要输出：reports/instruction_template_report.json。
写入行为：不会改写 datasets/ 或 benchmark 索引，只写模板验证报告。
所属流程：instruction 构建 2-1；应在 benchmark 构建完成后运行。
推荐运行命令：
  python scripts/2-instruction/2-1_build_instruction_templates.py \
    --benchmark-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geohazard_instruction_common import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_TEMPLATE_CONFIG,
    load_template_config,
    project_path_arg,
    to_repo_rel,
    validate_templates,
    write_template_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验多源滑坡任务指令模板 YAML。")
    parser.add_argument("--template-config", type=project_path_arg, default=DEFAULT_TEMPLATE_CONFIG, help="任务指令模板 YAML 路径。")
    parser.add_argument("--benchmark-dir", type=project_path_arg, default=DEFAULT_BENCHMARK_ROOT, help="目标 benchmark 目录，用于写报告。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_template_config(args.template_config)
    errors, warnings = validate_templates(config)
    report_path = args.benchmark_dir / "reports" / "instruction_template_report.json"
    write_template_report(report_path, config, errors, warnings)
    print(
        "任务指令模板校验完成: "
        f"templates={len(config.get('templates') or [])}, errors={len(errors)}, warnings={len(warnings)} -> "
        f"{to_repo_rel(report_path)}"
    )
    if errors:
        raise SystemExit("任务指令模板存在错误，请查看 instruction_template_report.json")


if __name__ == "__main__":
    main()
