#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 QPSALM 指标、QMEF 证据使用与 PMRD proposal 诊断 CSV。

用途：从验证/评估报告中抽取指标、modality reliability、query attention 和 proposal 诊断。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.export_tables --input outputs/RUN/eval_report.json
--output-dir outputs/RUN/tables
主要输入：一个或多个 QPSALM JSON 报告。
主要输出：metrics.csv、modality_reliability.csv、query_modality_attention.csv、proposal_diagnostics.csv、
analysis_tables_manifest.json。
写入行为：只写 --output-dir，不修改输入报告或 checkpoint。
所属流程：训练/评估后的论文表格和算法诊断导出。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.analysis_tables import export_analysis_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export QPSALM metrics, QMEF evidence, and PMRD proposal diagnostics.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to eval_report.json, validation_latest.json, or run_summary.json. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_analysis_tables(args.input, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
