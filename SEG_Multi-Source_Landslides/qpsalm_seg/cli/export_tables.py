#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 QPSALM 指标、多源 gate 与 proposal 诊断 CSV。

脚本作用：从 eval_report.json、validation_latest.json 或 run_summary.json 中抽取
Dice/IoU/Precision/Recall、modality gate summary 与样本级 proposal 选择诊断，
写成 CSV 表格。
主要输入：一个或多个 QPSALM JSON 报告。
主要输出：metrics.csv、modality_gates.csv、proposal_diagnostics.csv、
analysis_tables_manifest.json。
是否改写原始数据：不会，只写指定 output-dir。
典型用法：python -m qpsalm_seg.cli.export_tables --input outputs/.../eval_report.json --output-dir outputs/.../tables。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.analysis_tables import export_analysis_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export QPSALM metrics, modality gates, and proposal diagnostics to CSV.")
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
