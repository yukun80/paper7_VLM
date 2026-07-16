#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 D-1 zero-shot 与四路混合过拟合的统一工程门禁。

用途：只读核验两个既有 run 的协议、输入哈希、population、checkpoint 与生成文件。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.validate_d_minus_one --zero-shot-dir <dir> --overfit-dir <dir>
--output <dir>/d_minus_one_gate.json
输入：zero-shot 输出目录、overfit 输出目录。
输出：单个原子写入的 D-1 gate JSON。
写入行为：只写 --output，不运行模型、benchmark、CUDA 或训练。
所属流程：M5/M6 D-1 工程验收。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.d_minus_one import (
    validate_d_minus_one_runs,
    validate_d_minus_one_gate,
    write_d_minus_one_gate,
)
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the complete D-1 gate.")
    parser.add_argument("--zero-shot-dir", required=True)
    parser.add_argument("--overfit-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_d_minus_one_runs(
        args.zero_shot_dir,
        args.overfit_dir,
    )
    output = resolve_project_path(args.output) or Path(args.output)
    if report["errors"]:
        write_d_minus_one_gate(output, report)
    else:
        candidate = output.with_name(f".{output.name}.candidate")
        try:
            write_d_minus_one_gate(candidate, report)
            validate_d_minus_one_gate(candidate)
            candidate.replace(output)
        finally:
            candidate.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
