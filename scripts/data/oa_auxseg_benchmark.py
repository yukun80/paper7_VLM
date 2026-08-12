#!/usr/bin/env python3
"""用途：构建、验证、汇总或 smoke OA-AuxSeg Benchmark。

命令：python scripts/data/oa_auxseg_benchmark.py {build,validate,summary,smoke} --help
输入：只读 HDF5 数据根或既有 Benchmark；输出：新 Benchmark 或只读报告。
写入：build 拒绝覆盖并原子发布；其他命令不修改 Benchmark。
所属能力：Spatial Perception 数据生产。
"""

from __future__ import annotations

import argparse
from typing import Sequence

from oa_groundrag.data.oa_auxseg import builder, smoke, summary, validator


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "summary", "smoke"))
    arguments, remaining = parser.parse_known_args(argv)
    commands = {
        "build": builder.main,
        "validate": validator.main,
        "summary": summary.main,
        "smoke": smoke.main,
    }
    return commands[arguments.command](remaining)


if __name__ == "__main__":
    raise SystemExit(main())
