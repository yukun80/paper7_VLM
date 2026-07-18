#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 strict cache migration thin workflow.

用途：转发 side-by-side M3 v2 到当前 M3 v3 的严格迁移命令。
推荐调用：``qpsalm-segdesc cache migrate``。
输入：旧 cache、当前 Description v4/Bridge v7 与 segmentation cache v3。
输出：新 cache、migration audit、manifest 与 validation_report.json。
写入行为：仅由 data.cache_migration 创建新 output-dir，不修改旧 artifact。
工作流阶段：M3 artifact migration orchestration。
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from ..data.cache_migration import migrate_cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly migrate an M3 v2 Description Vision Cache to M3 v3"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-cache", required=True)
    parser.add_argument("--description-benchmark", required=True)
    parser.add_argument("--bridge-benchmark", required=True)
    parser.add_argument("--segmentation-vision-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    report = migrate_cache(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
