#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚合 M4 五种 baseline 对 full MGRR 的三 seed 正式门禁。

用途：深度重算五份 `qpsalm-compare-description-runs` gate，确认 baseline 枚举完整、共享同一
full-MGRR candidate/Bridge/population，并要求五份 2/3 seed gate 全部通过。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.validate_m4_region_encoder_suite --gate crop_only=outputs/.../gate.json
--gate full_image_box=outputs/.../gate.json --gate masked_pooling=outputs/.../gate.json
--gate roi_replay_only=outputs/.../gate.json --gate mgrr_no_context=outputs/.../gate.json
--output outputs/qpsalm_description/m4_region_encoder_suite_gate.json
输入：五份当前 `qpsalm_description_seed_gate_v12_strict_json_finite`、保留原始
eval/retrieval/ERFS input bindings 的三 seed comparison gate。
输出：原子写入五 baseline suite gate；任一比较未通过时返回非零并保留报告。
写入行为：只写 --output，不修改模型、评估目录、benchmark、cache 或 datasets。
所属流程：M4 科学准入；不能以单一 crop-only 比较替代完整六 encoder 消融。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.workflows.gates import (
    run_m4_suite_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all five M4 region-encoder baseline gates."
    )
    parser.add_argument(
        "--gate",
        action="append",
        required=True,
        metavar="ENCODER=PATH",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_m4_suite_gate(args.gate, output=args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if report["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
