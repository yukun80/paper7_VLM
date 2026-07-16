#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并并深度重放 D4 各 held-out fold 的 train predictions。

用途：将各 fold predicted JSONL 合并为唯一 OOF train index，并重放源分区、
segmentation checkpoint、Vision Cache v3 index 指纹、expert source row 和 mask。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.merge_oof_predictions --fold-manifest
outputs/qpsalm_description/oof_folds_small/fold_manifest.json --input
outputs/qpsalm_description/predicted_fold_0/predicted_train_0.jsonl --input
outputs/qpsalm_description/predicted_fold_1/predicted_train_1.jsonl --input
outputs/qpsalm_description/predicted_fold_2/predicted_train_2.jsonl --output
outputs/qpsalm_description/predicted_train_oof.jsonl
主要输出：OOF JSONL 及同目录、同 stem 的深度 validation report。
写入行为：只原子写入 --output 与其 report；不移动 mask 或修改 fold 结果。
所属流程：M6 D4。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.predicted_regions import merge_oof_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and replay-audit train OOF predictions"
    )
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge_oof_predictions(
        fold_manifest=args.fold_manifest,
        input_indexes=list(args.input),
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
