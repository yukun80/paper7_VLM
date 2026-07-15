#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 D4 train predicted-mask 使用的 parent-level OOF folds。

用途：按 dataset 与 modality family 分层，将 Bridge train parent 划入固定 holdout fold，
并为每个 fold 发布排除该 fold 的 segmentation train index。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.build_oof_folds --segmentation-index
benchmark/multisource_landslide_v2_small/indexes/instruction_train.jsonl --bridge-index
benchmark/landslide_region_description_v1_small/indexes/expert_train.jsonl --num-folds 3
--output-dir outputs/qpsalm_description/oof_folds_small --overwrite-output
主要输出：fold_manifest.json、fold_<n>_train.jsonl 和 fold_<n>_holdout.jsonl。
写入行为：只写 --output-dir；不会训练模型或改写 benchmark。
所属流程：M6 D4；只在 expert_train 冻结后运行，每个 fold checkpoint 必须使用对应
fold_<n>_train.jsonl 从头训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from qpsalm_seg.description.oof import build_oof_fold_indexes
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited segmentation OOF folds")
    parser.add_argument("--segmentation-index", required=True)
    parser.add_argument("--bridge-index", required=True)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if output.exists():
        if not args.overwrite_output:
            raise FileExistsError(f"output 已存在，使用 --overwrite-output: {output}")
        shutil.rmtree(output)
    report = build_oof_fold_indexes(
        segmentation_index=args.segmentation_index,
        bridge_index=args.bridge_index,
        output_dir=output,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    print(json.dumps({
        "manifest": str(output / "fold_manifest.json"),
        "num_parents": report["num_parents"],
        "num_folds": report["num_folds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
