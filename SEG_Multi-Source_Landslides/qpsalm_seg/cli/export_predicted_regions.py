#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 D4 使用的离线预测区域。

用途：用固定分割 checkpoint 生成 predicted-mask 描述课程索引；train 强制 out-of-fold。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.export_predicted_regions --segmentation-config
SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --preset qwen_psalm_full
--checkpoint outputs/qpsalm_v2/RUN/checkpoint_best.pt --source-index
benchmark/landslide_region_description_v1_small/indexes/expert_val.jsonl --split val
--output-dir outputs/qpsalm_description/predicted_val --device cuda
主要输出：原尺寸 .npy mask、qpsalm_predicted_region_v2_checkpoint_bound JSONL 和 report.json。
写入行为：只写 --output-dir；train 未提供 fold 审计信息时会拒绝运行。
所属流程：M6 D4 离线 predicted-mask curriculum。
"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.description.workflows.oof import (
    OOFLaunchError,
    run_predicted_region_export,
)
from qpsalm_seg.presets import PRESET_CHOICES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fold-audited predicted regions.")
    parser.add_argument("--segmentation-config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--vision-feature-cache", default=None)
    parser.add_argument("--train-index", default=None, help="Fold checkpoint/cache protocol train index.")
    parser.add_argument("--val-index", default=None, help="Fold checkpoint/cache protocol holdout index.")
    parser.add_argument(
        "--prediction-index", default=None,
        help="Rows to infer; train OOF must be the manifest holdout index.",
    )
    parser.add_argument("--fold-manifest", default=None)
    parser.add_argument("--checkpoint-fold", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = run_predicted_region_export(
            segmentation_config=args.segmentation_config,
            preset=args.preset,
            checkpoint=args.checkpoint,
            source_index=args.source_index,
            split=args.split,
            vision_feature_cache=args.vision_feature_cache,
            train_index=args.train_index,
            val_index=args.val_index,
            prediction_index=args.prediction_index,
            fold_manifest=args.fold_manifest,
            checkpoint_fold=args.checkpoint_fold,
            threshold=args.threshold,
            max_parents=args.max_parents,
            device_name=args.device,
            output_dir=args.output_dir,
            overwrite_output=args.overwrite_output,
        )
    except OOFLaunchError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
