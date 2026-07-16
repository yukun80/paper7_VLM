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
from pathlib import Path
import shutil

import torch

from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_preset(load_config(args.segmentation_config), args.preset)
    config = apply_config_overrides(config, {
        "vision_feature_cache": args.vision_feature_cache,
        "train_index": args.train_index,
        "val_index": args.val_index,
        "modality_dropout": 0.0,
        "train_hflip_prob": 0.0,
        "train_vflip_prob": 0.0,
    })
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    try:
        validate_output_replacement_safety(output, {
            "checkpoint": args.checkpoint,
            "source-index": args.source_index,
            "vision-feature-cache": args.vision_feature_cache,
            "train-index": args.train_index,
            "val-index": args.val_index,
            "prediction-index": args.prediction_index,
            "fold-manifest": args.fold_manifest,
        })
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if output.exists() and not output.is_dir():
        raise SystemExit(f"predicted-region output-dir 不是目录: {output}")
    if output.is_dir() and any(output.iterdir()) and not args.overwrite_output:
        raise SystemExit(
            "predicted-region output-dir 已非空；请改用新目录或显式 --overwrite-output"
        )
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    from qpsalm_seg.description.predicted_regions import export_predicted_regions

    report = export_predicted_regions(
        segmentation_config=config,
        checkpoint=args.checkpoint,
        source_index=args.source_index,
        split=args.split,
        output_dir=output,
        device=torch.device(args.device),
        threshold=args.threshold,
        fold_manifest=args.fold_manifest,
        checkpoint_fold=args.checkpoint_fold,
        prediction_index=args.prediction_index,
        max_parents=args.max_parents,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
