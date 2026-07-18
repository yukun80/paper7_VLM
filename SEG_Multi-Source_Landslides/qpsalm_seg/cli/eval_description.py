#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估 segmentation-grounded description checkpoint。

用途：分别执行 GT-mask oracle、固定预测 mask 或端到端分割后描述评价。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_description --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --stage bridge_expert
--seed 42 --checkpoint outputs/qpsalm_description/RUN/checkpoint_best.pt --split val
--evaluation-mode gt_mask --device cuda --output-dir outputs/qpsalm_description/RUN/eval_gt
默认行为：独立评估默认覆盖完整 split 并对全部样本生成；smoke 必须显式传入
--max-val-samples 和 --max-generate-samples 的正整数上限。
主要输出：eval_report.json、raw_generations.jsonl 和 counterfactual_generations.jsonl。
写入行为：只写 --output-dir，不修复或覆盖原始模型输出。
所属流程：M6 描述评价；主结构指标只使用未修复 raw JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.description.protocols.config import (
    DESCRIPTION_EVAL_MODES,
    DESCRIPTION_STAGES,
    load_segdesc_config,
)
from qpsalm_seg.description.workflows.evaluate import (
    DescriptionEvaluationLaunchError,
    run_description_evaluation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation-grounded descriptions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=DESCRIPTION_STAGES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "val", "test"], default="val")
    parser.add_argument("--evaluation-mode", choices=DESCRIPTION_EVAL_MODES, default=None)
    parser.add_argument(
        "--source-dataset",
        choices=["RSIEval"],
        default=None,
        help="仅用于冻结 rsicap_caption/test 的 RSIEval-only population",
    )
    parser.add_argument(
        "--region-source",
        choices=["gt_global_mask"],
        default=None,
        help="M6 GT/end-to-end 可比性所需的 frozen Bridge region-source filter",
    )
    parser.add_argument("--region-protocol", choices=["vision_only", "assisted"], default=None)
    parser.add_argument(
        "--region-encoder",
        choices=[
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        ],
        default=None,
    )
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument("--segmentation-mask-threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--max-val-samples", type=int, default=0,
        help="评估样本上限；0 表示完整 split（独立评估默认）",
    )
    parser.add_argument(
        "--max-generate-samples", type=int, default=0,
        help="生成样本上限；0 表示对全部评估样本生成（独立评估默认）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--counterfactual-samples", type=int, default=None)
    parser.add_argument(
        "--cycle-localization-samples", type=int, default=None,
        help="-1 关闭；0 对全部可定位 expert rows 运行；正整数为上限",
    )
    parser.add_argument("--no-counterfactuals", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_segdesc_config(args.config, {
        "stage": args.stage,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "evaluation_source_dataset": args.source_dataset,
        "evaluation_region_source": args.region_source,
        "region_protocol": args.region_protocol,
        "region_encoder": args.region_encoder,
        "predicted_index": args.predicted_index,
        "segmentation_mask_threshold": args.segmentation_mask_threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_val_samples": args.max_val_samples,
        "max_generate_samples": args.max_generate_samples,
        "max_new_tokens": args.max_new_tokens,
        "counterfactual_samples": args.counterfactual_samples,
        "cycle_localization_samples": args.cycle_localization_samples,
        "output_dir": args.output_dir,
    })
    try:
        report = run_description_evaluation(
            config,
            config_reference=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            device_name=args.device,
            run_counterfactuals=not args.no_counterfactuals,
            overwrite_output=args.overwrite_output,
        )
    except DescriptionEvaluationLaunchError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({
        "eval_report": str(Path(config.training.output_dir) / "eval_report.json"),
        "checkpoint_step": report["checkpoint_step"],
        "stage": config.training.stage,
        "split": args.split,
        "mode": config.evaluation.evaluation_mode,
        "num_samples": report["num_samples"],
        "generation_coverage": report["generation_coverage"],
        "generation_metrics": report["generation_metrics"],
        "same_image_retrieval": report["same_image_retrieval"],
        "cycle_localization": report["cycle_localization"],
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
