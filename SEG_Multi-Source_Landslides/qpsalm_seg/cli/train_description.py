#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 segmentation-grounded description 模型。

用途：运行 D-1、D0-D4 的独立描述训练，保存 qpsalm_segdesc_v1 best/last 权重。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.train_description --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --stage overfit
--seed 42 --device cuda --max-steps 100
--output-dir outputs/qpsalm_description/overfit_seed42 --overwrite-output
主要输入：已验证的 M1/M2 benchmark、description vision cache v1 和分割 checkpoint。
主要输出：checkpoint_best.pt、checkpoint_last.pt、validation 与 raw generation 报告。
写入行为：只写 --output-dir；不会改写 benchmark、cache 或分割 checkpoint。
所属流程：M6 描述训练；所有命令由用户手动运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import traceback

from qpsalm_seg.description.config import DESCRIPTION_STAGES, load_segdesc_config
from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation-grounded region description.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=DESCRIPTION_STAGES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--region-protocol", choices=["vision_only", "assisted"], default=None)
    parser.add_argument(
        "--region-encoder",
        choices=[
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        ],
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-generate-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument(
        "--predicted-val-index",
        default=None,
        help="D4 固定 val prediction index；不得与 OOF train index 混用",
    )
    parser.add_argument(
        "--d4-curriculum-gate",
        default=None,
        help="前一档 fixed expert-val 通过后发布的相邻升档 gate",
    )
    parser.add_argument(
        "--d-minus-one-gate",
        default=None,
        help="D0 必需的当前 D-1 v7 统一工程门禁",
    )
    parser.add_argument(
        "--predicted-mask-fraction",
        type=float,
        choices=[0.25, 0.50, 0.75],
        default=None,
        help="D4 预注册 predicted-mask curriculum tier",
    )
    parser.add_argument(
        "--d4-curriculum-sampling-seed",
        type=int,
        default=None,
        help="跨模型 seed 固定 D4 predicted-row population 的独立非负 seed",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--initialize-from", default=None, help="Load model weights only for a new D-stage.")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite_output:
        raise SystemExit("--resume 不能与 --overwrite-output 同时使用")
    if args.resume and args.initialize_from:
        raise SystemExit("--resume 不能与 --initialize-from 同时使用")
    config = load_segdesc_config(args.config, {
        "stage": args.stage,
        "seed": args.seed,
        "region_protocol": args.region_protocol,
        "region_encoder": args.region_encoder,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_generate_samples": args.max_generate_samples,
        "learning_rate": args.learning_rate,
        "amp_dtype": args.amp_dtype,
        "val_interval": args.val_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "predicted_index": args.predicted_index,
        "predicted_val_index": args.predicted_val_index,
        "d_minus_one_gate": args.d_minus_one_gate,
        "d4_curriculum_gate": args.d4_curriculum_gate,
        "predicted_mask_fraction": args.predicted_mask_fraction,
        "d4_curriculum_sampling_seed": args.d4_curriculum_sampling_seed,
        "output_dir": args.output_dir,
    })
    output = resolve_project_path(config.output_dir) or Path(config.output_dir)
    output_resolved = output.resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise SystemExit(f"description training output-dir 不是目录: {output}")
    protected = {
        "config": args.config,
        "segmentation-config": config.segmentation_config,
        "segmentation-checkpoint": config.segmentation_checkpoint,
        "segmentation-vision-cache": config.segmentation_vision_cache,
        "description-vision-cache": config.description_vision_cache,
        "description-benchmark": config.description_benchmark,
        "bridge-benchmark": config.bridge_benchmark,
        "initialize-from": args.initialize_from,
        "predicted-index": config.predicted_index,
        "predicted-val-index": config.predicted_val_index,
        "d-minus-one-gate": config.d_minus_one_gate,
        "d4-curriculum-gate": config.d4_curriculum_gate,
    }
    try:
        validate_output_replacement_safety(output, protected)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.resume:
        source = resolve_project_path(args.resume) or Path(args.resume)
        if not output.is_dir() or source.resolve(strict=False).parent != output_resolved:
            raise SystemExit(
                "--resume 必须使用同一非空 output-dir 内的 checkpoint，"
                "以保留历史、数据审计和失败记录"
            )
    elif output.is_dir() and any(output.iterdir()) and not args.overwrite_output:
        raise SystemExit("新 run 的 output-dir 已非空；请改用新目录或显式 --overwrite-output")
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    from qpsalm_seg.description.trainer import (
        DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        train_description,
    )
    from qpsalm_seg.description.common import write_json
    from qpsalm_seg.description.run_artifacts import (
        DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
        build_training_completion_report,
        prepare_training_attempt,
        validate_terminal_checkpoint_provenance,
    )
    from qpsalm_seg.description.checkpoint import inspect_segdesc_checkpoint

    attempt_audit = prepare_training_attempt(output, resume=bool(args.resume))

    try:
        report = train_description(
            config,
            device_name=args.device,
            resume=args.resume,
            initialize_from=args.initialize_from,
        )
        optional_artifacts = {
            "validation_best": output / "validation_best.json",
            "d_minus_one_overfit_validation": (
                output / "d_minus_one_overfit_validation.json"
            ),
        }
        if report.get("checkpoint_best"):
            optional_artifacts["checkpoint_best"] = report["checkpoint_best"]
        checkpoint_provenance = inspect_segdesc_checkpoint(
            report["checkpoint_last"]
        )
        terminal_checkpoint_audit = validate_terminal_checkpoint_provenance(
            checkpoint_provenance,
            checkpoint=report["checkpoint_last"],
            expected_step=int(report["steps"]),
            expected_stage=config.stage,
            progress_key="training_progress",
            expected_progress_protocol=DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
            progress_artifact=output / "training_progress_latest.json",
            progress_artifact_name="training_progress_latest",
            history_artifact=output / "train_history.jsonl",
            history_artifact_name="train_history",
        )
        completion = build_training_completion_report(
            protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
            report={
                **report,
                "attempt_audit": attempt_audit,
                "terminal_checkpoint_audit": terminal_checkpoint_audit,
            },
            required_artifacts={
                "checkpoint_last": report["checkpoint_last"],
                "dataset_summary": output / "dataset_summary.json",
                "resolved_config": output / "resolved_config.json",
                "train_history": output / "train_history.jsonl",
                "training_progress_latest": (
                    output / "training_progress_latest.json"
                ),
                "trainable_parameter_manifest": (
                    output / "trainable_parameter_manifest.json"
                ),
            },
            optional_artifacts=optional_artifacts,
        )
        write_json(output / "training_report.json", completion)
    except BaseException as exc:
        failure = {
            "protocol": "qpsalm_description_training_failure_v2_attempt_bound",
            "stage": config.stage,
            "attempt_audit": attempt_audit,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure_report.json", failure)
        raise
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
