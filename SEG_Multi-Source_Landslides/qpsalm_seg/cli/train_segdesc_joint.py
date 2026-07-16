#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交替训练 segmentation 与 description 双 Adapter。

用途：运行 M7 三 DataLoader 交替训练并执行 monitor segmentation retention gate。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.train_segdesc_joint --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --seed 42 --device cuda
--initialize-from outputs/qpsalm_description/M6/checkpoint_best.pt
--region-stage predicted_mask --predicted-mask-fraction 0.75
--d4-final-acceptance-gate outputs/qpsalm_description/M6/d4_final_m7_gate.json
--m6-acceptance-gate outputs/qpsalm_description/M6/m6_acceptance_gate.json
--output-dir outputs/qpsalm_description/joint_seed42 --overwrite-output
主要输入：冻结的 M1/M2 benchmark、description cache、原分割 checkpoint。
主要输出：qpsalm_segdesc_v1 best/last、三任务历史和 monitor retention 报告。
写入行为：只写 --output-dir；不会构造混合 collate 或改写源 benchmark。
所属流程：M7；需先完成专家 Bridge 与 M6 Small 门禁。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import traceback

from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint segmentation-description alternating trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--segmentation-batch-size", type=int, default=None)
    parser.add_argument("--description-batch-size", type=int, default=None)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="每个 optimizer step 对同一任务累积的 microbatch 数。",
    )
    parser.add_argument(
        "--train-shared-segmentation-dense",
        action="store_true",
        default=None,
        help="Ablation only: also update SANE/QMEF/PMRD and controller projections.",
    )
    parser.add_argument("--region-stage", choices=["bridge_auto", "bridge_expert", "predicted_mask"], default=None)
    parser.add_argument("--predicted-index", default=None)
    parser.add_argument(
        "--predicted-val-index",
        default=None,
        help="M7 predicted-mask region loader 的固定 val prediction index",
    )
    parser.add_argument(
        "--d4-final-acceptance-gate",
        default=None,
        help="75% predicted tier 自身通过 fixed expert-val 后发布的 M7 gate",
    )
    parser.add_argument(
        "--m6-acceptance-gate",
        default=None,
        help="GT/fixed/end-to-end、cycle 与 D4 final 共同通过的 M6 gate",
    )
    parser.add_argument(
        "--predicted-mask-fraction",
        type=float,
        choices=[0.75],
        default=None,
        help="M7 predicted-mask 主路线固定沿用已验收的 D4 75% tier",
    )
    parser.add_argument(
        "--d4-curriculum-sampling-seed",
        type=int,
        default=None,
        help="必须与 D4 三档相同的固定 predicted-row selection seed",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--initialize-from", required=False, help="Required for a new M7 run: load the accepted M6 model weights.")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite_output:
        raise SystemExit("--resume 不能与 --overwrite-output 同时使用")
    if args.resume and args.initialize_from:
        raise SystemExit("--resume 不能与 --initialize-from 同时使用")
    if not args.resume and not args.initialize_from:
        raise SystemExit("新 M7 run 必须提供 --initialize-from；续训请使用 --resume")
    config = load_segdesc_config(args.config, {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "val_interval": args.val_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "joint_segmentation_batch_size": args.segmentation_batch_size,
        "joint_description_batch_size": args.description_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "joint_train_shared_segmentation_dense": args.train_shared_segmentation_dense,
        "joint_region_stage": args.region_stage,
        "predicted_index": args.predicted_index,
        "predicted_val_index": args.predicted_val_index,
        "d4_final_acceptance_gate": args.d4_final_acceptance_gate,
        "m6_acceptance_gate": args.m6_acceptance_gate,
        "predicted_mask_fraction": args.predicted_mask_fraction,
        "d4_curriculum_sampling_seed": args.d4_curriculum_sampling_seed,
        "output_dir": args.output_dir,
    })
    output = resolve_project_path(config.output_dir) or Path(config.output_dir)
    if output.exists() and not output.is_dir():
        raise SystemExit(f"M7 output-dir 不是目录: {output}")
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
        "d4-final-acceptance-gate": config.d4_final_acceptance_gate,
        "m6-acceptance-gate": config.m6_acceptance_gate,
    }
    try:
        validate_output_replacement_safety(output, protected)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    if args.resume:
        source = resolve_project_path(args.resume) or Path(args.resume)
        if not output.is_dir() or source.resolve(strict=False).parent != output.resolve(strict=False):
            raise SystemExit(
                "--resume 必须使用同一非空 output-dir 内的 checkpoint，"
                "以保留 baseline、progress 和历史"
            )
    elif output.is_dir() and any(output.iterdir()) and not args.overwrite_output:
        raise SystemExit("新 M7 run 的 output-dir 已非空；请改用新目录或显式 --overwrite-output")
    output.mkdir(parents=True, exist_ok=True)
    from qpsalm_seg.description.joint_trainer import (
        JOINT_PROGRESS_PROTOCOL,
        train_joint_segdesc,
    )
    from qpsalm_seg.description.common import write_json
    from qpsalm_seg.description.run_artifacts import (
        JOINT_TRAINING_COMPLETION_PROTOCOL,
        build_training_completion_report,
        prepare_training_attempt,
        validate_terminal_checkpoint_provenance,
    )
    from qpsalm_seg.description.checkpoint import inspect_segdesc_checkpoint

    attempt_audit = prepare_training_attempt(output, resume=bool(args.resume))

    try:
        report = train_joint_segdesc(
            config,
            device_name=args.device,
            resume=args.resume,
            initialize_from=args.initialize_from,
        )
        optional_artifacts = {
            "joint_gradient_gate": output / "joint_gradient_gate.json",
            "joint_validation_latest": output / "joint_validation_latest.json",
        }
        if report.get("checkpoint_best"):
            optional_artifacts["checkpoint_best"] = report["checkpoint_best"]
            optional_artifacts["validation_best"] = (
                output / "joint_validation_best.json"
            )
        checkpoint_provenance = inspect_segdesc_checkpoint(
            report["checkpoint_last"]
        )
        terminal_checkpoint_audit = validate_terminal_checkpoint_provenance(
            checkpoint_provenance,
            checkpoint=report["checkpoint_last"],
            expected_step=int(report["steps"]),
            expected_stage="joint",
            progress_key="joint_progress",
            expected_progress_protocol=JOINT_PROGRESS_PROTOCOL,
            progress_artifact=output / "joint_coverage_latest.json",
            progress_artifact_name="joint_coverage_latest",
            history_artifact=output / "joint_history.jsonl",
            history_artifact_name="joint_history",
        )
        completion = build_training_completion_report(
            protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
            report={
                **report,
                "attempt_audit": attempt_audit,
                "terminal_checkpoint_audit": terminal_checkpoint_audit,
            },
            required_artifacts={
                "checkpoint_last": report["checkpoint_last"],
                "joint_coverage_latest": output / "joint_coverage_latest.json",
                "joint_history": output / "joint_history.jsonl",
                "joint_manifest": output / "joint_manifest.json",
                "segmentation_monitor_baseline": (
                    output / "segmentation_monitor_baseline.json"
                ),
            },
            optional_artifacts=optional_artifacts,
        )
        write_json(output / "training_report.json", completion)
    except BaseException as exc:
        write_json(output / "failure_report.json", {
            "protocol": "qpsalm_segdesc_joint_training_failure_v2_attempt_bound",
            "region_stage": config.joint_region_stage,
            "attempt_audit": attempt_audit,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
