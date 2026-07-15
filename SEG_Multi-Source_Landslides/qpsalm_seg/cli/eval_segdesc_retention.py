#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行 M7 full-val segmentation retention 门禁。

用途：在与基线相同的完整 val 上评估 joint checkpoint，验证 positive Dice 下降不超过 0.01。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_segdesc_retention --config
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml --checkpoint
outputs/qpsalm_description/JOINT/checkpoint_best.pt --baseline-eval-report
outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json
--device cuda --output-dir outputs/qpsalm_description/JOINT/retention_full_val
主要输出：joint_segmentation_eval.json 和 retention_gate.json。
写入行为：只写 --output-dir；不以 monitor 指标替代 full-val 门禁。
注意：--max-samples 仅供 smoke；非零时只报告 preliminary_passed，正式 passed 恒为 false。
所属流程：M7 最终验收。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.engine.evaluator import SAMPLE_IDENTITY_FIELDS, SAMPLE_IDENTITY_PROTOCOL
from qpsalm_seg.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-val segmentation retention.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-eval-report", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _positive_dice(report: dict) -> float:
    return float((((report.get("metrics") or {}).get("positive_only") or {}).get("dice")) or 0.0)


def _num_samples(report: dict) -> int:
    return int((report.get("coverage") or {}).get("num_samples") or 0)


def _sample_population(report: dict) -> dict:
    value = (report.get("coverage") or {}).get("sample_population") or {}
    return value if isinstance(value, dict) else {}


def build_retention_gate(
    baseline: dict,
    report: dict,
    *,
    split: str,
    max_samples: int,
    checkpoint: str,
    checkpoint_step: int,
    checkpoint_metadata: dict,
    maximum_allowed_drop: float,
) -> dict:
    """Build the formal gate without accepting count-only population matching."""
    baseline_dice = _positive_dice(baseline)
    current_dice = _positive_dice(report)
    drop = baseline_dice - current_dice
    baseline_n = _num_samples(baseline)
    current_n = _num_samples(report)
    baseline_threshold = float(baseline.get("threshold", 0.5))
    current_threshold = float(report.get("threshold", 0.5))
    baseline_population = _sample_population(baseline)
    current_population = _sample_population(report)
    baseline_population_hash = str(baseline_population.get("sha256") or "")
    current_population_hash = str(current_population.get("sha256") or "")
    population_schema_valid = bool(
        baseline_population.get("protocol") == SAMPLE_IDENTITY_PROTOCOL
        and current_population.get("protocol") == SAMPLE_IDENTITY_PROTOCOL
        and tuple(baseline_population.get("fields") or ()) == tuple(SAMPLE_IDENTITY_FIELDS)
        and tuple(current_population.get("fields") or ()) == tuple(SAMPLE_IDENTITY_FIELDS)
    )
    population_protocol_match = population_schema_valid
    population_identity_valid = bool(
        baseline_population.get("complete")
        and baseline_population.get("unique")
        and current_population.get("complete")
        and current_population.get("unique")
    )
    population_counts_valid = bool(
        int(baseline_population.get("num_records", -1)) == baseline_n
        and int(baseline_population.get("num_unique_sample_ids", -1)) == baseline_n
        and int(current_population.get("num_records", -1)) == current_n
        and int(current_population.get("num_unique_sample_ids", -1)) == current_n
    )
    same_sample_population = bool(
        population_protocol_match
        and population_identity_valid
        and population_counts_valid
        and baseline_population_hash
        and baseline_population_hash == current_population_hash
    )
    full_split = int(max_samples) == 0
    same_population_size = baseline_n > 0 and current_n == baseline_n
    same_threshold = abs(current_threshold - baseline_threshold) <= 1.0e-12
    checkpoint_stage = str((checkpoint_metadata.get("metadata") or {}).get("stage") or "")
    joint_checkpoint = checkpoint_stage == "joint"
    preliminary_passed = drop <= float(maximum_allowed_drop)
    scientific_gate_eligible = (
        split == "val"
        and full_split
        and same_population_size
        and same_sample_population
        and same_threshold
        and joint_checkpoint
    )
    return {
        "protocol": "qpsalm_segdesc_retention_v3",
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "checkpoint_metadata": checkpoint_metadata,
        "split": split,
        "baseline_num_samples": baseline_n,
        "joint_num_samples": current_n,
        "full_split_requested": full_split,
        "same_population_size": same_population_size,
        "baseline_sample_population": baseline_population,
        "joint_sample_population": current_population,
        "population_protocol_match": population_protocol_match,
        "population_schema_valid": population_schema_valid,
        "population_identity_valid": population_identity_valid,
        "population_counts_valid": population_counts_valid,
        "same_sample_population": same_sample_population,
        "baseline_threshold": baseline_threshold,
        "joint_threshold": current_threshold,
        "same_threshold": same_threshold,
        "joint_checkpoint": joint_checkpoint,
        "baseline_positive_dice": baseline_dice,
        "joint_positive_dice": current_dice,
        "absolute_drop": drop,
        "maximum_allowed_drop": float(maximum_allowed_drop),
        "preliminary_passed": preliminary_passed,
        "scientific_gate_eligible": scientific_gate_eligible,
        "passed": scientific_gate_eligible and preliminary_passed,
    }


def main() -> None:
    args = parse_args()
    config = load_segdesc_config(args.config)
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if args.overwrite_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = resolve_project_path(args.baseline_eval_report) or Path(args.baseline_eval_report)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    from qpsalm_seg.description.checkpoint import load_segdesc_checkpoint
    from qpsalm_seg.description.common import description_device, write_json
    from qpsalm_seg.description.runtime import build_segdesc_model
    from qpsalm_seg.engine.common import build_eval_loader
    from qpsalm_seg.engine.evaluator import evaluate

    device = description_device(args.device)
    model, _migration = build_segdesc_model(config, device)
    step, metadata = load_segdesc_checkpoint(args.checkpoint, model)
    segmentation_config = replace(
        model.segmentation.config,
        max_val_samples=args.max_samples or None,
    )
    loader = build_eval_loader(segmentation_config, args.split)
    with model.controller.adapter_scope("default"):
        report = evaluate(
            model.segmentation,
            loader,
            device,
            threshold=segmentation_config.eval_threshold,
        )
    gate = build_retention_gate(
        baseline,
        report,
        split=args.split,
        max_samples=args.max_samples,
        checkpoint=args.checkpoint,
        checkpoint_step=step,
        checkpoint_metadata=metadata,
        maximum_allowed_drop=config.segmentation_retention_max_drop,
    )
    write_json(output / "joint_segmentation_eval.json", report)
    write_json(output / "retention_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()
