#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate all required QPSALM instruction/visual ablations in one model load.

用途：同一 checkpoint 单进程运行 normal、三种 instruction 和多种 visual evidence 消融。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval_ablation_suite --config CONFIG --preset qwen_psalm_full
--checkpoint CHECKPOINT --vision-feature-cache CACHE --device cuda
--visual-remove terrain --visual-remove sar --output-dir outputs/RUN/ablation_suite
主要输入：benchmark-v2 val/test、同一 checkpoint 和匹配的 vision cache v3。
主要输出：每个条件的 eval_report/manifest 与 ablation_evidence.json。
写入行为：写入 --output-dir；--overwrite-output 会清空该目录。
所属流程：真实 integration 通过后、三 seed 模块准入之前的 Qwen 真实性评估。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from qpsalm_seg.cli.ablation_report import build_ablation_report, load_eval_bundle
from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset
from qpsalm_seg.runtime import torch_preflight
from qpsalm_seg.schema import MODALITY_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the strict QPSALM ablation suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="qwen_psalm_full")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--vision-feature-cache", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--eval-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visual-remove", action="append", required=True, choices=MODALITY_FAMILIES)
    parser.add_argument("--include-image-text-delta", action="store_true")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--torch-timeout", type=int, default=120)
    parser.add_argument("--skip-torch-preflight", action="store_true")
    return parser.parse_args()


def _condition_name(instruction: str, visual: str) -> str:
    if instruction != "normal":
        return f"instruction_{instruction.replace('-', '_')}"
    if visual != "normal":
        return f"visual_{visual.replace(':', '_').replace('-', '_')}"
    return "normal"


def main() -> None:
    args = parse_args()
    if not args.skip_torch_preflight:
        ok, message = torch_preflight(timeout=args.torch_timeout)
        if not ok:
            raise RuntimeError(f"PyTorch runtime is not ready: {message}")
        print(f"torch_preflight={message}")

    from qpsalm_seg.engine.checkpoint import load_checkpoint
    from qpsalm_seg.engine.common import (
        build_eval_loader,
        build_model,
        resolve_device,
        set_seed,
        utc_now,
        write_json,
    )
    from qpsalm_seg.engine.evaluator import evaluate
    from qpsalm_seg.paths import resolve_project_path

    config = apply_preset(load_config(args.config), args.preset)
    config = apply_config_overrides(config, {
        "benchmark_dir": args.benchmark_dir,
        "vision_feature_cache": args.vision_feature_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_val_samples": args.max_val_samples,
        "max_val_batches": args.max_val_batches,
        "eval_threshold": args.eval_threshold,
        "seed": args.seed,
    })
    config = replace(config, instruction_ablation="normal", visual_ablation="normal")
    set_seed(config.seed)
    device = resolve_device(args.device)
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if output.exists():
        if not args.overwrite_output:
            raise FileExistsError(f"ablation suite output exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    model = build_model(config, device)
    checkpoint_step = load_checkpoint(args.checkpoint, model)

    removals = sorted(set(args.visual_remove))
    conditions = [
        ("normal", "normal"),
        ("shuffled", "normal"),
        ("fixed-generic", "normal"),
        ("no-semantic", "normal"),
        ("normal", "shuffled"),
        ("normal", "text-only"),
        *(("normal", f"remove:{family}") for family in removals),
    ]
    if args.include_image_text_delta:
        conditions.append(("normal", "image-text-delta"))
    bundles = {}
    for instruction, visual in conditions:
        condition_config = replace(
            config,
            instruction_ablation=instruction,
            visual_ablation=visual,
        )
        if hasattr(model.controller, "visual_ablation"):
            model.controller.visual_ablation = visual
        if model.vision_bank is not None:
            model.vision_bank.set_visual_ablation(visual)
        loader = build_eval_loader(condition_config, args.split)
        report = evaluate(
            model,
            loader,
            device,
            max_batches=(
                condition_config.max_val_batches
                if condition_config.max_val_batches and condition_config.max_val_batches > 0
                else None
            ),
            visual_dir=None,
            num_visualizations=0,
            threshold=float(condition_config.eval_threshold),
            threshold_sweep=condition_config.threshold_sweep,
        )
        report["checkpoint_step"] = checkpoint_step
        name = _condition_name(instruction, visual)
        condition_dir = output / name
        condition_dir.mkdir(parents=True)
        write_json(condition_dir / "eval_report.json", report)
        write_json(condition_dir / "eval_manifest.json", {
            "created_at_utc": utc_now(),
            "created_by": "qpsalm-eval-ablation-suite",
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "split": args.split,
            "preset": condition_config.preset,
            "resolved_config": dict(condition_config.__dict__),
        })
        bundles[(instruction, visual)] = load_eval_bundle(condition_dir)
        overall = (report.get("metrics") or {}).get("overall") or {}
        print(json.dumps({
            "condition": name,
            "iou": overall.get("iou"),
            "dice": overall.get("dice"),
            "n": overall.get("n"),
        }, ensure_ascii=False))

    normal = bundles[("normal", "normal")]
    instruction_bundles = {
        name: bundles[(name, "normal")]
        for name in ("shuffled", "fixed-generic", "no-semantic")
    }
    visual_bundles = {
        "shuffled": (bundles[("normal", "shuffled")], None),
        "text-only": (bundles[("normal", "text-only")], None),
        **{
            f"remove:{family}": (bundles[("normal", f"remove:{family}")], family)
            for family in removals
        },
    }
    strategy = bundles.get(("normal", "image-text-delta"))
    evidence = build_ablation_report(
        normal,
        instruction_bundles,
        visual_bundles,
        strategy_bundle=strategy,
        min_delta=args.min_delta,
    )
    write_json(output / "ablation_evidence.json", evidence)
    print(json.dumps({
        "output": str(output / "ablation_evidence.json"),
        **evidence["acceptance"],
    }, ensure_ascii=False))
    if not evidence["acceptance"]["passed"]:
        raise SystemExit("ablation evidence gate failed")


if __name__ == "__main__":
    main()
