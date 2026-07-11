#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 SANE/QMEF/PMRD 模型。

用途：按 YAML runtime 配置和 Python preset 执行训练、周期验证、checkpoint 与可视化。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.train --config
SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml
--preset sane_qmef_pmrd --device cuda --condition-embedding-cache CACHE.pt
--train-index TRAIN.jsonl --val-index VAL.jsonl --output-dir outputs/RUN --skip-torch-preflight
主要输入：配置、preset、核心 train/val 索引和 Qwen 文本/可选视觉缓存。
主要输出：checkpoint_best.pt、checkpoint_last.pt、训练日志、验证报告与 mask 可视化。
写入行为：写入 --output-dir；--overwrite-output 会清空该运行目录。
所属流程：主模型训练；通常优先使用 scripts/run_qwen_phase1_full.sh 编排完整流程。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset
from qpsalm_seg.runtime import torch_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Source Qwen-PSALM-Seg.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None, help="Fallback canvas when the preset disables size buckets.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--controller", choices=["qwen", "qwen_cache", "cached_qwen", "text_probe"], default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--condition-embedding-cache", default=None)
    parser.add_argument("--visual-evidence-cache", default=None)
    parser.add_argument("--allow-qwen-cpu", action="store_true")
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-recent-checkpoints", type=int, default=None)
    parser.add_argument("--visualize-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--modality-dropout", type=float, default=None)
    parser.add_argument("--boundary-loss-weight", type=float, default=None)
    parser.add_argument("--missing-modality-consistency-weight", type=float, default=None)
    parser.add_argument("--eval-threshold", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--torch-timeout", type=int, default=120)
    parser.add_argument("--skip-torch-preflight", action="store_true")
    parser.add_argument("--print-full-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_preset(load_config(args.config), args.preset)
    config = apply_config_overrides(
        config,
        {
            "benchmark_dir": args.benchmark_dir,
            "batch_size": args.batch_size,
            "target_size": args.target_size,
            "num_workers": args.num_workers,
            "max_steps": args.max_steps,
            "num_epochs": args.num_epochs,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_val_batches": args.max_val_batches,
            "train_index": args.train_index,
            "val_index": args.val_index,
            "output_dir": args.output_dir,
            "controller": args.controller,
            "qwen_model_path": args.qwen_model_path,
            "condition_embedding_cache": args.condition_embedding_cache,
            "visual_evidence_cache": args.visual_evidence_cache,
            "allow_qwen_cpu": True if args.allow_qwen_cpu else None,
            "val_interval": args.val_interval,
            "save_interval": args.save_interval,
            "keep_recent_checkpoints": args.keep_recent_checkpoints,
            "visualize_interval": args.visualize_interval,
            "log_interval": args.log_interval,
            "num_visualizations": args.num_visualizations,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "grad_accum_steps": args.grad_accum_steps,
            "seed": args.seed,
            "modality_dropout": args.modality_dropout,
            "boundary_loss_weight": args.boundary_loss_weight,
            "missing_modality_consistency_weight": args.missing_modality_consistency_weight,
            "eval_threshold": args.eval_threshold,
        },
    )
    if not args.skip_torch_preflight:
        ok, message = torch_preflight(timeout=args.torch_timeout)
        if not ok:
            raise RuntimeError(f"PyTorch runtime is not ready: {message}")
        print(f"torch_preflight={message}")
    if args.overwrite_output:
        output_path = config.output_path()
        if output_path.exists():
            shutil.rmtree(output_path)
    from qpsalm_seg.train_eval import train

    result = train(config, device_name=args.device, resume=args.resume)
    if args.print_full_report:
        payload = result
    else:
        history = result.get("history") or []
        payload = {
            "output_dir": result.get("output_dir"),
            "steps": result.get("steps"),
            "last_train": history[-1] if history else None,
        }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
