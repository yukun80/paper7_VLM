#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-PSALM-Seg checkpoint 验证入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpsalm_seg.config import load_config, parse_combo_loss_weights
from qpsalm_seg.runtime import torch_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Multi-Source Qwen-PSALM-Seg checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--controller", choices=["qwen", "qwen_cache", "cached_qwen", "text_probe"], default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--allow-qwen-cpu", action="store_true")
    parser.add_argument("--condition-embedding-cache", default=None)
    parser.add_argument("--num-visualizations", type=int, default=None)
    parser.add_argument("--train-hflip-prob", type=float, default=None)
    parser.add_argument("--train-vflip-prob", type=float, default=None)
    parser.add_argument("--use-box-prior", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-focal-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--boundary-loss-weight", type=float, default=None)
    parser.add_argument("--condition-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--selection-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--foreground-bce-pos-weight", type=float, default=None)
    parser.add_argument("--mask-bce-weight", type=float, default=None)
    parser.add_argument("--mask-dice-weight", type=float, default=None)
    parser.add_argument("--mask-tversky-weight", type=float, default=None)
    parser.add_argument("--tversky-alpha", type=float, default=None)
    parser.add_argument("--tversky-beta", type=float, default=None)
    parser.add_argument("--proposal-cls-weight", type=float, default=None)
    parser.add_argument("--condition-cls-weight", type=float, default=None)
    parser.add_argument("--proposal-mask-weight", type=float, default=None)
    parser.add_argument("--empty-mask-suppression-weight", type=float, default=None)
    parser.add_argument("--empty-proposal-suppression-weight", type=float, default=None)
    parser.add_argument("--proposal-positive-weight", type=float, default=None)
    parser.add_argument("--condition-positive-weight", type=float, default=None)
    parser.add_argument("--query-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--proposal-mask-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--gate-entropy-loss-weight", type=float, default=None)
    parser.add_argument("--proposal-soft-target-topk", type=int, default=None)
    parser.add_argument("--proposal-soft-target-temperature", type=float, default=None)
    parser.add_argument("--query-usage-balance-loss-weight", type=float, default=None)
    parser.add_argument("--selection-proposal-weight", type=float, default=None)
    parser.add_argument("--selection-condition-weight", type=float, default=None)
    parser.add_argument("--selection-temperature", type=float, default=None)
    parser.add_argument("--final-foreground-gate-weight", type=float, default=None)
    parser.add_argument(
        "--final-mask-fusion",
        choices=["weighted_average", "topk_weighted_average", "topk_noisy_or"],
        default=None,
    )
    parser.add_argument("--final-topk", type=int, default=None)
    parser.add_argument("--final-noisy-or-epsilon", type=float, default=None)
    parser.add_argument(
        "--canonical-combo-loss-weights",
        default=None,
        help="Canonical combo loss weights, e.g. 'dem+s2=2.5,dem+s1+s2=1.5'.",
    )
    parser.add_argument("--eval-threshold", type=float, default=None)
    parser.add_argument("--torch-timeout", type=int, default=120)
    parser.add_argument("--skip-torch-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config,
        overrides={
            "batch_size": args.batch_size,
            "target_size": args.target_size,
            "num_workers": args.num_workers,
            "max_val_samples": args.max_val_samples,
            "max_val_batches": args.max_val_batches,
            "val_index": args.val_index,
            "output_dir": args.output_dir,
            "controller": args.controller,
            "qwen_model_path": args.qwen_model_path,
            "allow_qwen_cpu": True if args.allow_qwen_cpu else None,
            "condition_embedding_cache": args.condition_embedding_cache,
            "num_visualizations": args.num_visualizations,
            "train_hflip_prob": args.train_hflip_prob,
            "train_vflip_prob": args.train_vflip_prob,
            "use_box_prior": args.use_box_prior,
            "use_focal_loss": args.use_focal_loss,
            "boundary_loss_weight": args.boundary_loss_weight,
            "condition_ranking_loss_weight": args.condition_ranking_loss_weight,
            "selection_ranking_loss_weight": args.selection_ranking_loss_weight,
            "foreground_bce_pos_weight": args.foreground_bce_pos_weight,
            "mask_bce_weight": args.mask_bce_weight,
            "mask_dice_weight": args.mask_dice_weight,
            "mask_tversky_weight": args.mask_tversky_weight,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "proposal_cls_weight": args.proposal_cls_weight,
            "condition_cls_weight": args.condition_cls_weight,
            "proposal_mask_weight": args.proposal_mask_weight,
            "empty_mask_suppression_weight": args.empty_mask_suppression_weight,
            "empty_proposal_suppression_weight": args.empty_proposal_suppression_weight,
            "proposal_positive_weight": args.proposal_positive_weight,
            "condition_positive_weight": args.condition_positive_weight,
            "query_diversity_loss_weight": args.query_diversity_loss_weight,
            "proposal_mask_diversity_loss_weight": args.proposal_mask_diversity_loss_weight,
            "gate_entropy_loss_weight": args.gate_entropy_loss_weight,
            "proposal_soft_target_topk": args.proposal_soft_target_topk,
            "proposal_soft_target_temperature": args.proposal_soft_target_temperature,
            "query_usage_balance_loss_weight": args.query_usage_balance_loss_weight,
            "selection_proposal_weight": args.selection_proposal_weight,
            "selection_condition_weight": args.selection_condition_weight,
            "selection_temperature": args.selection_temperature,
            "final_foreground_gate_weight": args.final_foreground_gate_weight,
            "final_mask_fusion": args.final_mask_fusion,
            "final_topk": args.final_topk,
            "final_noisy_or_epsilon": args.final_noisy_or_epsilon,
            "canonical_combo_loss_weights": parse_combo_loss_weights(args.canonical_combo_loss_weights),
            "eval_threshold": args.eval_threshold,
        },
    )
    if not args.skip_torch_preflight:
        ok, message = torch_preflight(timeout=args.torch_timeout)
        if not ok:
            raise RuntimeError(
                f"PyTorch runtime is not ready: {message}. "
                "Run qpsalm-check-env or `python -m qpsalm_seg.cli.check_env` for details."
            )
        print(f"torch_preflight={message}")
    import torch
    from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate, resolve_repo_path
    from qpsalm_seg.qwen_cache import assert_qwen_cache_coverage
    from qpsalm_seg.train_eval import build_model, evaluate, load_checkpoint, resolve_device, utc_now, write_json

    device = resolve_device(args.device)
    cache_report = assert_qwen_cache_coverage(config, splits=(args.split,))
    if cache_report.get("ok"):
        print(
            "qwen_cache_coverage="
            f"required_texts={cache_report['required']['num_texts']} "
            f"cached_texts={cache_report['cache']['num_texts']} "
            f"backend={cache_report['cache'].get('backend')}"
        )
    ds = MultiSourceLandslideDataset(config, split=args.split, max_samples=config.max_val_samples)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=qpsalm_collate,
    )
    model = build_model(config, device)
    step = load_checkpoint(args.checkpoint, model)
    out_dir = resolve_repo_path(config.output_dir) or Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(
        model,
        loader,
        device,
        max_batches=config.max_val_batches if config.max_val_batches and config.max_val_batches > 0 else None,
        visual_dir=out_dir / "eval_visualizations",
        num_visualizations=config.num_visualizations,
        threshold=float(config.eval_threshold),
        threshold_sweep=config.threshold_sweep,
    )
    report["checkpoint_step"] = step
    write_json(
        out_dir / "eval_manifest.json",
        {
            "created_at_utc": utc_now(),
            "created_by": "qpsalm-eval",
            "eval_dir": str(out_dir),
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": step,
            "device": args.device,
            "split": args.split,
            "eval_report": str(out_dir / "eval_report.json"),
            "resolved_config": dict(config.__dict__),
        },
    )
    (out_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
