#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-PSALM-Seg 训练入口。"""

from __future__ import annotations

import argparse
import json

from qpsalm_seg.config import (
    LOSS_STAGE_CHOICES,
    apply_config_overrides,
    apply_loss_stage,
    load_config,
    parse_combo_loss_weights,
)
from qpsalm_seg.runtime import torch_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Source Qwen-PSALM-Seg.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--controller", choices=["qwen", "qwen_cache", "cached_qwen", "text_probe"], default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--allow-qwen-cpu", action="store_true")
    parser.add_argument("--condition-embedding-cache", default=None)
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
    parser.add_argument("--evidence-positive-weight", type=float, default=None)
    parser.add_argument("--query-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--proposal-mask-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--gate-entropy-loss-weight", type=float, default=None)
    parser.add_argument("--proposal-soft-target-topk", type=int, default=None)
    parser.add_argument("--proposal-soft-target-temperature", type=float, default=None)
    parser.add_argument("--query-usage-balance-loss-weight", type=float, default=None)
    parser.add_argument("--evidence-cls-weight", type=float, default=None)
    parser.add_argument("--evidence-ranking-loss-weight", type=float, default=None)
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
    parser.add_argument("--modality-dropout", type=float, default=None)
    parser.add_argument("--use-gsd-film", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-spatial-modality-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-query-modality-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--query-modality-feature-weight", type=float, default=None)
    parser.add_argument("--use-evidence-reasoning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evidence-reasoning-weight", type=float, default=None)
    parser.add_argument("--selection-evidence-weight", type=float, default=None)
    parser.add_argument("--use-visual-evidence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-evidence-cache", default=None)
    parser.add_argument("--visual-evidence-weight", type=float, default=None)
    parser.add_argument("--visual-evidence-feature-weight", type=float, default=None)
    parser.add_argument("--loss-stage", choices=LOSS_STAGE_CHOICES, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--torch-timeout", type=int, default=120)
    parser.add_argument("--skip-torch-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "batch_size": args.batch_size,
        "target_size": args.target_size,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_val_batches": args.max_val_batches,
        "train_index": args.train_index,
        "val_index": args.val_index,
        "output_dir": args.output_dir,
        "controller": args.controller,
        "qwen_model_path": args.qwen_model_path,
        "allow_qwen_cpu": True if args.allow_qwen_cpu else None,
        "condition_embedding_cache": args.condition_embedding_cache,
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
        "evidence_positive_weight": args.evidence_positive_weight,
        "query_diversity_loss_weight": args.query_diversity_loss_weight,
        "proposal_mask_diversity_loss_weight": args.proposal_mask_diversity_loss_weight,
        "gate_entropy_loss_weight": args.gate_entropy_loss_weight,
        "proposal_soft_target_topk": args.proposal_soft_target_topk,
        "proposal_soft_target_temperature": args.proposal_soft_target_temperature,
        "query_usage_balance_loss_weight": args.query_usage_balance_loss_weight,
        "evidence_cls_weight": args.evidence_cls_weight,
        "evidence_ranking_loss_weight": args.evidence_ranking_loss_weight,
        "selection_proposal_weight": args.selection_proposal_weight,
        "selection_condition_weight": args.selection_condition_weight,
        "selection_temperature": args.selection_temperature,
        "final_foreground_gate_weight": args.final_foreground_gate_weight,
        "final_mask_fusion": args.final_mask_fusion,
        "final_topk": args.final_topk,
        "final_noisy_or_epsilon": args.final_noisy_or_epsilon,
        "canonical_combo_loss_weights": parse_combo_loss_weights(args.canonical_combo_loss_weights),
        "eval_threshold": args.eval_threshold,
        "modality_dropout": args.modality_dropout,
        "use_gsd_film": args.use_gsd_film,
        "use_spatial_modality_gate": args.use_spatial_modality_gate,
        "use_query_modality_attention": args.use_query_modality_attention,
        "query_modality_feature_weight": args.query_modality_feature_weight,
        "use_evidence_reasoning": args.use_evidence_reasoning,
        "evidence_reasoning_weight": args.evidence_reasoning_weight,
        "selection_evidence_weight": args.selection_evidence_weight,
        "use_visual_evidence": args.use_visual_evidence,
        "visual_evidence_cache": args.visual_evidence_cache,
        "visual_evidence_weight": args.visual_evidence_weight,
        "visual_evidence_feature_weight": args.visual_evidence_feature_weight,
    }
    config = load_config(args.config)
    config = apply_loss_stage(config, args.loss_stage or config.loss_stage)
    config = apply_config_overrides(config, overrides)
    if not args.skip_torch_preflight:
        ok, message = torch_preflight(timeout=args.torch_timeout)
        if not ok:
            raise RuntimeError(
                f"PyTorch runtime is not ready: {message}. "
                "Run qpsalm-check-env or `python -m qpsalm_seg.cli.check_env` for details."
            )
        print(f"torch_preflight={message}")
    from qpsalm_seg.train_eval import train

    result = train(config, device_name=args.device, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
