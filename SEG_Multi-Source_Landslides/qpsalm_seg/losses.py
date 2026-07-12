#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact valid-region losses for proposal-set landslide segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qpsalm_seg.matching import assign_proposals, pairwise_component_cost
from qpsalm_seg.schema import ModalityBatch, SegmentationOutput


def _valid_like(valid_mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference)
    valid = valid_mask.to(device=reference.device, dtype=reference.dtype)
    if valid.shape[-2:] != reference.shape[-2:]:
        valid = F.interpolate(valid, size=reference.shape[-2:], mode="nearest")
    if valid.ndim != reference.ndim:
        raise ValueError(f"valid_mask/reference 维数不一致: {valid.shape} vs {reference.shape}")
    if valid.shape[1] == 1 and reference.shape[1] != 1:
        valid = valid.expand(-1, reference.shape[1], -1, -1)
    return valid


def _masked_mean_per_sample(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    valid = _valid_like(valid_mask, values)
    dims = tuple(range(1, values.ndim))
    return (values * valid).sum(dim=dims) / valid.sum(dim=dims).clamp_min(1.0e-6)


def dice_loss_per_sample_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    valid = _valid_like(valid_mask, probs)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target * valid).sum(dim=dims)
    denominator = (probs * valid).sum(dim=dims) + (target * valid).sum(dim=dims)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def dice_scores_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if target.shape[1] == 1 and probs.shape[1] != 1:
        target = target.expand(-1, probs.shape[1], -1, -1)
    valid = _valid_like(valid_mask, probs)
    intersection = (probs * target * valid).sum(dim=(2, 3))
    denominator = (probs * valid).sum(dim=(2, 3)) + (target * valid).sum(dim=(2, 3))
    return (2.0 * intersection + eps) / (denominator + eps)


def weighted_bce_per_sample_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if pos_weight > 1.0:
        loss = loss * torch.where(target >= 0.5, loss.new_tensor(float(pos_weight)), loss.new_tensor(1.0))
    return _masked_mean_per_sample(loss, valid_mask)


def boundary_loss_per_sample_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    kernel_size: int = 3,
) -> torch.Tensor:
    pad = kernel_size // 2
    probs = torch.sigmoid(logits)
    target = (target >= 0.5).to(probs.dtype)
    pred_boundary = (
        F.max_pool2d(probs, kernel_size, stride=1, padding=pad)
        + F.max_pool2d(-probs, kernel_size, stride=1, padding=pad)
    ).clamp(0.0, 1.0)
    target_boundary = (
        F.max_pool2d(target, kernel_size, stride=1, padding=pad)
        + F.max_pool2d(-target, kernel_size, stride=1, padding=pad)
    ).clamp(0.0, 1.0)
    boundary_valid = valid_mask
    if valid_mask is not None:
        valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
        boundary_valid = 1.0 - F.max_pool2d(1.0 - valid, kernel_size, stride=1, padding=pad)
    return _masked_mean_per_sample((pred_boundary - target_boundary).abs(), boundary_valid)


def proposal_set_losses(
    output: SegmentationOutput,
    batch: ModalityBatch,
    *,
    final_bce_weight: float = 1.0,
    final_dice_weight: float = 1.0,
    proposal_set_weight: float = 0.75,
    coarse_proposal_weight: float = 0.25,
    verifier_weight: float = 0.25,
    boundary_weight: float = 0.0,
    missing_modality_consistency: torch.Tensor | None = None,
    consistency_weight: float = 0.0,
    min_component_area_fraction: float = 5.0e-5,
    min_component_area_pixels: int = 4,
) -> dict[str, torch.Tensor]:
    """Final mask + component-set assignment + unified verifier + subset consistency."""

    final_logits = output.final_mask_logits
    proposal_masks = output.proposals.mask_logits
    coarse_masks = output.proposals.coarse_mask_logits
    relevance_logits = output.proposals.relevance_logits
    target = batch.mask.to(final_logits.device)
    valid = batch.valid_mask.to(final_logits.device)
    batch_size, _num_queries = relevance_logits.shape
    final_bce = weighted_bce_per_sample_with_logits(final_logits, target, valid_mask=valid).mean()
    final_dice = dice_loss_per_sample_with_logits(final_logits, target, valid_mask=valid).mean()
    zero = final_logits.sum() * 0.0
    proposal_terms: list[torch.Tensor] = []
    coarse_terms: list[torch.Tensor] = []
    coverage_terms: list[torch.Tensor] = []
    coarse_coverage_terms: list[torch.Tensor] = []
    verifier_terms: list[torch.Tensor] = []
    verifier_pos_weights: list[torch.Tensor] = []
    assignments = []

    for batch_index in range(batch_size):
        assignment = assign_proposals(
            proposal_masks[batch_index],
            relevance_logits[batch_index],
            target[batch_index, 0],
            valid[batch_index, 0],
            min_area_fraction=min_component_area_fraction,
            min_area_pixels=min_component_area_pixels,
        )
        assignments.append(assignment)
        if assignment.matched_queries.numel():
            rows, cols = assignment.matched_queries, assignment.matched_components
            coarse_bce, coarse_dice, coarse_cost = pairwise_component_cost(
                coarse_masks[batch_index], assignment.component_masks, valid[batch_index, 0]
            )
            if assignment.coverage_mode:
                refined_cost = assignment.pair_bce + assignment.pair_dice_loss
                coverage_terms.append(refined_cost.min(dim=0).values.mean())
                coarse_coverage_terms.append(coarse_cost.min(dim=0).values.mean())
            else:
                proposal_terms.append(
                    (assignment.pair_bce[rows, cols] + assignment.pair_dice_loss[rows, cols]).mean()
                )
                coarse_terms.append((coarse_bce[rows, cols] + coarse_dice[rows, cols]).mean())
        positive_count = assignment.relevance_targets.sum()
        negative_count = assignment.relevance_targets.numel() - positive_count
        pos_weight = (
            (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 8.0)
            if positive_count > 0 else relevance_logits.new_tensor(1.0)
        )
        verifier_pos_weights.append(pos_weight)
        verifier_terms.append(
            F.binary_cross_entropy_with_logits(
                relevance_logits[batch_index],
                assignment.relevance_targets,
                pos_weight=pos_weight,
            )
        )

    relevance_targets = torch.stack([item.relevance_targets for item in assignments])
    refined_proposal = torch.stack(proposal_terms).mean() if proposal_terms else zero
    coarse_proposal = torch.stack(coarse_terms).mean() if coarse_terms else zero
    proposal_set = refined_proposal + float(coarse_proposal_weight) * coarse_proposal
    refined_coverage = torch.stack(coverage_terms).mean() if coverage_terms else zero
    coarse_coverage = torch.stack(coarse_coverage_terms).mean() if coarse_coverage_terms else zero
    coverage = refined_coverage + float(coarse_proposal_weight) * coarse_coverage
    verifier = torch.stack(verifier_terms).mean() if verifier_terms else zero
    boundary = boundary_loss_per_sample_with_logits(final_logits, target, valid).mean() if boundary_weight > 0.0 else zero
    consistency = missing_modality_consistency if missing_modality_consistency is not None else zero
    total = (
        float(final_bce_weight) * final_bce
        + float(final_dice_weight) * final_dice
        + float(proposal_set_weight) * (proposal_set + coverage)
        + float(verifier_weight) * verifier
        + float(boundary_weight) * boundary
        + float(consistency_weight) * consistency
    )
    def stack_metric(name: str) -> torch.Tensor:
        return torch.stack([getattr(item, name) for item in assignments]).detach()

    matched_mean_dice = []
    oracle_matched_queries = []
    oracle_matched_dice = []
    for item in assignments:
        if item.matched_queries.numel():
            matched_values = item.pair_dice[item.matched_queries, item.matched_components]
            matched_mean_dice.append(matched_values.mean())
            best_pair = int(torch.argmax(matched_values).item())
            oracle_matched_queries.append(item.matched_queries[best_pair])
            oracle_matched_dice.append(matched_values[best_pair])
        else:
            matched_mean_dice.append(zero)
            oracle_matched_queries.append(
                torch.tensor(-1, device=final_logits.device, dtype=torch.long)
            )
            oracle_matched_dice.append(zero)
    return {
        "loss": total,
        "loss_mask_bce": final_bce.detach(),
        "loss_mask_dice": final_dice.detach(),
        "loss_proposal_set": proposal_set.detach(),
        "loss_proposal_coarse": coarse_proposal.detach(),
        "loss_proposal_coverage": coverage.detach(),
        "loss_proposal_coverage_coarse": coarse_coverage.detach(),
        "loss_semantic_verifier": verifier.detach(),
        "loss_boundary": boundary.detach(),
        "loss_missing_modality_consistency": consistency.detach(),
        "proposal_matched_mean_dice": torch.stack(matched_mean_dice).detach(),
        "proposal_oracle_matched_query": torch.stack(oracle_matched_queries).detach(),
        "proposal_oracle_matched_dice": torch.stack(oracle_matched_dice).detach(),
        "proposal_component_recall": stack_metric("component_recall"),
        "proposal_component_precision": stack_metric("component_precision"),
        "proposal_unmatched_rejection": stack_metric("unmatched_rejection"),
        "proposal_relevance_ap": stack_metric("relevance_ap"),
        "proposal_relevance_auc": stack_metric("relevance_auc"),
        "proposal_union_dice": stack_metric("proposal_union_dice"),
        "proposal_merge_error_rate": stack_metric("merge_error_rate"),
        "proposal_duplicate_error_rate": stack_metric("duplicate_error_rate"),
        "proposal_missed_component_rate": stack_metric("missed_component_rate"),
        "proposal_relevance_targets": relevance_targets.detach(),
        "proposal_target_mass": relevance_targets.sum(1).detach(),
        "proposal_target_positive_count": relevance_targets.sum(1).detach(),
        "proposal_verifier_pos_weight": torch.stack(verifier_pos_weights).detach(),
        "proposal_component_count": final_logits.new_tensor(
            [float(item.component_masks.shape[0]) for item in assignments]
        ),
        "proposal_matching_coverage_mode": final_logits.new_tensor(
            [float(item.coverage_mode) for item in assignments]
        ),
    }
