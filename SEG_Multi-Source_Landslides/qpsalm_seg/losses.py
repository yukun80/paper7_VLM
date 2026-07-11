#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact valid-region losses for proposal-set landslide segmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

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


def _component_masks(
    target: torch.Tensor,
    valid: torch.Tensor,
    min_area_fraction: float,
    min_area_pixels: int,
) -> torch.Tensor:
    from scipy import ndimage

    binary = ((target >= 0.5) & (valid >= 0.5)).detach().cpu().numpy().astype(np.uint8)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    valid_area = int((valid >= 0.5).sum().item())
    threshold = max(int(min_area_pixels), int(round(valid_area * float(min_area_fraction))))
    components = [
        torch.from_numpy(labels == label_id).to(device=target.device, dtype=target.dtype)
        for label_id in range(1, int(count) + 1)
        if int((labels == label_id).sum()) >= threshold
    ]
    if not components and bool(binary.any()):
        components.append(((target >= 0.5) & (valid >= 0.5)).to(target.dtype))
    return torch.stack(components) if components else target.new_zeros((0, *target.shape[-2:]))


def _pairwise_component_cost(
    logits: torch.Tensor,
    components: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = valid.to(logits.dtype)
    probs = torch.sigmoid(logits) * valid
    components = components * valid
    intersection = torch.einsum("qhw,khw->qk", probs, components)
    pred_area = probs.sum(dim=(1, 2))[:, None]
    target_area = components.sum(dim=(1, 2))[None]
    dice = 1.0 - (2.0 * intersection + 1.0e-6) / (pred_area + target_area + 1.0e-6)
    valid_area = valid.sum().clamp_min(1.0)
    base = (F.softplus(logits) * valid).sum(dim=(1, 2)) / valid_area
    positive = torch.einsum("qhw,khw->qk", logits * valid, components) / valid_area
    bce = base[:, None] - positive
    return bce, dice, bce + dice


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
    """Final mask + hybrid set matching + one verifier + missing-modality consistency."""
    from scipy.optimize import linear_sum_assignment

    final_logits = output.final_mask_logits
    proposal_masks = output.proposals.mask_logits
    coarse_masks = output.proposals.coarse_mask_logits
    relevance_logits = output.proposals.relevance_logits
    target = batch.mask.to(final_logits.device)
    valid = batch.valid_mask.to(final_logits.device)
    batch_size, num_queries = relevance_logits.shape
    final_bce = weighted_bce_per_sample_with_logits(final_logits, target, valid_mask=valid).mean()
    final_dice = dice_loss_per_sample_with_logits(final_logits, target, valid_mask=valid).mean()
    zero = final_logits.sum() * 0.0
    proposal_terms: list[torch.Tensor] = []
    coarse_terms: list[torch.Tensor] = []
    coverage_terms: list[torch.Tensor] = []
    verifier_terms: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    component_counts = final_logits.new_zeros((batch_size,))
    coverage_modes = final_logits.new_zeros((batch_size,))
    full_dice_scores = dice_scores_with_logits(proposal_masks, target, valid_mask=valid)
    best_query = full_dice_scores.argmax(dim=1)

    for batch_index in range(batch_size):
        relevance_target = torch.zeros_like(relevance_logits[batch_index])
        components = _component_masks(
            target[batch_index, 0],
            valid[batch_index, 0],
            min_component_area_fraction,
            min_component_area_pixels,
        )
        component_count = int(components.shape[0])
        component_counts[batch_index] = float(component_count)
        if component_count:
            bce_cost, dice_cost, matching_cost = _pairwise_component_cost(
                proposal_masks[batch_index], components, valid[batch_index, 0]
            )
            rows, cols = linear_sum_assignment(matching_cost.detach().float().cpu().numpy())
            row_index = torch.as_tensor(rows, dtype=torch.long, device=final_logits.device)
            col_index = torch.as_tensor(cols, dtype=torch.long, device=final_logits.device)
            relevance_target = relevance_target.scatter(0, row_index, 1.0)
            proposal_terms.append((bce_cost[row_index, col_index] + dice_cost[row_index, col_index]).mean())
            coarse_bce, coarse_dice, _ = _pairwise_component_cost(
                coarse_masks[batch_index], components, valid[batch_index, 0]
            )
            coarse_terms.append((coarse_bce[row_index, col_index] + coarse_dice[row_index, col_index]).mean())
            if component_count > num_queries:
                coverage_modes[batch_index] = 1.0
                coverage_terms.append(1.0 - (1.0 - dice_cost).max(dim=0).values.mean())
                relevance_target = torch.ones_like(relevance_target)
        verifier_terms.append(
            F.binary_cross_entropy_with_logits(
                relevance_logits[batch_index],
                relevance_target,
                pos_weight=relevance_logits.new_tensor(2.0),
            )
        )
        target_rows.append(relevance_target)

    relevance_targets = torch.stack(target_rows)
    refined_proposal = torch.stack(proposal_terms).mean() if proposal_terms else zero
    coarse_proposal = torch.stack(coarse_terms).mean() if coarse_terms else zero
    proposal_set = refined_proposal + float(coarse_proposal_weight) * coarse_proposal
    coverage = torch.stack(coverage_terms).mean() if coverage_terms else zero
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
    positive = component_counts > 0
    relevance_top = relevance_logits.argmax(1)
    best_query_accuracy = (
        (relevance_top[positive] == best_query[positive]).float().mean()
        if positive.any()
        else zero
    )
    positive_target_accuracy = (
        relevance_targets[positive].gather(1, relevance_top[positive][:, None]).float().mean()
        if positive.any()
        else zero
    )
    best_query_dice = full_dice_scores.gather(1, best_query[:, None]).squeeze(1)
    return {
        "loss": total,
        "loss_mask_bce": final_bce.detach(),
        "loss_mask_dice": final_dice.detach(),
        "loss_proposal_set": proposal_set.detach(),
        "loss_proposal_coarse": coarse_proposal.detach(),
        "loss_proposal_coverage": coverage.detach(),
        "loss_semantic_verifier": verifier.detach(),
        "loss_boundary": boundary.detach(),
        "loss_missing_modality_consistency": consistency.detach(),
        "best_query": best_query.detach(),
        "best_query_dice": best_query_dice.detach(),
        "verifier_best_query_accuracy": best_query_accuracy.detach(),
        "verifier_positive_target_accuracy": positive_target_accuracy.detach(),
        "proposal_target_mass": relevance_targets.sum(1).detach(),
        "proposal_target_positive_count": relevance_targets.sum(1).detach(),
        "proposal_component_count": component_counts.detach(),
        "proposal_matching_coverage_mode": coverage_modes.detach(),
    }
