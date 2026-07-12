#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Component-set proposal assignment shared by losses and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def calibrated_relevance_gates(logits: torch.Tensor) -> torch.Tensor:
    """Convert per-query relevance logits to query-count-stable union gates."""
    offset = logits.new_tensor(float(max(1, logits.shape[-1] - 1))).log()
    return torch.sigmoid(logits - offset)


@dataclass
class ProposalAssignment:
    component_masks: torch.Tensor
    matched_queries: torch.Tensor
    matched_components: torch.Tensor
    relevance_targets: torch.Tensor
    pair_bce: torch.Tensor
    pair_dice_loss: torch.Tensor
    pair_dice: torch.Tensor
    coverage_mode: bool
    component_recall: torch.Tensor
    component_precision: torch.Tensor
    unmatched_rejection: torch.Tensor
    merge_error_rate: torch.Tensor
    duplicate_error_rate: torch.Tensor
    missed_component_rate: torch.Tensor
    relevance_ap: torch.Tensor
    relevance_auc: torch.Tensor
    proposal_union_dice: torch.Tensor


def component_masks(
    target: torch.Tensor,
    valid: torch.Tensor,
    min_area_fraction: float,
    min_area_pixels: int,
) -> torch.Tensor:
    binary = ((target >= 0.5) & (valid >= 0.5)).detach().cpu().numpy().astype(np.uint8)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    valid_area = int((valid >= 0.5).sum().item())
    threshold = max(int(min_area_pixels), int(round(valid_area * float(min_area_fraction))))
    masks = [
        torch.from_numpy(labels == label_id).to(device=target.device, dtype=target.dtype)
        for label_id in range(1, int(count) + 1)
        if int((labels == label_id).sum()) >= threshold
    ]
    if not masks and bool(binary.any()):
        masks.append(((target >= 0.5) & (valid >= 0.5)).to(target.dtype))
    return torch.stack(masks) if masks else target.new_zeros((0, *target.shape[-2:]))


def pairwise_component_cost(
    logits: torch.Tensor,
    components: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.float()
    components = components.float()
    valid = valid.float()
    probs = torch.sigmoid(logits) * valid
    components = components * valid
    intersection = torch.einsum("qhw,khw->qk", probs, components)
    pred_area = probs.sum((1, 2))[:, None]
    target_area = components.sum((1, 2))[None]
    dice_loss = 1.0 - (2.0 * intersection + 1.0e-6) / (pred_area + target_area + 1.0e-6)
    valid_area = valid.sum().clamp_min(1.0)
    base = (F.softplus(logits) * valid).sum((1, 2)) / valid_area
    positive = torch.einsum("qhw,khw->qk", logits * valid, components) / valid_area
    bce = base[:, None] - positive
    return bce, dice_loss, bce + dice_loss


def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum()
    if positives <= 0:
        return scores.new_tensor(1.0 if (scores < 0).all() else 0.0)
    order = torch.argsort(scores, descending=True)
    ranked = targets[order].float()
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1, device=scores.device)
    return (precision * ranked).sum() / positives


def binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive = scores[targets > 0.5]
    negative = scores[targets <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.new_tensor(0.5)
    comparison = (positive[:, None] > negative[None]).float()
    ties = (positive[:, None] == negative[None]).float() * 0.5
    return (comparison + ties).mean()


def assign_proposals(
    proposal_logits: torch.Tensor,
    relevance_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_area_fraction: float = 5.0e-5,
    min_area_pixels: int = 4,
    match_threshold: float = 0.5,
    precomputed_components: torch.Tensor | None = None,
) -> ProposalAssignment:
    components = (
        precomputed_components.to(device=target.device, dtype=target.dtype)
        if precomputed_components is not None
        else component_masks(target, valid, min_area_fraction, min_area_pixels)
    )
    num_queries = int(proposal_logits.shape[0])
    relevance_target = relevance_logits.new_zeros((num_queries,))
    empty = torch.empty((0,), dtype=torch.long, device=proposal_logits.device)
    if components.shape[0] == 0:
        zero = relevance_logits.new_tensor(0.0)
        rejection = (torch.sigmoid(relevance_logits) < 0.5).float().mean()
        empty_matrix = proposal_logits.new_zeros((num_queries, 0))
        return ProposalAssignment(
            components, empty, empty, relevance_target, empty_matrix, empty_matrix, empty_matrix,
            False, zero, zero, rejection, zero, zero, zero,
            binary_average_precision(relevance_logits, relevance_target),
            binary_auc(relevance_logits, relevance_target),
            zero,
        )
    bce, dice_loss, cost = pairwise_component_cost(proposal_logits, components, valid)
    coverage_mode = int(components.shape[0]) > num_queries
    if coverage_mode:
        matched_components = torch.arange(
            components.shape[0], dtype=torch.long, device=proposal_logits.device
        )
        matched_queries = cost.argmin(dim=0)
    else:
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        matched_queries = torch.as_tensor(rows, dtype=torch.long, device=proposal_logits.device)
        matched_components = torch.as_tensor(cols, dtype=torch.long, device=proposal_logits.device)
    relevance_target[matched_queries] = 1.0
    dice = 1.0 - dice_loss
    covered = dice.max(dim=0).values >= match_threshold
    query_covers = dice >= match_threshold
    predicted_positive = torch.sigmoid(relevance_logits) >= 0.5
    useful_positive = predicted_positive & query_covers.any(dim=1)
    component_recall = covered.float().mean()
    component_precision = useful_positive.sum().float() / predicted_positive.sum().clamp_min(1)
    unmatched = relevance_target < 0.5
    rejection = (
        (torch.sigmoid(relevance_logits[unmatched]) < 0.5).float().mean()
        if unmatched.any() else relevance_logits.new_tensor(1.0)
    )
    merge_error = (query_covers.sum(dim=1) > 1).float().mean()
    duplicate_error = (query_covers.sum(dim=0) > 1).float().mean()
    gates = calibrated_relevance_gates(relevance_logits)[..., None, None]
    proposal_probs = (torch.sigmoid(proposal_logits) * gates).clamp(1.0e-6, 1.0 - 1.0e-6)
    union = 1.0 - torch.prod(1.0 - proposal_probs, dim=0)
    semantic_target = components.amax(dim=0)
    valid_float = valid.to(union.dtype)
    intersection = (union * semantic_target * valid_float).sum()
    union_dice = (2.0 * intersection + 1.0e-6) / (
        (union * valid_float).sum() + (semantic_target * valid_float).sum() + 1.0e-6
    )
    return ProposalAssignment(
        component_masks=components,
        matched_queries=matched_queries,
        matched_components=matched_components,
        relevance_targets=relevance_target,
        pair_bce=bce,
        pair_dice_loss=dice_loss,
        pair_dice=dice,
        coverage_mode=coverage_mode,
        component_recall=component_recall,
        component_precision=component_precision,
        unmatched_rejection=rejection,
        merge_error_rate=merge_error,
        duplicate_error_rate=duplicate_error,
        missed_component_rate=1.0 - component_recall,
        relevance_ap=binary_average_precision(relevance_logits, relevance_target),
        relevance_auc=binary_auc(relevance_logits, relevance_target),
        proposal_union_dice=union_dice,
    )
