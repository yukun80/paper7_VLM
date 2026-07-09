#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-PSALM-Seg 损失函数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """二值 Dice loss，适配滑坡小目标不平衡。"""
    return dice_loss_per_sample_with_logits(logits, target, eps=eps).mean()


def dice_loss_per_sample_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """返回每个样本的二值 Dice loss，形状为 [B]。"""
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tversky loss：beta 更大时更重罚漏检，适合滑坡小目标低 recall 场景。"""
    return tversky_loss_per_sample_with_logits(logits, target, alpha=alpha, beta=beta, eps=eps).mean()


def tversky_loss_per_sample_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """返回每个样本的 Tversky loss，形状为 [B]。"""
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - score


def dice_scores_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """返回每个 proposal 与 GT 的 soft Dice score。"""
    probs = torch.sigmoid(logits)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.expand(-1, probs.shape[1], -1, -1)
    inter = (probs * target).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (2.0 * inter + eps) / (denom + eps)


def per_query_weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """返回每个 query 的 mask BCE，形状为 [B,Q]。"""
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.expand(-1, logits.shape[1], -1, -1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if pos_weight > 1.0:
        weight = torch.ones_like(loss)
        weight = torch.where(target >= 0.5, weight * float(pos_weight), weight)
        loss = loss * weight
    return loss.mean(dim=(2, 3))


def tversky_losses_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """返回每个 query 的 Tversky loss，形状为 [B,Q]。"""
    probs = torch.sigmoid(logits)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.expand(-1, probs.shape[1], -1, -1)
    tp = (probs * target).sum(dim=(2, 3))
    fp = (probs * (1.0 - target)).sum(dim=(2, 3))
    fn = ((1.0 - probs) * target).sum(dim=(2, 3))
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - score


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """可选 focal loss。"""
    return focal_loss_per_sample_with_logits(logits, target, gamma=gamma, alpha=alpha).mean()


def focal_loss_per_sample_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """返回每个样本的 focal loss，形状为 [B]。"""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    weight = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (weight * (1.0 - pt).pow(gamma) * bce).mean(dim=tuple(range(1, logits.ndim)))


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """前景加权 BCE；pos_weight>1 可缓解小目标被背景压制。"""
    return weighted_bce_per_sample_with_logits(logits, target, pos_weight=pos_weight).mean()


def weighted_bce_per_sample_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """返回每个样本的前景加权 BCE，形状为 [B]。"""
    if pos_weight <= 1.0:
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return loss.mean(dim=tuple(range(1, logits.ndim)))
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.ones_like(loss)
    weight = torch.where(target >= 0.5, weight * float(pos_weight), weight)
    return (loss * weight).mean(dim=tuple(range(1, logits.ndim)))


def boundary_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """轻量边界损失：比较预测概率与 GT 的 soft boundary map。"""
    return boundary_loss_per_sample_with_logits(logits, target, kernel_size=kernel_size).mean()


def boundary_loss_per_sample_with_logits(logits: torch.Tensor, target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """返回每个样本的轻量边界损失，形状为 [B]。"""
    pad = kernel_size // 2
    probs = torch.sigmoid(logits)
    target = (target >= 0.5).to(probs.dtype)
    pred_max = F.max_pool2d(probs, kernel_size=kernel_size, stride=1, padding=pad)
    pred_min = -F.max_pool2d(-probs, kernel_size=kernel_size, stride=1, padding=pad)
    gt_max = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=pad)
    gt_min = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=pad)
    pred_boundary = (pred_max - pred_min).clamp(0, 1)
    gt_boundary = (gt_max - gt_min).clamp(0, 1)
    return (pred_boundary - gt_boundary).abs().mean(dim=tuple(range(1, logits.ndim)))


def normalized_sample_weights(
    sample_weights: torch.Tensor | None,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    """返回均值约为 1 的 [B] 样本权重，避免整体 loss 尺度随配置漂移。"""
    if sample_weights is None:
        return torch.ones((batch_size,), dtype=dtype, device=device)
    weights = sample_weights.to(device=device, dtype=dtype).view(-1)
    if weights.numel() != batch_size:
        return torch.ones((batch_size,), dtype=dtype, device=device)
    weights = torch.where(torch.isfinite(weights) & (weights > 0), weights, torch.ones_like(weights))
    return weights / weights.mean().clamp_min(eps)


def weighted_sample_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """按 [B] 权重对 per-sample loss 求均值。"""
    values = values.view(-1)
    weights = weights.to(device=values.device, dtype=values.dtype).view(-1)
    if values.numel() != weights.numel():
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def query_diversity_loss(query_embeddings: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """轻量 query 正交正则，缓解多个 mask token 学成同一 proposal。"""
    if query_embeddings.ndim != 3 or query_embeddings.shape[1] <= 1:
        return query_embeddings.sum() * 0.0
    q = F.normalize(query_embeddings.float(), dim=-1, eps=eps)
    sim = torch.matmul(q, q.transpose(1, 2))
    eye = torch.eye(sim.shape[-1], dtype=torch.bool, device=sim.device).unsqueeze(0)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return sim.sum() * 0.0
    return off_diag.pow(2).mean().to(query_embeddings.dtype)


def proposal_mask_diversity_loss(
    pred_masks: torch.Tensor,
    is_empty: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """约束非空样本的 proposal mask 不要全部学成同一空间响应。"""
    if pred_masks.ndim != 4 or pred_masks.shape[1] <= 1:
        return pred_masks.sum() * 0.0
    non_empty = (is_empty <= 0.5).view(-1)
    if not non_empty.any():
        return pred_masks.sum() * 0.0
    probs = torch.sigmoid(pred_masks[non_empty]).flatten(2).float()
    probs = probs - probs.mean(dim=-1, keepdim=True)
    probs = F.normalize(probs, dim=-1, eps=eps)
    sim = torch.matmul(probs, probs.transpose(1, 2))
    eye = torch.eye(sim.shape[-1], dtype=torch.bool, device=sim.device).unsqueeze(0)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return pred_masks.sum() * 0.0
    return off_diag.clamp_min(0.0).pow(2).mean().to(pred_masks.dtype)


def modality_gate_entropy_loss(
    gate_weights: torch.Tensor | None,
    active_mask: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """鼓励多模态样本的 gate 保持可用证据多样性，缓解早期单模态塌缩。"""
    if gate_weights is None or active_mask is None:
        return None
    if gate_weights.ndim != 2 or active_mask.ndim != 2:
        return gate_weights.sum() * 0.0
    active = active_mask.to(gate_weights.device, dtype=gate_weights.dtype)
    multi_source = active.sum(dim=1) > 1.5
    if not multi_source.any():
        return gate_weights.sum() * 0.0
    weights = (gate_weights[multi_source] * active[multi_source]).float().clamp_min(0.0)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = weights / denom
    entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1)
    max_entropy = active[multi_source].float().sum(dim=1).clamp_min(2.0).log()
    normalized_entropy = entropy / max_entropy.clamp_min(eps)
    return (1.0 - normalized_entropy).mean().to(gate_weights.dtype)


def build_soft_proposal_targets(
    dice_scores: torch.Tensor,
    non_empty: torch.Tensor,
    topk: int = 1,
    temperature: float = 0.10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """按 proposal-GT Dice 构造 top-k soft targets，缓解单 query 正样本塌缩。"""
    targets = torch.zeros_like(dice_scores)
    if dice_scores.ndim != 2 or dice_scores.shape[1] == 0 or not non_empty.any():
        return targets
    k = max(1, min(int(topk), int(dice_scores.shape[1])))
    rows = torch.nonzero(non_empty, as_tuple=False).flatten()
    scores = dice_scores.detach()[rows]
    top_values, top_indices = torch.topk(scores, k=k, dim=1)
    if k == 1:
        soft_values = torch.ones_like(top_values)
    else:
        temp = max(float(temperature), eps)
        soft_values = torch.softmax(top_values / temp, dim=1)
        soft_values = soft_values / soft_values.max(dim=1, keepdim=True).values.clamp_min(eps)
    targets[rows.unsqueeze(1), top_indices] = soft_values.to(targets.dtype)
    return targets


def soft_target_ranking_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    non_empty: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """用 soft proposal targets 监督 query ranking；top-k=1 时退化为 hard CE。"""
    if not non_empty.any():
        return logits.sum() * 0.0
    return soft_target_ranking_loss_per_sample(logits, soft_targets, non_empty, eps=eps).mean()


def soft_target_ranking_loss_per_sample(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    non_empty: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """返回非空样本的 query ranking loss，形状为 [N_non_empty]。"""
    if not non_empty.any():
        return logits.new_zeros((0,))
    target = soft_targets[non_empty].float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    log_prob = F.log_softmax(logits[non_empty].float(), dim=1)
    return -(target * log_prob).sum(dim=1).to(logits.dtype)


def query_usage_balance_loss(
    selection_weights: torch.Tensor | None,
    non_empty: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """batch 级 query 使用均衡正则，缓解 selection 长期集中到单个 mask token。"""
    if selection_weights is None:
        return None, None
    if selection_weights.ndim != 2 or selection_weights.shape[1] <= 1:
        zero = selection_weights.sum() * 0.0
        return zero, zero
    if not non_empty.any():
        zero = selection_weights.sum() * 0.0
        return zero, zero
    weights = selection_weights[non_empty].float().clamp_min(eps)
    usage = weights.mean(dim=0)
    usage = usage / usage.sum().clamp_min(eps)
    entropy = -(usage * usage.clamp_min(eps).log()).sum()
    max_entropy = usage.new_tensor(float(usage.numel())).log().clamp_min(eps)
    normalized_entropy = entropy / max_entropy
    loss = 1.0 - normalized_entropy
    return loss.to(selection_weights.dtype), normalized_entropy.to(selection_weights.dtype)


def segmentation_losses(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    is_empty: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    use_focal: bool = False,
    boundary_weight: float = 0.0,
    condition_ranking_weight: float = 0.1,
    foreground_bce_pos_weight: float = 1.0,
    mask_bce_weight: float = 1.0,
    mask_dice_weight: float = 1.0,
    mask_tversky_weight: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    proposal_cls_weight: float = 0.2,
    condition_cls_weight: float = 0.2,
    proposal_mask_weight: float = 0.5,
    empty_mask_suppression_weight: float = 0.0,
    empty_proposal_suppression_weight: float = 0.0,
    proposal_positive_weight: float = 1.0,
    condition_positive_weight: float = 1.0,
    query_diversity_loss_weight: float = 0.0,
    selection_ranking_loss_weight: float = 0.0,
    proposal_mask_diversity_loss_weight: float = 0.0,
    gate_entropy_loss_weight: float = 0.0,
    proposal_soft_target_topk: int = 1,
    proposal_soft_target_temperature: float = 0.10,
    query_usage_balance_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """组合最终 mask、proposal mask 和 proposal/condition 分类损失。"""
    final_logits = outputs["final_mask_logits"]
    pred_masks = outputs["pred_masks"]
    proposal_logits = outputs["proposal_logits"]
    condition_scores = outputs["condition_scores"]
    weights = normalized_sample_weights(
        sample_weights,
        batch_size=int(target_mask.shape[0]),
        dtype=final_logits.dtype,
        device=final_logits.device,
    )

    mask_bce_per_sample = (
        focal_loss_per_sample_with_logits(final_logits, target_mask)
        if use_focal
        else weighted_bce_per_sample_with_logits(final_logits, target_mask, pos_weight=float(foreground_bce_pos_weight))
    )
    mask_bce = weighted_sample_mean(mask_bce_per_sample, weights)
    mask_dice = weighted_sample_mean(dice_loss_per_sample_with_logits(final_logits, target_mask), weights)
    mask_tversky = (
        weighted_sample_mean(
            tversky_loss_per_sample_with_logits(
                final_logits,
                target_mask,
                alpha=float(tversky_alpha),
                beta=float(tversky_beta),
            ),
            weights,
        )
        if mask_tversky_weight and mask_tversky_weight > 0
        else final_logits.sum() * 0.0
    )

    dice_scores = dice_scores_with_logits(pred_masks, target_mask)
    best_query = dice_scores.argmax(dim=1)
    non_empty = (is_empty <= 0.5).view(-1)
    proposal_target = build_soft_proposal_targets(
        dice_scores,
        non_empty,
        topk=int(proposal_soft_target_topk),
        temperature=float(proposal_soft_target_temperature),
    )
    proposal_class_weight = (
        proposal_logits.new_tensor(float(proposal_positive_weight))
        if proposal_positive_weight and proposal_positive_weight > 1.0
        else None
    )
    proposal_fg_logits = proposal_logits[..., 1] - proposal_logits[..., 0]
    proposal_cls_per_query = F.binary_cross_entropy_with_logits(
        proposal_fg_logits,
        proposal_target,
        pos_weight=proposal_class_weight,
        reduction="none",
    )
    proposal_cls = weighted_sample_mean(proposal_cls_per_query.mean(dim=1), weights)
    condition_pos_weight = (
        condition_scores.new_tensor(float(condition_positive_weight))
        if condition_positive_weight and condition_positive_weight > 1.0
        else None
    )
    condition_cls_per_query = F.binary_cross_entropy_with_logits(
        condition_scores,
        proposal_target,
        pos_weight=condition_pos_weight,
        reduction="none",
    )
    condition_cls = weighted_sample_mean(condition_cls_per_query.mean(dim=1), weights)
    if non_empty.any():
        condition_rank_per_sample = soft_target_ranking_loss_per_sample(condition_scores, proposal_target, non_empty)
        condition_rank = weighted_sample_mean(condition_rank_per_sample, weights[non_empty])
        condition_rank_acc = (
            condition_scores[non_empty].argmax(dim=1) == best_query[non_empty]
        ).float().mean()
    else:
        condition_rank = final_logits.sum() * 0.0
        condition_rank_acc = final_logits.sum() * 0.0

    selection_logits = outputs.get("selection_logits")
    if (
        selection_logits is not None
        and selection_ranking_loss_weight
        and selection_ranking_loss_weight > 0
        and non_empty.any()
    ):
        selection_rank_per_sample = soft_target_ranking_loss_per_sample(selection_logits, proposal_target, non_empty)
        selection_rank = weighted_sample_mean(selection_rank_per_sample, weights[non_empty])
        selection_rank_acc = (
            selection_logits[non_empty].argmax(dim=1) == best_query[non_empty]
        ).float().mean()
    else:
        selection_rank = final_logits.sum() * 0.0
        selection_rank_acc = final_logits.sum() * 0.0
    selection_weights = outputs.get("selection_weights")
    if selection_weights is None and selection_logits is not None:
        selection_weights = torch.softmax(selection_logits, dim=1)
    usage_balance_value, usage_entropy_value = query_usage_balance_loss(selection_weights, non_empty)
    query_usage_balance = (
        usage_balance_value
        if usage_balance_value is not None
        and query_usage_balance_loss_weight
        and query_usage_balance_loss_weight > 0
        else final_logits.sum() * 0.0
    )
    query_usage_entropy = (
        usage_entropy_value
        if usage_entropy_value is not None
        else final_logits.sum() * 0.0
    )

    if non_empty.any():
        proposal_mass = proposal_target[non_empty].sum(dim=1).clamp_min(1.0e-6)
        mask_bce_per_query = per_query_weighted_bce_with_logits(
            pred_masks,
            target_mask,
            pos_weight=float(foreground_bce_pos_weight),
        )
        mask_dice_per_query = 1.0 - dice_scores
        if mask_tversky_weight and mask_tversky_weight > 0:
            mask_tversky_per_query = tversky_losses_with_logits(
                pred_masks,
                target_mask,
                alpha=float(tversky_alpha),
                beta=float(tversky_beta),
            )
        else:
            mask_tversky_per_query = None
        proposal_mask_bce_per_sample = (
            mask_bce_per_query[non_empty] * proposal_target[non_empty]
        ).sum(dim=1).div(proposal_mass)
        proposal_mask_dice_per_sample = (
            mask_dice_per_query[non_empty] * proposal_target[non_empty]
        ).sum(dim=1).div(proposal_mass)
        proposal_mask_bce = weighted_sample_mean(proposal_mask_bce_per_sample, weights[non_empty])
        proposal_mask_dice = weighted_sample_mean(proposal_mask_dice_per_sample, weights[non_empty])
        if mask_tversky_weight and mask_tversky_weight > 0:
            proposal_mask_tversky_per_sample = (
                mask_tversky_per_query[non_empty] * proposal_target[non_empty]
            ).sum(dim=1).div(proposal_mass)
            proposal_mask_tversky = weighted_sample_mean(proposal_mask_tversky_per_sample, weights[non_empty])
        else:
            proposal_mask_tversky = final_logits.sum() * 0.0
    else:
        proposal_mask_bce = final_logits.sum() * 0.0
        proposal_mask_dice = final_logits.sum() * 0.0
        proposal_mask_tversky = final_logits.sum() * 0.0

    empty = ~non_empty
    if empty.any():
        empty_mask_per_sample = torch.sigmoid(final_logits[empty]).mean(dim=tuple(range(1, final_logits.ndim)))
        empty_proposal_per_sample = torch.sigmoid(pred_masks[empty]).mean(dim=tuple(range(1, pred_masks.ndim)))
        empty_mask = weighted_sample_mean(empty_mask_per_sample, weights[empty])
        empty_proposal = weighted_sample_mean(empty_proposal_per_sample, weights[empty])
    else:
        empty_mask = final_logits.sum() * 0.0
        empty_proposal = final_logits.sum() * 0.0

    diversity = (
        query_diversity_loss(outputs["query_embeddings"])
        if query_diversity_loss_weight and query_diversity_loss_weight > 0 and "query_embeddings" in outputs
        else final_logits.sum() * 0.0
    )
    mask_diversity = (
        proposal_mask_diversity_loss(pred_masks, is_empty)
        if proposal_mask_diversity_loss_weight and proposal_mask_diversity_loss_weight > 0
        else final_logits.sum() * 0.0
    )
    gate_entropy_value = modality_gate_entropy_loss(
        outputs.get("modality_gate_weights_for_loss"),
        outputs.get("modality_active_mask_for_loss"),
    )
    gate_entropy = (
        gate_entropy_value
        if gate_entropy_value is not None and gate_entropy_loss_weight and gate_entropy_loss_weight > 0
        else final_logits.sum() * 0.0
    )

    boundary = (
        weighted_sample_mean(boundary_loss_per_sample_with_logits(final_logits, target_mask), weights)
        * float(boundary_weight)
        if boundary_weight and boundary_weight > 0
        else final_logits.sum() * 0.0
    )

    total = (
        float(mask_bce_weight) * mask_bce
        + float(mask_dice_weight) * mask_dice
        + float(mask_tversky_weight) * mask_tversky
        + float(proposal_cls_weight) * proposal_cls
        + float(condition_cls_weight) * condition_cls
        + float(proposal_mask_weight)
        * (proposal_mask_bce + proposal_mask_dice + float(mask_tversky_weight) * proposal_mask_tversky)
        + float(condition_ranking_weight) * condition_rank
        + float(selection_ranking_loss_weight) * selection_rank
        + float(empty_mask_suppression_weight) * empty_mask
        + float(empty_proposal_suppression_weight) * empty_proposal
        + float(query_diversity_loss_weight) * diversity
        + float(proposal_mask_diversity_loss_weight) * mask_diversity
        + float(gate_entropy_loss_weight) * gate_entropy
        + float(query_usage_balance_loss_weight) * query_usage_balance
        + boundary
    )
    return {
        "loss": total,
        "loss_mask_bce": mask_bce.detach(),
        "loss_mask_dice": mask_dice.detach(),
        "loss_mask_tversky": mask_tversky.detach(),
        "loss_proposal_cls": proposal_cls.detach(),
        "loss_condition_cls": condition_cls.detach(),
        "loss_condition_rank": condition_rank.detach(),
        "loss_selection_rank": selection_rank.detach(),
        "condition_rank_acc": condition_rank_acc.detach(),
        "selection_rank_acc": selection_rank_acc.detach(),
        "proposal_target_mass": proposal_target.sum(dim=1).detach(),
        "proposal_target_positive_count": (proposal_target > 1.0e-4).float().sum(dim=1).detach(),
        "loss_proposal_mask": (
            proposal_mask_bce + proposal_mask_dice + float(mask_tversky_weight) * proposal_mask_tversky
        ).detach(),
        "loss_empty_mask": empty_mask.detach(),
        "loss_empty_proposal": empty_proposal.detach(),
        "loss_query_diversity": diversity.detach(),
        "loss_proposal_mask_diversity": mask_diversity.detach(),
        "loss_gate_entropy": gate_entropy.detach(),
        "loss_query_usage_balance": query_usage_balance.detach(),
        "query_usage_entropy": query_usage_entropy.detach(),
        "loss_boundary": boundary.detach(),
        "sample_weight_normalized_mean": weights.detach(),
        "sample_weight_normalized_min": weights.min().detach(),
        "sample_weight_normalized_max": weights.max().detach(),
        "best_query": best_query.detach(),
        "best_query_dice": dice_scores.gather(1, best_query.view(-1, 1)).squeeze(1).detach(),
    }
