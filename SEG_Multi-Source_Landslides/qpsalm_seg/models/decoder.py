#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PSALM-style mask proposal decoder。"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import MLP


class ConditionAwareProposalScorer(nn.Module):
    """用 condition embedding 对 proposals 进行可学习匹配打分。"""

    def __init__(self, decoder_dim: int) -> None:
        super().__init__()
        d = int(decoder_dim)
        self.query_proj = MLP(d)
        self.condition_proj = MLP(d)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(d * 3),
            nn.Linear(d * 3, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0), dtype=torch.float32))

    def forward(self, query_embeddings: torch.Tensor, condition_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        q_proj = self.query_proj(query_embeddings)
        c_proj = self.condition_proj(condition_embedding).unsqueeze(1).expand_as(q_proj)
        q_norm = F.normalize(q_proj, dim=-1)
        c_norm = F.normalize(c_proj, dim=-1)
        cosine_logits = torch.einsum("bqd,bqd->bq", q_norm, c_norm)
        pair = torch.cat([q_proj, c_proj, q_proj * c_proj], dim=-1)
        pair_logits = self.pair_head(pair).squeeze(-1)
        scale = self.logit_scale.exp().clamp(0.1, 20.0)
        return {
            "condition_scores": cosine_logits * scale + pair_logits,
            "condition_cosine_scores": cosine_logits,
            "condition_pair_logits": pair_logits,
            "condition_logit_scale": scale,
        }


class PSALMConditionAwareMaskDecoder(nn.Module):
    """用 task context 生成 proposals，用 condition embedding 对 proposals 打分。"""

    def __init__(
        self,
        decoder_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        selection_proposal_weight: float = 1.0,
        selection_condition_weight: float = 1.0,
        selection_temperature: float = 1.0,
        final_foreground_gate_weight: float = 0.0,
        final_mask_fusion: str = "weighted_average",
        final_topk: int = 3,
        final_noisy_or_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.selection_proposal_weight = float(selection_proposal_weight)
        self.selection_condition_weight = float(selection_condition_weight)
        self.selection_temperature = max(1.0e-3, float(selection_temperature))
        self.final_foreground_gate_weight = float(final_foreground_gate_weight)
        self.final_mask_fusion = str(final_mask_fusion)
        self.final_topk = max(1, int(final_topk))
        self.final_noisy_or_epsilon = max(1.0e-8, float(final_noisy_or_epsilon))
        self.mask_tokens = nn.Parameter(torch.randn(num_queries, decoder_dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(num_queries, decoder_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.context_to_query = nn.Linear(decoder_dim, decoder_dim)
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.mask_embed = MLP(decoder_dim)
        self.proposal_head = nn.Linear(decoder_dim, 2)
        self.condition_scorer = ConditionAwareProposalScorer(decoder_dim)

    def _fuse_final_masks(
        self,
        pred_masks: torch.Tensor,
        selection_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把多个 proposal 合成为最终语义 mask。

        ``weighted_average`` 保留旧行为；top-k 分支用于一张遥感 patch 中存在多个
        滑坡斑块时避免单一 softmax 平均把候选区域抹平。
        """
        # CUDA autocast may keep ``selection_logits`` in bf16 while softmax/top-k
        # produce fp32 tensors. Build all fusion weights in fp32 first so scatter
        # and noisy-or stay dtype-stable, then cast the exported weights back.
        selection_logits_fp32 = selection_logits.float()
        pred_masks_fp32 = pred_masks.float()
        weights_fp32 = torch.softmax(selection_logits_fp32 / float(self.selection_temperature), dim=1)
        mode = self.final_mask_fusion.strip().lower().replace("-", "_")
        if mode in {"weighted_average", "softmax_average", "average"}:
            logits = torch.einsum("bq,bqhw->bhw", weights_fp32, pred_masks_fp32).unsqueeze(1)
            selected = torch.argmax(selection_logits, dim=1, keepdim=True)
            return logits.to(pred_masks.dtype), weights_fp32.to(selection_logits.dtype), selected

        k = min(self.final_topk, pred_masks.shape[1])
        top_logits, top_indices = torch.topk(selection_logits_fp32, k=k, dim=1)
        top_masks = torch.gather(
            pred_masks_fp32,
            dim=1,
            index=top_indices[:, :, None, None].expand(-1, -1, pred_masks.shape[-2], pred_masks.shape[-1]),
        )
        top_weights = torch.softmax(top_logits / float(self.selection_temperature), dim=1)
        full_weights_fp32 = torch.zeros_like(selection_logits_fp32).scatter(1, top_indices, top_weights)

        if mode in {"topk_weighted_average", "topk_average"}:
            logits = torch.einsum("bq,bqhw->bhw", full_weights_fp32, pred_masks_fp32).unsqueeze(1)
            return logits.to(pred_masks.dtype), full_weights_fp32.to(selection_logits.dtype), top_indices
        if mode in {"topk_noisy_or", "noisy_or", "topk_union"}:
            probs = torch.sigmoid(top_masks).clamp(
                min=float(self.final_noisy_or_epsilon),
                max=1.0 - float(self.final_noisy_or_epsilon),
            )
            # top-k logits 只负责筛选候选；标准 noisy-or 不再用 softmax 权重稀释像素概率。
            union_prob = 1.0 - torch.prod(1.0 - probs, dim=1, keepdim=True)
            union_prob = union_prob.clamp(
                min=float(self.final_noisy_or_epsilon),
                max=1.0 - float(self.final_noisy_or_epsilon),
            )
            logits = torch.logit(union_prob)
            return logits.to(pred_masks.dtype), full_weights_fp32.to(selection_logits.dtype), top_indices
        raise ValueError(
            "Unsupported final_mask_fusion="
            f"{self.final_mask_fusion!r}; expected weighted_average, topk_weighted_average, or topk_noisy_or."
        )

    def forward(
        self,
        mask_features: torch.Tensor,
        memory_features: torch.Tensor,
        proposal_context: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz = mask_features.shape[0]
        memory = memory_features.flatten(2).transpose(1, 2)
        queries = self.mask_tokens.unsqueeze(0).expand(bsz, -1, -1)
        queries = queries + self.query_pos.unsqueeze(0)
        queries = queries + self.context_to_query(proposal_context).unsqueeze(1)
        decoded = self.decoder(tgt=queries, memory=memory)
        decoded = self.query_norm(decoded)

        mask_embed = self.mask_embed(decoded)
        pred_masks = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_features)
        proposal_logits = self.proposal_head(decoded)

        score_out = self.condition_scorer(decoded, condition_embedding)
        condition_scores = score_out["condition_scores"]
        proposal_fg_logits = proposal_logits[..., 1] - proposal_logits[..., 0]
        selection_logits = (
            float(self.selection_proposal_weight) * proposal_fg_logits
            + float(self.selection_condition_weight) * condition_scores
        )
        final_mask_logits, weights, selected_query_indices = self._fuse_final_masks(pred_masks, selection_logits)
        foreground_gate_logits = proposal_fg_logits.max(dim=1).values
        if self.final_foreground_gate_weight:
            final_mask_logits = final_mask_logits + float(self.final_foreground_gate_weight) * foreground_gate_logits.view(
                bsz, 1, 1, 1
            )
        return {
            "pred_masks": pred_masks,
            "proposal_logits": proposal_logits,
            "proposal_fg_logits": proposal_fg_logits,
            "condition_scores": condition_scores,
            "selection_logits": selection_logits,
            "selection_weights": weights,
            "selected_query_indices": selected_query_indices,
            "foreground_gate_logits": foreground_gate_logits,
            "condition_cosine_scores": score_out["condition_cosine_scores"],
            "condition_pair_logits": score_out["condition_pair_logits"],
            "condition_logit_scale": score_out["condition_logit_scale"].detach(),
            "final_mask_logits": final_mask_logits,
            "query_embeddings": decoded,
        }
