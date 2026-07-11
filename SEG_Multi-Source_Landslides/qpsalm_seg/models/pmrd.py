#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Proposal-Set Mask Refinement Decoder (PMRD)."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import EvidenceFeatures, ProposalSet

from .common import ConvBlock, MLP


class ProposalSetMaskRefinementDecoder(nn.Module):
    """Generate PSALM mask proposals and refine them with mask-aware evidence replay."""

    def __init__(self, dim: int, num_queries: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.mask_tokens = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_position = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.task_to_query = nn.Linear(dim, dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.coarse_decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(1, int(num_layers)))
        self.query_norm = nn.LayerNorm(dim)
        self.position_proj = nn.Linear(2, dim)
        self.coarse_mask_embed = MLP(dim)
        self.detail_branch = ConvBlock(dim, dim)
        self.region_update = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.refine_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.refine_norm = nn.LayerNorm(dim)
        self.final_mask_embed = MLP(dim)

    def _position_tokens(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(height * width, 2)
        return self.position_proj(coords)

    def _memory(
        self,
        feature: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = feature.shape
        memory = feature.flatten(2).transpose(1, 2)
        memory = memory + self._position_tokens(height, width, feature.device, feature.dtype)[None]
        padding = ~(valid[:, 0].flatten(1) > 1.0e-4)
        if padding.all(dim=1).any():
            padding = padding.clone()
            padding[padding.all(dim=1), 0] = False
        return memory, padding

    def propose(
        self,
        evidence: EvidenceFeatures,
        task_token: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = task_token.shape[0]
        memory, padding = self._memory(evidence.fused_low, evidence.fused_low_valid)
        queries = self.mask_tokens[None].expand(batch_size, -1, -1)
        queries = queries + self.query_position[None] + self.task_to_query(task_token)[:, None]
        decoded = self.coarse_decoder(queries, memory, memory_key_padding_mask=padding)
        decoded = self.query_norm(decoded)
        mask_embed = self.coarse_mask_embed(decoded)
        coarse = torch.einsum("bqd,bdhw->bqhw", mask_embed, evidence.fused_high)
        coarse = F.interpolate(coarse, size=target_hw, mode="bilinear", align_corners=False)
        return decoded, coarse

    @staticmethod
    def _region_pool(
        coarse_masks: torch.Tensor,
        detail: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        masks = F.interpolate(coarse_masks, size=detail.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        masks = masks * valid
        mass = masks.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
        weights = masks / mass
        return torch.einsum("bqhw,bdhw->bqd", weights, detail)

    def refine(
        self,
        coarse_queries: torch.Tensor,
        coarse_masks: torch.Tensor,
        query_evidence: torch.Tensor,
        evidence: EvidenceFeatures,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.detail_branch(evidence.fused_high)
        region = self._region_pool(coarse_masks, detail, evidence.fused_high_valid)
        update = self.region_update(torch.cat([coarse_queries, region, query_evidence], dim=-1))
        queries = self.refine_norm(coarse_queries + update)
        memory, padding = self._memory(evidence.fused_mid, evidence.fused_mid_valid)
        attended, _ = self.refine_attention(queries, memory, memory, key_padding_mask=padding, need_weights=False)
        queries = self.refine_norm(queries + attended)
        mask_embed = self.final_mask_embed(queries)
        masks = torch.einsum("bqd,bdhw->bqhw", mask_embed, detail)
        masks = F.interpolate(masks, size=target_hw, mode="bilinear", align_corners=False)
        return queries, masks

    @staticmethod
    def relevance_gates(relevance_logits: torch.Tensor) -> torch.Tensor:
        """Calibrate neutral logits to about one active query instead of Q half-active queries."""
        num_queries = int(relevance_logits.shape[1])
        offset = math.log(float(max(1, num_queries - 1)))
        return torch.sigmoid(relevance_logits.float() - offset)

    @classmethod
    def compose_final_mask(cls, mask_logits: torch.Tensor, relevance_logits: torch.Tensor) -> torch.Tensor:
        """Union relevant proposals without softmax competition or query-count saturation."""
        mask_probs = torch.sigmoid(mask_logits).float()
        relevance = cls.relevance_gates(relevance_logits)[:, :, None, None]
        proposal_probs = (mask_probs * relevance).clamp(1.0e-6, 1.0 - 1.0e-6)
        union = 1.0 - torch.prod(1.0 - proposal_probs, dim=1, keepdim=True)
        return torch.logit(union.clamp(1.0e-6, 1.0 - 1.0e-6)).to(mask_logits.dtype)

    def build_proposal_set(
        self,
        masks: torch.Tensor,
        coarse_masks: torch.Tensor,
        relevance: torch.Tensor,
        queries: torch.Tensor,
        query_evidence: torch.Tensor,
        modality_weights: torch.Tensor,
        spatial_entropy: torch.Tensor,
    ) -> ProposalSet:
        return ProposalSet(
            mask_logits=masks,
            coarse_mask_logits=coarse_masks,
            relevance_logits=relevance,
            query_embeddings=queries,
            query_evidence=query_evidence,
            query_modality_attention=modality_weights,
            query_spatial_entropy=spatial_entropy,
        )
