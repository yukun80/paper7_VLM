#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PMRD: Qwen-query proposal set and query-specific high-resolution refinement."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.matching import calibrated_relevance_gates
from qpsalm_seg.schema import EvidenceFeatures, ProposalSet

from .common import MLP


class ProposalSetMaskRefinementDecoder(nn.Module):
    def __init__(self, dim: int, num_queries: int, num_layers: int, num_heads: int, query_chunk_size: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.query_chunk_size = int(query_chunk_size)
        layer = nn.TransformerDecoderLayer(dim, num_heads, dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.coarse_decoder = nn.TransformerDecoder(layer, max(1, int(num_layers)))
        self.query_norm = nn.LayerNorm(dim)
        self.position_proj = nn.Linear(2, dim)
        self.coarse_mask_embed = MLP(dim)
        self.coarse_mask_bias = nn.Linear(dim, 1)
        self.region_update = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.refine_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.refine_norm = nn.LayerNorm(dim)
        self.final_mask_embed = MLP(dim)
        self.final_mask_bias = nn.Linear(dim, 1)
        for layer in (self.coarse_mask_bias, self.final_mask_bias):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, -2.0)

    def _position_tokens(self, h, w, device, dtype):
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return self.position_proj(torch.stack([xx, yy], -1).reshape(h * w, 2))

    def _memory(self, feature, valid):
        b, _, h, w = feature.shape
        memory = feature.flatten(2).transpose(1, 2) + self._position_tokens(h, w, feature.device, feature.dtype)[None]
        padding = ~(valid[:, 0].flatten(1) > 1e-4)
        if padding.all(1).any():
            padding = padding.clone()
            padding[padding.all(1), 0] = False
        return memory, padding

    def propose(self, evidence: EvidenceFeatures, semantic, target_hw):
        memory, padding = self._memory(evidence.fused_low, evidence.fused_low_valid)
        if semantic.mask_query_states is None:
            raise ValueError("PMRD v2 要求 controller 提供 mask_query_states")
        queries = semantic.mask_query_states
        if queries.shape[1] != self.num_queries:
            raise ValueError(f"controller mask query count={queries.shape[1]} expected={self.num_queries}")
        decoded = self.query_norm(self.coarse_decoder(queries, memory, memory_key_padding_mask=padding))
        embed = self.coarse_mask_embed(decoded)
        coarse = (
            torch.einsum("bqd,bdhw->bqhw", embed, evidence.fused_high)
            + self.coarse_mask_bias(decoded)[..., None]
        )
        return decoded, F.interpolate(coarse, target_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _region_pool(coarse_masks, query_detail, valid):
        h, w = query_detail.shape[-2:]
        masks = F.interpolate(coarse_masks, (h, w), mode="bilinear", align_corners=False).sigmoid()
        masks = masks * valid[:, None, 0]
        weights = masks / masks.sum((2, 3), keepdim=True).clamp_min(1e-6)
        return torch.einsum("bqhw,bqdhw->bqd", weights, query_detail)

    def _query_detail(self, evidence, modality_weights, coarse_masks, start, end):
        weights = modality_weights[:, start:end] * evidence.modality_active[:, None].to(modality_weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        valid = evidence.modality_valid_detail[:, None].to(weights.dtype)
        pixel_weights = weights[:, :, :, None, None, None] * valid
        pixel_weights = pixel_weights / pixel_weights.sum(2, keepdim=True).clamp_min(1e-6)
        # Preserve QMEF's null-evidence decision after local valid normalization.
        pixel_weights = pixel_weights * evidence.real_reliability_mass[:, None, None, None, None, None].to(
            weights.dtype
        )
        query_detail = (pixel_weights * evidence.modality_detail[:, None]).sum(2)
        spatial = F.interpolate(
            coarse_masks[:, start:end], query_detail.shape[-2:], mode="bilinear", align_corners=False
        ).sigmoid()
        return query_detail * spatial[:, :, None]

    def refine(self, coarse_queries, coarse_masks, query_evidence, modality_weights, evidence, target_hw):
        valid_detail = (evidence.modality_valid_detail * evidence.modality_active[:, :, None, None, None]).amax(1)
        region_chunks = []
        for start in range(0, coarse_queries.shape[1], self.query_chunk_size):
            end = min(coarse_queries.shape[1], start + self.query_chunk_size)
            detail_chunk = self._query_detail(evidence, modality_weights, coarse_masks, start, end)
            region_chunks.append(self._region_pool(coarse_masks[:, start:end], detail_chunk, valid_detail))
        region = torch.cat(region_chunks, 1)
        queries = self.refine_norm(coarse_queries + self.region_update(torch.cat([coarse_queries, region, query_evidence], -1)))
        memory, padding = self._memory(evidence.fused_mid, evidence.fused_mid_valid)
        attended, _ = self.refine_attention(queries, memory, memory, key_padding_mask=padding, need_weights=False)
        queries = self.refine_norm(queries + attended)
        embeds = self.final_mask_embed(queries)
        chunks = []
        for start in range(0, queries.shape[1], self.query_chunk_size):
            end = min(queries.shape[1], start + self.query_chunk_size)
            detail_chunk = self._query_detail(evidence, modality_weights, coarse_masks, start, end)
            chunks.append(
                torch.einsum("bqd,bqdhw->bqhw", embeds[:, start:end], detail_chunk)
                + self.final_mask_bias(queries[:, start:end])[..., None]
            )
        masks = torch.cat(chunks, 1)
        return queries, F.interpolate(masks, target_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def relevance_gates(logits):
        return calibrated_relevance_gates(logits.float())

    @classmethod
    def compose_final_mask(cls, masks, relevance):
        probs = (torch.sigmoid(masks).float() * cls.relevance_gates(relevance)[:, :, None, None]).clamp(1e-6, 1 - 1e-6)
        union = 1 - torch.prod(1 - probs, 1, keepdim=True)
        return torch.logit(union.clamp(1e-6, 1 - 1e-6)).to(masks.dtype)

    def build_proposal_set(
        self, masks, coarse, relevance, queries, query_evidence,
        modality_weights, scale_weights, entropy,
    ):
        return ProposalSet(
            masks, coarse, relevance, queries, query_evidence,
            modality_weights, scale_weights, entropy,
        )
