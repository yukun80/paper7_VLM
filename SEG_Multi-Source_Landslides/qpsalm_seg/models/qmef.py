#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Guided Multi-Source Evidence Fusion (QMEF)."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import EvidenceFeatures, MultiScaleFeatures, SemanticEvidence

from .common import MLP


class ScaleAwareDeformableAggregator(nn.Module):
    """Lightweight learned sampling from a native feature map to a reference grid."""

    def __init__(self, dim: int, num_points: int = 4) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.context_proj = nn.Linear(dim * 2, dim)
        self.offset_head = nn.Conv2d(dim, self.num_points * 2, kernel_size=3, padding=1)
        self.weight_head = nn.Conv2d(dim, self.num_points, kernel_size=3, padding=1)

    @staticmethod
    def _base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1)[None]

    def forward(
        self,
        feature: torch.Tensor,
        valid_mask: torch.Tensor,
        target_hw: tuple[int, int],
        metadata_token: torch.Tensor,
        semantic_token: torch.Tensor,
        gsd_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature4d = feature[None]
        target_h, target_w = target_hw
        base_feature = F.interpolate(feature4d, size=target_hw, mode="bilinear", align_corners=False)
        context_token = self.context_proj(torch.cat([metadata_token, semantic_token], dim=-1))
        context = base_feature + context_token[None, :, None, None]
        offsets = self.offset_head(context).view(1, self.num_points, 2, target_h, target_w)
        offsets = offsets.permute(0, 1, 3, 4, 2).tanh()
        ratio = max(0.25, min(4.0, float(gsd_ratio)))
        pixel_step = offsets.new_tensor([2.0 / max(target_w - 1, 1), 2.0 / max(target_h - 1, 1)])
        offsets = offsets * pixel_step * ratio
        grid = self._base_grid(target_h, target_w, feature.device, feature.dtype)[:, None] + offsets
        flat_grid = grid.reshape(self.num_points, target_h, target_w, 2)
        repeated = feature4d.expand(self.num_points, -1, -1, -1)
        sampled = F.grid_sample(repeated, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        valid4d = valid_mask[None].to(feature.dtype).expand(self.num_points, -1, -1, -1)
        sampled_valid = F.grid_sample(valid4d, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        logits = self.weight_head(context).float()
        logits = logits.masked_fill(sampled_valid[:, 0][None] <= 1.0e-4, -1.0e4)
        weights = torch.softmax(logits, dim=1).to(sampled.dtype)
        aligned = (sampled[None] * weights[:, :, None]).sum(dim=1)
        aligned_valid = (sampled_valid[:, 0][None] * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        return aligned.squeeze(0), aligned_valid.squeeze(0)


class QwenGuidedEvidenceFusion(nn.Module):
    """Reliability prior, native-scale alignment, query-spatial attention and one verifier."""

    def __init__(self, dim: int, deformable_points: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.high_align = ScaleAwareDeformableAggregator(dim, deformable_points)
        self.mid_align = ScaleAwareDeformableAggregator(dim, deformable_points)
        self.low_align = ScaleAwareDeformableAggregator(dim, deformable_points)
        self.semantic_film = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.query_proj = nn.Linear(dim, dim)
        self.spatial_key_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.verifier_query = MLP(dim)
        self.verifier_evidence = MLP(dim)
        self.verifier_head = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    @staticmethod
    def _target_shapes(reference_hw: tuple[int, int]) -> dict[str, tuple[int, int]]:
        height, width = reference_hw
        return {
            "high": (max(1, math.ceil(height / 4)), max(1, math.ceil(width / 4))),
            "mid": (max(1, math.ceil(height / 8)), max(1, math.ceil(width / 8))),
            "low": (max(1, math.ceil(height / 16)), max(1, math.ceil(width / 16))),
        }

    @staticmethod
    def _gsd_ratio(native: float | None, aligned: float | None) -> float:
        if native is None or aligned is None or native <= 0.0 or aligned <= 0.0:
            return 1.0
        return float(native / aligned)

    def forward(self, features: MultiScaleFeatures, semantic: SemanticEvidence) -> EvidenceFeatures:
        shapes = self._target_shapes(features.reference_hw)
        batch_size = len(features.samples)
        max_modalities = max(len(sample) for sample in features.samples)
        device = semantic.global_token.device
        dtype = semantic.global_token.dtype
        high = torch.zeros((batch_size, max_modalities, self.dim, *shapes["high"]), device=device, dtype=dtype)
        mid = torch.zeros((batch_size, max_modalities, self.dim, *shapes["mid"]), device=device, dtype=dtype)
        low = torch.zeros((batch_size, max_modalities, self.dim, *shapes["low"]), device=device, dtype=dtype)
        mid_valid = torch.zeros((batch_size, max_modalities, 1, *shapes["mid"]), device=device, dtype=dtype)
        active = torch.zeros((batch_size, max_modalities), device=device, dtype=torch.bool)
        reliability_logits = torch.full((batch_size, max_modalities), -1.0e4, device=device, dtype=dtype)
        names: list[list[str]] = []
        film_scale, film_shift = self.semantic_film(semantic.global_token).chunk(2, dim=-1)
        film_scale = 0.1 * torch.tanh(film_scale)
        film_shift = 0.1 * torch.tanh(film_shift)
        for batch_index, sample in enumerate(features.samples):
            sample_names: list[str] = []
            for modality_index, pyramid in enumerate(sample):
                sample_names.append(pyramid.instance.name)
                active[batch_index, modality_index] = bool(pyramid.active)
                if not pyramid.active:
                    continue
                ratio = self._gsd_ratio(pyramid.instance.native_gsd_m, pyramid.instance.aligned_gsd_m)
                aligned_high, _ = self.high_align(
                    pyramid.high,
                    pyramid.high_valid,
                    shapes["high"],
                    pyramid.metadata_token,
                    semantic.global_token[batch_index],
                    ratio,
                )
                aligned_mid, valid_mid = self.mid_align(
                    pyramid.mid,
                    pyramid.mid_valid,
                    shapes["mid"],
                    pyramid.metadata_token,
                    semantic.global_token[batch_index],
                    ratio,
                )
                aligned_low, _ = self.low_align(
                    pyramid.low,
                    pyramid.low_valid,
                    shapes["low"],
                    pyramid.metadata_token,
                    semantic.global_token[batch_index],
                    ratio,
                )
                scale = 1.0 + film_scale[batch_index, :, None, None]
                shift = film_shift[batch_index, :, None, None]
                high[batch_index, modality_index] = aligned_high * scale + shift
                mid[batch_index, modality_index] = aligned_mid * scale + shift
                low[batch_index, modality_index] = aligned_low * scale + shift
                mid_valid[batch_index, modality_index] = valid_mid
                pooled = aligned_low.mean(dim=(1, 2))
                reliability_input = torch.cat(
                    [pooled, pyramid.metadata_token, semantic.global_token[batch_index]],
                    dim=-1,
                )
                quality_prior = math.log(max(float(pyramid.instance.quality), 1.0e-3))
                reliability_logits[batch_index, modality_index] = self.reliability_head(reliability_input).squeeze(-1) + quality_prior
            names.append(sample_names)
        reliability_logits = reliability_logits.masked_fill(~active, -1.0e4)
        reliability = torch.softmax(reliability_logits.float(), dim=1).to(dtype)
        reliability = reliability * active.to(dtype)
        reliability = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        def fuse(stack: torch.Tensor) -> torch.Tensor:
            return (stack * reliability[:, :, None, None, None]).sum(dim=1)

        fused_high = fuse(high)
        fused_mid = fuse(mid)
        fused_low = fuse(low)
        fused_mid_valid = (mid_valid * reliability[:, :, None, None, None]).sum(dim=1).clamp(0.0, 1.0)
        fused_high_valid = F.interpolate(fused_mid_valid, size=shapes["high"], mode="nearest")
        fused_low_valid = F.interpolate(fused_mid_valid, size=shapes["low"], mode="nearest")
        return EvidenceFeatures(
            fused_high=fused_high,
            fused_mid=fused_mid,
            fused_low=fused_low,
            fused_high_valid=fused_high_valid,
            fused_mid_valid=fused_mid_valid,
            fused_low_valid=fused_low_valid,
            modality_high=high,
            modality_mid=mid,
            modality_low=low,
            modality_valid_mid=mid_valid,
            modality_active=active,
            reliability_logits=reliability_logits,
            reliability_weights=reliability,
            modality_names=names,
        )

    def attend_queries(
        self,
        queries: torch.Tensor,
        evidence: EvidenceFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_modalities, dim, height, width = evidence.modality_mid.shape
        flat = evidence.modality_mid.reshape(batch_size * num_modalities, dim, height, width)
        keys = self.spatial_key_proj(flat).view(batch_size, num_modalities, dim, height, width)
        values = self.spatial_value_proj(flat).view(batch_size, num_modalities, dim, height, width)
        projected_queries = self.query_proj(queries)
        logits = torch.einsum("bqd,bmdhw->bqmhw", projected_queries, keys) / math.sqrt(float(dim))
        logits = logits + torch.log(evidence.reliability_weights.clamp_min(1.0e-6))[:, None, :, None, None]
        valid = evidence.modality_valid_mid[:, None, :, 0] > 1.0e-4
        active = evidence.modality_active[:, None, :, None, None]
        logits = logits.masked_fill(~(valid & active), -1.0e4)
        flat_logits = logits.flatten(2)
        attention = torch.softmax(flat_logits.float(), dim=-1).to(values.dtype).view_as(logits)
        context = torch.einsum("bqmhw,bmdhw->bqd", attention, values)
        modality_weights = attention.sum(dim=(3, 4))
        safe = attention.float().clamp_min(1.0e-8)
        entropy = -(safe * safe.log()).flatten(2).sum(dim=2)
        entropy = entropy / math.log(max(num_modalities * height * width, 2))
        return context, modality_weights, entropy.clamp(0.0, 1.0).to(queries.dtype)

    def verify(
        self,
        queries: torch.Tensor,
        query_evidence: torch.Tensor,
        semantic: SemanticEvidence,
    ) -> torch.Tensor:
        query = self.verifier_query(queries)
        semantic_token = semantic.global_token[:, None].expand_as(query)
        evidence = self.verifier_evidence(query_evidence)
        features = torch.cat([query, semantic_token, evidence, query * evidence], dim=-1)
        return self.verifier_head(features).squeeze(-1)
