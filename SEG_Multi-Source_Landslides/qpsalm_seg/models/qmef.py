#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QMEF: valid, rejectable and query-conditioned multi-scale evidence fusion."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import MODALITY_FAMILIES, EvidenceFeatures, MultiScaleFeatures, SemanticEvidence

from .common import MLP


def valid_weighted_pool(feature: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = valid.to(feature.dtype)
    mass = mask.sum().clamp_min(1.0)
    pooled = (feature * mask).sum((1, 2)) / mass
    coverage = mask.mean()
    return pooled, coverage


class ScaleAwareDeformableAggregator(nn.Module):
    def __init__(self, dim: int, num_points: int = 4) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.context_proj = nn.Linear(dim * 2, dim)
        self.offset_head = nn.Conv2d(dim, self.num_points * 2, 3, padding=1)
        self.weight_head = nn.Conv2d(dim, self.num_points, 3, padding=1)

    @staticmethod
    def _base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], -1)[None]

    @staticmethod
    def _resize_pad(feature: torch.Tensor, valid: torch.Tensor, target_hw: tuple[int, int], transform=None):
        target_h, target_w = target_hw
        source_h, source_w = feature.shape[-2:]
        if transform:
            reference_h, reference_w = transform["target_hw"]
            resized_reference_h, resized_reference_w = transform["resized_hw"]
            resized_h = max(1, min(target_h, int(round(resized_reference_h / reference_h * target_h))))
            resized_w = max(1, min(target_w, int(round(resized_reference_w / reference_w * target_w))))
        else:
            scale = min(target_h / max(source_h, 1), target_w / max(source_w, 1))
            resized_h = max(1, min(target_h, int(round(source_h * scale))))
            resized_w = max(1, min(target_w, int(round(source_w * scale))))
        resized = F.interpolate(feature[None], (resized_h, resized_w), mode="bilinear", align_corners=False)
        resized_valid = F.interpolate(valid[None].float(), (resized_h, resized_w), mode="nearest")
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        padding = (left, target_w - resized_w - left, top, target_h - resized_h - top)
        return F.pad(resized, padding), F.pad(resized_valid, padding)

    def forward(self, feature, valid_mask, target_hw, metadata_token, semantic_token, gsd_ratio, reference_transform=None):
        base, source_valid = self._resize_pad(feature, valid_mask, target_hw, reference_transform)
        context = base + self.context_proj(torch.cat([metadata_token, semantic_token]))[None, :, None, None]
        h, w = target_hw
        offsets = self.offset_head(context).view(1, self.num_points, 2, h, w).permute(0, 1, 3, 4, 2).tanh()
        step = offsets.new_tensor([2 / max(w - 1, 1), 2 / max(h - 1, 1)])
        offsets = offsets * step * max(0.25, min(4.0, float(gsd_ratio)))
        grid = self._base_grid(h, w, feature.device, feature.dtype)[:, None] + offsets
        grid = grid.reshape(self.num_points, h, w, 2)
        sampled = F.grid_sample(base.expand(self.num_points, -1, -1, -1), grid, align_corners=True)
        sampled_valid = F.grid_sample(
            source_valid.to(feature.dtype).expand(self.num_points, -1, -1, -1), grid, align_corners=True
        )
        logits = self.weight_head(context).float().masked_fill(sampled_valid[:, 0][None] <= 1e-4, -1e4)
        weights = torch.softmax(logits, 1).to(sampled.dtype)
        aligned = (sampled[None] * weights[:, :, None]).sum(1)
        aligned_valid = (sampled_valid[:, 0][None] * weights).sum(1, keepdim=True).clamp(0, 1)
        return aligned[0] * aligned_valid[0], aligned_valid[0]


class QwenGuidedEvidenceFusion(nn.Module):
    """Align native features, estimate null-aware reliability, and replay evidence per query."""

    def __init__(self, dim: int, deformable_points: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_points = int(deformable_points)
        self.align = nn.ModuleDict({name: ScaleAwareDeformableAggregator(dim, deformable_points) for name in ("detail", "high", "mid", "low")})
        self.semantic_film = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.reliability_head = nn.Sequential(nn.LayerNorm(dim * 3 + 1), nn.Linear(dim * 3 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.null_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.query_proj = nn.Linear(dim, dim)
        self.family_anchor_proj = nn.Linear(dim, dim)
        self.key_proj = nn.ModuleDict({name: nn.Conv2d(dim, dim, 1) for name in ("high", "mid", "low")})
        self.value_proj = nn.ModuleDict({name: nn.Conv2d(dim, dim, 1) for name in ("high", "mid", "low")})
        self.scale_embedding = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.query_offsets = nn.Linear(dim, 3 * self.num_points * 2)
        self.query_point_bias = nn.Linear(dim, 3 * self.num_points)
        nn.init.zeros_(self.query_offsets.weight)
        nn.init.zeros_(self.query_offsets.bias)
        if self.num_points <= 1:
            pattern = torch.zeros((1, 2), dtype=torch.float32)
        else:
            angles = torch.arange(self.num_points - 1, dtype=torch.float32) * (
                2.0 * math.pi / (self.num_points - 1)
            )
            pattern = torch.cat([torch.zeros((1, 2)), torch.stack([angles.cos(), angles.sin()], -1)], 0)
        self.register_buffer("point_pattern", pattern, persistent=False)
        self.verifier_query = MLP(dim)
        self.verifier_evidence = MLP(dim)
        self.verifier_semantic = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU())
        self.verifier_head = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))

    @staticmethod
    def _target_shapes(reference_hw):
        h, w = reference_hw
        return {
            "detail": (max(1, math.ceil(h / 2)), max(1, math.ceil(w / 2))),
            "high": (max(1, math.ceil(h / 4)), max(1, math.ceil(w / 4))),
            "mid": (max(1, math.ceil(h / 8)), max(1, math.ceil(w / 8))),
            "low": (max(1, math.ceil(h / 16)), max(1, math.ceil(w / 16))),
        }

    @staticmethod
    def _ratio(native, aligned):
        return float(native / aligned) if native and aligned and native > 0 and aligned > 0 else 1.0

    def forward(
        self,
        features: MultiScaleFeatures,
        semantic: SemanticEvidence,
        *,
        enable_semantic: bool = True,
        enable_reliability: bool = True,
    ) -> EvidenceFeatures:
        shapes = self._target_shapes(features.reference_hw)
        b = len(features.samples)
        m = max(len(sample) for sample in features.samples)
        device, dtype = semantic.global_token.device, semantic.global_token.dtype
        stacks = {name: torch.zeros((b, m, self.dim, *shape), device=device, dtype=dtype) for name, shape in shapes.items()}
        valids = {name: torch.zeros((b, m, 1, *shape), device=device, dtype=dtype) for name, shape in shapes.items()}
        active = torch.zeros((b, m), device=device, dtype=torch.bool)
        logits = torch.full((b, m), -1e4, device=device, dtype=dtype)
        coverage = torch.zeros((b, m), device=device, dtype=dtype)
        modality_anchors = torch.zeros((b, m, self.dim), device=device, dtype=dtype)
        names: list[list[str]] = []
        semantic_global = semantic.global_token if enable_semantic else torch.zeros_like(semantic.global_token)
        if enable_semantic:
            film_scale, film_shift = self.semantic_film(semantic_global).chunk(2, -1)
        else:
            film_scale = film_shift = torch.zeros_like(semantic_global)
        for bi, sample in enumerate(features.samples):
            sample_names = []
            for mi, pyramid in enumerate(sample):
                sample_names.append(pyramid.instance.name)
                active[bi, mi] = pyramid.active
                if not pyramid.active:
                    continue
                ratio = self._ratio(pyramid.instance.native_gsd_m, pyramid.instance.aligned_gsd_m)
                family_index = MODALITY_FAMILIES.index(pyramid.instance.family) + 1
                family_semantic = (
                    semantic.evidence_anchors[bi, family_index]
                    if enable_semantic and semantic.evidence_anchors is not None
                    else semantic_global[bi]
                )
                modality_anchors[bi, mi] = family_semantic
                for scale in stacks:
                    aligned, aligned_valid = self.align[scale](
                        getattr(pyramid, scale), getattr(pyramid, f"{scale}_valid"), shapes[scale],
                        pyramid.metadata_token, semantic_global[bi], ratio,
                        pyramid.instance.metadata.get("reference_resize_transform"),
                    )
                    conditioned = (
                        aligned * (1 + 0.1 * torch.tanh(film_scale[bi])[:, None, None])
                        + 0.1 * torch.tanh(film_shift[bi])[:, None, None]
                    )
                    # Geometry validity is a hard spatial contract. FiLM may
                    # modulate valid evidence but must not recreate padding/nodata.
                    stacks[scale][bi, mi] = torch.where(
                        aligned_valid > 1.0e-4,
                        conditioned * aligned_valid,
                        torch.zeros_like(conditioned),
                    )
                    valids[scale][bi, mi] = aligned_valid
                pooled, _low_coverage = valid_weighted_pool(stacks["low"][bi, mi], valids["low"][bi, mi])
                cov = pyramid.instance.valid_mask.to(device=device, dtype=dtype).mean().clamp(0, 1)
                coverage[bi, mi] = cov
                if enable_reliability:
                    reliability_input = torch.cat([pooled, pyramid.metadata_token, family_semantic, cov[None]])
                    quality_prior = math.log(max(float(pyramid.instance.quality), 1e-3))
                    coverage_cap = torch.log(cov.clamp_min(1e-3))
                    logits[bi, mi] = self.reliability_head(reliability_input).squeeze() + quality_prior + coverage_cap
            names.append(sample_names)
        logits = logits.masked_fill(~active, -1e4)
        if enable_reliability:
            null_logits = self.null_head(semantic_global).squeeze(-1)
            distribution = torch.softmax(torch.cat([logits.float(), null_logits[:, None].float()], 1), 1).to(dtype)
            reliability, null = distribution[:, :-1] * active.to(dtype), distribution[:, -1]
        else:
            counts = active.sum(1, keepdim=True).clamp_min(1).to(dtype)
            reliability = active.to(dtype) / counts
            null = torch.zeros((b,), device=device, dtype=dtype)
            logits = torch.where(active, torch.zeros_like(logits), torch.full_like(logits, -1e4))
        real_mass = reliability.sum(1)

        def fuse(name):
            return (stacks[name] * reliability[:, :, None, None, None]).sum(1)

        fused = {name: fuse(name) for name in stacks}
        fused_valid = {
            name: (
                valids[name]
                * active[:, :, None, None, None].to(valids[name].dtype)
            ).amax(1)
            for name in valids
        }
        return EvidenceFeatures(
            fused_high=fused["high"], fused_mid=fused["mid"], fused_low=fused["low"],
            fused_high_valid=fused_valid["high"], fused_mid_valid=fused_valid["mid"], fused_low_valid=fused_valid["low"],
            modality_high=stacks["high"], modality_mid=stacks["mid"], modality_low=stacks["low"], modality_detail=stacks["detail"],
            modality_valid_high=valids["high"], modality_valid_mid=valids["mid"], modality_valid_low=valids["low"], modality_valid_detail=valids["detail"],
            modality_active=active, reliability_logits=logits, reliability_weights=reliability,
            null_reliability=null, real_reliability_mass=real_mass, coverage_ratio=coverage,
            modality_semantic_anchors=modality_anchors,
            modality_names=names,
        )

    @staticmethod
    def _mask_reference(coarse_masks: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(coarse_masks.float())
        h, w = probability.shape[-2:]
        y = torch.linspace(-1, 1, h, device=probability.device, dtype=probability.dtype)
        x = torch.linspace(-1, 1, w, device=probability.device, dtype=probability.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mass = probability.sum((2, 3)).clamp_min(1.0e-6)
        ref_x = (probability * xx).sum((2, 3)) / mass
        ref_y = (probability * yy).sum((2, 3)) / mass
        return torch.stack([ref_x, ref_y], -1)

    @staticmethod
    def _sample_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        b, m, channels, h, w = values.shape
        q, points = grid.shape[1:3]
        expanded_grid = grid[:, None].expand(-1, m, -1, -1, -1).reshape(b * m, q * points, 1, 2)
        sampled = F.grid_sample(
            values.reshape(b * m, channels, h, w), expanded_grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        return sampled[:, :, :, 0].view(b, m, channels, q, points).permute(0, 3, 1, 4, 2)

    def attend_queries(self, queries, evidence: EvidenceFeatures, coarse_masks=None):
        if coarse_masks is None:
            raise ValueError("query-conditioned deformable attention requires coarse_masks")
        score_parts = []
        value_parts = []
        valid_parts = []
        sampling_grids = []
        reference = self._mask_reference(coarse_masks).to(queries.dtype)
        learned_offsets = self.query_offsets(queries).view(
            *queries.shape[:2], 3, self.num_points, 2
        ).tanh()
        point_bias = self.query_point_bias(queries).view(*queries.shape[:2], 3, self.num_points)
        radii = queries.new_tensor((0.20, 0.35, 0.50))
        family_keys = self.family_anchor_proj(evidence.modality_semantic_anchors)
        for scale_index, scale in enumerate(("high", "mid", "low")):
            features = getattr(evidence, f"modality_{scale}")
            valid = getattr(evidence, f"modality_valid_{scale}")
            b, m, d, h, w = features.shape
            flat = features.reshape(b * m, d, h, w)
            keys = self.key_proj[scale](flat).view(b, m, d, h, w)
            values = self.value_proj[scale](flat).view(b, m, d, h, w)
            projected = self.query_proj(queries + self.scale_embedding[scale_index])
            pattern = self.point_pattern.to(queries.dtype)[None, None]
            offsets = pattern * radii[scale_index] + 0.25 * learned_offsets[:, :, scale_index]
            grid = (reference[:, :, None] + offsets).clamp(-1, 1)
            sampling_grids.append(grid)
            sampled_keys = self._sample_maps(keys, grid)
            sampled_values = self._sample_maps(values, grid)
            sampled_valid = self._sample_maps(valid.to(values.dtype), grid)[..., 0] > 1.0e-4
            score = torch.einsum("bqd,bqmpd->bqmp", projected, sampled_keys) / math.sqrt(d)
            score = score + torch.einsum("bqd,bmd->bqm", projected, family_keys)[:, :, :, None] / math.sqrt(d)
            score = score + torch.log(evidence.reliability_weights.clamp_min(1e-8))[:, None, :, None]
            score = score + point_bias[:, :, scale_index, None]
            score = score.masked_fill(~sampled_valid, -1e4)
            score_parts.append(score)
            value_parts.append(sampled_values)
            valid_parts.append(sampled_valid)
        scores = torch.stack(score_parts, 2)
        sampled_values = torch.stack(value_parts, 2)
        sampled_valid = torch.stack(valid_parts, 2)
        flat_score = scores.flatten(2)
        flat_valid = sampled_valid.flatten(2)
        flat_weight = torch.softmax(flat_score.float(), -1) * flat_valid.float()
        flat_weight = flat_weight / flat_weight.sum(-1, keepdim=True).clamp_min(1.0e-8)
        weight = flat_weight.view(
            queries.shape[0], queries.shape[1], len(score_parts),
            evidence.modality_mid.shape[1], self.num_points,
        ).to(sampled_values.dtype)
        context = (sampled_values * weight[..., None]).sum((2, 3, 4))
        context = context * evidence.real_reliability_mass[:, None, None].to(context.dtype)
        modality_mass = weight.sum((2, 4))
        scale_mass = weight.sum((3, 4))
        safe = flat_weight.clamp_min(1.0e-8)
        entropy = (
            -(safe * safe.log()).sum(-1)
            / math.log(max(flat_weight.shape[-1], 2))
        ).clamp(0, 1).to(queries.dtype)
        return context, modality_mass, scale_mass, entropy, reference, torch.stack(sampling_grids, 2)

    def verify(self, queries, query_evidence, semantic):
        query = self.verifier_query(queries)
        semantic_token = self.verifier_semantic(
            torch.cat([semantic.condition_token, semantic.global_token], -1)
        )[:, None].expand_as(query)
        evidence = self.verifier_evidence(query_evidence)
        return self.verifier_head(torch.cat([query, semantic_token, evidence, query * evidence], -1)).squeeze(-1)
