#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MGRR: multi-granularity region replay over task-neutral SANE features."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from qpsalm_seg.schema import (
    ModalityPyramid,
    MultisourceBackboneState,
    RegionEvidenceState,
)


RegionProtocol = Literal["assisted", "vision_only"]
MGRRAblation = Literal["full", "no_context", "roi_replay_only"]
MGRR_PROTOCOL = "qpsalm_mgrr_v2_multiscale_grid_replay"
MGRR_ROI_GRID_SIZES = ((7, 7), (7, 7), (4, 4), (2, 2))


def rasterize_region_geometry(
    geometry: dict,
    valid_mask: torch.Tensor,
    *,
    explicit_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert full-image/box/mask/null geometry to one valid-canvas binary mask."""
    valid = valid_mask.float()
    if valid.ndim == 2:
        valid = valid[None]
    if valid.ndim != 3 or valid.shape[0] != 1:
        raise ValueError(f"valid_mask 必须是 [1,H,W]，当前 {tuple(valid.shape)}")
    region_type = geometry.get("type")
    if not region_type:
        region_type = "mask" if explicit_mask is not None else "null"
    region_type = str(region_type)
    if region_type == "null":
        return torch.zeros_like(valid)
    if region_type == "full_image":
        return (valid > 0.5).float()
    if region_type == "mask":
        if explicit_mask is None:
            raise ValueError("mask geometry 需要 explicit_mask")
        mask = explicit_mask.float()
        if mask.ndim == 2:
            mask = mask[None]
        if mask.shape != valid.shape:
            mask = F.interpolate(mask[None], size=valid.shape[-2:], mode="nearest")[0]
        return ((mask > 0.5) & (valid > 0.5)).float()
    if region_type == "box":
        height, width = valid.shape[-2:]
        bbox = geometry.get("bbox_xyxy_pixel_half_open")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            normalized = geometry.get("bbox_xyxy_normalized")
            if not isinstance(normalized, (list, tuple)) or len(normalized) != 4:
                raise ValueError("box geometry 缺少 pixel/normalized bbox")
            bbox = [
                round(float(normalized[0]) * width), round(float(normalized[1]) * height),
                round(float(normalized[2]) * width), round(float(normalized[3]) * height),
            ]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"box geometry 面积非法: {bbox}")
        result = torch.zeros_like(valid)
        result[:, y1:y2, x1:x2] = 1.0
        return result * (valid > 0.5)
    raise ValueError(f"未知 region geometry type={region_type!r}")


def _masked_pool(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=feature.dtype)
    return (feature * weight).sum((-2, -1)) / weight.sum((-2, -1)).clamp_min(1.0)


def _component_masks(
    mask: torch.Tensor,
    *,
    max_components: int,
    coverage_target: float,
) -> tuple[list[torch.Tensor], torch.Tensor | None, float, int]:
    binary = mask.detach().float().cpu().numpy() > 0.5
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    components: list[tuple[int, np.ndarray]] = []
    for label_id in range(1, int(count) + 1):
        value = labels == label_id
        components.append((int(value.sum()), value))
    components.sort(key=lambda item: -item[0])
    total = max(sum(area for area, _ in components), 1)
    selected: list[torch.Tensor] = []
    selected_area = 0
    residual = np.zeros_like(binary, dtype=bool)
    for index, (area, value) in enumerate(components):
        if index < max_components and (selected_area / total < coverage_target or not selected):
            selected.append(torch.from_numpy(value).to(device=mask.device, dtype=mask.dtype))
            selected_area += area
        else:
            residual |= value
    residual_tensor = (
        torch.from_numpy(residual).to(device=mask.device, dtype=mask.dtype)
        if residual.any() else None
    )
    return (
        selected,
        residual_tensor,
        float(selected_area / total) if components else 0.0,
        len(components),
    )


def _bbox_pool(
    feature: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    coordinates = torch.nonzero(mask[0] > 0.5, as_tuple=False)
    if coordinates.numel() == 0:
        return feature.new_zeros(feature.shape[0])
    y1, x1 = coordinates.min(0).values
    y2, x2 = coordinates.max(0).values + 1
    crop = feature[:, int(y1):int(y2), int(x1):int(x2)]
    if valid_mask is None:
        return crop.mean((-2, -1))
    valid = valid_mask[:, int(y1):int(y2), int(x1):int(x2)].to(crop.dtype)
    return (crop * valid).sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1.0)


def _geometry_values(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    binary = (mask[0] > 0.5) & (valid[0] > 0.5)
    coordinates = torch.nonzero(binary, as_tuple=False)
    height, width = binary.shape
    if coordinates.numel() == 0:
        return mask.new_zeros(10)
    y1, x1 = coordinates.min(0).values.float()
    y2, x2 = (coordinates.max(0).values + 1).float()
    center = coordinates.float().mean(0)
    area_ratio = binary.float().sum() / valid.float().sum().clamp_min(1.0)
    bbox_area = ((y2 - y1) * (x2 - x1)) / max(height * width, 1)
    aspect = (x2 - x1) / (y2 - y1).clamp_min(1.0)
    return torch.stack([
        area_ratio,
        x1 / max(width, 1), y1 / max(height, 1),
        x2 / max(width, 1), y2 / max(height, 1),
        center[1] / max(width - 1, 1), center[0] / max(height - 1, 1),
        bbox_area, torch.log1p(aspect), mask.new_tensor(1.0),
    ])


def _context_ring(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    area = max(float((mask > 0.5).sum().item()), 1.0)
    radius = max(3, min(32, int(round(math.sqrt(area / math.pi) * 0.5))))
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(mask[None].float(), kernel, stride=1, padding=radius)[0]
    return ((dilated > 0.5) & ~(mask > 0.5) & (valid > 0.5)).to(mask.dtype)


class MultiGranularityRegionReplay(nn.Module):
    """Replay exact masks, component RoIs, context and modality evidence per region."""

    def __init__(
        self,
        dim: int,
        *,
        max_components: int = 8,
        component_coverage: float = 0.9,
        ablation: MGRRAblation = "full",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_components = int(max_components)
        self.component_coverage = float(component_coverage)
        if ablation not in {"full", "no_context", "roi_replay_only"}:
            raise ValueError(f"未知 MGRR ablation={ablation!r}")
        self.ablation = ablation
        self.exact_project = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.component_project = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.context_project = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.global_project = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.geometry_project = nn.Sequential(nn.Linear(10, dim), nn.GELU(), nn.LayerNorm(dim))
        self.modality_project = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.LayerNorm(dim))
        self.region_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.roi_queries = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.roi_scale_embedding = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.null_region = nn.Parameter(torch.randn(dim) * 0.02)
        self.null_evidence = nn.Parameter(torch.randn(dim) * 0.02)
        self.reliability = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.output = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.LayerNorm(dim))

    @staticmethod
    def _align(value: torch.Tensor, target_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
        kwargs = {"size": target_hw, "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        return F.interpolate(value[None], **kwargs)[0]

    @staticmethod
    def _roi_grid_tokens(
        feature: torch.Tensor,
        valid_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an RoI grid using reference-canvas coordinates."""
        coordinates = torch.nonzero(reference_mask[0] > 0.5, as_tuple=False)
        count = int(grid_size[0] * grid_size[1])
        if coordinates.numel() == 0:
            return (
                feature.new_zeros((count, feature.shape[0])),
                torch.zeros(count, dtype=torch.bool, device=feature.device),
            )
        reference_h, reference_w = reference_mask.shape[-2:]
        minimum = coordinates.min(0).values.to(feature.dtype)
        maximum = (coordinates.max(0).values + 1).to(feature.dtype)
        ys = (
            torch.arange(grid_size[0], device=feature.device, dtype=feature.dtype) + 0.5
        ) / grid_size[0]
        xs = (
            torch.arange(grid_size[1], device=feature.device, dtype=feature.dtype) + 0.5
        ) / grid_size[1]
        ys = minimum[0] + ys * (maximum[0] - minimum[0])
        xs = minimum[1] + xs * (maximum[1] - minimum[1])
        # Pixel-centre normalization for grid_sample(..., align_corners=False).
        normalized_y = 2.0 * ys / max(reference_h, 1) - 1.0
        normalized_x = 2.0 * xs / max(reference_w, 1) - 1.0
        grid_y, grid_x = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], -1)[None]
        sampled = F.grid_sample(
            feature.float()[None], grid.float(), mode="bilinear", padding_mode="zeros",
            align_corners=False,
        )[0].flatten(1).T.to(feature.dtype)
        sampled_valid = F.grid_sample(
            valid_mask.float()[None], grid.float(), mode="nearest",
            padding_mode="zeros", align_corners=False,
        )[0, 0].flatten() > 0.5
        return sampled, sampled_valid

    def _multi_scale_roi_replay(
        self,
        pyramid: ModalityPyramid,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale_values = tuple(zip(
            (pyramid.detail, pyramid.high, pyramid.mid, pyramid.low),
            (pyramid.detail_valid, pyramid.high_valid, pyramid.mid_valid, pyramid.low_valid),
            MGRR_ROI_GRID_SIZES,
        ))
        token_rows = []
        valid_rows = []
        for scale_index, (feature, valid, grid_size) in enumerate(scale_values):
            tokens, token_valid = self._roi_grid_tokens(
                feature, valid, reference_mask, grid_size
            )
            tokens = tokens + self.roi_scale_embedding[scale_index].to(tokens.dtype)
            token_rows.append(tokens)
            valid_rows.append(token_valid)
        tokens = torch.cat(token_rows, 0)
        token_valid = torch.cat(valid_rows, 0)
        if not bool(token_valid.any()):
            return tokens.new_zeros(self.dim)
        queries = self.roi_queries.to(tokens.dtype)
        logits = queries @ tokens.T / math.sqrt(self.dim)
        logits = logits.masked_fill(~token_valid[None], -1.0e4)
        replay_queries = torch.softmax(logits.float(), -1).to(tokens.dtype) @ tokens
        return replay_queries.mean(0)

    def _component_replay(
        self,
        pyramid: ModalityPyramid,
        components: list[torch.Tensor],
        residual: torch.Tensor | None,
        aligned_feature: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = []
        weights = []
        for component in components:
            value = component[None]
            effective = value * valid_mask
            exact = _masked_pool(aligned_feature, effective)
            roi = self._multi_scale_roi_replay(pyramid, value)
            component_ring = _context_ring(value, valid_mask)
            local_context = _masked_pool(aligned_feature, component_ring * valid_mask)
            contrast = exact - local_context
            if self.ablation in {"no_context", "roi_replay_only"}:
                contrast = torch.zeros_like(contrast)
            if self.ablation == "roi_replay_only":
                exact = torch.zeros_like(exact)
            tokens.append(self.component_project(torch.cat([exact, roi, contrast], -1)))
            weights.append(float(value.sum().item()))
        if residual is not None and self.ablation != "roi_replay_only":
            value = residual[None]
            exact = _masked_pool(aligned_feature, value * valid_mask)
            # Residual components can be far apart. Their union bbox is not a
            # valid RoI, so residual evidence is exact-mask pooled only.
            tokens.append(self.component_project(torch.cat([
                exact,
                exact.new_zeros(exact.shape),
                exact.new_zeros(exact.shape),
            ], -1)))
            weights.append(float(value.sum().item()))
        if not tokens:
            return aligned_feature.new_zeros(self.dim), []
        weight = aligned_feature.new_tensor(weights)
        weight = weight / weight.sum().clamp_min(1.0)
        return (
            (torch.stack(tokens) * weight[:, None]).sum(0),
            tokens,
        )

    def forward(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        *,
        region_valid_mask: torch.Tensor | None = None,
        protocol: RegionProtocol = "vision_only",
    ) -> RegionEvidenceState:
        if protocol not in {"assisted", "vision_only"}:
            raise ValueError(f"未知 MGRR protocol={protocol!r}")
        if region_masks.ndim == 3:
            region_masks = region_masks[:, None]
        if region_masks.ndim != 4:
            raise ValueError(f"region_masks 必须为 [B,R,H,W]，当前 {tuple(region_masks.shape)}")
        batch_size, region_count, height, width = region_masks.shape
        if batch_size != len(backbone.features.samples) or (height, width) != backbone.reference_hw:
            raise ValueError("region mask batch/size 与 MultisourceBackboneState 不一致")
        device = backbone.valid_mask.device
        masks = region_masks.to(device=device, dtype=torch.float32)
        valid = (
            region_valid_mask.to(device=device, dtype=torch.float32)
            if region_valid_mask is not None else backbone.valid_mask.to(device=device, dtype=torch.float32)
        )
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.shape != (batch_size, 1, height, width):
            raise ValueError(f"region_valid_mask shape 非法: {tuple(valid.shape)}")

        region_rows = []
        context_rows = []
        geometry_rows = []
        modality_rows: list[list[torch.Tensor]] = []
        reliability_rows: list[torch.Tensor] = []
        sequence_rows: list[list[torch.Tensor]] = []
        component_counts = masks.new_zeros((batch_size, region_count))
        selected_component_counts = masks.new_zeros((batch_size, region_count))
        component_coverage = masks.new_zeros((batch_size, region_count))
        residual_area_ratio = masks.new_zeros((batch_size, region_count))
        max_modalities = max((len(sample) for sample in backbone.features.samples), default=1)
        for batch_index, pyramids in enumerate(backbone.features.samples):
            sample_regions = []
            sample_contexts = []
            sample_geometries = []
            sample_modalities = []
            sample_reliability = []
            sample_sequences = []
            for region_index in range(region_count):
                region = ((masks[batch_index, region_index:region_index + 1] > 0.5) & (valid[batch_index] > 0.5)).float()
                is_present = bool(region.any())
                components, residual, replay_coverage, total_component_count = _component_masks(
                    region[0],
                    max_components=self.max_components,
                    coverage_target=self.component_coverage,
                )
                component_counts[batch_index, region_index] = total_component_count
                selected_component_counts[batch_index, region_index] = len(components)
                component_coverage[batch_index, region_index] = replay_coverage
                residual_area_ratio[batch_index, region_index] = max(
                    0.0, 1.0 - replay_coverage
                )
                ring = _context_ring(region, valid[batch_index]) if is_present else valid[batch_index]
                geometry = self.geometry_project(_geometry_values(region, valid[batch_index]))
                if protocol == "vision_only":
                    geometry = torch.zeros_like(geometry)
                modality_tokens = []
                global_tokens = []
                exact_tokens = []
                replay_tokens = []
                context_tokens = []
                component_token_rows: list[list[torch.Tensor]] = []
                coverages = []
                for pyramid in pyramids:
                    detail = self._align(pyramid.detail, (height, width))
                    detail_valid = self._align(pyramid.detail_valid.float(), (height, width), "nearest")
                    high = self._align(pyramid.high, (height, width))
                    high_valid = self._align(pyramid.high_valid.float(), (height, width), "nearest")
                    effective_region = region * detail_valid
                    effective_ring = ring * detail_valid
                    exact_detail = _masked_pool(detail, effective_region)
                    exact_high = _masked_pool(high, region * high_valid)
                    context_raw = _masked_pool(detail, effective_ring)
                    contrast = exact_detail - context_raw
                    if self.ablation in {"no_context", "roi_replay_only"}:
                        contrast = torch.zeros_like(contrast)
                    exact = self.exact_project(torch.cat([
                        exact_detail, exact_high, contrast,
                    ], -1))
                    context = self.context_project(context_raw)
                    replay, component_tokens = self._component_replay(
                        pyramid, components, residual, detail, detail_valid
                    )
                    global_token = self.global_project(_masked_pool(
                        self._align(pyramid.low, (height, width)),
                        self._align(pyramid.low_valid.float(), (height, width), "nearest")
                        * valid[batch_index],
                    ))
                    coverage_ratio = effective_region.sum() / region.sum().clamp_min(1.0)
                    if self.ablation == "no_context":
                        context = torch.zeros_like(context)
                    elif self.ablation == "roi_replay_only":
                        exact = torch.zeros_like(exact)
                        context = torch.zeros_like(context)
                        global_token = torch.zeros_like(global_token)
                    token = self.modality_project(torch.cat([
                        exact, replay, context, pyramid.metadata_token,
                    ], -1))
                    modality_tokens.append(token)
                    global_tokens.append(global_token)
                    exact_tokens.append(exact)
                    replay_tokens.append(replay)
                    context_tokens.append(context)
                    component_token_rows.append(component_tokens)
                    coverages.append(coverage_ratio)
                if modality_tokens:
                    tokens = torch.stack(modality_tokens)
                    coverage_tensor = torch.stack(coverages).to(tokens.dtype)
                    query = self.region_query + geometry
                    logits = self.reliability(torch.cat([
                        tokens,
                        query[None].expand_as(tokens),
                        coverage_tensor[:, None],
                    ], -1)).squeeze(-1)
                    null_logit = self.reliability(torch.cat([
                        self.null_evidence,
                        query,
                        query.new_tensor([0.0]),
                    ], -1)[None]).squeeze()
                    if not is_present:
                        logits = logits.new_full(logits.shape, -1.0e4)
                        null_logit = null_logit * 0.0
                    weights = torch.softmax(torch.cat([logits, null_logit[None]]).float(), 0).to(tokens.dtype)
                    fused = (tokens * weights[:-1, None]).sum(0) + self.null_evidence * weights[-1]
                    real_weights = weights[:-1]
                    global_token = (torch.stack(global_tokens) * real_weights[:, None]).sum(0)
                    exact_token = (torch.stack(exact_tokens) * real_weights[:, None]).sum(0)
                    replay_token = (torch.stack(replay_tokens) * real_weights[:, None]).sum(0)
                    context_token = (torch.stack(context_tokens) * real_weights[:, None]).sum(0)
                    if not is_present:
                        global_token = torch.stack(global_tokens).mean(0)
                    padded = F.pad(tokens, (0, 0, 0, max_modalities - tokens.shape[0]))
                    padded_weights = torch.cat([
                        weights[:-1],
                        weights.new_zeros(max_modalities - tokens.shape[0]),
                        weights[-1:],
                    ])
                else:
                    fused = self.null_evidence
                    global_token = self.null_evidence
                    exact_token = self.null_evidence
                    replay_token = self.null_evidence
                    context_token = self.null_evidence
                    padded = fused.new_zeros((max_modalities, self.dim))
                    padded_weights = fused.new_zeros(max_modalities + 1)
                    padded_weights[-1] = 1.0
                region_token = self.output(torch.cat([
                    self.null_region if not is_present else self.region_query,
                    fused,
                    context_token,
                    geometry,
                ], -1))
                if is_present and modality_tokens:
                    if self.ablation == "roi_replay_only":
                        sequence = [region_token, replay_token, geometry]
                    elif self.ablation == "no_context":
                        sequence = [
                            region_token, global_token, exact_token, replay_token, geometry,
                        ]
                    else:
                        sequence = [
                            region_token, global_token, exact_token, replay_token,
                            context_token, geometry,
                        ]
                    sequence.extend(modality_tokens)
                    max_component_slots = max(
                        (len(values) for values in component_token_rows), default=0
                    )
                    for component_index in range(max_component_slots):
                        available = [
                            modality_index
                            for modality_index, values in enumerate(component_token_rows)
                            if component_index < len(values)
                        ]
                        slot_weights = real_weights[available]
                        slot_weights = slot_weights / slot_weights.sum().clamp_min(1.0e-6)
                        sequence.append(sum(
                            (
                                component_token_rows[modality_index][component_index]
                                * slot_weights[offset]
                                for offset, modality_index in enumerate(available)
                            ),
                            start=region_token.new_zeros(self.dim),
                        ))
                else:
                    # Global context remains visible for a no-target decision,
                    # while all local evidence is represented by explicit nulls.
                    sequence = [global_token, self.null_region, geometry, self.null_evidence]
                sequence_tensor = torch.stack(sequence)
                sample_regions.append(region_token)
                sample_contexts.append(context_token)
                sample_geometries.append(geometry)
                sample_modalities.append(padded)
                sample_reliability.append(padded_weights)
                sample_sequences.append(sequence_tensor)
            region_rows.append(torch.stack(sample_regions))
            context_rows.append(torch.stack(sample_contexts))
            geometry_rows.append(torch.stack(sample_geometries))
            modality_rows.append(sample_modalities)
            reliability_rows.append(torch.stack(sample_reliability))
            sequence_rows.append(sample_sequences)

        reliability = torch.stack(reliability_rows)
        flat_sequences = [value for sample in sequence_rows for value in sample]
        sequence_tokens = pad_sequence(flat_sequences, batch_first=True).reshape(
            batch_size, region_count, -1, self.dim
        )
        sequence_mask = pad_sequence(
            [torch.ones(value.shape[0], dtype=torch.bool, device=device) for value in flat_sequences],
            batch_first=True,
            padding_value=False,
        ).reshape(batch_size, region_count, -1)
        return RegionEvidenceState(
            backbone=backbone,
            region_masks=masks,
            region_valid_mask=valid,
            region_tokens=torch.stack(region_rows),
            region_sequence_tokens=sequence_tokens,
            region_sequence_mask=sequence_mask,
            context_tokens=torch.stack(context_rows),
            geometry_tokens=torch.stack(geometry_rows),
            modality_tokens=torch.stack([torch.stack(row) for row in modality_rows]),
            diagnostics={
                "modality_reliability": reliability,
                "null_reliability": reliability[..., -1],
                "component_count": component_counts,
                "selected_component_count": selected_component_counts,
                "component_coverage": component_coverage,
                "residual_area_ratio": residual_area_ratio,
                "roi_grid_sample_count": masks.new_tensor(float(sum(
                    height * width for height, width in MGRR_ROI_GRID_SIZES
                ))),
                "roi_query_count": masks.new_tensor(float(self.roi_queries.shape[0])),
                "region_sequence_length": sequence_mask.sum(-1),
                "protocol_assisted": masks.new_tensor(float(protocol == "assisted")),
                "ablation_full": masks.new_tensor(float(self.ablation == "full")),
                "ablation_no_context": masks.new_tensor(float(self.ablation == "no_context")),
                "ablation_roi_replay_only": masks.new_tensor(
                    float(self.ablation == "roi_replay_only")
                ),
            },
        )
