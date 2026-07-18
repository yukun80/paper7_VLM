#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt description vision cache v1 into a task-neutral MGRR backbone state."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import (
    MODALITY_FAMILY_IDS,
    ActiveModalitySubset,
    ModalityInstance,
    ModalityPyramid,
    MultiScaleFeatures,
    MultisourceBackboneState,
    TaskNeutralVisualEvidence,
)

from ..protocols.cache import description_cache_key
class DescriptionCacheBackboneEncoder(nn.Module):
    """Project cached Qwen layers without routing single images through segmentation SANE."""

    def __init__(self, bank: Any, dim: int) -> None:
        super().__init__()
        self.bank = bank
        self.dim = int(dim)
        channels = int(bank.manifest["spatial_channels"])
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(1, dim), nn.GELU())
            for _ in range(4)
        ])
        self.family_embedding = nn.Embedding(len(MODALITY_FAMILY_IDS), dim)
        self.view_quality = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))

    @staticmethod
    def _family(view: dict) -> str:
        families = [str(value) for value in view.get("source_families") or []]
        return families[0] if len(set(families)) == 1 and families[0] in MODALITY_FAMILY_IDS else "unknown"

    def _pyramid(self, view: dict, device: torch.device) -> ModalityPyramid:
        projected = [
            adapter(feature[None].to(device=device, dtype=next(adapter.parameters()).dtype))[0]
            for adapter, feature in zip(self.adapters, view["spatial_features"])
        ]
        valid_shallow = view["valid_mask"].to(device=device, dtype=projected[0].dtype)
        valids = [
            F.interpolate(valid_shallow[None], size=value.shape[-2:], mode="nearest")[0]
            for value in projected
        ]
        family = self._family(view)
        family_id = MODALITY_FAMILY_IDS.get(family, 0)
        quality_flags = tuple(str(value) for value in view.get("quality_flags") or [])
        quality = 0.5 if quality_flags else 1.0
        metadata_token = self.family_embedding.weight[family_id] + self.view_quality(
            projected[0].new_tensor([quality, float(not quality_flags)])
        )
        instance = ModalityInstance(
            name=str(view["name"]),
            family=family,
            sensor="qwen_vision_cache",
            product_type="rendered_view",
            band_names=("cached_view",),
            band_metadata=({},),
            orbit="unknown",
            units="feature_space",
            signed=False,
            image=projected[0].new_zeros((1, 1, 1)),
            valid_mask=projected[0].new_ones((1, 1, 1)),
            native_gsd_m=None,
            aligned_gsd_m=None,
            quality=quality,
            metadata={
                "source_modalities": tuple(view.get("source_modalities") or ()),
                "render_transform": dict(view.get("render_transform") or {}),
                "content_hash": str(view.get("content_hash") or ""),
            },
        )
        return ModalityPyramid(
            instance=instance,
            detail=projected[0], high=projected[1], mid=projected[2], low=projected[3],
            detail_valid=valids[0], high_valid=valids[1], mid_valid=valids[2], low_valid=valids[3],
            metadata_token=metadata_token,
            active=True,
        )

    def forward(
        self,
        requests: Sequence[Sequence[str]],
        *,
        include_spatial: bool = True,
    ) -> MultisourceBackboneState:
        if not requests:
            raise ValueError("DescriptionCacheBackboneEncoder requests 不能为空")
        device = next(self.parameters()).device
        render_size = int(self.bank.manifest.get("render_size") or 256)
        samples = []
        subsets = []
        metadata = []
        token_sequences = []
        family_sequences = []
        segments = []
        sample_valid = []
        for component, parent_id in requests:
            record = self.bank.record(component, parent_id)
            views = list(record["views"])
            pyramids = [self._pyramid(view, device) for view in views] if include_spatial else []
            samples.append(pyramids)
            names = tuple(str(view["name"]) for view in views)
            subsets.append(ActiveModalitySubset(
                active_names=names,
                dropped_names=(),
                signature=f"description-full:{description_cache_key(component, parent_id)}",
                is_full=True,
            ))
            metadata.append({
                "component": component,
                "parent_sample_id": parent_id,
                "source_ref": record["source_ref"],
                "cache_key": record["lookup_key"],
                "render_transforms": [dict(view.get("render_transform") or {}) for view in views],
                "spatial_features_loaded": bool(include_spatial),
            })
            tokens = [view["view_tokens"].to(device=device) for view in views]
            token_sequence = torch.cat(tokens, 0)
            token_sequences.append(token_sequence)
            family_sequences.append(torch.cat([
                torch.full(
                    (token.shape[0],),
                    MODALITY_FAMILY_IDS.get(self._family(view), 0),
                    dtype=torch.long,
                    device=device,
                )
                for token, view in zip(tokens, views)
            ]))
            segments.append([
                (str(view.get("description") or ""), int(token.shape[0]))
                for token, view in zip(tokens, views)
            ])
            union_valid = torch.stack([
                F.interpolate(
                    view["valid_mask"].to(device=device, dtype=torch.float32)[None],
                    size=(render_size, render_size),
                    mode="nearest",
                )[0]
                for view in views
            ]).amax(0)
            sample_valid.append(union_valid)

        counts = tuple(int(value.shape[0]) for value in token_sequences)
        max_tokens = max(counts)
        token_dim = int(self.bank.manifest["token_dim"])
        padded_tokens = token_sequences[0].new_zeros((len(requests), max_tokens, token_dim))
        token_mask = torch.zeros((len(requests), max_tokens), dtype=torch.bool, device=device)
        family_ids = torch.zeros((len(requests), max_tokens), dtype=torch.long, device=device)
        for index, (tokens, families) in enumerate(zip(token_sequences, family_sequences)):
            length = tokens.shape[0]
            padded_tokens[index, :length] = tokens
            token_mask[index, :length] = True
            family_ids[index, :length] = families
        visual = TaskNeutralVisualEvidence(
            tokens=padded_tokens,
            token_mask=token_mask,
            family_ids=family_ids,
            token_counts=counts,
            view_segments=segments,
            cache_keys=tuple(description_cache_key(component, parent) for component, parent in requests),
            cache_format=str(self.bank.manifest["format"]),
        )
        return MultisourceBackboneState(
            features=MultiScaleFeatures(samples=samples, reference_hw=(render_size, render_size)),
            valid_mask=torch.stack(sample_valid),
            active_subsets=tuple(subsets),
            metadata=tuple(metadata),
            reference_hw=(render_size, render_size),
            use_full_evidence=True,
            visual_evidence=visual,
        )
