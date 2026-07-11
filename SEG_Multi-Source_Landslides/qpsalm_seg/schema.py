#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed contracts shared by the data pipeline and SANE/QMEF/PMRD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import torch


@dataclass
class ModalityInstance:
    """A single sensor product with its physical and spatial semantics."""

    name: str
    family: str
    sensor: str
    band_names: tuple[str, ...]
    orbit: str
    image: torch.Tensor
    valid_mask: torch.Tensor
    native_gsd_m: float | None
    aligned_gsd_m: float | None
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ModalityInstance":
        return ModalityInstance(
            name=self.name,
            family=self.family,
            sensor=self.sensor,
            band_names=self.band_names,
            orbit=self.orbit,
            image=self.image.to(device),
            valid_mask=self.valid_mask.to(device),
            native_gsd_m=self.native_gsd_m,
            aligned_gsd_m=self.aligned_gsd_m,
            quality=self.quality,
            metadata=self.metadata,
        )


@dataclass
class ModalityBatch(Mapping[str, Any]):
    """Variable-cardinality multimodal batch with a common segmentation canvas."""

    instances: list[list[ModalityInstance]]
    mask: torch.Tensor
    valid_mask: torch.Tensor
    metadata: list[dict[str, Any]]
    proposal_context_text: list[str]
    condition_prompt_text: list[str]
    evidence_reasoning_text: list[str]
    visual_evidence_key: list[str]
    visual_preview: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.mask.shape[0])

    @property
    def reference_hw(self) -> tuple[int, int]:
        return int(self.mask.shape[-2]), int(self.mask.shape[-1])

    @property
    def availability(self) -> torch.Tensor:
        families = ("optical", "multispectral", "sar", "terrain", "deformation")
        result = torch.zeros((self.batch_size, len(families)), dtype=torch.float32)
        for batch_index, sample in enumerate(self.instances):
            available = {item.family for item in sample}
            for family_index, family in enumerate(families):
                result[batch_index, family_index] = float(family in available)
        return result

    def __getitem__(self, key: str) -> Any:
        if key == "availability":
            return self.availability
        if key == "modalities":
            raise KeyError("Typed ModalityBatch does not expose fixed canonical modality tensors")
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "instances",
                "mask",
                "valid_mask",
                "metadata",
                "proposal_context_text",
                "condition_prompt_text",
                "evidence_reasoning_text",
                "visual_evidence_key",
                "visual_preview",
                "availability",
            )
        )

    def __len__(self) -> int:
        return 10


@dataclass
class ModalityPyramid:
    """Native-scale features for one modality instance."""

    instance: ModalityInstance
    high: torch.Tensor
    mid: torch.Tensor
    low: torch.Tensor
    high_valid: torch.Tensor
    mid_valid: torch.Tensor
    low_valid: torch.Tensor
    metadata_token: torch.Tensor
    active: bool = True


@dataclass
class MultiScaleFeatures:
    """Per-sample collection of native-scale modality pyramids."""

    samples: list[list[ModalityPyramid]]
    reference_hw: tuple[int, int]


@dataclass
class SemanticEvidence:
    """Unified task, condition, reasoning and optional visual evidence tokens."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    task_token: torch.Tensor
    condition_token: torch.Tensor
    global_token: torch.Tensor
    visual_token_count: int = 0


@dataclass
class EvidenceFeatures:
    """QMEF output consumed by PMRD."""

    fused_high: torch.Tensor
    fused_mid: torch.Tensor
    fused_low: torch.Tensor
    fused_high_valid: torch.Tensor
    fused_mid_valid: torch.Tensor
    fused_low_valid: torch.Tensor
    modality_high: torch.Tensor
    modality_mid: torch.Tensor
    modality_low: torch.Tensor
    modality_valid_mid: torch.Tensor
    modality_active: torch.Tensor
    reliability_logits: torch.Tensor
    reliability_weights: torch.Tensor
    modality_names: list[list[str]]


@dataclass
class ProposalSet:
    """PSALM-style proposal set before semantic union."""

    mask_logits: torch.Tensor
    coarse_mask_logits: torch.Tensor
    relevance_logits: torch.Tensor
    query_embeddings: torch.Tensor
    query_evidence: torch.Tensor
    query_modality_attention: torch.Tensor
    query_spatial_entropy: torch.Tensor


@dataclass
class SegmentationOutput(Mapping[str, torch.Tensor]):
    """Typed model result with a small mapping view for diagnostics/export code."""

    final_mask_logits: torch.Tensor
    proposals: ProposalSet
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    losses: dict[str, torch.Tensor] = field(default_factory=dict)

    def _mapping(self) -> dict[str, torch.Tensor]:
        relevance = self.proposals.relevance_logits
        num_queries = int(relevance.shape[1])
        relevance_offset = torch.log(relevance.new_tensor(float(max(1, num_queries - 1))))
        values = {
            "final_mask_logits": self.final_mask_logits,
            "proposal_mask_logits": self.proposals.mask_logits,
            "proposal_coarse_mask_logits": self.proposals.coarse_mask_logits,
            "proposal_relevance_logits": relevance,
            "proposal_relevance_gates": torch.sigmoid(relevance - relevance_offset),
            "query_embeddings": self.proposals.query_embeddings,
            "query_modality_attention": self.proposals.query_modality_attention,
            "query_spatial_entropy": self.proposals.query_spatial_entropy,
            **self.diagnostics,
            **self.losses,
        }
        return values

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            return self._mapping()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping().get(key, default)

    def update_losses(self, values: dict[str, torch.Tensor]) -> None:
        self.losses.update(values)
