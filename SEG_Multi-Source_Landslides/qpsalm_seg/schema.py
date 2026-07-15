#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed contracts shared by the data pipeline and SANE/QMEF/PMRD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import torch

from qpsalm_seg.matching import calibrated_relevance_gates


MODALITY_FAMILIES = ("optical", "multispectral", "sar", "terrain", "deformation")
MODALITY_FAMILY_IDS = {"unknown": 0, **{name: index + 1 for index, name in enumerate(MODALITY_FAMILIES)}}


@dataclass
class ModalityInstance:
    """A single sensor product with its physical and spatial semantics."""

    name: str
    family: str
    sensor: str
    product_type: str
    band_names: tuple[str, ...]
    band_metadata: tuple[dict[str, Any], ...]
    orbit: str
    units: str
    signed: bool
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
            product_type=self.product_type,
            band_names=self.band_names,
            band_metadata=self.band_metadata,
            orbit=self.orbit,
            units=self.units,
            signed=self.signed,
            image=self.image.to(device),
            valid_mask=self.valid_mask.to(device),
            native_gsd_m=self.native_gsd_m,
            aligned_gsd_m=self.aligned_gsd_m,
            quality=self.quality,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ActiveModalitySubset:
    """Single source of truth for the modalities visible to one student sample."""

    active_names: tuple[str, ...]
    dropped_names: tuple[str, ...]
    signature: str
    is_full: bool


@dataclass
class ModalityBatch:
    """Variable-cardinality multimodal batch with a common segmentation canvas."""

    instances: list[list[ModalityInstance]]
    full_instances: list[list[ModalityInstance]]
    active_subsets: list[ActiveModalitySubset]
    mask: torch.Tensor
    valid_mask: torch.Tensor
    metadata: list[dict[str, Any]]
    proposal_context_text: list[str]
    condition_prompt_text: list[str]
    evidence_reasoning_text: list[str]
    full_proposal_context_text: list[str]
    full_condition_prompt_text: list[str]
    full_evidence_reasoning_text: list[str]
    visual_evidence_key: list[str]
    component_masks: list[torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.mask.shape[0])

    @property
    def reference_hw(self) -> tuple[int, int]:
        return int(self.mask.shape[-2]), int(self.mask.shape[-1])

    def select(self, indices: list[int] | tuple[int, ...]) -> "ModalityBatch":
        """Select samples without rebuilding or mutating their modality payloads."""
        selected = [int(index) for index in indices]
        if not selected:
            raise ValueError("ModalityBatch.select requires at least one index")
        if min(selected) < 0 or max(selected) >= self.batch_size:
            raise IndexError(f"batch indices out of range: {selected} for batch_size={self.batch_size}")
        tensor_indices = torch.tensor(selected, device=self.mask.device, dtype=torch.long)

        def pick(values):
            return [values[index] for index in selected]

        return ModalityBatch(
            instances=pick(self.instances),
            full_instances=pick(self.full_instances),
            active_subsets=pick(self.active_subsets),
            mask=self.mask.index_select(0, tensor_indices),
            valid_mask=self.valid_mask.index_select(0, tensor_indices),
            metadata=pick(self.metadata),
            proposal_context_text=pick(self.proposal_context_text),
            condition_prompt_text=pick(self.condition_prompt_text),
            evidence_reasoning_text=pick(self.evidence_reasoning_text),
            full_proposal_context_text=pick(self.full_proposal_context_text),
            full_condition_prompt_text=pick(self.full_condition_prompt_text),
            full_evidence_reasoning_text=pick(self.full_evidence_reasoning_text),
            visual_evidence_key=pick(self.visual_evidence_key),
            component_masks=(pick(self.component_masks) if self.component_masks is not None else None),
        )

    @property
    def availability(self) -> torch.Tensor:
        result = torch.zeros((self.batch_size, len(MODALITY_FAMILIES)), dtype=torch.float32)
        for batch_index, sample in enumerate(self.instances):
            available = {item.family for item in sample}
            for family_index, family in enumerate(MODALITY_FAMILIES):
                result[batch_index, family_index] = float(family in available)
        return result

    def pin_memory(self) -> "ModalityBatch":
        """Pin tensor payloads without letting PyTorch coerce this typed batch to dict."""
        pinned_instances: dict[int, ModalityInstance] = {}

        def pin_instance(item: ModalityInstance) -> ModalityInstance:
            key = id(item)
            if key not in pinned_instances:
                pinned_instances[key] = ModalityInstance(
                    name=item.name,
                    family=item.family,
                    sensor=item.sensor,
                    product_type=item.product_type,
                    band_names=item.band_names,
                    band_metadata=item.band_metadata,
                    orbit=item.orbit,
                    units=item.units,
                    signed=item.signed,
                    image=item.image.pin_memory(),
                    valid_mask=item.valid_mask.pin_memory(),
                    native_gsd_m=item.native_gsd_m,
                    aligned_gsd_m=item.aligned_gsd_m,
                    quality=item.quality,
                    metadata=item.metadata,
                )
            return pinned_instances[key]

        return ModalityBatch(
            instances=[[pin_instance(item) for item in sample] for sample in self.instances],
            full_instances=[[pin_instance(item) for item in sample] for sample in self.full_instances],
            active_subsets=self.active_subsets,
            mask=self.mask.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
            metadata=self.metadata,
            proposal_context_text=self.proposal_context_text,
            condition_prompt_text=self.condition_prompt_text,
            evidence_reasoning_text=self.evidence_reasoning_text,
            full_proposal_context_text=self.full_proposal_context_text,
            full_condition_prompt_text=self.full_condition_prompt_text,
            full_evidence_reasoning_text=self.full_evidence_reasoning_text,
            visual_evidence_key=self.visual_evidence_key,
            component_masks=(
                [value.pin_memory() for value in self.component_masks]
                if self.component_masks is not None else None
            ),
        )

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "instances",
                "full_instances",
                "active_subsets",
                "mask",
                "valid_mask",
                "metadata",
                "proposal_context_text",
                "condition_prompt_text",
                "evidence_reasoning_text",
                "full_proposal_context_text",
                "full_condition_prompt_text",
                "full_evidence_reasoning_text",
                "visual_evidence_key",
                "component_masks",
                "availability",
            )
        )

    def __len__(self) -> int:
        return 15


@dataclass
class ModalityPyramid:
    """Native-scale features for one modality instance."""

    instance: ModalityInstance
    high: torch.Tensor
    detail: torch.Tensor
    mid: torch.Tensor
    low: torch.Tensor
    high_valid: torch.Tensor
    detail_valid: torch.Tensor
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
class TaskNeutralVisualEvidence:
    """Parent-level visual tokens selected only by the active modality subset."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    family_ids: torch.Tensor
    token_counts: tuple[int, ...]
    view_segments: list[list[tuple[str, int]]]
    cache_keys: tuple[str, ...]
    cache_format: str


@dataclass
class MultisourceBackboneState:
    """Task-neutral SANE state reusable by segmentation and region description."""

    features: MultiScaleFeatures
    valid_mask: torch.Tensor
    active_subsets: tuple[ActiveModalitySubset, ...]
    metadata: tuple[dict[str, Any], ...]
    reference_hw: tuple[int, int]
    use_full_evidence: bool
    visual_evidence: TaskNeutralVisualEvidence | None = None


@dataclass
class SemanticEvidence:
    """Unified task, condition, reasoning and optional visual evidence tokens."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    task_token: torch.Tensor
    condition_token: torch.Tensor
    global_token: torch.Tensor
    mask_query_states: torch.Tensor | None = None
    evidence_anchors: torch.Tensor | None = None
    visual_token_count: int = 0
    sequence_lengths: tuple[int, ...] = ()
    visual_token_counts: tuple[int, ...] = ()
    visual_delta_norm: torch.Tensor | None = None


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
    modality_detail: torch.Tensor
    modality_valid_high: torch.Tensor
    modality_valid_mid: torch.Tensor
    modality_valid_low: torch.Tensor
    modality_valid_detail: torch.Tensor
    modality_active: torch.Tensor
    reliability_logits: torch.Tensor
    reliability_weights: torch.Tensor
    null_reliability: torch.Tensor
    real_reliability_mass: torch.Tensor
    coverage_ratio: torch.Tensor
    modality_semantic_anchors: torch.Tensor
    modality_names: list[list[str]]


@dataclass
class SegmentationState:
    """Segmentation-specific semantic and QMEF state over a backbone encoding."""

    backbone: MultisourceBackboneState
    semantic: SemanticEvidence
    evidence: EvidenceFeatures


@dataclass
class RegionEvidenceState:
    """Region-conditioned evidence state populated by MGRR without altering SANE."""

    backbone: MultisourceBackboneState
    region_masks: torch.Tensor
    region_valid_mask: torch.Tensor
    region_tokens: torch.Tensor | None = None
    region_sequence_tokens: torch.Tensor | None = None
    region_sequence_mask: torch.Tensor | None = None
    context_tokens: torch.Tensor | None = None
    geometry_tokens: torch.Tensor | None = None
    modality_tokens: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ProposalSet:
    """PSALM-style proposal set before semantic union."""

    mask_logits: torch.Tensor
    coarse_mask_logits: torch.Tensor
    relevance_logits: torch.Tensor
    query_embeddings: torch.Tensor
    query_evidence: torch.Tensor
    query_modality_attention: torch.Tensor
    query_scale_attention: torch.Tensor
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
        values = {
            "final_mask_logits": self.final_mask_logits,
            "proposal_mask_logits": self.proposals.mask_logits,
            "proposal_coarse_mask_logits": self.proposals.coarse_mask_logits,
            "proposal_relevance_logits": relevance,
            "proposal_relevance_gates": calibrated_relevance_gates(relevance),
            "query_embeddings": self.proposals.query_embeddings,
            "query_modality_attention": self.proposals.query_modality_attention,
            "query_scale_attention": self.proposals.query_scale_attention,
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
