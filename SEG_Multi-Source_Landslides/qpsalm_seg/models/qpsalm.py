#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SANE -> QMEF -> PMRD assembly for Multi-Source Qwen-PSALM-Seg."""

from __future__ import annotations

import torch
from torch import nn

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.losses import proposal_set_losses
from qpsalm_seg.schema import ModalityBatch, SegmentationOutput, SemanticEvidence

from .pmrd import ProposalSetMaskRefinementDecoder
from .qmef import QwenGuidedEvidenceFusion
from .sane import SensorAwareNativeScaleEncoder
from .visual_evidence import CachedQwenVisualEvidenceBank


class SemanticEvidenceController(nn.Module):
    """Fuse task, condition, reasoning and visual tokens into one evidence object."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.type_embedding = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.pool_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        task: torch.Tensor,
        condition: torch.Tensor,
        reasoning: torch.Tensor,
        visual_tokens: torch.Tensor | None,
        visual_mask: torch.Tensor | None = None,
    ) -> SemanticEvidence:
        tokens = [
            task + self.type_embedding[0],
            condition + self.type_embedding[1],
            reasoning + self.type_embedding[2],
        ]
        visual_count = 0
        if visual_tokens is not None:
            if visual_tokens.ndim == 2:
                visual_tokens = visual_tokens[:, None]
            visual_count = int(visual_tokens.shape[1])
            visual_tokens = visual_tokens + self.type_embedding[3][None, None]
            tokens.extend(visual_tokens.unbind(dim=1))
        stacked = torch.stack(tokens, dim=1)
        token_mask = torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
        if visual_count and visual_mask is not None:
            token_mask[:, -visual_count:] = visual_mask.to(device=stacked.device, dtype=torch.bool)
        query = self.pool_token.expand(stacked.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            query,
            stacked,
            stacked,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        global_token = self.norm(pooled[:, 0] + task)
        return SemanticEvidence(
            tokens=stacked,
            token_mask=token_mask,
            task_token=task,
            condition_token=condition,
            global_token=global_token,
            visual_token_count=visual_count,
        )


class MultiSourceQwenPSALMSeg(nn.Module):
    """Three-module research model for instruction-conditioned landslide masks."""

    def __init__(self, config: QPSalmConfig, controller: nn.Module) -> None:
        super().__init__()
        self.config = config
        dim = int(config.decoder_dim)
        self.controller = controller
        self.semantic_controller = SemanticEvidenceController(dim, int(config.num_heads))
        self.sane = SensorAwareNativeScaleEncoder(
            dim,
            modality_dropout=float(config.modality_dropout),
        )
        self.qmef = QwenGuidedEvidenceFusion(
            dim,
            deformable_points=int(getattr(config, "deformable_points", 4)),
        )
        self.pmrd = ProposalSetMaskRefinementDecoder(
            dim,
            num_queries=int(config.num_mask_tokens),
            num_layers=int(config.num_decoder_layers),
            num_heads=int(config.num_heads),
        )
        visual_cache = getattr(config, "visual_evidence_cache", None)
        self.visual_cache = CachedQwenVisualEvidenceBank(visual_cache, dim) if visual_cache else None

    def _semantic_evidence(self, batch: ModalityBatch, device: torch.device) -> SemanticEvidence:
        task = self.controller(batch.proposal_context_text, device=device)
        condition = self.controller(batch.condition_prompt_text, device=device)
        reasoning = self.controller(batch.evidence_reasoning_text, device=device)
        visual_tokens = None
        visual_mask = None
        if self.visual_cache is not None:
            visual_tokens, visual_mask = self.visual_cache(batch.visual_evidence_key, device=device)
            visual_tokens = visual_tokens.to(task.dtype)
        return self.semantic_controller(task, condition, reasoning, visual_tokens, visual_mask)

    def _decode(
        self,
        batch: ModalityBatch,
        semantic: SemanticEvidence,
        apply_modality_dropout: bool,
    ) -> SegmentationOutput:
        pyramids = self.sane(batch, apply_dropout=apply_modality_dropout)
        evidence = self.qmef(pyramids, semantic)
        coarse_queries, coarse_masks = self.pmrd.propose(evidence, semantic.task_token, batch.reference_hw)
        if bool(getattr(self.config, "use_query_spatial_attention", True)):
            query_evidence, modality_weights, spatial_entropy = self.qmef.attend_queries(coarse_queries, evidence)
        else:
            pooled = evidence.fused_mid.mean(dim=(2, 3))
            query_evidence = pooled[:, None].expand_as(coarse_queries)
            modality_weights = evidence.reliability_weights[:, None].expand(-1, coarse_queries.shape[1], -1)
            spatial_entropy = coarse_queries.new_zeros(coarse_queries.shape[:2])
        if bool(getattr(self.config, "use_mask_refinement", True)):
            queries, masks = self.pmrd.refine(
                coarse_queries,
                coarse_masks,
                query_evidence,
                evidence,
                batch.reference_hw,
            )
        else:
            queries, masks = coarse_queries, coarse_masks
        if queries.shape[1] == 1 and float(getattr(self.config, "semantic_verifier_loss_weight", 0.0)) <= 0.0:
            relevance = masks.new_full((masks.shape[0], 1), 8.0)
        else:
            relevance = self.qmef.verify(queries, query_evidence, semantic)
        proposals = self.pmrd.build_proposal_set(
            masks,
            coarse_masks,
            relevance,
            queries,
            query_evidence,
            modality_weights,
            spatial_entropy,
        )
        final_logits = self.pmrd.compose_final_mask(masks, relevance)
        return SegmentationOutput(
            final_mask_logits=final_logits,
            proposals=proposals,
            diagnostics={
                "modality_reliability_weights": evidence.reliability_weights.detach(),
                "modality_active": evidence.modality_active.detach(),
                "modality_reliability_logits": evidence.reliability_logits.detach(),
                "query_spatial_entropy_mean": spatial_entropy.mean(dim=1).detach(),
                "query_modality_attention_peak": modality_weights.max(dim=2).values.mean(dim=1).detach(),
            },
        )

    @staticmethod
    def _consistency_loss(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = (torch.sigmoid(student) - torch.sigmoid(teacher)).square() * valid
        return values.sum() / valid.sum().clamp_min(1.0)

    def forward(self, batch: ModalityBatch) -> SegmentationOutput:
        if not isinstance(batch, ModalityBatch):
            raise TypeError(f"MultiSourceQwenPSALMSeg expects ModalityBatch, got {type(batch).__name__}")
        device = next(self.parameters()).device
        semantic = self._semantic_evidence(batch, device)
        consistency_weight = float(getattr(self.config, "missing_modality_consistency_weight", 0.0))
        teacher_logits = None
        if self.training and consistency_weight > 0.0 and any(len(items) > 1 for items in batch.instances):
            with torch.no_grad():
                teacher_logits = self._decode(batch, semantic, apply_modality_dropout=False).final_mask_logits.detach()
        output = self._decode(batch, semantic, apply_modality_dropout=True)
        valid = batch.valid_mask.to(device=device, dtype=output.final_mask_logits.dtype)
        consistency = (
            self._consistency_loss(output.final_mask_logits, teacher_logits, valid)
            if teacher_logits is not None
            else output.final_mask_logits.sum() * 0.0
        )
        output.update_losses(
            proposal_set_losses(
                output,
                batch,
                final_bce_weight=float(getattr(self.config, "final_bce_weight", 1.0)),
                final_dice_weight=float(getattr(self.config, "final_dice_weight", 1.0)),
                proposal_set_weight=float(getattr(self.config, "proposal_set_loss_weight", 0.75)),
                coarse_proposal_weight=float(getattr(self.config, "coarse_proposal_loss_weight", 0.25)),
                verifier_weight=float(getattr(self.config, "semantic_verifier_loss_weight", 0.25)),
                boundary_weight=float(getattr(self.config, "boundary_loss_weight", 0.0)),
                missing_modality_consistency=consistency,
                consistency_weight=consistency_weight,
                min_component_area_fraction=float(getattr(self.config, "min_component_area_fraction", 5.0e-5)),
                min_component_area_pixels=int(getattr(self.config, "min_component_area_pixels", 4)),
            )
        )
        return output
