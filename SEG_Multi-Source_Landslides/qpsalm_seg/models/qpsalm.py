#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SANE -> QMEF -> PMRD assembly for benchmark-v2 experiments."""

from __future__ import annotations

import torch
from torch import nn

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.controllers import (
    build_controller,
    local_model_revision,
    local_processor_revision,
    validate_qwen_model_dir,
)
from qpsalm_seg.losses import proposal_set_losses
from qpsalm_seg.schema import (
    ActiveModalitySubset,
    ModalityBatch,
    MultisourceBackboneState,
    SegmentationOutput,
    SegmentationState,
    SemanticEvidence,
    TaskNeutralVisualEvidence,
)

from .pmrd import ProposalSetMaskRefinementDecoder
from .qmef import QwenGuidedEvidenceFusion
from .sane import SensorAwareNativeScaleEncoder
from .vision_cache import QwenVisionFeatureBank, vision_input_protocol


class MultiSourceQwenPSALMSeg(nn.Module):
    """Strict v2 model: subset-first evidence, Qwen queries, proposal-set masks."""

    def __init__(self, config: QPSalmConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        dim = int(config.decoder_dim)
        cache_path = config.vision_feature_cache
        needs_cache = bool(config.use_pretrained_sane or config.controller == "qwen_mask_query")
        if needs_cache and not cache_path:
            raise ValueError(f"preset={config.preset} 需要 vision_feature_cache v3")
        self.vision_bank = (
            QwenVisionFeatureBank(
                cache_path,
                dim,
                ram_budget_gib=float(config.vision_cache_ram_budget_gib),
                visual_ablation=config.visual_ablation,
            )
            if needs_cache and cache_path else None
        )
        if self.vision_bank is not None:
            expected_input_protocol = vision_input_protocol(config)
            if self.vision_bank.manifest.get("input_protocol") != expected_input_protocol:
                raise ValueError(
                    "Qwen vision cache input protocol 与模型不一致: "
                    f"cache={self.vision_bank.manifest.get('input_protocol')} "
                    f"model={expected_input_protocol}"
                )
            cached_view_tokens = int(self.vision_bank.manifest.get("view_tokens_per_view") or 0)
            if cached_view_tokens < int(config.qwen_view_tokens_per_view):
                raise ValueError(
                    f"Qwen vision cache 每 view 仅有 {cached_view_tokens} tokens，"
                    f"模型请求 {config.qwen_view_tokens_per_view}"
                )
        if self.vision_bank is not None and self.vision_bank.manifest.get("backend") != "hash-smoke":
            model_dir = validate_qwen_model_dir(config.qwen_model_path)
            expected = {
                "model_revision": local_model_revision(model_dir),
                "processor_revision": local_processor_revision(model_dir),
            }
            mismatched = {
                key: {"cache": self.vision_bank.manifest.get(key), "local": value}
                for key, value in expected.items()
                if self.vision_bank.manifest.get(key) != value
            }
            if mismatched:
                raise ValueError(f"Qwen vision cache revision 与本地模型不一致: {mismatched}")
        self.controller = build_controller(config, device, self.vision_bank)
        self.sane = SensorAwareNativeScaleEncoder(
            dim,
            pretrained_bank=self.vision_bank if config.use_pretrained_sane else None,
        )
        self.qmef = QwenGuidedEvidenceFusion(dim, deformable_points=int(config.deformable_points))
        self.pmrd = ProposalSetMaskRefinementDecoder(
            dim,
            num_queries=int(config.num_mask_tokens),
            num_layers=int(config.num_decoder_layers),
            num_heads=int(config.num_heads),
            query_chunk_size=int(config.query_chunk_size),
        )

    @staticmethod
    def _full_subsets(batch: ModalityBatch) -> list[ActiveModalitySubset]:
        return [
            ActiveModalitySubset(
                active_names=tuple(item.name for item in instances),
                dropped_names=(),
                signature="teacher-full",
                is_full=True,
            )
            for instances in batch.full_instances
        ]

    def _task_neutral_visual_evidence(
        self, batch: ModalityBatch, *, use_full: bool,
    ) -> TaskNeutralVisualEvidence | None:
        if self.vision_bank is None:
            return None
        device = next(self.parameters()).device
        subsets = self._full_subsets(batch) if use_full else batch.active_subsets
        tokens, mask, counts, family_ids, segments = self.vision_bank.tokens_for(
            batch.visual_evidence_key,
            subsets,
            device,
            int(self.config.qwen_view_tokens_per_view),
        )
        return TaskNeutralVisualEvidence(
            tokens=tokens,
            token_mask=mask,
            family_ids=family_ids,
            token_counts=tuple(int(value) for value in counts),
            view_segments=segments,
            cache_keys=tuple(str(value) for value in batch.visual_evidence_key),
            cache_format=str(self.vision_bank.manifest.get("format") or "unknown"),
        )

    def encode_multisource(
        self,
        batch: ModalityBatch,
        *,
        use_full: bool = False,
        include_visual_tokens: bool = True,
    ) -> MultisourceBackboneState:
        """Encode task-neutral multisource evidence once for downstream tasks."""
        if not isinstance(batch, ModalityBatch):
            raise TypeError(f"expected ModalityBatch, got {type(batch).__name__}")
        pyramids = self.sane(batch, use_full=use_full)
        device = next(self.parameters()).device
        metadata_fields = {
            "sample_id", "parent_sample_id", "dataset_name", "split", "original_size",
            "resize_transform", "raw_modality_combo", "modality_family_combo",
        }
        task_neutral_metadata = tuple(
            {key: value for key, value in row.items() if key in metadata_fields}
            for row in batch.metadata
        )
        return MultisourceBackboneState(
            features=pyramids,
            valid_mask=batch.valid_mask.to(device=device, non_blocking=True),
            active_subsets=tuple(self._full_subsets(batch) if use_full else batch.active_subsets),
            metadata=task_neutral_metadata,
            reference_hw=batch.reference_hw,
            use_full_evidence=bool(use_full),
            visual_evidence=(
                self._task_neutral_visual_evidence(batch, use_full=use_full)
                if include_visual_tokens else None
            ),
        )

    def build_segmentation_state(
        self,
        batch: ModalityBatch,
        *,
        use_full: bool = False,
        backbone: MultisourceBackboneState | None = None,
    ) -> SegmentationState:
        """Build segmentation-only semantic/QMEF state over a reusable backbone."""
        semantic = self.controller.encode_batch(batch, use_full=use_full)
        backbone_state = backbone or self.encode_multisource(
            batch, use_full=use_full, include_visual_tokens=False
        )
        if backbone_state.use_full_evidence != bool(use_full):
            raise ValueError("backbone evidence subset 与 segmentation state 不一致")
        evidence = self.qmef(
            backbone_state.features,
            semantic,
            enable_semantic=bool(self.config.use_qmef),
            enable_reliability=bool(self.config.use_qmef),
        )
        return SegmentationState(backbone=backbone_state, semantic=semantic, evidence=evidence)

    def segment_from_state(self, state: SegmentationState) -> SegmentationOutput:
        """Decode PMRD proposals without re-encoding SANE or Qwen evidence."""
        semantic = state.semantic
        evidence = state.evidence
        reference_hw = state.backbone.reference_hw
        coarse_queries, coarse_masks = self.pmrd.propose(evidence, semantic, reference_hw)
        if self.config.use_query_spatial_attention:
            (
                query_evidence,
                modality_weights,
                scale_weights,
                spatial_entropy,
                sampling_reference,
                sampling_grid,
            ) = self.qmef.attend_queries(coarse_queries, evidence, coarse_masks)
        else:
            pooled = evidence.fused_mid.mean(dim=(2, 3))
            query_evidence = pooled[:, None].expand_as(coarse_queries)
            modality_weights = evidence.reliability_weights[:, None].expand(-1, coarse_queries.shape[1], -1)
            scale_weights = coarse_queries.new_full((*coarse_queries.shape[:2], 3), 1.0 / 3.0)
            spatial_entropy = coarse_queries.new_zeros(coarse_queries.shape[:2])
            sampling_reference = coarse_queries.new_zeros((*coarse_queries.shape[:2], 2))
            sampling_grid = coarse_queries.new_zeros((
                *coarse_queries.shape[:2], 3, int(self.config.deformable_points), 2
            ))
        if self.config.use_mask_refinement:
            queries, masks = self.pmrd.refine(
                coarse_queries, coarse_masks, query_evidence, modality_weights, evidence, reference_hw
            )
        else:
            queries, masks = coarse_queries, coarse_masks
        if queries.shape[1] == 1 and self.config.semantic_verifier_loss_weight <= 0:
            relevance = masks.new_full((masks.shape[0], 1), 8.0)
        else:
            relevance = self.qmef.verify(queries, query_evidence, semantic)
        proposals = self.pmrd.build_proposal_set(
            masks, coarse_masks, relevance, queries, query_evidence,
            modality_weights, scale_weights, spatial_entropy,
        )
        sequence_lengths = masks.new_tensor(semantic.sequence_lengths, dtype=torch.long)
        visual_token_counts = masks.new_tensor(semantic.visual_token_counts, dtype=torch.long)
        max_sequence_length = sequence_lengths.max().clamp_min(1)
        padding_ratio = 1.0 - sequence_lengths.float().sum() / (
            max(sequence_lengths.numel(), 1) * max_sequence_length.float()
        )
        return SegmentationOutput(
            final_mask_logits=self.pmrd.compose_final_mask(masks, relevance),
            proposals=proposals,
            diagnostics={
                "modality_reliability_weights": evidence.reliability_weights.detach(),
                "modality_reliability_logits": evidence.reliability_logits.detach(),
                "modality_coverage_ratio": evidence.coverage_ratio.detach(),
                "modality_semantic_anchor_norm": evidence.modality_semantic_anchors.float().norm(dim=-1).detach(),
                "modality_active": evidence.modality_active.detach(),
                "null_evidence_weight": evidence.null_reliability.detach(),
                "real_evidence_mass": evidence.real_reliability_mass.detach(),
                "query_spatial_entropy_mean": spatial_entropy.mean(1).detach(),
                "query_modality_attention_peak": modality_weights.max(2).values.mean(1).detach(),
                "query_scale_attention_peak": scale_weights.max(2).values.mean(1).detach(),
                "query_sampling_reference": sampling_reference.detach(),
                "query_sampling_grid": sampling_grid.detach(),
                "controller_sequence_lengths": sequence_lengths,
                "controller_visual_token_counts": visual_token_counts,
                "controller_tokens_per_sample": sequence_lengths.float().mean(),
                "controller_padding_ratio": padding_ratio,
                **(
                    {"visual_evidence_delta_norm": semantic.visual_delta_norm.detach()}
                    if semantic.visual_delta_norm is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _consistency_loss(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = (torch.sigmoid(student) - torch.sigmoid(teacher)).square() * valid
        return values.sum() / valid.sum().clamp_min(1.0)

    def _teacher_mask_logits(self, batch: ModalityBatch) -> torch.Tensor:
        modules = list(self.modules())
        training_states = [module.training for module in modules]
        try:
            for module in modules:
                module.training = False
            with torch.no_grad():
                state = self.build_segmentation_state(batch, use_full=True)
                return self.segment_from_state(state).final_mask_logits.detach()
        finally:
            for module, training in zip(modules, training_states):
                module.training = training

    def forward(self, batch: ModalityBatch) -> SegmentationOutput:
        if not isinstance(batch, ModalityBatch):
            raise TypeError(f"expected ModalityBatch, got {type(batch).__name__}")
        consistency_weight = float(self.config.missing_modality_consistency_weight)
        dropped_indices = [
            index for index, subset in enumerate(batch.active_subsets) if not subset.is_full
        ]
        # Build the trainable student graph before running the stateful Qwen
        # controller in no-grad teacher mode. This prevents teacher inference
        # state from changing the QLoRA path used by the student forward.
        state = self.build_segmentation_state(batch, use_full=False)
        output = self.segment_from_state(state)
        teacher_logits = None
        if self.training and consistency_weight > 0 and dropped_indices:
            teacher_logits = self._teacher_mask_logits(batch.select(dropped_indices))
        valid = batch.valid_mask.to(
            device=output.final_mask_logits.device,
            dtype=output.final_mask_logits.dtype,
            non_blocking=True,
        )
        consistency = (
            self._consistency_loss(
                output.final_mask_logits.index_select(
                    0,
                    torch.tensor(dropped_indices, device=output.final_mask_logits.device),
                ),
                teacher_logits,
                valid.index_select(
                    0,
                    torch.tensor(dropped_indices, device=valid.device),
                ),
            )
            if teacher_logits is not None
            else output.final_mask_logits.sum() * 0.0
        )
        output.diagnostics["teacher_sample_count"] = output.final_mask_logits.new_tensor(
            float(len(dropped_indices))
        )
        output.diagnostics["teacher_sample_fraction"] = output.final_mask_logits.new_tensor(
            float(len(dropped_indices)) / max(batch.batch_size, 1)
        )
        output.update_losses(
            proposal_set_losses(
                output,
                batch,
                final_bce_weight=self.config.final_bce_weight,
                final_dice_weight=self.config.final_dice_weight,
                proposal_set_weight=self.config.proposal_set_loss_weight,
                coarse_proposal_weight=self.config.coarse_proposal_loss_weight,
                verifier_weight=self.config.semantic_verifier_loss_weight,
                boundary_weight=self.config.boundary_loss_weight,
                missing_modality_consistency=consistency,
                consistency_weight=consistency_weight,
                min_component_area_fraction=self.config.min_component_area_fraction,
                min_component_area_pixels=self.config.min_component_area_pixels,
            )
        )
        return output
