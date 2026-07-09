#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-Source Qwen-PSALM-Seg 总装模型。"""

from __future__ import annotations

import torch
from torch import nn

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.data import CANONICAL_MODALITIES, GSD_TOKENS
from qpsalm_seg.losses import segmentation_losses

from .decoder import PSALMConditionAwareMaskDecoder
from .fusion import MultiScaleFeatureFusion
from .modality import MultiSourceAdapterBank


class MultiSourceQwenPSALMSeg(nn.Module):
    """面向多源遥感滑坡 instruction segmentation 的 VLM-Seg + PSALM 原型。"""

    def __init__(self, config: QPSalmConfig, controller: nn.Module) -> None:
        super().__init__()
        self.config = config
        d = int(config.decoder_dim)
        self.controller = controller
        self.gsd_embedding = nn.Embedding(len(GSD_TOKENS), d)
        self.adapters = MultiSourceAdapterBank(d, float(config.modality_dropout))
        self.feature_fusion = MultiScaleFeatureFusion(d, use_box_prior=bool(config.use_box_prior))
        self.decoder = PSALMConditionAwareMaskDecoder(
            decoder_dim=d,
            num_queries=int(config.num_mask_tokens),
            num_layers=int(config.num_decoder_layers),
            num_heads=int(config.num_heads),
            selection_proposal_weight=float(config.selection_proposal_weight),
            selection_condition_weight=float(config.selection_condition_weight),
            selection_temperature=float(config.selection_temperature),
            final_foreground_gate_weight=float(config.final_foreground_gate_weight),
            final_mask_fusion=str(config.final_mask_fusion),
            final_topk=int(config.final_topk),
            final_noisy_or_epsilon=float(config.final_noisy_or_epsilon),
        )

    def _encode_texts(self, batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        proposal_text = batch.get("proposal_context_text") or batch.get("condition_text")
        condition_text = batch.get("condition_prompt_text") or batch.get("condition_text")
        if proposal_text is None or condition_text is None:
            raise KeyError("batch must contain proposal_context_text/condition_prompt_text or condition_text")
        proposal_context = self.controller(proposal_text, device=device)  # type: ignore[arg-type]
        condition_embedding = self.controller(condition_text, device=device)  # type: ignore[arg-type]
        gsd_id = batch["gsd_id"].to(device)  # type: ignore[union-attr]
        gsd = self.gsd_embedding(gsd_id)
        return proposal_context + gsd, condition_embedding + gsd

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        modalities = {name: tensor.to(device) for name, tensor in batch["modalities"].items()}  # type: ignore[index]
        availability = batch["availability"].to(device)  # type: ignore[union-attr]
        proposal_context, condition_embedding = self._encode_texts(batch, device)

        modality_out = self.adapters(
            modalities,
            availability,
            condition_embedding,
            proposal_context=proposal_context,
        )
        bbox_prior = batch["bbox_prior"].to(device) if "bbox_prior" in batch else None  # type: ignore[union-attr]
        fusion_out = self.feature_fusion(
            modality_out["fused"],
            bbox_prior=bbox_prior,
            modality_features=modality_out.get("stacked_features"),
            gate_weights=modality_out.get("modality_gate_weights"),
            condition_embedding=condition_embedding,
        )
        outputs = self.decoder(
            mask_features=fusion_out["mask_features"],
            memory_features=fusion_out["memory_features"],
            proposal_context=proposal_context,
            condition_embedding=condition_embedding,
        )
        outputs["modality_gate_weights"] = modality_out["modality_gate_weights"].detach()
        outputs["modality_active_mask"] = modality_out["modality_active_mask"].detach()
        outputs["modality_gate_logits"] = modality_out["modality_gate_logits"].detach()
        outputs["modality_feature_norms"] = modality_out["modality_feature_norms"].detach()
        outputs["modality_gate_feature_norms"] = modality_out["modality_gate_feature_norms"].detach()
        for scale_name in ("high", "mid", "low"):
            key = f"scale_gate_{scale_name}"
            if key in fusion_out:
                outputs[key] = fusion_out[key].detach()
        outputs["modality_gate_weights_for_loss"] = modality_out["modality_gate_weights"]
        outputs["modality_active_mask_for_loss"] = modality_out["modality_active_mask"]

        if "mask" in batch:
            target = batch["mask"].to(device)  # type: ignore[union-attr]
            is_empty = batch["is_empty"].to(device)  # type: ignore[union-attr]
            sample_weights = batch.get("sample_weight")
            sample_weights = sample_weights.to(device) if sample_weights is not None else None  # type: ignore[union-attr]
            outputs.update(
                segmentation_losses(
                    outputs,
                    target,
                    is_empty,
                    sample_weights=sample_weights,
                    use_focal=self.config.use_focal_loss,
                    boundary_weight=self.config.boundary_loss_weight,
                    condition_ranking_weight=self.config.condition_ranking_loss_weight,
                    selection_ranking_loss_weight=self.config.selection_ranking_loss_weight,
                    foreground_bce_pos_weight=self.config.foreground_bce_pos_weight,
                    mask_bce_weight=self.config.mask_bce_weight,
                    mask_dice_weight=self.config.mask_dice_weight,
                    mask_tversky_weight=self.config.mask_tversky_weight,
                    tversky_alpha=self.config.tversky_alpha,
                    tversky_beta=self.config.tversky_beta,
                    proposal_cls_weight=self.config.proposal_cls_weight,
                    condition_cls_weight=self.config.condition_cls_weight,
                    proposal_mask_weight=self.config.proposal_mask_weight,
                    empty_mask_suppression_weight=self.config.empty_mask_suppression_weight,
                    empty_proposal_suppression_weight=self.config.empty_proposal_suppression_weight,
                    proposal_positive_weight=self.config.proposal_positive_weight,
                    condition_positive_weight=self.config.condition_positive_weight,
                    query_diversity_loss_weight=self.config.query_diversity_loss_weight,
                    proposal_mask_diversity_loss_weight=self.config.proposal_mask_diversity_loss_weight,
                    gate_entropy_loss_weight=self.config.gate_entropy_loss_weight,
                    proposal_soft_target_topk=self.config.proposal_soft_target_topk,
                    proposal_soft_target_temperature=self.config.proposal_soft_target_temperature,
                    query_usage_balance_loss_weight=self.config.query_usage_balance_loss_weight,
                )
            )
        return outputs
