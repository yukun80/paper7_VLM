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
from .visual_evidence import CachedQwenVisualEvidenceBank, VisualEvidenceAdapter
from .modality import MultiSourceAdapterBank


class MultiSourceQwenPSALMSeg(nn.Module):
    """面向多源遥感滑坡 instruction segmentation 的 VLM-Seg + PSALM 原型。"""

    def __init__(self, config: QPSalmConfig, controller: nn.Module) -> None:
        super().__init__()
        self.config = config
        d = int(config.decoder_dim)
        self.controller = controller
        self.gsd_embedding = nn.Embedding(len(GSD_TOKENS), d)
        self.gsd_continuous_proj = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.gsd_feature_film = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
        )
        self.adapters = MultiSourceAdapterBank(d, float(config.modality_dropout))
        self.feature_fusion = MultiScaleFeatureFusion(
            d,
            use_box_prior=bool(config.use_box_prior),
            use_spatial_modality_gate=bool(getattr(config, "use_spatial_modality_gate", True)),
        )
        self.visual_evidence = VisualEvidenceAdapter(d) if bool(getattr(config, "use_visual_evidence", False)) else None
        visual_cache = getattr(config, "visual_evidence_cache", None)
        self.visual_evidence_cache = (
            CachedQwenVisualEvidenceBank(visual_cache, d)
            if bool(getattr(config, "use_visual_evidence", False)) and visual_cache
            else None
        )
        self.decoder = PSALMConditionAwareMaskDecoder(
            decoder_dim=d,
            num_queries=int(config.num_mask_tokens),
            num_layers=int(config.num_decoder_layers),
            num_heads=int(config.num_heads),
            selection_proposal_weight=float(config.selection_proposal_weight),
            selection_condition_weight=float(config.selection_condition_weight),
            selection_evidence_weight=float(getattr(config, "selection_evidence_weight", 0.25)),
            selection_visual_evidence_weight=float(getattr(config, "selection_visual_evidence_weight", 0.15)),
            selection_temperature=float(config.selection_temperature),
            final_foreground_gate_weight=float(config.final_foreground_gate_weight),
            final_mask_fusion=str(config.final_mask_fusion),
            final_topk=int(config.final_topk),
            final_noisy_or_epsilon=float(config.final_noisy_or_epsilon),
            use_query_modality_attention=bool(getattr(config, "use_query_modality_attention", True)),
            query_modality_feature_weight=float(getattr(config, "query_modality_feature_weight", 0.35)),
        )

    def _gsd_context(self, batch: dict[str, object], device: torch.device) -> torch.Tensor:
        gsd_id = batch["gsd_id"].to(device)  # type: ignore[union-attr]
        gsd = self.gsd_embedding(gsd_id)
        gsd_continuous = batch.get("gsd_continuous")
        if gsd_continuous is None:
            gsd_continuous = torch.zeros((gsd.shape[0], 2), dtype=gsd.dtype, device=device)
        else:
            gsd_continuous = gsd_continuous.to(device=device, dtype=gsd.dtype)  # type: ignore[union-attr]
        return gsd + self.gsd_continuous_proj(gsd_continuous)

    def _apply_gsd_film(self, features: torch.Tensor, gsd_context: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.config, "use_gsd_film", True)):
            return features
        scale, shift = self.gsd_feature_film(gsd_context).chunk(2, dim=-1)
        scale = 0.10 * torch.tanh(scale)
        shift = 0.10 * torch.tanh(shift)
        if features.ndim == 4:
            view_shape = (features.shape[0], features.shape[1], 1, 1)
        elif features.ndim == 5:
            view_shape = (features.shape[0], 1, features.shape[2], 1, 1)
        else:
            return features
        return features * (1.0 + scale.view(view_shape).to(features.dtype)) + shift.view(view_shape).to(features.dtype)

    def _encode_texts(
        self,
        batch: dict[str, object],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proposal_text = batch.get("proposal_context_text") or batch.get("condition_text")
        condition_text = batch.get("condition_prompt_text") or batch.get("condition_text")
        evidence_text = batch.get("evidence_reasoning_text") or batch.get("condition_text")
        if proposal_text is None or condition_text is None:
            raise KeyError("batch must contain proposal_context_text/condition_prompt_text or condition_text")
        if evidence_text is None:
            evidence_text = condition_text
        proposal_context = self.controller(proposal_text, device=device)  # type: ignore[arg-type]
        condition_embedding = self.controller(condition_text, device=device)  # type: ignore[arg-type]
        if bool(getattr(self.config, "use_evidence_reasoning", True)):
            evidence_embedding = self.controller(evidence_text, device=device)  # type: ignore[arg-type]
        else:
            evidence_embedding = torch.zeros_like(condition_embedding)
        gsd_context = self._gsd_context(batch, device)
        return (
            proposal_context + gsd_context,
            condition_embedding + gsd_context,
            evidence_embedding + gsd_context,
            gsd_context,
        )

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        modalities = {name: tensor.to(device) for name, tensor in batch["modalities"].items()}  # type: ignore[index]
        availability = batch["availability"].to(device)  # type: ignore[union-attr]
        proposal_context, condition_embedding, evidence_embedding, gsd_context = self._encode_texts(batch, device)
        evidence_enabled = bool(getattr(self.config, "use_evidence_reasoning", True))
        evidence_weight = float(getattr(self.config, "evidence_reasoning_weight", 0.35)) if evidence_enabled else 0.0
        if evidence_enabled and evidence_weight:
            modality_gate_condition = condition_embedding + evidence_weight * evidence_embedding
        else:
            modality_gate_condition = condition_embedding
        bbox_prior = batch["bbox_prior"].to(device) if "bbox_prior" in batch else None  # type: ignore[union-attr]
        visual_out: dict[str, torch.Tensor] = {}
        visual_evidence_embedding: torch.Tensor | None = None
        visual_weight = float(getattr(self.config, "visual_evidence_weight", 0.25))
        if self.visual_evidence is not None and "visual_preview" in batch:
            visual_preview = batch["visual_preview"].to(device)  # type: ignore[union-attr]
            visual_out = self.visual_evidence(visual_preview)
            visual_evidence_embedding = visual_weight * visual_out["visual_evidence_embedding"].to(proposal_context.dtype)
        if self.visual_evidence_cache is not None:
            keys = batch.get("visual_evidence_key")
            if keys is None:
                raise KeyError("batch must contain visual_evidence_key when visual_evidence_cache is configured")
            cached_visual = self.visual_evidence_cache(keys, device=device)  # type: ignore[arg-type]
            cached_visual = cached_visual.to(proposal_context.dtype)
            visual_evidence_embedding = (
                visual_weight * cached_visual
                if visual_evidence_embedding is None
                else visual_evidence_embedding + visual_weight * cached_visual
            )
            visual_out["qwen_visual_evidence_embedding"] = cached_visual

        modality_out = self.adapters(
            modalities,
            availability,
            modality_gate_condition,
            proposal_context=proposal_context,
        )
        modality_out["fused"] = self._apply_gsd_film(modality_out["fused"], gsd_context)
        if "stacked_features" in modality_out:
            modality_out["stacked_features"] = self._apply_gsd_film(modality_out["stacked_features"], gsd_context)
        fusion_out = self.feature_fusion(
            modality_out["fused"],
            bbox_prior=bbox_prior,
            modality_features=modality_out.get("stacked_features"),
            gate_weights=modality_out.get("modality_gate_weights"),
            condition_embedding=condition_embedding,
        )
        if visual_out:
            feature_weight = float(getattr(self.config, "visual_evidence_feature_weight", 0.15))
            visual_features = visual_out["visual_evidence_features"].to(fusion_out["mask_features"].dtype)
            if visual_features.shape[-2:] != fusion_out["mask_features"].shape[-2:]:
                visual_features = torch.nn.functional.interpolate(
                    visual_features,
                    size=fusion_out["mask_features"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fusion_out["mask_features"] = fusion_out["mask_features"] + feature_weight * visual_features
            memory_visual = torch.nn.functional.interpolate(
                visual_features,
                size=fusion_out["memory_features"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fusion_out["memory_features"] = fusion_out["memory_features"] + feature_weight * memory_visual
        outputs = self.decoder(
            mask_features=fusion_out["mask_features"],
            memory_features=fusion_out["memory_features"],
            proposal_context=proposal_context,
            condition_embedding=condition_embedding,
            evidence_embedding=evidence_embedding if evidence_enabled else None,
            visual_evidence_embedding=visual_evidence_embedding,
            modality_features=modality_out.get("stacked_features"),
            modality_active_mask=modality_out.get("modality_active_mask"),
            global_modality_gate=modality_out.get("modality_gate_weights"),
        )
        outputs["modality_gate_weights"] = modality_out["modality_gate_weights"].detach()
        outputs["modality_active_mask"] = modality_out["modality_active_mask"].detach()
        outputs["modality_gate_logits"] = modality_out["modality_gate_logits"].detach()
        outputs["modality_feature_norms"] = modality_out["modality_feature_norms"].detach()
        outputs["modality_gate_feature_norms"] = modality_out["modality_gate_feature_norms"].detach()
        if evidence_enabled:
            outputs["evidence_embedding_norm"] = evidence_embedding.detach().float().pow(2).mean(dim=1).sqrt()
        if visual_out:
            for key in (
                "visual_evidence_attention_mean",
                "visual_evidence_attention_max",
                "visual_evidence_embedding",
                "qwen_visual_evidence_embedding",
            ):
                if key in visual_out:
                    outputs[key] = visual_out[key].detach()
        for scale_name in ("high", "mid", "low"):
            key = f"scale_gate_{scale_name}"
            if key in fusion_out:
                outputs[key] = fusion_out[key].detach()
            global_key = f"global_scale_gate_{scale_name}"
            if global_key in fusion_out:
                outputs[global_key] = fusion_out[global_key].detach()
            for suffix in ("entropy", "peak"):
                spatial_key = f"spatial_gate_{scale_name}_{suffix}"
                if spatial_key in fusion_out:
                    outputs[spatial_key] = fusion_out[spatial_key].detach()
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
                    evidence_positive_weight=self.config.evidence_positive_weight,
                    query_diversity_loss_weight=self.config.query_diversity_loss_weight,
                    evidence_cls_weight=self.config.evidence_cls_weight if evidence_enabled else 0.0,
                    evidence_ranking_loss_weight=self.config.evidence_ranking_loss_weight if evidence_enabled else 0.0,
                    visual_evidence_cls_weight=(
                        self.config.visual_evidence_cls_weight if visual_evidence_embedding is not None else 0.0
                    ),
                    visual_evidence_ranking_loss_weight=(
                        self.config.visual_evidence_ranking_loss_weight if visual_evidence_embedding is not None else 0.0
                    ),
                    proposal_mask_diversity_loss_weight=self.config.proposal_mask_diversity_loss_weight,
                    gate_entropy_loss_weight=self.config.gate_entropy_loss_weight,
                    proposal_soft_target_topk=self.config.proposal_soft_target_topk,
                    proposal_soft_target_temperature=self.config.proposal_soft_target_temperature,
                    query_usage_balance_loss_weight=self.config.query_usage_balance_loss_weight,
                )
            )
        return outputs
