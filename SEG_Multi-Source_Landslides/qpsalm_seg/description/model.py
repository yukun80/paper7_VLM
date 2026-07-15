#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential segmentation-grounded description model with a named PEFT adapter."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.schema import (
    ModalityBatch,
    MultisourceBackboneState,
    RegionEvidenceState,
    SegmentationOutput,
)

from .mgrr import MultiGranularityRegionReplay, RegionProtocol
from .region_baselines import SingleVectorRegionPooling


DESCRIPTION_SEQUENCE_PROTOCOL = "qpsalm_description_causal_v2_multigranularity_tokens"
DESCRIPTION_ADAPTER_NAME = "desc_adapter"


@dataclass
class DescriptionForwardOutput:
    loss: torch.Tensor | None
    per_sample_loss: torch.Tensor | None
    logits: torch.Tensor
    labels: torch.Tensor | None
    region_state: RegionEvidenceState
    sequence_lengths: tuple[int, ...]


@dataclass
class SegmentThenDescribeOutput:
    segmentation: SegmentationOutput
    description: DescriptionForwardOutput


class SegmentationGroundedDescriptionModel(nn.Module):
    """Reuse SANE state, then activate desc_adapter for causal JSON generation."""

    def __init__(
        self,
        segmentation: MultiSourceQwenPSALMSeg,
        *,
        description_backbone: nn.Module | None = None,
        max_components: int = 8,
        component_coverage: float = 0.9,
        region_encoder: str = "mgrr",
    ) -> None:
        super().__init__()
        self.segmentation = segmentation
        self.description_backbone = description_backbone
        controller = segmentation.controller
        if not hasattr(controller, "adapter_scope") or not hasattr(controller, "tokenizer"):
            raise TypeError("Description Controller 需要 qwen_mask_query controller")
        controller.ensure_named_adapter(DESCRIPTION_ADAPTER_NAME)
        self.region_encoder_name = str(region_encoder)
        mgrr_ablation = {
            "mgrr": "full",
            "mgrr_no_context": "no_context",
            "roi_replay_only": "roi_replay_only",
        }.get(self.region_encoder_name)
        self.mgrr = (
            MultiGranularityRegionReplay(
                int(segmentation.config.decoder_dim),
                max_components=max_components,
                component_coverage=component_coverage,
                ablation=mgrr_ablation,
            )
            if mgrr_ablation is not None
            else SingleVectorRegionPooling(
                int(segmentation.config.decoder_dim), self.region_encoder_name
            )
        )
        hidden = int(controller.hidden_size)
        dim = int(segmentation.config.decoder_dim)
        self.region_to_hidden = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.description_view_to_hidden = copy.deepcopy(controller.view_to_hidden)
        self.region_type = nn.Parameter(torch.randn(hidden) * 0.02)
        self.instruction_type = nn.Parameter(torch.randn(hidden) * 0.02)
        self.visual_type = nn.Parameter(torch.randn(hidden) * 0.02)
        self.alignment_text_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.alignment_temperature = nn.Parameter(torch.tensor(0.07))

    @property
    def controller(self):
        return self.segmentation.controller

    def build_region_state(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        *,
        region_valid_mask: torch.Tensor | None = None,
        protocol: RegionProtocol = "vision_only",
    ) -> RegionEvidenceState:
        if self.region_encoder_name in {"mgrr", "mgrr_no_context", "roi_replay_only"}:
            return self.mgrr(
                backbone,
                region_masks,
                region_valid_mask=region_valid_mask,
                protocol=protocol,
            )
        return self.mgrr(
            backbone,
            region_masks,
            region_valid_mask=region_valid_mask,
        )

    def encode_description_requests(
        self, requests: Sequence[tuple[str, str]],
    ) -> MultisourceBackboneState:
        if self.description_backbone is None:
            raise RuntimeError("模型未配置 DescriptionCacheBackboneEncoder")
        return self.description_backbone(requests)

    def _token_ids(self, text: str, *, append_eos: bool) -> torch.Tensor:
        tokenizer = self.controller.tokenizer
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        values = [int(value) for value in ids]
        if append_eos and tokenizer.eos_token_id is not None:
            values.append(int(tokenizer.eos_token_id))
        if not values:
            fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            values = [int(fallback)]
        return torch.tensor(values, dtype=torch.long, device=next(self.parameters()).device)

    @staticmethod
    def _instruction_prompt(instruction: str, structured_output: bool) -> str:
        if structured_output:
            response_contract = (
                "Return exactly one JSON object following qpsalm_description_output_v1. "
                "Use unavailable or insufficient_evidence when the visual evidence does not support a claim."
            )
        else:
            response_contract = "Return only the requested English text without JSON or extra commentary."
        return "Task: " + instruction.strip() + "\n" + response_contract + "\nAnswer:"

    def _visual_tokens_for_sample(
        self, backbone: MultisourceBackboneState, sample_index: int,
    ) -> torch.Tensor | None:
        visual = backbone.visual_evidence
        if visual is None:
            return None
        selected = visual.tokens[sample_index][visual.token_mask[sample_index]]
        if not selected.numel():
            return None
        projected = self.description_view_to_hidden(selected.float())
        return projected + self.visual_type

    def _build_sequences(
        self,
        region_state: RegionEvidenceState,
        instructions: Sequence[str],
        targets: Sequence[str] | None,
        structured_outputs: Sequence[bool],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, ...]]:
        region_tokens = region_state.region_sequence_tokens
        region_mask = region_state.region_sequence_mask
        if region_tokens is None:
            summary = region_state.region_tokens
            if summary is None:
                raise ValueError("RegionEvidenceState 缺少 region tokens")
            region_tokens = summary[:, :, None]
            region_mask = torch.ones(
                region_tokens.shape[:3], dtype=torch.bool, device=region_tokens.device
            )
        if region_tokens.ndim != 4 or region_tokens.shape[1] != 1:
            raise ValueError("Description v1 每个样本要求恰好一个 region token sequence")
        if region_mask is None or region_mask.shape != region_tokens.shape[:3]:
            raise ValueError("region token sequence mask 与 token shape 不一致")
        batch_size = int(region_tokens.shape[0])
        if (
            len(instructions) != batch_size
            or len(structured_outputs) != batch_size
            or (targets is not None and len(targets) != batch_size)
        ):
            raise ValueError("instructions/targets 数量与 region batch 不一致")
        embedding = self.controller.model.get_input_embeddings()
        sequences = []
        labels = []
        lengths = []
        for index in range(batch_size):
            instruction_ids = self._token_ids(
                self._instruction_prompt(instructions[index], bool(structured_outputs[index])),
                append_eos=False,
            )
            instruction_embedding = embedding(instruction_ids) + self.instruction_type
            chunks = [instruction_embedding]
            visual = self._visual_tokens_for_sample(region_state.backbone, index)
            if visual is not None:
                chunks.append(visual.to(instruction_embedding.dtype))
            selected_region_tokens = region_tokens[index, 0][region_mask[index, 0]]
            region = self.region_to_hidden(selected_region_tokens.float()) + self.region_type
            chunks.append(region.to(instruction_embedding.dtype))
            prefix = torch.cat(chunks, 0)
            if targets is not None:
                target_ids = self._token_ids(targets[index], append_eos=True)
                target_embedding = embedding(target_ids)
                sequence = torch.cat([prefix, target_embedding], 0)
                label = torch.cat([
                    target_ids.new_full((prefix.shape[0],), -100), target_ids,
                ])
            else:
                sequence = prefix
                label = None
            sequences.append(sequence)
            labels.append(label)
            lengths.append(int(sequence.shape[0]))
        padded = pad_sequence(sequences, batch_first=True)
        attention = torch.arange(padded.shape[1], device=padded.device)[None] < torch.tensor(
            lengths, device=padded.device
        )[:, None]
        padded_labels = (
            pad_sequence(labels, batch_first=True, padding_value=-100)
            if targets is not None else None
        )
        return padded, attention, padded_labels, tuple(lengths)

    def describe_from_state(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        instructions: Sequence[str],
        *,
        target_texts: Sequence[str] | None = None,
        region_valid_mask: torch.Tensor | None = None,
        protocol: RegionProtocol = "vision_only",
        structured_output: bool | Sequence[bool] = True,
    ) -> DescriptionForwardOutput:
        region_state = self.build_region_state(
            backbone, region_masks, region_valid_mask=region_valid_mask, protocol=protocol
        )
        structured_outputs = (
            [bool(structured_output)] * len(instructions)
            if isinstance(structured_output, bool) else list(structured_output)
        )
        inputs, attention, labels, lengths = self._build_sequences(
            region_state, instructions, target_texts, structured_outputs
        )
        with self.controller.adapter_scope(DESCRIPTION_ADAPTER_NAME):
            outputs = self.controller.model(
                inputs_embeds=inputs,
                attention_mask=attention,
                return_dict=True,
                use_cache=False,
            )
        per_sample_loss = None
        loss = None
        if labels is not None:
            shift_logits = outputs.logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
            )
            token_valid = shift_labels != -100
            per_sample_loss = (token_loss * token_valid).sum(1) / token_valid.sum(1).clamp_min(1)
            loss = per_sample_loss.mean()
        return DescriptionForwardOutput(
            loss=loss,
            per_sample_loss=per_sample_loss,
            logits=outputs.logits,
            labels=labels,
            region_state=region_state,
            sequence_lengths=lengths,
        )

    def region_alignment_loss(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        phrases: Sequence[str],
        *,
        region_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric region-text contrastive loss for DIOR annotated candidate regions."""
        region, text = self.region_alignment_embeddings(
            backbone, region_masks, phrases, region_valid_mask=region_valid_mask
        )
        temperature = self.alignment_temperature.float().clamp(0.01, 1.0)
        logits = region @ text.T / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        )
        return loss, logits

    def region_alignment_embeddings(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        phrases: Sequence[str],
        *,
        region_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized MGRR region and Qwen phrase embeddings."""
        region_state = self.build_region_state(
            backbone, region_masks, region_valid_mask=region_valid_mask, protocol="vision_only"
        )
        regions = region_state.region_tokens
        if regions is None or regions.shape[1] != 1 or len(phrases) != regions.shape[0]:
            raise ValueError("DIOR alignment requires one region and one phrase per sample")
        embedding = self.controller.model.get_input_embeddings()
        sequences = []
        for phrase in phrases:
            ids = self._token_ids(phrase, append_eos=False)
            sequences.append(embedding(ids) + self.instruction_type)
        inputs = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([value.shape[0] for value in sequences], device=inputs.device)
        attention = torch.arange(inputs.shape[1], device=inputs.device)[None] < lengths[:, None]
        with self.controller.adapter_scope(DESCRIPTION_ADAPTER_NAME):
            outputs = self.controller.model(
                inputs_embeds=inputs,
                attention_mask=attention,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=1,
            )
        hidden = outputs.hidden_states[-1]
        text_hidden = torch.stack([
            hidden[index, int(length) - 1]
            for index, length in enumerate(lengths.tolist())
        ])
        text = self.alignment_text_projection(text_hidden.float())
        region = regions[:, 0]
        text = F.normalize(text.float(), dim=-1)
        region = F.normalize(region.float(), dim=-1)
        return region, text

    def segment_then_describe(
        self,
        batch: ModalityBatch,
        instructions: Sequence[str],
        *,
        target_texts: Sequence[str] | None = None,
        threshold: float = 0.5,
        protocol: RegionProtocol = "vision_only",
        structured_output: bool | Sequence[bool] = True,
    ) -> SegmentThenDescribeOutput:
        backbone = self.segmentation.encode_multisource(
            batch, use_full=False, include_visual_tokens=True
        )
        segmentation_state = self.segmentation.build_segmentation_state(
            batch, use_full=False, backbone=backbone
        )
        segmentation = self.segmentation.segment_from_state(segmentation_state)
        predicted = (torch.sigmoid(segmentation.final_mask_logits) >= float(threshold)).float()
        description = self.describe_from_state(
            backbone,
            predicted,
            instructions,
            target_texts=target_texts,
            region_valid_mask=batch.valid_mask,
            protocol=protocol,
            structured_output=structured_output,
        )
        return SegmentThenDescribeOutput(segmentation=segmentation, description=description)

    @torch.no_grad()
    def generate_from_state(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        instruction: str,
        *,
        max_new_tokens: int = 256,
        protocol: RegionProtocol = "vision_only",
        structured_output: bool = True,
    ) -> str:
        if region_masks.shape[0] != 1:
            raise ValueError("Description v1 autoregressive generation currently requires batch_size=1")
        region_state = self.build_region_state(backbone, region_masks, protocol=protocol)
        inputs, attention, _labels, _lengths = self._build_sequences(
            region_state, [instruction], None, [bool(structured_output)]
        )
        generated: list[int] = []
        embedding = self.controller.model.get_input_embeddings()
        eos = self.controller.tokenizer.eos_token_id
        with self.controller.adapter_scope(DESCRIPTION_ADAPTER_NAME):
            for _ in range(int(max_new_tokens)):
                output = self.controller.model(
                    inputs_embeds=inputs,
                    attention_mask=attention,
                    return_dict=True,
                    use_cache=False,
                )
                token = int(output.logits[0, -1].float().argmax().item())
                if eos is not None and token == int(eos):
                    break
                generated.append(token)
                token_tensor = torch.tensor([[token]], device=inputs.device)
                inputs = torch.cat([inputs, embedding(token_tensor)], 1)
                attention = torch.cat([
                    attention, torch.ones((1, 1), dtype=torch.bool, device=attention.device)
                ], 1)
        return self.controller.tokenizer.decode(generated, skip_special_tokens=True)
