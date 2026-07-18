"""Auxiliary generated-description to segmentation cycle-localization metric."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
from typing import Any, Iterable

import numpy as np
import torch

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.schema import ModalityBatch
from qpsalm_seg.visualize import restore_mask_to_original

from ..protocols.region_geometry import project_native_region_mask_to_cache
from ..data.datasets import (
    bridge_region_metadata,
    description_row_sample_id,
    end_to_end_region_support,
)
from .metrics import bootstrap_mean_ci, finite_mean
from .targets import EndToEndTargetResolver
from ..data.source_binding import build_segmentation_instruction_source_binding


CYCLE_LOCALIZATION_PROTOCOL = "qpsalm_cycle_localization_v1_raw_text_grounding"
CYCLE_PROMPT_PROTOCOL = "qpsalm_cycle_localization_prompt_v1"


def cycle_prompt_batch(batch: ModalityBatch, generated_texts: list[str]) -> ModalityBatch:
    """Replace semantic prompts with raw model text without mutating sensor inputs."""
    if len(generated_texts) != batch.batch_size:
        raise ValueError(
            "cycle prompt 数量与 batch 不一致: "
            f"texts={len(generated_texts)} batch={batch.batch_size}"
        )
    normalized = [str(value).strip() for value in generated_texts]
    if any(not value for value in normalized):
        raise ValueError("cycle localization 不接受空 generated text")
    proposal = [
        "Instruction: Segment the landslide region described by the following "
        f"raw generated text. Generated description: {value}"
        for value in normalized
    ]
    condition = [
        "Condition: the landslide region identified by this raw generated "
        f"description: {value}"
        for value in normalized
    ]
    reasoning = [
        f"{original} Ground only the generated description against the active "
        "sensor evidence; output an empty mask when the description has no "
        "supported target."
        for original in batch.evidence_reasoning_text
    ]
    full_reasoning = [
        f"{original} Ground only the generated description against the active "
        "sensor evidence; output an empty mask when the description has no "
        "supported target."
        for original in batch.full_evidence_reasoning_text
    ]
    return replace(
        batch,
        proposal_context_text=proposal,
        condition_prompt_text=condition,
        evidence_reasoning_text=reasoning,
        full_proposal_context_text=list(proposal),
        full_condition_prompt_text=list(condition),
        full_evidence_reasoning_text=full_reasoning,
    )


def cycle_region_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute binary cycle IoU with explicit empty-target behavior."""
    predicted = prediction.detach().bool()
    expected = target.detach().bool()
    if predicted.shape != expected.shape:
        raise ValueError(
            f"cycle prediction/target shape 不一致: {predicted.shape} vs {expected.shape}"
        )
    if valid_mask is None:
        valid = torch.ones_like(expected, dtype=torch.bool)
    else:
        valid = valid_mask.detach().bool()
        if valid.shape != expected.shape:
            raise ValueError(
                f"cycle valid-mask shape 不一致: {valid.shape} vs {expected.shape}"
            )
    predicted &= valid
    expected &= valid
    intersection = int((predicted & expected).sum().item())
    union = int((predicted | expected).sum().item())
    target_pixels = int(expected.sum().item())
    predicted_pixels = int(predicted.sum().item())
    return {
        "region_iou": intersection / union if union else 1.0,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "target_pixels": target_pixels,
        "predicted_pixels": predicted_pixels,
        "target_empty": target_pixels == 0,
        "prediction_empty": predicted_pixels == 0,
        "empty_target_correct": target_pixels == 0 and predicted_pixels == 0,
    }


class CycleLocalizationProvider:
    """Run the frozen segmentation adapter with raw generated text as its prompt."""

    def __init__(self, model: Any, split: str, threshold: float) -> None:
        config = replace(
            model.segmentation.config,
            modality_dropout=0.0,
            train_hflip_prob=0.0,
            train_vflip_prob=0.0,
        )
        self.dataset = MultiSourceLandslideDataset(config, split)
        self.segmentation_source_binding = (
            build_segmentation_instruction_source_binding(
                config, split, self.dataset.rows
            )
        )
        self.resolver = EndToEndTargetResolver(self.dataset.rows)
        self.model = model
        self.threshold = float(threshold)
        self.source_rows = 0
        self.eligible_rows = 0
        self.exclusion_counts: Counter[str] = Counter()
        self.runtime_skip_counts: Counter[str] = Counter()
        self._resolved_by_sample: dict[str, dict[str, Any]] = {}

    def prepare(self, rows: Iterable[dict[str, Any]]) -> None:
        """Freeze the exact cycle-eligible Bridge population before generation."""
        self.source_rows = 0
        self.eligible_rows = 0
        self.exclusion_counts.clear()
        self.runtime_skip_counts.clear()
        self._resolved_by_sample.clear()
        for row in rows:
            self.source_rows += 1
            sample_id = description_row_sample_id(row)
            supported, reason = end_to_end_region_support(row)
            if not supported:
                self.exclusion_counts[str(reason)] += 1
                continue
            metadata = bridge_region_metadata(row)
            audit = self.resolver.resolve(metadata)
            if sample_id in self._resolved_by_sample:
                raise ValueError(f"cycle localization sample_id 重复: {sample_id}")
            self._resolved_by_sample[sample_id] = audit
            self.eligible_rows += 1

    def eligible(self, sample_id: str) -> bool:
        return str(sample_id) in self._resolved_by_sample

    @torch.no_grad()
    def localize(
        self,
        metadata: dict[str, Any],
        raw_generated_text: str,
        output_hw: tuple[int, int],
        *,
        return_source: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | tuple[
        torch.Tensor, dict[str, Any], torch.Tensor
    ]:
        sample_id = str(metadata.get("sample_id") or "")
        audit = self._resolved_by_sample.get(sample_id)
        if audit is None:
            raise KeyError(f"sample 不在预冻结 cycle population: {sample_id}")
        item = self.dataset[int(audit["dataset_index"])]
        batch = cycle_prompt_batch(qpsalm_collate([item]), [raw_generated_text])
        with self.model.controller.adapter_scope("default"):
            output = self.model.segmentation(batch)
        canvas = (
            torch.sigmoid(output.final_mask_logits[0, 0].float()).detach().cpu().numpy()
            >= self.threshold
        ).astype(np.uint8)
        segmentation_transform = dict(item["metadata"].get("resize_transform") or {})
        restored = restore_mask_to_original(canvas, segmentation_transform)
        if restored is None:
            raise ValueError(f"cycle mask 无法恢复原尺寸: sample={sample_id}")
        original_mask = torch.from_numpy(restored.astype(np.float32))[None]
        parent = str(audit["parent_sample_id"])
        cache_record = self.model.description_backbone.bank.record(
            "multisource_parent", parent
        )
        description_transform = dict(cache_record["views"][0]["render_transform"])
        description_mask, source_mapping = project_native_region_mask_to_cache(
            original_mask, description_transform
        )
        if tuple(description_mask.shape[-2:]) != tuple(output_hw):
            raise ValueError(
                "cycle description mask canvas 不一致: "
                f"parent={parent} observed={tuple(description_mask.shape[-2:])} "
                f"expected={tuple(output_hw)}"
            )
        device = next(self.model.parameters()).device
        audit = {
            "protocol": CYCLE_PROMPT_PROTOCOL,
            "target_mapping": audit,
            "mask_threshold": self.threshold,
            "generated_text_sha256": hashlib.sha256(
                str(raw_generated_text).encode("utf-8")
            ).hexdigest(),
            "generated_text_characters": len(str(raw_generated_text)),
            "segmentation_resize_transform": segmentation_transform,
            "description_render_transform": description_transform,
            "source_to_render_mapping": source_mapping,
        }
        if return_source:
            return (
                description_mask.to(device=device),
                audit,
                original_mask.clone(),
            )
        return description_mask.to(device=device), audit


def summarize_cycle_localization(
    rows: list[dict[str, Any]],
    provider: CycleLocalizationProvider,
    *,
    requested: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_sample_id"])].append(float(row["region_iou"]))
    parent_values = [
        sum(values) / len(values) for _parent, values in sorted(grouped.items())
    ]
    present = [float(row["region_iou"]) for row in rows if not row["target_empty"]]
    absent = [row for row in rows if row["target_empty"]]
    target = provider.eligible_rows if requested == 0 else min(
        int(requested), provider.eligible_rows
    )
    return {
        "protocol": CYCLE_LOCALIZATION_PROTOCOL,
        "role": "auxiliary_self_consistency_only",
        "primary_evidence_replaced": False,
        "input_text": "raw_unrepaired_generation",
        "source_bridge_rows": provider.source_rows,
        "eligible_bridge_rows": provider.eligible_rows,
        "requested": int(requested),
        "target_evaluations": target,
        "evaluated_samples": len(rows),
        "evaluated_parents": len(parent_values),
        "coverage_complete": len(rows) == target,
        "excluded_by_reason": dict(sorted(provider.exclusion_counts.items())),
        "runtime_skipped_by_reason": dict(
            sorted(provider.runtime_skip_counts.items())
        ),
        "segmentation_source_binding": provider.segmentation_source_binding,
        "parent_macro_region_iou": finite_mean(parent_values),
        "parent_bootstrap_region_iou_ci": bootstrap_mean_ci(
            parent_values, seed=int(seed), samples=10000
        ),
        "present_region_iou": finite_mean(present),
        "empty_target_accuracy": (
            sum(int(row["empty_target_correct"]) for row in absent) / len(absent)
            if absent else None
        ),
        "num_present": len(present),
        "num_empty_target": len(absent),
    }
