#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark 样本的共享单样本推理服务。

用途：为 Gradio demo 和 PPT 图库复用同一套 checkpoint、模态子集与 prompt 覆盖逻辑。
运行方式：不作为独立入口；由 ``qpsalm-demo`` 或 ``qpsalm-curate-gallery`` 调用。
输入：benchmark-v2 的 val/test 样本、Qwen vision cache v3 和 v5 checkpoint。
输出：模型预测、原尺寸 mask、指标与可视化所需诊断信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.data.dataset import subset_signature
from qpsalm_seg.data.prompts import build_prompt_triplet, condition_text, instruction_text
from qpsalm_seg.data.transforms import resize_pad_tensor, valid_mask_from_transform
from qpsalm_seg.engine.checkpoint import load_checkpoint
from qpsalm_seg.engine.common import amp_dtype, autocast_enabled, build_model, resolve_device
from qpsalm_seg.indexing import family_combo, raw_modality_combo
from qpsalm_seg.matching import component_masks
from qpsalm_seg.metrics import batch_binary_metrics
from qpsalm_seg.schema import ActiveModalitySubset, ModalityBatch
from qpsalm_seg.visualize import restore_mask_to_original


@dataclass(frozen=True)
class CatalogEntry:
    sample_id: str
    parent_sample_id: str
    dataset_name: str
    task_family: str
    family_combo: str
    raw_combo: str
    instruction: str
    modality_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "parent_sample_id": self.parent_sample_id,
            "dataset_name": self.dataset_name,
            "task_family": self.task_family,
            "family_combo": self.family_combo,
            "raw_combo": self.raw_combo,
            "instruction": self.instruction,
            "modality_names": list(self.modality_names),
        }


@dataclass
class PredictionResult:
    sample_id: str
    checkpoint_step: int
    batch: ModalityBatch
    probability: np.ndarray
    final_mask: np.ndarray
    selected_proposal: np.ndarray
    selected_query: int
    ground_truth: np.ndarray
    valid_mask: np.ndarray
    restored_final_mask: np.ndarray | None
    metrics: dict[str, float]
    metrics_are_reference_only: bool
    latency_seconds: float
    diagnostics: dict[str, Any]


def _active_evidence_valid(item: dict[str, Any], active_names: set[str]) -> torch.Tensor:
    target_size = int(item["metadata"]["target_size"])
    transform = item["metadata"].get("resize_transform")
    canvas_valid = valid_mask_from_transform(transform)
    evidence_valid = torch.zeros_like(canvas_valid)
    for instance in item["full_instances"]:
        if instance.name not in active_names:
            continue
        aligned, _ = resize_pad_tensor(instance.valid_mask, target_size, "nearest")
        evidence_valid = torch.maximum(evidence_valid, (aligned >= 0.5).to(evidence_valid.dtype))
    return canvas_valid * evidence_valid


def override_inference_item(
    item: dict[str, Any],
    row: dict[str, Any],
    config: QPSalmConfig,
    *,
    active_modalities: Iterable[str] | None = None,
    instruction_override: str | None = None,
    condition_override: str | None = None,
) -> dict[str, Any]:
    """返回新的推理 item；不会改写 Dataset row 或原 item。"""
    full_instances = list(item["full_instances"])
    available = tuple(instance.name for instance in full_instances)
    modality_source = available if active_modalities is None else active_modalities
    requested = tuple(dict.fromkeys(str(value) for value in modality_source))
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"样本不存在所选模态: {unknown}; available={list(available)}")
    if not requested:
        raise ValueError("推理至少需要一个活动模态")
    active_set = set(requested)
    active_instances = [instance for instance in full_instances if instance.name in active_set]
    ordered_active = tuple(sorted(instance.name for instance in active_instances))
    dropped = tuple(sorted(set(available) - set(ordered_active)))
    subset = ActiveModalitySubset(
        active_names=ordered_active,
        dropped_names=dropped,
        signature=subset_signature(ordered_active),
        is_full=not dropped,
    )
    valid_mask = _active_evidence_valid(item, active_set)
    if not bool((valid_mask >= 0.5).any()):
        raise ValueError(f"所选模态没有有效像素: {list(ordered_active)}")

    original_instruction = instruction_text(row)
    original_condition = condition_text(row)
    task = str(instruction_override).strip() if instruction_override else original_instruction
    condition = str(condition_override).strip() if condition_override else (
        task if instruction_override else original_condition
    )
    is_custom = task != original_instruction or condition != original_condition
    proposal, condition_prompt, reasoning = build_prompt_triplet(
        row,
        active_instances,
        subset_signature=subset.signature,
        ablation="normal",
        instruction_override=task,
        condition_override=condition,
    )
    full_signature = subset_signature([instance.name for instance in full_instances])
    full_proposal, full_condition, full_reasoning = build_prompt_triplet(
        row,
        full_instances,
        subset_signature=full_signature,
        ablation="normal",
        instruction_override=task,
        condition_override=condition,
    )
    metadata = {
        **item["metadata"],
        "instruction": task,
        "condition": condition,
        "instruction_is_custom": is_custom,
        "gt_is_reference_only": is_custom,
        "active_subset": subset.signature,
        "active_modalities": list(ordered_active),
        "raw_combo": "+".join(ordered_active),
        "family_combo": "+".join(sorted({instance.family for instance in active_instances})),
        "sensor_combo": "+".join(sorted({instance.sensor for instance in active_instances})),
        "product_combo": "+".join(sorted({instance.product_type for instance in active_instances})),
        "valid_coverage": float(valid_mask.mean().item()),
    }
    components = component_masks(
        item["mask"][0],
        valid_mask[0],
        float(config.min_component_area_fraction),
        int(config.min_component_area_pixels),
    )
    return {
        **item,
        "instances": active_instances,
        "active_subset": subset,
        "valid_mask": valid_mask,
        "metadata": metadata,
        "proposal_context_text": proposal,
        "condition_prompt_text": condition_prompt,
        "evidence_reasoning_text": reasoning,
        "full_proposal_context_text": full_proposal,
        "full_condition_prompt_text": full_condition,
        "full_evidence_reasoning_text": full_reasoning,
        "component_masks": components,
    }


class InferenceSession:
    """一次加载模型，并对 benchmark 中的多个样本重复推理。"""

    def __init__(
        self,
        config: QPSalmConfig,
        *,
        split: str,
        checkpoint: str | Path | None,
        device: str | torch.device,
    ) -> None:
        if split not in {"val", "test"}:
            raise ValueError("交互推理只支持 val/test split")
        self.config = config
        self.split = split
        self.device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
        self.dataset = MultiSourceLandslideDataset(config, split)
        self._sample_to_index: dict[str, int] = {}
        self._catalog: list[CatalogEntry] = []
        for index, row in enumerate(self.dataset.rows):
            sample_id = str(row.get("sample_id"))
            if sample_id in self._sample_to_index:
                raise ValueError(f"split 中 sample_id 重复: {sample_id}")
            self._sample_to_index[sample_id] = index
            modality_names = tuple(sorted(
                str(name) for name, value in (row.get("modalities") or {}).items()
                if isinstance(value, dict) and value.get("available", True)
            ))
            self._catalog.append(CatalogEntry(
                sample_id=sample_id,
                parent_sample_id=str(row.get("parent_sample_id") or sample_id),
                dataset_name=str(row.get("dataset_name") or "unknown"),
                task_family=str(row.get("task_family") or "unknown"),
                family_combo=family_combo(row),
                raw_combo=raw_modality_combo(row),
                instruction=instruction_text(row),
                modality_names=modality_names,
            ))
        self.model = build_model(config, self.device)
        self.checkpoint_step = load_checkpoint(checkpoint, self.model) if checkpoint else 0
        self.model.eval()

    @property
    def catalog(self) -> tuple[CatalogEntry, ...]:
        return tuple(self._catalog)

    def filter_catalog(
        self,
        *,
        dataset_name: str | None = None,
        family_combo_name: str | None = None,
        task_family: str | None = None,
        query: str | None = None,
    ) -> list[CatalogEntry]:
        needle = str(query or "").strip().lower()
        return [
            entry for entry in self._catalog
            if (not dataset_name or dataset_name == "all" or entry.dataset_name == dataset_name)
            and (not family_combo_name or family_combo_name == "all" or entry.family_combo == family_combo_name)
            and (not task_family or task_family == "all" or entry.task_family == task_family)
            and (not needle or needle in entry.sample_id.lower() or needle in entry.instruction.lower())
        ]

    def sample_defaults(self, sample_id: str) -> dict[str, Any]:
        entry = self._catalog[self._sample_to_index[sample_id]]
        row = self.dataset.rows[self._sample_to_index[sample_id]]
        return {
            **entry.as_dict(),
            "condition": condition_text(row),
        }

    @torch.no_grad()
    def predict(
        self,
        sample_id: str,
        *,
        instruction: str | None = None,
        condition: str | None = None,
        active_modalities: Iterable[str] | None = None,
        threshold: float = 0.5,
    ) -> PredictionResult:
        if sample_id not in self._sample_to_index:
            raise KeyError(f"split={self.split} 中不存在 sample_id={sample_id!r}")
        index = self._sample_to_index[sample_id]
        row = self.dataset.rows[index]
        item = override_inference_item(
            self.dataset[index],
            row,
            self.config,
            active_modalities=active_modalities,
            instruction_override=instruction,
            condition_override=condition,
        )
        batch = qpsalm_collate([item])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=amp_dtype(self.config, self.device),
            enabled=autocast_enabled(self.config, self.device),
        ):
            outputs = self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started

        probability = torch.sigmoid(outputs["final_mask_logits"])[0, 0].detach().float().cpu().numpy()
        valid = (batch.valid_mask[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
        final = (probability >= float(threshold)).astype(np.uint8) * valid
        relevance = outputs["proposal_relevance_logits"][0].detach().float().cpu()
        selected_query = int(torch.argmax(relevance).item())
        proposal_probability = torch.sigmoid(outputs["proposal_mask_logits"][0, selected_query]).detach().float().cpu().numpy()
        selected = (proposal_probability >= float(threshold)).astype(np.uint8) * valid
        ground_truth = (batch.mask[0, 0].cpu().numpy() >= 0.5).astype(np.uint8) * valid
        metric = batch_binary_metrics(
            outputs["final_mask_logits"].detach().cpu(),
            batch.mask.cpu(),
            threshold=float(threshold),
            valid_mask=batch.valid_mask.cpu(),
        )[0]
        metadata = batch.metadata[0]
        modality_names = [instance.name for instance in batch.instances[0]]
        reliability = outputs.get("modality_reliability_weights")
        selected_attention = outputs.get("query_modality_attention")
        diagnostics = {
            "instruction": metadata.get("instruction"),
            "condition": metadata.get("condition"),
            "active_modalities": modality_names,
            "selected_query": selected_query,
            "selected_relevance_logit": float(relevance[selected_query].item()),
            "mask_area": int(final.sum()),
            "valid_pixels": int(valid.sum()),
            "null_evidence_weight": _scalar_at(outputs.get("null_evidence_weight"), 0),
            "real_evidence_mass": _scalar_at(outputs.get("real_evidence_mass"), 0),
            "modality_reliability": _named_row(reliability, modality_names, 0),
            "selected_query_modality_attention": _named_query_row(
                selected_attention, modality_names, 0, selected_query
            ),
        }
        return PredictionResult(
            sample_id=sample_id,
            checkpoint_step=self.checkpoint_step,
            batch=batch,
            probability=probability,
            final_mask=final,
            selected_proposal=selected,
            selected_query=selected_query,
            ground_truth=ground_truth,
            valid_mask=valid,
            restored_final_mask=restore_mask_to_original(final, metadata.get("resize_transform")),
            metrics=metric,
            metrics_are_reference_only=bool(metadata.get("gt_is_reference_only")),
            latency_seconds=latency,
            diagnostics=diagnostics,
        )


def _scalar_at(value: Any, index: int) -> float | None:
    return float(value.detach().float().cpu()[index].item()) if torch.is_tensor(value) else None


def _named_row(value: Any, names: list[str], index: int) -> dict[str, float] | None:
    if not torch.is_tensor(value):
        return None
    row = value.detach().float().cpu()[index]
    return {name: float(row[offset].item()) for offset, name in enumerate(names) if offset < row.numel()}


def _named_query_row(
    value: Any,
    names: list[str],
    sample_index: int,
    query_index: int,
) -> dict[str, float] | None:
    if not torch.is_tensor(value):
        return None
    row = value.detach().float().cpu()[sample_index, query_index]
    return {name: float(row[offset].item()) for offset, name in enumerate(names) if offset < row.numel()}
