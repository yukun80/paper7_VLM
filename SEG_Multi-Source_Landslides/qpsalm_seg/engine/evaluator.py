#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation/test evaluator for segmentation, proposal sets and instruction sensitivity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from qpsalm_seg.metrics import MetricAccumulator, batch_binary_metrics
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.visualize import save_visualizations

from .diagnostics import (
    average_dicts,
    collect_proposal_records,
    collect_query_attention_records,
    collect_reliability_records,
    compute_proposal_summary,
    compute_query_attention_summary,
    compute_reliability_summary,
    loss_log_values,
    metric_metadata_with_scale,
    paired_instruction_summary,
)
from .common import amp_dtype, autocast_enabled
from .threshold import (
    canvas_original_metric_delta,
    compute_threshold_sweep_report,
    normalize_thresholds,
    restored_original_space_metrics,
)


SAMPLE_IDENTITY_PROTOCOL = "qpsalm_segmentation_eval_population_v1"
SAMPLE_IDENTITY_FIELDS = (
    "sample_id",
    "parent_sample_id",
    "dataset_name",
    "template_id",
    "task_family",
    "instruction",
    "referring_category",
    "target_mask_path",
    "active_subset",
    "original_size",
    "mask_original_size",
    "target_size",
    "resize_transform",
    "prompt_version",
    "instruction_ablation",
)


def evaluation_population_identity(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an order-independent identity for the exact evaluated sample population."""
    canonical_rows: list[str] = []
    sample_ids: list[str] = []
    incomplete_indices: list[int] = []
    for index, row in enumerate(metadata):
        sample_id = str(row.get("sample_id") or "").strip()
        parent_id = str(row.get("parent_sample_id") or "").strip()
        if not sample_id or not parent_id:
            incomplete_indices.append(index)
        sample_ids.append(sample_id)
        payload = {field: row.get(field) for field in SAMPLE_IDENTITY_FIELDS}
        canonical_rows.append(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    duplicate_ids = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if sample_id and count > 1
    )
    digest = hashlib.sha256("\n".join(sorted(canonical_rows)).encode("utf-8")).hexdigest()
    return {
        "protocol": SAMPLE_IDENTITY_PROTOCOL,
        "fields": list(SAMPLE_IDENTITY_FIELDS),
        "sha256": digest,
        "num_records": len(canonical_rows),
        "num_unique_sample_ids": len({value for value in sample_ids if value}),
        "complete": not incomplete_indices,
        "unique": not duplicate_ids and len(sample_ids) == len(set(sample_ids)),
        "incomplete_record_indices": incomplete_indices,
        "duplicate_sample_ids": duplicate_ids,
    }


@torch.no_grad()
def evaluate(
    model: MultiSourceQwenPSALMSeg,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    visual_dir: Path | None = None,
    num_visualizations: int = 0,
    visualize_all: bool = False,
    export_multimodal_overview: bool = False,
    threshold: float = 0.5,
    threshold_sweep=None,
) -> dict[str, Any]:
    model.eval()
    canvas_acc, original_acc = MetricAccumulator(), MetricAccumulator()
    sweep = {value: MetricAccumulator() for value in normalize_thresholds(threshold_sweep)}
    losses, loss_components = [], []
    reliability_records, query_records, proposal_records, saved = [], [], [], []
    evaluated_metadata: list[dict[str, Any]] = []
    coverage = {
        "family_combos": Counter(), "raw_combos": Counter(), "sensor_combos": Counter(),
        "product_combos": Counter(), "target_area_px_bins": Counter(),
        "target_area_fraction_bins": Counter(), "ground_area_m2_bins": Counter(),
    }
    processed_batches = processed_samples = 0
    autocast = autocast_enabled(model.config, device)
    dtype = amp_dtype(model.config, device)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        processed_batches += 1
        processed_samples += batch.batch_size
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
            outputs = model(batch)
        if "loss" in outputs:
            losses.append(float(outputs["loss"].detach().cpu()))
            loss_components.append(loss_log_values(outputs))
        logits = outputs["final_mask_logits"].detach().cpu()
        target = batch.mask.detach().cpu()
        valid = batch.valid_mask.detach().cpu()
        metadata = metric_metadata_with_scale(batch.metadata, target, valid)
        evaluated_metadata.extend(dict(row) for row in metadata)
        for row in metadata:
            for field, source in (
                ("family_combos", "family_combo"), ("raw_combos", "raw_combo"),
                ("sensor_combos", "sensor_combo"), ("product_combos", "product_combo"),
                ("target_area_px_bins", "target_area_px_bin"),
                ("target_area_fraction_bins", "target_area_fraction_bin"),
                ("ground_area_m2_bins", "ground_area_m2_bin"),
            ):
                coverage[field][str(row.get(source, "unknown"))] += 1
        metrics = batch_binary_metrics(logits, target, threshold=threshold, valid_mask=valid)
        canvas_acc.update(metrics, metadata)
        original_acc.update(restored_original_space_metrics(logits, target, metadata, threshold, valid), metadata)
        for value, accumulator in sweep.items():
            accumulator.update(batch_binary_metrics(logits, target, threshold=value, valid_mask=valid), metadata)
        reliability_records.extend(collect_reliability_records(outputs, metadata))
        query_records.extend(collect_query_attention_records(outputs, metadata))
        proposal_records.extend(
            collect_proposal_records(outputs, target, valid, metadata, metrics, threshold=threshold)
        )
        if visual_dir is not None and (visualize_all or len(saved) < num_visualizations):
            remaining = batch.batch_size if visualize_all else max(0, num_visualizations - len(saved))
            saved.extend(save_visualizations(
                batch, outputs, visual_dir, remaining, f"eval_b{batch_index}", threshold,
                export_multimodal_overview,
            ))
        del outputs
        if device.type == "cuda" and batch_index % 50 == 49:
            torch.cuda.empty_cache()

    metrics, original_metrics = canvas_acc.compute(), original_acc.compute()
    visual_counts = {}
    if visual_dir is not None:
        for name, relative in (
            ("num_multimodal_overviews", "multimodal_overviews"),
            ("num_mask_export_pngs", "mask_exports"),
            ("num_restored_mask_export_pngs", "mask_exports_original_size"),
        ):
            directory = visual_dir / relative
            visual_counts[name] = len(list(directory.rglob("*.png"))) if directory.exists() else 0
    overview_dir = visual_dir / "multimodal_overviews" if visual_dir is not None else None
    mask_export_dir = visual_dir / "mask_exports" if visual_dir is not None else None
    restored_mask_export_dir = visual_dir / "mask_exports_original_size" if visual_dir is not None else None
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "loss_components": average_dicts(loss_components),
        "threshold": float(threshold),
        "coverage": {
            "num_batches": processed_batches, "num_samples": processed_samples,
            "sample_population": evaluation_population_identity(evaluated_metadata),
            **{key: dict(sorted(value.items())) for key, value in coverage.items()},
            "max_batches": max_batches,
        },
        "threshold_sweep": compute_threshold_sweep_report(sweep),
        "metrics": metrics,
        "metrics_original_size": original_metrics,
        "canvas_vs_original_delta": canvas_original_metric_delta(metrics, original_metrics),
        "modality_reliability_summary": compute_reliability_summary(reliability_records),
        "query_modality_attention_summary": compute_query_attention_summary(query_records),
        "proposal_diagnostics": {
            "records": proposal_records,
            "summary": compute_proposal_summary(proposal_records),
        },
        "instruction_sensitivity": paired_instruction_summary(proposal_records),
        "visualizations": saved,
        "visualization_export": {
            "visualize_all": bool(visualize_all),
            "export_multimodal_overview": bool(export_multimodal_overview),
            "num_diagnostic_pngs": len(saved),
            **visual_counts,
            "visualization_dir": str(visual_dir) if visual_dir else None,
            "visualization_manifest_path": str(visual_dir / "visualization_manifest.jsonl") if visual_dir else None,
            "multimodal_overview_dir": str(overview_dir) if overview_dir else None,
            "mask_export_dir": str(mask_export_dir) if mask_export_dir else None,
            "restored_mask_export_dir": str(restored_mask_export_dir) if restored_mask_export_dir else None,
        },
    }
