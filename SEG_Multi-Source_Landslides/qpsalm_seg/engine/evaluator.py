#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation/test evaluator for segmentation, proposal sets and instruction sensitivity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
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
SEGMENTATION_EVAL_MANIFEST_PROTOCOL = (
    "qpsalm_segmentation_eval_manifest_v3_replay_config_bound"
)
SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL = (
    "qpsalm_segmentation_eval_report_binding_v2_replay_config"
)
SEGMENTATION_PREDICTION_POPULATION_PROTOCOL = (
    "qpsalm_segmentation_prediction_population_v1_binary_sha256"
)
SEGMENTATION_PREDICTION_FIELDS = (
    "sample_id",
    "parent_sample_id",
    "shape",
    "prediction_sha256",
    "target_sha256",
    "valid_sha256",
)
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


def segmentation_prediction_population(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Bind every thresholded prediction, target and valid mask by sample."""

    normalized = [
        {field: row.get(field) for field in SEGMENTATION_PREDICTION_FIELDS}
        for row in rows
    ]
    normalized.sort(key=lambda row: str(row.get("sample_id") or ""))
    sample_ids = [str(row.get("sample_id") or "") for row in normalized]
    parent_ids = [str(row.get("parent_sample_id") or "") for row in normalized]
    canonical = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in normalized
    ]
    return {
        "protocol": SEGMENTATION_PREDICTION_POPULATION_PROTOCOL,
        "fields": list(SEGMENTATION_PREDICTION_FIELDS),
        "threshold": float(threshold),
        "num_records": len(normalized),
        "num_unique_sample_ids": len(set(sample_ids)),
        "complete": all(sample_ids) and all(parent_ids),
        "unique": len(sample_ids) == len(set(sample_ids)),
        "sha256": hashlib.sha256(
            "\n".join(canonical).encode("utf-8")
        ).hexdigest(),
        "rows": normalized,
    }


def validate_segmentation_prediction_population(value: Any) -> dict[str, Any]:
    """Recompute a serialized prediction population instead of trusting its summary."""

    if not isinstance(value, dict):
        raise ValueError("segmentation prediction population 必须是 object")
    if value.get("protocol") != SEGMENTATION_PREDICTION_POPULATION_PROTOCOL:
        raise ValueError("segmentation prediction population protocol 不兼容")
    if tuple(value.get("fields") or ()) != SEGMENTATION_PREDICTION_FIELDS:
        raise ValueError("segmentation prediction population fields 不兼容")
    if value.get("complete") is not True or value.get("unique") is not True:
        raise ValueError("segmentation prediction population 必须 complete 且 unique")
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("segmentation prediction population rows 非法")
    for index, row in enumerate(rows):
        if set(row) != set(SEGMENTATION_PREDICTION_FIELDS):
            raise ValueError(f"segmentation prediction row={index} fields 非法")
        shape = row.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        ):
            raise ValueError(f"segmentation prediction row={index} shape 非法")
        for field in ("prediction_sha256", "target_sha256", "valid_sha256"):
            digest = row.get(field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"segmentation prediction row={index} {field} 非法")
    threshold = float(value.get("threshold"))
    if not math.isfinite(threshold):
        raise ValueError("segmentation prediction population threshold 必须有限")
    rebuilt = segmentation_prediction_population(
        rows,
        threshold=threshold,
    )
    if rebuilt != value:
        raise ValueError("segmentation prediction population 汇总与逐行重算不一致")
    return rebuilt


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
    prediction_records: list[dict[str, Any]] = []
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
        probabilities = torch.sigmoid(logits.float())
        for sample_index, row in enumerate(metadata):
            valid_binary = valid[sample_index, 0] >= 0.5
            prediction_binary = (
                probabilities[sample_index, 0] >= float(threshold)
            ) & valid_binary
            target_binary = (target[sample_index, 0] >= 0.5) & valid_binary
            mask_shape = list(prediction_binary.shape)
            prediction_records.append({
                "sample_id": str(row.get("sample_id") or ""),
                "parent_sample_id": str(row.get("parent_sample_id") or ""),
                "shape": mask_shape,
                "prediction_sha256": hashlib.sha256(
                    prediction_binary.to(torch.uint8).contiguous().numpy().tobytes()
                ).hexdigest(),
                "target_sha256": hashlib.sha256(
                    target_binary.to(torch.uint8).contiguous().numpy().tobytes()
                ).hexdigest(),
                "valid_sha256": hashlib.sha256(
                    valid_binary.to(torch.uint8).contiguous().numpy().tobytes()
                ).hexdigest(),
            })
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
        "prediction_population": segmentation_prediction_population(
            prediction_records,
            threshold=threshold,
        ),
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
