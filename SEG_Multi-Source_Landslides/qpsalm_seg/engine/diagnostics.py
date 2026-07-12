#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training windows and SANE/QMEF/PMRD evaluation diagnostics."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


LOSS_KEYS = (
    "loss_mask_bce", "loss_mask_dice", "loss_boundary", "loss_proposal_set",
    "loss_proposal_coarse", "loss_proposal_coverage", "loss_semantic_verifier",
    "loss_proposal_coverage_coarse", "loss_missing_modality_consistency",
)
ASSIGNMENT_KEYS = (
    "proposal_matched_mean_dice", "proposal_component_recall", "proposal_component_precision",
    "proposal_unmatched_rejection", "proposal_relevance_ap", "proposal_relevance_auc",
    "proposal_union_dice", "proposal_merge_error_rate", "proposal_duplicate_error_rate",
    "proposal_missed_component_rate",
)


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def loss_log_values(outputs) -> dict[str, float]:
    values = {key: scalar(outputs[key]) for key in (*LOSS_KEYS, *ASSIGNMENT_KEYS) if key in outputs}
    for source, target in (
        ("proposal_target_positive_count", "proposal_target_positive_count"),
        ("proposal_component_count", "proposal_component_count"),
        ("proposal_matching_coverage_mode", "proposal_matching_coverage_fraction"),
        ("proposal_verifier_pos_weight", "proposal_verifier_pos_weight"),
    ):
        if source in outputs:
            values[target] = scalar(outputs[source])
    return values


def training_signal_values(outputs) -> dict[str, float]:
    values: dict[str, float] = {}
    if "modality_reliability_weights" in outputs:
        distribution = outputs["modality_reliability_weights"].detach().float().cpu()
        if "null_evidence_weight" in outputs:
            distribution = torch.cat([distribution, outputs["null_evidence_weight"].detach().float().cpu()[:, None]], 1)
        safe = distribution.clamp_min(1.0e-8)
        values["modality_reliability_entropy"] = float((-(safe * safe.log()).sum(1)).mean())
        values["modality_reliability_peak"] = float(safe.max(1).values.mean())
    for key in ("null_evidence_weight", "real_evidence_mass", "visual_evidence_delta_norm"):
        if key in outputs:
            values[key] = scalar(outputs[key])
    if "modality_active" in outputs:
        values["active_modality_count"] = scalar(outputs["modality_active"].float().sum(1))
    if "query_modality_attention" in outputs:
        attention = outputs["query_modality_attention"].detach().float().cpu().clamp_min(1.0e-8)
        values["query_modality_attention_entropy"] = float((-(attention * attention.log()).sum(-1)).mean())
        values["query_modality_attention_peak"] = float(attention.max(-1).values.mean())
    if "query_scale_attention" in outputs:
        attention = outputs["query_scale_attention"].detach().float().cpu().clamp_min(1.0e-8)
        values["query_scale_attention_entropy"] = float((-(attention * attention.log()).sum(-1)).mean())
        values["query_scale_attention_peak"] = float(attention.max(-1).values.mean())
    if "proposal_relevance_logits" in outputs:
        scores = outputs["proposal_relevance_logits"].detach().float().cpu()
        values.update({
            "proposal_relevance_mean": float(scores.mean()),
            "proposal_relevance_max": float(scores.max()),
            "top_query_mean": float(torch.argmax(scores, 1).float().mean()),
        })
    if "proposal_relevance_gates" in outputs:
        gates = outputs["proposal_relevance_gates"].detach().float().cpu()
        values["proposal_relevance_gate_mean"] = float(gates.mean())
        values["proposal_relevance_gate_max"] = float(gates.max())
    return values


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows)
        for key in sorted({key for row in rows for key in row})
    } if rows else {}


TRAIN_LOG_KEYS = (
    "loss", "iou", "dice", "proposal_matched_mean_dice", "proposal_component_recall",
    "proposal_component_precision", "proposal_unmatched_rejection", "proposal_relevance_ap",
    "proposal_union_dice", "proposal_target_positive_count", "proposal_component_count",
    "proposal_verifier_pos_weight",
    "proposal_matching_coverage_fraction", "proposal_relevance_max", "proposal_relevance_gate_mean",
    "modality_reliability_entropy", "modality_reliability_peak", "query_modality_attention_entropy",
    "query_modality_attention_peak", "query_scale_attention_entropy", "query_scale_attention_peak",
    "active_modality_count", "null_evidence_weight", "real_evidence_mass",
    "visual_evidence_delta_norm",
)


def summarize_train_window(rows: list[dict[str, Any]], elapsed: float) -> dict[str, float]:
    summary = {
        key: sum(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))
        / sum(isinstance(row.get(key), (int, float)) for row in rows)
        for key in TRAIN_LOG_KEYS
        if any(isinstance(row.get(key), (int, float)) for row in rows)
    }
    summary["steps_per_sec"] = len(rows) / max(elapsed, 1.0e-6)
    if rows:
        summary["lr"] = float(rows[-1].get("lr", 0.0))
    return summary


def format_train_window(start: int, end: int, count: int, values: dict[str, float]) -> str:
    parts = [
        f"steps={start}-{end}", f"n={count}", f"loss={values.get('loss', 0):.4f}",
        f"iou={values.get('iou', 0):.4f}", f"dice={values.get('dice', 0):.4f}",
        f"matched_dice={values.get('proposal_matched_mean_dice', 0):.4f}",
        f"lr={values.get('lr', 0):.2e}", f"sps={values.get('steps_per_sec', 0):.2f}",
    ]
    labels = {
        "proposal_component_recall": "compR", "proposal_component_precision": "compP",
        "proposal_unmatched_rejection": "reject", "proposal_relevance_ap": "relAP",
        "proposal_union_dice": "unionD", "proposal_target_positive_count": "posQ",
        "proposal_component_count": "components", "proposal_matching_coverage_fraction": "coverage",
        "proposal_verifier_pos_weight": "posW",
        "top_query_mean": "topQ", "modality_reliability_entropy": "relH",
        "query_modality_attention_peak": "qAttnP", "active_modality_count": "activeM",
        "query_scale_attention_peak": "qScaleP",
        "null_evidence_weight": "null", "real_evidence_mass": "realM",
        "visual_evidence_delta_norm": "visDelta",
    }
    for key, label in labels.items():
        if key in values:
            parts.append(f"{label}={values[key]:.3f}")
    return "train " + " ".join(parts)


def _named(values: torch.Tensor, names: list[str]) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(names[:values.numel()])}


def _group_fields(record: dict[str, Any]) -> tuple[str, ...]:
    return (
        f"family_combo={record.get('family_combo', 'unknown')}",
        f"raw_combo={record.get('raw_combo', 'unknown')}",
        f"sensor_combo={record.get('sensor_combo', 'unknown')}",
        f"product_combo={record.get('product_combo', 'unknown')}",
        f"task_family={record.get('task_family', 'unknown')}",
        f"target_area_px_bin={record.get('target_area_px_bin', 'unknown')}",
        f"ground_area_m2_bin={record.get('ground_area_m2_bin', 'unknown')}",
    )


def collect_reliability_records(outputs, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if "modality_reliability_weights" not in outputs:
        return []
    weights = outputs["modality_reliability_weights"].detach().float().cpu()
    active = outputs.get("modality_active")
    active = active.detach().float().cpu() if active is not None else None
    coverage = outputs.get("modality_coverage_ratio")
    coverage = coverage.detach().float().cpu() if coverage is not None else None
    anchor_norm = outputs.get("modality_semantic_anchor_norm")
    anchor_norm = anchor_norm.detach().float().cpu() if anchor_norm is not None else None
    null = outputs.get("null_evidence_weight")
    null = null.detach().float().cpu() if null is not None else None
    real_mass = outputs.get("real_evidence_mass")
    real_mass = real_mass.detach().float().cpu() if real_mass is not None else None
    visual_delta = outputs.get("visual_evidence_delta_norm")
    visual_delta = visual_delta.detach().float().cpu() if visual_delta is not None else None
    records = []
    for index, meta in enumerate(metadata):
        names = [str(value) for value in meta.get("active_modalities") or []]
        row = {**meta, "weights": _named(weights[index], names)}
        if active is not None:
            row["active"] = _named(active[index], names)
        if coverage is not None:
            row["coverage"] = _named(coverage[index], names)
        if anchor_norm is not None:
            row["semantic_anchor_norm"] = _named(anchor_norm[index], names)
        if null is not None:
            row["null_evidence_weight"] = float(null[index])
        if real_mass is not None:
            row["real_evidence_mass"] = float(real_mass[index])
        if visual_delta is not None:
            row["visual_evidence_delta_norm"] = float(visual_delta[index])
        records.append(row)
    return records


def collect_query_attention_records(outputs, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if "query_modality_attention" not in outputs:
        return []
    weights = outputs["query_modality_attention"].detach().float().cpu()
    relevance = outputs["proposal_relevance_logits"].detach().float().cpu()
    records = []
    for index, meta in enumerate(metadata):
        names = [str(value) for value in meta.get("active_modalities") or []]
        selected = int(relevance[index].argmax())
        records.append({
            **meta,
            "mean_query_weights": _named(weights[index].mean(0), names),
            "selected_query_weights": _named(weights[index, selected], names),
            "entropy": scalar(outputs["query_spatial_entropy_mean"][index]) if "query_spatial_entropy_mean" in outputs else None,
            "peak": scalar(outputs["query_modality_attention_peak"][index]) if "query_modality_attention_peak" in outputs else None,
        })
    return records


def _aggregate_named(rows, field):
    names = sorted({name for row in rows for name in (row.get(field) or {})})
    return {name: sum(float((row.get(field) or {}).get(name, 0)) for row in rows) / len(rows) for name in names}


def _group_summary(records, *, query: bool):
    if not records:
        return {}
    groups = {"overall": records}
    for record in records:
        for group in _group_fields(record):
            groups.setdefault(group, []).append(record)
    result = {}
    for name, rows in sorted(groups.items()):
        if query:
            result[name] = {
                "n": len(rows), "mean_query_weights": _aggregate_named(rows, "mean_query_weights"),
                "mean_selected_query_weights": _aggregate_named(rows, "selected_query_weights"),
                "mean_entropy": sum(float(row["entropy"]) for row in rows if row.get("entropy") is not None)
                / max(1, sum(row.get("entropy") is not None for row in rows)),
                "mean_peak": sum(float(row["peak"]) for row in rows if row.get("peak") is not None)
                / max(1, sum(row.get("peak") is not None for row in rows)),
            }
        else:
            result[name] = {
                "n": len(rows), "mean_weights": _aggregate_named(rows, "weights"),
                "mean_active": _aggregate_named(rows, "active"),
                "mean_coverage": _aggregate_named(rows, "coverage"),
                "mean_semantic_anchor_norm": _aggregate_named(rows, "semantic_anchor_norm"),
                "mean_null_evidence_weight": sum(float(row.get("null_evidence_weight", 0)) for row in rows) / len(rows),
                "mean_real_evidence_mass": sum(float(row.get("real_evidence_mass", 0)) for row in rows) / len(rows),
                "mean_visual_evidence_delta_norm": (
                    sum(float(row["visual_evidence_delta_norm"]) for row in rows if "visual_evidence_delta_norm" in row)
                    / max(1, sum("visual_evidence_delta_norm" in row for row in rows))
                ),
            }
    return result


def compute_reliability_summary(records):
    return _group_summary(records, query=False)


def compute_query_attention_summary(records):
    return _group_summary(records, query=True)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _area_px_bin(area):
    if area <= 0: return "empty"
    if area <= 16: return "tiny_le_16px"
    if area <= 64: return "small_17_64px"
    if area <= 256: return "medium_65_256px"
    if area <= 1024: return "large_257_1024px"
    return "very_large_gt_1024px"


def _fraction_bin(value):
    if value <= 0: return "empty"
    if value <= 0.001: return "tiny_le_0.1pct"
    if value <= 0.01: return "small_0.1_1pct"
    if value <= 0.05: return "medium_1_5pct"
    if value <= 0.20: return "large_5_20pct"
    return "very_large_gt_20pct"


def _ground_bin(value):
    if value is None: return "unknown"
    if value <= 0: return "empty"
    if value <= 100: return "tiny_le_100m2"
    if value <= 1000: return "small_100_1k_m2"
    if value <= 10000: return "medium_1k_10k_m2"
    if value <= 100000: return "large_10k_100k_m2"
    return "very_large_gt_100k_m2"


def metric_metadata_with_scale(
    metadata: list[dict[str, Any]],
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    binary = target.detach().float().cpu() >= 0.5
    valid = (
        valid_mask.detach().float().cpu() >= 0.5
        if valid_mask is not None else torch.ones_like(binary, dtype=torch.bool)
    )
    enriched = []
    for index, meta in enumerate(metadata):
        row = dict(meta)
        area = float((binary[index, 0] & valid[index, 0]).sum())
        valid_area = float(valid[index, 0].sum())
        fraction = area / max(valid_area, 1.0)
        transform = row.get("resize_transform") or {}
        scale, gsd = _number(transform.get("scale")), _number(row.get("gsd_m"))
        ground = area / (scale * scale) * gsd * gsd if scale and gsd and scale > 0 and gsd > 0 else None
        row.update({
            "target_area_px": area, "target_area_fraction": fraction, "metric_valid_area_px": valid_area,
            "target_area_px_bin": _area_px_bin(area), "target_area_fraction_bin": _fraction_bin(fraction),
            "ground_area_m2": ground, "ground_area_m2_bin": _ground_bin(ground),
        })
        enriched.append(row)
    return enriched


def _packed_mask(mask: torch.Tensor, size: int = 16) -> str:
    value = F.interpolate(mask[None, None].float(), (size, size), mode="nearest")[0, 0]
    packed = np.packbits((value >= 0.5).to(torch.uint8).cpu().numpy().reshape(-1))
    return packed.tobytes().hex()


def _packed_iou(first: str, second: str) -> float:
    left = np.unpackbits(np.frombuffer(bytes.fromhex(first), dtype=np.uint8)).astype(bool)
    right = np.unpackbits(np.frombuffer(bytes.fromhex(second), dtype=np.uint8)).astype(bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def collect_proposal_records(outputs, target, valid, metadata, sample_metrics, threshold=0.5):
    if "proposal_relevance_logits" not in outputs:
        return []
    relevance = outputs["proposal_relevance_logits"].detach().float().cpu()
    final = torch.sigmoid(outputs["final_mask_logits"].detach().float().cpu())
    relevance_targets = outputs.get("proposal_relevance_targets")
    relevance_targets = relevance_targets.detach().float().cpu() if relevance_targets is not None else torch.zeros_like(relevance)
    report_names = {
        "proposal_matched_mean_dice": "matched_mean_dice",
        "proposal_component_recall": "component_recall",
        "proposal_component_precision": "component_precision",
        "proposal_unmatched_rejection": "unmatched_rejection",
        "proposal_relevance_ap": "relevance_ap",
        "proposal_relevance_auc": "relevance_auc",
        "proposal_union_dice": "proposal_union_dice",
        "proposal_merge_error_rate": "merge_error_rate",
        "proposal_duplicate_error_rate": "duplicate_error_rate",
        "proposal_missed_component_rate": "missed_component_rate",
        "proposal_component_count": "component_count",
        "proposal_matching_coverage_mode": "coverage_mode",
    }
    assignments = {
        report_name: outputs[output_name].detach().float().cpu()
        for output_name, report_name in report_names.items()
        if output_name in outputs
    }
    proposal_masks = torch.sigmoid(outputs["proposal_mask_logits"].detach().float().cpu())
    oracle_queries = outputs.get("proposal_oracle_matched_query")
    oracle_queries = oracle_queries.detach().long().cpu() if oracle_queries is not None else None
    oracle_dice = outputs.get("proposal_oracle_matched_dice")
    oracle_dice = oracle_dice.detach().float().cpu() if oracle_dice is not None else None
    scale_attention = outputs.get("query_scale_attention")
    scale_attention = scale_attention.detach().float().cpu() if scale_attention is not None else None
    records = []
    for index, meta in enumerate(metadata):
        selected = int(relevance[index].argmax())
        oracle = int(oracle_queries[index]) if oracle_queries is not None else -1
        matched = relevance_targets[index] >= 0.5
        if bool(matched.any()):
            order = torch.argsort(relevance[index], descending=True)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(
                1, relevance.shape[1] + 1, dtype=torch.float32
            )
            matched_mean_rank = float(ranks[matched].mean())
            matched_rank_score = (
                1.0
                if relevance.shape[1] <= 1
                else 1.0 - (matched_mean_rank - 1.0) / (relevance.shape[1] - 1.0)
            )
        else:
            matched_mean_rank = None
            matched_rank_score = None
        sample_valid = valid[index, 0] >= 0.5
        prediction = (final[index, 0] >= float(threshold)) & sample_valid
        target_binary = (target[index, 0] >= 0.5) & sample_valid
        metric = sample_metrics[index] if index < len(sample_metrics) else {}
        row = {
            **meta, "num_queries": int(relevance.shape[1]), "selected_query": selected,
            "selected_is_matched": float(relevance_targets[index, selected] >= 0.5),
            "selected_relevance_logit": float(relevance[index, selected]),
            "oracle_matched_query": oracle,
            "oracle_matched_dice": float(oracle_dice[index]) if oracle_dice is not None else None,
            "oracle_relevance_logit": float(relevance[index, oracle]) if oracle >= 0 else None,
            "matched_relevance_mean_rank": matched_mean_rank,
            "matched_relevance_rank_score": matched_rank_score,
            "final_dice": metric.get("dice"), "final_iou": metric.get("iou"),
            "final_precision": metric.get("precision"), "final_recall": metric.get("recall"),
            "target_area": float(target_binary.sum()),
            "final_mask_area": float(prediction.sum()),
            "selected_mask_area": float(
                ((proposal_masks[index, selected] >= float(threshold)) & (valid[index, 0] >= 0.5)).sum()
            ),
            "oracle_mask_area": float(
                ((proposal_masks[index, oracle] >= float(threshold)) & (valid[index, 0] >= 0.5)).sum()
            ) if oracle >= 0 else 0.0,
            "prediction_hash": hashlib.sha256(prediction.to(torch.uint8).numpy().tobytes()).hexdigest(),
            "prediction_signature_16": _packed_mask(prediction),
            "target_signature_16": _packed_mask(target_binary),
        }
        if scale_attention is not None:
            for scale_index, scale_name in enumerate(("high", "mid", "low")):
                row[f"selected_scale_attention_{scale_name}"] = float(
                    scale_attention[index, selected, scale_index]
                )
        for name, values in assignments.items():
            row[name] = float(values[index])
        records.append(row)
    return records


def compute_proposal_summary(records):
    if not records:
        return {}
    groups = {"overall": records}
    for record in records:
        for group in _group_fields(record):
            groups.setdefault(group, []).append(record)
    excluded = {"sample_id", "parent_sample_id", "dataset_name", "template_id", "task_family", "prediction_hash"}
    result = {}
    for group, rows in sorted(groups.items()):
        numeric = sorted({key for row in rows for key, value in row.items() if key not in excluded and isinstance(value, (int, float))})
        result[group] = {"n": len(rows), **{
            f"mean_{key}": sum(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))
            / sum(isinstance(row.get(key), (int, float)) for row in rows)
            for key in numeric
        }}
    return result


def paired_instruction_summary(records):
    parents: dict[str, list[dict[str, Any]]] = {}
    no_target = []
    for record in records:
        if record.get("task_family") == "referring_landslide_segmentation":
            parents.setdefault(str(record.get("parent_sample_id")), []).append(record)
        elif record.get("task_family") == "no_target_segmentation":
            no_target.append(record)
    paired = [rows for rows in parents.values() if len({row.get("target_signature_16") for row in rows}) >= 2]
    target_ious = []
    prediction_ious = []
    for rows in paired:
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                target_ious.append(_packed_iou(rows[left]["target_signature_16"], rows[right]["target_signature_16"]))
                prediction_ious.append(_packed_iou(
                    rows[left]["prediction_signature_16"], rows[right]["prediction_signature_16"]
                ))
    target_contrast = sum(1.0 - value for value in target_ious) / len(target_ious) if target_ious else None
    prediction_contrast = (
        sum(1.0 - value for value in prediction_ious) / len(prediction_ious) if prediction_ious else None
    )
    return {
        "num_paired_parents": len(paired),
        "num_paired_comparisons": len(target_ious),
        "paired_prediction_difference_rate": (
            sum(len({row.get('prediction_hash') for row in rows}) >= 2 for rows in paired) / len(paired)
            if paired else None
        ),
        "mean_paired_target_iou_16": sum(target_ious) / len(target_ious) if target_ious else None,
        "mean_paired_prediction_iou_16": sum(prediction_ious) / len(prediction_ious) if prediction_ious else None,
        "mean_target_contrast_16": target_contrast,
        "mean_prediction_contrast_16": prediction_contrast,
        "instruction_contrast_ratio_16": (
            prediction_contrast / target_contrast
            if prediction_contrast is not None and target_contrast is not None and target_contrast > 1.0e-6
            else None
        ),
        "num_no_target": len(no_target),
        "no_target_empty_prediction_rate": (
            sum(float(row.get("final_mask_area", 1) == 0) for row in no_target) / len(no_target)
            if no_target else None
        ),
        "no_target_mean_false_positive_pixels": (
            sum(float(row.get("final_mask_area", 0)) for row in no_target) / len(no_target)
            if no_target else None
        ),
        "no_target_mean_unmatched_rejection": (
            sum(float(row.get("unmatched_rejection", 0)) for row in no_target) / len(no_target)
            if no_target else None
        ),
    }
