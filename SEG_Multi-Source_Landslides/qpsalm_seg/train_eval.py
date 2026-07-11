#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练、验证、checkpoint 与模型构建公共逻辑。"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import QPSalmConfig, save_config
from .controllers import build_controller
from .data import (
    MultiSourceLandslideDataset,
    SizeBucketBatchSampler,
    qpsalm_collate,
)
from .indexing import canonical_modality_combo, normalization_combo, raw_modality_combo, sensor_combo
from .losses import dice_scores_with_logits
from .metrics import MetricAccumulator, batch_binary_metrics
from .models import MultiSourceQwenPSALMSeg
from .paths import resolve_repo_path
from .qwen_cache import assert_qwen_cache_coverage
from .visualize import restore_mask_to_original, save_visualizations


CHECKPOINT_FORMAT = "qpsalm_sane_qmef_pmrd_v1"
EXCLUDED_FROZEN_PREFIXES = ("controller.model.",)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def utc_now() -> str:
    """返回 UTC ISO 时间戳，用于实验 manifest。"""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """原子写 torch checkpoint，避免中断时留下半截 .pt 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_standalone_train_manifest(out_dir: Path, config: QPSalmConfig, device_name: str, resume: str | None) -> None:
    """直接调用 qpsalm-train 时写入稳定的 run manifest。"""
    path = out_dir / "run_manifest.json"
    if path.exists():
        return
    write_json(
        path,
        {
            "created_at_utc": utc_now(),
            "created_by": "qpsalm-train",
            "preset": config.preset,
            "run_dir": str(out_dir),
            "device": device_name,
            "resume": resume,
            "checkpoint_last": str(out_dir / "checkpoint_last.pt"),
            "validation_latest": str(out_dir / "validation_latest.json"),
            "resolved_config": dict(config.__dict__),
        },
    )


def resolve_device(requested: str) -> torch.device:
    """解析设备；CUDA 不可见时给出清晰错误。"""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False in this session.")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested --device {requested}, but CUDA is unavailable in this session.")
        return torch.device(requested)
    return torch.device(requested)


def build_dataloaders(config: QPSalmConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = MultiSourceLandslideDataset(
        config,
        split="train",
        max_samples=config.max_train_samples,
        shuffle_seed=config.seed,
    )
    val_ds = MultiSourceLandslideDataset(
        config,
        split="val",
        max_samples=config.max_val_samples,
        shuffle_seed=None,
    )
    loader_common = {
        "num_workers": config.num_workers,
        "collate_fn": qpsalm_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if getattr(config, "size_buckets", []):
        train_loader = DataLoader(
            train_ds,
            batch_sampler=SizeBucketBatchSampler(
                train_ds,
                config.batch_size,
                shuffle=True,
                seed=config.seed,
            ),
            **loader_common,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=SizeBucketBatchSampler(
                val_ds,
                config.batch_size,
                shuffle=False,
                seed=config.seed,
            ),
            **loader_common,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **loader_common)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **loader_common)
    return train_loader, val_loader


def build_model(config: QPSalmConfig, device: torch.device) -> MultiSourceQwenPSALMSeg:
    controller = build_controller(config, device)
    model = MultiSourceQwenPSALMSeg(config, controller)
    model.to(device)
    return model


def cosine_lr(step: int, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


LOSS_LOG_KEYS = [
    "loss_mask_bce",
    "loss_mask_dice",
    "loss_boundary",
    "loss_proposal_set",
    "loss_proposal_coarse",
    "loss_proposal_coverage",
    "loss_semantic_verifier",
    "loss_missing_modality_consistency",
]


def scalar_tensor(value: torch.Tensor) -> float:
    """把标量或 batch tensor 汇总为 float，用于日志 JSON。"""
    return float(value.detach().float().mean().cpu().item())


def loss_log_values(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    """提取训练/验证可读的 loss 组件和 best-query 统计。"""
    values: dict[str, float] = {}
    for key in LOSS_LOG_KEYS:
        if key in outputs:
            values[key] = scalar_tensor(outputs[key])
    if "best_query_dice" in outputs:
        values["best_query_dice"] = scalar_tensor(outputs["best_query_dice"])
    if "best_query" in outputs:
        values["best_query_mean"] = scalar_tensor(outputs["best_query"])
    if "verifier_best_query_accuracy" in outputs:
        values["verifier_best_query_accuracy"] = scalar_tensor(outputs["verifier_best_query_accuracy"])
    if "verifier_positive_target_accuracy" in outputs:
        values["verifier_positive_target_accuracy"] = scalar_tensor(outputs["verifier_positive_target_accuracy"])
    if "proposal_target_mass" in outputs:
        values["proposal_target_mass"] = scalar_tensor(outputs["proposal_target_mass"])
    if "proposal_target_positive_count" in outputs:
        values["proposal_target_positive_count"] = scalar_tensor(outputs["proposal_target_positive_count"])
    if "proposal_component_count" in outputs:
        values["proposal_component_count"] = scalar_tensor(outputs["proposal_component_count"])
    if "proposal_matching_coverage_mode" in outputs:
        values["proposal_matching_coverage_fraction"] = scalar_tensor(outputs["proposal_matching_coverage_mode"])
    return values


def training_signal_values(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    """记录 SANE/QMEF/PMRD 的高信号训练诊断。"""
    values: dict[str, float] = {}
    if "modality_reliability_weights" in outputs:
        reliability = outputs["modality_reliability_weights"].detach().float().cpu()
        safe = reliability.clamp_min(1.0e-8)
        values["modality_reliability_entropy"] = float((-(safe * safe.log()).sum(dim=1)).mean().item())
        values["modality_reliability_peak"] = float(safe.max(dim=1).values.mean().item())
    if "modality_active" in outputs:
        active = outputs["modality_active"].detach().float().cpu()
        values["active_modality_count"] = float(active.sum(dim=1).mean().item())
    if "query_modality_attention" in outputs:
        attention = outputs["query_modality_attention"].detach().float().cpu().clamp_min(1.0e-8)
        values["query_modality_attention_entropy"] = float((-(attention * attention.log()).sum(dim=-1)).mean().item())
        values["query_modality_attention_peak"] = float(attention.max(dim=-1).values.mean().item())
    if "proposal_relevance_logits" in outputs:
        scores = outputs["proposal_relevance_logits"].detach().float().cpu()
        values["proposal_relevance_mean"] = float(scores.mean().item())
        values["proposal_relevance_max"] = float(scores.max().item())
        values["top_query_mean"] = float(torch.argmax(scores, dim=1).float().mean().item())
    if "proposal_relevance_gates" in outputs:
        gates = outputs["proposal_relevance_gates"].detach().float().cpu()
        values["proposal_relevance_gate_mean"] = float(gates.mean().item())
        values["proposal_relevance_gate_max"] = float(gates.max().item())
    if "final_mask_logits" in outputs:
        logits = outputs["final_mask_logits"].detach().float().cpu()
        values["final_logit_mean"] = float(logits.mean().item())
        values["final_logit_std"] = float(logits.std(unbiased=False).item())
    return values


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    """对同结构标量 dict 求均值。"""
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values:
            out[key] = sum(values) / len(values)
    return out


TRAIN_LOG_KEYS = [
    "loss",
    "iou",
    "dice",
    "best_query_dice",
    "verifier_best_query_accuracy",
    "verifier_positive_target_accuracy",
    "proposal_target_positive_count",
    "proposal_component_count",
    "proposal_matching_coverage_fraction",
    "proposal_relevance_max",
    "proposal_relevance_gate_mean",
    "proposal_relevance_gate_max",
    "top_query_mean",
    "modality_reliability_entropy",
    "modality_reliability_peak",
    "query_modality_attention_entropy",
    "query_modality_attention_peak",
    "active_modality_count",
]


def summarize_train_window(rows: list[dict[str, Any]], elapsed_sec: float) -> dict[str, float]:
    """把最近若干 step 聚合成高信号终端日志。"""
    summary: dict[str, float] = {}
    for key in TRAIN_LOG_KEYS:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = sum(values) / len(values)
    summary["steps_per_sec"] = len(rows) / max(float(elapsed_sec), 1.0e-6)
    if rows and isinstance(rows[-1].get("lr"), (int, float)):
        summary["lr"] = float(rows[-1]["lr"])
    return summary


def format_train_window(start_step: int, end_step: int, count: int, summary: dict[str, float]) -> str:
    """格式化窗口训练日志，避免每个 step 打印刷屏。"""
    parts = [
        f"steps={start_step}-{end_step}",
        f"n={count}",
        f"loss={summary.get('loss', 0.0):.4f}",
        f"iou={summary.get('iou', 0.0):.4f}",
        f"dice={summary.get('dice', 0.0):.4f}",
        f"best_q_dice={summary.get('best_query_dice', 0.0):.4f}",
        f"lr={summary.get('lr', 0.0):.2e}",
        f"sps={summary.get('steps_per_sec', 0.0):.2f}",
    ]
    optional = [
        ("bestQ_acc", "verifier_best_query_accuracy", ".3f"),
        ("target_acc", "verifier_positive_target_accuracy", ".3f"),
        ("posQ", "proposal_target_positive_count", ".1f"),
        ("components", "proposal_component_count", ".1f"),
        ("coverage", "proposal_matching_coverage_fraction", ".2f"),
        ("top_q", "top_query_mean", ".1f"),
        ("rel_max", "proposal_relevance_max", ".2f"),
        ("relGate", "proposal_relevance_gate_mean", ".2f"),
        ("relH", "modality_reliability_entropy", ".2f"),
        ("relP", "modality_reliability_peak", ".2f"),
        ("qAttnH", "query_modality_attention_entropy", ".2f"),
        ("qAttnP", "query_modality_attention_peak", ".2f"),
        ("activeM", "active_modality_count", ".1f"),
    ]
    for label, key, fmt in optional:
        if key in summary:
            parts.append(f"{label}={summary[key]:{fmt}}")
    return "train " + " ".join(parts)


def _gate_row_to_dict(values: torch.Tensor, names: list[str] | None = None) -> dict[str, float]:
    """把单样本模态 gate 张量转成稳定 JSON 字段。"""
    labels = names or [f"modality_{index}" for index in range(int(values.numel()))]
    return {str(labels[index]): float(values[index].item()) for index in range(min(len(labels), int(values.numel())))}


def collect_reliability_records(outputs: dict[str, torch.Tensor], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """收集 QMEF 的样本级模态可靠性先验。"""
    if "modality_reliability_weights" not in outputs:
        return []
    weights = outputs["modality_reliability_weights"].detach().float().cpu()
    active = outputs.get("modality_active")
    active_cpu = active.detach().float().cpu() if active is not None else None
    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        names = [str(item.get("name", f"modality_{j}")) for j, item in enumerate(meta.get("raw_modalities") or [])]
        record = {
            "sample_id": meta.get("sample_id", ""),
            "canonical_combo": meta.get("canonical_combo", "unknown"),
            "raw_combo": meta.get("raw_combo", "unknown"),
            "sensor_combo": meta.get("sensor_combo", "unknown"),
            "normalization_combo": meta.get("normalization_combo", "unknown"),
            "gsd_token": meta.get("gsd_token", "unknown"),
            "target_area_px_bin": meta.get("target_area_px_bin", "unknown"),
            "target_area_fraction_bin": meta.get("target_area_fraction_bin", "unknown"),
            "ground_area_m2_bin": meta.get("ground_area_m2_bin", "unknown"),
            "condition_prompt": meta.get("condition_prompt", "unknown"),
            "weights": _gate_row_to_dict(weights[idx], names),
        }
        if active_cpu is not None:
            record["active"] = _gate_row_to_dict(active_cpu[idx], names)
        records.append(record)
    return records


def collect_query_modality_attention_records(outputs: dict[str, torch.Tensor], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """收集每个 mask query 的模态注意力，诊断 proposal 是否按实例选择证据源。"""
    if "query_modality_attention" not in outputs:
        return []
    weights = outputs["query_modality_attention"].detach().float().cpu()
    if weights.ndim != 3:
        return []
    mean_weights = weights.mean(dim=1)
    entropy = outputs.get("query_spatial_entropy_mean")
    peak = outputs.get("query_modality_attention_peak")
    entropy_cpu = entropy.detach().float().cpu() if entropy is not None else None
    peak_cpu = peak.detach().float().cpu() if peak is not None else None
    best_query = outputs.get("best_query")
    best_query_cpu = best_query.detach().long().cpu() if best_query is not None else None
    relevance_logits = outputs.get("proposal_relevance_logits")
    selected_query_cpu = (
        torch.argmax(relevance_logits.detach().float().cpu(), dim=1)
        if relevance_logits is not None
        else None
    )
    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        names = [str(item.get("name", f"modality_{j}")) for j, item in enumerate(meta.get("raw_modalities") or [])]
        record: dict[str, Any] = {
            "sample_id": meta.get("sample_id", ""),
            "canonical_combo": meta.get("canonical_combo", "unknown"),
            "raw_combo": meta.get("raw_combo", "unknown"),
            "sensor_combo": meta.get("sensor_combo", "unknown"),
            "normalization_combo": meta.get("normalization_combo", "unknown"),
            "gsd_token": meta.get("gsd_token", "unknown"),
            "target_area_px_bin": meta.get("target_area_px_bin", "unknown"),
            "target_area_fraction_bin": meta.get("target_area_fraction_bin", "unknown"),
            "ground_area_m2_bin": meta.get("ground_area_m2_bin", "unknown"),
            "condition_prompt": meta.get("condition_prompt", "unknown"),
            "mean_query_weights": _gate_row_to_dict(mean_weights[idx], names),
        }
        if entropy_cpu is not None:
            record["entropy"] = float(entropy_cpu[idx].item())
        if peak_cpu is not None:
            record["peak"] = float(peak_cpu[idx].item())
        if selected_query_cpu is not None:
            selected = int(selected_query_cpu[idx].item())
            record["selected_query"] = selected
            record["selected_query_weights"] = _gate_row_to_dict(weights[idx, selected], names)
        if best_query_cpu is not None:
            best = int(best_query_cpu[idx].item())
            record["best_query"] = best
            record["best_query_weights"] = _gate_row_to_dict(weights[idx, best], names)
        records.append(record)
    return records


def _average_named_values(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    """对 records 中 weights/active 这类 name->float 字典求均值。"""
    names = sorted(
        {
            str(name)
            for record in records
            for name in ((record.get(field) or {}).keys() if isinstance(record.get(field), dict) else [])
        }
    )
    out: dict[str, float] = {}
    for name in names:
        values = [
            float((record.get(field) or {}).get(name, 0.0))
            for record in records
        ]
        out[name] = sum(values) / len(values) if values else 0.0
    return out


def _average_numeric_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def compute_reliability_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按 overall/combo/condition 聚合 QMEF 模态可靠性。"""
    if not records:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {"overall": records}
    for record in records:
        groups.setdefault(f"canonical_combo={record.get('canonical_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"raw_combo={record.get('raw_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"sensor_combo={record.get('sensor_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"normalization_combo={record.get('normalization_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"gsd_token={record.get('gsd_token', 'unknown')}", []).append(record)
        groups.setdefault(f"target_area_px_bin={record.get('target_area_px_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"target_area_fraction_bin={record.get('target_area_fraction_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"ground_area_m2_bin={record.get('ground_area_m2_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"condition={record.get('condition_prompt', 'unknown')}", []).append(record)
    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        summary[name] = {
            "n": len(rows),
            "mean_weights": _average_named_values(rows, "weights"),
            "mean_active": _average_named_values(rows, "active"),
        }
    return summary


def compute_query_modality_attention_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按 overall/combo/condition 聚合 query-level modality attention。"""
    if not records:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {"overall": records}
    for record in records:
        groups.setdefault(f"canonical_combo={record.get('canonical_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"raw_combo={record.get('raw_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"sensor_combo={record.get('sensor_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"normalization_combo={record.get('normalization_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"gsd_token={record.get('gsd_token', 'unknown')}", []).append(record)
        groups.setdefault(f"target_area_px_bin={record.get('target_area_px_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"target_area_fraction_bin={record.get('target_area_fraction_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"ground_area_m2_bin={record.get('ground_area_m2_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"condition={record.get('condition_prompt', 'unknown')}", []).append(record)
    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        summary[name] = {
            "n": len(rows),
            "mean_query_weights": _average_named_values(rows, "mean_query_weights"),
            "mean_selected_query_weights": _average_named_values(rows, "selected_query_weights"),
            "mean_best_query_weights": _average_named_values(rows, "best_query_weights"),
            "mean_entropy": _average_numeric_field(rows, "entropy"),
            "mean_peak": _average_numeric_field(rows, "peak"),
        }
    return summary


def _matrix_rank_of_col(matrix: torch.Tensor | None, row: int, col: int) -> float | None:
    if matrix is None:
        return None
    order = torch.argsort(matrix[row], descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), dtype=order.dtype)
    return float(ranks[int(col)].item() + 1)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def target_area_px_bin(area_px: float) -> str:
    """按训练 canvas 像素面积分层，定位小目标/大目标分割差异。"""
    if area_px <= 0:
        return "empty"
    if area_px <= 32:
        return "tiny_1_32px"
    if area_px <= 128:
        return "small_33_128px"
    if area_px <= 512:
        return "medium_129_512px"
    if area_px <= 2048:
        return "large_513_2048px"
    return "very_large_gt_2048px"


def target_area_fraction_bin(frac: float) -> str:
    """按 mask 占 target canvas 比例分层。"""
    if frac <= 0:
        return "empty"
    if frac <= 0.001:
        return "tiny_le_0.1pct"
    if frac <= 0.005:
        return "small_0.1_0.5pct"
    if frac <= 0.02:
        return "medium_0.5_2pct"
    if frac <= 0.10:
        return "large_2_10pct"
    return "very_large_gt_10pct"


def ground_area_m2_bin(area_m2: float | None) -> str:
    """按估算地面面积分层；GSD/resize 缺失时返回 unknown。"""
    if area_m2 is None:
        return "unknown"
    if area_m2 <= 0:
        return "empty"
    if area_m2 <= 100.0:
        return "tiny_le_100m2"
    if area_m2 <= 1000.0:
        return "small_100_1k_m2"
    if area_m2 <= 10000.0:
        return "medium_1k_10k_m2"
    if area_m2 <= 100000.0:
        return "large_10k_100k_m2"
    return "very_large_gt_100k_m2"


def metric_metadata_with_scale(metadata: list[dict[str, Any]], target_mask: torch.Tensor) -> list[dict[str, Any]]:
    """为指标聚合补充 target area/GSD/ground area 分层字段。"""
    target = (target_mask.detach().float().cpu() >= 0.5)
    height = int(target.shape[-2])
    width = int(target.shape[-1])
    canvas_area = max(1.0, float(height * width))
    enriched: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        item = dict(meta)
        area_px = float(target[idx, 0].sum().item()) if idx < target.shape[0] else 0.0
        area_fraction = area_px / canvas_area
        transform = item.get("resize_transform") if isinstance(item.get("resize_transform"), dict) else {}
        scale = _safe_float(transform.get("scale") if isinstance(transform, dict) else None)
        gsd = _safe_float(item.get("gsd_m"))
        ground_area = None
        if gsd is not None and gsd > 0 and scale is not None and scale > 0:
            original_pixel_area = area_px / (scale * scale)
            ground_area = original_pixel_area * gsd * gsd
        item["target_area_px"] = area_px
        item["target_area_fraction"] = area_fraction
        item["target_area_px_bin"] = target_area_px_bin(area_px)
        item["target_area_fraction_bin"] = target_area_fraction_bin(area_fraction)
        item["ground_area_m2"] = ground_area
        item["ground_area_m2_bin"] = ground_area_m2_bin(ground_area)
        enriched.append(item)
    return enriched


def collect_proposal_records(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    metadata: list[dict[str, Any]],
    sample_metrics: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """记录 PMRD proposal set 与统一 semantic verifier 的样本级诊断。"""
    if "proposal_mask_logits" not in outputs or "proposal_relevance_logits" not in outputs:
        return []
    proposal_masks = outputs["proposal_mask_logits"].detach().float().cpu()
    target = target_mask.detach().float().cpu()
    valid = valid_mask.detach().float().cpu()
    dice_scores = dice_scores_with_logits(proposal_masks, target, valid_mask=valid)
    best_query = outputs.get("best_query")
    best_query_cpu = best_query.detach().long().cpu() if best_query is not None else dice_scores.argmax(dim=1)
    relevance = outputs["proposal_relevance_logits"].detach().float().cpu()
    selected_query = torch.argmax(relevance, dim=1)
    final_probs = torch.sigmoid(outputs["final_mask_logits"].detach().float().cpu())
    proposal_mask_probs = torch.sigmoid(proposal_masks)

    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        selected = int(selected_query[idx].item())
        best = int(best_query_cpu[idx].item())
        metrics = sample_metrics[idx] if idx < len(sample_metrics) else {}
        record: dict[str, Any] = {
            "sample_id": meta.get("sample_id", ""),
            "dataset_name": meta.get("dataset_name", "unknown"),
            "template_id": meta.get("template_id", "unknown"),
            "task_family": meta.get("task_family", "unknown"),
            "raw_combo": meta.get("raw_combo", "unknown"),
            "canonical_combo": meta.get("canonical_combo", "unknown"),
            "sensor_combo": meta.get("sensor_combo", "unknown"),
            "normalization_combo": meta.get("normalization_combo", "unknown"),
            "condition_prompt": meta.get("condition_prompt", "unknown"),
            "gsd_token": meta.get("gsd_token", "unknown"),
            "target_area_px_bin": meta.get("target_area_px_bin", "unknown"),
            "target_area_fraction_bin": meta.get("target_area_fraction_bin", "unknown"),
            "ground_area_m2": meta.get("ground_area_m2"),
            "ground_area_m2_bin": meta.get("ground_area_m2_bin", "unknown"),
            "best_query": best,
            "selected_query": selected,
            "selected_matches_best": float(selected == best),
            "best_query_dice": float(dice_scores[idx, best].item()),
            "selected_query_dice": float(dice_scores[idx, selected].item()),
            "selected_relevance_logit": float(relevance[idx, selected].item()),
            "best_relevance_logit": float(relevance[idx, best].item()),
            "best_query_relevance_rank": _matrix_rank_of_col(relevance, idx, best),
            "final_dice": metrics.get("dice"),
            "final_iou": metrics.get("iou"),
            "final_precision": metrics.get("precision"),
            "final_recall": metrics.get("recall"),
            "target_area": float((target[idx, 0] >= 0.5).sum().item()),
            "final_mask_area": float((final_probs[idx, 0] >= 0.5).sum().item()),
            "selected_mask_area": float((proposal_mask_probs[idx, selected] >= 0.5).sum().item()),
            "best_mask_area": float((proposal_mask_probs[idx, best] >= 0.5).sum().item()),
        }
        record["relevance_gap_selected_minus_best"] = (
            record["selected_relevance_logit"] - record["best_relevance_logit"]
        )
        record["dice_gap_selected_minus_best"] = record["selected_query_dice"] - record["best_query_dice"]
        records.append(record)
    return records


def compute_proposal_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按 overall/combo/condition 聚合 proposal selection 诊断。"""
    if not records:
        return {}
    numeric_fields = [
        "selected_matches_best",
        "best_query_dice",
        "selected_query_dice",
        "selected_relevance_logit",
        "best_relevance_logit",
        "best_query_relevance_rank",
        "relevance_gap_selected_minus_best",
        "dice_gap_selected_minus_best",
        "final_dice",
        "final_iou",
        "final_precision",
        "final_recall",
        "target_area",
        "ground_area_m2",
        "final_mask_area",
        "selected_mask_area",
        "best_mask_area",
    ]
    groups: dict[str, list[dict[str, Any]]] = {"overall": records}
    for record in records:
        groups.setdefault(f"canonical_combo={record.get('canonical_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"raw_combo={record.get('raw_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"sensor_combo={record.get('sensor_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"normalization_combo={record.get('normalization_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"condition={record.get('condition_prompt', 'unknown')}", []).append(record)
        groups.setdefault(f"gsd_token={record.get('gsd_token', 'unknown')}", []).append(record)
        groups.setdefault(f"target_area_px_bin={record.get('target_area_px_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"target_area_fraction_bin={record.get('target_area_fraction_bin', 'unknown')}", []).append(record)
        groups.setdefault(f"ground_area_m2_bin={record.get('ground_area_m2_bin', 'unknown')}", []).append(record)
    summary: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        item: dict[str, Any] = {"n": len(rows)}
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in rows
                if isinstance(row.get(field), (int, float))
            ]
            if values:
                item[f"mean_{field}"] = sum(values) / len(values)
        summary[group] = item
    return summary


def normalize_thresholds(values: list[float] | tuple[float, ...] | None) -> list[float]:
    """规范化阈值扫描列表，保证在 [0,1] 内且顺序稳定。"""
    if not values:
        return []
    cleaned = sorted({round(float(value), 4) for value in values if 0.0 <= float(value) <= 1.0})
    return cleaned


def compute_threshold_sweep_report(accumulators: dict[float, MetricAccumulator]) -> dict[str, Any]:
    """汇总不同二值化阈值下的 overall 与分组指标。"""
    if not accumulators:
        return {}
    by_threshold: dict[str, dict[str, float]] = {}
    groups_by_threshold: dict[str, dict[str, dict[str, float]]] = {}
    best_by_dice: dict[str, Any] | None = None
    best_by_iou: dict[str, Any] | None = None
    best_by_dice_per_group: dict[str, dict[str, Any]] = {}
    best_by_iou_per_group: dict[str, dict[str, Any]] = {}
    for threshold, acc in sorted(accumulators.items()):
        groups = acc.compute()
        overall = groups.get("overall")
        if not overall:
            continue
        key = f"{threshold:.2f}"
        by_threshold[key] = overall
        groups_by_threshold[key] = groups
        row = {"threshold": threshold, **overall}
        if best_by_dice is None or float(overall.get("dice", -1.0)) > float(best_by_dice.get("dice", -1.0)):
            best_by_dice = row
        if best_by_iou is None or float(overall.get("iou", -1.0)) > float(best_by_iou.get("iou", -1.0)):
            best_by_iou = row
        for group_name, group_values in groups.items():
            group_dice = float(group_values.get("dice", -1.0))
            group_iou = float(group_values.get("iou", -1.0))
            group_row = {"threshold": threshold, **group_values}
            prev_dice = best_by_dice_per_group.get(group_name)
            if prev_dice is None or group_dice > float(prev_dice.get("dice", -1.0)):
                best_by_dice_per_group[group_name] = group_row
            prev_iou = best_by_iou_per_group.get(group_name)
            if prev_iou is None or group_iou > float(prev_iou.get("iou", -1.0)):
                best_by_iou_per_group[group_name] = group_row
    return {
        "overall_by_threshold": by_threshold,
        "groups_by_threshold": groups_by_threshold,
        "best_by_dice": best_by_dice,
        "best_by_iou": best_by_iou,
        "best_by_dice_per_group": dict(sorted(best_by_dice_per_group.items())),
        "best_by_iou_per_group": dict(sorted(best_by_iou_per_group.items())),
    }


def restored_original_space_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    metadata: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, float]]:
    """将有效 canvas 反变换回 source H/W 后逐样本计算指标。"""
    probabilities = torch.sigmoid(logits.detach().float().cpu())
    targets = target.detach().float().cpu()
    records: list[dict[str, float]] = []
    for index, meta in enumerate(metadata):
        transform = meta.get("resize_transform")
        pred = (probabilities[index, 0].numpy() >= float(threshold)).astype(np.uint8)
        gt = (targets[index, 0].numpy() >= 0.5).astype(np.uint8)
        restored_pred = restore_mask_to_original(pred, transform)
        restored_gt = restore_mask_to_original(gt, transform)
        if restored_pred is None or restored_gt is None:
            restored_pred, restored_gt = pred, gt
        pred_tensor = torch.from_numpy(restored_pred).float()[None, None]
        gt_tensor = torch.from_numpy(restored_gt).float()[None, None]
        pred_logits = torch.where(pred_tensor > 0.5, torch.full_like(pred_tensor, 20.0), torch.full_like(pred_tensor, -20.0))
        records.extend(batch_binary_metrics(pred_logits, gt_tensor, threshold=0.5))
    return records


def canvas_original_metric_delta(
    canvas: dict[str, dict[str, float]],
    original: dict[str, dict[str, float]],
) -> dict[str, float]:
    """报告 target canvas 与原尺寸 overall 指标差值，便于发现恢复偏差。"""
    canvas_overall = canvas.get("overall") or {}
    original_overall = original.get("overall") or {}
    return {
        key: float(canvas_overall.get(key, 0.0)) - float(original_overall.get(key, 0.0))
        for key in ("dice", "iou", "precision", "recall")
        if key in canvas_overall and key in original_overall
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
    threshold_sweep: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator()
    original_acc = MetricAccumulator()
    sweep_thresholds = normalize_thresholds(threshold_sweep)
    sweep_accumulators = {value: MetricAccumulator() for value in sweep_thresholds}
    loss_values: list[float] = []
    loss_components: list[dict[str, float]] = []
    reliability_records: list[dict[str, Any]] = []
    query_attention_records: list[dict[str, Any]] = []
    proposal_records: list[dict[str, Any]] = []
    saved: list[str] = []
    processed_batches = 0
    processed_samples = 0
    canonical_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    gsd_counts: Counter[str] = Counter()
    target_area_px_bin_counts: Counter[str] = Counter()
    target_area_fraction_bin_counts: Counter[str] = Counter()
    ground_area_m2_bin_counts: Counter[str] = Counter()
    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    iterator = enumerate(loader)
    for batch_idx, batch in iterator:
        if max_batches is not None and batch_idx >= max_batches:
            break
        processed_batches += 1
        processed_samples += len(batch["metadata"])
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(batch)
        if "loss" in outputs:
            loss_values.append(float(outputs["loss"].detach().cpu().item()))
            loss_components.append(loss_log_values(outputs))
        logits_cpu = outputs["final_mask_logits"].detach().cpu()
        mask_cpu = batch["mask"].detach().cpu()
        metric_metadata = metric_metadata_with_scale(batch["metadata"], mask_cpu)
        for meta in metric_metadata:
            canonical_counts[str(meta.get("canonical_combo", "unknown"))] += 1
            raw_counts[str(meta.get("raw_combo", "unknown"))] += 1
            sensor_counts[str(meta.get("sensor_combo", "unknown"))] += 1
            normalization_counts[str(meta.get("normalization_combo", "unknown"))] += 1
            gsd_counts[str(meta.get("gsd_token", "unknown"))] += 1
            target_area_px_bin_counts[str(meta.get("target_area_px_bin", "unknown"))] += 1
            target_area_fraction_bin_counts[str(meta.get("target_area_fraction_bin", "unknown"))] += 1
            ground_area_m2_bin_counts[str(meta.get("ground_area_m2_bin", "unknown"))] += 1
        valid_cpu = batch["valid_mask"].detach().cpu()
        metrics = batch_binary_metrics(
            logits_cpu,
            mask_cpu,
            threshold=float(threshold),
            valid_mask=valid_cpu,
        )
        acc.update(metrics, metric_metadata)
        original_metrics = restored_original_space_metrics(
            logits_cpu,
            mask_cpu,
            metric_metadata,
            threshold=float(threshold),
        )
        original_acc.update(original_metrics, metric_metadata)
        for sweep_value, sweep_acc in sweep_accumulators.items():
            sweep_metrics = batch_binary_metrics(
                logits_cpu,
                mask_cpu,
                threshold=sweep_value,
                valid_mask=valid_cpu,
            )
            sweep_acc.update(sweep_metrics, metric_metadata)
        reliability_records.extend(collect_reliability_records(outputs, metric_metadata))
        query_attention_records.extend(collect_query_modality_attention_records(outputs, metric_metadata))
        proposal_records.extend(
            collect_proposal_records(outputs, batch["mask"], batch["valid_mask"], metric_metadata, metrics)
        )
        should_save_visuals = visual_dir is not None and (visualize_all or len(saved) < num_visualizations)
        if should_save_visuals:
            max_items = len(batch["metadata"]) if visualize_all else num_visualizations - len(saved)
            saved.extend(
                save_visualizations(
                    batch,
                    outputs,
                    visual_dir,
                    max_items=max_items,
                    prefix=f"val_b{batch_idx}",
                    threshold=float(threshold),
                    export_multimodal_overview=bool(export_multimodal_overview),
                )
            )
        del outputs
        if device.type == "cuda" and batch_idx % 50 == 49:
            torch.cuda.empty_cache()
    groups = acc.compute()
    original_groups = original_acc.compute()
    overview_dir = visual_dir / "multimodal_overviews" if visual_dir is not None else None
    manifest_path = visual_dir / "visualization_manifest.jsonl" if visual_dir is not None else None
    mask_export_dir = visual_dir / "mask_exports" if visual_dir is not None else None
    restored_mask_export_dir = visual_dir / "mask_exports_original_size" if visual_dir is not None else None
    overview_count = len(list(overview_dir.glob("*.png"))) if overview_dir is not None and overview_dir.exists() else 0
    mask_export_count = (
        len(list(mask_export_dir.rglob("*.png"))) if mask_export_dir is not None and mask_export_dir.exists() else 0
    )
    restored_mask_export_count = (
        len(list(restored_mask_export_dir.rglob("*.png")))
        if restored_mask_export_dir is not None and restored_mask_export_dir.exists()
        else 0
    )
    return {
        "loss": sum(loss_values) / len(loss_values) if loss_values else None,
        "loss_components": average_dicts(loss_components),
        "threshold": float(threshold),
        "coverage": {
            "num_batches": processed_batches,
            "num_samples": processed_samples,
            "canonical_combos": dict(sorted(canonical_counts.items())),
            "raw_combos": dict(sorted(raw_counts.items())),
            "sensor_combos": dict(sorted(sensor_counts.items())),
            "normalization_combos": dict(sorted(normalization_counts.items())),
            "gsd_tokens": dict(sorted(gsd_counts.items())),
            "target_area_px_bins": dict(sorted(target_area_px_bin_counts.items())),
            "target_area_fraction_bins": dict(sorted(target_area_fraction_bin_counts.items())),
            "ground_area_m2_bins": dict(sorted(ground_area_m2_bin_counts.items())),
            "max_batches": max_batches,
        },
        "threshold_sweep": compute_threshold_sweep_report(sweep_accumulators),
        "metrics": groups,
        "metrics_original_size": original_groups,
        "canvas_vs_original_delta": canvas_original_metric_delta(groups, original_groups),
        "modality_reliability_summary": compute_reliability_summary(reliability_records),
        "query_modality_attention_summary": compute_query_modality_attention_summary(query_attention_records),
        "proposal_diagnostics": {
            "records": proposal_records,
            "summary": compute_proposal_summary(proposal_records),
        },
        "visualizations": saved,
        "visualization_export": {
            "visualize_all": bool(visualize_all),
            "export_multimodal_overview": bool(export_multimodal_overview),
            "num_diagnostic_pngs": len(saved),
            "num_multimodal_overviews": int(overview_count),
            "num_mask_export_pngs": int(mask_export_count),
            "num_restored_mask_export_pngs": int(restored_mask_export_count),
            "visualization_dir": str(visual_dir) if visual_dir is not None else None,
            "visualization_manifest_path": str(manifest_path) if manifest_path is not None else None,
            "multimodal_overview_dir": str(overview_dir) if overview_dir is not None else None,
            "mask_export_dir": str(mask_export_dir) if mask_export_dir is not None else None,
            "restored_mask_export_dir": str(restored_mask_export_dir) if restored_mask_export_dir is not None else None,
        },
    }


def save_checkpoint(
    path: Path,
    model: MultiSourceQwenPSALMSeg,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: QPSalmConfig,
    update_last: bool = True,
    include_optimizer: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        key: value
        for key, value in model.state_dict().items()
        if not any(key.startswith(prefix) for prefix in EXCLUDED_FROZEN_PREFIXES)
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "model_state": model_state,
        "excluded_frozen_prefixes": list(EXCLUDED_FROZEN_PREFIXES),
        "config": config.__dict__,
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    atomic_torch_save(payload, path)
    if update_last:
        last_path = path.parent / "checkpoint_last.pt"
        if last_path != path:
            atomic_torch_save(payload, last_path)


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    prefix = "checkpoint_step_"
    if not stem.startswith(prefix):
        return -1
    try:
        return int(stem[len(prefix) :])
    except ValueError:
        return -1


def prune_step_checkpoints(out_dir: Path, keep_recent: int) -> list[str]:
    """只保留最近 N 个 checkpoint_step_*.pt；best/last 始终不受影响。"""
    if keep_recent < 0:
        return []
    step_paths = sorted(
        [path for path in out_dir.glob("checkpoint_step_*.pt") if _checkpoint_step(path) >= 0],
        key=_checkpoint_step,
    )
    remove = step_paths[: max(0, len(step_paths) - int(keep_recent))]
    removed: list[str] = []
    for path in remove:
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            continue
    return removed


def load_best_validation(path: Path) -> float:
    """断点续训时读取已有最佳验证 Dice。"""
    if not path.exists():
        return -1.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return -1.0
    value = payload.get("selection_score") if isinstance(payload, dict) else None
    if not isinstance(value, (int, float)):
        positive = (payload.get("metrics") or {}).get("positive_only") if isinstance(payload, dict) else None
        value = positive.get("dice") if isinstance(positive, dict) else None
    return float(value) if isinstance(value, (int, float)) else -1.0


def validation_selection_score(report: dict[str, Any], metric_name: str) -> float:
    """从拆分指标选择 checkpoint，避免 negative 样本掩盖前景退化。"""
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    overall = metrics.get("overall") if isinstance(metrics.get("overall"), dict) else {}
    positive = metrics.get("positive_only") if isinstance(metrics.get("positive_only"), dict) else {}
    if metric_name == "overall_dice":
        return float(overall.get("dice", 0.0))
    if metric_name != "positive_only_dice":
        raise ValueError(f"未知 checkpoint_metric={metric_name!r}; expected positive_only_dice or overall_dice")
    return float(positive.get("dice", overall.get("dice", 0.0)))


def load_checkpoint(path: str | Path, model: MultiSourceQwenPSALMSeg, optimizer: torch.optim.Optimizer | None = None) -> int:
    ckpt_path = Path(path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"无法读取 checkpoint: {ckpt_path}. "
            "文件可能是不完整写入或格式损坏；请改用 checkpoint_best.pt、checkpoint_step_*.pt，"
            "或删除损坏文件后重跑。"
        ) from exc
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 checkpoint 格式: {ckpt.get('format')!r}; expected {CHECKPOINT_FORMAT}. "
            "本次架构重构不兼容旧 checkpoint，请重新训练。"
        )
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    prefixes = tuple(str(item) for item in ckpt.get("excluded_frozen_prefixes") or [])
    illegal_missing = [key for key in incompatible.missing_keys if not any(key.startswith(prefix) for prefix in prefixes)]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint 与当前 SANE/QMEF/PMRD 架构不一致: "
            f"missing={illegal_missing[:8]} unexpected={incompatible.unexpected_keys[:8]}"
        )
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("step", 0))


def load_existing_history(path: Path, start_step: int) -> list[dict[str, Any]]:
    """断点续训时保留 checkpoint 前的历史日志。"""
    if start_step <= 0 or not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    history: list[dict[str, Any]] = []
    for row in loaded:
        if not isinstance(row, dict):
            continue
        try:
            row_step = int(row.get("step", -1))
        except (TypeError, ValueError):
            continue
        if row_step < start_step:
            history.append(row)
    return history


def dataset_combo_report(dataset: Any) -> dict[str, Any]:
    """汇总实际可训练 rows 的模态与传感器组合。"""
    rows = list(getattr(dataset, "rows", []) or [])
    canonical = Counter(canonical_modality_combo(row) for row in rows)
    raw = Counter(raw_modality_combo(row) for row in rows)
    sensors = Counter(sensor_combo(row) for row in rows)
    normalizations = Counter(normalization_combo(row) for row in rows)
    return {
        "num_rows": len(rows),
        "canonical_combos": dict(sorted(canonical.items())),
        "raw_combos": dict(sorted(raw.items())),
        "sensor_combos": dict(sorted(sensors.items())),
        "normalization_combos": dict(sorted(normalizations.items())),
    }


def train(config: QPSalmConfig, device_name: str, resume: str | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(device_name)
    out_dir = resolve_repo_path(config.output_dir) or Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_report = assert_qwen_cache_coverage(config, splits=("train", "val"))
    if cache_report.get("ok"):
        print(
            "qwen_cache_coverage="
            f"required_texts={cache_report['required']['num_texts']} "
            f"cached_texts={cache_report['cache']['num_texts']} "
            f"backend={cache_report['cache'].get('backend')}"
        )

    train_loader, val_loader = build_dataloaders(config)
    if len(train_loader) == 0:
        raise RuntimeError("训练集为空：核心模板过滤后没有可用样本。")
    grad_accum_steps = max(1, int(getattr(config, "grad_accum_steps", 1) or 1))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    if config.max_steps is None or int(config.max_steps) <= 0:
        if config.num_epochs is None or int(config.num_epochs) <= 0:
            raise ValueError("max_steps 为空时必须设置正整数 num_epochs")
        config.max_steps = int(config.num_epochs) * steps_per_epoch
    config.max_steps = int(config.max_steps)
    save_config(out_dir / "resolved_config.yaml", config)
    write_standalone_train_manifest(out_dir, config, device_name=device_name, resume=resume)
    print(
        f"dataset train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"batch_size={config.batch_size} grad_accum_steps={grad_accum_steps} "
        f"target_size={config.target_size} steps_per_epoch={steps_per_epoch} "
        f"max_steps={config.max_steps} estimated_epochs={config.max_steps / steps_per_epoch:.2f}"
    )
    train_combo_report = dataset_combo_report(train_loader.dataset)
    val_combo_report = dataset_combo_report(val_loader.dataset)
    print(
        "dataset_combos "
        f"train={train_combo_report['canonical_combos']} "
        f"val={val_combo_report['canonical_combos']}"
    )
    print(
        "dataset_sensor_normalization "
        f"train_sensors={train_combo_report['sensor_combos']} "
        f"train_norms={train_combo_report['normalization_combos']}"
    )
    model = build_model(config, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    start_step = load_checkpoint(resume, model, optimizer) if resume else 0

    scaler_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    history_path = out_dir / "train_history.json"
    history: list[dict[str, Any]] = load_existing_history(history_path, start_step) if resume else []
    best_validation_path = out_dir / "validation_best.json"
    best_selection_score = load_best_validation(best_validation_path) if resume else -1.0
    step = start_step
    train_iter = iter(train_loader)
    log_interval = int(getattr(config, "log_interval", 20) or 0)
    log_window: list[dict[str, Any]] = []
    log_window_start = time.perf_counter()
    pbar = tqdm(total=max(0, config.max_steps - start_step), desc="qpsalm-train", dynamic_ncols=True)
    while step < config.max_steps:
        model.train()
        lr_mult = cosine_lr(step, config.max_steps, config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_mult
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        step_metrics: list[dict[str, float]] = []
        step_loss_logs: list[dict[str, float]] = []
        step_signal_logs: list[dict[str, float]] = []
        outputs: dict[str, torch.Tensor] | None = None
        batch: dict[str, Any] | None = None
        for _micro_step in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=scaler_enabled):
                outputs = model(batch)
                loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().cpu().item())}")
            accumulated_loss += float(loss.detach().cpu().item())
            step_metrics.extend(
                batch_binary_metrics(
                    outputs["final_mask_logits"].detach().cpu(),
                    batch["mask"].detach().cpu(),
                    threshold=float(config.eval_threshold),
                    valid_mask=batch["valid_mask"].detach().cpu(),
                )
            )
            step_loss_logs.append(loss_log_values(outputs))
            step_signal_logs.append(training_signal_values(outputs))
            (loss / float(grad_accum_steps)).backward()
        assert outputs is not None and batch is not None
        if config.grad_clip and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()

        mean_dice = sum(item["dice"] for item in step_metrics) / max(1, len(step_metrics))
        mean_iou = sum(item["iou"] for item in step_metrics) / max(1, len(step_metrics))
        row = {
            "step": step,
            "loss": accumulated_loss / float(grad_accum_steps),
            "lr": config.lr * lr_mult,
            "dice": mean_dice,
            "iou": mean_iou,
            "grad_accum_steps": float(grad_accum_steps),
        }
        row.update(average_dicts(step_loss_logs))
        row.update(average_dicts(step_signal_logs))
        history.append(row)
        log_window.append(row)
        outputs = None
        batch = None
        loss = None  # type: ignore[assignment]

        step += 1
        pbar.update(1)
        should_log = (
            log_interval > 0
            and (
                step == start_step + 1
                or step % log_interval == 0
                or step == config.max_steps
            )
        )
        if should_log:
            elapsed = time.perf_counter() - log_window_start
            summary = summarize_train_window(log_window, elapsed)
            pbar.set_postfix(
                {
                    "loss": f"{summary.get('loss', 0.0):.3f}",
                    "iou": f"{summary.get('iou', 0.0):.3f}",
                    "dice": f"{summary.get('dice', 0.0):.3f}",
                    "lr": f"{summary.get('lr', 0.0):.1e}",
                },
                refresh=False,
            )
            window_start_step = int(log_window[0]["step"])
            window_end_step = int(log_window[-1]["step"])
            tqdm.write(format_train_window(window_start_step, window_end_step, len(log_window), summary))
            log_window = []
            log_window_start = time.perf_counter()

        should_validate = step % config.val_interval == 0 or step == config.max_steps
        if should_validate:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            should_visualize = (
                config.num_visualizations > 0
                and (step % max(1, int(config.visualize_interval)) == 0 or step == config.max_steps)
            )
            visual_dir = out_dir / "visualizations" / f"step_{step:06d}" if should_visualize else None
            if config.max_val_batches is None or config.max_val_batches <= 0:
                max_val_batches = None
            else:
                max_val_batches = max(1, int(config.max_val_batches))
            val_report = evaluate(
                model,
                val_loader,
                device,
                max_batches=max_val_batches,
                visual_dir=visual_dir,
                num_visualizations=config.num_visualizations if should_visualize else 0,
                threshold=float(config.eval_threshold),
                threshold_sweep=config.threshold_sweep,
            )
            val_report["step"] = step
            overall = (val_report.get("metrics") or {}).get("overall") or {}
            checkpoint_metric = str(getattr(config, "checkpoint_metric", "positive_only_dice"))
            selection_score = validation_selection_score(val_report, checkpoint_metric)
            is_best = selection_score > best_selection_score
            if is_best:
                best_selection_score = selection_score
            val_report["selection_metric"] = checkpoint_metric
            val_report["selection_score"] = selection_score
            val_report["is_best"] = bool(is_best)
            val_report["best_so_far"] = {"metric": checkpoint_metric, "score": best_selection_score}
            tqdm.write(
                "val "
                f"step={step} "
                f"iou={float(overall.get('iou', 0.0)):.4f} "
                f"dice={float(overall.get('dice', 0.0)):.4f} "
                f"precision={float(overall.get('precision', 0.0)):.4f} "
                f"recall={float(overall.get('recall', 0.0)):.4f} "
                f"select={checkpoint_metric}:{selection_score:.4f} "
                f"best={best_selection_score:.4f} "
                f"n={int((val_report.get('coverage') or {}).get('num_samples') or overall.get('n') or 0)} "
                f"combos={len((val_report.get('coverage') or {}).get('canonical_combos') or {})} "
                f"visualize={int(config.num_visualizations if should_visualize else 0)}"
            )
            coverage = val_report.get("coverage") if isinstance(val_report.get("coverage"), dict) else {}
            if max_val_batches is not None or len(coverage.get("canonical_combos") or {}) <= 1:
                tqdm.write(
                    "val_coverage_warning "
                    f"max_val_batches={max_val_batches} "
                    f"canonical_combos={coverage.get('canonical_combos') or {}}"
                )
            if bool(getattr(config, "save_step_validation_reports", False)):
                (out_dir / f"validation_step_{step:06d}.json").write_text(
                    json.dumps(val_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            (out_dir / "validation_latest.json").write_text(
                json.dumps(val_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if is_best:
                best_validation_path.write_text(
                    json.dumps(val_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                save_checkpoint(
                    out_dir / "checkpoint_best.pt",
                    model,
                    optimizer,
                    step,
                    config,
                    update_last=False,
                    include_optimizer=False,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        save_interval = int(getattr(config, "save_interval", 0) or 0)
        if (save_interval > 0 and step % save_interval == 0) or step == config.max_steps:
            save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, step, config, update_last=False)
            if bool(getattr(config, "save_step_checkpoints", False)):
                save_checkpoint(out_dir / f"checkpoint_step_{step:06d}.pt", model, optimizer, step, config, update_last=False)
                removed_checkpoints = prune_step_checkpoints(
                    out_dir,
                    keep_recent=int(getattr(config, "keep_recent_checkpoints", 2)),
                )
                if removed_checkpoints:
                    tqdm.write(
                        "checkpoint_prune "
                        f"keep_recent={int(getattr(config, 'keep_recent_checkpoints', 2))} "
                        f"removed={len(removed_checkpoints)}"
                    )

    pbar.close()
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": out_dir.as_posix(), "steps": step, "history": history[-5:]}
