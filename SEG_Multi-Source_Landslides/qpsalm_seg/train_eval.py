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
    CANONICAL_MODALITIES,
    MultiSourceLandslideDataset,
    canonical_modality_combo,
    normalization_combo,
    qpsalm_collate,
    raw_modality_combo,
    resolve_repo_path,
    sensor_combo,
)
from .losses import dice_scores_with_logits
from .metrics import MetricAccumulator, batch_binary_metrics
from .model import MultiSourceQwenPSALMSeg
from .qwen_cache import assert_qwen_cache_coverage
from .visualize import save_visualizations


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
    """直接调用 qpsalm-train 时补充 run_manifest；run_phase1 已写过则保留。"""
    path = out_dir / "run_manifest.json"
    if path.exists():
        return
    write_json(
        path,
        {
            "created_at_utc": utc_now(),
            "created_by": "qpsalm-train",
            "mode": "standalone",
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
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=qpsalm_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=qpsalm_collate,
        pin_memory=torch.cuda.is_available(),
    )
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
    "loss_mask_tversky",
    "loss_proposal_cls",
    "loss_condition_cls",
    "loss_condition_rank",
    "loss_selection_rank",
    "loss_proposal_mask",
    "loss_empty_mask",
    "loss_empty_proposal",
    "loss_query_diversity",
    "loss_proposal_mask_diversity",
    "loss_gate_entropy",
    "loss_query_usage_balance",
    "loss_boundary",
    "sample_weight_normalized_mean",
    "sample_weight_normalized_min",
    "sample_weight_normalized_max",
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
    if "condition_rank_acc" in outputs:
        values["condition_rank_acc"] = scalar_tensor(outputs["condition_rank_acc"])
    if "selection_rank_acc" in outputs:
        values["selection_rank_acc"] = scalar_tensor(outputs["selection_rank_acc"])
    if "proposal_target_mass" in outputs:
        values["proposal_target_mass"] = scalar_tensor(outputs["proposal_target_mass"])
    if "proposal_target_positive_count" in outputs:
        values["proposal_target_positive_count"] = scalar_tensor(outputs["proposal_target_positive_count"])
    if "query_usage_entropy" in outputs:
        values["query_usage_entropy"] = scalar_tensor(outputs["query_usage_entropy"])
    return values


def training_signal_values(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    """记录轻量 proposal/gate 诊断，便于不看图时追踪训练动态。"""
    values: dict[str, float] = {}
    if "modality_gate_weights" in outputs:
        gate = outputs["modality_gate_weights"].detach().float().cpu()
        for idx, name in enumerate(CANONICAL_MODALITIES):
            values[f"gate_{name}"] = float(gate[:, idx].mean().item())
        gate_safe = gate.clamp_min(1.0e-8)
        values["gate_entropy"] = float((-(gate_safe * gate_safe.log()).sum(dim=1)).mean().item())
    for scale_name in ("high", "mid", "low"):
        key = f"scale_gate_{scale_name}"
        if key in outputs:
            gate = outputs[key].detach().float().cpu()
            for idx, name in enumerate(CANONICAL_MODALITIES):
                values[f"{key}_{name}"] = float(gate[:, idx].mean().item())
            gate_safe = gate.clamp_min(1.0e-8)
            values[f"{key}_entropy"] = float((-(gate_safe * gate_safe.log()).sum(dim=1)).mean().item())
    if "modality_active_mask" in outputs:
        active = outputs["modality_active_mask"].detach().float().cpu()
        values["active_modality_count"] = float(active.sum(dim=1).mean().item())
    if "modality_feature_norms" in outputs:
        norms = outputs["modality_feature_norms"].detach().float().cpu()
        for idx, name in enumerate(CANONICAL_MODALITIES):
            values[f"featnorm_{name}"] = float(norms[:, idx].mean().item())
    if "modality_gate_feature_norms" in outputs:
        norms = outputs["modality_gate_feature_norms"].detach().float().cpu()
        for idx, name in enumerate(CANONICAL_MODALITIES):
            values[f"gate_featnorm_{name}"] = float(norms[:, idx].mean().item())
    if "proposal_logits" in outputs:
        proposal_prob = torch.softmax(outputs["proposal_logits"].detach().float().cpu(), dim=-1)[..., 1]
        values["proposal_fg_prob_mean"] = float(proposal_prob.mean().item())
        values["proposal_fg_prob_max"] = float(proposal_prob.max().item())
    if "proposal_fg_logits" in outputs:
        fg_logits = outputs["proposal_fg_logits"].detach().float().cpu()
        values["proposal_fg_logit_mean"] = float(fg_logits.mean().item())
        values["proposal_fg_logit_max"] = float(fg_logits.max().item())
    if "condition_scores" in outputs:
        scores = outputs["condition_scores"].detach().float().cpu()
        values["condition_score_mean"] = float(scores.mean().item())
        values["condition_score_max"] = float(scores.max().item())
    if "condition_cosine_scores" in outputs:
        scores = outputs["condition_cosine_scores"].detach().float().cpu()
        values["condition_cosine_mean"] = float(scores.mean().item())
        values["condition_cosine_max"] = float(scores.max().item())
    if "condition_pair_logits" in outputs:
        scores = outputs["condition_pair_logits"].detach().float().cpu()
        values["condition_pair_logit_mean"] = float(scores.mean().item())
        values["condition_pair_logit_max"] = float(scores.max().item())
    if "condition_logit_scale" in outputs:
        values["condition_logit_scale"] = scalar_tensor(outputs["condition_logit_scale"])
    if "selection_logits" in outputs:
        scores = outputs["selection_logits"].detach().float().cpu()
        values["selection_logit_mean"] = float(scores.mean().item())
        values["selection_logit_max"] = float(scores.max().item())
        values["top_query_mean"] = float(torch.argmax(scores, dim=1).float().mean().item())
        values["top_query_score_mean"] = float(torch.max(scores, dim=1).values.mean().item())
    if "selection_weights" in outputs:
        weights = outputs["selection_weights"].detach().float().cpu().clamp_min(1.0e-8)
        values["selection_entropy"] = float((-(weights * weights.log()).sum(dim=1)).mean().item())
    if "foreground_gate_logits" in outputs:
        gate_logits = outputs["foreground_gate_logits"].detach().float().cpu()
        values["foreground_gate_logit_mean"] = float(gate_logits.mean().item())
        values["foreground_gate_logit_max"] = float(gate_logits.max().item())
    elif "proposal_logits" in outputs and "condition_scores" in outputs:
        proposal_prob = torch.softmax(outputs["proposal_logits"].detach().float().cpu(), dim=-1)[..., 1]
        scores = proposal_prob + outputs["condition_scores"].detach().float().cpu()
        values["top_query_mean"] = float(torch.argmax(scores, dim=1).float().mean().item())
        values["top_query_score_mean"] = float(torch.max(scores, dim=1).values.mean().item())
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
    "condition_rank_acc",
    "selection_rank_acc",
    "proposal_target_positive_count",
    "query_usage_entropy",
    "proposal_fg_prob_max",
    "top_query_mean",
    "gate_entropy",
    "active_modality_count",
    "sample_weight_raw_mean",
    "sample_weight_raw_max",
    "sample_weight_normalized_max",
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
        ("rank_acc", "condition_rank_acc", ".3f"),
        ("sel_acc", "selection_rank_acc", ".3f"),
        ("posQ", "proposal_target_positive_count", ".1f"),
        ("qUseH", "query_usage_entropy", ".2f"),
        ("top_q", "top_query_mean", ".1f"),
        ("fg_max", "proposal_fg_prob_max", ".3f"),
        ("gateH", "gate_entropy", ".2f"),
        ("activeM", "active_modality_count", ".1f"),
        ("wRaw", "sample_weight_raw_mean", ".2f"),
        ("wMax", "sample_weight_raw_max", ".2f"),
        ("wNormMax", "sample_weight_normalized_max", ".2f"),
    ]
    for label, key, fmt in optional:
        if key in summary:
            parts.append(f"{label}={summary[key]:{fmt}}")
    return "train " + " ".join(parts)


def _gate_row_to_dict(values: torch.Tensor) -> dict[str, float]:
    """把单样本模态 gate 张量转成稳定 JSON 字段。"""
    return {
        name: float(values[idx].item())
        for idx, name in enumerate(CANONICAL_MODALITIES)
    }


def collect_gate_records(outputs: dict[str, torch.Tensor], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """收集验证/eval 阶段的 condition-aware modality gate。"""
    if "modality_gate_weights" not in outputs:
        return []
    weights = outputs["modality_gate_weights"].detach().float().cpu()
    active = outputs.get("modality_active_mask")
    active_cpu = active.detach().float().cpu() if active is not None else None
    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        record = {
            "sample_id": meta.get("sample_id", ""),
            "canonical_combo": meta.get("canonical_combo", "unknown"),
            "raw_combo": meta.get("raw_combo", "unknown"),
            "sensor_combo": meta.get("sensor_combo", "unknown"),
            "normalization_combo": meta.get("normalization_combo", "unknown"),
            "condition_prompt": meta.get("condition_prompt", "unknown"),
            "weights": _gate_row_to_dict(weights[idx]),
        }
        if active_cpu is not None:
            record["active"] = _gate_row_to_dict(active_cpu[idx])
        records.append(record)
    return records


def _average_named_values(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    """对 records 中 weights/active 这类 name->float 字典求均值。"""
    out: dict[str, float] = {}
    for name in CANONICAL_MODALITIES:
        values = [
            float((record.get(field) or {}).get(name, 0.0))
            for record in records
        ]
        out[name] = sum(values) / len(values) if values else 0.0
    return out


def compute_gate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按 overall/combo/condition 聚合模态 gate，便于分析多源证据使用。"""
    if not records:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {"overall": records}
    for record in records:
        groups.setdefault(f"canonical_combo={record.get('canonical_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"raw_combo={record.get('raw_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"sensor_combo={record.get('sensor_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"normalization_combo={record.get('normalization_combo', 'unknown')}", []).append(record)
        groups.setdefault(f"condition={record.get('condition_prompt', 'unknown')}", []).append(record)
    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        summary[name] = {
            "n": len(rows),
            "mean_weights": _average_named_values(rows, "weights"),
            "mean_active": _average_named_values(rows, "active"),
        }
    return summary


def _optional_matrix(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    value = outputs.get(key)
    if value is None:
        return None
    return value.detach().float().cpu()


def _matrix_value(matrix: torch.Tensor | None, row: int, col: int, default: float | None = None) -> float | None:
    if matrix is None:
        return default
    return float(matrix[row, col].item())


def collect_proposal_records(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    metadata: list[dict[str, Any]],
    sample_metrics: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """记录样本级 proposal 选择诊断，便于分析 condition scorer 是否选对 query。"""
    if "pred_masks" not in outputs or "proposal_logits" not in outputs:
        return []
    pred_masks = outputs["pred_masks"].detach().float().cpu()
    target = target_mask.detach().float().cpu()
    dice_scores = dice_scores_with_logits(pred_masks, target)
    best_query = outputs.get("best_query")
    best_query_cpu = best_query.detach().long().cpu() if best_query is not None else dice_scores.argmax(dim=1)
    proposal_prob = torch.softmax(outputs["proposal_logits"].detach().float().cpu(), dim=-1)[..., 1]
    selection_logits = _optional_matrix(outputs, "selection_logits")
    if selection_logits is None:
        condition_scores = _optional_matrix(outputs, "condition_scores")
        selection_logits = proposal_prob + (condition_scores if condition_scores is not None else 0.0)
    selected_query = torch.argmax(selection_logits, dim=1)
    condition_scores = _optional_matrix(outputs, "condition_scores")
    condition_cosine = _optional_matrix(outputs, "condition_cosine_scores")
    condition_pair = _optional_matrix(outputs, "condition_pair_logits")
    final_probs = torch.sigmoid(outputs["final_mask_logits"].detach().float().cpu())
    proposal_mask_probs = torch.sigmoid(pred_masks)

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
            "best_query": best,
            "selected_query": selected,
            "selected_matches_best": float(selected == best),
            "best_query_dice": float(dice_scores[idx, best].item()),
            "selected_query_dice": float(dice_scores[idx, selected].item()),
            "selected_selection_logit": float(selection_logits[idx, selected].item()),
            "best_selection_logit": float(selection_logits[idx, best].item()),
            "selected_proposal_fg_prob": float(proposal_prob[idx, selected].item()),
            "best_proposal_fg_prob": float(proposal_prob[idx, best].item()),
            "selected_condition_score": _matrix_value(condition_scores, idx, selected),
            "best_condition_score": _matrix_value(condition_scores, idx, best),
            "selected_condition_cosine": _matrix_value(condition_cosine, idx, selected),
            "best_condition_cosine": _matrix_value(condition_cosine, idx, best),
            "selected_condition_pair_logit": _matrix_value(condition_pair, idx, selected),
            "best_condition_pair_logit": _matrix_value(condition_pair, idx, best),
            "final_dice": metrics.get("dice"),
            "final_iou": metrics.get("iou"),
            "final_precision": metrics.get("precision"),
            "final_recall": metrics.get("recall"),
            "target_area": float((target[idx, 0] >= 0.5).sum().item()),
            "final_mask_area": float((final_probs[idx, 0] >= 0.5).sum().item()),
            "selected_mask_area": float((proposal_mask_probs[idx, selected] >= 0.5).sum().item()),
            "best_mask_area": float((proposal_mask_probs[idx, best] >= 0.5).sum().item()),
        }
        record["selection_logit_gap_selected_minus_best"] = (
            record["selected_selection_logit"] - record["best_selection_logit"]
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
        "selected_selection_logit",
        "best_selection_logit",
        "selection_logit_gap_selected_minus_best",
        "dice_gap_selected_minus_best",
        "selected_proposal_fg_prob",
        "best_proposal_fg_prob",
        "selected_condition_score",
        "best_condition_score",
        "final_dice",
        "final_iou",
        "final_precision",
        "final_recall",
        "target_area",
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


@torch.no_grad()
def evaluate(
    model: MultiSourceQwenPSALMSeg,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    visual_dir: Path | None = None,
    num_visualizations: int = 0,
    threshold: float = 0.5,
    threshold_sweep: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator()
    sweep_thresholds = normalize_thresholds(threshold_sweep)
    sweep_accumulators = {value: MetricAccumulator() for value in sweep_thresholds}
    loss_values: list[float] = []
    loss_components: list[dict[str, float]] = []
    gate_records: list[dict[str, Any]] = []
    proposal_records: list[dict[str, Any]] = []
    saved: list[str] = []
    processed_batches = 0
    processed_samples = 0
    canonical_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    iterator = enumerate(loader)
    for batch_idx, batch in iterator:
        if max_batches is not None and batch_idx >= max_batches:
            break
        processed_batches += 1
        processed_samples += len(batch["metadata"])
        for meta in batch["metadata"]:
            canonical_counts[str(meta.get("canonical_combo", "unknown"))] += 1
            raw_counts[str(meta.get("raw_combo", "unknown"))] += 1
            sensor_counts[str(meta.get("sensor_combo", "unknown"))] += 1
            normalization_counts[str(meta.get("normalization_combo", "unknown"))] += 1
        outputs = model(batch)
        if "loss" in outputs:
            loss_values.append(float(outputs["loss"].detach().cpu().item()))
            loss_components.append(loss_log_values(outputs))
        logits_cpu = outputs["final_mask_logits"].detach().cpu()
        mask_cpu = batch["mask"].detach().cpu()
        metrics = batch_binary_metrics(logits_cpu, mask_cpu, threshold=float(threshold))
        acc.update(metrics, batch["metadata"])
        for sweep_value, sweep_acc in sweep_accumulators.items():
            sweep_metrics = batch_binary_metrics(logits_cpu, mask_cpu, threshold=sweep_value)
            sweep_acc.update(sweep_metrics, batch["metadata"])
        gate_records.extend(collect_gate_records(outputs, batch["metadata"]))
        proposal_records.extend(collect_proposal_records(outputs, batch["mask"], batch["metadata"], metrics))
        if visual_dir is not None and len(saved) < num_visualizations:
            saved.extend(
                save_visualizations(
                    batch,
                    outputs,
                    visual_dir,
                    max_items=num_visualizations - len(saved),
                    prefix=f"val_b{batch_idx}",
                    threshold=float(threshold),
                )
            )
    groups = acc.compute()
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
            "max_batches": max_batches,
        },
        "threshold_sweep": compute_threshold_sweep_report(sweep_accumulators),
        "metrics": groups,
        "modality_gate_summary": compute_gate_summary(gate_records),
        "proposal_diagnostics": {
            "records": proposal_records,
            "summary": compute_proposal_summary(proposal_records),
        },
        "visualizations": saved,
    }


def save_checkpoint(
    path: Path,
    model: MultiSourceQwenPSALMSeg,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: QPSalmConfig,
    update_last: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
    }
    atomic_torch_save(payload, path)
    if update_last:
        last_path = path.parent / "checkpoint_last.pt"
        if last_path != path:
            atomic_torch_save(payload, last_path)


def load_best_validation(path: Path) -> float:
    """断点续训时读取已有最佳验证 Dice。"""
    if not path.exists():
        return -1.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return -1.0
    overall = (payload.get("metrics") or {}).get("overall") if isinstance(payload, dict) else None
    value = overall.get("dice") if isinstance(overall, dict) else None
    return float(value) if isinstance(value, (int, float)) else -1.0


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
    model.load_state_dict(ckpt["model_state"], strict=True)
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


def dataset_combo_report(dataset: Any, config: QPSalmConfig) -> dict[str, Any]:
    """汇总实际可训练 rows 的组合和生效 loss 权重。"""
    rows = list(getattr(dataset, "rows", []) or [])
    canonical = Counter(canonical_modality_combo(row) for row in rows)
    raw = Counter(raw_modality_combo(row) for row in rows)
    sensors = Counter(sensor_combo(row) for row in rows)
    normalizations = Counter(normalization_combo(row) for row in rows)
    configured = dict(getattr(config, "canonical_combo_loss_weights", {}) or {})
    effective = {
        combo: float(configured.get(combo, configured.get(f"canonical_combo={combo}", 1.0)))
        for combo in sorted(canonical)
        if float(configured.get(combo, configured.get(f"canonical_combo={combo}", 1.0))) != 1.0
    }
    present_keys = set(canonical) | {f"canonical_combo={combo}" for combo in canonical}
    ignored = {
        key: value
        for key, value in sorted(configured.items())
        if key not in present_keys
    }
    return {
        "num_rows": len(rows),
        "canonical_combos": dict(sorted(canonical.items())),
        "raw_combos": dict(sorted(raw.items())),
        "sensor_combos": dict(sorted(sensors.items())),
        "normalization_combos": dict(sorted(normalizations.items())),
        "effective_canonical_combo_loss_weights": effective,
        "ignored_canonical_combo_loss_weights": ignored,
    }


def train(config: QPSalmConfig, device_name: str, resume: str | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(device_name)
    out_dir = resolve_repo_path(config.output_dir) or Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(out_dir / "resolved_config.yaml", config)
    write_standalone_train_manifest(out_dir, config, device_name=device_name, resume=resume)
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
    print(
        f"dataset train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"batch_size={config.batch_size} grad_accum_steps={max(1, int(config.grad_accum_steps))} "
        f"target_size={config.target_size}"
    )
    train_combo_report = dataset_combo_report(train_loader.dataset, config)
    val_combo_report = dataset_combo_report(val_loader.dataset, config)
    print(
        "dataset_combos "
        f"train={train_combo_report['canonical_combos']} "
        f"val={val_combo_report['canonical_combos']} "
        f"effective_weights={train_combo_report['effective_canonical_combo_loss_weights']} "
        f"ignored_weights={train_combo_report['ignored_canonical_combo_loss_weights']}"
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
    best_val_dice = load_best_validation(best_validation_path) if resume else -1.0
    step = start_step
    train_iter = iter(train_loader)
    log_interval = int(getattr(config, "log_interval", 20) or 0)
    log_window: list[dict[str, Any]] = []
    log_window_start = time.perf_counter()
    grad_accum_steps = max(1, int(getattr(config, "grad_accum_steps", 1) or 1))
    pbar = tqdm(total=max(0, config.max_steps - start_step), desc="qpsalm-train", dynamic_ncols=True)
    while step < config.max_steps:
        model.train()
        lr_mult = cosine_lr(step, config.max_steps, config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_mult
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
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
            (loss / float(grad_accum_steps)).backward()
        assert outputs is not None and batch is not None
        if config.grad_clip and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()

        metrics = batch_binary_metrics(
            outputs["final_mask_logits"].detach().cpu(),
            batch["mask"].detach().cpu(),
            threshold=float(config.eval_threshold),
        )
        row = {
            "step": step,
            "loss": accumulated_loss / float(grad_accum_steps),
            "lr": config.lr * lr_mult,
            "dice": metrics[0]["dice"],
            "iou": metrics[0]["iou"],
            "grad_accum_steps": float(grad_accum_steps),
        }
        if "sample_weight" in batch:
            raw_weights = batch["sample_weight"].detach().float().cpu()
            row["sample_weight_raw_mean"] = float(raw_weights.mean().item())
            row["sample_weight_raw_min"] = float(raw_weights.min().item())
            row["sample_weight_raw_max"] = float(raw_weights.max().item())
        row.update(loss_log_values(outputs))
        row.update(training_signal_values(outputs))
        history.append(row)
        log_window.append(row)

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

        if step % config.val_interval == 0 or step == config.max_steps:
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
            val_dice = float(overall.get("dice", 0.0))
            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
            val_report["is_best"] = bool(is_best)
            val_report["best_so_far"] = {"dice": best_val_dice}
            tqdm.write(
                "val "
                f"step={step} "
                f"iou={float(overall.get('iou', 0.0)):.4f} "
                f"dice={float(overall.get('dice', 0.0)):.4f} "
                f"precision={float(overall.get('precision', 0.0)):.4f} "
                f"recall={float(overall.get('recall', 0.0)):.4f} "
                f"best={best_val_dice:.4f} "
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
                save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, step, config, update_last=False)
        if step % config.save_interval == 0 or step == config.max_steps:
            save_checkpoint(out_dir / f"checkpoint_step_{step:06d}.pt", model, optimizer, step, config)

    pbar.close()
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": out_dir.as_posix(), "steps": step, "history": history[-5:]}
