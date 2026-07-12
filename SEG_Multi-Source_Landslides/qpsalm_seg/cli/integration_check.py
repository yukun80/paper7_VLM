#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run strict real-benchmark integration checks for SANE/QMEF/PMRD and Qwen.

用途：在真实 benchmark-v2 上验证 raw forward/backward，以及可选的 Qwen QLoRA BF16 单步。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.integration_check --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml
--mode all --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3
--device cuda --output outputs/qpsalm_v2/real_integration_report.json
主要输入：完整 small-v2 instruction train/val/test、可选 Qwen vision cache v3 和本地 Qwen 权重。
主要输出：包含样本、loss、梯度、cache subset、显存和验收状态的 JSON 报告。
写入行为：只写 --output，不保存 checkpoint，不修改 benchmark/cache。
所属流程：small-v2 正式三 seed 实验之前的真实数据与单卡验收门槛。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from qpsalm_seg.config import load_config
from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.engine.common import build_model, resolve_device, set_seed
from qpsalm_seg.metrics import batch_binary_metrics
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset


REPORT_FORMAT = "qpsalm_real_integration_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict real benchmark-v2 integration check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--mode", choices=("raw", "qwen", "all"), default="all")
    parser.add_argument("--raw-preset", choices=PRESET_CHOICES, default="raw_sane_qmef_pmrd")
    parser.add_argument("--qwen-preset", choices=PRESET_CHOICES, default="qwen_psalm_full")
    parser.add_argument("--vision-feature-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory-gib", type=float, default=23.0)
    parser.add_argument("--output", default="outputs/qpsalm_v2/real_integration_report.json")
    return parser.parse_args()


def _task_group(row: dict[str, Any]) -> str:
    family = str(row.get("task_family") or "")
    if family == "referring_landslide_segmentation":
        return "referring"
    if family == "no_target_segmentation":
        return "no_target"
    return "global"


def _is_multimodal(row: dict[str, Any]) -> bool:
    available = [
        value for value in (row.get("modalities") or {}).values()
        if isinstance(value, dict) and value.get("available", True)
    ]
    return len(available) > 1


def select_real_indices(dataset: MultiSourceLandslideDataset) -> dict[str, int]:
    """Select auditable global/referring/no-target rows, preferring multimodal evidence."""
    candidates: dict[str, list[int]] = {"global": [], "referring": [], "no_target": []}
    for index, row in enumerate(dataset.rows):
        candidates[_task_group(row)].append(index)
    missing = [name for name, values in candidates.items() if not values]
    if missing:
        raise RuntimeError(f"真实 integration split 缺少任务组: {missing}")
    selected = {}
    for name, values in candidates.items():
        selected[name] = next((index for index in values if _is_multimodal(dataset.rows[index])), values[0])
    return selected


def _sample_record(item: dict[str, Any], output, metric: dict[str, float]) -> dict[str, Any]:
    meta = item["metadata"]
    return {
        "sample_id": meta.get("sample_id"),
        "parent_sample_id": meta.get("parent_sample_id"),
        "task_family": meta.get("task_family"),
        "active_modalities": list(meta.get("active_modalities") or []),
        "full_modalities": list(meta.get("full_modalities") or []),
        "active_subset": meta.get("active_subset"),
        "valid_coverage": meta.get("valid_coverage"),
        "loss": float(output["loss"].detach().float().cpu()),
        "iou": float(metric["iou"]),
        "dice": float(metric["dice"]),
        "component_count": float(output["proposal_component_count"].detach().float().mean().cpu()),
        "visual_evidence_key": item["visual_evidence_key"],
    }


def _gradient_report(model: torch.nn.Module) -> dict[str, float]:
    values = {
        name: float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    finite = all(math.isfinite(value) for value in values.values())
    return {
        "num_parameters_with_grad": len(values),
        "gradient_norm_sum": sum(values.values()),
        "all_finite": finite,
    }


def run_raw_check(config, device: torch.device) -> dict[str, Any]:
    stable = replace(
        config,
        controller="text_probe",
        vision_feature_cache=None,
        use_pretrained_sane=False,
        modality_dropout=0.0,
        train_hflip_prob=0.0,
        train_vflip_prob=0.0,
        missing_modality_consistency_weight=0.0,
        num_workers=0,
    )
    dataset = MultiSourceLandslideDataset(stable, "train")
    indices = select_real_indices(dataset)
    model = build_model(stable, device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=stable.lr,
        weight_decay=stable.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    records = []
    for index in indices.values():
        item = dataset[index]
        batch = qpsalm_collate([item])
        output = model(batch)
        if not torch.isfinite(output["loss"]):
            raise RuntimeError(f"raw integration loss 非有限: sample={item['metadata']['sample_id']}")
        (output["loss"] / len(indices)).backward()
        metric = batch_binary_metrics(
            output["final_mask_logits"].detach().cpu(),
            batch.mask,
            valid_mask=batch.valid_mask,
        )[0]
        records.append(_sample_record(item, output, metric))
    gradients = _gradient_report(model)
    if not gradients["all_finite"] or gradients["gradient_norm_sum"] <= 0:
        raise RuntimeError(f"raw integration 梯度无效: {gradients}")
    optimizer.step()
    return {
        "status": "passed",
        "preset": stable.preset,
        "device": str(device),
        "selected_indices": indices,
        "samples": records,
        "gradients": gradients,
    }


def _select_multimodal_item(dataset: MultiSourceLandslideDataset) -> dict[str, Any]:
    for index, row in enumerate(dataset.rows):
        if _is_multimodal(row):
            return dataset[index]
    raise RuntimeError("Qwen integration 需要至少一个真实多模态样本")


def run_qwen_check(config, device: torch.device, max_memory_gib: float) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Qwen integration 必须显式使用 CUDA device")
    if config.controller != "qwen_mask_query" or not config.use_pretrained_sane:
        raise RuntimeError(
            f"Qwen integration 需要 qwen_mask_query + pretrained SANE，当前 preset={config.preset}"
        )
    strict = replace(
        config,
        modality_dropout=1.0,
        train_hflip_prob=0.0,
        train_vflip_prob=0.0,
        missing_modality_consistency_weight=0.0,
        batch_size=1,
        num_workers=0,
    )
    dataset = MultiSourceLandslideDataset(strict, "train")
    item = _select_multimodal_item(dataset)
    if item["active_subset"].is_full or not item["active_subset"].dropped_names:
        raise RuntimeError("Qwen integration 未形成真实 dropped-modality student subset")
    batch = qpsalm_collate([item])
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(strict, device).train()
    if model.vision_bank is None:
        raise RuntimeError("Qwen integration 缺少 vision feature bank")
    selected_views = model.vision_bank.selected_views_for(
        batch.visual_evidence_key[0], batch.active_subsets[0]
    )
    selected_sources = sorted({
        str(source)
        for view in selected_views
        for source in view.get("source_modalities") or []
    })
    dropped_sources = set(item["active_subset"].dropped_names) & set(selected_sources)
    if dropped_sources or set(selected_sources) - set(item["active_subset"].active_names):
        raise RuntimeError(
            f"Qwen cache subset 泄漏: selected={selected_sources} "
            f"active={item['active_subset'].active_names} dropped={item['active_subset'].dropped_names}"
        )
    _, token_mask, counts, family_ids, segments = model.vision_bank.tokens_for(
        batch.visual_evidence_key,
        batch.active_subsets,
        device,
        strict.qwen_view_tokens_per_view,
    )
    if not counts or counts[0] <= 0 or int(token_mask[0].sum()) != counts[0]:
        raise RuntimeError(f"Qwen integration active subset 没有视觉 tokens: counts={counts}")
    trainable_qwen = [
        name for name, parameter in model.controller.model.named_parameters()
        if parameter.requires_grad
    ]
    illegal_qwen = [name for name in trainable_qwen if "lora_" not in name]
    if illegal_qwen or not trainable_qwen:
        raise RuntimeError(
            f"Qwen QLoRA 参数隔离失败: trainable={trainable_qwen[:8]} illegal={illegal_qwen[:8]}"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=strict.lr, weight_decay=strict.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)
        loss = output["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("Qwen integration loss 非有限")
    loss.backward()
    gradients = _gradient_report(model)
    lora_gradient_sum = sum(
        float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in model.controller.model.named_parameters()
        if "lora_" in name and parameter.grad is not None
    )
    if not gradients["all_finite"] or gradients["gradient_norm_sum"] <= 0 or lora_gradient_sum <= 0:
        raise RuntimeError(
            f"Qwen integration 梯度无效: gradients={gradients} lora={lora_gradient_sum}"
        )
    torch.nn.utils.clip_grad_norm_(trainable, strict.grad_clip)
    optimizer.step()
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    if peak_reserved > float(max_memory_gib):
        raise RuntimeError(
            f"Qwen integration 峰值显存超过门槛: reserved={peak_reserved:.3f} GiB "
            f"limit={max_memory_gib:.3f} GiB"
        )
    metric = batch_binary_metrics(
        output["final_mask_logits"].detach().cpu(), batch.mask, valid_mask=batch.valid_mask
    )[0]
    return {
        "status": "passed",
        "preset": strict.preset,
        "device": str(device),
        "sample": _sample_record(item, output, metric),
        "active_subset": {
            "active": list(item["active_subset"].active_names),
            "dropped": list(item["active_subset"].dropped_names),
            "signature": item["active_subset"].signature,
        },
        "cache": {
            "visual_token_count": counts[0],
            "selected_source_modalities": selected_sources,
            "visual_family_ids": family_ids[0, :counts[0]].detach().cpu().tolist(),
            "view_segments": segments[0],
            "format": model.vision_bank.manifest["format"],
        },
        "qwen": {
            "num_trainable_lora_parameters": len(trainable_qwen),
            "lora_gradient_norm_sum": lora_gradient_sum,
        },
        "gradients": gradients,
        "memory": {
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "limit_gib": float(max_memory_gib),
        },
    }


def _write_report(path_ref: str, payload: dict[str, Any]) -> Path:
    path = resolve_project_path(path_ref) or Path(path_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    overrides = {"benchmark_dir": args.benchmark_dir}
    base = load_config(args.config, overrides=overrides)
    device = resolve_device(args.device)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "created_unix": time.time(),
        "benchmark_dir": str(base.benchmark_path()),
        "mode": args.mode,
        "seed": args.seed,
        "checks": {},
    }
    errors = []
    if args.mode in {"raw", "all"}:
        try:
            raw_config = apply_preset(base, args.raw_preset)
            report["checks"]["raw"] = run_raw_check(raw_config, device)
        except Exception as exc:
            report["checks"]["raw"] = {"status": "failed", "error": str(exc)}
            errors.append(f"raw: {exc}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.mode in {"qwen", "all"}:
        try:
            qwen_config = apply_preset(base, args.qwen_preset)
            cache = args.vision_feature_cache or qwen_config.vision_feature_cache
            qwen_config = replace(qwen_config, vision_feature_cache=cache)
            report["checks"]["qwen"] = run_qwen_check(qwen_config, device, args.max_memory_gib)
        except Exception as exc:
            report["checks"]["qwen"] = {"status": "failed", "error": str(exc)}
            errors.append(f"qwen: {exc}")
    report["acceptance"] = {
        "passed": not errors,
        "required_checks": ["raw", "qwen"] if args.mode == "all" else [args.mode],
        "errors": errors,
    }
    path = _write_report(args.output, report)
    print(json.dumps({"report": str(path), **report["acceptance"]}, ensure_ascii=False))
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
