#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run strict real-benchmark integration checks for SANE/QMEF/PMRD and Qwen.

用途：在真实 benchmark-v2 上验证 raw forward/backward，以及 Qwen QLoRA 动态序列连续训练。
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

from qpsalm_seg.config import AMP_DTYPES, QWEN_GRADIENT_CHECKPOINTING_MODES, load_config
from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.engine.common import (
    amp_dtype,
    autocast_enabled,
    build_model,
    create_grad_scaler,
    resolve_device,
    set_seed,
)
from qpsalm_seg.metrics import batch_binary_metrics
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset


REPORT_FORMAT = "qpsalm_real_integration_v2"
INTEGRATION_PROTOCOL_VERSION = "qwen_batch_gradient_v3"


class IntegrationFailure(RuntimeError):
    """Integration error that preserves completed diagnostics in the report."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict real benchmark-v2 integration check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--mode", choices=("raw", "qwen", "all"), default="all")
    parser.add_argument("--raw-preset", choices=PRESET_CHOICES, default="raw_sane_qmef_pmrd")
    parser.add_argument("--qwen-preset", choices=PRESET_CHOICES, default="qwen_psalm_full")
    parser.add_argument("--amp-dtype", choices=AMP_DTYPES, default=None)
    parser.add_argument("--vision-feature-cache", default=None)
    parser.add_argument(
        "--qwen-gradient-checkpointing",
        choices=QWEN_GRADIENT_CHECKPOINTING_MODES,
        default=None,
    )
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


def _row_families(row: dict[str, Any]) -> set[str]:
    return {
        str(value.get("family"))
        for value in (row.get("modalities") or {}).values()
        if isinstance(value, dict) and value.get("available", True) and value.get("family")
    }


def select_qwen_dynamic_indices(dataset: MultiSourceLandslideDataset) -> dict[str, int]:
    """Select deterministic evidence layouts that force different Qwen lengths."""
    selected: dict[str, int] = {}
    for index, row in enumerate(dataset.rows):
        families = _row_families(row)
        if "optical" not in selected and families == {"optical"}:
            selected["optical"] = index
        if (
            "multispectral" not in selected
            and "multispectral" in families
            and "deformation" not in families
        ):
            selected["multispectral"] = index
        if "multisource" not in selected and len(families) >= 3:
            selected["multisource"] = index
        if len(selected) == 3:
            break
    missing = [name for name in ("optical", "multispectral", "multisource") if name not in selected]
    if missing:
        raise RuntimeError(f"Qwen dynamic integration 缺少证据布局: {missing}")
    if len(set(selected.values())) != len(selected):
        raise RuntimeError(f"Qwen dynamic integration 样本不独立: {selected}")
    return selected


def select_stress_batch_indices(
    dataset: MultiSourceLandslideDataset,
    batch_size: int,
    *,
    multisource_only: bool,
) -> list[int]:
    """Select a same-bucket, positive, distinct-parent batch for a real stress gate."""
    grouped: dict[int, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        if multisource_only and len(_row_families(row)) < 3:
            continue
        if bool((row.get("mask") or {}).get("empty_mask")):
            continue
        if str(row.get("task_family") or "") == "no_target_segmentation":
            continue
        grouped.setdefault(int(dataset.bucket_size(index)), []).append(index)
    required = max(1, int(batch_size))
    for bucket in sorted(grouped, reverse=True):
        selected: list[int] = []
        parents: set[str] = set()
        ordered = sorted(
            grouped[bucket],
            key=lambda index: _stress_candidate_key(dataset, index),
            reverse=True,
        )
        for index in ordered:
            row = dataset.rows[index]
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            if parent in parents:
                continue
            parents.add(parent)
            selected.append(index)
            if len(selected) == required:
                return selected
    role = "multisource" if multisource_only else "spatial"
    raise RuntimeError(
        f"Qwen {role} memory gate 缺少 {required} 个同桶、正样本、不同 parent 的样本"
    )


def _stress_candidate_key(
    dataset: MultiSourceLandslideDataset, index: int
) -> tuple[int, int, int, str]:
    row = dataset.rows[index]
    mask = row.get("mask") or {}
    positive = int(not bool(mask.get("empty_mask")))
    task = str(row.get("task_family") or "")
    semantic_task = int(task != "no_target_segmentation")
    return (
        int(dataset.bucket_size(index)),
        positive,
        semantic_task,
        str(row.get("sample_id")),
    )


def _lora_gradient_sum(model: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in model.controller.model.named_parameters()
        if "lora_" in name and parameter.grad is not None
    )


def _module_gradient_report(module: torch.nn.Module, *, exclude_lora: bool = False) -> dict[str, Any]:
    norms = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if exclude_lora and "lora_" in name:
            continue
        norms.append(parameter.grad.detach().float().norm())
    if not norms:
        return {"num_parameters_with_grad": 0, "gradient_norm_sum": 0.0, "all_finite": True}
    values = torch.stack(norms)
    return {
        "num_parameters_with_grad": len(norms),
        "gradient_norm_sum": float(values.sum().cpu()),
        "all_finite": bool(torch.isfinite(values).all().cpu()),
    }


def _gradient_groups(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "qwen_lora": {
            "gradient_norm_sum": _lora_gradient_sum(model),
            "num_parameters_with_grad": sum(
                1 for name, parameter in model.controller.model.named_parameters()
                if "lora_" in name and parameter.grad is not None
            ),
        },
        "controller_aux": _module_gradient_report(model.controller, exclude_lora=True),
        "sane": _module_gradient_report(model.sane),
        "qmef": _module_gradient_report(model.qmef),
        "pmrd": _module_gradient_report(model.pmrd),
    }


def _cache_subset_record(model, batch, item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    selected_views = model.vision_bank.selected_views_for(
        batch.visual_evidence_key[0], batch.active_subsets[0]
    )
    selected_sources = sorted({
        str(source)
        for view in selected_views
        for source in view.get("source_modalities") or []
    })
    subset = item["active_subset"]
    dropped_sources = set(subset.dropped_names) & set(selected_sources)
    unexpected_sources = set(selected_sources) - set(subset.active_names)
    if dropped_sources or unexpected_sources:
        raise RuntimeError(
            f"Qwen cache subset 泄漏: selected={selected_sources} "
            f"active={subset.active_names} dropped={subset.dropped_names}"
        )
    _, token_mask, counts, family_ids, segments = model.vision_bank.tokens_for(
        batch.visual_evidence_key,
        batch.active_subsets,
        device,
        model.config.qwen_view_tokens_per_view,
    )
    if not counts or counts[0] <= 0 or int(token_mask[0].sum()) != counts[0]:
        raise RuntimeError(f"Qwen integration active subset 没有视觉 tokens: counts={counts}")
    return {
        "visual_token_count": int(counts[0]),
        "selected_source_modalities": selected_sources,
        "visual_family_ids": family_ids[0, :counts[0]].detach().cpu().tolist(),
        "view_segments": segments[0],
    }


def run_qwen_check(config, device: torch.device, max_memory_gib: float) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Qwen integration 必须显式使用 CUDA device")
    if config.controller != "qwen_mask_query" or not config.use_pretrained_sane:
        raise RuntimeError(
            f"Qwen integration 需要 qwen_mask_query + pretrained SANE，当前 preset={config.preset}"
        )
    strict = replace(
        config,
        train_hflip_prob=0.0,
        train_vflip_prob=0.0,
        missing_modality_consistency_weight=max(
            0.1, float(config.missing_modality_consistency_weight)
        ),
        num_workers=0,
    )
    full_dataset = MultiSourceLandslideDataset(replace(strict, modality_dropout=0.0), "train")
    dropped_dataset = MultiSourceLandslideDataset(replace(strict, modality_dropout=1.0), "train")
    dynamic_indices = select_qwen_dynamic_indices(full_dataset)
    single_items = [
        ("batch1-optical", full_dataset[dynamic_indices["optical"]]),
        ("batch1-multispectral", full_dataset[dynamic_indices["multispectral"]]),
        ("batch1-multisource", full_dataset[dynamic_indices["multisource"]]),
        ("batch1-dropped-multisource", dropped_dataset[dynamic_indices["multisource"]]),
    ]
    if single_items[-1][1]["active_subset"].is_full:
        raise RuntimeError("Qwen integration 未形成真实 dropped-modality student subset")
    stress_indices = {
        "max_spatial_bucket": select_stress_batch_indices(
            full_dataset, strict.batch_size, multisource_only=False
        ),
        "max_multisource_bucket": select_stress_batch_indices(
            full_dataset, strict.batch_size, multisource_only=True
        ),
    }
    full_stress_items = [full_dataset[index] for index in stress_indices["max_spatial_bucket"]]
    mixed_stress_items = [
        (full_dataset if offset % 2 == 0 else dropped_dataset)[index]
        for offset, index in enumerate(stress_indices["max_multisource_bucket"])
    ]
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(strict, device).train()
    if model.vision_bank is None:
        raise RuntimeError("Qwen integration 缺少 vision feature bank")
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
    scaler = create_grad_scaler(strict, device)
    autocast = autocast_enabled(strict, device)
    dtype = amp_dtype(strict, device)
    diagnostics: dict[str, Any] = {
        "protocol_version": INTEGRATION_PROTOCOL_VERSION,
        "selected_indices": dynamic_indices,
        "stress_indices": stress_indices,
        "single_batch_checks": [],
        "memory_gates": [],
        "optimizer_steps": [],
    }

    def run_probe(
        role: str,
        probe_items: list[dict[str, Any]],
        *,
        measure_memory: bool,
        take_optimizer_step: bool,
    ) -> dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        if measure_memory:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        batch = qpsalm_collate(probe_items)
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
            output = model(batch)
            loss = output["loss"]
        if not torch.isfinite(loss):
            raise IntegrationFailure(
                f"Qwen integration loss 非有限: role={role}", diagnostics
            )
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradients = _gradient_report(model)
        groups = _gradient_groups(model)
        lora_gradient = float(groups["qwen_lora"]["gradient_norm_sum"])
        sequence_lengths = output["controller_sequence_lengths"].detach().cpu().tolist()
        visual_counts = output["controller_visual_token_counts"].detach().cpu().tolist()
        parents = [str(item["metadata"]["parent_sample_id"]) for item in probe_items]
        record = {
            "role": role,
            "batch_size": batch.batch_size,
            "sample_ids": [str(item["metadata"]["sample_id"]) for item in probe_items],
            "parent_sample_ids": parents,
            "unique_parent_count": len(set(parents)),
            "bucket_size": int(probe_items[0]["metadata"]["target_size"]),
            "sequence_lengths": sequence_lengths,
            "visual_token_counts": visual_counts,
            "padding_ratio": (
                1.0 - sum(sequence_lengths) / max(len(sequence_lengths) * max(sequence_lengths), 1)
            ),
            "teacher_sample_count": float(output["teacher_sample_count"].detach().cpu()),
            "loss": float(loss.detach().float().cpu()),
            "gradients": gradients,
            "gradient_groups": groups,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024**3) if measure_memory else None
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / (1024**3) if measure_memory else None
            ),
        }
        if (
            not gradients["all_finite"]
            or gradients["gradient_norm_sum"] <= 0
            or not math.isfinite(lora_gradient)
            or lora_gradient <= 0
        ):
            diagnostics["failed_probe"] = record
            raise IntegrationFailure(
                f"Qwen integration 梯度无效: role={role} lora={lora_gradient}",
                diagnostics,
            )
        torch.nn.utils.clip_grad_norm_(trainable, strict.grad_clip)
        if take_optimizer_step:
            scaler.step(optimizer)
            scaler.update()
        elif scaler.is_enabled():
            scaler.update()
        return record

    for role, item in single_items:
        cache_record = _cache_subset_record(model, qpsalm_collate([item]), item, device)
        record = run_probe(role, [item], measure_memory=False, take_optimizer_step=False)
        record["cache"] = cache_record
        diagnostics["single_batch_checks"].append(record)

    diagnostics["memory_gates"].append(
        run_probe(
            "max_spatial_bucket",
            full_stress_items,
            measure_memory=True,
            take_optimizer_step=False,
        )
    )
    diagnostics["memory_gates"].append(
        run_probe(
            "max_multisource_bucket",
            mixed_stress_items,
            measure_memory=True,
            take_optimizer_step=False,
        )
    )
    diagnostics["optimizer_steps"].append(
        run_probe("optimizer-step-full", full_stress_items, measure_memory=False, take_optimizer_step=True)
    )
    diagnostics["optimizer_steps"].append(
        run_probe("optimizer-step-mixed", mixed_stress_items, measure_memory=False, take_optimizer_step=True)
    )

    sequence_lengths = {
        value
        for row in diagnostics["single_batch_checks"]
        for value in row["sequence_lengths"]
    }
    visual_counts = {
        value
        for row in diagnostics["single_batch_checks"]
        for value in row["visual_token_counts"]
    }
    if len(sequence_lengths) < 2 or len(visual_counts) < 2:
        raise RuntimeError(
            "Qwen dynamic integration 未形成变长序列: "
            f"sequence_lengths={sorted(sequence_lengths)} visual_counts={sorted(visual_counts)}"
        )
    if not any(row["teacher_sample_count"] > 0 for row in diagnostics["memory_gates"]):
        raise RuntimeError("Qwen dynamic integration 未覆盖 teacher/student consistency")
    peak_allocated = max(row["peak_allocated_gib"] for row in diagnostics["memory_gates"])
    peak_reserved = max(row["peak_reserved_gib"] for row in diagnostics["memory_gates"])
    if peak_reserved > float(max_memory_gib):
        raise RuntimeError(
            f"Qwen integration 峰值显存超过门槛: reserved={peak_reserved:.3f} GiB "
            f"limit={max_memory_gib:.3f} GiB"
        )
    return {
        "status": "passed",
        "protocol_version": INTEGRATION_PROTOCOL_VERSION,
        "preset": strict.preset,
        "device": str(device),
        **diagnostics,
        "sequence_length_diversity": sorted(sequence_lengths),
        "visual_token_count_diversity": sorted(visual_counts),
        "cache": {
            "format": model.vision_bank.manifest["format"],
        },
        "qwen": {
            "num_trainable_lora_parameters": len(trainable_qwen),
            "gradient_checkpointing": model.controller.gradient_checkpointing_mode,
            "gradient_checkpointing_kwargs": model.controller.gradient_checkpointing_kwargs,
            "amp_dtype": strict.amp_dtype,
            "trainable_lora_dtypes": sorted({
                str(parameter.dtype)
                for name, parameter in model.controller.model.named_parameters()
                if parameter.requires_grad and "lora_" in name
            }),
        },
        "memory": {
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "limit_gib": float(max_memory_gib),
            "underutilized_warning": peak_reserved < 18.0,
            "note": "nvidia-smi also includes quantized base weights, CUDA context and allocator overhead",
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
    if args.amp_dtype is not None:
        base = replace(base, amp_dtype=args.amp_dtype)
    device = resolve_device(args.device)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "created_unix": time.time(),
        "benchmark_dir": str(base.benchmark_path()),
        "mode": args.mode,
        "seed": args.seed,
        "amp_dtype": base.amp_dtype,
        "checks": {},
    }
    errors, warnings = [], []
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
            qwen_config = replace(
                qwen_config,
                vision_feature_cache=cache,
                qwen_gradient_checkpointing=(
                    args.qwen_gradient_checkpointing
                    or qwen_config.qwen_gradient_checkpointing
                ),
            )
            report["checks"]["qwen"] = run_qwen_check(qwen_config, device, args.max_memory_gib)
            memory = report["checks"]["qwen"].get("memory") or {}
            if memory.get("underutilized_warning"):
                warnings.append(
                    f"qwen peak_reserved_gib={memory.get('peak_reserved_gib', 0):.3f} < 18.0"
                )
        except Exception as exc:
            report["checks"]["qwen"] = {
                "status": "failed",
                "error": str(exc),
                **(exc.details if isinstance(exc, IntegrationFailure) else {}),
            }
            errors.append(f"qwen: {exc}")
    report["acceptance"] = {
        "passed": not errors,
        "required_checks": ["raw", "qwen"] if args.mode == "all" else [args.mode],
        "errors": errors,
        "warnings": warnings,
    }
    path = _write_report(args.output, report)
    print(json.dumps({"report": str(path), **report["acceptance"]}, ensure_ascii=False))
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
