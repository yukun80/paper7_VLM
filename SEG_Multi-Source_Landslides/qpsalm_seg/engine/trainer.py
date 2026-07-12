#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-GPU trainer with compact best/last checkpoint policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from collections import Counter
from typing import Any

import torch
from tqdm import tqdm

from qpsalm_seg.config import QPSalmConfig, save_config
from qpsalm_seg.indexing import family_combo, normalization_methods, product_combo, raw_modality_combo, sensor_combo
from qpsalm_seg.metrics import batch_binary_metric_tensors
from qpsalm_seg.paths import resolve_repo_path

from .checkpoint import load_checkpoint, prune_step_checkpoints, save_checkpoint
from .common import (
    amp_dtype,
    autocast_enabled,
    build_dataloaders,
    build_model,
    cosine_lr,
    create_grad_scaler,
    resolve_device,
    set_seed,
    utc_now,
    write_json,
)
from .diagnostics import (
    format_train_window,
    training_scalar_tensors,
)
from .evaluator import evaluate


def write_train_manifest(out_dir: Path, config: QPSalmConfig, device: str, resume: str | None) -> None:
    path = out_dir / "run_manifest.json"
    if not path.exists():
        write_json(path, {
            "created_at_utc": utc_now(), "created_by": "qpsalm-train", "preset": config.preset,
            "run_dir": str(out_dir), "device": device, "resume": resume,
            "checkpoint_last": str(out_dir / "checkpoint_last.pt"),
            "validation_latest": str(out_dir / "validation_latest.json"),
            "resolved_config": dict(config.__dict__),
        })


def validation_selection_score(report: dict[str, Any], metric_name: str) -> float:
    metrics = report.get("metrics") or {}
    overall, positive = metrics.get("overall") or {}, metrics.get("positive_only") or {}
    if metric_name == "overall_dice":
        return float(overall.get("dice", 0))
    if metric_name != "positive_only_dice":
        raise ValueError(f"未知 checkpoint_metric={metric_name!r}")
    return float(positive.get("dice", overall.get("dice", 0)))


def load_best_validation(path: Path) -> float:
    if not path.exists():
        return -1.0
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return -1.0
    return float(report.get("selection_score", -1.0))


def load_history(path: Path, start_step: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and (start_step <= 0 or int(row.get("step_end", -1)) < start_step):
            rows.append(row)
    return rows


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean_tensor_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: torch.stack([row[key] for row in rows if key in row]).mean()
        for key in keys
    }


def flush_train_window(
    rows: list[dict[str, torch.Tensor]],
    *,
    elapsed: float,
    sample_count: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    means = mean_tensor_rows(rows)
    keys = sorted(means)
    packed = torch.stack([means[key] for key in keys]).detach().float().cpu().tolist()
    summary = dict(zip(keys, packed))
    summary["steps_per_sec"] = len(rows) / max(elapsed, 1.0e-6)
    summary["samples_per_sec"] = sample_count / max(elapsed, 1.0e-6)
    if "controller_tokens_per_sample" in summary:
        summary["qwen_tokens_per_sec"] = (
            summary["controller_tokens_per_sample"] * summary["samples_per_sec"]
        )
    summary["lr"] = float(lr)
    summary["peak_reserved_gib"] = (
        torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    return summary


def dataset_combo_report(dataset) -> dict[str, Any]:
    rows = list(getattr(dataset, "rows", []) or [])
    counters = {
        "family_combos": Counter(family_combo(row) for row in rows),
        "raw_combos": Counter(raw_modality_combo(row) for row in rows),
        "sensor_combos": Counter(sensor_combo(row) for row in rows),
        "product_combos": Counter(product_combo(row) for row in rows),
        "normalization_methods": Counter(normalization_methods(row) for row in rows),
    }
    return {"num_rows": len(rows), **{key: dict(sorted(value.items())) for key, value in counters.items()}}


def train(config: QPSalmConfig, device_name: str, resume: str | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(device_name)
    out_dir = resolve_repo_path(config.output_dir) or Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_dataloaders(config)
    if len(train_loader) == 0:
        raise RuntimeError("训练集为空")
    grad_accum = max(1, int(config.grad_accum_steps))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    if config.max_steps is None or int(config.max_steps) <= 0:
        if config.num_epochs is None or int(config.num_epochs) <= 0:
            raise ValueError("max_steps 为空时必须设置 num_epochs")
        config.max_steps = int(config.num_epochs) * steps_per_epoch
    config.max_steps = int(config.max_steps)
    save_config(out_dir / "resolved_config.yaml", config)
    write_train_manifest(out_dir, config, device_name, resume)
    train_report, val_report = dataset_combo_report(train_loader.dataset), dataset_combo_report(val_loader.dataset)
    dataset_summary = {
        "train": train_report,
        "monitor_val": val_report,
        "full_val_rows": int(getattr(val_loader.dataset, "full_row_count", len(val_loader.dataset))),
        "monitor_sample_ids": [str(row.get("sample_id")) for row in val_loader.dataset.rows],
        "monitor_parent_ids": sorted({
            str(row.get("parent_sample_id") or row.get("sample_id"))
            for row in val_loader.dataset.rows
        }),
    }
    write_json(out_dir / "dataset_summary.json", dataset_summary)
    monitor_manifest = {
        "seed": config.seed + 1009,
        "limit": config.monitor_val_samples,
        "sample_ids": dataset_summary["monitor_sample_ids"],
        "parent_ids": dataset_summary["monitor_parent_ids"],
    }
    monitor_manifest_path = out_dir / "monitor_val_manifest.json"
    if resume and monitor_manifest_path.exists():
        observed = json.loads(monitor_manifest_path.read_text(encoding="utf-8"))
        if observed != monitor_manifest:
            raise RuntimeError("resume 的 monitor validation manifest 与当前数据选择不一致")
    write_json(monitor_manifest_path, monitor_manifest)
    print(
        f"[DATA] train={len(train_loader.dataset)} monitor_val={len(val_loader.dataset)} "
        f"full_val={dataset_summary['full_val_rows']} train_combos={len(train_report['family_combos'])} "
        f"val_combos={len(val_report['family_combos'])}"
    )
    print(
        f"[MODEL] preset={config.preset} controller={config.controller} "
        f"precision={config.amp_dtype} batch={config.batch_size} ga={grad_accum} "
        f"query_chunk={config.query_chunk_size} qwen_checkpoint={config.qwen_gradient_checkpointing} "
        f"qwen_base={('nf4' if config.qwen_4bit else config.amp_dtype) if config.controller == 'qwen_mask_query' else 'none'} "
        f"attention={config.qwen_attn_implementation if config.controller == 'qwen_mask_query' else 'none'} "
        f"target={config.target_size} "
        f"steps_per_epoch={steps_per_epoch} max_steps={config.max_steps}"
    )
    model = build_model(config, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    scaler = create_grad_scaler(config, device)
    start_step = load_checkpoint(resume, model, optimizer, scaler) if resume else 0
    history_path = out_dir / "train_history.jsonl"
    history = load_history(history_path, start_step) if resume else []
    if resume:
        write_history(history_path, history)
    best_path = out_dir / "validation_best.json"
    best_score = load_best_validation(best_path) if resume else -1.0
    step, iterator = start_step, iter(train_loader)
    window, window_elapsed, window_samples = [], 0.0, 0
    autocast, dtype = autocast_enabled(config, device), amp_dtype(config, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(
        total=max(0, config.max_steps - start_step),
        desc="qpsalm-train",
        dynamic_ncols=True,
        mininterval=max(0.5, float(config.progress_min_interval)),
    )
    while step < config.max_steps:
        optimizer_step_started = time.perf_counter()
        model.train()
        multiplier = cosine_lr(step, config.max_steps, config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        micro_rows: list[dict[str, torch.Tensor]] = []
        step_samples = 0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                outputs = model(batch)
                loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step={step}")
            row_tensors = {
                "loss": loss.detach().float(),
                **batch_binary_metric_tensors(
                    outputs["final_mask_logits"].detach(),
                    batch.mask.to(device=outputs["final_mask_logits"].device, non_blocking=True),
                    threshold=config.eval_threshold,
                    valid_mask=batch.valid_mask.to(
                        device=outputs["final_mask_logits"].device, non_blocking=True
                    ),
                ),
                **training_scalar_tensors(outputs),
            }
            micro_rows.append(row_tensors)
            step_samples += batch.batch_size
            scaler.scale(loss / grad_accum).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if (
            scaler.is_enabled()
            and config.controller == "qwen_mask_query"
            and step == start_step
        ):
            lora_gradients = [
                parameter.grad.detach().float().norm()
                for name, parameter in model.controller.model.named_parameters()
                if parameter.requires_grad and "lora_" in name and parameter.grad is not None
            ]
            lora_norm = (
                torch.stack(lora_gradients).sum()
                if lora_gradients else loss.new_tensor(0.0)
            )
            if not torch.isfinite(lora_norm) or float(lora_norm.detach().cpu()) <= 0:
                raise RuntimeError(
                    "FP16 unscale 后 LoRA 梯度为零或非有限；请改用 AMP_DTYPE=bf16，"
                    "并保留该运行的 integration report。"
                )
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        window.append(mean_tensor_rows(micro_rows))
        window_elapsed += time.perf_counter() - optimizer_step_started
        window_samples += step_samples
        step += 1
        progress.update(1)
        if config.log_interval > 0 and (step == start_step + 1 or step % config.log_interval == 0 or step == config.max_steps):
            summary = flush_train_window(
                window,
                elapsed=window_elapsed,
                sample_count=window_samples,
                lr=config.lr * multiplier,
                device=device,
            )
            record = {"step_start": step - len(window), "step_end": step - 1, **summary}
            history.append(record)
            append_history(history_path, record)
            progress.set_postfix({key: f"{summary.get(key, 0):.3f}" for key in ("loss", "iou", "dice")}, refresh=False)
            tqdm.write(format_train_window(step - len(window), step - 1, len(window), summary))
            window, window_elapsed, window_samples = [], 0.0, 0

        if step % config.val_interval == 0 or step == config.max_steps:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            visualize = config.num_visualizations > 0 and (
                step % max(1, config.visualize_interval) == 0 or step == config.max_steps
            )
            report = evaluate(
                model, val_loader, device,
                max_batches=config.max_val_batches if config.max_val_batches and config.max_val_batches > 0 else None,
                visual_dir=out_dir / "visualizations" / f"step_{step:06d}" if visualize else None,
                num_visualizations=config.num_visualizations if visualize else 0,
                threshold=config.eval_threshold, threshold_sweep=config.threshold_sweep,
            )
            report["step"] = step
            selection = validation_selection_score(report, config.checkpoint_metric)
            is_best = selection > best_score
            if is_best:
                best_score = selection
            report.update({
                "selection_metric": config.checkpoint_metric, "selection_score": selection,
                "is_best": is_best, "best_so_far": {"metric": config.checkpoint_metric, "score": best_score},
            })
            overall = (report.get("metrics") or {}).get("overall") or {}
            tqdm.write(
                f"[VAL] step={step} iou={overall.get('iou', 0):.4f} dice={overall.get('dice', 0):.4f} "
                f"precision={overall.get('precision', 0):.4f} recall={overall.get('recall', 0):.4f} "
                f"select={config.checkpoint_metric}:{selection:.4f} best={best_score:.4f} "
                f"n={report['coverage']['num_samples']} combos={len(report['coverage']['family_combos'])}"
            )
            write_json(out_dir / "validation_latest.json", report)
            if config.save_step_validation_reports:
                write_json(out_dir / f"validation_step_{step:06d}.json", report)
            if is_best:
                write_json(best_path, report)
                save_checkpoint(
                    out_dir / "checkpoint_best.pt", model, optimizer, step, config,
                    update_last=False, include_optimizer=False, scaler=scaler,
                )
                tqdm.write(f"[CKPT] saved=checkpoint_best.pt step={step} score={best_score:.4f}")

        if (config.save_interval > 0 and step % config.save_interval == 0) or step == config.max_steps:
            save_checkpoint(
                out_dir / "checkpoint_last.pt", model, optimizer, step, config,
                update_last=False, scaler=scaler,
            )
            tqdm.write(f"[CKPT] saved=checkpoint_last.pt step={step}")
            if config.save_step_checkpoints:
                save_checkpoint(
                    out_dir / f"checkpoint_step_{step:06d}.pt", model, optimizer, step, config,
                    update_last=False, scaler=scaler,
                )
                prune_step_checkpoints(out_dir, config.keep_recent_checkpoints)
    progress.close()
    if model.vision_bank is not None:
        write_json(out_dir / "vision_cache_runtime.json", {
            "hits": int(model.vision_bank.cache_hits),
            "misses": int(model.vision_bank.cache_misses),
            "loaded_shards": len(model.vision_bank._loaded),
            "loaded_bytes": int(model.vision_bank._loaded_bytes),
            "ram_budget_bytes": int(model.vision_bank.ram_budget_bytes),
        })
    return {"output_dir": str(out_dir), "steps": step, "history": history[-5:]}
