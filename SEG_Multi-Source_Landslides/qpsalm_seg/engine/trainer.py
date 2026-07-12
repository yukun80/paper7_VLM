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
from qpsalm_seg.metrics import batch_binary_metrics
from qpsalm_seg.paths import resolve_repo_path

from .checkpoint import load_checkpoint, prune_step_checkpoints, save_checkpoint
from .common import build_dataloaders, build_model, cosine_lr, resolve_device, set_seed, utc_now, write_json
from .diagnostics import (
    average_dicts,
    format_train_window,
    loss_log_values,
    summarize_train_window,
    training_signal_values,
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
    if start_step <= 0 or not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [row for row in rows if isinstance(row, dict) and int(row.get("step", -1)) < start_step]


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
    print(
        f"dataset train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"batch_size={config.batch_size} grad_accum_steps={grad_accum} target_size={config.target_size} "
        f"steps_per_epoch={steps_per_epoch} max_steps={config.max_steps} estimated_epochs={config.max_steps / steps_per_epoch:.2f}"
    )
    print(f"dataset_combos train={train_report['family_combos']} val={val_report['family_combos']}")
    print(
        f"dataset_sensor_normalization train_sensors={train_report['sensor_combos']} "
        f"train_products={train_report['product_combos']} train_norms={train_report['normalization_methods']}"
    )
    print(
        f"algorithm preset={config.preset} controller={config.controller} pretrained_sane={config.use_pretrained_sane} "
        f"qmef={config.use_qmef} pmrd={config.use_mask_refinement} queries={config.num_mask_tokens} "
        f"instruction_ablation={config.instruction_ablation} visual_ablation={config.visual_ablation}"
    )
    model = build_model(config, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    start_step = load_checkpoint(resume, model, optimizer) if resume else 0
    history_path = out_dir / "train_history.json"
    history = load_history(history_path, start_step) if resume else []
    best_path = out_dir / "validation_best.json"
    best_score = load_best_validation(best_path) if resume else -1.0
    step, iterator = start_step, iter(train_loader)
    window, window_start = [], time.perf_counter()
    autocast, dtype = device.type == "cuda", torch.bfloat16 if device.type == "cuda" else torch.float32
    progress = tqdm(total=max(0, config.max_steps - start_step), desc="qpsalm-train", dynamic_ncols=True)
    while step < config.max_steps:
        model.train()
        multiplier = cosine_lr(step, config.max_steps, config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        accumulated, metrics, loss_rows, signal_rows = 0.0, [], [], []
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
            accumulated += float(loss.detach().cpu())
            metrics.extend(batch_binary_metrics(
                outputs["final_mask_logits"].detach().cpu(), batch.mask.detach().cpu(),
                threshold=config.eval_threshold, valid_mask=batch.valid_mask.detach().cpu(),
            ))
            loss_rows.append(loss_log_values(outputs))
            signal_rows.append(training_signal_values(outputs))
            (loss / grad_accum).backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        row = {
            "step": step, "loss": accumulated / grad_accum, "lr": config.lr * multiplier,
            "dice": sum(value["dice"] for value in metrics) / len(metrics),
            "iou": sum(value["iou"] for value in metrics) / len(metrics),
            "grad_accum_steps": float(grad_accum),
            **average_dicts(loss_rows), **average_dicts(signal_rows),
        }
        history.append(row)
        window.append(row)
        step += 1
        progress.update(1)
        if config.log_interval > 0 and (step == start_step + 1 or step % config.log_interval == 0 or step == config.max_steps):
            summary = summarize_train_window(window, time.perf_counter() - window_start)
            progress.set_postfix({key: f"{summary.get(key, 0):.3f}" for key in ("loss", "iou", "dice")}, refresh=False)
            tqdm.write(format_train_window(int(window[0]["step"]), int(window[-1]["step"]), len(window), summary))
            write_json(history_path, history)
            window, window_start = [], time.perf_counter()

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
                f"val step={step} iou={overall.get('iou', 0):.4f} dice={overall.get('dice', 0):.4f} "
                f"precision={overall.get('precision', 0):.4f} recall={overall.get('recall', 0):.4f} "
                f"select={config.checkpoint_metric}:{selection:.4f} best={best_score:.4f} "
                f"n={report['coverage']['num_samples']} combos={len(report['coverage']['family_combos'])}"
            )
            write_json(out_dir / "validation_latest.json", report)
            if config.save_step_validation_reports:
                write_json(out_dir / f"validation_step_{step:06d}.json", report)
            if is_best:
                write_json(best_path, report)
                save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, step, config, update_last=False, include_optimizer=False)

        if (config.save_interval > 0 and step % config.save_interval == 0) or step == config.max_steps:
            save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, step, config, update_last=False)
            if config.save_step_checkpoints:
                save_checkpoint(out_dir / f"checkpoint_step_{step:06d}.pt", model, optimizer, step, config, update_last=False)
                prune_step_checkpoints(out_dir, config.keep_recent_checkpoints)
    progress.close()
    write_json(history_path, history)
    return {"output_dir": str(out_dir), "steps": step, "history": history[-5:]}
