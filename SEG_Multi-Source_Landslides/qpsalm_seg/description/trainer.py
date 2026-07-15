#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-GPU D-1/D0-D4 trainer for segmentation-grounded description."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from qpsalm_seg.paths import resolve_project_path

from .checkpoint import (
    initialize_segdesc_checkpoint,
    load_segdesc_checkpoint,
    save_segdesc_checkpoint,
)
from .common import (
    append_jsonl,
    build_description_dataset,
    build_description_loader,
    description_amp_dtype,
    description_device,
    description_scaler,
    move_description_batch,
    set_description_seed,
    validation_split,
    write_json,
)
from .config import SegDescConfig
from .evaluator import description_selection_score, evaluate_description
from .model import DESCRIPTION_ADAPTER_NAME
from .runtime import build_description_optimizer, build_segdesc_model


def _desc_adapter_parameters(model) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if f".{DESCRIPTION_ADAPTER_NAME}." in name and "lora_" in name
    ]


def _gradient_summary(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [value.grad.detach().float() for value in parameters if value.grad is not None]
    return {
        "num_parameters": len(parameters),
        "num_with_grad": len(gradients),
        "num_nonzero": sum(int(torch.count_nonzero(value).item() > 0) for value in gradients),
        "norm_sum": float(sum((value.norm() for value in gradients), start=torch.tensor(0.0, device=gradients[0].device)).cpu()) if gradients else 0.0,
        "all_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
    }


def _train_loss(model, batch: dict[str, Any], config: SegDescConfig) -> tuple[torch.Tensor, dict[str, float]]:
    backbone = model.encode_description_requests(batch["requests"])
    if config.stage == "dior_alignment":
        loss, logits = model.region_alignment_loss(
            backbone, batch["region_masks"], batch["target_texts"]
        )
        targets = torch.arange(logits.shape[0], device=logits.device)
        accuracy = 0.5 * (
            (logits.argmax(1) == targets).float().mean()
            + (logits.argmax(0) == targets).float().mean()
        )
        return loss, {"in_batch_retrieval_r1": float(accuracy.detach().cpu())}
    output = model.describe_from_state(
        backbone,
        batch["region_masks"],
        batch["instructions"],
        target_texts=batch["target_texts"],
        region_valid_mask=backbone.valid_mask,
        protocol=config.region_protocol,
        structured_output=batch["structured_outputs"],
    )
    if output.per_sample_loss is None:
        raise RuntimeError("description forward 未产生 per-sample loss")
    weights = batch["weights"]
    loss = (output.per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {
        "mean_sequence_length": sum(output.sequence_lengths) / max(len(output.sequence_lengths), 1),
    }


def _load_best(path: Path) -> float:
    if not path.is_file():
        return -math.inf
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("selection_score", -math.inf))
    except (json.JSONDecodeError, TypeError, ValueError):
        return -math.inf


def train_description(
    config: SegDescConfig,
    *,
    device_name: str,
    resume: str | None = None,
    initialize_from: str | None = None,
) -> dict[str, Any]:
    if resume and initialize_from:
        raise ValueError("--resume 与 --initialize-from 不能同时使用")
    set_description_seed(config.seed)
    device = description_device(device_name)
    output_dir = resolve_project_path(config.output_dir) or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, migration = build_segdesc_model(config, device)
    bank = model.description_backbone.bank
    train_dataset = build_description_dataset(config, bank, split="train", training=True)
    if not len(train_dataset):
        raise RuntimeError(f"description stage={config.stage} 训练集为空")
    if config.stage == "dior_alignment" and config.batch_size < 2:
        raise ValueError("dior_alignment 需要 batch_size >= 2 才能形成对比负样本")
    train_streams = {
        "main": {
            "config": config,
            "dataset": train_dataset,
            "loader": build_description_loader(train_dataset, config, training=True),
        }
    }
    stream_pattern = ("main",)
    if config.stage == "bridge_expert":
        # D3b keeps the three supervision types in independent DataLoaders.
        # This avoids mixing contrastive DIOR rows with causal JSON rows in one
        # collate while preserving the documented 60/20/20 task schedule.
        dior_config = replace(config, stage="dior_alignment")
        global_config = replace(config, stage="rsicap_caption")
        dior_dataset = build_description_dataset(
            dior_config, bank, split="train", training=True
        )
        global_dataset = build_description_dataset(
            global_config, bank, split="train", training=True
        )
        if not len(dior_dataset) or not len(global_dataset):
            raise RuntimeError("D3b 需要非空 DIOR 与 global-caption replay 数据")
        train_streams = {
            "bridge": {
                "config": config,
                "dataset": train_dataset,
                "loader": build_description_loader(train_dataset, config, training=True),
            },
            "dior": {
                "config": dior_config,
                "dataset": dior_dataset,
                "loader": build_description_loader(
                    dior_dataset, dior_config, training=True,
                    batch_size=max(2, int(config.batch_size)),
                ),
            },
            "global_caption": {
                "config": global_config,
                "dataset": global_dataset,
                "loader": build_description_loader(global_dataset, global_config, training=True),
            },
        }
        stream_pattern = tuple(config.bridge_expert_task_pattern or [
            "bridge", "bridge", "bridge", "dior", "global_caption",
        ])
    val_name = validation_split(config.stage)
    val_loader = None
    if val_name is not None:
        val_dataset = build_description_dataset(config, bank, split=val_name, training=False)
        if len(val_dataset):
            val_loader = build_description_loader(val_dataset, config, training=False)

    optimizer, scheduler = build_description_optimizer(model, config)
    scaler = description_scaler(config, device)
    start_step = 0
    resume_metadata: dict[str, Any] = {}
    if resume:
        start_step, resume_metadata = load_segdesc_checkpoint(
            resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
    elif initialize_from:
        _source_step, source_metadata = initialize_segdesc_checkpoint(initialize_from, model)
        resume_metadata = {
            "initialized_from": str(initialize_from),
            "source": source_metadata,
        }
    resolved = dict(config.__dict__)
    write_json(output_dir / "resolved_config.json", resolved)
    write_json(output_dir / "dataset_summary.json", {
        "stage": config.stage,
        "train_split": "train",
        "train_samples": len(train_dataset),
        "training_streams": {
            name: {
                "stage": value["config"].stage,
                "samples": len(value["dataset"]),
                "batch_size": value["loader"].batch_size,
            }
            for name, value in train_streams.items()
        },
        "stream_pattern": list(stream_pattern),
        "validation_split": val_name,
        "validation_samples": len(val_loader.dataset) if val_loader is not None else 0,
        "initialized_from": initialize_from,
    })
    print(
        f"[DESC-DATA] stage={config.stage} train={len(train_dataset)} "
        f"val={len(val_loader.dataset) if val_loader is not None else 0}"
    )
    print(
        f"[DESC-MODEL] protocol={config.region_protocol} precision={config.amp_dtype} "
        f"batch={config.batch_size} ga={config.grad_accum_steps} max_steps={config.max_steps}"
    )
    best_path = output_dir / "validation_best.json"
    best_score = float((resume_metadata.get("metadata") or {}).get("best_score", _load_best(best_path)))
    history_path = output_dir / "train_history.jsonl"
    validation_history = output_dir / "validation_history.jsonl"
    desc_parameters = _desc_adapter_parameters(model)
    if not desc_parameters:
        raise RuntimeError("description trainer 未找到 desc_adapter LoRA 参数")
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"
    iterators = {name: iter(value["loader"]) for name, value in train_streams.items()}
    epochs = {name: 0 for name in train_streams}
    micro_step = start_step * max(1, int(config.grad_accum_steps))
    progress = tqdm(total=config.max_steps, initial=start_step, desc="qpsalm-description")
    window_loss = window_samples = 0.0
    window_steps = 0
    window_started = time.perf_counter()
    first_gradient_checked = start_step > 0
    step = start_step
    model.train()
    while step < config.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_samples = 0
        auxiliary: dict[str, list[float]] = {}
        for _ in range(max(1, int(config.grad_accum_steps))):
            stream_name = stream_pattern[micro_step % len(stream_pattern)]
            stream = train_streams[stream_name]
            try:
                cpu_batch = next(iterators[stream_name])
            except StopIteration:
                epochs[stream_name] += 1
                stream["dataset"].set_epoch(epochs[stream_name])
                iterators[stream_name] = iter(stream["loader"])
                cpu_batch = next(iterators[stream_name])
            micro_step += 1
            batch = move_description_batch(cpu_batch, device)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=autocast
            ):
                loss, diagnostics = _train_loss(model, batch, stream["config"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"description loss 非有限: step={step}")
            scaler.scale(loss / max(1, int(config.grad_accum_steps))).backward()
            step_loss += float(loss.detach().cpu())
            step_samples += len(batch["metadata"])
            for name, value in diagnostics.items():
                auxiliary.setdefault(name, []).append(float(value))
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradients = _gradient_summary(desc_parameters)
        if not first_gradient_checked:
            if not gradients["all_finite"] or gradients["num_nonzero"] <= 0:
                raise RuntimeError(
                    "desc_adapter 首个 optimizer step 梯度无效；"
                    f"summary={gradients}。请先运行 description integration smoke。"
                )
            first_gradient_checked = True
            write_json(output_dir / "desc_adapter_gradient_gate.json", gradients)
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups for value in group["params"]],
            config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        progress.update(1)
        step_mean = step_loss / max(1, int(config.grad_accum_steps))
        window_loss += step_mean
        window_samples += step_samples
        window_steps += 1

        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            elapsed = time.perf_counter() - window_started
            row = {
                "step": step,
                "epochs": dict(epochs),
                "loss": window_loss / max(window_steps, 1),
                "samples_per_second": window_samples / max(elapsed, 1.0e-9),
                "learning_rates": {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups},
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
                **{name: sum(values) / len(values) for name, values in auxiliary.items()},
            }
            append_jsonl(history_path, row)
            tqdm.write(
                f"[DESC-TRAIN] step={step} loss={row['loss']:.4f} "
                f"sample_sps={row['samples_per_second']:.2f} peak_gib={row['peak_reserved_gib']:.2f}"
            )
            window_loss = window_samples = 0.0
            window_steps = 0
            window_started = time.perf_counter()

        validation_due = val_loader is not None and (
            step % config.val_interval == 0 or step == config.max_steps
        )
        if validation_due:
            report = evaluate_description(
                model,
                val_loader,
                config,
                device,
                split=str(val_name),
                output_dir=output_dir / "validation_latest",
                run_counterfactuals=False,
            )
            score = description_selection_score(report, config.stage, config.checkpoint_metric)
            record = {"step": step, "selection_score": score, "report": report}
            append_jsonl(validation_history, record)
            write_json(output_dir / "validation_latest.json", record)
            tqdm.write(f"[DESC-VAL] step={step} score={score:.4f} loss={report['mean_teacher_forced_loss']}")
            if score > best_score:
                best_score = score
                write_json(best_path, record)
                save_segdesc_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    model,
                    step=step,
                    segmentation_migration=migration,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metadata={"stage": config.stage, "best_score": best_score, "config": resolved},
                )
                tqdm.write(f"[DESC-CKPT] saved=checkpoint_best.pt step={step} score={score:.4f}")
            model.train()

        if step % config.save_interval == 0 or step == config.max_steps:
            save_segdesc_checkpoint(
                output_dir / "checkpoint_last.pt",
                model,
                step=step,
                segmentation_migration=migration,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metadata={"stage": config.stage, "best_score": best_score, "config": resolved},
            )
            tqdm.write(f"[DESC-CKPT] saved=checkpoint_last.pt step={step}")
    progress.close()
    return {
        "output_dir": str(output_dir),
        "stage": config.stage,
        "steps": step,
        "best_score": best_score if math.isfinite(best_score) else None,
        "checkpoint_best": str(output_dir / "checkpoint_best.pt") if (output_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
    }
