#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 alternating segmentation/global-caption/region-description training."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from qpsalm_seg.engine.common import build_dataloaders
from qpsalm_seg.engine.evaluator import evaluate as evaluate_segmentation
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
    write_json,
)
from .config import SegDescConfig
from .evaluator import evaluate_description
from .model import DESCRIPTION_ADAPTER_NAME
from .runtime import build_segdesc_model
from .trainer import _train_loss


class EpochConcatDataset(ConcatDataset):
    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


def _no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.casefold()


def build_joint_optimizer(model, config: SegDescConfig):
    """Build one named optimizer while keeping the segmentation trunk frozen by default."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        role = None
        if "lora_" in name and f".{DESCRIPTION_ADAPTER_NAME}." in name:
            role = "description_adapter"
        elif "lora_" in name and ".default." in name:
            role = "segmentation_adapter"
        elif name.startswith((
            "segmentation.sane.", "segmentation.qmef.", "segmentation.pmrd.",
            "segmentation.controller.text_type", "segmentation.controller.view_description_type",
            "segmentation.controller.view_attention_query", "segmentation.controller.evidence_anchors",
            "segmentation.controller.anchor_availability", "segmentation.controller.mask_embeddings",
            "segmentation.controller.view_to_hidden", "segmentation.controller.visual_family_embedding",
            "segmentation.controller.output_projection",
        )):
            if config.joint_train_shared_segmentation_dense:
                role = "segmentation_dense"
        elif name.startswith(("description_backbone.", "mgrr.")):
            role = "mgrr"
        elif name.startswith((
            "region_to_hidden.", "description_view_to_hidden.", "alignment_text_projection."
        )) or name in {"region_type", "instruction_type", "visual_type", "alignment_temperature"}:
            role = "description_projection"
        if role is None:
            continue
        parameter.requires_grad_(True)
        groups.setdefault((role, _no_decay(name, parameter)), []).append(parameter)
    required = {
        "segmentation_adapter", "description_adapter", "mgrr", "description_projection",
    }
    if config.joint_train_shared_segmentation_dense:
        required.add("segmentation_dense")
    missing = sorted(required - {role for role, _no_decay_value in groups})
    if missing:
        raise RuntimeError(f"joint optimizer 缺少参数组: {missing}")
    scales = {
        "segmentation_adapter": 0.1,
        "description_adapter": config.desc_adapter_lr_scale,
        "segmentation_dense": 0.25,
        "mgrr": 1.0,
        "description_projection": 0.5,
    }
    parameter_groups = []
    for (role, no_decay), parameters in sorted(groups.items()):
        parameter_groups.append({
            "name": role + ("_no_decay" if no_decay else "_decay"),
            "group_role": role,
            "params": parameters,
            "lr": config.learning_rate * scales[role],
            "lr_scale": scales[role],
            "weight_decay": 0.0 if no_decay or "adapter" in role else config.weight_decay,
        })
    optimizer = torch.optim.AdamW(parameter_groups)

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _positive_dice(report: dict[str, Any]) -> float:
    return float((((report.get("metrics") or {}).get("positive_only") or {}).get("dice")) or 0.0)


def _next(iterator, loader):
    try:
        return next(iterator), iterator, False
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator, True


def train_joint_segdesc(
    config: SegDescConfig,
    *,
    device_name: str,
    resume: str | None = None,
    initialize_from: str | None = None,
) -> dict[str, Any]:
    if resume and initialize_from:
        raise ValueError("joint --resume 与 --initialize-from 不能同时使用")
    if not resume and not initialize_from:
        raise ValueError(
            "正式 M7 必须使用 --initialize-from 加载已通过 M6 的 qpsalm_segdesc_v1 权重；"
            "--resume 仅用于同一 M7 run 续训"
        )
    set_description_seed(config.seed)
    device = description_device(device_name)
    output_dir = resolve_project_path(config.output_dir) or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, migration = build_segdesc_model(config, device)
    bank = model.description_backbone.bank

    seg_config = replace(
        model.segmentation.config,
        batch_size=int(config.joint_segmentation_batch_size),
        grad_accum_steps=1,
        max_train_samples=config.max_train_samples or None,
        monitor_val_samples=config.max_val_samples or None,
    )
    segmentation_train, segmentation_val = build_dataloaders(seg_config)
    global_configs = [replace(config, stage=stage) for stage in (config.joint_global_stages or ["mmrs_caption", "rsicap_caption"])]
    global_train_sets = [
        build_description_dataset(value, bank, split="train", training=True)
        for value in global_configs
    ]
    global_train = EpochConcatDataset(global_train_sets)
    global_loader = build_description_loader(
        global_train, config, training=True, batch_size=config.joint_description_batch_size
    )
    region_config = replace(config, stage=config.joint_region_stage)
    region_train = build_description_dataset(region_config, bank, split="train", training=True)
    region_loader = build_description_loader(
        region_train, region_config, training=True, batch_size=config.joint_description_batch_size
    )
    caption_val_config = replace(config, stage="rsicap_caption")
    caption_val = build_description_loader(
        build_description_dataset(caption_val_config, bank, split="dev", training=False),
        caption_val_config,
        training=False,
        batch_size=config.joint_description_batch_size,
    )
    region_val_name = "val" if config.joint_region_stage in {"bridge_expert", "predicted_mask"} else None
    region_val = (
        build_description_loader(
            build_description_dataset(region_config, bank, split=region_val_name, training=False),
            region_config,
            training=False,
            batch_size=config.joint_description_batch_size,
        )
        if region_val_name is not None else None
    )
    if not len(global_train) or not len(region_train):
        raise RuntimeError("M7 joint training 需要非空 global-caption 与 region-description 数据")

    optimizer, scheduler = build_joint_optimizer(model, config)
    scaler = description_scaler(config, device)
    start_step = 0
    resume_metadata = {}
    if resume:
        start_step, resume_metadata = load_segdesc_checkpoint(
            resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
    elif initialize_from:
        _source_step, source_metadata = initialize_segdesc_checkpoint(initialize_from, model)
        resume_metadata = {"initialized_from": initialize_from, "source": source_metadata}
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"

    with model.controller.adapter_scope("default"):
        baseline_report = evaluate_segmentation(
            model.segmentation, segmentation_val, device, threshold=model.segmentation.config.eval_threshold
        )
    baseline_dice = _positive_dice(baseline_report)
    write_json(output_dir / "segmentation_monitor_baseline.json", baseline_report)
    write_json(output_dir / "joint_manifest.json", {
        "protocol": "qpsalm_segdesc_joint_v1",
        "task_pattern": list(config.resolved_joint_task_pattern()),
        "gradient_accumulation_microbatches_per_optimizer_step": config.grad_accum_steps,
        "segmentation_train": len(segmentation_train.dataset),
        "global_caption_train": len(global_train),
        "region_description_train": len(region_train),
        "segmentation_monitor_baseline_positive_dice": baseline_dice,
        "retention_max_drop": config.segmentation_retention_max_drop,
        "resolved_config": dict(config.__dict__),
        "initialized_from": initialize_from,
        "shared_segmentation_dense_trainable": bool(
            config.joint_train_shared_segmentation_dense
        ),
    })
    print(
        f"[JOINT-DATA] segmentation={len(segmentation_train.dataset)} "
        f"global_caption={len(global_train)} region_description={len(region_train)}"
    )
    print(
        f"[JOINT-MODEL] pattern={config.joint_task_pattern} "
        f"shared_segmentation_dense={config.joint_train_shared_segmentation_dense} "
        f"baseline_positive_dice={baseline_dice:.4f}"
    )

    iterators = {
        "segmentation": iter(segmentation_train),
        "global_caption": iter(global_loader),
        "region_description": iter(region_loader),
    }
    loaders = {
        "segmentation": segmentation_train,
        "global_caption": global_loader,
        "region_description": region_loader,
    }
    epoch_by_task = {name: 0 for name in loaders}
    pattern = config.resolved_joint_task_pattern()
    grad_accum = max(1, int(config.grad_accum_steps))
    history_path = output_dir / "joint_history.jsonl"
    best_score = float((resume_metadata.get("metadata") or {}).get("best_score", -math.inf))
    progress = tqdm(total=config.max_steps, initial=start_step, desc="qpsalm-segdesc-joint")
    step = start_step
    window: list[dict[str, Any]] = []
    window_started = time.perf_counter()
    model.train()
    while step < config.max_steps:
        task = pattern[step % len(pattern)]
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[torch.Tensor] = []
        sample_count = 0
        for micro_index in range(grad_accum):
            raw_batch, iterators[task], restarted = _next(iterators[task], loaders[task])
            if restarted:
                epoch_by_task[task] += 1
                dataset = loaders[task].dataset
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch_by_task[task])
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast):
                if task == "segmentation":
                    with model.controller.adapter_scope("default"):
                        output = model.segmentation(raw_batch)
                    loss = output["loss"]
                    sample_count += raw_batch.batch_size
                else:
                    batch = move_description_batch(raw_batch, device)
                    active_config = caption_val_config if task == "global_caption" else region_config
                    loss, _diagnostics = _train_loss(model, batch, active_config)
                    sample_count += len(batch["metadata"])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"joint loss 非有限: step={step} micro={micro_index} task={task}"
                )
            micro_losses.append(loss.detach().float())
            scaler.scale(loss / grad_accum).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups for value in group["params"]],
            config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        progress.update(1)
        mean_loss = torch.stack(micro_losses).mean()
        window.append({"task": task, "loss": float(mean_loss.cpu()), "samples": sample_count})
        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            elapsed = time.perf_counter() - window_started
            by_task = {}
            for name in sorted(set(value["task"] for value in window)):
                values = [value["loss"] for value in window if value["task"] == name]
                by_task[name] = {"steps": len(values), "loss": sum(values) / len(values)}
            record = {
                "step": step,
                "by_task": by_task,
                "samples_per_second": sum(value["samples"] for value in window) / max(elapsed, 1.0e-9),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
            }
            append_jsonl(history_path, record)
            tqdm.write(
                f"[JOINT-TRAIN] step={step} sample_sps={record['samples_per_second']:.2f} "
                f"peak_gib={record['peak_reserved_gib']:.2f} losses={by_task}"
            )
            window = []
            window_started = time.perf_counter()

        if step % config.val_interval == 0 or step == config.max_steps:
            with model.controller.adapter_scope("default"):
                seg_report = evaluate_segmentation(
                    model.segmentation, segmentation_val, device,
                    threshold=model.segmentation.config.eval_threshold,
                )
            caption_report = evaluate_description(
                model, caption_val, caption_val_config, device, split="dev",
                output_dir=output_dir / "validation_caption_latest", run_counterfactuals=False,
            )
            region_report = (
                evaluate_description(
                    model, region_val, region_config, device, split=str(region_val_name),
                    output_dir=output_dir / "validation_region_latest", run_counterfactuals=False,
                )
                if region_val is not None else None
            )
            current_dice = _positive_dice(seg_report)
            drop = baseline_dice - current_dice
            caption_score = float((caption_report["generation_metrics"] or {}).get("caption_token_f1") or 0.0)
            region_score = float(
                ((region_report or {}).get("generation_metrics") or {}).get("structured_field_macro_f1") or 0.0
            )
            retention_passed = drop <= config.segmentation_retention_max_drop
            composite = caption_score + region_score + current_dice if retention_passed else -math.inf
            validation = {
                "step": step,
                "segmentation_positive_dice": current_dice,
                "segmentation_drop": drop,
                "retention_passed": retention_passed,
                "caption_token_f1": caption_score,
                "region_structured_field_macro_f1": region_score,
                "selection_score": composite if math.isfinite(composite) else None,
            }
            append_jsonl(output_dir / "joint_validation_history.jsonl", validation)
            write_json(output_dir / "joint_validation_latest.json", validation)
            tqdm.write(
                f"[JOINT-VAL] step={step} seg_dice={current_dice:.4f} drop={drop:.4f} "
                f"caption={caption_score:.4f} region={region_score:.4f} retention={retention_passed}"
            )
            if retention_passed and composite > best_score:
                best_score = composite
                save_segdesc_checkpoint(
                    output_dir / "checkpoint_best.pt", model, step=step,
                    segmentation_migration=migration, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler,
                    metadata={
                        "stage": "joint", "best_score": best_score,
                        "segmentation_monitor_baseline_positive_dice": baseline_dice,
                        "validation": validation, "config": dict(config.__dict__),
                    },
                )
                tqdm.write(f"[JOINT-CKPT] saved=checkpoint_best.pt step={step}")
            model.train()
        if step % config.save_interval == 0 or step == config.max_steps:
            save_segdesc_checkpoint(
                output_dir / "checkpoint_last.pt", model, step=step,
                segmentation_migration=migration, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler,
                metadata={
                    "stage": "joint", "best_score": best_score,
                    "segmentation_monitor_baseline_positive_dice": baseline_dice,
                    "config": dict(config.__dict__),
                },
            )
            tqdm.write(f"[JOINT-CKPT] saved=checkpoint_last.pt step={step}")
    progress.close()
    return {
        "output_dir": str(output_dir),
        "steps": step,
        "best_score": best_score if math.isfinite(best_score) else None,
        "checkpoint_best": str(output_dir / "checkpoint_best.pt") if (output_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
    }
