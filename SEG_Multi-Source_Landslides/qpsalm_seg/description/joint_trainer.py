#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 alternating segmentation/global-caption/region-description training."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from qpsalm_seg.engine.common import build_dataloaders
from qpsalm_seg.engine.evaluator import (
    SAMPLE_IDENTITY_FIELDS,
    SAMPLE_IDENTITY_PROTOCOL,
    evaluate as evaluate_segmentation,
)
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


def monitor_baseline_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Freeze the exact monitor population used before joint optimization."""
    coverage = dict(report.get("coverage") or {})
    population = dict(coverage.get("sample_population") or {})
    if (
        population.get("protocol") != SAMPLE_IDENTITY_PROTOCOL
        or tuple(population.get("fields") or ()) != tuple(SAMPLE_IDENTITY_FIELDS)
    ):
        raise RuntimeError("joint monitor baseline sample population protocol 或字段集合过期")
    if not population.get("complete") or not population.get("unique"):
        raise RuntimeError("joint monitor baseline sample population 不完整或包含重复 sample")
    identity = {
        "protocol": "qpsalm_segdesc_monitor_baseline_v1",
        "num_samples": int(coverage.get("num_samples") or 0),
        "sample_population": population,
        "threshold": float(report.get("threshold", 0.5)),
        "positive_dice": _positive_dice(report),
    }
    if identity["num_samples"] <= 0 or not str(population.get("sha256") or ""):
        raise RuntimeError("joint monitor baseline 缺少有效样本数或 population hash")
    if (
        int(population.get("num_records", -1)) != identity["num_samples"]
        or int(population.get("num_unique_sample_ids", -1)) != identity["num_samples"]
    ):
        raise RuntimeError("joint monitor baseline sample population 计数不一致")
    return identity


def monitor_retention_gate(
    baseline_identity: dict[str, Any],
    current_report: dict[str, Any],
    *,
    maximum_allowed_drop: float,
) -> dict[str, Any]:
    """Evaluate a periodic monitor without presenting it as the formal full-val gate."""
    current_identity = monitor_baseline_identity(current_report)
    same_population = (
        current_identity["num_samples"] == baseline_identity.get("num_samples")
        and current_identity["sample_population"] == baseline_identity.get("sample_population")
    )
    same_threshold = abs(
        float(current_identity["threshold"]) - float(baseline_identity.get("threshold", 0.5))
    ) <= 1.0e-12
    drop = float(baseline_identity["positive_dice"]) - float(current_identity["positive_dice"])
    passed = bool(
        same_population and same_threshold and drop <= float(maximum_allowed_drop)
    )
    return {
        "protocol": "qpsalm_segdesc_monitor_retention_v1",
        "monitor_only": True,
        "same_sample_population": same_population,
        "same_threshold": same_threshold,
        "baseline_positive_dice": float(baseline_identity["positive_dice"]),
        "current_positive_dice": float(current_identity["positive_dice"]),
        "absolute_drop": drop,
        "maximum_allowed_drop": float(maximum_allowed_drop),
        "passed": passed,
    }


def _next(iterator, loader):
    try:
        return next(iterator), iterator, False
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator, True


def _joint_gradient_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        role = str(group.get("group_role") or group.get("name") or "unknown")
        values = [
            parameter.grad.detach().float()
            for parameter in group["params"] if parameter.grad is not None
        ]
        current = by_role.setdefault(role, {
            "num_parameters": 0, "num_with_grad": 0, "num_nonzero": 0,
            "norm_sum": 0.0, "all_finite": True,
        })
        current["num_parameters"] += len(group["params"])
        current["num_with_grad"] += len(values)
        current["num_nonzero"] += sum(
            int(torch.count_nonzero(value).item() > 0) for value in values
        )
        current["norm_sum"] += sum(float(value.norm().cpu()) for value in values)
        current["all_finite"] &= all(bool(torch.isfinite(value).all()) for value in values)
    return by_role


def validate_joint_task_gradients(
    task: str,
    report: dict[str, dict[str, Any]],
    *,
    train_shared_segmentation_dense: bool,
) -> dict[str, Any]:
    required = {
        "segmentation": {"segmentation_adapter"},
        "global_caption": {"description_adapter", "description_projection"},
        "region_description": {"description_adapter", "mgrr", "description_projection"},
    }
    if task not in required:
        raise ValueError(f"未知 joint gradient task={task!r}")
    required_roles = set(required[task])
    if task == "segmentation" and train_shared_segmentation_dense:
        required_roles.add("segmentation_dense")
    all_roles = {
        "segmentation_adapter", "description_adapter", "mgrr",
        "description_projection", "segmentation_dense",
    }
    forbidden_roles = all_roles - required_roles
    missing_or_zero = sorted(
        role for role in required_roles
        if role not in report
        or int(report[role]["num_nonzero"]) <= 0
        or not bool(report[role]["all_finite"])
    )
    leaked = sorted(
        role for role in forbidden_roles
        if role in report and int(report[role]["num_nonzero"]) > 0
    )
    nonfinite = sorted(
        role for role, value in report.items()
        if int(value.get("num_with_grad", 0)) > 0 and not bool(value.get("all_finite", False))
    )
    return {
        "task": task,
        "required_roles": sorted(required_roles),
        "forbidden_roles": sorted(forbidden_roles),
        "missing_or_zero": missing_or_zero,
        "leaked_nonzero_roles": leaked,
        "nonfinite_roles": nonfinite,
        "passed": not missing_or_zero and not leaked and not nonfinite,
    }


def joint_optimizer_manifest(model, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    observed: set[int] = set()
    for group in optimizer.param_groups:
        names = []
        numel = 0
        for parameter in group["params"]:
            parameter_id = id(parameter)
            name = names_by_id.get(parameter_id)
            if name is None:
                raise RuntimeError("joint optimizer parameter 不属于模型")
            if parameter_id in observed:
                raise RuntimeError(f"joint optimizer parameter 重复分组: {name}")
            observed.add(parameter_id)
            names.append(name)
            numel += int(parameter.numel())
        groups.append({
            "name": str(group.get("name")),
            "role": str(group.get("group_role")),
            "learning_rate": float(group["lr"]),
            "lr_scale": float(group["lr_scale"]),
            "weight_decay": float(group["weight_decay"]),
            "num_parameters": len(names),
            "numel": numel,
            "parameter_names": sorted(names),
        })
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if expected != observed:
        raise RuntimeError("joint optimizer 参数集合与 requires_grad 不一致")
    return {
        "protocol": "qpsalm_segdesc_joint_optimizer_v1",
        "groups": groups,
        "total_numel": sum(value["numel"] for value in groups),
    }


def _dataset_parent_ids(dataset) -> set[str]:
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        return set().union(*(_dataset_parent_ids(value) for value in nested))
    return {
        str(row["parent_sample_id"])
        for row in getattr(dataset, "rows", [])
        if row.get("parent_sample_id")
    }


def _joint_progress_payload(
    *,
    step: int,
    task_steps: Counter,
    task_samples: Counter,
    parent_coverage: dict[str, set[str]],
    parent_populations: dict[str, set[str]],
) -> dict[str, Any]:
    return {
        "protocol": "qpsalm_segdesc_joint_progress_v1",
        "step": int(step),
        "optimizer_steps": dict(task_steps),
        "samples_seen": dict(task_samples),
        "parent_coverage": {
            task: {
                "covered": len(parent_coverage[task]),
                "population": len(parent_populations[task]),
                "fraction": len(parent_coverage[task]) / max(len(parent_populations[task]), 1),
                "population_sha256": hashlib.sha256(
                    "\n".join(sorted(parent_populations[task])).encode("utf-8")
                ).hexdigest(),
                "covered_sha256": hashlib.sha256(
                    "\n".join(sorted(parent_coverage[task])).encode("utf-8")
                ).hexdigest(),
                "parent_ids": sorted(parent_coverage[task]),
            }
            for task in parent_populations
        },
    }


def restore_joint_progress(
    saved_progress: dict[str, Any],
    parent_populations: dict[str, set[str]],
    *,
    checkpoint_step: int,
    required: bool,
) -> tuple[Counter, Counter, dict[str, set[str]]]:
    """Restore coverage only when it belongs to the exact current data populations."""
    if not saved_progress:
        if required:
            raise RuntimeError("M7 resume checkpoint 缺少 joint_progress")
        return Counter(), Counter(), {task: set() for task in parent_populations}
    if saved_progress.get("protocol") != "qpsalm_segdesc_joint_progress_v1":
        raise RuntimeError("M7 resume joint_progress protocol 不一致")
    if int(saved_progress.get("step", -1)) != int(checkpoint_step):
        raise RuntimeError("M7 resume checkpoint step 与 joint_progress step 不一致")
    saved_coverage = dict(saved_progress.get("parent_coverage") or {})
    if set(saved_coverage) != set(parent_populations):
        raise RuntimeError("M7 resume parent coverage task 集合不一致")
    restored_coverage: dict[str, set[str]] = {}
    for task, population in parent_populations.items():
        saved = dict(saved_coverage.get(task) or {})
        expected_hash = hashlib.sha256(
            "\n".join(sorted(population)).encode("utf-8")
        ).hexdigest()
        if str(saved.get("population_sha256") or "") != expected_hash:
            raise RuntimeError(f"M7 resume {task} parent population 已变化")
        if int(saved.get("population", -1)) != len(population):
            raise RuntimeError(f"M7 resume {task} parent population count 不一致")
        covered = {str(value) for value in (saved.get("parent_ids") or [])}
        unknown = sorted(covered - population)
        if unknown:
            raise RuntimeError(f"M7 resume {task} coverage 包含未知 parent: {unknown[:8]}")
        if int(saved.get("covered", -1)) != len(covered):
            raise RuntimeError(f"M7 resume {task} coverage count 不一致")
        covered_hash = hashlib.sha256(
            "\n".join(sorted(covered)).encode("utf-8")
        ).hexdigest()
        if str(saved.get("covered_sha256") or "") != covered_hash:
            raise RuntimeError(f"M7 resume {task} coverage hash 不一致")
        restored_coverage[task] = covered
    task_steps = Counter({
        str(key): int(value)
        for key, value in (saved_progress.get("optimizer_steps") or {}).items()
    })
    task_samples = Counter({
        str(key): int(value)
        for key, value in (saved_progress.get("samples_seen") or {}).items()
    })
    allowed_tasks = set(parent_populations)
    if set(task_steps) - allowed_tasks or set(task_samples) - allowed_tasks:
        raise RuntimeError("M7 resume progress 包含未知 task")
    if any(value < 0 for value in (*task_steps.values(), *task_samples.values())):
        raise RuntimeError("M7 resume progress 包含负计数")
    if sum(task_steps.values()) != int(checkpoint_step):
        raise RuntimeError("M7 resume optimizer step 总数与 checkpoint step 不一致")
    return task_steps, task_samples, restored_coverage


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
    optimizer_manifest = joint_optimizer_manifest(model, optimizer)
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
    parent_populations = {
        "segmentation": _dataset_parent_ids(segmentation_train.dataset),
        "global_caption": _dataset_parent_ids(global_train),
        "region_description": _dataset_parent_ids(region_train),
    }
    saved_progress = dict((resume_metadata.get("metadata") or {}).get("joint_progress") or {})
    task_steps, task_samples, parent_coverage = restore_joint_progress(
        saved_progress,
        parent_populations,
        checkpoint_step=start_step,
        required=bool(resume),
    )
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"

    baseline_path = output_dir / "segmentation_monitor_baseline.json"
    if resume:
        if not baseline_path.is_file():
            raise RuntimeError(
                "M7 resume 需要原 run 的 segmentation_monitor_baseline.json；"
                "禁止用已联合训练的 checkpoint 重新建立基线"
            )
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_identity = monitor_baseline_identity(baseline_report)
        saved_baseline_identity = dict(
            (resume_metadata.get("metadata") or {}).get("segmentation_monitor_baseline_identity")
            or {}
        )
        if baseline_identity != saved_baseline_identity:
            raise RuntimeError("M7 resume monitor baseline 与 checkpoint 身份不一致")
    else:
        with model.controller.adapter_scope("default"):
            baseline_report = evaluate_segmentation(
                model.segmentation, segmentation_val, device,
                threshold=model.segmentation.config.eval_threshold,
            )
        baseline_identity = monitor_baseline_identity(baseline_report)
        write_json(baseline_path, baseline_report)
    baseline_dice = float(baseline_identity["positive_dice"])
    joint_manifest = {
        "protocol": "qpsalm_segdesc_joint_v3_task_isolated",
        "task_pattern": list(config.resolved_joint_task_pattern()),
        "gradient_accumulation_microbatches_per_optimizer_step": config.grad_accum_steps,
        "segmentation_train": len(segmentation_train.dataset),
        "global_caption_train": len(global_train),
        "region_description_train": len(region_train),
        "segmentation_monitor_baseline_positive_dice": baseline_dice,
        "segmentation_monitor_baseline_identity": baseline_identity,
        "retention_max_drop": config.segmentation_retention_max_drop,
        "resolved_config": dict(config.__dict__),
        "initialized_from": initialize_from,
        "shared_segmentation_dense_trainable": bool(
            config.joint_train_shared_segmentation_dense
        ),
        "loader_batches_per_epoch": {
            "segmentation": len(segmentation_train),
            "global_caption": len(global_loader),
            "region_description": len(region_loader),
        },
        "parent_populations": {
            task: {
                "count": len(parents),
                "sha256": hashlib.sha256(
                    "\n".join(sorted(parents)).encode("utf-8")
                ).hexdigest(),
            }
            for task, parents in parent_populations.items()
        },
        "optimizer": optimizer_manifest,
        "monitor_retention_only": True,
        "formal_full_val_retention_required": True,
    }
    write_json(output_dir / "joint_manifest.json", joint_manifest)
    print(
        f"[JOINT-DATA] segmentation={len(segmentation_train.dataset)} "
        f"global_caption={len(global_train)} region_description={len(region_train)}"
    )
    print(
        f"[JOINT-MODEL] pattern={list(config.resolved_joint_task_pattern())} "
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
    # Recheck all three task paths after resume; this is cheap and proves that
    # adapter isolation still holds for the restored optimizer state.
    gradient_gate_seen: set[str] = set()
    gradient_gate_reports: dict[str, Any] = {}
    while step < config.max_steps:
        task = pattern[step % len(pattern)]
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[torch.Tensor] = []
        sample_count = 0
        step_parent_ids: set[str] = set()
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
                    step_parent_ids.update(
                        str(row["parent_sample_id"])
                        for row in raw_batch.metadata
                        if row.get("parent_sample_id")
                    )
                else:
                    batch = move_description_batch(raw_batch, device)
                    active_config = caption_val_config if task == "global_caption" else region_config
                    loss, _diagnostics = _train_loss(model, batch, active_config)
                    sample_count += len(batch["metadata"])
                    step_parent_ids.update(
                        str(row["parent_sample_id"])
                        for row in batch["metadata"]
                        if row.get("parent_sample_id")
                    )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"joint loss 非有限: step={step} micro={micro_index} task={task}"
                )
            micro_losses.append(loss.detach().float())
            scaler.scale(loss / grad_accum).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradient_report = _joint_gradient_report(optimizer)
        if task not in gradient_gate_seen:
            gate = validate_joint_task_gradients(
                task,
                gradient_report,
                train_shared_segmentation_dense=config.joint_train_shared_segmentation_dense,
            )
            if not gate["passed"]:
                raise RuntimeError(
                    f"joint {task} 梯度门禁失败: gate={gate} report={gradient_report}"
                )
            gradient_gate_seen.add(task)
            gradient_gate_reports[task] = {
                "validation": gate,
                "optimizer_roles": gradient_report,
            }
            if gradient_gate_seen == {
                "segmentation", "global_caption", "region_description",
            }:
                write_json(output_dir / "joint_gradient_gate.json", {
                    "protocol": "qpsalm_segdesc_joint_gradient_gate_v2",
                    "reports": gradient_gate_reports,
                    "passed": True,
                })
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups for value in group["params"]],
            config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        task_steps[task] += 1
        task_samples[task] += sample_count
        parent_coverage[task].update(step_parent_ids)
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
            joint_progress = _joint_progress_payload(
                step=step,
                task_steps=task_steps,
                task_samples=task_samples,
                parent_coverage=parent_coverage,
                parent_populations=parent_populations,
            )
            write_json(output_dir / "joint_coverage_latest.json", joint_progress)
            write_json(
                output_dir / "joint_manifest.json",
                {**joint_manifest, "progress": joint_progress},
            )
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
            retention = monitor_retention_gate(
                baseline_identity,
                seg_report,
                maximum_allowed_drop=config.segmentation_retention_max_drop,
            )
            drop = float(retention["absolute_drop"])
            caption_score = float((caption_report["generation_metrics"] or {}).get("caption_token_f1") or 0.0)
            region_score = float(
                ((region_report or {}).get("generation_metrics") or {}).get("structured_field_macro_f1") or 0.0
            )
            retention_passed = bool(retention["passed"])
            composite = caption_score + region_score + current_dice if retention_passed else -math.inf
            validation = {
                "step": step,
                "segmentation_positive_dice": current_dice,
                "segmentation_drop": drop,
                "retention_passed": retention_passed,
                "monitor_retention": retention,
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
                        "segmentation_monitor_baseline_identity": baseline_identity,
                        "validation": validation, "config": dict(config.__dict__),
                        "joint_progress": _joint_progress_payload(
                            step=step,
                            task_steps=task_steps,
                            task_samples=task_samples,
                            parent_coverage=parent_coverage,
                            parent_populations=parent_populations,
                        ),
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
                    "segmentation_monitor_baseline_identity": baseline_identity,
                    "config": dict(config.__dict__),
                    "joint_progress": _joint_progress_payload(
                        step=step,
                        task_steps=task_steps,
                        task_samples=task_samples,
                        parent_coverage=parent_coverage,
                        parent_populations=parent_populations,
                    ),
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
