#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 task-isolated joint-training orchestration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import torch
from tqdm import tqdm

from qpsalm_seg.engine.common import build_dataloaders
from qpsalm_seg.engine.evaluator import evaluate as evaluate_segmentation
from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import SegDescConfig
from ..data.loaders import (
    append_jsonl,
    build_description_dataset,
    build_description_loader,
    description_amp_dtype,
    description_device,
    description_scaler,
    move_description_batch,
    set_description_seed,
    set_loader_epoch,
    validate_predicted_training_indexes,
    write_json,
)
from ..evaluation.d4_curriculum import (
    D4_FINAL_FRACTION,
    validate_d4_final_acceptance_for_m7,
)
from ..evaluation.d_minus_one import revalidate_saved_d_minus_one_acceptance
from ..evaluation.m6_acceptance import (
    revalidate_saved_m6_acceptance,
    validate_m6_acceptance_for_m7,
)
from ..evaluation.runner import evaluate_description
from ..protocols.io import strict_json_loads
from .checkpoint import (
    initialize_segdesc_checkpoint,
    load_segdesc_checkpoint,
    read_segdesc_checkpoint_step,
    save_segdesc_checkpoint,
    validate_description_stage_lineage,
    validate_segmentation_migration_lineage,
    validate_resume_run_config,
)
from .engineering_gates import train_loss
from .joint_contracts import (
    JOINT_LOADER_SEED_OFFSETS,
    JOINT_RUN_PROTOCOL,
)
from .joint_lifecycle import (
    build_joint_initialization_audit,
    joint_progress_payload,
    restore_joint_progress,
    revalidate_joint_initialization_audit,
    validate_joint_checkpoint_execution,
    validate_m7_source_checkpoint,
)
from .joint_runtime import (
    EpochConcatDataset,
    build_joint_optimizer,
    dataset_parent_ids,
    joint_gradient_report,
    joint_loader_binding,
    joint_optimizer_manifest,
    monitor_baseline_identity,
    monitor_retention_gate,
    next_joint_loader_batch,
    positive_dice,
    region_data_audit,
    validate_joint_task_gradients,
)
from .run_artifacts import reconcile_resume_run, validate_checkpoint_run_completion
from .runtime import build_segdesc_model


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
    predicted_training_indexes = validate_predicted_training_indexes(
        config, stage=config.joint.joint_region_stage
    )
    if (
        config.joint.joint_region_stage == "predicted_mask"
        and not config.training.d4_final_acceptance_gate
    ):
        raise ValueError(
            "M7 predicted-mask 主路线必须提供 75% tier 的 --d4-final-acceptance-gate"
        )
    if (
        config.joint.joint_region_stage == "predicted_mask"
        and not config.training.m6_acceptance_gate
    ):
        raise ValueError(
            "M7 predicted-mask 主路线必须提供完整 --m6-acceptance-gate"
        )
    if (
        config.joint.joint_region_stage == "predicted_mask"
        and not math.isclose(
            float(config.data.predicted_mask_fraction),
            D4_FINAL_FRACTION,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "M7 predicted-mask 主路线必须显式使用 --predicted-mask-fraction 0.75"
        )
    set_description_seed(config.training.seed)
    device = description_device(device_name)
    output_dir = resolve_project_path(config.training.output_dir) or Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, migration = build_segdesc_model(config, device)
    bank = model.description_backbone.bank

    loader_seeds = {
        task: int(config.training.seed) + offset
        for task, offset in JOINT_LOADER_SEED_OFFSETS.items()
    }
    seg_config = replace(
        model.segmentation.config,
        batch_size=int(config.joint.joint_segmentation_batch_size),
        grad_accum_steps=1,
        max_train_samples=config.data.max_train_samples or None,
        monitor_val_samples=config.data.max_val_samples or None,
        # 每个 epoch 重建 worker，才能从 epoch/cursor 确定性重放其随机状态。
        persistent_workers=False,
    )
    segmentation_train, segmentation_val = build_dataloaders(seg_config)
    global_configs = [
        config.with_overrides(stage=stage)
        for stage in (
            config.joint.joint_global_stages or ["mmrs_caption", "rsicap_caption"]
        )
    ]
    global_train_sets = [
        build_description_dataset(value, bank, split="train", training=True)
        for value in global_configs
    ]
    global_train = EpochConcatDataset(global_train_sets)
    global_loader = build_description_loader(
        global_train,
        config,
        training=True,
        batch_size=config.joint.joint_description_batch_size,
        sampler_seed=loader_seeds["global_caption"],
    )
    region_config = config.with_overrides(stage=config.joint.joint_region_stage)
    region_train = build_description_dataset(region_config, bank, split="train", training=True)
    region_loader = build_description_loader(
        region_train,
        region_config,
        training=True,
        batch_size=config.joint.joint_description_batch_size,
        sampler_seed=loader_seeds["region_description"],
    )
    loaders = {
        "segmentation": segmentation_train,
        "global_caption": global_loader,
        "region_description": region_loader,
    }
    for task, loader in loaders.items():
        set_loader_epoch(loader, 0, loader_seed=loader_seeds[task])
    loader_bindings = {
        task: joint_loader_binding(
            task, loader, loader_seed=loader_seeds[task]
        )
        for task, loader in loaders.items()
    }
    caption_val_config = config.with_overrides(stage="rsicap_caption")
    caption_val = build_description_loader(
        build_description_dataset(caption_val_config, bank, split="dev", training=False),
        caption_val_config,
        training=False,
        batch_size=config.joint.joint_description_batch_size,
    )
    region_val_name = "val" if config.joint.joint_region_stage in {"bridge_expert", "predicted_mask"} else None
    region_val_config = (
        region_config.with_overrides(evaluation_mode="fixed_prediction")
        if config.joint.joint_region_stage == "predicted_mask"
        else region_config
    )
    region_val = (
        build_description_loader(
            build_description_dataset(
                region_val_config, bank, split=region_val_name, training=False
            ),
            region_val_config,
            training=False,
            batch_size=config.joint.joint_description_batch_size,
        )
        if region_val_name is not None else None
    )
    if not len(global_train) or not len(region_train):
        raise RuntimeError("M7 joint training 需要非空 global-caption 与 region-description 数据")
    if region_val is not None and not len(region_val.dataset):
        raise RuntimeError("M7 joint training 的固定 region val 集为空")
    validation_predicted_index_audit = (
        getattr(region_val.dataset, "predicted_index_audit", None)
        if region_val is not None else None
    )

    optimizer, scheduler = build_joint_optimizer(model, config)
    optimizer_manifest = joint_optimizer_manifest(model, optimizer)
    scaler = description_scaler(config, device)
    start_step = 0
    resume_metadata = {}
    resume_reconciliation: dict[str, Any] | None = None
    source_step: int | None = None
    source_metadata: dict[str, Any] | None = None
    if resume:
        start_step, resume_metadata = load_segdesc_checkpoint(
            resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        validate_resume_run_config(resume_metadata, config.to_dict())
    elif initialize_from:
        source_step, source_metadata = initialize_segdesc_checkpoint(
            initialize_from,
            model,
            expected_seed=config.training.seed,
            require_run_completion=True,
            run_completion_validator=validate_checkpoint_run_completion,
        )
        resume_metadata = {"initialized_from": initialize_from, "source": source_metadata}
    d_minus_one_source = (
        resume_metadata if resume else resume_metadata["source"]
    )
    d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
        (d_minus_one_source.get("metadata") or {}).get(
            "d_minus_one_acceptance"
        ),
        expected_description_benchmark=config.data.description_benchmark,
        expected_bridge_benchmark=config.data.bridge_benchmark,
        expected_unified_benchmark=config.data.unified_benchmark,
        expected_description_cache=config.model.description_vision_cache,
    )
    stage_lineage = validate_description_stage_lineage(
        (d_minus_one_source.get("metadata") or {}).get("stage_lineage"),
        expected_target_stage=config.joint.joint_region_stage,
    )
    current_region_data_audit = region_data_audit(region_train)
    initialization_audit = validate_m7_source_checkpoint(
        resume_metadata if resume else resume_metadata["source"],
        region_stage=config.joint.joint_region_stage,
        current_data_audit=current_region_data_audit,
        resume=bool(resume),
        expected_seed=config.training.seed,
    )
    d4_final_acceptance: dict[str, Any] | None = None
    m6_acceptance: dict[str, Any] | None = None
    if config.joint.joint_region_stage == "predicted_mask":
        if resume:
            saved_acceptance = dict(
                (resume_metadata.get("metadata") or {}).get(
                    "d4_final_acceptance"
                ) or {}
            )
            if not saved_acceptance:
                raise RuntimeError("M7 resume checkpoint 缺少 D4 final acceptance audit")
            acceptance_source = saved_acceptance.get("source_checkpoint") or ""
        else:
            saved_acceptance = None
            acceptance_source = initialize_from or ""
        d4_final_acceptance = validate_d4_final_acceptance_for_m7(
            config.training.d4_final_acceptance_gate,
            seed=config.training.seed,
            initialize_from=acceptance_source,
            expert_gate_audit=dict(
                getattr(region_train, "expert_gate_audit", None) or {}
            ),
            train_region_data_audit=current_region_data_audit,
            val_predicted_index_audit=dict(
                validation_predicted_index_audit or {}
            ),
        )
        if saved_acceptance is not None and saved_acceptance != d4_final_acceptance:
            raise RuntimeError("M7 resume D4 final gate audit 与 checkpoint 不一致")
        if resume:
            m6_acceptance = revalidate_saved_m6_acceptance(
                (resume_metadata.get("metadata") or {}).get("m6_acceptance"),
                seed=config.training.seed,
                train_region_data_audit=current_region_data_audit,
            )
        else:
            m6_acceptance = validate_m6_acceptance_for_m7(
                config.training.m6_acceptance_gate or "",
                seed=config.training.seed,
                initialize_from=initialize_from or "",
                train_region_data_audit=current_region_data_audit,
            )
        if (
            m6_acceptance.get("d4_final_acceptance")
            != d4_final_acceptance
            or m6_acceptance.get("d_minus_one_acceptance")
            != d_minus_one_acceptance
        ):
            raise RuntimeError("M7 的 D-1/D4 与完整 M6 acceptance 不一致")
    if resume:
        joint_initialization_audit = revalidate_joint_initialization_audit(
            (resume_metadata.get("metadata") or {}).get(
                "joint_initialization_audit"
            ),
            expected_seed=config.training.seed,
            region_stage=config.joint.joint_region_stage,
            region_data_audit=current_region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=migration,
            require_m6_binding=config.joint.joint_region_stage == "predicted_mask",
        )
        resume_migration_lineage = validate_segmentation_migration_lineage(
            migration, resume_metadata
        )
        if (
            (resume_metadata.get("metadata") or {}).get(
                "segmentation_migration_lineage"
            ) != resume_migration_lineage
            or joint_initialization_audit.get(
                "segmentation_migration_lineage"
            ) != resume_migration_lineage
        ):
            raise RuntimeError("M7 resume segmentation migration lineage 已漂移")
    else:
        if source_step is None or source_metadata is None or not initialize_from:
            raise RuntimeError("M7 initialize 缺少 source checkpoint loader provenance")
        joint_initialization_audit = build_joint_initialization_audit(
            initialize_from,
            expected_seed=config.training.seed,
            region_stage=config.joint.joint_region_stage,
            region_data_audit=current_region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=migration,
            source_step=source_step,
            source_initialization=source_metadata.get("initialization"),
            require_m6_binding=config.joint.joint_region_stage == "predicted_mask",
        )
    segmentation_migration_lineage = joint_initialization_audit[
        "segmentation_migration_lineage"
    ]
    parent_populations = {
        "segmentation": dataset_parent_ids(segmentation_train.dataset),
        "global_caption": dataset_parent_ids(global_train),
        "region_description": dataset_parent_ids(region_train),
    }
    saved_progress = dict((resume_metadata.get("metadata") or {}).get("joint_progress") or {})
    pattern = config.resolved_joint_task_pattern()
    grad_accum = max(1, int(config.training.grad_accum_steps))
    task_steps, task_samples, parent_coverage, loader_states = restore_joint_progress(
        saved_progress,
        parent_populations,
        loader_bindings,
        checkpoint_step=start_step,
        required=bool(resume),
        task_pattern=pattern,
        grad_accum_steps=grad_accum,
    )
    if resume:
        resume_reconciliation = reconcile_resume_run(
            output_dir,
            resume_checkpoint=resume,
            checkpoint_step=start_step,
            histories={
                "joint_history.jsonl": start_step > 0,
                "joint_validation_history.jsonl": False,
            },
            checkpoint_step_reader=read_segdesc_checkpoint_step,
        )
        write_json(output_dir / "joint_coverage_latest.json", saved_progress)
    resume_execution_audit = (
        validate_joint_checkpoint_execution(
            dict(resume_metadata.get("metadata") or {}),
            checkpoint_step=start_step,
        )
        if resume else None
    )
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.training.amp_dtype != "fp32"

    baseline_path = output_dir / "segmentation_monitor_baseline.json"
    if resume:
        if not baseline_path.is_file():
            raise RuntimeError(
                "M7 resume 需要原 run 的 segmentation_monitor_baseline.json；"
                "禁止用已联合训练的 checkpoint 重新建立基线"
            )
        baseline_report = strict_json_loads(
            baseline_path.read_text(encoding="utf-8")
        )
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
        "protocol": JOINT_RUN_PROTOCOL,
        "task_pattern": list(pattern),
        "gradient_accumulation_microbatches_per_optimizer_step": config.training.grad_accum_steps,
        "segmentation_train": len(segmentation_train.dataset),
        "global_caption_train": len(global_train),
        "global_caption_sampling_audit": getattr(
            global_train, "caption_sampling_audit", None
        ),
        "region_description_train": len(region_train),
        "region_expert_gate_audit": getattr(
            region_train, "expert_gate_audit", None
        ),
        "region_predicted_index_audit": getattr(
            region_train, "predicted_index_audit", None
        ),
        "predicted_training_indexes": predicted_training_indexes,
        "d4_final_acceptance": d4_final_acceptance,
        "m6_acceptance": m6_acceptance,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
        "segmentation_monitor_baseline_positive_dice": baseline_dice,
        "segmentation_monitor_baseline_identity": baseline_identity,
        "retention_max_drop": config.joint.segmentation_retention_max_drop,
        "resolved_config": config.to_dict(),
        "initialized_from": initialize_from,
        "initialization_audit": initialization_audit,
        "joint_initialization_audit": joint_initialization_audit,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "resume_execution_audit": resume_execution_audit,
        "resume_reconciliation": resume_reconciliation,
        "shared_segmentation_dense_trainable": bool(
            config.joint.joint_train_shared_segmentation_dense
        ),
        "loader_bindings": loader_bindings,
        "loader_batches_per_epoch": {
            task: int(binding["batches_per_epoch"])
            for task, binding in loader_bindings.items()
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
        f"shared_segmentation_dense={config.joint.joint_train_shared_segmentation_dense} "
        f"baseline_positive_dice={baseline_dice:.4f}"
    )

    # Iterators remain lazy so resume can reconstruct only the next requested stream.
    iterators = {task: None for task in loaders}
    history_path = output_dir / "joint_history.jsonl"
    saved_best_score = (resume_metadata.get("metadata") or {}).get("best_score")
    best_score = -math.inf if saved_best_score is None else float(saved_best_score)
    progress = tqdm(total=config.training.max_steps, initial=start_step, desc="qpsalm-segdesc-joint")
    step = start_step
    window: list[dict[str, Any]] = []
    window_started = time.perf_counter()
    model.train()
    # Recheck all three task paths after resume; this is cheap and proves that
    # adapter isolation still holds for the restored optimizer state.
    gradient_gate_seen: set[str] = set()
    gradient_gate_reports: dict[str, Any] = {}
    while step < config.training.max_steps:
        task = pattern[step % len(pattern)]
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[torch.Tensor] = []
        sample_count = 0
        step_parent_ids: set[str] = set()
        for micro_index in range(grad_accum):
            raw_batch, iterators[task] = next_joint_loader_batch(
                loaders[task],
                iterators[task],
                loader_states[task],
                loader_bindings[task],
            )
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
                    loss, _diagnostics = train_loss(model, batch, active_config)
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
        gradient_report = joint_gradient_report(optimizer)
        if task not in gradient_gate_seen:
            gate = validate_joint_task_gradients(
                task,
                gradient_report,
                train_shared_segmentation_dense=config.joint.joint_train_shared_segmentation_dense,
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
            config.training.max_grad_norm,
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
        if step == 1 or step % config.training.log_interval == 0 or step == config.training.max_steps:
            elapsed = time.perf_counter() - window_started
            by_task = {}
            for name in sorted(set(value["task"] for value in window)):
                values = [value["loss"] for value in window if value["task"] == name]
                by_task[name] = {"steps": len(values), "loss": sum(values) / len(values)}
            record = {
                "step": step,
                "by_task": by_task,
                "samples_per_second": sum(
                    value["samples"] for value in window
                ) / max(elapsed, 1.0e-9),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device.type == "cuda" else 0.0
                ),
            }
            append_jsonl(history_path, record)
            joint_progress = joint_progress_payload(
                step=step,
                task_steps=task_steps,
                task_samples=task_samples,
                parent_coverage=parent_coverage,
                parent_populations=parent_populations,
                loader_states=loader_states,
                loader_bindings=loader_bindings,
                task_pattern=pattern,
                grad_accum_steps=grad_accum,
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

        if step % config.training.val_interval == 0 or step == config.training.max_steps:
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
                    model, region_val, region_val_config, device, split=str(region_val_name),
                    output_dir=output_dir / "validation_region_latest", run_counterfactuals=False,
                )
                if region_val is not None else None
            )
            current_dice = positive_dice(seg_report)
            retention = monitor_retention_gate(
                baseline_identity,
                seg_report,
                maximum_allowed_drop=config.joint.segmentation_retention_max_drop,
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
                # 冻结 checkpoint_best 的选择证据，不能用随后覆盖的 latest 代替。
                write_json(
                    output_dir / "joint_validation_best.json",
                    validation,
                )
                save_segdesc_checkpoint(
                    output_dir / "checkpoint_best.pt", model, step=step,
                    segmentation_migration=migration, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler,
                    metadata={
                        "stage": "joint",
                        "checkpoint_role": "validation_best",
                        "best_score": (
                            best_score if math.isfinite(best_score) else None
                        ),
                        "joint_run_protocol": JOINT_RUN_PROTOCOL,
                        "joint_loader_bindings": loader_bindings,
                        "segmentation_monitor_baseline_positive_dice": baseline_dice,
                        "segmentation_monitor_baseline_identity": baseline_identity,
                        "validation": validation, "config": config.to_dict(),
                        "region_data_audit": current_region_data_audit,
                        "d4_final_acceptance": d4_final_acceptance,
                        "m6_acceptance": m6_acceptance,
                        "joint_initialization_audit": joint_initialization_audit,
                        "segmentation_migration_lineage": (
                            segmentation_migration_lineage
                        ),
                        "d_minus_one_acceptance": d_minus_one_acceptance,
                        "stage_lineage": stage_lineage,
                        "resume_reconciliation": resume_reconciliation,
                        "joint_progress": joint_progress_payload(
                            step=step,
                            task_steps=task_steps,
                            task_samples=task_samples,
                            parent_coverage=parent_coverage,
                            parent_populations=parent_populations,
                            loader_states=loader_states,
                            loader_bindings=loader_bindings,
                            task_pattern=pattern,
                            grad_accum_steps=grad_accum,
                        ),
                    },
                )
                tqdm.write(f"[JOINT-CKPT] saved=checkpoint_best.pt step={step}")
            model.train()
        if step % config.training.save_interval == 0 or step == config.training.max_steps:
            save_segdesc_checkpoint(
                output_dir / "checkpoint_last.pt", model, step=step,
                segmentation_migration=migration, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler,
                metadata={
                    "stage": "joint",
                    "checkpoint_role": "terminal_last",
                    "best_score": (
                        best_score if math.isfinite(best_score) else None
                    ),
                    "joint_run_protocol": JOINT_RUN_PROTOCOL,
                    "joint_loader_bindings": loader_bindings,
                    "segmentation_monitor_baseline_positive_dice": baseline_dice,
                    "segmentation_monitor_baseline_identity": baseline_identity,
                    "config": config.to_dict(),
                    "region_data_audit": current_region_data_audit,
                    "d4_final_acceptance": d4_final_acceptance,
                    "m6_acceptance": m6_acceptance,
                    "joint_initialization_audit": joint_initialization_audit,
                    "segmentation_migration_lineage": (
                        segmentation_migration_lineage
                    ),
                    "d_minus_one_acceptance": d_minus_one_acceptance,
                    "stage_lineage": stage_lineage,
                    "resume_reconciliation": resume_reconciliation,
                    "joint_progress": joint_progress_payload(
                        step=step,
                        task_steps=task_steps,
                        task_samples=task_samples,
                        parent_coverage=parent_coverage,
                        parent_populations=parent_populations,
                        loader_states=loader_states,
                        loader_bindings=loader_bindings,
                        task_pattern=pattern,
                        grad_accum_steps=grad_accum,
                    ),
                },
            )
            tqdm.write(f"[JOINT-CKPT] saved=checkpoint_last.pt step={step}")
    progress.close()
    return {
        "output_dir": str(output_dir),
        "stage": "joint",
        "steps": step,
        "best_score": best_score if math.isfinite(best_score) else None,
        "checkpoint_best": (
            str(output_dir / "checkpoint_best.pt")
            if (output_dir / "checkpoint_best.pt").is_file() else None
        ),
        "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "d4_final_acceptance": d4_final_acceptance,
        "m6_acceptance": m6_acceptance,
        "joint_initialization_audit": joint_initialization_audit,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "stage_lineage": stage_lineage,
        "resume_reconciliation": resume_reconciliation,
    }
