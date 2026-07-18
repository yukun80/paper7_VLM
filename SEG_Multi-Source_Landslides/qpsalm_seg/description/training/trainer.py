#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-GPU D-1/D0-D4 trainer for segmentation-grounded description."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import torch
from tqdm import tqdm

from qpsalm_seg.paths import resolve_project_path

from .checkpoint import (
    build_description_stage_lineage,
    initialize_segdesc_checkpoint,
    load_segdesc_checkpoint,
    read_segdesc_checkpoint_step,
    save_segdesc_checkpoint,
    validate_description_stage_lineage,
    validate_segmentation_migration_lineage,
    validate_resume_run_config,
    verify_segdesc_checkpoint_reload,
)
from ..data.loaders import (
    append_jsonl,
    build_description_dataset,
    build_description_loader,
    description_collator_audit,
    description_amp_dtype,
    description_device,
    description_scaler,
    move_description_batch,
    set_description_seed,
    validate_predicted_training_indexes,
    validation_split,
    write_json,
)
from ..data.artifact_readiness import ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL
from ..protocols.config import SegDescConfig
from ..evaluation.runner import description_selection_score, evaluate_description
from ..protocols.io import strict_json_loads
from ..protocols.stages import DESCRIPTION_STREAM_SEED_OFFSETS, get_stage_spec
from ..protocols.versions import (
    D0_CONSTRUCTION_CONTRACT_PROTOCOL,
    D0_PREFLIGHT_ACCEPTANCE_PROTOCOL,
)
from ..evaluation.d4_curriculum import validate_d4_curriculum_transition
from ..evaluation.d_minus_one import (
    revalidate_saved_d_minus_one_acceptance,
    validate_d_minus_one_gate,
)
from .runtime import (
    build_description_optimizer,
    build_segdesc_model,
    description_optimizer_audit,
    description_trainable_parameter_manifest,
)
from .run_artifacts import reconcile_resume_run, validate_checkpoint_run_completion
from .engineering_gates import (
    build_d_minus_one_overfit_validation,
    dataset_data_audit,
    desc_adapter_parameters,
    region_data_audit,
    train_loss,
)
from .gradient_gates import DescriptionGradientGateTracker
from .streams import (
    description_stream_binding,
    description_training_progress_payload,
    load_best,
    next_description_stream_batch,
    restore_description_training_progress,
)


def train_description(
    config: SegDescConfig,
    *,
    device_name: str,
    resume: str | None = None,
    initialize_from: str | None = None,
    artifact_readiness_acceptance: dict[str, Any] | None = None,
    d0_preflight_acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resume and initialize_from:
        raise ValueError("--resume 与 --initialize-from 不能同时使用")
    stage_spec = get_stage_spec(config.training.stage)
    if config.training.stage == "overfit":
        if (
            not isinstance(artifact_readiness_acceptance, dict)
            or artifact_readiness_acceptance.get("protocol")
            != ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL
            or artifact_readiness_acceptance.get("status")
            != "engineering-valid"
            or artifact_readiness_acceptance.get("errors") != []
        ):
            raise ValueError(
                "D-1 overfit 缺少当前 artifact readiness acceptance"
            )
    elif artifact_readiness_acceptance is not None:
        raise ValueError(
            "artifact readiness acceptance 只允许 D-1 overfit 直接消费"
        )
    if config.training.stage == "mmrs_caption":
        if (
            not isinstance(d0_preflight_acceptance, dict)
            or d0_preflight_acceptance.get("protocol")
            != D0_PREFLIGHT_ACCEPTANCE_PROTOCOL
            or d0_preflight_acceptance.get("status")
            != "engineering-valid"
            or d0_preflight_acceptance.get("errors") != []
        ):
            raise ValueError("D0 trainer 缺少当前 preflight acceptance")
    elif d0_preflight_acceptance is not None:
        raise ValueError("D0 preflight acceptance 只允许 mmrs_caption 消费")
    if stage_spec.initialization_kind == "segmentation_checkpoint" and initialize_from:
        raise ValueError(
            f"stage={config.training.stage} 必须从 segmentation checkpoint 新建，"
            "不能使用 --initialize-from"
        )
    if (
        stage_spec.initialization_kind == "previous_stage_checkpoint"
        and not (resume or initialize_from)
    ):
        raise ValueError(
            f"stage={config.training.stage} 必须从 {stage_spec.initialize_from_stage} "
            f"的 {stage_spec.initialize_from_checkpoint_role} checkpoint "
            "使用 --initialize-from，"
            "或使用同阶段 --resume"
        )
    predicted_training_indexes = validate_predicted_training_indexes(
        config, stage=config.training.stage
    )
    if (
        "d4_curriculum_transition" in stage_spec.gate_requirements
        and not config.training.d4_curriculum_gate
    ):
        raise ValueError(
            "D4 predicted-mask training 必须提供前一档通过的 --d4-curriculum-gate"
        )
    d_minus_one_acceptance: dict[str, Any] | None = None
    if config.training.stage == "mmrs_caption":
        d_minus_one_acceptance = validate_d_minus_one_gate(
            str(config.training.d_minus_one_gate or ""),
            expected_description_benchmark=config.data.description_benchmark,
            expected_bridge_benchmark=config.data.bridge_benchmark,
            expected_unified_benchmark=config.data.unified_benchmark,
            expected_description_cache=(
                config.model.description_vision_cache
            ),
        )
    set_description_seed(config.training.seed)
    device = description_device(device_name)
    output_dir = resolve_project_path(config.training.output_dir) or Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, migration = build_segdesc_model(config, device)
    bank = model.description_backbone.bank
    train_dataset = build_description_dataset(config, bank, split="train", training=True)
    if not len(train_dataset):
        raise RuntimeError(f"description stage={config.training.stage} 训练集为空")
    if config.training.stage == "dior_alignment" and config.training.batch_size < 2:
        raise ValueError("dior_alignment 需要 batch_size >= 2 才能形成对比负样本")
    train_streams = {
        "main": {
            "config": config,
            "dataset": train_dataset,
            "loader": build_description_loader(
                train_dataset,
                config,
                training=True,
                sampler_seed=(
                    int(config.training.seed)
                    + DESCRIPTION_STREAM_SEED_OFFSETS["main"]
                ),
            ),
        }
    }
    stream_pattern = ("main",)
    if config.training.stage == "bridge_expert":
        # D3b keeps the three supervision types in independent DataLoaders.
        # This avoids mixing contrastive DIOR rows with causal JSON rows in one
        # collate while preserving the documented 60/20/20 task schedule.
        dior_config = config.with_overrides(stage="dior_alignment")
        global_config = config.with_overrides(stage="rsicap_caption")
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
                "loader": build_description_loader(
                    train_dataset,
                    config,
                    training=True,
                    sampler_seed=(
                        int(config.training.seed)
                        + DESCRIPTION_STREAM_SEED_OFFSETS["bridge"]
                    ),
                ),
            },
            "dior": {
                "config": dior_config,
                "dataset": dior_dataset,
                "loader": build_description_loader(
                    dior_dataset, dior_config, training=True,
                    batch_size=max(2, int(config.training.batch_size)),
                    sampler_seed=(
                        int(config.training.seed)
                        + DESCRIPTION_STREAM_SEED_OFFSETS["dior"]
                    ),
                ),
            },
            "global_caption": {
                "config": global_config,
                "dataset": global_dataset,
                "loader": build_description_loader(
                    global_dataset,
                    global_config,
                    training=True,
                    sampler_seed=(
                        int(config.training.seed)
                        + DESCRIPTION_STREAM_SEED_OFFSETS["global_caption"]
                    ),
                ),
            },
        }
        stream_pattern = tuple(config.data.bridge_expert_task_pattern or [
            "bridge", "bridge", "bridge", "dior", "global_caption",
        ])
    val_name = validation_split(config.training.stage)
    val_loader = None
    validation_config = (
        config.with_overrides(evaluation_mode="fixed_prediction")
        if config.training.stage == "predicted_mask" else config
    )
    if val_name is not None:
        val_dataset = build_description_dataset(
            validation_config, bank, split=val_name, training=False
        )
        if len(val_dataset):
            val_loader = build_description_loader(
                val_dataset, validation_config, training=False
            )

    training_data_audits = {
        name: dataset_data_audit(value["dataset"])
        for name, value in train_streams.items()
    }
    d0_collator_audit = (
        description_collator_audit(next(iter(train_streams["main"]["loader"])))
        if d0_preflight_acceptance is not None else None
    )
    stream_loader_bindings = {
        name: description_stream_binding(
            name, train_streams[name], training_data_audits[name]
        )
        for name in train_streams
    }
    validation_data_audit = (
        dataset_data_audit(val_loader.dataset) if val_loader is not None else None
    )
    checkpoint_data_audit = {
        "protocol": "qpsalm_description_training_data_binding_v2_loader_bound",
        "stage_spec": stage_spec.to_dict(),
        "training_streams": training_data_audits,
        "stream_loader_bindings": stream_loader_bindings,
        "validation": validation_data_audit,
        "stream_pattern": list(stream_pattern),
        "artifact_readiness_acceptance": (
            artifact_readiness_acceptance
        ),
        "d0_preflight_acceptance": d0_preflight_acceptance,
        "d0_collator_audit": d0_collator_audit,
    }
    checkpoint_region_data_audit = region_data_audit(train_dataset)
    validation_predicted_index_audit = (
        getattr(val_loader.dataset, "predicted_index_audit", None)
        if val_loader is not None else None
    )

    optimizer, scheduler = build_description_optimizer(model, config)
    trainable_manifest = description_trainable_parameter_manifest(
        model, optimizer.param_groups, stage=config.training.stage
    )
    if d0_preflight_acceptance is not None:
        current_construction = {
            "protocol": D0_CONSTRUCTION_CONTRACT_PROTOCOL,
            "segmentation_migration": migration,
            "dataset": training_data_audits["main"],
            "collator": d0_collator_audit,
            "loader": {
                "batches": len(train_streams["main"]["loader"]),
                "num_workers": int(config.data.num_workers),
                "batch_sampler": type(
                    train_streams["main"]["loader"].batch_sampler
                ).__name__,
                "stream_binding": stream_loader_bindings["main"],
            },
            "trainable_parameters": trainable_manifest,
            "optimizer": description_optimizer_audit(optimizer, scheduler),
        }
        if current_construction != d0_preflight_acceptance.get(
            "construction_contract"
        ):
            raise RuntimeError(
                "D0 trainer 实际 model/data/collator/loader/optimizer "
                "构造偏离 preflight acceptance"
            )
    write_json(output_dir / "trainable_parameter_manifest.json", trainable_manifest)
    scaler = description_scaler(config, device)
    start_step = 0
    resume_metadata: dict[str, Any] = {}
    resume_reconciliation: dict[str, Any] | None = None
    d4_curriculum_transition: dict[str, Any] | None = None
    stage_lineage: dict[str, Any] | None = None
    segmentation_migration_lineage = validate_segmentation_migration_lineage(
        migration, {"segmentation_migration": migration}
    )
    resolved = config.to_dict()
    if resume:
        start_step, resume_metadata = load_segdesc_checkpoint(
            resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage=config.training.stage,
        )
        validate_resume_run_config(resume_metadata, resolved)
        resumed_migration_lineage = validate_segmentation_migration_lineage(
            migration, resume_metadata
        )
        if (
            (resume_metadata.get("metadata") or {}).get(
                "segmentation_migration_lineage"
            ) != resumed_migration_lineage
        ):
            raise RuntimeError(
                "resume checkpoint segmentation migration lineage 已漂移"
            )
        segmentation_migration_lineage = resumed_migration_lineage
        if config.training.stage != "overfit":
            saved_d_minus_one = revalidate_saved_d_minus_one_acceptance(
                (resume_metadata.get("metadata") or {}).get(
                    "d_minus_one_acceptance"
                ),
                expected_description_benchmark=config.data.description_benchmark,
                expected_bridge_benchmark=config.data.bridge_benchmark,
                expected_unified_benchmark=config.data.unified_benchmark,
                expected_description_cache=(
                    config.model.description_vision_cache
                ),
            )
            if (
                d_minus_one_acceptance is not None
                and saved_d_minus_one != d_minus_one_acceptance
            ):
                raise RuntimeError("D0 resume 的 D-1 gate 与 checkpoint 不一致")
            d_minus_one_acceptance = saved_d_minus_one
        stage_lineage = (resume_metadata.get("metadata") or {}).get(
            "stage_lineage"
        )
        if stage_spec.requires_initialize_from:
            stage_lineage = validate_description_stage_lineage(
                stage_lineage,
                expected_target_stage=config.training.stage,
            )
        saved_data_audit = (resume_metadata.get("metadata") or {}).get("data_audit")
        if saved_data_audit != checkpoint_data_audit:
            raise RuntimeError(
                "resume checkpoint 的 description data population/sampling policy "
                "与当前运行不一致"
            )
        if config.training.stage == "predicted_mask":
            saved_transition = dict(
                (resume_metadata.get("metadata") or {}).get(
                    "d4_curriculum_transition"
                ) or {}
            )
            if not saved_transition:
                raise RuntimeError("D4 resume checkpoint 缺少 curriculum transition audit")
            d4_curriculum_transition = validate_d4_curriculum_transition(
                config.training.d4_curriculum_gate,
                target_fraction=config.data.predicted_mask_fraction,
                seed=config.training.seed,
                initialize_from=saved_transition.get("source_checkpoint") or "",
                expert_gate_audit=dict(
                    getattr(train_dataset, "expert_gate_audit", None) or {}
                ),
                train_region_data_audit=dict(checkpoint_region_data_audit or {}),
                val_predicted_index_audit=dict(
                    validation_predicted_index_audit or {}
                ),
            )
            if saved_transition != d4_curriculum_transition:
                raise RuntimeError("D4 resume curriculum gate audit 与 checkpoint 不一致")
    elif initialize_from:
        _source_step, source_metadata = initialize_segdesc_checkpoint(
            initialize_from, model, target_stage=config.training.stage,
            expected_seed=config.training.seed,
            run_completion_validator=validate_checkpoint_run_completion,
            allow_same_stage_curriculum=(
                config.training.stage == "predicted_mask"
                and float(config.data.predicted_mask_fraction) > 0.25
            ),
        )
        resume_metadata = {
            "initialized_from": str(initialize_from),
            "source": source_metadata,
        }
        segmentation_migration_lineage = validate_segmentation_migration_lineage(
            migration, source_metadata
        )
        if (
            (source_metadata.get("metadata") or {}).get(
                "segmentation_migration_lineage"
            ) != segmentation_migration_lineage
        ):
            raise RuntimeError(
                "initialize-from checkpoint segmentation migration lineage 缺失或漂移"
            )
        d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
            (source_metadata.get("metadata") or {}).get(
                "d_minus_one_acceptance"
            ),
            expected_description_benchmark=config.data.description_benchmark,
            expected_bridge_benchmark=config.data.bridge_benchmark,
            expected_unified_benchmark=config.data.unified_benchmark,
            expected_description_cache=(
                config.model.description_vision_cache
            ),
        )
        stage_lineage = build_description_stage_lineage(
            source_metadata, target_stage=config.training.stage
        )
        if config.training.stage == "predicted_mask":
            d4_curriculum_transition = validate_d4_curriculum_transition(
                config.training.d4_curriculum_gate,
                target_fraction=config.data.predicted_mask_fraction,
                seed=config.training.seed,
                initialize_from=initialize_from,
                expert_gate_audit=dict(
                    getattr(train_dataset, "expert_gate_audit", None) or {}
                ),
                train_region_data_audit=dict(checkpoint_region_data_audit or {}),
                val_predicted_index_audit=dict(
                    validation_predicted_index_audit or {}
                ),
            )
    grad_accum = max(1, int(config.training.grad_accum_steps))
    saved_training_progress = (
        dict((resume_metadata.get("metadata") or {}).get("training_progress") or {})
        if resume else {}
    )
    stream_states = restore_description_training_progress(
        saved_training_progress,
        checkpoint_step=start_step,
        required=bool(resume),
        stream_pattern=stream_pattern,
        grad_accum_steps=grad_accum,
        train_streams=train_streams,
        stream_bindings=stream_loader_bindings,
    )
    if resume:
        resume_reconciliation = reconcile_resume_run(
            output_dir,
            resume_checkpoint=resume,
            checkpoint_step=start_step,
            histories={
                "train_history.jsonl": start_step > 0,
                "validation_history.jsonl": False,
            },
            checkpoint_step_reader=read_segdesc_checkpoint_step,
        )
        # Active progress must describe the restored state, not an uncheckpointed
        # tail that may have been written before the previous process stopped.
        write_json(output_dir / "training_progress_latest.json", saved_training_progress)
    write_json(output_dir / "resolved_config.json", resolved)
    write_json(output_dir / "dataset_summary.json", {
        "stage": config.training.stage,
        "stage_spec": stage_spec.to_dict(),
        "train_split": "train",
        "train_samples": len(train_dataset),
        "training_streams": {
            name: {
                "stage": value["config"].training.stage,
                "samples": len(value["dataset"]),
                "batch_size": (
                    value["loader"].batch_size
                    or getattr(value["loader"].batch_sampler, "batch_size", None)
                ),
                "caption_sampling_audit": getattr(
                    value["dataset"], "caption_sampling_audit", None
                ),
                "curriculum_audit": getattr(
                    value["dataset"], "curriculum_audit", None
                ),
                "data_audit": training_data_audits[name],
            }
            for name, value in train_streams.items()
        },
        "expert_gate_audit": getattr(train_dataset, "expert_gate_audit", None),
        "bridge_engineering_audit": getattr(
            train_dataset, "bridge_engineering_audit", None
        ),
        "description_engineering_audit": getattr(
            train_dataset, "description_engineering_audit", None
        ),
        "predicted_index_audit": getattr(train_dataset, "predicted_index_audit", None),
        "predicted_training_indexes": predicted_training_indexes,
        "d4_curriculum_transition": d4_curriculum_transition,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "validation_expert_gate_audit": (
            getattr(val_loader.dataset, "expert_gate_audit", None)
            if val_loader is not None else None
        ),
        "stream_pattern": list(stream_pattern),
        "validation_split": val_name,
        "validation_evaluation_mode": (
            validation_config.evaluation.evaluation_mode
        ),
        "validation_samples": len(val_loader.dataset) if val_loader is not None else 0,
        "validation_data_audit": validation_data_audit,
        "initialized_from": initialize_from,
        "resume_reconciliation": resume_reconciliation,
        "d_minus_one_sampling_audit": getattr(
            train_dataset, "d_minus_one_sampling_audit", None
        ),
        "artifact_readiness_acceptance": (
            artifact_readiness_acceptance
        ),
        "d0_preflight_acceptance": d0_preflight_acceptance,
    })
    print(
        f"[DESC-DATA] stage={config.training.stage} train={len(train_dataset)} "
        f"val={len(val_loader.dataset) if val_loader is not None else 0}"
    )
    print(
        f"[DESC-MODEL] protocol={config.model.region_protocol} "
        f"precision={config.training.amp_dtype} "
        f"batch={config.training.batch_size} "
        f"ga={config.training.grad_accum_steps} "
        f"max_steps={config.training.max_steps}"
    )
    best_path = output_dir / "validation_best.json"
    saved_best_score = (resume_metadata.get("metadata") or {}).get("best_score")
    best_score = (
        load_best(best_path)
        if saved_best_score is None
        else float(saved_best_score)
    )
    history_path = output_dir / "train_history.jsonl"
    validation_history = output_dir / "validation_history.jsonl"
    desc_parameters = desc_adapter_parameters(model)
    if not desc_parameters:
        raise RuntimeError("description trainer 未找到 desc_adapter LoRA 参数")
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.training.amp_dtype != "fp32"
    iterators = {name: None for name in train_streams}
    progress = tqdm(total=config.training.max_steps, initial=start_step, desc="qpsalm-description")
    window_loss = window_samples = 0.0
    window_steps = 0
    window_auxiliary: dict[str, list[float]] = {}
    window_started = time.perf_counter()
    # Resume 后也重新验证每条路径的梯度隔离，避免只信任旧进程内状态。
    gradient_gate = DescriptionGradientGateTracker(
        train_streams, run_stage=config.training.stage
    )
    last_validation_report: dict[str, Any] | None = None
    if (
        resume
        and config.training.stage == "overfit"
        and start_step == int(config.training.max_steps)
    ):
        gradient_gate.restore_completed(
            (resume_metadata.get("metadata") or {}).get("gradient_gate")
        )
        recovered_validation = strict_json_loads(
            (output_dir / "validation_latest/eval_report.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(recovered_validation, dict):
            raise RuntimeError(
                "D-1 terminal recovery validation report 必须为 JSON object"
            )
        last_validation_report = recovered_validation
    step = start_step
    model.train()
    while step < config.training.max_steps:
        stream_name = stream_pattern[step % len(stream_pattern)]
        stream = train_streams[stream_name]
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_samples = 0
        window_task_paths: set[str] = set()
        for _ in range(grad_accum):
            cpu_batch, iterators[stream_name] = next_description_stream_batch(
                stream,
                iterators[stream_name],
                stream_states[stream_name],
                stream_loader_bindings[stream_name],
            )
            batch = move_description_batch(cpu_batch, device)
            if config.training.stage == "overfit":
                window_task_paths.update(
                    gradient_gate.task_paths(batch["use_region_tokens"])
                )
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=autocast
            ):
                loss, diagnostics = train_loss(model, batch, stream["config"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"description loss 非有限: step={step}")
            scaler.scale(loss / grad_accum).backward()
            step_loss += float(loss.detach().cpu())
            step_samples += len(batch["metadata"])
            for name, value in diagnostics.items():
                window_auxiliary.setdefault(name, []).append(float(value))
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if not gradient_gate.stream_complete(stream_name):
            window_gradient_gate = gradient_gate.audit_window(
                model,
                optimizer,
                stream_name=stream_name,
                stream_stage=stream["config"].training.stage,
                observed_task_paths=window_task_paths,
            )
            if not window_gradient_gate["passed"]:
                raise RuntimeError(
                    "description stage-aware 梯度门禁失败；"
                    f"stream={stream_name} report={window_gradient_gate}"
                )
            write_json(
                output_dir / "description_gradient_gate.json",
                gradient_gate.payload(),
            )
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups for value in group["params"]],
            config.training.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        progress.update(1)
        step_mean = step_loss / grad_accum
        window_loss += step_mean
        window_samples += step_samples
        window_steps += 1

        if step == 1 or step % config.training.log_interval == 0 or step == config.training.max_steps:
            elapsed = time.perf_counter() - window_started
            row = {
                "step": step,
                "epochs": {
                    name: int(state["epoch"])
                    for name, state in stream_states.items()
                },
                "loss": window_loss / max(window_steps, 1),
                "samples_per_second": window_samples / max(elapsed, 1.0e-9),
                "learning_rates": {
                    str(group.get("name")): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device.type == "cuda" else 0.0
                ),
                "device_type": device.type,
                "device_index": device.index,
                "last_stream": stream_name,
                **{
                    name: sum(values) / len(values)
                    for name, values in window_auxiliary.items()
                },
            }
            append_jsonl(history_path, row)
            training_progress = description_training_progress_payload(
                step=step,
                stream_pattern=stream_pattern,
                grad_accum_steps=grad_accum,
                stream_states=stream_states,
                stream_bindings=stream_loader_bindings,
            )
            write_json(
                output_dir / "training_progress_latest.json", training_progress
            )
            tqdm.write(
                f"[DESC-TRAIN] step={step} loss={row['loss']:.4f} "
                f"sample_sps={row['samples_per_second']:.2f} peak_gib={row['peak_reserved_gib']:.2f}"
            )
            window_loss = window_samples = 0.0
            window_steps = 0
            window_auxiliary = {}
            window_started = time.perf_counter()

        validation_due = val_loader is not None and (
            step % config.training.val_interval == 0 or step == config.training.max_steps
        )
        if validation_due:
            report = evaluate_description(
                model,
                val_loader,
                validation_config,
                device,
                split=str(val_name),
                output_dir=output_dir / "validation_latest",
                run_counterfactuals=False,
            )
            last_validation_report = report
            score = description_selection_score(report, config.training.stage, config.training.checkpoint_metric)
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
                    metadata={
                        "stage": config.training.stage,
                        "checkpoint_role": "validation_best",
                        "best_score": (
                            best_score if math.isfinite(best_score) else None
                        ),
                        "config": resolved,
                        "data_audit": checkpoint_data_audit,
                        "region_data_audit": checkpoint_region_data_audit,
                        "d4_curriculum_transition": d4_curriculum_transition,
                        "d_minus_one_acceptance": d_minus_one_acceptance,
                        "stage_lineage": stage_lineage,
                        "segmentation_migration_lineage": (
                            segmentation_migration_lineage
                        ),
                        "resume_reconciliation": resume_reconciliation,
                        "training_progress": description_training_progress_payload(
                            step=step,
                            stream_pattern=stream_pattern,
                            grad_accum_steps=grad_accum,
                            stream_states=stream_states,
                            stream_bindings=stream_loader_bindings,
                        ),
                    },
                )
                tqdm.write(f"[DESC-CKPT] saved=checkpoint_best.pt step={step} score={score:.4f}")
            model.train()

        if step % config.training.save_interval == 0 or step == config.training.max_steps:
            if step == config.training.max_steps and not gradient_gate.complete:
                raise RuntimeError(
                    "description terminal checkpoint 拒绝保存："
                    "尚未观察全部必需 gradient task paths"
                )
            save_segdesc_checkpoint(
                output_dir / "checkpoint_last.pt",
                model,
                step=step,
                segmentation_migration=migration,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metadata={
                    "stage": config.training.stage,
                    "checkpoint_role": "terminal_last",
                    "best_score": (
                        best_score if math.isfinite(best_score) else None
                    ),
                    "config": resolved,
                    "data_audit": checkpoint_data_audit,
                    "region_data_audit": checkpoint_region_data_audit,
                    "d4_curriculum_transition": d4_curriculum_transition,
                    "d_minus_one_acceptance": d_minus_one_acceptance,
                    "stage_lineage": stage_lineage,
                    "segmentation_migration_lineage": (
                        segmentation_migration_lineage
                    ),
                    "resume_reconciliation": resume_reconciliation,
                    "gradient_gate": gradient_gate.payload(),
                    "training_progress": description_training_progress_payload(
                        step=step,
                        stream_pattern=stream_pattern,
                        grad_accum_steps=grad_accum,
                        stream_states=stream_states,
                        stream_bindings=stream_loader_bindings,
                    ),
                },
            )
            tqdm.write(f"[DESC-CKPT] saved=checkpoint_last.pt step={step}")
    progress.close()
    d_minus_one_report = None
    if config.training.stage == "overfit":
        checkpoint_path = output_dir / "checkpoint_last.pt"
        reloaded_step, strict_reload_audit = verify_segdesc_checkpoint_reload(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage="overfit",
        )
        if strict_reload_audit.get("segmentation_migration") != migration:
            raise RuntimeError(
                "D-1 strict reload 返回的 segmentation migration 与 live run 不一致"
            )

        def read_jsonl(path: Path) -> list[dict[str, Any]]:
            if not path.is_file():
                return []
            return [
                strict_json_loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        d_minus_one_report = build_d_minus_one_overfit_validation(
            config=config,
            sampling_audit=getattr(
                train_dataset, "d_minus_one_sampling_audit", None
            ),
            history_rows=read_jsonl(history_path),
            gradient_gate=gradient_gate.payload(),
            validation_report=last_validation_report,
            generation_rows=read_jsonl(
                output_dir / "validation_latest/raw_generations.jsonl"
            ),
            trainable_manifest=trainable_manifest,
            checkpoint_path=checkpoint_path,
            checkpoint_step=reloaded_step,
            device_type=device.type,
            segmentation_migration=migration,
            reload_audit=strict_reload_audit,
            artifact_readiness_acceptance=(
                artifact_readiness_acceptance
            ),
            source_files={
                "checkpoint": checkpoint_path,
                "dataset_summary": output_dir / "dataset_summary.json",
                "gradient_gate": output_dir / "description_gradient_gate.json",
                "raw_generations": (
                    output_dir / "validation_latest/raw_generations.jsonl"
                ),
                "resolved_config": output_dir / "resolved_config.json",
                "train_history": history_path,
                "trainable_manifest": (
                    output_dir / "trainable_parameter_manifest.json"
                ),
                "validation_report": (
                    output_dir / "validation_latest/eval_report.json"
                ),
                "artifact_readiness_report": Path(
                    resolve_project_path(
                        str(config.data.artifact_readiness_report)
                    )
                    or str(config.data.artifact_readiness_report)
                ),
            },
        )
        write_json(
            output_dir / "d_minus_one_overfit_validation.json",
            d_minus_one_report,
        )
    return {
        "output_dir": str(output_dir),
        "stage": config.training.stage,
        "steps": step,
        "best_score": best_score if math.isfinite(best_score) else None,
        "checkpoint_best": (
            str(output_dir / "checkpoint_best.pt")
            if (output_dir / "checkpoint_best.pt").is_file() else None
        ),
        "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
        "d_minus_one_overfit_validation": d_minus_one_report,
        "d4_curriculum_transition": d4_curriculum_transition,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "resume_reconciliation": resume_reconciliation,
    }
