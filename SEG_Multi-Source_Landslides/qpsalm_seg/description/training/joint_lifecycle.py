#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 source lineage, initialization, progress, and resume contracts."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
)
from ..evaluation.m6_acceptance import M6_ACCEPTANCE_AUDIT_PROTOCOL
from ..protocols.io import canonical_sha256
from .checkpoint import (
    inspect_segdesc_checkpoint,
    validate_segmentation_migration_lineage,
)
from .joint_contracts import (
    JOINT_INITIALIZATION_PROTOCOL,
    JOINT_LOADER_BINDING_PROTOCOL,
    JOINT_LOADER_CURSOR_PROTOCOL,
    JOINT_LOADER_SEED_OFFSETS,
    JOINT_PROGRESS_PROTOCOL,
    JOINT_RUN_PROTOCOL,
    JOINT_TASKS,
)
from .joint_runtime import initial_joint_loader_states
from .run_artifacts import validate_checkpoint_run_completion
from ..protocols.versions import DESCRIPTION_TRAINING_COMPLETION_PROTOCOL


def validate_m7_source_checkpoint(
    checkpoint_metadata: dict[str, Any],
    *,
    region_stage: str,
    current_data_audit: dict[str, Any],
    resume: bool,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Prove M7 weights and current region loader share the same accepted data."""
    metadata = dict(checkpoint_metadata.get("metadata") or {})
    expected_stage = "joint" if resume else region_stage
    if str(metadata.get("stage") or "") != expected_stage:
        raise RuntimeError(
            f"M7 {'resume' if resume else 'initialize'} checkpoint stage 必须为 "
            f"{expected_stage!r}"
        )
    expected_role = (
        "terminal_last"
        if resume or region_stage == "bridge_auto"
        else "validation_best"
    )
    if metadata.get("checkpoint_role") != expected_role:
        raise RuntimeError(
            "M7 source checkpoint role 非法: "
            f"expected={expected_role!r} "
            f"observed={metadata.get('checkpoint_role')!r}"
        )
    if resume and metadata.get("joint_run_protocol") != JOINT_RUN_PROTOCOL:
        raise RuntimeError(
            "M7 resume checkpoint joint run protocol 不一致；旧实验性 joint "
            "checkpoint 不能继续续训"
        )
    observed_audit = dict(metadata.get("region_data_audit") or {})
    if not observed_audit:
        raise RuntimeError(
            "M7 checkpoint 缺少显式 region_data_audit；旧 checkpoint 不兼容，"
            "必须由当前 D3/D4 trainer 重建"
        )
    if observed_audit != current_data_audit:
        raise RuntimeError("M7 checkpoint 与当前 region expert/predicted 数据绑定不一致")
    source_config = require_serialized_segdesc_config(
        metadata.get("config"), label="M7 source checkpoint config"
    )
    source_seed = serialized_segdesc_config_value(source_config, "seed")
    if expected_seed is not None and (
        source_seed is None or int(source_seed) != int(expected_seed)
    ):
        raise RuntimeError(
            "M7 checkpoint seed 与当前 run seed 不一致: "
            f"expected={int(expected_seed)} observed={source_seed!r}"
        )
    return {
        "source_stage": expected_stage,
        "source_checkpoint_role": expected_role,
        "region_data_audit": current_data_audit,
        "resume": bool(resume),
        "source_seed": int(source_seed) if source_seed is not None else None,
        "run_seed": int(expected_seed) if expected_seed is not None else None,
        "seed_match": expected_seed is None or int(source_seed) == int(expected_seed),
    }


def _resolved_checkpoint_path(value: str | Path, *, label: str) -> Path:
    path = resolve_project_path(value) or Path(value)
    if not path.is_file():
        raise RuntimeError(f"{label} 不存在: {value}")
    return path.resolve(strict=False)


def build_joint_initialization_audit(
    source_checkpoint: str | Path,
    *,
    expected_seed: int,
    region_stage: str,
    region_data_audit: dict[str, Any],
    d4_final_acceptance: dict[str, Any] | None,
    m6_acceptance: dict[str, Any] | None,
    segmentation_migration: dict[str, Any],
    source_step: int | None = None,
    source_initialization: dict[str, Any] | None = None,
    require_m6_binding: bool = False,
) -> dict[str, Any]:
    """Bind the exact pre-M7 payload instead of trusting copied gate metadata."""
    source_path = _resolved_checkpoint_path(
        source_checkpoint, label="M7 initialization source checkpoint"
    )
    try:
        provenance = inspect_segdesc_checkpoint(source_path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError("M7 initialization source checkpoint payload 无法重放") from exc
    metadata = dict(
        (provenance.get("checkpoint_metadata") or {}).get("metadata") or {}
    )
    segmentation_migration_lineage = validate_segmentation_migration_lineage(
        segmentation_migration,
        provenance["checkpoint_metadata"],
    )
    if str(metadata.get("stage") or "") != str(region_stage):
        raise RuntimeError("M7 initialization source checkpoint stage 不一致")
    expected_role = (
        "terminal_last" if region_stage == "bridge_auto" else "validation_best"
    )
    if metadata.get("checkpoint_role") != expected_role:
        raise RuntimeError(
            "M7 initialization source checkpoint role 不一致: "
            f"expected={expected_role!r} "
            f"observed={metadata.get('checkpoint_role')!r}"
        )
    try:
        source_run_completion = validate_checkpoint_run_completion(
            source_path,
            expected_completion_protocol=(
                DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
            ),
            expected_stage=region_stage,
            expected_role=expected_role,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "M7 initialization source training run 未成功完成"
        ) from exc
    if metadata.get("region_data_audit") != region_data_audit:
        raise RuntimeError("M7 initialization source checkpoint region data 已漂移")
    source_config = require_serialized_segdesc_config(
        metadata.get("config"), label="M7 initialization source config"
    )
    source_seed = serialized_segdesc_config_value(source_config, "seed")
    if source_seed is None or int(source_seed) != int(expected_seed):
        raise RuntimeError("M7 initialization source checkpoint seed 不一致")
    if source_step is not None and int(source_step) != int(
        provenance["checkpoint_step"]
    ):
        raise RuntimeError("M7 initialization loader 返回的 source step 不一致")

    if source_initialization is not None:
        if not isinstance(source_initialization, dict):
            raise RuntimeError("M7 initialize loader 缺少 initialization provenance")
        loaded_path = _resolved_checkpoint_path(
            source_initialization.get("source_checkpoint") or "",
            label="M7 loader initialization source checkpoint",
        )
        if (
            loaded_path != source_path
            or str(source_initialization.get("source_checkpoint_sha256") or "")
            != str(provenance["checkpoint_sha256"])
            or str(source_initialization.get("source_stage") or "")
            != str(region_stage)
            or source_initialization.get("seed_match") is not True
            or int(source_initialization.get("source_seed", -1))
            != int(expected_seed)
            or source_initialization.get("source_run_completion")
            != source_run_completion
        ):
            raise RuntimeError("M7 initialize loader provenance 与源 checkpoint 不一致")

    formal_m6_bound = bool(
        isinstance(d4_final_acceptance, dict)
        and d4_final_acceptance.get("passed") is True
        and isinstance(m6_acceptance, dict)
        and m6_acceptance.get("protocol") == M6_ACCEPTANCE_AUDIT_PROTOCOL
        and m6_acceptance.get("passed") is True
        and m6_acceptance.get("d4_final_acceptance") == d4_final_acceptance
    )
    if require_m6_binding and not formal_m6_bound:
        raise RuntimeError("正式 M7 initialization 缺少一致的 D4/M6 acceptance")

    d4_source = (
        _resolved_checkpoint_path(
            d4_final_acceptance.get("source_checkpoint") or "",
            label="D4 final source checkpoint",
        )
        if formal_m6_bound else None
    )
    m6_source = (
        _resolved_checkpoint_path(
            m6_acceptance.get("source_checkpoint") or "",
            label="M6 acceptance source checkpoint",
        )
        if formal_m6_bound else None
    )
    if formal_m6_bound and (
        d4_source != source_path
        or m6_source != source_path
        or str(d4_final_acceptance.get("source_checkpoint_sha256") or "")
        != str(provenance["checkpoint_sha256"])
        or str(m6_acceptance.get("source_checkpoint_sha256") or "")
        != str(provenance["checkpoint_sha256"])
    ):
        raise RuntimeError("M7 initialization source 不是 D4/M6 验收的 checkpoint")

    return {
        "protocol": JOINT_INITIALIZATION_PROTOCOL,
        "passed": True,
        "formal_m6_bound": formal_m6_bound,
        "target_stage": "joint",
        "source_stage": str(region_stage),
        "source_checkpoint_role": expected_role,
        "source_run_completion": source_run_completion,
        "seed": int(expected_seed),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": provenance["checkpoint_sha256"],
        "source_checkpoint_step": int(provenance["checkpoint_step"]),
        "source_checkpoint_metadata_sha256": canonical_sha256(
            provenance["checkpoint_metadata"]
        ),
        "source_model_state_inventory_sha256": provenance[
            "model_state_inventory_sha256"
        ],
        "region_data_audit_sha256": canonical_sha256(region_data_audit),
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "d4_final_gate": (
            d4_final_acceptance.get("gate") if formal_m6_bound else None
        ),
        "d4_final_gate_sha256": (
            d4_final_acceptance.get("gate_sha256") if formal_m6_bound else None
        ),
        "m6_acceptance_gate": (
            m6_acceptance.get("gate") if formal_m6_bound else None
        ),
        "m6_acceptance_gate_sha256": (
            m6_acceptance.get("gate_sha256") if formal_m6_bound else None
        ),
    }


def revalidate_joint_initialization_audit(
    saved: Any,
    *,
    expected_seed: int,
    region_stage: str,
    region_data_audit: dict[str, Any],
    d4_final_acceptance: dict[str, Any] | None,
    m6_acceptance: dict[str, Any] | None,
    segmentation_migration: dict[str, Any],
    require_m6_binding: bool = False,
) -> dict[str, Any]:
    """Reopen the pre-M7 checkpoint and reproduce the saved initialization audit."""
    if (
        not isinstance(saved, dict)
        or saved.get("protocol") != JOINT_INITIALIZATION_PROTOCOL
        or saved.get("passed") is not True
    ):
        raise RuntimeError("joint checkpoint 缺少当前 initialization source audit")
    rebuilt = build_joint_initialization_audit(
        saved.get("source_checkpoint") or "",
        expected_seed=expected_seed,
        region_stage=region_stage,
        region_data_audit=region_data_audit,
        d4_final_acceptance=d4_final_acceptance,
        m6_acceptance=m6_acceptance,
        segmentation_migration=segmentation_migration,
        source_step=int(saved.get("source_checkpoint_step", -1)),
        require_m6_binding=require_m6_binding,
    )
    if rebuilt != saved:
        raise RuntimeError("joint checkpoint initialization source audit 已漂移")
    return rebuilt


def joint_progress_payload(
    *,
    step: int,
    task_steps: Counter,
    task_samples: Counter,
    parent_coverage: dict[str, set[str]],
    parent_populations: dict[str, set[str]],
    loader_states: dict[str, dict[str, int]],
    loader_bindings: dict[str, dict[str, Any]],
    task_pattern: tuple[str, ...],
    grad_accum_steps: int,
) -> dict[str, Any]:
    tasks = set(parent_populations)
    if (
        tasks != set(loader_states)
        or tasks != set(loader_bindings)
        or tasks != set(parent_coverage)
    ):
        raise RuntimeError(
            "M7 progress 的 task/population/loader/coverage 集合不一致"
        )
    grad_accum_steps = max(1, int(grad_accum_steps))
    loader_cursors = {}
    for task in sorted(tasks):
        unknown_parents = parent_coverage[task] - parent_populations[task]
        if unknown_parents:
            raise RuntimeError(
                f"M7 {task} coverage 包含未知 parent: "
                f"{sorted(unknown_parents)[:8]}"
            )
        if (
            int(task_samples[task]) < len(parent_coverage[task])
            or (
                int(task_steps[task]) > 0
                and (
                    int(task_samples[task]) <= 0
                    or not parent_coverage[task]
                )
            )
        ):
            raise RuntimeError(
                f"M7 {task} samples/parent coverage 与 optimizer step 不一致"
            )
        state = loader_states[task]
        binding = loader_bindings[task]
        batches_per_epoch = int(binding["batches_per_epoch"])
        expected_total = int(task_steps[task]) * grad_accum_steps
        total = int(state["total_microbatches"])
        expected_epoch, expected_cursor = divmod(expected_total, batches_per_epoch)
        if total != expected_total or (
            int(state["epoch"]), int(state["batch_in_epoch"])
        ) != (expected_epoch, expected_cursor):
            raise RuntimeError(
                f"M7 {task} loader cursor 与 optimizer/microbatch 计数不一致"
            )
        loader_cursors[task] = {
            "protocol": JOINT_LOADER_CURSOR_PROTOCOL,
            "epoch": expected_epoch,
            "batch_in_epoch": expected_cursor,
            "total_microbatches": expected_total,
            "batches_per_epoch": batches_per_epoch,
            "loader_binding_sha256": binding["binding_sha256"],
        }
    return {
        "protocol": JOINT_PROGRESS_PROTOCOL,
        "step": int(step),
        "task_pattern": list(task_pattern),
        "task_pattern_sha256": canonical_sha256(list(task_pattern)),
        "grad_accum_steps": grad_accum_steps,
        "optimizer_steps": {task: int(task_steps[task]) for task in sorted(tasks)},
        "samples_seen": {task: int(task_samples[task]) for task in sorted(tasks)},
        "loader_cursors": loader_cursors,
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
                "population_parent_ids": sorted(parent_populations[task]),
                "parent_ids": sorted(parent_coverage[task]),
            }
            for task in parent_populations
        },
    }


def restore_joint_progress(
    saved_progress: dict[str, Any],
    parent_populations: dict[str, set[str]],
    loader_bindings: dict[str, dict[str, Any]],
    *,
    checkpoint_step: int,
    required: bool,
    task_pattern: tuple[str, ...],
    grad_accum_steps: int,
) -> tuple[Counter, Counter, dict[str, set[str]], dict[str, dict[str, int]]]:
    """Restore coverage and exact next-batch cursors for one unchanged M7 run."""
    tasks = set(parent_populations)
    if tasks != set(loader_bindings) or tasks != set(JOINT_TASKS):
        raise RuntimeError("M7 current loader/population task 集合不完整")
    if not task_pattern or set(task_pattern) != tasks:
        raise RuntimeError("M7 current task pattern 与 loader task 集合不一致")
    grad_accum_steps = max(1, int(grad_accum_steps))
    if not saved_progress:
        if required:
            raise RuntimeError("M7 resume checkpoint 缺少 joint_progress")
        return (
            Counter({task: 0 for task in tasks}),
            Counter({task: 0 for task in tasks}),
            {task: set() for task in tasks},
            initial_joint_loader_states(loader_bindings),
        )
    if saved_progress.get("protocol") != JOINT_PROGRESS_PROTOCOL:
        raise RuntimeError("M7 resume joint_progress protocol 不一致")
    if int(saved_progress.get("step", -1)) != int(checkpoint_step):
        raise RuntimeError("M7 resume checkpoint step 与 joint_progress step 不一致")
    if (
        tuple(saved_progress.get("task_pattern") or ()) != tuple(task_pattern)
        or saved_progress.get("task_pattern_sha256")
        != canonical_sha256(list(task_pattern))
    ):
        raise RuntimeError("M7 resume task pattern binding 不一致")
    if int(saved_progress.get("grad_accum_steps", -1)) != grad_accum_steps:
        raise RuntimeError("M7 resume grad accumulation binding 不一致")
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
        if saved.get("population_parent_ids") != sorted(population):
            raise RuntimeError(
                f"M7 resume {task} parent population list 不一致"
            )
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
    allowed_tasks = tasks
    if set(task_steps) - allowed_tasks or set(task_samples) - allowed_tasks:
        raise RuntimeError("M7 resume progress 包含未知 task")
    if any(value < 0 for value in (*task_steps.values(), *task_samples.values())):
        raise RuntimeError("M7 resume progress 包含负计数")
    if sum(task_steps.values()) != int(checkpoint_step):
        raise RuntimeError("M7 resume optimizer step 总数与 checkpoint step 不一致")
    expected_steps = Counter(
        task_pattern[index % len(task_pattern)]
        for index in range(int(checkpoint_step))
    )
    if any(int(task_steps[task]) != int(expected_steps[task]) for task in tasks):
        raise RuntimeError("M7 resume optimizer task steps 与 task pattern 不一致")

    saved_cursors = dict(saved_progress.get("loader_cursors") or {})
    if set(saved_cursors) != tasks:
        raise RuntimeError("M7 resume loader cursor task 集合不一致")
    loader_states: dict[str, dict[str, int]] = {}
    for task in sorted(tasks):
        cursor = dict(saved_cursors[task] or {})
        binding = loader_bindings[task]
        batches_per_epoch = int(binding["batches_per_epoch"])
        expected_total = int(task_steps[task]) * grad_accum_steps
        expected_epoch, expected_batch = divmod(expected_total, batches_per_epoch)
        if cursor.get("protocol") != JOINT_LOADER_CURSOR_PROTOCOL:
            raise RuntimeError(f"M7 resume {task} loader cursor protocol 不一致")
        if (
            int(cursor.get("batches_per_epoch", -1)) != batches_per_epoch
            or str(cursor.get("loader_binding_sha256") or "")
            != str(binding["binding_sha256"])
        ):
            raise RuntimeError(f"M7 resume {task} loader binding 已变化")
        observed = (
            int(cursor.get("epoch", -1)),
            int(cursor.get("batch_in_epoch", -1)),
            int(cursor.get("total_microbatches", -1)),
        )
        expected = (expected_epoch, expected_batch, expected_total)
        if observed != expected:
            raise RuntimeError(
                f"M7 resume {task} loader cursor 与 task step 不一致: "
                f"expected={expected} observed={observed}"
            )
        loader_states[task] = {
            "epoch": expected_epoch,
            "batch_in_epoch": expected_batch,
            "total_microbatches": expected_total,
        }
    return task_steps, task_samples, restored_coverage, loader_states


def validate_joint_checkpoint_execution(
    metadata: dict[str, Any],
    *,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Recompute the saved M7 schedule/cursor contract without loading datasets."""
    if metadata.get("joint_run_protocol") != JOINT_RUN_PROTOCOL:
        raise RuntimeError("M7 checkpoint joint run protocol 不一致")
    config = require_serialized_segdesc_config(
        metadata.get("config"), label="M7 joint checkpoint config"
    )
    run_seed = serialized_segdesc_config_value(config, "seed")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise RuntimeError("M7 checkpoint run seed 非法")
    bindings = metadata.get("joint_loader_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(JOINT_TASKS):
        raise RuntimeError("M7 checkpoint loader binding task 集合不完整")
    normalized_bindings: dict[str, dict[str, Any]] = {}
    for task in JOINT_TASKS:
        binding = bindings.get(task)
        if not isinstance(binding, dict):
            raise RuntimeError(f"M7 checkpoint {task} loader binding 非法")
        binding = dict(binding)
        observed_hash = str(binding.pop("binding_sha256", ""))
        if (
            binding.get("protocol") != JOINT_LOADER_BINDING_PROTOCOL
            or binding.get("task") != task
            or observed_hash != canonical_sha256(binding)
        ):
            raise RuntimeError(f"M7 checkpoint {task} loader binding hash/protocol 非法")
        batches_per_epoch = binding.get("batches_per_epoch")
        if (
            isinstance(batches_per_epoch, bool)
            or not isinstance(batches_per_epoch, int)
            or batches_per_epoch <= 0
        ):
            raise RuntimeError(f"M7 checkpoint {task} batches_per_epoch 非法")
        if binding.get("persistent_workers") is not False:
            raise RuntimeError(
                f"M7 checkpoint {task} 必须关闭 persistent_workers 才能重放 epoch"
            )
        if binding.get("loader_seed") != (
            int(run_seed) + JOINT_LOADER_SEED_OFFSETS[task]
        ):
            raise RuntimeError(
                f"M7 checkpoint {task} loader seed 未由 run seed 确定"
            )
        normalized_bindings[task] = {
            **binding,
            "binding_sha256": observed_hash,
        }

    progress = metadata.get("joint_progress")
    if not isinstance(progress, dict) or progress.get("protocol") != JOINT_PROGRESS_PROTOCOL:
        raise RuntimeError("M7 checkpoint joint progress protocol 不一致")
    if progress.get("step") != int(checkpoint_step):
        raise RuntimeError("M7 checkpoint joint progress step 不一致")
    pattern = tuple(serialized_segdesc_config_value(
        config, "joint_task_pattern"
    ) or (
        "segmentation", "global_caption", "segmentation", "region_description",
    ))
    if (
        not pattern
        or set(pattern) != set(JOINT_TASKS)
        or tuple(progress.get("task_pattern") or ()) != pattern
        or progress.get("task_pattern_sha256") != canonical_sha256(list(pattern))
    ):
        raise RuntimeError("M7 checkpoint task pattern binding 非法")
    grad_accum = serialized_segdesc_config_value(config, "grad_accum_steps")
    if (
        isinstance(grad_accum, bool)
        or not isinstance(grad_accum, int)
        or grad_accum <= 0
        or progress.get("grad_accum_steps") != grad_accum
    ):
        raise RuntimeError("M7 checkpoint grad accumulation binding 非法")

    expected_steps = Counter(
        pattern[index % len(pattern)] for index in range(int(checkpoint_step))
    )
    optimizer_steps = progress.get("optimizer_steps")
    samples_seen = progress.get("samples_seen")
    cursors = progress.get("loader_cursors")
    parent_coverage = progress.get("parent_coverage")
    if not all(isinstance(value, dict) for value in (
        optimizer_steps, samples_seen, cursors, parent_coverage,
    )) or any(set(value) != set(JOINT_TASKS) for value in (
        optimizer_steps, samples_seen, cursors, parent_coverage,
    )):
        raise RuntimeError(
            "M7 checkpoint task step/sample/cursor/coverage 集合不完整"
        )
    for task in JOINT_TASKS:
        step_count = optimizer_steps[task]
        sample_count = samples_seen[task]
        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or step_count != int(expected_steps[task])
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 0
        ):
            raise RuntimeError(f"M7 checkpoint {task} task count 与 schedule 不一致")
        coverage = parent_coverage[task]
        parent_ids = coverage.get("parent_ids") if isinstance(coverage, dict) else None
        population_parent_ids = (
            coverage.get("population_parent_ids")
            if isinstance(coverage, dict) else None
        )
        covered = coverage.get("covered") if isinstance(coverage, dict) else None
        population = coverage.get("population") if isinstance(coverage, dict) else None
        fraction = coverage.get("fraction") if isinstance(coverage, dict) else None
        if (
            not isinstance(coverage, dict)
            or not isinstance(parent_ids, list)
            or not isinstance(population_parent_ids, list)
            or any(not isinstance(value, str) or not value for value in parent_ids)
            or any(
                not isinstance(value, str) or not value
                for value in population_parent_ids
            )
            or len(parent_ids) != len(set(parent_ids))
            or len(population_parent_ids) != len(set(population_parent_ids))
            or parent_ids != sorted(parent_ids)
            or population_parent_ids != sorted(population_parent_ids)
            or isinstance(covered, bool)
            or not isinstance(covered, int)
            or covered != len(parent_ids)
            or isinstance(population, bool)
            or not isinstance(population, int)
            or population <= 0
            or population != len(population_parent_ids)
            or not 0 <= covered <= population
            or not set(parent_ids) <= set(population_parent_ids)
            or covered > sample_count
            or (step_count > 0 and (sample_count <= 0 or covered <= 0))
            or not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or not math.isclose(
                float(fraction), covered / population, abs_tol=1.0e-12
            )
            or coverage.get("population_sha256") != hashlib.sha256(
                "\n".join(sorted(population_parent_ids)).encode("utf-8")
            ).hexdigest()
            or coverage.get("covered_sha256") != hashlib.sha256(
                "\n".join(sorted(parent_ids)).encode("utf-8")
            ).hexdigest()
        ):
            raise RuntimeError(
                f"M7 checkpoint {task} parent coverage 计数/hash 非法"
            )
        binding = normalized_bindings[task]
        batches_per_epoch = int(binding["batches_per_epoch"])
        total = step_count * grad_accum
        epoch, batch_in_epoch = divmod(total, batches_per_epoch)
        cursor = cursors[task]
        expected_cursor = {
            "protocol": JOINT_LOADER_CURSOR_PROTOCOL,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "total_microbatches": total,
            "batches_per_epoch": batches_per_epoch,
            "loader_binding_sha256": binding["binding_sha256"],
        }
        if cursor != expected_cursor:
            raise RuntimeError(f"M7 checkpoint {task} loader cursor 无法由 schedule 重算")
    training_population_binding = {
        "protocol": "qpsalm_segdesc_joint_training_population_v1",
        "tasks": {},
    }
    for task in JOINT_TASKS:
        loader_contract = dict(normalized_bindings[task])
        loader_contract.pop("binding_sha256", None)
        loader_contract.pop("loader_seed", None)
        sampler_contract = dict(loader_contract.get("batch_sampler") or {})
        sampler_contract.pop("seed", None)
        loader_contract["batch_sampler"] = sampler_contract
        coverage = parent_coverage[task]
        training_population_binding["tasks"][task] = {
            "loader_contract": loader_contract,
            "loader_contract_sha256": canonical_sha256(loader_contract),
            "parent_population": {
                "count": int(coverage["population"]),
                "sha256": str(coverage["population_sha256"]),
                "parent_ids": list(coverage["population_parent_ids"]),
            },
        }
    training_population_binding["binding_sha256"] = canonical_sha256(
        training_population_binding
    )
    return {
        "protocol": "qpsalm_segdesc_joint_execution_audit_v1",
        "joint_run_protocol": JOINT_RUN_PROTOCOL,
        "joint_progress_protocol": JOINT_PROGRESS_PROTOCOL,
        "checkpoint_step": int(checkpoint_step),
        "task_pattern": list(pattern),
        "grad_accum_steps": grad_accum,
        "loader_binding_sha256": {
            task: normalized_bindings[task]["binding_sha256"]
            for task in JOINT_TASKS
        },
        "parent_coverage_sha256": canonical_sha256(parent_coverage),
        "training_population_binding": training_population_binding,
        "progress_sha256": canonical_sha256(progress),
        "passed": True,
    }
