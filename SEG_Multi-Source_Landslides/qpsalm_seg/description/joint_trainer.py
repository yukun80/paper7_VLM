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
    capture_training_rng_state,
    initialize_segdesc_checkpoint,
    inspect_segdesc_checkpoint,
    load_segdesc_checkpoint,
    read_segdesc_checkpoint_step,
    restore_training_rng_state,
    save_segdesc_checkpoint,
    validate_description_stage_lineage,
    validate_segmentation_migration_lineage,
    validate_resume_run_config,
)
from .common import (
    append_jsonl,
    build_description_dataset,
    build_description_loader,
    description_amp_dtype,
    description_device,
    description_scaler,
    move_description_batch,
    set_loader_epoch,
    set_description_seed,
    validate_predicted_training_indexes,
    write_json,
)
from .config import SegDescConfig
from .json_protocol import strict_json_loads
from .data import REGION_TRAINING_DATA_PROTOCOL
from .evaluator import evaluate_description
from .d4_curriculum import (
    D4_FINAL_FRACTION,
    validate_d4_final_acceptance_for_m7,
)
from .d_minus_one import revalidate_saved_d_minus_one_acceptance
from .m6_acceptance import (
    M6_ACCEPTANCE_AUDIT_PROTOCOL,
    revalidate_saved_m6_acceptance,
    validate_m6_acceptance_for_m7,
)
from .model import DESCRIPTION_ADAPTER_NAME
from .runtime import build_segdesc_model
from .run_artifacts import reconcile_resume_run
from .trainer import _dataset_data_audit, _train_loss


JOINT_RUN_PROTOCOL = "qpsalm_segdesc_joint_v7_strict_json_finite"
JOINT_INITIALIZATION_PROTOCOL = (
    "qpsalm_segdesc_joint_initialization_v4_run_completion_bound"
)
JOINT_PROGRESS_PROTOCOL = (
    "qpsalm_segdesc_joint_progress_v3_parent_population_list_bound"
)
JOINT_LOADER_BINDING_PROTOCOL = "qpsalm_segdesc_joint_loader_binding_v1"
JOINT_LOADER_CURSOR_PROTOCOL = "qpsalm_segdesc_joint_loader_cursor_v1"
JOINT_TASKS = ("segmentation", "global_caption", "region_description")
JOINT_LOADER_SEED_OFFSETS = {
    "segmentation": 710_011,
    "global_caption": 720_013,
    "region_description": 730_019,
}


class EpochConcatDataset(ConcatDataset):
    def __init__(self, datasets) -> None:
        super().__init__(datasets)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_rows_binding(dataset) -> dict[str, Any]:
    """Bind exact ordered rows, including boundaries inside ConcatDataset."""
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        children = [_dataset_rows_binding(value) for value in nested]
        payload = {
            "dataset_class": type(dataset).__name__,
            "children": children,
        }
        return {
            "num_rows": sum(int(value["num_rows"]) for value in children),
            "ordered_rows_sha256": _canonical_sha256(payload),
            "num_children": len(children),
        }
    rows = getattr(dataset, "rows", None)
    if not isinstance(rows, list):
        raise RuntimeError(
            f"M7 loader dataset 缺少可审计 rows: {type(dataset).__name__}"
        )
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(f"{index}:".encode("ascii"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return {
        "num_rows": len(rows),
        "ordered_rows_sha256": digest.hexdigest(),
        "num_children": 0,
    }


def _joint_loader_binding(
    task: str,
    loader,
    *,
    loader_seed: int,
) -> dict[str, Any]:
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise RuntimeError(f"M7 {task} loader 没有可训练 batch")
    batch_sampler = loader.batch_sampler
    sampler = {
        "class": type(batch_sampler).__name__,
        "protocol": getattr(batch_sampler, "protocol", None),
        "batch_size": getattr(batch_sampler, "batch_size", None),
        "seed": getattr(batch_sampler, "seed", None),
        "drop_last": getattr(batch_sampler, "drop_last", None),
        "shuffle": getattr(batch_sampler, "shuffle", None),
        "balance_tasks": getattr(batch_sampler, "balance_tasks", None),
        "task_weights": getattr(batch_sampler, "task_weights", None),
    }
    binding = {
        "protocol": JOINT_LOADER_BINDING_PROTOCOL,
        "task": task,
        "dataset": _dataset_rows_binding(loader.dataset),
        "batches_per_epoch": int(batches_per_epoch),
        "num_workers": int(loader.num_workers),
        "persistent_workers": bool(loader.persistent_workers),
        "prefetch_factor": loader.prefetch_factor,
        "loader_seed": int(loader_seed),
        "worker_seed_protocol": "loader_seed_plus_1000003_times_epoch",
        "batch_sampler": sampler,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def _initial_joint_loader_states(
    loader_bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    return {
        task: {"epoch": 0, "batch_in_epoch": 0, "total_microbatches": 0}
        for task in loader_bindings
    }


def _loader_iterator_at_cursor(
    loader,
    state: dict[str, int],
    binding: dict[str, Any],
):
    """Rebuild one epoch and skip consumed batches without perturbing model RNG."""
    set_loader_epoch(
        loader,
        int(state["epoch"]),
        loader_seed=int(binding["loader_seed"]),
    )
    iterator = iter(loader)
    cursor = int(state["batch_in_epoch"])
    if cursor <= 0:
        return iterator
    rng_state = capture_training_rng_state()
    try:
        for _ in range(cursor):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "M7 resume loader cursor 超出当前 epoch；loader binding 已失效"
                ) from exc
    finally:
        restore_training_rng_state(rng_state)
    return iterator


def _next_joint_loader_batch(
    loader,
    iterator,
    state: dict[str, int],
    binding: dict[str, Any],
):
    if iterator is None:
        iterator = _loader_iterator_at_cursor(loader, state, binding)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(
            "M7 loader 在已绑定 batches_per_epoch 前提前耗尽"
        ) from exc
    state["total_microbatches"] += 1
    state["batch_in_epoch"] += 1
    if state["batch_in_epoch"] == int(binding["batches_per_epoch"]):
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        iterator = None
    elif state["batch_in_epoch"] > int(binding["batches_per_epoch"]):
        raise RuntimeError("M7 loader cursor 超过 batches_per_epoch")
    return batch, iterator


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


def _region_data_audit(dataset) -> dict[str, Any]:
    population = _dataset_data_audit(dataset)
    return {
        "protocol": REGION_TRAINING_DATA_PROTOCOL,
        "stage": str(dataset.stage),
        "expert_gate_audit": getattr(dataset, "expert_gate_audit", None),
        "bridge_engineering_audit": getattr(
            dataset, "bridge_engineering_audit", None
        ),
        "predicted_index_audit": getattr(dataset, "predicted_index_audit", None),
        "curriculum_audit": getattr(dataset, "curriculum_audit", None),
        "population": {
            "protocol": population["protocol"],
            "stage": population["stage"],
            "split": population["split"],
            "num_samples": population["num_samples"],
            "num_parents": population["num_parents"],
            "population_sha256": population["population_sha256"],
        },
    }


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
    source_config = dict(metadata.get("config") or {})
    source_seed = source_config.get("seed")
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
    from .run_artifacts import (
        DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
        validate_checkpoint_run_completion,
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
    source_config = dict(metadata.get("config") or {})
    source_seed = source_config.get("seed")
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
        "source_checkpoint_metadata_sha256": _canonical_sha256(
            provenance["checkpoint_metadata"]
        ),
        "source_model_state_inventory_sha256": provenance[
            "model_state_inventory_sha256"
        ],
        "region_data_audit_sha256": _canonical_sha256(region_data_audit),
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


def _joint_progress_payload(
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
        "task_pattern_sha256": _canonical_sha256(list(task_pattern)),
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
            _initial_joint_loader_states(loader_bindings),
        )
    if saved_progress.get("protocol") != JOINT_PROGRESS_PROTOCOL:
        raise RuntimeError("M7 resume joint_progress protocol 不一致")
    if int(saved_progress.get("step", -1)) != int(checkpoint_step):
        raise RuntimeError("M7 resume checkpoint step 与 joint_progress step 不一致")
    if (
        tuple(saved_progress.get("task_pattern") or ()) != tuple(task_pattern)
        or saved_progress.get("task_pattern_sha256")
        != _canonical_sha256(list(task_pattern))
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
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("M7 checkpoint 缺少完整 config")
    run_seed = config.get("seed")
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
            or observed_hash != _canonical_sha256(binding)
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
    pattern = tuple(config.get("joint_task_pattern") or (
        "segmentation", "global_caption", "segmentation", "region_description",
    ))
    if (
        not pattern
        or set(pattern) != set(JOINT_TASKS)
        or tuple(progress.get("task_pattern") or ()) != pattern
        or progress.get("task_pattern_sha256") != _canonical_sha256(list(pattern))
    ):
        raise RuntimeError("M7 checkpoint task pattern binding 非法")
    grad_accum = config.get("grad_accum_steps")
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
            "loader_contract_sha256": _canonical_sha256(loader_contract),
            "parent_population": {
                "count": int(coverage["population"]),
                "sha256": str(coverage["population_sha256"]),
                "parent_ids": list(coverage["population_parent_ids"]),
            },
        }
    training_population_binding["binding_sha256"] = _canonical_sha256(
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
        "parent_coverage_sha256": _canonical_sha256(parent_coverage),
        "training_population_binding": training_population_binding,
        "progress_sha256": _canonical_sha256(progress),
        "passed": True,
    }


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
        config, stage=config.joint_region_stage
    )
    if (
        config.joint_region_stage == "predicted_mask"
        and not config.d4_final_acceptance_gate
    ):
        raise ValueError(
            "M7 predicted-mask 主路线必须提供 75% tier 的 --d4-final-acceptance-gate"
        )
    if (
        config.joint_region_stage == "predicted_mask"
        and not config.m6_acceptance_gate
    ):
        raise ValueError(
            "M7 predicted-mask 主路线必须提供完整 --m6-acceptance-gate"
        )
    if (
        config.joint_region_stage == "predicted_mask"
        and not math.isclose(
            float(config.predicted_mask_fraction),
            D4_FINAL_FRACTION,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "M7 predicted-mask 主路线必须显式使用 --predicted-mask-fraction 0.75"
        )
    set_description_seed(config.seed)
    device = description_device(device_name)
    output_dir = resolve_project_path(config.output_dir) or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, migration = build_segdesc_model(config, device)
    bank = model.description_backbone.bank

    loader_seeds = {
        task: int(config.seed) + offset
        for task, offset in JOINT_LOADER_SEED_OFFSETS.items()
    }
    seg_config = replace(
        model.segmentation.config,
        batch_size=int(config.joint_segmentation_batch_size),
        grad_accum_steps=1,
        max_train_samples=config.max_train_samples or None,
        monitor_val_samples=config.max_val_samples or None,
        # 每个 epoch 重建 worker，才能从 epoch/cursor 确定性重放其随机状态。
        persistent_workers=False,
    )
    segmentation_train, segmentation_val = build_dataloaders(seg_config)
    global_configs = [replace(config, stage=stage) for stage in (config.joint_global_stages or ["mmrs_caption", "rsicap_caption"])]
    global_train_sets = [
        build_description_dataset(value, bank, split="train", training=True)
        for value in global_configs
    ]
    global_train = EpochConcatDataset(global_train_sets)
    global_loader = build_description_loader(
        global_train,
        config,
        training=True,
        batch_size=config.joint_description_batch_size,
        sampler_seed=loader_seeds["global_caption"],
    )
    region_config = replace(config, stage=config.joint_region_stage)
    region_train = build_description_dataset(region_config, bank, split="train", training=True)
    region_loader = build_description_loader(
        region_train,
        region_config,
        training=True,
        batch_size=config.joint_description_batch_size,
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
        task: _joint_loader_binding(
            task, loader, loader_seed=loader_seeds[task]
        )
        for task, loader in loaders.items()
    }
    caption_val_config = replace(config, stage="rsicap_caption")
    caption_val = build_description_loader(
        build_description_dataset(caption_val_config, bank, split="dev", training=False),
        caption_val_config,
        training=False,
        batch_size=config.joint_description_batch_size,
    )
    region_val_name = "val" if config.joint_region_stage in {"bridge_expert", "predicted_mask"} else None
    region_val_config = (
        replace(region_config, evaluation_mode="fixed_prediction")
        if config.joint_region_stage == "predicted_mask"
        else region_config
    )
    region_val = (
        build_description_loader(
            build_description_dataset(
                region_val_config, bank, split=region_val_name, training=False
            ),
            region_val_config,
            training=False,
            batch_size=config.joint_description_batch_size,
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
        validate_resume_run_config(resume_metadata, dict(config.__dict__))
    elif initialize_from:
        source_step, source_metadata = initialize_segdesc_checkpoint(
            initialize_from,
            model,
            expected_seed=config.seed,
            require_run_completion=True,
        )
        resume_metadata = {"initialized_from": initialize_from, "source": source_metadata}
    d_minus_one_source = (
        resume_metadata if resume else resume_metadata["source"]
    )
    d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
        (d_minus_one_source.get("metadata") or {}).get(
            "d_minus_one_acceptance"
        ),
        expected_description_benchmark=config.description_benchmark,
    )
    stage_lineage = validate_description_stage_lineage(
        (d_minus_one_source.get("metadata") or {}).get("stage_lineage"),
        expected_target_stage=config.joint_region_stage,
    )
    current_region_data_audit = _region_data_audit(region_train)
    initialization_audit = validate_m7_source_checkpoint(
        resume_metadata if resume else resume_metadata["source"],
        region_stage=config.joint_region_stage,
        current_data_audit=current_region_data_audit,
        resume=bool(resume),
        expected_seed=config.seed,
    )
    d4_final_acceptance: dict[str, Any] | None = None
    m6_acceptance: dict[str, Any] | None = None
    if config.joint_region_stage == "predicted_mask":
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
            config.d4_final_acceptance_gate,
            seed=config.seed,
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
                seed=config.seed,
                train_region_data_audit=current_region_data_audit,
            )
        else:
            m6_acceptance = validate_m6_acceptance_for_m7(
                config.m6_acceptance_gate or "",
                seed=config.seed,
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
            expected_seed=config.seed,
            region_stage=config.joint_region_stage,
            region_data_audit=current_region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=migration,
            require_m6_binding=config.joint_region_stage == "predicted_mask",
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
            expected_seed=config.seed,
            region_stage=config.joint_region_stage,
            region_data_audit=current_region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=migration,
            source_step=source_step,
            source_initialization=source_metadata.get("initialization"),
            require_m6_binding=config.joint_region_stage == "predicted_mask",
        )
    segmentation_migration_lineage = joint_initialization_audit[
        "segmentation_migration_lineage"
    ]
    parent_populations = {
        "segmentation": _dataset_parent_ids(segmentation_train.dataset),
        "global_caption": _dataset_parent_ids(global_train),
        "region_description": _dataset_parent_ids(region_train),
    }
    saved_progress = dict((resume_metadata.get("metadata") or {}).get("joint_progress") or {})
    pattern = config.resolved_joint_task_pattern()
    grad_accum = max(1, int(config.grad_accum_steps))
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
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"

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
        "gradient_accumulation_microbatches_per_optimizer_step": config.grad_accum_steps,
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
        "retention_max_drop": config.segmentation_retention_max_drop,
        "resolved_config": dict(config.__dict__),
        "initialized_from": initialize_from,
        "initialization_audit": initialization_audit,
        "joint_initialization_audit": joint_initialization_audit,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "resume_execution_audit": resume_execution_audit,
        "resume_reconciliation": resume_reconciliation,
        "shared_segmentation_dense_trainable": bool(
            config.joint_train_shared_segmentation_dense
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
        f"shared_segmentation_dense={config.joint_train_shared_segmentation_dense} "
        f"baseline_positive_dice={baseline_dice:.4f}"
    )

    # Iterators remain lazy so resume can reconstruct only the next requested stream.
    iterators = {task: None for task in loaders}
    history_path = output_dir / "joint_history.jsonl"
    saved_best_score = (resume_metadata.get("metadata") or {}).get("best_score")
    best_score = -math.inf if saved_best_score is None else float(saved_best_score)
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
            raw_batch, iterators[task] = _next_joint_loader_batch(
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
                    model, region_val, region_val_config, device, split=str(region_val_name),
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
                        "validation": validation, "config": dict(config.__dict__),
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
                        "joint_progress": _joint_progress_payload(
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
        if step % config.save_interval == 0 or step == config.max_steps:
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
                    "config": dict(config.__dict__),
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
                    "joint_progress": _joint_progress_payload(
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
        "checkpoint_best": str(output_dir / "checkpoint_best.pt") if (output_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "d4_final_acceptance": d4_final_acceptance,
        "m6_acceptance": m6_acceptance,
        "joint_initialization_audit": joint_initialization_audit,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "stage_lineage": stage_lineage,
        "resume_reconciliation": resume_reconciliation,
    }
