#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 optimizer, task-isolated gradients, and deterministic loader replay."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch.utils.data import ConcatDataset

from qpsalm_seg.engine.evaluator import (
    SAMPLE_IDENTITY_FIELDS,
    SAMPLE_IDENTITY_PROTOCOL,
)

from ..protocols.config import SegDescConfig
from ..data.engineering_contracts import REGION_TRAINING_DATA_PROTOCOL
from ..data.loaders import set_loader_epoch
from ..modeling.model import DESCRIPTION_ADAPTER_NAME
from ..protocols.io import canonical_sha256
from .checkpoint import capture_training_rng_state, restore_training_rng_state
from .engineering_gates import dataset_data_audit
from .joint_contracts import JOINT_LOADER_BINDING_PROTOCOL


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
            if config.joint.joint_train_shared_segmentation_dense:
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
    if config.joint.joint_train_shared_segmentation_dense:
        required.add("segmentation_dense")
    missing = sorted(required - {role for role, _no_decay_value in groups})
    if missing:
        raise RuntimeError(f"joint optimizer 缺少参数组: {missing}")
    scales = {
        "segmentation_adapter": 0.1,
        "description_adapter": config.training.desc_adapter_lr_scale,
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
            "lr": config.training.learning_rate * scales[role],
            "lr_scale": scales[role],
            "weight_decay": 0.0 if no_decay or "adapter" in role else config.training.weight_decay,
        })
    optimizer = torch.optim.AdamW(parameter_groups)

    def schedule(step: int) -> float:
        if step < config.training.warmup_steps:
            return (step + 1) / max(config.training.warmup_steps, 1)
        progress = (
            step - config.training.warmup_steps
        ) / max(
            config.training.max_steps - config.training.warmup_steps, 1
        )
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def positive_dice(report: dict[str, Any]) -> float:
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
        "positive_dice": positive_dice(report),
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


def dataset_rows_binding(dataset) -> dict[str, Any]:
    """Bind exact ordered rows, including boundaries inside ConcatDataset."""
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        children = [dataset_rows_binding(value) for value in nested]
        payload = {
            "dataset_class": type(dataset).__name__,
            "children": children,
        }
        return {
            "num_rows": sum(int(value["num_rows"]) for value in children),
            "ordered_rows_sha256": canonical_sha256(payload),
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


def joint_loader_binding(
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
        "dataset": dataset_rows_binding(loader.dataset),
        "batches_per_epoch": int(batches_per_epoch),
        "num_workers": int(loader.num_workers),
        "persistent_workers": bool(loader.persistent_workers),
        "prefetch_factor": loader.prefetch_factor,
        "loader_seed": int(loader_seed),
        "worker_seed_protocol": "loader_seed_plus_1000003_times_epoch",
        "batch_sampler": sampler,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def initial_joint_loader_states(
    loader_bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    return {
        task: {"epoch": 0, "batch_in_epoch": 0, "total_microbatches": 0}
        for task in loader_bindings
    }


def loader_iterator_at_cursor(
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


def next_joint_loader_batch(
    loader,
    iterator,
    state: dict[str, int],
    binding: dict[str, Any],
):
    if iterator is None:
        iterator = loader_iterator_at_cursor(loader, state, binding)
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


def joint_gradient_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
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


def dataset_parent_ids(dataset) -> set[str]:
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        return set().union(*(dataset_parent_ids(value) for value in nested))
    return {
        str(row["parent_sample_id"])
        for row in getattr(dataset, "rows", [])
        if row.get("parent_sample_id")
    }


def region_data_audit(dataset) -> dict[str, Any]:
    population = dataset_data_audit(dataset)
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
