#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run strict real-benchmark integration checks for SANE/QMEF/PMRD and Qwen.

用途：在真实 benchmark-v2 上验证 raw forward/backward，以及代表性 batch 的 Qwen QLoRA 可训练性。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.integration_check --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml
--mode all --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3
--device cuda --output outputs/qpsalm_v2/real_integration_report.json
主要输入：完整 small-v2 instruction train/val/test、可选 Qwen vision cache v3 和本地 Qwen 权重。
主要输出：包含代表性 batch、loss、聚合 LoRA 梯度、参数更新、显存和验收状态的 JSON 报告。
写入行为：只写 --output，不保存 checkpoint，不修改 benchmark/cache。
所属流程：small-v2 正式三 seed 实验之前的真实数据与单卡验收门槛。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from qpsalm_seg.config import AMP_DTYPES, QWEN_GRADIENT_CHECKPOINTING_MODES, load_config
from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.data.samplers import task_group
from qpsalm_seg.engine.common import (
    amp_dtype,
    autocast_enabled,
    build_model,
    create_grad_scaler,
    resolve_device,
    set_seed,
)
from qpsalm_seg.metrics import batch_binary_metrics
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset


REPORT_FORMAT = "qpsalm_real_integration_v2"
INTEGRATION_PROTOCOL_VERSION = "qwen_trainability_v6"


class IntegrationFailure(RuntimeError):
    """Integration error that preserves completed diagnostics in the report."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def runtime_library_report(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("transformers", "peft", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        **packages,
    }
    if device.type == "cuda":
        report.update({
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        })
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict real benchmark-v2 integration check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--mode", choices=("raw", "qwen", "all"), default="all")
    parser.add_argument("--raw-preset", choices=PRESET_CHOICES, default="raw_sane_qmef_pmrd")
    parser.add_argument("--qwen-preset", choices=PRESET_CHOICES, default="qwen_psalm_full")
    parser.add_argument(
        "--qwen-check",
        choices=("launch", "diagnostic"),
        default="launch",
        help="launch runs one representative end-to-end batch; diagnostic adds a two-step controller-only probe.",
    )
    parser.add_argument("--amp-dtype", choices=AMP_DTYPES, default=None)
    parser.add_argument("--vision-feature-cache", default=None)
    parser.add_argument(
        "--qwen-gradient-checkpointing",
        choices=QWEN_GRADIENT_CHECKPOINTING_MODES,
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory-gib", type=float, default=23.0)
    parser.add_argument("--output", default="outputs/qpsalm_v2/real_integration_report.json")
    return parser.parse_args()


def _task_group(row: dict[str, Any]) -> str:
    family = str(row.get("task_family") or "")
    if family == "referring_landslide_segmentation":
        return "referring"
    if family == "no_target_segmentation":
        return "no_target"
    return "global"


def _is_multimodal(row: dict[str, Any]) -> bool:
    available = [
        value for value in (row.get("modalities") or {}).values()
        if isinstance(value, dict) and value.get("available", True)
    ]
    return len(available) > 1


def select_real_indices(dataset: MultiSourceLandslideDataset) -> dict[str, int]:
    """Select auditable global/referring/no-target rows, preferring multimodal evidence."""
    candidates: dict[str, list[int]] = {"global": [], "referring": [], "no_target": []}
    for index, row in enumerate(dataset.rows):
        candidates[_task_group(row)].append(index)
    missing = [name for name, values in candidates.items() if not values]
    if missing:
        raise RuntimeError(f"真实 integration split 缺少任务组: {missing}")
    selected = {}
    for name, values in candidates.items():
        selected[name] = next((index for index in values if _is_multimodal(dataset.rows[index])), values[0])
    return selected


def _sample_record(item: dict[str, Any], output, metric: dict[str, float]) -> dict[str, Any]:
    meta = item["metadata"]
    return {
        "sample_id": meta.get("sample_id"),
        "parent_sample_id": meta.get("parent_sample_id"),
        "task_family": meta.get("task_family"),
        "active_modalities": list(meta.get("active_modalities") or []),
        "full_modalities": list(meta.get("full_modalities") or []),
        "active_subset": meta.get("active_subset"),
        "valid_coverage": meta.get("valid_coverage"),
        "loss": float(output["loss"].detach().float().cpu()),
        "iou": float(metric["iou"]),
        "dice": float(metric["dice"]),
        "component_count": float(output["proposal_component_count"].detach().float().mean().cpu()),
        "visual_evidence_key": item["visual_evidence_key"],
    }


def _gradient_report(model: torch.nn.Module) -> dict[str, float]:
    values = {
        name: float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    finite = all(math.isfinite(value) for value in values.values())
    return {
        "num_parameters_with_grad": len(values),
        "gradient_norm_sum": sum(values.values()),
        "all_finite": finite,
    }


def run_raw_check(config, device: torch.device) -> dict[str, Any]:
    stable = replace(
        config,
        controller="text_probe",
        vision_feature_cache=None,
        use_pretrained_sane=False,
        modality_dropout=0.0,
        train_hflip_prob=0.0,
        train_vflip_prob=0.0,
        missing_modality_consistency_weight=0.0,
        num_workers=0,
    )
    dataset = MultiSourceLandslideDataset(stable, "train")
    indices = select_real_indices(dataset)
    model = build_model(stable, device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=stable.lr,
        weight_decay=stable.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    records = []
    for index in indices.values():
        item = dataset[index]
        batch = qpsalm_collate([item])
        output = model(batch)
        if not torch.isfinite(output["loss"]):
            raise RuntimeError(f"raw integration loss 非有限: sample={item['metadata']['sample_id']}")
        (output["loss"] / len(indices)).backward()
        metric = batch_binary_metrics(
            output["final_mask_logits"].detach().cpu(),
            batch.mask,
            valid_mask=batch.valid_mask,
        )[0]
        records.append(_sample_record(item, output, metric))
    gradients = _gradient_report(model)
    if not gradients["all_finite"] or gradients["gradient_norm_sum"] <= 0:
        raise RuntimeError(f"raw integration 梯度无效: {gradients}")
    optimizer.step()
    return {
        "status": "passed",
        "preset": stable.preset,
        "device": str(device),
        "selected_indices": indices,
        "samples": records,
        "gradients": gradients,
    }


def _row_families(row: dict[str, Any]) -> set[str]:
    return {
        str(value.get("family"))
        for value in (row.get("modalities") or {}).values()
        if isinstance(value, dict) and value.get("available", True) and value.get("family")
    }


def select_representative_batch_indices(
    dataset: MultiSourceLandslideDataset,
    batch_size: int,
) -> list[int]:
    """Select one costly, sampler-shaped multisource batch with distinct parents."""
    grouped: dict[tuple[int, int, str], list[int]] = {}
    for index, row in enumerate(dataset.rows):
        if len(_row_families(row)) < 3:
            continue
        if bool((row.get("mask") or {}).get("empty_mask")):
            continue
        if str(row.get("task_family") or "") == "no_target_segmentation":
            continue
        spatial_bucket = int(dataset.bucket_size(index))
        load_bucket = int(
            dataset.sequence_load_bucket(index)
            if hasattr(dataset, "sequence_load_bucket") else 0
        )
        grouped.setdefault((spatial_bucket, load_bucket, task_group(row)), []).append(index)
    required = max(1, int(batch_size))
    for bucket in sorted(grouped, key=lambda value: (value[0], value[1], value[2]), reverse=True):
        selected: list[int] = []
        parents: set[str] = set()
        for index in sorted(grouped[bucket], key=lambda value: str(dataset.rows[value].get("sample_id"))):
            row = dataset.rows[index]
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            if parent in parents:
                continue
            parents.add(parent)
            selected.append(index)
            if len(selected) == required:
                return selected
    raise RuntimeError(
        "Qwen representative gate 缺少 "
        f"{required} 个同空间/负载/任务组、正样本、多源、不同 parent 的样本"
    )


def lora_gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, list[tuple[str, float]]] = {"lora_A": [], "lora_B": []}
    for name, parameter in model.controller.model.named_parameters():
        matrix = "lora_A" if "lora_A." in name else "lora_B" if "lora_B." in name else None
        if matrix is None or parameter.grad is None:
            continue
        groups[matrix].append((name, float(parameter.grad.detach().float().norm().cpu())))

    def summarize(values: list[tuple[str, float]]) -> dict[str, Any]:
        norms = [value for _, value in values]
        return {
            "num_with_grad": len(values),
            "num_nonzero": sum(value > 0.0 for value in norms),
            "norm_sum": sum(norms),
            "all_finite": all(math.isfinite(value) for value in norms),
            "nonzero_names": [name for name, value in values if value > 0.0],
        }

    by_matrix = {name: summarize(values) for name, values in groups.items()}
    return {
        "by_matrix": by_matrix,
        "num_with_grad": sum(value["num_with_grad"] for value in by_matrix.values()),
        "num_nonzero": sum(value["num_nonzero"] for value in by_matrix.values()),
        "norm_sum": sum(value["norm_sum"] for value in by_matrix.values()),
        "all_finite": all(value["all_finite"] for value in by_matrix.values()),
    }




def snapshot_lora_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.controller.model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    }


def lora_parameter_update_summary(
    model: torch.nn.Module,
    before: dict[str, torch.Tensor],
) -> dict[str, Any]:
    deltas = []
    changed = 0
    by_matrix: dict[str, list[torch.Tensor]] = {"lora_A": [], "lora_B": []}
    for name, parameter in model.controller.model.named_parameters():
        if name not in before:
            continue
        delta = (parameter.detach().float().cpu() - before[name]).norm()
        deltas.append(delta)
        changed += int(float(delta) > 0.0)
        matrix = "lora_A" if "lora_A." in name else "lora_B" if "lora_B." in name else None
        if matrix is not None:
            by_matrix[matrix].append(delta)
    if not deltas:
        return {
            "num_parameters": 0,
            "num_changed": 0,
            "norm_sum": 0.0,
            "all_finite": True,
            "by_matrix": {},
        }
    values = torch.stack(deltas)
    return {
        "num_parameters": len(deltas),
        "num_changed": changed,
        "norm_sum": float(values.sum()),
        "all_finite": bool(torch.isfinite(values).all()),
        "by_matrix": {
            name: {
                "num_parameters": len(matrix_deltas),
                "num_changed": sum(float(value) > 0.0 for value in matrix_deltas),
                "norm_sum": float(torch.stack(matrix_deltas).sum()) if matrix_deltas else 0.0,
            }
            for name, matrix_deltas in by_matrix.items()
        },
    }


def _controller_probe_loss(semantic) -> torch.Tensor:
    masks = semantic.mask_query_states.float()
    anchors = semantic.evidence_anchors.float()
    mask_target = torch.linspace(-0.5, 0.5, masks.numel(), device=masks.device).reshape_as(masks)
    anchor_target = torch.linspace(0.25, -0.25, anchors.numel(), device=anchors.device).reshape_as(anchors)
    return torch.nn.functional.mse_loss(masks, mask_target) + 0.1 * torch.nn.functional.mse_loss(
        anchors, anchor_target
    )


def run_controller_trainability_probe(model, batch) -> dict[str, Any]:
    lora_parameters = [
        parameter
        for name, parameter in model.controller.model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(lora_parameters, lr=1.0e-3, weight_decay=0.0)
    steps = []
    for step in range(2):
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        before = snapshot_lora_parameters(model)
        with model.controller.trace_lora_execution() as execution:
            semantic = model.controller.encode_batch(batch, use_full=False)
            loss = _controller_probe_loss(semantic)
        if not torch.isfinite(loss):
            raise IntegrationFailure("Qwen controller-only probe loss 非有限")
        loss.backward()
        gradients = lora_gradient_report(model)
        optimizer.step()
        updates = lora_parameter_update_summary(model, before)
        record = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "executed_module_count": sum(value > 0 for value in execution.values()),
            "execution_counts": dict(execution),
            "gradients": gradients,
            "parameter_update": updates,
        }
        steps.append(record)
        if record["executed_module_count"] <= 0:
            raise IntegrationFailure(
                "Qwen controller-only probe 未执行任何 LoRA projection",
                {"controller_probe": {"steps": steps}},
            )
        expected = "lora_B" if step == 0 else "lora_A"
        if gradients["by_matrix"][expected]["num_nonzero"] <= 0:
            raise IntegrationFailure(
                f"Qwen controller-only probe 第 {step + 1} 步 {expected} 梯度为零",
                {"controller_probe": {"steps": steps}},
            )
        if updates["by_matrix"][expected]["num_changed"] <= 0:
            raise IntegrationFailure(
                f"Qwen controller-only probe 第 {step + 1} 步 {expected} 参数未更新",
                {"controller_probe": {"steps": steps}},
            )
    return {"status": "passed", "steps": steps}


@contextmanager
def trace_end_to_end_query_gradients(model):
    tensors: dict[str, list[torch.Tensor]] = {
        "qwen_hidden_states": [],
        "mask_query_states": [],
        "coarse_queries": [],
        "refined_queries": [],
    }

    def capture(name: str, predicate=None):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                return
            if predicate is not None and not predicate(output):
                return
            output.retain_grad()
            tensors[name].append(output)

        return hook

    def capture_input(name: str, predicate=None):
        def hook(_module, inputs):
            value = inputs[0]
            if not torch.is_tensor(value) or not value.requires_grad:
                return
            if predicate is not None and not predicate(value):
                return
            value.retain_grad()
            tensors[name].append(value)

        return hook

    handles = [
        model.controller.output_projection.register_forward_pre_hook(
            capture_input(
                "qwen_hidden_states",
                lambda value: value.ndim == 3 and value.shape[1] == model.controller.num_queries,
            )
        ),
        model.controller.output_projection.register_forward_hook(
            capture(
                "mask_query_states",
                lambda value: value.ndim == 3 and value.shape[1] == model.controller.num_queries,
            )
        ),
        model.pmrd.coarse_decoder.register_forward_hook(capture("coarse_queries")),
        model.pmrd.refine_norm.register_forward_hook(capture("refined_queries")),
    ]
    try:
        yield tensors
    finally:
        for handle in handles:
            handle.remove()


def query_gradient_report(tensors: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    report = {}
    for name, values in tensors.items():
        norms = [
            float(value.grad.detach().float().norm().cpu())
            for value in values
            if value.grad is not None
        ]
        report[name] = {
            "num_captured": len(values),
            "num_with_grad": len(norms),
            "num_nonzero": sum(value > 0.0 for value in norms),
            "norm_sum": sum(norms),
            "all_finite": all(math.isfinite(value) for value in norms),
        }
    return report


def run_student_only_end_to_end_probe(model, batch, *, autocast: bool, dtype: torch.dtype) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    original_config = model.config
    model.config = replace(model.config, missing_modality_consistency_weight=0.0)
    try:
        with trace_end_to_end_query_gradients(model) as query_tensors:
            with model.controller.trace_lora_execution() as execution:
                with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
                    output = model(batch)
                    loss = output["loss"]
        if not torch.isfinite(loss):
            raise IntegrationFailure("Qwen student-only end-to-end probe loss 非有限")
        loss.backward()
    finally:
        model.config = original_config
    gradients = lora_gradient_report(model)
    query_gradients = query_gradient_report(query_tensors)
    report = {
        "loss": float(loss.detach().float().cpu()),
        "executed_module_count": sum(value > 0 for value in execution.values()),
        "execution_counts": dict(execution),
        "lora_gradients": gradients,
        "query_gradients": query_gradients,
    }
    if report["executed_module_count"] <= 0:
        raise IntegrationFailure(
            "Qwen student-only probe 未执行任何 LoRA projection",
            {"student_only_probe": report},
        )
    if not gradients["all_finite"] or gradients["num_nonzero"] <= 0:
        raise IntegrationFailure(
            "Qwen controller 可训练，但 student-only segmentation loss 未到达 LoRA",
            {"student_only_probe": report},
        )
    if any(
        value["num_nonzero"] <= 0 or not value["all_finite"]
        for value in query_gradients.values()
    ):
        raise IntegrationFailure(
            "Qwen student-only mask/coarse/refined query 梯度链路中断",
            {"student_only_probe": report},
        )
    return {"status": "passed", **report}


def run_qwen_check(
    config,
    device: torch.device,
    max_memory_gib: float,
    *,
    diagnostic: bool = False,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Qwen integration 必须显式使用 CUDA device")
    if config.controller != "qwen_mask_query" or not config.use_pretrained_sane:
        raise RuntimeError(
            f"Qwen integration 需要 qwen_mask_query + pretrained SANE，当前 preset={config.preset}"
        )
    strict = replace(
        config,
        train_hflip_prob=0.0,
        train_vflip_prob=0.0,
        missing_modality_consistency_weight=max(
            0.1, float(config.missing_modality_consistency_weight)
        ),
        num_workers=0,
    )
    full_dataset = MultiSourceLandslideDataset(replace(strict, modality_dropout=0.0), "train")
    dropped_dataset = MultiSourceLandslideDataset(replace(strict, modality_dropout=1.0), "train")
    representative_indices = select_representative_batch_indices(
        full_dataset, strict.batch_size
    )
    representative_items = [
        (full_dataset if offset % 2 == 0 else dropped_dataset)[index]
        for offset, index in enumerate(representative_indices)
    ]
    if not any(not item["active_subset"].is_full for item in representative_items):
        raise RuntimeError("Qwen representative batch 未覆盖 dropped-modality student")
    model = build_model(strict, device).train()
    if model.vision_bank is None:
        raise RuntimeError("Qwen integration 缺少 vision feature bank")
    trainable_qwen = [
        name for name, parameter in model.controller.model.named_parameters()
        if parameter.requires_grad
    ]
    illegal_qwen = [name for name in trainable_qwen if "lora_" not in name]
    if illegal_qwen or not trainable_qwen:
        raise RuntimeError(
            f"Qwen QLoRA 参数隔离失败: trainable={trainable_qwen[:8]} illegal={illegal_qwen[:8]}"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=strict.lr, weight_decay=strict.weight_decay)
    scaler = create_grad_scaler(strict, device)
    autocast = autocast_enabled(strict, device)
    dtype = amp_dtype(strict, device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch = qpsalm_collate(representative_items)
    controller_probe = run_controller_trainability_probe(model, batch) if diagnostic else None
    student_only_probe = None
    if diagnostic:
        try:
            student_only_probe = run_student_only_end_to_end_probe(
                model,
                batch,
                autocast=autocast,
                dtype=dtype,
            )
        except IntegrationFailure as exc:
            raise IntegrationFailure(
                str(exc),
                {
                    "protocol_version": INTEGRATION_PROTOCOL_VERSION,
                    "controller_probe": controller_probe,
                    **exc.details,
                },
            ) from exc
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    lora_before = snapshot_lora_parameters(model)
    tap_context = trace_end_to_end_query_gradients(model) if diagnostic else nullcontext({})
    with tap_context as query_tensors:
        with model.controller.trace_lora_execution() as execution:
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
                output = model(batch)
                loss = output["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("Qwen representative batch loss 非有限")
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    gradients = _gradient_report(model)
    lora_gradients = lora_gradient_report(model)
    query_gradients = query_gradient_report(query_tensors) if diagnostic else None
    sequence_lengths = output["controller_sequence_lengths"].detach().cpu().tolist()
    visual_counts = output["controller_visual_token_counts"].detach().cpu().tolist()
    parents = [str(item["metadata"]["parent_sample_id"]) for item in representative_items]
    task_groups = sorted({task_group(full_dataset.rows[index]) for index in representative_indices})
    spatial_buckets = sorted({int(full_dataset.bucket_size(index)) for index in representative_indices})
    load_buckets = sorted({int(full_dataset.sequence_load_bucket(index)) for index in representative_indices})
    representative_batch = {
        "indices": representative_indices,
        "batch_size": batch.batch_size,
        "sample_ids": [str(item["metadata"]["sample_id"]) for item in representative_items],
        "parent_sample_ids": parents,
        "unique_parent_count": len(set(parents)),
        "task_groups": task_groups,
        "spatial_buckets": spatial_buckets,
        "sequence_load_buckets": load_buckets,
        "active_family_combos": [str(item["metadata"]["family_combo"]) for item in representative_items],
        "full_family_combos": [
            "+".join(sorted({instance.family for instance in item["full_instances"]}))
            for item in representative_items
        ],
        "full_count": sum(item["active_subset"].is_full for item in representative_items),
        "dropped_count": sum(not item["active_subset"].is_full for item in representative_items),
        "teacher_sample_count": float(output["teacher_sample_count"].detach().cpu()),
        "sequence_lengths": sequence_lengths,
        "visual_token_counts": visual_counts,
        "padding_ratio": 1.0 - sum(sequence_lengths) / max(
            len(sequence_lengths) * max(sequence_lengths), 1
        ),
        "loss": float(loss.detach().float().cpu()),
        "global_gradients": gradients,
        "lora_execution": {
            "executed_module_count": sum(value > 0 for value in execution.values()),
            "execution_counts": dict(execution),
        },
        "lora_gradients": lora_gradients,
        "lora_parameter_state": {
            "num_parameters": len(trainable_qwen),
            "num_requires_grad": sum(
                parameter.requires_grad
                for name, parameter in model.controller.model.named_parameters()
                if "lora_" in name
            ),
        },
        "lora_runtime_status": model.controller.lora_runtime_status(),
        **({"query_gradients": query_gradients} if query_gradients is not None else {}),
    }
    failure_details = {
        "protocol_version": INTEGRATION_PROTOCOL_VERSION,
        "representative_batch": representative_batch,
        **({"controller_probe": controller_probe} if controller_probe is not None else {}),
        **({"student_only_probe": student_only_probe} if student_only_probe is not None else {}),
    }
    if (
        batch.batch_size != int(strict.batch_size)
        or len(set(parents)) != batch.batch_size
        or len(task_groups) != 1
        or len(spatial_buckets) != 1
        or len(load_buckets) != 1
        or representative_batch["dropped_count"] <= 0
        or representative_batch["teacher_sample_count"] <= 0
    ):
        raise IntegrationFailure(
            "Qwen representative batch 不符合真实训练 batch 约束",
            failure_details,
        )
    if (
        representative_batch["lora_execution"]["executed_module_count"] <= 0
    ):
        raise IntegrationFailure(
            "Qwen representative batch 未执行任何 LoRA projection",
            failure_details,
        )
    if diagnostic and any(
        value["num_nonzero"] <= 0 or not value["all_finite"]
        for value in query_gradients.values()
    ):
        raise IntegrationFailure(
            "Qwen controller 可训练，但端到端 mask/coarse/refined query 梯度链路中断",
            failure_details,
        )
    if (
        not gradients["all_finite"]
        or gradients["gradient_norm_sum"] <= 0
        or not lora_gradients["all_finite"]
        or lora_gradients["num_nonzero"] <= 0
    ):
        raise IntegrationFailure(
            "Qwen LoRA projection 已执行，但端到端 segmentation loss 未产生有效 LoRA 梯度",
            failure_details,
        )
    torch.nn.utils.clip_grad_norm_(trainable, strict.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    lora_update = lora_parameter_update_summary(model, lora_before)
    representative_batch["lora_parameter_update"] = lora_update
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    representative_batch["peak_allocated_gib"] = peak_allocated
    representative_batch["peak_reserved_gib"] = peak_reserved
    if (
        not lora_update["all_finite"]
        or lora_update["num_changed"] <= 0
        or lora_update["norm_sum"] <= 0
    ):
        raise IntegrationFailure(
            f"Qwen representative batch LoRA 参数未更新: {lora_update}",
            failure_details,
        )
    if peak_reserved > float(max_memory_gib):
        raise IntegrationFailure(
            f"Qwen integration 峰值显存超过门槛: reserved={peak_reserved:.3f} GiB "
            f"limit={max_memory_gib:.3f} GiB",
            failure_details,
        )
    return {
        "status": "passed",
        "protocol_version": INTEGRATION_PROTOCOL_VERSION,
        "preset": strict.preset,
        "device": str(device),
        "representative_batch": representative_batch,
        **({"controller_probe": controller_probe} if controller_probe is not None else {}),
        **({"student_only_probe": student_only_probe} if student_only_probe is not None else {}),
        "cache": {
            "format": model.vision_bank.manifest["format"],
        },
        "qwen": {
            "num_trainable_lora_parameters": len(trainable_qwen),
            "gradient_checkpointing": model.controller.gradient_checkpointing_mode,
            "gradient_checkpointing_kwargs": model.controller.gradient_checkpointing_kwargs,
            "amp_dtype": strict.amp_dtype,
            "trainable_lora_dtypes": sorted({
                str(parameter.dtype)
                for name, parameter in model.controller.model.named_parameters()
                if parameter.requires_grad and "lora_" in name
            }),
            "runtime_status": model.controller.lora_runtime_status(),
        },
        "runtime_libraries": runtime_library_report(device),
        "memory": {
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "limit_gib": float(max_memory_gib),
            "underutilized_warning": peak_reserved < 18.0,
            "note": "nvidia-smi also includes quantized base weights, CUDA context and allocator overhead",
        },
    }


def _write_report(path_ref: str, payload: dict[str, Any]) -> Path:
    path = resolve_project_path(path_ref) or Path(path_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    overrides = {"benchmark_dir": args.benchmark_dir}
    base = load_config(args.config, overrides=overrides)
    if args.amp_dtype is not None:
        base = replace(base, amp_dtype=args.amp_dtype)
    device = resolve_device(args.device)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "created_unix": time.time(),
        "benchmark_dir": str(base.benchmark_path()),
        "mode": args.mode,
        "seed": args.seed,
        "amp_dtype": base.amp_dtype,
        "qwen_check": args.qwen_check,
        "checks": {},
    }
    errors, warnings = [], []
    if args.mode in {"raw", "all"}:
        try:
            raw_config = apply_preset(base, args.raw_preset)
            report["checks"]["raw"] = run_raw_check(raw_config, device)
        except Exception as exc:
            report["checks"]["raw"] = {"status": "failed", "error": str(exc)}
            errors.append(f"raw: {exc}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.mode in {"qwen", "all"}:
        try:
            qwen_config = apply_preset(base, args.qwen_preset)
            cache = args.vision_feature_cache or qwen_config.vision_feature_cache
            qwen_config = replace(
                qwen_config,
                vision_feature_cache=cache,
                qwen_gradient_checkpointing=(
                    args.qwen_gradient_checkpointing
                    or qwen_config.qwen_gradient_checkpointing
                ),
            )
            report["checks"]["qwen"] = run_qwen_check(
                qwen_config,
                device,
                args.max_memory_gib,
                diagnostic=args.qwen_check == "diagnostic",
            )
            memory = report["checks"]["qwen"].get("memory") or {}
            if memory.get("underutilized_warning"):
                warnings.append(
                    f"qwen peak_reserved_gib={memory.get('peak_reserved_gib', 0):.3f} < 18.0"
                )
        except Exception as exc:
            report["checks"]["qwen"] = {
                "status": "failed",
                "error": str(exc),
                **(exc.details if isinstance(exc, IntegrationFailure) else {}),
            }
            errors.append(f"qwen: {exc}")
    report["acceptance"] = {
        "passed": not errors,
        "required_checks": ["raw", "qwen"] if args.mode == "all" else [args.mode],
        "errors": errors,
        "warnings": warnings,
    }
    path = _write_report(args.output, report)
    print(json.dumps({"report": str(path), **report["acceptance"]}, ensure_ascii=False))
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
