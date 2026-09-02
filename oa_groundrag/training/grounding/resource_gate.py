"""M7 冻结最坏样本的真实 CUDA forward/backward 资源门。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import atomic_write_json, first_symlink_component
from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.training.vlm.cuda_telemetry import (
    CudaMicrobatchTelemetry,
    CudaTelemetryPolicy,
)
from oa_groundrag.training.vlm.input_pipeline import DeterministicBatchPlanner
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.vlm.processing import DescriptionCollator

from .config import Stage5Config
from .data import Stage5MixedDataset, Stage5MixedSampler
from .resource_profile import verify_stage5_resource_profile


WORST_CASE_GATE_SCHEMA = "rs_vlm.mask_grounded_worst_case_cuda_gate.v1"


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=False)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _metadata(
    *,
    selection_item: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    labels = batch.get("labels")
    pixels = batch.get("pixel_values")
    grid = batch.get("image_grid_thw")
    if not all(isinstance(value, Tensor) for value in (labels, pixels, grid)):
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "最坏样本 batch 缺少 labels/pixel/grid",
        )
    assert isinstance(labels, Tensor)
    assert isinstance(pixels, Tensor)
    assert isinstance(grid, Tensor)
    return {
        "probe_kind": "frozen_worst_case",
        "optimizer_step": 0,
        "micro_step": int(selection_item["sequence_index"]) + 1,
        "epoch": int(selection_item["epoch"]),
        "record_id": str(selection_item["record_id"]),
        "parent_id": str(selection_item["parent_id"]),
        "logical_role": str(selection_item["logical_role"]),
        "task_family": str(selection_item["task_family"]),
        "selection_reasons": list(selection_item["reasons"]),
        "input_tokens": int(batch["input_token_counts"][0]),
        "supervised_tokens": int(labels[:, 1:].ne(-100).sum()),
        "image_count": int(batch["image_counts"][0]),
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
        "pixel_numel": int(pixels.numel()),
        "image_grid_thw": grid.tolist(),
    }


def _selected_batches(
    *,
    config: Stage5Config,
    dataset: Stage5MixedDataset,
    collator: DescriptionCollator,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    verify_stage5_resource_profile(config)
    assert config.resource_contract is not None
    selection = read_json(
        config.resource_contract.profile_root / "worst_case_selection.json"
    )
    selected = tuple(selection["selected"])
    selected_by_index = {
        int(item["sequence_index"]): item for item in selected
    }
    if not selected_by_index or len(selected_by_index) != len(selected):
        raise ModelError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "最坏样本 selection 为空或 sequence index 重复",
        )
    planner = DeterministicBatchPlanner(
        dataset=dataset,
        sampler=Stage5MixedSampler(dataset, seed=config.run.seed),
        batch_size=1,
        max_epochs=config.training.epochs,
        start_epoch=0,
        start_sample_offset=0,
        max_batches=max(selected_by_index) + 1,
    )
    output: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for sequence_index in range(max(selected_by_index) + 1):
        plan = planner.next_batch()
        item = selected_by_index.get(sequence_index)
        if item is None:
            continue
        sample = plan.samples[0]
        if (
            sample.record_id != item["record_id"]
            or sample.parent_id != item["parent_id"]
            or sample.logical_role != item["logical_role"]
            or sample.task_family != item["task_family"]
            or plan.sample_epochs[0] != item["epoch"]
        ):
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "最坏样本 selection 与当前 sampler trace 漂移",
            )
        output.append((item, collator(plan.samples)))
    if len(output) != len(selected):
        raise AssertionError("未恢复全部最坏样本")
    return output


def run_worst_case_cuda_gate(
    *,
    config: Stage5Config,
    dataset: Stage5MixedDataset,
    collator: DescriptionCollator,
    model: Any,
    device: torch.device,
    policy: CudaTelemetryPolicy,
    output_root: Path,
) -> dict[str, Any]:
    """逐条 backward，不执行 optimizer step；失败证据保留在独立新根。"""

    root = Path(output_root)
    selected = _selected_batches(config=config, dataset=dataset, collator=collator)
    linked = first_symlink_component(root)
    if linked is not None or root.exists() or root.is_symlink():
        raise ModelError(
            ReasonCode.OUTPUT_EXISTS,
            f"最坏样本 CUDA gate 要求全新普通根：{root}",
        )
    root.mkdir(parents=True)
    telemetry = CudaMicrobatchTelemetry(policy=policy, device=device)
    telemetry.start(root, completed_microbatches=0)
    trainable = [
        parameter for parameter in model.model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "最坏样本 CUDA gate 没有可训练参数",
        )
    torch.manual_seed(config.run.seed)
    torch.cuda.manual_seed_all(config.run.seed)
    model.train()
    rows = []
    completed = 0
    try:
        for item, cpu_batch in selected:
            for parameter in trainable:
                parameter.grad = None
            metadata = _metadata(selection_item=item, batch=cpu_batch)
            batch = _move_batch(cpu_batch, device)
            telemetry.begin(metadata)
            result = model.forward(batch)
            telemetry.after_forward()
            loss = getattr(result, "loss", None)
            if (
                not isinstance(loss, Tensor)
                or loss.ndim != 0
                or not bool(torch.isfinite(loss))
            ):
                raise ModelError(
                    ReasonCode.NONFINITE_NUMBER,
                    "最坏样本 CUDA gate loss 非有限",
                )
            loss.backward()
            telemetry.after_backward()
            gradient_values = [
                parameter.grad
                for parameter in trainable
                if parameter.grad is not None
            ]
            if len(gradient_values) != len(trainable) or any(
                not bool(torch.isfinite(gradient).all())
                for gradient in gradient_values
            ):
                raise ModelError(
                    ReasonCode.NONFINITE_NUMBER,
                    "最坏样本 CUDA gate gradient 缺失或非有限",
                )
            gradient_norm = math.sqrt(
                sum(
                    float(gradient.detach().float().square().sum().cpu())
                    for gradient in gradient_values
                )
            )
            loss_value = float(loss.detach().cpu())
            del gradient_values, loss, result, batch, cpu_batch
            for parameter in trainable:
                parameter.grad = None
            telemetry.complete_after_release()
            rows.append(
                {
                    "sequence_index": item["sequence_index"],
                    "record_id": item["record_id"],
                    "reasons": item["reasons"],
                    "loss": loss_value,
                    "gradient_norm": gradient_norm,
                    "finite": True,
                }
            )
            completed += 1
    except BaseException as error:
        telemetry.persist_failure(
            error,
            last_completed_optimizer_step=0,
            last_completed_microbatches=completed,
        )
        raise
    report = {
        "schema_version": WORST_CASE_GATE_SCHEMA,
        "status": "passed",
        "resource_policy": policy.layout_identity(),
        "selected_count": len(rows),
        "results": rows,
        "telemetry_sha256": sha256_file(
            root / "cuda_resource_telemetry.jsonl"
        ),
        "allocator_identity_sha256": sha256_file(
            root / "cuda_resource_identity.json"
        ),
        "formal_acceptance": False,
        "scientific_acceptance": False,
    }
    atomic_write_json(root / "worst_case_gate_report.json", report)
    return report


def verify_worst_case_cuda_gate(
    *,
    config: Stage5Config,
    policy: CudaTelemetryPolicy,
    output_root: Path,
) -> dict[str, Any]:
    verify_stage5_resource_profile(config)
    root = Path(output_root)
    linked = first_symlink_component(root)
    if linked is not None or not root.is_dir() or root.is_symlink():
        raise ModelError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "最坏样本 CUDA gate 根不存在或含 symlink",
        )
    report = read_json(root / "worst_case_gate_report.json")
    if (
        report.get("schema_version") != WORST_CASE_GATE_SCHEMA
        or report.get("status") != "passed"
        or report.get("resource_policy") != policy.layout_identity()
        or report.get("selected_count") != len(report.get("results", ()))
        or report.get("selected_count") <= 0
        or report.get("telemetry_sha256")
        != sha256_file(root / "cuda_resource_telemetry.jsonl")
        or report.get("allocator_identity_sha256")
        != sha256_file(root / "cuda_resource_identity.json")
        or any(
            row.get("finite") is not True
            or not math.isfinite(float(row.get("loss", math.nan)))
            or not math.isfinite(float(row.get("gradient_norm", math.nan)))
            for row in report.get("results", ())
        )
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "最坏样本 CUDA gate report/telemetry 漂移",
        )
    return report
