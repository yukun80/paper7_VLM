"""OA-AuxSeg 评价指标与 checkpoint 评价入口。"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from oa_groundrag.data.oa_auxseg.dataset import atomic_write_json
from oa_groundrag.segmentation.checkpoint import model_from_checkpoint, read_checkpoint
from oa_groundrag.segmentation.config import resolve_runtime
from oa_groundrag.segmentation.contracts import OAAuxSegBatch, RuntimeConfig
from oa_groundrag.segmentation.data import (
    benchmark_contract_from_root,
    make_dataloader,
    prepare_collated_batch,
)
from oa_groundrag.segmentation.losses import bce_dice_loss
from oa_groundrag.segmentation.metrics import SegmentationMetrics
from oa_groundrag.segmentation.model import OAAuxSegModel
from oa_groundrag.segmentation.policy import (
    autocast_context as _autocast,
    prepare_policy_batch,
    validate_benchmark_registry as _validate_benchmark_registry,
    validate_inference_config as _validate_inference_config,
)
from oa_groundrag.segmentation.validity import resample_validity, validity_from_channels
from oa_groundrag.training.segmentation.progress import TrainingProgress

def _active_label(names: Sequence[str]) -> str:
    return "+".join(sorted(names)) if names else "none"
def _subset_category(
    active: Sequence[str], available: Sequence[str]
) -> str:
    if not active:
        return "none"
    if len(active) == 1:
        return "single"
    if len(active) == len(available):
        return "all"
    return "multi"
def _batch_conditional_weight_diagnostics(
    *,
    model: OAAuxSegModel,
    batch: OAAuxSegBatch,
    output: Any,
    active: Sequence[Sequence[str]],
) -> dict[str, Any]:
    covered: list[set[str]] = [set() for _ in active]
    for modality, packed in batch.auxiliaries.items():
        local_valid = (
            packed.pixel_valid.to(torch.bool)
            & packed.channel_valid.to(torch.bool)[:, :, None, None]
        ).flatten(1).any(dim=1)
        for local_index, sample_index in enumerate(
            packed.sample_indices.tolist()
        ):
            if bool(local_valid[local_index].item()):
                covered[int(sample_index)].add(modality)
    selected = [
        set(names) & covered[index]
        for index, names in enumerate(active)
    ]
    global_values: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for modality_index, modality in enumerate(model.modality_order):
        indices = [
            index
            for index, names in enumerate(selected)
            if modality in names
        ]
        counts[modality] = len(indices)
        global_values[modality] = (
            float(
                output.modality_weights[indices, modality_index]
                .detach()
                .float()
                .mean()
                .item()
            )
            if indices
            else None
        )
    auxiliary_indices = [
        index for index, names in enumerate(selected) if names
    ]
    counts["__null__"] = len(auxiliary_indices)
    global_values["__null__"] = (
        float(
            output.modality_weights[auxiliary_indices, -1]
            .detach()
            .float()
            .mean()
            .item()
        )
        if auxiliary_indices
        else None
    )
    stage_values: dict[str, dict[str, float | None]] = {}
    for stride, weight_map in zip(
        output.modality_weight_map_strides,
        output.modality_weight_maps,
        strict=True,
    ):
        coverage = torch.cat(
            [
                resample_validity(
                    validity_from_channels(
                        pixel_valid.unsqueeze(0),
                        channel_valid.unsqueeze(0),
                    ),
                    weight_map.shape[-2:],
                ).coverage
                for pixel_valid, channel_valid in zip(
                    batch.optical_pixel_valid,
                    batch.optical_channel_valid,
                    strict=True,
                )
            ],
            dim=0,
        ).float()
        summary = (
            weight_map.float() * coverage
        ).sum(dim=(2, 3)) / coverage.sum(dim=(2, 3)).clamp_min(1e-6)
        stage_values[f"stride{stride}"] = {
            name: (
                float(
                    summary[
                        (
                            [
                                index
                                for index, names in enumerate(selected)
                                if names
                            ]
                            if name == "__null__"
                            else [
                                index
                                for index, names in enumerate(selected)
                                if name in names
                            ]
                        ),
                        modality_index,
                    ]
                    .detach()
                    .mean()
                    .item()
                )
                if (
                    auxiliary_indices
                    if name == "__null__"
                    else any(name in names for names in selected)
                )
                else None
            )
            for modality_index, name in enumerate(
                model.modality_weight_order
            )
        }
    return {
        "counts": counts,
        "global": global_values,
        "by_stage": stage_values,
    }
def _metric_update_one(
    collection: dict[str, SegmentationMetrics],
    key: str,
    probability: Tensor,
    target: Tensor,
) -> None:
    collection.setdefault(key, SegmentationMetrics()).update(probability, target)
@torch.no_grad()
def evaluate_model(
    model: OAAuxSegModel,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    config: RuntimeConfig,
    progress: TrainingProgress | None = None,
    progress_label: str | None = None,
) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    was_training = model.training
    model.eval()
    overall = SegmentationMetrics()
    by_source: dict[str, SegmentationMetrics] = {}
    by_available: dict[str, SegmentationMetrics] = {}
    by_active: dict[str, SegmentationMetrics] = {}
    total_loss = 0.0
    total_samples = 0
    weight_sum = torch.zeros(
        len(model.modality_weight_order), dtype=torch.float64
    )
    conditional_weight_sum: Counter[str] = Counter()
    conditional_weight_count: Counter[str] = Counter()
    stage_conditional_sum: dict[int, Counter[str]] = {
        stride: Counter() for stride in model.modality_weight_map_strides
    }
    stage_conditional_count: dict[int, Counter[str]] = {
        stride: Counter() for stride in model.modality_weight_map_strides
    }
    source_conditional_sum: dict[str, Counter[str]] = defaultdict(Counter)
    source_conditional_count: dict[str, Counter[str]] = defaultdict(Counter)
    if progress is not None and progress_label is not None:
        progress.start_evaluation(
            label=progress_label, total_batches=len(loader)
        )
    try:
        for batch_index, collated in enumerate(loader, start=1):
            prepared = prepare_collated_batch(collated).to(
                device, non_blocking=device.type == "cuda"
            )
            model_batch, available, active = prepare_policy_batch(
                prepared.model,
                variant=model.variant,
                subset_sampler=None,
            )
            with _autocast(config, device):
                output = model(model_batch)
                loss, _ = bce_dice_loss(output.mask_logits, prepared.mask)
            batch_size = prepared.model.batch_size
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            probability = output.mask_probability.float()
            target = prepared.mask
            overall.update(probability, target)
            weight_sum += (
                output.modality_weights.detach().double().sum(dim=0).cpu()
            )
            covered_available: list[list[str]] = [
                [] for _ in range(batch_size)
            ]
            for modality, packed in model_batch.auxiliaries.items():
                local_valid = (
                    packed.pixel_valid.to(torch.bool)
                    & packed.channel_valid.to(torch.bool)[:, :, None, None]
                ).flatten(1).any(dim=1)
                for local_index, sample_index in enumerate(
                    packed.sample_indices.tolist()
                ):
                    if bool(local_valid[local_index].item()):
                        covered_available[int(sample_index)].append(modality)
            stage_summaries: list[Tensor] = []
            for weight_map in output.modality_weight_maps:
                coverage = torch.cat(
                    [
                        resample_validity(
                            validity_from_channels(
                                pixel_valid.unsqueeze(0),
                                channel_valid.unsqueeze(0),
                            ),
                            weight_map.shape[-2:],
                        ).coverage
                        for pixel_valid, channel_valid in zip(
                            model_batch.optical_pixel_valid,
                            model_batch.optical_channel_valid,
                            strict=True,
                        )
                    ],
                    dim=0,
                ).float()
                stage_summaries.append(
                    (
                        weight_map.float() * coverage
                    ).sum(dim=(2, 3))
                    / coverage.sum(dim=(2, 3)).clamp_min(1e-6)
                )
            for index in range(batch_size):
                covered_names = tuple(covered_available[index])
                source = str(prepared.metadata[index]["source"])
                if covered_names:
                    null_index = len(model.modality_order)
                    null_value = float(
                        output.modality_weights[index, null_index]
                        .detach()
                        .item()
                    )
                    conditional_weight_sum["__null__"] += null_value
                    conditional_weight_count["__null__"] += 1
                    source_conditional_sum[source]["__null__"] += null_value
                    source_conditional_count[source]["__null__"] += 1
                    for stage_index, stride in enumerate(
                        model.modality_weight_map_strides
                    ):
                        stage_null = float(
                            stage_summaries[stage_index][index, null_index]
                            .detach()
                            .item()
                        )
                        stage_conditional_sum[stride][
                            "__null__"
                        ] += stage_null
                        stage_conditional_count[stride]["__null__"] += 1
                for modality in covered_names:
                    modality_index = model.modality_index[modality]
                    value = float(
                        output.modality_weights[index, modality_index]
                        .detach()
                        .item()
                    )
                    conditional_weight_sum[modality] += value
                    conditional_weight_count[modality] += 1
                    source_conditional_sum[source][modality] += value
                    source_conditional_count[source][modality] += 1
                    for stage_index, stride in enumerate(
                        model.modality_weight_map_strides
                    ):
                        stage_value = float(
                            stage_summaries[stage_index][
                                index, modality_index
                            ]
                            .detach()
                            .item()
                        )
                        stage_conditional_sum[stride][modality] += stage_value
                        stage_conditional_count[stride][modality] += 1
                _metric_update_one(
                    by_source,
                    source,
                    probability[index : index + 1],
                    target[index : index + 1],
                )
                _metric_update_one(
                    by_available,
                    _active_label(available[index]),
                    probability[index : index + 1],
                    target[index : index + 1],
                )
                _metric_update_one(
                    by_active,
                    _active_label(active[index]),
                    probability[index : index + 1],
                    target[index : index + 1],
                )
            if progress is not None and progress_label is not None:
                progress.update_evaluation(
                    batch=batch_index,
                    running_loss=total_loss / max(total_samples, 1),
                    metrics=overall.compute(),
                )
    except BaseException:
        if progress is not None:
            progress.abort_evaluation()
        raise
    finally:
        if was_training:
            model.train()
    duration_seconds = time.perf_counter() - evaluation_started
    result = {
        "loss": total_loss / max(total_samples, 1),
        "duration_seconds": duration_seconds,
        "overall": overall.compute(),
        "by_source": {
            key: value.compute() for key, value in sorted(by_source.items())
        },
        "by_available_modality_signature": {
            key: value.compute() for key, value in sorted(by_available.items())
        },
        "by_active_subset": {
            key: value.compute() for key, value in sorted(by_active.items())
        },
        "mean_modality_weights": {
            name: float(weight_sum[index].item() / max(total_samples, 1))
            for index, name in enumerate(model.modality_weight_order)
        },
        "conditional_mean_modality_weights": {
            name: (
                conditional_weight_sum[name]
                / conditional_weight_count[name]
                if conditional_weight_count[name]
                else None
            )
            for name in model.modality_weight_order
        },
        "conditional_weight_counts": {
            name: conditional_weight_count[name]
            for name in model.modality_weight_order
        },
        "conditional_stage_modality_weights": {
            f"stride{stride}": {
                name: (
                    stage_conditional_sum[stride][name]
                    / stage_conditional_count[stride][name]
                    if stage_conditional_count[stride][name]
                    else None
                )
                for name in model.modality_weight_order
            }
            for stride in model.modality_weight_map_strides
        },
        "conditional_modality_weights_by_source": {
            source: {
                name: (
                    source_conditional_sum[source][name]
                    / source_conditional_count[source][name]
                    if source_conditional_count[source][name]
                    else None
                )
                for name in model.modality_weight_order
            }
            for source in sorted(source_conditional_sum)
        },
    }
    if progress is not None and progress_label is not None:
        progress.finish_evaluation(
            label=progress_label,
            result=result,
            duration_seconds=duration_seconds,
        )
    return result
def run_evaluation(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    split: str,
    output_path: Path,
) -> dict[str, Any]:
    benchmark_root, _, _, device = resolve_runtime(config, repo_root)
    benchmark_contract = benchmark_contract_from_root(benchmark_root)
    benchmark_hash = str(benchmark_contract["index_sha256"])
    payload = read_checkpoint(
        checkpoint_path,
        expected_benchmark_contract=benchmark_contract,
    )
    model = model_from_checkpoint(payload, device=device)
    _validate_inference_config(payload, model, config)
    _validate_benchmark_registry(model, benchmark_root)
    loader = make_dataloader(
        benchmark_root,
        split=split,
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    report = {
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": int(payload["step"]),
        "benchmark_index_sha256": benchmark_hash,
        "benchmark_contract": dict(benchmark_contract),
        "backbone_sha256": model.backbone_sha256,
        "metrics": evaluate_model(model, loader, device=device, config=config),
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"评价输出已存在，拒绝覆盖：{output_path}")
    atomic_write_json(output_path, report)
    return report

__all__ = ["evaluate_model", "run_evaluation"]
