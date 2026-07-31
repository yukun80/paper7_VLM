"""Phase 2 OA-AuxSeg 统一训练、评价、smoke、过拟合与推理引擎。"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from scripts.phase1_benchmark_build.benchmark_common import (
    BenchmarkDataset,
    atomic_write_json,
    atomic_write_jsonl,
    collate_benchmark_samples,
    read_jsonl,
    sha256_file,
)

from .checkpoint import (
    maximum_parameter_change,
    model_from_checkpoint,
    module_parameter_snapshot,
    optimizer_state_to_device,
    read_checkpoint,
    restore_training_state,
    save_training_checkpoint,
)
from .contracts import (
    CONFIG_SCHEMA_VERSION,
    INFERENCE_SCHEMA_VERSION,
    OAAuxSegBatch,
    OAAuxSegConfig,
    RuntimeConfig,
    SUPPORTED_BACKBONE,
    TRAINING_REPORT_SCHEMA_VERSION,
    VARIANTS,
)
from .data import (
    AuxiliarySubsetSampler,
    PreparedBatch,
    StatefulTrainingBatcher,
    available_auxiliaries_by_sample,
    benchmark_contract_from_root,
    filter_auxiliaries,
    make_dataloader,
    prepare_collated_batch,
    registry_from_benchmark,
)
from .losses import bce_dice_loss
from .metrics import SegmentationMetrics
from .model import OAAuxSegModel
from .progress import TrainingProgress, format_duration
from .validity import resample_validity, validity_from_channels


def load_runtime_config(path: Path) -> RuntimeConfig:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("运行配置顶层必须是 JSON object")
    return RuntimeConfig.from_dict(value)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_training_config(
    config: RuntimeConfig, *, capacity_overfit: bool
) -> RuntimeConfig:
    if not capacity_overfit:
        return config
    return config.with_overrides(
        weight_decay=0.0,
        eval_interval=100,
        checkpoint_interval=100,
        min_lr_ratio=0.10,
        grad_clip=5.0,
        modality_dropout=0.0,
        train_sampler="uniform",
        optical_stochastic_depth=0.0,
        auxiliary_drop_path=0.0,
        decoder_dropout=0.0,
        use_bf16=False,
        gradient_checkpointing=False,
    )


def resolve_runtime(
    config: RuntimeConfig, repo_root: Path
) -> tuple[Path, Path, Path, torch.device]:
    benchmark_root = config.resolve_path(config.benchmark_root, repo_root)
    output_dir = config.resolve_path(config.output_dir, repo_root)
    backbone_weights = config.resolve_path(config.backbone_weights, repo_root)
    if not (benchmark_root / "index.jsonl").is_file():
        raise FileNotFoundError(f"Benchmark index 不存在：{benchmark_root / 'index.jsonl'}")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前 torch.cuda.is_available() 为 False")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return benchmark_root, output_dir, backbone_weights, device


def require_local_backbone(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            "Phase 2 v6 真实训练要求本地 torchvision ConvNeXt-Small 官方 "
            f"state_dict，当前不存在：{path}"
        )


def make_optimizer(
    model: OAAuxSegModel, config: RuntimeConfig
) -> torch.optim.AdamW:
    backbone_modules = (
        model.backbone,
        model.optical_rgb_stem,
        model.optical_stem_norm,
    )
    backbone_parameters = [
        parameter
        for module in backbone_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    no_decay_ids: set[int] = set()
    normalization_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
    )
    for module_name, module in model.named_modules():
        if isinstance(module, normalization_types):
            no_decay_ids.update(
                id(parameter)
                for parameter in module.parameters(recurse=False)
                if parameter.requires_grad
            )
        for parameter_name, parameter in module.named_parameters(
            recurse=False
        ):
            qualified = (
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )
            if parameter_name == "bias" or "layer_scale" in qualified:
                no_decay_ids.add(id(parameter))
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not new_parameters:
        raise RuntimeError("optimizer 参数分组为空")
    groups = []
    for family, parameters, learning_rate in (
        ("backbone", backbone_parameters, config.backbone_lr),
        ("new", new_parameters, config.new_lr),
    ):
        for decay_label, use_decay in (
            ("decay", True),
            ("no_decay", False),
        ):
            selected = [
                parameter
                for parameter in parameters
                if (id(parameter) not in no_decay_ids) == use_decay
            ]
            if selected:
                groups.append(
                    {
                        "params": selected,
                        "lr": learning_rate,
                        "weight_decay": (
                            config.weight_decay if use_decay else 0.0
                        ),
                        "group_name": f"{family}_{decay_label}",
                    }
                )
    return torch.optim.AdamW(
        groups,
        weight_decay=0.0,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    values: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name.startswith("backbone_"):
            values["backbone"] = float(group["lr"])
        elif name.startswith("new_"):
            values["new"] = float(group["lr"])
    if set(values) != {"backbone", "new"}:
        raise RuntimeError("optimizer 缺少 backbone/new LR 参数组")
    return values["backbone"], values["new"]


def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    # bf16 不需要动态 loss scaling；仍保存 AMP state 以固定 checkpoint 合同。
    return torch.amp.GradScaler("cuda", enabled=False)


def _autocast(config: RuntimeConfig, device: torch.device):
    if device.type == "cuda" and config.use_bf16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("配置要求 bf16，但当前 GPU 不支持")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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


def prepare_policy_batch(
    batch: OAAuxSegBatch,
    *,
    variant: str,
    subset_sampler: AuxiliarySubsetSampler | None,
) -> tuple[OAAuxSegBatch, list[tuple[str, ...]], list[tuple[str, ...]]]:
    available = available_auxiliaries_by_sample(batch)
    if variant == "optical_only":
        active = [tuple() for _ in available]
    elif variant == "proposed_dropout" and subset_sampler is not None:
        active = subset_sampler.sample(available)
    else:
        active = available
    return filter_auxiliaries(batch, active), available, active


def _metric_update_one(
    collection: dict[str, SegmentationMetrics],
    key: str,
    probability: Tensor,
    target: Tensor,
) -> None:
    collection.setdefault(key, SegmentationMetrics()).update(probability, target)


def _validate_inference_config(
    payload: Mapping[str, Any],
    model: OAAuxSegModel,
    config: RuntimeConfig,
) -> None:
    stored = RuntimeConfig.from_dict(
        payload["training_state"]["runtime_config"]
    )
    if model.variant != config.variant or stored.variant != config.variant:
        raise ValueError("运行配置 variant 与 checkpoint 不一致")
    if (
        config.backbone != SUPPORTED_BACKBONE
        or stored.backbone != config.backbone
    ):
        raise ValueError("运行配置 backbone 与 checkpoint 不一致")
    if stored.normalization != config.normalization:
        raise ValueError("运行配置 normalization 与 checkpoint 不一致")
    if (
        model.config.region_threshold != config.region_threshold
        or model.config.min_region_area != config.min_region_area
    ):
        raise ValueError("运行配置区域提取合同与 checkpoint 不一致")


def _validate_benchmark_registry(
    model: OAAuxSegModel,
    benchmark_root: Path,
) -> None:
    if model.registry != registry_from_benchmark(benchmark_root):
        raise ValueError("checkpoint registry 与当前 Benchmark index 不一致")


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


def _gradient_modules(model: OAAuxSegModel) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {
        "extra_band_optical_stems": model.optical_stems,
        "direct_concat_extra_stems": model.direct_stems,
    }
    modules.update(
        {
            f"auxiliary_adapter_{name}": module
            for name, module in model.auxiliary_adapters.items()
        }
    )
    for stride, auxiliary_stage, quality_selector, cmnext_selector, fusion in zip(
        (4, 8, 16, 32),
        model.auxiliary_stages,
        model.quality_selectors,
        model.cmnext_selectors,
        model.progressive_fusions,
        strict=True,
    ):
        modules[f"mspa_stride{stride}"] = auxiliary_stage
        modules[f"quality_selector_stride{stride}"] = quality_selector
        modules[f"cmnext_selector_stride{stride}"] = cmnext_selector
        modules[f"frm_stride{stride}"] = fusion.rectification
        modules[f"ffm_stride{stride}"] = fusion.fusion
    return modules


def _gradient_norms(
    modules: Mapping[str, nn.Module],
) -> dict[str, float]:
    """在设备端聚合后一次回传，避免逐模块/逐参数 CUDA 同步。"""

    names: list[str] = []
    totals: list[Tensor] = []
    fallback_device = next(
        (
            parameter.device
            for module in modules.values()
            for parameter in module.parameters()
        ),
        torch.device("cpu"),
    )
    for name, module in modules.items():
        total: Tensor | None = None
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            squared = parameter.grad.detach().float().square().sum()
            total = squared if total is None else total + squared
        if total is None:
            total = torch.zeros((), device=fallback_device)
        names.append(name)
        totals.append(total)
    if not totals:
        return {}
    values = torch.stack(totals).sqrt().detach().cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(names, values, strict=True)
    }


def _required_auxiliary_modules(
    model: OAAuxSegModel,
) -> tuple[str, ...]:
    return (
        ("extra_band_optical_stems",)
        + tuple(
            f"auxiliary_adapter_{name}" for name in model.modality_order
        )
        + tuple(
            f"{family}_stride{stride}"
            for stride in (4, 8, 16, 32)
            for family in ("mspa", "quality_selector", "frm", "ffm")
        )
    )


def _capacity_acceptance(
    *,
    loss_history: Sequence[float],
    metrics: Mapping[str, Any],
    gradient_max: Mapping[str, float],
    parameter_updates: Mapping[str, float],
    required_modules: Sequence[str],
) -> dict[str, bool]:
    window = min(10, len(loss_history))
    initial = sum(loss_history[:window]) / max(window, 1)
    final = sum(loss_history[-window:]) / max(window, 1)
    overall = metrics["overall"]
    return {
        "loss_drop_at_least_90_percent": (
            1.0 - final / max(initial, 1e-12) >= 0.90
        ),
        "micro_dice_at_least_0_95": overall["dice"] >= 0.95,
        "positive_dice_at_least_0_90": (
            overall["positive_only_dice"] >= 0.90
        ),
        "empty_mask_fpr_zero": (
            overall["no_target_false_positive_rate"] == 0
        ),
        "empty_mean_probability_at_most_0_01": (
            overall["empty_mean_foreground_probability"] <= 0.01
        ),
        "all_auxiliary_gradients_nonzero": all(
            gradient_max.get(name, 0.0) > 0 for name in required_modules
        ),
        "all_auxiliary_parameters_updated": all(
            parameter_updates.get(name, 0.0) > 0
            for name in required_modules
        ),
    }


def _validation_is_better(
    candidate: Mapping[str, Any],
    best: Mapping[str, Any] | None,
) -> bool:
    if best is None:
        return True
    candidate_key = (
        float(candidate["dice"]),
        -float(candidate["loss"]),
        -float(candidate["no_target_false_positive_rate"]),
    )
    best_key = (
        float(best["dice"]),
        -float(best["loss"]),
        -float(best["no_target_false_positive_rate"]),
    )
    return candidate_key > best_key


def _finalization_candidate(
    row: Mapping[str, Any],
    *,
    location: str,
) -> dict[str, float | int]:
    validation = row.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"{location}.validation 必须是对象")
    overall = validation.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError(f"{location}.validation.overall 必须是对象")
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{location}.step 必须是非负整数")
    try:
        return {
            "step": step,
            "dice": float(overall["dice"]),
            "loss": float(validation["loss"]),
            "no_target_false_positive_rate": float(
                overall["no_target_false_positive_rate"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{location} 缺少合法的 best selection 指标") from error


def _same_selection(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    required = (
        "step",
        "dice",
        "loss",
        "no_target_false_positive_rate",
    )
    try:
        if int(first["step"]) != int(second["step"]):
            return False
        return all(
            math.isclose(
                float(first[name]),
                float(second[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in required[1:]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _read_finalization_log(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"训练日志不存在：{path}")
    rows = [dict(row) for row in read_jsonl(path)]
    if not rows:
        raise ValueError("训练日志不能为空")
    previous_step = -1
    for index, row in enumerate(rows):
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"train_log[{index}].step 必须是非负整数")
        if step < previous_step:
            raise ValueError(
                "训练日志 step 必须单调不下降："
                f"train_log[{index - 1}]={previous_step}, "
                f"train_log[{index}]={step}"
            )
        previous_step = step
    return rows


def _checkpoint_training_diagnostics(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    step = int(payload["step"])
    state = payload["training_state"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint training_state 必须是对象")
    loss_history = state.get("loss_history")
    if not isinstance(loss_history, list) or not loss_history:
        raise ValueError("checkpoint loss_history 必须是非空数组")
    if len(loss_history) != step:
        raise ValueError(
            "checkpoint loss_history 长度与 step 不一致："
            f"{len(loss_history)} != {step}"
        )
    losses = [float(value) for value in loss_history]
    window = min(10, len(losses))
    initial_window = sum(losses[:window]) / window
    final_window = sum(losses[-window:]) / window
    training_seconds = float(state.get("training_step_seconds", 0.0))
    timed_steps = int(state.get("training_steps_timed", 0))
    samples_seen = int(state.get("training_samples_seen", 0))
    clipped_steps = int(state.get("clipped_steps", 0))
    return {
        "step": step,
        "loss_history_length": len(losses),
        "initial_loss_window_mean": initial_window,
        "final_loss_window_mean": final_window,
        "loss_drop_fraction": (
            1.0 - final_window / max(initial_window, 1e-12)
        ),
        "initial_ema_loss": state.get("initial_ema_loss"),
        "final_ema_loss": state.get("ema_loss"),
        "gradient_max_norms": dict(state.get("gradient_max", {})),
        "parameter_max_changes": dict(
            state.get("parameter_change_max", {})
        ),
        "training_step_seconds": training_seconds,
        "training_steps_timed": timed_steps,
        "training_samples_seen": samples_seen,
        "average_steps_per_second": (
            timed_steps / training_seconds if training_seconds > 0 else None
        ),
        "average_samples_per_second": (
            samples_seen / training_seconds if training_seconds > 0 else None
        ),
        "gradient_clipped_steps": clipped_steps,
        "gradient_clipped_fraction": (
            clipped_steps / max(timed_steps, 1)
        ),
        "gradient_clip_scale_mean": (
            float(state.get("clip_scale_sum", 0.0))
            / max(timed_steps, 1)
        ),
        "gradient_clip_scale_min": float(
            state.get("clip_scale_min", 1.0)
        ),
        "training_subset_counts": dict(
            state.get("training_subset_counts", {})
        ),
        "training_subset_reason_counts": dict(
            state.get("training_subset_reason_counts", {})
        ),
        "training_native_auxiliary_samples": int(
            state.get("training_native_auxiliary_samples", 0)
        ),
        "training_active_auxiliary_samples": int(
            state.get("training_active_auxiliary_samples", 0)
        ),
        "validation_count": len(state.get("validation_trajectory", [])),
    }


def _validate_finalization_evidence(
    *,
    config: RuntimeConfig,
    rows: Sequence[Mapping[str, Any]],
    selected_payload: Mapping[str, Any],
    last_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("训练日志不能为空")
    selected_step = int(selected_payload["step"])
    last_checkpoint_step = int(last_payload["step"])
    last_logged_step = int(rows[-1]["step"])
    if not (
        0 <= selected_step <= last_checkpoint_step <= last_logged_step
        <= config.max_steps
    ):
        raise ValueError(
            "训练步数关系不合法："
            f"selected={selected_step}, last_checkpoint={last_checkpoint_step}, "
            f"last_logged={last_logged_step}, max={config.max_steps}"
        )

    selected_state = selected_payload["training_state"]
    last_state = last_payload["training_state"]
    selected_config = RuntimeConfig.from_dict(
        selected_state["runtime_config"]
    )
    last_config = RuntimeConfig.from_dict(last_state["runtime_config"])
    if selected_config.to_dict() != config.to_dict():
        raise ValueError("final checkpoint runtime config 与当前配置不一致")
    if last_config.to_dict() != config.to_dict():
        raise ValueError("last checkpoint runtime config 与当前配置不一致")
    if selected_payload["model_contract"] != last_payload["model_contract"]:
        raise ValueError("best/last checkpoint model contract 不一致")

    selected_best = selected_state.get("best_selection")
    last_best = last_state.get("best_selection")
    if not isinstance(selected_best, Mapping):
        raise ValueError("final checkpoint 缺少 best_selection")
    if not isinstance(last_best, Mapping):
        raise ValueError("last checkpoint 缺少 best_selection")
    if int(selected_best.get("step", -1)) != selected_step:
        raise ValueError("final checkpoint step 与 best_selection.step 不一致")
    if not _same_selection(selected_best, last_best):
        raise ValueError("best/last checkpoint 的 best_selection 不一致")

    computed_best: dict[str, float | int] | None = None
    selected_rows: list[Mapping[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "validation" not in row:
            continue
        candidate = _finalization_candidate(
            row, location=f"train_log[{index}]"
        )
        expected_is_new_best = _validation_is_better(
            candidate, computed_best
        )
        if row.get("is_new_best") is not expected_is_new_best:
            raise ValueError(
                f"train_log[{index}].is_new_best 与既有选择规则不一致"
            )
        if expected_is_new_best:
            computed_best = candidate
        if int(candidate["step"]) == selected_step:
            selected_rows.append(row)
        validation_rows.append(dict(row))
    if computed_best is None:
        raise ValueError("训练日志中没有 validation 记录")
    if not _same_selection(computed_best, selected_best):
        raise ValueError("训练日志计算得到的 best selection 与 checkpoint 不一致")
    if len(selected_rows) != 1:
        raise ValueError(
            "训练日志中 final checkpoint step 必须且只能有一条 validation："
            f"step={selected_step}, count={len(selected_rows)}"
        )
    selected_row = selected_rows[0]
    if selected_row.get("is_new_best") is not True:
        raise ValueError("final checkpoint 对应日志没有 is_new_best=true")
    log_selection = _finalization_candidate(
        selected_row, location=f"train_log@step{selected_step}"
    )
    if not _same_selection(log_selection, selected_best):
        raise ValueError("final checkpoint 与同一步日志指标不一致")

    selected_trajectory = selected_state.get("validation_trajectory")
    if not isinstance(selected_trajectory, list):
        raise ValueError("final checkpoint validation_trajectory 必须是数组")
    trajectory_matches = [
        item
        for item in selected_trajectory
        if isinstance(item, Mapping)
        and int(item.get("step", -1)) == selected_step
    ]
    if len(trajectory_matches) != 1:
        raise ValueError("final checkpoint trajectory 缺少唯一的 best step")
    trajectory_item = trajectory_matches[0]
    if trajectory_item.get("is_new_best") is not True:
        raise ValueError("final checkpoint trajectory 的 best step 标记错误")
    trajectory_candidate = _finalization_candidate(
        trajectory_item,
        location=f"checkpoint.validation_trajectory@step{selected_step}",
    )
    if not _same_selection(trajectory_candidate, selected_best):
        raise ValueError("checkpoint trajectory 与 best_selection 指标不一致")
    if trajectory_item.get("validation") != selected_row.get("validation"):
        raise ValueError("checkpoint trajectory 与 train log validation 不一致")

    return {
        "selected_checkpoint_step": selected_step,
        "last_checkpoint_step": last_checkpoint_step,
        "last_logged_step": last_logged_step,
        "best_selection": dict(selected_best),
        "selection_validation_snapshot": dict(selected_row["validation"]),
        "validation_trajectory": validation_rows,
        "validation_count": len(validation_rows),
        "logged_but_not_checkpointed": {
            "after_step": last_checkpoint_step,
            "through_step": last_logged_step,
            "step_count": last_logged_step - last_checkpoint_step,
            "included_in_final_weights": False,
        },
    }


def _metric_replay_delta(
    stored: Any,
    fresh: Any,
    *,
    location: str = "$",
) -> Any:
    if isinstance(stored, Mapping) and isinstance(fresh, Mapping):
        if set(stored) != set(fresh):
            raise ValueError(
                f"{location} fresh/stored metric keys 不一致："
                f"stored={sorted(stored)} fresh={sorted(fresh)}"
            )
        return {
            str(key): _metric_replay_delta(
                stored[key],
                fresh[key],
                location=f"{location}.{key}",
            )
            for key in stored
        }
    if (
        isinstance(stored, (int, float))
        and not isinstance(stored, bool)
        and isinstance(fresh, (int, float))
        and not isinstance(fresh, bool)
    ):
        return float(fresh) - float(stored)
    if stored != fresh:
        raise ValueError(
            f"{location} fresh/stored metric structure 不一致："
            f"{stored!r} != {fresh!r}"
        )
    return None


def _acceptance_collated(
    benchmark_root: Path,
    *,
    normalization: str,
    batch_size: int,
) -> dict[str, Any]:
    dataset = BenchmarkDataset(
        benchmark_root,
        split="train",
        auxiliary_policy="all",
        normalization=normalization,
    )
    if len(dataset) < batch_size:
        raise ValueError(
            f"acceptance batch 需要 {batch_size} 条 train，实际仅 {len(dataset)}"
        )
    registry = registry_from_benchmark(benchmark_root)

    def row_modalities(row: Mapping[str, Any]) -> tuple[str, ...]:
        names = set(str(name) for name in row["auxiliaries"])
        return tuple(
            name for name in registry.auxiliary_order if name in names
        )

    def row_features(row: Mapping[str, Any]) -> set[tuple[str, ...]]:
        optical = tuple(
            str(name) for name in row["optical"]["channel_names"]
        )
        modalities = row_modalities(row)
        features = {
            ("source", str(row["source"])),
            ("optical", *optical),
        }
        features.update(("auxiliary", name) for name in modalities)
        if not modalities:
            features.add(("auxiliary_combination", "none"))
        elif len(modalities) >= 2:
            features.add(("auxiliary_combination", *modalities))
        return features

    features_by_index = [
        row_features(row) for row in dataset.rows
    ]
    required_features = set().union(*features_by_index)
    chosen: list[int] = []
    uncovered = set(required_features)
    while uncovered:
        candidates = [
            (
                len(features & uncovered),
                -index,
                index,
            )
            for index, features in enumerate(features_by_index)
            if index not in chosen
        ]
        gain, _, best_index = max(candidates, default=(0, 0, -1))
        if gain <= 0 or best_index < 0:
            raise RuntimeError("acceptance batch 无法覆盖当前 Benchmark 合同")
        chosen.append(best_index)
        uncovered -= features_by_index[best_index]
        if len(chosen) > batch_size:
            raise RuntimeError(
                "acceptance batch_size 不足以覆盖当前 Benchmark 合同"
            )
    for index in range(len(dataset)):
        if len(chosen) >= batch_size:
            break
        if index not in chosen:
            chosen.append(index)
    if len(chosen) != batch_size:
        raise RuntimeError("无法构建指定大小的 acceptance batch")
    collated = collate_benchmark_samples([dataset[index] for index in chosen])
    actual_optical = {
        tuple(str(name) for name in names)
        for names in collated["optical_channel_names"]
    }
    if actual_optical != set(registry.optical_signatures):
        raise RuntimeError("acceptance batch 未覆盖全部光学通道签名")
    auxiliary_names = set(collated["auxiliaries"])
    expected_auxiliaries = set(registry.auxiliary_order)
    if auxiliary_names != expected_auxiliaries:
        raise RuntimeError(
            "acceptance batch 未覆盖全部已注册辅助模态："
            f"actual={sorted(auxiliary_names)} "
            f"expected={sorted(expected_auxiliaries)}"
        )
    present_names: list[list[str]] = [[] for _ in chosen]
    for name, packed in collated["auxiliaries"].items():
        for sample_index in packed["sample_indices"].tolist():
            present_names[int(sample_index)].append(str(name))
    present_combinations = {
        tuple(
            name
            for name in registry.auxiliary_order
            if name in set(names)
        )
        for names in present_names
    }
    expected_multi = {
        row_modalities(row)
        for row in dataset.rows
        if len(row_modalities(row)) >= 2
    }
    if tuple() not in present_combinations or not expected_multi.issubset(
        present_combinations
    ):
        raise RuntimeError("acceptance batch 未同时覆盖无辅助与多辅助样本")
    return collated


@torch.no_grad()
def checkpoint_reload_difference(
    *,
    checkpoint_path: Path,
    model: OAAuxSegModel,
    collated: Mapping[str, Any],
    benchmark_contract: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    prepared = prepare_collated_batch(collated).to(
        device, non_blocking=device.type == "cuda"
    )
    model_batch, _, _ = prepare_policy_batch(
        prepared.model, variant=model.variant, subset_sampler=None
    )
    model.eval()
    before = model(model_batch)
    payload = read_checkpoint(
        checkpoint_path,
        expected_benchmark_contract=benchmark_contract,
    )
    reloaded = model_from_checkpoint(payload, device=device)
    reloaded.eval()
    after = reloaded(model_batch)
    differences = {
        "mask_logits": float(
            (before.mask_logits.float() - after.mask_logits.float())
            .abs()
            .max()
            .item()
        ),
        "mask_probability": float(
            (before.mask_probability.float() - after.mask_probability.float())
            .abs()
            .max()
            .item()
        ),
        "modality_weights": float(
            (before.modality_weights.float() - after.modality_weights.float())
            .abs()
            .max()
            .item()
        ),
    }
    differences["modality_weight_maps"] = max(
        float(
            (first.float() - second.float()).abs().max().item()
        )
        for first, second in zip(
            before.modality_weight_maps,
            after.modality_weight_maps,
            strict=True,
        )
    )
    return differences


def _new_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{path}")
    path.mkdir(parents=True, exist_ok=True)


def _training_state(
    *,
    config: RuntimeConfig,
    loss_history: Sequence[float],
    ema_loss: float | None,
    initial_ema_loss: float | None,
    gradient_max: Mapping[str, float],
    parameter_change_max: Mapping[str, float],
    training_step_seconds: float,
    training_samples_seen: int,
    training_steps_timed: int,
    clipped_steps: int,
    clip_scale_sum: float,
    clip_scale_min: float,
    validation_trajectory: Sequence[Mapping[str, Any]],
    best_selection: Mapping[str, Any] | None,
    training_subset_counts: Mapping[str, int],
    training_subset_reason_counts: Mapping[str, int],
    training_native_auxiliary_samples: int,
    training_active_auxiliary_samples: int,
) -> dict[str, Any]:
    return {
        "runtime_config": config.to_dict(),
        "loss_history": list(loss_history),
        "ema_loss": ema_loss,
        "initial_ema_loss": initial_ema_loss,
        "gradient_max": dict(gradient_max),
        "parameter_change_max": dict(parameter_change_max),
        "training_step_seconds": float(training_step_seconds),
        "training_samples_seen": int(training_samples_seen),
        "training_steps_timed": int(training_steps_timed),
        "clipped_steps": int(clipped_steps),
        "clip_scale_sum": float(clip_scale_sum),
        "clip_scale_min": float(clip_scale_min),
        "validation_trajectory": [
            dict(item) for item in validation_trajectory
        ],
        "best_selection": (
            dict(best_selection) if best_selection is not None else None
        ),
        "training_subset_counts": dict(training_subset_counts),
        "training_subset_reason_counts": dict(
            training_subset_reason_counts
        ),
        "training_native_auxiliary_samples": int(
            training_native_auxiliary_samples
        ),
        "training_active_auxiliary_samples": int(
            training_active_auxiliary_samples
        ),
    }


def run_training(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    capacity_overfit: bool,
    resume_checkpoint: Path | None = None,
    progress: TrainingProgress | None = None,
) -> dict[str, Any]:
    terminal = (
        progress
        if progress is not None
        else TrainingProgress(log_interval=config.log_interval)
    )
    try:
        return _run_training_impl(
            config,
            repo_root=repo_root,
            capacity_overfit=capacity_overfit,
            resume_checkpoint=resume_checkpoint,
            progress=terminal,
        )
    finally:
        terminal.close()


def finalize_training_run(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    termination_reason: str,
    progress: TrainingProgress | None = None,
) -> dict[str, Any]:
    """只读冻结人工终止训练，并用 final best 生成 train/val 工程报告。"""

    terminal = (
        progress
        if progress is not None
        else TrainingProgress(log_interval=config.log_interval)
    )
    try:
        return _finalize_training_run_impl(
            config,
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            termination_reason=termination_reason,
            progress=terminal,
        )
    finally:
        terminal.close()


def _asset_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _finalize_training_run_impl(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    termination_reason: str,
    progress: TrainingProgress,
) -> dict[str, Any]:
    if termination_reason != "project_owner_manual_stop":
        raise ValueError(
            "finalize 仅接受 "
            "termination_reason=project_owner_manual_stop"
        )
    wall_started = time.perf_counter()
    benchmark_root, output_dir, _, device = resolve_runtime(config, repo_root)
    output_dir = output_dir.resolve()
    selected_checkpoint_path = Path(checkpoint_path).resolve()
    expected_best_path = (output_dir / "checkpoint_best.pt").resolve()
    last_checkpoint_path = (output_dir / "checkpoint_last.pt").resolve()
    train_log_path = (output_dir / "train_log.jsonl").resolve()
    training_report_path = (output_dir / "training_report.json").resolve()
    if selected_checkpoint_path != expected_best_path:
        raise ValueError(
            "finalize 只允许冻结配置输出根中的 checkpoint_best.pt："
            f"{selected_checkpoint_path} != {expected_best_path}"
        )
    for label, path in (
        ("checkpoint_best.pt", selected_checkpoint_path),
        ("checkpoint_last.pt", last_checkpoint_path),
        ("train_log.jsonl", train_log_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 不存在：{path}")
    if training_report_path.exists():
        raise FileExistsError(
            f"训练报告已存在，拒绝覆盖：{training_report_path}"
        )

    input_paths = {
        "checkpoint_best": selected_checkpoint_path,
        "checkpoint_last": last_checkpoint_path,
        "train_log": train_log_path,
    }
    input_stats_before = {
        name: _asset_stat(path) for name, path in input_paths.items()
    }
    progress.phase(
        "[finalize] loading checkpoint/log evidence without optimizer state "
        f"output={output_dir}"
    )
    benchmark_contract = benchmark_contract_from_root(benchmark_root)
    rows = _read_finalization_log(train_log_path)
    selected_payload = read_checkpoint(
        selected_checkpoint_path,
        expected_benchmark_contract=benchmark_contract,
    )
    last_payload = read_checkpoint(
        last_checkpoint_path,
        expected_benchmark_contract=benchmark_contract,
    )
    evidence = _validate_finalization_evidence(
        config=config,
        rows=rows,
        selected_payload=selected_payload,
        last_payload=last_payload,
    )
    selected_checkpoint_sha256 = sha256_file(selected_checkpoint_path)
    train_log_sha256 = sha256_file(train_log_path)

    model = model_from_checkpoint(selected_payload, device=device)
    _validate_inference_config(selected_payload, model, config)
    _validate_benchmark_registry(model, benchmark_root)
    train_loader = make_dataloader(
        benchmark_root,
        split="train",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    validation_loader = make_dataloader(
        benchmark_root,
        split="val",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    set_global_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress.phase(
        "[finalize] evaluating final checkpoint on train/val only; "
        "test remains sealed"
    )
    train_metrics = evaluate_model(
        model,
        train_loader,
        device=device,
        config=config,
        progress=progress,
        progress_label="finalize-train",
    )
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        device=device,
        config=config,
        progress=progress,
        progress_label="finalize-val",
    )
    selection_snapshot = evidence["selection_validation_snapshot"]
    validation_replay_delta = _metric_replay_delta(
        selection_snapshot,
        validation_metrics,
        location="$.validation_metrics",
    )

    progress.phase(
        "[finalize] checking strict checkpoint reload output consistency"
    )
    reload_started = time.perf_counter()
    reload_difference = checkpoint_reload_difference(
        checkpoint_path=selected_checkpoint_path,
        model=model,
        collated=_acceptance_collated(
            benchmark_root,
            normalization=config.normalization,
            batch_size=config.batch_size,
        ),
        benchmark_contract=benchmark_contract,
        device=device,
    )
    reload_seconds = time.perf_counter() - reload_started
    input_stats_after = {
        name: _asset_stat(path) for name, path in input_paths.items()
    }
    if input_stats_after != input_stats_before:
        raise RuntimeError(
            "finalize 期间 checkpoint 或 train log 的 size/mtime 发生变化，"
            "拒绝生成报告"
        )

    selected_step = int(evidence["selected_checkpoint_step"])
    last_checkpoint_step = int(evidence["last_checkpoint_step"])
    last_logged_step = int(evidence["last_logged_step"])
    train_sample_count = int(train_metrics["overall"]["sample_count"])
    validation_sample_count = int(
        validation_metrics["overall"]["sample_count"]
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    engineering_checks = {
        "config_checkpoint_benchmark_identity_consistent": True,
        "best_last_log_selection_evidence_consistent": True,
        "train_sample_count_matches_loader": (
            train_sample_count == len(train_loader.dataset)
        ),
        "validation_sample_count_matches_loader": (
            validation_sample_count == len(validation_loader.dataset)
        ),
        "checkpoint_reload_within_1e_6": (
            max(reload_difference.values()) <= 1e-6
        ),
        "input_checkpoints_and_log_unchanged": True,
        "test_split_not_evaluated": True,
    }
    engineering_checks_passed = all(engineering_checks.values())
    selected_diagnostics = _checkpoint_training_diagnostics(
        selected_payload
    )
    last_diagnostics = _checkpoint_training_diagnostics(last_payload)
    report: dict[str, Any] = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "runtime_config_schema_version": CONFIG_SCHEMA_VERSION,
        "command": "finalize",
        "status": "completed",
        "completion_mode": termination_reason,
        "training_declared_complete": True,
        "automatic_early_stopping": False,
        "scheduled_max_steps_reached": (
            last_logged_step >= config.max_steps
        ),
        "stopped_early": last_logged_step < config.max_steps,
        "resume_required": False,
        "configured_max_steps": config.max_steps,
        "maximum_steps": config.max_steps,
        "steps": last_logged_step,
        "last_logged_step": last_logged_step,
        "last_checkpoint_step": last_checkpoint_step,
        "selected_checkpoint_step": selected_step,
        "weights_include_training_through_step": selected_step,
        "logged_but_not_checkpointed": evidence[
            "logged_but_not_checkpointed"
        ],
        "selected_checkpoint_role": "final",
        "checkpoint": str(selected_checkpoint_path),
        "checkpoint_best": str(selected_checkpoint_path),
        "checkpoint_last": str(last_checkpoint_path),
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "selected_checkpoint_size_bytes": input_stats_before[
            "checkpoint_best"
        ]["size_bytes"],
        "train_log": str(train_log_path),
        "train_log_sha256": train_log_sha256,
        "train_log_row_count": len(rows),
        "selection_rule": [
            "maximize_validation_dice",
            "minimize_validation_loss_on_tie",
            "minimize_no_target_false_positive_rate_on_tie",
        ],
        "best_step": selected_step,
        "best_selection": evidence["best_selection"],
        "selection_validation_snapshot": selection_snapshot,
        "validation_trajectory": evidence["validation_trajectory"],
        "validation_replay_delta": validation_replay_delta,
        "benchmark_index_sha256": str(
            benchmark_contract["index_sha256"]
        ),
        "benchmark_contract": dict(benchmark_contract),
        "model_contract": dict(selected_payload["model_contract"]),
        "backbone_sha256": model.backbone_sha256,
        "variant": config.variant,
        "backbone": config.backbone,
        "architecture": model.model_contract()["architecture"],
        "modality_weight_order": list(model.modality_weight_order),
        "region_feature_dim": model.region_feature_dim,
        "training_diagnostics": {
            "sources": {
                "complete_run_tail": "train_log.jsonl",
                "last_recoverable_state": "checkpoint_last.pt",
                "final_weight_state": "checkpoint_best.pt",
            },
            "selected_checkpoint": selected_diagnostics,
            "last_checkpoint": last_diagnostics,
            "run_log": {
                "last_logged_step": last_logged_step,
                "row_count": len(rows),
                "validation_count": int(evidence["validation_count"]),
            },
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "checkpoint_reload_seconds": reload_seconds,
        "evaluation_scope": "train_val_engineering",
        "test_evaluated": False,
        "engineering_checks": engineering_checks,
        "engineering_checks_passed": engineering_checks_passed,
        "acceptance_scope": "engineering_runtime_only",
        "acceptance_enforced": False,
        "acceptance_passed": None,
        "gate_a_evaluated": False,
        "formal_acceptance": False,
        "formal_acceptance_reason": "gate_a_not_evaluated",
        "input_asset_stats_before": input_stats_before,
        "input_asset_stats_after": input_stats_after,
        "evaluation_duration_seconds": {
            "train": train_metrics["duration_seconds"],
            "validation": validation_metrics["duration_seconds"],
        },
        "peak_cuda_memory_bytes": peak_memory,
        "peak_cuda_memory_gib": (
            peak_memory / (1024**3) if peak_memory is not None else None
        ),
        "wall_elapsed_seconds": time.perf_counter() - wall_started,
        "training_report": str(training_report_path),
    }
    atomic_write_json(training_report_path, report)
    if not engineering_checks_passed:
        failed = [
            name
            for name, passed in engineering_checks.items()
            if not passed
        ]
        raise RuntimeError(
            "人工定版报告已写入，但工程检查未通过：" + ", ".join(failed)
        )
    progress.phase(
        "[finalize] completed "
        f"selected_step={selected_step} last_logged_step={last_logged_step} "
        f"report={training_report_path}"
    )
    return report


def _run_training_impl(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    capacity_overfit: bool,
    resume_checkpoint: Path | None,
    progress: TrainingProgress,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    config = effective_training_config(
        config, capacity_overfit=capacity_overfit
    )
    if config.num_workers != 0:
        raise ValueError(
            "v6 可恢复定长训练 batcher 要求 num_workers=0；"
            "评价与推理仍可使用多 worker"
        )
    benchmark_root, output_dir, backbone_weights, device = resolve_runtime(
        config, repo_root
    )
    command_label = "overfit" if capacity_overfit else "train"
    gpu_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    progress.phase(
        "[setup] "
        f"command={command_label} variant={config.variant} "
        f"device={device.type} output={output_dir} loading runtime"
    )
    benchmark_contract = benchmark_contract_from_root(benchmark_root)
    benchmark_hash = str(benchmark_contract["index_sha256"])
    set_global_seed(config.seed)
    subset_sampler = (
        AuxiliarySubsetSampler(
            config.seed + 2,
            config.modality_dropout,
        )
        if config.variant == "proposed_dropout" and not capacity_overfit
        else None
    )
    if resume_checkpoint is None:
        require_local_backbone(backbone_weights)
        _new_output_directory(output_dir)
        registry = registry_from_benchmark(benchmark_root)
        model = OAAuxSegModel(
            OAAuxSegConfig(
                variant=config.variant,
                optical_stochastic_depth=config.optical_stochastic_depth,
                auxiliary_drop_path=config.auxiliary_drop_path,
                decoder_dropout=config.decoder_dropout,
                region_threshold=config.region_threshold,
                min_region_area=config.min_region_area,
            ),
            registry,
            backbone_weights=backbone_weights,
            gradient_checkpointing=config.gradient_checkpointing,
        ).to(device)
        start_step = 0
        loss_history: list[float] = []
        ema_loss: float | None = None
        initial_ema_loss: float | None = None
        gradient_max: dict[str, float] = defaultdict(float)
        parameter_change_max: dict[str, float] = defaultdict(float)
        training_step_seconds = 0.0
        training_samples_seen = 0
        training_steps_timed = 0
        clipped_steps = 0
        clip_scale_sum = 0.0
        clip_scale_min = 1.0
        validation_trajectory: list[dict[str, Any]] = []
        best_selection: dict[str, Any] | None = None
        training_subset_counts: Counter[str] = Counter()
        training_subset_reason_counts: Counter[str] = Counter()
        training_native_auxiliary_samples = 0
        training_active_auxiliary_samples = 0
        resume_payload = None
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_payload = read_checkpoint(
            Path(resume_checkpoint),
            expected_benchmark_contract=benchmark_contract,
        )
        model = model_from_checkpoint(resume_payload, device=device)
        model.set_gradient_checkpointing(config.gradient_checkpointing)
        if model.variant != config.variant:
            raise ValueError("resume checkpoint variant 与配置不一致")
        if model.registry != registry_from_benchmark(benchmark_root):
            raise ValueError("resume checkpoint registry 与 Benchmark 不一致")
        start_step = int(resume_payload["step"])
        state = resume_payload["training_state"]
        old_config = dict(state["runtime_config"])
        for name in (
            "variant",
            "backbone",
            "seed",
            "batch_size",
            "normalization",
            "max_steps",
            "backbone_lr",
            "new_lr",
            "weight_decay",
            "warmup_ratio",
            "min_lr_ratio",
            "modality_dropout",
            "train_sampler",
            "optical_stochastic_depth",
            "auxiliary_drop_path",
            "decoder_dropout",
            "use_bf16",
            "grad_clip",
            "gradient_checkpointing",
        ):
            if old_config[name] != getattr(config, name):
                raise ValueError(f"resume 配置字段 {name} 不一致")
        loss_history = [float(value) for value in state["loss_history"]]
        ema_loss = (
            None if state["ema_loss"] is None else float(state["ema_loss"])
        )
        initial_ema_loss = (
            None
            if state["initial_ema_loss"] is None
            else float(state["initial_ema_loss"])
        )
        gradient_max = defaultdict(
            float,
            {str(key): float(value) for key, value in state["gradient_max"].items()},
        )
        parameter_change_max = defaultdict(
            float,
            {
                str(key): float(value)
                for key, value in state.get(
                    "parameter_change_max", {}
                ).items()
            },
        )
        training_step_seconds = float(
            state.get("training_step_seconds", 0.0)
        )
        training_samples_seen = int(state.get("training_samples_seen", 0))
        training_steps_timed = int(state.get("training_steps_timed", 0))
        clipped_steps = int(state.get("clipped_steps", 0))
        clip_scale_sum = float(state.get("clip_scale_sum", 0.0))
        clip_scale_min = float(state.get("clip_scale_min", 1.0))
        validation_trajectory = [
            dict(item)
            for item in state.get("validation_trajectory", [])
        ]
        stored_best = state.get("best_selection")
        best_selection = (
            None if stored_best is None else dict(stored_best)
        )
        training_subset_counts = Counter(
            {
                str(key): int(value)
                for key, value in state.get(
                    "training_subset_counts", {}
                ).items()
            }
        )
        training_subset_reason_counts = Counter(
            {
                str(key): int(value)
                for key, value in state.get(
                    "training_subset_reason_counts", {}
                ).items()
            }
        )
        training_native_auxiliary_samples = int(
            state.get("training_native_auxiliary_samples", 0)
        )
        training_active_auxiliary_samples = int(
            state.get("training_active_auxiliary_samples", 0)
        )
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(
        optimizer,
        total_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        min_lr_ratio=config.min_lr_ratio,
    )
    scaler = make_scaler(device)
    train_batcher = StatefulTrainingBatcher(
        benchmark_root,
        batch_size=config.batch_size,
        normalization=config.normalization,
        seed=config.seed + 1,
        policy=config.train_sampler,
    )
    val_loader = make_dataloader(
        benchmark_root,
        split="val",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    train_eval_loader = make_dataloader(
        benchmark_root,
        split="train",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    if resume_payload is not None:
        restore_training_state(
            payload=resume_payload,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        optimizer_state_to_device(optimizer, device)
        if subset_sampler is not None:
            sampler_state = resume_payload["subset_sampler_state"]
            if sampler_state is None:
                raise ValueError("proposed_dropout checkpoint 缺少 sampler state")
            subset_sampler.load_state_dict(sampler_state)
        train_batcher.load_state_dict(
            resume_payload["training_batcher_state"]
        )
    modules = _gradient_modules(model)
    snapshots = {
        name: module_parameter_snapshot(module) for name, module in modules.items()
    }
    train_log_path = output_dir / "train_log.jsonl"
    logs: list[dict[str, Any]] = (
        [dict(item) for item in read_jsonl(train_log_path)]
        if resume_payload is not None and train_log_path.is_file()
        else []
    )
    checkpoint_path = output_dir / "checkpoint_last.pt"
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    if (
        resume_payload is not None
        and best_selection is not None
        and not best_checkpoint_path.is_file()
    ):
        raise FileNotFoundError(
            "resume checkpoint 记录了 best_selection，"
            f"但输出目录缺少 {best_checkpoint_path.name}"
        )
    checkpoint_total_seconds = 0.0
    speed_window: deque[tuple[float, int]] = deque(
        maxlen=max(config.log_interval, 10)
    )
    progress.announce_setup(
        command=command_label,
        variant=config.variant,
        device=device.type,
        gpu_name=gpu_name,
        train_samples=len(train_batcher.dataset),
        validation_samples=len(val_loader.dataset),
        batch_size=config.batch_size,
        total_steps=config.max_steps,
        start_step=start_step,
        eval_interval=config.eval_interval,
        checkpoint_interval=config.checkpoint_interval,
        output_dir=output_dir,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    progress.start_training(
        variant=config.variant,
        total_steps=config.max_steps,
        start_step=start_step,
    )

    def save_current_checkpoint(
        path: Path,
        *,
        step: int,
        training_state: Mapping[str, Any],
        label: str,
    ) -> tuple[float, int]:
        progress.phase(f"[checkpoint] step={step} saving {label} {path}")
        started = time.perf_counter()
        save_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            benchmark_contract=benchmark_contract,
            subset_sampler_state=(
                subset_sampler.state_dict()
                if subset_sampler is not None
                else None
            ),
            training_batcher_state=train_batcher.state_dict(),
            training_state=training_state,
        )
        duration = time.perf_counter() - started
        size = path.stat().st_size
        progress.phase(
            "[checkpoint] "
            f"step={step} saved {label} "
            f"time={format_duration(duration)} "
            f"size={size / (1024**3):.2f} GiB"
        )
        return duration, size

    actual_step = start_step
    early_capacity_acceptance: dict[str, bool] | None = None
    for step in range(start_step + 1, config.max_steps + 1):
        actual_step = step
        step_started = time.perf_counter()
        prepared = prepare_collated_batch(train_batcher.next()).to(
            device, non_blocking=device.type == "cuda"
        )
        model_batch, available, active = prepare_policy_batch(
            prepared.model,
            variant=config.variant,
            subset_sampler=subset_sampler,
        )
        for active_names, available_names in zip(
            active, available, strict=True
        ):
            training_native_auxiliary_samples += int(
                bool(available_names)
            )
            training_active_auxiliary_samples += int(bool(active_names))
            training_subset_counts[
                _subset_category(active_names, available_names)
            ] += 1
        if subset_sampler is not None:
            reasons = subset_sampler.last_reasons
        else:
            reasons = [
                (
                    "native_none"
                    if not available_names
                    else (
                        "active"
                        if active_names
                        else "variant_forced_none"
                    )
                )
                for active_names, available_names in zip(
                    active, available, strict=True
                )
            ]
        training_subset_reason_counts.update(reasons)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            output = model(model_batch)
            loss, components = bce_dice_loss(output.mask_logits, prepared.mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        current_gradient = _gradient_norms(modules)
        for name, value in current_gradient.items():
            gradient_max[name] = max(gradient_max[name], value)
        gradient_total = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        )
        clip_scale = min(
            1.0,
            config.grad_clip / (gradient_total + 1e-6),
        )
        clipped_steps += int(clip_scale < 1.0)
        clip_scale_sum += clip_scale
        clip_scale_min = min(clip_scale_min, clip_scale)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_value = float(loss.detach().item())
        bce_value = float(components["bce"].detach().item())
        dice_loss_value = float(components["dice"].detach().item())
        step_seconds = time.perf_counter() - step_started
        batch_size = prepared.model.batch_size
        training_step_seconds += step_seconds
        training_samples_seen += batch_size
        training_steps_timed += 1
        speed_window.append((step_seconds, batch_size))
        rolling_seconds = sum(item[0] for item in speed_window)
        rolling_steps_per_second = len(speed_window) / max(
            rolling_seconds, 1e-12
        )
        rolling_samples_per_second = sum(
            item[1] for item in speed_window
        ) / max(rolling_seconds, 1e-12)
        estimated_remaining_seconds = (
            config.max_steps - step
        ) / max(rolling_steps_per_second, 1e-12)
        peak_cuda_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        peak_cuda_memory_gib = (
            peak_cuda_memory_bytes / (1024**3)
            if peak_cuda_memory_bytes is not None
            else None
        )
        loss_history.append(loss_value)
        ema_loss = (
            loss_value
            if ema_loss is None
            else 0.95 * ema_loss + 0.05 * loss_value
        )
        if initial_ema_loss is None:
            initial_ema_loss = ema_loss
        progress.update_training(
            step=step,
            loss=loss_value,
            ema_loss=float(ema_loss),
            bce=bce_value,
            dice_loss=dice_loss_value,
            learning_rates=optimizer_learning_rates(optimizer),
            samples_per_second=rolling_samples_per_second,
            estimated_remaining_seconds=estimated_remaining_seconds,
            peak_cuda_memory_gib=peak_cuda_memory_gib,
        )
        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            active_counts = Counter(_active_label(names) for names in active)
            available_counts = Counter(_active_label(names) for names in available)
            conditional_weights = _batch_conditional_weight_diagnostics(
                model=model,
                batch=model_batch,
                output=output,
                active=active,
            )
            logs.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "bce": bce_value,
                    "dice_loss": dice_loss_value,
                    "ema_loss": ema_loss,
                    "lr": [float(group["lr"]) for group in optimizer.param_groups],
                    "step_time_seconds": step_seconds,
                    "rolling_steps_per_second": rolling_steps_per_second,
                    "rolling_samples_per_second": rolling_samples_per_second,
                    "estimated_remaining_seconds": estimated_remaining_seconds,
                    "training_elapsed_seconds": training_step_seconds,
                    "wall_elapsed_seconds": time.perf_counter() - wall_started,
                    "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
                    "peak_cuda_memory_gib": peak_cuda_memory_gib,
                    "gradient_total_before_clip": gradient_total,
                    "gradient_clip_scale": clip_scale,
                    "gradient_was_clipped": clip_scale < 1.0,
                    "gradient_norms": current_gradient,
                    "available_subsets": dict(sorted(available_counts.items())),
                    "active_subsets": dict(sorted(active_counts.items())),
                    "subset_category_counts_cumulative": dict(
                        sorted(training_subset_counts.items())
                    ),
                    "subset_reason_counts_cumulative": (
                        dict(
                            sorted(
                                training_subset_reason_counts.items()
                            )
                        )
                    ),
                    "conditional_modality_weights": conditional_weights,
                    "mean_modality_weights": {
                        name: float(
                            output.modality_weights[:, index]
                            .detach()
                            .float()
                            .mean()
                            .item()
                        )
                        for index, name in enumerate(
                            model.modality_weight_order
                        )
                    },
                    "modality_weight_entropy": float(
                        (
                            -output.modality_weights.detach().float()
                            .clamp_min(1e-12)
                            * output.modality_weights.detach().float()
                            .clamp_min(1e-12)
                            .log()
                        )
                        .sum(dim=1)
                        .mean()
                        .item()
                    ),
                    "stage_modality_weight_entropy": {
                        f"stride{stride}": float(
                            (
                                -weight_map.detach().float()
                                .clamp_min(1e-12)
                                * weight_map.detach().float()
                                .clamp_min(1e-12)
                                .log()
                            )
                            .sum(dim=1)
                            .mean()
                            .item()
                        )
                        for stride, weight_map in zip(
                            output.modality_weight_map_strides,
                            output.modality_weight_maps,
                            strict=True,
                        )
                    },
                    "stage_null_weight": {
                        f"stride{stride}": float(
                            weight_map[:, -1]
                            .detach()
                            .float()
                            .mean()
                            .item()
                        )
                        for stride, weight_map in zip(
                            output.modality_weight_map_strides,
                            output.modality_weight_maps,
                            strict=True,
                        )
                    },
                }
            )
            atomic_write_jsonl(train_log_path, logs)
        should_evaluate = (
            step % config.eval_interval == 0 or step == config.max_steps
        )
        is_new_best = False
        if should_evaluate:
            for name, module in modules.items():
                parameter_change_max[name] = max(
                    parameter_change_max[name],
                    maximum_parameter_change(module, snapshots[name]),
                )
            capacity_metrics = None
            if capacity_overfit:
                capacity_metrics = evaluate_model(
                    model,
                    train_eval_loader,
                    device=device,
                    config=config,
                    progress=progress,
                    progress_label=f"train@step{step}",
                )
                early_capacity_acceptance = _capacity_acceptance(
                    loss_history=loss_history,
                    metrics=capacity_metrics,
                    gradient_max=gradient_max,
                    parameter_updates=parameter_change_max,
                    required_modules=_required_auxiliary_modules(model),
                )
            validation = evaluate_model(
                model,
                val_loader,
                device=device,
                config=config,
                progress=progress,
                progress_label=f"val@step{step}",
            )
            overall = validation["overall"]
            candidate = {
                "step": step,
                "dice": float(overall["dice"]),
                "loss": float(validation["loss"]),
                "no_target_false_positive_rate": float(
                    overall["no_target_false_positive_rate"]
                ),
            }
            is_new_best = _validation_is_better(
                candidate, best_selection
            )
            if is_new_best:
                best_selection = candidate
            trajectory_item: dict[str, Any] = {
                "step": step,
                "validation": validation,
                "is_new_best": is_new_best,
            }
            if capacity_metrics is not None:
                trajectory_item["capacity_train"] = capacity_metrics
                trajectory_item["capacity_acceptance"] = dict(
                    early_capacity_acceptance or {}
                )
            validation_trajectory.append(trajectory_item)
            logs.append(trajectory_item)
            atomic_write_jsonl(train_log_path, logs)
        elif step % config.checkpoint_interval == 0:
            for name, module in modules.items():
                parameter_change_max[name] = max(
                    parameter_change_max[name],
                    maximum_parameter_change(module, snapshots[name]),
                )
        training_state = _training_state(
            config=config,
            loss_history=loss_history,
            ema_loss=ema_loss,
            initial_ema_loss=initial_ema_loss,
            gradient_max=gradient_max,
            parameter_change_max=parameter_change_max,
            training_step_seconds=training_step_seconds,
            training_samples_seen=training_samples_seen,
            training_steps_timed=training_steps_timed,
            clipped_steps=clipped_steps,
            clip_scale_sum=clip_scale_sum,
            clip_scale_min=clip_scale_min,
            validation_trajectory=validation_trajectory,
            best_selection=best_selection,
            training_subset_counts=training_subset_counts,
            training_subset_reason_counts=(
                training_subset_reason_counts
            ),
            training_native_auxiliary_samples=(
                training_native_auxiliary_samples
            ),
            training_active_auxiliary_samples=(
                training_active_auxiliary_samples
            ),
        )
        if is_new_best:
            duration, size = save_current_checkpoint(
                best_checkpoint_path,
                step=step,
                training_state=training_state,
                label="best",
            )
            checkpoint_total_seconds += duration
            logs.append(
                {
                    "step": step,
                    "checkpoint_best": {
                        "path": str(best_checkpoint_path),
                        "duration_seconds": duration,
                        "size_bytes": size,
                    },
                }
            )
        capacity_passed = (
            capacity_overfit
            and early_capacity_acceptance is not None
            and all(early_capacity_acceptance.values())
        )
        should_checkpoint = (
            step % config.checkpoint_interval == 0
            or step == config.max_steps
            or capacity_passed
        )
        if should_checkpoint:
            duration, size = save_current_checkpoint(
                checkpoint_path,
                step=step,
                training_state=training_state,
                label="last",
            )
            checkpoint_total_seconds += duration
            logs.append(
                {
                    "step": step,
                    "checkpoint_last": {
                        "path": str(checkpoint_path),
                        "duration_seconds": duration,
                        "size_bytes": size,
                    },
                }
            )
        if is_new_best or should_checkpoint:
            atomic_write_jsonl(train_log_path, logs)
        if capacity_passed:
            progress.phase(
                f"[overfit] all capacity thresholds passed at step={step}"
            )
            break
    progress.finish_training()
    if not checkpoint_path.is_file():
        raise RuntimeError("训练结束后缺少 checkpoint")
    if train_batcher.samples_emitted != training_samples_seen:
        raise RuntimeError(
            "训练 batcher 与累计样本计数不一致："
            f"{train_batcher.samples_emitted} != {training_samples_seen}"
        )
    if (
        sum(training_subset_counts.values()) != training_samples_seen
        or sum(training_subset_reason_counts.values())
        != training_samples_seen
    ):
        raise RuntimeError("训练子集诊断计数与累计样本数不一致")
    if (
        training_active_auxiliary_samples
        > training_native_auxiliary_samples
    ):
        raise RuntimeError("实际激活辅助样本数不能超过原生辅助样本数")
    for name, module in modules.items():
        parameter_change_max[name] = max(
            parameter_change_max[name],
            maximum_parameter_change(module, snapshots[name]),
        )
    updates = dict(parameter_change_max)
    last_train_metrics = evaluate_model(
        model,
        train_eval_loader,
        device=device,
        config=config,
        progress=progress,
        progress_label="train-last",
    )
    last_validation_metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        config=config,
        progress=progress,
        progress_label="val-last",
    )
    selected_checkpoint_path = (
        checkpoint_path
        if capacity_overfit or not best_checkpoint_path.is_file()
        else best_checkpoint_path
    )
    if selected_checkpoint_path == best_checkpoint_path:
        selected_payload = read_checkpoint(
            selected_checkpoint_path,
            expected_benchmark_contract=benchmark_contract,
        )
        model = model_from_checkpoint(selected_payload, device=device)
        train_metrics = evaluate_model(
            model,
            train_eval_loader,
            device=device,
            config=config,
            progress=progress,
            progress_label="train-best",
        )
        validation_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            config=config,
            progress=progress,
            progress_label="val-best",
        )
    else:
        train_metrics = last_train_metrics
        validation_metrics = last_validation_metrics
    progress.phase(
        "[reload] checking selected checkpoint output consistency"
    )
    reload_started = time.perf_counter()
    reload_difference = checkpoint_reload_difference(
        checkpoint_path=selected_checkpoint_path,
        model=model,
        collated=_acceptance_collated(
            benchmark_root,
            normalization=config.normalization,
            batch_size=config.batch_size,
        ),
        benchmark_contract=benchmark_contract,
        device=device,
    )
    reload_seconds = time.perf_counter() - reload_started
    progress.phase(
        "[reload] "
        f"done time={format_duration(reload_seconds)} "
        f"max_abs_difference={max(reload_difference.values()):.3e}"
    )
    initial_window = (
        sum(loss_history[: min(10, len(loss_history))])
        / min(10, len(loss_history))
    )
    final_window = (
        sum(loss_history[-min(10, len(loss_history)) :])
        / min(10, len(loss_history))
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    subset_counts = dict(sorted(training_subset_counts.items()))
    subset_reason_counts = dict(
        sorted(training_subset_reason_counts.items())
    )
    subset_reason_summary = {
        "native_none": subset_reason_counts.get("native_none", 0),
        "dropout_restored": subset_reason_counts.get(
            "dropout_restored", 0
        ),
        "variant_forced_none": subset_reason_counts.get(
            "variant_forced_none", 0
        ),
        "single": subset_counts.get("single", 0),
        "multi": subset_counts.get("multi", 0),
        "all": subset_counts.get("all", 0),
    }
    active_auxiliary_exposures = sum(
        subset_counts.get(name, 0) for name in ("single", "multi", "all")
    )
    if active_auxiliary_exposures != training_active_auxiliary_samples:
        raise RuntimeError("辅助曝光分类计数与实际激活样本数不一致")
    subset_total = sum(subset_counts.values())
    conditional_active_auxiliary_fraction = (
        training_active_auxiliary_samples
        / training_native_auxiliary_samples
        if training_native_auxiliary_samples
        else None
    )
    training_report_path = output_dir / "training_report.json"
    report: dict[str, Any] = {
        "command": command_label,
        "schema_version": CONFIG_SCHEMA_VERSION,
        "variant": config.variant,
        "backbone": config.backbone,
        "architecture": model.model_contract()["architecture"],
        "gradient_checkpointing": config.gradient_checkpointing,
        "steps": actual_step,
        "maximum_steps": config.max_steps,
        "stopped_early": actual_step < config.max_steps,
        "benchmark_index_sha256": benchmark_hash,
        "benchmark_contract": dict(benchmark_contract),
        "backbone_sha256": model.backbone_sha256,
        "modality_weight_order": list(model.modality_weight_order),
        "region_feature_dim": model.region_feature_dim,
        "checkpoint": str(selected_checkpoint_path),
        "checkpoint_last": str(checkpoint_path),
        "checkpoint_best": (
            str(best_checkpoint_path)
            if best_checkpoint_path.is_file()
            else None
        ),
        "best_step": (
            int(best_selection["step"])
            if best_selection is not None
            else None
        ),
        "best_selection": best_selection,
        "validation_trajectory": validation_trajectory,
        "training_report": str(training_report_path),
        "initial_loss_window_mean": initial_window,
        "final_loss_window_mean": final_window,
        "loss_drop_fraction": 1.0 - final_window / max(initial_window, 1e-12),
        "initial_ema_loss": initial_ema_loss,
        "final_ema_loss": ema_loss,
        "ema_drop_fraction": (
            1.0 - float(ema_loss) / max(float(initial_ema_loss), 1e-12)
            if ema_loss is not None and initial_ema_loss is not None
            else None
        ),
        "gradient_max_norms": dict(sorted(gradient_max.items())),
        "parameter_max_changes": updates,
        "subset_counts": subset_counts,
        "subset_reason_counts": subset_reason_summary,
        "subset_reason_events": subset_reason_counts,
        "subset_sampler_cardinality_counts": (
            dict(sorted(subset_sampler.cardinality_counts.items()))
            if subset_sampler is not None
            else {}
        ),
        "subset_sampler_dropout_counts": (
            dict(sorted(subset_sampler.dropout_counts.items()))
            if subset_sampler is not None
            else {}
        ),
        "native_auxiliary_sample_count": (
            training_native_auxiliary_samples
        ),
        "active_auxiliary_sample_count": (
            training_active_auxiliary_samples
        ),
        "effective_auxiliary_exposure_fraction": (
            active_auxiliary_exposures / subset_total
            if subset_total
            else None
        ),
        "conditional_active_auxiliary_fraction": (
            conditional_active_auxiliary_fraction
        ),
        "train_sampler": config.train_sampler,
        "actual_batch_size": config.batch_size,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "last_train_metrics": last_train_metrics,
        "last_validation_metrics": last_validation_metrics,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "training_step_seconds": training_step_seconds,
        "training_steps_timed": training_steps_timed,
        "training_samples_seen": training_samples_seen,
        "gradient_clipped_steps": clipped_steps,
        "gradient_clipped_fraction": (
            clipped_steps / max(training_steps_timed, 1)
        ),
        "gradient_clip_scale_mean": (
            clip_scale_sum / max(training_steps_timed, 1)
        ),
        "gradient_clip_scale_min": clip_scale_min,
        "average_steps_per_second": (
            training_steps_timed / training_step_seconds
            if training_step_seconds > 0
            else None
        ),
        "average_samples_per_second": (
            training_samples_seen / training_step_seconds
            if training_step_seconds > 0
            else None
        ),
        "checkpoint_save_seconds": checkpoint_total_seconds,
        "checkpoint_reload_seconds": reload_seconds,
        "evaluation_duration_seconds": {
            "train": train_metrics["duration_seconds"],
            "validation": validation_metrics["duration_seconds"],
            "train_last": last_train_metrics["duration_seconds"],
            "validation_last": last_validation_metrics["duration_seconds"],
        },
        "peak_cuda_memory_bytes": peak_memory,
        "peak_cuda_memory_gib": (
            peak_memory / (1024**3) if peak_memory is not None else None
        ),
    }
    required_auxiliary_modules = _required_auxiliary_modules(model)
    if capacity_overfit:
        report["acceptance"] = _capacity_acceptance(
            loss_history=loss_history,
            metrics=train_metrics,
            gradient_max=gradient_max,
            parameter_updates=updates,
            required_modules=required_auxiliary_modules,
        )
    else:
        report["acceptance"] = {
            "ema_drop_at_least_50_percent": (
                report["ema_drop_fraction"] is not None
                and report["ema_drop_fraction"] >= 0.50
            ),
            "observed_none": subset_counts.get("none", 0) > 0,
            "observed_single": subset_counts.get("single", 0) > 0,
            "observed_all": subset_counts.get("all", 0) > 0,
            "all_native_auxiliary_samples_remain_active": (
                report["conditional_active_auxiliary_fraction"] is not None
                and report["conditional_active_auxiliary_fraction"] == 1.0
            ),
            "fixed_batch_sample_count_exact": (
                training_samples_seen
                == training_steps_timed * config.batch_size
            ),
            "reload_within_1e_6": max(reload_difference.values()) <= 1e-6,
        }
    should_enforce = capacity_overfit or (
        config.variant == "proposed_dropout" and subset_sampler is not None
    )
    acceptance_passed = all(report["acceptance"].values())
    report["acceptance_enforced"] = should_enforce
    report["acceptance_passed"] = (
        acceptance_passed if should_enforce else None
    )
    report["wall_elapsed_seconds"] = time.perf_counter() - wall_started
    atomic_write_json(training_report_path, report)
    if should_enforce and not acceptance_passed:
        failed = [
            name
            for name, passed in report["acceptance"].items()
            if not passed
        ]
        progress.phase(
            "[done] "
            f"acceptance=FAIL failed={','.join(failed)} "
            f"report={training_report_path}"
        )
        raise RuntimeError(
            "训练完成，但未通过 Phase 2 验收阈值："
            + ", ".join(failed)
        )
    return report


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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@torch.no_grad()
def run_inference(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    split: str,
    source: str | None,
    limit: int | None,
    output_dir: Path,
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
    model.eval()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"推理输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        loader = make_dataloader(
            benchmark_root,
            split=split,
            source=source,
            batch_size=config.batch_size,
            normalization=config.normalization,
            shuffle=False,
            num_workers=config.num_workers,
        )
        rows: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        seen = 0
        for collated in loader:
            prepared = prepare_collated_batch(collated).to(
                device, non_blocking=device.type == "cuda"
            )
            model_batch, available, active = prepare_policy_batch(
                prepared.model, variant=model.variant, subset_sampler=None
            )
            with _autocast(config, device):
                output = model(model_batch, return_regions=True)
            if output.candidate_regions is None or output.region_features is None:
                raise RuntimeError("推理未返回区域合同")
            for index, sample_id in enumerate(prepared.sample_ids):
                if limit is not None and seen >= limit:
                    break
                prefix = f"sample_{seen:06d}"
                regions = output.candidate_regions[index]
                region_masks = np.stack(
                    [region.mask.numpy().astype(np.uint8) for region in regions],
                    axis=0,
                ) if regions else np.empty(
                    (0, *prepared.model.spatial_size), dtype=np.uint8
                )
                arrays[f"{prefix}__mask_probability"] = (
                    output.mask_probability[index, 0].float().cpu().numpy()
                )
                arrays[f"{prefix}__global_mask"] = (
                    output.mask_probability[index, 0].float().cpu().numpy() >= 0.5
                ).astype(np.uint8)
                arrays[f"{prefix}__region_masks"] = region_masks
                arrays[f"{prefix}__region_features"] = (
                    output.region_features[index].float().cpu().numpy()
                )
                arrays[f"{prefix}__modality_weights"] = (
                    output.modality_weights[index].float().cpu().numpy()
                )
                weight_map_keys: dict[str, str] = {}
                for stride, weight_map in zip(
                    output.modality_weight_map_strides,
                    output.modality_weight_maps,
                    strict=True,
                ):
                    key = (
                        f"{prefix}__modality_weight_map_stride{stride}"
                    )
                    arrays[key] = (
                        weight_map[index].float().cpu().numpy()
                    )
                    weight_map_keys[str(stride)] = key
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source": str(prepared.metadata[index]["source"]),
                        "split": split,
                        "available_modalities": list(available[index]),
                        "active_modalities": list(active[index]),
                        "no_target_score": float(
                            output.no_target_score[index].float().item()
                        ),
                        "modality_names": list(output.modality_names),
                        "modality_weights": [
                            float(value)
                            for value in output.modality_weights[index]
                            .float()
                            .cpu()
                            .tolist()
                        ],
                        "array_keys": {
                            "mask_probability": f"{prefix}__mask_probability",
                            "global_mask": f"{prefix}__global_mask",
                            "region_masks": f"{prefix}__region_masks",
                            "region_features": f"{prefix}__region_features",
                            "modality_weights": f"{prefix}__modality_weights",
                            "modality_weight_maps": weight_map_keys,
                        },
                        "regions": [
                            {
                                "region_id": region.region_id,
                                "bbox_xyxy": list(region.bbox_xyxy),
                                "centroid_xy": list(region.centroid_xy),
                                "area_pixels": region.area_pixels,
                                "confidence": region.confidence,
                            }
                            for region in regions
                        ],
                    }
                )
                seen += 1
            if limit is not None and seen >= limit:
                break
        if not rows:
            raise ValueError("推理没有选择任何样本")
        atomic_write_jsonl(temporary_dir / "predictions.jsonl", rows)
        _atomic_npz(temporary_dir / "predictions.npz", arrays)
        atomic_write_json(
            temporary_dir / "manifest.json",
            {
                "schema_version": INFERENCE_SCHEMA_VERSION,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_step": int(payload["step"]),
                "benchmark_index_sha256": benchmark_hash,
                "benchmark_contract": dict(benchmark_contract),
                "backbone_sha256": model.backbone_sha256,
                "backbone": SUPPORTED_BACKBONE,
                "architecture": model.model_contract()["architecture"],
                "split": split,
                "source": source,
                "sample_count": len(rows),
                "modality_weight_order": list(
                    model.modality_weight_order
                ),
                "modality_weight_map_strides": list(
                    model.modality_weight_map_strides
                ),
                "modality_weight_map_summary": (
                    "coverage_pool_each_stage_then_equal_stage_mean"
                ),
                "region_feature_dim": model.region_feature_dim,
            },
        )
        temporary_dir.replace(output_dir)
    except BaseException:
        for child in temporary_dir.iterdir():
            child.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise
    return {
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "jsonl": str(output_dir / "predictions.jsonl"),
        "npz": str(output_dir / "predictions.npz"),
    }


def run_smoke(config: RuntimeConfig, *, repo_root: Path) -> dict[str, Any]:
    benchmark_root, output_dir, backbone_weights, device = resolve_runtime(
        config, repo_root
    )
    require_local_backbone(backbone_weights)
    _new_output_directory(output_dir)
    if config.batch_size != 8:
        raise ValueError("真实六消融 smoke 固定要求 device batch_size=8")
    set_global_seed(config.seed)
    registry = registry_from_benchmark(benchmark_root)
    collated = _acceptance_collated(
        benchmark_root,
        normalization=config.normalization,
        batch_size=config.batch_size,
    )
    prepared = prepare_collated_batch(collated).to(
        device, non_blocking=device.type == "cuda"
    )
    benchmark_contract = benchmark_contract_from_root(benchmark_root)
    benchmark_hash = str(benchmark_contract["index_sha256"])
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_name:
        temporary_root = Path(temporary_name)
        for variant in VARIANTS:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = OAAuxSegModel(
                OAAuxSegConfig(
                    variant=variant,
                    optical_stochastic_depth=(
                        config.optical_stochastic_depth
                    ),
                    auxiliary_drop_path=config.auxiliary_drop_path,
                    decoder_dropout=config.decoder_dropout,
                    region_threshold=config.region_threshold,
                    min_region_area=config.min_region_area,
                ),
                registry,
                backbone_weights=backbone_weights,
                gradient_checkpointing=config.gradient_checkpointing,
            ).to(device)
            variant_config = config.with_overrides(variant=variant, max_steps=1)
            optimizer = make_optimizer(model, variant_config)
            scheduler = make_scheduler(
                optimizer,
                total_steps=1,
                warmup_ratio=variant_config.warmup_ratio,
                min_lr_ratio=variant_config.min_lr_ratio,
            )
            scaler = make_scaler(device)
            model_batch, available, active = prepare_policy_batch(
                prepared.model,
                variant=variant,
                subset_sampler=None,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(variant_config, device):
                output = model(model_batch)
                loss, _ = bce_dice_loss(output.mask_logits, prepared.mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_total = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), variant_config.grad_clip
                )
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                training_peak = int(
                    torch.cuda.max_memory_allocated(device)
                )
            else:
                training_peak = None
            checkpoint_path = temporary_root / f"{variant}.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=1,
                benchmark_contract=benchmark_contract,
                subset_sampler_state=None,
                training_batcher_state={
                    "smoke": True,
                    "batch_size": config.batch_size,
                },
                training_state={"runtime_config": variant_config.to_dict()},
            )
            difference = checkpoint_reload_difference(
                checkpoint_path=checkpoint_path,
                model=model,
                collated=collated,
                benchmark_contract=benchmark_contract,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                reload_peak = int(
                    torch.cuda.max_memory_allocated(device)
                )
            else:
                reload_peak = None
            reports.append(
                {
                    "variant": variant,
                    "batch_size": prepared.model.batch_size,
                    "available_subsets": dict(
                        sorted(Counter(_active_label(item) for item in available).items())
                    ),
                    "active_subsets": dict(
                        sorted(Counter(_active_label(item) for item in active).items())
                    ),
                    "loss": float(loss.item()),
                    "gradient_total_before_clip": gradient_total,
                    "checkpoint_reload_max_abs_difference": difference,
                    "peak_cuda_memory_bytes": training_peak,
                    "peak_cuda_memory_gib": (
                        training_peak / (1024**3)
                        if training_peak is not None
                        else None
                    ),
                    "checkpoint_reload_peak_cuda_memory_bytes": (
                        reload_peak
                    ),
                    "checkpoint_reload_peak_cuda_memory_gib": (
                        reload_peak / (1024**3)
                        if reload_peak is not None
                        else None
                    ),
                }
            )
            del model, optimizer, output
    report = {
        "command": "smoke",
        "benchmark_index_sha256": benchmark_hash,
        "benchmark_contract": dict(benchmark_contract),
        "backbone_sha256": sha256_file(backbone_weights),
        "backbone": config.backbone,
        "gradient_checkpointing": config.gradient_checkpointing,
        "modality_weight_order": list(registry.modality_weight_order),
        "modality_weight_map_strides": [4, 8, 16, 32],
        "region_feature_dim": registry.region_feature_dim(128),
        "variants": reports,
        "acceptance": {
            "all_six_completed": len(reports) == len(VARIANTS),
            "all_reload_within_1e_6": all(
                max(item["checkpoint_reload_max_abs_difference"].values()) <= 1e-6
                for item in reports
            ),
            "gpu_memory_measured": device.type == "cuda",
            "all_peak_memory_below_23_gib": (
                all(
                    item["peak_cuda_memory_gib"] is not None
                    and item["peak_cuda_memory_gib"] < 23
                    for item in reports
                )
                if device.type == "cuda"
                else None
            ),
        },
    }
    atomic_write_json(output_dir / "smoke_report.json", report)
    core_passed = (
        report["acceptance"]["all_six_completed"]
        and report["acceptance"]["all_reload_within_1e_6"]
    )
    gpu_memory_passed = (
        device.type != "cuda"
        or report["acceptance"]["all_peak_memory_below_23_gib"] is True
    )
    if not core_passed or not gpu_memory_passed:
        raise RuntimeError("六消融 smoke 未通过全部验收阈值")
    return report
