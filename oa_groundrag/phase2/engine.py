"""Phase 2 OA-AuxSeg 统一训练、评价、smoke、过拟合与推理引擎。"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from scripts.phase1_benchmark_build.benchmark_common import (
    BenchmarkDataset,
    atomic_write_json,
    atomic_write_jsonl,
    collate_benchmark_samples,
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
    OAAuxSegBatch,
    OAAuxSegConfig,
    RuntimeConfig,
    SUPPORTED_BACKBONE,
    SUPPORTED_AUXILIARY_ORDER,
    VARIANTS,
)
from .data import (
    AuxiliarySubsetSampler,
    PreparedBatch,
    available_auxiliaries_by_sample,
    filter_auxiliaries,
    make_dataloader,
    prepare_collated_batch,
    registry_from_benchmark,
)
from .losses import bce_dice_loss
from .metrics import SegmentationMetrics
from .model import OAAuxSegModel


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
            "Phase 2 v4 真实训练要求本地 torchvision ConvNeXt-Small 官方 "
            f"state_dict，当前不存在：{path}"
        )


def make_optimizer(
    model: OAAuxSegModel, config: RuntimeConfig
) -> torch.optim.AdamW:
    backbone_modules = (
        model.backbone,
        model.optical_stem_norm,
        model.optical_stems,
    )
    backbone_parameters = [
        parameter
        for module in backbone_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not new_parameters:
        raise RuntimeError("optimizer 参数分组为空")
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_lr},
            {"params": new_parameters, "lr": config.new_lr},
        ],
        weight_decay=config.weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


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


class CyclingDataIterator:
    """记录当前 epoch 的 generator 起点与 batch offset，支持精确跳回。"""

    def __init__(
        self, loader: DataLoader[dict[str, Any]], generator: torch.Generator
    ) -> None:
        self.loader = loader
        self.generator = generator
        self.epoch = 0
        self.batch_offset = 0
        self.epoch_start_generator_state: Tensor | None = None
        self._iterator: Iterable[dict[str, Any]] | None = None

    def _start_epoch(self) -> None:
        self.epoch_start_generator_state = self.generator.get_state().clone()
        self._iterator = iter(self.loader)
        self.batch_offset = 0

    def next(self) -> dict[str, Any]:
        if self._iterator is None:
            self._start_epoch()
        try:
            value = next(self._iterator)  # type: ignore[arg-type]
        except StopIteration:
            self.epoch += 1
            self._start_epoch()
            value = next(self._iterator)  # type: ignore[arg-type]
        self.batch_offset += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        if self.epoch_start_generator_state is None:
            self._start_epoch()
        return {
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "epoch_start_generator_state": self.epoch_start_generator_state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        target_offset = int(state["batch_offset"])
        self.generator.set_state(state["epoch_start_generator_state"])
        self._start_epoch()
        for _ in range(target_offset):
            try:
                next(self._iterator)  # type: ignore[arg-type]
            except StopIteration as error:
                raise ValueError("checkpoint batch_offset 超出当前 DataLoader") from error
            self.batch_offset += 1


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
) -> dict[str, Any]:
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
    for collated in loader:
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
        weight_sum += output.modality_weights.detach().double().sum(dim=0).cpu()
        for index in range(batch_size):
            _metric_update_one(
                by_source,
                str(prepared.metadata[index]["source"]),
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
    if was_training:
        model.train()
    return {
        "loss": total_loss / max(total_samples, 1),
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
    }


def _gradient_modules(model: OAAuxSegModel) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {
        f"auxiliary_adapter_{name}": module
        for name, module in model.auxiliary_adapters.items()
    }
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


def _gradient_norm(module: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt().item())


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
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        source_indices[str(row["source"])].append(index)
    chosen = [
        max(
            indices,
            key=lambda index: (
                len(dataset.rows[index]["auxiliaries"]),
                -index,
            ),
        )
        for indices in source_indices.values()
    ]
    for index in range(len(dataset)):
        if len(chosen) >= batch_size:
            break
        if index not in chosen:
            chosen.append(index)
    if len(chosen) != batch_size:
        raise RuntimeError("无法构建指定大小的 acceptance batch")
    collated = collate_benchmark_samples([dataset[index] for index in chosen])
    channel_counts = {
        int(value.shape[0]) for value in collated["optical"]
    }
    if not {3, 4, 10, 12}.issubset(channel_counts):
        raise RuntimeError("acceptance batch 未覆盖 3/4/10/12 通道光学")
    auxiliary_names = set(collated["auxiliaries"])
    expected_auxiliaries = set(SUPPORTED_AUXILIARY_ORDER)
    if auxiliary_names != expected_auxiliaries:
        raise RuntimeError(
            "acceptance batch 未覆盖全部已注册辅助模态："
            f"actual={sorted(auxiliary_names)} "
            f"expected={sorted(expected_auxiliaries)}"
        )
    present_count = [0] * len(chosen)
    for packed in collated["auxiliaries"].values():
        for sample_index in packed["sample_indices"].tolist():
            present_count[int(sample_index)] += 1
    if 0 not in present_count or max(present_count) < 3:
        raise RuntimeError("acceptance batch 未同时覆盖无辅助与多辅助样本")
    return collated


@torch.no_grad()
def checkpoint_reload_difference(
    *,
    checkpoint_path: Path,
    model: OAAuxSegModel,
    collated: Mapping[str, Any],
    benchmark_index_sha256: str,
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
        expected_benchmark_index_sha256=benchmark_index_sha256,
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
    loader: CyclingDataIterator,
) -> dict[str, Any]:
    return {
        "runtime_config": config.to_dict(),
        "loss_history": list(loss_history),
        "ema_loss": ema_loss,
        "initial_ema_loss": initial_ema_loss,
        "gradient_max": dict(gradient_max),
        "loader_state": loader.state_dict(),
    }


def run_training(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    capacity_overfit: bool,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    benchmark_root, output_dir, backbone_weights, device = resolve_runtime(
        config, repo_root
    )
    benchmark_hash = sha256_file(benchmark_root / "index.jsonl")
    set_global_seed(config.seed)
    data_generator = torch.Generator().manual_seed(config.seed + 1)
    subset_sampler = (
        AuxiliarySubsetSampler(config.seed + 2, config.modality_dropout)
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
        resume_payload = None
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_payload = read_checkpoint(
            Path(resume_checkpoint),
            expected_benchmark_index_sha256=benchmark_hash,
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
            "modality_dropout",
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
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(
        optimizer, total_steps=config.max_steps, warmup_ratio=config.warmup_ratio
    )
    scaler = make_scaler(device)
    train_loader = make_dataloader(
        benchmark_root,
        split="train",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=True,
        num_workers=config.num_workers,
        generator=data_generator,
    )
    val_loader = make_dataloader(
        benchmark_root,
        split="val",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    cycling = CyclingDataIterator(train_loader, data_generator)
    if resume_payload is not None:
        data_generator.set_state(resume_payload["dataloader_generator_state"])
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
        cycling.load_state_dict(resume_payload["training_state"]["loader_state"])
    modules = _gradient_modules(model)
    snapshots = {
        name: module_parameter_snapshot(module) for name, module in modules.items()
    }
    logs: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "checkpoint_last.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(start_step + 1, config.max_steps + 1):
        prepared = prepare_collated_batch(cycling.next()).to(
            device, non_blocking=device.type == "cuda"
        )
        model_batch, available, active = prepare_policy_batch(
            prepared.model,
            variant=config.variant,
            subset_sampler=subset_sampler,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            output = model(model_batch)
            loss, components = bce_dice_loss(output.mask_logits, prepared.mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        current_gradient = {
            name: _gradient_norm(module) for name, module in modules.items()
        }
        for name, value in current_gradient.items():
            gradient_max[name] = max(gradient_max[name], value)
        gradient_total = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_value = float(loss.detach().item())
        loss_history.append(loss_value)
        ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value
        if initial_ema_loss is None:
            initial_ema_loss = ema_loss
        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            active_counts = Counter(_active_label(names) for names in active)
            available_counts = Counter(_active_label(names) for names in available)
            logs.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "bce": float(components["bce"].item()),
                    "dice_loss": float(components["dice"].item()),
                    "ema_loss": ema_loss,
                    "lr": [float(group["lr"]) for group in optimizer.param_groups],
                    "gradient_total_before_clip": gradient_total,
                    "gradient_norms": current_gradient,
                    "available_subsets": dict(sorted(available_counts.items())),
                    "active_subsets": dict(sorted(active_counts.items())),
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
            atomic_write_jsonl(output_dir / "train_log.jsonl", logs)
        training_state = _training_state(
            config=config,
            loss_history=loss_history,
            ema_loss=ema_loss,
            initial_ema_loss=initial_ema_loss,
            gradient_max=gradient_max,
            loader=cycling,
        )
        should_checkpoint = (
            step % config.checkpoint_interval == 0 or step == config.max_steps
        )
        if should_checkpoint:
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                benchmark_index_sha256=benchmark_hash,
                subset_sampler_state=(
                    subset_sampler.state_dict() if subset_sampler is not None else None
                ),
                dataloader_generator_state=data_generator.get_state(),
                training_state=training_state,
            )
        if step % config.eval_interval == 0 and step != config.max_steps:
            logs.append(
                {
                    "step": step,
                    "validation": evaluate_model(
                        model, val_loader, device=device, config=config
                    ),
                }
            )
            atomic_write_jsonl(output_dir / "train_log.jsonl", logs)
    if not checkpoint_path.is_file():
        raise RuntimeError("训练结束后缺少 checkpoint")
    train_eval_loader = make_dataloader(
        benchmark_root,
        split="train",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    train_metrics = evaluate_model(
        model, train_eval_loader, device=device, config=config
    )
    validation_metrics = evaluate_model(
        model, val_loader, device=device, config=config
    )
    reload_difference = checkpoint_reload_difference(
        checkpoint_path=checkpoint_path,
        model=model,
        collated=_acceptance_collated(
            benchmark_root,
            normalization=config.normalization,
            batch_size=config.batch_size,
        ),
        benchmark_index_sha256=benchmark_hash,
        device=device,
    )
    updates = {
        name: maximum_parameter_change(module, snapshots[name])
        for name, module in modules.items()
    }
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
    subset_counts = (
        dict(sorted(subset_sampler.counts.items()))
        if subset_sampler is not None
        else {}
    )
    report: dict[str, Any] = {
        "command": "overfit" if capacity_overfit else "train",
        "schema_version": CONFIG_SCHEMA_VERSION,
        "variant": config.variant,
        "backbone": config.backbone,
        "architecture": model.model_contract()["architecture"],
        "gradient_checkpointing": config.gradient_checkpointing,
        "steps": config.max_steps,
        "benchmark_index_sha256": benchmark_hash,
        "backbone_sha256": model.backbone_sha256,
        "modality_weight_order": list(model.modality_weight_order),
        "region_feature_dim": model.region_feature_dim,
        "checkpoint": str(checkpoint_path),
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
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "peak_cuda_memory_bytes": peak_memory,
        "peak_cuda_memory_gib": (
            peak_memory / (1024**3) if peak_memory is not None else None
        ),
    }
    required_auxiliary_modules = (
        tuple(
            f"auxiliary_adapter_{name}" for name in model.modality_order
        )
        + (
            "mspa_stride4",
            "quality_selector_stride4",
            "frm_stride4",
            "ffm_stride4",
            "mspa_stride8",
            "quality_selector_stride8",
            "frm_stride8",
            "ffm_stride8",
            "mspa_stride16",
            "quality_selector_stride16",
            "frm_stride16",
            "ffm_stride16",
            "mspa_stride32",
            "quality_selector_stride32",
            "frm_stride32",
            "ffm_stride32",
        )
    )
    if capacity_overfit:
        overall = train_metrics["overall"]
        report["acceptance"] = {
            "loss_drop_at_least_90_percent": report["loss_drop_fraction"] >= 0.90,
            "micro_dice_at_least_0_95": overall["dice"] >= 0.95,
            "positive_dice_at_least_0_90": overall["positive_only_dice"] >= 0.90,
            "empty_mask_fpr_zero": overall["no_target_false_positive_rate"] == 0,
            "empty_mean_probability_at_most_0_01": (
                overall["empty_mean_foreground_probability"] <= 0.01
            ),
            "all_auxiliary_gradients_nonzero": all(
                gradient_max.get(name, 0.0) > 0
                for name in required_auxiliary_modules
            ),
            "all_auxiliary_parameters_updated": all(
                updates.get(name, 0.0) > 0
                for name in required_auxiliary_modules
            ),
        }
    else:
        report["acceptance"] = {
            "ema_drop_at_least_50_percent": (
                report["ema_drop_fraction"] is not None
                and report["ema_drop_fraction"] >= 0.50
            ),
            "observed_none": subset_counts.get("none", 0) > 0,
            "observed_single": subset_counts.get("single", 0) > 0,
            "observed_multi_or_all": (
                subset_counts.get("multi", 0) + subset_counts.get("all", 0) > 0
            ),
            "reload_within_1e_6": max(reload_difference.values()) <= 1e-6,
        }
    atomic_write_json(output_dir / "training_report.json", report)
    should_enforce = capacity_overfit or (
        config.variant == "proposed_dropout" and subset_sampler is not None
    )
    if should_enforce and not all(report["acceptance"].values()):
        raise RuntimeError("训练完成，但未通过配置对应的 Phase 2 验收阈值")
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
    benchmark_hash = sha256_file(benchmark_root / "index.jsonl")
    payload = read_checkpoint(
        checkpoint_path,
        expected_benchmark_index_sha256=benchmark_hash,
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
    benchmark_hash = sha256_file(benchmark_root / "index.jsonl")
    payload = read_checkpoint(
        checkpoint_path,
        expected_benchmark_index_sha256=benchmark_hash,
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
                "schema_version": "oa_auxseg_inference_v4",
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_step": int(payload["step"]),
                "benchmark_index_sha256": benchmark_hash,
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
    benchmark_hash = sha256_file(benchmark_root / "index.jsonl")
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
                optimizer, total_steps=1, warmup_ratio=variant_config.warmup_ratio
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
                benchmark_index_sha256=benchmark_hash,
                subset_sampler_state=None,
                dataloader_generator_state=torch.Generator()
                .manual_seed(config.seed)
                .get_state(),
                training_state={"runtime_config": variant_config.to_dict()},
            )
            difference = checkpoint_reload_difference(
                checkpoint_path=checkpoint_path,
                model=model,
                collated=collated,
                benchmark_index_sha256=benchmark_hash,
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
