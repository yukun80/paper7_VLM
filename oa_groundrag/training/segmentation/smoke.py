"""OA-AuxSeg 六变体训练能力 smoke。"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from oa_groundrag.data.oa_auxseg.dataset import atomic_write_json, sha256_file
from oa_groundrag.evaluation.segmentation import _active_label
from oa_groundrag.segmentation.checkpoint import save_training_checkpoint
from oa_groundrag.segmentation.config import resolve_runtime
from oa_groundrag.segmentation.contracts import OAAuxSegConfig, RuntimeConfig, VARIANTS
from oa_groundrag.segmentation.data import (
    benchmark_contract_from_root,
    prepare_collated_batch,
    registry_from_benchmark,
)
from oa_groundrag.segmentation.losses import bce_dice_loss
from oa_groundrag.segmentation.model import OAAuxSegModel
from oa_groundrag.segmentation.policy import (
    autocast_context as _autocast,
    prepare_policy_batch,
)
from .engine import (
    _acceptance_collated,
    _new_output_directory,
    checkpoint_reload_difference,
    make_optimizer,
    make_scaler,
    make_scheduler,
    require_local_backbone,
    set_global_seed,
)

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

__all__ = ["run_smoke"]
