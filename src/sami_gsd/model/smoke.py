"""Bounded one-forward P2 orchestration and strict evidence reporting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from sami_gsd.contracts.model import SamiModelConfig
from sami_gsd.model.cache import QwenBackboneCache, compare_backbone_states
from sami_gsd.model.input_loader import (
    load_canonical_parent,
    load_model_parent,
    validate_benchmark_binding,
)
from sami_gsd.model.qwen_backbone import QwenBackboneWrapper
from sami_gsd.model.sensor_adapter import SensorAwareMultiImageAdapter
from sami_gsd.utilities.artifacts import atomic_write_json, sha256_file


def _gib(value: int) -> float:
    """Convert bytes to GiB without changing measurement semantics."""

    return value / (1024.0**3)


def run_model_smoke(
    config: SamiModelConfig,
    *,
    config_path: Path,
    repository_root: Path,
    benchmark_root: Path,
    parent_id: str,
    active_modality_ids: tuple[str, ...] | None,
    device: str,
    cache_dir: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    """Run one bounded official forward and atomically publish its P2 report."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing P2 smoke report: {output_path}")
    if config.active_profile != "S":
        raise ValueError("P2 acceptance smoke must run the frozen active Profile S")
    binding = validate_benchmark_binding(benchmark_root, config.benchmark)
    parent = load_canonical_parent(benchmark_root, parent_id)
    requested = active_modality_ids or tuple(modality.modality_id for modality in parent.modalities)
    loaded_parent = load_model_parent(
        benchmark_root,
        parent,
        requested_modality_ids=requested,
    )

    resolved_device = torch.device(device)
    cuda_index: int | None = None
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Profile S CUDA smoke requested but CUDA is unavailable")
        cuda_index = 0 if resolved_device.index is None else resolved_device.index
        torch.cuda.set_device(cuda_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.monotonic()
    backbone = QwenBackboneWrapper.from_pretrained(
        config,
        repository_root=repository_root,
        device=device,
    )
    adapter = SensorAwareMultiImageAdapter(config, backbone)
    batch = adapter.prepare((loaded_parent,), (requested,))
    cache: QwenBackboneCache | None = None
    if cache_dir is not None:
        if cache_dir.exists() and not cache_dir.is_dir():
            raise FileExistsError("P2 cache root exists and is not a directory")
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise FileExistsError("P2 equivalence smoke requires a new or empty cache directory")
        cache = QwenBackboneCache(cache_dir, schema_version=config.cache.schema_version)
    online = adapter.encode(batch, return_spatial_features=True, cache_store=cache)
    cache_equivalence: dict[str, Any] | None = None
    if cache is not None:
        cached = adapter.encode(batch, return_spatial_features=True, cache_store=cache)
        if not cached.from_cache:
            raise RuntimeError("second P2 encode did not reopen the new cache entry")
        cache_equivalence = compare_backbone_states(online, cached, config.cache.equivalence)
    if resolved_device.type == "cuda":
        assert cuda_index is not None
        torch.cuda.synchronize(cuda_index)
        peak_allocated_gib = _gib(torch.cuda.max_memory_allocated(cuda_index))
        peak_reserved_gib = _gib(torch.cuda.max_memory_reserved(cuda_index))
    else:
        peak_allocated_gib = None
        peak_reserved_gib = None
    elapsed = time.monotonic() - started
    errors: list[str] = []
    warnings: list[str] = []
    if peak_reserved_gib is None:
        warnings.append("profile_s_cuda_peak_not_recorded_on_non_cuda_device")
    elif peak_reserved_gib > config.smoke.profile_s_peak_limit_gib:
        errors.append(
            f"profile_s_peak_reserved_gib_exceeds_{config.smoke.profile_s_peak_limit_gib:g}"
        )
    report: dict[str, Any] = {
        "schema_version": config.smoke.output_schema_version,
        "status": "engineering_valid" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "config": {
            "schema_version": config.schema_version,
            "sha256": sha256_file(config_path),
            "active_profile": config.active_profile,
        },
        "benchmark": binding,
        "parent_id": parent_id,
        "requested_active_modality_ids": list(requested),
        "effective_view_order": [list(order) for order in online.view_order],
        "excluded_modalities": [
            [
                {
                    "modality_id": item.modality_id,
                    "availability_status": item.availability_status,
                    "reason": item.reason,
                }
                for item in group
            ]
            for group in online.excluded_modalities
        ],
        "state": {
            "schema_version": online.schema_version,
            "cache_key": online.cache_key,
            "model_fingerprint": online.model_fingerprint,
            "processor_fingerprint": online.processor_fingerprint,
            "qwen_code_revision": online.qwen_code_revision,
            "dtype": online.dtype,
            "views": [
                {
                    "view_id": view.view_id,
                    "role": view.role,
                    "language_tokens_shape": list(view.language_aligned_visual_tokens.shape),
                    "spatial_features": [
                        {"level": level.level, "shape": list(level.features.shape)}
                        for level in view.spatial_features
                    ],
                    "processor_grid_thw": list(view.transform.processor_grid_thw),
                    "merged_grid_hw": list(view.transform.merged_grid_hw),
                    "valid_pixels": int(view.valid_mask.sum().item()),
                }
                for view in online.views
            ],
        },
        "cache_equivalence": cache_equivalence,
        "profile_s_memory": {
            "device": str(resolved_device),
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "limit_gib": config.smoke.profile_s_peak_limit_gib,
            "recorded": peak_reserved_gib is not None,
            "within_limit": (
                None
                if peak_reserved_gib is None
                else peak_reserved_gib <= config.smoke.profile_s_peak_limit_gib
            ),
        },
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(output_path, report)
    return report


__all__ = ["run_model_smoke"]
