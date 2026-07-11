#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime configuration for the SANE/QMEF/PMRD research model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_project_path


CORE_TEMPLATES = [
    "generic_landslide_v1",
    "negative_aware_landslide_v1",
    "multisource_landslide_v1",
    "terrain_evidence_landslide_v1",
    "sar_terrain_landslide_v1",
    "insar_evidence_landslide_v1",
]


@dataclass
class QPSalmConfig:
    """One resolved configuration is the only algorithmic truth for a run."""

    benchmark_dir: str = "benchmark/multisource_landslide_v1_small"
    output_dir: str = "outputs/qpsalm_refactor"
    train_index: str = "indexes/instruction_train.jsonl"
    val_index: str = "indexes/instruction_val.jsonl"
    test_index: str = "indexes/instruction_test.jsonl"
    preset: str = "sane_qmef_pmrd"

    controller: str = "qwen"
    qwen_model_path: str = "models_zoo/Qwen3-VL-2B-Instruct"
    allow_qwen_cpu: bool = False
    condition_embedding_cache: str | None = None
    visual_evidence_cache: str | None = None

    target_size: int = 128
    size_buckets: list[int] = field(default_factory=list)
    max_native_size: int = 384
    batch_size: int = 1
    num_workers: int = 4
    max_train_samples: int | None = None
    max_val_samples: int | None = 256
    train_hflip_prob: float = 0.5
    train_vflip_prob: float = 0.5

    max_steps: int | None = 1000
    num_epochs: int | None = None
    val_interval: int = 100
    save_interval: int = 1000
    save_step_checkpoints: bool = False
    save_step_validation_reports: bool = False
    keep_recent_checkpoints: int = 1
    visualize_interval: int = 100
    log_interval: int = 20
    max_val_batches: int | None = 0
    num_visualizations: int = 8
    seed: int = 42
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    decoder_dim: int = 256
    num_mask_tokens: int = 16
    num_decoder_layers: int = 2
    num_heads: int = 8
    modality_dropout: float = 0.2
    deformable_points: int = 4
    use_query_spatial_attention: bool = True
    use_mask_refinement: bool = True

    final_bce_weight: float = 1.0
    final_dice_weight: float = 1.0
    proposal_set_loss_weight: float = 0.75
    coarse_proposal_loss_weight: float = 0.25
    semantic_verifier_loss_weight: float = 0.25
    missing_modality_consistency_weight: float = 0.0
    boundary_loss_weight: float = 0.0
    min_component_area_fraction: float = 5.0e-5
    min_component_area_pixels: int = 4

    eval_threshold: float = 0.5
    checkpoint_metric: str = "positive_only_dice"
    threshold_sweep: list[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    )
    core_templates: list[str] = field(default_factory=lambda: list(CORE_TEMPLATES))

    def benchmark_path(self) -> Path:
        path = resolve_project_path(self.benchmark_dir)
        if path is None:
            raise ValueError("benchmark_dir 不能为空")
        return path

    def output_path(self) -> Path:
        path = resolve_project_path(self.output_dir)
        if path is None:
            raise ValueError("output_dir 不能为空")
        return path

    def index_path(self, split: str) -> Path:
        rel = {"train": self.train_index, "val": self.val_index, "test": self.test_index}.get(split)
        if rel is None:
            raise ValueError(f"未知 split={split!r}; expected train/val/test")
        path = Path(rel)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] in {"indexes", "data", "reports"}:
            return self.benchmark_path() / path
        return resolve_project_path(path) or path


def _known_values(data: dict[str, Any]) -> dict[str, Any]:
    valid = {item.name for item in fields(QPSalmConfig)}
    return {key: value for key, value in data.items() if key in valid}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> QPSalmConfig:
    data: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"配置文件必须是 YAML dict: {config_path}")
        data.update(_known_values(loaded))
    if overrides:
        data.update(_known_values({key: value for key, value in overrides.items() if value is not None}))
    return QPSalmConfig(**data)


def apply_config_overrides(config: QPSalmConfig, overrides: dict[str, Any] | None = None) -> QPSalmConfig:
    if not overrides:
        return config
    data = {item.name: getattr(config, item.name) for item in fields(QPSalmConfig)}
    data.update(_known_values({key: value for key, value in overrides.items() if value is not None}))
    return QPSalmConfig(**data)


def save_config(path: Path, config: QPSalmConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {item.name: getattr(config, item.name) for item in fields(QPSalmConfig)}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
