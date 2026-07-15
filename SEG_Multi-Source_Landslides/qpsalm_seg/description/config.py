#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed configuration for description-only and later joint training stages."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from qpsalm_seg.paths import resolve_project_path


DESCRIPTION_STAGES = (
    "overfit", "mmrs_caption", "rsicap_caption", "dior_alignment",
    "bridge_auto", "bridge_expert", "predicted_mask",
)
DESCRIPTION_EVAL_MODES = ("gt_mask", "fixed_prediction", "end_to_end")
DEFAULT_JOINT_TASK_PATTERN = (
    "segmentation", "global_caption", "segmentation", "region_description",
)


@dataclass
class SegDescConfig:
    segmentation_config: str
    segmentation_preset: str
    segmentation_checkpoint: str
    segmentation_vision_cache: str
    description_vision_cache: str
    description_benchmark: str
    bridge_benchmark: str
    stage: str = "bridge_auto"
    region_protocol: str = "vision_only"
    region_encoder: str = "mgrr"
    seed: int = 42
    batch_size: int = 1
    grad_accum_steps: int = 4
    num_workers: int = 0
    max_steps: int = 1000
    warmup_steps: int = 50
    learning_rate: float = 1.0e-4
    desc_adapter_lr_scale: float = 0.2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    amp_dtype: str = "bf16"
    log_interval: int = 50
    val_interval: int = 250
    save_interval: int = 500
    max_train_samples: int = 0
    max_val_samples: int = 256
    max_generate_samples: int = 32
    max_new_tokens: int = 256
    evaluation_mode: str = "gt_mask"
    segmentation_mask_threshold: float = 0.5
    counterfactual_samples: int = 16
    counterfactual_modes: list[str] | None = None
    checkpoint_metric: str = "auto"
    rsicap_mmrs_fraction: float = 0.30
    bridge_expert_task_pattern: list[str] | None = None
    predicted_mask_fraction: float = 0.25
    output_dir: str = "outputs/qpsalm_description/run"
    predicted_index: str | None = None
    joint_global_stages: list[str] | None = None
    joint_region_stage: str = "bridge_expert"
    joint_task_pattern: list[str] | None = None
    joint_segmentation_batch_size: int = 1
    joint_description_batch_size: int = 1
    joint_train_shared_segmentation_dense: bool = False
    segmentation_retention_max_drop: float = 0.01

    def resolved_joint_task_pattern(self) -> tuple[str, ...]:
        return tuple(self.joint_task_pattern or DEFAULT_JOINT_TASK_PATTERN)

    def validate(self) -> None:
        if self.stage not in DESCRIPTION_STAGES:
            raise ValueError(f"未知 description stage={self.stage!r}")
        if self.region_protocol not in {"assisted", "vision_only"}:
            raise ValueError(f"未知 region_protocol={self.region_protocol!r}")
        if self.region_encoder not in {
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        }:
            raise ValueError(f"未知 region_encoder={self.region_encoder!r}")
        if self.amp_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"未知 amp_dtype={self.amp_dtype!r}")
        if self.evaluation_mode not in DESCRIPTION_EVAL_MODES:
            raise ValueError(f"未知 evaluation_mode={self.evaluation_mode!r}")
        if self.evaluation_mode == "end_to_end" and self.stage != "bridge_expert":
            raise ValueError("end_to_end evaluation 只支持 bridge_expert stage")
        if not 0.0 < float(self.segmentation_mask_threshold) < 1.0:
            raise ValueError("segmentation_mask_threshold 必须位于 (0,1)")
        for name in ("batch_size", "grad_accum_steps", "max_steps", "log_interval"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.stage == "predicted_mask" and not self.predicted_index:
            raise ValueError("predicted_mask stage 必须设置 predicted_index")
        if self.evaluation_mode == "fixed_prediction" and not self.predicted_index:
            raise ValueError("fixed_prediction evaluation 需要 predicted_index")
        for name in ("rsicap_mmrs_fraction", "predicted_mask_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} 必须位于 [0,1)，当前={value}")
        bridge_tasks = self.bridge_expert_task_pattern or [
            "bridge", "bridge", "bridge", "dior", "global_caption",
        ]
        if not bridge_tasks or set(bridge_tasks) - {"bridge", "dior", "global_caption"}:
            raise ValueError(f"bridge_expert_task_pattern 非法: {bridge_tasks}")
        global_stages = self.joint_global_stages or ["mmrs_caption", "rsicap_caption"]
        if set(global_stages) - {"mmrs_caption", "rsicap_caption"}:
            raise ValueError(f"joint_global_stages 非法: {global_stages}")
        if self.joint_region_stage not in {"bridge_auto", "bridge_expert", "predicted_mask"}:
            raise ValueError(f"joint_region_stage 非法: {self.joint_region_stage}")
        tasks = self.resolved_joint_task_pattern()
        if not tasks or set(tasks) - {"segmentation", "global_caption", "region_description"}:
            raise ValueError(f"joint_task_pattern 非法: {tasks}")


def load_segdesc_config(path_ref: str | Path, overrides: dict[str, Any] | None = None) -> SegDescConfig:
    path = resolve_project_path(path_ref)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"segdesc config 不存在: {path_ref}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("segdesc config root 必须为 object")
    known = {field.name for field in fields(SegDescConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"segdesc config 包含未知字段: {unknown}")
    config = SegDescConfig(**payload)
    values = {key: value for key, value in (overrides or {}).items() if value is not None}
    if values:
        invalid = sorted(set(values) - known)
        if invalid:
            raise ValueError(f"segdesc overrides 包含未知字段: {invalid}")
        config = replace(config, **values)
    config.validate()
    return config
