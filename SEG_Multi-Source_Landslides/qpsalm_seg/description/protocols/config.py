#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composable, versioned configuration for SegDesc training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from pathlib import Path
from typing import Any

import yaml

from qpsalm_seg.paths import resolve_project_path

from .stages import DESCRIPTION_STAGES, get_stage_spec


SEGDESC_CONFIG_PROTOCOL = "qpsalm_segdesc_config_v2"
DESCRIPTION_EVAL_MODES = ("gt_mask", "fixed_prediction", "end_to_end")
D4_PREDICTED_MASK_FRACTIONS = (0.25, 0.50, 0.75)
DEFAULT_JOINT_TASK_PATTERN = (
    "segmentation", "global_caption", "segmentation", "region_description",
)


@dataclass(frozen=True)
class ModelConfig:
    segmentation_config: str
    segmentation_preset: str
    segmentation_checkpoint: str
    segmentation_vision_cache: str
    description_vision_cache: str
    region_protocol: str = "vision_only"
    region_encoder: str = "mgrr"


@dataclass(frozen=True)
class DataConfig:
    description_benchmark: str
    bridge_benchmark: str
    unified_benchmark: str
    artifact_readiness_report: str | None = None
    num_workers: int = 0
    max_train_samples: int = 0
    max_val_samples: int = 256
    rsicap_mmrs_fraction: float = 0.30
    bridge_expert_task_pattern: list[str] | None = None
    predicted_mask_fraction: float = 0.25
    d4_curriculum_sampling_seed: int = 42
    predicted_index: str | None = None
    predicted_val_index: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = "bridge_auto"
    seed: int = 42
    batch_size: int = 1
    grad_accum_steps: int = 4
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
    checkpoint_metric: str = "auto"
    output_dir: str = "outputs/qpsalm_description/run"
    d_minus_one_gate: str | None = None
    d4_curriculum_gate: str | None = None
    d4_final_acceptance_gate: str | None = None
    m6_acceptance_gate: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_mode: str = "gt_mask"
    evaluation_source_dataset: str | None = None
    evaluation_region_source: str | None = None
    segmentation_mask_threshold: float = 0.5
    counterfactual_samples: int = 16
    counterfactual_modes: list[str] | None = None
    cycle_localization_samples: int = -1
    max_generate_samples: int = 32
    max_new_tokens: int = 256


@dataclass(frozen=True)
class JointConfig:
    joint_global_stages: list[str] | None = None
    joint_region_stage: str = "bridge_expert"
    joint_task_pattern: list[str] | None = None
    joint_segmentation_batch_size: int = 1
    joint_description_batch_size: int = 1
    joint_train_shared_segmentation_dense: bool = False
    segmentation_retention_max_drop: float = 0.01


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "joint": JointConfig,
}
_SECTION_FIELD_NAMES = [
    field.name
    for config_type in _SECTIONS.values()
    for field in fields(config_type)
]
_DUPLICATE_SECTION_FIELDS = {
    name for name in _SECTION_FIELD_NAMES
    if _SECTION_FIELD_NAMES.count(name) > 1
}
if _DUPLICATE_SECTION_FIELDS:
    raise RuntimeError(
        "SegDesc config section 字段重名: "
        f"{sorted(_DUPLICATE_SECTION_FIELDS)}"
    )
_FIELD_TO_SECTION = {
    field.name: section
    for section, config_type in _SECTIONS.items()
    for field in fields(config_type)
}


@dataclass(frozen=True)
class SegDescConfig:
    """Typed composition for the current SegDesc configuration protocol."""

    protocol: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    joint: JointConfig

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SegDescConfig":
        expected = {"protocol", *_SECTIONS}
        unknown = sorted(set(payload) - expected)
        missing = sorted(expected - set(payload))
        if unknown or missing:
            raise ValueError(
                "segdesc config v2 sections 不完整: "
                f"missing={missing} unknown={unknown}"
            )
        if payload["protocol"] != SEGDESC_CONFIG_PROTOCOL:
            observed_protocol = payload["protocol"]
            raise ValueError(
                "segdesc config protocol 不兼容: "
                f"{observed_protocol!r} != {SEGDESC_CONFIG_PROTOCOL!r}"
            )
        values: dict[str, Any] = {"protocol": payload["protocol"]}
        for section, config_type in _SECTIONS.items():
            section_payload = payload[section]
            if not isinstance(section_payload, dict):
                raise ValueError(f"segdesc config section={section} 必须是 object")
            known = {field.name for field in fields(config_type)}
            section_unknown = sorted(set(section_payload) - known)
            if section_unknown:
                raise ValueError(
                    f"segdesc config section={section} 未知字段: {section_unknown}"
                )
            values[section] = config_type(**section_payload)
        config = cls(**values)
        config.validate()
        return config

    def with_overrides(self, **overrides: Any) -> "SegDescConfig":
        values = {key: value for key, value in overrides.items() if value is not None}
        invalid = sorted(set(values) - set(_FIELD_TO_SECTION))
        if invalid:
            raise ValueError(f"segdesc overrides 包含未知字段: {invalid}")
        updated = self
        for section in _SECTIONS:
            section_values = {
                key: value
                for key, value in values.items()
                if _FIELD_TO_SECTION[key] == section
            }
            if section_values:
                updated = replace(
                    updated,
                    **{section: replace(getattr(updated, section), **section_values)},
                )
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_joint_task_pattern(self) -> tuple[str, ...]:
        return tuple(self.joint.joint_task_pattern or DEFAULT_JOINT_TASK_PATTERN)

    def validate(self) -> None:
        get_stage_spec(self.training.stage)

        def field_value(name: str) -> Any:
            section = _FIELD_TO_SECTION[name]
            return getattr(getattr(self, section), name)

        for name in (
            "learning_rate", "desc_adapter_lr_scale", "weight_decay",
            "max_grad_norm", "segmentation_mask_threshold",
            "rsicap_mmrs_fraction", "predicted_mask_fraction",
            "segmentation_retention_max_drop",
        ):
            value = float(field_value(name))
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数，当前={value!r}")
        for name in ("learning_rate", "desc_adapter_lr_scale", "max_grad_norm"):
            if float(field_value(name)) <= 0.0:
                raise ValueError(f"{name} 必须为正数")
        if float(self.training.weight_decay) < 0.0:
            raise ValueError("weight_decay 必须为非负数")
        if not 0.0 <= float(self.joint.segmentation_retention_max_drop) <= 1.0:
            raise ValueError("segmentation_retention_max_drop 必须位于 [0,1]")
        if self.model.region_protocol not in {"assisted", "vision_only"}:
            raise ValueError(f"未知 region_protocol={self.model.region_protocol!r}")
        if self.model.region_encoder not in {
            "mgrr", "mgrr_no_context", "roi_replay_only",
            "crop_only", "masked_pooling", "full_image_box",
        }:
            raise ValueError(f"未知 region_encoder={self.model.region_encoder!r}")
        if self.training.amp_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"未知 amp_dtype={self.training.amp_dtype!r}")
        if self.evaluation.evaluation_mode not in DESCRIPTION_EVAL_MODES:
            raise ValueError(f"未知 evaluation_mode={self.evaluation.evaluation_mode!r}")
        if self.evaluation.evaluation_mode == "end_to_end" and self.training.stage != "bridge_expert":
            raise ValueError("end_to_end evaluation 只支持 bridge_expert stage")
        if self.evaluation.evaluation_mode == "fixed_prediction" and self.training.stage != "predicted_mask":
            raise ValueError("fixed_prediction evaluation 只支持 predicted_mask stage")
        if self.evaluation.evaluation_source_dataset is not None and (
            self.training.stage != "rsicap_caption" or self.evaluation.evaluation_source_dataset != "RSIEval"
        ):
            raise ValueError(
                "evaluation_source_dataset 当前只允许 rsicap_caption/RSIEval 正式测试"
            )
        if self.evaluation.evaluation_region_source is not None and (
            self.evaluation.evaluation_region_source != "gt_global_mask"
            or self.training.stage != "bridge_expert"
            or self.evaluation.evaluation_mode not in {"gt_mask", "end_to_end"}
        ):
            raise ValueError(
                "evaluation_region_source 当前只允许 bridge_expert 的 "
                "GT-mask/end-to-end 使用 gt_global_mask"
            )
        if not 0.0 < float(self.evaluation.segmentation_mask_threshold) < 1.0:
            raise ValueError("segmentation_mask_threshold 必须位于 (0,1)")
        if int(self.evaluation.cycle_localization_samples) < -1:
            raise ValueError("cycle_localization_samples 必须为 -1、0 或正整数")
        if int(self.evaluation.cycle_localization_samples) >= 0 and (
            self.training.stage != "bridge_expert"
            or self.evaluation.evaluation_mode != "gt_mask"
            or self.model.region_protocol != "vision_only"
        ):
            raise ValueError(
                "cycle localization 只允许 frozen expert Bridge 的 Vision-only GT-mask 评价"
            )
        for name in (
            "batch_size", "grad_accum_steps", "max_steps", "log_interval",
            "val_interval", "save_interval", "joint_segmentation_batch_size",
            "joint_description_batch_size", "max_new_tokens",
        ):
            if int(field_value(name)) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        for name in (
            "seed", "warmup_steps", "num_workers", "max_train_samples",
            "max_val_samples", "max_generate_samples", "counterfactual_samples",
        ):
            if int(field_value(name)) < 0:
                raise ValueError(f"{name} 必须为非负整数")
        if int(self.data.d4_curriculum_sampling_seed) < 0:
            raise ValueError("d4_curriculum_sampling_seed 必须为非负整数")
        if self.training.stage == "predicted_mask" and not self.data.predicted_index:
            raise ValueError("predicted_mask stage 必须设置 predicted_index")
        if self.training.stage == "mmrs_caption" and not self.training.d_minus_one_gate:
            raise ValueError("D0 mmrs_caption 必须提供已通过的 d_minus_one_gate")
        if self.training.stage == "overfit":
            if int(self.training.max_steps) != 100:
                raise ValueError("D-1 overfit 固定要求 max_steps=100")
            if int(self.training.batch_size) < 2:
                raise ValueError("D-1 overfit 固定要求 batch_size>=2")
            if int(self.data.max_train_samples) not in {0, 64}:
                raise ValueError(
                    "D-1 overfit 固定要求 max_train_samples=64（0 仅表示采用协议默认 64）"
                )
        if self.evaluation.evaluation_mode == "fixed_prediction" and not self.data.predicted_index:
            raise ValueError("fixed_prediction evaluation 需要 predicted_index")
        for name in ("rsicap_mmrs_fraction", "predicted_mask_fraction"):
            value = float(field_value(name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} 必须位于 [0,1)，当前={value}")
        if (
            (self.training.stage == "predicted_mask" or self.joint.joint_region_stage == "predicted_mask")
            and not any(
                abs(float(self.data.predicted_mask_fraction) - expected) < 1.0e-12
                for expected in D4_PREDICTED_MASK_FRACTIONS
            )
        ):
            raise ValueError(
                "D4 predicted_mask_fraction 必须是预注册 curriculum tier "
                f"{D4_PREDICTED_MASK_FRACTIONS}"
            )
        bridge_tasks = self.data.bridge_expert_task_pattern or [
            "bridge", "bridge", "bridge", "dior", "global_caption",
        ]
        if not bridge_tasks or set(bridge_tasks) - {"bridge", "dior", "global_caption"}:
            raise ValueError(f"bridge_expert_task_pattern 非法: {bridge_tasks}")
        global_stages = self.joint.joint_global_stages or ["mmrs_caption", "rsicap_caption"]
        if set(global_stages) - {"mmrs_caption", "rsicap_caption"}:
            raise ValueError(f"joint_global_stages 非法: {global_stages}")
        if self.joint.joint_region_stage not in {"bridge_auto", "bridge_expert", "predicted_mask"}:
            raise ValueError(f"joint_region_stage 非法: {self.joint.joint_region_stage}")
        if self.joint.joint_region_stage == "predicted_mask" and not self.data.predicted_index:
            raise ValueError("joint_region_stage=predicted_mask 必须设置 predicted_index")
        tasks = self.resolved_joint_task_pattern()
        if not tasks or set(tasks) - {
            "segmentation", "global_caption", "region_description",
        }:
            raise ValueError(f"joint_task_pattern 非法: {tasks}")
        if set(tasks) != {"segmentation", "global_caption", "region_description"}:
            raise ValueError(
                "M7 joint_task_pattern 必须覆盖 segmentation、global_caption 和 "
                f"region_description，当前={tasks}"
            )


class SegDescConfigContractError(ValueError, RuntimeError):
    """A serialized config is not the exact current composed contract."""


def require_serialized_segdesc_config(
    value: Any,
    *,
    label: str = "segdesc checkpoint config",
) -> dict[str, Any]:
    """Validate and normalize a complete config-v2 artifact.

    YAML input may omit dataclass defaults. Saved run/checkpoint metadata may not:
    accepting a partial or formerly flat mapping would make resume and lineage
    comparisons depend on implicit defaults from the currently installed code.
    """
    if not isinstance(value, dict):
        raise SegDescConfigContractError(f"{label} 必须是完整 object")
    expected_sections = {"protocol", *_SECTIONS}
    if set(value) != expected_sections:
        raise SegDescConfigContractError(
            f"{label} 不是 {SEGDESC_CONFIG_PROTOCOL}: "
            f"missing={sorted(expected_sections - set(value))} "
            f"unknown={sorted(set(value) - expected_sections)}"
        )
    for section, config_type in _SECTIONS.items():
        section_value = value.get(section)
        expected_fields = {field.name for field in fields(config_type)}
        if not isinstance(section_value, dict) or set(section_value) != expected_fields:
            observed_fields = set(section_value) if isinstance(section_value, dict) else set()
            raise SegDescConfigContractError(
                f"{label}.{section} 字段不完整: "
                f"missing={sorted(expected_fields - observed_fields)} "
                f"unknown={sorted(observed_fields - expected_fields)}"
            )
    try:
        normalized = SegDescConfig.from_mapping(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise SegDescConfigContractError(f"{label} 非法: {exc}") from exc
    return normalized


def serialized_segdesc_config_value(
    value: Any,
    field_name: str,
    *,
    label: str = "segdesc checkpoint config",
) -> Any:
    """Read one named field without reintroducing a flat compatibility view."""
    section = _FIELD_TO_SECTION.get(field_name)
    if section is None:
        raise KeyError(f"未知 segdesc config field={field_name!r}")
    normalized = require_serialized_segdesc_config(value, label=label)
    return normalized[section][field_name]


def serialized_segdesc_config_without(
    value: Any,
    excluded_fields: set[str] | frozenset[str],
    *,
    label: str = "segdesc checkpoint config",
) -> dict[str, Any]:
    """Return a composed config with explicit fields removed for paired audits."""
    normalized = require_serialized_segdesc_config(value, label=label)
    unknown = sorted(set(excluded_fields) - set(_FIELD_TO_SECTION))
    if unknown:
        raise KeyError(f"未知 segdesc comparison fields={unknown}")
    for field_name in excluded_fields:
        section = _FIELD_TO_SECTION[field_name]
        del normalized[section][field_name]
    return normalized


def load_segdesc_config(
    path_ref: str | Path,
    overrides: dict[str, Any] | None = None,
) -> SegDescConfig:
    path = resolve_project_path(path_ref)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"segdesc config 不存在: {path_ref}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("segdesc config root 必须为 object")
    config = SegDescConfig.from_mapping(payload)
    return config.with_overrides(**(overrides or {}))
