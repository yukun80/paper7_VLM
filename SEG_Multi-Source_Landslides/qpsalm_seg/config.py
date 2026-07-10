#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-PSALM-Seg 配置读取。

脚本作用：集中管理训练、验证、数据读取和模型超参数，支持 YAML 配置与
CLI 覆盖项合并。
主要输入：SEG_Multi-Source_Landslides/configs/*.yaml。
主要输出：QPSalmConfig dataclass。
是否改写原始数据：不会。
典型用法：from qpsalm_seg.config import load_config。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


CORE_TEMPLATES = [
    "generic_landslide_v1",
    "negative_aware_landslide_v1",
    "multisource_landslide_v1",
    "terrain_evidence_landslide_v1",
    "sar_terrain_landslide_v1",
    "insar_evidence_landslide_v1",
]

LOSS_STAGE_CHOICES = ["full", "base", "proposal", "condition", "modality"]
LOSS_STAGE_PRESETS: dict[str, dict[str, Any]] = {
    "base": {
        "proposal_cls_weight": 0.0,
        "condition_cls_weight": 0.0,
        "proposal_mask_weight": 0.0,
        "condition_ranking_loss_weight": 0.0,
        "selection_ranking_loss_weight": 0.0,
        "empty_mask_suppression_weight": 0.0,
        "empty_proposal_suppression_weight": 0.0,
        "query_diversity_loss_weight": 0.0,
        "proposal_mask_diversity_loss_weight": 0.0,
        "gate_entropy_loss_weight": 0.0,
        "query_usage_balance_loss_weight": 0.0,
        "selection_condition_weight": 0.0,
        "selection_evidence_weight": 0.0,
        "evidence_cls_weight": 0.0,
        "evidence_ranking_loss_weight": 0.0,
        "final_foreground_gate_weight": 0.0,
        "final_mask_fusion": "weighted_average",
    },
    "proposal": {
        "proposal_cls_weight": 0.2,
        "condition_cls_weight": 0.0,
        "proposal_mask_weight": 0.5,
        "condition_ranking_loss_weight": 0.0,
        "selection_ranking_loss_weight": 0.0,
        "empty_mask_suppression_weight": 0.0,
        "empty_proposal_suppression_weight": 0.0,
        "query_diversity_loss_weight": 0.0,
        "proposal_mask_diversity_loss_weight": 0.0,
        "gate_entropy_loss_weight": 0.0,
        "query_usage_balance_loss_weight": 0.0,
        "proposal_soft_target_topk": 1,
        "selection_condition_weight": 0.0,
        "selection_evidence_weight": 0.0,
        "evidence_cls_weight": 0.0,
        "evidence_ranking_loss_weight": 0.0,
        "final_foreground_gate_weight": 0.0,
        "final_mask_fusion": "weighted_average",
    },
    "condition": {
        "proposal_cls_weight": 0.2,
        "condition_cls_weight": 0.2,
        "proposal_mask_weight": 0.5,
        "condition_ranking_loss_weight": 0.1,
        "selection_ranking_loss_weight": 0.2,
        "empty_mask_suppression_weight": 0.0,
        "empty_proposal_suppression_weight": 0.0,
        "query_diversity_loss_weight": 0.0,
        "proposal_mask_diversity_loss_weight": 0.0,
        "gate_entropy_loss_weight": 0.0,
        "query_usage_balance_loss_weight": 0.0,
        "proposal_soft_target_topk": 1,
        "selection_condition_weight": 0.5,
        "selection_evidence_weight": 0.0,
        "evidence_cls_weight": 0.0,
        "evidence_ranking_loss_weight": 0.0,
        "final_foreground_gate_weight": 0.0,
        "final_mask_fusion": "weighted_average",
    },
    "modality": {
        "proposal_cls_weight": 0.2,
        "condition_cls_weight": 0.2,
        "proposal_mask_weight": 0.5,
        "condition_ranking_loss_weight": 0.1,
        "selection_ranking_loss_weight": 0.2,
        "empty_mask_suppression_weight": 0.3,
        "empty_proposal_suppression_weight": 0.1,
        "query_diversity_loss_weight": 0.0,
        "proposal_mask_diversity_loss_weight": 0.0,
        "gate_entropy_loss_weight": 0.02,
        "query_usage_balance_loss_weight": 0.0,
        "proposal_soft_target_topk": 1,
        "selection_condition_weight": 0.5,
        "selection_evidence_weight": 0.0,
        "evidence_cls_weight": 0.0,
        "evidence_ranking_loss_weight": 0.0,
        "final_foreground_gate_weight": 0.0,
        "final_mask_fusion": "weighted_average",
    },
    "full": {},
}


@dataclass
class QPSalmConfig:
    """第一版原型配置。"""

    benchmark_dir: str = "benchmark/multisource_landslide_v1_small"
    output_dir: str = "outputs/qpsalm_small_qwen_core"
    train_index: str = "indexes/instruction_train.jsonl"
    val_index: str = "indexes/instruction_val.jsonl"
    test_index: str = "indexes/instruction_test.jsonl"
    controller: str = "qwen"
    qwen_model_path: str = "models_zoo/Qwen3-VL-2B-Instruct"
    allow_qwen_cpu: bool = False
    condition_embedding_cache: str | None = None

    target_size: int = 128
    batch_size: int = 1
    num_workers: int = 4
    max_steps: int = 1000
    val_interval: int = 100
    save_interval: int = 1000
    keep_recent_checkpoints: int = 2
    visualize_interval: int = 100
    log_interval: int = 20
    max_val_batches: int | None = 0
    num_visualizations: int = 8
    max_train_samples: int | None = None
    max_val_samples: int | None = 256
    seed: int = 42
    train_hflip_prob: float = 0.5
    train_vflip_prob: float = 0.5

    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    decoder_dim: int = 256
    num_mask_tokens: int = 16
    num_decoder_layers: int = 2
    num_heads: int = 8
    modality_dropout: float = 0.2
    use_gsd_film: bool = True
    use_spatial_modality_gate: bool = True
    use_query_modality_attention: bool = True
    query_modality_feature_weight: float = 0.35
    use_evidence_reasoning: bool = True
    evidence_reasoning_weight: float = 0.35
    selection_evidence_weight: float = 0.25
    use_visual_evidence: bool = True
    visual_evidence_cache: str | None = None
    visual_evidence_weight: float = 0.25
    visual_evidence_feature_weight: float = 0.15
    loss_stage: str = "full"
    use_focal_loss: bool = False
    use_box_prior: bool = False
    boundary_loss_weight: float = 0.0
    condition_ranking_loss_weight: float = 0.1
    selection_ranking_loss_weight: float = 0.2
    foreground_bce_pos_weight: float = 1.0
    mask_bce_weight: float = 1.0
    mask_dice_weight: float = 1.0
    mask_tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    proposal_cls_weight: float = 0.2
    condition_cls_weight: float = 0.2
    proposal_mask_weight: float = 0.5
    empty_mask_suppression_weight: float = 0.0
    empty_proposal_suppression_weight: float = 0.0
    proposal_positive_weight: float = 1.0
    condition_positive_weight: float = 1.0
    evidence_positive_weight: float = 1.0
    query_diversity_loss_weight: float = 0.0
    proposal_mask_diversity_loss_weight: float = 0.0
    gate_entropy_loss_weight: float = 0.0
    proposal_soft_target_topk: int = 1
    proposal_soft_target_temperature: float = 0.10
    query_usage_balance_loss_weight: float = 0.0
    evidence_cls_weight: float = 0.1
    evidence_ranking_loss_weight: float = 0.1
    selection_proposal_weight: float = 1.0
    selection_condition_weight: float = 1.0
    selection_temperature: float = 1.0
    final_foreground_gate_weight: float = 0.0
    final_mask_fusion: str = "weighted_average"
    final_topk: int = 3
    final_noisy_or_epsilon: float = 1.0e-5
    canonical_combo_loss_weights: dict[str, float] = field(default_factory=dict)
    eval_threshold: float = 0.5
    threshold_sweep: list[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    )
    core_templates: list[str] = field(default_factory=lambda: list(CORE_TEMPLATES))

    def benchmark_path(self) -> Path:
        return Path(self.benchmark_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def index_path(self, split: str) -> Path:
        if split == "train":
            rel = self.train_index
        elif split == "val":
            rel = self.val_index
        elif split == "test":
            rel = self.test_index
        else:
            raise ValueError(f"未知 split={split!r}; expected train/val/test")
        path = Path(rel)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] in {"indexes", "data", "reports"}:
            return self.benchmark_path() / path
        return path


def _clean_none_values(data: dict[str, Any]) -> dict[str, Any]:
    """YAML 中的 null 保留为 None，其余字段原样传入 dataclass。"""
    valid = {f.name for f in fields(QPSalmConfig)}
    return {key: value for key, value in data.items() if key in valid}


def parse_combo_loss_weights(value: Any) -> dict[str, float] | None:
    """解析 canonical combo loss weight 覆盖项。

    支持 YAML/JSON dict，也支持命令行里的 ``dem+s2=2.5,dem+s1+s2=1.5``。
    key 可以写 ``s1`` 或 ``canonical_combo=s1``，内部统一保留原始 combo 名。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        raw_items = value.items()
    else:
        text = str(value).strip()
        if not text:
            return {}
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            raw_items = loaded.items()
        else:
            pairs: list[tuple[str, str]] = []
            for chunk in text.split(","):
                item = chunk.strip()
                if not item:
                    continue
                if "=" not in item:
                    raise ValueError(f"combo loss weight 必须是 key=value: {item}")
                key, weight = item.split("=", 1)
                pairs.append((key.strip(), weight.strip()))
            raw_items = pairs
    weights: dict[str, float] = {}
    for key, weight in raw_items:
        combo = str(key).strip()
        if combo.startswith("canonical_combo="):
            combo = combo.split("=", 1)[1]
        if not combo:
            continue
        weights[combo] = float(weight)
    return weights


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> QPSalmConfig:
    """读取 YAML 配置，并应用 CLI 覆盖项。"""
    data: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"配置文件必须是 YAML dict: {config_path}")
        data.update(_clean_none_values(loaded))
    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    if "canonical_combo_loss_weights" in data:
        data["canonical_combo_loss_weights"] = parse_combo_loss_weights(data["canonical_combo_loss_weights"]) or {}
    return QPSalmConfig(**data)


def apply_config_overrides(config: QPSalmConfig, overrides: dict[str, Any] | None = None) -> QPSalmConfig:
    """对已有配置应用非 None 覆盖项，供 CLI 保持 preset/显式参数顺序。"""
    if not overrides:
        return config
    data = {f.name: getattr(config, f.name) for f in fields(QPSalmConfig)}
    data.update({key: value for key, value in overrides.items() if value is not None and key in data})
    if "canonical_combo_loss_weights" in data:
        data["canonical_combo_loss_weights"] = parse_combo_loss_weights(data["canonical_combo_loss_weights"]) or {}
    return QPSalmConfig(**data)


def apply_loss_stage(config: QPSalmConfig, stage: str | None) -> QPSalmConfig:
    """按实验阶段启用一组 loss/selection 默认项，便于建立 ablation 证据链。"""
    stage_name = str(stage or config.loss_stage or "full").strip().lower()
    if stage_name not in LOSS_STAGE_PRESETS:
        raise ValueError(f"未知 loss_stage={stage_name!r}; 可选: {', '.join(LOSS_STAGE_CHOICES)}")
    updates = {"loss_stage": stage_name, **LOSS_STAGE_PRESETS[stage_name]}
    return apply_config_overrides(config, updates)


def save_config(path: Path, config: QPSalmConfig) -> None:
    """保存实际运行配置，便于 checkpoint 复现。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {f.name: getattr(config, f.name) for f in fields(QPSalmConfig)}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
