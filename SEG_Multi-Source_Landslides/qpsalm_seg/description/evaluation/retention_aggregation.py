#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-seed aggregation and publication replay for M7 retention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import canonical_sha256
from .retention_contracts import (
    M7_RETENTION_SEED_GATE_PROTOCOL,
    json_object,
)
from .retention_validation import validate_m7_retention_gate


def aggregate_m7_retention_seed_gates(
    gate_paths: Sequence[str | Path],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Require three independent M7 checkpoints to pass one exact full-val baseline."""
    if len(gate_paths) != 3 or len(seeds) != 3:
        raise ValueError("M7 Small retention 必须恰好提供 3 个 gate 和 3 个 seed")
    normalized_seeds = [int(value) for value in seeds]
    if len(set(normalized_seeds)) != 3:
        raise ValueError("M7 Small retention 的 3 个 seed 必须互不相同")
    audits = [
        validate_m7_retention_gate(path, expected_seed=seed)
        for path, seed in zip(gate_paths, normalized_seeds, strict=True)
    ]
    uniqueness = {
        "gate_paths": [item["gate"] for item in audits],
        "gate_sha256": [item["gate_sha256"] for item in audits],
        "joint_checkpoint_sha256": [
            item["joint_checkpoint_sha256"] for item in audits
        ],
    }
    repeated = {
        label: values
        for label, values in uniqueness.items()
        if len(set(values)) != len(values)
    }
    if repeated:
        raise ValueError(f"M7 三种子产物不独立，存在重复 gate/checkpoint: {repeated}")

    baseline_signatures = {
        canonical_sha256({
            "binding": item["baseline_binding"],
            "checkpoint_replay": {
                key: value
                for key, value in item[
                    "baseline_checkpoint_replay_audit"
                ].items()
                if key not in {"frozen_report", "replay_report"}
            },
            "population": item["baseline_population"],
            "positive_dice": item["baseline_positive_dice"],
            "maximum_allowed_drop": item["maximum_allowed_drop"],
        })
        for item in audits
    }
    if len(baseline_signatures) != 1:
        raise ValueError("M7 三种子 retention 未使用完全相同的 full-val baseline")
    config_signatures = {
        item["joint_scientific_config_sha256"] for item in audits
    }
    if len(config_signatures) != 1:
        raise ValueError("M7 三种子 joint scientific config 不一致")
    joint_training_population_signatures = {
        canonical_sha256(item["joint_training_population_binding"])
        for item in audits
    }
    if len(joint_training_population_signatures) != 1:
        raise ValueError(
            "M7 三种子 segmentation/global/region 训练 population 不一致"
        )
    cache_content_signatures = {
        canonical_sha256({
            "manifest_sha256": item[
                "description_cache_artifact_provenance"
            ].get("manifest_sha256"),
            "validation_report_sha256": item[
                "description_cache_artifact_provenance"
            ].get("validation_report_sha256"),
            "shard_inventory_sha256": item[
                "description_cache_artifact_provenance"
            ].get("shard_inventory_sha256"),
        })
        for item in audits
    }
    if len(cache_content_signatures) != 1:
        raise ValueError(
            "M7 三种子未使用相同内容的 Description Vision Cache"
        )
    d4_data_signatures = {
        canonical_sha256({
            "frozen_gate_audit": item["d4_final_acceptance_audit"].get(
                "frozen_gate_audit"
            ),
            "source_train_region_data_audit": item[
                "d4_final_acceptance_audit"
            ].get("source_train_region_data_audit"),
            "source_val_predicted_index_audit": item[
                "d4_final_acceptance_audit"
            ].get("source_val_predicted_index_audit"),
        })
        for item in audits
    }
    if len(d4_data_signatures) != 1:
        raise ValueError("M7 三种子未使用同一 D4/Bridge train-val population")
    m6_population_signatures = {
        canonical_sha256({
            "frozen_gate_audit": item["m6_acceptance_audit"].get(
                "frozen_gate_audit"
            ),
            "evaluation_parent_populations": item[
                "m6_acceptance_audit"
            ].get("evaluation_parent_populations"),
            "segmentation_instruction_source_binding": item[
                "m6_acceptance_audit"
            ].get("segmentation_instruction_source_binding"),
        })
        for item in audits
    }
    if len(m6_population_signatures) != 1:
        raise ValueError("M7 三种子未使用同一 M6 GT/fixed/end-to-end expert population")
    d_minus_one_signatures = {
        canonical_sha256(item["d_minus_one_acceptance_audit"])
        for item in audits
    }
    if len(d_minus_one_signatures) != 1:
        raise ValueError("M7 三种子未继承同一个 D-1 acceptance")
    joint_population_hashes = {
        item["joint_population"]["sha256"] for item in audits
    }
    if len(joint_population_hashes) != 1:
        raise ValueError("M7 三种子 joint full-val population 不一致")

    passed = sum(int(item["passed"]) for item in audits)
    drops = [float(item["absolute_drop"]) for item in audits]
    dice = [float(item["joint_positive_dice"]) for item in audits]
    return {
        "protocol": M7_RETENTION_SEED_GATE_PROTOCOL,
        "required_seed_count": 3,
        "seeds": normalized_seeds,
        "all_seeds_distinct": True,
        "all_joint_checkpoints_unique": True,
        "same_full_val_baseline": True,
        "same_joint_config_except_seed": True,
        "same_joint_training_population": True,
        "same_description_vision_cache": True,
        "same_m6_accepted_data_population": True,
        "same_d_minus_one_acceptance": True,
        "same_joint_full_val_population": True,
        "seed_gates": audits,
        "statistics": {
            "joint_positive_dice_mean": sum(dice) / len(dice),
            "joint_positive_dice_min": min(dice),
            "joint_positive_dice_max": max(dice),
            "absolute_drop_mean": sum(drops) / len(drops),
            "absolute_drop_min": min(drops),
            "absolute_drop_max": max(drops),
        },
        "required_passed": 3,
        "num_passed": passed,
        "passed_all_three": passed == 3,
        "passed": passed == 3,
    }


def validate_m7_retention_seed_gate(
    path_ref: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Deep-recompute a published three-seed M7 retention aggregate."""
    path = resolve_project_path(path_ref) or Path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"M7 retention seed gate 不存在: {path}")
    payload = json_object(path, label="M7 retention seed gate")
    if payload.get("protocol") != M7_RETENTION_SEED_GATE_PROTOCOL:
        raise ValueError("M7 retention seed gate protocol 不兼容")
    audits = payload.get("seed_gates")
    seeds = payload.get("seeds")
    if (
        not isinstance(audits, list)
        or len(audits) != 3
        or not all(isinstance(value, dict) for value in audits)
        or not isinstance(seeds, list)
        or len(seeds) != 3
    ):
        raise ValueError("M7 retention seed gate 缺少完整三种子 bindings")
    gate_paths = [str(value.get("gate") or "") for value in audits]
    rebuilt = aggregate_m7_retention_seed_gates(
        gate_paths,
        seeds=[int(value) for value in seeds],
    )
    if rebuilt != payload:
        raise ValueError(
            "M7 retention seed gate 与绑定单 seed gates 的重新计算结果不一致"
        )
    return path.resolve(strict=False), rebuilt
