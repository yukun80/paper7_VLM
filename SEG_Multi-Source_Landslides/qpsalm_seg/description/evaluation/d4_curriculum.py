#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D4 predicted-mask curriculum transition and final M7 acceptance gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
)
from .comparison import (
    absolute_candidate_gate,
    validate_m4_region_encoder_suite_gate,
)
from .formal_inputs import (
    formal_seed_binding,
    load_evaluation_rows,
    validate_evaluation_checkpoint_provenance,
    validate_expert_binding,
)
from ..protocols.io import sha256_file as _sha256_file, strict_json_loads
from ..data.engineering_contracts import (
    BRIDGE_ENGINEERING_AUDIT_PROTOCOL,
    REGION_TRAINING_DATA_PROTOCOL,
    revalidate_predicted_index_audit,
)
from ..data.expert_contracts import load_frozen_scientific_gate
from .contracts import DESCRIPTION_EVALUATION_PROTOCOL
from .expert_factuality import EXPERT_FACTUALITY_PROTOCOL


D4_CURRICULUM_GATE_PROTOCOL = (
    "qpsalm_d4_curriculum_gate_v6_strict_json_finite"
)
D4_CURRICULUM_TRANSITIONS = {
    0.0: 0.25,
    0.25: 0.50,
    0.50: 0.75,
}
D4_FINAL_FRACTION = 0.75
D4_CURRICULUM_FRACTIONS = (0.0, 0.25, 0.50, D4_FINAL_FRACTION)
M4_SUITE_ACCEPTANCE_PROTOCOL = (
    "qpsalm_m4_suite_acceptance_v5_strict_json_finite"
)


def _resolved_file(path_ref: Any, *, label: str) -> Path:
    if not isinstance(path_ref, (str, Path)) or not str(path_ref).strip():
        raise ValueError(f"{label} 缺少 path")
    path = resolve_project_path(path_ref) or Path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path.resolve(strict=False)


def _fraction(value: Any, *, allow_zero: bool, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是合法 curriculum fraction: {value!r}") from exc
    allowed = set(D4_CURRICULUM_FRACTIONS)
    if not allow_zero:
        allowed.discard(0.0)
    if not any(math.isclose(normalized, item, abs_tol=1.0e-12) for item in allowed):
        raise ValueError(f"{label} 不在预注册 curriculum 中: {value!r}")
    return next(
        item for item in allowed
        if math.isclose(normalized, item, abs_tol=1.0e-12)
    )


def _load_gate(path_ref: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _resolved_file(path_ref, label="D4 curriculum gate")
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"D4 curriculum gate 不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("D4 curriculum gate 顶层必须是 object")
    if payload.get("protocol") != D4_CURRICULUM_GATE_PROTOCOL:
        raise ValueError("D4 curriculum gate protocol 不兼容")
    return path, payload


def _validate_region_training_audit(
    audit: Any,
    *,
    stage: str,
    expert_gate_audit: dict[str, Any],
    predicted_fraction: float | None,
) -> dict[str, Any]:
    if not isinstance(audit, dict):
        raise ValueError("D4 source/target checkpoint 缺少 region_data_audit")
    population = audit.get("population")
    population_samples = (
        population.get("num_samples") if isinstance(population, dict) else None
    )
    bridge_audit = audit.get("bridge_engineering_audit")
    cache_input = (
        bridge_audit.get("cache_input_fingerprint")
        if isinstance(bridge_audit, dict) else None
    )
    candidate_sha256 = (
        str(bridge_audit.get("candidate_index_sha256") or "")
        if isinstance(bridge_audit, dict) else ""
    )
    if (
        audit.get("protocol") != REGION_TRAINING_DATA_PROTOCOL
        or audit.get("stage") != stage
        or audit.get("expert_gate_audit") != expert_gate_audit
        or not isinstance(bridge_audit, dict)
        or bridge_audit.get("protocol")
        != BRIDGE_ENGINEERING_AUDIT_PROTOCOL
        or bridge_audit.get("status") != "expert_pilot_frozen"
        or bridge_audit.get("expert_truth_used") is not False
        or len(candidate_sha256) != 64
        or candidate_sha256
        != str(expert_gate_audit.get("candidate_index_sha256") or "")
        or not isinstance(cache_input, dict)
        or not str(cache_input.get("benchmark") or "").strip()
        or cache_input.get("index") != "indexes/candidate_all.jsonl"
        or isinstance(cache_input.get("size"), bool)
        or not isinstance(cache_input.get("size"), int)
        or int(cache_input["size"]) <= 0
        or cache_input.get("sha256") != candidate_sha256
        or not isinstance(population, dict)
        or population.get("protocol")
        != "qpsalm_description_dataset_population_v1"
        or population.get("stage") != stage
        or population.get("split") != "train"
        or isinstance(population_samples, bool)
        or not isinstance(population_samples, int)
        or population_samples <= 0
        or len(str(population.get("population_sha256") or "")) != 64
    ):
        raise ValueError(
            "D4 region training data audit 的 stage/population/frozen Bridge/"
            "cache-candidate 绑定非法"
        )
    predicted = audit.get("predicted_index_audit")
    curriculum = audit.get("curriculum_audit")
    if predicted_fraction is None:
        if predicted is not None or curriculum is not None:
            raise ValueError("D3b source checkpoint 不应携带 predicted curriculum audit")
    else:
        expected = _fraction(
            predicted_fraction,
            allow_zero=False,
            label="region audit predicted fraction",
        )
        if not isinstance(predicted, dict) or predicted.get("split") != "train":
            raise ValueError("D4 region training data audit 缺少 OOF train prediction")
        revalidate_predicted_index_audit(
            predicted,
            expected_split="train",
            expert_gate_audit=expert_gate_audit,
        )
        observed_fraction = _fraction(
            curriculum.get("requested_predicted_fraction")
            if isinstance(curriculum, dict) else None,
            allow_zero=False,
            label="region audit requested fraction",
        )
        selection_seed = (
            curriculum.get("selection_seed")
            if isinstance(curriculum, dict) else None
        )
        if (
            not isinstance(curriculum, dict)
            or curriculum.get("protocol")
            != "qpsalm_d4_predicted_mask_curriculum_v1"
            or curriculum.get("training_mix") is not True
            or isinstance(selection_seed, bool)
            or not isinstance(selection_seed, int)
            or selection_seed < 0
            or not math.isclose(
                observed_fraction,
                expected,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("D4 region training data audit 的 curriculum tier 不一致")
    return audit


def build_d4_curriculum_gate(
    *,
    evaluation_dir: str | Path,
    expert_report: str | Path,
    bridge_benchmark: str | Path,
    current_fraction: float,
    next_fraction: float | None,
    seed: int,
    m4_suite_gate: str | Path | None = None,
) -> dict[str, Any]:
    """Build one threshold-bound transition or final-M7 acceptance gate."""
    current = _fraction(
        current_fraction, allow_zero=True, label="current_fraction"
    )
    final_m7 = next_fraction is None
    if final_m7:
        if not math.isclose(current, D4_FINAL_FRACTION, abs_tol=1.0e-12):
            raise ValueError("final M7 acceptance 只允许 current_fraction=0.75")
        normalized_next = None
    else:
        normalized_next = _fraction(
            next_fraction, allow_zero=False, label="next_fraction"
        )
        if not math.isclose(
            D4_CURRICULUM_TRANSITIONS[current],
            normalized_next,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"D4 curriculum 只能顺序升档: current={current} next={normalized_next}"
            )

    root = resolve_project_path(evaluation_dir) or Path(evaluation_dir)
    rows, report = load_evaluation_rows(root, require_complete_generation=True)
    expected_stage = "bridge_expert" if current == 0.0 else "predicted_mask"
    expected_mode = "gt_mask" if current == 0.0 else "fixed_prediction"
    if (
        report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL
        or report.get("stage") != expected_stage
        or report.get("split") != "val"
        or report.get("evaluation_mode") != expected_mode
        or report.get("region_protocol") != "vision_only"
        or not rows
    ):
        raise ValueError(
            "D4 curriculum gate 需要完整 expert val 的 Vision-only GT/fixed evaluation"
        )
    frozen = load_frozen_scientific_gate(
        resolve_project_path(bridge_benchmark) or Path(bridge_benchmark)
    )
    if report.get("expert_gate_audit") != frozen["audit"]:
        raise ValueError("D4 curriculum evaluation 未绑定当前 frozen Bridge gate")
    seed_binding = formal_seed_binding(
        report, expected_seed=seed, label=f"d4_curriculum:{root}"
    )

    expert_path = _resolved_file(expert_report, label="D4 expert factuality report")
    expert = strict_json_loads(expert_path.read_text(encoding="utf-8"))
    if expert.get("protocol") != EXPERT_FACTUALITY_PROTOCOL:
        raise ValueError("D4 curriculum 需要当前 ERFS expert report")
    validate_expert_binding(
        expert,
        root,
        expert_report_path=expert_path,
        label="D4 curriculum",
    )
    if set(expert.get("per_parent_scores") or {}) != {
        str(row["parent_sample_id"]) for row in rows.values()
    }:
        raise ValueError("D4 curriculum ERFS parent population 与 evaluation 不一致")

    checkpoint_metadata = dict(report.get("checkpoint_metadata") or {})
    metadata = dict(checkpoint_metadata.get("metadata") or {})
    checkpoint_config = require_serialized_segdesc_config(
        metadata.get("config"), label="D4 curriculum checkpoint config"
    )
    if str(metadata.get("stage") or "") != expected_stage:
        raise ValueError("D4 curriculum checkpoint stage 与 evaluation 不一致")
    if current > 0.0 and not math.isclose(
        float(serialized_segdesc_config_value(
            checkpoint_config, "predicted_mask_fraction"
        )),
        current,
        abs_tol=1.0e-12,
    ):
        raise ValueError("D4 curriculum checkpoint 保存的 fraction 与当前档不一致")
    source_checkpoint = _resolved_file(
        report.get("checkpoint"), label="D4 source checkpoint"
    )
    source_sha256 = _sha256_file(source_checkpoint)
    if source_sha256 != str(report.get("checkpoint_sha256") or ""):
        raise ValueError("D4 source checkpoint SHA-256 已漂移")
    checkpoint_payload_provenance = validate_evaluation_checkpoint_provenance(
        root, report
    )
    source_region_audit = _validate_region_training_audit(
        metadata.get("region_data_audit"),
        stage=expected_stage,
        expert_gate_audit=frozen["audit"],
        predicted_fraction=None if current == 0.0 else current,
    )
    if current == 0.0:
        if m4_suite_gate is None:
            raise ValueError("D3b -> D4 25% 升档必须提供完整 M4 suite gate")
        suite_path, suite = validate_m4_region_encoder_suite_gate(m4_suite_gate)
        if suite.get("passed") is not True:
            raise ValueError("D3b -> D4 25% 要求 M4 五 baseline suite 全部通过")
        if suite.get("frozen_gate_audit") != frozen["audit"]:
            raise ValueError("M4 suite 与当前 frozen Bridge 不一致")
        suite_seeds = [int(value) for value in (suite.get("seeds") or [])]
        if suite_seeds.count(int(seed)) != 1:
            raise ValueError("M4 suite 不包含当前 D4 seed")
        seed_index = suite_seeds.index(int(seed))
        candidate_hashes = list(
            suite.get("candidate_main_checkpoint_sha256") or []
        )
        if (
            len(candidate_hashes) != len(suite_seeds)
            or candidate_hashes[seed_index] != source_sha256
        ):
            raise ValueError("D3b source checkpoint 不是 M4 suite 验收的 full-MGRR candidate")
        m4_acceptance = {
            "protocol": M4_SUITE_ACCEPTANCE_PROTOCOL,
            "suite_gate": str(suite_path),
            "suite_gate_sha256": _sha256_file(suite_path),
            "seed": int(seed),
            "candidate_checkpoint_sha256": source_sha256,
            "frozen_gate_audit": frozen["audit"],
            "passed": True,
        }
    else:
        inherited = dict(
            (metadata.get("d4_curriculum_transition") or {}).get(
                "m4_suite_acceptance"
            ) or {}
        )
        if (
            inherited.get("protocol") != M4_SUITE_ACCEPTANCE_PROTOCOL
            or inherited.get("passed") is not True
            or inherited.get("frozen_gate_audit") != frozen["audit"]
            or int(inherited.get("seed", -1)) != int(seed)
        ):
            raise ValueError("D4 source checkpoint 未继承当前 seed 的 M4 suite acceptance")
        inherited_path = _resolved_file(
            inherited.get("suite_gate"), label="inherited M4 suite gate"
        )
        if _sha256_file(inherited_path) != inherited.get("suite_gate_sha256"):
            raise ValueError("inherited M4 suite gate SHA-256 已漂移")
        m4_acceptance = inherited
    val_predicted_audit = report.get("predicted_index_audit")
    if current > 0.0 and not isinstance(val_predicted_audit, dict):
        raise ValueError("D4 fixed-prediction evaluation 缺少 val index audit")
    if current > 0.0:
        current_val_audit = revalidate_predicted_index_audit(
            val_predicted_audit,
            expected_split="val",
            expert_gate_audit=frozen["audit"],
        )
        segmentation_migration = dict(
            checkpoint_metadata.get("segmentation_migration") or {}
        )
        if (
            current_val_audit.get("segmentation_checkpoint_sha256")
            != segmentation_migration.get("source_sha256")
        ):
            raise ValueError(
                "D4 fixed prediction masks 与 description checkpoint 的 segmentation source 不一致"
            )
    if current == 0.0 and val_predicted_audit is not None:
        raise ValueError("D3b GT-mask curriculum gate 不应携带 predicted index audit")

    absolute_gate = absolute_candidate_gate(
        report, expert, frozen["thresholds"]
    )
    return {
        "protocol": D4_CURRICULUM_GATE_PROTOCOL,
        "purpose": "m7_acceptance" if final_m7 else "curriculum_transition",
        "current_fraction": current,
        "next_fraction": normalized_next,
        "seed": int(seed),
        "source_stage": expected_stage,
        "source_evaluation_mode": expected_mode,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_seed_binding": seed_binding,
        "source_checkpoint_payload_provenance": checkpoint_payload_provenance,
        "source_train_region_data_audit": source_region_audit,
        "source_val_predicted_index_audit": val_predicted_audit,
        "evaluation_dir": str(root.resolve(strict=False)),
        "evaluation_report": str((root / "eval_report.json").resolve(strict=False)),
        "evaluation_report_sha256": _sha256_file(root / "eval_report.json"),
        "raw_generations": str((root / "raw_generations.jsonl").resolve(strict=False)),
        "raw_generations_sha256": _sha256_file(root / "raw_generations.jsonl"),
        "expert_report": str(expert_path),
        "expert_report_sha256": _sha256_file(expert_path),
        "num_samples": len(rows),
        "num_expert_parents": len(expert.get("per_parent_scores") or {}),
        "frozen_gate_audit": frozen["audit"],
        "absolute_factuality_gate": absolute_gate,
        "m4_suite_acceptance": m4_acceptance,
        "passed": bool(absolute_gate["passed"]),
    }


def _validate_bound_gate_files(gate: dict[str, Any]) -> dict[str, str]:
    bindings = {
        "source_checkpoint": "source_checkpoint_sha256",
        "evaluation_report": "evaluation_report_sha256",
        "raw_generations": "raw_generations_sha256",
        "expert_report": "expert_report_sha256",
    }
    resolved: dict[str, str] = {}
    for path_field, hash_field in bindings.items():
        path = _resolved_file(gate.get(path_field), label=path_field)
        if _sha256_file(path) != str(gate.get(hash_field) or ""):
            raise ValueError(f"D4 curriculum {path_field} SHA-256 已漂移")
        resolved[path_field] = str(path)
    return resolved


def _rebuild_gate_from_bound_sources(
    gate: dict[str, Any],
    *,
    expert_gate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Recompute a gate instead of trusting editable derived fields.

    The current expert data loader has already validated the Bridge package.  Its
    audit names the exact evaluation gate; resolving the Bridge root from that
    path lets us rerun every source/evidence check before training starts.
    """
    evaluation_gate = _resolved_file(
        expert_gate_audit.get("evaluation_gate"),
        label="current frozen Bridge evaluation gate",
    )
    bridge_root = evaluation_gate.parent.parent
    rebuilt = build_d4_curriculum_gate(
        evaluation_dir=gate.get("evaluation_dir"),
        expert_report=gate.get("expert_report"),
        bridge_benchmark=bridge_root,
        current_fraction=gate.get("current_fraction"),
        next_fraction=gate.get("next_fraction"),
        seed=int(gate.get("seed", -1)),
        m4_suite_gate=(
            ((gate.get("m4_suite_acceptance") or {}).get("suite_gate"))
            if float(gate.get("current_fraction", -1.0)) == 0.0 else None
        ),
    )
    if rebuilt != gate:
        raise ValueError(
            "D4 curriculum gate 与其绑定的 eval/ERFS/Bridge 重新计算结果不一致"
        )
    return rebuilt


def validate_d4_curriculum_gate(
    gate_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Reopen and reproduce one published D4 transition/final gate.

    ``passed=false`` is a valid scientific result and remains publishable.  This
    validator only decides whether the JSON is an exact, current derivation of
    its bound evaluation, ERFS, Bridge, checkpoint and optional M4 suite inputs.
    """
    path, gate = _load_gate(gate_path)
    expert_gate_audit = gate.get("frozen_gate_audit")
    if not isinstance(expert_gate_audit, dict):
        raise ValueError("D4 curriculum gate 缺少 frozen Bridge audit")
    rebuilt = _rebuild_gate_from_bound_sources(
        gate,
        expert_gate_audit=expert_gate_audit,
    )
    _validate_bound_gate_files(rebuilt)
    return path, rebuilt


def validate_d4_curriculum_transition(
    gate_path: str | Path,
    *,
    target_fraction: float,
    seed: int,
    initialize_from: str | Path,
    expert_gate_audit: dict[str, Any],
    train_region_data_audit: dict[str, Any],
    val_predicted_index_audit: dict[str, Any],
) -> dict[str, Any]:
    """Authorize exactly one D4 tier using the prior tier's frozen val gate."""
    path, gate = _load_gate(gate_path)
    if gate.get("purpose") != "curriculum_transition" or gate.get("passed") is not True:
        raise ValueError("D4 tier 只能使用 passed curriculum_transition gate")
    current = _fraction(
        gate.get("current_fraction"), allow_zero=True, label="gate current_fraction"
    )
    target = _fraction(target_fraction, allow_zero=False, label="target_fraction")
    if not math.isclose(float(gate.get("next_fraction", -1.0)), target, abs_tol=1.0e-12):
        raise ValueError("D4 curriculum gate 的 next_fraction 与目标档不一致")
    if not math.isclose(D4_CURRICULUM_TRANSITIONS[current], target, abs_tol=1.0e-12):
        raise ValueError("D4 curriculum transition 不是预注册相邻档")
    if int(gate.get("seed", -1)) != int(seed):
        raise ValueError("D4 curriculum gate seed 与目标 run 不一致")
    if gate.get("frozen_gate_audit") != expert_gate_audit:
        raise ValueError("D4 curriculum gate 与当前 frozen Bridge 不一致")
    gate = _rebuild_gate_from_bound_sources(
        gate, expert_gate_audit=expert_gate_audit
    )
    resolved = _validate_bound_gate_files(gate)
    initialization_checkpoint = _resolved_file(
        initialize_from, label="D4 initialize-from checkpoint"
    )
    if (
        str(initialization_checkpoint) != resolved["source_checkpoint"]
        or _sha256_file(initialization_checkpoint)
        != str(gate.get("source_checkpoint_sha256") or "")
    ):
        raise ValueError("D4 initialize-from 不是 curriculum gate 评价的 checkpoint")
    target_train_audit = _validate_region_training_audit(
        train_region_data_audit,
        stage="predicted_mask",
        expert_gate_audit=expert_gate_audit,
        predicted_fraction=target,
    )
    if val_predicted_index_audit.get("split") != "val":
        raise ValueError("D4 target 缺少独立 fixed val prediction audit")
    if current > 0.0:
        source_train_audit = dict(gate["source_train_region_data_audit"])
        if (
            source_train_audit.get("expert_gate_audit")
            != target_train_audit.get("expert_gate_audit")
            or source_train_audit.get("predicted_index_audit")
            != target_train_audit.get("predicted_index_audit")
            or (source_train_audit.get("curriculum_audit") or {}).get(
                "selection_seed"
            )
            != (target_train_audit.get("curriculum_audit") or {}).get(
                "selection_seed"
            )
        ):
            raise ValueError("D4 相邻 tier 使用了不同 OOF train prediction population")
        if gate.get("source_val_predicted_index_audit") != val_predicted_index_audit:
            raise ValueError("D4 相邻 tier 使用了不同 fixed val prediction population")
    return {
        "protocol": D4_CURRICULUM_GATE_PROTOCOL,
        "gate": str(path),
        "gate_sha256": _sha256_file(path),
        "current_fraction": current,
        "target_fraction": target,
        "seed": int(seed),
        "source_checkpoint": resolved["source_checkpoint"],
        "source_checkpoint_sha256": gate["source_checkpoint_sha256"],
        "frozen_gate_audit": expert_gate_audit,
        "source_train_region_data_audit": gate[
            "source_train_region_data_audit"
        ],
        "source_val_predicted_index_audit": gate[
            "source_val_predicted_index_audit"
        ],
        "m4_suite_acceptance": gate["m4_suite_acceptance"],
        "passed": True,
    }


def validate_d4_final_acceptance_for_m7(
    gate_path: str | Path,
    *,
    seed: int,
    initialize_from: str | Path,
    expert_gate_audit: dict[str, Any],
    train_region_data_audit: dict[str, Any],
    val_predicted_index_audit: dict[str, Any],
) -> dict[str, Any]:
    """Require the trained 75%-predicted checkpoint itself to pass before M7."""
    path, gate = _load_gate(gate_path)
    if (
        gate.get("purpose") != "m7_acceptance"
        or gate.get("passed") is not True
        or gate.get("next_fraction") is not None
        or not math.isclose(
            float(gate.get("current_fraction", -1.0)),
            D4_FINAL_FRACTION,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("M7 只接受通过固定 val 的 D4 75% final gate")
    if int(gate.get("seed", -1)) != int(seed):
        raise ValueError("D4 final gate seed 与 M7 run 不一致")
    if gate.get("frozen_gate_audit") != expert_gate_audit:
        raise ValueError("D4 final gate 与当前 frozen Bridge 不一致")
    gate = _rebuild_gate_from_bound_sources(
        gate, expert_gate_audit=expert_gate_audit
    )
    resolved = _validate_bound_gate_files(gate)
    checkpoint = _resolved_file(initialize_from, label="M7 initialize-from checkpoint")
    if (
        str(checkpoint) != resolved["source_checkpoint"]
        or _sha256_file(checkpoint) != str(gate.get("source_checkpoint_sha256") or "")
    ):
        raise ValueError("M7 initialize-from 不是 D4 final gate 评价的 checkpoint")
    if gate.get("source_train_region_data_audit") != train_region_data_audit:
        raise ValueError("M7 region train data 与 D4 final gate 不一致")
    if gate.get("source_val_predicted_index_audit") != val_predicted_index_audit:
        raise ValueError("M7 region val data 与 D4 final gate 不一致")
    return {
        "protocol": D4_CURRICULUM_GATE_PROTOCOL,
        "gate": str(path),
        "gate_sha256": _sha256_file(path),
        "current_fraction": D4_FINAL_FRACTION,
        "seed": int(seed),
        "source_checkpoint": resolved["source_checkpoint"],
        "source_checkpoint_sha256": gate["source_checkpoint_sha256"],
        "frozen_gate_audit": expert_gate_audit,
        "source_train_region_data_audit": gate[
            "source_train_region_data_audit"
        ],
        "source_val_predicted_index_audit": gate[
            "source_val_predicted_index_audit"
        ],
        "m4_suite_acceptance": gate["m4_suite_acceptance"],
        "passed": True,
    }


def revalidate_saved_d4_final_acceptance(
    saved_audit: Any,
    *,
    seed: int,
    train_region_data_audit: dict[str, Any],
) -> dict[str, Any]:
    """Deep-check the M6 acceptance copied into an M7/joint checkpoint."""
    if not isinstance(saved_audit, dict) or saved_audit.get("passed") is not True:
        raise ValueError("joint checkpoint 缺少 passed D4 final acceptance audit")
    path, gate = _load_gate(saved_audit.get("gate"))
    val_audit = gate.get("source_val_predicted_index_audit")
    if not isinstance(val_audit, dict):
        raise ValueError("D4 final acceptance gate 缺少 fixed val prediction audit")
    expert_gate_audit = train_region_data_audit.get("expert_gate_audit")
    if not isinstance(expert_gate_audit, dict):
        raise ValueError("joint region data 缺少 frozen Bridge audit")
    rebuilt = validate_d4_final_acceptance_for_m7(
        path,
        seed=seed,
        initialize_from=saved_audit.get("source_checkpoint") or "",
        expert_gate_audit=expert_gate_audit,
        train_region_data_audit=train_region_data_audit,
        val_predicted_index_audit=val_audit,
    )
    if rebuilt != saved_audit:
        raise ValueError("joint checkpoint 保存的 D4 final acceptance audit 已漂移")
    return rebuilt
