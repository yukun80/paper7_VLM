#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 full-val retention 产物绑定与三种子聚合门禁。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from qpsalm_seg.engine.evaluator import (
    SAMPLE_IDENTITY_FIELDS,
    SAMPLE_IDENTITY_PROTOCOL,
    SEGMENTATION_EVAL_MANIFEST_PROTOCOL,
    SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL,
    validate_segmentation_prediction_population,
)
from qpsalm_seg.engine.threshold import normalize_thresholds
from qpsalm_seg.paths import resolve_project_path

from .checkpoint import (
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
    validate_description_stage_lineage,
)
from .json_protocol import strict_json_loads
from .d4_curriculum import (
    D4_FINAL_FRACTION,
    revalidate_saved_d4_final_acceptance,
)
from .d_minus_one import revalidate_saved_d_minus_one_acceptance
from .m6_acceptance import revalidate_saved_m6_acceptance
from .joint_trainer import (
    revalidate_joint_initialization_audit,
    validate_joint_checkpoint_execution,
)
from .vision_cache import revalidate_description_cache_artifact
from .run_artifacts import (
    JOINT_TRAINING_COMPLETION_PROTOCOL,
    validate_checkpoint_run_completion,
)


RETENTION_GATE_PROTOCOL = (
    "qpsalm_segdesc_retention_v22_run_completion_bound"
)
RETENTION_EVAL_BINDING_PROTOCOL = "qpsalm_segdesc_retention_eval_binding_v1"
BASELINE_CHECKPOINT_REPLAY_PROTOCOL = (
    "qpsalm_segdesc_baseline_checkpoint_replay_v2_eval_config_bound"
)
M7_RETENTION_SEED_GATE_PROTOCOL = (
    "qpsalm_segdesc_retention_seed_gate_v18_run_completion_bound"
)
M7_RUN_LOCAL_CONFIG_FIELDS = {
    "seed",
    "output_dir",
    "d4_final_acceptance_gate",
    "m6_acceptance_gate",
}
SEGMENTATION_METRIC_INPUT_POPULATION_PROTOCOL = (
    "qpsalm_segmentation_metric_input_population_v1_target_valid_bound"
)
SEGMENTATION_METRIC_INPUT_FIELDS = (
    "sample_id",
    "parent_sample_id",
    "shape",
    "target_sha256",
    "valid_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不是可读 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须是 object: {path}")
    return payload


def _bound_file(
    path_ref: Any,
    expected_sha256: Any,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(path_ref, str) or not path_ref.strip():
        raise ValueError(f"{label} 缺少 path")
    expected = str(expected_sha256 or "")
    if len(expected) != 64:
        raise ValueError(f"{label} 缺少 SHA-256")
    path = resolve_project_path(path_ref)
    if path is None or not path.is_file():
        raise ValueError(f"{label} 文件不存在: {path_ref}")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 已漂移: expected={expected} observed={observed}"
        )
    return path.resolve(strict=False), observed


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是数值: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须有限: {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 不能是 bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是整数: {value!r}") from exc
    if str(value).strip() not in {str(result), f"+{result}"} and not isinstance(value, int):
        raise ValueError(f"{label} 不是精确整数: {value!r}")
    return result


def _positive_dice(report: dict[str, Any]) -> float:
    return _finite_float(
        (((report.get("metrics") or {}).get("positive_only") or {}).get("dice")),
        label="eval positive Dice",
    )


def _sample_population(report: dict[str, Any]) -> dict[str, Any]:
    value = (report.get("coverage") or {}).get("sample_population") or {}
    if not isinstance(value, dict):
        raise ValueError("eval sample_population 必须是 object")
    return value


def _validate_population(
    population: dict[str, Any],
    *,
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if population.get("protocol") != SAMPLE_IDENTITY_PROTOCOL:
        errors.append("protocol")
    if tuple(population.get("fields") or ()) != tuple(SAMPLE_IDENTITY_FIELDS):
        errors.append("fields")
    if population.get("complete") is not True:
        errors.append("complete")
    if population.get("unique") is not True:
        errors.append("unique")
    if _integer(population.get("num_records", -1), label=f"{label}.num_records") != expected_count:
        errors.append("num_records")
    if _integer(
        population.get("num_unique_sample_ids", -1),
        label=f"{label}.num_unique_sample_ids",
    ) != expected_count:
        errors.append("num_unique_sample_ids")
    population_sha256 = str(population.get("sha256") or "")
    if len(population_sha256) != 64:
        errors.append("sha256")
    if errors:
        raise ValueError(f"{label} population 非法: {errors}")
    return {
        "protocol": SAMPLE_IDENTITY_PROTOCOL,
        "fields": list(SAMPLE_IDENTITY_FIELDS),
        "sha256": population_sha256,
        "num_records": expected_count,
    }


def _validate_eval_report_matches_gate(
    report: dict[str, Any],
    gate: dict[str, Any],
    *,
    prefix: str,
) -> None:
    expected_count = _integer(gate[f"{prefix}_num_samples"], label=f"{prefix}_num_samples")
    observed_count = _integer(
        (report.get("coverage") or {}).get("num_samples", -1),
        label=f"{prefix} report num_samples",
    )
    expected_population = gate[f"{prefix}_sample_population"]
    if not isinstance(expected_population, dict):
        raise ValueError(f"{prefix}_sample_population 必须是 object")
    observed_population = _sample_population(report)
    expected_threshold = _finite_float(gate[f"{prefix}_threshold"], label=f"{prefix}_threshold")
    observed_threshold = _finite_float(report.get("threshold", 0.5), label=f"{prefix} report threshold")
    expected_dice = _finite_float(
        gate[f"{prefix}_positive_dice"], label=f"{prefix}_positive_dice"
    )
    observed_dice = _positive_dice(report)
    prediction_population = validate_segmentation_prediction_population(
        report.get("prediction_population")
    )
    expected_prediction_sha256 = str(
        gate.get(f"{prefix}_prediction_population_sha256") or ""
    )
    if (
        observed_count != expected_count
        or observed_population != expected_population
        or abs(observed_threshold - expected_threshold) > 1.0e-12
        or abs(observed_dice - expected_dice) > 1.0e-12
        or prediction_population.get("sha256")
        != expected_prediction_sha256
        or abs(
            _finite_float(
                prediction_population.get("threshold"),
                label=f"{prefix} prediction population threshold",
            )
            - observed_threshold
        ) > 1.0e-12
    ):
        raise ValueError(f"{prefix} eval report 与 retention gate 内容不一致")


def segmentation_metric_input_population(
    prediction_population: dict[str, Any],
) -> dict[str, Any]:
    """Bind the exact target/valid bytes used to compute segmentation metrics."""

    validated = validate_segmentation_prediction_population(
        prediction_population
    )
    rows = [
        {field: row[field] for field in SEGMENTATION_METRIC_INPUT_FIELDS}
        for row in validated["rows"]
    ]
    rows.sort(key=lambda row: str(row["sample_id"]))
    return {
        "protocol": SEGMENTATION_METRIC_INPUT_POPULATION_PROTOCOL,
        "fields": list(SEGMENTATION_METRIC_INPUT_FIELDS),
        "num_records": len(rows),
        "sha256": _canonical_sha256(rows),
    }


def build_baseline_checkpoint_replay_audit(
    frozen_report: dict[str, Any],
    replay_report: dict[str, Any],
    *,
    baseline_binding: dict[str, Any],
    segmentation_migration: dict[str, Any],
    replay_report_path: str | Path,
) -> dict[str, Any]:
    """Prove the frozen baseline metrics replay from the declared checkpoint."""

    frozen_prediction = validate_segmentation_prediction_population(
        frozen_report.get("prediction_population")
    )
    replay_prediction = validate_segmentation_prediction_population(
        replay_report.get("prediction_population")
    )
    replay_path = resolve_project_path(replay_report_path) or Path(replay_report_path)
    if not replay_path.is_file():
        raise FileNotFoundError(f"baseline replay report 不存在: {replay_path}")
    replay_from_file = _json_object(replay_path, label="baseline replay report")
    if replay_from_file != replay_report:
        raise ValueError("baseline replay 内存结果与原子发布文件不一致")
    source_sha256 = str(segmentation_migration.get("source_sha256") or "")
    bound_checkpoint_sha256 = str(baseline_binding.get("checkpoint_sha256") or "")
    frozen_count = _integer(
        (frozen_report.get("coverage") or {}).get("num_samples", -1),
        label="frozen baseline count",
    )
    replay_count = _integer(
        (replay_report.get("coverage") or {}).get("num_samples", -1),
        label="replayed baseline count",
    )
    frozen_step = _integer(
        frozen_report.get("checkpoint_step", -1), label="frozen baseline step"
    )
    replay_step = _integer(
        replay_report.get("checkpoint_step", -1), label="replayed baseline step"
    )
    bound_step = _integer(
        baseline_binding.get("checkpoint_step", -1), label="bound baseline step"
    )
    migration_step = _integer(
        segmentation_migration.get("source_step", -1),
        label="segmentation migration source step",
    )
    frozen_evidence = {
        "metrics": frozen_report.get("metrics"),
        "metrics_original_size": frozen_report.get("metrics_original_size"),
        "threshold_sweep": frozen_report.get("threshold_sweep"),
    }
    replay_evidence = {
        "metrics": replay_report.get("metrics"),
        "metrics_original_size": replay_report.get("metrics_original_size"),
        "threshold_sweep": replay_report.get("threshold_sweep"),
    }
    checks = {
        "same_segmentation_checkpoint": bool(
            len(source_sha256) == 64
            and source_sha256 == bound_checkpoint_sha256
        ),
        "same_checkpoint_step": (
            frozen_step == replay_step == bound_step == migration_step
        ),
        "same_sample_count": frozen_count > 0 and frozen_count == replay_count,
        "same_sample_population": (
            _sample_population(frozen_report) == _sample_population(replay_report)
        ),
        "same_threshold": abs(
            _finite_float(frozen_report.get("threshold"), label="frozen threshold")
            - _finite_float(replay_report.get("threshold"), label="replay threshold")
        ) <= 1.0e-12,
        "same_prediction_population": (
            frozen_prediction == replay_prediction
        ),
        "same_metric_evidence": frozen_evidence == replay_evidence,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"baseline checkpoint replay 不一致: {failures}")
    return {
        "protocol": BASELINE_CHECKPOINT_REPLAY_PROTOCOL,
        "passed": True,
        "checks": checks,
        "checkpoint_sha256": source_sha256,
        "checkpoint_step": bound_step,
        "segmentation_source_identity": {
            name: segmentation_migration.get(name)
            for name in (
                "source_sha256",
                "source_format",
                "source_step",
                "allowed_prefixes",
            )
        },
        "num_samples": frozen_count,
        "sample_population_sha256": str(
            _sample_population(frozen_report).get("sha256") or ""
        ),
        "prediction_population_sha256": frozen_prediction["sha256"],
        "metric_evidence_sha256": _canonical_sha256(frozen_evidence),
        "frozen_report": {
            "path": str(baseline_binding.get("eval_report") or ""),
            "sha256": str(baseline_binding.get("eval_report_sha256") or ""),
        },
        "replay_report": {
            "path": str(replay_path.resolve(strict=False)),
            "sha256": _sha256_file(replay_path),
            "bytes": int(replay_path.stat().st_size),
        },
    }


def bind_joint_evaluation_report(
    gate: dict[str, Any],
    *,
    eval_report_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Finalize a retention gate only after binding its raw full-val report."""
    if gate.get("protocol") != RETENTION_GATE_PROTOCOL:
        raise ValueError("retention gate protocol 不支持 report binding")
    report_path = resolve_project_path(eval_report_path) or Path(eval_report_path)
    checkpoint = resolve_project_path(checkpoint_path) or Path(checkpoint_path)
    if not report_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("retention report binding 的 report/checkpoint 不存在")
    report = _json_object(report_path, label="joint segmentation eval report")
    _validate_eval_report_matches_gate(report, gate, prefix="joint")
    checkpoint_sha256 = _sha256_file(checkpoint)
    declared_checkpoint_sha256 = str(gate.get("joint_checkpoint_sha256") or "")
    if declared_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("joint checkpoint SHA-256 与 retention gate 不一致")
    finalized = dict(gate)
    finalized["joint_eval_binding"] = {
        "protocol": RETENTION_EVAL_BINDING_PROTOCOL,
        "eval_report": str(report_path.resolve(strict=False)),
        "eval_report_sha256": _sha256_file(report_path),
        "checkpoint": str(checkpoint.resolve(strict=False)),
        "checkpoint_sha256": checkpoint_sha256,
        "split": str(gate.get("split") or ""),
        "sample_population_sha256": str(
            (gate.get("joint_sample_population") or {}).get("sha256") or ""
        ),
    }
    finalized["formal_report_binding_complete"] = True
    finalized["passed"] = bool(
        finalized.get("scientific_gate_eligible") is True
        and finalized.get("preliminary_passed") is True
    )
    return finalized


def _validate_baseline_binding(gate: dict[str, Any]) -> dict[str, Any]:
    binding = gate.get("baseline_binding")
    if not isinstance(binding, dict) or binding.get("valid") is not True:
        raise ValueError("retention gate 缺少有效 baseline_binding")
    if str(binding.get("split") or "") != "val":
        raise ValueError("retention baseline 必须绑定 val split")
    report_path, report_sha256 = _bound_file(
        binding.get("eval_report"),
        binding.get("eval_report_sha256"),
        label="baseline eval report",
    )
    manifest_path, manifest_sha256 = _bound_file(
        binding.get("eval_manifest"),
        binding.get("eval_manifest_sha256"),
        label="baseline eval manifest",
    )
    checkpoint_path, checkpoint_sha256 = _bound_file(
        binding.get("checkpoint"),
        binding.get("checkpoint_sha256"),
        label="baseline checkpoint",
    )
    report = _json_object(report_path, label="baseline eval report")
    prediction_population = validate_segmentation_prediction_population(
        report.get("prediction_population")
    )
    manifest = _json_object(manifest_path, label="baseline eval manifest")
    _validate_eval_report_matches_gate(report, gate, prefix="baseline")
    resolved = dict(manifest.get("resolved_config") or {})
    report_binding = manifest.get("eval_report_binding")
    if not isinstance(report_binding, dict):
        raise ValueError("baseline eval manifest 缺少 report 字节绑定")
    report_threshold = _finite_float(
        report.get("threshold"), label="baseline report threshold"
    )
    resolved_threshold = _finite_float(
        resolved.get("eval_threshold"), label="baseline resolved threshold"
    )
    binding_threshold = _finite_float(
        report_binding.get("eval_threshold"),
        label="baseline manifest report-binding threshold",
    )
    resolved_sweep = normalize_thresholds(resolved.get("threshold_sweep"))
    binding_sweep = normalize_thresholds(
        report_binding.get("threshold_sweep")
    )
    sweep_report = report.get("threshold_sweep") or {}
    observed_sweep_keys = set(
        dict(sweep_report.get("overall_by_threshold") or {})
    ) if isinstance(sweep_report, dict) else set()
    expected_sweep_keys = {f"{value:.2f}" for value in resolved_sweep}
    manifest_checkpoint = resolve_project_path(str(manifest.get("checkpoint") or ""))
    manifest_report_path = resolve_project_path(str(report_binding.get("path") or ""))
    if (
        manifest.get("protocol") != SEGMENTATION_EVAL_MANIFEST_PROTOCOL
        or manifest.get("created_by") != "qpsalm-eval"
        or report_binding.get("protocol")
        != SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL
        or manifest_report_path is None
        or manifest_report_path.resolve(strict=False) != report_path
        or str(report_binding.get("sha256") or "") != report_sha256
        or str(report_binding.get("prediction_population_sha256") or "")
        != str(prediction_population.get("sha256") or "")
        or not (0.0 <= report_threshold <= 1.0)
        or abs(report_threshold - resolved_threshold) > 1.0e-12
        or abs(report_threshold - binding_threshold) > 1.0e-12
        or abs(
            report_threshold
            - _finite_float(
                binding.get("eval_threshold"),
                label="retention baseline binding threshold",
            )
        ) > 1.0e-12
        or abs(
            report_threshold
            - _finite_float(
                prediction_population.get("threshold"),
                label="baseline prediction population threshold",
            )
        ) > 1.0e-12
        or not isinstance(resolved.get("threshold_sweep"), list)
        or not isinstance(report_binding.get("threshold_sweep"), list)
        or not isinstance(report.get("threshold_sweep"), dict)
        or binding_sweep != resolved_sweep
        or binding.get("threshold_sweep") != resolved_sweep
        or observed_sweep_keys != expected_sweep_keys
        or _integer(
            report_binding.get("bytes", -1), label="baseline report bytes"
        ) != int(report_path.stat().st_size)
        or str(manifest.get("split") or "") != "val"
        or manifest_checkpoint is None
        or manifest_checkpoint.resolve(strict=False) != checkpoint_path
        or str(manifest.get("checkpoint_sha256") or "") != checkpoint_sha256
        or _integer(manifest.get("checkpoint_step", -1), label="baseline manifest step")
        != _integer(binding.get("checkpoint_step", -2), label="baseline binding step")
        or _integer(report.get("checkpoint_step", -3), label="baseline report step")
        != _integer(binding.get("checkpoint_step", -2), label="baseline binding step")
        or str(resolved.get("instruction_ablation") or "normal") != "normal"
        or str(resolved.get("visual_ablation") or "normal") != "normal"
    ):
        raise ValueError("baseline eval manifest/report/checkpoint 绑定不一致")
    return {
        "eval_report": str(report_path),
        "eval_report_sha256": report_sha256,
        "eval_report_bytes": int(report_path.stat().st_size),
        "eval_report_manifest_binding": report_binding,
        "prediction_population": {
            key: value
            for key, value in prediction_population.items()
            if key != "rows"
        },
        "eval_manifest": str(manifest_path),
        "eval_manifest_sha256": manifest_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": _integer(
            binding.get("checkpoint_step"), label="baseline checkpoint step"
        ),
        "eval_threshold": report_threshold,
        "threshold_sweep": resolved_sweep,
    }


def validate_m7_retention_gate(
    gate_path: str | Path,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Deep-validate one formal M7 full-val retention artifact."""
    path = resolve_project_path(gate_path) or Path(gate_path)
    if not path.is_file():
        raise ValueError(f"M7 retention gate 不存在: {gate_path}")
    gate = _json_object(path, label="M7 retention gate")
    if gate.get("protocol") != RETENTION_GATE_PROTOCOL:
        raise ValueError(
            f"M7 retention gate protocol 不兼容: {gate.get('protocol')!r}; "
            f"expected={RETENTION_GATE_PROTOCOL!r}"
        )
    seed_values = {
        "CLI seed": expected_seed,
        "gate expected_seed": gate.get("expected_seed"),
        "joint checkpoint seed": gate.get("joint_checkpoint_seed"),
        "checkpoint config seed": (
            (((gate.get("checkpoint_metadata") or {}).get("metadata") or {}).get("config") or {}).get("seed")
        ),
    }
    try:
        normalized_seeds = {
            label: _integer(value, label=label) for label, value in seed_values.items()
        }
    except ValueError as exc:
        raise ValueError(f"M7 retention seed binding 不完整: {seed_values}") from exc
    if len(set(normalized_seeds.values())) != 1 or gate.get("seed_match") is not True:
        raise ValueError(f"M7 retention seed binding 不一致: {seed_values}")
    metadata = dict((gate.get("checkpoint_metadata") or {}).get("metadata") or {})
    checkpoint_config = dict(metadata.get("config") or {})
    try:
        joint_execution_audit = validate_joint_checkpoint_execution(
            metadata,
            checkpoint_step=_integer(
                gate.get("checkpoint_step"), label="joint checkpoint step"
            ),
        )
    except RuntimeError as exc:
        raise ValueError("M7 retention joint execution contract 非法") from exc
    if (
        gate.get("joint_execution_contract_valid") is not True
        or gate.get("joint_execution_audit") != joint_execution_audit
    ):
        raise ValueError("M7 retention gate 未绑定可重算的 joint execution contract")
    protocol_assets = (gate.get("checkpoint_metadata") or {}).get(
        "description_protocol_assets"
    )
    if (
        protocol_assets != description_protocol_assets_spec()
        or gate.get("description_protocol_assets_current") is not True
    ):
        raise ValueError("M7 retention joint checkpoint ontology/schema binding 已漂移")
    checkpoint_path, checkpoint_sha256 = _bound_file(
        gate.get("checkpoint"),
        gate.get("joint_checkpoint_sha256"),
        label="joint checkpoint",
    )
    try:
        checkpoint_provenance = inspect_segdesc_checkpoint(checkpoint_path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError("M7 retention joint checkpoint payload 无法重放") from exc
    if (
        gate.get("checkpoint_metadata")
        != checkpoint_provenance["checkpoint_metadata"]
        or _integer(gate.get("checkpoint_step"), label="joint checkpoint step")
        != checkpoint_provenance["checkpoint_step"]
        or checkpoint_sha256 != checkpoint_provenance["checkpoint_sha256"]
    ):
        raise ValueError("M7 retention joint checkpoint step/metadata 与 payload 不一致")
    try:
        joint_run_completion = validate_checkpoint_run_completion(
            checkpoint_path,
            expected_completion_protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
            expected_stage="joint",
            expected_role="validation_best",
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError("M7 retention joint training run 未成功完成") from exc
    if gate.get("joint_run_completion_audit") != joint_run_completion:
        raise ValueError("M7 retention joint run completion binding 已漂移")
    checkpoint_architecture = dict(
        checkpoint_provenance["checkpoint_metadata"].get(
            "description_architecture_spec"
        ) or {}
    )
    try:
        description_cache_artifact_provenance = (
            revalidate_description_cache_artifact(
                checkpoint_architecture.get(
                    "description_cache_artifact_binding"
                )
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "M7 retention joint checkpoint Description Vision Cache artifact 无法重放"
        ) from exc
    if (
        str(metadata.get("stage") or "") != "joint"
        or metadata.get("checkpoint_role") != "validation_best"
        or gate.get("joint_checkpoint") is not True
    ):
        raise ValueError("M7 retention 只接受 joint validation_best checkpoint")
    if (
        checkpoint_config.get("joint_region_stage") != "predicted_mask"
        or not math.isclose(
            _finite_float(
                checkpoint_config.get("predicted_mask_fraction"),
                label="joint predicted_mask_fraction",
            ),
            D4_FINAL_FRACTION,
            abs_tol=1.0e-12,
        )
        or gate.get("predicted_mask_main_route") is not True
    ):
        raise ValueError("M7 retention 只接受 D4 75% predicted-mask 主路线")
    region_data_audit = metadata.get("region_data_audit")
    if not isinstance(region_data_audit, dict):
        raise ValueError("joint checkpoint 缺少 region_data_audit")
    d4_final_acceptance = revalidate_saved_d4_final_acceptance(
        metadata.get("d4_final_acceptance"),
        seed=expected_seed,
        train_region_data_audit=region_data_audit,
    )
    if (
        gate.get("d4_final_acceptance_audit") != d4_final_acceptance
        or gate.get("d4_final_acceptance_valid") is not True
    ):
        raise ValueError("M7 retention gate 未绑定 joint checkpoint 的 D4 final acceptance")
    m6_acceptance = revalidate_saved_m6_acceptance(
        metadata.get("m6_acceptance"),
        seed=expected_seed,
        train_region_data_audit=region_data_audit,
    )
    if (
        gate.get("m6_acceptance_audit") != m6_acceptance
        or gate.get("m6_acceptance_valid") is not True
        or m6_acceptance.get("d4_final_acceptance") != d4_final_acceptance
    ):
        raise ValueError("M7 retention gate 未绑定 joint checkpoint 的完整 M6 acceptance")
    try:
        joint_initialization_audit = revalidate_joint_initialization_audit(
            metadata.get("joint_initialization_audit"),
            expected_seed=expected_seed,
            region_stage="predicted_mask",
            region_data_audit=region_data_audit,
            d4_final_acceptance=d4_final_acceptance,
            m6_acceptance=m6_acceptance,
            segmentation_migration=dict(
                checkpoint_provenance["checkpoint_metadata"].get(
                    "segmentation_migration"
                ) or {}
            ),
            require_m6_binding=True,
        )
    except RuntimeError as exc:
        raise ValueError(
            "M7 retention joint initialization source 无法重放"
        ) from exc
    if (
        gate.get("joint_initialization_audit")
        != joint_initialization_audit
        or gate.get("joint_initialization_valid") is not True
        or metadata.get("segmentation_migration_lineage")
        != joint_initialization_audit.get("segmentation_migration_lineage")
    ):
        raise ValueError("M7 retention gate 未绑定可重放的 joint initialization source")
    checkpoint_description_benchmark = checkpoint_config.get(
        "description_benchmark"
    )
    if not isinstance(checkpoint_description_benchmark, str) or not (
        checkpoint_description_benchmark.strip()
    ):
        raise ValueError("M7 joint checkpoint 缺少 description_benchmark binding")
    d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
        metadata.get("d_minus_one_acceptance"),
        expected_description_benchmark=checkpoint_description_benchmark,
    )
    if (
        gate.get("d_minus_one_acceptance_audit") != d_minus_one_acceptance
        or gate.get("d_minus_one_acceptance_valid") is not True
        or m6_acceptance.get("d_minus_one_acceptance")
        != d_minus_one_acceptance
    ):
        raise ValueError("M7 retention gate 未绑定 joint checkpoint 的 D-1 acceptance")
    stage_lineage = validate_description_stage_lineage(
        metadata.get("stage_lineage"),
        expected_target_stage="predicted_mask",
    )
    if (
        gate.get("stage_lineage_audit") != stage_lineage
        or gate.get("stage_lineage_valid") is not True
    ):
        raise ValueError("M7 retention gate 未绑定完整 D0-D4 stage lineage")

    binding = gate.get("joint_eval_binding")
    if not isinstance(binding, dict) or binding.get("protocol") != RETENTION_EVAL_BINDING_PROTOCOL:
        raise ValueError("M7 retention gate 缺少 joint eval report binding")
    report_path, report_sha256 = _bound_file(
        binding.get("eval_report"),
        binding.get("eval_report_sha256"),
        label="joint eval report",
    )
    binding_checkpoint, binding_checkpoint_sha256 = _bound_file(
        binding.get("checkpoint"),
        binding.get("checkpoint_sha256"),
        label="joint eval checkpoint",
    )
    if (
        binding_checkpoint != checkpoint_path
        or binding_checkpoint_sha256 != checkpoint_sha256
        or str(binding.get("split") or "") != "val"
        or str(binding.get("sample_population_sha256") or "")
        != str((gate.get("joint_sample_population") or {}).get("sha256") or "")
    ):
        raise ValueError("joint eval binding 与 retention gate 不一致")
    joint_report = _json_object(report_path, label="joint eval report")
    _validate_eval_report_matches_gate(joint_report, gate, prefix="joint")

    baseline_binding = _validate_baseline_binding(gate)
    migration = gate.get("segmentation_migration")
    checkpoint_migration = (gate.get("checkpoint_metadata") or {}).get(
        "segmentation_migration"
    )
    if not isinstance(migration, dict) or migration != checkpoint_migration:
        raise ValueError("joint checkpoint segmentation migration audit 不一致")
    if (
        str(migration.get("source_sha256") or "")
        != baseline_binding["checkpoint_sha256"]
        or gate.get("baseline_source_checkpoint_match") is not True
    ):
        raise ValueError("joint checkpoint 未继承当前 retention baseline")
    replay_audit = gate.get("baseline_checkpoint_replay_audit")
    if not isinstance(replay_audit, dict):
        raise ValueError("M7 retention gate 缺少 baseline checkpoint replay audit")
    replay_report_path = (replay_audit.get("replay_report") or {}).get("path")
    replay_path, _replay_sha256 = _bound_file(
        replay_report_path,
        (replay_audit.get("replay_report") or {}).get("sha256"),
        label="baseline replay report",
    )
    frozen_report = _json_object(
        Path(baseline_binding["eval_report"]), label="baseline eval report"
    )
    replay_report = _json_object(replay_path, label="baseline replay report")
    rebuilt_replay_audit = build_baseline_checkpoint_replay_audit(
        frozen_report,
        replay_report,
        baseline_binding=baseline_binding,
        segmentation_migration=migration,
        replay_report_path=replay_path,
    )
    if (
        replay_audit != rebuilt_replay_audit
        or gate.get("baseline_checkpoint_replayed") is not True
    ):
        raise ValueError("M7 retention baseline checkpoint replay audit 已漂移")

    baseline_count = _integer(gate.get("baseline_num_samples"), label="baseline_num_samples")
    joint_count = _integer(gate.get("joint_num_samples"), label="joint_num_samples")
    if baseline_count <= 0 or joint_count != baseline_count:
        raise ValueError("M7 retention full-val 样本数不一致")
    baseline_population = gate.get("baseline_sample_population")
    joint_population = gate.get("joint_sample_population")
    if not isinstance(baseline_population, dict) or not isinstance(joint_population, dict):
        raise ValueError("M7 retention 缺少 sample population")
    baseline_population_audit = _validate_population(
        baseline_population, expected_count=baseline_count, label="baseline"
    )
    joint_population_audit = _validate_population(
        joint_population, expected_count=joint_count, label="joint"
    )
    if baseline_population != joint_population:
        raise ValueError("M7 retention baseline/joint sample population 不一致")
    baseline_metric_inputs = segmentation_metric_input_population(
        frozen_report.get("prediction_population")
    )
    joint_metric_inputs = segmentation_metric_input_population(
        joint_report.get("prediction_population")
    )
    if (
        baseline_metric_inputs != joint_metric_inputs
        or gate.get("baseline_metric_input_population")
        != baseline_metric_inputs
        or gate.get("joint_metric_input_population") != joint_metric_inputs
        or gate.get("same_metric_input_population") is not True
    ):
        raise ValueError(
            "M7 retention baseline/joint target 或 valid-mask population 不一致"
        )

    baseline_threshold = _finite_float(gate.get("baseline_threshold"), label="baseline_threshold")
    joint_threshold = _finite_float(gate.get("joint_threshold"), label="joint_threshold")
    baseline_dice = _finite_float(gate.get("baseline_positive_dice"), label="baseline_positive_dice")
    joint_dice = _finite_float(gate.get("joint_positive_dice"), label="joint_positive_dice")
    maximum_drop = _finite_float(gate.get("maximum_allowed_drop"), label="maximum_allowed_drop")
    observed_drop = _finite_float(gate.get("absolute_drop"), label="absolute_drop")
    if not (0.0 <= baseline_dice <= 1.0 and 0.0 <= joint_dice <= 1.0):
        raise ValueError("M7 retention positive Dice 超出 [0, 1]")
    if maximum_drop < 0.0 or abs(baseline_threshold - joint_threshold) > 1.0e-12:
        raise ValueError("M7 retention threshold/drop 配置非法")
    recomputed_drop = baseline_dice - joint_dice
    recomputed_preliminary = recomputed_drop <= maximum_drop
    if (
        abs(observed_drop - recomputed_drop) > 1.0e-12
        or gate.get("preliminary_passed") is not recomputed_preliminary
    ):
        raise ValueError("M7 retention Dice drop 计算与 gate 不一致")
    required_true = (
        "full_split_requested",
        "same_population_size",
        "population_protocol_match",
        "population_schema_valid",
        "population_identity_valid",
        "population_counts_valid",
        "same_sample_population",
        "same_metric_input_population",
        "same_threshold",
        "scientific_gate_eligible",
        "formal_report_binding_complete",
        "predicted_mask_main_route",
        "d4_final_acceptance_valid",
        "d_minus_one_acceptance_valid",
        "stage_lineage_valid",
        "m6_acceptance_valid",
        "joint_initialization_valid",
        "description_protocol_assets_current",
        "baseline_checkpoint_replayed",
    )
    missing_true = [name for name in required_true if gate.get(name) is not True]
    if str(gate.get("split") or "") != "val" or missing_true:
        raise ValueError(f"M7 retention 不是可发布的 full-val gate: {missing_true}")
    if gate.get("baseline_comparison_mode") != "frozen_full_report":
        raise ValueError("M7 正式 retention 必须使用 frozen full baseline report")
    recomputed_passed = bool(recomputed_preliminary)
    if gate.get("passed") is not recomputed_passed:
        raise ValueError("M7 retention passed 与绑定证据不一致")

    # 除 seed 外，三条链必须使用完全相同的联合训练配置。
    scientific_config = {
        key: value
        for key, value in checkpoint_config.items()
        if key not in M7_RUN_LOCAL_CONFIG_FIELDS
    }
    return {
        "seed": int(expected_seed),
        "gate": str(path.resolve(strict=False)),
        "gate_sha256": _sha256_file(path),
        "joint_eval_report": str(report_path),
        "joint_eval_report_sha256": report_sha256,
        "joint_checkpoint": str(checkpoint_path),
        "joint_checkpoint_sha256": checkpoint_sha256,
        "joint_checkpoint_step": _integer(
            gate.get("checkpoint_step"), label="joint checkpoint step"
        ),
        "joint_checkpoint_payload_provenance": checkpoint_provenance,
        "description_cache_artifact_provenance": (
            description_cache_artifact_provenance
        ),
        "joint_scientific_config_sha256": _canonical_sha256(scientific_config),
        "joint_execution_audit": joint_execution_audit,
        "joint_training_population_binding": joint_execution_audit[
            "training_population_binding"
        ],
        "m6_acceptance_audit": m6_acceptance,
        "joint_initialization_audit": joint_initialization_audit,
        "d4_final_acceptance_audit": d4_final_acceptance,
        "d_minus_one_acceptance_audit": d_minus_one_acceptance,
        "stage_lineage_audit": stage_lineage,
        "description_protocol_assets": protocol_assets,
        "baseline_binding": baseline_binding,
        "baseline_checkpoint_replay_audit": rebuilt_replay_audit,
        "baseline_population": baseline_population_audit,
        "joint_population": joint_population_audit,
        "metric_input_population": baseline_metric_inputs,
        "baseline_positive_dice": baseline_dice,
        "joint_positive_dice": joint_dice,
        "absolute_drop": recomputed_drop,
        "maximum_allowed_drop": maximum_drop,
        "passed": recomputed_passed,
    }


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
        _canonical_sha256({
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
        _canonical_sha256(item["joint_training_population_binding"])
        for item in audits
    }
    if len(joint_training_population_signatures) != 1:
        raise ValueError(
            "M7 三种子 segmentation/global/region 训练 population 不一致"
        )
    cache_content_signatures = {
        _canonical_sha256({
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
        _canonical_sha256({
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
        _canonical_sha256({
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
        _canonical_sha256(item["d_minus_one_acceptance_audit"])
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
    payload = _json_object(path, label="M7 retention seed gate")
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
