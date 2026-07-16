"""D-1 zero-shot plus mixed-overfit engineering acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT
from qpsalm_seg.paths import resolve_project_path

from .checkpoint import (
    D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
    DESCRIPTION_PROTOCOL_ASSETS,
    SEGMENTATION_STATE_PREFIXES,
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
)
from .json_protocol import strict_json_loads
from .run_artifacts import (
    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
    validate_training_completion_report,
)
from .vision_cache import revalidate_description_cache_artifact
from .zero_shot import (
    DESCRIPTION_BUILDER_VERSION,
    ZERO_SHOT_INPUT_PROTOCOL,
    ZERO_SHOT_PROTOCOL,
    _input_audit,
)


D_MINUS_ONE_GATE_PROTOCOL = (
    "qpsalm_d_minus_one_engineering_gate_v7_training_completion_bound"
)
OVERFIT_PROTOCOL = (
    "qpsalm_d_minus_one_overfit_validation_v5_strict_json_finite"
)
OVERFIT_PROTOCOL_ASSET_SOURCES = {
    "description_ontology": DESCRIPTION_PROTOCOL_ASSETS[0],
    "description_record_schema": DESCRIPTION_PROTOCOL_ASSETS[1],
    "description_output_schema": DESCRIPTION_PROTOCOL_ASSETS[2],
}
OVERFIT_SOURCE_NAMES = {
    "checkpoint", "dataset_summary", "gradient_gate", "raw_generations",
    "resolved_config", "train_history", "trainable_manifest",
    "validation_report", *OVERFIT_PROTOCOL_ASSET_SOURCES,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"D-1 缺少 {label}: {path}")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"D-1 {label} 不是合法 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"D-1 {label} 必须为 JSON object: {path}")
    return value


def _bound_file(path_value: Any, sha_value: Any) -> bool:
    path = resolve_project_path(str(path_value or ""))
    return bool(
        path is not None
        and path.is_file()
        and isinstance(sha_value, str)
        and len(sha_value) == 64
        and _sha256_file(path) == sha_value
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    try:
        rows = [
            strict_json_loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_zero_shot_report(report: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the zero-shot input selection and revalidate all bound files."""
    input_audit = dict(report.get("input_audit") or {})
    model_audit = dict(report.get("model_audit") or {})
    num_samples = int(report.get("num_samples", 0))
    raw_path = resolve_project_path(str(report.get("raw_generations") or ""))
    raw_rows = _read_jsonl(raw_path) if raw_path is not None and raw_path.is_file() else None
    current_input: dict[str, Any] | None = None
    selected_rows: list[dict[str, Any]] | None = None
    try:
        selected_rows, current_input = _input_audit(
            input_audit.get("benchmark_root") or "",
            str(input_audit.get("split") or ""),
            int(input_audit.get("requested_max_samples", 0)),
            int(input_audit.get("sampling_seed", -1)),
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        current_input = None
        selected_rows = None
    model_dir = resolve_project_path(str(model_audit.get("model_dir") or ""))
    metadata_hashes = dict(model_audit.get("metadata_file_sha256") or {})
    model_files_bound = bool(model_dir is not None and model_dir.is_dir())
    if model_files_bound:
        model_files_bound = all(
            _bound_file(model_dir / name, sha256)
            for name, sha256 in metadata_hashes.items()
        ) and "config.json" in metadata_hashes
    raw_ids = (
        [str(row.get("sample_id") or "") for row in raw_rows]
        if raw_rows is not None else []
    )
    selected_ids = (
        [str(row.get("sample_id") or "") for row in selected_rows]
        if selected_rows is not None else []
    )
    reported_checks = dict(report.get("checks") or {})
    checks = {
        "protocol_current": report.get("protocol") == ZERO_SHOT_PROTOCOL,
        "reported_engineering_valid": (
            report.get("status") == "engineering-valid"
            and not (report.get("errors") or [])
            and bool(reported_checks)
            and all(value is True for value in reported_checks.values())
        ),
        "population_is_32_to_64": 32 <= num_samples <= 64,
        "input_protocol_current": input_audit.get("protocol")
        == ZERO_SHOT_INPUT_PROTOCOL,
        "input_builder_current": input_audit.get("builder_version")
        == DESCRIPTION_BUILDER_VERSION,
        "input_rebuild_matches": (
            current_input is not None and input_audit == current_input
        ),
        "raw_generations_bound": _bound_file(
            report.get("raw_generations"), report.get("raw_generations_sha256")
        ),
        "raw_generation_population_matches": (
            raw_rows is not None
            and len(raw_rows) == num_samples
            and raw_ids == selected_ids
            and len(raw_ids) == len(set(raw_ids))
            and all(raw_ids)
            and all(bool(str(row.get("prediction") or "").strip()) for row in raw_rows)
        ),
        "model_metadata_files_bound": model_files_bound,
        "model_metadata_snapshot_matches": (
            bool(metadata_hashes)
            and model_audit.get("metadata_snapshot_sha256")
            == _canonical_sha256(metadata_hashes)
        ),
        "statistics_seed_matches_sampling": (
            report.get("statistics_seed") == input_audit.get("sampling_seed")
        ),
        "no_region_capability_claim": report.get("region_capability_claimed") is False,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "protocol": "qpsalm_d_minus_one_zero_shot_revalidation_v1",
        "status": "engineering-valid" if not errors else "engineering-invalid",
        "checks": checks,
        "errors": errors,
    }


def validate_d_minus_one_overfit_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Deeply revalidate the D-1 overfit subgate from its immutable sources."""
    sampling = dict(report.get("sampling_audit") or {})
    observations = dict(report.get("observations") or {})
    bindings = dict(report.get("source_bindings") or {})
    reported_checks = dict(report.get("checks") or {})
    binding_inventory_current = set(bindings) == OVERFIT_SOURCE_NAMES
    binding_files_current = binding_inventory_current and all(
        isinstance(value, dict)
        and _bound_file(value.get("path"), value.get("sha256"))
        for value in bindings.values()
    )

    def bound_path(name: str) -> Path | None:
        value = dict(bindings.get(name) or {})
        path = resolve_project_path(str(value.get("path") or ""))
        return path if path is not None and path.is_file() else None

    def bound_json(name: str) -> dict[str, Any] | None:
        path = bound_path(name)
        if path is None:
            return None
        try:
            value = strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    history_path = bound_path("train_history")
    history_rows = _read_jsonl(history_path) if history_path is not None else None
    raw_path = bound_path("raw_generations")
    generation_rows = _read_jsonl(raw_path) if raw_path is not None else None
    validation = bound_json("validation_report")
    gradient_gate = bound_json("gradient_gate")
    manifest = bound_json("trainable_manifest")
    resolved_config = bound_json("resolved_config")
    dataset_summary = bound_json("dataset_summary")

    losses = []
    peak_reserved_gib = 0.0
    device_types: set[str] = set()
    if history_rows is not None:
        for row in history_rows:
            loss = row.get("loss")
            peak = row.get("peak_reserved_gib")
            if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
                losses.append(float(loss))
            if isinstance(peak, (int, float)) and math.isfinite(float(peak)):
                peak_reserved_gib = max(peak_reserved_gib, float(peak))
            device_types.add(str(row.get("device_type") or ""))
    categories = sorted({
        str(row.get("d_minus_one_category"))
        for row in (generation_rows or [])
        if row.get("d_minus_one_category")
    })
    parameter_names = [
        str(name)
        for group in ((manifest or {}).get("groups") or [])
        for name in (group.get("parameter_names") or [])
    ]
    migration = dict(observations.get("segmentation_migration") or {})
    reload_audit = dict(observations.get("strict_reload_audit") or {})
    migration_path = resolve_project_path(str(migration.get("source_path") or ""))
    checkpoint_binding = dict(bindings.get("checkpoint") or {})
    raw_binding = dict(bindings.get("raw_generations") or {})
    current_protocol_assets = description_protocol_assets_spec()
    reported_protocol_assets = observations.get("description_protocol_assets")
    current_asset_specs = dict(current_protocol_assets.get("assets") or {})
    protocol_asset_bindings_current = all(
        dict(bindings.get(name) or {}).get("sha256")
        == dict(current_asset_specs.get(reference) or {}).get("sha256")
        and (
            (path := bound_path(name)) is not None
            and int(path.stat().st_size)
            == int(dict(current_asset_specs.get(reference) or {}).get("bytes", -1))
        )
        for name, reference in OVERFIT_PROTOCOL_ASSET_SOURCES.items()
    )
    checkpoint_payload_provenance: dict[str, Any] | None = None
    description_cache_artifact_provenance: dict[str, Any] | None = None
    checkpoint_payload_error: str | None = None
    checkpoint_path = bound_path("checkpoint")
    if checkpoint_path is not None:
        try:
            checkpoint_payload_provenance = inspect_segdesc_checkpoint(
                checkpoint_path
            )
            checkpoint_architecture = dict(
                checkpoint_payload_provenance["checkpoint_metadata"].get(
                    "description_architecture_spec"
                ) or {}
            )
            description_cache_artifact_provenance = (
                revalidate_description_cache_artifact(
                    checkpoint_architecture.get(
                        "description_cache_artifact_binding"
                    )
                )
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            checkpoint_payload_error = f"{type(exc).__name__}: {exc}"
    expected_report_errors = [
        name for name, passed in reported_checks.items() if passed is not True
    ]
    checks = {
        "protocol_current": report.get("protocol") == OVERFIT_PROTOCOL,
        "reported_status_consistent": (
            report.get("status") == "engineering-valid"
            and report.get("overfit_subgate_passed") is True
            and report.get("d_minus_one_complete") is False
            and report.get("pending_external_subgates")
            == ["native_qwen_zero_shot_baseline"]
            and report.get("errors") == expected_report_errors == []
            and bool(reported_checks)
        ),
        "candidate_not_expert": (
            report.get("candidate_supervision_is_expert_truth") is False
            and sampling.get("expert_truth_used") is False
        ),
        "source_binding_inventory_current": binding_inventory_current,
        "source_binding_files_current": binding_files_current,
        "description_protocol_assets_revalidated": (
            reported_protocol_assets == current_protocol_assets
            and protocol_asset_bindings_current
        ),
        "history_recomputed": (
            history_rows is not None
            and len(losses) == len(history_rows)
            and len(losses) >= 2
            and losses[0] == observations.get("initial_logged_loss")
            and losses[-1] == observations.get("final_logged_loss")
            and peak_reserved_gib == observations.get("peak_reserved_gib")
            and losses[-1] < losses[0]
        ),
        "cuda_memory_recomputed": (
            device_types == {"cuda"}
            and observations.get("device_type") == "cuda"
            and 0.0 < peak_reserved_gib <= 24.0
        ),
        "resolved_config_matches": (
            resolved_config is not None
            and resolved_config.get("stage") == "overfit"
            and int(resolved_config.get("batch_size", 0))
            == int(observations.get("batch_size", -1)) > 1
            and int(resolved_config.get("max_steps", 0))
            == int(observations.get("max_steps", -1))
            == int(observations.get("checkpoint_step", -2))
        ),
        "dataset_sampling_matches": (
            dataset_summary is not None
            and dataset_summary.get("d_minus_one_sampling_audit") == sampling
        ),
        "gradient_gate_revalidated": (
            gradient_gate is not None
            and gradient_gate.get("passed") is True
            and gradient_gate.get("all_required_streams_checked") is True
        ),
        "desc_adapter_manifest_revalidated": (
            bool(parameter_names)
            and any("desc_adapter" in name and "lora_" in name for name in parameter_names)
            and not any(
                "lora_A.default" in name or "lora_B.default" in name
                for name in parameter_names
            )
        ),
        "generation_metrics_revalidated": (
            validation is not None
            and dict(validation.get("generation_metrics") or {})
            == dict(observations.get("generation_metrics") or {})
        ),
        "generation_categories_revalidated": (
            generation_rows is not None
            and categories == ["box", "global", "mask", "null"]
            and categories == sorted(observations.get("generated_categories") or [])
        ),
        "checkpoint_binding_matches_observation": (
            checkpoint_binding.get("path") == observations.get("checkpoint")
            and checkpoint_binding.get("sha256")
            == observations.get("checkpoint_sha256")
        ),
        "generation_binding_matches_observation": (
            raw_binding.get("path") == observations.get("raw_generations")
            and raw_binding.get("sha256")
            == observations.get("raw_generations_sha256")
        ),
        "segmentation_migration_revalidated": (
            migration.get("source_format") == CHECKPOINT_FORMAT
            and tuple(migration.get("allowed_prefixes") or ())
            == SEGMENTATION_STATE_PREFIXES
            and migration_path is not None
            and _bound_file(migration_path, migration.get("source_sha256"))
        ),
        "strict_reload_probe_revalidated": (
            reload_audit.get("protocol")
            == "qpsalm_segdesc_strict_reload_probe_v1"
            and reload_audit.get("passed") is True
            and reload_audit.get("checkpoint") == checkpoint_binding.get("path")
            and reload_audit.get("checkpoint_sha256")
            == checkpoint_binding.get("sha256")
            and reload_audit.get("checkpoint_step")
            == observations.get("checkpoint_step")
            and reload_audit.get("before_sha256")
            == reload_audit.get("restored_sha256")
            and reload_audit.get("corrupted_sha256")
            != reload_audit.get("restored_sha256")
            and reload_audit.get("segmentation_migration") == migration
        ),
        "checkpoint_payload_revalidated": (
            checkpoint_payload_error is None
            and checkpoint_payload_provenance
            == observations.get("checkpoint_payload_provenance")
            and checkpoint_payload_provenance is not None
            and checkpoint_payload_provenance.get("checkpoint_sha256")
            == checkpoint_binding.get("sha256")
            and checkpoint_payload_provenance.get("checkpoint_step")
            == observations.get("checkpoint_step")
            and (
                checkpoint_payload_provenance.get("checkpoint_metadata") or {}
            ).get("segmentation_migration") == migration
            and (
                (
                    checkpoint_payload_provenance.get("checkpoint_metadata")
                    or {}
                ).get("metadata") or {}
            ).get("stage") == "overfit"
            and (
                (
                    checkpoint_payload_provenance.get("checkpoint_metadata")
                    or {}
                ).get("metadata") or {}
            ).get("checkpoint_role") == "terminal_last"
            and observations.get("checkpoint_payload_error") is None
        ),
        "description_cache_artifact_revalidated": (
            description_cache_artifact_provenance
            == observations.get("description_cache_artifact_provenance")
            and description_cache_artifact_provenance is not None
            and (
                description_cache_artifact_provenance.get("shard_replay")
                or {}
            ).get("all_verified") is True
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "protocol": "qpsalm_d_minus_one_overfit_revalidation_v1",
        "status": "engineering-valid" if not errors else "engineering-invalid",
        "checks": checks,
        "errors": errors,
    }


def validate_d_minus_one_runs(
    zero_shot_dir: str | Path,
    overfit_dir: str | Path,
) -> dict[str, Any]:
    """Validate two completed runs without rerunning model inference."""
    zero_dir = resolve_project_path(zero_shot_dir) or Path(zero_shot_dir)
    overfit = resolve_project_path(overfit_dir) or Path(overfit_dir)
    zero_path = zero_dir / "eval_report.json"
    overfit_path = overfit / "d_minus_one_overfit_validation.json"
    completion_path = overfit / "training_report.json"
    zero = _load_report(zero_path, "zero-shot eval_report.json")
    mixed = _load_report(overfit_path, "overfit validation report")
    completion: dict[str, Any] | None = None
    completion_error: str | None = None
    try:
        completion = validate_training_completion_report(
            completion_path,
            expected_protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        completion_error = f"{type(exc).__name__}: {exc}"
    zero_input = dict(zero.get("input_audit") or {})
    sampling = dict(mixed.get("sampling_audit") or {})
    observations = dict(mixed.get("observations") or {})
    overfit_bindings = dict(mixed.get("source_bindings") or {})
    completion_artifacts = dict(
        (completion or {}).get("artifacts") or {}
    )
    terminal_audit = dict(
        (completion or {}).get("terminal_checkpoint_audit") or {}
    )

    def completion_matches_overfit_source(
        completion_name: str,
        overfit_name: str,
    ) -> bool:
        completion_binding = dict(
            completion_artifacts.get(completion_name) or {}
        )
        source_binding = dict(overfit_bindings.get(overfit_name) or {})
        completion_bound = resolve_project_path(
            str(completion_binding.get("path") or "")
        )
        source_bound = resolve_project_path(
            str(source_binding.get("path") or "")
        )
        return bool(
            completion_bound is not None
            and source_bound is not None
            and completion_bound.resolve(strict=False)
            == source_bound.resolve(strict=False)
            and completion_binding.get("sha256")
            == source_binding.get("sha256")
        )

    zero_revalidation = validate_zero_shot_report(zero)
    overfit_revalidation = validate_d_minus_one_overfit_report(mixed)
    zero_samples = int(zero.get("num_samples", 0))
    mixed_samples = int(sampling.get("selected_samples", 0))
    checks = {
        "zero_shot_protocol_current": zero.get("protocol") == ZERO_SHOT_PROTOCOL,
        "zero_shot_engineering_valid": (
            zero.get("status") == "engineering-valid"
            and not (zero.get("errors") or [])
            and zero_revalidation.get("status") == "engineering-valid"
        ),
        "zero_shot_population_is_32_to_64": 32 <= zero_samples <= 64,
        "zero_shot_has_no_region_claim": zero.get("region_capability_claimed") is False,
        "zero_shot_raw_generations_bound": _bound_file(
            zero.get("raw_generations"), zero.get("raw_generations_sha256")
        ),
        "zero_shot_description_index_bound": _bound_file(
            zero_input.get("index"), zero_input.get("index_sha256")
        ),
        "zero_shot_description_validation_bound": _bound_file(
            zero_input.get("validation_report"),
            zero_input.get("validation_report_sha256"),
        ),
        "zero_shot_materialized_images_bound": (
            int(zero_input.get("materialized_images", -1)) == zero_samples
            and len(str(
                zero_input.get("materialized_image_population_sha256") or ""
            )) == 64
        ),
        "overfit_protocol_current": mixed.get("protocol") == OVERFIT_PROTOCOL,
        "overfit_engineering_valid": (
            mixed.get("status") == "engineering-valid"
            and mixed.get("overfit_subgate_passed") is True
            and not (mixed.get("errors") or [])
            and overfit_revalidation.get("status") == "engineering-valid"
        ),
        "overfit_training_completion_revalidated": (
            completion is not None and completion_error is None
        ),
        "overfit_training_completion_is_terminal_overfit": (
            completion is not None
            and completion.get("stage") == "overfit"
            and terminal_audit.get("stage") == "overfit"
            and terminal_audit.get("checkpoint_role") == "terminal_last"
            and completion.get("steps") == observations.get("checkpoint_step")
        ),
        "overfit_training_completion_checkpoint_bound": (
            completion_matches_overfit_source("checkpoint_last", "checkpoint")
            and terminal_audit.get("checkpoint_sha256")
            == observations.get("checkpoint_sha256")
        ),
        "overfit_training_completion_sources_bound": all(
            completion_matches_overfit_source(completion_name, source_name)
            for completion_name, source_name in (
                ("dataset_summary", "dataset_summary"),
                ("resolved_config", "resolved_config"),
                ("train_history", "train_history"),
                ("trainable_parameter_manifest", "trainable_manifest"),
            )
        ),
        "overfit_training_completion_validation_bound": (
            isinstance(
                completion_artifacts.get("d_minus_one_overfit_validation"),
                dict,
            )
            and Path(str(
                completion_artifacts[
                    "d_minus_one_overfit_validation"
                ].get("path") or ""
            )).resolve(strict=False) == overfit_path.resolve(strict=False)
            and completion_artifacts[
                "d_minus_one_overfit_validation"
            ].get("sha256") == _sha256_file(overfit_path)
        ),
        "overfit_failure_report_absent": not (
            overfit / "failure_report.json"
        ).exists(),
        "overfit_population_is_32_to_64": 32 <= mixed_samples <= 64,
        "overfit_candidate_not_expert": (
            mixed.get("candidate_supervision_is_expert_truth") is False
            and sampling.get("expert_truth_used") is False
        ),
        "overfit_checkpoint_bound": _bound_file(
            observations.get("checkpoint"), observations.get("checkpoint_sha256")
        ),
        "overfit_raw_generations_bound": _bound_file(
            observations.get("raw_generations"),
            observations.get("raw_generations_sha256"),
        ),
        "overfit_description_index_bound": _bound_file(
            sampling.get("description_index"),
            sampling.get("description_index_sha256"),
        ),
        "overfit_description_validation_bound": _bound_file(
            sampling.get("description_validation_report"),
            sampling.get("description_validation_report_sha256"),
        ),
        "overfit_bridge_index_bound": _bound_file(
            sampling.get("bridge_index"), sampling.get("bridge_index_sha256")
        ),
        "overfit_bridge_validation_bound": _bound_file(
            sampling.get("bridge_validation_report"),
            sampling.get("bridge_validation_report_sha256"),
        ),
        "overfit_description_protocol_assets_current": (
            (mixed.get("observations") or {}).get("description_protocol_assets")
            == description_protocol_assets_spec()
            and overfit_revalidation.get("checks", {}).get(
                "description_protocol_assets_revalidated"
            ) is True
        ),
        "same_description_validation_report": (
            bool(zero_input.get("validation_report_sha256"))
            and zero_input.get("validation_report_sha256")
            == sampling.get("description_validation_report_sha256")
        ),
        "same_sampling_seed": (
            isinstance(zero_input.get("sampling_seed"), int)
            and zero_input.get("sampling_seed") == sampling.get("sampling_seed")
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "protocol": D_MINUS_ONE_GATE_PROTOCOL,
        "status": "engineering-valid" if not errors else "engineering-invalid",
        "d_minus_one_complete": not errors,
        "checks": checks,
        "errors": errors,
        "zero_shot": {
            "report": str(zero_path.resolve(strict=False)),
            "report_sha256": _sha256_file(zero_path),
            "num_samples": zero_samples,
            "caption_token_f1": zero.get("caption_token_f1"),
            "population_sha256": zero_input.get("population_sha256"),
            "materialized_image_population_sha256": zero_input.get(
                "materialized_image_population_sha256"
            ),
            "revalidation": zero_revalidation,
        },
        "overfit": {
            "report": str(overfit_path.resolve(strict=False)),
            "report_sha256": _sha256_file(overfit_path),
            "training_report": str(completion_path.resolve(strict=False)),
            "training_report_sha256": (
                _sha256_file(completion_path)
                if completion_path.is_file()
                else None
            ),
            "training_completion_protocol": (
                completion.get("protocol") if completion is not None else None
            ),
            "training_completion_error": completion_error,
            "terminal_checkpoint_audit": (
                terminal_audit if completion is not None else None
            ),
            "num_samples": mixed_samples,
            "checkpoint_sha256": observations.get("checkpoint_sha256"),
            "bridge_status": sampling.get("bridge_status"),
            "expert_truth_used": False,
            "revalidation": overfit_revalidation,
        },
    }


def validate_d_minus_one_gate(
    path: str | Path,
    *,
    expected_description_benchmark: str | Path | None = None,
) -> dict[str, Any]:
    """Deep-recompute a published D-1 gate and return its lineage binding."""
    gate_path = resolve_project_path(path) or Path(path)
    gate = _load_report(gate_path, "D-1 gate")
    if gate.get("protocol") != D_MINUS_ONE_GATE_PROTOCOL:
        raise ValueError(
            "D-1 gate protocol 不兼容: "
            f"observed={gate.get('protocol')!r} expected={D_MINUS_ONE_GATE_PROTOCOL!r}"
        )
    zero_report = resolve_project_path(
        str((gate.get("zero_shot") or {}).get("report") or "")
    )
    overfit_report = resolve_project_path(
        str((gate.get("overfit") or {}).get("report") or "")
    )
    if (
        zero_report is None
        or not zero_report.is_file()
        or overfit_report is None
        or not overfit_report.is_file()
    ):
        raise ValueError("D-1 gate 缺少已绑定的 zero-shot/overfit report")
    recomputed = validate_d_minus_one_runs(
        zero_report.parent,
        overfit_report.parent,
    )
    if gate != recomputed:
        raise ValueError("D-1 gate 与当前绑定源重新计算结果不一致")
    if (
        gate.get("status") != "engineering-valid"
        or gate.get("d_minus_one_complete") is not True
        or gate.get("errors")
    ):
        raise ValueError("D-1 gate 尚未 engineering-valid")
    zero_payload = _load_report(zero_report, "zero-shot eval_report.json")
    zero_input = dict(zero_payload.get("input_audit") or {})
    description_root = resolve_project_path(
        str(zero_input.get("benchmark_root") or "")
    )
    validation_path = resolve_project_path(
        str(zero_input.get("validation_report") or "")
    )
    if (
        description_root is None
        or not description_root.is_dir()
        or validation_path is None
        or not validation_path.is_file()
        or validation_path
        != (description_root / "reports/validation_report.json").resolve(
            strict=False
        )
        or zero_input.get("builder_version") != DESCRIPTION_BUILDER_VERSION
        or _sha256_file(validation_path)
        != zero_input.get("validation_report_sha256")
    ):
        raise ValueError("D-1 gate 的 Description M1.1 source binding 非法")
    if expected_description_benchmark is not None:
        expected_root = resolve_project_path(expected_description_benchmark)
        if (
            expected_root is None
            or expected_root.resolve(strict=False)
            != description_root.resolve(strict=False)
        ):
            raise ValueError(
                "D-1 gate 与当前 description_benchmark 不是同一 M1.1 source"
            )
    return {
        "protocol": D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
        "passed": True,
        "gate": str(gate_path.resolve(strict=False)),
        "gate_sha256": _sha256_file(gate_path),
        "zero_shot_report": str(zero_report.resolve(strict=False)),
        "zero_shot_report_sha256": _sha256_file(zero_report),
        "overfit_report": str(overfit_report.resolve(strict=False)),
        "overfit_report_sha256": _sha256_file(overfit_report),
        "overfit_training_report": (gate.get("overfit") or {}).get(
            "training_report"
        ),
        "overfit_training_report_sha256": (gate.get("overfit") or {}).get(
            "training_report_sha256"
        ),
        "sampling_seed": (zero_payload.get("input_audit") or {}).get(
            "sampling_seed"
        ),
        "zero_shot_population_sha256": (gate.get("zero_shot") or {}).get(
            "population_sha256"
        ),
        "zero_shot_materialized_image_population_sha256": (
            gate.get("zero_shot") or {}
        ).get("materialized_image_population_sha256"),
        "overfit_checkpoint_sha256": (gate.get("overfit") or {}).get(
            "checkpoint_sha256"
        ),
        "description_source": {
            "protocol": "qpsalm_d_minus_one_description_source_v1",
            "benchmark_root": str(description_root.resolve(strict=False)),
            "builder_version": DESCRIPTION_BUILDER_VERSION,
            "validation_report": str(validation_path.resolve(strict=False)),
            "validation_report_sha256": _sha256_file(validation_path),
        },
    }


def revalidate_saved_d_minus_one_acceptance(
    saved: Any,
    *,
    expected_description_benchmark: str | Path | None = None,
) -> dict[str, Any]:
    """Reject edited/stale D-1 acceptance copied through later checkpoints."""
    if not isinstance(saved, dict) or (
        saved.get("protocol") != D_MINUS_ONE_ACCEPTANCE_PROTOCOL
        or saved.get("passed") is not True
    ):
        raise ValueError("checkpoint 缺少当前 D-1 acceptance")
    current = validate_d_minus_one_gate(
        str(saved.get("gate") or ""),
        expected_description_benchmark=expected_description_benchmark,
    )
    if saved != current:
        raise ValueError("checkpoint 的 D-1 acceptance 与当前 gate 不一致")
    return current


def write_d_minus_one_gate(path: str | Path, report: dict[str, Any]) -> None:
    target = resolve_project_path(path) or Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
