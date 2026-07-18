"""D-1 zero-shot plus mixed-overfit engineering acceptance."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT
from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
)
from ..data.artifact_readiness import (
    ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL,
    revalidate_saved_artifact_readiness_acceptance,
)
from ..data.loaders import DMinusOneTaskPathBatchSampler
from ..training.checkpoint import (
    D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
    SEGMENTATION_STATE_PREFIXES,
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
)
from ..protocols.io import (
    atomic_write_json,
    sha256_file as _sha256_file,
    strict_json_loads,
)
from ..protocols.versions import (
    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
    D_MINUS_ONE_GATE_PROTOCOL,
    D_MINUS_ONE_OVERFIT_PROTOCOL as OVERFIT_PROTOCOL,
    D_MINUS_ONE_OVERFIT_PROTOCOL_ASSET_SOURCES as OVERFIT_PROTOCOL_ASSET_SOURCES,
    D_MINUS_ONE_OVERFIT_SOURCE_NAMES as OVERFIT_SOURCE_NAMES,
)
from ..protocols.gates import (
    audit_causal_label_history,
    d_minus_one_gradient_gate_passed,
    strict_reload_state_replay_passed,
    structured_generation_audits_current,
)
from ..training.run_artifacts import validate_training_completion_report
from ..data.vision_cache import revalidate_description_cache_artifact
from .zero_shot import (
    DESCRIPTION_BUILDER_VERSION,
    ZERO_SHOT_PROTOCOL,
)
from .readiness_replay import replay_overfit_artifact_readiness
from .d_minus_one_contracts import (
    bound_file_matches,
    load_d_minus_one_report,
    read_strict_jsonl,
    validate_zero_shot_report,
)


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
        and bound_file_matches(value.get("path"), value.get("sha256"))
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
    history_rows = read_strict_jsonl(history_path) if history_path is not None else None
    raw_path = bound_path("raw_generations")
    generation_rows = read_strict_jsonl(raw_path) if raw_path is not None else None
    validation = bound_json("validation_report")
    gradient_gate = bound_json("gradient_gate")
    manifest = bound_json("trainable_manifest")
    resolved_config = bound_json("resolved_config")
    dataset_summary = bound_json("dataset_summary")
    try:
        composed_config = require_serialized_segdesc_config(
            resolved_config, label="D-1 resolved_config"
        )
    except (TypeError, ValueError):
        composed_config = None

    readiness_path = bound_path("artifact_readiness_report")
    artifact_readiness_acceptance, artifact_readiness_error = (
        replay_overfit_artifact_readiness(
            observations, composed_config, readiness_path
        )
    )

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
    causal_history_audit = audit_causal_label_history(history_rows)
    categories = sorted({
        str(row.get("d_minus_one_category"))
        for row in (generation_rows or [])
        if row.get("d_minus_one_category")
    })
    generated_route_policy: dict[str, dict[str, bool]] = {}
    inconsistent_generated_routes: set[str] = set()
    for row in generation_rows or []:
        category = str(row.get("d_minus_one_category") or "")
        if not category:
            continue
        route = {
            "structured_output": row.get("structured_output"),
            "use_region_tokens": row.get("use_region_tokens"),
        }
        if any(type(value) is not bool for value in route.values()):
            inconsistent_generated_routes.add(category)
            continue
        previous = generated_route_policy.setdefault(category, route)
        if previous != route:
            inconsistent_generated_routes.add(category)
    expected_generated_route_policy = {
        "global": {
            "structured_output": False,
            "use_region_tokens": False,
        },
        "box": {
            "structured_output": False,
            "use_region_tokens": True,
        },
        "mask": {
            "structured_output": True,
            "use_region_tokens": True,
        },
        "null": {
            "structured_output": True,
            "use_region_tokens": True,
        },
    }
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
    checkpoint_training_metadata = dict(
        (
            (checkpoint_payload_provenance or {}).get("checkpoint_metadata")
            or {}
        ).get("metadata") or {}
    )
    checkpoint_data_audit = dict(
        checkpoint_training_metadata.get("data_audit") or {}
    )
    main_stream_binding = dict(
        (checkpoint_data_audit.get("stream_loader_bindings") or {}).get(
            "main"
        ) or {}
    )
    main_batch_sampler = dict(main_stream_binding.get("batch_sampler") or {})
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
        "artifact_readiness_revalidated": (
            artifact_readiness_error is None
            and artifact_readiness_acceptance is not None
            and artifact_readiness_acceptance.get("protocol")
            == ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL
            and artifact_readiness_acceptance
            == observations.get("artifact_readiness_acceptance")
            and artifact_readiness_acceptance
            == (dataset_summary or {}).get(
                "artifact_readiness_acceptance"
            )
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
        "causal_label_history_revalidated": (
            all(causal_history_audit.values())
            and causal_history_audit
            == observations.get("causal_label_history_audit")
            and reported_checks.get("causal_labels_runtime_replayed") is True
        ),
        "cuda_memory_recomputed": (
            device_types == {"cuda"}
            and observations.get("device_type") == "cuda"
            and 0.0 < peak_reserved_gib <= 24.0
        ),
        "resolved_config_matches": (
            composed_config is not None
            and serialized_segdesc_config_value(
                composed_config, "stage"
            ) == "overfit"
            and int(serialized_segdesc_config_value(
                composed_config, "batch_size"
            ))
            == int(observations.get("batch_size", -1)) > 1
            and int(serialized_segdesc_config_value(
                composed_config, "max_steps"
            ))
            == int(observations.get("max_steps", -1))
            == int(observations.get("checkpoint_step", -2))
            == 100
        ),
        "dataset_sampling_matches": (
            dataset_summary is not None
            and dataset_summary.get("d_minus_one_sampling_audit") == sampling
            and sampling.get("selected_samples") == 64
        ),
        "output_format_region_route_revalidated": (
            sampling.get("category_region_token_policy") == {
                "global": False,
                "box": True,
                "mask": True,
                "null": True,
            }
            and not inconsistent_generated_routes
            and generated_route_policy == expected_generated_route_policy
            and generated_route_policy
            == observations.get("generated_route_policy")
            and reported_checks.get(
                "output_format_region_route_separated"
            ) is True
        ),
        "gradient_window_sampler_revalidated": (
            main_batch_sampler.get("class")
            == DMinusOneTaskPathBatchSampler.__name__
            and main_batch_sampler.get("protocol")
            == DMinusOneTaskPathBatchSampler.protocol
            and composed_config is not None
            and int(main_batch_sampler.get("batch_size", -1))
            == int(serialized_segdesc_config_value(
                composed_config, "batch_size"
            ))
            and int(main_batch_sampler.get("gradient_window_batches", -1))
            == int(serialized_segdesc_config_value(
                composed_config, "grad_accum_steps"
            ))
            and main_batch_sampler.get("drop_last") is False
            and reported_checks.get(
                "gradient_windows_are_task_path_homogeneous"
            ) is True
        ),
        "gradient_gate_revalidated": (
            gradient_gate is not None
            and d_minus_one_gradient_gate_passed(gradient_gate)
            and gradient_gate == observations.get("gradient_gate")
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
        "structured_generation_protocol_revalidated": (
            generation_rows is not None
            and structured_generation_audits_current(generation_rows)
            and reported_checks.get(
                "structured_generation_protocol_bound"
            ) is True
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
            and bound_file_matches(migration_path, migration.get("source_sha256"))
        ),
        "strict_reload_probe_revalidated": (
            strict_reload_state_replay_passed(reload_audit)
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
            and (
                (
                    checkpoint_payload_provenance.get("checkpoint_metadata")
                    or {}
                ).get("metadata") or {}
            ).get("gradient_gate") == gradient_gate
            and observations.get("checkpoint_payload_error") is None
            and (
                (
                    (
                        checkpoint_payload_provenance.get(
                            "checkpoint_metadata"
                        ) or {}
                    ).get("metadata") or {}
                ).get("data_audit") or {}
            ).get("artifact_readiness_acceptance")
            == artifact_readiness_acceptance
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
        "artifact_readiness_error": artifact_readiness_error,
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
    zero = load_d_minus_one_report(zero_path, "zero-shot eval_report.json")
    mixed = load_d_minus_one_report(overfit_path, "overfit validation report")
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
        "zero_shot_population_is_exactly_64": zero_samples == 64,
        "zero_shot_budget_is_exactly_64": (
            zero_input.get("requested_max_samples") == 64
            and zero_input.get("selected_samples") == 64
        ),
        "zero_shot_has_no_region_claim": zero.get("region_capability_claimed") is False,
        "zero_shot_raw_generations_bound": bound_file_matches(
            zero.get("raw_generations"), zero.get("raw_generations_sha256")
        ),
        "zero_shot_description_index_bound": bound_file_matches(
            zero_input.get("index"), zero_input.get("index_sha256")
        ),
        "zero_shot_description_validation_bound": bound_file_matches(
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
        "overfit_population_is_exactly_64": mixed_samples == 64,
        "overfit_budget_is_exactly_100_steps": (
            observations.get("max_steps") == 100
            and observations.get("checkpoint_step") == 100
        ),
        "overfit_candidate_not_expert": (
            mixed.get("candidate_supervision_is_expert_truth") is False
            and sampling.get("expert_truth_used") is False
        ),
        "overfit_checkpoint_bound": bound_file_matches(
            observations.get("checkpoint"), observations.get("checkpoint_sha256")
        ),
        "overfit_raw_generations_bound": bound_file_matches(
            observations.get("raw_generations"),
            observations.get("raw_generations_sha256"),
        ),
        "overfit_description_index_bound": bound_file_matches(
            sampling.get("description_index"),
            sampling.get("description_index_sha256"),
        ),
        "overfit_description_validation_bound": bound_file_matches(
            sampling.get("description_validation_report"),
            sampling.get("description_validation_report_sha256"),
        ),
        "overfit_bridge_index_bound": bound_file_matches(
            sampling.get("bridge_index"), sampling.get("bridge_index_sha256")
        ),
        "overfit_bridge_validation_bound": bound_file_matches(
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
            "artifact_readiness_acceptance": observations.get(
                "artifact_readiness_acceptance"
            ),
            "expert_truth_used": False,
            "revalidation": overfit_revalidation,
        },
    }


def validate_d_minus_one_gate(
    path: str | Path,
    *,
    expected_description_benchmark: str | Path | None = None,
    expected_bridge_benchmark: str | Path | None = None,
    expected_unified_benchmark: str | Path | None = None,
    expected_description_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Deep-recompute a published D-1 gate and return its lineage binding."""
    gate_path = resolve_project_path(path) or Path(path)
    gate = load_d_minus_one_report(gate_path, "D-1 gate")
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
    zero_payload = load_d_minus_one_report(
        zero_report, "zero-shot eval_report.json"
    )
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
    expected_artifact_inputs = (
        expected_bridge_benchmark,
        expected_unified_benchmark,
        expected_description_cache,
    )
    if any(value is not None for value in expected_artifact_inputs):
        if (
            expected_description_benchmark is None
            or any(value is None for value in expected_artifact_inputs)
        ):
            raise ValueError(
                "D-1 artifact readiness 当前输入必须完整提供"
            )
        revalidate_saved_artifact_readiness_acceptance(
            (gate.get("overfit") or {}).get(
                "artifact_readiness_acceptance"
            ),
            expected_description_benchmark=expected_description_benchmark,
            expected_bridge_benchmark=expected_bridge_benchmark,
            expected_unified_benchmark=expected_unified_benchmark,
            expected_description_cache=expected_description_cache,
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
        "artifact_readiness_acceptance": (gate.get("overfit") or {}).get(
            "artifact_readiness_acceptance"
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
    expected_bridge_benchmark: str | Path | None = None,
    expected_unified_benchmark: str | Path | None = None,
    expected_description_cache: str | Path | None = None,
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
        expected_bridge_benchmark=expected_bridge_benchmark,
        expected_unified_benchmark=expected_unified_benchmark,
        expected_description_cache=expected_description_cache,
    )
    if saved != current:
        raise ValueError("checkpoint 的 D-1 acceptance 与当前 gate 不一致")
    return current


def write_d_minus_one_gate(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)
