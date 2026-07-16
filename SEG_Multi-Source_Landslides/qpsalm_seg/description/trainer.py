#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-GPU D-1/D0-D4 trainer for segmentation-grounded description."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT
from qpsalm_seg.paths import resolve_project_path

from .checkpoint import (
    SEGMENTATION_STATE_PREFIXES,
    build_description_stage_lineage,
    capture_training_rng_state,
    description_protocol_assets_spec,
    initialize_segdesc_checkpoint,
    inspect_segdesc_checkpoint,
    load_segdesc_checkpoint,
    read_segdesc_checkpoint_step,
    restore_training_rng_state,
    save_segdesc_checkpoint,
    validate_description_stage_lineage,
    validate_segmentation_migration_lineage,
    validate_resume_run_config,
    verify_segdesc_checkpoint_reload,
)
from .common import (
    append_jsonl,
    build_description_dataset,
    build_description_loader,
    description_amp_dtype,
    description_device,
    description_scaler,
    move_description_batch,
    set_description_seed,
    set_loader_epoch,
    validate_predicted_training_indexes,
    validation_split,
    write_json,
)
from .config import SegDescConfig
from .data import REGION_TRAINING_DATA_PROTOCOL
from .evaluator import description_selection_score, evaluate_description
from .json_protocol import strict_json_loads
from .d4_curriculum import validate_d4_curriculum_transition
from .d_minus_one import (
    OVERFIT_PROTOCOL_ASSET_SOURCES,
    OVERFIT_SOURCE_NAMES,
    OVERFIT_PROTOCOL,
    revalidate_saved_d_minus_one_acceptance,
    validate_d_minus_one_gate,
)
from .model import DESCRIPTION_ADAPTER_NAME
from .model import DESCRIPTION_SEQUENCE_PROTOCOL
from .runtime import (
    build_description_optimizer,
    build_segdesc_model,
    description_trainable_parameter_manifest,
)
from .run_artifacts import reconcile_resume_run
from .vision_cache import revalidate_description_cache_artifact


DESCRIPTION_TRAINING_PROGRESS_PROTOCOL = (
    "qpsalm_description_training_progress_v1_loader_cursor_bound"
)
DESCRIPTION_STREAM_CURSOR_PROTOCOL = "qpsalm_description_stream_cursor_v1"
DESCRIPTION_STREAM_BINDING_PROTOCOL = "qpsalm_description_stream_binding_v1"


def _desc_adapter_parameters(model) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if f".{DESCRIPTION_ADAPTER_NAME}." in name and "lora_" in name
    ]


def _gradient_summary(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [value.grad.detach().float() for value in parameters if value.grad is not None]
    return {
        "num_parameters": len(parameters),
        "num_with_grad": len(gradients),
        "num_nonzero": sum(int(torch.count_nonzero(value).item() > 0) for value in gradients),
        "norm_sum": float(sum((value.norm() for value in gradients), start=torch.tensor(0.0, device=gradients[0].device)).cpu()) if gradients else 0.0,
        "all_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
    }


def _description_step_gradient_gate(
    model,
    optimizer: torch.optim.Optimizer,
    *,
    run_stage: str,
    stream_name: str,
    stream_stage: str,
) -> dict[str, Any]:
    named = list(model.named_parameters())

    def summary(predicate) -> dict[str, Any]:
        return _gradient_summary([
            parameter for name, parameter in named if predicate(name)
        ])

    modules = {
        "desc_adapter": summary(
            lambda name: f".{DESCRIPTION_ADAPTER_NAME}." in name and "lora_" in name
        ),
        "description_backbone": summary(
            lambda name: name.startswith("description_backbone.")
        ),
        "mgrr": summary(lambda name: name.startswith("mgrr.")),
        "region_projection": summary(
            lambda name: name.startswith("region_to_hidden.")
        ),
        "global_visual_projection": summary(
            lambda name: name.startswith("description_view_to_hidden.")
        ),
        "alignment": summary(
            lambda name: name.startswith("alignment_text_projection.")
            or name == "alignment_temperature"
        ),
    }
    if run_stage == "bridge_expert" and stream_name == "dior":
        required_nonzero = {
            "desc_adapter", "description_backbone", "mgrr", "alignment",
        }
        required_zero = {"region_projection", "global_visual_projection"}
    elif run_stage == "bridge_expert" and stream_name == "global_caption":
        required_nonzero = {
            "desc_adapter", "global_visual_projection",
        }
        required_zero = {"mgrr", "region_projection", "alignment"}
    elif stream_stage == "dior_alignment":
        required_nonzero = {"description_backbone", "mgrr", "alignment"}
        required_zero = {
            "desc_adapter", "region_projection", "global_visual_projection",
        }
    elif stream_stage in {"mmrs_caption", "rsicap_caption"}:
        required_nonzero = {
            "desc_adapter", "global_visual_projection",
        }
        required_zero = {"mgrr", "region_projection", "alignment"}
    else:
        required_nonzero = {
            "desc_adapter", "description_backbone", "mgrr", "region_projection",
        }
        required_zero = {"alignment"}
    checks = {
        **{
            f"{name}_nonzero": modules[name]["num_nonzero"] > 0
            for name in sorted(required_nonzero)
        },
        **{
            f"{name}_zero": modules[name]["num_nonzero"] == 0
            for name in sorted(required_zero)
        },
        "all_trainable_gradients_finite": all(
            _gradient_summary(list(group["params"]))["all_finite"]
            for group in optimizer.param_groups
        ),
    }
    return {
        "run_stage": run_stage,
        "stream_name": stream_name,
        "stream_stage": stream_stage,
        "required_nonzero": sorted(required_nonzero),
        "required_zero": sorted(required_zero),
        "modules": modules,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _dataset_data_audit(dataset) -> dict[str, Any]:
    """Bind resume/checkpoints to the exact task population and sampling policy."""
    identities = []
    for row in dataset.rows:
        answers = [
            {
                "text_sha256": next((
                    str(source.get("source_text_sha256"))
                    for source in answer.get("source_provenance", [])
                    if source.get("source_text_sha256")
                ), hashlib.sha256(str(answer.get("text") or "").encode()).hexdigest()),
                "caption_quality_weight": answer.get("caption_quality_weight"),
            }
            for answer in row.get("answers", [])
        ]
        identities.append({
            "sample_id": str(row.get("sample_id") or row.get("bridge_record_id") or ""),
            "parent_sample_id": str(row.get("parent_sample_id") or ""),
            "split": str(row.get("split") or dataset.split),
            "task_family": str(row.get("task_family") or ""),
            "source_dataset": str(row.get("source_dataset") or row.get("dataset_name") or ""),
            "target_status": row.get("target_status"),
            "region_id": row.get("region_id"),
            "region_source": row.get("region_source"),
            "region_mask_path": (row.get("region_mask") or {}).get("path"),
            "region_mask_sha256": (row.get("region_mask") or {}).get("sha256"),
            "answers": answers,
            "d_minus_one_category": row.get("_d_minus_one_category"),
            "d_minus_one_item_kind": row.get("_d_minus_one_item_kind"),
            "d_minus_one_target_authority": row.get(
                "_d_minus_one_target_authority"
            ),
        })
    identities.sort(key=lambda value: value["sample_id"])
    sample_ids = [value["sample_id"] for value in identities]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("description data audit 要求非空且唯一 sample_id")
    encoded = json.dumps(
        identities,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "protocol": "qpsalm_description_dataset_population_v1",
        "stage": dataset.stage,
        "split": dataset.split,
        "num_samples": len(identities),
        "num_parents": len({value["parent_sample_id"] for value in identities}),
        "population_sha256": hashlib.sha256(encoded).hexdigest(),
        "tasks": dict(sorted(Counter(value["task_family"] for value in identities).items())),
        "sources": dict(sorted(Counter(value["source_dataset"] for value in identities).items())),
        "caption_sampling_audit": getattr(dataset, "caption_sampling_audit", None),
        "curriculum_audit": getattr(dataset, "curriculum_audit", None),
        "expert_gate_audit": getattr(dataset, "expert_gate_audit", None),
        "bridge_engineering_audit": getattr(
            dataset, "bridge_engineering_audit", None
        ),
        "description_engineering_audit": getattr(
            dataset, "description_engineering_audit", None
        ),
        "predicted_index_audit": getattr(dataset, "predicted_index_audit", None),
        "d_minus_one_sampling_audit": getattr(
            dataset, "d_minus_one_sampling_audit", None
        ),
    }


def build_d_minus_one_overfit_validation(
    *,
    config: SegDescConfig,
    sampling_audit: dict[str, Any] | None,
    history_rows: list[dict[str, Any]],
    gradient_gate: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
    generation_rows: list[dict[str, Any]],
    trainable_manifest: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_step: int,
    device_type: str,
    segmentation_migration: dict[str, Any],
    reload_audit: dict[str, Any],
    source_files: dict[str, Path],
) -> dict[str, Any]:
    """Assemble a machine-auditable D-1 overfit sub-gate.

    This deliberately does not claim the separate native-Qwen zero-shot run is
    complete. Candidate Bridge text is accepted only as engineering overfit
    supervision and is never relabeled as expert truth.
    """
    sampling = dict(sampling_audit or {})
    category_counts = dict(sampling.get("category_counts") or {})
    losses = [
        float(row["loss"])
        for row in history_rows
        if isinstance(row.get("loss"), (int, float))
        and math.isfinite(float(row["loss"]))
    ]
    peak_reserved_gib = max(
        [
            float(row.get("peak_reserved_gib", 0.0))
            for row in history_rows
            if isinstance(row.get("peak_reserved_gib", 0.0), (int, float))
            and math.isfinite(float(row.get("peak_reserved_gib", 0.0)))
        ]
        or [0.0]
    )
    metrics = dict((validation_report or {}).get("generation_metrics") or {})
    generated_categories = sorted({
        str(row.get("d_minus_one_category"))
        for row in generation_rows if row.get("d_minus_one_category")
    })
    parameter_names = [
        str(name)
        for group in trainable_manifest.get("groups", [])
        for name in group.get("parameter_names", [])
    ]
    def file_sha256(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    required_sources = set(OVERFIT_SOURCE_NAMES)
    bound_source_files = dict(source_files)
    for name, reference in OVERFIT_PROTOCOL_ASSET_SOURCES.items():
        asset_path = resolve_project_path(reference)
        if asset_path is None or not asset_path.is_file():
            raise FileNotFoundError(f"D-1 缺少 description protocol asset: {reference}")
        bound_source_files[name] = asset_path
    source_bindings = {
        name: {
            "path": str(path.resolve(strict=False)),
            "sha256": file_sha256(path),
        }
        for name, path in sorted(bound_source_files.items())
    }
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint_payload_provenance: dict[str, Any] | None = None
    description_cache_artifact_provenance: dict[str, Any] | None = None
    checkpoint_payload_error: str | None = None
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
    generation_path = source_files.get("raw_generations")
    generation_sha256 = (
        file_sha256(generation_path) if generation_path is not None else None
    )
    migration_path = resolve_project_path(
        str(segmentation_migration.get("source_path") or "")
    )
    migration_source_bound = bool(
        migration_path is not None
        and migration_path.is_file()
        and file_sha256(migration_path)
        == segmentation_migration.get("source_sha256")
    )
    reload_checkpoint = resolve_project_path(
        str(reload_audit.get("checkpoint") or "")
    )
    runtime_device_types = {
        str(row.get("device_type") or "") for row in history_rows
    }
    protocol_assets = description_protocol_assets_spec()
    protocol_asset_specs = dict(protocol_assets.get("assets") or {})
    checks = {
        "sample_count_is_32_to_64": 32 <= int(
            sampling.get("selected_samples", 0)
        ) <= 64,
        "global_box_mask_null_present": (
            set(category_counts) == {"global", "box", "mask", "null"}
            and all(int(category_counts.get(name, 0)) > 0 for name in category_counts)
        ),
        "candidate_not_expert": (
            sampling.get("expert_truth_used") is False
            and sampling.get("bridge_target_authority")
            == "deterministic_rule_candidate_not_expert"
        ),
        "benchmark_inputs_bound": all(
            isinstance(sampling.get(name), str)
            and len(str(sampling.get(name))) == 64
            for name in (
                "description_index_sha256",
                "description_validation_report_sha256",
                "bridge_index_sha256",
                "bridge_validation_report_sha256",
            )
        ),
        "description_builder_current": sampling.get("description_builder_version")
        == "description_benchmark_m1_v4_answer_trace",
        "description_protocol_assets_bound": all(
            source_bindings[name].get("sha256")
            == dict(protocol_asset_specs.get(reference) or {}).get("sha256")
            for name, reference in OVERFIT_PROTOCOL_ASSET_SOURCES.items()
        ),
        "bridge_builder_current": sampling.get("bridge_builder_version")
        == "landslide_bridge_m2_v7_expert_review_replay_bound",
        "bridge_status_allows_candidate_engineering": sampling.get("bridge_status")
        in {"awaiting_expert_review", "expert_pilot_frozen"},
        "different_native_image_sizes_present": int(
            sampling.get("num_native_source_sizes", 0)
        ) >= 2,
        "batch_size_greater_than_one": int(config.batch_size) > 1,
        "all_steps_completed": int(checkpoint_step) == int(config.max_steps),
        "finite_loss_history": bool(losses) and len(losses) == len(history_rows),
        "loss_decreased": len(losses) >= 2 and losses[-1] < losses[0],
        "stage_gradient_gate_passed": bool((gradient_gate or {}).get("passed")),
        "segmentation_checkpoint_migration_current": (
            segmentation_migration.get("source_format") == CHECKPOINT_FORMAT
            and tuple(segmentation_migration.get("allowed_prefixes") or ())
            == SEGMENTATION_STATE_PREFIXES
        ),
        "segmentation_checkpoint_source_bound": migration_source_bound,
        "strict_checkpoint_reload_passed": (
            reload_audit.get("protocol")
            == "qpsalm_segdesc_strict_reload_probe_v1"
            and reload_audit.get("passed") is True
            and reload_checkpoint is not None
            and reload_checkpoint.resolve(strict=False)
            == checkpoint_path.resolve(strict=False)
            and reload_audit.get("checkpoint_sha256") == checkpoint_sha256
            and int(reload_audit.get("checkpoint_step", -1))
            == int(checkpoint_step)
            and reload_audit.get("before_sha256")
            == reload_audit.get("restored_sha256")
            and reload_audit.get("corrupted_sha256")
            != reload_audit.get("restored_sha256")
            and reload_audit.get("segmentation_migration")
            == segmentation_migration
        ),
        "checkpoint_payload_replayed": (
            checkpoint_payload_provenance is not None
            and checkpoint_payload_error is None
            and checkpoint_payload_provenance.get("checkpoint_sha256")
            == checkpoint_sha256
            and int(
                checkpoint_payload_provenance.get("checkpoint_step", -1)
            ) == int(checkpoint_step)
            and (
                checkpoint_payload_provenance.get("checkpoint_metadata") or {}
            ).get("segmentation_migration") == segmentation_migration
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
        ),
        "description_cache_artifact_replayed": (
            description_cache_artifact_provenance is not None
            and (
                description_cache_artifact_provenance.get("shard_replay")
                or {}
            ).get("all_verified") is True
        ),
        "desc_adapter_only": (
            any("desc_adapter" in name and "lora_" in name for name in parameter_names)
            and not any(
                "lora_A.default" in name or "lora_B.default" in name
                for name in parameter_names
            )
        ),
        "causal_sequence_protocol_bound": (
            DESCRIPTION_SEQUENCE_PROTOCOL
            == "qpsalm_description_causal_v4_stage_separated"
        ),
        "generation_covers_four_categories": set(generated_categories)
        == {"global", "box", "mask", "null"},
        "caption_generation_observed": int(metrics.get("num_caption", 0)) > 0,
        "structured_generation_observed": int(metrics.get("num_structured", 0)) > 0,
        "raw_json_parser_smoke_passed": float(
            metrics.get("raw_json_parse_rate", 0.0)
        ) > 0.0,
        "raw_schema_smoke_passed": float(
            metrics.get("raw_schema_valid_rate", 0.0)
        ) > 0.0,
        "nonempty_summary_smoke_passed": float(
            metrics.get("summary_nonempty_rate", 0.0)
        ) > 0.0,
        "cuda_memory_measurement_observed": (
            str(device_type) == "cuda"
            and runtime_device_types == {"cuda"}
            and peak_reserved_gib > 0.0
        ),
        "peak_reserved_memory_within_24gib": (
            0.0 < peak_reserved_gib <= 24.0
        ),
        "checkpoint_is_bound": checkpoint_sha256 is not None,
        "raw_generations_are_bound": generation_sha256 is not None,
        "all_runtime_sources_are_bound": (
            set(source_bindings) == required_sources
            and all(
                isinstance(value.get("sha256"), str)
                and len(value["sha256"]) == 64
                for value in source_bindings.values()
            )
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    overfit_passed = not errors
    return {
        "protocol": OVERFIT_PROTOCOL,
        "stage": "overfit",
        "status": "engineering-valid" if overfit_passed else "engineering-invalid",
        "overfit_subgate_passed": overfit_passed,
        "d_minus_one_complete": False,
        "pending_external_subgates": ["native_qwen_zero_shot_baseline"],
        "candidate_supervision_is_expert_truth": False,
        "sampling_audit": sampling,
        "source_bindings": source_bindings,
        "checks": checks,
        "errors": errors,
        "observations": {
            "initial_logged_loss": losses[0] if losses else None,
            "final_logged_loss": losses[-1] if losses else None,
            "peak_reserved_gib": peak_reserved_gib,
            "device_type": str(device_type),
            "batch_size": int(config.batch_size),
            "max_steps": int(config.max_steps),
            "generated_categories": generated_categories,
            "generation_metrics": metrics,
            "description_protocol_assets": protocol_assets,
            "segmentation_migration": dict(segmentation_migration),
            "strict_reload_audit": dict(reload_audit),
            "checkpoint": str(checkpoint_path.resolve(strict=False)),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_payload_provenance": checkpoint_payload_provenance,
            "checkpoint_payload_error": checkpoint_payload_error,
            "description_cache_artifact_provenance": (
                description_cache_artifact_provenance
            ),
            "raw_generations": (
                str(generation_path.resolve(strict=False))
                if generation_path is not None else None
            ),
            "raw_generations_sha256": generation_sha256,
        },
    }


def _region_data_audit(dataset) -> dict[str, Any] | None:
    """Return the M7-facing identity of a region-supervision population.

    The complete training audit also contains stream schedules and validation
    populations. M7 compares only frozen expert/prediction bindings against
    its current region loader, so checkpoint this compact contract explicitly.
    """
    if dataset.stage not in {"bridge_auto", "bridge_expert", "predicted_mask"}:
        return None
    population = _dataset_data_audit(dataset)
    return {
        "protocol": REGION_TRAINING_DATA_PROTOCOL,
        "stage": str(dataset.stage),
        "expert_gate_audit": getattr(dataset, "expert_gate_audit", None),
        "bridge_engineering_audit": getattr(
            dataset, "bridge_engineering_audit", None
        ),
        "predicted_index_audit": getattr(dataset, "predicted_index_audit", None),
        "curriculum_audit": getattr(dataset, "curriculum_audit", None),
        "population": {
            "protocol": population["protocol"],
            "stage": population["stage"],
            "split": population["split"],
            "num_samples": population["num_samples"],
            "num_parents": population["num_parents"],
            "population_sha256": population["population_sha256"],
        },
    }


def _train_loss(model, batch: dict[str, Any], config: SegDescConfig) -> tuple[torch.Tensor, dict[str, float]]:
    backbone = model.encode_description_requests(
        batch["requests"],
        include_spatial=config.stage not in {"mmrs_caption", "rsicap_caption"},
    )
    if config.stage == "dior_alignment":
        loss, logits, positive_mask = model.region_alignment_loss(
            backbone,
            batch["region_masks"],
            batch["target_texts"],
            parent_ids=[str(row["parent_sample_id"]) for row in batch["metadata"]],
        )
        accuracy = 0.5 * (
            positive_mask[
                torch.arange(logits.shape[0], device=logits.device), logits.argmax(1)
            ].float().mean()
            + positive_mask[
                logits.argmax(0), torch.arange(logits.shape[0], device=logits.device)
            ].float().mean()
        )
        return loss, {"in_batch_retrieval_r1": float(accuracy.detach().cpu())}
    output = model.describe_from_state(
        backbone,
        batch["region_masks"],
        batch["instructions"],
        target_texts=batch["target_texts"],
        region_valid_mask=backbone.valid_mask,
        protocol=config.region_protocol,
        structured_output=batch["structured_outputs"],
    )
    if output.per_sample_loss is None:
        raise RuntimeError("description forward 未产生 per-sample loss")
    weights = batch["weights"]
    loss = (output.per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {
        "mean_sequence_length": sum(output.sequence_lengths) / max(len(output.sequence_lengths), 1),
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _description_stream_binding(
    name: str,
    stream: dict[str, Any],
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    loader = stream["loader"]
    seed = int(getattr(loader, "_qpsalm_loader_seed", stream["config"].seed))
    set_loader_epoch(loader, 0, loader_seed=seed)
    batches = len(loader)
    if batches <= 0:
        raise RuntimeError(f"description stream={name} 没有可训练 batch")
    sampler = loader.batch_sampler
    binding = {
        "protocol": DESCRIPTION_STREAM_BINDING_PROTOCOL,
        "stream": name,
        "stage": str(stream["config"].stage),
        "dataset_audit_sha256": _canonical_sha256(data_audit),
        "dataset_samples": len(stream["dataset"]),
        "epoch_zero_batches": int(batches),
        "loader_seed": seed,
        "num_workers": int(loader.num_workers),
        "persistent_workers": bool(loader.persistent_workers),
        "batch_sampler": {
            "class": type(sampler).__name__,
            "protocol": getattr(sampler, "protocol", None),
            "batch_size": getattr(sampler, "batch_size", None),
            "seed": getattr(sampler, "seed", None),
            "drop_last": getattr(sampler, "drop_last", None),
        },
    }
    if binding["persistent_workers"]:
        raise RuntimeError(
            f"description stream={name} 必须关闭 persistent_workers 才能重放 cursor"
        )
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def _initial_description_stream_states(
    bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "epoch": 0,
            "batch_in_epoch": 0,
            "total_microbatches": 0,
            "batches_per_epoch": int(binding["epoch_zero_batches"]),
            "completed_epoch_batches": [],
        }
        for name, binding in bindings.items()
    }


def _description_training_progress_payload(
    *,
    step: int,
    stream_pattern: tuple[str, ...],
    grad_accum_steps: int,
    stream_states: dict[str, dict[str, Any]],
    stream_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    streams = set(stream_bindings)
    if not stream_pattern or set(stream_pattern) != streams or set(stream_states) != streams:
        raise RuntimeError("description progress stream pattern/binding/state 集合不一致")
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer_steps = Counter(
        stream_pattern[index % len(stream_pattern)] for index in range(int(step))
    )
    cursors = {}
    for name in sorted(streams):
        state = stream_states[name]
        completed = [int(value) for value in state["completed_epoch_batches"]]
        total = int(state["total_microbatches"])
        expected_total = int(optimizer_steps[name]) * grad_accum_steps
        if (
            len(completed) != int(state["epoch"])
            or any(value <= 0 for value in completed)
            or sum(completed) + int(state["batch_in_epoch"]) != total
            or total != expected_total
            or int(state["batches_per_epoch"]) <= int(state["batch_in_epoch"])
        ):
            raise RuntimeError(f"description stream={name} cursor 与 step 不一致")
        cursors[name] = {
            "protocol": DESCRIPTION_STREAM_CURSOR_PROTOCOL,
            "epoch": int(state["epoch"]),
            "batch_in_epoch": int(state["batch_in_epoch"]),
            "total_microbatches": total,
            "batches_per_epoch": int(state["batches_per_epoch"]),
            "completed_epoch_batches": completed,
            "stream_binding_sha256": stream_bindings[name]["binding_sha256"],
        }
    return {
        "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        "step": int(step),
        "stream_pattern": list(stream_pattern),
        "stream_pattern_sha256": _canonical_sha256(list(stream_pattern)),
        "grad_accum_steps": grad_accum_steps,
        "optimizer_steps": {
            name: int(optimizer_steps[name]) for name in sorted(streams)
        },
        "stream_cursors": cursors,
    }


def _restore_description_training_progress(
    saved: dict[str, Any],
    *,
    checkpoint_step: int,
    required: bool,
    stream_pattern: tuple[str, ...],
    grad_accum_steps: int,
    train_streams: dict[str, dict[str, Any]],
    stream_bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not saved:
        if required:
            raise RuntimeError("description resume checkpoint 缺少 training_progress")
        return _initial_description_stream_states(stream_bindings)
    streams = set(stream_bindings)
    grad_accum_steps = max(1, int(grad_accum_steps))
    if (
        saved.get("protocol") != DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
        or saved.get("step") != int(checkpoint_step)
    ):
        raise RuntimeError("description resume training_progress protocol/step 不一致")
    if (
        tuple(saved.get("stream_pattern") or ()) != stream_pattern
        or saved.get("stream_pattern_sha256")
        != _canonical_sha256(list(stream_pattern))
        or saved.get("grad_accum_steps") != grad_accum_steps
    ):
        raise RuntimeError("description resume stream pattern/grad accumulation 已变化")
    optimizer_steps = saved.get("optimizer_steps")
    cursors = saved.get("stream_cursors")
    if (
        not isinstance(optimizer_steps, dict)
        or not isinstance(cursors, dict)
        or set(optimizer_steps) != streams
        or set(cursors) != streams
        or set(train_streams) != streams
    ):
        raise RuntimeError("description resume stream progress 集合不完整")
    expected_steps = Counter(
        stream_pattern[index % len(stream_pattern)]
        for index in range(int(checkpoint_step))
    )
    restored = {}
    for name in sorted(streams):
        if optimizer_steps[name] != int(expected_steps[name]):
            raise RuntimeError(f"description resume stream={name} optimizer steps 非法")
        cursor = cursors[name]
        if not isinstance(cursor, dict) or cursor.get("protocol") != DESCRIPTION_STREAM_CURSOR_PROTOCOL:
            raise RuntimeError(f"description resume stream={name} cursor protocol 非法")
        if cursor.get("stream_binding_sha256") != stream_bindings[name]["binding_sha256"]:
            raise RuntimeError(f"description resume stream={name} loader/data binding 已变化")
        completed = cursor.get("completed_epoch_batches")
        if not isinstance(completed, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in completed
        ):
            raise RuntimeError(f"description resume stream={name} epoch history 非法")
        integer_fields = (
            "epoch", "batch_in_epoch", "total_microbatches", "batches_per_epoch",
        )
        if any(
            isinstance(cursor.get(field), bool)
            or not isinstance(cursor.get(field), int)
            or int(cursor[field]) < 0
            for field in integer_fields
        ):
            raise RuntimeError(f"description resume stream={name} cursor 字段非法")
        expected_total = int(expected_steps[name]) * grad_accum_steps
        if (
            len(completed) != cursor["epoch"]
            or sum(completed) + cursor["batch_in_epoch"] != cursor["total_microbatches"]
            or cursor["total_microbatches"] != expected_total
            or cursor["batches_per_epoch"] <= cursor["batch_in_epoch"]
        ):
            raise RuntimeError(f"description resume stream={name} cursor 与 step 不一致")
        loader = train_streams[name]["loader"]
        set_loader_epoch(
            loader,
            cursor["epoch"],
            loader_seed=int(stream_bindings[name]["loader_seed"]),
        )
        if len(loader) != cursor["batches_per_epoch"]:
            raise RuntimeError(f"description resume stream={name} 当前 epoch batch 数已变化")
        restored[name] = {
            "epoch": cursor["epoch"],
            "batch_in_epoch": cursor["batch_in_epoch"],
            "total_microbatches": cursor["total_microbatches"],
            "batches_per_epoch": cursor["batches_per_epoch"],
            "completed_epoch_batches": list(completed),
        }
    return restored


def _description_iterator_at_cursor(
    stream: dict[str, Any],
    state: dict[str, Any],
    binding: dict[str, Any],
):
    loader = stream["loader"]
    set_loader_epoch(
        loader,
        int(state["epoch"]),
        loader_seed=int(binding["loader_seed"]),
    )
    if len(loader) != int(state["batches_per_epoch"]):
        raise RuntimeError("description stream 当前 epoch batch 数与 cursor 不一致")
    iterator = iter(loader)
    cursor = int(state["batch_in_epoch"])
    if cursor <= 0:
        return iterator
    rng_state = capture_training_rng_state()
    try:
        for _ in range(cursor):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("description stream cursor 超出当前 epoch") from exc
    finally:
        restore_training_rng_state(rng_state)
    return iterator


def _next_description_stream_batch(
    stream: dict[str, Any],
    iterator,
    state: dict[str, Any],
    binding: dict[str, Any],
):
    if iterator is None:
        iterator = _description_iterator_at_cursor(stream, state, binding)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("description stream 在 batches_per_epoch 前提前耗尽") from exc
    state["total_microbatches"] += 1
    state["batch_in_epoch"] += 1
    if state["batch_in_epoch"] == state["batches_per_epoch"]:
        state["completed_epoch_batches"].append(state["batches_per_epoch"])
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        loader = stream["loader"]
        set_loader_epoch(
            loader,
            state["epoch"],
            loader_seed=int(binding["loader_seed"]),
        )
        state["batches_per_epoch"] = len(loader)
        if state["batches_per_epoch"] <= 0:
            raise RuntimeError("description stream 新 epoch 没有可训练 batch")
        iterator = None
    return batch, iterator


def _load_best(path: Path) -> float:
    if not path.is_file():
        return -math.inf
    try:
        return float(
            strict_json_loads(path.read_text(encoding="utf-8")).get(
                "selection_score", -math.inf
            )
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return -math.inf


def train_description(
    config: SegDescConfig,
    *,
    device_name: str,
    resume: str | None = None,
    initialize_from: str | None = None,
) -> dict[str, Any]:
    if resume and initialize_from:
        raise ValueError("--resume 与 --initialize-from 不能同时使用")
    staged_successors = {
        "rsicap_caption", "dior_alignment", "bridge_auto",
        "bridge_expert", "predicted_mask",
    }
    if config.stage in {"overfit", "mmrs_caption"} and initialize_from:
        raise ValueError(
            f"stage={config.stage} 必须从 segmentation checkpoint 新建，"
            "不能使用 --initialize-from"
        )
    if config.stage in staged_successors and not (resume or initialize_from):
        raise ValueError(
            f"stage={config.stage} 必须使用前一阶段 --initialize-from，"
            "或使用同阶段 --resume"
        )
    predicted_training_indexes = validate_predicted_training_indexes(
        config, stage=config.stage
    )
    if config.stage == "predicted_mask" and not config.d4_curriculum_gate:
        raise ValueError(
            "D4 predicted-mask training 必须提供前一档通过的 --d4-curriculum-gate"
        )
    d_minus_one_acceptance: dict[str, Any] | None = None
    if config.stage == "mmrs_caption":
        d_minus_one_acceptance = validate_d_minus_one_gate(
            str(config.d_minus_one_gate or ""),
            expected_description_benchmark=config.description_benchmark,
        )
    set_description_seed(config.seed)
    device = description_device(device_name)
    output_dir = resolve_project_path(config.output_dir) or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, migration = build_segdesc_model(config, device)
    bank = model.description_backbone.bank
    train_dataset = build_description_dataset(config, bank, split="train", training=True)
    if not len(train_dataset):
        raise RuntimeError(f"description stage={config.stage} 训练集为空")
    if config.stage == "dior_alignment" and config.batch_size < 2:
        raise ValueError("dior_alignment 需要 batch_size >= 2 才能形成对比负样本")
    train_streams = {
        "main": {
            "config": config,
            "dataset": train_dataset,
            "loader": build_description_loader(
                train_dataset,
                config,
                training=True,
                sampler_seed=int(config.seed) + 11_003,
            ),
        }
    }
    stream_pattern = ("main",)
    if config.stage == "bridge_expert":
        # D3b keeps the three supervision types in independent DataLoaders.
        # This avoids mixing contrastive DIOR rows with causal JSON rows in one
        # collate while preserving the documented 60/20/20 task schedule.
        dior_config = replace(config, stage="dior_alignment")
        global_config = replace(config, stage="rsicap_caption")
        dior_dataset = build_description_dataset(
            dior_config, bank, split="train", training=True
        )
        global_dataset = build_description_dataset(
            global_config, bank, split="train", training=True
        )
        if not len(dior_dataset) or not len(global_dataset):
            raise RuntimeError("D3b 需要非空 DIOR 与 global-caption replay 数据")
        train_streams = {
            "bridge": {
                "config": config,
                "dataset": train_dataset,
                "loader": build_description_loader(
                    train_dataset,
                    config,
                    training=True,
                    sampler_seed=int(config.seed) + 11_003,
                ),
            },
            "dior": {
                "config": dior_config,
                "dataset": dior_dataset,
                "loader": build_description_loader(
                    dior_dataset, dior_config, training=True,
                    batch_size=max(2, int(config.batch_size)),
                    sampler_seed=int(config.seed) + 21_013,
                ),
            },
            "global_caption": {
                "config": global_config,
                "dataset": global_dataset,
                "loader": build_description_loader(
                    global_dataset,
                    global_config,
                    training=True,
                    sampler_seed=int(config.seed) + 31_019,
                ),
            },
        }
        stream_pattern = tuple(config.bridge_expert_task_pattern or [
            "bridge", "bridge", "bridge", "dior", "global_caption",
        ])
    val_name = validation_split(config.stage)
    val_loader = None
    validation_config = (
        replace(config, evaluation_mode="fixed_prediction")
        if config.stage == "predicted_mask" else config
    )
    if val_name is not None:
        val_dataset = build_description_dataset(
            validation_config, bank, split=val_name, training=False
        )
        if len(val_dataset):
            val_loader = build_description_loader(
                val_dataset, validation_config, training=False
            )

    training_data_audits = {
        name: _dataset_data_audit(value["dataset"])
        for name, value in train_streams.items()
    }
    stream_loader_bindings = {
        name: _description_stream_binding(
            name, train_streams[name], training_data_audits[name]
        )
        for name in train_streams
    }
    validation_data_audit = (
        _dataset_data_audit(val_loader.dataset) if val_loader is not None else None
    )
    checkpoint_data_audit = {
        "protocol": "qpsalm_description_training_data_binding_v2_loader_bound",
        "training_streams": training_data_audits,
        "stream_loader_bindings": stream_loader_bindings,
        "validation": validation_data_audit,
        "stream_pattern": list(stream_pattern),
    }
    checkpoint_region_data_audit = _region_data_audit(train_dataset)
    validation_predicted_index_audit = (
        getattr(val_loader.dataset, "predicted_index_audit", None)
        if val_loader is not None else None
    )

    optimizer, scheduler = build_description_optimizer(model, config)
    trainable_manifest = description_trainable_parameter_manifest(
        model, optimizer.param_groups, stage=config.stage
    )
    write_json(output_dir / "trainable_parameter_manifest.json", trainable_manifest)
    scaler = description_scaler(config, device)
    start_step = 0
    resume_metadata: dict[str, Any] = {}
    resume_reconciliation: dict[str, Any] | None = None
    d4_curriculum_transition: dict[str, Any] | None = None
    stage_lineage: dict[str, Any] | None = None
    segmentation_migration_lineage = validate_segmentation_migration_lineage(
        migration, {"segmentation_migration": migration}
    )
    resolved = dict(config.__dict__)
    if resume:
        start_step, resume_metadata = load_segdesc_checkpoint(
            resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage=config.stage,
        )
        validate_resume_run_config(resume_metadata, resolved)
        resumed_migration_lineage = validate_segmentation_migration_lineage(
            migration, resume_metadata
        )
        if (
            (resume_metadata.get("metadata") or {}).get(
                "segmentation_migration_lineage"
            ) != resumed_migration_lineage
        ):
            raise RuntimeError(
                "resume checkpoint segmentation migration lineage 已漂移"
            )
        segmentation_migration_lineage = resumed_migration_lineage
        if config.stage != "overfit":
            saved_d_minus_one = revalidate_saved_d_minus_one_acceptance(
                (resume_metadata.get("metadata") or {}).get(
                    "d_minus_one_acceptance"
                ),
                expected_description_benchmark=config.description_benchmark,
            )
            if (
                d_minus_one_acceptance is not None
                and saved_d_minus_one != d_minus_one_acceptance
            ):
                raise RuntimeError("D0 resume 的 D-1 gate 与 checkpoint 不一致")
            d_minus_one_acceptance = saved_d_minus_one
        stage_lineage = (resume_metadata.get("metadata") or {}).get(
            "stage_lineage"
        )
        if config.stage in staged_successors:
            stage_lineage = validate_description_stage_lineage(
                stage_lineage,
                expected_target_stage=config.stage,
            )
        saved_data_audit = (resume_metadata.get("metadata") or {}).get("data_audit")
        if saved_data_audit != checkpoint_data_audit:
            raise RuntimeError(
                "resume checkpoint 的 description data population/sampling policy "
                "与当前运行不一致"
            )
        if config.stage == "predicted_mask":
            saved_transition = dict(
                (resume_metadata.get("metadata") or {}).get(
                    "d4_curriculum_transition"
                ) or {}
            )
            if not saved_transition:
                raise RuntimeError("D4 resume checkpoint 缺少 curriculum transition audit")
            d4_curriculum_transition = validate_d4_curriculum_transition(
                config.d4_curriculum_gate,
                target_fraction=config.predicted_mask_fraction,
                seed=config.seed,
                initialize_from=saved_transition.get("source_checkpoint") or "",
                expert_gate_audit=dict(
                    getattr(train_dataset, "expert_gate_audit", None) or {}
                ),
                train_region_data_audit=dict(checkpoint_region_data_audit or {}),
                val_predicted_index_audit=dict(
                    validation_predicted_index_audit or {}
                ),
            )
            if saved_transition != d4_curriculum_transition:
                raise RuntimeError("D4 resume curriculum gate audit 与 checkpoint 不一致")
    elif initialize_from:
        _source_step, source_metadata = initialize_segdesc_checkpoint(
            initialize_from, model, target_stage=config.stage,
            expected_seed=config.seed,
            allow_same_stage_curriculum=(
                config.stage == "predicted_mask"
                and float(config.predicted_mask_fraction) > 0.25
            ),
        )
        resume_metadata = {
            "initialized_from": str(initialize_from),
            "source": source_metadata,
        }
        segmentation_migration_lineage = validate_segmentation_migration_lineage(
            migration, source_metadata
        )
        if (
            (source_metadata.get("metadata") or {}).get(
                "segmentation_migration_lineage"
            ) != segmentation_migration_lineage
        ):
            raise RuntimeError(
                "initialize-from checkpoint segmentation migration lineage 缺失或漂移"
            )
        d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
            (source_metadata.get("metadata") or {}).get(
                "d_minus_one_acceptance"
            ),
            expected_description_benchmark=config.description_benchmark,
        )
        stage_lineage = build_description_stage_lineage(
            source_metadata, target_stage=config.stage
        )
        if config.stage == "predicted_mask":
            d4_curriculum_transition = validate_d4_curriculum_transition(
                config.d4_curriculum_gate,
                target_fraction=config.predicted_mask_fraction,
                seed=config.seed,
                initialize_from=initialize_from,
                expert_gate_audit=dict(
                    getattr(train_dataset, "expert_gate_audit", None) or {}
                ),
                train_region_data_audit=dict(checkpoint_region_data_audit or {}),
                val_predicted_index_audit=dict(
                    validation_predicted_index_audit or {}
                ),
            )
    grad_accum = max(1, int(config.grad_accum_steps))
    saved_training_progress = (
        dict((resume_metadata.get("metadata") or {}).get("training_progress") or {})
        if resume else {}
    )
    stream_states = _restore_description_training_progress(
        saved_training_progress,
        checkpoint_step=start_step,
        required=bool(resume),
        stream_pattern=stream_pattern,
        grad_accum_steps=grad_accum,
        train_streams=train_streams,
        stream_bindings=stream_loader_bindings,
    )
    if resume:
        resume_reconciliation = reconcile_resume_run(
            output_dir,
            resume_checkpoint=resume,
            checkpoint_step=start_step,
            histories={
                "train_history.jsonl": start_step > 0,
                "validation_history.jsonl": False,
            },
            checkpoint_step_reader=read_segdesc_checkpoint_step,
        )
        # Active progress must describe the restored state, not an uncheckpointed
        # tail that may have been written before the previous process stopped.
        write_json(output_dir / "training_progress_latest.json", saved_training_progress)
    write_json(output_dir / "resolved_config.json", resolved)
    write_json(output_dir / "dataset_summary.json", {
        "stage": config.stage,
        "train_split": "train",
        "train_samples": len(train_dataset),
        "training_streams": {
            name: {
                "stage": value["config"].stage,
                "samples": len(value["dataset"]),
                "batch_size": (
                    value["loader"].batch_size
                    or getattr(value["loader"].batch_sampler, "batch_size", None)
                ),
                "caption_sampling_audit": getattr(
                    value["dataset"], "caption_sampling_audit", None
                ),
                "curriculum_audit": getattr(
                    value["dataset"], "curriculum_audit", None
                ),
                "data_audit": training_data_audits[name],
            }
            for name, value in train_streams.items()
        },
        "expert_gate_audit": getattr(train_dataset, "expert_gate_audit", None),
        "bridge_engineering_audit": getattr(
            train_dataset, "bridge_engineering_audit", None
        ),
        "description_engineering_audit": getattr(
            train_dataset, "description_engineering_audit", None
        ),
        "predicted_index_audit": getattr(train_dataset, "predicted_index_audit", None),
        "predicted_training_indexes": predicted_training_indexes,
        "d4_curriculum_transition": d4_curriculum_transition,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "validation_expert_gate_audit": (
            getattr(val_loader.dataset, "expert_gate_audit", None)
            if val_loader is not None else None
        ),
        "stream_pattern": list(stream_pattern),
        "validation_split": val_name,
        "validation_evaluation_mode": validation_config.evaluation_mode,
        "validation_samples": len(val_loader.dataset) if val_loader is not None else 0,
        "validation_data_audit": validation_data_audit,
        "initialized_from": initialize_from,
        "resume_reconciliation": resume_reconciliation,
        "d_minus_one_sampling_audit": getattr(
            train_dataset, "d_minus_one_sampling_audit", None
        ),
    })
    print(
        f"[DESC-DATA] stage={config.stage} train={len(train_dataset)} "
        f"val={len(val_loader.dataset) if val_loader is not None else 0}"
    )
    print(
        f"[DESC-MODEL] protocol={config.region_protocol} precision={config.amp_dtype} "
        f"batch={config.batch_size} ga={config.grad_accum_steps} max_steps={config.max_steps}"
    )
    best_path = output_dir / "validation_best.json"
    saved_best_score = (resume_metadata.get("metadata") or {}).get("best_score")
    best_score = (
        _load_best(best_path)
        if saved_best_score is None
        else float(saved_best_score)
    )
    history_path = output_dir / "train_history.jsonl"
    validation_history = output_dir / "validation_history.jsonl"
    desc_parameters = _desc_adapter_parameters(model)
    if not desc_parameters:
        raise RuntimeError("description trainer 未找到 desc_adapter LoRA 参数")
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"
    iterators = {name: None for name in train_streams}
    progress = tqdm(total=config.max_steps, initial=start_step, desc="qpsalm-description")
    window_loss = window_samples = 0.0
    window_steps = 0
    window_auxiliary: dict[str, list[float]] = {}
    window_started = time.perf_counter()
    # Resume 后也重新验证每条路径的梯度隔离，避免只信任旧进程内状态。
    checked_gradient_streams: set[str] = set()
    gradient_gate_reports: dict[str, Any] = {}
    last_validation_report: dict[str, Any] | None = None
    step = start_step
    model.train()
    while step < config.max_steps:
        stream_name = stream_pattern[step % len(stream_pattern)]
        stream = train_streams[stream_name]
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_samples = 0
        for _ in range(grad_accum):
            cpu_batch, iterators[stream_name] = _next_description_stream_batch(
                stream,
                iterators[stream_name],
                stream_states[stream_name],
                stream_loader_bindings[stream_name],
            )
            batch = move_description_batch(cpu_batch, device)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=autocast
            ):
                loss, diagnostics = _train_loss(model, batch, stream["config"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"description loss 非有限: step={step}")
            scaler.scale(loss / grad_accum).backward()
            step_loss += float(loss.detach().cpu())
            step_samples += len(batch["metadata"])
            for name, value in diagnostics.items():
                window_auxiliary.setdefault(name, []).append(float(value))
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if stream_name not in checked_gradient_streams:
            gradient_gate = _description_step_gradient_gate(
                model,
                optimizer,
                run_stage=config.stage,
                stream_name=stream_name,
                stream_stage=stream["config"].stage,
            )
            gradient_gate_reports[stream_name] = gradient_gate
            if not gradient_gate["passed"]:
                raise RuntimeError(
                    "description stage-aware 梯度门禁失败；"
                    f"stream={stream_name} report={gradient_gate}"
                )
            checked_gradient_streams.add(stream_name)
            write_json(output_dir / "description_gradient_gate.json", {
                "protocol": "qpsalm_description_gradient_gate_v2_stage_aware",
                "run_stage": config.stage,
                "required_streams": sorted(train_streams),
                "checked_streams": sorted(checked_gradient_streams),
                "all_required_streams_checked": (
                    checked_gradient_streams == set(train_streams)
                ),
                "streams": gradient_gate_reports,
                "passed": checked_gradient_streams == set(train_streams),
            })
        torch.nn.utils.clip_grad_norm_(
            [value for group in optimizer.param_groups for value in group["params"]],
            config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        progress.update(1)
        step_mean = step_loss / grad_accum
        window_loss += step_mean
        window_samples += step_samples
        window_steps += 1

        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            elapsed = time.perf_counter() - window_started
            row = {
                "step": step,
                "epochs": {
                    name: int(state["epoch"])
                    for name, state in stream_states.items()
                },
                "loss": window_loss / max(window_steps, 1),
                "samples_per_second": window_samples / max(elapsed, 1.0e-9),
                "learning_rates": {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups},
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
                "device_type": device.type,
                "device_index": device.index,
                "last_stream": stream_name,
                **{
                    name: sum(values) / len(values)
                    for name, values in window_auxiliary.items()
                },
            }
            append_jsonl(history_path, row)
            training_progress = _description_training_progress_payload(
                step=step,
                stream_pattern=stream_pattern,
                grad_accum_steps=grad_accum,
                stream_states=stream_states,
                stream_bindings=stream_loader_bindings,
            )
            write_json(
                output_dir / "training_progress_latest.json", training_progress
            )
            tqdm.write(
                f"[DESC-TRAIN] step={step} loss={row['loss']:.4f} "
                f"sample_sps={row['samples_per_second']:.2f} peak_gib={row['peak_reserved_gib']:.2f}"
            )
            window_loss = window_samples = 0.0
            window_steps = 0
            window_auxiliary = {}
            window_started = time.perf_counter()

        validation_due = val_loader is not None and (
            step % config.val_interval == 0 or step == config.max_steps
        )
        if validation_due:
            report = evaluate_description(
                model,
                val_loader,
                validation_config,
                device,
                split=str(val_name),
                output_dir=output_dir / "validation_latest",
                run_counterfactuals=False,
            )
            last_validation_report = report
            score = description_selection_score(report, config.stage, config.checkpoint_metric)
            record = {"step": step, "selection_score": score, "report": report}
            append_jsonl(validation_history, record)
            write_json(output_dir / "validation_latest.json", record)
            tqdm.write(f"[DESC-VAL] step={step} score={score:.4f} loss={report['mean_teacher_forced_loss']}")
            if score > best_score:
                best_score = score
                write_json(best_path, record)
                save_segdesc_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    model,
                    step=step,
                    segmentation_migration=migration,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metadata={
                        "stage": config.stage,
                        "checkpoint_role": "validation_best",
                        "best_score": (
                            best_score if math.isfinite(best_score) else None
                        ),
                        "config": resolved,
                        "data_audit": checkpoint_data_audit,
                        "region_data_audit": checkpoint_region_data_audit,
                        "d4_curriculum_transition": d4_curriculum_transition,
                        "d_minus_one_acceptance": d_minus_one_acceptance,
                        "stage_lineage": stage_lineage,
                        "segmentation_migration_lineage": (
                            segmentation_migration_lineage
                        ),
                        "resume_reconciliation": resume_reconciliation,
                        "training_progress": _description_training_progress_payload(
                            step=step,
                            stream_pattern=stream_pattern,
                            grad_accum_steps=grad_accum,
                            stream_states=stream_states,
                            stream_bindings=stream_loader_bindings,
                        ),
                    },
                )
                tqdm.write(f"[DESC-CKPT] saved=checkpoint_best.pt step={step} score={score:.4f}")
            model.train()

        if step % config.save_interval == 0 or step == config.max_steps:
            save_segdesc_checkpoint(
                output_dir / "checkpoint_last.pt",
                model,
                step=step,
                segmentation_migration=migration,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metadata={
                    "stage": config.stage,
                    "checkpoint_role": "terminal_last",
                    "best_score": (
                        best_score if math.isfinite(best_score) else None
                    ),
                    "config": resolved,
                    "data_audit": checkpoint_data_audit,
                    "region_data_audit": checkpoint_region_data_audit,
                    "d4_curriculum_transition": d4_curriculum_transition,
                    "d_minus_one_acceptance": d_minus_one_acceptance,
                    "stage_lineage": stage_lineage,
                    "segmentation_migration_lineage": (
                        segmentation_migration_lineage
                    ),
                    "resume_reconciliation": resume_reconciliation,
                    "training_progress": _description_training_progress_payload(
                        step=step,
                        stream_pattern=stream_pattern,
                        grad_accum_steps=grad_accum,
                        stream_states=stream_states,
                        stream_bindings=stream_loader_bindings,
                    ),
                },
            )
            tqdm.write(f"[DESC-CKPT] saved=checkpoint_last.pt step={step}")
    progress.close()
    d_minus_one_report = None
    if config.stage == "overfit":
        checkpoint_path = output_dir / "checkpoint_last.pt"
        reloaded_step, strict_reload_audit = verify_segdesc_checkpoint_reload(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage="overfit",
        )
        if strict_reload_audit.get("segmentation_migration") != migration:
            raise RuntimeError(
                "D-1 strict reload 返回的 segmentation migration 与 live run 不一致"
            )

        def read_jsonl(path: Path) -> list[dict[str, Any]]:
            if not path.is_file():
                return []
            return [
                strict_json_loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        d_minus_one_report = build_d_minus_one_overfit_validation(
            config=config,
            sampling_audit=getattr(
                train_dataset, "d_minus_one_sampling_audit", None
            ),
            history_rows=read_jsonl(history_path),
            gradient_gate={
                "protocol": "qpsalm_description_gradient_gate_v2_stage_aware",
                "passed": checked_gradient_streams == set(train_streams),
                "streams": gradient_gate_reports,
            },
            validation_report=last_validation_report,
            generation_rows=read_jsonl(
                output_dir / "validation_latest/raw_generations.jsonl"
            ),
            trainable_manifest=trainable_manifest,
            checkpoint_path=checkpoint_path,
            checkpoint_step=reloaded_step,
            device_type=device.type,
            segmentation_migration=migration,
            reload_audit=strict_reload_audit,
            source_files={
                "checkpoint": checkpoint_path,
                "dataset_summary": output_dir / "dataset_summary.json",
                "gradient_gate": output_dir / "description_gradient_gate.json",
                "raw_generations": (
                    output_dir / "validation_latest/raw_generations.jsonl"
                ),
                "resolved_config": output_dir / "resolved_config.json",
                "train_history": history_path,
                "trainable_manifest": (
                    output_dir / "trainable_parameter_manifest.json"
                ),
                "validation_report": (
                    output_dir / "validation_latest/eval_report.json"
                ),
            },
        )
        write_json(
            output_dir / "d_minus_one_overfit_validation.json",
            d_minus_one_report,
        )
    return {
        "output_dir": str(output_dir),
        "stage": config.stage,
        "steps": step,
        "best_score": best_score if math.isfinite(best_score) else None,
        "checkpoint_best": str(output_dir / "checkpoint_best.pt") if (output_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
        "d_minus_one_overfit_validation": d_minus_one_report,
        "d4_curriculum_transition": d4_curriculum_transition,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
        "segmentation_migration_lineage": segmentation_migration_lineage,
        "resume_reconciliation": resume_reconciliation,
    }
