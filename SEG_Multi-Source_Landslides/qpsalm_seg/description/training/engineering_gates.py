"""D-1 engineering gates, dataset identity and task loss contracts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT
from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import SegDescConfig
from ..data.artifact_readiness import (
    ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL,
)
from ..data.loaders import DMinusOneTaskPathBatchSampler
from ..data.engineering_contracts import REGION_TRAINING_DATA_PROTOCOL
from ..data.vision_cache import revalidate_description_cache_artifact
from ..modeling.model import DESCRIPTION_ADAPTER_NAME
from ..protocols.gates import (
    audit_causal_label_history,
    d_minus_one_gradient_gate_passed,
    strict_reload_state_replay_passed,
    structured_generation_audits_current,
)
from ..protocols.io import sha256_file
from ..protocols.stages import get_stage_spec
from ..protocols.versions import (
    DESCRIPTION_SEQUENCE_PROTOCOL,
    DESCRIPTION_GRADIENT_GATE_PROTOCOL,
    D_MINUS_ONE_OVERFIT_PROTOCOL as OVERFIT_PROTOCOL,
    D_MINUS_ONE_OVERFIT_PROTOCOL_ASSET_SOURCES as OVERFIT_PROTOCOL_ASSET_SOURCES,
    D_MINUS_ONE_OVERFIT_SOURCE_NAMES as OVERFIT_SOURCE_NAMES,
)
from .checkpoint import (
    SEGMENTATION_STATE_PREFIXES,
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
)


def desc_adapter_parameters(model) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if f".{DESCRIPTION_ADAPTER_NAME}." in name and "lora_" in name
    ]


def gradient_summary(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [
        value.grad.detach().float()
        for value in parameters
        if value.grad is not None
    ]
    norm_sum = (
        float(sum(
            (value.norm() for value in gradients),
            start=torch.tensor(0.0, device=gradients[0].device),
        ).cpu())
        if gradients else 0.0
    )
    return {
        "num_parameters": len(parameters),
        "num_with_grad": len(gradients),
        "num_nonzero": sum(
            int(torch.count_nonzero(value).item() > 0)
            for value in gradients
        ),
        "norm_sum": norm_sum,
        "all_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
    }


def description_step_gradient_gate(
    model,
    optimizer: torch.optim.Optimizer,
    *,
    run_stage: str,
    stream_name: str,
    stream_stage: str,
    observed_task_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Audit one accumulation window against the paths it actually executed.

    D-1 interleaves global and structured examples in one shuffled stream.  A
    global-only window must prove that MGRR stayed inactive, while a region
    window must prove the opposite; neither window is allowed to stand in for
    a task path that was not present in its microbatches.
    """
    named = list(model.named_parameters())

    def summary(predicate) -> dict[str, Any]:
        return gradient_summary([
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
    task_paths = set(observed_task_paths or ())
    allowed_task_paths = {"global_caption", "region_description"}
    if run_stage == "overfit":
        if not task_paths or task_paths - allowed_task_paths:
            raise ValueError(
                "D-1 gradient gate 需要当前 accumulation window 的显式 "
                f"task paths，observed={sorted(task_paths)}"
            )
        required_nonzero = {"desc_adapter", "global_visual_projection"}
        required_zero = {"alignment"}
        if "region_description" in task_paths:
            required_nonzero.update({
                "description_backbone", "mgrr", "region_projection",
            })
        else:
            # 全局 caption 不允许因为同属 overfit stage 而偷偷经过区域路径。
            required_zero.update({
                "description_backbone", "mgrr", "region_projection",
            })
    elif task_paths:
        raise ValueError(
            "observed_task_paths 仅用于 D-1 mixed overfit stream"
        )
    elif run_stage == "bridge_expert" and stream_name == "dior":
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
            gradient_summary(list(group["params"]))["all_finite"]
            for group in optimizer.param_groups
        ),
    }
    return {
        "protocol": DESCRIPTION_GRADIENT_GATE_PROTOCOL,
        "run_stage": run_stage,
        "stream_name": stream_name,
        "stream_stage": stream_stage,
        "observed_task_paths": sorted(task_paths),
        "required_nonzero": sorted(required_nonzero),
        "required_zero": sorted(required_zero),
        "modules": modules,
        "checks": checks,
        "passed": all(checks.values()),
    }


def causal_label_audit(output: Any, *, eos_token_id: int | None) -> dict[str, Any]:
    """Replay the exact prefix/target/padding label contract of one causal batch."""
    labels = output.labels
    lengths = tuple(int(value) for value in output.sequence_lengths)
    if labels is None or labels.ndim != 2:
        raise RuntimeError("causal description forward 缺少二维 labels")
    if labels.shape[0] != len(lengths) or output.logits.shape[:2] != labels.shape:
        raise RuntimeError("causal labels/logits/sequence_lengths shape 不一致")
    if eos_token_id is None:
        raise RuntimeError("causal description tokenizer 缺少 eos_token_id")

    prefix_masked = True
    target_contiguous = True
    padding_masked = True
    eos_supervised = True
    supervised_counts: list[int] = []
    prefix_lengths: list[int] = []
    for index, length in enumerate(lengths):
        if not 0 < length <= labels.shape[1]:
            raise RuntimeError(
                f"causal sequence length 越界: sample={index} length={length}"
            )
        row = labels[index]
        active = row[:length]
        supervised = torch.nonzero(active != -100, as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise RuntimeError(f"causal target 没有监督 token: sample={index}")
        first = int(supervised[0].item())
        last = int(supervised[-1].item())
        prefix_lengths.append(first)
        supervised_counts.append(int(supervised.numel()))
        prefix_masked &= first > 0 and bool((active[:first] == -100).all())
        target_contiguous &= (
            last == length - 1
            and int(supervised.numel()) == length - first
            and bool((active[first:length] != -100).all())
        )
        padding_masked &= bool((row[length:] == -100).all())
        eos_supervised &= int(active[-1].item()) == int(eos_token_id)

    checks = {
        "prefix_masked": prefix_masked,
        "target_contiguous": target_contiguous,
        "padding_masked": padding_masked,
        "eos_supervised": eos_supervised,
    }
    return {
        "protocol": "qpsalm_description_causal_label_audit_v1_runtime_batch",
        "num_samples": len(lengths),
        "min_prefix_tokens": min(prefix_lengths),
        "min_supervised_tokens": min(supervised_counts),
        "checks": checks,
        "passed": all(checks.values()),
    }


def dataset_data_audit(dataset) -> dict[str, Any]:
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
            "d_minus_one_use_region_tokens": (
                str(row.get("_d_minus_one_category") or "") != "global"
                if row.get("_d_minus_one_category") is not None else None
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
        "stage_spec": get_stage_spec(dataset.stage).to_dict(),
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
    artifact_readiness_acceptance: dict[str, Any] | None,
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
    generated_route_policy: dict[str, dict[str, bool]] = {}
    inconsistent_generated_routes: set[str] = set()
    for row in generation_rows:
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
        for group in trainable_manifest.get("groups", [])
        for name in group.get("parameter_names", [])
    ]
    def file_sha256(path: Path) -> str | None:
        if not path.is_file():
            return None
        return sha256_file(path)

    required_sources = set(OVERFIT_SOURCE_NAMES)
    artifact_acceptance = dict(artifact_readiness_acceptance or {})
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
    checkpoint_architecture: dict[str, Any] = {}
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
    causal_history_audit = audit_causal_label_history(history_rows)
    checks = {
        "artifact_readiness_consumed": (
            artifact_acceptance.get("protocol")
            == ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL
            and artifact_acceptance.get("status") == "engineering-valid"
            and artifact_acceptance.get("expert_truth_used") is False
            and artifact_acceptance.get("errors") == []
            and source_bindings.get("artifact_readiness_report", {}).get(
                "path"
            ) == artifact_acceptance.get("report")
            and source_bindings.get("artifact_readiness_report", {}).get(
                "sha256"
            ) == artifact_acceptance.get("report_sha256")
        ),
        "sample_count_is_exactly_64": int(
            sampling.get("selected_samples", 0)
        ) == 64,
        "global_box_mask_null_present": (
            set(category_counts) == {"global", "box", "mask", "null"}
            and all(int(category_counts.get(name, 0)) > 0 for name in category_counts)
        ),
        "output_format_region_route_separated": (
            sampling.get("category_region_token_policy") == {
                "global": False,
                "box": True,
                "mask": True,
                "null": True,
            }
            and not inconsistent_generated_routes
            and generated_route_policy == expected_generated_route_policy
        ),
        "gradient_windows_are_task_path_homogeneous": (
            main_batch_sampler.get("class")
            == DMinusOneTaskPathBatchSampler.__name__
            and main_batch_sampler.get("protocol")
            == DMinusOneTaskPathBatchSampler.protocol
            and int(main_batch_sampler.get("batch_size", -1))
            == int(config.training.batch_size)
            and int(main_batch_sampler.get("gradient_window_batches", -1))
            == int(config.training.grad_accum_steps)
            and main_batch_sampler.get("drop_last") is False
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
        "batch_size_greater_than_one": int(config.training.batch_size) > 1,
        "all_steps_completed": int(checkpoint_step) == int(config.training.max_steps),
        "max_steps_is_exactly_100": int(config.training.max_steps) == 100,
        "finite_loss_history": bool(losses) and len(losses) == len(history_rows),
        "loss_decreased": len(losses) >= 2 and losses[-1] < losses[0],
        "stage_gradient_gate_passed": d_minus_one_gradient_gate_passed(
            gradient_gate
        ),
        "segmentation_checkpoint_migration_current": (
            segmentation_migration.get("source_format") == CHECKPOINT_FORMAT
            and tuple(segmentation_migration.get("allowed_prefixes") or ())
            == SEGMENTATION_STATE_PREFIXES
        ),
        "segmentation_checkpoint_source_bound": migration_source_bound,
        "strict_checkpoint_reload_passed": (
            strict_reload_state_replay_passed(reload_audit)
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
            and (
                (
                    checkpoint_payload_provenance.get("checkpoint_metadata")
                    or {}
                ).get("metadata") or {}
            ).get("gradient_gate") == gradient_gate
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
            checkpoint_architecture.get("description_sequence_protocol")
            == DESCRIPTION_SEQUENCE_PROTOCOL
        ),
        "causal_labels_runtime_replayed": all(
            causal_history_audit.values()
        ),
        "generation_covers_four_categories": set(generated_categories)
        == {"global", "box", "mask", "null"},
        "caption_generation_observed": int(metrics.get("num_caption", 0)) > 0,
        "structured_generation_observed": int(metrics.get("num_structured", 0)) > 0,
        "structured_generation_protocol_bound": (
            structured_generation_audits_current(generation_rows)
        ),
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
            "batch_size": int(config.training.batch_size),
            "max_steps": int(config.training.max_steps),
            "generated_categories": generated_categories,
            "generated_route_policy": generated_route_policy,
            "generation_metrics": metrics,
            "causal_label_history_audit": causal_history_audit,
            "description_protocol_assets": protocol_assets,
            "artifact_readiness_acceptance": artifact_acceptance,
            "segmentation_migration": dict(segmentation_migration),
            "strict_reload_audit": dict(reload_audit),
            "gradient_gate": dict(gradient_gate or {}),
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


def region_data_audit(dataset) -> dict[str, Any] | None:
    """Return the M7-facing identity of a region-supervision population.

    The complete training audit also contains stream schedules and validation
    populations. M7 compares only frozen expert/prediction bindings against
    its current region loader, so checkpoint this compact contract explicitly.
    """
    if dataset.stage not in {"bridge_auto", "bridge_expert", "predicted_mask"}:
        return None
    population = dataset_data_audit(dataset)
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


def train_loss(model, batch: dict[str, Any], config: SegDescConfig) -> tuple[torch.Tensor, dict[str, float]]:
    backbone = model.encode_description_requests(
        batch["requests"],
        include_spatial=config.training.stage not in {"mmrs_caption", "rsicap_caption"},
    )
    if config.training.stage == "dior_alignment":
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
        protocol=config.model.region_protocol,
        structured_output=batch["structured_outputs"],
        use_region_tokens=batch["use_region_tokens"],
    )
    if output.per_sample_loss is None:
        raise RuntimeError("description forward 未产生 per-sample loss")
    label_audit = causal_label_audit(
        output,
        eos_token_id=getattr(model.controller.tokenizer, "eos_token_id", None),
    )
    if label_audit["passed"] is not True:
        raise RuntimeError(f"causal label 运行时门禁失败: {label_audit}")
    weights = batch["weights"]
    loss = (output.per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {
        "mean_sequence_length": sum(output.sequence_lengths) / max(len(output.sequence_lengths), 1),
        "causal_label_audit_passed": float(label_audit["passed"]),
        **{
            f"causal_{name}": float(value)
            for name, value in label_audit["checks"].items()
        },
    }
