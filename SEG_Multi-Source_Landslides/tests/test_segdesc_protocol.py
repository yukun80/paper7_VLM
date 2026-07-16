#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmentation-grounded description M5-M7 协议测试。

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest
SEG_Multi-Source_Landslides/tests/test_segdesc_protocol.py -v
写入行为：仅使用合成张量和临时目录，不加载 Qwen、benchmark 或 checkpoint。
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from qpsalm_seg.controllers import QwenMaskQueryController
from qpsalm_seg.engine.evaluator import (
    SAMPLE_IDENTITY_FIELDS,
    SAMPLE_IDENTITY_PROTOCOL,
    SEGMENTATION_EVAL_MANIFEST_PROTOCOL,
    SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL,
    segmentation_prediction_population,
)
from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT as SEGMENTATION_CHECKPOINT_FORMAT
from qpsalm_seg.schema import ActiveModalitySubset, ModalityBatch
from qpsalm_seg.paths import validate_output_replacement_safety
from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.description.counterfactuals import counterfactual_region_masks
from qpsalm_seg.description.data import (
    BRIDGE_BUILDER_VERSION,
    BRIDGE_ENGINEERING_AUDIT_PROTOCOL,
    BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
    BRIDGE_EXPERT_REPLAY_PROTOCOL,
    DescriptionTaskDataset,
    DESCRIPTION_ENGINEERING_AUDIT_PROTOCOL,
    FROZEN_GATE_COUNTERFACTUAL_MODES,
    FROZEN_GATE_SCIENTIFIC_PROTOCOLS,
    FROZEN_GATE_THRESHOLD_KEYS,
    REGION_INPUT_SOURCE_PROTOCOL,
    REGION_TRAINING_DATA_PROTOCOL,
    _caption_source_weights,
    _stable_weighted_index,
    require_engineering_bridge,
    require_engineering_description,
    _validate_expert_rows,
    cross_parent_region_swap_candidates,
    end_to_end_region_support,
    evaluation_region_source_population_sha256,
    filter_evaluation_region_source,
    filter_evaluation_source,
    require_frozen_expert_bridge,
    same_parent_region_swap_candidates,
    select_d_minus_one_mixture,
    validate_predicted_index,
)
from qpsalm_seg.description.metrics import (
    DescriptionMetricAccumulator,
    structured_disagreement,
    unsupported_claim_counts,
)
import qpsalm_seg.description.output_protocol as output_protocol
from qpsalm_seg.description.output_protocol import parse_description_output
from qpsalm_seg.description.json_protocol import strict_json_loads
from qpsalm_seg.description.predicted_regions import (
    FIXED_PREDICTION_ARTIFACT_PROTOCOL,
    OOF_MERGE_PROTOCOL,
    PREDICTED_REGION_FORMAT,
    export_predicted_regions,
    merge_oof_predictions,
    revalidate_oof_merged_index,
    validate_oof_checkpoint_binding,
)
from qpsalm_seg.description.expert_factuality import (
    EXPERT_FAMILIES,
    EXPERT_FACTUALITY_PROTOCOL,
    aggregate_expert_factuality,
    build_expert_review_template,
    revalidate_expert_factuality_report,
)
from qpsalm_seg.description.evaluator import (
    COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL,
    DESCRIPTION_EVALUATION_PROTOCOL,
    END_TO_END_TARGET_PROTOCOL,
    EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
    EVALUATION_POPULATION_FIELDS,
    EndToEndTargetResolver,
    _same_image_retrieval,
    build_evaluation_publication_audit,
    counterfactual_input_change_audit,
    evaluation_mask_artifact_inventory,
    evaluation_population_sha256,
    revalidate_evaluation_publication,
    validate_evaluation_checkpoint_binding,
    write_evaluation_mask_artifact,
)
from qpsalm_seg.description.target_audit import (
    build_segmentation_instruction_source_binding,
)
from qpsalm_seg.description.model import (
    SegmentationGroundedDescriptionModel,
    alignment_positive_mask,
    multi_positive_alignment_loss,
)
from qpsalm_seg.description.runtime import (
    description_parameter_groups,
    description_trainable_parameter_manifest,
)
from qpsalm_seg.description.trainer import (
    DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
    _description_stream_binding,
    _description_training_progress_payload,
    _description_step_gradient_gate,
    _next_description_stream_batch,
    _restore_description_training_progress,
    build_d_minus_one_overfit_validation,
)
from qpsalm_seg.description.backbone import (
    DescriptionCacheBackboneEncoder,
    restore_region_mask_from_cache,
    transform_region_mask_to_cache,
)
from qpsalm_seg.description.vision_cache import (
    DESCRIPTION_CACHE_BUILDER_VERSION,
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DescriptionVisionFeatureBank,
    description_cache_key,
)
from qpsalm_seg.description.common import (
    EpochShuffleBatchSampler,
    ParentGroupedRegionBatchSampler,
    append_jsonl,
    predicted_index_for_dataset,
    set_loader_epoch,
    validate_predicted_training_indexes,
    write_json,
)
from qpsalm_seg.description.run_artifacts import (
    CHECKPOINT_RUN_COMPLETION_PROTOCOL,
    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
    JOINT_TRAINING_COMPLETION_PROTOCOL,
    RESUME_RECONCILIATION_PROTOCOL,
    TERMINAL_CHECKPOINT_AUDIT_PROTOCOL,
    build_training_completion_report,
    prepare_training_attempt,
    reconcile_resume_run,
    validate_checkpoint_run_completion,
    validate_terminal_checkpoint_provenance,
)
from qpsalm_seg.description.joint_trainer import (
    JOINT_INITIALIZATION_PROTOCOL,
    JOINT_LOADER_SEED_OFFSETS,
    JOINT_LOADER_BINDING_PROTOCOL,
    JOINT_PROGRESS_PROTOCOL,
    JOINT_RUN_PROTOCOL,
    _initial_joint_loader_states,
    _joint_loader_binding,
    _joint_progress_payload,
    _next_joint_loader_batch,
    build_joint_initialization_audit,
    build_joint_optimizer,
    joint_optimizer_manifest,
    monitor_baseline_identity,
    monitor_retention_gate,
    restore_joint_progress,
    revalidate_joint_initialization_audit,
    validate_joint_checkpoint_execution,
    validate_m7_source_checkpoint,
    validate_joint_task_gradients,
)
from qpsalm_seg.description.checkpoint import (
    DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
    FROZEN_QWEN_PREFIX,
    SEGDESC_CHECKPOINT_FORMAT,
    SEGDESC_CHECKPOINT_PROVENANCE_PROTOCOL,
    SEGMENTATION_MIGRATION_LINEAGE_PROTOCOL,
    SEGMENTATION_STATE_PREFIXES,
    build_description_stage_lineage,
    capture_training_rng_state,
    description_protocol_assets_spec,
    initialize_segdesc_checkpoint,
    inspect_segdesc_checkpoint,
    load_segdesc_checkpoint,
    restore_training_rng_state,
    save_segdesc_checkpoint,
    validate_description_stage_lineage,
    validate_segmentation_migration_lineage,
    validate_resume_run_config,
    verify_segdesc_checkpoint_reload,
)
from qpsalm_seg.description.d_minus_one import (
    D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
    D_MINUS_ONE_GATE_PROTOCOL,
    validate_d_minus_one_gate,
    validate_d_minus_one_overfit_report,
    validate_d_minus_one_runs,
)
from qpsalm_seg.description.cycle_localization import (
    CYCLE_LOCALIZATION_PROTOCOL,
    CYCLE_PROMPT_PROTOCOL,
    CycleLocalizationProvider,
    cycle_prompt_batch,
    cycle_region_iou,
    summarize_cycle_localization,
)
from qpsalm_seg.description.caption_metrics import (
    _bertscore_model_audit,
    _metric_summary,
    caption_metric_population,
)
from qpsalm_seg.description.caption_human_review import (
    aggregate_caption_human_reviews,
    build_caption_human_review_template,
)
from qpsalm_seg.description.zero_shot import (
    ZERO_SHOT_INPUT_PROTOCOL,
    ZERO_SHOT_PROTOCOL,
    _input_audit,
)
from qpsalm_seg.description.oof import build_oof_fold_indexes, load_oof_manifest
from qpsalm_seg.description.comparison import (
    M4_BASELINE_REGION_ENCODERS,
    M4_SEED_GATE_PROTOCOL,
    M4_SUITE_GATE_PROTOCOL,
    _m4_cross_seed_training_population_contract,
    _counterfactual_gate,
    _formal_seed_binding,
    _rows,
    _validate_three_seed_artifact_uniqueness,
    _validate_evaluation_checkpoint_provenance,
    _validate_paired_evaluation_reports,
    aggregate_m4_region_encoder_reports,
)
from qpsalm_seg.description.retention import (
    BASELINE_CHECKPOINT_REPLAY_PROTOCOL,
    M7_RETENTION_SEED_GATE_PROTOCOL,
    aggregate_m7_retention_seed_gates,
    bind_joint_evaluation_report,
    build_baseline_checkpoint_replay_audit,
    segmentation_metric_input_population,
    validate_m7_retention_gate,
    validate_m7_retention_seed_gate,
)
from qpsalm_seg.description.d4_curriculum import (
    D4_CURRICULUM_GATE_PROTOCOL,
    M4_SUITE_ACCEPTANCE_PROTOCOL,
    build_d4_curriculum_gate,
    validate_d4_curriculum_gate,
    validate_d4_curriculum_transition,
    validate_d4_final_acceptance_for_m7,
)
from qpsalm_seg.description.m6_acceptance import (
    M6_ACCEPTANCE_AUDIT_PROTOCOL,
    build_m6_acceptance_gate,
    validate_m6_acceptance_for_m7,
    validate_m6_acceptance_gate,
)
from qpsalm_seg.cli.eval_segdesc_retention import (
    baseline_eval_binding,
    build_retention_gate,
)


def valid_target(status: str = "absent") -> dict:
    return {
        "schema_version": "qpsalm_description_output_v1",
        "target_status": status,
        "region": {
            "location": "unavailable", "size_class": "unavailable",
            "shape": "unavailable", "elongation": "unavailable",
            "compactness": "unavailable", "fragmentation": "unavailable",
        },
        "evidence": {
            "surface_observation": "unavailable", "terrain_support": "unavailable",
            "sar_support": "unavailable", "deformation_support": "unavailable",
            "surrounding_context": "unavailable", "evidence_sufficiency": "unavailable",
        },
        "summary": "No target is present.",
    }


def publish_synthetic_evaluation(root: Path, report: dict) -> dict:
    """Attach the standalone CLI terminal audit to a lightweight test report."""
    raw_path = root / "raw_generations.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (root / "counterfactual_generations.jsonl").write_text(
        "", encoding="utf-8"
    )
    checkpoint = root / "synthetic_publication_checkpoint.bin"
    checkpoint.write_bytes(b"synthetic-publication-checkpoint")
    report.setdefault("num_samples", len(rows))
    report.setdefault("num_generated", len(rows))
    coverage = report.setdefault("generation_coverage", {})
    coverage.setdefault("requested", 0)
    coverage.setdefault("eligible_samples", int(report["num_samples"]))
    coverage.setdefault("generated_samples", len(rows))
    coverage.setdefault(
        "fraction", len(rows) / max(int(report["num_samples"]), 1)
    )
    coverage.setdefault("complete", len(rows) == int(report["num_samples"]))
    coverage.setdefault("population_sha256", evaluation_population_sha256(rows))
    coverage.setdefault(
        "population_identity_fields", list(EVALUATION_POPULATION_FIELDS)
    )
    report.setdefault("counterfactual_sensitivity", {})
    report.setdefault("end_to_end_coverage", None)
    report.setdefault("cycle_localization", None)
    report.update({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": 1,
        "checkpoint_metadata": {},
        "checkpoint_binding": {
            "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
        },
    })
    report["publication_audit"] = build_evaluation_publication_audit(
        root, report
    )
    (root / "eval_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return report


def refresh_synthetic_evaluation_publication(root: Path, report: dict) -> dict:
    """Rebind an intentionally edited synthetic report before deeper replay tests."""
    report.pop("publication_audit", None)
    report["publication_audit"] = build_evaluation_publication_audit(
        root, report
    )
    return report


def write_synthetic_region_input_source(
    root: Path,
    *,
    sample_id: str,
    parent_sample_id: str,
    region_id: str,
    region_source: str,
    mask: np.ndarray,
) -> tuple[dict, Path]:
    source_dir = root / "source_masks"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.npy"
    np.save(path, mask.astype(np.uint8), allow_pickle=False)
    shape = list(mask.shape)
    return {
        "protocol": REGION_INPUT_SOURCE_PROTOCOL,
        "sample_id": sample_id,
        "parent_sample_id": parent_sample_id,
        "region_id": region_id,
        "region_source": region_source,
        "cache_lookup_key": description_cache_key(
            "multisource_parent", parent_sample_id
        ),
        "cache_fingerprint": hashlib.sha256("|".join([
            DESCRIPTION_CACHE_PROTOCOL,
            description_cache_key("multisource_parent", parent_sample_id),
            hashlib.sha256(b"synthetic-source").hexdigest(),
            "synthetic-model-revision",
            "synthetic-processor-revision",
            hashlib.sha256(b"synthetic-view").hexdigest(),
        ]).encode()).hexdigest(),
        "render_transform": {
            "source_h": shape[0],
            "source_w": shape[1],
            "resized_h": shape[0],
            "resized_w": shape[1],
            "pad_top": 0,
            "pad_left": 0,
            "size": shape[0],
        },
        "source_mask": {
            "kind": "binary_npy",
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": int(path.stat().st_size),
            "shape": shape,
            "positive_pixels": int(mask.sum()),
        },
    }, path


def write_synthetic_description_cache_binding(
    root: Path,
    *,
    variant: str = "",
) -> dict:
    """Create a minimal deep-validated cache for formal checkpoint fixtures."""
    if (root / "validation_report.json").is_file():
        return DescriptionVisionFeatureBank(root).artifact_binding()
    root.mkdir(parents=True, exist_ok=True)
    parent = "parent-1"
    component = "multisource_parent"
    key = description_cache_key(component, parent)
    source_hash = hashlib.sha256(
        f"synthetic-source:{variant}".encode()
    ).hexdigest()
    view_hash = hashlib.sha256(
        f"synthetic-view:{variant}".encode()
    ).hexdigest()
    cache_fingerprint = hashlib.sha256("|".join([
        DESCRIPTION_CACHE_PROTOCOL,
        key,
        source_hash,
        "synthetic-model-revision",
        "synthetic-processor-revision",
        view_hash,
    ]).encode()).hexdigest()
    record = {
        "lookup_key": key,
        "component": component,
        "parent_sample_id": parent,
        "source_ref": f"benchmark/synthetic{variant}.png",
        "source_content_hash": source_hash,
        "source_cache": None,
        "cache_fingerprint": cache_fingerprint,
        "views": [{
            "content_hash": view_hash,
            "spatial_features": [
                torch.zeros(2, size, size, dtype=torch.float16)
                for size in (4, 3, 2, 1)
            ],
            "view_tokens": torch.zeros(2, 4, dtype=torch.float16),
            "valid_mask": torch.ones(1, 4, 4, dtype=torch.float16),
            "render_transform": {
                "source_h": 4,
                "source_w": 4,
                "resized_h": 4,
                "resized_w": 4,
                "pad_top": 0,
                "pad_left": 0,
                "size": 4,
            },
        }],
    }
    shard = root / "shard_00000.pt"
    torch.save({"format": DESCRIPTION_CACHE_FORMAT, "records": [record]}, shard)
    shard_fingerprint = {
        "path": shard.name,
        "size": int(shard.stat().st_size),
        "records": 1,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    }
    input_fingerprints = {
        component: {
            "benchmark": "benchmark/synthetic_description",
            "index": "indexes/all.jsonl",
            "size": 1,
            "sha256": hashlib.sha256(
                f"synthetic-index:{variant}".encode()
            ).hexdigest(),
            "validation_report": "reports/validation_report.json",
            "validation_report_size": 1,
            "validation_report_sha256": hashlib.sha256(
                f"synthetic-validation:{variant}".encode()
            ).hexdigest(),
            "validation_builder_version": "synthetic_validation_v1",
            "validation_status": "engineering-valid",
        },
    }
    manifest = {
        "format": DESCRIPTION_CACHE_FORMAT,
        "protocol": DESCRIPTION_CACHE_PROTOCOL,
        "builder_version": DESCRIPTION_CACHE_BUILDER_VERSION,
        "renderer_version": "synthetic-renderer",
        "render_size": 4,
        "model_revision": "synthetic-model-revision",
        "processor_revision": "synthetic-processor-revision",
        "layers": [5, 11, 17, 23],
        "spatial_sizes": [4, 3, 2, 1],
        "view_tokens_per_view": 2,
        "spatial_channels": 2,
        "token_dim": 4,
        "backend": "hash-smoke",
        "input_fingerprints": input_fingerprints,
        "source_cache_provenance": {
            "provided": False,
            "path": None,
            "manifest_sha256": None,
            "metadata_fingerprint": None,
            "file_count": None,
            "reused_records": 0,
            "isolation_unchanged": True,
        },
        "num_samples": 1,
        "components": [component],
        "lookup": {key: {
            "shard": 0,
            "index": 0,
            "component": component,
            "parent_sample_id": parent,
        }},
        "shards": [shard.name],
        "shard_fingerprints": [shard_fingerprint],
        "shard_size": 1,
        "forbidden_state": [
            "instruction", "condition", "region_geometry", "segmentation_state",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    builder_bank = DescriptionVisionFeatureBank(
        root, require_validation_report=False
    )
    report = builder_bank.validate_all(
        expected_input_fingerprints=input_fingerprints
    )
    if report["errors"]:
        raise AssertionError(f"synthetic description cache invalid: {report['errors']}")
    (root / "validation_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return DescriptionVisionFeatureBank(root).artifact_binding()


def write_synthetic_segdesc_checkpoint(
    path: Path,
    checkpoint_metadata: dict,
    *,
    step: int,
    state_token: float = 1.0,
) -> None:
    """Write the smallest current checkpoint whose metadata can be replayed."""
    architecture = checkpoint_metadata.setdefault("description_architecture_spec", {})
    if "description_cache_artifact_binding" not in architecture:
        architecture["description_cache_artifact_binding"] = (
            write_synthetic_description_cache_binding(
                path.parent / f".{path.stem}_description_cache"
            )
        )
    checkpoint_metadata.setdefault("segmentation_architecture_spec", {})
    checkpoint_metadata.setdefault("metadata", {})
    try:
        migration_lineage = validate_segmentation_migration_lineage(
            checkpoint_metadata.get("segmentation_migration"),
            {"segmentation_migration": checkpoint_metadata.get(
                "segmentation_migration"
            )},
        )
    except RuntimeError:
        migration_lineage = None
    if migration_lineage is not None:
        checkpoint_metadata["metadata"].setdefault(
            "segmentation_migration_lineage", migration_lineage
        )
    state = {"synthetic.weight": torch.tensor([float(state_token)])}
    torch.save({
        "format": SEGDESC_CHECKPOINT_FORMAT,
        "step": int(step),
        "model_state": state,
        "required_state_keys": sorted(state),
        "frozen_qwen_prefix": FROZEN_QWEN_PREFIX,
        "adapter_names": ["default", "desc_adapter"],
        "description_sequence_protocol": "qpsalm_description_causal_v4_stage_separated",
        "description_protocol_assets": checkpoint_metadata[
            "description_protocol_assets"
        ],
        "description_architecture_spec": checkpoint_metadata[
            "description_architecture_spec"
        ],
        "segmentation_migration": checkpoint_metadata["segmentation_migration"],
        "segmentation_architecture_spec": checkpoint_metadata[
            "segmentation_architecture_spec"
        ],
        "metadata": checkpoint_metadata["metadata"],
    }, path)


def publish_synthetic_description_run_completion(
    selected_checkpoint: Path,
    *,
    stage: str,
    role: str,
    step: int,
) -> dict:
    """Publish a minimal successful run around a synthetic selected checkpoint."""
    root = selected_checkpoint.parent
    payload = torch.load(
        selected_checkpoint, map_location="cpu", weights_only=False
    )
    progress = {
        "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        "step": int(step),
    }
    if role == "terminal_last":
        payload["metadata"]["checkpoint_role"] = "terminal_last"
        payload["metadata"]["training_progress"] = progress
        torch.save(payload, selected_checkpoint)
        last_checkpoint = selected_checkpoint
    else:
        payload["metadata"]["checkpoint_role"] = "validation_best"
        torch.save(payload, selected_checkpoint)
        last_checkpoint = root / "checkpoint_last.pt"
        terminal_payload = copy.deepcopy(payload)
        terminal_payload["metadata"]["checkpoint_role"] = "terminal_last"
        terminal_payload["metadata"]["training_progress"] = progress
        torch.save(terminal_payload, last_checkpoint)

    history_path = root / "train_history.jsonl"
    history_path.write_text(
        json.dumps({"step": int(step), "loss": 0.1}) + "\n",
        encoding="utf-8",
    )
    progress_path = root / "training_progress_latest.json"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    required = {
        "checkpoint_last": last_checkpoint,
        "dataset_summary": root / "dataset_summary.json",
        "resolved_config": root / "resolved_config.json",
        "train_history": history_path,
        "training_progress_latest": progress_path,
        "trainable_parameter_manifest": (
            root / "trainable_parameter_manifest.json"
        ),
    }
    for name in (
        "dataset_summary.json",
        "resolved_config.json",
        "trainable_parameter_manifest.json",
    ):
        (root / name).write_text("{}\n", encoding="utf-8")
    optional = {}
    if role == "validation_best":
        selection_path = root / "validation_best.json"
        selection_path.write_text(
            json.dumps({"step": int(step), "selection_score": 0.5}),
            encoding="utf-8",
        )
        optional = {
            "checkpoint_best": selected_checkpoint,
            "validation_best": selection_path,
        }
    terminal = validate_terminal_checkpoint_provenance(
        inspect_segdesc_checkpoint(last_checkpoint),
        checkpoint=last_checkpoint,
        expected_step=int(step),
        expected_stage=stage,
        progress_key="training_progress",
        expected_progress_protocol=DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        progress_artifact=progress_path,
        progress_artifact_name="training_progress_latest",
        history_artifact=history_path,
        history_artifact_name="train_history",
    )
    completion = build_training_completion_report(
        protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
        report={
            "stage": stage,
            "steps": int(step),
            "terminal_checkpoint_audit": terminal,
        },
        required_artifacts=required,
        optional_artifacts=optional,
    )
    (root / "training_report.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    return validate_checkpoint_run_completion(
        selected_checkpoint,
        expected_completion_protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
        expected_stage=stage,
        expected_role=role,
    )


def publish_synthetic_joint_run_completion(
    selected_checkpoint: Path,
    *,
    step: int,
) -> dict:
    """Publish a minimal completed joint run around validation_best."""
    root = selected_checkpoint.parent
    payload = torch.load(
        selected_checkpoint, map_location="cpu", weights_only=False
    )
    payload["metadata"]["checkpoint_role"] = "validation_best"
    torch.save(payload, selected_checkpoint)
    progress = dict(payload["metadata"]["joint_progress"])
    last_checkpoint = root / "checkpoint_last.pt"
    terminal_payload = copy.deepcopy(payload)
    terminal_payload["metadata"]["checkpoint_role"] = "terminal_last"
    torch.save(terminal_payload, last_checkpoint)
    progress_path = root / "joint_coverage_latest.json"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    history_path = root / "joint_history.jsonl"
    history_path.write_text(
        json.dumps({"step": int(step), "loss": 0.1}) + "\n",
        encoding="utf-8",
    )
    selection_path = root / "joint_validation_best.json"
    selection_path.write_text(
        json.dumps({"step": int(step), "selection_score": 0.5}),
        encoding="utf-8",
    )
    joint_manifest = root / "joint_manifest.json"
    monitor_baseline = root / "segmentation_monitor_baseline.json"
    joint_manifest.write_text("{}\n", encoding="utf-8")
    monitor_baseline.write_text("{}\n", encoding="utf-8")
    terminal = validate_terminal_checkpoint_provenance(
        inspect_segdesc_checkpoint(last_checkpoint),
        checkpoint=last_checkpoint,
        expected_step=int(step),
        expected_stage="joint",
        progress_key="joint_progress",
        expected_progress_protocol=JOINT_PROGRESS_PROTOCOL,
        progress_artifact=progress_path,
        progress_artifact_name="joint_coverage_latest",
        history_artifact=history_path,
        history_artifact_name="joint_history",
    )
    completion = build_training_completion_report(
        protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
        report={
            "stage": "joint",
            "steps": int(step),
            "terminal_checkpoint_audit": terminal,
        },
        required_artifacts={
            "checkpoint_last": last_checkpoint,
            "joint_coverage_latest": progress_path,
            "joint_history": history_path,
            "joint_manifest": joint_manifest,
            "segmentation_monitor_baseline": monitor_baseline,
        },
        optional_artifacts={
            "checkpoint_best": selected_checkpoint,
            "validation_best": selection_path,
        },
    )
    (root / "training_report.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    return validate_checkpoint_run_completion(
        selected_checkpoint,
        expected_completion_protocol=JOINT_TRAINING_COMPLETION_PROTOCOL,
        expected_stage="joint",
        expected_role="validation_best",
    )


def frozen_scientific_gate(bindings: dict[str, str]) -> dict:
    return {
        "protocol": "landslide_bridge_evaluation_gate_v2",
        "builder_version": "landslide_bridge_m2_v7_expert_review_replay_bound",
        "frozen": True,
        "status": "frozen_after_pilot",
        "bindings": bindings,
        "thresholds": {key: 0.5 for key in FROZEN_GATE_THRESHOLD_KEYS},
        "scientific_protocol": {
            **FROZEN_GATE_SCIENTIFIC_PROTOCOLS,
            "bootstrap": {
                "unit": "parent", "confidence": 0.95,
                "samples": 10000, "seed": 42,
            },
            "counterfactual_minimum_effective_parents": {
                mode: 1 for mode in FROZEN_GATE_COUNTERFACTUAL_MODES
            },
        },
    }


def synthetic_bridge_expert_row(parent: str, split: str) -> dict:
    return {
        "bridge_record_id": f"expert::{split}::{parent}",
        "parent_sample_id": parent,
        "split": split,
        "instruction": "Describe the reviewed landslide region.",
        "task_family": "region_description_expert",
        "target_status": "present",
        "region_id": "global",
        "region_source": "gt_global_mask",
        "dataset_name": "synthetic_dataset",
        "modality_family_combo": "optical+terrain",
        "review": {"status": "accepted"},
        "expert_target": {
            "structured_output": valid_target("present"),
            "summary": "A reviewed landslide region is present.",
        },
    }


def synthetic_predicted_expert_rows() -> list[dict]:
    return [
        synthetic_bridge_expert_row("train-parent-0", "train"),
        synthetic_bridge_expert_row("train-parent-1", "train"),
        synthetic_bridge_expert_row("parent-1", "val"),
        synthetic_bridge_expert_row("test-parent-0", "test"),
    ]


def write_bound_frozen_bridge(
    bridge: Path,
    *,
    expert_rows: list[dict] | None = None,
) -> None:
    """Create the smallest v7 Bridge whose raw and derived artifacts replay."""
    (bridge / "reports").mkdir(parents=True, exist_ok=True)
    (bridge / "manifests").mkdir(exist_ok=True)
    (bridge / "indexes").mkdir(exist_ok=True)

    rows = [copy.deepcopy(row) for row in (expert_rows or [])]
    present_splits = {str(row.get("split")) for row in rows}
    for split in ("train", "val", "test"):
        if split not in present_splits:
            rows.append(synthetic_bridge_expert_row(f"{split}-filler", split))
    rows.sort(key=lambda row: (
        str(row["split"]), str(row["parent_sample_id"]),
        str(row["bridge_record_id"]),
    ))

    binding_paths = {
        "pilot_parent_manifest_sha256": bridge / "manifests/pilot_parent_manifest.jsonl",
        "review_selection_sha256": bridge / "manifests/review_selection.jsonl",
        "candidate_index_sha256": bridge / "indexes/candidate_all.jsonl",
    }
    for index, path in enumerate(binding_paths.values()):
        if not path.is_file():
            path.write_text(json.dumps({"index": index}) + "\n", encoding="utf-8")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_jsonl(path: Path, values: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )

    def artifact(path: Path, *, records: int | None = None) -> dict:
        result = {
            "path": str(path.resolve(strict=False)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if records is not None:
            result["records"] = records
        return result

    write_jsonl(binding_paths["review_selection_sha256"], [
        {
            "review_item_id": f"review::{row['bridge_record_id']}",
            "bridge_record_id": row["bridge_record_id"],
        }
        for row in rows
    ])

    expert_all = bridge / "indexes/expert_all.jsonl"
    pending = bridge / "indexes/pending_arbitration.jsonl"
    write_jsonl(expert_all, rows)
    write_jsonl(pending, [])
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, values in split_rows.items():
        write_jsonl(bridge / f"indexes/expert_{split}.jsonl", values)

    review_items = len(rows)
    review_source_dir = bridge / "review_sources"
    review_source_dir.mkdir(exist_ok=True)
    reviewer_paths = {}
    for reviewer in ("reviewer_1", "reviewer_2"):
        path = review_source_dir / f"{reviewer}.jsonl"
        write_jsonl(path, [
            {
                "review_item_id": f"review::{row['bridge_record_id']}",
                "reviewer_id": reviewer,
                "decision": "accept",
            }
            for row in rows
        ])
        reviewer_paths[reviewer] = path
    gate_source = review_source_dir / "evaluation_gate_frozen.json"
    gate_payload = frozen_scientific_gate({
        name: sha256(path) for name, path in binding_paths.items()
    })
    gate_source.write_text(json.dumps(gate_payload, sort_keys=True), encoding="utf-8")
    gate_output = bridge / "manifests/evaluation_gate_manifest.json"
    published_gate = copy.deepcopy(gate_payload)
    published_gate["source_file"] = str(gate_source.resolve(strict=False))
    gate_output.write_text(json.dumps(published_gate, sort_keys=True), encoding="utf-8")

    merge_binding = {
        "protocol": BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
        "builder_version": BRIDGE_BUILDER_VERSION,
        "sources": {
            "reviewer_1": artifact(reviewer_paths["reviewer_1"], records=review_items),
            "reviewer_2": artifact(reviewer_paths["reviewer_2"], records=review_items),
            "arbitration": None,
            "evaluation_gate_source": artifact(gate_source),
        },
        "outputs": {
            "expert_all": artifact(expert_all, records=len(rows)),
            "expert_train": artifact(
                bridge / "indexes/expert_train.jsonl", records=len(split_rows["train"])
            ),
            "expert_val": artifact(
                bridge / "indexes/expert_val.jsonl", records=len(split_rows["val"])
            ),
            "expert_test": artifact(
                bridge / "indexes/expert_test.jsonl", records=len(split_rows["test"])
            ),
            "pending_arbitration": artifact(pending, records=0),
            "evaluation_gate": artifact(gate_output),
        },
    }
    review_report_path = bridge / "reports/expert_review_report.json"
    review_report_path.write_text(json.dumps({
        "builder_version": BRIDGE_BUILDER_VERSION,
        "status": "complete",
        "frozen_evaluation_gate": True,
        "errors": [],
        "review_items": review_items,
        "expert_records": len(rows),
        "pending_arbitration": 0,
        "final_decisions": {"accept": len(rows)},
        "expert_artifact_binding": merge_binding,
    }, sort_keys=True), encoding="utf-8")
    validation_binding = {
        "protocol": BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
        "builder_version": BRIDGE_BUILDER_VERSION,
        "review_report": artifact(review_report_path),
        "merge_artifacts": merge_binding,
        "semantic_replay": {
            "protocol": BRIDGE_EXPERT_REPLAY_PROTOCOL,
            "candidate_index": artifact(
                binding_paths["candidate_index_sha256"], records=1
            ),
            "review_selection": artifact(
                binding_paths["review_selection_sha256"], records=review_items
            ),
            "review_items": review_items,
            "disputed_review_items": 0,
            "expert_records": len(rows),
            "pending_arbitration": 0,
            "final_decisions": {"accept": len(rows)},
            "review_report_statistics_verified": True,
        },
    }
    (bridge / "reports/validation_report.json").write_text(json.dumps({
        "builder_version": BRIDGE_BUILDER_VERSION,
        "mode": "small",
        "status": "expert_pilot_frozen",
        "require_expert_complete": True,
        "errors": [],
        "expert_artifact_binding": validation_binding,
    }, sort_keys=True), encoding="utf-8")


def write_synthetic_expert_factuality(evaluation: Path) -> Path:
    """Create two immutable reviewer sources and a revalidated ERFS report."""
    templates = build_expert_review_template(evaluation)
    review_paths = []
    for reviewer in ("reviewer_1", "reviewer_2"):
        path = evaluation / f"{reviewer}.jsonl"
        rows = [
            {
                **json.loads(json.dumps(template)),
                "reviewer_id": reviewer,
                "family_scores": {family: 1.0 for family in EXPERT_FAMILIES},
                "claims": [
                    {**claim, "support": "supported"}
                    for claim in template["claims"]
                ],
            }
            for template in templates
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        review_paths.append(path)
    report = aggregate_expert_factuality(
        evaluation,
        review_paths,
        seed=42,
        minimum_reviewers=2,
    )
    output = evaluation / "expert_factuality_report.json"
    output.write_text(json.dumps(report), encoding="utf-8")
    revalidate_expert_factuality_report(output, evaluation_dir=evaluation)
    return output


def write_synthetic_predicted_artifacts(
    root: Path,
    *,
    bridge: Path,
    expert_gate_audit: dict,
) -> tuple[dict, dict, Path]:
    """Publish tiny but fully replayable OOF-train and fixed-val artifacts."""
    artifact_root = root / "predicted_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    marker = artifact_root / "artifact_refs.json"
    if marker.is_file():
        references = json.loads(marker.read_text(encoding="utf-8"))
        train_audit = validate_predicted_index(
            Path(references["train_index"]),
            split="train",
            expert_gate_audit=expert_gate_audit,
        )
        val_audit = validate_predicted_index(
            Path(references["val_index"]),
            split="val",
            expert_gate_audit=expert_gate_audit,
        )
        return train_audit, val_audit, Path(references["fixed_checkpoint"])

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def canonical_sha256(value: object) -> str:
        return hashlib.sha256(json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    frozen_rows = synthetic_predicted_expert_rows()
    train_sources = [row for row in frozen_rows if row["split"] == "train"]
    val_sources = [row for row in frozen_rows if row["split"] == "val"]
    expert_train = bridge / "indexes/expert_train.jsonl"
    expert_val = bridge / "indexes/expert_val.jsonl"
    self_check_rows = {
        "train": [json.loads(line) for line in expert_train.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()],
        "val": [json.loads(line) for line in expert_val.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()],
    }
    if self_check_rows != {"train": train_sources, "val": val_sources}:
        raise AssertionError("synthetic frozen Bridge expert indexes drifted")

    segmentation_source = artifact_root / "instruction_train.jsonl"
    segmentation_source.write_text("".join(
        json.dumps({
            "sample_id": f"instruction::{row['parent_sample_id']}",
            "parent_sample_id": row["parent_sample_id"],
            "split": "train",
        }) + "\n"
        for row in train_sources
    ), encoding="utf-8")
    folds_root = artifact_root / "folds"
    manifest = build_oof_fold_indexes(
        segmentation_index=segmentation_source,
        bridge_index=expert_train,
        output_dir=folds_root,
        num_folds=2,
        seed=42,
    )
    manifest_path = folds_root / "fold_manifest.json"
    fold_inputs = []
    source_by_parent = {row["parent_sample_id"]: row for row in train_sources}
    for fold, metadata in manifest["folds"].items():
        train_path = Path(metadata["train_index"])
        holdout_path = Path(metadata["holdout_index"])

        def fingerprint(path: Path) -> dict:
            return {
                "reference": str(path),
                "status": "present",
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }

        checkpoint = artifact_root / f"oof_segmentation_fold_{fold}.pt"
        torch.save({
            "format": SEGMENTATION_CHECKPOINT_FORMAT,
            "step": 50,
            "config": {
                "train_index": str(train_path),
                "val_index": str(holdout_path),
            },
            "evidence_protocol": {
                "input_protocol": {
                    "index_fingerprints": {
                        "train": fingerprint(train_path),
                        "val": fingerprint(holdout_path),
                        "test": fingerprint(holdout_path),
                    },
                },
            },
        }, checkpoint)
        checkpoint_audit = validate_oof_checkpoint_binding(
            checkpoint=checkpoint,
            fold_manifest=manifest_path,
            checkpoint_fold=fold,
            prediction_index=holdout_path,
        )
        fold_output = artifact_root / f"predicted_fold_{fold}"
        prediction_rows = []
        for parent, assigned in manifest["parent_to_fold"].items():
            if assigned != fold:
                continue
            mask = fold_output / f"masks/train/{parent}.npy"
            mask.parent.mkdir(parents=True, exist_ok=True)
            np = __import__("numpy")
            np.save(mask, np.eye(2, dtype=np.uint8), allow_pickle=False)
            source = source_by_parent[parent]
            prediction_rows.append({
                **source,
                "schema_version": PREDICTED_REGION_FORMAT,
                "bridge_record_id": f"predicted::{parent}::50",
                "region_id": "predicted_global",
                "region_source": "predicted_proposal",
                "region_mask": {
                    "path": str(mask),
                    "sha256": sha256(mask),
                    "shape": [2, 2],
                    "threshold": 0.5,
                },
                "prediction_provenance": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "checkpoint_step": 50,
                    "split": "train",
                    "fold_manifest": checkpoint_audit["fold_manifest"],
                    "fold_manifest_sha256": checkpoint_audit[
                        "fold_manifest_sha256"
                    ],
                    "checkpoint_fold": fold,
                    "out_of_fold_verified": True,
                    "fold_audit": checkpoint_audit,
                    "source_bridge_record_id": source["bridge_record_id"],
                    "source_expert_record_sha256": canonical_sha256(source),
                },
            })
        prediction_index = fold_output / f"predicted_train_{fold}.jsonl"
        prediction_index.write_text(
            "".join(json.dumps(row) + "\n" for row in prediction_rows),
            encoding="utf-8",
        )
        fold_inputs.append(prediction_index)
    merged = artifact_root / "predicted_train_oof.jsonl"
    merge_oof_predictions(
        fold_manifest=manifest_path,
        input_indexes=fold_inputs,
        output=merged,
    )
    train_audit = validate_predicted_index(
        merged, split="train", expert_gate_audit=expert_gate_audit
    )

    fixed_checkpoint = artifact_root / "fixed_segmentation.pt"
    torch.save({
        "format": SEGMENTATION_CHECKPOINT_FORMAT,
        "step": 6000,
    }, fixed_checkpoint)
    val_source = val_sources[0]
    fixed_dir = artifact_root / "fixed_val"
    val_mask = fixed_dir / "masks/val/parent-1.npy"
    val_mask.parent.mkdir(parents=True)
    np = __import__("numpy")
    np.save(val_mask, np.eye(2, dtype=np.uint8), allow_pickle=False)
    val_row = {
        **val_source,
        "schema_version": PREDICTED_REGION_FORMAT,
        "bridge_record_id": "predicted::parent-1::6000",
        "region_id": "predicted_global",
        "region_source": "predicted_proposal",
        "region_mask": {
            "path": str(val_mask),
            "sha256": sha256(val_mask),
            "shape": [2, 2],
            "threshold": 0.5,
        },
        "prediction_provenance": {
            "checkpoint": str(fixed_checkpoint),
            "checkpoint_sha256": sha256(fixed_checkpoint),
            "checkpoint_step": 6000,
            "split": "val",
            "fold_manifest": None,
            "fold_manifest_sha256": None,
            "checkpoint_fold": None,
            "out_of_fold_verified": True,
            "fold_audit": None,
            "source_bridge_record_id": val_source["bridge_record_id"],
            "source_expert_record_sha256": canonical_sha256(val_source),
        },
    }
    fixed_index = fixed_dir / "predicted_val.jsonl"
    fixed_index.write_text(json.dumps(val_row) + "\n", encoding="utf-8")
    mask_inventory = [{
        "parent_sample_id": "parent-1",
        "path": str(val_mask),
        "sha256": sha256(val_mask),
    }]
    (fixed_dir / "report.json").write_text(json.dumps({
        "format": PREDICTED_REGION_FORMAT,
        "validation_protocol": FIXED_PREDICTION_ARTIFACT_PROTOCOL,
        "split": "val",
        "requested_max_parents": 0,
        "num_parents": 1,
        "num_eligible_parents": 1,
        "population_complete": True,
        "population_sha256": canonical_sha256(["parent-1"]),
        "mask_inventory_sha256": canonical_sha256(mask_inventory),
        "mask_bytes": val_mask.stat().st_size,
        "checkpoint": str(fixed_checkpoint),
        "checkpoint_sha256": sha256(fixed_checkpoint),
        "checkpoint_step": 6000,
        "source_bridge_index": str(expert_val),
        "source_bridge_index_sha256": sha256(expert_val),
        "expert_gate_audit": expert_gate_audit,
        "index": str(fixed_index),
        "index_sha256": sha256(fixed_index),
    }), encoding="utf-8")
    val_audit = validate_predicted_index(
        fixed_index, split="val", expert_gate_audit=expert_gate_audit
    )
    marker.write_text(json.dumps({
        "train_index": str(merged),
        "val_index": str(fixed_index),
        "fixed_checkpoint": str(fixed_checkpoint),
    }), encoding="utf-8")
    return train_audit, val_audit, fixed_checkpoint


def write_synthetic_d4_curriculum_gate(
    root: Path,
    *,
    current_fraction: float,
    next_fraction: float | None,
    seed: int = 42,
    bridge_root: Path | None = None,
    d_minus_one_acceptance: dict | None = None,
    m4_candidate_checkpoint_sha256: str = "synthetic-upstream",
    build_curriculum_gate: bool = True,
    predicted_artifact_root: Path | None = None,
) -> tuple[Path, dict, Path, dict, dict, dict | None]:
    """Create a complete hash-bound formal D4 source chain in a temp tree."""

    root.mkdir(parents=True, exist_ok=True)

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    bridge = bridge_root or (root / "bridge")
    if not (bridge / "reports/validation_report.json").is_file():
        write_bound_frozen_bridge(
            bridge,
            expert_rows=synthetic_predicted_expert_rows(),
        )
    expert_gate_audit = require_frozen_expert_bridge(bridge)

    if current_fraction > 0.0:
        predicted_train_audit, predicted_val_audit, fixed_segmentation = (
            write_synthetic_predicted_artifacts(
                predicted_artifact_root or root,
                bridge=bridge,
                expert_gate_audit=expert_gate_audit,
            )
        )
        segmentation_source_sha256 = sha256(fixed_segmentation)
    else:
        predicted_train_audit = None
        predicted_val_audit = None
        fixed_segmentation = root / "segmentation_source.pt"
        torch.save({
            "format": SEGMENTATION_CHECKPOINT_FORMAT,
            "step": 6000,
        }, fixed_segmentation)
        segmentation_source_sha256 = sha256(fixed_segmentation)

    evaluation = root / "evaluation"
    evaluation.mkdir()
    checkpoint = root / "source_checkpoint.pt"
    source_stage = "bridge_expert" if current_fraction == 0.0 else "predicted_mask"
    candidate_path = bridge / "indexes/candidate_all.jsonl"
    candidate_sha256 = expert_gate_audit["candidate_index_sha256"]
    train_audit = {
        "protocol": REGION_TRAINING_DATA_PROTOCOL,
        "stage": source_stage,
        "expert_gate_audit": expert_gate_audit,
        "bridge_engineering_audit": {
            "protocol": BRIDGE_ENGINEERING_AUDIT_PROTOCOL,
            "status": "expert_pilot_frozen",
            "expert_truth_used": False,
            "candidate_index_sha256": candidate_sha256,
            "cache_input_fingerprint": {
                "benchmark": str(bridge),
                "index": "indexes/candidate_all.jsonl",
                "size": candidate_path.stat().st_size,
                "sha256": candidate_sha256,
            },
        },
        "predicted_index_audit": (
            predicted_train_audit
        ),
        "curriculum_audit": (
            None if current_fraction == 0.0 else {
                "protocol": "qpsalm_d4_predicted_mask_curriculum_v1",
                "requested_predicted_fraction": current_fraction,
                "selection_seed": 42,
                "training_mix": True,
            }
        ),
        "population": {
            "protocol": "qpsalm_description_dataset_population_v1",
            "stage": source_stage,
            "split": "train",
            "num_samples": 1,
            "num_parents": 1,
            "population_sha256": hashlib.sha256(
                f"{source_stage}:{current_fraction}".encode("utf-8")
            ).hexdigest(),
        },
    }
    val_audit = predicted_val_audit
    migration = {
        "source_path": str(fixed_segmentation.resolve(strict=False)),
        "source_sha256": segmentation_source_sha256,
        "source_format": SEGMENTATION_CHECKPOINT_FORMAT,
        "source_step": 6000,
        "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
    }
    checkpoint_metadata = {
        "segmentation_migration": migration,
        "description_protocol_assets": description_protocol_assets_spec(),
        "metadata": {
            "stage": source_stage,
            "checkpoint_role": (
                "terminal_last"
                if source_stage == "bridge_auto" else "validation_best"
            ),
            "config": {
                "seed": seed,
                "predicted_mask_fraction": current_fraction,
                "description_benchmark": (
                    (d_minus_one_acceptance or {}).get(
                        "description_source", {}
                    ).get("benchmark_root")
                ),
            },
            "region_data_audit": train_audit,
        },
    }
    if d_minus_one_acceptance is not None:
        checkpoint_metadata["metadata"].update({
            "d_minus_one_acceptance": d_minus_one_acceptance,
            "stage_lineage": synthetic_stage_lineage(
                seed=seed,
                d_minus_one_acceptance=d_minus_one_acceptance,
                target_stage=source_stage,
                predicted_predecessors=(
                    2 if current_fraction == 0.75
                    else 1 if current_fraction == 0.50 else 0
                ),
            ),
        })
    if current_fraction > 0.0:
        synthetic_m4_gate = root / "inherited_m4_suite_gate.json"
        synthetic_m4_gate.write_text("{}\n", encoding="utf-8")
        checkpoint_metadata["metadata"]["d4_curriculum_transition"] = {
            "m4_suite_acceptance": {
                "protocol": M4_SUITE_ACCEPTANCE_PROTOCOL,
                "suite_gate": str(synthetic_m4_gate),
                "suite_gate_sha256": sha256(synthetic_m4_gate),
                "seed": seed,
                "candidate_checkpoint_sha256": m4_candidate_checkpoint_sha256,
                "frozen_gate_audit": expert_gate_audit,
                "passed": True,
            },
        }
    write_synthetic_segdesc_checkpoint(
        checkpoint,
        checkpoint_metadata,
        step=100,
        state_token=float(seed) + float(current_fraction),
    )
    run_completion = publish_synthetic_description_run_completion(
        checkpoint,
        stage=source_stage,
        role=(
            "terminal_last"
            if source_stage == "bridge_auto" else "validation_best"
        ),
        step=100,
    )
    baseline_payload = valid_target("present")
    baseline_payload["region"].update({
        "location": "center",
        "size_class": "medium",
        "shape": "irregular",
        "elongation": "moderate",
        "compactness": "moderate",
        "fragmentation": "few_components",
    })
    baseline_payload["evidence"].update({
        "surface_observation": "A bright exposed-soil scar is visible.",
        "terrain_support": "supports",
        "sar_support": "supports",
        "deformation_support": "insufficient_evidence",
        "surrounding_context": "The region lies on a steep slope.",
        "evidence_sufficiency": "partial",
    })
    baseline_payload["summary"] = "A medium irregular landslide scar is present."
    changed_payload = valid_target("absent")
    baseline_text = json.dumps(baseline_payload)
    changed_text = json.dumps(changed_payload)
    row = {key: None for key in EVALUATION_POPULATION_FIELDS}
    preview = evaluation / "review_preview.png"
    preview.write_bytes(b"synthetic-review-preview")
    row.update({
        "sample_id": "sample-1",
        "parent_sample_id": "parent-1",
        "task_family": "region_description_expert",
        "target_status": "present",
        "region_id": (
            "global" if current_fraction == 0.0 else "predicted_global"
        ),
        "region_source": (
            "gt_global_mask" if current_fraction == 0.0 else "predicted_proposal"
        ),
        "split": "val",
        "evaluation_mode": (
            "gt_mask" if current_fraction == 0.0 else "fixed_prediction"
        ),
        "instruction": "Describe the region.",
        "target_text": baseline_text,
        "reference_texts": ["Reviewed target."],
        "has_unavailable_modality": True,
        "visual_preview_path": str(preview),
        "raw_generation": baseline_text,
        "raw_metrics": {
            "raw_schema_valid": True,
            "raw_field_accuracy": 1.0,
        },
    })
    region_mask = np.zeros((4, 4), dtype=np.uint8)
    region_mask.reshape(-1)[:10] = 1
    region_source_binding, region_source_path = write_synthetic_region_input_source(
        evaluation,
        sample_id="sample-1",
        parent_sample_id="parent-1",
        region_id=str(row["region_id"]),
        region_source=str(row["region_source"]),
        mask=region_mask,
    )
    region_artifact = write_evaluation_mask_artifact(
        evaluation,
        role="region_input",
        sample_id="sample-1",
        mask=region_mask,
    )
    row["region_input_mask_artifact"] = region_artifact
    row["region_input_source_binding"] = region_source_binding
    row["region_mask_path"] = str(region_source_path)
    row["region_area_fraction"] = float(region_mask.mean())
    raw_path = evaluation / "raw_generations.jsonl"
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    evaluation_mode = "gt_mask" if current_fraction == 0.0 else "fixed_prediction"
    report = {
        "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
        "stage": source_stage,
        "split": "val",
        "evaluation_mode": evaluation_mode,
        "region_protocol": "vision_only",
        "expert_gate_audit": expert_gate_audit,
        "predicted_index_audit": val_audit,
        "num_samples": 1,
        "num_generated": 1,
        "evaluation_limit_audit": {
            "protocol": "qpsalm_description_evaluation_limit_v1",
            "requested_max_samples": 0,
            "full_population_requested": True,
            "dataset_rows_evaluated": 1,
        },
        "generation_coverage": {
            "requested": 0,
            "eligible_samples": 1,
            "generated_samples": 1,
            "fraction": 1.0,
            "complete": True,
            "population_sha256": evaluation_population_sha256([row]),
            "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
        },
        "evaluation_mask_artifacts": evaluation_mask_artifact_inventory(
            [region_artifact]
        ),
        "statistics_protocol": {"runtime_seed": seed},
        "generation_metrics": {
            "target_status": {
                "macro_f1": 0.9,
                "per_label": {
                    "present": {"recall": 0.9},
                    "absent": {"recall": 0.9},
                },
                "false_description_rate": 0.1,
                "positive_false_rejection_rate": 0.1,
            },
        },
        "counterfactual_sensitivity": {
            mode: {
                "requested": 1,
                "n": 1,
                "num_effective_parents": 1,
                "aggregation_unit": "parent",
                "coverage_complete": True,
            }
            for mode in FROZEN_GATE_COUNTERFACTUAL_MODES
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": 100,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_binding": {
            "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
            "evaluation_mode": evaluation_mode,
            "evaluation_data_stage": source_stage,
            "checkpoint_stage": source_stage,
            "checkpoint_role": (
                "terminal_last"
                if source_stage == "bridge_auto" else "validation_best"
            ),
            "expected_checkpoint_role": (
                "terminal_last"
                if source_stage == "bridge_auto" else "validation_best"
            ),
            "saved_segmentation_migration": migration,
            "runtime_segmentation_migration": migration,
            "segmentation_source_sha256_match": True,
            "checkpoint_training_seed": seed,
            "evaluation_seed": seed,
            "seed_match": True,
            "run_completion": run_completion,
        },
    }
    report_path = evaluation / "eval_report.json"
    baseline_score = 1.0 - structured_disagreement(
        baseline_payload, baseline_payload
    )
    changed_score = 1.0 - structured_disagreement(
        changed_payload, baseline_payload
    )
    sensitivity = structured_disagreement(baseline_payload, changed_payload)
    baseline_claims = unsupported_claim_counts(
        baseline_payload, baseline_payload
    )[1]
    changed_claims = unsupported_claim_counts(
        changed_payload, baseline_payload
    )[1]

    def synthetic_counterfactual(mode: str) -> dict:
        mask_mode = mode in {
            "shuffled_mask", "region_swap", "cross_parent_region_swap",
        }
        donor = None
        if mode == "region_swap":
            donor = {
                "protocol": "qpsalm_same_parent_region_swap_v1",
                "parent_sample_id": "parent-1",
                "alternate_sample_id": "sample-2",
            }
        elif mode == "cross_parent_region_swap":
            donor = {
                "protocol": "qpsalm_cross_parent_region_swap_v1",
                "target_parent_sample_id": "parent-1",
                "donor_parent_sample_id": "parent-2",
                "donor_sample_id": "sample-2",
            }
        elif mode == "cross_parent_modality_swap":
            donor = {
                "protocol": "qpsalm_cross_parent_modality_donor_v1",
                "target_parent_sample_id": "parent-1",
                "donor_parent_sample_id": "parent-2",
                "common_modality_families": ["optical"],
                "applied_swap": {
                    "protocol": "qpsalm_cross_parent_modality_swap_v1",
                    "donor_parent_sample_id": "parent-2",
                    "modality_family": "optical",
                },
            }
        return {
            "sample_id": "sample-1",
            "parent_sample_id": "parent-1",
            "mode": mode,
            "counterfactual_input": donor,
            "input_change_audit": {
                "protocol": COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL,
                "mode": mode,
                "baseline_region_mask_sha256": "1" * 64,
                "counterfactual_region_mask_sha256": (
                    "2" * 64 if mask_mode else "1" * 64
                ),
                "baseline_backbone_state_sha256": "3" * 64,
                "counterfactual_backbone_state_sha256": (
                    "3" * 64 if mask_mode else "4" * 64
                ),
                "changed_dimensions": [
                    "region_mask" if mask_mode else "backbone_state"
                ],
                "changed": True,
            },
            "baseline_generation": baseline_text,
            "counterfactual_generation": changed_text,
            "sensitivity": sensitivity,
            "baseline_target_score": baseline_score,
            "counterfactual_target_score": changed_score,
            "target_score_delta": changed_score - baseline_score,
            "factual_claim_count_delta": float(changed_claims - baseline_claims),
        }

    (evaluation / "counterfactual_generations.jsonl").write_text(
        "".join(
            json.dumps(synthetic_counterfactual(mode)) + "\n"
            for mode in FROZEN_GATE_COUNTERFACTUAL_MODES
        ),
        encoding="utf-8",
    )
    report["publication_audit"] = build_evaluation_publication_audit(
        evaluation, report
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    expert_path = write_synthetic_expert_factuality(evaluation)
    gate = (
        build_d4_curriculum_gate(
            evaluation_dir=evaluation,
            expert_report=expert_path,
            bridge_benchmark=bridge,
            current_fraction=current_fraction,
            next_fraction=next_fraction,
            seed=seed,
        )
        if build_curriculum_gate
        else {}
    )
    gate_path = root / "d4_curriculum_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return (
        gate_path,
        gate,
        checkpoint,
        expert_gate_audit,
        train_audit,
        val_audit,
    )


def synthetic_retention_population(sha256: str = "a" * 64) -> dict:
    return {
        "protocol": SAMPLE_IDENTITY_PROTOCOL,
        "fields": list(SAMPLE_IDENTITY_FIELDS),
        "sha256": sha256,
        "complete": True,
        "unique": True,
        "num_records": 10,
        "num_unique_sample_ids": 10,
    }


def synthetic_segmentation_prediction_population(tag: str) -> dict:
    rows = []
    for index in range(10):
        rows.append({
            "sample_id": f"sample-{index}",
            "parent_sample_id": f"parent-{index}",
            "shape": [16, 16],
            "prediction_sha256": hashlib.sha256(
                f"prediction:{tag}:{index}".encode()
            ).hexdigest(),
            "target_sha256": hashlib.sha256(
                f"target:{index}".encode()
            ).hexdigest(),
            "valid_sha256": hashlib.sha256(
                f"valid:{index}".encode()
            ).hexdigest(),
        })
    return segmentation_prediction_population(rows, threshold=0.5)


def write_synthetic_d_minus_one_acceptance(root: Path) -> dict:
    """Create one current, deeply revalidatable D-1 acceptance fixture."""
    root.mkdir(parents=True, exist_ok=True)
    existing_gate = root / "d_minus_one_gate.json"
    if existing_gate.is_file():
        return validate_d_minus_one_gate(existing_gate)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    description = root / "description"
    (description / "indexes").mkdir(parents=True)
    (description / "reports").mkdir()
    (description / "data/dev/rsicap").mkdir(parents=True)
    description_rows = []
    for index in range(32):
        image_path = description / f"data/dev/rsicap/parent-{index}.png"
        image_path.write_bytes(f"synthetic-image-{index}".encode("utf-8"))
        description_rows.append({
            "sample_id": f"sample-{index}",
            "parent_sample_id": f"parent-{index}",
            "source_dataset": "RSICap",
            "task_family": "global_caption",
            "split": "dev",
            "instruction": "Describe the image.",
            "answers": [{"text": f"caption {index}"}],
            "visual_ref": {
                "path": str(image_path),
                "sha256": sha(image_path),
                "storage_mode": "materialized_copy",
            },
        })
    description_index = description / "indexes/dev.jsonl"
    description_index.write_text(
        "".join(json.dumps(row) + "\n" for row in description_rows),
        encoding="utf-8",
    )
    description_validation = description / "reports/validation_report.json"
    description_validation.write_text(json.dumps({
        "builder_version": "description_benchmark_m1_v4_answer_trace",
        "verified_perceptual_duplicate_cross_split_groups": 0,
        "errors": [],
    }), encoding="utf-8")
    selected, input_audit = _input_audit(
        description, "dev", max_samples=32, seed=42
    )

    zero_dir = root / "zero"
    zero_dir.mkdir()
    zero_raw = zero_dir / "raw_generations.jsonl"
    zero_raw.write_text("".join(
        json.dumps({
            "sample_id": row["sample_id"],
            "prediction": "nonempty synthetic caption",
        }) + "\n"
        for row in selected
    ), encoding="utf-8")
    model_dir = root / "qwen"
    model_dir.mkdir()
    model_config = model_dir / "config.json"
    model_config.write_text("{}\n", encoding="utf-8")
    metadata_hashes = {"config.json": sha(model_config)}
    zero_report = {
        "protocol": ZERO_SHOT_PROTOCOL,
        "status": "engineering-valid",
        "errors": [],
        "checks": {"synthetic_complete": True},
        "num_samples": 32,
        "caption_token_f1": 0.1,
        "statistics_seed": 42,
        "region_capability_claimed": False,
        "raw_generations": str(zero_raw),
        "raw_generations_sha256": sha(zero_raw),
        "input_audit": input_audit,
        "model_audit": {
            "model_dir": str(model_dir),
            "metadata_file_sha256": metadata_hashes,
            "metadata_snapshot_sha256": hashlib.sha256(json.dumps(
                metadata_hashes, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
        },
    }
    (zero_dir / "eval_report.json").write_text(
        json.dumps(zero_report), encoding="utf-8"
    )

    overfit_dir = root / "overfit"
    overfit_dir.mkdir()
    checkpoint = overfit_dir / "checkpoint_last.pt"
    segmentation = root / "segmentation.pt"
    segmentation.write_bytes(b"synthetic-segmentation-checkpoint")
    bridge_index = root / "candidate_all.jsonl"
    bridge_index.write_text("{}\n", encoding="utf-8")
    bridge_validation = root / "bridge_validation.json"
    bridge_validation.write_text("{}\n", encoding="utf-8")
    sampling = {
        "selected_samples": 32,
        "category_counts": {"global": 8, "box": 8, "mask": 8, "null": 8},
        "num_native_source_sizes": 2,
        "expert_truth_used": False,
        "bridge_target_authority": "deterministic_rule_candidate_not_expert",
        "sampling_seed": 42,
        "description_builder_version": "description_benchmark_m1_v4_answer_trace",
        "bridge_builder_version": "landslide_bridge_m2_v7_expert_review_replay_bound",
        "bridge_status": "awaiting_expert_review",
        "description_index": str(description_index),
        "description_index_sha256": sha(description_index),
        "description_validation_report": str(description_validation),
        "description_validation_report_sha256": sha(description_validation),
        "bridge_index": str(bridge_index),
        "bridge_index_sha256": sha(bridge_index),
        "bridge_validation_report": str(bridge_validation),
        "bridge_validation_report_sha256": sha(bridge_validation),
    }
    history = [
        {
            "step": 1,
            "loss": 2.0,
            "peak_reserved_gib": 20.0,
            "device_type": "cuda",
        },
        {
            "step": 100,
            "loss": 0.2,
            "peak_reserved_gib": 21.0,
            "device_type": "cuda",
        },
    ]
    history_path = overfit_dir / "train_history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row) + "\n" for row in history),
        encoding="utf-8",
    )
    generations = [
        {"d_minus_one_category": name}
        for name in ("global", "box", "mask", "null")
    ]
    overfit_raw = overfit_dir / "raw_generations.jsonl"
    overfit_raw.write_text(
        "".join(json.dumps(row) + "\n" for row in generations),
        encoding="utf-8",
    )
    metrics = {
        "num_caption": 16,
        "num_structured": 16,
        "raw_json_parse_rate": 0.5,
        "raw_schema_valid_rate": 0.25,
        "summary_nonempty_rate": 0.5,
    }
    validation_path = overfit_dir / "eval_report.json"
    validation_path.write_text(
        json.dumps({"generation_metrics": metrics}), encoding="utf-8"
    )
    gradient_path = overfit_dir / "description_gradient_gate.json"
    gradient_path.write_text(json.dumps({
        "passed": True, "all_required_streams_checked": True,
    }), encoding="utf-8")
    manifest = {"groups": [{"parameter_names": [
        "controller.model.layer.lora_A.desc_adapter.weight",
        "controller.model.layer.lora_B.desc_adapter.weight",
    ]}]}
    manifest_path = overfit_dir / "trainable_parameter_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = overfit_dir / "resolved_config.json"
    config_path.write_text(json.dumps({
        "stage": "overfit", "batch_size": 2, "max_steps": 100,
    }), encoding="utf-8")
    summary_path = overfit_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps({
        "d_minus_one_sampling_audit": sampling,
    }), encoding="utf-8")
    migration = {
        "source_path": str(segmentation),
        "source_sha256": sha(segmentation),
        "source_format": "qpsalm_sane_qmef_pmrd_v5",
        "source_step": 10,
        "allowed_prefixes": ["controller.", "sane.", "qmef.", "pmrd."],
    }
    write_synthetic_segdesc_checkpoint(
        checkpoint,
        {
            "description_protocol_assets": description_protocol_assets_spec(),
            "segmentation_migration": migration,
            "metadata": {
                "stage": "overfit",
                "checkpoint_role": "terminal_last",
                "training_progress": {
                    "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
                    "step": 100,
                },
                "config": {"seed": 42, "batch_size": 2, "max_steps": 100},
            },
        },
        step=100,
    )
    reload_audit = {
        "protocol": "qpsalm_segdesc_strict_reload_probe_v1",
        "passed": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_step": 100,
        "before_sha256": "a" * 64,
        "corrupted_sha256": "b" * 64,
        "restored_sha256": "a" * 64,
        "segmentation_migration": migration,
    }
    overfit_report = build_d_minus_one_overfit_validation(
        config=SimpleNamespace(batch_size=2, max_steps=100),
        sampling_audit=sampling,
        history_rows=history,
        gradient_gate={"passed": True},
        validation_report={"generation_metrics": metrics},
        generation_rows=generations,
        trainable_manifest=manifest,
        checkpoint_path=checkpoint,
        checkpoint_step=100,
        device_type="cuda",
        segmentation_migration=migration,
        reload_audit=reload_audit,
        source_files={
            "checkpoint": checkpoint,
            "dataset_summary": summary_path,
            "gradient_gate": gradient_path,
            "raw_generations": overfit_raw,
            "resolved_config": config_path,
            "train_history": history_path,
            "trainable_manifest": manifest_path,
            "validation_report": validation_path,
        },
    )
    (overfit_dir / "d_minus_one_overfit_validation.json").write_text(
        json.dumps(overfit_report), encoding="utf-8"
    )
    progress_path = overfit_dir / "training_progress_latest.json"
    progress_path.write_text(json.dumps({
        "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        "step": 100,
    }), encoding="utf-8")
    terminal = validate_terminal_checkpoint_provenance(
        inspect_segdesc_checkpoint(checkpoint),
        checkpoint=checkpoint,
        expected_step=100,
        expected_stage="overfit",
        progress_key="training_progress",
        expected_progress_protocol=DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        progress_artifact=progress_path,
        progress_artifact_name="training_progress_latest",
        history_artifact=history_path,
        history_artifact_name="train_history",
    )
    completion = build_training_completion_report(
        protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
        report={
            "output_dir": str(overfit_dir),
            "stage": "overfit",
            "steps": 100,
            "checkpoint_last": str(checkpoint),
            "terminal_checkpoint_audit": terminal,
        },
        required_artifacts={
            "checkpoint_last": checkpoint,
            "dataset_summary": summary_path,
            "resolved_config": config_path,
            "train_history": history_path,
            "training_progress_latest": progress_path,
            "trainable_parameter_manifest": manifest_path,
        },
        optional_artifacts={
            "d_minus_one_overfit_validation": (
                overfit_dir / "d_minus_one_overfit_validation.json"
            ),
        },
    )
    (overfit_dir / "training_report.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    gate = validate_d_minus_one_runs(zero_dir, overfit_dir)
    gate_path = root / "d_minus_one_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return validate_d_minus_one_gate(gate_path)


def synthetic_stage_lineage(
    *,
    seed: int,
    d_minus_one_acceptance: dict,
    target_stage: str = "predicted_mask",
    predicted_predecessors: int = 2,
) -> dict:
    prefixes = {
        "rsicap_caption": ["mmrs_caption"],
        "dior_alignment": ["mmrs_caption", "rsicap_caption"],
        "bridge_auto": ["mmrs_caption", "rsicap_caption", "dior_alignment"],
        "bridge_expert": [
            "mmrs_caption", "rsicap_caption", "dior_alignment", "bridge_auto",
        ],
        "predicted_mask": [
            "mmrs_caption", "rsicap_caption", "dior_alignment", "bridge_auto",
            "bridge_expert",
        ] + ["predicted_mask"] * int(predicted_predecessors),
    }
    stages = prefixes[target_stage]

    def canonical(value) -> str:
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    acceptance_sha = canonical(d_minus_one_acceptance)
    entries = []
    for index, stage in enumerate(stages):
        role = "terminal_last" if stage == "bridge_auto" else "validation_best"
        checkpoint_path = f"/synthetic/{index}-{stage}.pt"
        checkpoint_sha256 = hashlib.sha256(
            f"checkpoint:{seed}:{index}:{stage}".encode()
        ).hexdigest()
        run_completion = {
            "protocol": CHECKPOINT_RUN_COMPLETION_PROTOCOL,
            "passed": True,
            "training_report": {
                "path": f"/synthetic/{index}-{stage}/training_report.json",
                "sha256": hashlib.sha256(
                    f"completion:{seed}:{index}:{stage}".encode()
                ).hexdigest(),
                "bytes": 1,
            },
            "completion_protocol": DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
            "stage": stage,
            "checkpoint_role": role,
            "checkpoint_step": 100,
            "selected_artifact_name": (
                "checkpoint_last" if role == "terminal_last" else "checkpoint_best"
            ),
            "selected_checkpoint": {
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
                "bytes": 1,
            },
            "selection_report": (
                None
                if role == "terminal_last"
                else {
                    "path": f"/synthetic/{index}-{stage}/validation_best.json",
                    "sha256": hashlib.sha256(
                        f"selection:{seed}:{index}:{stage}".encode()
                    ).hexdigest(),
                    "bytes": 1,
                }
            ),
        }
        entries.append({
            "stage": stage,
            "checkpoint_role": role,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "seed": seed,
            "region_encoder": "mgrr",
            "config_sha256": hashlib.sha256(
                f"config:{seed}:{index}:{stage}".encode()
            ).hexdigest(),
            "controlled_config_sha256": hashlib.sha256(
                f"controlled:{index}:{stage}".encode()
            ).hexdigest(),
            "data_audit_sha256": hashlib.sha256(
                f"data:{index}:{stage}".encode()
            ).hexdigest(),
            "region_data_audit_sha256": hashlib.sha256(
                f"region:{index}:{stage}".encode()
            ).hexdigest(),
            "d_minus_one_acceptance_sha256": acceptance_sha,
            "run_completion": run_completion,
            "run_completion_sha256": canonical(run_completion),
        })
    return {
        "protocol": DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
        "target_stage": target_stage,
        "entries": entries,
        "lineage_sha256": canonical(entries),
    }


def write_synthetic_retention_baseline(
    root: Path,
    *,
    population: dict,
    checkpoint: Path | None = None,
) -> tuple[dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint or root / "segmentation.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"accepted-segmentation-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "checkpoint_step": 6000,
        "threshold": 0.5,
        "coverage": {
            "num_samples": 10,
            "sample_population": population,
        },
        "prediction_population": (
            synthetic_segmentation_prediction_population("baseline")
        ),
        "metrics": {"positive_only": {"dice": 0.5}},
        "threshold_sweep": {"overall_by_threshold": {}},
    }
    report_path = root / "eval_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    report_binding = {
        "protocol": SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL,
        "path": str(report_path.resolve(strict=False)),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "bytes": int(report_path.stat().st_size),
        "prediction_population_sha256": report[
            "prediction_population"
        ]["sha256"],
        "eval_threshold": 0.5,
        "threshold_sweep": [],
    }
    (root / "eval_manifest.json").write_text(json.dumps({
        "protocol": SEGMENTATION_EVAL_MANIFEST_PROTOCOL,
        "created_by": "qpsalm-eval",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": 6000,
        "split": "val",
        "preset": "qwen_psalm_full",
        "resolved_config": {
            "instruction_ablation": "normal",
            "visual_ablation": "normal",
            "eval_threshold": 0.5,
            "threshold_sweep": [],
        },
        "eval_report_binding": report_binding,
    }, sort_keys=True), encoding="utf-8")
    return report, baseline_eval_binding(report_path, report, split="val")


def synthetic_retention_segmentation_checkpoint(root: Path) -> Path:
    checkpoint = root / "shared_predicted_artifacts/fixed_segmentation.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": SEGMENTATION_CHECKPOINT_FORMAT,
        "step": 6000,
    }, checkpoint)
    return checkpoint


def write_synthetic_m6_acceptance(
    root: Path,
    *,
    seed: int,
    bridge_root: Path,
    d_minus_one_root: Path,
    predicted_artifact_root: Path | None = None,
) -> tuple[Path, dict, Path, dict]:
    """Create a deeply revalidatable GT/fixed/end-to-end M6 gate."""
    root.mkdir(parents=True, exist_ok=True)
    d_minus_one = write_synthetic_d_minus_one_acceptance(d_minus_one_root)
    segmentation_index = bridge_root.parent / "segmentation_instruction_val.jsonl"
    segmentation_rows = [{
        "schema_version": "qpsalm_instruction_v2",
        "sample_id": "seg-global-1",
        "parent_sample_id": "parent-1",
        "split": "val",
        "source_level": "patch",
        "task_family": "global_landslide_segmentation",
        "quality_flags": [],
        "mask": {"positive_pixels": 10, "empty_mask": False},
        "modalities": {"optical": {"available": True}},
    }]
    segmentation_index.write_text(
        "".join(json.dumps(row) + "\n" for row in segmentation_rows),
        encoding="utf-8",
    )
    segmentation_source_binding = build_segmentation_instruction_source_binding(
        SimpleNamespace(
            task_families=["global_landslide_segmentation"],
            index_path=lambda _split: segmentation_index,
        ),
        "val",
        segmentation_rows,
    )
    resolver = EndToEndTargetResolver(segmentation_rows)

    (
        _gt_transition_gate,
        _gt_transition_payload,
        gt_checkpoint,
        _expert_gate_audit,
        _gt_train_audit,
        _gt_val_audit,
    ) = write_synthetic_d4_curriculum_gate(
        root / "gt_source",
        current_fraction=0.0,
        next_fraction=0.25,
        seed=seed,
        bridge_root=bridge_root,
        d_minus_one_acceptance=d_minus_one,
        build_curriculum_gate=False,
    )
    gt_eval = root / "gt_source/evaluation"
    gt_raw = gt_eval / "raw_generations.jsonl"
    gt_report_path = gt_eval / "eval_report.json"
    gt_expert_path = gt_eval / "expert_factuality_report.json"
    gt_row = json.loads(gt_raw.read_text(encoding="utf-8"))
    gt_row["region_source"] = "gt_global_mask"
    gt_row["region_id"] = "global"
    gt_raw.write_text(json.dumps(gt_row) + "\n", encoding="utf-8")
    gt_report = json.loads(gt_report_path.read_text(encoding="utf-8"))
    gt_report["generation_coverage"] = {
        "requested": 0,
        "eligible_samples": 1,
        "generated_samples": 1,
        "fraction": 1.0,
        "complete": True,
        "population_sha256": evaluation_population_sha256([gt_row]),
        "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
    }
    gt_report["region_source_filter_audit"] = {
        "protocol": "qpsalm_description_region_source_filter_v1",
        "region_source": "gt_global_mask",
        "rows_before_filter": 1,
        "rows_after_filter": 1,
        "excluded_rows": 0,
        "population_sha256": evaluation_region_source_population_sha256(
            [gt_row]
        ),
    }
    gt_cycle_path = gt_eval / "cycle_localization.jsonl"
    gt_mapping = resolver.resolve(gt_row)
    raw_gt_generation = str(gt_row["raw_generation"])
    cycle_target_mask = np.zeros((4, 4), dtype=np.uint8)
    cycle_target_mask.reshape(-1)[:10] = 1
    cycle_prediction_mask = np.zeros((4, 4), dtype=np.uint8)
    cycle_prediction_mask.reshape(-1)[:8] = 1
    cycle_prediction_artifact = write_evaluation_mask_artifact(
        gt_eval,
        role="cycle_prediction",
        sample_id="sample-1",
        mask=cycle_prediction_mask,
    )
    cycle_target_artifact = write_evaluation_mask_artifact(
        gt_eval,
        role="cycle_target",
        sample_id="sample-1",
        mask=cycle_target_mask,
    )
    cycle_source_artifact = write_evaluation_mask_artifact(
        gt_eval,
        role="cycle_source",
        sample_id="sample-1",
        mask=cycle_prediction_mask,
    )
    cycle_valid_artifact = write_evaluation_mask_artifact(
        gt_eval,
        role="cycle_valid",
        sample_id="sample-1",
        mask=np.ones((4, 4), dtype=np.uint8),
    )
    gt_cycle_path.write_text(json.dumps({
        "sample_id": "sample-1",
        "parent_sample_id": "parent-1",
        "region_iou": 0.8,
        "intersection_pixels": 8,
        "union_pixels": 10,
        "target_pixels": 10,
        "predicted_pixels": 8,
        "target_empty": False,
        "prediction_empty": False,
        "empty_target_correct": False,
        "prediction_mask_artifact": cycle_prediction_artifact,
        "target_mask_artifact": cycle_target_artifact,
        "source_mask_artifact": cycle_source_artifact,
        "valid_mask_artifact": cycle_valid_artifact,
        "cycle_audit": {
            "protocol": CYCLE_PROMPT_PROTOCOL,
            "target_mapping": gt_mapping,
            "mask_threshold": 0.5,
            "generated_text_sha256": hashlib.sha256(
                raw_gt_generation.encode("utf-8")
            ).hexdigest(),
            "generated_text_characters": len(raw_gt_generation),
            "segmentation_resize_transform": {"target_size": 16},
            "description_render_transform": {
                "source_h": 4,
                "source_w": 4,
                "resized_h": 4,
                "resized_w": 4,
                "pad_top": 0,
                "pad_left": 0,
                "size": 4,
            },
        },
    }) + "\n", encoding="utf-8")
    gt_report["cycle_localization"] = {
        "protocol": CYCLE_LOCALIZATION_PROTOCOL,
        "role": "auxiliary_self_consistency_only",
        "source_bridge_rows": 1,
        "eligible_bridge_rows": 1,
        "requested": 0,
        "target_evaluations": 1,
        "coverage_complete": True,
        "evaluated_samples": 1,
        "evaluated_parents": 1,
        "parent_macro_region_iou": 0.8,
        "segmentation_source_binding": segmentation_source_binding,
    }
    gt_report["evaluation_mask_artifacts"] = evaluation_mask_artifact_inventory([
        gt_row["region_input_mask_artifact"],
        cycle_prediction_artifact,
        cycle_target_artifact,
        cycle_source_artifact,
        cycle_valid_artifact,
    ])
    gt_report.pop("publication_audit", None)
    gt_report["publication_audit"] = build_evaluation_publication_audit(
        gt_eval, gt_report
    )
    gt_report_path.write_text(json.dumps(gt_report), encoding="utf-8")
    gt_expert_path = write_synthetic_expert_factuality(gt_eval)

    gt_checkpoint_sha256 = hashlib.sha256(gt_checkpoint.read_bytes()).hexdigest()
    (
        d4_gate,
        _d4_payload,
        d4_checkpoint,
        _d4_expert_gate_audit,
        d4_train_audit,
        _d4_val_audit,
    ) = write_synthetic_d4_curriculum_gate(
        root / "d4_final",
        current_fraction=0.75,
        next_fraction=None,
        seed=seed,
        bridge_root=bridge_root,
        d_minus_one_acceptance=d_minus_one,
        m4_candidate_checkpoint_sha256=gt_checkpoint_sha256,
        predicted_artifact_root=predicted_artifact_root,
    )
    fixed_eval = root / "d4_final/evaluation"
    fixed_expert = fixed_eval / "expert_factuality_report.json"

    end_eval = root / "end_to_end"
    end_eval.mkdir()
    fixed_row = json.loads(
        (fixed_eval / "raw_generations.jsonl").read_text(encoding="utf-8")
    )
    end_row = dict(fixed_row)
    end_row["evaluation_mode"] = "end_to_end"
    end_row["region_source"] = "gt_global_mask"
    end_row["region_id"] = "global"
    end_region_mask = np.zeros((4, 4), dtype=np.uint8)
    end_region_mask.reshape(-1)[:10] = 1
    end_region_artifact = write_evaluation_mask_artifact(
        end_eval,
        role="region_input",
        sample_id="sample-1",
        mask=end_region_mask,
    )
    end_source_artifact = write_evaluation_mask_artifact(
        end_eval,
        role="end_to_end_source",
        sample_id="sample-1",
        mask=end_region_mask,
    )
    end_source_binding = {
        **dict(end_row["region_input_source_binding"]),
        "region_id": "global",
        "region_source": "gt_global_mask",
        "source_mask": {
            "kind": "evaluation_artifact",
            "artifact": end_source_artifact,
            "shape": list(end_source_artifact["shape"]),
            "positive_pixels": int(end_source_artifact["positive_pixels"]),
        },
    }
    end_mapping = {
        **resolver.resolve(end_row),
        "mask_threshold": 0.5,
        "segmentation_resize_transform": {"target_size": 16},
        "description_render_transform": {"target_size": 16},
        "original_mask_shape": [16, 16],
        "region_input_mask_artifact": end_region_artifact,
        "region_input_source_binding": end_source_binding,
    }
    end_row["end_to_end_segmentation_target"] = end_mapping
    end_row["region_input_mask_artifact"] = end_region_artifact
    end_row["region_input_source_binding"] = end_source_binding
    end_row["region_area_fraction"] = float(end_region_mask.mean())
    end_raw = end_eval / "raw_generations.jsonl"
    end_raw.write_text(json.dumps(end_row) + "\n", encoding="utf-8")
    fixed_report = json.loads(
        (fixed_eval / "eval_report.json").read_text(encoding="utf-8")
    )
    end_report = dict(fixed_report)
    end_report["stage"] = "bridge_expert"
    end_report["evaluation_mode"] = "end_to_end"
    end_report["predicted_index_audit"] = None
    end_report["region_source_filter_audit"] = {
        "protocol": "qpsalm_description_region_source_filter_v1",
        "region_source": "gt_global_mask",
        "rows_before_filter": 1,
        "rows_after_filter": 1,
        "excluded_rows": 0,
        "population_sha256": evaluation_region_source_population_sha256(
            [end_row]
        ),
    }
    end_report["generation_coverage"] = {
        "requested": 0,
        "eligible_samples": 1,
        "generated_samples": 1,
        "fraction": 1.0,
        "complete": True,
        "population_sha256": evaluation_population_sha256([end_row]),
        "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
    }
    end_report["evaluation_mask_artifacts"] = evaluation_mask_artifact_inventory(
        [end_region_artifact, end_source_artifact]
    )
    end_report["end_to_end_coverage"] = {
        "protocol": END_TO_END_TARGET_PROTOCOL,
        "source_bridge_rows": 1,
        "eligible_bridge_rows_before_limit": 1,
        "evaluated_rows": 1,
        "excluded_by_reason": {},
        "mapping_counts": {"global_instruction": 1},
        "unique_segmentation_inferences": 1,
        "mask_threshold": 0.5,
        "segmentation_source_binding": segmentation_source_binding,
    }
    end_binding = dict(end_report["checkpoint_binding"])
    end_binding.update({
        "evaluation_mode": "end_to_end",
        "evaluation_data_stage": "bridge_expert",
    })
    end_report["checkpoint_binding"] = end_binding
    end_report_path = end_eval / "eval_report.json"
    (end_eval / "counterfactual_generations.jsonl").write_text(
        (fixed_eval / "counterfactual_generations.jsonl").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (end_eval / "end_to_end_target_audit.jsonl").write_text(
        json.dumps(end_mapping) + "\n", encoding="utf-8"
    )
    preview_source = fixed_eval / "review_preview.png"
    preview_target = end_eval / "review_preview.png"
    preview_target.write_bytes(preview_source.read_bytes())
    end_row["visual_preview_path"] = str(preview_target)
    end_raw.write_text(json.dumps(end_row) + "\n", encoding="utf-8")
    end_report["generation_coverage"]["population_sha256"] = (
        evaluation_population_sha256([end_row])
    )
    end_report.pop("publication_audit", None)
    end_report["publication_audit"] = build_evaluation_publication_audit(
        end_eval, end_report
    )
    end_report_path.write_text(json.dumps(end_report), encoding="utf-8")
    end_expert_path = write_synthetic_expert_factuality(end_eval)

    gate = build_m6_acceptance_gate(
        gt_evaluation_dir=gt_eval,
        gt_expert_report=gt_expert_path,
        fixed_evaluation_dir=fixed_eval,
        fixed_expert_report=fixed_expert,
        end_to_end_evaluation_dir=end_eval,
        end_to_end_expert_report=end_expert_path,
        bridge_benchmark=bridge_root,
        d4_final_gate=d4_gate,
        seed=seed,
    )
    gate_path = root / "m6_acceptance_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    validate_m6_acceptance_gate(gate_path)
    audit = validate_m6_acceptance_for_m7(
        gate_path,
        seed=seed,
        initialize_from=d4_checkpoint,
        train_region_data_audit=d4_train_audit,
    )
    return gate_path, audit, d4_checkpoint, d4_train_audit


def synthetic_joint_execution_fields(
    *, step: int, seed: int = 42, population_variant: str = "",
) -> dict:
    pattern = (
        "segmentation", "global_caption", "segmentation", "region_description",
    )
    tasks = ("segmentation", "global_caption", "region_description")
    bindings = {}
    for index, task in enumerate(tasks):
        binding = {
            "protocol": JOINT_LOADER_BINDING_PROTOCOL,
            "task": task,
            "dataset": {
                "num_rows": 10 + index,
                "ordered_rows_sha256": hashlib.sha256(
                    f"{task}{population_variant}".encode()
                ).hexdigest(),
                "num_children": 0,
            },
            "batches_per_epoch": 7 + index,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "loader_seed": int(seed) + JOINT_LOADER_SEED_OFFSETS[task],
            "worker_seed_protocol": "loader_seed_plus_1000003_times_epoch",
            "batch_sampler": {
                "class": "SyntheticBatchSampler",
                "protocol": "synthetic_epoch_addressable_v1",
                "batch_size": 1,
                "seed": 42 + index,
                "drop_last": False,
                "shuffle": True,
                "balance_tasks": task == "segmentation",
                "task_weights": None,
            },
        }
        binding["binding_sha256"] = hashlib.sha256(json.dumps(
            binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        bindings[task] = binding
    task_steps = {
        task: sum(pattern[index % len(pattern)] == task for index in range(step))
        for task in tasks
    }
    loader_states = {}
    for task in tasks:
        total = task_steps[task]
        epoch, cursor = divmod(total, bindings[task]["batches_per_epoch"])
        loader_states[task] = {
            "epoch": epoch,
            "batch_in_epoch": cursor,
            "total_microbatches": total,
        }
    populations = {
        task: {f"{task}-parent{population_variant}"} for task in tasks
    }
    progress = _joint_progress_payload(
        step=step,
        task_steps=task_steps,
        task_samples=task_steps,
        parent_coverage={task: set(value) for task, value in populations.items()},
        parent_populations=populations,
        loader_states=loader_states,
        loader_bindings=bindings,
        task_pattern=pattern,
        grad_accum_steps=1,
    )
    return {
        "joint_run_protocol": JOINT_RUN_PROTOCOL,
        "joint_loader_bindings": bindings,
        "joint_progress": progress,
        "config": {
            "seed": int(seed),
            "joint_task_pattern": list(pattern),
            "grad_accum_steps": 1,
        },
    }


def write_synthetic_retention_gate(
    root: Path,
    *,
    seed: int,
    baseline: dict,
    baseline_binding: dict,
    population: dict,
    joint_dice: float = 0.495,
    checkpoint_bytes: bytes | None = None,
    cache_variant: str = "",
    joint_population_variant: str = "",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (
        m6_gate,
        m6_acceptance,
        d4_checkpoint,
        region_data_audit,
    ) = write_synthetic_m6_acceptance(
        root / "accepted_m6",
        seed=seed,
        bridge_root=root.parent / "shared_m6_bridge",
        d_minus_one_root=root.parent / "shared_d_minus_one",
        predicted_artifact_root=root.parent / "shared_predicted_artifacts",
    )
    d4_final_acceptance = m6_acceptance["d4_final_acceptance"]
    d_minus_one_acceptance = m6_acceptance["d_minus_one_acceptance"]
    stage_lineage = synthetic_stage_lineage(
        seed=seed,
        d_minus_one_acceptance=d_minus_one_acceptance,
    )
    migration = inspect_segdesc_checkpoint(d4_checkpoint)[
        "checkpoint_metadata"
    ]["segmentation_migration"]
    joint_initialization_audit = build_joint_initialization_audit(
        d4_checkpoint,
        expected_seed=seed,
        region_stage="predicted_mask",
        region_data_audit=region_data_audit,
        d4_final_acceptance=d4_final_acceptance,
        m6_acceptance=m6_acceptance,
        segmentation_migration=migration,
        source_step=100,
        require_m6_binding=True,
    )
    checkpoint = root / "checkpoint_best.pt"
    candidate = {
        "threshold": 0.5,
        "coverage": {
            "num_samples": 10,
            "sample_population": population,
        },
        "prediction_population": synthetic_segmentation_prediction_population(
            f"joint:{seed}"
        ),
        "metrics": {"positive_only": {"dice": joint_dice}},
    }
    checkpoint_metadata = {
        "description_protocol_assets": description_protocol_assets_spec(),
        "description_architecture_spec": {
            "description_cache_artifact_binding": (
                write_synthetic_description_cache_binding(
                    root / ".checkpoint_best_description_cache",
                    variant=cache_variant,
                )
            ),
        },
        "metadata": {
            "stage": "joint",
            "checkpoint_role": "validation_best",
            **synthetic_joint_execution_fields(
                step=100,
                seed=seed,
                population_variant=joint_population_variant,
            ),
            "config": {
                "seed": seed,
                "description_benchmark": d_minus_one_acceptance[
                    "description_source"
                ]["benchmark_root"],
                "joint_region_stage": "predicted_mask",
                "predicted_mask_fraction": 0.75,
                "d4_curriculum_sampling_seed": 42,
                "d4_final_acceptance_gate": str(
                    m6_acceptance["d4_final_acceptance"]["gate"]
                ),
                "m6_acceptance_gate": str(m6_gate),
                "output_dir": str(root),
                "grad_accum_steps": 1,
            },
            "region_data_audit": region_data_audit,
            "d4_final_acceptance": d4_final_acceptance,
            "m6_acceptance": m6_acceptance,
            "joint_initialization_audit": joint_initialization_audit,
            "segmentation_migration_lineage": joint_initialization_audit[
                "segmentation_migration_lineage"
            ],
            "d_minus_one_acceptance": d_minus_one_acceptance,
            "stage_lineage": stage_lineage,
        },
        "segmentation_migration": migration,
    }
    if checkpoint_bytes is None:
        write_synthetic_segdesc_checkpoint(
            checkpoint,
            checkpoint_metadata,
            step=100,
            state_token=float(seed),
        )
        joint_run_completion = publish_synthetic_joint_run_completion(
            checkpoint,
            step=100,
        )
    else:
        # 用于证明重复/伪造 artifact 会在 payload 重放阶段被拒绝。
        checkpoint.write_bytes(checkpoint_bytes)
        joint_run_completion = None
    baseline_replay_path = root / "baseline_segmentation_replay.json"
    baseline_replay_path.write_text(
        json.dumps(baseline, sort_keys=True), encoding="utf-8"
    )
    baseline_checkpoint_replay_audit = build_baseline_checkpoint_replay_audit(
        baseline,
        baseline,
        baseline_binding=baseline_binding,
        segmentation_migration=migration,
        replay_report_path=baseline_replay_path,
    )
    gate = build_retention_gate(
        baseline,
        candidate,
        split="val",
        max_samples=0,
        checkpoint=str(checkpoint),
        checkpoint_step=100,
        checkpoint_metadata=checkpoint_metadata,
        maximum_allowed_drop=0.01,
        baseline_binding=baseline_binding,
        expected_seed=seed,
        d4_final_acceptance_audit=d4_final_acceptance,
        m6_acceptance_audit=m6_acceptance,
        joint_initialization_audit=joint_initialization_audit,
        d_minus_one_acceptance_audit=d_minus_one_acceptance,
        stage_lineage_audit=stage_lineage,
        baseline_checkpoint_replay_audit=baseline_checkpoint_replay_audit,
    )
    gate["joint_checkpoint_sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    gate["joint_run_completion_audit"] = joint_run_completion
    report_path = root / "joint_segmentation_eval.json"
    report_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
    gate = bind_joint_evaluation_report(
        gate,
        eval_report_path=report_path,
        checkpoint_path=checkpoint,
    )
    gate_path = root / "retention_gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    return gate_path


class FakePeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ParameterDict({
            "default": nn.Parameter(torch.ones(2, 2)),
            "desc_adapter": nn.Parameter(torch.ones(2, 2)),
        })
        self.active_adapters = ("default",)
        self.peft_config = {"default": object(), "desc_adapter": object()}

    def set_adapter(self, name: str) -> None:
        self.active_adapters = (name,)
        for key, parameter in self.lora_A.items():
            parameter.requires_grad_(key == name)


class AdapterScopeHarness:
    adapter_scope = QwenMaskQueryController.adapter_scope

    def __init__(self) -> None:
        self.model = FakePeftModel()

    def ensure_named_adapter(self, _name: str) -> None:
        return None


class FakeSegDescCheckpointModel(nn.Module):
    def __init__(self, region_encoder: str) -> None:
        super().__init__()
        self.region_encoder_name = region_encoder
        self.shared = nn.Linear(4, 4)
        self.adapter_bank = nn.ModuleDict({
            "desc_adapter": nn.ModuleDict({
                "lora_A": nn.Linear(4, 4, bias=False),
            }),
        })
        self.mgrr = nn.Linear(4, 4) if region_encoder == "mgrr" else nn.Sequential(
            nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4)
        )
        self.segmentation = SimpleNamespace(config=SimpleNamespace(decoder_dim=4))
        self.controller = SimpleNamespace(
            model=SimpleNamespace(peft_config={"default": object(), "desc_adapter": object()})
        )


class FakeTokenizer:
    eos_token_id = 3

    def __call__(self, _text: str, *, add_special_tokens: bool = False) -> dict:
        del add_special_tokens
        return {"input_ids": [1, 2]}

    @staticmethod
    def decode(token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(value) for value in token_ids)


class SequenceProtocolHarness(nn.Module):
    _token_ids = SegmentationGroundedDescriptionModel._token_ids
    _instruction_prompt = staticmethod(SegmentationGroundedDescriptionModel._instruction_prompt)
    _visual_tokens_for_sample = SegmentationGroundedDescriptionModel._visual_tokens_for_sample
    _build_sequences = SegmentationGroundedDescriptionModel._build_sequences

    def __init__(self) -> None:
        super().__init__()
        language_model = nn.Module()
        language_model.embedding = nn.Embedding(8, 4)
        language_model.get_input_embeddings = lambda: language_model.embedding
        self.controller = SimpleNamespace(model=language_model, tokenizer=FakeTokenizer())
        self.description_view_to_hidden = nn.Linear(3, 4, bias=False)
        self.region_to_hidden = nn.Linear(2, 4, bias=False)
        self.instruction_type = nn.Parameter(torch.zeros(4))
        self.visual_type = nn.Parameter(torch.zeros(4))
        self.region_type = nn.Parameter(torch.zeros(4))


class FakeAutoregressiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.generated_tokens = (4, 5, 3)
        self.calls: list[dict] = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")
        length = int(
            inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        )
        token = self.generated_tokens[min(len(self.calls), len(self.generated_tokens) - 1)]
        logits = self.embedding.weight.new_full((1, length, 8), -100.0)
        logits[0, -1, token] = 100.0
        self.calls.append({
            "used_inputs_embeds": inputs_embeds is not None,
            "used_input_ids": input_ids is not None,
            "attention_length": int(kwargs["attention_mask"].shape[1]),
            "used_cache": kwargs.get("past_key_values") is not None,
        })
        return SimpleNamespace(logits=logits, past_key_values=("synthetic-cache",))


class GenerationController:
    def __init__(self) -> None:
        self.model = FakeAutoregressiveModel()
        self.tokenizer = FakeTokenizer()
        self.adapter_calls: list[str] = []

    @contextmanager
    def adapter_scope(self, name: str):
        self.adapter_calls.append(name)
        yield


class GenerationProtocolHarness(SequenceProtocolHarness):
    generate_from_state = SegmentationGroundedDescriptionModel.generate_from_state

    def __init__(self) -> None:
        super().__init__()
        self.controller = GenerationController()
        self.region_state = SimpleNamespace(
            backbone=SimpleNamespace(visual_evidence=None),
            region_sequence_tokens=torch.ones(1, 1, 2, 2),
            region_sequence_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            region_tokens=None,
        )

    def _description_region_state(self, *_args, **_kwargs):
        return self.region_state


class StageParameterHarness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segmentation = nn.Module()
        self.segmentation.controller = nn.Module()
        self.segmentation.controller.lora_A = nn.Module()
        self.segmentation.controller.lora_A.default = nn.Linear(2, 2, bias=False)
        self.segmentation.controller.lora_A.desc_adapter = nn.Linear(2, 2, bias=False)
        self.description_backbone = nn.Linear(2, 2)
        self.mgrr = nn.Linear(2, 2)
        self.region_to_hidden = nn.Linear(2, 2)
        self.description_view_to_hidden = nn.Linear(2, 2)
        self.alignment_text_projection = nn.Linear(2, 2)
        self.region_type = nn.Parameter(torch.zeros(2))
        self.instruction_type = nn.Parameter(torch.zeros(2))
        self.visual_type = nn.Parameter(torch.zeros(2))
        self.alignment_temperature = nn.Parameter(torch.tensor(0.07))


class RegionBypassHarness:
    _description_region_state = SegmentationGroundedDescriptionModel._description_region_state

    def __init__(self) -> None:
        self.segmentation = SimpleNamespace(config=SimpleNamespace(decoder_dim=2))

    def build_region_state(self, *_args, **_kwargs):
        raise RuntimeError("region replay executed")


class SegDescProtocolTest(unittest.TestCase):
    def test_output_replacement_cannot_delete_bound_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            nested = output / "source.jsonl"
            sibling = root / "source.jsonl"
            nested.parent.mkdir(parents=True)
            nested.write_text("{}\n", encoding="utf-8")
            sibling.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                validate_output_replacement_safety(
                    output, {"sibling": sibling, "optional": None}
                )["sibling"],
                str(sibling.resolve(strict=False)),
            )
            with self.assertRaisesRegex(ValueError, "路径重叠"):
                validate_output_replacement_safety(
                    output, {"nested-source": nested}
                )
            with self.assertRaisesRegex(ValueError, "路径重叠"):
                validate_output_replacement_safety(
                    nested, {"same-source": nested}
                )
            with self.assertRaisesRegex(ValueError, "路径重叠"):
                validate_output_replacement_safety(
                    root / "benchmark/cache", {"benchmark": root / "benchmark"}
                )

    def test_resume_run_reconciles_uncheckpointed_history_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "checkpoint_best.pt"
            last = root / "checkpoint_last.pt"
            best.write_bytes(b"best-step-4")
            last.write_bytes(b"last-step-5")
            history = root / "train_history.jsonl"
            original = b"".join(
                (json.dumps({"step": step, "loss": float(step)}) + "\n").encode()
                for step in (1, 4, 6)
            )
            history.write_bytes(original)

            steps = {best.name: 4, last.name: 5}
            audit = reconcile_resume_run(
                root,
                resume_checkpoint=last,
                checkpoint_step=5,
                histories={"train_history.jsonl": True},
                checkpoint_step_reader=lambda path: steps[path.name],
            )
            self.assertEqual(
                json.loads((root / "resume_reconciliation.json").read_text())["protocol"],
                RESUME_RECONCILIATION_PROTOCOL,
            )
            history_audit = audit["histories"]["train_history.jsonl"]
            self.assertEqual(history_audit["rows_retained"], 2)
            self.assertEqual(history_audit["rows_archived"], 1)
            self.assertEqual(
                [json.loads(line)["step"] for line in history.read_text().splitlines()],
                [1, 4],
            )
            archive = Path(history_audit["archive"]["path"])
            self.assertEqual(archive.read_bytes(), original)

    def test_resume_run_rejects_older_sibling_checkpoint_and_bad_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "checkpoint_best.pt"
            last = root / "checkpoint_last.pt"
            best.write_bytes(b"best")
            last.write_bytes(b"last")
            (root / "train_history.jsonl").write_text(
                '{"step": 2}\n{"step": 2}\n', encoding="utf-8"
            )
            steps = {best.name: 2, last.name: 3}
            with self.assertRaisesRegex(RuntimeError, "最新可恢复状态"):
                reconcile_resume_run(
                    root,
                    resume_checkpoint=best,
                    checkpoint_step=2,
                    histories={"train_history.jsonl": True},
                    checkpoint_step_reader=lambda path: steps[path.name],
                )
            with self.assertRaisesRegex(ValueError, "严格递增"):
                reconcile_resume_run(
                    root,
                    resume_checkpoint=last,
                    checkpoint_step=3,
                    histories={"train_history.jsonl": True},
                    checkpoint_step_reader=lambda path: steps[path.name],
                )

    def test_training_attempt_archives_failure_and_completion_binds_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = root / "failure_report.json"
            failure.write_text(
                json.dumps({"protocol": "failure", "message": "interrupted"}),
                encoding="utf-8",
            )
            attempt = prepare_training_attempt(root, resume=True)
            self.assertIsNotNone(attempt["archived_failure"])
            self.assertFalse(failure.exists())
            self.assertTrue((root / "failure_history.json").is_file())

            checkpoint = root / "checkpoint_last.pt"
            history = root / "train_history.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            history.write_text('{"step": 1}\n', encoding="utf-8")
            completion = build_training_completion_report(
                protocol="completion-v2",
                report={"steps": 1, "checkpoint_last": str(checkpoint)},
                required_artifacts={
                    "checkpoint_last": checkpoint,
                    "train_history": history,
                },
                optional_artifacts={"missing_optional": root / "missing.json"},
            )
            self.assertEqual(completion["terminal_status"], "completed")
            self.assertEqual(
                completion["artifacts"]["checkpoint_last"]["bytes"],
                len(b"checkpoint"),
            )
            self.assertIsNone(completion["artifacts"]["missing_optional"])
            provenance = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "checkpoint_step": 1,
                "checkpoint_metadata": {
                    "metadata": {
                        "stage": "overfit",
                        "checkpoint_role": "terminal_last",
                        "training_progress": {
                            "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
                            "step": 1,
                        },
                    },
                },
            }
            progress_artifact = root / "training_progress_latest.json"
            progress_artifact.write_text(
                json.dumps(
                    provenance["checkpoint_metadata"]["metadata"][
                        "training_progress"
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            terminal = validate_terminal_checkpoint_provenance(
                provenance,
                checkpoint=checkpoint,
                expected_step=1,
                expected_stage="overfit",
                progress_key="training_progress",
                expected_progress_protocol=(
                    DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
                ),
                progress_artifact=progress_artifact,
                progress_artifact_name="training_progress_latest",
                history_artifact=history,
                history_artifact_name="train_history",
            )
            self.assertEqual(
                terminal["protocol"], TERMINAL_CHECKPOINT_AUDIT_PROTOCOL
            )
            wrong_role = copy.deepcopy(provenance)
            wrong_role["checkpoint_metadata"]["metadata"][
                "checkpoint_role"
            ] = "validation_best"
            with self.assertRaisesRegex(RuntimeError, "terminal_last"):
                validate_terminal_checkpoint_provenance(
                    wrong_role,
                    checkpoint=checkpoint,
                    expected_step=1,
                    expected_stage="overfit",
                    progress_key="training_progress",
                    expected_progress_protocol=(
                        DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
                    ),
                    progress_artifact=progress_artifact,
                    progress_artifact_name="training_progress_latest",
                    history_artifact=history,
                    history_artifact_name="train_history",
                )
            bound_terminal = build_training_completion_report(
                protocol="completion-v3",
                report={"steps": 1, "terminal_checkpoint_audit": terminal},
                required_artifacts={
                    "checkpoint_last": checkpoint,
                    "training_progress_latest": progress_artifact,
                    "train_history": history,
                },
            )
            self.assertEqual(
                bound_terminal["terminal_checkpoint_audit"][
                    "checkpoint_sha256"
                ],
                bound_terminal["artifacts"]["checkpoint_last"]["sha256"],
            )
            mismatched_terminal = copy.deepcopy(terminal)
            mismatched_terminal["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "artifact binding"):
                build_training_completion_report(
                    protocol="completion-v3",
                    report={
                        "steps": 1,
                        "terminal_checkpoint_audit": mismatched_terminal,
                    },
                    required_artifacts={"checkpoint_last": checkpoint},
                )
            drifted = copy.deepcopy(provenance)
            drifted["checkpoint_metadata"]["metadata"][
                "training_progress"
            ]["step"] = 2
            with self.assertRaisesRegex(RuntimeError, "progress"):
                validate_terminal_checkpoint_provenance(
                    drifted,
                    checkpoint=checkpoint,
                    expected_step=1,
                    expected_stage="overfit",
                    progress_key="training_progress",
                    expected_progress_protocol=(
                        DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
                    ),
                    progress_artifact=progress_artifact,
                    progress_artifact_name="training_progress_latest",
                    history_artifact=history,
                    history_artifact_name="train_history",
                )
            with self.assertRaisesRegex(ValueError, "保留字段"):
                build_training_completion_report(
                    protocol="completion-v3",
                    report={"steps": 1, "protocol": "forged"},
                    required_artifacts={"checkpoint_last": checkpoint},
                )
            (root / "training_report.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "不允许再次 resume"):
                prepare_training_attempt(root, resume=True)

    def test_caption_human_review_is_blind_complete_and_parent_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(2):
                image = root / f"image_{index}.png"
                image.write_bytes(f"synthetic-image-{index}".encode("utf-8"))
                rows.append({
                    "sample_id": f"rsieval_{index}",
                    "parent_sample_id": f"parent_{index}",
                    "task_family": "global_caption",
                    "source_dataset": "RSIEval",
                    "visual_image_path": str(image),
                    "split": "test",
                    "evaluation_mode": "gt_mask",
                    "instruction": "Describe the image.",
                    "target_text": f"reference {index}",
                    "reference_texts": [f"reference {index}"],
                    "raw_generation": f"prediction {index}",
                })
            raw_path = root / "raw_generations.jsonl"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = {
                "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
                "stage": "rsicap_caption",
                "split": "test",
                "num_samples": len(rows),
                "num_generated": len(rows),
                "evaluation_limit_audit": {
                    "protocol": "qpsalm_description_evaluation_limit_v1",
                    "requested_max_samples": 0,
                    "full_population_requested": True,
                    "dataset_rows_evaluated": len(rows),
                },
                "source_filter_audit": {
                    "protocol": "qpsalm_description_evaluation_source_filter_v1",
                    "stage": "rsicap_caption",
                    "split": "test",
                    "source_dataset": "RSIEval",
                    "rows_before_filter": 3,
                    "rows_after_filter": len(rows),
                },
                "generation_coverage": {
                    "complete": True,
                    "eligible_samples": len(rows),
                    "generated_samples": len(rows),
                    "population_sha256": evaluation_population_sha256(rows),
                    "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
                },
            }
            publish_synthetic_evaluation(root, report)
            templates = build_caption_human_review_template(
                root, expected_samples=2
            )
            self.assertEqual(len(templates), 2)
            self.assertTrue(all(row["reference_target_hidden"] for row in templates))
            self.assertTrue(all("reference_texts" not in row for row in templates))

            review_paths = []
            reviewer_scores = {
                "reviewer_1": {"factuality": 5, "detail": 4, "readability": 5},
                "reviewer_2": {"factuality": 3, "detail": 2, "readability": 4},
            }
            for reviewer, scores in reviewer_scores.items():
                path = root / f"{reviewer}.jsonl"
                completed = [
                    {
                        **json.loads(json.dumps(row)),
                        "reviewer_id": reviewer,
                        "scores": dict(scores),
                    }
                    for row in templates
                ]
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in completed),
                    encoding="utf-8",
                )
                review_paths.append(path)
            report = aggregate_caption_human_reviews(
                root,
                review_paths,
                seed=42,
                expected_samples=2,
            )
            self.assertEqual(report["status"], "expert_review_complete")
            self.assertEqual(report["num_parents"], 2)
            self.assertEqual(report["metrics"]["factuality"]["parent_macro_1_to_5"], 4.0)
            self.assertEqual(report["metrics"]["detail"]["parent_macro_1_to_5"], 3.0)
            self.assertEqual(report["metrics"]["readability"]["parent_macro_1_to_5"], 4.5)
            self.assertFalse(report["checkpoint_selection_allowed"])

            corrupted = [json.loads(line) for line in review_paths[0].read_text(
                encoding="utf-8"
            ).splitlines()]
            corrupted[0]["model_generation"] = "rewritten"
            review_paths[0].write_text(
                "".join(json.dumps(row) + "\n" for row in corrupted),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "冻结字段"):
                aggregate_caption_human_reviews(
                    root,
                    review_paths,
                    seed=42,
                    expected_samples=2,
                )

    def test_caption_metric_population_is_complete_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "sample_id": f"rsieval_{index}",
                    "parent_sample_id": f"parent_{index}",
                    "task_family": "global_caption",
                    "source_dataset": "RSIEval",
                    "split": "test",
                    "evaluation_mode": "gt_mask",
                    "instruction": "Describe the image.",
                    "target_text": f"reference {index}",
                    "reference_texts": [
                        f"reference {index}", f"alternate {index}"
                    ],
                    "raw_generation": f"prediction {index}",
                }
                for index in range(2)
            ]
            raw_path = root / "raw_generations.jsonl"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = {
                "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
                "stage": "rsicap_caption",
                "split": "test",
                "num_samples": len(rows),
                "num_generated": len(rows),
                "evaluation_limit_audit": {
                    "protocol": "qpsalm_description_evaluation_limit_v1",
                    "requested_max_samples": 0,
                    "full_population_requested": True,
                    "dataset_rows_evaluated": len(rows),
                },
                "source_filter_audit": {
                    "protocol": "qpsalm_description_evaluation_source_filter_v1",
                    "stage": "rsicap_caption",
                    "split": "test",
                    "source_dataset": "RSIEval",
                    "rows_before_filter": 3,
                    "rows_after_filter": len(rows),
                },
                "generation_coverage": {
                    "complete": True,
                    "eligible_samples": len(rows),
                    "generated_samples": len(rows),
                    "population_sha256": evaluation_population_sha256(rows),
                    "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
                },
            }
            report_path = root / "eval_report.json"
            publish_synthetic_evaluation(root, report)

            selected, audit = caption_metric_population(
                root, expected_samples=2
            )
            self.assertEqual([row["sample_id"] for row in selected], [
                "rsieval_0", "rsieval_1"
            ])
            self.assertEqual(audit["num_samples"], 2)
            self.assertEqual(audit["num_parents"], 2)
            self.assertEqual(len(audit["metric_population_sha256"]), 64)
            self.assertEqual(
                audit["evaluation_publication_audit"],
                report["publication_audit"],
            )

            report["generation_coverage"]["complete"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "完整 frozen generation"):
                caption_metric_population(root, expected_samples=2)
            report["generation_coverage"]["complete"] = True
            report_path.write_text(json.dumps(report), encoding="utf-8")
            rows[0]["target_text"] = "drifted reference"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "publication artifact/report"):
                caption_metric_population(root, expected_samples=2)

    def test_caption_metric_summary_and_local_model_audit(self) -> None:
        summary = _metric_summary(0.75, [0.5, 1.0], seed=42)
        self.assertEqual(summary["corpus_score"], 0.75)
        self.assertEqual(summary["parent_macro"], 0.75)
        self.assertEqual(summary["parent_macro_bootstrap_95ci"]["n"], 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"synthetic-weights")
            model_dir, audit = _bertscore_model_audit(root)
            self.assertEqual(model_dir, root)
            self.assertIn("config.json", audit["binding_files"])
            self.assertIn("model.safetensors", audit["binding_files"])
            self.assertEqual(len(audit["model_snapshot_sha256"]), 64)

    def test_caption_sampling_is_quality_weighted_and_source_balanced(self) -> None:
        rows = [
            {
                "sample_id": f"nwpu_{index}",
                "source_dataset": "MMRS-NWPU-Caption",
            }
            for index in range(3)
        ] + [{"sample_id": "ucm_0", "source_dataset": "MMRS-UCM-Caption"}]
        weights, audit = _caption_source_weights(
            rows, stage="mmrs_caption", rsicap_mmrs_fraction=0.3
        )
        self.assertAlmostEqual(sum(weights[f"nwpu_{index}"] for index in range(3)), 2.0)
        self.assertAlmostEqual(weights["ucm_0"], 2.0)
        self.assertAlmostEqual(audit["row_weight_mean"], 1.0)
        d1_rows = rows + [
            {"sample_id": "rsicap_0", "source_dataset": "RSICap"},
            {"sample_id": "rsicap_1", "source_dataset": "RSICap"},
        ]
        d1_weights, d1_audit = _caption_source_weights(
            d1_rows, stage="rsicap_caption", rsicap_mmrs_fraction=0.3
        )
        self.assertAlmostEqual(
            sum(d1_weights[row["sample_id"]] for row in d1_rows[:4]), 1.8
        )
        self.assertAlmostEqual(
            sum(d1_weights[row["sample_id"]] for row in d1_rows[4:]), 4.2
        )
        self.assertEqual(d1_audit["group_total_mass"], {"mmrs": 0.3, "rsicap": 0.7})
        for epoch in range(5):
            self.assertEqual(
                _stable_weighted_index(42, epoch, "sample", [0.0, 1.0]), 1
            )

    def test_rsieval_evaluation_source_filter_is_explicit_and_test_only(self) -> None:
        rows = [
            {"sample_id": "eval", "source_dataset": "RSIEval", "task_family": "global_caption"},
            {"sample_id": "cap", "source_dataset": "RSICap", "task_family": "global_caption"},
        ]
        selected, audit = filter_evaluation_source(
            rows,
            stage="rsicap_caption",
            split="test",
            source_dataset="RSIEval",
        )
        self.assertEqual([row["sample_id"] for row in selected], ["eval"])
        self.assertEqual(audit["rows_before_filter"], 2)
        self.assertEqual(audit["rows_after_filter"], 1)
        with self.assertRaisesRegex(ValueError, "split=test"):
            filter_evaluation_source(
                rows,
                stage="rsicap_caption",
                split="dev",
                source_dataset="RSIEval",
            )
        source = Path(
            "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml"
        )
        config = load_segdesc_config(source, {
            "stage": "rsicap_caption",
            "evaluation_source_dataset": "RSIEval",
        })
        self.assertEqual(config.evaluation_source_dataset, "RSIEval")
        with self.assertRaisesRegex(ValueError, "rsicap_caption/RSIEval"):
            load_segdesc_config(source, {
                "stage": "bridge_auto",
                "evaluation_source_dataset": "RSIEval",
            })

    @staticmethod
    def _joint_gradient_role(nonzero: int) -> dict:
        return {
            "num_parameters": 2,
            "num_with_grad": int(nonzero > 0),
            "num_nonzero": nonzero,
            "norm_sum": float(nonzero),
            "all_finite": True,
        }

    def test_joint_gradient_gate_distinguishes_global_and_region_description(self) -> None:
        global_report = {
            "segmentation_adapter": self._joint_gradient_role(0),
            "description_adapter": self._joint_gradient_role(1),
            "mgrr": self._joint_gradient_role(0),
            "description_projection": self._joint_gradient_role(1),
        }
        gate = validate_joint_task_gradients(
            "global_caption", global_report,
            train_shared_segmentation_dense=False,
        )
        self.assertTrue(gate["passed"])
        region_gate = validate_joint_task_gradients(
            "region_description", global_report,
            train_shared_segmentation_dense=False,
        )
        self.assertFalse(region_gate["passed"])
        self.assertEqual(region_gate["missing_or_zero"], ["mgrr"])

        region_report = dict(global_report)
        region_report["mgrr"] = self._joint_gradient_role(1)
        self.assertTrue(validate_joint_task_gradients(
            "region_description", region_report,
            train_shared_segmentation_dense=False,
        )["passed"])
        leaked = dict(global_report)
        leaked["segmentation_adapter"] = self._joint_gradient_role(1)
        leak_gate = validate_joint_task_gradients(
            "global_caption", leaked,
            train_shared_segmentation_dense=False,
        )
        self.assertFalse(leak_gate["passed"])
        self.assertEqual(leak_gate["leaked_nonzero_roles"], ["segmentation_adapter"])

    def test_joint_optimizer_manifest_covers_exact_trainable_parameters(self) -> None:
        model = StageParameterHarness()
        config = SimpleNamespace(
            joint_train_shared_segmentation_dense=False,
            desc_adapter_lr_scale=0.2,
            learning_rate=1.0e-4,
            weight_decay=0.01,
            warmup_steps=2,
            max_steps=10,
        )
        optimizer, _scheduler = build_joint_optimizer(model, config)
        manifest = joint_optimizer_manifest(model, optimizer)
        self.assertEqual(manifest["protocol"], "qpsalm_segdesc_joint_optimizer_v1")
        self.assertEqual(
            {group["role"] for group in manifest["groups"]},
            {"segmentation_adapter", "description_adapter", "mgrr", "description_projection"},
        )
        listed = {
            name for group in manifest["groups"] for name in group["parameter_names"]
        }
        self.assertEqual(
            listed,
            {name for name, value in model.named_parameters() if value.requires_grad},
        )

    @staticmethod
    def _monitor_report(population_hash: str, dice: float) -> dict:
        return {
            "threshold": 0.5,
            "coverage": {
                "num_samples": 2,
                "sample_population": {
                    "protocol": "qpsalm_segmentation_eval_population_v1",
                    "fields": list(SAMPLE_IDENTITY_FIELDS),
                    "sha256": population_hash,
                    "num_records": 2,
                    "num_unique_sample_ids": 2,
                    "complete": True,
                    "unique": True,
                    "incomplete_record_indices": [],
                    "duplicate_sample_ids": [],
                },
            },
            "metrics": {"positive_only": {"dice": dice}},
        }

    def test_joint_monitor_retention_freezes_population_identity(self) -> None:
        baseline = monitor_baseline_identity(self._monitor_report("population-a", 0.60))
        passed = monitor_retention_gate(
            baseline,
            self._monitor_report("population-a", 0.595),
            maximum_allowed_drop=0.01,
        )
        self.assertTrue(passed["passed"])
        changed = monitor_retention_gate(
            baseline,
            self._monitor_report("population-b", 0.80),
            maximum_allowed_drop=0.01,
        )
        self.assertFalse(changed["passed"])
        self.assertFalse(changed["same_sample_population"])

    def test_joint_progress_resume_rejects_changed_population(self) -> None:
        populations = {
            "segmentation": {"s1", "s2"},
            "global_caption": {"g1"},
            "region_description": {"r1", "r2"},
        }
        pattern = ("segmentation", "global_caption", "region_description")
        bindings = {
            task: {
                "batches_per_epoch": 3,
                "binding_sha256": hashlib.sha256(task.encode()).hexdigest(),
            }
            for task in populations
        }
        states = _initial_joint_loader_states(bindings)
        for state in states.values():
            state.update({"batch_in_epoch": 2, "total_microbatches": 2})
        progress = _joint_progress_payload(
            step=3,
            task_steps={"segmentation": 1, "global_caption": 1, "region_description": 1},
            task_samples={"segmentation": 2, "global_caption": 1, "region_description": 2},
            parent_coverage={
                "segmentation": {"s1"},
                "global_caption": {"g1"},
                "region_description": {"r1"},
            },
            parent_populations=populations,
            loader_states=states,
            loader_bindings=bindings,
            task_pattern=pattern,
            grad_accum_steps=2,
        )
        self.assertEqual(progress["protocol"], JOINT_PROGRESS_PROTOCOL)
        steps, samples, coverage, restored_states = restore_joint_progress(
            progress,
            populations,
            bindings,
            checkpoint_step=3,
            required=True,
            task_pattern=pattern,
            grad_accum_steps=2,
        )
        self.assertEqual(sum(steps.values()), 3)
        self.assertEqual(samples["segmentation"], 2)
        self.assertEqual(coverage["region_description"], {"r1"})
        self.assertEqual(restored_states["global_caption"]["batch_in_epoch"], 2)
        changed = dict(populations)
        changed["region_description"] = {"r1", "r3"}
        with self.assertRaisesRegex(RuntimeError, "population"):
            restore_joint_progress(
                progress,
                changed,
                bindings,
                checkpoint_step=3,
                required=True,
                task_pattern=pattern,
                grad_accum_steps=2,
            )
        with self.assertRaisesRegex(RuntimeError, "step"):
            restore_joint_progress(
                progress,
                populations,
                bindings,
                checkpoint_step=4,
                required=True,
                task_pattern=pattern,
                grad_accum_steps=2,
            )
        drifted_bindings = copy.deepcopy(bindings)
        drifted_bindings["global_caption"]["binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "loader binding"):
            restore_joint_progress(
                progress,
                populations,
                drifted_bindings,
                checkpoint_step=3,
                required=True,
                task_pattern=pattern,
                grad_accum_steps=2,
            )
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            restore_joint_progress(
                {**progress, "protocol": "qpsalm_segdesc_joint_progress_v1"},
                populations,
                bindings,
                checkpoint_step=3,
                required=True,
                task_pattern=pattern,
                grad_accum_steps=2,
            )

    def test_joint_execution_replays_parent_coverage_hashes(self) -> None:
        metadata = synthetic_joint_execution_fields(step=100)
        audit = validate_joint_checkpoint_execution(
            metadata, checkpoint_step=100
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["parent_coverage_sha256"]), 64)

        drifted = copy.deepcopy(metadata)
        coverage = drifted["joint_progress"]["parent_coverage"][
            "region_description"
        ]
        coverage["parent_ids"].append("forged-parent")
        with self.assertRaisesRegex(RuntimeError, "parent coverage"):
            validate_joint_checkpoint_execution(
                drifted, checkpoint_step=100
            )

    def test_joint_loader_cursor_resume_matches_uninterrupted_next_batch(self) -> None:
        class RandomRows(Dataset):
            def __init__(self) -> None:
                self.rows = [
                    {"sample_id": f"s{index}", "parent_sample_id": f"p{index}"}
                    for index in range(9)
                ]
                self.epoch = 0

            def set_epoch(self, epoch: int) -> None:
                self.epoch = int(epoch)

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict:
                return {
                    "index": index,
                    "epoch": self.epoch,
                    "python": random.random(),
                    "numpy": float(np.random.random()),
                    "torch": float(torch.rand(())),
                }

        def make_loader() -> DataLoader:
            dataset = RandomRows()
            generator = torch.Generator().manual_seed(1234)
            loader = DataLoader(
                dataset,
                batch_sampler=EpochShuffleBatchSampler(
                    dataset, 2, seed=1234, drop_last=False
                ),
                generator=generator,
            )
            set_loader_epoch(loader, 0, loader_seed=1234)
            return loader

        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        loader = make_loader()
        binding = _joint_loader_binding(
            "global_caption", loader, loader_seed=1234
        )
        state = {"epoch": 0, "batch_in_epoch": 0, "total_microbatches": 0}
        iterator = None
        # Five batches finish epoch 0; two more leave a nonzero cursor in epoch 1.
        for _ in range(7):
            _batch, iterator = _next_joint_loader_batch(
                loader, iterator, state, binding
            )
        saved_state = copy.deepcopy(state)
        saved_rng = capture_training_rng_state()
        expected, _iterator = _next_joint_loader_batch(
            loader, iterator, state, binding
        )

        resumed_loader = make_loader()
        resumed_binding = _joint_loader_binding(
            "global_caption", resumed_loader, loader_seed=1234
        )
        self.assertEqual(binding, resumed_binding)
        restore_training_rng_state(saved_rng)
        observed, _ = _next_joint_loader_batch(
            resumed_loader, None, saved_state, resumed_binding
        )
        for field in ("index", "epoch", "python", "numpy", "torch"):
            self.assertTrue(torch.equal(expected[field], observed[field]), field)

    def test_description_stream_resume_matches_uninterrupted_next_batch(self) -> None:
        class RandomRows(Dataset):
            def __init__(self) -> None:
                self.rows = [
                    {"sample_id": f"d{index}", "parent_sample_id": f"p{index}"}
                    for index in range(9)
                ]
                self.epoch = 0

            def set_epoch(self, epoch: int) -> None:
                self.epoch = int(epoch)

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict:
                return {
                    "index": index,
                    "epoch": self.epoch,
                    "python": random.random(),
                    "numpy": float(np.random.random()),
                    "torch": float(torch.rand(())),
                }

        def make_stream() -> dict:
            dataset = RandomRows()
            loader = DataLoader(
                dataset,
                batch_sampler=EpochShuffleBatchSampler(
                    dataset, 2, seed=2234, drop_last=False
                ),
                generator=torch.Generator().manual_seed(2234),
            )
            set_loader_epoch(loader, 0, loader_seed=2234)
            return {
                "dataset": dataset,
                "loader": loader,
                "config": SimpleNamespace(seed=42, stage="mmrs_caption"),
            }

        audit = {
            "protocol": "synthetic_description_dataset_population_v1",
            "population_sha256": "a" * 64,
            "num_samples": 9,
        }
        random.seed(191)
        np.random.seed(192)
        torch.manual_seed(193)
        stream = make_stream()
        binding = _description_stream_binding("main", stream, audit)
        state = {
            "epoch": 0,
            "batch_in_epoch": 0,
            "total_microbatches": 0,
            "batches_per_epoch": binding["epoch_zero_batches"],
            "completed_epoch_batches": [],
        }
        iterator = None
        for _ in range(7):
            _batch, iterator = _next_description_stream_batch(
                stream, iterator, state, binding
            )
        progress = _description_training_progress_payload(
            step=7,
            stream_pattern=("main",),
            grad_accum_steps=1,
            stream_states={"main": state},
            stream_bindings={"main": binding},
        )
        self.assertEqual(progress["protocol"], DESCRIPTION_TRAINING_PROGRESS_PROTOCOL)
        saved_rng = capture_training_rng_state()
        expected, _ = _next_description_stream_batch(
            stream, iterator, state, binding
        )

        resumed_stream = make_stream()
        resumed_binding = _description_stream_binding(
            "main", resumed_stream, audit
        )
        restored = _restore_description_training_progress(
            progress,
            checkpoint_step=7,
            required=True,
            stream_pattern=("main",),
            grad_accum_steps=1,
            train_streams={"main": resumed_stream},
            stream_bindings={"main": resumed_binding},
        )
        restore_training_rng_state(saved_rng)
        observed, _ = _next_description_stream_batch(
            resumed_stream,
            None,
            restored["main"],
            resumed_binding,
        )
        for field in ("index", "epoch", "python", "numpy", "torch"):
            self.assertTrue(torch.equal(expected[field], observed[field]), field)
        with self.assertRaisesRegex(RuntimeError, "缺少 training_progress"):
            _restore_description_training_progress(
                {},
                checkpoint_step=7,
                required=True,
                stream_pattern=("main",),
                grad_accum_steps=1,
                train_streams={"main": resumed_stream},
                stream_bindings={"main": resumed_binding},
            )

    def test_m7_source_checkpoint_is_bound_to_current_region_data(self) -> None:
        audit = {
            "protocol": REGION_TRAINING_DATA_PROTOCOL,
            "stage": "predicted_mask",
            "expert_gate_audit": {
                "status": "expert_pilot_frozen",
                "candidate_index_sha256": "b" * 64,
            },
            "bridge_engineering_audit": {
                "protocol": BRIDGE_ENGINEERING_AUDIT_PROTOCOL,
                "status": "expert_pilot_frozen",
                "expert_truth_used": False,
                "candidate_index_sha256": "b" * 64,
                "cache_input_fingerprint": {
                    "benchmark": "benchmark/bridge",
                    "index": "indexes/candidate_all.jsonl",
                    "size": 1,
                    "sha256": "b" * 64,
                },
            },
            "predicted_index_audit": {"index_sha256": "a" * 64},
        }
        initialized = validate_m7_source_checkpoint(
            {"metadata": {
                "stage": "predicted_mask",
                "checkpoint_role": "validation_best",
                "region_data_audit": audit,
                "config": {"seed": 42},
            }},
            region_stage="predicted_mask",
            current_data_audit=audit,
            resume=False,
            expected_seed=42,
        )
        self.assertEqual(initialized["source_stage"], "predicted_mask")
        self.assertTrue(initialized["seed_match"])
        resumed = validate_m7_source_checkpoint(
            {"metadata": {
                "stage": "joint",
                "checkpoint_role": "terminal_last",
                "region_data_audit": audit,
                "joint_run_protocol": JOINT_RUN_PROTOCOL,
                "config": {"seed": 42},
            }},
            region_stage="predicted_mask",
            current_data_audit=audit,
            resume=True,
            expected_seed=42,
        )
        self.assertTrue(resumed["resume"])
        with self.assertRaisesRegex(RuntimeError, "run seed"):
            validate_m7_source_checkpoint(
                {"metadata": {
                    "stage": "predicted_mask",
                    "checkpoint_role": "validation_best",
                    "region_data_audit": audit,
                    "config": {"seed": 123},
                }},
                region_stage="predicted_mask",
                current_data_audit=audit,
                resume=False,
                expected_seed=42,
            )
        with self.assertRaisesRegex(RuntimeError, "数据绑定"):
            validate_m7_source_checkpoint(
                {"metadata": {
                    "stage": "predicted_mask",
                    "checkpoint_role": "validation_best",
                    "region_data_audit": {**audit, "predicted_index_audit": None},
                }},
                region_stage="predicted_mask",
                current_data_audit=audit,
                resume=False,
            )
        with self.assertRaisesRegex(RuntimeError, "stage"):
            validate_m7_source_checkpoint(
                {"metadata": {
                    "stage": "bridge_auto",
                    "checkpoint_role": "terminal_last",
                    "region_data_audit": audit,
                }},
                region_stage="predicted_mask",
                current_data_audit=audit,
                resume=False,
            )
        with self.assertRaisesRegex(RuntimeError, "缺少显式 region_data_audit"):
            validate_m7_source_checkpoint(
                {"metadata": {
                    "stage": "predicted_mask",
                    "checkpoint_role": "validation_best",
                    "data_audit": audit,
                }},
                region_stage="predicted_mask",
                current_data_audit=audit,
                resume=False,
            )

    def test_segdesc_stage_migration_keeps_one_segmentation_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "segmentation.pt"
            second = root / "relocated_segmentation.pt"
            first.write_bytes(b"same-segmentation-checkpoint")
            second.write_bytes(first.read_bytes())
            digest = hashlib.sha256(first.read_bytes()).hexdigest()

            def migration(path: Path, sha256: str = digest) -> dict:
                return {
                    "source_path": str(path),
                    "source_sha256": sha256,
                    "source_format": SEGMENTATION_CHECKPOINT_FORMAT,
                    "source_step": 6000,
                    "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
                }

            audit = validate_segmentation_migration_lineage(
                migration(first),
                {"segmentation_migration": migration(second)},
            )
            self.assertEqual(
                audit["protocol"], SEGMENTATION_MIGRATION_LINEAGE_PROTOCOL
            )
            self.assertTrue(audit["passed"])
            with self.assertRaisesRegex(RuntimeError, "不同"):
                validate_segmentation_migration_lineage(
                    migration(first),
                    {"segmentation_migration": migration(second, "f" * 64)},
                )
            second.write_bytes(b"drifted")
            with self.assertRaisesRegex(RuntimeError, "bytes 已漂移"):
                validate_segmentation_migration_lineage(
                    migration(first),
                    {"segmentation_migration": migration(second)},
                )

    def test_m7_initialization_audit_replays_exact_m6_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (_gate, m6, source, region_audit) = write_synthetic_m6_acceptance(
                root / "m6",
                seed=42,
                bridge_root=root / "bridge",
                d_minus_one_root=root / "d_minus_one",
                predicted_artifact_root=root / "predicted",
            )
            d4 = m6["d4_final_acceptance"]
            segmentation_migration = inspect_segdesc_checkpoint(source)[
                "checkpoint_metadata"
            ]["segmentation_migration"]
            audit = build_joint_initialization_audit(
                source,
                expected_seed=42,
                region_stage="predicted_mask",
                region_data_audit=region_audit,
                d4_final_acceptance=d4,
                m6_acceptance=m6,
                segmentation_migration=segmentation_migration,
                source_step=100,
                require_m6_binding=True,
            )
            self.assertTrue(audit["formal_m6_bound"])
            self.assertEqual(
                revalidate_joint_initialization_audit(
                    audit,
                    expected_seed=42,
                    region_stage="predicted_mask",
                    region_data_audit=region_audit,
                    d4_final_acceptance=d4,
                    m6_acceptance=m6,
                    segmentation_migration=segmentation_migration,
                    require_m6_binding=True,
                ),
                audit,
            )
            tampered = dict(audit)
            tampered["source_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "已漂移"):
                revalidate_joint_initialization_audit(
                    tampered,
                    expected_seed=42,
                    region_stage="predicted_mask",
                    region_data_audit=region_audit,
                    d4_final_acceptance=d4,
                    m6_acceptance=m6,
                    segmentation_migration=segmentation_migration,
                    require_m6_binding=True,
                )
            with self.assertRaisesRegex(RuntimeError, "different|不同"):
                build_joint_initialization_audit(
                    source,
                    expected_seed=42,
                    region_stage="predicted_mask",
                    region_data_audit=region_audit,
                    d4_final_acceptance=d4,
                    m6_acceptance=m6,
                    segmentation_migration={"source_sha256": "f" * 64},
                    source_step=100,
                    require_m6_binding=True,
                )
            source.write_bytes(source.read_bytes() + b"source-drift")
            with self.assertRaisesRegex(RuntimeError, "D4/M6"):
                revalidate_joint_initialization_audit(
                    audit,
                    expected_seed=42,
                    region_stage="predicted_mask",
                    region_data_audit=region_audit,
                    d4_final_acceptance=d4,
                    m6_acceptance=m6,
                    segmentation_migration=segmentation_migration,
                    require_m6_binding=True,
                )

    def test_dior_sampler_places_same_parent_candidates_in_batches(self) -> None:
        class SizedDataset:
            rows = [
                {"parent_sample_id": parent, "sample_id": f"{parent}_{index}"}
                for parent, count in (
                    ("p1", 3), ("p2", 3), ("p3", 2),
                    ("single_1", 1), ("single_2", 1), ("single_3", 1), ("single_4", 1),
                )
                for index in range(count)
            ]
            epoch = 0

            def __len__(self) -> int:
                return len(self.rows)

        source = SizedDataset()
        sampler = ParentGroupedRegionBatchSampler(
            source, 4, seed=42, drop_last=True
        )
        first = list(iter(sampler))
        self.assertTrue(first)
        self.assertEqual(len(first), len(sampler))
        self.assertTrue(all(len(batch) == 4 for batch in first))
        for batch in first:
            parents = [source.rows[index]["parent_sample_id"] for index in batch]
            self.assertLess(len(set(parents)), len(parents))
        self.assertEqual(first, list(iter(sampler)))

    def test_global_caption_cache_path_skips_spatial_feature_projection(self) -> None:
        class FakeDescriptionBank:
            manifest = {
                "spatial_channels": 2,
                "render_size": 8,
                "token_dim": 3,
                "format": "synthetic_description_cache",
            }

            @staticmethod
            def record(_component: str, _parent: str) -> dict:
                return {
                    "lookup_key": "synthetic-key",
                    "source_ref": {"kind": "single_image"},
                    "views": [{
                        "name": "rgb",
                        "source_families": ["optical"],
                        "source_modalities": ["rgb"],
                        "quality_flags": [],
                        "render_transform": {},
                        "content_hash": "a" * 64,
                        "valid_mask": torch.ones(1, 8, 8),
                        "view_tokens": torch.ones(2, 3),
                        "description": "RGB view",
                        "spatial_features": [
                            torch.ones(2, size, size) for size in (8, 4, 2, 1)
                        ],
                    }],
                }

        encoder = DescriptionCacheBackboneEncoder(FakeDescriptionBank(), dim=4)
        global_state = encoder(
            [("single_image", "parent_001")], include_spatial=False
        )
        self.assertEqual(global_state.features.samples, [[]])
        self.assertFalse(global_state.metadata[0]["spatial_features_loaded"])
        self.assertEqual(global_state.visual_evidence.token_counts, (2,))
        region_state = encoder(
            [("single_image", "parent_001")], include_spatial=True
        )
        self.assertEqual(len(region_state.features.samples[0]), 1)
        self.assertTrue(region_state.metadata[0]["spatial_features_loaded"])

    def test_demo_overlay_restores_cache_padding_before_source_display(self) -> None:
        source = torch.zeros(1, 3, 6)
        source[:, 1:, 2:5] = 1.0
        transform = {
            "source_h": 3,
            "source_w": 6,
            "resized_h": 3,
            "resized_w": 6,
            "pad_top": 2,
            "pad_left": 1,
            "size": 8,
        }
        cached = transform_region_mask_to_cache(source, transform)
        self.assertEqual(tuple(cached.shape), (1, 8, 8))
        restored = restore_region_mask_from_cache(cached, transform)
        self.assertTrue(torch.equal(restored, source))
        with self.assertRaisesRegex(ValueError, "canvas"):
            restore_region_mask_from_cache(torch.zeros(1, 4, 4), transform)

    def test_global_caption_sequence_excludes_region_replay_tokens(self) -> None:
        model = SequenceProtocolHarness()
        backbone = SimpleNamespace(visual_evidence=SimpleNamespace(
            tokens=torch.ones(1, 2, 3),
            token_mask=torch.ones(1, 2, dtype=torch.bool),
        ))
        region_state = SimpleNamespace(
            backbone=backbone,
            region_sequence_tokens=torch.ones(1, 1, 3, 2),
            region_sequence_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            region_tokens=None,
        )
        global_sequence, _, _, global_lengths = model._build_sequences(
            region_state, ["Describe the image."], None, [False]
        )
        region_sequence, _, _, region_lengths = model._build_sequences(
            region_state, ["Describe the selected region."], None, [True]
        )
        self.assertEqual(global_sequence.shape[1], 4)
        self.assertEqual(global_lengths, (4,))
        self.assertEqual(region_sequence.shape[1], 7)
        self.assertEqual(region_lengths, (7,))

    def test_description_trainable_modules_follow_curriculum_stage(self) -> None:
        def trainable(stage: str) -> set[str]:
            model = StageParameterHarness()
            config = SimpleNamespace(
                stage=stage,
                learning_rate=1.0e-4,
                desc_adapter_lr_scale=0.2,
                weight_decay=0.01,
            )
            description_parameter_groups(model, config)
            return {name for name, value in model.named_parameters() if value.requires_grad}

        global_names = trainable("mmrs_caption")
        self.assertTrue(any("desc_adapter" in name for name in global_names))
        self.assertTrue(any(name.startswith("description_view_to_hidden.") for name in global_names))
        self.assertIn("instruction_type", global_names)
        self.assertIn("visual_type", global_names)
        self.assertFalse(any(name.startswith("mgrr.") for name in global_names))
        self.assertFalse(any(name.startswith("description_backbone.") for name in global_names))
        self.assertNotIn("region_type", global_names)

        alignment_names = trainable("dior_alignment")
        self.assertTrue(any(name.startswith("mgrr.") for name in alignment_names))
        self.assertTrue(any(name.startswith("description_backbone.") for name in alignment_names))
        self.assertTrue(any(name.startswith("alignment_text_projection.") for name in alignment_names))
        self.assertFalse(any("desc_adapter" in name for name in alignment_names))
        self.assertIn("instruction_type", alignment_names)
        self.assertFalse(any(name.startswith("region_to_hidden.") for name in alignment_names))
        self.assertNotIn("region_type", alignment_names)
        self.assertNotIn("visual_type", alignment_names)

        bridge_names = trainable("bridge_auto")
        for prefix in (
            "description_backbone.", "mgrr.", "region_to_hidden.",
            "description_view_to_hidden.",
        ):
            self.assertTrue(any(name.startswith(prefix) for name in bridge_names))
        self.assertFalse(any(
            name.startswith("alignment_text_projection.") for name in bridge_names
        ))
        self.assertIn("region_type", bridge_names)
        self.assertIn("instruction_type", bridge_names)
        self.assertIn("visual_type", bridge_names)
        expert_names = trainable("bridge_expert")
        self.assertTrue(any("desc_adapter" in name for name in expert_names))
        self.assertTrue(any(
            name.startswith("alignment_text_projection.") for name in expert_names
        ))
        self.assertIn("alignment_temperature", expert_names)

    def test_d2_gradient_gate_keeps_active_desc_adapter_frozen(self) -> None:
        model = StageParameterHarness()
        config = SimpleNamespace(
            stage="dior_alignment",
            learning_rate=1.0e-4,
            desc_adapter_lr_scale=0.2,
            weight_decay=0.01,
        )
        groups = description_parameter_groups(model, config)
        optimizer = torch.optim.SGD(groups)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        gate = _description_step_gradient_gate(
            model,
            optimizer,
            run_stage="dior_alignment",
            stream_name="main",
            stream_stage="dior_alignment",
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["modules"]["desc_adapter"]["num_nonzero"], 0)

    def test_global_caption_state_skips_region_replay(self) -> None:
        model = RegionBypassHarness()
        mask = torch.zeros(2, 1, 8, 8)
        backbone = SimpleNamespace(valid_mask=torch.ones_like(mask))
        state = model._description_region_state(
            backbone,
            mask,
            region_valid_mask=None,
            protocol="vision_only",
            structured_outputs=[False, False],
        )
        self.assertEqual(state.region_tokens.shape, (2, 1, 2))
        self.assertTrue(bool(state.diagnostics["global_caption_region_replay_skipped"].all()))
        with self.assertRaisesRegex(RuntimeError, "region replay executed"):
            model._description_region_state(
                backbone,
                mask,
                region_valid_mask=None,
                protocol="vision_only",
                structured_outputs=[False, True],
            )

    def test_trainable_parameter_manifest_matches_optimizer_groups(self) -> None:
        model = StageParameterHarness()
        config = SimpleNamespace(
            stage="mmrs_caption",
            learning_rate=1.0e-4,
            desc_adapter_lr_scale=0.2,
            weight_decay=0.01,
        )
        groups = description_parameter_groups(model, config)
        manifest = description_trainable_parameter_manifest(
            model, groups, stage=config.stage
        )
        self.assertEqual(
            manifest["protocol"], "qpsalm_description_trainable_parameters_v1"
        )
        self.assertEqual(manifest["stage"], "mmrs_caption")
        flattened = {
            name
            for group in manifest["groups"]
            for name in group["parameter_names"]
        }
        self.assertEqual(
            flattened,
            {name for name, value in model.named_parameters() if value.requires_grad},
        )

    def test_m6_acceptance_binds_three_modes_and_rejects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path, audit, d4_checkpoint, region_data_audit = (
                write_synthetic_m6_acceptance(
                    root / "m6",
                    seed=42,
                    bridge_root=root / "bridge",
                    d_minus_one_root=root / "d_minus_one",
                )
            )
            self.assertEqual(audit["protocol"], M6_ACCEPTANCE_AUDIT_PROTOCOL)
            self.assertTrue(audit["passed"])
            self.assertEqual(
                audit["evaluation_parent_populations"]["gt_mask"],
                audit["evaluation_parent_populations"]["end_to_end"],
            )
            self.assertEqual(
                validate_m6_acceptance_for_m7(
                    gate_path,
                    seed=42,
                    initialize_from=d4_checkpoint,
                    train_region_data_audit=region_data_audit,
                ),
                audit,
            )
            original_gate = gate_path.read_text(encoding="utf-8")
            failed_gate = json.loads(original_gate)
            failed_check = next(iter(failed_gate["checks"]))
            failed_gate["checks"][failed_check] = False
            failed_gate["errors"] = [failed_check]
            failed_gate["passed"] = False
            failed_gate["status"] = "engineering-invalid"
            gate_path.write_text(json.dumps(failed_gate), encoding="utf-8")
            with patch(
                "qpsalm_seg.description.m6_acceptance.build_m6_acceptance_gate",
                return_value=failed_gate,
            ):
                replayed_path, replayed_gate = validate_m6_acceptance_gate(gate_path)
                self.assertEqual(replayed_path, gate_path.resolve(strict=False))
                self.assertFalse(replayed_gate["passed"])
                with self.assertRaisesRegex(ValueError, "不能授权 M7"):
                    validate_m6_acceptance_for_m7(
                        gate_path,
                        seed=42,
                        initialize_from=d4_checkpoint,
                        train_region_data_audit=region_data_audit,
                    )
            gate_path.write_text(original_gate, encoding="utf-8")
            fixed_eval_report = root / "m6/d4_final/evaluation/eval_report.json"
            original_fixed_eval = fixed_eval_report.read_text(encoding="utf-8")
            forged_fixed_eval = json.loads(original_fixed_eval)
            forged_fixed_eval["checkpoint_metadata"]["metadata"][
                "untrusted_report_only_field"
            ] = True
            refresh_synthetic_evaluation_publication(
                fixed_eval_report.parent, forged_fixed_eval
            )
            fixed_eval_report.write_text(
                json.dumps(forged_fixed_eval), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "step/metadata 与 payload 不一致"):
                validate_m6_acceptance_gate(gate_path)
            fixed_eval_report.write_text(original_fixed_eval, encoding="utf-8")
            drifted_assets = description_protocol_assets_spec()
            drifted_assets["assets"][
                "configs/qpsalm_description_output_v1.schema.json"
            ]["sha256"] = "0" * 64
            with patch(
                "qpsalm_seg.description.m6_acceptance.description_protocol_assets_spec",
                return_value=drifted_assets,
            ):
                with self.assertRaisesRegex(ValueError, "ontology/schema binding 已漂移"):
                    validate_m6_acceptance_gate(gate_path)
            reviewer = root / "m6/end_to_end/reviewer_1.jsonl"
            original_review = reviewer.read_text(encoding="utf-8")
            reviewer.write_text(original_review + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewer source|ERFS"):
                validate_m6_acceptance_gate(gate_path)
            reviewer.write_text(original_review, encoding="utf-8")
            counterfactual = (
                root / "m6/end_to_end/counterfactual_generations.jsonl"
            )
            original_counterfactual = counterfactual.read_text(encoding="utf-8")
            changed_rows = [
                json.loads(line)
                for line in original_counterfactual.splitlines()
                if line.strip()
            ]
            changed_rows[0]["target_score_delta"] = 0.5
            counterfactual.write_text(
                "".join(json.dumps(row) + "\n" for row in changed_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "publication|重新计算|反事实|不一致"
            ):
                validate_m6_acceptance_gate(gate_path)
            counterfactual.write_text(original_counterfactual, encoding="utf-8")
            changed_rows[0]["target_score_delta"] = json.loads(
                original_counterfactual.splitlines()[0]
            )["target_score_delta"]
            input_audit = changed_rows[0]["input_change_audit"]
            input_audit["counterfactual_region_mask_sha256"] = input_audit[
                "baseline_region_mask_sha256"
            ]
            counterfactual.write_text(
                "".join(json.dumps(row) + "\n" for row in changed_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "publication|fingerprint|input-change"
            ):
                validate_m6_acceptance_gate(gate_path)
            counterfactual.write_text(original_counterfactual, encoding="utf-8")
            segmentation_index = root / "segmentation_instruction_val.jsonl"
            original_index = segmentation_index.read_text(encoding="utf-8")
            segmentation_index.write_text(original_index + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "instruction source index.*漂移"):
                validate_m6_acceptance_gate(gate_path)
            segmentation_index.write_text(original_index, encoding="utf-8")
            target_audit = root / "m6/end_to_end/end_to_end_target_audit.jsonl"
            original_target_audit = target_audit.read_text(encoding="utf-8")
            changed_target = json.loads(original_target_audit)
            changed_target["segmentation_sample_id"] = "tampered-segmentation-row"
            target_audit.write_text(
                json.dumps(changed_target) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "publication|standalone audit|重放"
            ):
                validate_m6_acceptance_gate(gate_path)
            target_audit.write_text(original_target_audit, encoding="utf-8")
            cycle_path = root / "m6/gt_source/evaluation/cycle_localization.jsonl"
            original_cycle = cycle_path.read_text(encoding="utf-8")
            changed_cycle = json.loads(original_cycle)
            changed_cycle["cycle_audit"]["generated_text_sha256"] = "0" * 64
            cycle_path.write_text(json.dumps(changed_cycle) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "publication|cycle.*raw generation"
            ):
                validate_m6_acceptance_gate(gate_path)
            cycle_path.write_text(original_cycle, encoding="utf-8")
            changed_cycle = json.loads(original_cycle)
            changed_cycle.update({
                "region_iou": 0.7,
                "intersection_pixels": 7,
                "union_pixels": 10,
                "target_pixels": 10,
                "predicted_pixels": 7,
            })
            cycle_path.write_text(json.dumps(changed_cycle) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "publication|mask artifacts|像素计数"
            ):
                validate_m6_acceptance_gate(gate_path)
            cycle_path.write_text(original_cycle, encoding="utf-8")
            gt_report_path = root / "m6/gt_source/evaluation/eval_report.json"
            original_gt_report = gt_report_path.read_text(encoding="utf-8")
            changed_cycle = json.loads(original_cycle)
            prediction_path = (
                cycle_path.parent
                / changed_cycle["prediction_mask_artifact"]["path"]
            )
            original_prediction_mask = prediction_path.read_bytes()
            forged_prediction_artifact = write_evaluation_mask_artifact(
                cycle_path.parent,
                role="cycle_prediction",
                sample_id="sample-1",
                mask=np.zeros((4, 4), dtype=np.uint8),
            )
            changed_cycle.update({
                "region_iou": 0.0,
                "intersection_pixels": 0,
                "union_pixels": 10,
                "target_pixels": 10,
                "predicted_pixels": 0,
                "target_empty": False,
                "prediction_empty": True,
                "empty_target_correct": False,
                "prediction_mask_artifact": forged_prediction_artifact,
            })
            cycle_path.write_text(
                json.dumps(changed_cycle) + "\n", encoding="utf-8"
            )
            changed_report = json.loads(original_gt_report)
            changed_report["evaluation_mask_artifacts"] = (
                evaluation_mask_artifact_inventory([
                    json.loads(
                        (root / "m6/gt_source/evaluation/raw_generations.jsonl")
                        .read_text(encoding="utf-8")
                    )["region_input_mask_artifact"],
                    forged_prediction_artifact,
                    changed_cycle["target_mask_artifact"],
                    changed_cycle["source_mask_artifact"],
                    changed_cycle["valid_mask_artifact"],
                ])
            )
            refresh_synthetic_evaluation_publication(
                gt_report_path.parent, changed_report
            )
            gt_report_path.write_text(
                json.dumps(changed_report), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "M3 valid/source projection/region"
            ):
                validate_m6_acceptance_gate(gate_path)
            prediction_path.write_bytes(original_prediction_mask)
            cycle_path.write_text(original_cycle, encoding="utf-8")
            gt_report_path.write_text(original_gt_report, encoding="utf-8")
            changed_cycle = json.loads(original_cycle)
            cycle_paths = {
                role: cycle_path.parent / changed_cycle[field]["path"]
                for role, field in {
                    "cycle_prediction": "prediction_mask_artifact",
                    "cycle_target": "target_mask_artifact",
                    "cycle_valid": "valid_mask_artifact",
                }.items()
            }
            original_cycle_masks = {
                role: path.read_bytes() for role, path in cycle_paths.items()
            }
            forged_cycle_artifacts = {
                role: write_evaluation_mask_artifact(
                    cycle_path.parent,
                    role=role,
                    sample_id="sample-1",
                    mask=np.zeros((4, 4), dtype=np.uint8),
                )
                for role in cycle_paths
            }
            changed_cycle.update({
                "region_iou": 1.0,
                "intersection_pixels": 0,
                "union_pixels": 0,
                "target_pixels": 0,
                "predicted_pixels": 0,
                "target_empty": True,
                "prediction_empty": True,
                "empty_target_correct": True,
                "prediction_mask_artifact": forged_cycle_artifacts[
                    "cycle_prediction"
                ],
                "target_mask_artifact": forged_cycle_artifacts["cycle_target"],
                "valid_mask_artifact": forged_cycle_artifacts["cycle_valid"],
            })
            cycle_path.write_text(
                json.dumps(changed_cycle) + "\n", encoding="utf-8"
            )
            changed_report = json.loads(original_gt_report)
            changed_report["evaluation_mask_artifacts"] = (
                evaluation_mask_artifact_inventory([
                    json.loads(
                        (root / "m6/gt_source/evaluation/raw_generations.jsonl")
                        .read_text(encoding="utf-8")
                    )["region_input_mask_artifact"],
                    forged_cycle_artifacts["cycle_prediction"],
                    forged_cycle_artifacts["cycle_target"],
                    changed_cycle["source_mask_artifact"],
                    forged_cycle_artifacts["cycle_valid"],
                ])
            )
            refresh_synthetic_evaluation_publication(
                gt_report_path.parent, changed_report
            )
            gt_report_path.write_text(
                json.dumps(changed_report), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "M3 valid"):
                validate_m6_acceptance_gate(gate_path)
            for role, path in cycle_paths.items():
                path.write_bytes(original_cycle_masks[role])
            cycle_path.write_text(original_cycle, encoding="utf-8")
            gt_report_path.write_text(original_gt_report, encoding="utf-8")
            gt_raw = root / "m6/gt_source/evaluation/raw_generations.jsonl"
            gt_generation = json.loads(gt_raw.read_text(encoding="utf-8"))
            region_mask_path = (
                gt_raw.parent / gt_generation["region_input_mask_artifact"]["path"]
            )
            original_region_mask = region_mask_path.read_bytes()
            np.save(
                region_mask_path,
                np.zeros((4, 4), dtype=np.uint8),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "mask artifact.*漂移"):
                validate_m6_acceptance_gate(gate_path)
            region_mask_path.write_bytes(original_region_mask)
            end_raw = root / "m6/end_to_end/raw_generations.jsonl"
            end_raw.write_text(
                end_raw.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "publication|重新计算|绑定|sample_id"
            ):
                validate_m6_acceptance_gate(gate_path)

    def test_region_source_filter_freezes_exact_gt_global_population(self) -> None:
        rows = [
            {
                "bridge_record_id": "gt-1",
                "parent_sample_id": "parent-1",
                "region_id": "global",
                "region_source": "gt_global_mask",
            },
            {
                "bridge_record_id": "pseudo-1",
                "parent_sample_id": "parent-1",
                "region_id": "component-1",
                "region_source": "pseudo_instance_component",
            },
        ]
        selected, audit = filter_evaluation_region_source(
            rows,
            stage="bridge_expert",
            split="val",
            training=False,
            evaluation_mode="gt_mask",
            region_source="gt_global_mask",
        )
        self.assertEqual([row["bridge_record_id"] for row in selected], ["gt-1"])
        self.assertEqual(audit["excluded_rows"], 1)
        self.assertEqual(
            audit["population_sha256"],
            evaluation_region_source_population_sha256(selected),
        )
        with self.assertRaisesRegex(ValueError, "region-source filter"):
            filter_evaluation_region_source(
                rows,
                stage="predicted_mask",
                split="val",
                training=False,
                evaluation_mode="fixed_prediction",
                region_source="gt_global_mask",
            )

    def test_retention_requires_exact_sample_population_identity(self) -> None:
        population = {
            "protocol": "qpsalm_segmentation_eval_population_v1",
            "fields": list(SAMPLE_IDENTITY_FIELDS),
            "sha256": "a" * 64,
            "complete": True,
            "unique": True,
            "num_records": 10,
            "num_unique_sample_ids": 10,
        }
        baseline = {
            "threshold": 0.5,
            "coverage": {"num_samples": 10, "sample_population": population},
            "prediction_population": (
                synthetic_segmentation_prediction_population("baseline")
            ),
            "metrics": {"positive_only": {"dice": 0.50}},
        }
        candidate = {
            "threshold": 0.5,
            "coverage": {"num_samples": 10, "sample_population": dict(population)},
            "prediction_population": (
                synthetic_segmentation_prediction_population("joint")
            ),
            "metrics": {"positive_only": {"dice": 0.495}},
        }
        d4_final_acceptance = {"passed": True, "current_fraction": 0.75}
        m6_acceptance = {
            "protocol": M6_ACCEPTANCE_AUDIT_PROTOCOL,
            "passed": True,
            "d4_final_acceptance": d4_final_acceptance,
        }
        d_minus_one_acceptance = {
            "protocol": D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
            "passed": True,
        }
        stage_lineage = {
            "protocol": DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
            "target_stage": "predicted_mask",
        }
        baseline_replay_audit = {
            "protocol": BASELINE_CHECKPOINT_REPLAY_PROTOCOL,
            "passed": True,
            "checkpoint_sha256": "seg-source",
        }
        joint_initialization = {
            "protocol": JOINT_INITIALIZATION_PROTOCOL,
            "passed": True,
            "formal_m6_bound": True,
            "segmentation_migration_lineage": {"passed": True},
        }
        protocol_assets = description_protocol_assets_spec()
        joint_execution = synthetic_joint_execution_fields(step=100)
        gate = build_retention_gate(
            baseline, candidate, split="val", max_samples=0,
            checkpoint="joint.pt", checkpoint_step=100,
            checkpoint_metadata={
                "description_protocol_assets": protocol_assets,
                "metadata": {"stage": "joint",
                    "joint_initialization_audit": joint_initialization,
                    "segmentation_migration_lineage": {"passed": True},
                    **joint_execution, "config": {
                    "seed": 42,
                    "joint_region_stage": "predicted_mask",
                    "predicted_mask_fraction": 0.75,
                    "grad_accum_steps": 1,
                }},
                "segmentation_migration": {"source_sha256": "seg-source"},
            },
            maximum_allowed_drop=0.01,
            baseline_binding={"valid": True, "checkpoint_sha256": "seg-source"},
            expected_seed=42,
            d4_final_acceptance_audit=d4_final_acceptance,
            m6_acceptance_audit=m6_acceptance,
            joint_initialization_audit=joint_initialization,
            d_minus_one_acceptance_audit=d_minus_one_acceptance,
            stage_lineage_audit=stage_lineage,
            baseline_checkpoint_replay_audit=baseline_replay_audit,
        )
        self.assertTrue(gate["scientific_gate_eligible"])
        self.assertTrue(gate["preliminary_passed"])
        self.assertFalse(gate["formal_report_binding_complete"])
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["seed_match"])
        self.assertTrue(gate["joint_execution_contract_valid"])
        self.assertTrue(gate["joint_initialization_valid"])
        self.assertTrue(gate["same_metric_input_population"])
        self.assertEqual(
            gate["baseline_comparison_mode"], "frozen_full_report"
        )
        candidate["coverage"]["sample_population"]["sha256"] = "b" * 64
        gate = build_retention_gate(
            baseline, candidate, split="val", max_samples=0,
            checkpoint="joint.pt", checkpoint_step=100,
            checkpoint_metadata={
                "description_protocol_assets": protocol_assets,
                "metadata": {"stage": "joint",
                    "joint_initialization_audit": joint_initialization,
                    "segmentation_migration_lineage": {"passed": True},
                    **joint_execution, "config": {
                    "joint_region_stage": "predicted_mask",
                    "predicted_mask_fraction": 0.75,
                    "grad_accum_steps": 1,
                }},
                "segmentation_migration": {"source_sha256": "seg-source"},
            },
            maximum_allowed_drop=0.01,
            baseline_binding={"valid": True, "checkpoint_sha256": "seg-source"},
            d4_final_acceptance_audit=d4_final_acceptance,
            m6_acceptance_audit=m6_acceptance,
            joint_initialization_audit=joint_initialization,
            d_minus_one_acceptance_audit=d_minus_one_acceptance,
            stage_lineage_audit=stage_lineage,
            baseline_checkpoint_replay_audit=baseline_replay_audit,
        )
        self.assertFalse(gate["scientific_gate_eligible"])
        self.assertFalse(gate["passed"])

        candidate["coverage"]["sample_population"]["sha256"] = "a" * 64
        changed_source = build_retention_gate(
            baseline, candidate, split="val", max_samples=0,
            checkpoint="joint.pt", checkpoint_step=100,
            checkpoint_metadata={
                "description_protocol_assets": protocol_assets,
                "metadata": {"stage": "joint",
                    "joint_initialization_audit": joint_initialization,
                    "segmentation_migration_lineage": {"passed": True},
                    **joint_execution, "config": {
                    "joint_region_stage": "predicted_mask",
                    "predicted_mask_fraction": 0.75,
                    "grad_accum_steps": 1,
                }},
                "segmentation_migration": {"source_sha256": "different"},
            },
            maximum_allowed_drop=0.01,
            baseline_binding={"valid": True, "checkpoint_sha256": "seg-source"},
            d4_final_acceptance_audit=d4_final_acceptance,
            m6_acceptance_audit=m6_acceptance,
            joint_initialization_audit=joint_initialization,
            d_minus_one_acceptance_audit=d_minus_one_acceptance,
            stage_lineage_audit=stage_lineage,
            baseline_checkpoint_replay_audit=baseline_replay_audit,
        )
        self.assertFalse(changed_source["baseline_source_checkpoint_match"])
        self.assertFalse(changed_source["scientific_gate_eligible"])
        self.assertFalse(changed_source["passed"])

        wrong_seed = build_retention_gate(
            baseline, candidate, split="val", max_samples=0,
            checkpoint="joint.pt", checkpoint_step=100,
            checkpoint_metadata={
                "description_protocol_assets": protocol_assets,
                "metadata": {"stage": "joint",
                    "joint_initialization_audit": joint_initialization,
                    "segmentation_migration_lineage": {"passed": True},
                    **joint_execution, "config": {
                    "seed": 123,
                    "joint_region_stage": "predicted_mask",
                    "predicted_mask_fraction": 0.75,
                    "grad_accum_steps": 1,
                }},
                "segmentation_migration": {"source_sha256": "seg-source"},
            },
            maximum_allowed_drop=0.01,
            baseline_binding={"valid": True, "checkpoint_sha256": "seg-source"},
            expected_seed=42,
            d4_final_acceptance_audit=d4_final_acceptance,
            m6_acceptance_audit=m6_acceptance,
            joint_initialization_audit=joint_initialization,
            d_minus_one_acceptance_audit=d_minus_one_acceptance,
            stage_lineage_audit=stage_lineage,
            baseline_checkpoint_replay_audit=baseline_replay_audit,
        )
        self.assertFalse(wrong_seed["seed_match"])
        self.assertFalse(wrong_seed["scientific_gate_eligible"])

    def test_retention_metric_inputs_bind_target_and_valid_not_prediction(self) -> None:
        baseline = synthetic_segmentation_prediction_population("baseline")
        joint = synthetic_segmentation_prediction_population("joint")
        self.assertEqual(
            segmentation_metric_input_population(baseline),
            segmentation_metric_input_population(joint),
        )
        rows = copy.deepcopy(joint["rows"])
        rows[0]["target_sha256"] = "f" * 64
        drifted = segmentation_prediction_population(rows, threshold=0.5)
        self.assertNotEqual(
            segmentation_metric_input_population(baseline),
            segmentation_metric_input_population(drifted),
        )

    def test_retention_smoke_is_preliminary_live_limited_comparison(self) -> None:
        population = synthetic_retention_population()
        baseline = {
            "threshold": 0.5,
            "coverage": {"num_samples": 10, "sample_population": population},
            "prediction_population": (
                synthetic_segmentation_prediction_population("baseline")
            ),
            "metrics": {"positive_only": {"dice": 0.50}},
        }
        candidate = {
            "threshold": 0.5,
            "coverage": {
                "num_samples": 10,
                "sample_population": dict(population),
            },
            "prediction_population": (
                synthetic_segmentation_prediction_population("joint")
            ),
            "metrics": {"positive_only": {"dice": 0.495}},
        }
        gate = build_retention_gate(
            baseline,
            candidate,
            split="val",
            max_samples=10,
            checkpoint="joint.pt",
            checkpoint_step=0,
            checkpoint_metadata={},
            maximum_allowed_drop=0.01,
            baseline_binding={},
        )
        self.assertEqual(
            gate["baseline_comparison_mode"], "live_limited_replay"
        )
        self.assertTrue(gate["preliminary_passed"])
        self.assertFalse(gate["scientific_gate_eligible"])
        self.assertFalse(gate["passed"])

    def test_m7_retention_three_seed_gate_binds_independent_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = synthetic_retention_population()
            baseline, binding = write_synthetic_retention_baseline(
                root / "baseline", population=population,
                checkpoint=synthetic_retention_segmentation_checkpoint(root),
            )
            gates = [
                write_synthetic_retention_gate(
                    root / f"seed_{seed}",
                    seed=seed,
                    baseline=baseline,
                    baseline_binding=binding,
                    population=population,
                )
                for seed in (42, 123, 3407)
            ]
            report = aggregate_m7_retention_seed_gates(
                gates, seeds=(42, 123, 3407)
            )
            self.assertEqual(report["protocol"], M7_RETENTION_SEED_GATE_PROTOCOL)
            self.assertEqual(report["num_passed"], 3)
            self.assertTrue(report["passed_all_three"])
            self.assertTrue(report["all_joint_checkpoints_unique"])
            self.assertTrue(report["same_description_vision_cache"])
            self.assertTrue(report["same_joint_training_population"])
            aggregate_path = root / "m7_retention_seed_gate.json"
            write_json(aggregate_path, report)
            _path, replayed = validate_m7_retention_seed_gate(aggregate_path)
            self.assertEqual(replayed, report)
            original_single_gate = gates[0].read_text(encoding="utf-8")
            gates[0].write_text(original_single_gate + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重新计算结果不一致"):
                validate_m7_retention_seed_gate(aggregate_path)
            gates[0].write_text(original_single_gate, encoding="utf-8")

            drifted_assets = description_protocol_assets_spec()
            drifted_assets["assets"][
                "configs/qpsalm_description_output_v1.schema.json"
            ]["sha256"] = "0" * 64
            with patch(
                "qpsalm_seg.description.retention.description_protocol_assets_spec",
                return_value=drifted_assets,
            ):
                with self.assertRaisesRegex(ValueError, "ontology/schema binding 已漂移"):
                    validate_m7_retention_gate(gates[0], expected_seed=42)

            with self.assertRaisesRegex(ValueError, "seed binding"):
                aggregate_m7_retention_seed_gates(
                    gates, seeds=(42, 999, 3407)
                )

            training_drift_gate = write_synthetic_retention_gate(
                root / "seed_123_training_population_drift",
                seed=123,
                baseline=baseline,
                baseline_binding=binding,
                population=population,
                joint_population_variant="-drift",
            )
            with self.assertRaisesRegex(ValueError, "训练 population"):
                aggregate_m7_retention_seed_gates(
                    [gates[0], training_drift_gate, gates[2]],
                    seeds=(42, 123, 3407),
                )

            gates[1] = write_synthetic_retention_gate(
                root / "seed_123_cache_drift",
                seed=123,
                baseline=baseline,
                baseline_binding=binding,
                population=population,
                cache_variant="-drift",
            )
            with self.assertRaisesRegex(
                ValueError, "Description Vision Cache"
            ):
                aggregate_m7_retention_seed_gates(
                    gates, seeds=(42, 123, 3407)
                )

            gate = json.loads(gates[0].read_text(encoding="utf-8"))
            cache_binding = gate["checkpoint_metadata"][
                "description_architecture_spec"
            ]["description_cache_artifact_binding"]
            shard = Path(cache_binding["cache_dir"]) / "shard_00000.pt"
            shard.write_bytes(shard.read_bytes() + b"m7-cache-drift")
            with self.assertRaisesRegex(
                ValueError, "Description Vision Cache artifact"
            ):
                validate_m7_retention_gate(gates[0], expected_seed=42)

    def test_m7_retention_three_seed_gate_rejects_duplicate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = synthetic_retention_population()
            baseline, binding = write_synthetic_retention_baseline(
                root / "baseline", population=population,
                checkpoint=synthetic_retention_segmentation_checkpoint(root),
            )
            duplicate = b"same-joint-checkpoint-state"
            gates = [
                write_synthetic_retention_gate(
                    root / f"seed_{seed}",
                    seed=seed,
                    baseline=baseline,
                    baseline_binding=binding,
                    population=population,
                    checkpoint_bytes=(duplicate if seed != 3407 else None),
                )
                for seed in (42, 123, 3407)
            ]
            with self.assertRaisesRegex(ValueError, "payload"):
                aggregate_m7_retention_seed_gates(
                    gates, seeds=(42, 123, 3407)
                )

    def test_m7_retention_requires_all_seeds_and_one_full_val_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = synthetic_retention_population()
            baseline, binding = write_synthetic_retention_baseline(
                root / "baseline", population=population,
                checkpoint=synthetic_retention_segmentation_checkpoint(root),
            )
            gates = [
                write_synthetic_retention_gate(
                    root / f"seed_{seed}",
                    seed=seed,
                    baseline=baseline,
                    baseline_binding=binding,
                    population=population,
                    joint_dice=(0.48 if seed == 123 else 0.495),
                )
                for seed in (42, 123, 3407)
            ]
            report = aggregate_m7_retention_seed_gates(
                gates, seeds=(42, 123, 3407)
            )
            self.assertEqual(report["num_passed"], 2)
            self.assertFalse(report["passed"])

            drift_population = synthetic_retention_population("b" * 64)
            drift_baseline, drift_binding = write_synthetic_retention_baseline(
                root / "drift_baseline",
                population=drift_population,
                checkpoint=Path(binding["checkpoint"]),
            )
            gates[1] = write_synthetic_retention_gate(
                root / "seed_123_drift",
                seed=123,
                baseline=drift_baseline,
                baseline_binding=drift_binding,
                population=drift_population,
            )
            with self.assertRaisesRegex(ValueError, "full-val baseline"):
                aggregate_m7_retention_seed_gates(
                    gates, seeds=(42, 123, 3407)
                )

    def test_retention_baseline_is_bound_to_segmentation_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "segmentation.pt"
            checkpoint.write_bytes(b"accepted-segmentation-checkpoint")
            report_path = root / "eval_report.json"
            report = {
                "checkpoint_step": 6000,
                "threshold": 0.5,
                "prediction_population": (
                    synthetic_segmentation_prediction_population("baseline")
                ),
                "threshold_sweep": {"overall_by_threshold": {}},
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            checkpoint_hash = __import__("hashlib").sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            report_binding = {
                "protocol": SEGMENTATION_EVAL_REPORT_BINDING_PROTOCOL,
                "path": str(report_path.resolve(strict=False)),
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "bytes": int(report_path.stat().st_size),
                "prediction_population_sha256": report[
                    "prediction_population"
                ]["sha256"],
                "eval_threshold": 0.5,
                "threshold_sweep": [],
            }
            (root / "eval_manifest.json").write_text(json.dumps({
                "protocol": SEGMENTATION_EVAL_MANIFEST_PROTOCOL,
                "created_by": "qpsalm-eval",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_step": 6000,
                "split": "val",
                "preset": "qwen_psalm_full",
                "resolved_config": {
                    "instruction_ablation": "normal",
                    "visual_ablation": "normal",
                    "eval_threshold": 0.5,
                    "threshold_sweep": [],
                },
                "eval_report_binding": report_binding,
            }), encoding="utf-8")
            binding = baseline_eval_binding(report_path, report, split="val")
            self.assertTrue(binding["valid"])
            self.assertEqual(binding["checkpoint_sha256"], checkpoint_hash)
            self.assertEqual(binding["eval_threshold"], 0.5)
            self.assertEqual(binding["threshold_sweep"], [])
            manifest_path = root / "eval_manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            drifted_manifest = json.loads(original_manifest)
            drifted_manifest["resolved_config"]["eval_threshold"] = 0.4
            manifest_path.write_text(
                json.dumps(drifted_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "eval_threshold"):
                baseline_eval_binding(report_path, report, split="val")
            manifest_path.write_text(original_manifest, encoding="utf-8")
            tampered_report = copy.deepcopy(report)
            tampered_report["tampered"] = True
            report_path.write_text(json.dumps(tampered_report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "内存 report"):
                baseline_eval_binding(report_path, report, split="val")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            checkpoint.write_bytes(b"changed-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
                baseline_eval_binding(report_path, report, split="val")

    def test_retention_replays_baseline_prediction_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = synthetic_retention_population()
            checkpoint = root / "segmentation.pt"
            checkpoint.write_bytes(b"baseline-segmentation")
            baseline, binding = write_synthetic_retention_baseline(
                root / "baseline",
                population=population,
                checkpoint=checkpoint,
            )
            replay = copy.deepcopy(baseline)
            rows = copy.deepcopy(replay["prediction_population"]["rows"])
            rows[0]["prediction_sha256"] = "f" * 64
            replay["prediction_population"] = segmentation_prediction_population(
                rows, threshold=0.5
            )
            replay_path = root / "baseline_replay.json"
            replay_path.write_text(json.dumps(replay), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "same_prediction_population"
            ):
                build_baseline_checkpoint_replay_audit(
                    baseline,
                    replay,
                    baseline_binding=binding,
                    segmentation_migration={
                        "source_sha256": binding["checkpoint_sha256"],
                        "source_step": 6000,
                    },
                    replay_report_path=replay_path,
                )

    def test_description_evaluation_binds_data_stage_and_segmentation_source(self) -> None:
        migration = {"source_sha256": "a" * 64}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        d_minus_one_acceptance = write_synthetic_d_minus_one_acceptance(
            Path(temporary.name) / "d_minus_one"
        )
        predicted_lineage = synthetic_stage_lineage(
            seed=42,
            d_minus_one_acceptance=d_minus_one_acceptance,
        )
        bridge_lineage = synthetic_stage_lineage(
            seed=42,
            d_minus_one_acceptance=d_minus_one_acceptance,
            target_stage="bridge_expert",
        )
        description_benchmark = d_minus_one_acceptance[
            "description_source"
        ]["benchmark_root"]
        fixed_config = SimpleNamespace(
            stage="predicted_mask",
            evaluation_mode="fixed_prediction",
            seed=42,
            description_benchmark=description_benchmark,
        )
        fixed = validate_evaluation_checkpoint_binding(
            fixed_config,
            {
                "metadata": {
                    "stage": "predicted_mask", "config": {"seed": 42},
                    "checkpoint_role": "validation_best",
                    "d_minus_one_acceptance": d_minus_one_acceptance,
                    "stage_lineage": predicted_lineage,
                },
                "segmentation_migration": migration,
            },
            migration,
            {"segmentation_checkpoint_sha256": "a" * 64},
        )
        self.assertTrue(fixed["fixed_prediction_segmentation_source_match"])
        end_to_end = validate_evaluation_checkpoint_binding(
            SimpleNamespace(
                stage="bridge_expert",
                evaluation_mode="end_to_end",
                seed=42,
                description_benchmark=description_benchmark,
            ),
            {
                "metadata": {
                    "stage": "predicted_mask", "config": {"seed": 42},
                    "checkpoint_role": "validation_best",
                    "d_minus_one_acceptance": d_minus_one_acceptance,
                    "stage_lineage": predicted_lineage,
                },
                "segmentation_migration": migration,
            },
            migration,
            None,
        )
        self.assertEqual(end_to_end["checkpoint_stage"], "predicted_mask")
        wrong_role = {
            "metadata": {
                "stage": "predicted_mask",
                "checkpoint_role": "terminal_last",
                "config": {"seed": 42},
                "d_minus_one_acceptance": d_minus_one_acceptance,
                "stage_lineage": predicted_lineage,
            },
            "segmentation_migration": migration,
        }
        with self.assertRaisesRegex(RuntimeError, "checkpoint role"):
            validate_evaluation_checkpoint_binding(
                fixed_config,
                wrong_role,
                migration,
                {"segmentation_checkpoint_sha256": "a" * 64},
            )
        with self.assertRaisesRegex(RuntimeError, "同一 segmentation"):
            validate_evaluation_checkpoint_binding(
                fixed_config,
                {
                    "metadata": {
                        "stage": "predicted_mask", "config": {"seed": 42},
                        "checkpoint_role": "validation_best",
                        "d_minus_one_acceptance": d_minus_one_acceptance,
                        "stage_lineage": predicted_lineage,
                    },
                    "segmentation_migration": migration,
                },
                migration,
                {"segmentation_checkpoint_sha256": "b" * 64},
            )
        with self.assertRaisesRegex(RuntimeError, "checkpoint stage"):
            validate_evaluation_checkpoint_binding(
                fixed_config,
                {
                    "metadata": {
                        "stage": "bridge_expert", "config": {"seed": 42},
                        "checkpoint_role": "validation_best",
                        "d_minus_one_acceptance": d_minus_one_acceptance,
                        "stage_lineage": bridge_lineage,
                    },
                    "segmentation_migration": migration,
                },
                migration,
                {"segmentation_checkpoint_sha256": "a" * 64},
            )
        with self.assertRaisesRegex(RuntimeError, "训练 seed"):
            validate_evaluation_checkpoint_binding(
                fixed_config,
                {
                    "metadata": {
                        "stage": "predicted_mask", "config": {"seed": 123},
                        "checkpoint_role": "validation_best",
                        "d_minus_one_acceptance": d_minus_one_acceptance,
                        "stage_lineage": predicted_lineage,
                    },
                    "segmentation_migration": migration,
                },
                migration,
                {"segmentation_checkpoint_sha256": "a" * 64},
            )

    def test_adapter_scope_restores_optimizer_trainability(self) -> None:
        controller = AdapterScopeHarness()
        controller.model.lora_A["default"].requires_grad_(False)
        controller.model.lora_A["desc_adapter"].requires_grad_(True)
        with controller.adapter_scope("desc_adapter"):
            self.assertEqual(controller.model.active_adapters, ("desc_adapter",))
            self.assertTrue(controller.model.lora_A["desc_adapter"].requires_grad)
        self.assertEqual(controller.model.active_adapters, ("default",))
        self.assertFalse(controller.model.lora_A["default"].requires_grad)
        self.assertTrue(controller.model.lora_A["desc_adapter"].requires_grad)
        controller.model.zero_grad(set_to_none=True)
        with controller.adapter_scope("desc_adapter"):
            loss = controller.model.lora_A[
                controller.model.active_adapters[0]
            ].square().sum()
        loss.backward()
        self.assertIsNone(controller.model.lora_A["default"].grad)
        self.assertGreater(
            float(controller.model.lora_A["desc_adapter"].grad.abs().sum()), 0.0
        )
        controller.model.lora_A["desc_adapter"].requires_grad_(False)
        with controller.adapter_scope("desc_adapter"):
            self.assertFalse(
                controller.model.lora_A["desc_adapter"].requires_grad
            )

    def test_causal_labels_mask_prefix_padding_and_supervise_eos(self) -> None:
        model = SequenceProtocolHarness()
        region_state = SimpleNamespace(
            backbone=SimpleNamespace(visual_evidence=SimpleNamespace(
                tokens=torch.ones(2, 2, 3),
                token_mask=torch.tensor([[True, True], [True, False]]),
            )),
            region_sequence_tokens=torch.ones(2, 1, 3, 2),
            region_sequence_mask=torch.ones(2, 1, 3, dtype=torch.bool),
            region_tokens=None,
        )
        _inputs, attention, labels, lengths = model._build_sequences(
            region_state,
            ["Caption the image.", "Describe the region."],
            ["global target", "structured target"],
            [False, True],
        )
        self.assertEqual(lengths, (7, 9))
        self.assertIsNotNone(labels)
        self.assertTrue(torch.equal(labels[0, :4], torch.full((4,), -100)))
        self.assertTrue(torch.equal(labels[0, 4:7], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(labels[0, 7:], torch.full((2,), -100)))
        self.assertTrue(torch.equal(labels[1, :6], torch.full((6,), -100)))
        self.assertTrue(torch.equal(labels[1, 6:9], torch.tensor([1, 2, 3])))
        self.assertTrue(bool(attention[0, :7].all()))
        self.assertFalse(bool(attention[0, 7:].any()))
        self.assertTrue(bool(attention[1].all()))

    def test_cycle_localization_uses_raw_text_and_explicit_empty_iou(self) -> None:
        batch = ModalityBatch(
            instances=[[]],
            full_instances=[[]],
            active_subsets=[],
            mask=torch.zeros(1, 1, 4, 4),
            valid_mask=torch.ones(1, 1, 4, 4),
            metadata=[{"sample_id": "sample"}],
            proposal_context_text=["original proposal"],
            condition_prompt_text=["original condition"],
            evidence_reasoning_text=["original reasoning"],
            full_proposal_context_text=["original full proposal"],
            full_condition_prompt_text=["original full condition"],
            full_evidence_reasoning_text=["original full reasoning"],
            visual_evidence_key=["key"],
        )
        raw = '{"summary":"raw region description"}'
        cycled = cycle_prompt_batch(batch, [raw])
        self.assertEqual(batch.proposal_context_text, ["original proposal"])
        self.assertIn(raw, cycled.proposal_context_text[0])
        self.assertIn(raw, cycled.condition_prompt_text[0])
        self.assertIn("original reasoning", cycled.evidence_reasoning_text[0])
        with self.assertRaisesRegex(ValueError, "空 generated text"):
            cycle_prompt_batch(batch, [""])

        empty = cycle_region_iou(
            torch.zeros(1, 4, 4), torch.zeros(1, 4, 4)
        )
        self.assertEqual(empty["region_iou"], 1.0)
        self.assertTrue(empty["empty_target_correct"])
        prediction = torch.zeros(1, 4, 4)
        target = torch.zeros(1, 4, 4)
        prediction[:, :2, :2] = 1
        target[:, 1:3, 1:3] = 1
        partial = cycle_region_iou(prediction, target)
        self.assertAlmostEqual(partial["region_iou"], 1.0 / 7.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            cycle_region_iou(prediction, torch.zeros(1, 2, 2))

    def test_cycle_localization_summary_is_parent_macro_and_auxiliary(self) -> None:
        provider = SimpleNamespace(
            source_rows=5,
            eligible_rows=4,
            exclusion_counts={"unsupported": 1},
            runtime_skip_counts={"empty_raw_generation": 1},
            segmentation_source_binding={"protocol": "synthetic"},
        )
        rows = [
            {
                "parent_sample_id": "p1", "region_iou": 1.0,
                "target_empty": False, "empty_target_correct": False,
            },
            {
                "parent_sample_id": "p1", "region_iou": 0.0,
                "target_empty": False, "empty_target_correct": False,
            },
            {
                "parent_sample_id": "p2", "region_iou": 1.0,
                "target_empty": True, "empty_target_correct": True,
            },
        ]
        report = summarize_cycle_localization(
            rows, provider, requested=3, seed=42
        )
        self.assertEqual(report["protocol"], CYCLE_LOCALIZATION_PROTOCOL)
        self.assertEqual(report["role"], "auxiliary_self_consistency_only")
        self.assertFalse(report["primary_evidence_replaced"])
        self.assertAlmostEqual(report["parent_macro_region_iou"], 0.75)
        self.assertEqual(report["empty_target_accuracy"], 1.0)
        self.assertTrue(report["coverage_complete"])

    def test_cycle_provider_runs_default_adapter_and_restores_canvas(self) -> None:
        class Controller:
            def __init__(self) -> None:
                self.adapters = []

            @contextmanager
            def adapter_scope(self, name: str):
                self.adapters.append(name)
                yield

        class Segmenter:
            def __init__(self) -> None:
                self.batch = None

            def __call__(self, batch):
                self.batch = batch
                logits = torch.full((1, 1, 4, 4), -8.0)
                logits[:, :, :2, :2] = 8.0
                return SimpleNamespace(final_mask_logits=logits)

        class Bank:
            @staticmethod
            def record(_component: str, _parent: str) -> dict:
                return {"views": [{"render_transform": {
                    "source_h": 4, "source_w": 4,
                    "resized_h": 4, "resized_w": 4,
                    "pad_top": 0, "pad_left": 0, "size": 4,
                }}]}

        class Model:
            def __init__(self) -> None:
                self.controller = Controller()
                self.segmentation = Segmenter()
                self.description_backbone = SimpleNamespace(bank=Bank())
                self.anchor = nn.Parameter(torch.zeros(()))

            def parameters(self):
                return iter([self.anchor])

        item = {
            "instances": [], "full_instances": [],
            "active_subset": SimpleNamespace(),
            "mask": torch.zeros(1, 4, 4),
            "valid_mask": torch.ones(1, 4, 4),
            "metadata": {"resize_transform": {
                "source_hw": [4, 4], "resized_hw": [4, 4],
                "target_hw": [4, 4], "pad_top": 0, "pad_left": 0,
            }},
            "proposal_context_text": "original",
            "condition_prompt_text": "original",
            "evidence_reasoning_text": "reasoning",
            "full_proposal_context_text": "original full",
            "full_condition_prompt_text": "original full",
            "full_evidence_reasoning_text": "full reasoning",
            "visual_evidence_key": "key",
            "component_masks": torch.zeros(0, 4, 4),
        }
        model = Model()
        provider = CycleLocalizationProvider.__new__(CycleLocalizationProvider)
        provider.dataset = [item]
        provider.model = model
        provider.threshold = 0.5
        provider._resolved_by_sample = {
            "bridge": {
                "dataset_index": 0,
                "parent_sample_id": "parent",
                "segmentation_sample_id": "seg",
            }
        }
        mask, audit = provider.localize(
            {"sample_id": "bridge"}, "raw generated description", (4, 4)
        )
        self.assertEqual(model.controller.adapters, ["default"])
        self.assertIn(
            "raw generated description",
            model.segmentation.batch.proposal_context_text[0],
        )
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(audit["protocol"], "qpsalm_cycle_localization_prompt_v1")
        self.assertEqual(audit["generated_text_characters"], 25)

    def test_cycle_localization_config_is_expert_vision_only_gt_mask(self) -> None:
        path = Path(
            "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml"
        )
        config = load_segdesc_config(path, {
            "stage": "bridge_expert",
            "evaluation_mode": "gt_mask",
            "region_protocol": "vision_only",
            "cycle_localization_samples": 8,
        })
        self.assertEqual(config.cycle_localization_samples, 8)
        with self.assertRaisesRegex(ValueError, "frozen expert Bridge"):
            load_segdesc_config(path, {
                "stage": "bridge_auto",
                "evaluation_mode": "gt_mask",
                "region_protocol": "vision_only",
                "cycle_localization_samples": 8,
            })

    def test_d_minus_one_sampling_uses_real_four_way_inputs(self) -> None:
        description_rows = []
        bridge_rows = []
        for index in range(12):
            description_rows.extend([
                {
                    "sample_id": f"global-{index}",
                    "parent_sample_id": f"global-parent-{index}",
                    "task_family": "global_caption",
                    "region_geometry": {"type": "full_image"},
                    "visual_ref": {"height": 256, "width": 256},
                },
                {
                    "sample_id": f"box-{index}",
                    "parent_sample_id": f"box-parent-{index}",
                    "task_family": "region_referring_expression",
                    "region_geometry": {"type": "box"},
                    "visual_ref": {"height": 512, "width": 512},
                },
            ])
            bridge_rows.extend([
                {
                    "bridge_record_id": f"mask-{index}",
                    "parent_sample_id": f"mask-parent-{index}",
                    "split": "train",
                    "region_source": "pseudo_instance_component",
                    "target_status": "present",
                    "region_mask": {"path": f"mask-{index}.npy"},
                    "visual_ref": {"original_size": [1024, 1024]},
                },
                {
                    "bridge_record_id": f"null-{index}",
                    "parent_sample_id": f"null-parent-{index}",
                    "split": "train",
                    "region_source": "no_target",
                    "target_status": "absent",
                    "region_mask": None,
                    "visual_ref": {"original_size": [512, 512]},
                },
            ])
        selected, audit = select_d_minus_one_mixture(
            description_rows, bridge_rows, count=32, seed=42
        )
        self.assertEqual(len(selected), 32)
        self.assertEqual(
            [row["_d_minus_one_category"] for row in selected[:4]],
            ["global", "box", "mask", "null"],
        )
        self.assertEqual(
            audit["category_counts"],
            {"global": 8, "box": 8, "mask": 8, "null": 8},
        )
        self.assertGreaterEqual(audit["num_native_source_sizes"], 2)
        self.assertFalse(audit["expert_truth_used"])
        repeated, repeated_audit = select_d_minus_one_mixture(
            description_rows, bridge_rows, count=32, seed=42
        )
        self.assertEqual(
            [row.get("sample_id") or row.get("bridge_record_id") for row in selected],
            [row.get("sample_id") or row.get("bridge_record_id") for row in repeated],
        )
        self.assertEqual(audit, repeated_audit)
        with self.assertRaisesRegex(ValueError, "32-64"):
            select_d_minus_one_mixture(
                description_rows, bridge_rows, count=16, seed=42
            )

    def test_d_minus_one_overfit_report_never_claims_zero_shot_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint_last.pt"
            segmentation_checkpoint = root / "segmentation.pt"
            segmentation_checkpoint.write_bytes(b"synthetic-segmentation")
            raw_generations = root / "raw_generations.jsonl"
            raw_generations.write_text("{}\n", encoding="utf-8")
            source_files = {
                "checkpoint": checkpoint,
                "dataset_summary": root / "dataset_summary.json",
                "gradient_gate": root / "description_gradient_gate.json",
                "raw_generations": raw_generations,
                "resolved_config": root / "resolved_config.json",
                "train_history": root / "train_history.jsonl",
                "trainable_manifest": root / "trainable_parameter_manifest.json",
                "validation_report": root / "eval_report.json",
            }
            for name, path in source_files.items():
                if name not in {"checkpoint", "raw_generations"}:
                    path.write_text("{}\n", encoding="utf-8")

            def sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            migration = {
                "source_path": str(segmentation_checkpoint),
                "source_sha256": sha(segmentation_checkpoint),
                "source_format": "qpsalm_sane_qmef_pmrd_v5",
                "source_step": 10,
                "allowed_prefixes": [
                    "controller.", "sane.", "qmef.", "pmrd.",
                ],
            }
            write_synthetic_segdesc_checkpoint(
                checkpoint,
                {
                    "description_protocol_assets": (
                        description_protocol_assets_spec()
                    ),
                    "segmentation_migration": migration,
                    "metadata": {
                        "stage": "overfit",
                        "checkpoint_role": "terminal_last",
                        "config": {
                            "seed": 42, "batch_size": 2, "max_steps": 100,
                        },
                    },
                },
                step=100,
            )

            config = SimpleNamespace(batch_size=2, max_steps=100)
            sampling = {
                "selected_samples": 32,
                "category_counts": {
                    "global": 8, "box": 8, "mask": 8, "null": 8,
                },
                "num_native_source_sizes": 3,
                "expert_truth_used": False,
                "bridge_target_authority": (
                    "deterministic_rule_candidate_not_expert"
                ),
                "description_builder_version": (
                    "description_benchmark_m1_v4_answer_trace"
                ),
                "bridge_builder_version": (
                    "landslide_bridge_m2_v7_expert_review_replay_bound"
                ),
                "bridge_status": "awaiting_expert_review",
                "description_index_sha256": "a" * 64,
                "description_validation_report_sha256": "b" * 64,
                "bridge_index_sha256": "c" * 64,
                "bridge_validation_report_sha256": "d" * 64,
            }
            report = build_d_minus_one_overfit_validation(
                config=config,
                sampling_audit=sampling,
                history_rows=[
                    {"loss": 2.0, "peak_reserved_gib": 20.0, "device_type": "cuda"},
                    {"loss": 0.2, "peak_reserved_gib": 21.0, "device_type": "cuda"},
                ],
                gradient_gate={"passed": True},
                validation_report={"generation_metrics": {
                    "num_caption": 16,
                    "num_structured": 16,
                    "raw_json_parse_rate": 0.5,
                    "raw_schema_valid_rate": 0.25,
                    "summary_nonempty_rate": 0.5,
                }},
                generation_rows=[
                    {"d_minus_one_category": name}
                    for name in ("global", "box", "mask", "null")
                ],
                trainable_manifest={"groups": [{"parameter_names": [
                    "controller.model.layer.lora_A.desc_adapter.weight",
                    "controller.model.layer.lora_B.desc_adapter.weight",
                ]}]},
                checkpoint_path=checkpoint,
                checkpoint_step=100,
                device_type="cuda",
                segmentation_migration=migration,
                reload_audit={
                    "protocol": "qpsalm_segdesc_strict_reload_probe_v1",
                    "passed": True,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha(checkpoint),
                    "checkpoint_step": 100,
                    "before_sha256": "a" * 64,
                    "corrupted_sha256": "b" * 64,
                    "restored_sha256": "a" * 64,
                    "segmentation_migration": migration,
                },
                source_files=source_files,
            )
            self.assertEqual(
                set(report["source_bindings"]),
                {
                    "checkpoint", "dataset_summary", "description_ontology",
                    "description_output_schema", "description_record_schema",
                    "gradient_gate", "raw_generations", "resolved_config",
                    "train_history", "trainable_manifest", "validation_report",
                },
            )

            # 即使旧 schema 文件自身哈希正确，当前协议资产重放也必须拒绝它。
            stale_schema = root / "stale_output_schema.json"
            stale_schema.write_text("{}\n", encoding="utf-8")
            stale_report = json.loads(json.dumps(report))
            stale_report["source_bindings"]["description_output_schema"] = {
                "path": str(stale_schema),
                "sha256": sha(stale_schema),
            }
            stale_report["observations"]["description_protocol_assets"]["assets"][
                "configs/qpsalm_description_output_v1.schema.json"
            ] = {
                "sha256": sha(stale_schema),
                "bytes": stale_schema.stat().st_size,
            }
            stale_validation = validate_d_minus_one_overfit_report(stale_report)
            self.assertEqual(stale_validation["status"], "engineering-invalid")
            self.assertIn(
                "description_protocol_assets_revalidated",
                stale_validation["errors"],
            )
        self.assertEqual(report["status"], "engineering-valid")
        self.assertTrue(report["overfit_subgate_passed"])
        self.assertFalse(report["d_minus_one_complete"])
        self.assertEqual(
            report["pending_external_subgates"],
            ["native_qwen_zero_shot_baseline"],
        )
        self.assertFalse(report["candidate_supervision_is_expert_truth"])

    def test_d_minus_one_zero_shot_input_and_combined_gate_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "description"
            (benchmark / "indexes").mkdir(parents=True)
            (benchmark / "reports").mkdir(parents=True)
            (benchmark / "data/dev/rsicap").mkdir(parents=True)

            def sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            rows = []
            for index in range(40):
                image_path = benchmark / f"data/dev/rsicap/parent-{index}.png"
                image_path.write_bytes(
                    f"synthetic-image-{index}".encode("utf-8")
                )
                rows.append({
                    "sample_id": f"sample-{index}",
                    "parent_sample_id": f"parent-{index}",
                    "source_dataset": "RSICap",
                    "task_family": "global_caption",
                    "split": "dev",
                    "instruction": "Describe the image.",
                    "answers": [{"text": f"caption {index}"}],
                    "visual_ref": {
                        "path": str(image_path),
                        "sha256": sha(image_path),
                        "storage_mode": "materialized_copy",
                    },
                })
            (benchmark / "indexes/dev.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            validation_path = benchmark / "reports/validation_report.json"
            validation_path.write_text(json.dumps({
                "builder_version": "description_benchmark_m1_v4_answer_trace",
                "verified_perceptual_duplicate_cross_split_groups": 0,
                "errors": [],
            }), encoding="utf-8")
            selected, input_audit = _input_audit(
                benchmark, "dev", max_samples=32, seed=42
            )
            self.assertEqual(len(selected), 32)
            self.assertEqual(input_audit["selected_samples"], 32)
            validation_hash = __import__("hashlib").sha256(
                validation_path.read_bytes()
            ).hexdigest()
            self.assertEqual(
                input_audit["validation_report_sha256"], validation_hash
            )
            selected_image = Path(selected[0]["visual_ref"]["path"])
            original_image = selected_image.read_bytes()
            selected_image.write_bytes(b"late-image-mutation")
            with self.assertRaisesRegex(RuntimeError, "image SHA"):
                _input_audit(benchmark, "dev", max_samples=32, seed=42)
            selected_image.write_bytes(original_image)

            zero_dir = root / "zero"
            overfit_dir = root / "overfit"
            zero_dir.mkdir()
            overfit_dir.mkdir()
            zero_raw = zero_dir / "raw_generations.jsonl"
            overfit_raw = overfit_dir / "raw_generations.jsonl"
            checkpoint = overfit_dir / "checkpoint_last.pt"
            bridge_index = root / "candidate_all.jsonl"
            description_index = benchmark / "indexes/dev.jsonl"
            bridge_validation = root / "bridge_validation.json"

            zero_raw.write_text("".join(
                json.dumps({
                    "sample_id": row["sample_id"],
                    "prediction": "synthetic nonempty caption",
                }) + "\n"
                for row in selected
            ), encoding="utf-8")
            model_dir = root / "qwen"
            model_dir.mkdir()
            model_config = model_dir / "config.json"
            model_config.write_text("{}\n", encoding="utf-8")
            metadata_hashes = {"config.json": sha(model_config)}
            metadata_snapshot = hashlib.sha256(json.dumps(
                metadata_hashes, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            (zero_dir / "eval_report.json").write_text(json.dumps({
                "protocol": ZERO_SHOT_PROTOCOL,
                "status": "engineering-valid",
                "errors": [],
                "checks": {"synthetic_complete": True},
                "num_samples": 32,
                "caption_token_f1": 0.1,
                "statistics_seed": 42,
                "region_capability_claimed": False,
                "raw_generations": str(zero_raw),
                "raw_generations_sha256": sha(zero_raw),
                "input_audit": input_audit,
                "model_audit": {
                    "model_dir": str(model_dir),
                    "metadata_file_sha256": metadata_hashes,
                    "metadata_snapshot_sha256": metadata_snapshot,
                },
            }), encoding="utf-8")

            segmentation_checkpoint = root / "segmentation.pt"
            segmentation_checkpoint.write_bytes(b"segmentation")
            bridge_index.write_text("{}\n", encoding="utf-8")
            bridge_validation.write_text("{}", encoding="utf-8")
            sampling = {
                "selected_samples": 32,
                "category_counts": {
                    "global": 8, "box": 8, "mask": 8, "null": 8,
                },
                "num_native_source_sizes": 2,
                "expert_truth_used": False,
                "bridge_target_authority": (
                    "deterministic_rule_candidate_not_expert"
                ),
                "sampling_seed": 42,
                "description_builder_version": (
                    "description_benchmark_m1_v4_answer_trace"
                ),
                "bridge_builder_version": (
                    "landslide_bridge_m2_v7_expert_review_replay_bound"
                ),
                "description_validation_report_sha256": validation_hash,
                "description_validation_report": str(validation_path),
                "description_index": str(description_index),
                "description_index_sha256": sha(description_index),
                "bridge_validation_report": str(bridge_validation),
                "bridge_validation_report_sha256": sha(bridge_validation),
                "bridge_index": str(bridge_index),
                "bridge_index_sha256": sha(bridge_index),
                "bridge_status": "awaiting_expert_review",
            }
            history_rows = [
                {
                    "step": 1,
                    "loss": 2.0,
                    "peak_reserved_gib": 20.0,
                    "device_type": "cuda",
                },
                {
                    "step": 100,
                    "loss": 0.2,
                    "peak_reserved_gib": 21.0,
                    "device_type": "cuda",
                },
            ]
            (overfit_dir / "train_history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in history_rows),
                encoding="utf-8",
            )
            generation_rows = [
                {"d_minus_one_category": name}
                for name in ("global", "box", "mask", "null")
            ]
            overfit_raw.write_text(
                "".join(json.dumps(row) + "\n" for row in generation_rows),
                encoding="utf-8",
            )
            generation_metrics = {
                "num_caption": 16,
                "num_structured": 16,
                "raw_json_parse_rate": 0.5,
                "raw_schema_valid_rate": 0.25,
                "summary_nonempty_rate": 0.5,
            }
            (overfit_dir / "eval_report.json").write_text(json.dumps({
                "generation_metrics": generation_metrics,
            }), encoding="utf-8")
            (overfit_dir / "description_gradient_gate.json").write_text(json.dumps({
                "passed": True,
                "all_required_streams_checked": True,
            }), encoding="utf-8")
            manifest = {"groups": [{"parameter_names": [
                "controller.model.layer.lora_A.desc_adapter.weight",
                "controller.model.layer.lora_B.desc_adapter.weight",
            ]}]}
            (overfit_dir / "trainable_parameter_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (overfit_dir / "resolved_config.json").write_text(json.dumps({
                "stage": "overfit", "batch_size": 2, "max_steps": 100,
            }), encoding="utf-8")
            (overfit_dir / "dataset_summary.json").write_text(json.dumps({
                "d_minus_one_sampling_audit": sampling,
            }), encoding="utf-8")
            migration = {
                "source_path": str(segmentation_checkpoint),
                "source_sha256": sha(segmentation_checkpoint),
                "source_format": "qpsalm_sane_qmef_pmrd_v5",
                "source_step": 10,
                "allowed_prefixes": [
                    "controller.", "sane.", "qmef.", "pmrd.",
                ],
            }
            write_synthetic_segdesc_checkpoint(
                checkpoint,
                {
                    "description_protocol_assets": (
                        description_protocol_assets_spec()
                    ),
                    "segmentation_migration": migration,
                    "metadata": {
                        "stage": "overfit",
                        "checkpoint_role": "terminal_last",
                        "training_progress": {
                            "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
                            "step": 100,
                        },
                        "config": {
                            "seed": 42, "batch_size": 2, "max_steps": 100,
                        },
                    },
                },
                step=100,
            )
            reload_audit = {
                "protocol": "qpsalm_segdesc_strict_reload_probe_v1",
                "passed": True,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha(checkpoint),
                "checkpoint_step": 100,
                "before_sha256": "a" * 64,
                "corrupted_sha256": "b" * 64,
                "restored_sha256": "a" * 64,
                "segmentation_migration": migration,
            }
            source_files = {
                "checkpoint": checkpoint,
                "dataset_summary": overfit_dir / "dataset_summary.json",
                "gradient_gate": overfit_dir / "description_gradient_gate.json",
                "raw_generations": overfit_raw,
                "resolved_config": overfit_dir / "resolved_config.json",
                "train_history": overfit_dir / "train_history.jsonl",
                "trainable_manifest": (
                    overfit_dir / "trainable_parameter_manifest.json"
                ),
                "validation_report": overfit_dir / "eval_report.json",
            }
            report = build_d_minus_one_overfit_validation(
                config=SimpleNamespace(batch_size=2, max_steps=100),
                sampling_audit=sampling,
                history_rows=history_rows,
                gradient_gate={"passed": True},
                validation_report={"generation_metrics": generation_metrics},
                generation_rows=generation_rows,
                trainable_manifest=manifest,
                checkpoint_path=checkpoint,
                checkpoint_step=100,
                device_type="cuda",
                segmentation_migration=migration,
                reload_audit=reload_audit,
                source_files=source_files,
            )
            overfit_report = overfit_dir / "d_minus_one_overfit_validation.json"
            overfit_report.write_text(json.dumps(report), encoding="utf-8")
            progress_path = overfit_dir / "training_progress_latest.json"
            progress_path.write_text(json.dumps({
                "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
                "step": 100,
            }), encoding="utf-8")
            terminal = validate_terminal_checkpoint_provenance(
                inspect_segdesc_checkpoint(checkpoint),
                checkpoint=checkpoint,
                expected_step=100,
                expected_stage="overfit",
                progress_key="training_progress",
                expected_progress_protocol=(
                    DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
                ),
                progress_artifact=progress_path,
                progress_artifact_name="training_progress_latest",
                history_artifact=overfit_dir / "train_history.jsonl",
                history_artifact_name="train_history",
            )
            completion = build_training_completion_report(
                protocol=DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
                report={
                    "output_dir": str(overfit_dir),
                    "stage": "overfit",
                    "steps": 100,
                    "checkpoint_last": str(checkpoint),
                    "terminal_checkpoint_audit": terminal,
                },
                required_artifacts={
                    "checkpoint_last": checkpoint,
                    "dataset_summary": overfit_dir / "dataset_summary.json",
                    "resolved_config": overfit_dir / "resolved_config.json",
                    "train_history": overfit_dir / "train_history.jsonl",
                    "training_progress_latest": progress_path,
                    "trainable_parameter_manifest": (
                        overfit_dir / "trainable_parameter_manifest.json"
                    ),
                },
                optional_artifacts={
                    "d_minus_one_overfit_validation": overfit_report,
                },
            )
            completion_path = overfit_dir / "training_report.json"
            completion_path.write_text(
                json.dumps(completion), encoding="utf-8"
            )
            gate = validate_d_minus_one_runs(zero_dir, overfit_dir)
            self.assertTrue(gate["d_minus_one_complete"])
            self.assertEqual(gate["status"], "engineering-valid")
            self.assertEqual(gate["protocol"], D_MINUS_ONE_GATE_PROTOCOL)
            gate_path = root / "d_minus_one_gate.json"
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            acceptance = validate_d_minus_one_gate(
                gate_path,
                expected_description_benchmark=benchmark,
            )
            self.assertEqual(
                acceptance["protocol"],
                D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
            )
            self.assertEqual(
                acceptance["description_source"]["benchmark_root"],
                str(benchmark.resolve(strict=False)),
            )
            self.assertEqual(
                acceptance[
                    "zero_shot_materialized_image_population_sha256"
                ],
                input_audit["materialized_image_population_sha256"],
            )
            self.assertEqual(
                acceptance["overfit_training_report"],
                str(completion_path.resolve(strict=False)),
            )
            original_completion = completion_path.read_bytes()
            completion_path.unlink()
            incomplete = validate_d_minus_one_runs(zero_dir, overfit_dir)
            self.assertFalse(incomplete["d_minus_one_complete"])
            self.assertIn(
                "overfit_training_completion_revalidated",
                incomplete["errors"],
            )
            completion_path.write_bytes(original_completion)
            other_benchmark = root / "different_description"
            other_benchmark.mkdir()
            with self.assertRaisesRegex(ValueError, "不是同一 M1.1 source"):
                validate_d_minus_one_gate(
                    gate_path,
                    expected_description_benchmark=other_benchmark,
                )
            cache_binding = report["observations"][
                "checkpoint_payload_provenance"
            ]["checkpoint_metadata"]["description_architecture_spec"][
                "description_cache_artifact_binding"
            ]
            cache_shard = Path(cache_binding["cache_dir"]) / "shard_00000.pt"
            cache_shard.write_bytes(cache_shard.read_bytes() + b"d-minus-one-drift")
            cache_drifted = validate_d_minus_one_overfit_report(report)
            self.assertIn(
                "description_cache_artifact_revalidated",
                cache_drifted["errors"],
            )
            checkpoint.write_bytes(b"drifted")
            drifted = validate_d_minus_one_runs(zero_dir, overfit_dir)
            self.assertFalse(drifted["d_minus_one_complete"])
            self.assertIn("overfit_checkpoint_bound", drifted["errors"])

    def test_cached_autoregressive_generation_uses_desc_adapter_until_eos(self) -> None:
        model = GenerationProtocolHarness()
        text = model.generate_from_state(
            SimpleNamespace(),
            torch.ones(1, 1, 8, 8),
            "Describe the region.",
            max_new_tokens=5,
        )
        self.assertEqual(text, "4 5")
        self.assertEqual(model.controller.adapter_calls, ["desc_adapter"])
        calls = model.controller.model.calls
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0]["used_inputs_embeds"])
        self.assertFalse(calls[0]["used_cache"])
        self.assertTrue(all(call["used_input_ids"] for call in calls[1:]))
        self.assertTrue(all(call["used_cache"] for call in calls[1:]))
        self.assertEqual(
            [call["attention_length"] for call in calls],
            [4, 5, 6],
        )

    def test_invalid_raw_json_scores_zero_even_when_repair_is_valid(self) -> None:
        target = json.dumps(valid_target())
        metric = DescriptionMetricAccumulator()
        row = metric.update(
            prediction='{"target_status":"absent","summary":"partial"}',
            target_text=target,
            references=[target],
            structured=True,
            metadata={"sample_id": "synthetic", "task_family": "no_target_response"},
        )
        report = metric.compute()
        self.assertFalse(row["raw_schema_valid"])
        self.assertTrue(row["repair_attempted"])
        self.assertTrue(row["repair_schema_valid"])
        self.assertEqual(report["raw_schema_valid_rate"], 0.0)
        self.assertEqual(report["raw_json_invalid_rate"], 1.0)
        self.assertEqual(report["repair_schema_valid_rate"], 1.0)
        self.assertEqual(report["repair_success_rate"], 1.0)
        self.assertEqual(report["repaired_only_field_score"], 1.0)
        self.assertEqual(report["empty_description_rate"], 0.0)
        self.assertEqual(report["no_factual_claim_samples"], 1)
        self.assertLess(report["structured_field_macro_f1"], 1.0)
        unparseable = parse_description_output("not a JSON object")
        self.assertEqual(len(unparseable.parse_errors), 1)
        self.assertTrue(unparseable.parse_errors[0].startswith("json_parse:"))

        valid = json.dumps(valid_target())
        wrapped = parse_description_output("Result:\n" + valid)
        self.assertIsNone(wrapped.parsed)
        self.assertFalse(wrapped.schema_valid)
        self.assertIn("root:extracted_wrapper_for_repair", wrapped.repair_actions)
        self.assertTrue(parse_description_output(json.dumps(wrapped.repaired)).schema_valid)

        empty_summary = valid_target()
        empty_summary["summary"] = ""
        empty = parse_description_output(json.dumps(empty_summary))
        self.assertFalse(empty.schema_valid)
        self.assertTrue(any("summary" in error for error in empty.parse_errors))

        nonfinite = valid_target()
        nonfinite_raw = json.dumps({**nonfinite, "confidence": float("nan")})
        rejected = parse_description_output(nonfinite_raw)
        self.assertIsNone(rejected.parsed)
        self.assertFalse(rejected.schema_valid)
        self.assertTrue(any("non-standard JSON" in error for error in rejected.parse_errors))
        self.assertNotIn("confidence", rejected.repaired)

    def test_json_artifact_writers_reject_nonfinite_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "report.json"
            history_path = root / "history.jsonl"
            write_json(report_path, {"value": 1.0})
            append_jsonl(history_path, {"step": 1, "loss": 0.5})
            original_report = report_path.read_bytes()
            original_history = history_path.read_bytes()

            with self.assertRaises(ValueError):
                write_json(report_path, {"value": float("nan")})
            with self.assertRaises(ValueError):
                append_jsonl(history_path, {"step": 2, "loss": float("inf")})
            with self.assertRaisesRegex(
                json.JSONDecodeError, "non-standard JSON numeric constant"
            ):
                strict_json_loads('{"value": Infinity}')

            self.assertEqual(report_path.read_bytes(), original_report)
            self.assertEqual(history_path.read_bytes(), original_history)
            self.assertFalse(report_path.with_suffix(".json.tmp").exists())

    def test_absent_output_rejects_region_attributes_in_both_schema_validators(self) -> None:
        hallucinated = valid_target("absent")
        hallucinated["region"].update({
            "location": "upper_left",
            "shape": "elongated",
            "elongation": "high",
        })
        hallucinated["evidence"].update({
            "terrain_support": "supports",
            "evidence_sufficiency": "sufficient",
        })
        raw = json.dumps(hallucinated)

        parsed = parse_description_output(raw)
        self.assertFalse(parsed.schema_valid)
        self.assertTrue(any("region.location" in error for error in parsed.parse_errors))
        self.assertTrue(any("evidence.terrain_support" in error for error in parsed.parse_errors))
        self.assertIn("region.location:absent_to_unavailable", parsed.repair_actions)
        self.assertIn("region.shape:absent_to_unavailable", parsed.repair_actions)
        self.assertTrue(all(value == "unavailable" for value in parsed.repaired["region"].values()))
        self.assertEqual(parsed.repaired["evidence"]["terrain_support"], "unavailable")
        self.assertEqual(parsed.repaired["evidence"]["evidence_sufficiency"], "unavailable")
        self.assertTrue(parse_description_output(json.dumps(parsed.repaired)).schema_valid)

        # jsonschema 未安装时，内置校验器必须执行完全相同的条件语义。
        with patch.object(output_protocol, "Draft202012Validator", None):
            fallback = output_protocol.parse_description_output(raw)
            self.assertFalse(fallback.schema_valid)
            self.assertTrue(any("region.location" in error for error in fallback.parse_errors))

            present = valid_target("present")
            present["region"].update({
                "location": "upper_left",
                "shape": "elongated",
                "elongation": "high",
            })
            self.assertTrue(
                output_protocol.parse_description_output(json.dumps(present)).schema_valid
            )

        metric = DescriptionMetricAccumulator()
        row = metric.update(
            prediction=raw,
            target_text=json.dumps(valid_target("absent")),
            references=[json.dumps(valid_target("absent"))],
            structured=True,
            metadata={"sample_id": "absent_hallucination", "task_family": "no_target_response"},
        )
        report = metric.compute()
        self.assertFalse(row["raw_schema_valid"])
        self.assertEqual(row["raw_field_accuracy"], 0.0)
        self.assertEqual(report["target_status"]["per_label"]["absent"]["recall"], 0.0)
        self.assertEqual(report["target_status"]["false_description_rate"], 1.0)

        # Schema-valid status 也不能掩盖自由文本 evidence 中的 unsupported claim。
        text_claim = valid_target("absent")
        text_claim["evidence"]["surface_observation"] = (
            "A landslide scar is clearly visible inside the region."
        )
        semantic_metric = DescriptionMetricAccumulator()
        semantic_row = semantic_metric.update(
            prediction=json.dumps(text_claim),
            target_text=json.dumps(valid_target("absent")),
            references=[json.dumps(valid_target("absent"))],
            structured=True,
            metadata={"sample_id": "absent_text_claim", "task_family": "no_target_response"},
        )
        semantic_report = semantic_metric.compute()
        self.assertTrue(semantic_row["raw_schema_valid"])
        self.assertEqual(semantic_row["unsupported_claims"], 1)
        self.assertTrue(semantic_row["false_description"])
        self.assertEqual(
            semantic_report["target_status"]["per_label"]["absent"]["recall"],
            1.0,
        )
        self.assertEqual(
            semantic_report["target_status"]["false_description_rate"],
            1.0,
        )

    def test_structured_metrics_report_raw_summary_quality(self) -> None:
        target = json.dumps(valid_target("present"))
        metric = DescriptionMetricAccumulator()
        row = metric.update(
            prediction=target,
            target_text=target,
            references=[target],
            structured=True,
            metadata={"sample_id": "summary", "task_family": "region_description_expert"},
        )
        report = metric.compute()
        self.assertEqual(row["summary_token_f1"], 1.0)
        self.assertTrue(row["summary_exact_match"])
        self.assertEqual(report["summary_token_f1"], 1.0)
        self.assertEqual(report["summary_nonempty_rate"], 1.0)

    def test_target_status_macro_uses_only_labels_present_in_evaluation(self) -> None:
        metric = DescriptionMetricAccumulator()
        for status in ("present", "absent"):
            target = json.dumps(valid_target(status))
            metric.update(
                prediction=target,
                target_text=target,
                references=[target],
                structured=True,
                metadata={"sample_id": status, "task_family": "bridge"},
            )
        status = metric.compute()["target_status"]
        self.assertEqual(status["active_labels"], ["present", "absent"])
        self.assertEqual(status["macro_f1"], 1.0)
        self.assertEqual(status["balanced_accuracy"], 1.0)

    def test_region_counterfactuals_preserve_canvas(self) -> None:
        mask = torch.zeros(2, 1, 8, 12)
        mask[0, :, 2:5, 3:7] = 1
        mask[1, :, 1:4, 8:10] = 1
        for mode in ("full_mask", "zero_mask", "shuffled_mask"):
            changed = counterfactual_region_masks(mask, mode)
            self.assertEqual(changed.shape, mask.shape)
            self.assertTrue(bool(torch.isfinite(changed).all()))
        self.assertEqual(counterfactual_region_masks(mask, "shuffled_mask").sum(), mask.sum())
        with self.assertRaisesRegex(ValueError, "同一 parent"):
            counterfactual_region_masks(mask, "region_swap")

    def test_region_swap_candidates_are_real_regions_from_same_parent(self) -> None:
        rows = [{
            "bridge_record_id": "current", "parent_sample_id": "p1",
            "region_id": "global", "region_source": "gt_global_mask",
            "target_status": "present", "region_mask": {"path": "global.npy"},
        }]
        catalog = [
            *rows,
            {
                "bridge_record_id": "same-parent", "parent_sample_id": "p1",
                "region_id": "component-1", "region_source": "pseudo_instance_component",
                "target_status": "present", "region_mask": {"path": "component.npy"},
            },
            {
                "bridge_record_id": "null", "parent_sample_id": "p1",
                "region_id": "no-target", "region_source": "no_target",
                "target_status": "absent", "region_mask": None,
            },
            {
                "bridge_record_id": "other-parent", "parent_sample_id": "p2",
                "region_id": "component-2", "region_source": "pseudo_instance_component",
                "target_status": "present", "region_mask": {"path": "other.npy"},
            },
        ]
        selected = same_parent_region_swap_candidates(
            rows, "current", catalog=catalog
        )
        self.assertEqual(
            [description_row["bridge_record_id"] for description_row in selected],
            ["same-parent"],
        )
        self.assertEqual(
            same_parent_region_swap_candidates(rows, "missing", catalog=catalog),
            [],
        )
        cross_parent = cross_parent_region_swap_candidates(
            rows, "current", catalog=catalog
        )
        self.assertEqual(
            [value["bridge_record_id"] for value in cross_parent],
            ["other-parent"],
        )
        self.assertNotEqual(
            cross_parent[0]["parent_sample_id"], rows[0]["parent_sample_id"]
        )

    def test_cross_parent_donor_resolution_works_with_batch_size_one(self) -> None:
        class FakeBank:
            def record(self, _component: str, parent: str) -> dict:
                family = "optical" if parent != "p3" else "terrain"
                return {"views": [{"source_families": [family]}]}

        rows = [
            {"bridge_record_id": "s1", "parent_sample_id": "p1"},
            {"bridge_record_id": "s1-view", "parent_sample_id": "p1"},
            {"bridge_record_id": "s2", "parent_sample_id": "p2"},
            {"bridge_record_id": "s3", "parent_sample_id": "p3"},
        ]
        dataset = DescriptionTaskDataset.__new__(DescriptionTaskDataset)
        dataset.stage = "bridge_expert"
        dataset.rows = rows
        dataset._rows_by_sample_id = {
            row["bridge_record_id"]: row for row in rows
        }
        dataset._request_family_cache = {}
        dataset.vision_bank = FakeBank()
        donor = dataset.cross_parent_modality_swap_request("s1")
        self.assertIsNotNone(donor)
        request, audit = donor
        self.assertEqual(request, ("multisource_parent", "p2"))
        self.assertEqual(audit["target_parent_sample_id"], "p1")
        self.assertEqual(audit["donor_parent_sample_id"], "p2")
        self.assertEqual(audit["common_modality_families"], ["optical"])

    def test_region_swap_loads_candidate_geometry_without_expert_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            numpy = __import__("numpy")
            donor_path = root / "donor.npy"
            numpy.save(donor_path, numpy.eye(4, dtype=numpy.uint8), allow_pickle=False)
            donor_hash = __import__("hashlib").sha256(donor_path.read_bytes()).hexdigest()

            class FakeBank:
                @staticmethod
                def record(_component: str, _parent: str) -> dict:
                    return {"views": [{"render_transform": {
                        "source_h": 4, "source_w": 4,
                        "resized_h": 4, "resized_w": 4,
                        "pad_top": 0, "pad_left": 0, "size": 4,
                    }}]}

            current = {
                "bridge_record_id": "current", "parent_sample_id": "p1",
                "region_id": "global", "region_source": "gt_global_mask",
                "target_status": "present", "region_mask": {"path": "current.npy"},
            }
            same_parent = {
                "bridge_record_id": "same", "parent_sample_id": "p1",
                "region_id": "component", "region_source": "pseudo_instance_component",
                "target_status": "present",
                "region_mask": {
                    "path": str(donor_path), "sha256": donor_hash, "shape": [4, 4],
                },
            }
            cross_parent = {
                **same_parent,
                "bridge_record_id": "cross", "parent_sample_id": "p2",
            }
            dataset = DescriptionTaskDataset.__new__(DescriptionTaskDataset)
            dataset.stage = "bridge_expert"
            dataset.vision_bank = FakeBank()
            dataset.rows = [current]
            dataset._rows_by_sample_id = {"current": current}
            dataset._region_swap_catalog = [current, same_parent, cross_parent]
            dataset._verified_mask_hashes = {}
            reference = torch.zeros(1, 4, 4)
            same = dataset.same_parent_region_swap("current", reference)
            cross = dataset.cross_parent_region_swap("current", reference)
            self.assertIsNotNone(same)
            self.assertIsNotNone(cross)
            self.assertEqual(same[1]["alternate_sample_id"], "same")
            self.assertEqual(cross[1]["donor_parent_sample_id"], "p2")

    def test_expert_data_requires_current_frozen_bridge_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory)
            (bridge / "reports").mkdir()
            (bridge / "manifests").mkdir()
            (bridge / "indexes").mkdir()
            binding_paths = {
                "pilot_parent_manifest_sha256": bridge / "manifests/pilot_parent_manifest.jsonl",
                "review_selection_sha256": bridge / "manifests/review_selection.jsonl",
                "candidate_index_sha256": bridge / "indexes/candidate_all.jsonl",
            }
            for index, path in enumerate(binding_paths.values()):
                path.write_text(json.dumps({"index": index}) + "\n", encoding="utf-8")
            report_path = bridge / "reports/validation_report.json"
            report_path.write_text(json.dumps({
                "status": "awaiting_expert_review",
                "require_expert_complete": False,
                "errors": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expert_pilot_frozen"):
                require_frozen_expert_bridge(bridge)

            write_bound_frozen_bridge(bridge)
            audit = require_frozen_expert_bridge(bridge)
            self.assertEqual(audit["status"], "expert_pilot_frozen")
            binding_paths["candidate_index_sha256"].write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "hash"):
                require_frozen_expert_bridge(bridge)

    def test_bridge_auto_replays_live_candidate_projection_without_expert_truth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory)
            (bridge / "reports").mkdir()
            (bridge / "indexes").mkdir()
            candidate = {
                "bridge_record_id": "bridge-1",
                "parent_sample_id": "parent-1",
                "split": "train",
                "region_source": "gt_global_mask",
                "candidate": {
                    "protocol": "landslide_bridge_rule_candidate_v1",
                    "is_expert_truth": False,
                    "summary": "rule candidate",
                },
            }
            candidate_path = bridge / "indexes/candidate_all.jsonl"
            auto_path = bridge / "indexes/auto_train.jsonl"
            encoded = json.dumps(candidate) + "\n"
            candidate_path.write_text(encoded, encoding="utf-8")
            auto_path.write_text(encoded, encoding="utf-8")
            report_path = bridge / "reports/validation_report.json"
            report_path.write_text(
                json.dumps({
                    "builder_version": BRIDGE_BUILDER_VERSION,
                    "status": "awaiting_expert_review",
                    "pilot_protocol_complete": True,
                    "records": 1,
                    "parents": 1,
                    "records_by_region_source": {"gt_global_mask": 1},
                    "errors": [],
                }),
                encoding="utf-8",
            )
            bank = SimpleNamespace(manifest={
                "input_fingerprints": {
                    "multisource_parent": {
                        "benchmark": str(bridge),
                        "index": "indexes/candidate_all.jsonl",
                        "size": candidate_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            candidate_path.read_bytes()
                        ).hexdigest(),
                        "validation_report": "reports/validation_report.json",
                        "validation_report_size": report_path.stat().st_size,
                        "validation_report_sha256": hashlib.sha256(
                            report_path.read_bytes()
                        ).hexdigest(),
                        "validation_builder_version": BRIDGE_BUILDER_VERSION,
                        "validation_status": "awaiting_expert_review",
                    },
                },
            })
            audit = require_engineering_bridge(bridge, bank)
            self.assertEqual(
                audit["protocol"], BRIDGE_ENGINEERING_AUDIT_PROTOCOL
            )
            self.assertFalse(audit["expert_truth_used"])

            drifted = copy.deepcopy(candidate)
            drifted["candidate"]["summary"] = "late mutation"
            auto_path.write_text(json.dumps(drifted) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "偏离 candidate"):
                require_engineering_bridge(bridge, bank)

            drifted["candidate"]["is_expert_truth"] = True
            forged = json.dumps(drifted) + "\n"
            candidate_path.write_text(forged, encoding="utf-8")
            auto_path.write_text(forged, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expert truth"):
                require_engineering_bridge(bridge, bank)

            drifted["candidate"]["is_expert_truth"] = False
            valid_drift = json.dumps(drifted) + "\n"
            candidate_path.write_text(valid_drift, encoding="utf-8")
            auto_path.write_text(valid_drift, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Cache binding"):
                require_engineering_bridge(bridge, bank)

    def test_description_indexes_replay_cache_bound_all_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "indexes").mkdir()
            (root / "reports").mkdir()
            row = {
                "sample_id": "sample-1",
                "parent_sample_id": "parent-1",
                "split": "train",
                "task_family": "global_caption",
                "answers": [{
                    "text": "accepted caption",
                    "caption_quality_weight": 1.0,
                }],
            }
            for name, rows in {
                "all": [row],
                "train": [row],
                "dev": [],
                "test": [],
                "train_eligible": [row],
            }.items():
                (root / f"indexes/{name}.jsonl").write_text(
                    "".join(json.dumps(value) + "\n" for value in rows),
                    encoding="utf-8",
                )
            report = root / "reports/validation_report.json"
            report.write_text(json.dumps({
                "builder_version": "description_benchmark_m1_v4_answer_trace",
                "num_records": 1,
                "deep_checked_records": 1,
                "num_parents": 1,
                "decoded_unique_images": 1,
                "materialized_files": 1,
                "train_eligible_records": 1,
                "verified_perceptual_duplicate_cross_split_groups": 0,
                "errors": [],
            }), encoding="utf-8")
            all_path = root / "indexes/all.jsonl"
            bank = SimpleNamespace(manifest={
                "input_fingerprints": {
                    "single_image": {
                        "benchmark": str(root),
                        "index": "indexes/all.jsonl",
                        "size": all_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            all_path.read_bytes()
                        ).hexdigest(),
                        "validation_report": "reports/validation_report.json",
                        "validation_report_size": report.stat().st_size,
                        "validation_report_sha256": hashlib.sha256(
                            report.read_bytes()
                        ).hexdigest(),
                        "validation_builder_version": (
                            "description_benchmark_m1_v4_answer_trace"
                        ),
                        "validation_status": "engineering-valid",
                    },
                },
            })
            audit = require_engineering_description(root, bank)
            self.assertEqual(
                audit["protocol"], DESCRIPTION_ENGINEERING_AUDIT_PROTOCOL
            )

            eligible = root / "indexes/train_eligible.jsonl"
            drifted = copy.deepcopy(row)
            drifted["answers"][0]["text"] = "late mutation"
            eligible.write_text(json.dumps(drifted) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "train_eligible"):
                require_engineering_description(root, bank)

            eligible.write_text(json.dumps(row) + "\n", encoding="utf-8")
            all_path.write_text(
                all_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Cache binding"):
                require_engineering_description(root, bank)

    def test_frozen_bridge_replays_review_sources_and_expert_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory) / "bridge"
            write_bound_frozen_bridge(bridge)
            require_frozen_expert_bridge(bridge)

            reviewer = bridge / "review_sources/reviewer_1.jsonl"
            original_reviewer = reviewer.read_text(encoding="utf-8")
            reviewer.write_text(
                original_reviewer + json.dumps({
                    "review_item_id": "late-edit",
                    "reviewer_id": "reviewer_1",
                    "decision": "accept",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "hash"):
                require_frozen_expert_bridge(bridge)
            reviewer.write_text(original_reviewer, encoding="utf-8")

            expert_val = bridge / "indexes/expert_val.jsonl"
            original_val = expert_val.read_text(encoding="utf-8")
            expert_val.write_text(
                original_val + original_val,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "投影|hash"):
                require_frozen_expert_bridge(bridge)

    def test_expert_and_d4_rows_never_fall_back_to_candidate(self) -> None:
        row = {
            "bridge_record_id": "expert-1",
            "region_source": "gt_global_mask",
            "review": {"status": "accepted"},
            "expert_target": {
                "structured_output": valid_target("present"),
                "summary": "Reviewed target.",
            },
        }
        _validate_expert_rows([row], stage="bridge_expert", split="train")
        missing = dict(row)
        missing.pop("expert_target")
        with self.assertRaisesRegex(ValueError, "人工审核 target"):
            _validate_expert_rows([missing], stage="bridge_expert", split="train")

        predicted = {
            **row,
            "bridge_record_id": "predicted-1",
            "region_source": "predicted_proposal",
            "prediction_provenance": {"out_of_fold_verified": False},
        }
        with self.assertRaisesRegex(ValueError, "OOF"):
            _validate_expert_rows([predicted], stage="predicted_mask", split="train")
        predicted["prediction_provenance"]["out_of_fold_verified"] = True
        _validate_expert_rows([predicted], stage="predicted_mask", split="train")

        dataset = DescriptionTaskDataset.__new__(DescriptionTaskDataset)
        dataset.stage = "predicted_mask"
        with self.assertRaisesRegex(ValueError, "禁止回退"):
            dataset._bridge_item(missing)

    def test_fixed_prediction_index_and_mask_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def sha256(path: Path) -> str:
                return __import__("hashlib").sha256(path.read_bytes()).hexdigest()

            numpy = __import__("numpy")
            mask_path = root / "masks/val/parent-1.npy"
            mask_path.parent.mkdir(parents=True)
            numpy.save(mask_path, numpy.eye(4, dtype=numpy.uint8), allow_pickle=False)

            bridge = root / "bridge"
            (bridge / "indexes").mkdir(parents=True)
            (bridge / "manifests").mkdir()
            (bridge / "reports").mkdir()
            source = {
                "bridge_record_id": "expert-val-1",
                "parent_sample_id": "parent-1",
                "split": "val",
                "instruction": "Describe the region.",
                "task_family": "region_description_expert",
                "target_status": "present",
                "region_id": "global",
                "region_source": "gt_global_mask",
                "review": {"status": "accepted"},
                "expert_target": {
                    "structured_output": valid_target("present"),
                    "summary": "Reviewed target.",
                },
            }
            write_bound_frozen_bridge(bridge, expert_rows=[source])
            gate_audit = require_frozen_expert_bridge(bridge)
            source_index = bridge / "indexes/expert_val.jsonl"

            segmentation_checkpoint = root / "segmentation.pt"
            torch.save({
                "format": SEGMENTATION_CHECKPOINT_FORMAT,
                "step": 6000,
            }, segmentation_checkpoint)

            def canonical_sha256(value: object) -> str:
                return __import__("hashlib").sha256(json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()

            row = {
                **source,
                "schema_version": PREDICTED_REGION_FORMAT,
                "bridge_record_id": "predicted::parent-1::6000",
                "region_id": "predicted_global",
                "region_source": "predicted_proposal",
                "region_mask": {
                    "path": str(mask_path),
                    "sha256": sha256(mask_path),
                    "shape": [4, 4],
                    "threshold": 0.5,
                },
                "prediction_provenance": {
                    "checkpoint": str(segmentation_checkpoint),
                    "checkpoint_sha256": sha256(segmentation_checkpoint),
                    "checkpoint_step": 6000,
                    "split": "val",
                    "fold_manifest": None,
                    "fold_manifest_sha256": None,
                    "checkpoint_fold": None,
                    "out_of_fold_verified": True,
                    "fold_audit": None,
                    "source_bridge_record_id": source["bridge_record_id"],
                    "source_expert_record_sha256": canonical_sha256(source),
                },
            }
            index_path = root / "predicted_val.jsonl"
            index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            mask_inventory = [{
                "parent_sample_id": "parent-1",
                "path": str(mask_path),
                "sha256": sha256(mask_path),
            }]
            (root / "report.json").write_text(json.dumps({
                "format": PREDICTED_REGION_FORMAT,
                "validation_protocol": FIXED_PREDICTION_ARTIFACT_PROTOCOL,
                "split": "val",
                "requested_max_parents": 0,
                "num_parents": 1,
                "num_eligible_parents": 1,
                "population_complete": True,
                "population_sha256": canonical_sha256(["parent-1"]),
                "mask_inventory_sha256": canonical_sha256(mask_inventory),
                "mask_bytes": mask_path.stat().st_size,
                "index": str(index_path),
                "index_sha256": sha256(index_path),
                "source_bridge_index": str(source_index),
                "source_bridge_index_sha256": sha256(source_index),
                "expert_gate_audit": gate_audit,
                "checkpoint": str(segmentation_checkpoint),
                "checkpoint_sha256": sha256(segmentation_checkpoint),
                "checkpoint_step": 6000,
            }), encoding="utf-8")
            audit = validate_predicted_index(
                index_path, split="val", expert_gate_audit=gate_audit
            )
            self.assertEqual(audit["index_sha256"], sha256(index_path))
            self.assertEqual(
                audit["segmentation_checkpoint_sha256"],
                sha256(segmentation_checkpoint),
            )
            stale_mask = root / "masks/val/stale.npy"
            numpy.save(
                stale_mask, numpy.zeros((4, 4), dtype=numpy.uint8),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "未绑定|目录"):
                validate_predicted_index(
                    index_path, split="val", expert_gate_audit=gate_audit
                )
            stale_mask.unlink()
            incomplete = root / "report.json.part"
            incomplete.write_bytes(b"incomplete-publication")
            with self.assertRaisesRegex(ValueError, "\.part"):
                validate_predicted_index(
                    index_path, split="val", expert_gate_audit=gate_audit
                )
            incomplete.unlink()
            numpy.save(
                mask_path, numpy.zeros((4, 4), dtype=numpy.uint8), allow_pickle=False
            )
            with self.assertRaisesRegex(ValueError, "mask hash"):
                validate_predicted_index(
                    index_path, split="val", expert_gate_audit=gate_audit
                )
            numpy.save(
                mask_path, numpy.eye(4, dtype=numpy.uint8), allow_pickle=False
            )
            index_path.write_text(
                index_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "path/hash"):
                validate_predicted_index(
                    index_path, split="val", expert_gate_audit=gate_audit
                )

            class FakeBank:
                @staticmethod
                def record(_component: str, _parent: str) -> dict:
                    return {"views": [{"render_transform": {
                        "source_h": 4, "source_w": 4,
                        "resized_h": 4, "resized_w": 4,
                        "pad_top": 0, "pad_left": 0, "size": 4,
                    }}]}

            dataset = DescriptionTaskDataset.__new__(DescriptionTaskDataset)
            dataset.stage = "predicted_mask"
            dataset.vision_bank = FakeBank()
            dataset._verified_mask_hashes = {}
            item = dataset._bridge_item(row)
            self.assertEqual(tuple(item["region_mask"].shape), (1, 4, 4))
            self.assertEqual(
                item["region_input_source_binding"]["protocol"],
                REGION_INPUT_SOURCE_PROTOCOL,
            )
            self.assertEqual(
                item["region_input_source_binding"]["source_mask"]["path"],
                str(mask_path),
            )
            numpy.save(mask_path, numpy.zeros((4, 4), dtype=numpy.uint8), allow_pickle=False)
            dataset._verified_mask_hashes = {}
            with self.assertRaisesRegex(ValueError, "hash"):
                dataset._bridge_item(row)

    def test_formal_counterfactual_gate_requires_negative_paired_ci(self) -> None:
        report = {"counterfactual_sensitivity": {
            "shuffled_mask": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.1},
            },
            "region_swap": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.2},
            },
            "cross_parent_region_swap": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.15},
            },
            "cross_parent_modality_swap": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.05}
            },
            "modality_removal": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_factual_claim_count_delta_ci": {"high": -0.1}
            },
        }}
        self.assertTrue(_counterfactual_gate(report)["passed"])
        report["counterfactual_sensitivity"]["region_swap"][
            "paired_target_score_delta_ci"
        ]["high"] = 0.01
        self.assertFalse(_counterfactual_gate(report)["passed"])
        report["counterfactual_sensitivity"]["region_swap"][
            "paired_target_score_delta_ci"
        ]["high"] = -0.2
        report["counterfactual_sensitivity"]["region_swap"]["n"] = 7
        self.assertFalse(_counterfactual_gate(report)["passed"])

    def test_counterfactual_input_audit_fingerprints_mask_and_backbone(self) -> None:
        def state(token: float, active_names: tuple[str, ...]):
            return SimpleNamespace(
                features=SimpleNamespace(samples=[[]]),
                valid_mask=torch.ones(1, 1, 2, 2),
                active_subsets=(ActiveModalitySubset(
                    active_names=active_names,
                    dropped_names=(),
                    signature="+".join(active_names),
                    is_full=True,
                ),),
                metadata=({
                    "component": "synthetic",
                    "parent_sample_id": "parent",
                    "cache_key": "cache",
                },),
                reference_hw=(2, 2),
                use_full_evidence=False,
                visual_evidence=SimpleNamespace(
                    tokens=torch.full((1, 2, 3), token),
                    token_mask=torch.ones(1, 2, dtype=torch.bool),
                    family_ids=torch.ones(1, 2, dtype=torch.long),
                    token_counts=(2,),
                    view_segments=[[('rgb', 2)]],
                    cache_keys=("cache",),
                    cache_format="synthetic",
                ),
            )

        baseline = state(0.0, ("rgb",))
        changed_state = state(1.0, ("rgb",))
        mask = torch.zeros(1, 1, 2, 2)
        state_audit = counterfactual_input_change_audit(
            mode="cross_parent_modality_swap",
            baseline_state=baseline,
            counterfactual_state=changed_state,
            baseline_mask=mask,
            counterfactual_mask=mask.clone(),
        )
        self.assertEqual(state_audit["changed_dimensions"], ["backbone_state"])
        mask_audit = counterfactual_input_change_audit(
            mode="shuffled_mask",
            baseline_state=baseline,
            counterfactual_state=baseline,
            baseline_mask=mask,
            counterfactual_mask=torch.ones_like(mask),
        )
        self.assertEqual(mask_audit["changed_dimensions"], ["region_mask"])
        self.assertTrue(mask_audit["changed"])

    def test_formal_counterfactual_gate_recomputes_frozen_parent_statistics(self) -> None:
        modes = tuple(FROZEN_GATE_COUNTERFACTUAL_MODES)
        rows = [
            {
                "sample_id": f"{mode}_{parent}",
                "parent_sample_id": parent,
                "mode": mode,
                "target_score_delta": -1.0,
                "factual_claim_count_delta": -1.0,
            }
            for mode in modes
            for parent in ("parent_a", "parent_b")
        ]
        report = {"counterfactual_sensitivity": {
            mode: {
                "requested": 2,
                "n": 2,
                "num_effective_parents": 2,
                "aggregation_unit": "parent",
                # 正式门禁必须忽略运行时 CI，并用冻结 seed 重新计算。
                "paired_target_score_delta_ci": {"high": 1.0},
                "paired_factual_claim_count_delta_ci": {"high": 1.0},
            }
            for mode in modes
        }}
        scientific = frozen_scientific_gate({})["scientific_protocol"]
        scientific["counterfactual_minimum_effective_parents"] = {
            mode: 2 for mode in modes
        }
        gate = _counterfactual_gate(report, rows, scientific)
        self.assertTrue(gate["passed"])
        self.assertEqual(
            gate["frozen_parent_statistics"]["region_swap"]["num_effective_parents"],
            2,
        )

    def test_formal_comparison_rejects_pre_gate_bound_evaluation_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw_generations.jsonl").write_text("", encoding="utf-8")
            (root / "counterfactual_generations.jsonl").write_text(
                "", encoding="utf-8"
            )
            report = {
                "protocol": "qpsalm_description_evaluation_v3",
                "num_samples": 0,
                "num_generated": 0,
                "generation_coverage": {
                    "complete": True,
                    "population_sha256": evaluation_population_sha256([]),
                },
            }
            (root / "eval_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "atomic_artifact_bound"):
                _rows(root, require_complete_generation=True)
            checkpoint = root / "checkpoint.pt"
            segmentation = root / "segmentation.pt"
            segmentation.write_bytes(b"comparison-segmentation")
            migration = {
                "source_path": str(segmentation),
                "source_sha256": hashlib.sha256(
                    segmentation.read_bytes()
                ).hexdigest(),
                "source_format": SEGMENTATION_CHECKPOINT_FORMAT,
                "source_step": 6000,
                "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
            }
            report["protocol"] = DESCRIPTION_EVALUATION_PROTOCOL
            report["generation_coverage"].update({
                "requested": 0,
                "eligible_samples": 0,
                "generated_samples": 0,
                "fraction": 0.0,
                "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
            })
            report.update({
                "evaluation_mask_artifacts": evaluation_mask_artifact_inventory([]),
                "evaluation_limit_audit": {
                    "protocol": "qpsalm_description_evaluation_limit_v1",
                    "requested_max_samples": 0,
                    "full_population_requested": True,
                    "dataset_rows_evaluated": 0,
                },
                "stage": "bridge_expert",
                "evaluation_mode": "gt_mask",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "",
                "checkpoint_step": 17,
                "checkpoint_metadata": {
                    "description_protocol_assets": description_protocol_assets_spec(),
                    "metadata": {
                        "stage": "bridge_expert", "config": {"seed": 42},
                    },
                    "segmentation_migration": migration,
                },
                "statistics_protocol": {"runtime_seed": 42},
                "counterfactual_sensitivity": {},
                "checkpoint_binding": {
                    "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
                    "checkpoint_stage": "bridge_expert",
                    "evaluation_data_stage": "bridge_expert",
                    "evaluation_mode": "gt_mask",
                    "saved_segmentation_migration": migration,
                    "segmentation_source_sha256_match": True,
                    "checkpoint_training_seed": 42,
                    "evaluation_seed": 42,
                    "seed_match": True,
                },
            })
            write_synthetic_segdesc_checkpoint(
                checkpoint,
                report["checkpoint_metadata"],
                step=17,
            )
            report["checkpoint_sha256"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            report["publication_audit"] = build_evaluation_publication_audit(
                root, report
            )
            (root / "eval_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            rows, observed = _rows(root, require_complete_generation=True)
            self.assertEqual(rows, {})
            self.assertEqual(observed["protocol"], report["protocol"])
            report["evaluation_limit_audit"].update({
                "requested_max_samples": 16,
                "full_population_requested": False,
            })
            (root / "eval_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "max-val-samples 0"):
                _rows(root, require_complete_generation=True)

    def test_evaluation_publication_reopens_bound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"synthetic-evaluation-checkpoint")
            row = {
                "sample_id": "sample-1",
                "parent_sample_id": "parent-1",
                "split": "val",
                "evaluation_mode": "gt_mask",
                "raw_generation": "initial generation",
            }
            raw_path = root / "raw_generations.jsonl"
            raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "counterfactual_generations.jsonl").write_text(
                "", encoding="utf-8"
            )
            report = {
                "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
                "stage": "bridge_expert",
                "split": "val",
                "evaluation_mode": "gt_mask",
                "num_samples": 1,
                "num_generated": 1,
                "generation_coverage": {
                    "requested": 0,
                    "eligible_samples": 1,
                    "generated_samples": 1,
                    "fraction": 1.0,
                    "complete": True,
                    "population_sha256": evaluation_population_sha256([row]),
                    "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
                },
                "counterfactual_sensitivity": {},
                "end_to_end_coverage": None,
                "cycle_localization": None,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "checkpoint_step": 7,
                "checkpoint_metadata": {},
                "checkpoint_binding": {
                    "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
                },
            }
            report["publication_audit"] = build_evaluation_publication_audit(
                root, report
            )
            rebuilt = revalidate_evaluation_publication(root, report)
            self.assertEqual(rebuilt, report["publication_audit"])
            row["raw_generation"] = "drifted generation"
            raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact/report 已漂移"):
                revalidate_evaluation_publication(root, report)

    def test_paired_evaluation_requires_same_segmentation_source(self) -> None:
        gate_audit = {"status": "expert_pilot_frozen"}
        population = "p" * 64

        def report(source: str) -> dict:
            return {
                "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
                "stage": "bridge_expert",
                "split": "test",
                "evaluation_mode": "gt_mask",
                "region_protocol": "vision_only",
                "num_samples": 2,
                "evaluation_limit_audit": {
                    "protocol": "qpsalm_description_evaluation_limit_v1",
                    "requested_max_samples": 0,
                    "full_population_requested": True,
                    "dataset_rows_evaluated": 2,
                },
                "expert_gate_audit": gate_audit,
                "generation_coverage": {"population_sha256": population},
                "checkpoint_binding": {
                    "saved_segmentation_migration": {
                        "source_sha256": source,
                    },
                },
            }

        paired = _validate_paired_evaluation_reports(
            report("a" * 64), report("a" * 64), expert_gate_audit=gate_audit
        )
        self.assertEqual(paired["segmentation_source_sha256"], "a" * 64)
        with self.assertRaisesRegex(ValueError, "同一 segmentation source"):
            _validate_paired_evaluation_reports(
                report("a" * 64), report("b" * 64),
                expert_gate_audit=gate_audit,
            )

    def test_m4_formal_pair_binds_training_controls_and_shared_d1(self) -> None:
        gate_audit = {"status": "expert_pilot_frozen"}

        def canonical_sha(payload) -> str:
            return hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()

        def dataset_audit(stage: str, split: str, tag: str) -> dict:
            return {
                "protocol": "qpsalm_description_dataset_population_v1",
                "stage": stage,
                "split": split,
                "num_samples": 2,
                "num_parents": 2,
                "population_sha256": hashlib.sha256(tag.encode()).hexdigest(),
                "tasks": {tag: 2},
                "sources": {"synthetic": 2},
                "caption_sampling_audit": {"protocol": "not_applicable"},
                "curriculum_audit": None,
                "expert_gate_audit": (
                    gate_audit if stage == "bridge_expert" else None
                ),
                "bridge_engineering_audit": None,
                "description_engineering_audit": None,
                "predicted_index_audit": None,
                "d_minus_one_sampling_audit": None,
            }

        def loader_binding(
            name: str,
            stage: str,
            audit: dict,
            loader_seed: int,
        ) -> dict:
            binding = {
                "protocol": "qpsalm_description_stream_binding_v1",
                "stream": name,
                "stage": stage,
                "dataset_audit_sha256": canonical_sha(audit),
                "dataset_samples": audit["num_samples"],
                "epoch_zero_batches": 1,
                "loader_seed": loader_seed,
                "num_workers": 0,
                "persistent_workers": False,
                "batch_sampler": {
                    "class": "EpochShuffleBatchSampler",
                    "protocol": "qpsalm_epoch_shuffle_batch_sampler_v1",
                    "batch_size": 2,
                    "seed": loader_seed,
                    "drop_last": False,
                },
            }
            binding["binding_sha256"] = canonical_sha(binding)
            return binding

        def training_audits(seed: int) -> tuple[dict, dict]:
            streams = {
                "bridge": dataset_audit(
                    "bridge_expert", "train", "bridge-train"
                ),
                "dior": dataset_audit(
                    "dior_alignment", "train", "dior-train"
                ),
                "global_caption": dataset_audit(
                    "rsicap_caption", "train", "caption-train"
                ),
            }
            bindings = {
                "bridge": loader_binding(
                    "bridge", "bridge_expert", streams["bridge"], seed + 11_003
                ),
                "dior": loader_binding(
                    "dior", "dior_alignment", streams["dior"], seed + 21_013
                ),
                "global_caption": loader_binding(
                    "global_caption", "rsicap_caption",
                    streams["global_caption"], seed + 31_019,
                ),
            }
            data_audit = {
                "protocol": (
                    "qpsalm_description_training_data_binding_v2_loader_bound"
                ),
                "training_streams": streams,
                "stream_loader_bindings": bindings,
                "validation": dataset_audit(
                    "bridge_expert", "val", "bridge-val"
                ),
                "stream_pattern": [
                    "bridge", "bridge", "bridge", "dior", "global_caption",
                ],
            }
            bridge = streams["bridge"]
            region_audit = {
                "protocol": REGION_TRAINING_DATA_PROTOCOL,
                "stage": "bridge_expert",
                "expert_gate_audit": bridge["expert_gate_audit"],
                "bridge_engineering_audit": None,
                "predicted_index_audit": None,
                "curriculum_audit": None,
                "population": {
                    key: bridge[key]
                    for key in (
                        "protocol", "stage", "split", "num_samples",
                        "num_parents", "population_sha256",
                    )
                },
            }
            return data_audit, region_audit

        data_audit, region_data_audit = training_audits(42)

        def report(encoder: str, checkpoint_sha: str) -> dict:
            entries = []
            for stage in (
                "mmrs_caption", "rsicap_caption", "dior_alignment", "bridge_auto",
            ):
                entries.append({
                    "stage": stage,
                    "checkpoint_role": (
                        "terminal_last"
                        if stage == "bridge_auto" else "validation_best"
                    ),
                    "seed": 42,
                    "checkpoint": f"/{stage}/{encoder}.pt",
                    "checkpoint_sha256": (
                        "d" * 64 if stage == "rsicap_caption"
                        else hashlib.sha256(f"{stage}:{encoder}".encode()).hexdigest()
                    ),
                    "config_sha256": hashlib.sha256(
                        f"config:{stage}:{encoder}".encode()
                    ).hexdigest(),
                    "controlled_config_sha256": hashlib.sha256(
                        f"controlled:{stage}".encode()
                    ).hexdigest(),
                    "data_audit_sha256": "a" * 64,
                    "region_data_audit_sha256": "b" * 64,
                    "d_minus_one_acceptance_sha256": "c" * 64,
                })
                entry = entries[-1]
                run_completion = {
                    "protocol": CHECKPOINT_RUN_COMPLETION_PROTOCOL,
                    "passed": True,
                    "training_report": {
                        "path": f"/{stage}/{encoder}/training_report.json",
                        "sha256": hashlib.sha256(
                            f"completion:{stage}:{encoder}".encode()
                        ).hexdigest(),
                        "bytes": 1,
                    },
                    "completion_protocol": (
                        DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
                    ),
                    "stage": stage,
                    "checkpoint_role": entry["checkpoint_role"],
                    "checkpoint_step": 100,
                    "selected_artifact_name": (
                        "checkpoint_last"
                        if stage == "bridge_auto" else "checkpoint_best"
                    ),
                    "selected_checkpoint": {
                        "path": entry["checkpoint"],
                        "sha256": entry["checkpoint_sha256"],
                        "bytes": 1,
                    },
                    "selection_report": (
                        None
                        if stage == "bridge_auto"
                        else {
                            "path": f"/{stage}/{encoder}/validation_best.json",
                            "sha256": hashlib.sha256(
                                f"selection:{stage}:{encoder}".encode()
                            ).hexdigest(),
                            "bytes": 1,
                        }
                    ),
                }
                entry["run_completion"] = run_completion
                entry["run_completion_sha256"] = canonical_sha(
                    run_completion
                )
            lineage = {
                "protocol": DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
                "target_stage": "bridge_expert",
                "entries": entries,
                "lineage_sha256": canonical_sha(entries),
            }
            migration = {"source_sha256": "s" * 64}
            return {
                "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
                "stage": "bridge_expert",
                "split": "val",
                "evaluation_mode": "gt_mask",
                "region_protocol": "vision_only",
                "num_samples": 2,
                "evaluation_limit_audit": {
                    "protocol": "qpsalm_description_evaluation_limit_v1",
                    "requested_max_samples": 0,
                    "full_population_requested": True,
                    "dataset_rows_evaluated": 2,
                },
                "expert_gate_audit": gate_audit,
                "generation_coverage": {"population_sha256": "p" * 64},
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_metadata": {
                    "description_protocol_assets": description_protocol_assets_spec(),
                    "metadata": {
                        "stage": "bridge_expert",
                        "config": {
                            "seed": 42,
                            "region_encoder": encoder,
                            "output_dir": f"outputs/{encoder}",
                            "learning_rate": 1.0e-4,
                        },
                        "data_audit": data_audit,
                        "region_data_audit": region_data_audit,
                        "stage_lineage": lineage,
                    },
                    "segmentation_migration": migration,
                },
                "statistics_protocol": {"runtime_seed": 42},
                "checkpoint_binding": {
                    "saved_segmentation_migration": migration,
                    "checkpoint_training_seed": 42,
                    "evaluation_seed": 42,
                    "seed_match": True,
                },
            }

        baseline = report("crop_only", "1" * 64)
        candidate = report("mgrr", "2" * 64)
        audit = _validate_paired_evaluation_reports(
            baseline,
            candidate,
            expert_gate_audit=gate_audit,
            expected_seed=42,
        )
        self.assertEqual(
            audit["training_control"]["shared_d1_checkpoint_sha256"],
            "d" * 64,
        )
        seed_123_data, seed_123_region = training_audits(123)
        seed_42_contract = _m4_cross_seed_training_population_contract(
            data_audit, region_data_audit, expected_seed=42
        )
        seed_123_contract = _m4_cross_seed_training_population_contract(
            seed_123_data, seed_123_region, expected_seed=123
        )
        self.assertEqual(seed_42_contract, seed_123_contract)
        drift_data = json.loads(json.dumps(seed_123_data))
        drift_data["training_streams"]["dior"]["population_sha256"] = "e" * 64
        drift_loader = drift_data["stream_loader_bindings"]["dior"]
        drift_loader["dataset_audit_sha256"] = canonical_sha(
            drift_data["training_streams"]["dior"]
        )
        drift_loader["binding_sha256"] = canonical_sha({
            key: value for key, value in drift_loader.items()
            if key != "binding_sha256"
        })
        drift_contract = _m4_cross_seed_training_population_contract(
            drift_data, seed_123_region, expected_seed=123
        )
        self.assertNotEqual(seed_42_contract, drift_contract)
        drift = json.loads(json.dumps(candidate))
        drift["checkpoint_metadata"]["metadata"]["config"]["learning_rate"] = 2.0e-4
        with self.assertRaisesRegex(ValueError, "训练配置"):
            _validate_paired_evaluation_reports(
                baseline,
                drift,
                expert_gate_audit=gate_audit,
                expected_seed=42,
            )
        drift = json.loads(json.dumps(candidate))
        lineage = drift["checkpoint_metadata"]["metadata"]["stage_lineage"]
        lineage["entries"][1]["checkpoint_sha256"] = "e" * 64
        lineage["lineage_sha256"] = canonical_sha(lineage["entries"])
        with self.assertRaisesRegex(ValueError, "D1 upstream"):
            _validate_paired_evaluation_reports(
                baseline,
                drift,
                expert_gate_audit=gate_audit,
                expected_seed=42,
            )

    def test_m4_suite_requires_all_five_baselines_and_one_candidate_set(self) -> None:
        seeds = (42, 123, 3407)
        candidate_main = tuple(f"candidate-main-{seed}" for seed in seeds)
        candidate_retrieval = tuple(
            f"candidate-retrieval-{seed}" for seed in seeds
        )

        def report(encoder: str) -> dict:
            return {
                "protocol": M4_SEED_GATE_PROTOCOL,
                "frozen_gate_audit": {"gate": "shared"},
                "pairs": [
                    {
                        "seed": seed,
                        "paired_evaluation": {"training_control": {
                            "baseline_region_encoder": encoder,
                            "candidate_region_encoder": "mgrr",
                        }},
                    }
                    for seed in seeds
                ],
                "artifact_checkpoint_fingerprints": {
                    "candidate_main": list(candidate_main),
                    "candidate_retrieval": list(candidate_retrieval),
                },
                "same_evaluation_population_across_seeds": True,
                "same_retrieval_population_across_seeds": True,
                "same_scientific_config_across_seeds": True,
                "same_training_population_across_seeds": True,
                "cross_seed_training_population_sha256": "a" * 64,
                "passed_fraction_gate": True,
            }

        reports = {
            encoder: report(encoder) for encoder in M4_BASELINE_REGION_ENCODERS
        }
        gate = aggregate_m4_region_encoder_reports(reports)
        self.assertEqual(gate["protocol"], M4_SUITE_GATE_PROTOCOL)
        self.assertEqual(gate["num_baselines"], 5)
        self.assertTrue(gate["passed"])

        incomplete = dict(reports)
        incomplete.pop("crop_only")
        with self.assertRaisesRegex(ValueError, "五种 baseline"):
            aggregate_m4_region_encoder_reports(incomplete)
        drift = json.loads(json.dumps(reports))
        drift["crop_only"]["artifact_checkpoint_fingerprints"][
            "candidate_main"
        ][0] = "different-candidate"
        with self.assertRaisesRegex(ValueError, "同一组三 seed"):
            aggregate_m4_region_encoder_reports(drift)
        drift = json.loads(json.dumps(reports))
        drift["crop_only"]["cross_seed_training_population_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "artifacts/population"):
            aggregate_m4_region_encoder_reports(drift)

    def test_formal_seed_binding_rejects_relabelled_artifact(self) -> None:
        report = {
            "checkpoint_sha256": "a" * 64,
            "checkpoint_metadata": {
                "description_protocol_assets": description_protocol_assets_spec(),
                "metadata": {"config": {"seed": 42}},
            },
            "checkpoint_binding": {
                "checkpoint_training_seed": 42,
                "evaluation_seed": 42,
                "seed_match": True,
            },
            "statistics_protocol": {"runtime_seed": 42},
        }
        binding = _formal_seed_binding(
            report, expected_seed=42, label="synthetic",
        )
        self.assertEqual(binding["checkpoint_config_seed"], 42)
        stale_assets = json.loads(json.dumps(report))
        stale_assets["checkpoint_metadata"]["description_protocol_assets"]["assets"][
            "configs/qpsalm_description_output_v1.schema.json"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ontology/schema binding 已漂移"):
            _formal_seed_binding(
                stale_assets, expected_seed=42, label="synthetic stale schema",
            )
        with self.assertRaisesRegex(ValueError, "seed binding 不一致"):
            _formal_seed_binding(
                report, expected_seed=123, label="synthetic relabel",
            )

    def test_three_seed_gate_rejects_duplicate_checkpoint_slots(self) -> None:
        def pair(index: int) -> dict:
            def value(offset: int) -> dict:
                return {"checkpoint_sha256": f"{index + offset:064x}"}

            return {"artifact_seed_binding": {
                "main_evaluation": {
                    "baseline": value(1), "candidate": value(11),
                },
                "retrieval_evaluation": {
                    "baseline": value(21), "candidate": value(31),
                },
            }}

        pairs = [pair(0), pair(100), pair(200)]
        fingerprints = _validate_three_seed_artifact_uniqueness(pairs)
        self.assertEqual(len(fingerprints["candidate_main"]), 3)
        pairs[2]["artifact_seed_binding"]["main_evaluation"]["candidate"] = (
            pairs[0]["artifact_seed_binding"]["main_evaluation"]["candidate"]
        )
        with self.assertRaisesRegex(ValueError, "重复 candidate_main checkpoint"):
            _validate_three_seed_artifact_uniqueness(pairs)

    def test_structured_disagreement_detects_region_change(self) -> None:
        first = valid_target("present")
        second = valid_target("present")
        second["region"]["shape"] = "elongated"
        self.assertGreater(structured_disagreement(first, second), 0.0)

    def test_same_image_retrieval_reports_parent_level_scores(self) -> None:
        region = [torch.eye(4)]
        text = [torch.eye(4)]
        report = _same_image_retrieval(
            region, text, ["parent_a", "parent_a", "parent_b", "parent_b"]
        )
        self.assertEqual(report["mean_r1"], 1.0)
        self.assertEqual(report["mean_r5"], 1.0)
        self.assertEqual(report["normalized_phrase_match"], 1.0)
        self.assertEqual(report["modifier_accuracy"], 1.0)
        self.assertEqual(report["mean_ranking_margin"], 1.0)
        self.assertEqual(report["aggregation_unit"], "parent")
        self.assertEqual(report["per_parent_mean_r1"], {"parent_a": 1.0, "parent_b": 1.0})

    def test_same_image_retrieval_treats_duplicate_phrases_as_multi_positive(self) -> None:
        region = [torch.eye(3)]
        text = [torch.eye(3)[torch.tensor([1, 0, 2])]]
        report = _same_image_retrieval(
            region,
            text,
            ["parent_a", "parent_a", "parent_a"],
            ["landslide scar", "landslide scar", "road"],
        )
        self.assertEqual(report["num_ambiguous_phrase_queries"], 2)
        self.assertEqual(report["mean_r1"], 1.0)
        self.assertEqual(report["mean_r5"], 1.0)
        self.assertGreater(report["mean_ranking_margin"], 0.0)

    def test_alignment_loss_does_not_treat_same_parent_duplicate_phrase_as_negative(self) -> None:
        logits = torch.tensor([
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        positives = alignment_positive_mask(
            ["landslide scar", "landslide scar", "road"],
            ["parent_a", "parent_a", "parent_a"],
            device=logits.device,
        )
        loss = multi_positive_alignment_loss(logits, positives)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertLess(float(loss), 0.05)

    def test_end_to_end_region_targets_never_fall_back_to_wrong_global_mask(self) -> None:
        rows = [
            {
                "sample_id": "global_positive",
                "parent_sample_id": "parent_positive",
                "task_family": "global_landslide_segmentation",
                "mask": {"positive_pixels": 50, "empty_mask": False},
            },
            {
                "sample_id": "referring_positive_instruction",
                "parent_sample_id": "parent_positive",
                "parent_referring_target_sample_id": "ref_target_1",
                "task_family": "referring_landslide_segmentation",
                "mask": {"positive_pixels": 10, "empty_mask": False},
            },
            {
                "sample_id": "referring_absent_instruction",
                "parent_sample_id": "parent_positive",
                "parent_referring_target_sample_id": "ref_target_absent",
                "task_family": "no_target_segmentation",
                "mask": {"positive_pixels": 0, "empty_mask": True},
            },
            {
                "sample_id": "global_empty",
                "parent_sample_id": "parent_empty",
                "task_family": "global_landslide_segmentation",
                "mask": {"positive_pixels": 0, "empty_mask": True},
            },
        ]
        resolver = EndToEndTargetResolver(rows)
        referring = resolver.resolve({
            "parent_sample_id": "parent_positive",
            "region_id": "referring_region",
            "region_source": "gt_referring_mask",
            "source_region_aliases": [{"sample_id": "ref_target_1"}],
        })
        self.assertEqual(referring["segmentation_sample_id"], "referring_positive_instruction")
        absent = resolver.resolve({
            "parent_sample_id": "parent_positive",
            "region_id": "absent_region",
            "region_source": "no_target",
            "source_region_aliases": [{"sample_id": "ref_target_absent"}],
        })
        self.assertEqual(absent["segmentation_task_family"], "no_target_segmentation")
        empty_global = resolver.resolve({
            "parent_sample_id": "parent_empty",
            "region_id": "no_target",
            "region_source": "no_target",
            "source_region_aliases": [],
        })
        self.assertEqual(empty_global["mapping_kind"], "empty_global_instruction")
        supported, reason = end_to_end_region_support({
            "region_source": "pseudo_instance_component",
            "source_region_aliases": [],
        })
        self.assertFalse(supported)
        self.assertEqual(reason, "component_without_language_target")
        with self.assertRaisesRegex(KeyError, "referring alias"):
            resolver.resolve({
                "parent_sample_id": "parent_positive",
                "region_id": "component_001",
                "region_source": "pseudo_instance_component",
                "source_region_aliases": [],
            })
        with self.assertRaisesRegex(KeyError, "global target.*非空"):
            resolver.resolve({
                "parent_sample_id": "parent_positive",
                "region_id": "no_target",
                "region_source": "no_target",
                "source_region_aliases": [],
            })

    def test_train_prediction_requires_out_of_fold_protocol(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-fold|fold"):
            export_predicted_regions(
                segmentation_config=None,
                checkpoint="missing.pt",
                source_index="missing.jsonl",
                split="train",
                output_dir="unused",
                device=torch.device("cpu"),
                threshold=0.5,
            )

    def test_oof_fold_indexes_are_parent_isolated_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmentation = root / "instruction_train.jsonl"
            bridge_root = root / "bridge"
            (bridge_root / "indexes").mkdir(parents=True)
            (bridge_root / "manifests").mkdir()
            (bridge_root / "reports").mkdir()
            bridge = bridge_root / "indexes/expert_train.jsonl"
            parents = [f"parent_{index}" for index in range(9)]
            segmentation_rows = [
                {
                    "sample_id": f"instruction_{parent}_{view}",
                    "parent_sample_id": parent,
                    "split": "train",
                }
                for parent in parents
                for view in range(2)
            ]
            bridge_rows = [
                {
                    "bridge_record_id": f"bridge_{parent}",
                    "parent_sample_id": parent,
                    "split": "train",
                    "region_source": "gt_global_mask",
                    "dataset_name": "dataset_a" if index < 6 else "dataset_b",
                    "modality_family_combo": "optical" if index % 2 else "multispectral+terrain",
                    "review": {"status": "accepted"},
                    "expert_target": {
                        "structured_output": valid_target("present"),
                        "summary": "reviewed target",
                    },
                }
                for index, parent in enumerate(parents)
            ]
            segmentation.write_text(
                "".join(json.dumps(row) + "\n" for row in segmentation_rows), encoding="utf-8"
            )
            bridge.write_text(
                "".join(json.dumps(row) + "\n" for row in bridge_rows), encoding="utf-8"
            )
            write_bound_frozen_bridge(bridge_root, expert_rows=bridge_rows)

            def sha256(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            first = build_oof_fold_indexes(
                segmentation_index=segmentation,
                bridge_index=bridge,
                output_dir=root / "folds_a",
                num_folds=3,
                seed=42,
            )
            second = build_oof_fold_indexes(
                segmentation_index=segmentation,
                bridge_index=bridge,
                output_dir=root / "folds_b",
                num_folds=3,
                seed=42,
            )
            self.assertEqual(first["parent_to_fold"], second["parent_to_fold"])
            self.assertEqual(
                first["expert_gate_audit"]["status"], "expert_pilot_frozen"
            )
            loaded = load_oof_manifest(root / "folds_a/fold_manifest.json")
            for fold, metadata in loaded["folds"].items():
                train_rows = [
                    json.loads(line)
                    for line in Path(metadata["train_index"]).read_text(encoding="utf-8").splitlines()
                ]
                train_parents = {row["parent_sample_id"] for row in train_rows}
                held_out = {
                    parent for parent, assigned in loaded["parent_to_fold"].items()
                    if assigned == fold
                }
                self.assertTrue(held_out)
                self.assertFalse(train_parents & held_out)

            # Merge 必须从 checkpoint payload 与 Vision Cache v3 index 指纹重放，
            # 不能只信任 predicted row 中复制的 out_of_fold_verified。
            source_by_parent = {
                row["parent_sample_id"]: row for row in bridge_rows
            }
            fold_predictions = []
            for fold, metadata in loaded["folds"].items():
                train_path = Path(metadata["train_index"])
                holdout_path = Path(metadata["holdout_index"])

                def fingerprint(path: Path) -> dict:
                    return {
                        "reference": str(path),
                        "status": "present",
                        "size": path.stat().st_size,
                        "sha256": sha256(path),
                    }

                checkpoint = root / f"seg_fold_{fold}.pt"
                torch.save({
                    "format": SEGMENTATION_CHECKPOINT_FORMAT,
                    "step": 25,
                    "config": {
                        "train_index": str(train_path),
                        "val_index": str(holdout_path),
                    },
                    "evidence_protocol": {
                        "input_protocol": {
                            "index_fingerprints": {
                                "train": fingerprint(train_path),
                                "val": fingerprint(holdout_path),
                                "test": fingerprint(holdout_path),
                            },
                        },
                    },
                }, checkpoint)
                checkpoint_audit = validate_oof_checkpoint_binding(
                    checkpoint=checkpoint,
                    fold_manifest=root / "folds_a/fold_manifest.json",
                    checkpoint_fold=fold,
                    prediction_index=holdout_path,
                )
                fold_output = root / f"predicted_fold_{fold}"
                prediction_rows = []
                held_out = sorted(
                    parent for parent, assigned in loaded["parent_to_fold"].items()
                    if assigned == fold
                )
                for parent in held_out:
                    mask = fold_output / f"masks/train/{parent}.npy"
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    __import__("numpy").save(
                        mask,
                        __import__("numpy").eye(4, dtype=__import__("numpy").uint8),
                        allow_pickle=False,
                    )
                    source = source_by_parent[parent]
                    prediction_rows.append({
                        **source,
                        "schema_version": PREDICTED_REGION_FORMAT,
                        "bridge_record_id": f"predicted::{parent}::25",
                        "region_id": "predicted_global",
                        "region_source": "predicted_proposal",
                        "region_mask": {
                            "path": str(mask),
                            "sha256": sha256(mask),
                            "shape": [4, 4],
                            "threshold": 0.5,
                        },
                        "prediction_provenance": {
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": sha256(checkpoint),
                            "checkpoint_step": 25,
                            "split": "train",
                            "fold_manifest": checkpoint_audit["fold_manifest"],
                            "fold_manifest_sha256": checkpoint_audit[
                                "fold_manifest_sha256"
                            ],
                            "checkpoint_fold": fold,
                            "out_of_fold_verified": True,
                            "fold_audit": checkpoint_audit,
                            "source_bridge_record_id": source["bridge_record_id"],
                            "source_expert_record_sha256": __import__("hashlib").sha256(
                                json.dumps(
                                    source,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                        },
                    })
                prediction_index = fold_output / f"predicted_train_{fold}.jsonl"
                prediction_index.write_text(
                    "".join(json.dumps(row) + "\n" for row in prediction_rows),
                    encoding="utf-8",
                )
                fold_predictions.append(prediction_index)
            merged_path = root / "predicted_train_oof.jsonl"
            merge_report = merge_oof_predictions(
                fold_manifest=root / "folds_a/fold_manifest.json",
                input_indexes=fold_predictions,
                output=merged_path,
            )
            self.assertEqual(merge_report["protocol"], OOF_MERGE_PROTOCOL)
            replay = revalidate_oof_merged_index(
                merged_path,
                expected_expert_gate_audit=loaded["expert_gate_audit"],
            )
            self.assertEqual(replay["num_parents"], len(parents))
            stale_mask = fold_predictions[0].parent / "masks/train/stale.npy"
            __import__("numpy").save(
                stale_mask,
                __import__("numpy").zeros((4, 4), dtype=__import__("numpy").uint8),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "未绑定|目录"):
                revalidate_oof_merged_index(merged_path)
            stale_mask.unlink()
            incomplete = fold_predictions[0].parent / "report.json.part"
            incomplete.write_bytes(b"incomplete-fold-publication")
            with self.assertRaisesRegex(ValueError, "\.part"):
                revalidate_oof_merged_index(merged_path)
            incomplete.unlink()
            checkpoint_to_mutate = root / "seg_fold_0.pt"
            checkpoint_payload = torch.load(
                checkpoint_to_mutate, map_location="cpu", weights_only=False
            )
            checkpoint_payload["step"] = 26
            torch.save(checkpoint_payload, checkpoint_to_mutate)
            with self.assertRaisesRegex(ValueError, "fold audit|checkpoint"):
                revalidate_oof_merged_index(merged_path)

            # 即使攻击者同步更新 manifest 中的文件 hash，错误 train partition 也会失败。
            manifest_path = root / "folds_b/fold_manifest.json"
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            fold_zero = tampered["folds"]["0"]
            train_zero = Path(fold_zero["train_index"])
            held_out_zero = next(
                parent for parent, assigned in tampered["parent_to_fold"].items()
                if assigned == "0"
            )
            leaked = next(
                row for row in segmentation_rows
                if row["parent_sample_id"] == held_out_zero
            )
            train_zero.write_text(
                train_zero.read_text(encoding="utf-8") + json.dumps(leaked) + "\n",
                encoding="utf-8",
            )
            fold_zero["train_index_sha256"] = sha256(train_zero)
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "精确排除分区"):
                load_oof_manifest(manifest_path)

    def test_segdesc_config_rejects_unknown_region_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml")
            payload = __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
            payload["region_encoder"] = "union_bbox"
            path = Path(directory) / "invalid.yaml"
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "region_encoder"):
                load_segdesc_config(path)

    def test_segdesc_config_rejects_nonfinite_runtime_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(
                "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml"
            )
            payload = __import__("yaml").safe_load(
                source.read_text(encoding="utf-8")
            )
            path = Path(directory) / "nonfinite.yaml"
            for name, value in (
                ("learning_rate", float("nan")),
                ("segmentation_retention_max_drop", float("inf")),
            ):
                invalid = {**payload, name: value}
                path.write_text(
                    __import__("yaml").safe_dump(invalid), encoding="utf-8"
                )
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, f"{name}.*有限数"
                ):
                    load_segdesc_config(path)

    def test_d0_config_requires_d_minus_one_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(
                "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml"
            )
            payload = __import__("yaml").safe_load(
                source.read_text(encoding="utf-8")
            )
            payload.update({"stage": "mmrs_caption", "d_minus_one_gate": None})
            path = Path(directory) / "d0.yaml"
            path.write_text(
                __import__("yaml").safe_dump(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "d_minus_one_gate"):
                load_segdesc_config(path)
            payload["d_minus_one_gate"] = "outputs/d_minus_one_gate.json"
            path.write_text(
                __import__("yaml").safe_dump(payload), encoding="utf-8"
            )
            self.assertEqual(
                load_segdesc_config(path).d_minus_one_gate,
                "outputs/d_minus_one_gate.json",
            )

    def test_d4_config_accepts_only_preregistered_curriculum_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml")
            payload = __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
            payload.update({
                "stage": "predicted_mask",
                "predicted_index": "outputs/synthetic_predicted.jsonl",
                "predicted_mask_fraction": 0.4,
            })
            path = Path(directory) / "invalid_d4.yaml"
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "curriculum tier"):
                load_segdesc_config(path)
            payload["predicted_mask_fraction"] = 0.5
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            self.assertEqual(load_segdesc_config(path).predicted_mask_fraction, 0.5)

    def test_d4_training_separates_oof_train_and_fixed_val_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_index = root / "predicted_train_oof.jsonl"
            val_index = root / "predicted_val.jsonl"
            train_index.write_text("{}\n", encoding="utf-8")
            val_index.write_text("{}\n", encoding="utf-8")
            config = SimpleNamespace(
                predicted_index=str(train_index),
                predicted_val_index=str(val_index),
            )
            audit = validate_predicted_training_indexes(
                config, stage="predicted_mask"
            )
            self.assertEqual(audit["train"], str(train_index.resolve()))
            self.assertEqual(audit["val"], str(val_index.resolve()))
            self.assertEqual(
                predicted_index_for_dataset(config, split="train", training=True),
                str(train_index),
            )
            self.assertEqual(
                predicted_index_for_dataset(config, split="val", training=False),
                str(val_index),
            )

            missing_val = SimpleNamespace(
                predicted_index=str(train_index), predicted_val_index=None,
            )
            with self.assertRaisesRegex(ValueError, "fixed val"):
                validate_predicted_training_indexes(
                    missing_val, stage="predicted_mask"
                )
            same = SimpleNamespace(
                predicted_index=str(train_index),
                predicted_val_index=str(train_index),
            )
            with self.assertRaisesRegex(ValueError, "不同产物"):
                validate_predicted_training_indexes(same, stage="predicted_mask")

    def test_d4_curriculum_transition_is_adjacent_and_checkpoint_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                gate_path,
                gate,
                checkpoint,
                expert_gate,
                train_audit,
                val_audit,
            ) = write_synthetic_d4_curriculum_gate(
                root, current_fraction=0.25, next_fraction=0.50
            )
            validated_path, validated_gate = validate_d4_curriculum_gate(gate_path)
            self.assertEqual(validated_path, gate_path.resolve(strict=False))
            self.assertEqual(validated_gate, gate)
            target_train_audit = {
                **train_audit,
                "curriculum_audit": {
                    **train_audit["curriculum_audit"],
                    "requested_predicted_fraction": 0.50,
                },
                "population": {
                    **train_audit["population"],
                    "population_sha256": "b" * 64,
                },
            }
            audit = validate_d4_curriculum_transition(
                gate_path,
                target_fraction=0.50,
                seed=42,
                initialize_from=checkpoint,
                expert_gate_audit=expert_gate,
                train_region_data_audit=target_train_audit,
                val_predicted_index_audit=val_audit,
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["current_fraction"], 0.25)
            self.assertEqual(audit["target_fraction"], 0.50)
            mismatched_sha256 = (
                "0" * 64
                if expert_gate["candidate_index_sha256"] != "0" * 64
                else "1" * 64
            )
            mismatched_cache_audit = copy.deepcopy(target_train_audit)
            mismatched_cache_audit["bridge_engineering_audit"].update({
                "candidate_index_sha256": mismatched_sha256,
                "cache_input_fingerprint": {
                    **mismatched_cache_audit[
                        "bridge_engineering_audit"
                    ]["cache_input_fingerprint"],
                    "sha256": mismatched_sha256,
                },
            })
            with self.assertRaisesRegex(ValueError, "cache-candidate"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.50,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=mismatched_cache_audit,
                    val_predicted_index_audit=val_audit,
                )
            with self.assertRaisesRegex(ValueError, "next_fraction"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.75,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=target_train_audit,
                    val_predicted_index_audit=val_audit,
                )
            fixed_row = json.loads(
                Path(val_audit["index"]).read_text(encoding="utf-8").splitlines()[0]
            )
            fixed_mask = Path(fixed_row["region_mask"]["path"])
            fixed_mask_bytes = fixed_mask.read_bytes()
            np = __import__("numpy")
            np.save(fixed_mask, np.zeros((2, 2), dtype=np.uint8), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "mask hash|重放"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.50,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=target_train_audit,
                    val_predicted_index_audit=val_audit,
                )
            fixed_mask.write_bytes(fixed_mask_bytes)
            evaluation_root = root / "evaluation"
            evaluation_row = json.loads(
                (evaluation_root / "raw_generations.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            evaluation_mask = (
                evaluation_root / evaluation_row["region_input_mask_artifact"]["path"]
            )
            evaluation_mask_bytes = evaluation_mask.read_bytes()
            np.save(
                evaluation_mask,
                np.zeros((4, 4), dtype=np.uint8),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "mask artifact.*漂移"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.50,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=target_train_audit,
                    val_predicted_index_audit=val_audit,
                )
            evaluation_mask.write_bytes(evaluation_mask_bytes)
            orphan_part = evaluation_root / "mask_artifacts/orphan.npy.part"
            orphan_part.write_bytes(b"incomplete-atomic-write")
            with self.assertRaisesRegex(ValueError, "临时|未绑定"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.50,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=target_train_audit,
                    val_predicted_index_audit=val_audit,
                )
            orphan_part.unlink()
            checkpoint.write_bytes(b"drifted-checkpoint")
            with self.assertRaisesRegex(ValueError, "path/hash|SHA-256"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.50,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=target_train_audit,
                    val_predicted_index_audit=val_audit,
                )

    def test_d4_first_tier_requires_complete_m4_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "M4 suite gate"):
                write_synthetic_d4_curriculum_gate(
                    Path(directory),
                    current_fraction=0.0,
                    next_fraction=0.25,
                )

    def test_formal_region_mask_is_replayed_from_bound_source_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic_d4_curriculum_gate(
                root, current_fraction=0.25, next_fraction=0.50
            )
            evaluation = root / "evaluation"
            raw_path = evaluation / "raw_generations.jsonl"
            report_path = evaluation / "eval_report.json"
            row = json.loads(raw_path.read_text(encoding="utf-8"))
            # 重新写成文件、metadata、population 全部自洽的错误输入；source mask 保持不变。
            forged = write_evaluation_mask_artifact(
                evaluation,
                role="region_input",
                sample_id=str(row["sample_id"]),
                mask=np.zeros((4, 4), dtype=np.uint8),
            )
            row["region_input_mask_artifact"] = forged
            row["region_area_fraction"] = 0.0
            raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["evaluation_mask_artifacts"] = evaluation_mask_artifact_inventory(
                [forged]
            )
            report["generation_coverage"]["population_sha256"] = (
                evaluation_population_sha256([row])
            )
            refresh_synthetic_evaluation_publication(evaluation, report)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source/cache transform.*不一致"):
                build_d4_curriculum_gate(
                    evaluation_dir=evaluation,
                    expert_report=evaluation / "expert_factuality_report.json",
                    bridge_benchmark=root / "bridge",
                    current_fraction=0.25,
                    next_fraction=0.50,
                    seed=42,
                )

    def test_formal_region_transform_must_match_bound_m3_cache_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic_d4_curriculum_gate(
                root, current_fraction=0.25, next_fraction=0.50
            )
            evaluation = root / "evaluation"
            raw_path = evaluation / "raw_generations.jsonl"
            report_path = evaluation / "eval_report.json"
            row = json.loads(raw_path.read_text(encoding="utf-8"))
            source_path = Path(
                row["region_input_source_binding"]["source_mask"]["path"]
            )
            source = torch.from_numpy(
                np.load(source_path, allow_pickle=False).astype(np.float32)
            )[None]
            forged_transform = {
                "source_h": 4,
                "source_w": 4,
                "resized_h": 2,
                "resized_w": 2,
                "pad_top": 1,
                "pad_left": 1,
                "size": 4,
            }
            forged_mask = transform_region_mask_to_cache(
                source, forged_transform
            )
            forged_artifact = write_evaluation_mask_artifact(
                evaluation,
                role="region_input",
                sample_id=str(row["sample_id"]),
                mask=forged_mask,
            )
            row["region_input_mask_artifact"] = forged_artifact
            row["region_input_source_binding"]["render_transform"] = (
                forged_transform
            )
            row["region_area_fraction"] = float(forged_mask.mean())
            raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["evaluation_mask_artifacts"] = (
                evaluation_mask_artifact_inventory([forged_artifact])
            )
            report["generation_coverage"]["population_sha256"] = (
                evaluation_population_sha256([row])
            )
            refresh_synthetic_evaluation_publication(evaluation, report)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "M3 cache record"):
                build_d4_curriculum_gate(
                    evaluation_dir=evaluation,
                    expert_report=evaluation / "expert_factuality_report.json",
                    bridge_benchmark=root / "bridge",
                    current_fraction=0.25,
                    next_fraction=0.50,
                    seed=42,
                )

    def test_d4_final_gate_binds_m7_to_evaluated_75_percent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                gate_path,
                payload,
                checkpoint,
                expert_gate,
                train_audit,
                val_audit,
            ) = write_synthetic_d4_curriculum_gate(
                root, current_fraction=0.75, next_fraction=None
            )
            audit = validate_d4_final_acceptance_for_m7(
                gate_path,
                seed=42,
                initialize_from=checkpoint,
                expert_gate_audit=expert_gate,
                train_region_data_audit=train_audit,
                val_predicted_index_audit=val_audit,
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["current_fraction"], 0.75)
            wrong_fraction_audit = {
                **train_audit,
                "curriculum_audit": {
                    **train_audit["curriculum_audit"],
                    "requested_predicted_fraction": 0.25,
                },
            }
            with self.assertRaisesRegex(ValueError, "region train data"):
                validate_d4_final_acceptance_for_m7(
                    gate_path,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=wrong_fraction_audit,
                    val_predicted_index_audit=val_audit,
                )
            payload["purpose"] = "curriculum_transition"
            gate_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "75% final gate"):
                validate_d4_final_acceptance_for_m7(
                    gate_path,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=train_audit,
                    val_predicted_index_audit=val_audit,
                )

    def test_d4_gate_derived_fields_are_recomputed_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                gate_path,
                payload,
                checkpoint,
                expert_gate,
                train_audit,
                val_audit,
            ) = write_synthetic_d4_curriculum_gate(
                root, current_fraction=0.25, next_fraction=0.50
            )
            target_train_audit = {
                **train_audit,
                "curriculum_audit": {
                    **train_audit["curriculum_audit"],
                    "requested_predicted_fraction": 0.50,
                },
                "population": {
                    **train_audit["population"],
                    "population_sha256": "b" * 64,
                },
            }
            payload["source_train_region_data_audit"] = {
                **train_audit,
                "predicted_index_audit": {"split": "train", "index": "forged"},
            }
            gate_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重新计算结果"):
                validate_d4_curriculum_gate(gate_path)
            with self.assertRaisesRegex(ValueError, "重新计算结果"):
                validate_d4_curriculum_transition(
                    gate_path,
                    target_fraction=0.50,
                    seed=42,
                    initialize_from=checkpoint,
                    expert_gate_audit=expert_gate,
                    train_region_data_audit=target_train_audit,
                    val_predicted_index_audit=val_audit,
                )

    def test_d4_same_stage_initialization_requires_explicit_curriculum_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "d4_25.pt"
            source = FakeSegDescCheckpointModel("mgrr")
            target = FakeSegDescCheckpointModel("mgrr")
            save_segdesc_checkpoint(
                checkpoint,
                source,
                step=25,
                segmentation_migration={"source_sha256": "a" * 64},
                metadata={
                    "stage": "predicted_mask",
                    "checkpoint_role": "validation_best",
                    "config": {"seed": 42, "predicted_mask_fraction": 0.25},
                },
            )
            publish_synthetic_description_run_completion(
                checkpoint,
                stage="predicted_mask",
                role="validation_best",
                step=25,
            )
            with self.assertRaisesRegex(RuntimeError, "expected_source"):
                initialize_segdesc_checkpoint(
                    checkpoint,
                    target,
                    target_stage="predicted_mask",
                    expected_seed=42,
                )
            step, report = initialize_segdesc_checkpoint(
                checkpoint,
                target,
                target_stage="predicted_mask",
                expected_seed=42,
                allow_same_stage_curriculum=True,
            )
            self.assertEqual(step, 25)
            self.assertTrue(report["initialization"]["same_stage_curriculum"])
            (checkpoint.parent / "training_report.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "source run"):
                initialize_segdesc_checkpoint(
                    checkpoint,
                    target,
                    target_stage="predicted_mask",
                    expected_seed=42,
                    allow_same_stage_curriculum=True,
                )

    def test_fixed_prediction_mode_requires_predicted_mask_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml")
            payload = __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
            payload.update({
                "stage": "bridge_expert",
                "evaluation_mode": "fixed_prediction",
                "predicted_index": "outputs/synthetic_predicted.jsonl",
            })
            path = Path(directory) / "invalid_fixed_stage.yaml"
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predicted_mask stage"):
                load_segdesc_config(path)

    def test_joint_default_pattern_is_fifty_twenty_five_twenty_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml")
            payload = __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
            payload.pop("joint_task_pattern", None)
            path = Path(directory) / "default_joint_pattern.yaml"
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            config = load_segdesc_config(path)
            self.assertEqual(
                config.resolved_joint_task_pattern(),
                ("segmentation", "global_caption", "segmentation", "region_description"),
            )

    def test_initialize_can_replace_only_region_encoder_while_resume_stays_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = FakeSegDescCheckpointModel("mgrr")
            target = FakeSegDescCheckpointModel("crop_only")
            target_region_before = {
                key: value.detach().clone() for key, value in target.mgrr.state_dict().items()
            }
            save_segdesc_checkpoint(
                path,
                source,
                step=7,
                segmentation_migration={"source": "synthetic"},
            )
            step, report = initialize_segdesc_checkpoint(path, target)
            self.assertEqual(step, 7)
            self.assertTrue(report["initialization"]["region_encoder_reinitialized"])
            self.assertTrue(torch.equal(source.shared.weight, target.shared.weight))
            for key, value in target.mgrr.state_dict().items():
                self.assertTrue(torch.equal(value, target_region_before[key]))
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                load_segdesc_checkpoint(path, target)

    def test_description_stage_transition_and_resume_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d0.pt"
            source = FakeSegDescCheckpointModel("mgrr")
            save_segdesc_checkpoint(
                path,
                source,
                step=5,
                segmentation_migration={"source": "synthetic"},
                metadata={
                    "stage": "mmrs_caption", "config": {"seed": 42},
                    "checkpoint_role": "validation_best",
                    "d_minus_one_acceptance": {
                        "protocol": D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
                        "passed": True,
                    },
                },
            )
            publish_synthetic_description_run_completion(
                path,
                stage="mmrs_caption",
                role="validation_best",
                step=5,
            )
            _step, initialization = initialize_segdesc_checkpoint(
                path,
                FakeSegDescCheckpointModel("mgrr"),
                target_stage="rsicap_caption",
                expected_seed=42,
            )
            lineage = build_description_stage_lineage(
                initialization, target_stage="rsicap_caption"
            )
            self.assertEqual(lineage["protocol"], DESCRIPTION_STAGE_LINEAGE_PROTOCOL)
            self.assertEqual(
                [entry["stage"] for entry in lineage["entries"]],
                ["mmrs_caption"],
            )
            self.assertTrue(
                lineage["entries"][0]["run_completion"]["passed"]
            )
            self.assertEqual(
                validate_description_stage_lineage(
                    lineage, expected_target_stage="rsicap_caption"
                ),
                lineage,
            )
            with self.assertRaisesRegex(RuntimeError, "seed lineage"):
                initialize_segdesc_checkpoint(
                    path,
                    FakeSegDescCheckpointModel("mgrr"),
                    target_stage="rsicap_caption",
                    expected_seed=123,
                )
            with self.assertRaisesRegex(RuntimeError, "顺序非法"):
                initialize_segdesc_checkpoint(
                    path,
                    FakeSegDescCheckpointModel("mgrr"),
                    target_stage="bridge_auto",
                )
            with self.assertRaisesRegex(RuntimeError, "stage 不一致"):
                load_segdesc_checkpoint(
                    path,
                    FakeSegDescCheckpointModel("mgrr"),
                    expected_stage="rsicap_caption",
                )

    def test_d3b_initialization_requires_d3a_terminal_checkpoint_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for role in (None, "validation_best", "terminal_last"):
                label = role or "missing"
                checkpoint = root / f"d3a_{label}.pt"
                metadata = {
                    "stage": "bridge_auto",
                    "config": {"seed": 42},
                }
                if role is not None:
                    metadata["checkpoint_role"] = role
                save_segdesc_checkpoint(
                    checkpoint,
                    FakeSegDescCheckpointModel("mgrr"),
                    step=100,
                    segmentation_migration={"source": "synthetic"},
                    metadata=metadata,
                )
                if role == "terminal_last":
                    publish_synthetic_description_run_completion(
                        checkpoint,
                        stage="bridge_auto",
                        role="terminal_last",
                        step=100,
                    )
                    _step, report = initialize_segdesc_checkpoint(
                        checkpoint,
                        FakeSegDescCheckpointModel("mgrr"),
                        target_stage="bridge_expert",
                        expected_seed=42,
                    )
                    self.assertEqual(
                        report["metadata"]["checkpoint_role"], "terminal_last"
                    )
                else:
                    with self.assertRaisesRegex(RuntimeError, "checkpoint role"):
                        initialize_segdesc_checkpoint(
                            checkpoint,
                            FakeSegDescCheckpointModel("mgrr"),
                            target_stage="bridge_expert",
                            expected_seed=42,
                        )

    def test_resume_requires_exact_saved_run_config(self) -> None:
        config = {
            "stage": "bridge_auto",
            "max_steps": 100,
            "grad_accum_steps": 4,
            "joint_task_pattern": [
                "segmentation", "global_caption",
                "segmentation", "region_description",
            ],
        }
        audit = validate_resume_run_config(
            {"metadata": {"config": dict(config)}}, config
        )
        self.assertTrue(audit["matched"])
        with self.assertRaisesRegex(RuntimeError, "max_steps"):
            validate_resume_run_config(
                {"metadata": {"config": dict(config)}},
                {**config, "max_steps": 200},
            )
        with self.assertRaisesRegex(RuntimeError, "缺少完整 config"):
            validate_resume_run_config({"metadata": {}}, config)

    def test_checkpoint_reload_restores_weights_and_resume_state_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = FakeSegDescCheckpointModel("mgrr")
            with torch.no_grad():
                source.shared.weight.fill_(2.5)
            source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
            source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1)
            complete = root / "complete.pt"
            save_segdesc_checkpoint(
                complete,
                source,
                step=11,
                segmentation_migration={"source": "synthetic"},
                optimizer=source_optimizer,
                scheduler=source_scheduler,
                metadata={"stage": "overfit"},
            )
            probe_step, reload_audit = verify_segdesc_checkpoint_reload(
                complete,
                source,
                optimizer=source_optimizer,
                scheduler=source_scheduler,
                scaler=None,
                expected_stage="overfit",
            )
            self.assertEqual(probe_step, 11)
            self.assertTrue(reload_audit["passed"])
            self.assertEqual(
                reload_audit["before_sha256"],
                reload_audit["restored_sha256"],
            )
            self.assertNotEqual(
                reload_audit["corrupted_sha256"],
                reload_audit["restored_sha256"],
            )
            target = FakeSegDescCheckpointModel("mgrr")
            target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
            target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
            step, _report = load_segdesc_checkpoint(
                complete,
                target,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
            )
            self.assertEqual(step, 11)
            for key, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]))

            weights_only = root / "weights_only.pt"
            save_segdesc_checkpoint(
                weights_only,
                source,
                step=12,
                segmentation_migration={"source": "synthetic"},
            )
            missing_target = FakeSegDescCheckpointModel("mgrr")
            with self.assertRaisesRegex(RuntimeError, "optimizer_state"):
                load_segdesc_checkpoint(
                    weights_only,
                    missing_target,
                    optimizer=torch.optim.SGD(missing_target.parameters(), lr=0.1),
                )

            payload = torch.load(complete, map_location="cpu", weights_only=False)
            payload["required_state_keys"] = payload["required_state_keys"][:-1]
            corrupt = root / "corrupt_inventory.pt"
            torch.save(payload, corrupt)
            with self.assertRaisesRegex(RuntimeError, "required_state_keys"):
                load_segdesc_checkpoint(corrupt, FakeSegDescCheckpointModel("mgrr"))

            payload = torch.load(complete, map_location="cpu", weights_only=False)
            payload["description_protocol_assets"]["assets"][
                "configs/qpsalm_description_output_v1.schema.json"
            ]["sha256"] = "0" * 64
            stale_schema = root / "stale_schema.pt"
            torch.save(payload, stale_schema)
            with self.assertRaisesRegex(RuntimeError, "ontology/schema"):
                load_segdesc_checkpoint(
                    stale_schema, FakeSegDescCheckpointModel("mgrr")
                )

    def test_checkpoint_metadata_rejects_nonfinite_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.pt"
            model = FakeSegDescCheckpointModel("mgrr")
            with self.assertRaisesRegex(RuntimeError, "finite.*JSON-compatible"):
                save_segdesc_checkpoint(
                    invalid,
                    model,
                    step=1,
                    segmentation_migration={"source": "synthetic"},
                    metadata={"stage": "overfit", "best_score": float("nan")},
                )
            self.assertFalse(invalid.exists())
            self.assertEqual(list(root.glob(".invalid.pt.*.tmp")), [])

            valid = root / "valid.pt"
            save_segdesc_checkpoint(
                valid,
                model,
                step=1,
                segmentation_migration={"source": "synthetic"},
                metadata={"stage": "overfit", "best_score": None},
            )
            payload = torch.load(valid, map_location="cpu", weights_only=False)
            payload["metadata"]["best_score"] = float("inf")
            torch.save(payload, valid)
            with self.assertRaisesRegex(RuntimeError, "finite.*JSON-compatible"):
                load_segdesc_checkpoint(
                    valid,
                    FakeSegDescCheckpointModel("mgrr"),
                )

    def test_checkpoint_resume_restores_process_rng_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            source = FakeSegDescCheckpointModel("mgrr")
            source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
            source_scheduler = torch.optim.lr_scheduler.StepLR(
                source_optimizer, step_size=1
            )
            random.seed(401)
            np.random.seed(402)
            torch.manual_seed(403)
            save_segdesc_checkpoint(
                path,
                source,
                step=4,
                segmentation_migration={"source": "synthetic"},
                optimizer=source_optimizer,
                scheduler=source_scheduler,
                metadata={"stage": "joint"},
            )
            expected = (
                random.random(),
                float(np.random.random()),
                torch.rand(4),
            )
            random.seed(901)
            np.random.seed(902)
            torch.manual_seed(903)
            target = FakeSegDescCheckpointModel("mgrr")
            target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
            target_scheduler = torch.optim.lr_scheduler.StepLR(
                target_optimizer, step_size=1
            )
            load_segdesc_checkpoint(
                path,
                target,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
            )
            observed = (
                random.random(),
                float(np.random.random()),
                torch.rand(4),
            )
            self.assertEqual(expected[0], observed[0])
            self.assertEqual(expected[1], observed[1])
            self.assertTrue(torch.equal(expected[2], observed[2]))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.pop("training_rng_state")
            legacy = Path(directory) / "legacy_rng.pt"
            torch.save(payload, legacy)
            legacy_target = FakeSegDescCheckpointModel("mgrr")
            with self.assertRaisesRegex(RuntimeError, "RNG state"):
                load_segdesc_checkpoint(
                    legacy,
                    legacy_target,
                    optimizer=torch.optim.SGD(legacy_target.parameters(), lr=0.1),
                )

    def test_formal_checkpoint_provenance_replays_payload_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "formal.pt"
            segmentation = root / "segmentation.pt"
            segmentation.write_bytes(b"formal-segmentation")
            migration = {
                "source_path": str(segmentation),
                "source_sha256": hashlib.sha256(
                    segmentation.read_bytes()
                ).hexdigest(),
                "source_format": SEGMENTATION_CHECKPOINT_FORMAT,
                "source_step": 6000,
                "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
            }
            migration_lineage = validate_segmentation_migration_lineage(
                migration, {"segmentation_migration": migration}
            )
            model = FakeSegDescCheckpointModel("mgrr")
            save_segdesc_checkpoint(
                checkpoint,
                model,
                step=23,
                segmentation_migration=migration,
                metadata={
                    "stage": "bridge_expert",
                    "config": {"seed": 42},
                    "segmentation_migration_lineage": migration_lineage,
                },
            )
            provenance = inspect_segdesc_checkpoint(checkpoint)
            self.assertEqual(
                provenance["protocol"],
                SEGDESC_CHECKPOINT_PROVENANCE_PROTOCOL,
            )
            self.assertEqual(provenance["checkpoint_step"], 23)
            self.assertEqual(
                provenance["checkpoint_metadata"]["metadata"]["stage"],
                "bridge_expert",
            )
            self.assertGreater(provenance["model_state_keys"], 0)

    def test_formal_checkpoint_provenance_replays_bound_description_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "formal_cache_bound.pt"
            segmentation = root / "segmentation.pt"
            segmentation.write_bytes(b"formal-cache-segmentation")
            migration = {
                "source_path": str(segmentation),
                "source_sha256": hashlib.sha256(
                    segmentation.read_bytes()
                ).hexdigest(),
                "source_format": SEGMENTATION_CHECKPOINT_FORMAT,
                "source_step": 6000,
                "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
            }
            migration_lineage = validate_segmentation_migration_lineage(
                migration, {"segmentation_migration": migration}
            )
            checkpoint_metadata = {
                "description_protocol_assets": description_protocol_assets_spec(),
                "segmentation_migration": migration,
                "metadata": {
                    "stage": "bridge_expert",
                    "checkpoint_role": "validation_best",
                    "config": {"seed": 42},
                    "segmentation_migration_lineage": migration_lineage,
                },
            }
            write_synthetic_segdesc_checkpoint(
                checkpoint,
                checkpoint_metadata,
                step=23,
            )
            run_completion = publish_synthetic_description_run_completion(
                checkpoint,
                stage="bridge_expert",
                role="validation_best",
                step=23,
            )
            checkpoint_sha256 = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            report = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": 23,
                "checkpoint_metadata": checkpoint_metadata,
                "stage": "bridge_expert",
                "evaluation_mode": "gt_mask",
                "statistics_protocol": {"runtime_seed": 42},
                "checkpoint_binding": {
                    "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
                    "checkpoint_stage": "bridge_expert",
                    "checkpoint_role": "validation_best",
                    "expected_checkpoint_role": "validation_best",
                    "evaluation_data_stage": "bridge_expert",
                    "evaluation_mode": "gt_mask",
                    "saved_segmentation_migration": migration,
                    "segmentation_source_sha256_match": True,
                    "checkpoint_training_seed": 42,
                    "evaluation_seed": 42,
                    "seed_match": True,
                    "run_completion": run_completion,
                },
            }
            audit = _validate_evaluation_checkpoint_provenance(root, report)
            self.assertTrue(
                audit["description_cache_artifact_provenance"]
                ["shard_replay"]["all_verified"]
            )
            completion_path = checkpoint.parent / "training_report.json"
            completion_bytes = completion_path.read_bytes()
            completion_path.unlink()
            with self.assertRaisesRegex(ValueError, "训练 run"):
                _validate_evaluation_checkpoint_provenance(root, report)
            completion_path.write_bytes(completion_bytes)

            binding = checkpoint_metadata["description_architecture_spec"][
                "description_cache_artifact_binding"
            ]
            shard = Path(binding["cache_dir"]) / "shard_00000.pt"
            shard.write_bytes(shard.read_bytes() + b"drift")
            with self.assertRaisesRegex(
                ValueError, "Description Vision Cache artifact"
            ):
                _validate_evaluation_checkpoint_provenance(root, report)

    def test_expert_factuality_is_aggregated_by_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview = root / "synthetic_preview.png"
            preview.write_bytes(b"synthetic-review-asset")
            generations = [
                {
                    "sample_id": sample,
                    "parent_sample_id": "parent_1",
                    "raw_metrics": {"raw_schema_valid": True},
                    "raw_generation": json.dumps(valid_target("present")),
                    "instruction": "Describe the selected landslide region.",
                    "visual_preview_path": str(preview),
                    "has_unavailable_modality": sample == "sample_a",
                }
                for sample in ("sample_a", "sample_b")
            ]
            (root / "raw_generations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
            )
            report = {
                "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
                "num_samples": len(generations),
                "generation_coverage": {"complete": True},
                "evaluation_limit_audit": {
                    "protocol": "qpsalm_description_evaluation_limit_v1",
                    "requested_max_samples": 0,
                    "full_population_requested": True,
                    "dataset_rows_evaluated": len(generations),
                },
                "evaluation_mode": "gt_mask",
            }
            publish_synthetic_evaluation(root, report)
            templates = build_expert_review_template(root)
            review_paths = []
            for reviewer in ("reviewer_1", "reviewer_2"):
                path = root / f"{reviewer}.jsonl"
                rows = [{
                    **template,
                    "reviewer_id": reviewer,
                    "family_scores": {
                        "target_status": 1.0,
                        "region_geometry": 1.0,
                        "surface": 1.0,
                        "terrain": 1.0,
                        "sar": 0.5,
                        "deformation": 0.5,
                        "surrounding_context": 0.5,
                        "summary": 0.5,
                    },
                    "claims": [
                        {**claim, "support": "supported"}
                        for claim in template["claims"]
                    ],
                } for template in templates]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                review_paths.append(path)
            report = aggregate_expert_factuality(root, review_paths, seed=42)
            self.assertEqual(report["num_parents"], 1)
            self.assertAlmostEqual(report["expert_region_factuality_score"], 0.75)
            self.assertEqual(report["expert_unsupported_claim_rate"], 0.0)
            self.assertEqual(report["unavailable_modality_num_samples"], 1)
            self.assertEqual(
                report["unavailable_modality_unsupported_claim_rate"], 0.0
            )
            self.assertAlmostEqual(
                report["field_agreement"]["target_status"]["exact_agreement"], 1.0
            )
            self.assertAlmostEqual(
                report["field_agreement"]["summary"]["krippendorff_alpha_nominal"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
