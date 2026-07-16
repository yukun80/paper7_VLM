#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit segmentation checkpoint migration and unified segdesc checkpoint I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.paths import resolve_project_path

from .model import (
    DESCRIPTION_ADAPTER_NAME,
    DESCRIPTION_SEQUENCE_PROTOCOL,
    SegmentationGroundedDescriptionModel,
)
from .mgrr import MGRR_PROTOCOL, MGRR_ROI_GRID_SIZES


SEGDESC_CHECKPOINT_FORMAT = "qpsalm_segdesc_v1"
D_MINUS_ONE_ACCEPTANCE_PROTOCOL = (
    "qpsalm_d_minus_one_acceptance_v5_training_completion_bound"
)
SEGDESC_CHECKPOINT_PROVENANCE_PROTOCOL = (
    "qpsalm_segdesc_checkpoint_provenance_v3_segmentation_lineage_bound"
)
TRAINING_RNG_STATE_PROTOCOL = "qpsalm_segdesc_training_rng_state_v1"
SEGMENTATION_MIGRATION_LINEAGE_PROTOCOL = (
    "qpsalm_segmentation_migration_lineage_v1_source_bytes_bound"
)
SEGMENTATION_STATE_PREFIXES = ("controller.", "sane.", "qmef.", "pmrd.")
FROZEN_QWEN_PREFIX = "segmentation.controller.model."
SEGMENTATION_ARCHITECTURE_FIELDS = (
    "preset", "controller", "qwen_model_path", "qwen_4bit",
    "qwen_lora_rank", "qwen_lora_alpha", "qwen_lora_dropout",
    "qwen_lora_last_n_layers", "qwen_lora_trainable",
    "qwen_max_text_tokens", "qwen_view_tokens_per_view", "qwen_view_pooling",
    "qwen_attn_implementation", "decoder_dim", "num_mask_tokens",
    "num_decoder_layers", "num_heads", "deformable_points",
    "use_pretrained_sane", "use_qmef", "use_query_spatial_attention",
    "use_mask_refinement",
)
REGION_ARCHITECTURE_FIELDS = (
    "region_encoder", "mgrr_protocol", "mgrr_max_components",
    "mgrr_component_coverage", "mgrr_roi_grid_sizes", "mgrr_roi_query_count",
)
DESCRIPTION_STAGE_PREDECESSOR = {
    "rsicap_caption": "mmrs_caption",
    "dior_alignment": "rsicap_caption",
    "bridge_auto": "dior_alignment",
    "bridge_expert": "bridge_auto",
    "predicted_mask": "bridge_expert",
}
DESCRIPTION_STAGE_LINEAGE_PROTOCOL = (
    "qpsalm_description_stage_lineage_v3_run_completion_bound"
)
DESCRIPTION_CHECKPOINT_ROLES = {"validation_best", "terminal_last"}
DESCRIPTION_STAGE_CHECKPOINT_ROLE = {
    "mmrs_caption": "validation_best",
    "rsicap_caption": "validation_best",
    "dior_alignment": "validation_best",
    "bridge_auto": "terminal_last",
    "bridge_expert": "validation_best",
    "predicted_mask": "validation_best",
}
DESCRIPTION_VARIANT_CONFIG_FIELDS = {"region_encoder", "output_dir"}
DESCRIPTION_PROTOCOL_ASSETS = (
    "configs/description_ontology_v1.yaml",
    "configs/qpsalm_description_record_v2.schema.json",
    "configs/qpsalm_description_output_v1.schema.json",
)
DESCRIPTION_LINEAGE_STAGE_PREFIX = {
    "rsicap_caption": ("mmrs_caption",),
    "dior_alignment": ("mmrs_caption", "rsicap_caption"),
    "bridge_auto": (
        "mmrs_caption", "rsicap_caption", "dior_alignment",
    ),
    "bridge_expert": (
        "mmrs_caption", "rsicap_caption", "dior_alignment", "bridge_auto",
    ),
    "predicted_mask": (
        "mmrs_caption", "rsicap_caption", "dior_alignment", "bridge_auto",
        "bridge_expert",
    ),
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_description_stage_lineage(
    value: Any,
    *,
    expected_target_stage: str | None = None,
) -> dict[str, Any]:
    """Validate exact D0-D4 order, identity hashes and shared D-1 ancestry."""
    if (
        not isinstance(value, dict)
        or value.get("protocol") != DESCRIPTION_STAGE_LINEAGE_PROTOCOL
        or not isinstance(value.get("entries"), list)
    ):
        raise RuntimeError("description stage lineage protocol/entries 非法")
    target_stage = str(value.get("target_stage") or "")
    if expected_target_stage is not None and target_stage != expected_target_stage:
        raise RuntimeError(
            "description stage lineage target 不一致: "
            f"expected={expected_target_stage!r} observed={target_stage!r}"
        )
    expected_prefix = DESCRIPTION_LINEAGE_STAGE_PREFIX.get(target_stage)
    if expected_prefix is None:
        raise RuntimeError(f"description stage lineage target 非法: {target_stage!r}")
    entries = [dict(item) for item in value["entries"] if isinstance(item, dict)]
    if len(entries) != len(value["entries"]):
        raise RuntimeError("description stage lineage entry 必须全部为 object")
    stages = tuple(str(item.get("stage") or "") for item in entries)
    if target_stage == "predicted_mask":
        valid_order = (
            len(stages) >= len(expected_prefix)
            and stages[:len(expected_prefix)] == expected_prefix
            and all(stage == "predicted_mask" for stage in stages[len(expected_prefix):])
        )
    else:
        valid_order = stages == expected_prefix
    if not valid_order:
        raise RuntimeError(
            "description stage lineage 顺序非法: "
            f"target={target_stage!r} stages={stages}"
        )
    seeds: set[int] = set()
    acceptance_hashes: set[str] = set()
    checkpoint_hashes: list[str] = []
    required_hash_fields = (
        "checkpoint_sha256", "config_sha256", "controlled_config_sha256",
        "data_audit_sha256", "region_data_audit_sha256",
        "d_minus_one_acceptance_sha256", "run_completion_sha256",
    )
    for entry in entries:
        if not str(entry.get("checkpoint") or ""):
            raise RuntimeError("description stage lineage entry 缺少 checkpoint path")
        stage = str(entry.get("stage") or "")
        expected_role = DESCRIPTION_STAGE_CHECKPOINT_ROLE.get(stage)
        if entry.get("checkpoint_role") != expected_role:
            raise RuntimeError(
                "description stage lineage checkpoint role 非法: "
                f"stage={stage!r} expected={expected_role!r} "
                f"observed={entry.get('checkpoint_role')!r}"
            )
        from .run_artifacts import CHECKPOINT_RUN_COMPLETION_PROTOCOL
        run_completion = entry.get("run_completion")
        expected_selected_artifact = (
            "checkpoint_last"
            if expected_role == "terminal_last" else "checkpoint_best"
        )
        expected_selection_binding = run_completion.get(
            "selection_report"
        ) if isinstance(run_completion, dict) else None
        if (
            not isinstance(run_completion, dict)
            or run_completion.get("protocol")
            != CHECKPOINT_RUN_COMPLETION_PROTOCOL
            or run_completion.get("passed") is not True
            or run_completion.get("stage") != stage
            or run_completion.get("checkpoint_role") != expected_role
            or (run_completion.get("selected_checkpoint") or {}).get(
                "sha256"
            ) != entry.get("checkpoint_sha256")
            or entry.get("run_completion_sha256")
            != _canonical_sha256(run_completion)
            or run_completion.get("selected_artifact_name")
            != expected_selected_artifact
            or not isinstance(run_completion.get("training_report"), dict)
            or (
                expected_role == "validation_best"
                and not isinstance(expected_selection_binding, dict)
            )
            or (
                expected_role == "terminal_last"
                and expected_selection_binding is not None
            )
        ):
            raise RuntimeError(
                "description stage lineage run completion binding 非法"
            )
        for field in required_hash_fields:
            observed = entry.get(field)
            if not isinstance(observed, str) or len(observed) != 64:
                raise RuntimeError(
                    f"description stage lineage entry 缺少有效 {field}"
                )
        try:
            seeds.add(int(entry.get("seed")))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("description stage lineage seed 非法") from exc
        checkpoint_hashes.append(str(entry["checkpoint_sha256"]))
        acceptance_hashes.add(str(entry["d_minus_one_acceptance_sha256"]))
    if len(seeds) != 1 or len(acceptance_hashes) != 1:
        raise RuntimeError("description stage lineage seed 或 D-1 ancestry 不一致")
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise RuntimeError("description stage lineage 重复使用同一 checkpoint")
    if value.get("lineage_sha256") != _canonical_sha256(entries):
        raise RuntimeError("description stage lineage canonical hash 不一致")
    return {
        "protocol": DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
        "target_stage": target_stage,
        "entries": entries,
        "lineage_sha256": value["lineage_sha256"],
    }


def build_description_stage_lineage(
    source_report: dict[str, Any],
    *,
    target_stage: str,
) -> dict[str, Any]:
    """Extend the immutable D0-D4 initialization chain by one checkpoint."""
    source_metadata = dict(source_report.get("metadata") or {})
    initialization = dict(source_report.get("initialization") or {})
    source_stage = str(source_metadata.get("stage") or "")
    source_checkpoint = str(initialization.get("source_checkpoint") or "")
    source_sha256 = str(initialization.get("source_checkpoint_sha256") or "")
    run_completion = initialization.get("source_run_completion")
    from .run_artifacts import CHECKPOINT_RUN_COMPLETION_PROTOCOL
    if (
        not source_stage
        or not source_checkpoint
        or len(source_sha256) != 64
        or initialization.get("source_stage") != source_stage
        or initialization.get("target_stage") != target_stage
        or not isinstance(run_completion, dict)
        or run_completion.get("protocol")
        != CHECKPOINT_RUN_COMPLETION_PROTOCOL
        or run_completion.get("passed") is not True
    ):
        raise RuntimeError("description stage lineage 缺少有效 initialize source binding")
    prior = source_metadata.get("stage_lineage")
    if prior is None:
        entries: list[dict[str, Any]] = []
    else:
        validated_prior = validate_description_stage_lineage(
            prior,
            expected_target_stage=source_stage,
        )
        entries = [dict(value) for value in validated_prior["entries"]]
    source_config = dict(source_metadata.get("config") or {})
    d_minus_one_acceptance = source_metadata.get("d_minus_one_acceptance")
    if (
        not isinstance(d_minus_one_acceptance, dict)
        or d_minus_one_acceptance.get("protocol")
        != D_MINUS_ONE_ACCEPTANCE_PROTOCOL
        or d_minus_one_acceptance.get("passed") is not True
    ):
        raise RuntimeError("source checkpoint 缺少已通过的 D-1 acceptance")
    controlled_config = {
        key: value
        for key, value in source_config.items()
        if key not in DESCRIPTION_VARIANT_CONFIG_FIELDS
    }
    entry = {
        "stage": source_stage,
        "checkpoint_role": source_metadata.get("checkpoint_role"),
        "checkpoint": source_checkpoint,
        "checkpoint_sha256": source_sha256,
        "seed": source_config.get("seed"),
        "region_encoder": source_config.get("region_encoder"),
        "config_sha256": _canonical_sha256(source_config),
        "controlled_config_sha256": _canonical_sha256(controlled_config),
        "data_audit_sha256": _canonical_sha256(source_metadata.get("data_audit")),
        "region_data_audit_sha256": _canonical_sha256(
            source_metadata.get("region_data_audit")
        ),
        "d_minus_one_acceptance_sha256": _canonical_sha256(
            d_minus_one_acceptance
        ),
        "run_completion": dict(run_completion),
        "run_completion_sha256": _canonical_sha256(run_completion),
    }
    entries.append(entry)
    lineage = {
        "protocol": DESCRIPTION_STAGE_LINEAGE_PROTOCOL,
        "target_stage": target_stage,
        "entries": entries,
        "lineage_sha256": _canonical_sha256(entries),
    }
    return validate_description_stage_lineage(
        lineage,
        expected_target_stage=target_stage,
    )


def validate_resume_run_config(
    checkpoint_report: dict[str, Any],
    current_config: dict[str, Any],
) -> dict[str, Any]:
    """Require a resume to preserve the exact scheduler/data execution contract."""
    metadata = dict(checkpoint_report.get("metadata") or {})
    saved = metadata.get("config")
    if not isinstance(saved, dict) or not saved:
        raise RuntimeError(
            "resume checkpoint 缺少完整 config；旧 checkpoint 不兼容，"
            "只能作为明确允许的 --initialize-from 源"
        )
    current = dict(current_config)
    if saved != current:
        keys = sorted(set(saved) | set(current))
        changed = {
            key: {"checkpoint": saved.get(key), "current": current.get(key)}
            for key in keys if saved.get(key) != current.get(key)
        }
        raise RuntimeError(
            "resume 必须保持同一 run 的完整 config；"
            f"changed={changed}。跨协议或超参数变更请新建 run"
        )
    return {
        "protocol": "qpsalm_segdesc_resume_config_binding_v1",
        "config": current,
        "matched": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def description_protocol_assets_spec() -> dict[str, Any]:
    """Return the current byte-level ontology/schema contract."""
    assets = {}
    for reference in DESCRIPTION_PROTOCOL_ASSETS:
        path = resolve_project_path(reference)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"segdesc checkpoint 缺少 description protocol asset: {reference}"
            )
        assets[reference] = {
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return {
        "protocol": "qpsalm_description_protocol_assets_v1",
        "assets": assets,
    }


def _checkpoint_metadata_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the exact non-tensor payload returned by the normal loader."""
    names = (
        "segmentation_migration",
        "description_protocol_assets",
        "description_architecture_spec",
        "segmentation_architecture_spec",
        "metadata",
    )
    report: dict[str, Any] = {}
    for name in names:
        value = payload.get(name)
        if not isinstance(value, dict):
            raise RuntimeError(
                f"segdesc checkpoint non-tensor metadata 缺少 object: {name}"
            )
        report[name] = dict(value)
    try:
        # Formal gates publish this exact projection as JSON; reject a checkpoint
        # that could only be represented through Python's NaN/Infinity extension.
        json.dumps(report, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "segdesc checkpoint non-tensor metadata 必须是 finite、标准 JSON-compatible"
        ) from exc
    best_score = report["metadata"].get("best_score")
    if best_score is not None and (
        isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise RuntimeError(
            "segdesc checkpoint metadata.best_score 必须是 finite number 或 null"
        )
    checkpoint_role = report["metadata"].get("checkpoint_role")
    if (
        checkpoint_role is not None
        and checkpoint_role not in DESCRIPTION_CHECKPOINT_ROLES
    ):
        raise RuntimeError(
            "segdesc checkpoint metadata.checkpoint_role 非法: "
            f"{checkpoint_role!r}"
        )
    return report


def inspect_segdesc_checkpoint(path: str | Path) -> dict[str, Any]:
    """Replay formal provenance from a checkpoint without constructing the model.

    Tensor storages are memory-mapped so M4/M6/M7 gates can verify the saved
    stage, lineage and protocol assets without duplicating a full model load.
    """
    resolved = resolve_project_path(path) or Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"segdesc checkpoint 不存在: {resolved}")
    try:
        payload = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"segdesc checkpoint provenance 无法读取当前 zip checkpoint: {resolved}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "formal provenance 只接受当前 segdesc checkpoint format: "
            f"observed={getattr(payload, 'get', lambda _key: None)('format')!r}"
        )
    if payload.get("description_sequence_protocol") != DESCRIPTION_SEQUENCE_PROTOCOL:
        raise RuntimeError("formal provenance description sequence protocol 不一致")
    if payload.get("description_protocol_assets") != description_protocol_assets_spec():
        raise RuntimeError("formal provenance ontology/schema protocol assets 不一致")
    checkpoint_metadata = _checkpoint_metadata_report(payload)
    if tuple(sorted(payload.get("adapter_names") or [])) != tuple(
        sorted(("default", DESCRIPTION_ADAPTER_NAME))
    ):
        raise RuntimeError("formal provenance adapter names 不一致")
    state = _checkpoint_state(payload)
    if not state:
        raise RuntimeError("formal provenance checkpoint model_state 不能为空")
    for name in ("description_architecture_spec", "segmentation_architecture_spec"):
        if not isinstance(payload.get(name), dict):
            raise RuntimeError(f"formal provenance checkpoint 缺少 {name}")
    if not isinstance(payload.get("segmentation_migration"), dict):
        raise RuntimeError("formal provenance checkpoint 缺少 segmentation_migration")
    if not isinstance(payload.get("metadata"), dict):
        raise RuntimeError("formal provenance checkpoint metadata 必须是 object")
    segmentation_migration_lineage = validate_segmentation_migration_lineage(
        payload.get("segmentation_migration"),
        {"segmentation_migration": payload.get("segmentation_migration")},
    )
    if (
        (payload.get("metadata") or {}).get("segmentation_migration_lineage")
        != segmentation_migration_lineage
    ):
        raise RuntimeError(
            "formal provenance checkpoint segmentation migration lineage 缺失或漂移"
        )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError(f"formal provenance checkpoint step 非法: {step!r}")
    state_keys = sorted(state)
    return {
        "protocol": SEGDESC_CHECKPOINT_PROVENANCE_PROTOCOL,
        "checkpoint": str(resolved.resolve(strict=False)),
        "checkpoint_sha256": _sha256_file(resolved),
        "checkpoint_step": int(step),
        "checkpoint_metadata": checkpoint_metadata,
        "model_state_keys": len(state_keys),
        "model_state_inventory_sha256": _canonical_sha256(state_keys),
        "segmentation_migration_lineage": segmentation_migration_lineage,
    }


def read_segdesc_checkpoint_step(path: str | Path) -> int:
    """Read only the current-format checkpoint step for same-run ordering.

    ``mmap=True`` keeps sibling best/last comparison from materializing a second
    copy of the model while a resume checkpoint is already resident in memory.
    """

    resolved = resolve_project_path(path) or Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"segdesc checkpoint 不存在: {resolved}")
    try:
        payload = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise RuntimeError(f"无法读取 segdesc checkpoint step: {resolved}") from exc
    if not isinstance(payload, dict) or payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "resume sibling 只接受当前 segdesc checkpoint format: "
            f"{getattr(payload, 'get', lambda _key: None)('format')!r}"
        )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError(f"segdesc checkpoint step 非法: {step!r}")
    return int(step)


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_segmentation_checkpoint(
    checkpoint_path: str | Path,
    segmentation: MultiSourceQwenPSALMSeg,
) -> dict[str, Any]:
    """Load qpsalm_sane_qmef_pmrd_v5 after an explicit state-key whitelist audit."""
    path = resolve_project_path(checkpoint_path) or Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"只允许迁移 {CHECKPOINT_FORMAT}，当前为 {payload.get('format')!r}"
        )
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("segmentation checkpoint 缺少 model_state")
    illegal = [key for key in state if not key.startswith(SEGMENTATION_STATE_PREFIXES)]
    if illegal:
        raise RuntimeError(f"segmentation checkpoint 包含白名单外参数: {illegal[:8]}")
    step = load_checkpoint(path, segmentation)
    return {
        "source_path": str(path.resolve(strict=False)),
        "source_sha256": _sha256_file(path),
        "source_format": CHECKPOINT_FORMAT,
        "source_step": step,
        "allowed_prefixes": list(SEGMENTATION_STATE_PREFIXES),
    }


def validate_segmentation_migration_lineage(
    current_migration: Any,
    source_checkpoint_report: Any,
) -> dict[str, Any]:
    """Prove every SegDesc stage descends from the same segmentation bytes."""
    current = dict(current_migration or {}) if isinstance(current_migration, dict) else {}
    source_report = (
        dict(source_checkpoint_report or {})
        if isinstance(source_checkpoint_report, dict) else {}
    )
    source = dict(source_report.get("segmentation_migration") or {})
    if not current or not source:
        raise RuntimeError("segdesc checkpoint 缺少 segmentation migration lineage")

    fields = ("source_sha256", "source_format", "source_step", "allowed_prefixes")
    current_identity = {name: current.get(name) for name in fields}
    source_identity = {name: source.get(name) for name in fields}
    for label, identity in (
        ("current", current_identity), ("source checkpoint", source_identity),
    ):
        sha256 = str(identity.get("source_sha256") or "")
        if len(sha256) != 64:
            raise RuntimeError(f"{label} segmentation migration 缺少 SHA-256")
    if current_identity != source_identity:
        raise RuntimeError(
            "SegDesc stage 使用了不同的原始 segmentation checkpoint: "
            f"current={current_identity} source={source_identity}"
        )

    # 正式运行保存绝对 source_path；若仍可访问，就必须逐字节复验。
    for label, migration in (("current", current), ("source checkpoint", source)):
        path_ref = migration.get("source_path")
        if not path_ref:
            raise RuntimeError(f"{label} segmentation migration 缺少 source_path")
        path = resolve_project_path(path_ref) or Path(str(path_ref))
        if (
            not path.is_file()
            or _sha256_file(path) != current_identity["source_sha256"]
        ):
            raise RuntimeError(f"{label} segmentation migration source bytes 已漂移")
    return {
        "protocol": SEGMENTATION_MIGRATION_LINEAGE_PROTOCOL,
        "segmentation_source_identity": current_identity,
        "source_bytes_revalidated": True,
        "passed": True,
    }


def _required_state(model: SegmentationGroundedDescriptionModel) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if ".lora_" in key or not key.startswith(FROZEN_QWEN_PREFIX)
    }


def _adapter_names(model: SegmentationGroundedDescriptionModel) -> tuple[str, ...]:
    configs = getattr(model.controller.model, "peft_config", None)
    if not isinstance(configs, dict):
        raise RuntimeError("统一 segdesc checkpoint 需要 PEFT model")
    names = tuple(sorted(str(value) for value in configs))
    expected = ("default", DESCRIPTION_ADAPTER_NAME)
    if names != tuple(sorted(expected)):
        raise RuntimeError(f"Adapter 名称必须严格为 {expected}，当前为 {names}")
    return names


def _checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Validate the checkpoint's declared tensor inventory before any load."""
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("segdesc checkpoint 缺少 model_state")
    declared = payload.get("required_state_keys")
    if not isinstance(declared, list) or declared != sorted(state):
        raise RuntimeError("segdesc checkpoint required_state_keys 与 model_state 不一致")
    if payload.get("frozen_qwen_prefix") != FROZEN_QWEN_PREFIX:
        raise RuntimeError("segdesc checkpoint frozen Qwen prefix 不一致")
    return state


def _restore_resume_state(
    payload: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
) -> None:
    """Resume is strict; cross-stage weight initialization uses another API."""
    requested = (
        ("optimizer_state", optimizer, optimizer is not None),
        ("scheduler_state", scheduler, scheduler is not None),
        (
            "grad_scaler_state",
            scaler,
            scaler is not None and scaler.is_enabled(),
        ),
    )
    for key, target, required in requested:
        if not required:
            continue
        if key not in payload:
            raise RuntimeError(f"resume checkpoint 缺少 {key}")
        target.load_state_dict(payload[key])
    if any(required for _key, _target, required in requested):
        restore_training_rng_state(payload.get("training_rng_state"))


def capture_training_rng_state() -> dict[str, Any]:
    """Snapshot every process-level RNG used by resumable SegDesc training."""
    cuda_available = bool(torch.cuda.is_available())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else []
    return {
        "protocol": TRAINING_RNG_STATE_PROTOCOL,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "cuda_available": cuda_available,
        "cuda_device_count": len(cuda_states),
    }


def restore_training_rng_state(value: Any) -> None:
    """Strictly restore a current RNG snapshot; old resume checkpoints are rejected."""
    if not isinstance(value, dict) or value.get("protocol") != TRAINING_RNG_STATE_PROTOCOL:
        raise RuntimeError(
            "resume checkpoint 缺少当前 training RNG state；旧实验 checkpoint "
            "不能用于同一 run 精确续训"
        )
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if any(name not in value for name in required):
        raise RuntimeError("resume checkpoint training RNG state 字段不完整")
    saved_cuda = bool(value.get("cuda_available"))
    current_cuda = bool(torch.cuda.is_available())
    cuda_states = list(value.get("torch_cuda") or [])
    if saved_cuda != current_cuda:
        raise RuntimeError(
            "resume checkpoint CUDA RNG 环境不一致: "
            f"checkpoint={saved_cuda} current={current_cuda}"
        )
    if saved_cuda and (
        int(value.get("cuda_device_count", -1)) != torch.cuda.device_count()
        or len(cuda_states) != torch.cuda.device_count()
    ):
        raise RuntimeError("resume checkpoint CUDA device/RNG state 数量不一致")
    try:
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch_cpu"])
        if saved_cuda:
            torch.cuda.set_rng_state_all(cuda_states)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("resume checkpoint training RNG state 无法恢复") from exc


def _description_architecture_spec(
    model: SegmentationGroundedDescriptionModel,
) -> dict[str, Any]:
    bank = getattr(getattr(model, "description_backbone", None), "bank", None)
    manifest = dict(getattr(bank, "manifest", {}) or {})
    artifact_binding = (
        bank.artifact_binding()
        if bank is not None and callable(getattr(bank, "artifact_binding", None))
        else None
    )
    return {
        "region_encoder": model.region_encoder_name,
        "mgrr_protocol": (
            MGRR_PROTOCOL if isinstance(model.mgrr, torch.nn.Module)
            and hasattr(model.mgrr, "roi_queries") else None
        ),
        "mgrr_max_components": getattr(model.mgrr, "max_components", None),
        "mgrr_component_coverage": getattr(model.mgrr, "component_coverage", None),
        "mgrr_roi_grid_sizes": (
            [list(value) for value in MGRR_ROI_GRID_SIZES]
            if hasattr(model.mgrr, "roi_queries") else None
        ),
        "mgrr_roi_query_count": (
            int(model.mgrr.roi_queries.shape[0])
            if hasattr(model.mgrr, "roi_queries") else None
        ),
        "decoder_dim": int(model.segmentation.config.decoder_dim),
        "description_sequence_protocol": DESCRIPTION_SEQUENCE_PROTOCOL,
        "description_cache_protocol": manifest.get("format"),
        "description_cache_renderer_version": manifest.get("renderer_version"),
        "description_cache_model_revision": manifest.get("model_revision"),
        "description_cache_processor_revision": manifest.get("processor_revision"),
        "description_cache_spatial_channels": manifest.get("spatial_channels"),
        "description_cache_layers": manifest.get("layers"),
        "description_cache_artifact_binding": artifact_binding,
    }


def _segmentation_architecture_spec(
    model: SegmentationGroundedDescriptionModel,
) -> dict[str, Any]:
    config = model.segmentation.config
    return {
        name: getattr(config, name, None)
        for name in SEGMENTATION_ARCHITECTURE_FIELDS
    }


def save_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    step: int,
    segmentation_migration: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    state = _required_state(model)
    payload: dict[str, Any] = {
        "format": SEGDESC_CHECKPOINT_FORMAT,
        "step": int(step),
        "model_state": state,
        "required_state_keys": sorted(state),
        "frozen_qwen_prefix": FROZEN_QWEN_PREFIX,
        "adapter_names": list(_adapter_names(model)),
        "description_sequence_protocol": DESCRIPTION_SEQUENCE_PROTOCOL,
        "description_protocol_assets": description_protocol_assets_spec(),
        "description_architecture_spec": _description_architecture_spec(model),
        "segmentation_migration": dict(segmentation_migration),
        "segmentation_architecture_spec": _segmentation_architecture_spec(model),
        "training_rng_state": capture_training_rng_state(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["grad_scaler_state"] = scaler.state_dict()
    _checkpoint_metadata_report(payload)
    _atomic_save(payload, Path(path))


def load_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    expected_stage: str | None = None,
) -> tuple[int, dict[str, Any]]:
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 segdesc checkpoint={payload.get('format')!r}; expected={SEGDESC_CHECKPOINT_FORMAT}"
        )
    if payload.get("description_sequence_protocol") != DESCRIPTION_SEQUENCE_PROTOCOL:
        raise RuntimeError("segdesc description sequence protocol 不一致")
    if payload.get("description_protocol_assets") != description_protocol_assets_spec():
        raise RuntimeError("segdesc ontology/schema protocol assets 不一致")
    checkpoint_metadata = _checkpoint_metadata_report(payload)
    source_stage = str((payload.get("metadata") or {}).get("stage") or "")
    if expected_stage is not None and source_stage != expected_stage:
        raise RuntimeError(
            f"resume checkpoint stage 不一致: expected={expected_stage!r} "
            f"observed={source_stage!r}"
        )
    if payload.get("description_architecture_spec") != _description_architecture_spec(model):
        raise RuntimeError(
            "segdesc description architecture 不一致；resume 必须保持 region encoder 完全相同"
        )
    if payload.get("segmentation_architecture_spec") != _segmentation_architecture_spec(model):
        raise RuntimeError(
            "segdesc segmentation architecture 不一致；Qwen/SANE/QMEF/PMRD 参数语义不可迁移"
        )
    if tuple(sorted(payload.get("adapter_names") or [])) != _adapter_names(model):
        raise RuntimeError("segdesc adapter names 不一致")
    expected_state = _required_state(model)
    state = _checkpoint_state(payload)
    observed_keys = set(state)
    if observed_keys != set(expected_state):
        missing = sorted(set(expected_state) - observed_keys)
        unexpected = sorted(observed_keys - set(expected_state))
        raise RuntimeError(
            f"segdesc checkpoint state 不一致: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    illegal_missing = [
        key for key in incompatible.missing_keys
        if not (key.startswith(FROZEN_QWEN_PREFIX) and ".lora_" not in key)
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"segdesc checkpoint 加载不完整: missing={illegal_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    _restore_resume_state(
        payload, optimizer=optimizer, scheduler=scheduler, scaler=scaler
    )
    return int(payload.get("step", 0)), checkpoint_metadata


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash one dense tensor without depending on torch serialization metadata."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(
        list(tensor.shape), separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_segdesc_checkpoint_reload(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    expected_stage: str,
) -> tuple[int, dict[str, Any]]:
    """Prove that a strict reload restores a deliberately corrupted adapter tensor.

    Merely loading a just-saved checkpoint into the unchanged live model cannot
    demonstrate restoration.  This probe first verifies a description-adapter
    sentinel against the checkpoint, mutates it, then exercises the normal strict
    model and resume-state loader and verifies exact byte restoration.
    """
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "reload probe 仅支持当前 segdesc checkpoint format: "
            f"{payload.get('format')!r}"
        )
    _checkpoint_metadata_report(payload)
    state = _checkpoint_state(payload)
    named_parameters = dict(model.named_parameters())
    candidates = [
        name
        for name in sorted(state)
        if (
            f".{DESCRIPTION_ADAPTER_NAME}." in name
            and "lora_" in name
            and name in named_parameters
            and named_parameters[name].is_floating_point()
            and named_parameters[name].numel() > 0
        )
    ]
    if not candidates:
        raise RuntimeError("reload probe 未找到可变的 desc_adapter LoRA 哨兵参数")
    sentinel_name = candidates[0]
    sentinel = named_parameters[sentinel_name]
    checkpoint_sha256 = _sha256_file(resolved)
    expected_sha256 = _tensor_sha256(state[sentinel_name])
    before_sha256 = _tensor_sha256(sentinel)
    if before_sha256 != expected_sha256:
        raise RuntimeError(
            "reload probe 开始前 live model 已与刚保存 checkpoint 不一致: "
            f"parameter={sentinel_name}"
        )
    with torch.no_grad():
        sentinel.reshape(-1)[0].add_(1.0)
    corrupted_sha256 = _tensor_sha256(sentinel)
    if corrupted_sha256 == expected_sha256:
        raise RuntimeError("reload probe 未能改变 desc_adapter 哨兵参数")
    step, metadata = load_segdesc_checkpoint(
        resolved,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_stage=expected_stage,
    )
    restored_sha256 = _tensor_sha256(named_parameters[sentinel_name])
    passed = restored_sha256 == expected_sha256
    if not passed:
        raise RuntimeError(
            "strict segdesc checkpoint reload 未恢复 desc_adapter 哨兵参数"
        )
    return step, {
        "protocol": "qpsalm_segdesc_strict_reload_probe_v1",
        "passed": True,
        "checkpoint": str(resolved.resolve(strict=False)),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(step),
        "expected_stage": str(expected_stage),
        "sentinel_parameter": sentinel_name,
        "before_sha256": before_sha256,
        "corrupted_sha256": corrupted_sha256,
        "restored_sha256": restored_sha256,
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "grad_scaler_state_requested": bool(
            scaler is not None and scaler.is_enabled()
        ),
        "segmentation_migration": dict(
            metadata.get("segmentation_migration") or {}
        ),
    }


def initialize_segdesc_checkpoint(
    path: str | Path,
    model: SegmentationGroundedDescriptionModel,
    *,
    target_stage: str | None = None,
    expected_seed: int | None = None,
    allow_same_stage_curriculum: bool = False,
    require_run_completion: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Load a new D-stage, allowing only an explicit region-encoder replacement."""
    resolved = resolve_project_path(path) or Path(path)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != SEGDESC_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"不支持 segdesc checkpoint={payload.get('format')!r}; expected={SEGDESC_CHECKPOINT_FORMAT}"
        )
    if payload.get("description_sequence_protocol") != DESCRIPTION_SEQUENCE_PROTOCOL:
        raise RuntimeError("segdesc description sequence protocol 不一致")
    if payload.get("description_protocol_assets") != description_protocol_assets_spec():
        raise RuntimeError("segdesc initialize ontology/schema protocol assets 不一致")
    checkpoint_metadata = _checkpoint_metadata_report(payload)
    source_metadata = dict(payload.get("metadata") or {})
    source_stage = str(source_metadata.get("stage") or "")
    source_config = dict(source_metadata.get("config") or {})
    source_seed = source_config.get("seed")
    source_run_completion: dict[str, Any] | None = None
    if expected_seed is not None and (
        source_seed is None or int(source_seed) != int(expected_seed)
    ):
        raise RuntimeError(
            "跨 stage initialize seed lineage 不一致: "
            f"expected={int(expected_seed)} observed={source_seed!r}"
        )
    if target_stage is not None:
        expected_source = DESCRIPTION_STAGE_PREDECESSOR.get(target_stage)
        if expected_source is None:
            raise RuntimeError(
                f"stage={target_stage!r} 必须从 segmentation checkpoint 新建，"
                "不能使用 segdesc --initialize-from"
            )
        same_stage_curriculum = bool(
            allow_same_stage_curriculum
            and target_stage == "predicted_mask"
            and source_stage == "predicted_mask"
        )
        if source_stage != expected_source and not same_stage_curriculum:
            raise RuntimeError(
                "跨 stage initialize 顺序非法: "
                f"target={target_stage!r} expected_source={expected_source!r} "
                f"observed_source={source_stage!r}"
            )
        expected_role = DESCRIPTION_STAGE_CHECKPOINT_ROLE.get(source_stage)
        if source_metadata.get("checkpoint_role") != expected_role:
            # 角色来自 checkpoint 本体而非文件名；随后再绑定同目录成功 completion。
            # D3a 无人工 validation，只有它使用 terminal_last；其他阶段使用 best。
            raise RuntimeError(
                "跨 stage initialize checkpoint role 非法: "
                f"source_stage={source_stage!r} expected={expected_role!r} "
                f"observed={source_metadata.get('checkpoint_role')!r}"
            )
        from .run_artifacts import (
            DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
            validate_checkpoint_run_completion,
        )
        try:
            source_run_completion = validate_checkpoint_run_completion(
                resolved,
                expected_completion_protocol=(
                    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
                ),
                expected_stage=source_stage,
                expected_role=expected_role,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "跨 stage initialize source run 尚未成功完成或 completion 已漂移"
            ) from exc
    if require_run_completion and source_run_completion is None:
        expected_role = DESCRIPTION_STAGE_CHECKPOINT_ROLE.get(source_stage)
        if expected_role is None:
            raise RuntimeError(
                f"initialize source stage 没有完成角色定义: {source_stage!r}"
            )
        from .run_artifacts import (
            DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
            validate_checkpoint_run_completion,
        )
        try:
            source_run_completion = validate_checkpoint_run_completion(
                resolved,
                expected_completion_protocol=(
                    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
                ),
                expected_stage=source_stage,
                expected_role=expected_role,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "initialize source run 尚未成功完成或 completion 已漂移"
            ) from exc
    if tuple(sorted(payload.get("adapter_names") or [])) != _adapter_names(model):
        raise RuntimeError("segdesc adapter names 不一致")
    if payload.get("segmentation_architecture_spec") != _segmentation_architecture_spec(model):
        raise RuntimeError("segdesc initialize segmentation architecture 不一致")
    source_spec = dict(payload.get("description_architecture_spec") or {})
    target_spec = _description_architecture_spec(model)
    non_region_source = {
        key: value for key, value in source_spec.items()
        if key not in REGION_ARCHITECTURE_FIELDS
    }
    non_region_target = {
        key: value for key, value in target_spec.items()
        if key not in REGION_ARCHITECTURE_FIELDS
    }
    if non_region_source != non_region_target:
        raise RuntimeError(
            "segdesc initialize architecture 不一致: "
            f"source={non_region_source} target={non_region_target}"
        )
    source_state = _checkpoint_state(payload)
    target_state = _required_state(model)
    region_changed = source_spec.get("region_encoder") != target_spec["region_encoder"]
    if not region_changed:
        source_region_spec = {
            key: source_spec.get(key) for key in REGION_ARCHITECTURE_FIELDS
        }
        target_region_spec = {
            key: target_spec.get(key) for key in REGION_ARCHITECTURE_FIELDS
        }
        if source_region_spec != target_region_spec:
            raise RuntimeError(
                "segdesc initialize MGRR protocol 不一致；同 region encoder 不允许静默迁移: "
                f"source={source_region_spec} target={target_region_spec}"
            )
    loaded = {}
    skipped_source = []
    for key, value in source_state.items():
        if region_changed and key.startswith("mgrr."):
            skipped_source.append(key)
            continue
        target = target_state.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            raise RuntimeError(f"segdesc initialize 非 region 参数不匹配: {key}")
        loaded[key] = value
    expected_loaded = {
        key for key in target_state
        if not (region_changed and key.startswith("mgrr."))
    }
    if set(loaded) != expected_loaded:
        missing = sorted(expected_loaded - set(loaded))
        unexpected = sorted(set(loaded) - expected_loaded)
        raise RuntimeError(
            f"segdesc initialize state 不完整: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(loaded, strict=False)
    allowed_missing = {
        key for key in target_state if region_changed and key.startswith("mgrr.")
    } | {
        key for key in model.state_dict()
        if key.startswith(FROZEN_QWEN_PREFIX) and ".lora_" not in key
    }
    illegal_missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"segdesc initialize 加载不完整: missing={illegal_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return int(payload.get("step", 0)), {
        **checkpoint_metadata,
        "initialization": {
            "source_checkpoint": str(resolved),
            "source_checkpoint_sha256": _sha256_file(resolved),
            "source_stage": source_stage or None,
            "target_stage": target_stage,
            "source_seed": int(source_seed) if source_seed is not None else None,
            "target_seed": int(expected_seed) if expected_seed is not None else None,
            "seed_match": expected_seed is None or int(source_seed) == int(expected_seed),
            "same_stage_curriculum": bool(
                allow_same_stage_curriculum
                and target_stage == "predicted_mask"
                and source_stage == "predicted_mask"
            ),
            "source_region_encoder": source_spec.get("region_encoder"),
            "target_region_encoder": target_spec["region_encoder"],
            "region_encoder_reinitialized": region_changed,
            "skipped_source_region_keys": sorted(skipped_source),
            "initialized_target_region_keys": sorted(allowed_missing & set(target_state)),
            "source_run_completion": source_run_completion,
        },
    }
