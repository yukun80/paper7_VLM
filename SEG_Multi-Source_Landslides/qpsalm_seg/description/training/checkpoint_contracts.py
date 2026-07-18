"""Strict SegDesc checkpoint metadata, lineage and provenance contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
    serialized_segdesc_config_without,
)
from ..modeling.model import DESCRIPTION_ADAPTER_NAME
from ..protocols.stages import DESCRIPTION_STAGES, get_stage_spec
from ..protocols.versions import (
    CHECKPOINT_RUN_COMPLETION_PROTOCOL,
    DESCRIPTION_PROTOCOL_ASSETS,
    DESCRIPTION_SEQUENCE_PROTOCOL,
    D_MINUS_ONE_ACCEPTANCE_PROTOCOL,
    STRUCTURED_GENERATION_PROTOCOL,
)
from ..protocols.io import (
    canonical_sha256 as _canonical_sha256,
    sha256_file as _sha256_file,
)


SEGDESC_CHECKPOINT_FORMAT = "qpsalm_segdesc_v1"
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
    stage: get_stage_spec(stage).initialize_from_stage
    for stage in DESCRIPTION_STAGES
    if get_stage_spec(stage).initialize_from_stage is not None
}
DESCRIPTION_STAGE_LINEAGE_PROTOCOL = (
    "qpsalm_description_stage_lineage_v3_run_completion_bound"
)
DESCRIPTION_CHECKPOINT_ROLES = {"validation_best", "terminal_last"}
DESCRIPTION_STAGE_CHECKPOINT_ROLE = {
    stage: get_stage_spec(stage).checkpoint_role
    for stage in DESCRIPTION_STAGES
}
DESCRIPTION_VARIANT_CONFIG_FIELDS = {"region_encoder", "output_dir"}
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
    source_config = require_serialized_segdesc_config(
        source_metadata.get("config"),
        label="description lineage source config",
    )
    d_minus_one_acceptance = source_metadata.get("d_minus_one_acceptance")
    if (
        not isinstance(d_minus_one_acceptance, dict)
        or d_minus_one_acceptance.get("protocol")
        != D_MINUS_ONE_ACCEPTANCE_PROTOCOL
        or d_minus_one_acceptance.get("passed") is not True
    ):
        raise RuntimeError("source checkpoint 缺少已通过的 D-1 acceptance")
    controlled_config = serialized_segdesc_config_without(
        source_config,
        DESCRIPTION_VARIANT_CONFIG_FIELDS,
        label="description lineage source config",
    )
    entry = {
        "stage": source_stage,
        "checkpoint_role": source_metadata.get("checkpoint_role"),
        "checkpoint": source_checkpoint,
        "checkpoint_sha256": source_sha256,
        "seed": serialized_segdesc_config_value(source_config, "seed"),
        "region_encoder": serialized_segdesc_config_value(
            source_config, "region_encoder"
        ),
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
    try:
        saved = require_serialized_segdesc_config(
            metadata.get("config"), label="resume checkpoint config"
        )
        current = require_serialized_segdesc_config(
            current_config, label="current run config"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "resume checkpoint 缺少完整 composed config v2；旧 checkpoint "
            "不兼容，只能按当前迁移协议作为显式 --initialize-from 源"
        ) from exc
    if saved != current:
        changed: dict[str, dict[str, Any]] = {}
        if saved.get("protocol") != current.get("protocol"):
            changed["protocol"] = {
                "checkpoint": saved.get("protocol"),
                "current": current.get("protocol"),
            }
        for section in ("model", "data", "training", "evaluation", "joint"):
            saved_section = dict(saved[section])
            current_section = dict(current[section])
            for key in sorted(set(saved_section) | set(current_section)):
                if saved_section.get(key) != current_section.get(key):
                    changed[f"{section}.{key}"] = {
                        "checkpoint": saved_section.get(key),
                        "current": current_section.get(key),
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


def checkpoint_metadata_report(payload: dict[str, Any]) -> dict[str, Any]:
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
        # StageSpec contains tuple fields: normalize them through the exact JSON
        # boundary now so initial publication and later replay cannot disagree as
        # tuple versus list while representing identical artifact metadata.
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
        canonical_report = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "segdesc checkpoint non-tensor metadata 必须是 finite、标准 JSON-compatible"
        ) from exc
    if not isinstance(canonical_report, dict):
        raise RuntimeError("segdesc checkpoint non-tensor metadata canonical root 非法")
    report = canonical_report
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
    if checkpoint_role not in DESCRIPTION_CHECKPOINT_ROLES:
        raise RuntimeError(
            "segdesc checkpoint metadata.checkpoint_role 非法: "
            f"{checkpoint_role!r}"
        )
    stage = str(report["metadata"].get("stage") or "")
    if stage not in {*DESCRIPTION_STAGES, "joint"}:
        raise RuntimeError(
            f"segdesc checkpoint metadata.stage 非法: {stage!r}"
        )
    config = require_serialized_segdesc_config(
        report["metadata"].get("config"),
        label="segdesc checkpoint metadata.config",
    )
    if stage != "joint" and serialized_segdesc_config_value(
        config, "stage"
    ) != stage:
        raise RuntimeError(
            "segdesc checkpoint metadata.stage 与 composed config.training.stage 不一致"
        )
    migration_lineage = validate_segmentation_migration_lineage(
        report["segmentation_migration"],
        {"segmentation_migration": report["segmentation_migration"]},
    )
    if (
        report["metadata"].get("segmentation_migration_lineage")
        != migration_lineage
    ):
        raise RuntimeError(
            "segdesc checkpoint metadata.segmentation_migration_lineage "
            "缺失或与 segmentation_migration 不一致"
        )
    report["metadata"]["config"] = config
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
    checkpoint_metadata = checkpoint_metadata_report(payload)
    if tuple(sorted(payload.get("adapter_names") or [])) != tuple(
        sorted(("default", DESCRIPTION_ADAPTER_NAME))
    ):
        raise RuntimeError("formal provenance adapter names 不一致")
    state = checkpoint_state(payload)
    if not state:
        raise RuntimeError("formal provenance checkpoint model_state 不能为空")
    for name in ("description_architecture_spec", "segmentation_architecture_spec"):
        if not isinstance(payload.get(name), dict):
            raise RuntimeError(f"formal provenance checkpoint 缺少 {name}")
    if (
        payload["description_architecture_spec"].get(
            "description_sequence_protocol"
        )
        != DESCRIPTION_SEQUENCE_PROTOCOL
    ):
        raise RuntimeError(
            "formal provenance architecture description sequence protocol 不一致"
        )
    if (
        payload["description_architecture_spec"].get(
            "structured_generation_protocol"
        )
        != STRUCTURED_GENERATION_PROTOCOL
    ):
        raise RuntimeError(
            "formal provenance structured generation protocol 不一致"
        )
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



def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
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
