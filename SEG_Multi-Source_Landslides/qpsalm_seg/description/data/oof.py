#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic parent-level folds for out-of-fold predicted masks."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path, to_project_ref

from .expert_contracts import require_frozen_expert_bridge, validate_expert_rows
from ..protocols.io import (
    atomic_write_json as _atomic_json,
    atomic_write_jsonl as _atomic_jsonl,
    canonical_sha256 as _canonical_sha256,
    sha256_file as _sha256,
    strict_json_loads,
)


OOF_FOLD_FORMAT = "qpsalm_segmentation_oof_folds_v3_source_partition_replayed"
OOF_PARTITION_REPLAY_PROTOCOL = (
    "qpsalm_segmentation_oof_partition_replay_v1_source_rows_exact"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        strict_json_loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_rank(seed: int, namespace: str, parent: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{parent}".encode()).hexdigest()


def _parent_id(row: dict[str, Any]) -> str:
    parent = str(row.get("parent_sample_id") or row.get("sample_id") or "")
    if not parent:
        raise ValueError("OOF segmentation row 缺少 parent_sample_id/sample_id")
    return parent


def _validate_segmentation_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("segmentation OOF source index 不能为空")
    non_train = [
        str(row.get("sample_id") or "unknown")
        for row in rows
        if str(row.get("split")) != "train"
    ]
    if non_train:
        raise ValueError(
            "OOF fold source 只能包含 split=train 的 segmentation rows: "
            f"count={len(non_train)} examples={non_train[:8]}"
        )
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("OOF segmentation source sample_id 缺失或重复")
    for row in rows:
        _parent_id(row)


def _expert_parent_metadata(
    bridge_rows: list[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    validate_expert_rows(bridge_rows, stage="bridge_expert", split="train")
    non_train = [
        str(row.get("bridge_record_id") or "unknown")
        for row in bridge_rows
        if str(row.get("split")) != "train"
    ]
    if non_train:
        raise ValueError(
            "OOF Bridge source 只能包含 split=train 的 expert rows: "
            f"count={len(non_train)} examples={non_train[:8]}"
        )
    parent_metadata: dict[str, tuple[str, str]] = {}
    for row in bridge_rows:
        if row.get("region_source") != "gt_global_mask":
            continue
        if not isinstance(row.get("expert_target"), dict):
            raise ValueError(
                "D4 OOF folds 只接受已冻结 expert_train 中带 expert_target 的 gt_global_mask"
            )
        parent = str(row.get("parent_sample_id") or "")
        if not parent:
            raise ValueError("Bridge gt_global_mask row 缺少 parent_sample_id")
        stratum = (
            str(row.get("dataset_name") or "unknown"),
            str(row.get("modality_family_combo") or "unknown"),
        )
        previous = parent_metadata.setdefault(parent, stratum)
        if previous != stratum:
            raise ValueError(f"Bridge parent metadata 不一致: {parent}")
    return parent_metadata


def _assign_parent_folds(
    parent_metadata: dict[str, tuple[str, str]], *, num_folds: int, seed: int,
) -> dict[str, str]:
    if len(parent_metadata) < num_folds:
        raise ValueError(
            f"可用 Bridge train parents={len(parent_metadata)} 少于 num_folds={num_folds}"
        )
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for parent, stratum in parent_metadata.items():
        strata[stratum].append(parent)
    parent_to_fold: dict[str, str] = {}
    for stratum, parents in sorted(strata.items()):
        namespace = "|".join(stratum)
        ordered = sorted(
            parents, key=lambda value: _stable_rank(seed, namespace, value)
        )
        offset = int(_stable_rank(seed, "offset", namespace)[:8], 16) % num_folds
        for index, parent in enumerate(ordered):
            parent_to_fold[parent] = str((index + offset) % num_folds)
    return parent_to_fold


def _partition_binding(
    segmentation_rows: list[dict[str, Any]],
    parent_to_fold: dict[str, str],
) -> dict[str, Any]:
    segmentation_parents = sorted({_parent_id(row) for row in segmentation_rows})
    return {
        "protocol": OOF_PARTITION_REPLAY_PROTOCOL,
        "num_source_records": len(segmentation_rows),
        "num_source_parents": len(segmentation_parents),
        "source_sample_ids_sha256": _canonical_sha256(
            [str(row["sample_id"]) for row in segmentation_rows]
        ),
        "source_parent_population_sha256": _canonical_sha256(segmentation_parents),
        "eligible_parent_assignment_sha256": _canonical_sha256(parent_to_fold),
    }


def build_oof_fold_indexes(
    *,
    segmentation_index: str | Path,
    bridge_index: str | Path,
    output_dir: str | Path,
    num_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Build parent-isolated train/holdout indexes and their immutable audit manifest."""
    if num_folds < 2:
        raise ValueError("num_folds 必须至少为 2")
    segmentation_path = resolve_project_path(segmentation_index) or Path(segmentation_index)
    bridge_path = resolve_project_path(bridge_index) or Path(bridge_index)
    output = resolve_project_path(output_dir) or Path(output_dir)
    bridge_root = bridge_path.parent.parent
    expected_bridge_index = bridge_root / "indexes/expert_train.jsonl"
    if bridge_path.resolve(strict=False) != expected_bridge_index.resolve(strict=False):
        raise ValueError(
            "D4 OOF bridge-index 必须是冻结 Bridge 的 indexes/expert_train.jsonl"
        )
    expert_gate_audit = require_frozen_expert_bridge(bridge_root)
    segmentation_rows = _read_jsonl(segmentation_path)
    bridge_rows = _read_jsonl(bridge_path)
    _validate_segmentation_rows(segmentation_rows)
    parent_metadata = _expert_parent_metadata(bridge_rows)

    segmentation_parents = {
        _parent_id(row)
        for row in segmentation_rows
    }
    missing = sorted(set(parent_metadata) - segmentation_parents)
    if missing:
        raise ValueError(f"segmentation index 缺少 Bridge parents: {missing[:8]}")

    parent_to_fold = _assign_parent_folds(
        parent_metadata, num_folds=num_folds, seed=seed
    )

    output.mkdir(parents=True, exist_ok=True)
    folds: dict[str, dict[str, Any]] = {}
    for fold_index in range(num_folds):
        fold = str(fold_index)
        holdout_parents = {
            parent for parent, assigned in parent_to_fold.items() if assigned == fold
        }
        if not holdout_parents:
            raise ValueError(f"fold={fold} 没有 holdout parent")
        train_rows = [
            row for row in segmentation_rows
            if _parent_id(row) not in holdout_parents
        ]
        holdout_rows = [
            row for row in segmentation_rows
            if _parent_id(row) in holdout_parents
        ]
        train_parents = {
            _parent_id(row) for row in train_rows
        }
        observed_holdout_parents = {
            _parent_id(row) for row in holdout_rows
        }
        if train_parents & holdout_parents or observed_holdout_parents != holdout_parents:
            raise RuntimeError(f"fold={fold} parent isolation 构建失败")
        train_path = output / f"fold_{fold}_train.jsonl"
        holdout_path = output / f"fold_{fold}_holdout.jsonl"
        _atomic_jsonl(train_path, train_rows)
        _atomic_jsonl(holdout_path, holdout_rows)
        folds[fold] = {
            "held_out_fold": fold,
            "num_holdout_parents": len(holdout_parents),
            "num_train_parents": len(train_parents),
            "num_train_records": len(train_rows),
            "num_holdout_records": len(holdout_rows),
            "train_index": to_project_ref(train_path),
            "train_index_sha256": _sha256(train_path),
            "holdout_index": to_project_ref(holdout_path),
            "holdout_index_sha256": _sha256(holdout_path),
        }
    manifest = {
        "protocol": OOF_FOLD_FORMAT,
        "seed": int(seed),
        "num_folds": int(num_folds),
        "source_segmentation_index": to_project_ref(segmentation_path),
        "source_segmentation_index_sha256": _sha256(segmentation_path),
        "source_bridge_index": to_project_ref(bridge_path),
        "source_bridge_index_sha256": _sha256(bridge_path),
        "expert_gate_audit": expert_gate_audit,
        "num_parents": len(parent_to_fold),
        "parent_to_fold": dict(sorted(parent_to_fold.items())),
        "partition_binding": _partition_binding(
            segmentation_rows, dict(sorted(parent_to_fold.items()))
        ),
        "folds": folds,
    }
    _atomic_json(output / "fold_manifest.json", manifest)
    return manifest


def load_oof_manifest(path_ref: str | Path) -> dict[str, Any]:
    """Load and replay the exact source-to-fold partition.

    Hashes alone only prove that a declared file did not change.  This loader also
    reconstructs the deterministic assignment from the current frozen expert
    source, then compares every train/holdout row with the source segmentation
    index.  A self-consistent but scientifically wrong edited manifest therefore
    fails closed.
    """
    path = resolve_project_path(path_ref) or Path(path_ref)
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != OOF_FOLD_FORMAT:
        raise ValueError(
            f"train predicted mask 需要 {OOF_FOLD_FORMAT}, observed={payload.get('protocol')!r}"
        )
    if not isinstance(payload.get("parent_to_fold"), dict) or not isinstance(payload.get("folds"), dict):
        raise ValueError("OOF manifest 缺少 parent_to_fold/folds")
    source_segmentation = resolve_project_path(
        str(payload.get("source_segmentation_index") or "")
    )
    source_bridge = resolve_project_path(str(payload.get("source_bridge_index") or ""))
    if source_segmentation is None or not source_segmentation.is_file():
        raise FileNotFoundError("OOF manifest source segmentation index 缺失")
    if source_bridge is None or not source_bridge.is_file():
        raise FileNotFoundError("OOF manifest source Bridge index 缺失")
    if _sha256(source_segmentation) != payload.get("source_segmentation_index_sha256"):
        raise ValueError("OOF source segmentation index 已变化")
    if _sha256(source_bridge) != payload.get("source_bridge_index_sha256"):
        raise ValueError("OOF source Bridge expert index 已变化")
    current_gate = require_frozen_expert_bridge(source_bridge.parent.parent)
    if payload.get("expert_gate_audit") != current_gate:
        raise ValueError("OOF manifest 与当前 frozen Bridge gate 不一致")
    segmentation_rows = _read_jsonl(source_segmentation)
    bridge_rows = _read_jsonl(source_bridge)
    _validate_segmentation_rows(segmentation_rows)
    try:
        num_folds = int(payload.get("num_folds", 0))
        seed = int(payload["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OOF manifest seed/num_folds 非法") from error
    if num_folds < 2:
        raise ValueError("OOF manifest num_folds 必须至少为 2")
    parent_metadata = _expert_parent_metadata(bridge_rows)
    expected_parent_to_fold = dict(sorted(_assign_parent_folds(
        parent_metadata, num_folds=num_folds, seed=seed
    ).items()))
    observed_parent_to_fold = {
        str(parent): str(fold)
        for parent, fold in payload["parent_to_fold"].items()
    }
    if observed_parent_to_fold != expected_parent_to_fold:
        raise ValueError("OOF manifest parent_to_fold 与源 Bridge 确定性重算不一致")
    if int(payload.get("num_parents", -1)) != len(expected_parent_to_fold):
        raise ValueError("OOF manifest num_parents 与确定性重算不一致")
    expected_partition_binding = _partition_binding(
        segmentation_rows, expected_parent_to_fold
    )
    if payload.get("partition_binding") != expected_partition_binding:
        raise ValueError("OOF manifest partition_binding 与当前源索引重算不一致")
    segmentation_parents = {_parent_id(row) for row in segmentation_rows}
    missing = sorted(set(expected_parent_to_fold) - segmentation_parents)
    if missing:
        raise ValueError(f"OOF source segmentation index 缺少 Bridge parents: {missing[:8]}")

    valid_folds = {str(index) for index in range(num_folds)}
    if set(payload["folds"]) != valid_folds:
        raise ValueError("OOF manifest fold inventory 不完整")
    if set(observed_parent_to_fold.values()) - valid_folds:
        raise ValueError("OOF manifest parent_to_fold 包含未知 fold")
    seen_paths: set[Path] = set()
    for fold, values in payload["folds"].items():
        if not isinstance(values, dict) or str(values.get("held_out_fold")) != fold:
            raise ValueError(f"OOF fold metadata 非法: {fold}")
        indexes: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
        for prefix in ("train", "holdout"):
            index_path = resolve_project_path(str(values.get(f"{prefix}_index") or ""))
            if index_path is None or not index_path.is_file():
                raise FileNotFoundError(f"OOF fold={fold} {prefix} index 缺失")
            resolved = index_path.resolve(strict=False)
            if resolved in seen_paths:
                raise ValueError(f"OOF fold indexes 复用了同一文件: {index_path}")
            seen_paths.add(resolved)
            if _sha256(index_path) != values.get(f"{prefix}_index_sha256"):
                raise ValueError(f"OOF fold={fold} {prefix} index 已变化")
            indexes[prefix] = (index_path, _read_jsonl(index_path))
        holdout_parents = {
            parent
            for parent, assigned in expected_parent_to_fold.items()
            if assigned == fold
        }
        expected_train_rows = [
            row for row in segmentation_rows if _parent_id(row) not in holdout_parents
        ]
        expected_holdout_rows = [
            row for row in segmentation_rows if _parent_id(row) in holdout_parents
        ]
        observed_train_rows = indexes["train"][1]
        observed_holdout_rows = indexes["holdout"][1]
        if observed_train_rows != expected_train_rows:
            raise ValueError(
                f"OOF fold={fold} train index 不是源 segmentation rows 的精确排除分区"
            )
        if observed_holdout_rows != expected_holdout_rows:
            raise ValueError(
                f"OOF fold={fold} holdout index 不是源 segmentation rows 的精确保留分区"
            )
        train_parents = {_parent_id(row) for row in observed_train_rows}
        observed_holdout_parents = {_parent_id(row) for row in observed_holdout_rows}
        expected_counts = {
            "num_holdout_parents": len(holdout_parents),
            "num_train_parents": len(train_parents),
            "num_train_records": len(expected_train_rows),
            "num_holdout_records": len(expected_holdout_rows),
        }
        for name, expected in expected_counts.items():
            if int(values.get(name, -1)) != expected:
                raise ValueError(
                    f"OOF fold={fold} {name} 与分区重算不一致: "
                    f"manifest={values.get(name)!r} expected={expected}"
                )
        if train_parents & holdout_parents:
            raise ValueError(f"OOF fold={fold} train index 泄漏 held-out parent")
        if observed_holdout_parents != holdout_parents:
            raise ValueError(f"OOF fold={fold} holdout parent 覆盖不完整")
    return payload


def oof_manifest_artifact_binding(path_ref: str | Path) -> dict[str, Any]:
    """Return a compact binding only after the complete partition replay passes."""
    path = resolve_project_path(path_ref) or Path(path_ref)
    payload = load_oof_manifest(path)
    return {
        "protocol": OOF_PARTITION_REPLAY_PROTOCOL,
        "manifest": to_project_ref(path),
        "manifest_sha256": _sha256(path),
        "source_segmentation_index": payload["source_segmentation_index"],
        "source_segmentation_index_sha256": payload[
            "source_segmentation_index_sha256"
        ],
        "source_bridge_index": payload["source_bridge_index"],
        "source_bridge_index_sha256": payload["source_bridge_index_sha256"],
        "expert_gate_audit": payload["expert_gate_audit"],
        "seed": int(payload["seed"]),
        "num_folds": int(payload["num_folds"]),
        "num_parents": int(payload["num_parents"]),
        "partition_binding": payload["partition_binding"],
        "fold_inventory_sha256": _canonical_sha256(payload["folds"]),
    }
