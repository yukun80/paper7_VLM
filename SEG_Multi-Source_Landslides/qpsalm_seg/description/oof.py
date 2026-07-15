#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic parent-level folds for out-of-fold predicted masks."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path, to_project_ref


OOF_FOLD_FORMAT = "qpsalm_segmentation_oof_folds_v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stable_rank(seed: int, namespace: str, parent: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{parent}".encode()).hexdigest()


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
    segmentation_rows = _read_jsonl(segmentation_path)
    bridge_rows = _read_jsonl(bridge_path)
    if not segmentation_rows:
        raise ValueError("segmentation OOF source index 不能为空")
    non_train = [
        str(row.get("sample_id") or "unknown")
        for row in segmentation_rows if str(row.get("split")) != "train"
    ]
    if non_train:
        raise ValueError(
            "OOF fold source 只能包含 split=train 的 segmentation rows: "
            f"count={len(non_train)} examples={non_train[:8]}"
        )
    sample_ids = [str(row.get("sample_id") or "") for row in segmentation_rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("OOF segmentation source sample_id 缺失或重复")

    parent_metadata: dict[str, tuple[str, str]] = {}
    for row in bridge_rows:
        if str(row.get("split")) != "train" or row.get("region_source") != "gt_global_mask":
            continue
        if not isinstance(row.get("expert_target"), dict):
            raise ValueError(
                "D4 OOF folds 只接受已冻结 expert_train 中带 expert_target 的 gt_global_mask"
            )
        parent = str(row["parent_sample_id"])
        stratum = (
            str(row.get("dataset_name") or "unknown"),
            str(row.get("modality_family_combo") or "unknown"),
        )
        previous = parent_metadata.setdefault(parent, stratum)
        if previous != stratum:
            raise ValueError(f"Bridge parent metadata 不一致: {parent}")
    if len(parent_metadata) < num_folds:
        raise ValueError(
            f"可用 Bridge train parents={len(parent_metadata)} 少于 num_folds={num_folds}"
        )

    segmentation_parents = {
        str(row.get("parent_sample_id") or row.get("sample_id"))
        for row in segmentation_rows
    }
    missing = sorted(set(parent_metadata) - segmentation_parents)
    if missing:
        raise ValueError(f"segmentation index 缺少 Bridge parents: {missing[:8]}")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for parent, stratum in parent_metadata.items():
        strata[stratum].append(parent)
    parent_to_fold: dict[str, str] = {}
    for stratum, parents in sorted(strata.items()):
        namespace = "|".join(stratum)
        ordered = sorted(parents, key=lambda value: _stable_rank(seed, namespace, value))
        offset = int(_stable_rank(seed, "offset", namespace)[:8], 16) % num_folds
        for index, parent in enumerate(ordered):
            parent_to_fold[parent] = str((index + offset) % num_folds)

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
            if str(row.get("parent_sample_id") or row.get("sample_id")) not in holdout_parents
        ]
        holdout_rows = [
            row for row in segmentation_rows
            if str(row.get("parent_sample_id") or row.get("sample_id")) in holdout_parents
        ]
        train_parents = {
            str(row.get("parent_sample_id") or row.get("sample_id")) for row in train_rows
        }
        observed_holdout_parents = {
            str(row.get("parent_sample_id") or row.get("sample_id")) for row in holdout_rows
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
        "num_parents": len(parent_to_fold),
        "parent_to_fold": dict(sorted(parent_to_fold.items())),
        "folds": folds,
    }
    _atomic_json(output / "fold_manifest.json", manifest)
    return manifest


def load_oof_manifest(path_ref: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_ref) or Path(path_ref)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != OOF_FOLD_FORMAT:
        raise ValueError(
            f"train predicted mask 需要 {OOF_FOLD_FORMAT}, observed={payload.get('protocol')!r}"
        )
    if not isinstance(payload.get("parent_to_fold"), dict) or not isinstance(payload.get("folds"), dict):
        raise ValueError("OOF manifest 缺少 parent_to_fold/folds")
    return payload
