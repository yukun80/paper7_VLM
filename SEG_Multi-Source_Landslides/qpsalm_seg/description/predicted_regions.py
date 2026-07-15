#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline, fold-audited segmentation predictions for D4 description curriculum."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.engine.checkpoint import load_checkpoint
from qpsalm_seg.engine.common import build_model
from qpsalm_seg.paths import resolve_project_path, to_project_ref
from qpsalm_seg.visualize import restore_mask_to_original

from .oof import load_oof_manifest


PREDICTED_REGION_FORMAT = "qpsalm_predicted_region_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_train_index(checkpoint: str | Path) -> tuple[Path, str]:
    checkpoint_path = resolve_project_path(checkpoint) or Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    train_ref = config.get("train_index")
    if not train_ref:
        raise ValueError("OOF segmentation checkpoint 缺少 config.train_index 审计字段")
    train_path = resolve_project_path(str(train_ref)) or Path(str(train_ref))
    if not train_path.is_file():
        raise FileNotFoundError(f"OOF checkpoint 声明的 train index 不存在: {train_path}")
    return train_path, _sha256(train_path)


def _validate_oof_checkpoint(
    checkpoint: str | Path,
    manifest: dict[str, Any],
    checkpoint_fold: str,
    prediction_index: str | Path,
) -> dict[str, Any]:
    fold = (manifest.get("folds") or {}).get(str(checkpoint_fold))
    if not isinstance(fold, dict):
        raise ValueError(f"OOF manifest 不包含 held-out fold={checkpoint_fold}")
    expected_path = resolve_project_path(str(fold.get("train_index") or ""))
    if expected_path is None or not expected_path.is_file():
        raise FileNotFoundError(f"OOF fold train index 不存在: {fold.get('train_index')}")
    expected_hash = str(fold.get("train_index_sha256") or "")
    if _sha256(expected_path) != expected_hash:
        raise ValueError(f"OOF fold train index 已变化: fold={checkpoint_fold}")
    observed_path, observed_hash = _checkpoint_train_index(checkpoint)
    if observed_hash != expected_hash:
        raise ValueError(
            "checkpoint 不是由对应 held-out fold 的排除索引训练: "
            f"fold={checkpoint_fold} checkpoint_train={observed_path}"
        )
    holdout_path = resolve_project_path(str(fold.get("holdout_index") or ""))
    prediction_path = resolve_project_path(prediction_index) or Path(prediction_index)
    expected_holdout_hash = str(fold.get("holdout_index_sha256") or "")
    if holdout_path is None or not holdout_path.is_file():
        raise FileNotFoundError(f"OOF fold holdout index 不存在: {fold.get('holdout_index')}")
    if _sha256(holdout_path) != expected_holdout_hash:
        raise ValueError(f"OOF fold holdout index 已变化: fold={checkpoint_fold}")
    if not prediction_path.is_file() or _sha256(prediction_path) != expected_holdout_hash:
        raise ValueError(
            "train OOF prediction 必须读取 manifest 对应的 holdout index: "
            f"fold={checkpoint_fold} prediction_index={prediction_path}"
        )
    return {
        "held_out_fold": str(checkpoint_fold),
        "train_index": str(fold["train_index"]),
        "train_index_sha256": expected_hash,
        "checkpoint_train_index": to_project_ref(observed_path),
        "prediction_index": to_project_ref(prediction_path),
        "prediction_index_sha256": expected_holdout_hash,
    }


@torch.no_grad()
def export_predicted_regions(
    *,
    segmentation_config,
    checkpoint: str | Path,
    source_index: str | Path,
    split: str,
    output_dir: str | Path,
    device: torch.device,
    threshold: float,
    fold_manifest: str | Path | None = None,
    checkpoint_fold: str | None = None,
    prediction_index: str | Path | None = None,
    max_parents: int = 0,
) -> dict[str, Any]:
    if split == "train" and (
        fold_manifest is None or checkpoint_fold is None or prediction_index is None
    ):
        raise ValueError(
            "train predicted-mask curriculum 必须提供 --fold-manifest、--checkpoint-fold "
            "和 --prediction-index，"
            "禁止使用 in-fold prediction"
        )
    fold_payload = load_oof_manifest(fold_manifest) if split == "train" else None
    fold_by_parent = {
        str(key): str(value)
        for key, value in ((fold_payload or {}).get("parent_to_fold") or {}).items()
    }
    fold_audit = (
        _validate_oof_checkpoint(
            checkpoint, fold_payload, str(checkpoint_fold), prediction_index
        )
        if split == "train" else None
    )
    source_path = resolve_project_path(source_index) or Path(source_index)
    source_rows = _read_jsonl(source_path)
    canonical: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        if str(row.get("split")) != split or row.get("region_source") != "gt_global_mask":
            continue
        if split == "train" and not isinstance(row.get("expert_target"), dict):
            raise ValueError(
                "train predicted-mask curriculum 只允许从 expert_train 导出，"
                "source row 缺少 expert_target"
            )
        parent = str(row["parent_sample_id"])
        if split == "train":
            if parent not in fold_by_parent:
                raise ValueError(f"fold manifest 缺少 train parent={parent}")
            if fold_by_parent[parent] != str(checkpoint_fold):
                continue
        canonical.setdefault(parent, row)
    parents = sorted(canonical)
    if max_parents > 0:
        parents = parents[:max_parents]
    dataset_config = (
        replace(segmentation_config, train_index=str(prediction_index))
        if split == "train" and prediction_index is not None else segmentation_config
    )
    dataset = MultiSourceLandslideDataset(dataset_config, split)
    indices: dict[str, int] = {}
    priorities: dict[str, int] = {}
    for index, row in enumerate(dataset.rows):
        parent = str(row.get("parent_sample_id") or row.get("sample_id"))
        priority = 0 if row.get("task_family") == "global_landslide_segmentation" else 1
        if parent not in priorities or priority < priorities[parent]:
            priorities[parent] = priority
            indices[parent] = index
    missing = sorted(set(parents) - set(indices))
    if missing:
        raise KeyError(f"segmentation index 缺少 Bridge parents: {missing[:8]}")
    model = build_model(segmentation_config, device)
    checkpoint_step = load_checkpoint(checkpoint, model)
    model.eval()
    target_dir = resolve_project_path(output_dir) or Path(output_dir)
    mask_dir = target_dir / "masks" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for parent in parents:
        item = dataset[indices[parent]]
        batch = qpsalm_collate([item])
        with model.controller.adapter_scope("default"):
            output = model(batch)
        canvas = (torch.sigmoid(output.final_mask_logits[0, 0].float()).cpu().numpy() >= threshold).astype(np.uint8)
        restored = restore_mask_to_original(canvas, item["metadata"]["resize_transform"])
        if restored is None:
            raise ValueError(f"无法恢复 predicted mask 原尺寸: parent={parent}")
        destination = mask_dir / f"{parent}.npy"
        temporary = destination.with_suffix(".npy.part")
        with temporary.open("wb") as handle:
            np.save(handle, restored.astype(np.uint8), allow_pickle=False)
        temporary.replace(destination)
        source = canonical[parent]
        row = {
            **source,
            "schema_version": PREDICTED_REGION_FORMAT,
            "bridge_record_id": f"predicted::{parent}::{checkpoint_step}",
            "region_id": "predicted_global",
            "region_source": "predicted_proposal",
            "region_mask": {
                "path": to_project_ref(destination),
                "sha256": _sha256(destination),
                "shape": list(restored.shape),
                "threshold": float(threshold),
            },
            "prediction_provenance": {
                "checkpoint": str(checkpoint),
                "checkpoint_step": checkpoint_step,
                "split": split,
                "fold_manifest": str(fold_manifest) if fold_manifest is not None else None,
                "checkpoint_fold": checkpoint_fold,
                "out_of_fold_verified": split != "train" or fold_by_parent[parent] == str(checkpoint_fold),
                "fold_audit": fold_audit,
            },
        }
        output_rows.append(row)
    index_path = target_dir / f"predicted_{split}{'_' + str(checkpoint_fold) if checkpoint_fold else ''}.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    report = {
        "format": PREDICTED_REGION_FORMAT,
        "split": split,
        "num_parents": len(output_rows),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_fold": checkpoint_fold,
        "out_of_fold_verified": split != "train" or bool(fold_audit),
        "index": str(index_path),
    }
    (target_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
