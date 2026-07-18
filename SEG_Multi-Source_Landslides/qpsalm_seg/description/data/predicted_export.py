#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint-bound fixed and OOF predicted-region export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.engine.checkpoint import load_checkpoint
from qpsalm_seg.engine.common import build_model
from qpsalm_seg.paths import (
    resolve_project_path,
    to_project_ref,
)
from qpsalm_seg.visualize import restore_mask_to_original

from .expert_contracts import require_frozen_expert_bridge, validate_expert_rows
from .oof import load_oof_manifest
from .predicted_checkpoint import validate_oof_checkpoint
from .predicted_validation import revalidate_fixed_predicted_index
from .predicted_contracts import (
    FIXED_PREDICTION_ARTIFACT_PROTOCOL,
    PREDICTED_REGION_FORMAT,
    atomic_write_prediction_json,
    atomic_write_prediction_jsonl,
    read_prediction_jsonl,
)
from ..protocols.io import canonical_sha256, sha256_file


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
        validate_oof_checkpoint(
            checkpoint,
            fold_payload,
            str(checkpoint_fold),
            prediction_index,
            fold_manifest=fold_manifest,
        )
        if split == "train" else None
    )
    source_path = resolve_project_path(source_index) or Path(source_index)
    bridge_root = source_path.parent.parent
    expected_source = bridge_root / f"indexes/expert_{split}.jsonl"
    if source_path.resolve(strict=False) != expected_source.resolve(strict=False):
        raise ValueError(
            f"predicted-mask source-index 必须是 frozen Bridge 的 expert_{split}.jsonl"
        )
    expert_gate_audit = require_frozen_expert_bridge(bridge_root)
    if split == "train":
        manifest_source = resolve_project_path(
            str((fold_payload or {}).get("source_bridge_index") or "")
        )
        if (
            manifest_source is None
            or manifest_source.resolve(strict=False) != source_path.resolve(strict=False)
            or str((fold_payload or {}).get("source_bridge_index_sha256") or "")
            != sha256_file(source_path)
            or (fold_payload or {}).get("expert_gate_audit") != expert_gate_audit
        ):
            raise ValueError("OOF manifest 与当前 frozen expert_train/gate 绑定不一致")
    source_rows = read_prediction_jsonl(source_path)
    validate_expert_rows(source_rows, stage="bridge_expert", split=split)
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
    eligible_parents = sorted(canonical)
    parents = list(eligible_parents)
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
    checkpoint_path = resolve_project_path(checkpoint) or Path(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if split == "train" and (
        int((fold_audit or {}).get("checkpoint_step", -1)) != checkpoint_step
        or str((fold_audit or {}).get("checkpoint_sha256") or "")
        != checkpoint_sha256
    ):
        raise RuntimeError("OOF checkpoint 在审计与模型加载之间发生变化")
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
        try:
            destination.resolve(strict=False).relative_to(
                mask_dir.resolve(strict=False)
            )
        except ValueError as exc:
            raise ValueError(
                f"predicted parent 不能映射为安全 mask 文件名: {parent!r}"
            ) from exc
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
                "sha256": sha256_file(destination),
                "shape": list(restored.shape),
                "threshold": float(threshold),
            },
            "prediction_provenance": {
                "checkpoint": to_project_ref(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": checkpoint_step,
                "split": split,
                "fold_manifest": (
                    (fold_audit or {}).get("fold_manifest")
                    if split == "train" else None
                ),
                "fold_manifest_sha256": (
                    (fold_audit or {}).get("fold_manifest_sha256")
                    if split == "train" else None
                ),
                "checkpoint_fold": checkpoint_fold,
                "out_of_fold_verified": split != "train" or fold_by_parent[parent] == str(checkpoint_fold),
                "fold_audit": fold_audit,
                "source_bridge_record_id": str(
                    source.get("bridge_record_id") or source.get("sample_id") or ""
                ),
                "source_expert_record_sha256": canonical_sha256(source),
            },
        }
        output_rows.append(row)
    index_path = target_dir / f"predicted_{split}{'_' + str(checkpoint_fold) if checkpoint_fold else ''}.jsonl"
    atomic_write_prediction_jsonl(index_path, output_rows)
    report = {
        "format": PREDICTED_REGION_FORMAT,
        "validation_protocol": FIXED_PREDICTION_ARTIFACT_PROTOCOL,
        "split": split,
        "requested_max_parents": int(max_parents),
        "num_parents": len(output_rows),
        "num_eligible_parents": len(eligible_parents),
        "population_complete": parents == eligible_parents,
        "population_sha256": canonical_sha256(parents),
        "mask_inventory_sha256": canonical_sha256([
            {
                "parent_sample_id": str(row["parent_sample_id"]),
                "path": str(row["region_mask"]["path"]),
                "sha256": str(row["region_mask"]["sha256"]),
            }
            for row in output_rows
        ]),
        "mask_bytes": sum(
            int((resolve_project_path(str(row["region_mask"]["path"]))
                 or Path(str(row["region_mask"]["path"]))).stat().st_size)
            for row in output_rows
        ),
        "checkpoint": to_project_ref(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_fold": checkpoint_fold,
        "out_of_fold_verified": split != "train" or bool(fold_audit),
        "source_bridge_index": to_project_ref(source_path),
        "source_bridge_index_sha256": sha256_file(source_path),
        "expert_gate_audit": expert_gate_audit,
        "fold_manifest": to_project_ref(
            resolve_project_path(fold_manifest) or Path(fold_manifest)
        ) if fold_manifest is not None else None,
        "fold_manifest_sha256": sha256_file(
            resolve_project_path(fold_manifest) or Path(fold_manifest)
        ) if fold_manifest is not None else None,
        "index": to_project_ref(index_path),
        "index_sha256": sha256_file(index_path),
    }
    report_path = target_dir / "report.json"
    atomic_write_prediction_json(report_path, report)
    if split in {"val", "test"}:
        # 发布端立即执行与 D4/M6/M7 消费端相同的全量重放；任何半成品都非零失败。
        revalidate_fixed_predicted_index(
            index_path,
            split=split,
            expected_expert_gate_audit=expert_gate_audit,
            require_complete=False,
        )
    return report
