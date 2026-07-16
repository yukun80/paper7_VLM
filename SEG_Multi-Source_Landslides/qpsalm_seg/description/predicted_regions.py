#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline, fold-audited segmentation predictions for D4 description curriculum."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from qpsalm_seg.engine.common import build_model
from qpsalm_seg.paths import (
    resolve_project_path,
    to_project_ref,
    validate_output_replacement_safety,
)
from qpsalm_seg.visualize import restore_mask_to_original

from .data import _validate_expert_rows, require_frozen_expert_bridge
from .json_protocol import strict_json_loads
from .oof import load_oof_manifest, oof_manifest_artifact_binding


PREDICTED_REGION_FORMAT = "qpsalm_predicted_region_v2_checkpoint_bound"
OOF_CHECKPOINT_BINDING_PROTOCOL = (
    "qpsalm_segmentation_oof_checkpoint_binding_v1_cache_index_replayed"
)
OOF_MERGE_PROTOCOL = (
    "qpsalm_predicted_region_oof_merge_v4_exact_fold_publications_replayed"
)
FIXED_PREDICTION_ARTIFACT_PROTOCOL = (
    "qpsalm_fixed_predicted_region_artifact_v3_exact_mask_directory_bound"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        strict_json_loads(line)
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
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_index_fingerprint(
    *,
    fingerprint: Any,
    expected_path: Path,
    split: str,
) -> dict[str, Any]:
    if not isinstance(fingerprint, dict):
        raise ValueError(f"OOF checkpoint Vision Cache v3 缺少 {split} index fingerprint")
    reference = resolve_project_path(str(fingerprint.get("reference") or ""))
    expected_hash = _sha256(expected_path)
    expected_size = int(expected_path.stat().st_size)
    if (
        fingerprint.get("status") != "present"
        or reference is None
        or reference.resolve(strict=False) != expected_path.resolve(strict=False)
        or str(fingerprint.get("sha256") or "") != expected_hash
        or int(fingerprint.get("size") or -1) != expected_size
    ):
        raise ValueError(
            f"OOF checkpoint Vision Cache v3 的 {split} index 未绑定精确 fold index"
        )
    return {
        "reference": to_project_ref(reference),
        "sha256": expected_hash,
        "size": expected_size,
    }


def _inspect_oof_segmentation_checkpoint(
    checkpoint: str | Path,
    *,
    expected_train_path: Path,
    expected_holdout_path: Path,
) -> dict[str, Any]:
    checkpoint_path = resolve_project_path(checkpoint) or Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OOF segmentation checkpoint 不存在: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "OOF segmentation checkpoint format 非法: "
            f"observed={getattr(payload, 'get', lambda *_: None)('format')!r} "
            f"expected={CHECKPOINT_FORMAT!r}"
        )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("OOF segmentation checkpoint 缺少完整 config")
    train_path = resolve_project_path(str(config.get("train_index") or ""))
    val_path = resolve_project_path(str(config.get("val_index") or ""))
    if (
        train_path is None
        or train_path.resolve(strict=False) != expected_train_path.resolve(strict=False)
    ):
        raise ValueError("OOF checkpoint config.train_index 不是精确 fold train index")
    if (
        val_path is None
        or val_path.resolve(strict=False) != expected_holdout_path.resolve(strict=False)
    ):
        raise ValueError("OOF checkpoint config.val_index 不是精确 fold holdout index")
    evidence = payload.get("evidence_protocol")
    input_protocol = (
        evidence.get("input_protocol") if isinstance(evidence, dict) else None
    )
    fingerprints = (
        input_protocol.get("index_fingerprints")
        if isinstance(input_protocol, dict)
        else None
    )
    if not isinstance(fingerprints, dict):
        raise ValueError("OOF checkpoint 缺少 Vision Cache v3 input_protocol")
    cache_indexes = {
        "train": _checkpoint_index_fingerprint(
            fingerprint=fingerprints.get("train"),
            expected_path=expected_train_path,
            split="train",
        ),
        "val": _checkpoint_index_fingerprint(
            fingerprint=fingerprints.get("val"),
            expected_path=expected_holdout_path,
            split="val",
        ),
    }
    try:
        step = int(payload.get("step", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("OOF checkpoint step 非法") from error
    if step <= 0:
        raise ValueError("OOF checkpoint 必须来自至少一个已完成 optimizer step")
    return {
        "checkpoint": to_project_ref(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_step": step,
        "config_train_index": to_project_ref(train_path),
        "config_val_index": to_project_ref(val_path),
        "vision_cache_index_fingerprints": cache_indexes,
    }


def _validate_oof_checkpoint(
    checkpoint: str | Path,
    manifest: dict[str, Any],
    checkpoint_fold: str,
    prediction_index: str | Path,
    *,
    fold_manifest: str | Path,
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
    holdout_path = resolve_project_path(str(fold.get("holdout_index") or ""))
    prediction_path = resolve_project_path(prediction_index) or Path(prediction_index)
    expected_holdout_hash = str(fold.get("holdout_index_sha256") or "")
    if holdout_path is None or not holdout_path.is_file():
        raise FileNotFoundError(f"OOF fold holdout index 不存在: {fold.get('holdout_index')}")
    if _sha256(holdout_path) != expected_holdout_hash:
        raise ValueError(f"OOF fold holdout index 已变化: fold={checkpoint_fold}")
    if (
        not prediction_path.is_file()
        or prediction_path.resolve(strict=False) != holdout_path.resolve(strict=False)
        or _sha256(prediction_path) != expected_holdout_hash
    ):
        raise ValueError(
            "train OOF prediction 必须读取 manifest 对应的 holdout index: "
            f"fold={checkpoint_fold} prediction_index={prediction_path}"
        )
    checkpoint_audit = _inspect_oof_segmentation_checkpoint(
        checkpoint,
        expected_train_path=expected_path,
        expected_holdout_path=holdout_path,
    )
    manifest_binding = oof_manifest_artifact_binding(fold_manifest)
    return {
        "protocol": OOF_CHECKPOINT_BINDING_PROTOCOL,
        "held_out_fold": str(checkpoint_fold),
        "fold_manifest": manifest_binding["manifest"],
        "fold_manifest_sha256": manifest_binding["manifest_sha256"],
        "train_index": str(fold["train_index"]),
        "train_index_sha256": expected_hash,
        "holdout_index": str(fold["holdout_index"]),
        "holdout_index_sha256": expected_holdout_hash,
        "prediction_index": to_project_ref(prediction_path),
        "prediction_index_sha256": expected_holdout_hash,
        **checkpoint_audit,
    }


def validate_oof_checkpoint_binding(
    *,
    checkpoint: str | Path,
    fold_manifest: str | Path,
    checkpoint_fold: str,
    prediction_index: str | Path,
) -> dict[str, Any]:
    """Replay fold sources, checkpoint config and Vision Cache v3 index binding."""
    manifest = load_oof_manifest(fold_manifest)
    return _validate_oof_checkpoint(
        checkpoint,
        manifest,
        str(checkpoint_fold),
        prediction_index,
        fold_manifest=fold_manifest,
    )


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
            != _sha256(source_path)
            or (fold_payload or {}).get("expert_gate_audit") != expert_gate_audit
        ):
            raise ValueError("OOF manifest 与当前 frozen expert_train/gate 绑定不一致")
    source_rows = _read_jsonl(source_path)
    _validate_expert_rows(source_rows, stage="bridge_expert", split=split)
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
    checkpoint_sha256 = _sha256(checkpoint_path)
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
                "sha256": _sha256(destination),
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
                "source_expert_record_sha256": _canonical_sha256(source),
            },
        }
        output_rows.append(row)
    index_path = target_dir / f"predicted_{split}{'_' + str(checkpoint_fold) if checkpoint_fold else ''}.jsonl"
    _atomic_jsonl(index_path, output_rows)
    report = {
        "format": PREDICTED_REGION_FORMAT,
        "validation_protocol": FIXED_PREDICTION_ARTIFACT_PROTOCOL,
        "split": split,
        "requested_max_parents": int(max_parents),
        "num_parents": len(output_rows),
        "num_eligible_parents": len(eligible_parents),
        "population_complete": parents == eligible_parents,
        "population_sha256": _canonical_sha256(parents),
        "mask_inventory_sha256": _canonical_sha256([
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
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_fold": checkpoint_fold,
        "out_of_fold_verified": split != "train" or bool(fold_audit),
        "source_bridge_index": to_project_ref(source_path),
        "source_bridge_index_sha256": _sha256(source_path),
        "expert_gate_audit": expert_gate_audit,
        "fold_manifest": to_project_ref(
            resolve_project_path(fold_manifest) or Path(fold_manifest)
        ) if fold_manifest is not None else None,
        "fold_manifest_sha256": _sha256(
            resolve_project_path(fold_manifest) or Path(fold_manifest)
        ) if fold_manifest is not None else None,
        "index": to_project_ref(index_path),
        "index_sha256": _sha256(index_path),
    }
    report_path = target_dir / "report.json"
    _atomic_json(report_path, report)
    if split in {"val", "test"}:
        # 发布端立即执行与 D4/M6/M7 消费端相同的全量重放；任何半成品都非零失败。
        revalidate_fixed_predicted_index(
            index_path,
            split=split,
            expected_expert_gate_audit=expert_gate_audit,
            require_complete=False,
        )
    return report


def revalidate_fixed_predicted_index(
    index_path: str | Path,
    *,
    split: str,
    expected_expert_gate_audit: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    """Deeply replay a fixed val/test prediction publication before use."""
    if split not in {"val", "test"}:
        raise ValueError("fixed predicted artifact replay 只接受 val/test")
    path = resolve_project_path(index_path) or Path(index_path)
    report_path = path.parent / "report.json"
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError("fixed predicted index/report 缺失")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != PREDICTED_REGION_FORMAT
        or report.get("validation_protocol") != FIXED_PREDICTION_ARTIFACT_PROTOCOL
        or str(report.get("split")) != split
    ):
        raise ValueError("fixed predicted report protocol/split 非法")
    reported_index = resolve_project_path(str(report.get("index") or ""))
    if (
        reported_index is None
        or reported_index.resolve(strict=False) != path.resolve(strict=False)
        or str(report.get("index_sha256") or "") != _sha256(path)
    ):
        raise ValueError("fixed predicted index path/hash 已漂移")
    source_path = resolve_project_path(str(report.get("source_bridge_index") or ""))
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError("fixed predicted report 绑定的 expert source 缺失")
    expected_source = source_path.parent.parent / f"indexes/expert_{split}.jsonl"
    if (
        source_path.resolve(strict=False) != expected_source.resolve(strict=False)
        or str(report.get("source_bridge_index_sha256") or "") != _sha256(source_path)
    ):
        raise ValueError("fixed predicted report 未绑定当前 expert split index")
    current_gate = require_frozen_expert_bridge(source_path.parent.parent)
    if (
        report.get("expert_gate_audit") != current_gate
        or current_gate != expected_expert_gate_audit
    ):
        raise ValueError("fixed predicted report 与当前 frozen Bridge gate 不一致")
    source_rows = _read_jsonl(source_path)
    _validate_expert_rows(source_rows, stage="bridge_expert", split=split)
    source_by_parent: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        if str(source.get("split")) != split:
            raise ValueError("fixed predicted expert source 包含错误 split")
        if source.get("region_source") != "gt_global_mask":
            continue
        parent = str(source.get("parent_sample_id") or "")
        previous = source_by_parent.setdefault(parent, source)
        if not parent or previous != source:
            raise ValueError("fixed predicted expert source 的 global parent 缺失或重复")
    if not source_by_parent:
        raise ValueError("fixed predicted expert source 没有 gt_global_mask parent")

    checkpoint_path = resolve_project_path(str(report.get("checkpoint") or ""))
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError("fixed predicted segmentation checkpoint 缺失")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != str(report.get("checkpoint_sha256") or ""):
        raise ValueError("fixed predicted segmentation checkpoint hash 已漂移")
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("format") != CHECKPOINT_FORMAT
        or int(checkpoint_payload.get("step", -1)) <= 0
        or int(checkpoint_payload.get("step", -1))
        != int(report.get("checkpoint_step", -2))
    ):
        raise ValueError("fixed predicted segmentation checkpoint payload/step 非法")

    rows = _read_jsonl(path)
    _revalidate_prediction_publication_directory(path, rows, split=split)
    _validate_expert_rows(rows, stage="predicted_mask", split=split)
    by_parent: dict[str, dict[str, Any]] = {}
    mask_paths: set[str] = set()
    mask_inventory: list[dict[str, str]] = []
    mask_bytes = 0
    for row in rows:
        parent = str(row.get("parent_sample_id") or "")
        provenance = row.get("prediction_provenance")
        source = source_by_parent.get(parent)
        if (
            row.get("schema_version") != PREDICTED_REGION_FORMAT
            or str(row.get("split")) != split
            or row.get("region_source") != "predicted_proposal"
            or row.get("region_id") != "predicted_global"
            or not isinstance(provenance, dict)
            or source is None
        ):
            raise ValueError(f"fixed predicted row protocol/source 非法: {parent}")
        if (
            str(provenance.get("checkpoint") or "") != to_project_ref(checkpoint_path)
            or str(provenance.get("checkpoint_sha256") or "") != checkpoint_sha256
            or int(provenance.get("checkpoint_step", -1))
            != int(checkpoint_payload["step"])
            or str(provenance.get("split")) != split
            or provenance.get("out_of_fold_verified") is not True
            or provenance.get("checkpoint_fold") is not None
            or provenance.get("fold_audit") is not None
            or str(provenance.get("source_bridge_record_id") or "")
            != str(source.get("bridge_record_id") or source.get("sample_id") or "")
            or str(provenance.get("source_expert_record_sha256") or "")
            != _canonical_sha256(source)
            or row.get("expert_target") != source.get("expert_target")
            or row.get("review") != source.get("review")
        ):
            raise ValueError(f"fixed predicted row provenance/source 已漂移: {parent}")
        if parent in by_parent:
            raise ValueError(f"fixed predicted parent 重复: {parent}")
        expected_mask_path = path.parent / "masks" / split / f"{parent}.npy"
        mask_path, size, mask_sha256 = _validate_prediction_mask(
            row,
            parent,
            expected_path=expected_mask_path,
        )
        if mask_path in mask_paths:
            raise ValueError(f"fixed predicted mask 被多个 parent 复用: {mask_path}")
        mask_paths.add(mask_path)
        mask_bytes += size
        mask_inventory.append({
            "parent_sample_id": parent,
            "path": str(row["region_mask"]["path"]),
            "sha256": mask_sha256,
        })
        by_parent[parent] = row
    eligible_parents = sorted(source_by_parent)
    try:
        requested_max_parents = int(report.get("requested_max_parents", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("fixed predicted requested_max_parents 非法") from error
    if requested_max_parents < 0:
        raise ValueError("fixed predicted report 缺少 requested_max_parents")
    expected_parents = (
        eligible_parents[:requested_max_parents]
        if requested_max_parents > 0
        else eligible_parents
    )
    observed_parents = sorted(by_parent)
    if observed_parents != expected_parents:
        raise ValueError("fixed predicted index 未按确定性 prefix 覆盖 expert split parents")
    population_complete = expected_parents == eligible_parents
    if require_complete and not population_complete:
        raise ValueError("正式 fixed predicted index 必须完整覆盖 expert split global parents")
    expected_report = {
        "num_parents": len(observed_parents),
        "num_eligible_parents": len(eligible_parents),
        "population_complete": population_complete,
        "population_sha256": _canonical_sha256(observed_parents),
        "mask_inventory_sha256": _canonical_sha256(mask_inventory),
        "mask_bytes": mask_bytes,
    }
    if {name: report.get(name) for name in expected_report} != expected_report:
        raise ValueError("fixed predicted report population/mask statistics 未通过重放")
    return {
        "protocol": FIXED_PREDICTION_ARTIFACT_PROTOCOL,
        "report": to_project_ref(report_path),
        "report_sha256": _sha256(report_path),
        "index": to_project_ref(path),
        "index_sha256": _sha256(path),
        "split": split,
        "segmentation_checkpoint": to_project_ref(checkpoint_path),
        "segmentation_checkpoint_sha256": checkpoint_sha256,
        "segmentation_checkpoint_step": int(checkpoint_payload["step"]),
        "source_bridge_index": to_project_ref(source_path),
        "source_bridge_index_sha256": _sha256(source_path),
        "num_parents": len(observed_parents),
        "population_complete": population_complete,
        "population_sha256": expected_report["population_sha256"],
        "mask_inventory_sha256": expected_report["mask_inventory_sha256"],
    }


def _validate_prediction_mask(
    row: dict[str, Any],
    parent: str,
    *,
    expected_path: Path | None = None,
) -> tuple[str, int, str]:
    mask = row.get("region_mask")
    if not isinstance(mask, dict) or not mask.get("path"):
        raise ValueError(f"OOF prediction 缺少 region_mask.path: {parent}")
    path = resolve_project_path(str(mask["path"])) or Path(str(mask["path"]))
    if (
        expected_path is not None
        and path.resolve(strict=False) != expected_path.resolve(strict=False)
    ):
        raise ValueError(
            f"fixed predicted mask path 不在确定性 publication 目录: {parent}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"OOF prediction mask 不存在: {parent} path={path}")
    observed_hash = _sha256(path)
    if observed_hash != str(mask.get("sha256") or ""):
        raise ValueError(f"OOF prediction mask hash 不一致: {parent}")
    values = np.load(path, allow_pickle=False)
    try:
        expected_shape = tuple(int(value) for value in mask.get("shape", []))
    except (TypeError, ValueError) as error:
        raise ValueError(f"OOF prediction mask shape metadata 非法: {parent}") from error
    if values.ndim != 2 or tuple(values.shape) != expected_shape:
        raise ValueError(
            f"OOF prediction mask shape 不一致: {parent} "
            f"observed={tuple(values.shape)} expected={expected_shape}"
        )
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"OOF prediction mask 不是 binary: {parent}")
    return (
        str(path.resolve(strict=False)),
        int(path.stat().st_size),
        observed_hash,
    )


def _revalidate_prediction_publication_directory(
    index_path: Path,
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> None:
    """Require one deterministic mask file per indexed parent and no leftovers."""
    root = index_path.parent.resolve(strict=False)
    mask_root = (root / "masks" / split).resolve(strict=False)
    expected_files: set[str] = set()
    for row in rows:
        parent = str(row.get("parent_sample_id") or "")
        mask = row.get("region_mask")
        path = resolve_project_path(
            str(mask.get("path") or "") if isinstance(mask, dict) else ""
        )
        expected = (mask_root / f"{parent}.npy").resolve(strict=False)
        try:
            expected.relative_to(mask_root)
        except ValueError as exc:
            raise ValueError(
                f"predicted parent 逃逸 publication mask 目录: {parent!r}"
            ) from exc
        if not parent or path is None or path.resolve(strict=False) != expected:
            raise ValueError(
                f"predicted mask path 不在确定性 publication 目录: {parent!r}"
            )
        expected_files.add(str(expected))
    observed_files = {
        str(candidate.resolve(strict=False))
        for candidate in mask_root.rglob("*")
        if candidate.is_file()
    } if mask_root.is_dir() else set()
    if observed_files != expected_files:
        raise ValueError(
            "predicted mask 目录存在未绑定、临时或缺失文件"
        )
    part_files = sorted(
        str(candidate.relative_to(root))
        for candidate in root.rglob("*.part")
        if candidate.is_file()
    )
    if part_files:
        raise ValueError(
            f"predicted publication 残留 .part 文件: {part_files[:8]}"
        )


def _audit_oof_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    fold_manifest: str | Path,
) -> dict[str, Any]:
    parent_to_fold = {
        str(parent): str(fold)
        for parent, fold in manifest["parent_to_fold"].items()
    }
    source_bridge = resolve_project_path(str(manifest["source_bridge_index"]))
    if source_bridge is None or not source_bridge.is_file():
        raise FileNotFoundError("OOF source expert_train 在 row replay 时缺失")
    source_by_parent: dict[str, dict[str, Any]] = {}
    for source in _read_jsonl(source_bridge):
        if source.get("region_source") != "gt_global_mask":
            continue
        parent = str(source.get("parent_sample_id") or "")
        previous = source_by_parent.setdefault(parent, source)
        if previous != source:
            raise ValueError(f"OOF source expert_train 存在重复 gt_global_mask parent={parent}")

    merged: dict[str, dict[str, Any]] = {}
    fold_counts: Counter[str] = Counter()
    mask_paths: set[str] = set()
    mask_inventory: list[dict[str, Any]] = []
    mask_bytes = 0
    checkpoint_audits: dict[str, dict[str, Any]] = {}
    live_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        parent = str(row.get("parent_sample_id") or "")
        provenance = row.get("prediction_provenance")
        if row.get("schema_version") != PREDICTED_REGION_FORMAT:
            raise ValueError(f"predicted region format 非法: {parent}")
        if (
            str(row.get("split")) != "train"
            or row.get("region_source") != "predicted_proposal"
            or row.get("region_id") != "predicted_global"
        ):
            raise ValueError(f"OOF merge 只接受 train predicted_global row: {parent}")
        if not parent or parent not in parent_to_fold:
            raise ValueError(f"prediction parent 不在 fold manifest: {parent}")
        if not isinstance(provenance, dict):
            raise ValueError(f"prediction 缺少 provenance: {parent}")
        fold = parent_to_fold[parent]
        if str(provenance.get("checkpoint_fold")) != fold:
            raise ValueError(f"prediction fold 归属错误: {parent}")
        if provenance.get("out_of_fold_verified") is not True:
            raise ValueError(f"prediction 未通过 OOF checkpoint 审计: {parent}")
        source = source_by_parent.get(parent)
        if source is None:
            raise ValueError(f"prediction parent 不在当前 expert_train gt_global_mask: {parent}")
        if (
            str(provenance.get("source_bridge_record_id") or "")
            != str(source.get("bridge_record_id") or source.get("sample_id") or "")
            or str(provenance.get("source_expert_record_sha256") or "")
            != _canonical_sha256(source)
            or row.get("expert_target") != source.get("expert_target")
            or row.get("review") != source.get("review")
        ):
            raise ValueError(f"prediction expert source record 绑定不一致: {parent}")
        checkpoint_ref = str(provenance.get("checkpoint") or "")
        cache_key = (fold, checkpoint_ref)
        if cache_key not in live_cache:
            expected_holdout = (manifest["folds"][fold] or {}).get("holdout_index")
            live_cache[cache_key] = _validate_oof_checkpoint(
                checkpoint_ref,
                manifest,
                fold,
                str(expected_holdout or ""),
                fold_manifest=fold_manifest,
            )
        live_audit = live_cache[cache_key]
        if provenance.get("fold_audit") != live_audit:
            raise ValueError(f"prediction fold audit 未通过当前 artifact 重放: {parent}")
        if (
            str(provenance.get("checkpoint_sha256") or "")
            != live_audit["checkpoint_sha256"]
            or int(provenance.get("checkpoint_step", -1))
            != int(live_audit["checkpoint_step"])
            or str(provenance.get("fold_manifest") or "")
            != str(live_audit["fold_manifest"])
            or str(provenance.get("fold_manifest_sha256") or "")
            != str(live_audit["fold_manifest_sha256"])
        ):
            raise ValueError(f"prediction checkpoint/manifest binding 不一致: {parent}")
        previous_audit = checkpoint_audits.setdefault(fold, live_audit)
        if previous_audit != live_audit:
            raise ValueError(f"同一 fold 使用了多个 segmentation checkpoint: fold={fold}")
        if parent in merged:
            raise ValueError(f"OOF prediction parent 重复: {parent}")
        mask_path, size, mask_sha256 = _validate_prediction_mask(row, parent)
        if mask_path in mask_paths:
            raise ValueError(f"多个 parent 复用了同一个 predicted mask: {mask_path}")
        mask_paths.add(mask_path)
        mask_bytes += size
        mask_inventory.append({
            "parent_sample_id": parent,
            "path": mask_path,
            "sha256": mask_sha256,
            "bytes": size,
        })
        fold_counts[fold] += 1
        merged[parent] = row
    _validate_expert_rows(
        list(merged.values()), stage="predicted_mask", split="train"
    )
    missing = sorted(set(parent_to_fold) - set(merged))
    unexpected = sorted(set(merged) - set(parent_to_fold))
    if missing or unexpected:
        raise ValueError(
            "OOF prediction parent 覆盖不精确: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"examples={(missing + unexpected)[:8]}"
        )
    expected_fold_counts = Counter(parent_to_fold.values())
    if fold_counts != expected_fold_counts:
        raise ValueError(
            "OOF prediction fold 覆盖不一致: "
            f"observed={dict(fold_counts)} expected={dict(expected_fold_counts)}"
        )
    if set(checkpoint_audits) != set(manifest["folds"]):
        raise ValueError("OOF prediction 未绑定每个 fold 的唯一 checkpoint")
    ordered_rows = [merged[parent] for parent in sorted(merged)]
    return {
        "rows": ordered_rows,
        "num_parents": len(ordered_rows),
        "parents_per_fold": dict(sorted(fold_counts.items())),
        "num_unique_masks": len(mask_paths),
        "mask_bytes": mask_bytes,
        "mask_inventory_sha256": _canonical_sha256(
            sorted(mask_inventory, key=lambda value: value["parent_sample_id"])
        ),
        "fold_checkpoint_audits": dict(sorted(checkpoint_audits.items())),
        "population_sha256": _canonical_sha256(
            [str(row["parent_sample_id"]) for row in ordered_rows]
        ),
    }


def merge_oof_predictions(
    *,
    fold_manifest: str | Path,
    input_indexes: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Merge OOF rows only after replaying sources, folds, checkpoints and masks."""
    if not input_indexes:
        raise ValueError("OOF merge 至少需要一个 fold prediction index")
    manifest = load_oof_manifest(fold_manifest)
    manifest_binding = oof_manifest_artifact_binding(fold_manifest)
    input_bindings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_inputs: set[Path] = set()
    for path_ref in input_indexes:
        path = resolve_project_path(path_ref) or Path(path_ref)
        if not path.is_file():
            raise FileNotFoundError(f"OOF prediction index 不存在: {path}")
        resolved = path.resolve(strict=False)
        if resolved in seen_inputs:
            raise ValueError(f"OOF merge 重复输入 index: {path}")
        seen_inputs.add(resolved)
        input_rows = _read_jsonl(path)
        if not input_rows:
            raise ValueError(f"OOF prediction index 不能为空: {path}")
        _revalidate_prediction_publication_directory(
            path, input_rows, split="train"
        )
        rows.extend(input_rows)
        input_bindings.append({
            "path": to_project_ref(path),
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "num_records": len(input_rows),
        })
    audit = _audit_oof_prediction_rows(
        rows, manifest=manifest, fold_manifest=fold_manifest
    )
    output_path = resolve_project_path(output) or Path(output)
    protected = {
        "fold-manifest": fold_manifest,
        **{
            f"input-{index}": path
            for index, path in enumerate(input_indexes)
        },
    }
    validate_output_replacement_safety(output_path, protected)
    validate_output_replacement_safety(
        output_path.with_suffix(".report.json"), protected
    )
    _atomic_jsonl(output_path, audit["rows"])
    report = {
        "protocol": OOF_MERGE_PROTOCOL,
        "fold_manifest_artifact": manifest_binding,
        "expert_gate_audit": manifest["expert_gate_audit"],
        "num_folds": int(manifest["num_folds"]),
        "num_parents": audit["num_parents"],
        "parents_per_fold": audit["parents_per_fold"],
        "num_unique_masks": audit["num_unique_masks"],
        "mask_bytes": audit["mask_bytes"],
        "mask_inventory_sha256": audit["mask_inventory_sha256"],
        "population_sha256": audit["population_sha256"],
        "fold_checkpoint_audits": audit["fold_checkpoint_audits"],
        "inputs": input_bindings,
        "output": to_project_ref(output_path),
        "output_sha256": _sha256(output_path),
        "output_bytes": int(output_path.stat().st_size),
    }
    _atomic_json(output_path.with_suffix(".report.json"), report)
    return report


def revalidate_oof_merged_index(
    index_path: str | Path,
    *,
    expected_expert_gate_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a published train OOF index before D4/M7 consumes it."""
    path = resolve_project_path(index_path) or Path(index_path)
    report_path = path.with_suffix(".report.json")
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError("OOF merged index/report 缺失")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol") != OOF_MERGE_PROTOCOL:
        raise ValueError(
            f"OOF merge report protocol 非法: {report.get('protocol')!r}"
        )
    reported_output = resolve_project_path(str(report.get("output") or ""))
    if (
        reported_output is None
        or reported_output.resolve(strict=False) != path.resolve(strict=False)
        or str(report.get("output_sha256") or "") != _sha256(path)
        or int(report.get("output_bytes", -1)) != int(path.stat().st_size)
    ):
        raise ValueError("OOF merged index path/hash/bytes 已漂移")
    manifest_binding = report.get("fold_manifest_artifact")
    if not isinstance(manifest_binding, dict):
        raise ValueError("OOF merge report 缺少 fold manifest artifact binding")
    manifest_path = resolve_project_path(str(manifest_binding.get("manifest") or ""))
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError("OOF merge report 绑定的 fold manifest 缺失")
    current_manifest_binding = oof_manifest_artifact_binding(manifest_path)
    if manifest_binding != current_manifest_binding:
        raise ValueError("OOF fold manifest artifact 在 merge 后发生变化")
    manifest = load_oof_manifest(manifest_path)
    if (
        report.get("expert_gate_audit") != manifest["expert_gate_audit"]
        or (
            expected_expert_gate_audit is not None
            and report.get("expert_gate_audit") != expected_expert_gate_audit
        )
    ):
        raise ValueError("OOF merge report 与当前 frozen expert gate 不一致")
    inputs = report.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("OOF merge report 缺少输入 index bindings")
    for binding in inputs:
        if not isinstance(binding, dict):
            raise ValueError("OOF merge input binding 非法")
        input_path = resolve_project_path(str(binding.get("path") or ""))
        if (
            input_path is None
            or not input_path.is_file()
            or _sha256(input_path) != str(binding.get("sha256") or "")
            or int(input_path.stat().st_size) != int(binding.get("bytes", -1))
            or len(_read_jsonl(input_path)) != int(binding.get("num_records", -1))
        ):
            raise ValueError("OOF merge 的 fold prediction input 已漂移")
        _revalidate_prediction_publication_directory(
            input_path, _read_jsonl(input_path), split="train"
        )
    audit = _audit_oof_prediction_rows(
        _read_jsonl(path), manifest=manifest, fold_manifest=manifest_path
    )
    expected = {
        "num_folds": int(manifest["num_folds"]),
        "num_parents": audit["num_parents"],
        "parents_per_fold": audit["parents_per_fold"],
        "num_unique_masks": audit["num_unique_masks"],
        "mask_bytes": audit["mask_bytes"],
        "mask_inventory_sha256": audit["mask_inventory_sha256"],
        "population_sha256": audit["population_sha256"],
        "fold_checkpoint_audits": audit["fold_checkpoint_audits"],
    }
    observed = {name: report.get(name) for name in expected}
    if observed != expected:
        raise ValueError("OOF merge report 统计/checkpoint provenance 未通过当前重放")
    return {
        "protocol": OOF_MERGE_PROTOCOL,
        "report": to_project_ref(report_path),
        "report_sha256": _sha256(report_path),
        "index": to_project_ref(path),
        "index_sha256": _sha256(path),
        "fold_manifest_artifact": current_manifest_binding,
        "fold_checkpoint_audits": audit["fold_checkpoint_audits"],
        "num_parents": audit["num_parents"],
        "population_sha256": audit["population_sha256"],
        "mask_inventory_sha256": audit["mask_inventory_sha256"],
    }
