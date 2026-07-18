#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact OOF fold-index and segmentation-checkpoint bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT
from qpsalm_seg.paths import resolve_project_path, to_project_ref

from .oof import load_oof_manifest, oof_manifest_artifact_binding
from .predicted_contracts import OOF_CHECKPOINT_BINDING_PROTOCOL
from ..protocols.io import sha256_file


def _checkpoint_index_fingerprint(
    *,
    fingerprint: Any,
    expected_path: Path,
    split: str,
) -> dict[str, Any]:
    if not isinstance(fingerprint, dict):
        raise ValueError(f"OOF checkpoint Vision Cache v3 缺少 {split} index fingerprint")
    reference = resolve_project_path(str(fingerprint.get("reference") or ""))
    expected_hash = sha256_file(expected_path)
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
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_step": step,
        "config_train_index": to_project_ref(train_path),
        "config_val_index": to_project_ref(val_path),
        "vision_cache_index_fingerprints": cache_indexes,
    }


def validate_oof_checkpoint(
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
    if sha256_file(expected_path) != expected_hash:
        raise ValueError(f"OOF fold train index 已变化: fold={checkpoint_fold}")
    holdout_path = resolve_project_path(str(fold.get("holdout_index") or ""))
    prediction_path = resolve_project_path(prediction_index) or Path(prediction_index)
    expected_holdout_hash = str(fold.get("holdout_index_sha256") or "")
    if holdout_path is None or not holdout_path.is_file():
        raise FileNotFoundError(f"OOF fold holdout index 不存在: {fold.get('holdout_index')}")
    if sha256_file(holdout_path) != expected_holdout_hash:
        raise ValueError(f"OOF fold holdout index 已变化: fold={checkpoint_fold}")
    if (
        not prediction_path.is_file()
        or prediction_path.resolve(strict=False) != holdout_path.resolve(strict=False)
        or sha256_file(prediction_path) != expected_holdout_hash
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
    return validate_oof_checkpoint(
        checkpoint,
        manifest,
        str(checkpoint_fold),
        prediction_index,
        fold_manifest=fold_manifest,
    )
