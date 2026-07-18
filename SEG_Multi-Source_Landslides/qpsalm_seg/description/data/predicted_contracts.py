#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predicted-region protocol identifiers and deterministic artifact I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..protocols.io import (
    atomic_write_json,
    atomic_write_jsonl,
    strict_json_loads,
)


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


def read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        strict_json_loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write_prediction_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Preserve the established insertion-order JSONL byte contract."""
    atomic_write_jsonl(path, rows, sort_keys=False)


def atomic_write_prediction_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    atomic_write_json(path, payload)
