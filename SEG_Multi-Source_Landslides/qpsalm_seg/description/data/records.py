"""Task-row projection and evaluation-population contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..protocols.output import canonical_description_json
from .sampling import description_row_sample_id, stable_subset


def append_fraction(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    fraction: float,
    *,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    """Keep every primary row and add a deterministic secondary fraction."""
    if not primary or fraction <= 0 or not secondary:
        return list(primary)
    requested = round(len(primary) * float(fraction) / max(1.0 - float(fraction), 1.0e-8))
    return list(primary) + stable_subset(secondary, requested, seed, namespace)


def structured_text(record: dict[str, Any], *, expert: bool) -> str:
    if expert:
        target = record.get("expert_target") or {}
        structured = dict(target.get("structured_output") or {})
        summary = str(target.get("summary") or "")
    else:
        candidate = record.get("candidate") or {}
        structured = dict(candidate.get("structured_output") or {})
        summary = str(candidate.get("summary") or "")
    if not summary.strip():
        raise ValueError(
            "Bridge structured target 缺少非空 summary: "
            f"{description_row_sample_id(record)}"
        )
    output = {
        "schema_version": "qpsalm_description_output_v1",
        "target_status": structured.get("target_status", record.get("target_status", "uncertain")),
        "region": structured.get("region") or {},
        "evidence": structured.get("evidence") or {},
        "summary": summary,
    }
    try:
        text = canonical_description_json(output)
    except ValueError as exc:
        raise ValueError(
            "Bridge structured target 不符合 qpsalm_description_output_v1: "
            f"sample={description_row_sample_id(record)} errors={exc}"
        ) from exc
    return text


def has_unavailable_modality(row: dict[str, Any]) -> bool:
    evidence = row.get("modality_evidence") or {}
    if isinstance(evidence, dict):
        values = evidence.values()
    elif isinstance(evidence, (list, tuple)):
        values = evidence
    else:
        return False
    for value in values:
        if not isinstance(value, dict):
            continue
        level = str(value.get("evidence_level") or "")
        status = str(value.get("status") or value.get("availability") or "")
        if level.startswith("C_") or status in {"unavailable", "insufficient"}:
            return True
    return False


def bridge_region_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Build target identity metadata without loading cache features or masks."""
    review_responses = ((row.get("review") or {}).get("reviewer_responses") or [])
    panel_paths = sorted({
        str(value.get("panel_path"))
        for value in review_responses
        if isinstance(value, dict) and value.get("panel_path")
    })
    preview_paths = (row.get("visual_ref") or {}).get("preview_paths") or {}
    return {
        "sample_id": str(row["bridge_record_id"]),
        "parent_sample_id": str(row["parent_sample_id"]),
        "task_family": str(row["task_family"]),
        "target_status": str(row.get("target_status") or "uncertain"),
        "source_dataset": str(row.get("dataset_name") or "unknown"),
        "region_pair_id": None,
        "region_id": str(row.get("region_id") or "unknown"),
        "region_source": str(row.get("region_source") or "unknown"),
        "source_region_aliases": [
            dict(value) for value in (row.get("source_region_aliases") or [])
            if isinstance(value, dict)
        ],
        "region_mask_path": (row.get("region_mask") or {}).get("path"),
        "expert_review_panel_path": panel_paths[0] if panel_paths else None,
        "visual_preview_path": preview_paths.get("visual"),
        "multimodal_preview_path": preview_paths.get("modalities"),
        "has_unavailable_modality": has_unavailable_modality(row),
    }


def filter_evaluation_source(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    split: str,
    source_dataset: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Freeze a source-specific independent test population before limiting."""
    if source_dataset is None:
        return rows, None
    if stage != "rsicap_caption" or split != "test" or source_dataset != "RSIEval":
        raise ValueError(
            "source-specific description evaluation 只允许 "
            "stage=rsicap_caption split=test source=RSIEval"
        )
    selected = [
        row for row in rows
        if str(row.get("source_dataset") or "") == source_dataset
        and str(row.get("task_family") or "") == "global_caption"
    ]
    if not selected:
        raise ValueError("RSIEval source filter 产生空 evaluation population")
    return selected, {
        "protocol": "qpsalm_description_evaluation_source_filter_v1",
        "stage": stage,
        "split": split,
        "source_dataset": source_dataset,
        "rows_before_filter": len(rows),
        "rows_after_filter": len(selected),
    }


def evaluation_region_source_population_sha256(
    rows: list[dict[str, Any]],
) -> str:
    """Hash the exact region identity population selected before eval limiting."""
    identities = sorted(
        (
            str(row.get("sample_id") or description_row_sample_id(row)),
            str(row.get("parent_sample_id") or ""),
            str(row.get("region_id") or ""),
            str(row.get("region_source") or ""),
        )
        for row in rows
    )
    return hashlib.sha256(
        json.dumps(
            identities,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def filter_evaluation_region_source(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    split: str,
    training: bool,
    evaluation_mode: str,
    region_source: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Freeze the GT-global-mask oracle population before sample limiting."""
    if region_source is None:
        return rows, None
    if (
        training
        or stage != "bridge_expert"
        or split not in {"val", "test"}
        or evaluation_mode not in {"gt_mask", "end_to_end"}
        or region_source != "gt_global_mask"
    ):
        raise ValueError(
            "region-source filter 只允许 frozen bridge_expert val/test 的 "
            "GT-mask/end-to-end gt_global_mask"
        )
    selected = [
        row for row in rows
        if str(row.get("region_source") or "") == region_source
    ]
    if not selected:
        raise ValueError("region-source filter 产生空 evaluation population")
    sample_ids = [description_row_sample_id(row) for row in selected]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("region-source filter 后 sample_id 必须唯一")
    return selected, {
        "protocol": "qpsalm_description_region_source_filter_v1",
        "stage": stage,
        "split": split,
        "evaluation_mode": evaluation_mode,
        "region_source": region_source,
        "rows_before_filter": len(rows),
        "rows_after_filter": len(selected),
        "excluded_rows": len(rows) - len(selected),
        "population_sha256": evaluation_region_source_population_sha256(selected),
    }

