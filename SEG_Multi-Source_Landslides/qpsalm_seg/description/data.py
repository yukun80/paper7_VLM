#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-aware datasets for global, region-alignment and Landslide Bridge description."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import MODALITY_FAMILY_IDS

from .backbone import transform_region_mask_to_cache
from .vision_cache import DescriptionVisionFeatureBank


DescriptionStage = Literal[
    "overfit", "mmrs_caption", "rsicap_caption", "dior_alignment",
    "bridge_auto", "bridge_expert", "predicted_mask",
]


def description_row_sample_id(row: dict[str, Any]) -> str:
    """Return the stable task identity used by evaluation metadata."""
    value = row.get("sample_id") or row.get("bridge_record_id")
    if not value:
        raise ValueError("description row 缺少 sample_id/bridge_record_id")
    return str(value)


def same_parent_region_swap_candidates(
    rows: list[dict[str, Any]],
    sample_id: str,
    *,
    catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic real-region candidates from the same parent.

    The alternate catalog may contain unreviewed Bridge region geometry, but
    its text target is never consumed.  Null/no-target rows are not regions and
    therefore cannot be used as a region-swap shortcut.
    """
    by_id = {description_row_sample_id(row): row for row in rows}
    current = by_id.get(str(sample_id))
    if current is None:
        return []
    parent = str(current.get("parent_sample_id") or "")
    current_region_id = str(current.get("region_id") or "")
    current_mask = (current.get("region_mask") or {}).get("path")
    candidates = []
    for row in catalog if catalog is not None else rows:
        candidate_id = description_row_sample_id(row)
        if candidate_id == str(sample_id):
            continue
        if str(row.get("parent_sample_id") or "") != parent:
            continue
        geometry = row.get("region_geometry") or {}
        has_box = geometry.get("type") == "box"
        has_mask = bool((row.get("region_mask") or {}).get("path"))
        if not (has_box or has_mask):
            continue
        if str(row.get("target_status") or "present") != "present":
            continue
        candidate_region_id = str(row.get("region_id") or "")
        candidate_mask = (row.get("region_mask") or {}).get("path")
        if (
            candidate_region_id
            and candidate_region_id == current_region_id
            and candidate_mask == current_mask
        ):
            continue
        candidates.append(row)
    return sorted(
        candidates,
        key=lambda row: (
            str(row.get("region_source") or "") == str(current.get("region_source") or ""),
            str(row.get("region_id") or ""),
            description_row_sample_id(row),
        ),
    )


def end_to_end_region_support(row: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a Bridge row has an identifiable segmentation target.

    Global masks always map to the global segmentation instruction. Referring
    masks and pseudo components are valid only when inventory deduplication
    attached at least one referring-target alias. A pseudo component without
    such an alias has no language target for the segmentation model and must
    not silently fall back to whole-image segmentation.
    """
    source = str(row.get("region_source") or "unknown")
    if source == "gt_global_mask":
        return True, "global_instruction"
    aliases = [
        value for value in (row.get("source_region_aliases") or [])
        if isinstance(value, dict) and value.get("sample_id")
    ]
    if source == "gt_referring_mask":
        return (bool(aliases), "referring_alias" if aliases else "missing_referring_alias")
    if source == "pseudo_instance_component":
        return (
            bool(aliases),
            "component_with_referring_alias" if aliases else "component_without_language_target",
        )
    if source == "no_target":
        # Empty parents can map to an empty global instruction even when there
        # is no explicit no-target referring alias. The resolver verifies this.
        return True, "no_target_alias_or_empty_global"
    return False, f"unsupported_region_source:{source}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 非法 JSONL") from exc
    return rows


def _stable_index(seed: int, epoch: int, sample_id: str, length: int) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).hexdigest()
    return int(digest[:16], 16) % max(length, 1)


def _stable_subset(rows: list[dict[str, Any]], count: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{namespace}:{row.get('sample_id') or row.get('bridge_record_id')}".encode()
        ).hexdigest(),
    )
    return ranked[:max(0, int(count))]


def _append_fraction(
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
    return list(primary) + _stable_subset(secondary, requested, seed, namespace)


def _structured_text(record: dict[str, Any], *, expert: bool) -> str:
    if expert:
        target = record.get("expert_target") or {}
        structured = dict(target.get("structured_output") or {})
        summary = str(target.get("summary") or "")
    else:
        candidate = record.get("candidate") or {}
        structured = dict(candidate.get("structured_output") or {})
        summary = str(candidate.get("summary") or "")
    output = {
        "schema_version": "qpsalm_description_output_v1",
        "target_status": structured.get("target_status", record.get("target_status", "uncertain")),
        "region": structured.get("region") or {},
        "evidence": structured.get("evidence") or {},
        "summary": summary,
    }
    return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    }


class DescriptionTaskDataset(Dataset):
    """One task family per dataset instance; joint training uses separate DataLoaders."""

    def __init__(
        self,
        *,
        stage: DescriptionStage,
        split: str,
        vision_bank: DescriptionVisionFeatureBank,
        description_benchmark: str | Path,
        bridge_benchmark: str | Path,
        predicted_index: str | Path | None = None,
        seed: int = 42,
        max_samples: int = 0,
        training: bool = False,
        evaluation_mode: str = "gt_mask",
        rsicap_mmrs_fraction: float = 0.30,
        predicted_mask_fraction: float = 0.25,
    ) -> None:
        self.stage = stage
        self.split = split
        self.vision_bank = vision_bank
        self.seed = int(seed)
        self.epoch = 0
        self.training = bool(training)
        self.evaluation_mode = str(evaluation_mode)
        self.end_to_end_exclusion_counts: Counter[str] = Counter()
        self.end_to_end_source_count = 0
        self.end_to_end_eligible_count = 0
        description_dir = resolve_project_path(description_benchmark)
        bridge_dir = resolve_project_path(bridge_benchmark)
        if description_dir is None or bridge_dir is None:
            raise ValueError("description/bridge benchmark 路径不能为空")
        if stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}:
            index_name = "train_eligible.jsonl" if split == "train" else f"{split}.jsonl"
            rows = _read_jsonl(description_dir / f"indexes/{index_name}")
            if stage == "mmrs_caption":
                rows = [
                    row for row in rows
                    if row["task_family"] == "global_caption" and str(row["source_dataset"]).startswith("MMRS-")
                ]
            elif stage == "rsicap_caption":
                rsicap_rows = [
                    row for row in rows
                    if row["task_family"] == "global_caption" and row["source_dataset"] in {"RSICap", "RSIEval"}
                ]
                if self.training and split == "train":
                    mmrs_rows = [
                        row for row in _read_jsonl(description_dir / "indexes/train_eligible.jsonl")
                        if row["task_family"] == "global_caption"
                        and str(row["source_dataset"]).startswith("MMRS-")
                    ]
                    rows = _append_fraction(
                        rsicap_rows, mmrs_rows, rsicap_mmrs_fraction,
                        seed=self.seed, namespace="d1_rsicap_mmrs",
                    )
                else:
                    rows = rsicap_rows
            else:
                rows = [row for row in rows if row["task_family"] == "region_referring_expression"]
        elif stage in {"bridge_auto", "overfit"}:
            rows = _read_jsonl(bridge_dir / "indexes/auto_train.jsonl")
            if stage == "bridge_auto" and split != "train":
                rows = []
        elif stage == "bridge_expert":
            path = bridge_dir / f"indexes/expert_{split}.jsonl"
            rows = _read_jsonl(path)
        elif stage == "predicted_mask":
            if predicted_index is None:
                raise ValueError("predicted_mask stage 需要独立离线 --predicted-index")
            path = resolve_project_path(predicted_index)
            rows = _read_jsonl(path)
            rows = [row for row in rows if row.get("split") == split]
            if self.training and split == "train":
                expert_path = bridge_dir / "indexes/expert_train.jsonl"
                if not expert_path.is_file():
                    raise FileNotFoundError(
                        "D4 GT/predicted curriculum 需要已冻结 indexes/expert_train.jsonl"
                    )
                expert_rows = _read_jsonl(expert_path)
                requested_predicted = round(
                    len(expert_rows) * float(predicted_mask_fraction)
                    / max(1.0 - float(predicted_mask_fraction), 1.0e-8)
                )
                rows = expert_rows + _stable_subset(
                    rows, requested_predicted, self.seed, "d4_predicted_masks"
                )
        else:
            raise ValueError(f"未知 description stage={stage!r}")
        if self.evaluation_mode == "end_to_end":
            if stage != "bridge_expert":
                raise ValueError("end_to_end evaluation 只支持 bridge_expert stage")
            self.end_to_end_source_count = len(rows)
            supported_rows = []
            for row in rows:
                supported, reason = end_to_end_region_support(row)
                if supported:
                    supported_rows.append(row)
                else:
                    self.end_to_end_exclusion_counts[reason] += 1
            rows = supported_rows
            self.end_to_end_eligible_count = len(rows)
        rows.sort(key=lambda row: str(
            row.get("sample_id") or row.get("bridge_record_id") or ""
        ))
        if stage == "overfit":
            rows = rows[: min(64, len(rows))]
        if max_samples > 0:
            rows = _stable_subset(rows, max_samples, self.seed, f"{stage}:{split}:limit")
            rows.sort(key=lambda row: str(
                row.get("sample_id") or row.get("bridge_record_id") or ""
            ))
        self.rows = rows
        self._rows_by_sample_id: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            sample_id = description_row_sample_id(row)
            if sample_id in self._rows_by_sample_id:
                raise ValueError(f"description dataset sample_id 重复: {sample_id}")
            self._rows_by_sample_id[sample_id] = row
        self._region_swap_catalog = self.rows
        if stage in {"bridge_auto", "bridge_expert", "predicted_mask", "overfit"}:
            candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
            if candidate_path.is_file():
                self._region_swap_catalog = [
                    row for row in _read_jsonl(candidate_path)
                    if str(row.get("split") or "") == self.split
                ]
        self._request_family_cache: dict[tuple[str, str], set[str]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def same_parent_region_swap(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Load and identify another real region from the same image."""
        candidates = same_parent_region_swap_candidates(
            self.rows,
            sample_id,
            catalog=self._region_swap_catalog,
        )
        current = reference_mask.detach().cpu()
        for row in candidates:
            item = (
                self._description_item(row)
                if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
                else self._bridge_item(row)
            )
            alternate = item["region_mask"].detach().cpu()
            if alternate.shape != current.shape:
                continue
            if not torch.equal(alternate, current):
                return alternate, {
                    "protocol": "qpsalm_same_parent_region_swap_v1",
                    "parent_sample_id": str(row["parent_sample_id"]),
                    "alternate_sample_id": description_row_sample_id(row),
                    "alternate_region_id": str(row.get("region_id") or "unknown"),
                    "alternate_region_source": str(
                        row.get("region_source") or "region_geometry"
                    ),
                    "alternate_mask_path": (row.get("region_mask") or {}).get("path"),
                }
        return None

    def same_parent_region_swap_mask(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compatibility convenience returning only the resolved mask."""
        resolved = self.same_parent_region_swap(sample_id, reference_mask)
        return resolved[0] if resolved is not None else None

    def _request_for_row(self, row: dict[str, Any]) -> tuple[str, str]:
        component = (
            "single_image"
            if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
            else "multisource_parent"
        )
        return component, str(row["parent_sample_id"])

    def _request_families(self, request: tuple[str, str]) -> set[str]:
        cached = self._request_family_cache.get(request)
        if cached is not None:
            return cached
        record = self.vision_bank.record(*request)
        families = set()
        for view in record["views"]:
            source = [str(value) for value in view.get("source_families") or []]
            family = (
                source[0]
                if source and len(set(source)) == 1 and source[0] in MODALITY_FAMILY_IDS
                else "unknown"
            )
            families.add(family)
        self._request_family_cache[request] = families
        return families

    def cross_parent_modality_swap_request(
        self,
        sample_id: str,
    ) -> tuple[tuple[str, str], dict[str, Any]] | None:
        """Select a deterministic donor parent sharing at least one view family."""
        current = self._rows_by_sample_id.get(str(sample_id))
        if current is None:
            return None
        current_parent = str(current["parent_sample_id"])
        current_request = self._request_for_row(current)
        current_families = self._request_families(current_request)
        seen_parents = {current_parent}
        for row in sorted(self.rows, key=description_row_sample_id):
            donor_parent = str(row.get("parent_sample_id") or "")
            if not donor_parent or donor_parent in seen_parents:
                continue
            seen_parents.add(donor_parent)
            donor_request = self._request_for_row(row)
            common = sorted(current_families & self._request_families(donor_request))
            if not common:
                continue
            return donor_request, {
                "protocol": "qpsalm_cross_parent_modality_donor_v1",
                "target_parent_sample_id": current_parent,
                "donor_parent_sample_id": donor_parent,
                "common_modality_families": common,
            }
        return None

    def _description_item(self, row: dict[str, Any]) -> dict[str, Any]:
        answers = [
            answer for answer in row.get("answers", [])
            if self.split != "train" or float(answer.get("caption_quality_weight", 1.0)) > 0
        ]
        if not answers:
            raise ValueError(f"description record 没有可训练 answer: {row['sample_id']}")
        answer = answers[_stable_index(self.seed, self.epoch, str(row["sample_id"]), len(answers))]
        visual = row["visual_ref"]
        width, height = int(visual["width"]), int(visual["height"])
        geometry = row["region_geometry"]
        source_mask = torch.zeros((1, height, width), dtype=torch.float32)
        if geometry["type"] == "full_image":
            source_mask.fill_(1.0)
        elif geometry["type"] == "box":
            x1, y1, x2, y2 = [int(value) for value in geometry["bbox_xyxy_pixel_half_open"]]
            source_mask[:, y1:y2, x1:x2] = 1.0
        elif geometry["type"] not in {"null"}:
            raise ValueError(f"M1 record region type 暂不支持: {geometry['type']}")
        cache = self.vision_bank.record("single_image", str(row["parent_sample_id"]))
        region = transform_region_mask_to_cache(source_mask, cache["views"][0]["render_transform"])
        return {
            "request": ("single_image", str(row["parent_sample_id"])),
            "region_mask": region,
            "instruction": str(row["instruction"]),
            "target_text": str(answer["text"]),
            "reference_texts": [str(value["text"]) for value in answers],
            "structured_output": False,
            "weight": float(answer.get("caption_quality_weight", 1.0)),
            "sample_id": str(row["sample_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "task_family": str(row["task_family"]),
            "target_status": str(row.get("target_status") or "present"),
            "source_dataset": str(row.get("source_dataset") or "unknown"),
            "region_pair_id": row.get("region_pair_id"),
        }

    def _bridge_item(self, row: dict[str, Any]) -> dict[str, Any]:
        cache = self.vision_bank.record("multisource_parent", str(row["parent_sample_id"]))
        transform = cache["views"][0]["render_transform"]
        if row.get("region_mask"):
            path = resolve_project_path(row["region_mask"]["path"])
            values = np.load(path)
            if values.ndim == 2:
                values = values[None]
            source_mask = torch.from_numpy((values > 0).astype(np.float32))
        else:
            source_h = int(transform["source_h"])
            source_w = int(transform["source_w"])
            source_mask = torch.zeros((1, source_h, source_w), dtype=torch.float32)
        region = transform_region_mask_to_cache(source_mask, transform)
        # Predicted-mask rows inherit the reviewed target from their source row.
        # Falling back to the deterministic candidate here would make fixed-mask
        # evaluation measure a different target than the GT-mask expert run.
        expert = self.stage == "bridge_expert" or isinstance(row.get("expert_target"), dict)
        return {
            "request": ("multisource_parent", str(row["parent_sample_id"])),
            "region_mask": region,
            "instruction": str(row["instruction"]),
            "target_text": _structured_text(row, expert=expert),
            "reference_texts": [_structured_text(row, expert=expert)],
            "structured_output": True,
            "weight": 1.0,
            **bridge_region_metadata(row),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return (
            self._description_item(row)
            if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
            else self._bridge_item(row)
        )


def collate_description(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("description batch 不能为空")
    shapes = {tuple(item["region_mask"].shape) for item in items}
    if len(shapes) != 1:
        raise ValueError(f"description region canvas 必须一致: {sorted(shapes)}")
    return {
        "requests": [item["request"] for item in items],
        "region_masks": torch.stack([item["region_mask"] for item in items]),
        "instructions": [item["instruction"] for item in items],
        "target_texts": [item["target_text"] for item in items],
        "reference_texts": [item["reference_texts"] for item in items],
        "structured_outputs": [bool(item["structured_output"]) for item in items],
        "weights": torch.tensor([float(item["weight"]) for item in items], dtype=torch.float32),
        "metadata": [{
            key: item[key]
            for key in (
                "sample_id", "parent_sample_id", "task_family", "target_status",
                "source_dataset", "region_pair_id", "region_id", "region_source",
                "source_region_aliases", "region_mask_path", "expert_review_panel_path",
                "visual_preview_path", "multimodal_preview_path",
            )
            if key in item
        } for item in items],
    }
