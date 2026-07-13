#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 eval proposal records 中确定性选择 PPT 候选样本。"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Any


def _positive(record: dict[str, Any]) -> bool:
    return float(record.get("target_area", record.get("target_area_px", 0)) or 0) > 0


def _stratum(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("dataset_name") or "unknown"),
        str(record.get("family_combo") or "unknown"),
        str(record.get("task_family") or "unknown"),
        "positive" if _positive(record) else "negative",
    )


def _hamming(left: str, right: str) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a != b for a, b in zip(left, right)) / max(len(left), 1)


def select_gallery_records(
    records: list[dict[str, Any]],
    *,
    max_items: int = 120,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """选择强/典型/失败、负样本、指令对和弱模态案例。"""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if isinstance(record, dict) and record.get("sample_id"):
            grouped[_stratum(record)].append(record)

    candidates: list[tuple[int, str, str, dict[str, Any]]] = []
    for key, values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: (float(item.get("final_dice", 0)), str(item["sample_id"])))
        stratum = " | ".join(key)
        if key[-1] == "positive":
            picks = (
                ("failure", values[0]),
                ("typical", values[len(values) // 2]),
                ("strong", values[-1]),
            )
            for category, record in picks:
                priority = {"strong": 10, "typical": 20, "failure": 30}[category]
                candidates.append((priority, category, stratum, record))
        else:
            candidates.append((40, "negative_correct", stratum, values[-1]))
            if float(values[0].get("final_dice", 1.0)) < 1.0:
                candidates.append((41, "negative_false_positive", stratum, values[0]))

    # 同一 parent 的 referring targets 用于展示语言条件改变 mask。
    referring: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("task_family") == "referring_landslide_segmentation":
            referring[str(record.get("parent_sample_id") or record.get("sample_id"))].append(record)
    pair_candidates = []
    for parent, values in referring.items():
        if len(values) < 2:
            continue
        best = None
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                contrast = _hamming(str(left.get("target_signature_16") or ""), str(right.get("target_signature_16") or ""))
                key = (contrast, str(left.get("sample_id")), str(right.get("sample_id")))
                if best is None or key > best[0]:
                    best = (key, left, right)
        if best is not None and best[0][0] > 0:
            pair_candidates.append((best[0][0], parent, best[1], best[2]))
    for _, parent, left, right in sorted(pair_candidates, reverse=True)[:6]:
        candidates.append((5, "instruction_pair", f"parent={parent}", left))
        candidates.append((5, "instruction_pair", f"parent={parent}", right))

    # 给 Sen12 与小目标显式标签，确保弱点不会被高分光学样本淹没，同时限制专题数量。
    sen12 = sorted(
        (record for record in records if _positive(record) and record.get("dataset_name") == "Sen12Landslides"),
        key=lambda item: (float(item.get("final_dice", 0)), str(item.get("sample_id"))),
    )
    for record in _low_median_high(sen12):
        candidates.append((35, "weak_modality_sen12", "Sen12Landslides", record))
    for area_bin in ("tiny_le_16px", "small_17_64px"):
        values = sorted(
            (record for record in records if _positive(record) and record.get("target_area_px_bin") == area_bin),
            key=lambda item: (float(item.get("final_dice", 0)), str(item.get("sample_id"))),
        )
        for record in _low_median_high(values):
            candidates.append((36, "small_target", area_bin, record))

    rng = random.Random(seed)
    candidates.sort(key=lambda item: (item[0], item[1], item[2], str(item[3].get("sample_id"))))
    # 同优先级内稳定扰动，避免永远只取索引头部的数据源位置。
    blocks: dict[int, list[tuple[int, str, str, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        blocks[candidate[0]].append(candidate)
    ordered = []
    for priority in sorted(blocks):
        block = blocks[priority]
        rng.shuffle(block)
        ordered.extend(block)

    selected: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_parent_task: set[tuple[str, str]] = set()
    sample_to_selected: dict[str, dict[str, Any]] = {}
    parent_task_to_selected: dict[tuple[str, str], dict[str, Any]] = {}
    for _, category, stratum, record in ordered:
        sample_id = str(record.get("sample_id"))
        parent_task = (
            str(record.get("parent_sample_id") or sample_id),
            str(record.get("task_family") or "unknown"),
        )
        is_pair = category == "instruction_pair"
        if sample_id in seen_samples:
            existing = sample_to_selected[sample_id]
            existing.setdefault("gallery_tags", []).append(category)
            continue
        if parent_task in seen_parent_task and not is_pair:
            existing = parent_task_to_selected[parent_task]
            existing.setdefault("gallery_tags", []).append(category)
            continue
        output = {
            **record,
            "gallery_category": category,
            "gallery_tags": [category],
            "gallery_stratum": stratum,
        }
        selected.append(output)
        seen_samples.add(sample_id)
        sample_to_selected[sample_id] = output
        if not is_pair:
            seen_parent_task.add(parent_task)
            parent_task_to_selected[parent_task] = output
        if len(selected) >= max(1, int(max_items)):
            break
    return selected


def _low_median_high(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not values:
        return []
    return [values[index] for index in sorted({0, len(values) // 2, len(values) - 1})]
