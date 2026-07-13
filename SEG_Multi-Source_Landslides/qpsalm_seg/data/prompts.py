#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-free natural-language prompts derived only from the active modality subset."""

from __future__ import annotations

import copy
import math
from typing import Any

from qpsalm_seg.schema import ModalityInstance


PROMPT_VERSION = "qpsalm_prompt_v3"
GRID_POSITIONS = (
    ("upper-left", "upper", "upper-right"),
    ("left", "center", "right"),
    ("lower-left", "lower", "lower-right"),
)
GRID_COORDINATES = {
    name: (row, column)
    for row, values in enumerate(GRID_POSITIONS)
    for column, name in enumerate(values)
}


def flip_grid_position(value: str, *, hflip: bool, vflip: bool) -> str:
    if value not in GRID_COORDINATES:
        return value
    row, column = GRID_COORDINATES[value]
    if hflip:
        column = 2 - column
    if vflip:
        row = 2 - row
    return GRID_POSITIONS[row][column]


def transform_spatial_instruction(row: dict[str, Any], *, hflip: bool, vflip: bool) -> dict[str, Any]:
    if not hflip and not vflip:
        return row
    target = row.get("referring_target") or {}
    category = str(target.get("category") or "")
    grounding = target.get("grounding") or {}
    grid = str(grounding.get("grid") or "")
    if category not in {"position", "no_target"} or grid not in GRID_COORDINATES:
        return row
    transformed = copy.deepcopy(row)
    target = transformed["referring_target"]
    new_grid = flip_grid_position(grid, hflip=hflip, vflip=vflip)
    target.setdefault("grounding", {})["grid"] = new_grid
    target["grounding"]["augmentation_source_grid"] = grid
    target["subtype"] = f"position_{new_grid}" if category == "no_target" else new_grid
    instruction = dict(transformed.get("instruction") or {})
    position = new_grid.replace("-", " ")
    if category == "no_target":
        instruction["text"] = (
            f"Segment landslide regions in the {position} part of the image. "
            "If none are present there, output an empty mask."
        )
    else:
        instruction["text"] = f"Segment the landslide regions in the {position} part of the image."
    zh = {
        "upper-left": "左上部", "upper": "上部", "upper-right": "右上部",
        "left": "左侧", "center": "中心", "right": "右侧",
        "lower-left": "左下部", "lower": "下部", "lower-right": "右下部",
    }[new_grid]
    instruction["text_zh"] = (
        f"分割图像{zh}区域内的滑坡；如果该区域不存在滑坡，输出空掩膜。"
        if category == "no_target" else f"分割图像{zh}的滑坡区域。"
    )
    transformed["instruction"] = instruction
    return transformed


def gsd_description(value: Any) -> str:
    try:
        gsd = float(value)
    except (TypeError, ValueError):
        return "unknown ground sampling distance"
    if not math.isfinite(gsd) or gsd <= 0:
        return "unknown ground sampling distance"
    if gsd < 1:
        return f"approximately {gsd:g} meter sub-meter imagery"
    return f"approximately {gsd:g} meters per pixel"


def instruction_text(row: dict[str, Any]) -> str:
    return str((row.get("instruction") or {}).get("text") or "Segment all landslide regions.")


def condition_text(row: dict[str, Any]) -> str:
    family = str(row.get("task_family") or "global_landslide_segmentation")
    template = str(row.get("template_id") or "")
    if family == "no_target_segmentation":
        return "the landslide target described by the instruction, which may be absent"
    if family == "referring_landslide_segmentation":
        return "the landslide region selected by the spatial, scale, morphology, or count condition"
    if family == "multisource_evidence_segmentation":
        if template in {"deformation_evidence_landslide_v2", "insar_evidence_landslide_v2"}:
            return "landslide regions supported by deformation, terrain, and optical evidence"
        if template == "sar_terrain_landslide_v2":
            return "landslide regions supported by SAR, terrain, and multispectral evidence"
        if template == "terrain_evidence_landslide_v2":
            return "landslide regions consistent with optical and terrain evidence"
        return "landslide regions supported by the currently available remote-sensing evidence"
    return "all landslide regions"


def evidence_roles(instances: list[ModalityInstance]) -> list[str]:
    roles: list[str] = []
    for family in sorted({item.family for item in instances}):
        if family == "optical":
            roles.append("optical imagery provides scar texture, exposed soil, and vegetation disruption")
        elif family == "multispectral":
            roles.append("multispectral imagery provides vegetation, soil, moisture, and spectral contrast")
        elif family == "sar":
            roles.append("SAR provides roughness, moisture, and structural backscatter evidence")
        elif family == "terrain":
            roles.append("terrain products constrain slope plausibility, source area, and runout context")
        elif family == "deformation":
            roles.append("signed deformation provides activity evidence and is not ordinary texture")
    return roles


def build_prompt_triplet(
    row: dict[str, Any],
    instances: list[ModalityInstance],
    *,
    subset_signature: str,
    ablation: str = "normal",
    instruction_override: str | None = None,
    condition_override: str | None = None,
) -> tuple[str, str, str]:
    """Build task/condition/reasoning without dataset or preprocessing shortcuts."""
    del subset_signature
    if ablation not in {"normal", "shuffled", "fixed-generic", "no-semantic"}:
        raise ValueError(f"未知 instruction_ablation={ablation!r}")
    task = str(instruction_override).strip() if instruction_override else instruction_text(row)
    condition = str(condition_override).strip() if condition_override else condition_text(row)
    if ablation == "fixed-generic":
        task = "Segment all landslide regions."
        condition = "all landslide regions"
    elif ablation == "no-semantic":
        task = "Produce the requested binary segmentation mask."
        condition = "the requested target region"
    families = sorted({item.family for item in instances})
    roles = evidence_roles(instances)
    scale = gsd_description((row.get("spatial") or {}).get("gsd_m"))
    proposal = (
        f"Instruction: {task} Target: {condition}. Available evidence families: "
        f"{', '.join(families)}. Image scale: {scale}."
    )
    condition_prompt = f"Condition: {condition}."
    if ablation == "no-semantic":
        reasoning = "Use only the active sensor evidence to verify candidate masks; the requested target may be absent."
    else:
        reasoning = (
            f"Use only the available evidence ({'; '.join(roles)}) to verify mask proposals. "
            "Reject roads, riverbeds, shadows, terraces, continuous bare slopes, and unrelated exposed soil. "
            "An instruction may describe no target; in that case prefer an empty mask."
        )
    return proposal, condition_prompt, reasoning
