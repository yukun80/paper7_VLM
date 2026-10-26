"""Stage 4 v2 永久测试的小型 CPU-only 合成 fixture。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oa_groundrag.data.grounded.region_contracts import (
    REGION_RECORD_SCHEMA,
    annotation_state_template,
    empty_description_template,
)
from oa_groundrag.grounding.evidence import render_binary_mask, render_mask_overlay
from oa_groundrag.grounding.outputs import REGION_OUTPUT_SCHEMA_VERSION


def target_output() -> dict[str, Any]:
    return {
        "schema_version": REGION_OUTPUT_SCHEMA_VERSION,
        "target_status": "target_present",
        "target_appearance": {
            "tone": "mixed brown and green tones",
            "texture": "heterogeneous texture",
            "vegetation_or_exposure": "partly exposed",
            "homogeneity": "heterogeneous",
            "boundary_visibility": "partly visible",
        },
        "target_morphology": {
            "shape": "irregular",
            "fragmentation": "multiple visible parts",
            "qualitative_orientation": "diagonal visual elongation",
        },
        "surrounding_environment": {
            "land_cover": ["vegetation"],
            "nearby_objects": ["bare ground"],
            "visible_terrain_context": ["sloping terrain"],
            "human_disturbance": [],
        },
        "region_context_contrast": {
            "tone_contrast": "moderate",
            "texture_contrast": "visible",
            "vegetation_contrast": "partial",
            "boundary_transition": "gradual",
            "adjacency": ["vegetation"],
        },
        "possible_confusers": ["bare soil"],
        "evidence_sufficiency": "limited",
        "short_summary": "The annotated region is visually heterogeneous relative to its surroundings.",
        "limitations": ["Fine material properties cannot be determined from this image."],
    }


def no_target_output() -> dict[str, Any]:
    return {
        "schema_version": REGION_OUTPUT_SCHEMA_VERSION,
        "target_status": "no_target",
        "target_appearance": {
            "tone": "not_applicable",
            "texture": "not_applicable",
            "vegetation_or_exposure": "not_applicable",
            "homogeneity": "not_applicable",
            "boundary_visibility": "not_applicable",
        },
        "target_morphology": {
            "shape": "not_applicable",
            "fragmentation": "not_applicable",
            "qualitative_orientation": "not_applicable",
        },
        "surrounding_environment": {
            "land_cover": [], "nearby_objects": [], "visible_terrain_context": [],
            "human_disturbance": [],
        },
        "region_context_contrast": {
            "tone_contrast": "not_applicable",
            "texture_contrast": "not_applicable",
            "vegetation_contrast": "not_applicable",
            "boundary_transition": "not_applicable",
            "adjacency": [],
        },
        "possible_confusers": [],
        "evidence_sufficiency": "insufficient",
        "short_summary": "No target region is specified by the empty mask.",
        "limitations": ["The empty mask provides no target region to describe."],
    }


def target_record(root: Path, *, record_id: str = "mgr_fixture") -> dict[str, Any]:
    optical = Image.new("RGB", (12, 10), (40, 80, 120))
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:7, 3:9] = True
    crop = optical.crop((2, 1, 10, 8))
    paths = {
        "optical_full": "assets/optical_full/a.png",
        "binary_mask": "assets/binary_mask/a.png",
        "context_crop": "assets/context_crop/a.png",
        "audit_overlay": "assets/audit_overlay/a.png",
    }
    for relative in paths.values():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    optical.save(root / paths["optical_full"], format="PNG")
    render_binary_mask(mask).save(root / paths["binary_mask"], format="PNG")
    crop.save(root / paths["context_crop"], format="PNG")
    render_mask_overlay(optical, mask, alpha=0.45).save(root / paths["audit_overlay"], format="PNG")
    return {
        "schema_version": REGION_RECORD_SCHEMA,
        "record_id": record_id,
        "parent_id": "parent-1",
        "parent_identity_status": "known",
        "sample_id": "sample-1",
        "source": "synthetic",
        "split": "val",
        "source_identity": {"record_sha256": "0" * 64},
        "mask_source": "oa_auxseg_benchmark_gt",
        "target_status": "target_present",
        "representation_mode": "full_plus_mask_plus_crop",
        "assets": paths,
        "formal_model_input_roles": ["optical_full", "binary_mask", "context_crop"],
        "audit_only_roles": ["audit_overlay"],
        "program_facts": {
            "image": {"width_pixels": 12, "height_pixels": 10},
            "mask": {"crop_window_xyxy_pixel_half_open": [2, 1, 10, 8]},
            "coordinate_basis": "top_left_pixel_xy_half_open",
            "mask_renderer": "binary_mask_png_l_0_255.v1",
            "difficulty_proxy": None,
            "counterfactual": None,
        },
        "description": empty_description_template(),
        "annotation": annotation_state_template(),
    }
