"""Synthetic P1 fixtures with no dependency on project datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SHA = "a" * 64


def canonical_parent_payload() -> dict[str, Any]:
    """Return one complete, valid Canonical Parent v3 mapping."""

    payload: dict[str, Any] = {
        "schema_version": "sami_canonical_parent_v3",
        "parent_id": "synthetic-parent-001",
        "source": {
            "dataset": "synthetic",
            "record_id": "record-001",
            "scene_id": "scene-001",
            "event_id": None,
            "region_id": "region-001",
            "source_group_id": "group-001",
        },
        "split": "train",
        "reference_canvas": {
            "reference_modality_id": "optical-001",
            "coordinate_space": "reference_pixel_half_open",
            "original_hw": [32, 48],
            "canvas_hw": [32, 48],
            "valid_mask_path": "assets/synthetic-parent-001/valid.npy",
            "transform_chain": [
                {
                    "operation": "identity",
                    "input_hw": [32, 48],
                    "output_hw": [32, 48],
                    "interpolation": "not_applicable",
                    "invertible": True,
                    "parameters": {},
                }
            ],
            "inverse_transform_available": True,
            "crs": None,
            "geotransform": None,
        },
        "modalities": [
            {
                "modality_id": "optical-001",
                "family": "optical",
                "sensor": "synthetic-sensor",
                "product_type": "orthorectified-image",
                "band_names": ["red", "green", "blue"],
                "band_metadata": [
                    {"name": "red", "wavelength_nm": 665.0, "polarization": None, "units": "reflectance"},
                    {"name": "green", "wavelength_nm": 560.0, "polarization": None, "units": "reflectance"},
                    {"name": "blue", "wavelength_nm": 490.0, "polarization": None, "units": "reflectance"},
                ],
                "orbit": None,
                "acquisition_time": "2024-01-01T00:00:00Z",
                "time_range": None,
                "native_gsd_m": 10.0,
                "aligned_gsd_m": 10.0,
                "units": "reflectance",
                "signed": False,
                "sign_convention": None,
                "normalization": {"method": "none", "parameters": {}, "statistics_source": None},
                "quality": {"status": "usable", "flags": [], "notes": None},
                "availability_status": "present_valid",
                "valid_coverage": 1.0,
                "native_asset_path": "assets/synthetic-parent-001/optical.npy",
                "aligned_asset_path": "assets/synthetic-parent-001/optical.npy",
                "valid_mask_path": "assets/synthetic-parent-001/valid.npy",
                "source_to_reference_transform": [
                    {
                        "operation": "identity",
                        "input_hw": [32, 48],
                        "output_hw": [32, 48],
                        "interpolation": "not_applicable",
                        "invertible": True,
                        "parameters": {},
                    }
                ],
                "reference_to_source_transform": [
                    {
                        "operation": "identity",
                        "input_hw": [32, 48],
                        "output_hw": [32, 48],
                        "interpolation": "not_applicable",
                        "invertible": True,
                        "parameters": {},
                    }
                ],
                "alignment_status": "reference",
                "render_policy": {"mode": "rgb", "channels": ["red", "green", "blue"], "clip_percentiles": None},
                "hashes": {"native": SHA, "aligned": SHA, "valid": SHA},
            }
        ],
        "annotations": {
            "global_landslide_mask": {"path": "assets/synthetic-parent-001/mask.npy", "sha256": SHA},
            "global_mask_origin": "official",
            "global_target_status": "positive",
            "referring_regions": [
                {
                    "region_id": "region-001",
                    "expression": "the central landslide region",
                    "mask_ref": {"path": "assets/synthetic-parent-001/mask.npy", "sha256": SHA},
                    "bbox_half_open": [4, 5, 20, 24],
                    "annotation_origin": "official",
                }
            ],
            "no_target_eligibility": False,
            "region_fact_refs": [],
        },
        "provenance": {
            "source_registry_key": "synthetic",
            "source_paths": ["datasets/synthetic/image.bin"],
            "source_record_sha256": SHA,
            "scanner_version": "synthetic-test-v1",
            "derivation_steps": [],
        },
        "license": {
            "source_key": "synthetic",
            "license_status": "verified",
            "license_name": "CC0-1.0",
            "license_url_or_document": "licenses/synthetic.txt",
            "allowed_for_training": True,
            "allowed_for_evaluation": True,
            "allowed_for_redistribution": True,
            "academic_only": False,
            "attribution": "Synthetic fixture generated by the test suite.",
            "reviewed_by": "test-suite",
            "review_date": "2026-07-20",
        },
        "hashes": {"source_record_sha256": SHA, "assets": {"mask": SHA}},
        "annotation_status": "gold",
    }
    return deepcopy(payload)


def task_view_payload() -> dict[str, Any]:
    """Return one complete, valid T2 task-view mapping."""

    return {
        "task_id": "task-001",
        "parent_id": "synthetic-parent-001",
        "task_type": "t2_referring",
        "instruction": "Segment the central landslide region.",
        "target_status": "positive",
        "region_geometry": {
            "coordinate_space": "reference_pixel_half_open",
            "region_id": "region-001",
            "bbox_half_open": [4, 5, 20, 24],
        },
        "target_mask_ref": {"path": "assets/synthetic-parent-001/mask.npy", "sha256": SHA},
        "target_box_ref": [4, 5, 20, 24],
        "answer_ref": None,
        "annotation_origin": "official",
        "weight": 1.0,
    }
