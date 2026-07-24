"""Synthetic, filesystem-isolated fixtures for focused Benchmark v4 tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sami_gsd.utilities.artifacts import sha256_file


def make_synthetic_source(root: Path) -> tuple[dict[str, Any], Path]:
    datasets_root = root / "datasets"
    source_root = datasets_root / "synthetic"
    (source_root / "indexes").mkdir(parents=True)
    (source_root / "hdf5/image/train").mkdir(parents=True)
    (source_root / "hdf5/mask/train").mkdir(parents=True)

    schema_path = source_root / "channel_schema.json"
    schema_path.write_text(
        json.dumps({"channels": ["zero_valid", "zero_missing"]}),
        encoding="utf-8",
    )
    image_path = source_root / "hdf5/image/train/sample_1.h5"
    mask_path = source_root / "hdf5/mask/train/sample_1.h5"
    image = np.zeros((2, 3, 3), dtype=np.float32)
    image[0, 1, 1] = 0.0
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("image", data=image)
        handle.create_dataset(
            "channel_valid",
            data=np.asarray([1, 0], dtype=np.uint8),
        )
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    with h5py.File(mask_path, "w") as handle:
        handle.create_dataset("mask", data=mask)

    index_path = source_root / "indexes/train.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "sample_key": "sample_1",
                "image_hdf5": "hdf5/image/train/sample_1.h5",
                "mask_hdf5": "hdf5/mask/train/sample_1.h5",
                "split": "train",
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    channels = [
        {
            "index": 0,
            "channel_key": "zero_valid",
            "display_name": "Zero valid",
            "modality_family": "other",
            "physical_unit": None,
            "wavelength_nm": None,
            "wavelength_known": False,
            "gsd_m": None,
            "gsd_known": False,
            "normalization": "zscore_valid_pixels",
            "validity_source": "channel_valid",
            "schema_version": "sami_channel_descriptor_v1",
        },
        {
            "index": 1,
            "channel_key": "zero_missing",
            "display_name": "Zero missing",
            "modality_family": "other",
            "physical_unit": None,
            "wavelength_nm": None,
            "wavelength_known": False,
            "gsd_m": None,
            "gsd_known": False,
            "normalization": "zscore_valid_pixels",
            "validity_source": "channel_valid",
            "schema_version": "sami_channel_descriptor_v1",
        },
    ]
    source = {
        "source_key": "synthetic",
        "ingestion_status": "ready",
        "source_root": "datasets/synthetic",
        "hdf5_base": "datasets/synthetic",
        "indexes": [
            {
                "logical_path": "datasets/synthetic/indexes/train.jsonl",
                "source_declared_split": "train",
                "canonical_split": "train",
            }
        ],
        "sample_id_field": "sample_key",
        "image_path_field": "image_hdf5",
        "mask_path_field": "mask_hdf5",
        "row_split_field": "split",
        "split_assurance": "source_declared_unverified",
        "evaluation_eligibility": "exploratory",
        "group_field": None,
        "group_kind": "unknown",
        "group_completeness": "unavailable",
        "group_evidence": [],
        "duplicate_component_field": None,
        "duplicate_evidence_level": "unavailable",
        "channel_schema": "datasets/synthetic/channel_schema.json",
        "channel_schema_sha256": sha256_file(schema_path),
        "channels": channels,
        "validity": {
            "valid_mask_key": None,
            "valid_mask_owner": "absent",
            "pixel_valid_key": None,
            "pixel_valid_owner": "absent",
            "channel_valid_key": "/channel_valid",
            "channel_valid_owner": "image_hdf5",
            "label_valid_semantics": "full grid",
            "input_pixel_valid_derivation": (
                "broadcast_present_channels_over_full_grid"
            ),
            "notes": ["synthetic fixture"],
        },
        "registered_rgb": None,
        "provenance": {
            "source_name": "Synthetic",
            "source_document": "datasets/synthetic/channel_schema.json",
            "citation_key": "synthetic",
            "upstream_url": None,
            "provenance_notes": "Focused unit fixture only.",
        },
        "risks": [],
    }
    return source, datasets_root
