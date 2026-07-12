#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark-v2 referring/no-target protocol tests.

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest
SEG_Multi-Source_Landslides/tests/test_benchmark_v2.py -v
写入行为：不写 benchmark 或 outputs。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPTS = REPO_ROOT / "scripts" / "1-benchmark"
sys.path.insert(0, str(BENCHMARK_SCRIPTS))

from geohazard_benchmark_common import make_referring_target_sample, modality_entry  # noqa: E402
from geohazard_referring_common import occupied_grid_positions  # noqa: E402

_PREPROCESS_SPEC = importlib.util.spec_from_file_location(
    "qpsalm_benchmark_preprocess", BENCHMARK_SCRIPTS / "1-4_preprocess_samples.py"
)
assert _PREPROCESS_SPEC is not None and _PREPROCESS_SPEC.loader is not None
_PREPROCESS = importlib.util.module_from_spec(_PREPROCESS_SPEC)
_PREPROCESS_SPEC.loader.exec_module(_PREPROCESS)


class ReferringCounterfactualTest(unittest.TestCase):
    def test_sentinel2_band_names_and_physics_are_canonical(self) -> None:
        modality = modality_entry(
            REPO_ROOT.parent / "datasets" / "placeholder.h5",
            fmt="hdf5",
            band_names=["B1", "B2", "B8A", "B12"],
            native_gsd_m=10,
            family="multispectral",
            sensor="sentinel2",
            product_type="surface_reflectance",
            units="reflectance",
            signed=False,
        )
        self.assertEqual(modality["band_names"], ["B01", "B02", "B8A", "B12"])
        physics = {item["name"]: item for item in modality["band_metadata"]}
        self.assertEqual(physics["B01"]["native_gsd_m"], 60.0)
        self.assertEqual(physics["B02"]["center_wavelength_nm"], 490.0)
        self.assertEqual(physics["B8A"]["native_gsd_m"], 20.0)

    def test_normalization_uses_product_type_not_raw_modality_name(self) -> None:
        values = np.asarray([[[-60.0, -20.0], [0.0, 20.0]]], dtype=np.float32)
        normalized, metadata = _PREPROCESS.normalize_modality(
            "future_sensor_product",
            values,
            {"dataset_name": "synthetic"},
            {"family": "sar", "product_type": "sar_backscatter", "units": "dB"},
            {},
        )
        self.assertEqual(metadata["method"], "linear_clip_scale")
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_occupied_grid_uses_all_pixels_not_top_candidates(self) -> None:
        mask = np.zeros((1, 9, 9), dtype=np.uint8)
        for y, x in ((1, 1), (1, 4), (1, 7), (4, 1), (7, 7)):
            mask[:, y, x] = 1
        occupied = occupied_grid_positions(mask)
        self.assertEqual(
            occupied,
            {"upper-left", "upper", "upper-right", "left", "lower-right"},
        )
        self.assertNotIn("center", occupied)

    def test_referring_row_preserves_parent_mask_for_counterfactual_audit(self) -> None:
        parent = {
            "sample_id": "parent", "task_type": "landslide_segmentation",
            "source_key": "source", "mask": {"path": "benchmark/x/mask.npy", "format": "npy"},
        }
        target = {
            "target_id": "ref_no_target", "category": "no_target", "subtype": "position_center",
            "target_mask": {"path": "benchmark/x/empty.npy", "format": "npy"},
            "grounding": {"grid": "center"}, "confidence": "rule_high",
        }
        row = make_referring_target_sample(parent, target)
        self.assertEqual(row["parent_mask"], parent["mask"])
        self.assertNotIn("mask", row)


if __name__ == "__main__":
    unittest.main()
