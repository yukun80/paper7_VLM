#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Landslide Bridge M2 协议与合成事实测试。

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest SEG_Multi-Source_Landslides/tests/test_landslide_bridge.py -v
写入行为：只在临时目录创建合成 npy，不修改 benchmark、datasets 或 outputs。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPTS = REPO_ROOT / "scripts/4-landslide-bridge"
sys.path.insert(0, str(BRIDGE_SCRIPTS))

from landslide_bridge_common import (  # noqa: E402
    cohen_kappa,
    connected_components,
    geometry_from_mask,
    krippendorff_alpha_nominal,
    load_config,
    validate_bridge_structured_target,
)


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BRIDGE_SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INVENTORY = load_script("qpsalm_bridge_inventory", "4-1_inventory_regions.py")
FACTS = load_script("qpsalm_bridge_facts", "4-2_extract_region_facts.py")
CANDIDATES = load_script("qpsalm_bridge_candidates", "4-3_build_candidate_descriptions.py")
MERGE = load_script("qpsalm_bridge_merge", "4-5_merge_expert_reviews.py")


class LandslideBridgeProtocolTest(unittest.TestCase):
    @staticmethod
    def _valid_structured_target(status: str = "present") -> dict:
        return {
            "target_status": status,
            "region": {
                "location": "center", "size_class": "small", "shape": "irregular",
                "elongation": "moderate", "compactness": "moderate",
                "fragmentation": "single",
            },
            "evidence": {
                "surface_observation": "A visible surface anomaly is present.",
                "terrain_support": "insufficient_evidence",
                "sar_support": "unavailable",
                "deformation_support": "unavailable",
                "surrounding_context": "Context is available.",
                "evidence_sufficiency": "partial",
            },
        }

    def test_schema_and_config_parse(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "configs/qpsalm_landslide_region_description_v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(load_config()["version"], "landslide_bridge_v1")

    def test_eight_connected_components_and_area_filter(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1, 1] = 1
        mask[2, 2] = 1
        mask[6, 6] = 1
        components = connected_components(mask, np.ones_like(mask), min_pixels=2, min_fraction=0.0)
        self.assertEqual(len(components), 1)
        self.assertEqual(int(components[0].sum()), 2)

    def test_absent_geometry_is_explicitly_unavailable(self) -> None:
        geometry = geometry_from_mask(None, np.ones((6, 7), dtype=np.uint8))
        self.assertEqual(geometry["area_pixels"], 0)
        self.assertEqual(geometry["location"], "unavailable")
        self.assertIsNone(geometry["bbox_xyxy_pixel_half_open"])

    def _evidence_item(
        self, root: Path, values: np.ndarray, valid: np.ndarray, *,
        family: str, units: str, normalization: str,
    ) -> dict:
        value_path = root / f"{family}_values.npy"
        valid_path = root / f"{family}_valid.npy"
        np.save(value_path, values.astype(np.float32))
        np.save(valid_path, valid.astype(np.uint8))
        return {
            "path": str(value_path), "available": True, "family": family,
            "sensor": "synthetic", "product_type": "synthetic", "band_names": ["band_0"],
            "units": units, "normalization": {"method": normalization},
            "valid_mask": {"path": str(valid_path)},
        }

    def test_evidence_levels_a_b_and_c(self) -> None:
        config = load_config()
        region = np.zeros((16, 16), dtype=np.uint8)
        region[5:10, 5:10] = 1
        values = np.linspace(0, 10, 256, dtype=np.float32).reshape(1, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = self._evidence_item(
                root, values, np.ones((16, 16)), family="terrain", units="m",
                normalization="preserve_physical_values",
            )
            relative = self._evidence_item(
                root, values, np.ones((16, 16)), family="optical", units="digital_number",
                normalization="preserve_rgb_values",
            )
            unavailable = self._evidence_item(
                root, values, np.zeros((16, 16)), family="sar", units="normalized",
                normalization="linear_clip_scale",
            )
            self.assertEqual(FACTS.modality_evidence(physical, region, config)["evidence_level"], "A_physical")
            self.assertEqual(FACTS.modality_evidence(relative, region, config)["evidence_level"], "B_normalized_relative")
            self.assertEqual(FACTS.modality_evidence(unavailable, region, config)["evidence_level"], "C_unavailable")

    def test_pilot_quota_rescaling_preserves_total(self) -> None:
        quotas = INVENTORY._split_quotas(30, load_config())
        self.assertEqual(sum(quotas.values()), 30)
        self.assertEqual(set(quotas), {"train", "val", "test"})

    def test_candidate_never_claims_expert_truth(self) -> None:
        record = {
            "target_status": "absent",
            "structured_targets": {"target_status": "absent", "region": {}, "evidence": {}},
            "candidate": {}, "provenance": {},
        }
        candidate = CANDIDATES.build_candidate(record)
        self.assertFalse(candidate["candidate"]["is_expert_truth"])
        self.assertIn("absent", candidate["candidate"]["summary"].casefold())

    def test_revision_requires_exact_double_review_agreement(self) -> None:
        left = {
            "decision": "revise", "corrected_structured_targets": {"target_status": "present"},
            "revised_summary": "A reviewed summary.",
        }
        self.assertTrue(MERGE._same_revision(left, dict(left)))
        changed = dict(left, revised_summary="A different summary.")
        self.assertFalse(MERGE._same_revision(left, changed))

    def test_expert_structured_target_is_schema_and_gt_status_constrained(self) -> None:
        target = self._valid_structured_target()
        self.assertEqual(
            validate_bridge_structured_target(target, expected_target_status="present"), []
        )
        invalid = json.loads(json.dumps(target))
        invalid["region"]["size_class"] = "huge"
        self.assertTrue(validate_bridge_structured_target(invalid))
        self.assertTrue(
            validate_bridge_structured_target(target, expected_target_status="absent")
        )

    def test_frozen_gate_rejects_out_of_range_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps({
                "protocol": "landslide_bridge_evaluation_gate_v1",
                "status": "frozen_after_pilot",
                "frozen": True,
                "thresholds": {
                    "no_target_rejection": 1.2,
                    "unsupported_claim_rate": 0.1,
                    "expert_fact_score": 0.7,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
                MERGE._load_frozen_gate(str(path))

    def test_agreement_statistics(self) -> None:
        self.assertEqual(cohen_kappa(["accept", "reject"], ["accept", "reject"]), 1.0)
        alpha = krippendorff_alpha_nominal([["accept", "accept"], ["reject", "reject"]])
        self.assertEqual(alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
