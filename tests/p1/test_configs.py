"""Repository-owned P1.1 configuration tests."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import yaml

from sami_gsd.contracts.config import load_audit_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    """Validate live source configs and frozen ontology fields."""

    def test_live_audit_configs_are_strict_and_fail_closed(self) -> None:
        """Both modes bind nine sources without training eligibility."""

        for mode in ("small", "full"):
            with self.subTest(mode=mode):
                config = load_audit_config(REPOSITORY_ROOT / "configs" / f"benchmark_v3_{mode}.yaml")
                self.assertEqual(config.mode, mode)
                self.assertEqual(len(config.sources), 9)
                self.assertTrue(all(not source.license.allowed_for_training for source in config.sources))
                self.assertTrue(
                    all(
                        not source.license.allowed_for_training
                        for source in config.sources
                        if source.license.license_status == "unknown"
                    )
                )

    def test_scene_region_ontology_declares_every_required_policy(self) -> None:
        """Every frozen ontology field has all required attributes."""

        path = REPOSITORY_ROOT / "configs" / "scene_region_ontology_v2.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        required_fields = {
            "vegetation",
            "water_system",
            "farmland",
            "bare_soil",
            "exposed_rock",
            "road",
            "railway",
            "bridge",
            "building",
            "settlement",
            "valley",
            "channel",
            "ridge",
            "slope_position",
            "target_location",
            "target_shape",
            "boundary_clarity",
            "surface_disturbance",
            "vegetation_disturbance",
            "internal_texture",
            "relation_to_river",
            "relation_to_road",
            "relation_to_settlement",
            "alternative_explanation",
            "evidence_limitation",
        }
        required_attributes = {
            "kind",
            "allowed_values",
            "synonyms",
            "direct_observation_or_inference",
            "permitted_source_views",
            "forbidden_without_metadata",
            "evaluation_metric",
        }
        self.assertEqual(set(payload["fields"]), required_fields)
        self.assertTrue(all(set(field) == required_attributes for field in payload["fields"].values()))

    def test_root_package_exposes_only_sami_gsd_cli(self) -> None:
        """The greenfield distribution has exactly one console entrypoint."""

        payload = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(payload["project"]["scripts"], {"sami-gsd": "sami_gsd.cli:main"})


if __name__ == "__main__":
    unittest.main()
