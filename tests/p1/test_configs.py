"""Repository-owned P1 configuration tests."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from sami_gsd.contracts.config import BenchmarkAuditConfig, load_audit_config


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
                components = {
                    component.component_key: component
                    for source in config.sources
                    for component in source.language_components
                }
                self.assertEqual(
                    tuple(components),
                    (
                        "mmrs_1m:rsicd",
                        "mmrs_1m:ucm",
                        "mmrs_1m:sydney",
                        "mmrs_1m:nwpu",
                        "mmrs_1m:rsitmd",
                        "mmrs_1m:dior_rsvg",
                        "rsgpt:rsicap",
                        "rsgpt:rsieval",
                    ),
                )
                self.assertTrue(all(not component.license.allowed_for_training for component in components.values()))

    def test_aggregate_language_license_cannot_authorize_components(self) -> None:
        """A shared raw container must never override per-component decisions."""

        payload = yaml.safe_load(
            (REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml").read_text(encoding="utf-8")
        )
        mmrs = next(source for source in payload["sources"] if source["source_key"] == "mmrs_1m")
        mmrs["license"].update(
            {
                "license_status": "verified",
                "license_name": "synthetic-approved",
                "license_url_or_document": "licenses/mmrs.txt",
                "allowed_for_training": True,
                "allowed_for_evaluation": True,
                "reviewed_by": "test-suite",
                "review_date": "2026-07-21",
            }
        )
        with self.assertRaisesRegex(ValidationError, "cannot authorize component use"):
            BenchmarkAuditConfig.model_validate(payload)

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
