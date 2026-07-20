"""Read-only deterministic source-audit integration tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.data.audit import audit_sources
from sami_gsd.utilities.artifacts import atomic_write_json


def build_audit_config(*, training_allowed: bool = False) -> BenchmarkAuditConfig:
    """Build a minimal unknown-license audit config."""

    payload: dict[str, Any] = {
        "schema_version": "sami_benchmark_audit_config_v3",
        "benchmark_name": "SAMI Landslide Grounded Benchmark v3",
        "mode": "small",
        "seed": 42,
        "benchmark_relative_path": "sami_landslide_v3/small",
        "datasets_root": {"env": "SAMI_TEST_DATASETS_ROOT", "relative_to": "repository_root", "default": "datasets"},
        "benchmark_root": {"env": "SAMI_TEST_BENCHMARK_ROOT", "relative_to": "repository_root", "default": "benchmark"},
        "audit": {"hash_algorithm": "sha256", "include_hidden": False, "follow_symlinks": False},
        "sources": [
            {
                "source_key": "synthetic",
                "display_name": "Synthetic source",
                "local_path": "synthetic",
                "enabled": True,
                "allowed_task_roles": ["inventory"],
                "license": {
                    "source_key": "synthetic",
                    "license_status": "unknown",
                    "license_name": "unknown",
                    "license_url_or_document": None,
                    "allowed_for_training": training_allowed,
                    "allowed_for_evaluation": False,
                    "allowed_for_redistribution": False,
                    "academic_only": False,
                    "attribution": "Synthetic audit fixture; no use authorized.",
                    "reviewed_by": None,
                    "review_date": None,
                },
            }
        ],
    }
    return BenchmarkAuditConfig.model_validate(payload)


class AuditTests(unittest.TestCase):
    """Verify audit determinism, immutability and publication behavior."""

    def test_unknown_license_fails_closed_for_training(self) -> None:
        """Unknown sources can be inventoried but cannot train."""

        with self.assertRaisesRegex(ValidationError, "training eligibility"):
            build_audit_config(training_allowed=True)

    def test_audit_is_read_only_atomic_and_repeatable(self) -> None:
        """Independent runs over identical bytes yield one aggregate hash."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets_root = root / "datasets"
            source_root = datasets_root / "synthetic"
            source_root.mkdir(parents=True)
            (source_root / "b.bin").write_bytes(b"beta")
            (source_root / "a.bin").write_bytes(b"alpha")
            before = {path.name: path.read_bytes() for path in source_root.iterdir()}

            first = audit_sources(build_audit_config(), datasets_root=datasets_root, output_dir=root / "audit-one")
            second = audit_sources(build_audit_config(), datasets_root=datasets_root, output_dir=root / "audit-two")
            after = {path.name: path.read_bytes() for path in source_root.iterdir()}

            self.assertEqual(before, after)
            self.assertEqual(first["aggregate_sha256"], second["aggregate_sha256"])
            self.assertEqual(first["errors"], [])
            self.assertEqual(list(root.rglob("*.part-*")), [])

            inventory = json.loads((root / "audit-one" / "inventory.json").read_text(encoding="utf-8"))
            paths = [record["logical_path"] for record in inventory["sources"][0]["files"]]
            self.assertEqual(paths, ["datasets/synthetic/a.bin", "datasets/synthetic/b.bin"])
            self.assertTrue(all(not Path(path).is_absolute() for path in paths))

            registry = yaml.safe_load((root / "audit-one" / "source_registry.yaml").read_text(encoding="utf-8"))
            self.assertFalse(registry["entries"][0]["allowed_for_training"])
            report = json.loads((root / "audit-one" / "license_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["training_eligible_unknown_count"], 0)
            self.assertEqual(report["unknown_license_sources"], ["synthetic"])

    def test_audit_refuses_existing_output(self) -> None:
        """An existing audit directory is never overwriteable scratch."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets_root = root / "datasets"
            (datasets_root / "synthetic").mkdir(parents=True)
            output_dir = root / "audit"
            audit_sources(build_audit_config(), datasets_root=datasets_root, output_dir=output_dir)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                audit_sources(build_audit_config(), datasets_root=datasets_root, output_dir=output_dir)

    def test_audit_rejects_a_symlinked_source_root(self) -> None:
        """Configured source roots cannot escape through a symbolic link."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets_root = root / "datasets"
            datasets_root.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (datasets_root / "synthetic").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                audit_sources(build_audit_config(), datasets_root=datasets_root, output_dir=root / "audit")

    def test_strict_json_rejects_nan_and_existing_target(self) -> None:
        """Published JSON is finite and accepted targets are not replaced."""

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                atomic_write_json(target, {"metric": float("nan")})
            atomic_write_json(target, {"errors": []})
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                atomic_write_json(target, {"errors": ["replacement"]})


if __name__ == "__main__":
    unittest.main()
