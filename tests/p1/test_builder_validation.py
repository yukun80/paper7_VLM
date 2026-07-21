"""End-to-end synthetic Small build, validation and repeat-hash tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sami_gsd.contracts.config import BenchmarkAuditConfig, load_audit_config
from sami_gsd.data.builder import build_canonical_benchmark
from sami_gsd.data.validation import validate_benchmark_payload, validate_published_benchmark
from tests.p1.test_materialization import spatial_input


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def synthetic_build_config() -> BenchmarkAuditConfig:
    """Replace live sources with one synthetic technical source."""

    payload = load_audit_config(REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml").model_dump(mode="json")
    payload["build"]["materialization"]["canvas_hw"] = [8, 8]
    payload["build"]["small_max_parents_per_source"] = 8
    payload["sources"] = [
        {
            "source_key": "synthetic",
            "enabled": True,
            "task_roles": ["inventory", "t1", "t2"],
            "provenance": {
                "source_key": "synthetic",
                "source_name": "Synthetic test source",
                "source_root": "datasets/synthetic",
                "source_document": None,
                "citation_key": "synthetic",
                "upstream_url": None,
                "provenance_notes": "synthetic local research fixture",
            },
        }
    ]
    return BenchmarkAuditConfig.model_validate(payload)


class BuilderValidationTests(unittest.TestCase):
    """Verify a complete new directory can pass every engineering P1 gate."""

    def test_two_clean_builds_have_identical_aggregate_and_replay_validation(self) -> None:
        """Atomic builds are byte-stable and the independent validator reopens them."""

        config = synthetic_build_config()
        first_input = spatial_input()
        second_input = replace(
            spatial_input(),
            parent_id="synthetic-parent-002",
            source=spatial_input().source.model_copy(
                update={
                    "record_id": "record-002",
                    "scene_id": "scene-002",
                    "region_id": "region-002",
                    "source_group_id": "group-002",
                }
            ),
            source_record_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "build-one"
            second_root = root / "build-two"
            first = build_canonical_benchmark(
                config,
                parent_inputs=(first_input, second_input),
                description_records=(),
                output_dir=first_root,
                schemas_root=REPOSITORY_ROOT / "schemas",
            )
            second = build_canonical_benchmark(
                config,
                parent_inputs=(second_input, first_input),
                description_records=(),
                output_dir=second_root,
                schemas_root=REPOSITORY_ROOT / "schemas",
            )
            self.assertEqual(first["aggregate_sha256"], second["aggregate_sha256"])
            self.assertEqual(first["output_sha256"], second["output_sha256"])
            replay = validate_published_benchmark(first_root, schemas_root=REPOSITORY_ROOT / "schemas")
            self.assertEqual(replay["errors"], [])
            self.assertEqual(replay["verified_duplicate_cross_split_count"], 0)
            self.assertEqual(replay["provenance_binding_error_count"], 0)
            self.assertEqual(list(first_root.rglob("*.part-*")), [])

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                build_canonical_benchmark(
                    config,
                    parent_inputs=(first_input,),
                    description_records=(),
                    output_dir=first_root,
                    schemas_root=REPOSITORY_ROOT / "schemas",
                )

            summary_path = first_root / "reports/summary_report.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["valid_pixel_count"] = 0
            summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
            tampered = validate_benchmark_payload(first_root, schemas_root=REPOSITORY_ROOT / "schemas")
            self.assertIn("summary_report_replay_mismatch:valid_pixel_count", tampered["errors"])


if __name__ == "__main__":
    unittest.main()
