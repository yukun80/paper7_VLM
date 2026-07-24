from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sami_gsd.data.benchmark_v4 import (
    BenchmarkV4BuildError,
    _NormalizationAccumulator,
    _StatisticsAccumulator,
    _assert_source_population,
    _channel_catalog,
    _materialize_observation,
    _reject_machine_paths,
)
from sami_gsd.data.benchmark_v4_validation import (
    _ReplayAccumulator,
    _report_payload,
    _validate_artifact_tree,
    _validate_materialized_assets,
)
from sami_gsd.data.hdf5_sources_v4 import iter_source_observations
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from tests.p1.v4_test_support import make_synthetic_source


class BuilderValidatorHelperTests(unittest.TestCase):
    def test_streaming_statistics_use_only_valid_present_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, datasets_root = make_synthetic_source(Path(directory))
            observation = next(iter_source_observations(source, datasets_root))
        record = observation.to_source_record_payload()
        normalization = _NormalizationAccumulator()
        normalization.update(observation, record)
        payload = normalization.finalize(source_index_sha256="0" * 64)
        self.assertEqual(len(payload["statistics"]), 1)
        self.assertEqual(payload["statistics"][0]["channel_key"], "zero_valid")
        self.assertEqual(payload["statistics"][0]["mean"], 0.0)

        statistics = _StatisticsAccumulator()
        statistics.update(observation, record)
        stats = statistics.statistics_payload(
            normalization_binding_sha256="1" * 64
        )
        self.assertEqual(stats["positive_count"], 1)
        self.assertEqual(
            stats["eligibility_counts"],
            {"strict": 0, "exploratory": 1, "train_only": 0},
        )
        self.assertEqual(stats["strict_generalization_status"], "unavailable")

    def test_machine_path_scan_allows_hdf5_keys_only(self) -> None:
        _reject_machine_paths(
            {
                "dataset_key": "/image",
                "validity": {"channel_valid_key": "/channel_valid"},
            }
        )
        with self.assertRaisesRegex(ValueError, "machine path"):
            _reject_machine_paths({"logical_path": "/home/user/source.h5"})

    def test_population_binding_rejects_truncated_source(self) -> None:
        source = {
            "source_key": "synthetic",
            "expected_pair_count": 2,
            "expected_source_split_counts": {"train": 2},
            "expected_positive_count": 1,
            "expected_no_target_count": 1,
        }
        with self.assertRaisesRegex(BenchmarkV4BuildError, "drifts"):
            _assert_source_population(
                source,
                observed_pair_count=1,
                observed_split_counts={"train": 1},
                observed_positive_count=1,
                observed_no_target_count=0,
            )

    def test_validator_artifact_replay_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory)
            artifact = benchmark / "indexes/source_records.jsonl"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(canonical_json_bytes({"record": 1}))
            bindings = {artifact.relative_to(benchmark).as_posix(): sha256_file(artifact)}
            manifest = {
                "artifact_bindings": bindings,
                "aggregate_sha256": sha256_bytes(
                    canonical_json_bytes(dict(sorted(bindings.items())))
                ),
            }
            errors: list[str] = []
            _validate_artifact_tree(benchmark, manifest, errors=errors)
            self.assertIn("missing_artifact:manifests/benchmark_manifest.json", errors)
            errors.clear()
            (benchmark / "manifests").mkdir()
            (benchmark / "manifests/benchmark_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            artifact.write_text("corrupt\n", encoding="utf-8")
            _validate_artifact_tree(benchmark, manifest, errors=errors)
            self.assertIn(
                "artifact_sha256_mismatch:indexes/source_records.jsonl",
                errors,
            )

    def test_materialization_copies_hdf5_and_channel_catalog_is_explicit(self) -> None:
        from sami_gsd.data.benchmark_v4 import _JsonlWriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, datasets_root = make_synthetic_source(root)
            observation = next(iter_source_observations(source, datasets_root))
            record = observation.to_source_record_payload()
            staging = root / "benchmark"
            staging.mkdir()
            materialized: dict[str, dict[str, object]] = {}
            with _JsonlWriter(staging / "manifests/materialized_assets.jsonl") as writer:
                _materialize_observation(
                    observation,
                    record,
                    staging=staging,
                    writer=writer,
                    materialized=materialized,
                )
            self.assertEqual(len(materialized), 2)
            for relative, row in materialized.items():
                copied = staging / relative
                self.assertTrue(copied.is_file())
                self.assertEqual(sha256_file(copied), row["sha256"])
                self.assertNotEqual(copied.stat().st_ino, observation.image_source_path.stat().st_ino)
            materialization_path = staging / "manifests/materialized_assets.jsonl"
            manifest = {
                "materialization_index_sha256": sha256_file(materialization_path),
                "materialized_asset_count": len(materialized),
                "materialized_size_bytes": sum(
                    int(row["size_bytes"]) for row in materialized.values()
                ),
            }
            errors: list[str] = []
            count, size = _validate_materialized_assets(
                staging,
                datasets_root,
                [record],
                manifest,
                errors=errors,
            )
            self.assertEqual(errors, [])
            self.assertEqual(count, 2)
            self.assertEqual(size, manifest["materialized_size_bytes"])

            catalog = _channel_catalog([source])
            self.assertEqual(
                [entry["channel_key"] for entry in catalog["entries"]],
                ["zero_missing", "zero_valid"],
            )
            self.assertEqual(
                catalog["ordering_rule"],
                "channel_token_is_lexicographic_channel_key_not_tensor_order",
            )

    def test_validation_report_is_finite_and_contract_bound(self) -> None:
        report = _report_payload(
            errors=[],
            warnings=[],
            manifest_sha256="0" * 64,
            artifact_count=10,
            materialized_asset_count=2,
            materialized_size_bytes=123,
            source_record_count=1,
            parent_count=1,
            split_counts={"train": 1},
            assurance_counts={"source_declared_unverified": 1},
            eligibility_counts={"exploratory": 1},
            normalization_binding="1" * 64,
        )
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["schema_version"], "sami_benchmark_validation_report_v4")

    def test_independent_statistics_replay_uses_contract_hash_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, datasets_root = make_synthetic_source(Path(directory))
            observation = next(iter_source_observations(source, datasets_root))
        record = observation.to_source_record_payload()
        replay = _ReplayAccumulator()
        replay.update(observation, record)
        statistics = replay.statistics(
            normalization_binding_sha256="1" * 64,
        )
        self.assertEqual(statistics["source_record_count"], 1)
        core = dict(statistics)
        aggregate = core.pop("aggregate_sha256")
        core.pop("schema_version")
        self.assertEqual(
            aggregate,
            sha256_bytes(canonical_json_bytes(core)),
        )


if __name__ == "__main__":
    unittest.main()
