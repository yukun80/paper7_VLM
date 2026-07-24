from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from sami_gsd.contracts.benchmark_v4 import (
    ChannelDescriptorV1,
    validate_dataset_logical_path,
)
from sami_gsd.contracts.benchmark_v4_config import load_benchmark_v4_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkV4ConfigTests(unittest.TestCase):
    def test_production_config_preserves_five_ready_sources(self) -> None:
        config = load_benchmark_v4_config(
            REPOSITORY_ROOT / "configs/benchmark_v4_small.yaml",
            repository_root=REPOSITORY_ROOT,
        )
        self.assertEqual(config.protocol, "sami_hdf5_materialized_copy_v1")
        ready = [source for source in config.sources if source.ingestion_status == "ready"]
        self.assertEqual(len(ready), 5)
        self.assertEqual(sum(len(source.channels) for source in ready), 29)
        self.assertTrue(
            all(
                not channel.wavelength_known
                and channel.wavelength_nm is None
                and not channel.gsd_known
                and channel.gsd_m is None
                for source in ready
                for channel in source.channels
            )
        )
        lmhld = next(source for source in ready if source.source_key == "lmhld")
        self.assertEqual(
            tuple(item.canonical_split for item in lmhld.indexes),
            ("train", "val", "test"),
        )
        landslidebench = next(
            source
            for source in ready
            if source.source_key == "landslidebench_agent"
        )
        self.assertEqual(
            landslidebench.known_location_cross_split_conflict_count,
            311,
        )
        self.assertEqual(landslidebench.expected_pair_count, 2130)
        self.assertEqual(
            landslidebench.expected_source_split_counts,
            {"train": 1701, "val": 210, "test": 219},
        )
        self.assertTrue(
            all(
                source.known_location_cross_split_conflict_count is None
                for source in config.sources
                if source.source_key != "landslidebench_agent"
            )
        )
        sen12 = next(
            source
            for source in config.sources
            if source.source_key == "sen12_landslides"
        )
        self.assertIsNone(sen12.split_assurance)
        self.assertIsNone(sen12.evaluation_eligibility)
        self.assertEqual(sen12.expected_pair_count, 0)
        self.assertEqual(sen12.expected_source_split_counts, {})

    def test_config_rejects_missing_native_train_index(self) -> None:
        source_path = REPOSITORY_ROOT / "configs/benchmark_v4_small.yaml"
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        tampered = deepcopy(payload)
        gdcld = next(
            source
            for source in tampered["sources"]
            if source["source_key"] == "gdcld"
        )
        gdcld["indexes"] = gdcld["indexes"][1:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(tampered, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "index selection/order"):
                load_benchmark_v4_config(
                    path,
                    repository_root=REPOSITORY_ROOT,
                )

    def test_config_rejects_conflict_count_drift(self) -> None:
        source_path = REPOSITORY_ROOT / "configs/benchmark_v4_small.yaml"
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        tampered = deepcopy(payload)
        landslidebench = next(
            source
            for source in tampered["sources"]
            if source["source_key"] == "landslidebench_agent"
        )
        landslidebench["known_location_cross_split_conflict_count"] = 310
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(tampered, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflict count"):
                load_benchmark_v4_config(
                    path,
                    repository_root=REPOSITORY_ROOT,
                )

    def test_config_rejects_unknown_root_key(self) -> None:
        source_path = REPOSITORY_ROOT / "configs/benchmark_v4_small.yaml"
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "extra"):
                load_benchmark_v4_config(
                    path,
                    repository_root=REPOSITORY_ROOT,
                )

    def test_unknown_metadata_state_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "wavelength_known"):
            ChannelDescriptorV1(
                index=0,
                channel_key="invalid",
                display_name="Invalid",
                modality_family="optical",
                physical_unit=None,
                wavelength_nm=665.0,
                wavelength_known=False,
                gsd_m=None,
                gsd_known=False,
                normalization="zscore_valid_pixels",
                validity_source="channel_valid",
            )

    def test_logical_path_rejects_absolute_and_traversal(self) -> None:
        for value in (
            "/home/user/source.h5",
            "datasets/source/../source.h5",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_dataset_logical_path(value, require_hdf5=True)


if __name__ == "__main__":
    unittest.main()
