from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from sami_gsd.data.hdf5_sources_v4 import (
    Hdf5SourceV4Error,
    iter_source_observations,
    resolve_benchmark_logical_path,
    resolve_dataset_logical_path,
)
from tests.p1.v4_test_support import make_synthetic_source


class Hdf5SourceV4Tests(unittest.TestCase):
    def test_valid_zero_and_missing_zero_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, datasets_root = make_synthetic_source(Path(directory))
            observations = list(iter_source_observations(source, datasets_root))
        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation.channel_valid.tolist(), [True, False])
        self.assertTrue(bool(observation.pixel_valid[0].all()))
        self.assertFalse(bool(observation.pixel_valid[1].any()))
        self.assertTrue(
            bool(np.all(observation.image_values[0][observation.pixel_valid[0]] >= 0))
        )
        record = observation.to_source_record_payload()
        self.assertEqual(record["canonical_split"], "train")
        self.assertEqual(record["evaluation_eligibility"], "exploratory")
        self.assertTrue(
            record["image"]["source_logical_path"].startswith("datasets/")
        )
        self.assertTrue(
            record["image"]["benchmark_logical_path"].startswith(
                "benchmark/sami_landslide_hdf5_v4/small/assets/synthetic/"
            )
        )
        self.assertGreater(record["image"]["size_bytes"], 0)
        self.assertNotIn("/home/", str(record))

    def test_source_record_schema_rejects_native_split_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, datasets_root = make_synthetic_source(Path(directory))
            observation = next(iter_source_observations(source, datasets_root))
        record = observation.to_source_record_payload()
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "schemas/hdf5_source_record_v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        validator.validate(record)
        tampered = deepcopy(record)
        tampered["source_declared_split"] = "val"
        with self.assertRaises(ValidationError):
            validator.validate(tampered)

    def test_native_split_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, datasets_root = make_synthetic_source(Path(directory))
            source["indexes"][0]["source_declared_split"] = "val"
            with self.assertRaisesRegex(Hdf5SourceV4Error, "rewritten|split drift"):
                list(iter_source_observations(source, datasets_root))

    def test_root_containment_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                resolve_dataset_logical_path(
                    "datasets/source/../../escape.h5",
                    root,
                )
            with self.assertRaises(ValueError):
                resolve_benchmark_logical_path(
                    "benchmark/sami_landslide_hdf5_v4/small/assets/../../escape.h5",
                    root,
                )


if __name__ == "__main__":
    unittest.main()
