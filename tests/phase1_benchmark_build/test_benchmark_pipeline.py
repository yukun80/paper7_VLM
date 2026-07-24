"""阶段 1B Benchmark 的五源合成 fixture 与破坏检测测试。"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "scripts/phase1_benchmark_build"
sys.path.insert(0, str(SCRIPT_ROOT))

from benchmark_common import (  # noqa: E402
    BenchmarkDataset,
    collate_benchmark_samples,
    read_jsonl,
    resize_binary_mask,
    resize_continuous_with_validity,
)


def _load_numbered(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_numbered("phase1_builder", "1_1_build_benchmark.py")
VALIDATOR = _load_numbered("phase1_validator", "1_2_validate_benchmark.py")
SMOKE = _load_numbered("phase1_smoke", "1_4_smoke_dataloader.py")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _schema(path: Path, version: str, names: list[str]) -> None:
    _write_json(
        path,
        {
            "schema_version": version,
            "channels": [
                {"index": index, "name": name} for index, name in enumerate(names)
            ],
        },
    )


def _h5_pair(
    image_path: Path,
    mask_path: Path,
    channels: int,
    *,
    pixel_valid: bool = False,
    label_valid: bool = False,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(channels * 16, dtype=np.float32).reshape(channels, 4, 4)
    values[:, 0, 0] = np.nan
    valid = np.ones(values.shape, dtype=np.uint8)
    valid[:, 0, 0] = 0
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("image", data=values)
        handle.create_dataset("channel_valid", data=np.ones(channels, dtype=np.uint8))
        if pixel_valid:
            handle.create_dataset("pixel_valid", data=valid)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    with h5py.File(mask_path, "w") as handle:
        handle.create_dataset("mask", data=mask)
        if label_valid:
            label = np.ones((4, 4), dtype=np.uint8)
            label[0, 0] = 0
            handle.create_dataset("valid_mask", data=label)


def build_fixture(root: Path) -> None:
    gdcld = root / "GDCLD"
    _schema(
        gdcld / "hdf5/channel_schema.json",
        "gdcld_rgb_mask_valid_hdf5_v1",
        ["Red", "Green", "Blue"],
    )
    _h5_pair(
        gdcld / "hdf5/image/train/gd.h5",
        gdcld / "hdf5/mask/train/gd.h5",
        3,
        label_valid=True,
    )
    _write_jsonl(
        gdcld / "jsonl/sample_index_train.jsonl",
        [
            {
                "sample_key": "gd",
                "split": "train",
                "image_hdf5": "hdf5/image/train/gd.h5",
                "mask_hdf5": "hdf5/mask/train/gd.h5",
                "has_landslide": True,
                "positive_pixel_count": 4,
                "source_origin": "future_work",
                "region": "fixture",
                "scene": "scene",
                "record_sha256": "gd-record",
            }
        ],
    )

    lmhld = root / "LMHLD"
    _schema(
        lmhld / "hdf5/channel_schema.json",
        "lmhld_blue_green_red_nir_v1",
        ["Blue", "Green", "Red", "NIR"],
    )
    _h5_pair(
        lmhld / "hdf5/image/train/lm.h5",
        lmhld / "hdf5/mask/train/lm.h5",
        4,
    )
    _write_jsonl(
        lmhld / "jsonl/sample_index_train.jsonl",
        [
            {
                "sample_key": "lm",
                "split": "train",
                "image_hdf5": "hdf5/image/train/lm.h5",
                "mask_hdf5": "hdf5/mask/train/lm.h5",
                "has_landslide": True,
                "positive_pixel_count": 4,
                "source_index": 0,
                "record_sha256": "lm-record",
            }
        ],
    )

    landslidebench = root / "LandslideBench_agent"
    _schema(
        landslidebench / "hdf5/channel_schema.json",
        "landslidebench_agent_rgb_mask_text_v1",
        ["R", "G", "B"],
    )
    _h5_pair(
        landslidebench / "hdf5/image/lb.h5",
        landslidebench / "hdf5/mask/lb.h5",
        3,
    )
    _write_jsonl(
        landslidebench / "jsonl/sample_index_train.jsonl",
        [
            {
                "sample_key": "lb",
                "split": "train",
                "image_hdf5": "hdf5/image/lb.h5",
                "mask_hdf5": "hdf5/mask/lb.h5",
                "has_landslide": True,
                "mask_positive_pixel_count": 4,
                "location_key": "location",
            }
        ],
    )

    l4s = root / "landslide4sense"
    _schema(
        l4s / "hdf5/channel_schema.json",
        "landslide4sense_fixed_channels_v1",
        [f"B{index:02d}" for index in range(1, 13)] + ["slope", "DEM"],
    )
    _h5_pair(
        l4s / "hdf5/image/landslide/l4.h5",
        l4s / "hdf5/mask/landslide/l4.h5",
        14,
    )
    _write_jsonl(
        l4s / "hdf5/conversion_manifest.jsonl",
        [
            {
                "sample_key": "l4",
                "sample_id": 1,
                "image_hdf5": "image/landslide/l4.h5",
                "mask_hdf5": "mask/landslide/l4.h5",
                "positive_pixel_count": 4,
                "subset": "landslide",
                "status": "converted",
            }
        ],
    )

    multimodal = root / "multimodal-landslide-dataset"
    _schema(
        multimodal / "hdf5/channel_schema.json",
        "multimodal_landslide_rgb_dem_insar_hdf5_v1",
        ["Red", "Green", "Blue", "DEM", "InSAR_mean_LOS_velocity_encoded"],
    )
    _h5_pair(
        multimodal / "hdf5/image/train/mm.h5",
        multimodal / "hdf5/mask/train/mm.h5",
        5,
        pixel_valid=True,
        label_valid=True,
    )
    _write_jsonl(
        multimodal / "jsonl/sample_index_train.jsonl",
        [
            {
                "sample_key": "mm",
                "source_key": "MM_1",
                "split": "train",
                "image_hdf5": "hdf5/image/train/mm.h5",
                "mask_hdf5": "hdf5/mask/train/mm.h5",
                "has_landslide": True,
                "positive_pixel_count": 4,
                "record_sha256": "mm-record",
            }
        ],
    )


class BenchmarkPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.datasets = self.root / "datasets"
        build_fixture(self.datasets)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, name: str, mode: str = "small") -> Path:
        return BUILDER.build_benchmark(
            datasets_root=self.datasets,
            output_base=self.root / name,
            mode=mode,
            patch_size=8,
            small_per_source=1,
            seed=20260724,
            split_seed=20260724,
            shard_target_mib=1,
        )

    def test_resize_clears_invalid_and_mask_is_binary(self) -> None:
        values = np.ones((2, 4, 4), dtype=np.float32)
        values[0, 0, 0] = np.nan
        valid = np.ones_like(values, dtype=np.uint8)
        valid[:, 0, 0] = 0
        resized, resized_valid, _ = resize_continuous_with_validity(
            values, valid, np.ones(2, dtype=np.uint8), 8
        )
        self.assertTrue(np.isfinite(resized).all())
        self.assertTrue(np.all(resized[resized_valid == 0] == 0))
        mask = resize_binary_mask(
            np.array([[0, 1], [1, 0]], dtype=np.uint8), None, 8
        )
        self.assertEqual(mask.shape, (1, 8, 8))
        self.assertLessEqual(set(np.unique(mask).tolist()), {0, 1})

    def test_five_source_build_is_deterministic_and_loadable(self) -> None:
        first = self._build("first")
        second = self._build("second")
        first_rows = read_jsonl(first / "index.jsonl")
        second_rows = read_jsonl(second / "index.jsonl")
        self.assertEqual(
            [row["sample_id"] for row in first_rows],
            [row["sample_id"] for row in second_rows],
        )
        self.assertEqual(
            (first / "index.jsonl").read_bytes(),
            (second / "index.jsonl").read_bytes(),
        )
        report = VALIDATOR.validate_benchmark(first, deep=True)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["checked_samples"], 5)
        dataset = BenchmarkDataset(first)
        batch = collate_benchmark_samples([dataset[index] for index in range(5)])
        self.assertEqual(tuple(batch["mask"].shape), (5, 1, 8, 8))
        self.assertEqual(
            sorted(batch["auxiliaries"]),
            ["dem", "insar_velocity", "slope"],
        )
        smoke = SMOKE.smoke(first)
        self.assertEqual(smoke["status"], "pass")
        with self.assertRaises(FileExistsError):
            self._build("first")

    def test_small_and_full_share_schema(self) -> None:
        small = self._build("small-build", "small")
        full = self._build("full-build", "full")
        small_keys = set(read_jsonl(small / "index.jsonl")[0])
        full_keys = set(read_jsonl(full / "index.jsonl")[0])
        self.assertEqual(small_keys, full_keys)

    def test_validator_detects_missing_shard_and_illegal_mask(self) -> None:
        valid = self._build("valid")
        missing = self.root / "missing"
        shutil.copytree(valid, missing)
        first_row = read_jsonl(missing / "index.jsonl")[0]
        (missing / first_row["storage"]["shard"]).unlink()
        report = VALIDATOR.validate_benchmark(missing, deep=True)
        self.assertTrue(any("不存在" in error or "缺失" in error for error in report["errors"]))

        illegal = self.root / "illegal"
        shutil.copytree(valid, illegal)
        first_row = read_jsonl(illegal / "index.jsonl")[0]
        with h5py.File(illegal / first_row["storage"]["shard"], "r+") as handle:
            handle["mask"][int(first_row["storage"]["row"]), 0, 0, 0] = 2
        report = VALIDATOR.validate_benchmark(illegal, deep=True)
        self.assertTrue(any("mask" in error for error in report["errors"]))

    def test_validator_detects_shape_and_validity_errors(self) -> None:
        valid = self._build("valid-shape")
        corrupt = self.root / "corrupt-shape"
        shutil.copytree(valid, corrupt)
        first_row = read_jsonl(corrupt / "index.jsonl")[0]
        first_row["optical"]["shape"] = [99, 8, 8]
        index_rows = read_jsonl(corrupt / "index.jsonl")
        index_rows[0] = first_row
        _write_jsonl(corrupt / "index.jsonl", index_rows)
        shard_path = corrupt / first_row["storage"]["shard"]
        with h5py.File(shard_path, "r+") as handle:
            handle["optical_pixel_valid"][
                int(first_row["storage"]["row"]), 0, 0, 0
            ] = 2
        report = VALIDATOR.validate_benchmark(corrupt, deep=True)
        self.assertTrue(any("shape" in error for error in report["errors"]))
        self.assertTrue(any("validity" in error for error in report["errors"]))

    def test_validator_detects_corrupt_index(self) -> None:
        valid = self._build("valid-index")
        corrupt = self.root / "corrupt-index"
        shutil.copytree(valid, corrupt)
        (corrupt / "index.jsonl").write_text("{broken\n", encoding="utf-8")
        report = VALIDATOR.validate_benchmark(corrupt, deep=True)
        self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
