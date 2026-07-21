"""P1.3 strict source-adapter registry and bounded extraction tests."""

from __future__ import annotations

import binascii
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from pydantic import ValidationError

from sami_gsd.contracts.config import load_audit_config
from sami_gsd.contracts.sources import CanonicalParentCandidate
from sami_gsd.data.adapters import audit_source_samples, build_source_adapter_registry
from sami_gsd.data.adapters.formats import (
    read_geotiff_header,
    read_hdf5_dataset_header,
    read_image_header,
    read_netcdf_variable_header,
    read_npy_header,
)
from sami_gsd.data.adapters.registry import SourceAdapterRegistry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Build one checksummed PNG chunk for a tiny synthetic fixture."""

    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload))


def write_png(path: Path, *, height: int = 2, width: int = 3, grayscale: bool = False) -> None:
    """Write a deterministic RGB or grayscale PNG without Pillow."""

    channels = 1 if grayscale else 3
    color_type = 0 if grayscale else 2
    rows = b"".join(b"\x00" + bytes([row + 1]) * (width * channels) for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_npy(path: Path, *, shape: tuple[int, ...], dtype: str = "<f4") -> None:
    """Write a minimal NPY v1 array adequate for header-only extraction."""

    mapping = {"descr": dtype, "fortran_order": False, "shape": shape}
    header = repr(mapping)
    padding = 64 - ((10 + len(header) + 1) % 64)
    encoded_header = (header + " " * padding + "\n").encode("latin1")
    item_size = int(dtype[-1])
    item_count = 1
    for dimension in shape:
        item_count *= dimension
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded_header)) + encoded_header + bytes(item_count * item_size))


def write_hdf5(path: Path, *, key: str, shape: tuple[int, ...], dtype: str) -> None:
    """Write one deterministic HDF5 dataset for header-only adapter tests."""

    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(key, data=np.zeros(shape, dtype=dtype))
        dataset.attrs["fixture"] = "p1"


def write_geotiff(path: Path, *, channels: int, dtype: str, nodata: int | float) -> None:
    """Write a small co-registered GeoTIFF with explicit CRS/transform/nodata."""

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=3,
        width=4,
        count=channels,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(100.0, 30.0, 0.1, 0.1),
        nodata=nodata,
    ) as dataset:
        dataset.write(np.zeros((channels, 3, 4), dtype=dtype))


def write_netcdf(path: Path) -> None:
    """Write one metadata-only NetCDF fixture with an explicit time dimension."""

    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("y", 3)
        dataset.createDimension("x", 4)
        variable = dataset.createVariable("MASK", "u1", ("time", "y", "x"))
        variable.setncattr("annotated", "false")


def build_live_layout_fixture(root: Path) -> Path:
    """Create the five implemented layouts plus four explicit blocked roots."""

    datasets = root / "datasets"
    for source in (
        "GDCLD",
        "LMHLD",
        "Sen12Landslides",
        "landslide4sense",
        "multimodal-landslide-dataset",
        "LandslideBench_agent",
        "MMRS-1M",
        "RSGPT",
        "DisasterM3",
    ):
        (datasets / source).mkdir(parents=True)

    write_png(datasets / "GDCLD/train_data/a.tif")
    write_png(datasets / "GDCLD/train_label/a.tif", grayscale=True)

    lmhld = datasets / "LMHLD/LMHLD_dataset_different_patch_sizes/Region_32"
    write_npy(lmhld / "train_images.npy", shape=(2, 4, 2, 3))
    write_npy(lmhld / "train_labels.npy", shape=(2, 1, 2, 3))

    landslide4sense = datasets / "landslide4sense/TrainData"
    write_hdf5(landslide4sense / "img/image_1.h5", key="img", shape=(3, 4, 14), dtype="float32")
    write_hdf5(landslide4sense / "mask/mask_1.h5", key="mask", shape=(3, 4), dtype="uint8")

    multimodal = datasets / "multimodal-landslide-dataset/multimodal-landslide-dataset"
    multimodal.mkdir(parents=True, exist_ok=True)
    (multimodal / "train.txt").write_text("Loess_000001.tif\n", encoding="utf-8")
    write_geotiff(multimodal / "rgb/Loess_000001.tif", channels=3, dtype="int16", nodata=-9999)
    write_geotiff(multimodal / "dem/Loess_000001.tif", channels=1, dtype="float32", nodata=-9999)
    write_geotiff(multimodal / "insar_vel/Loess_000001.tif", channels=1, dtype="uint16", nodata=65535)
    write_geotiff(multimodal / "label/Loess_000001.tif", channels=1, dtype="uint8", nodata=255)

    write_netcdf(datasets / "Sen12Landslides/s2/fixture.nc")

    landslidebench = datasets / "LandslideBench_agent"
    write_png(landslidebench / "images/debris1_Level_16.png")
    write_png(landslidebench / "mask/debris1_Level_16.png", grayscale=True)
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "images/debris1_Level_16.png"},
                    {"type": "text", "text": "audit fixture"},
                ],
            }
        ]
    }
    (landslidebench / "qwen3vl_landslide_train.jsonl").write_text(
        json.dumps(row, allow_nan=False) + "\n", encoding="utf-8"
    )

    mmrs = datasets / "MMRS-1M"
    write_png(mmrs / "caption/nwpu_caption/images/airplane/a.png")
    mmrs_index = mmrs / "json/caption/caption_nwpu.json"
    mmrs_index.parent.mkdir(parents=True)
    mmrs_index.write_text(
        json.dumps(
            [
                {
                    "image": "data/caption/nwpu_caption/images/airplane/a.png",
                    "conversations": [
                        {"from": "human", "value": "Describe the image."},
                        {"from": "gpt", "value": "An airplane."},
                    ],
                }
            ],
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    rsgpt = datasets / "RSGPT"
    write_png(rsgpt / "dataset/RSICap/images/a.png")
    rsgpt_index = rsgpt / "dataset/RSICap/captions.json"
    rsgpt_index.parent.mkdir(parents=True, exist_ok=True)
    rsgpt_index.write_text(
        json.dumps(
            {"annotations": [{"filename": "a.png", "text_output": "A scene."}]},
            allow_nan=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return datasets


class SourceFormatTests(unittest.TestCase):
    """Verify header probes never depend on filename suffixes or decoded arrays."""

    def test_png_signature_and_npy_header_are_read_without_optional_dependencies(self) -> None:
        """A disguised PNG and virtual NPY record retain exact shapes."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.tif"
            array = root / "array.npy"
            write_png(image, height=4, width=5)
            write_npy(array, shape=(3, 4, 5, 6))
            image_header = read_image_header(image)
            npy_header = read_npy_header(array)
            self.assertEqual((image_header.container, image_header.height, image_header.width), ("png", 4, 5))
            self.assertEqual(npy_header.shape, (3, 4, 5, 6))
            self.assertEqual(npy_header.dtype, "<f4")

    def test_declared_data_extra_reads_hdf5_geotiff_and_netcdf_metadata_only(self) -> None:
        """Spatial containers expose grids and attributes without returning arrays."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdf5_path = root / "sample.h5"
            tiff_path = root / "sample.tif"
            netcdf_path = root / "sample.nc"
            write_hdf5(hdf5_path, key="img", shape=(3, 4, 2), dtype="float32")
            write_geotiff(tiff_path, channels=2, dtype="int16", nodata=-9999)
            write_netcdf(netcdf_path)

            hdf5_header = read_hdf5_dataset_header(hdf5_path, internal_key="img")
            tiff_header = read_geotiff_header(tiff_path)
            netcdf_header = read_netcdf_variable_header(netcdf_path, internal_key="MASK")
            self.assertEqual(hdf5_header.shape, (3, 4, 2))
            self.assertEqual((tiff_header.height, tiff_header.width, tiff_header.channel_count), (3, 4, 2))
            self.assertEqual(tiff_header.crs, "EPSG:4326")
            self.assertEqual(netcdf_header.shape, (2, 3, 4))


class SourceRegistryTests(unittest.TestCase):
    """Verify unique coverage and the absence of a generic fallback adapter."""

    def test_registry_covers_exactly_the_nine_live_config_keys(self) -> None:
        """Small/full config source keys map one-to-one to stable descriptors."""

        registry = build_source_adapter_registry()
        config = load_audit_config(REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml")
        self.assertEqual(registry.keys(), tuple(sorted(source.source_key for source in config.sources)))
        self.assertEqual(sum(item.implementation_status == "implemented" for item in registry.descriptors()), 7)
        self.assertEqual(sum(item.implementation_status == "blocked" for item in registry.descriptors()), 2)
        with self.assertRaisesRegex(KeyError, "no source adapter"):
            registry.get("legacy_fallback")

    def test_duplicate_adapter_registration_is_rejected(self) -> None:
        """Two implementations cannot silently compete for one source key."""

        source = build_source_adapter_registry().get("gdcld")
        registry = SourceAdapterRegistry()
        registry.register(source)
        with self.assertRaisesRegex(ValueError, "duplicate source adapter"):
            registry.register(source)


class SourceAuditIntegrationTests(unittest.TestCase):
    """Exercise deterministic extraction, fail-closed blockers and raw immutability."""

    def test_all_nine_sources_are_accounted_and_repeat_hash_is_stable(self) -> None:
        """Five sampled sources plus four blockers produce no extraction errors."""

        with tempfile.TemporaryDirectory() as directory:
            datasets = build_live_layout_fixture(Path(directory))
            config = load_audit_config(REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml")
            before = {path.relative_to(datasets).as_posix(): path.read_bytes() for path in datasets.rglob("*") if path.is_file()}
            first = audit_source_samples(config, datasets_root=datasets, limit_per_source=1)
            second = audit_source_samples(config, datasets_root=datasets, limit_per_source=1)
            after = {path.relative_to(datasets).as_posix(): path.read_bytes() for path in datasets.rglob("*") if path.is_file()}

            self.assertEqual(before, after)
            self.assertEqual(first["aggregate_sha256"], second["aggregate_sha256"])
            self.assertEqual(first["source_count"], 9)
            self.assertEqual(first["implemented_source_count"], 7)
            self.assertEqual(first["sampled_source_count"], 7)
            self.assertEqual(first["blocked_source_count"], 2)
            self.assertEqual(first["missing_source_count"], 0)
            self.assertEqual(first["errors"], [])
            sampled = [source for source in first["sources"] if source["status"] == "sampled"]
            self.assertTrue(all(source["raw_bytes_unchanged"] for source in sampled))
            self.assertTrue(all(source["training_eligible"] is False for source in first["sources"]))
            self.assertTrue(
                all(not Path(source["logical_root"]).is_absolute() for source in first["sources"])
            )

    def test_candidate_contract_rejects_training_promotion(self) -> None:
        """P1.3 projections cannot be edited into training records."""

        with tempfile.TemporaryDirectory() as directory:
            datasets = build_live_layout_fixture(Path(directory))
            config = load_audit_config(REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml")
            adapter = build_source_adapter_registry().get("gdcld")
            source_config = next(source for source in config.sources if source.source_key == "gdcld")
            candidate = adapter.extract_samples(datasets / "GDCLD", source_config, limit=1)[0].canonical_candidate
            payload = candidate.model_dump(mode="json")
            payload["training_eligible"] = True
            with self.assertRaises(ValidationError):
                CanonicalParentCandidate.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
