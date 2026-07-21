"""Synthetic single-time Sen12 source-loader tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

from sami_gsd.contracts.config import SourceConfig
from sami_gsd.data.materialize import materialize_spatial_parent
from sami_gsd.data.source_loaders.sen12 import load_sen12_parents
def sen12_source_config() -> SourceConfig:
    """Return a provenance-only synthetic configuration with the real source key."""

    return SourceConfig.model_validate(
        {
            "source_key": "sen12_landslides",
            "enabled": True,
            "task_roles": ["inventory", "t1"],
            "provenance": {
                "source_key": "sen12_landslides",
                "source_name": "Synthetic Sen12 fixture",
                "source_root": "datasets/Sen12Landslides",
                "source_document": None,
                "citation_key": "sen12_landslides",
                "upstream_url": None,
                "provenance_notes": "synthetic local research fixture",
            },
        }
    )


def write_sen12_file(
    path: Path,
    *,
    satellite: str,
    annotated: str = "True",
    event_date: str = "2020-01-10",
) -> None:
    """Write one paired time/x/y NetCDF record with source-like metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 3)
        time = dataset.createVariable("time", "i4", ("time",))
        time[:] = [0, 9, 30]
        time.units = "days since 2020-01-01"
        time.calendar = "proleptic_gregorian"
        dataset.createVariable("x", "f8", ("x",))[:] = [100, 110, 120, 130]
        dataset.createVariable("y", "f8", ("y",))[:] = [200, 190, 180]
        spatial_ref = dataset.createVariable("spatial_ref", "i8")
        spatial_ref.GeoTransform = "100.0 10.0 0.0 200.0 0.0 -10.0"
        dataset.annotated = annotated
        dataset.event_date = event_date
        dataset.pre_post_dates = "{'pre': 0, 'post': 2}"
        dataset.satellite = satellite
        dataset.crs = "EPSG:32632"

        names = (
            ("B02", "i2"),
            ("B03", "i2"),
            ("B04", "i2"),
            ("B05", "i2"),
            ("B06", "i2"),
            ("B07", "i2"),
            ("B08", "i2"),
            ("B8A", "i2"),
            ("B11", "i2"),
            ("B12", "i2"),
            ("SCL", "i2"),
        ) if satellite == "s2" else (("VV", "f4"), ("VH", "f4"))
        for offset, (name, dtype) in enumerate(names):
            variable = dataset.createVariable(name, dtype, ("time", "x", "y"))
            variable[:] = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3) + offset
        if satellite == "s2":
            dataset.variables["SCL"][:] = 4
            dataset.variables["SCL"][1, 0, 0] = 9
        dem = dataset.createVariable("DEM", "i2", ("time", "x", "y"))
        dem[:] = 100
        mask = dataset.createVariable("MASK", "u1", ("time", "x", "y"))
        values = np.zeros((3, 4, 3), dtype=np.uint8)
        values[:, 2:, 1:] = 1
        mask[:] = values


class Sen12LoaderTests(unittest.TestCase):
    """Verify one event-nearest slice is selected without change inputs."""

    def test_paired_single_time_record_materializes_with_cloud_validity(self) -> None:
        """S2/ASC/DSC nearest dates and DEM share one explicit source grid."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            write_sen12_file(source_root / "s2/event_s2_1.nc", satellite="s2")
            write_sen12_file(source_root / "s1asc/event_s1asc_1.nc", satellite="s1-asc")
            write_sen12_file(source_root / "s1dsc/event_s1dsc_1.nc", satellite="s1-dsc")
            parents = load_sen12_parents(sen12_source_config(), source_root=source_root, limit=1)
            self.assertEqual(len(parents), 1)
            parent = parents[0]
            self.assertEqual(tuple(modality.modality_id for modality in parent.modalities), (
                "s2_optical", "s1_ascending", "s1_descending", "dem"
            ))
            self.assertTrue(all("2020-01-10" in (modality.acquisition_time or "2020-01-10") for modality in parent.modalities))
            self.assertEqual(parent.global_mask.shape, (3, 4))
            self.assertEqual(parent.modalities[0].valid[0, 0], 0)
            self.assertNotIn("pre_post", repr(parent))

            benchmark_root = Path(directory) / "benchmark"
            materialized = materialize_spatial_parent(parent, benchmark_root=benchmark_root, canvas_hw=(8, 8))
            self.assertEqual(materialized.parent.reference_canvas.crs, "EPSG:32632")
            self.assertEqual(materialized.parent.annotations.global_mask_origin, "official")
            self.assertEqual(materialized.parent.modalities[1].units, "dB")

    def test_unannotated_triplet_is_skipped(self) -> None:
        """String 'False' is parsed explicitly rather than treated as truthy."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            write_sen12_file(source_root / "s2/event_s2_1.nc", satellite="s2", annotated="False")
            write_sen12_file(source_root / "s1asc/event_s1asc_1.nc", satellite="s1-asc", annotated="False")
            write_sen12_file(source_root / "s1dsc/event_s1dsc_1.nc", satellite="s1-dsc", annotated="False")
            self.assertEqual(load_sen12_parents(sen12_source_config(), source_root=source_root, limit=1), ())

    def test_bounded_selection_round_robins_across_events(self) -> None:
        """Small selection does not consume a lexical prefix from one event."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            for event in ("alpha", "beta", "gamma"):
                for sample in ("1", "2"):
                    write_sen12_file(source_root / f"s2/{event}_s2_{sample}.nc", satellite="s2")
                    write_sen12_file(source_root / f"s1asc/{event}_s1asc_{sample}.nc", satellite="s1-asc")
                    write_sen12_file(source_root / f"s1dsc/{event}_s1dsc_{sample}.nc", satellite="s1-dsc")
            first = load_sen12_parents(sen12_source_config(), source_root=source_root, limit=3)
            second = load_sen12_parents(sen12_source_config(), source_root=source_root, limit=3)
            self.assertEqual(
                tuple(parent.parent_id for parent in first),
                ("sen12-alpha-1", "sen12-beta-1", "sen12-gamma-1"),
            )
            self.assertEqual(
                tuple(parent.source.event_id for parent in first),
                tuple(parent.source.event_id for parent in second),
            )

    def test_multiple_event_dates_choose_one_deterministic_nearest_date(self) -> None:
        """Comma-separated source dates select the unique minimum-offset contemporaneous triple."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            for relative, satellite in (
                ("s2/event_s2_1.nc", "s2"),
                ("s1asc/event_s1asc_1.nc", "s1-asc"),
                ("s1dsc/event_s1dsc_1.nc", "s1-dsc"),
            ):
                write_sen12_file(
                    source_root / relative,
                    satellite=satellite,
                    event_date="2019-01-01, 2020-01-10",
                )
            parents = load_sen12_parents(sen12_source_config(), source_root=source_root, limit=1)
            self.assertEqual(parents[0].source.event_id, "event:2020-01-10")

    def test_out_of_window_record_is_excluded_without_blocking_the_source(self) -> None:
        """A sample with no contemporaneous acquisition is a record-level technical exclusion."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            for relative, satellite in (
                ("s2/event_s2_1.nc", "s2"),
                ("s1asc/event_s1asc_1.nc", "s1-asc"),
                ("s1dsc/event_s1dsc_1.nc", "s1-dsc"),
            ):
                write_sen12_file(
                    source_root / relative,
                    satellite=satellite,
                    event_date="2010-01-01",
                )
            self.assertEqual(load_sen12_parents(sen12_source_config(), source_root=source_root, limit=1), ())

    def test_missing_event_date_literal_is_excluded_without_blocking_the_source(self) -> None:
        """The corpus literal ``None`` is missing metadata, not a malformed ISO date."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            for relative, satellite in (
                ("s2/event_s2_1.nc", "s2"),
                ("s1asc/event_s1asc_1.nc", "s1-asc"),
                ("s1dsc/event_s1dsc_1.nc", "s1-dsc"),
            ):
                write_sen12_file(
                    source_root / relative,
                    satellite=satellite,
                    event_date="None",
                )
            self.assertEqual(load_sen12_parents(sen12_source_config(), source_root=source_root, limit=1), ())

    def test_zero_valid_s2_reference_is_excluded_without_blocking_the_source(self) -> None:
        """An all-cloud selected S2 slice cannot become a zero-valid reference canvas."""

        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "Sen12Landslides"
            for relative, satellite in (
                ("s2/event_s2_1.nc", "s2"),
                ("s1asc/event_s1asc_1.nc", "s1-asc"),
                ("s1dsc/event_s1dsc_1.nc", "s1-dsc"),
            ):
                write_sen12_file(source_root / relative, satellite=satellite)
            with netCDF4.Dataset(source_root / "s2/event_s2_1.nc", mode="a") as dataset:
                dataset.variables["SCL"][1, :, :] = 9
            self.assertEqual(load_sen12_parents(sen12_source_config(), source_root=source_root, limit=1), ())


if __name__ == "__main__":
    unittest.main()
