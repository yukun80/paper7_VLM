from __future__ import annotations

import contextlib
import io
import unittest

from sami_gsd.cli import build_parser


class BenchmarkV4CliTests(unittest.TestCase):
    def test_build_and_validate_routes_are_explicit(self) -> None:
        parser = build_parser()
        build = parser.parse_args(
            [
                "benchmark",
                "build",
                "--datasets-root",
                "/tmp/datasets",
                "--benchmark-root",
                "/tmp/benchmark",
            ]
        )
        self.assertEqual(build.benchmark_command, "build")
        validate = parser.parse_args(
            [
                "benchmark",
                "validate",
                "--benchmark",
                "/tmp/benchmark/sami_landslide_hdf5_v4/small",
                "--datasets-root",
                "/tmp/datasets",
                "--output",
                "/tmp/report.json",
            ]
        )
        self.assertEqual(validate.benchmark_command, "validate")

    def test_retired_model_route_is_not_exposed(self) -> None:
        parser = build_parser()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(["model", "smoke"])


if __name__ == "__main__":
    unittest.main()
