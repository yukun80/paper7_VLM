"""SAMI-GroundSegDesc segmentation-only command-line entrypoint.

用途：构建或独立验证 P1 HDF5 Benchmark v4 Small。
推荐命令：``python -m sami_gsd.cli benchmark --help``。
输入：严格 YAML、只读 HDF5 source、全新 benchmark/output 路径。
输出：自包含 HDF5 benchmark 或仓库 ``outputs/`` 下的独立验证报告。
写行为：所有入口 fail-on-existing；阶段：P1。模型命令在 P1 验收前不暴露。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sami_gsd import __version__


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_benchmark_build(arguments: argparse.Namespace) -> int:
    from sami_gsd.contracts.benchmark_v4_config import load_benchmark_v4_config
    from sami_gsd.data.benchmark_v4 import build_benchmark_v4

    repository_root = _repository_root()
    config_path = arguments.config.resolve()
    config = load_benchmark_v4_config(
        config_path,
        repository_root=repository_root,
    )
    result = build_benchmark_v4(
        config,
        config_path=config_path,
        datasets_root=arguments.datasets_root.resolve(),
        benchmark_root=arguments.benchmark_root.resolve(),
        schemas_root=repository_root / "schemas",
        source_contract_path=repository_root / config.source_contract_path,
        source_inventory_path=repository_root / config.source_inventory_path,
    )
    print(json.dumps(result, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


def _run_benchmark_validate(arguments: argparse.Namespace) -> int:
    from sami_gsd.data.benchmark_v4_validation import validate_benchmark_v4

    report = validate_benchmark_v4(
        arguments.benchmark.resolve(),
        datasets_root=arguments.datasets_root.resolve(),
        schemas_root=arguments.schemas_root.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0 if not report["errors"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Create the P1-only parser without reading config, data, or HDF5."""

    repository_root = _repository_root()
    parser = argparse.ArgumentParser(
        prog="sami-gsd",
        description="SAMI-GroundSegDesc HDF5-first segmentation CLI",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser(
        "benchmark",
        help="P1 Benchmark v4 operations",
    )
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
    )

    build = benchmark_commands.add_parser(
        "build",
        help="build a new self-contained Benchmark v4 Small",
    )
    build.add_argument(
        "--config",
        type=Path,
        default=repository_root / "configs/benchmark_v4_small.yaml",
        help="strict Benchmark v4 YAML",
    )
    build.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="read-only directory corresponding to logical datasets/",
    )
    build.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="benchmark parent; output v4/small must not exist",
    )
    build.set_defaults(handler=_run_benchmark_build)

    validate = benchmark_commands.add_parser(
        "validate",
        help="independently replay an existing Benchmark v4",
    )
    validate.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="exact immutable sami_landslide_hdf5_v4/small directory",
    )
    validate.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="read-only directory corresponding to logical datasets/",
    )
    validate.add_argument(
        "--schemas-root",
        type=Path,
        default=repository_root / "schemas",
        help="repository Benchmark v4 schema directory",
    )
    validate.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new report path below repository outputs/",
    )
    validate.set_defaults(handler=_run_benchmark_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
