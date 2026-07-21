"""SAMI-GroundSegDesc 唯一命令行入口。

用途：P1 raw audit、Canonical Benchmark v3 原子构建与独立验证。
推荐命令：``sami-gsd data build --config configs/benchmark_v3_small.yaml``。
输入：严格校验 YAML 与只读 raw source；输出：配置指定的新 Benchmark 目录和可重放报告。
写行为：拒绝覆盖既有目录且不修改 raw data；工作流阶段：P1。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sami_gsd import __version__
from sami_gsd.contracts.config import BenchmarkAuditConfig, load_audit_config, resolve_root
from sami_gsd.contracts.language import DescriptionSourceRecord
from sami_gsd.data.audit import audit_sources
from sami_gsd.data.builder import build_canonical_benchmark
from sami_gsd.data.language_subset import build_description_subset
from sami_gsd.data.source_loaders import load_resolved_spatial_parents
from sami_gsd.data.validation import validate_published_benchmark


def _repository_root() -> Path:
    """Return the source-checkout root for portable config defaults."""

    return Path(__file__).resolve().parents[2]


def _run_data_audit(arguments: argparse.Namespace) -> int:
    """Execute the read-only P1.1 audit command."""

    config_path = arguments.config.resolve()
    config = load_audit_config(config_path)
    repository_root = _repository_root()
    datasets_root = resolve_root(
        config.datasets_root,
        repository_root=repository_root,
        override=arguments.datasets_root,
    )
    manifest = audit_sources(
        config,
        datasets_root=datasets_root,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(manifest, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


def _runtime_roots(arguments: argparse.Namespace) -> tuple[BenchmarkAuditConfig, Path, Path]:
    """Resolve one validated config and runtime-only dataset/benchmark roots."""

    config = load_audit_config(arguments.config.resolve())
    repository_root = _repository_root()
    datasets_root = resolve_root(
        config.datasets_root,
        repository_root=repository_root,
        override=getattr(arguments, "datasets_root", None),
    )
    benchmark_root = resolve_root(
        config.benchmark_root,
        repository_root=repository_root,
        override=getattr(arguments, "benchmark_root", None),
    )
    return config, datasets_root, benchmark_root


def _run_data_build(arguments: argparse.Namespace) -> int:
    """Build a new P1 benchmark only after source/license preflight passes."""

    config, datasets_root, benchmark_root = _runtime_roots(arguments)
    parent_inputs = load_resolved_spatial_parents(config, datasets_root=datasets_root)
    subset = build_description_subset(
        config,
        datasets_root=datasets_root,
        limit_per_component=config.build.small_max_parents_per_source,
    )
    description_records = tuple(
        DescriptionSourceRecord.model_validate(record) for record in subset["records"]
    )
    manifest = build_canonical_benchmark(
        config,
        parent_inputs=parent_inputs,
        description_records=description_records,
        output_dir=benchmark_root / config.benchmark_relative_path,
        schemas_root=_repository_root() / "schemas",
    )
    print(json.dumps(manifest, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


def _run_data_validate(arguments: argparse.Namespace) -> int:
    """Replay validation and manifest hashes without modifying the benchmark."""

    config, _, benchmark_root = _runtime_roots(arguments)
    report = validate_published_benchmark(
        benchmark_root / config.benchmark_relative_path,
        schemas_root=_repository_root() / "schemas",
    )
    print(json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0 if not report["errors"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the sole project CLI parser without reading config or data."""

    parser = argparse.ArgumentParser(prog="sami-gsd", description="SAMI-GroundSegDesc greenfield CLI")
    parser.add_argument("--version", action="version", version=__version__)
    top_level = parser.add_subparsers(dest="command", required=True)

    data_parser = top_level.add_parser("data", help="raw-source and license operations")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    audit_parser = data_commands.add_parser("audit", help="read-only deterministic raw-source audit")
    audit_parser.add_argument("--config", type=Path, required=True, help="Benchmark v3 audit YAML")
    audit_parser.add_argument("--datasets-root", type=Path, help="runtime-only override for the raw dataset root")
    audit_parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; overwrite is forbidden")
    audit_parser.set_defaults(handler=_run_data_audit)

    build_parser = data_commands.add_parser("build", help="atomically build Canonical Benchmark v3")
    build_parser.add_argument("--config", type=Path, required=True, help="Benchmark v3 YAML")
    build_parser.add_argument("--datasets-root", type=Path, help="runtime-only raw dataset root override")
    build_parser.add_argument("--benchmark-root", type=Path, help="runtime-only benchmark root override")
    build_parser.set_defaults(handler=_run_data_build)

    validate_parser = data_commands.add_parser("validate", help="independently replay a published benchmark")
    validate_parser.add_argument("--config", type=Path, required=True, help="Benchmark v3 YAML")
    validate_parser.add_argument("--datasets-root", type=Path, help="accepted for shared root resolution")
    validate_parser.add_argument("--benchmark-root", type=Path, help="runtime-only benchmark root override")
    validate_parser.set_defaults(handler=_run_data_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch one explicitly registered command."""

    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
