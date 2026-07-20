"""SAMI-GroundSegDesc 唯一命令行入口。

用途：P1.1 只读审计 raw source 与许可证门禁。
推荐命令：``sami-gsd data audit --config configs/benchmark_v3_small.yaml --output-dir <new-dir>``。
输入：严格校验的 YAML 配置及只读数据根；输出：原子发布的 inventory、registry、license report 和 manifest。
写行为：拒绝覆盖既有输出目录，不修改 raw data；工作流阶段：P1.1。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sami_gsd import __version__
from sami_gsd.contracts.config import load_audit_config, resolve_root
from sami_gsd.data.audit import audit_sources


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch one explicitly registered command."""

    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
