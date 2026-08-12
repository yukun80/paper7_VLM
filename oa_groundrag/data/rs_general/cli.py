"""RS-GeneralDesc audit/build/repackage/validate/export 薄 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .builder import audit_sources, build_benchmark
from .io import atomic_write_json
from .config import load_build_config, load_export_config
from .errors import RSGeneralDescError, ReasonCode
from .exporter import export_qwen
from .repackage import repackage_benchmark
from .validator import validate_benchmark


def _report(value: dict[str, object], path: Path | None) -> None:
    if path is not None:
        if path.exists() or path.is_symlink():
            raise RSGeneralDescError(
                ReasonCode.OUTPUT_EXISTS,
                f"拒绝覆盖已有 report：{path}",
            )
        atomic_write_json(path, value)
        summary = {
            "report": str(path),
            "schema_version": value.get("schema_version"),
            "status": value.get("status"),
            "errors": len(value.get("errors", []))
            if isinstance(value.get("errors"), list)
            else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 RS-GeneralDesc 三库视觉证据→文本多任务 Benchmark："
            "审计、构建、验证和 task-aware Qwen 导出。"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="只读审计配置中的真实 source")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--deep", action="store_true")
    audit.add_argument("--report", type=Path)

    build = subparsers.add_parser(
        "build",
        help="按配置构建 RS-GeneralDesc external train/val",
    )
    build.add_argument("--config", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="只读验证已发布产物")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--deep", action="store_true")
    validate.add_argument("--report", type=Path)

    repackage = subparsers.add_parser(
        "repackage",
        help="将锁定身份的冻结 External payload 重发布为 native v1",
    )
    repackage.add_argument("--source-root", type=Path, required=True)
    repackage.add_argument("--target-root", type=Path, required=True)
    repackage.add_argument("--expected-manifest-sha256", required=True)
    repackage.add_argument("--expected-build-id", required=True)
    repackage.add_argument("--expected-payload-sha256", required=True)
    repackage.add_argument("--expected-hash-manifest-sha256", required=True)

    export = subparsers.add_parser(
        "export",
        help="从 canonical 生成显式任务集合的 task-aware Qwen messages",
    )
    export.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        config = load_build_config(args.config)
        _report(audit_sources(config, deep=args.deep), args.report)
        return 0
    if args.command == "build":
        config = load_build_config(args.config)
        target = build_benchmark(config)
        print(f"构建完成：{target}")
        return 0
    if args.command == "validate":
        report = validate_benchmark(args.root, deep=args.deep)
        _report(report, args.report)
        return 1 if report["errors"] else 0
    if args.command == "repackage":
        target = repackage_benchmark(
            args.source_root,
            args.target_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_build_id=args.expected_build_id,
            expected_payload_sha256=args.expected_payload_sha256,
            expected_hash_manifest_sha256=(
                args.expected_hash_manifest_sha256
            ),
        )
        print(f"原生重发布完成：{target}")
        return 0
    if args.command == "export":
        config = load_export_config(args.config)
        target = export_qwen(config)
        print(f"Qwen 导出完成：{target}")
        return 0
    raise AssertionError(args.command)


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except RSGeneralDescError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    except Exception as error:
        print(f"UNEXPECTED_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
