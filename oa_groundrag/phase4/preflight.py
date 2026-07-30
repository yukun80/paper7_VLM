"""Benchmark/model/output 的轻量 preflight；不做 full validation。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oa_groundrag.phase3.common import (
    first_symlink_component,
    portable_relative_path,
    read_json,
)

from .config import Phase4Config
from .contracts import (
    SUPPORTED_CANONICAL_SCHEMA,
    SUPPORTED_MANIFEST_SCHEMA,
    DataMode,
)
from .errors import PreflightError, ReasonCode


@dataclass(frozen=True)
class BenchmarkIdentity:
    root: Path
    manifest_schema: str
    canonical_schema: str
    build_id: str
    payload_sha256: str
    benchmark_scope: str
    source_roots_embedded: bool
    deep_validation_saved: bool
    formal_acceptance_eligible: bool
    formal_acceptance_blockers: tuple[str, ...]
    record_count: int
    parent_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_schema": self.manifest_schema,
            "canonical_schema": self.canonical_schema,
            "build_id": self.build_id,
            "payload_sha256": self.payload_sha256,
            "benchmark_scope": self.benchmark_scope,
            "source_roots_embedded": self.source_roots_embedded,
            "deep_validation_saved": self.deep_validation_saved,
            "formal_acceptance_eligible": self.formal_acceptance_eligible,
            "formal_acceptance_blockers": list(
                self.formal_acceptance_blockers
            ),
            "record_count": self.record_count,
            "parent_count": self.parent_count,
        }


def _regular_file(root: Path, relative: str, *, location: str) -> Path:
    pure = portable_relative_path(relative, location=location)
    path = root.joinpath(*pure.parts)
    linked = first_symlink_component(path)
    if linked is not None:
        raise PreflightError(
            ReasonCode.OUTPUT_LINK,
            f"{location}: 路径含链接组件 {linked}",
        )
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise PreflightError(
            ReasonCode.PATH_ESCAPE,
            f"{location}: 路径逃逸",
        ) from error
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PreflightError(
            ReasonCode.OUTPUT_LINK,
            f"{location}: 必须是普通单链接文件",
        )
    return path


def inspect_benchmark_identity(config: Phase4Config) -> BenchmarkIdentity:
    root = config.data.benchmark_root
    linked = first_symlink_component(root)
    if linked is not None:
        raise PreflightError(
            ReasonCode.OUTPUT_LINK,
            f"Benchmark root 含链接组件：{linked}",
        )
    if root.is_symlink() or not root.is_dir():
        raise PreflightError(
            ReasonCode.ASSET_MISSING,
            f"Benchmark root 不存在或不是普通目录：{root}",
        )
    if ".staging" in root.name or root.name.startswith("."):
        raise PreflightError(
            ReasonCode.OUTPUT_LINK,
            f"Benchmark root 疑似 staging：{root}",
        )
    manifest = read_json(_regular_file(root, "manifest.json", location="manifest"))
    expected_identity = {
        "schema_version": config.data.expected_manifest_schema,
        "canonical_schema_version": config.data.expected_canonical_schema,
        "build_id": config.data.expected_build_id,
        "payload_root_sha256": config.data.expected_payload_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_identity.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "Benchmark identity 与配置不一致",
            details=mismatches,
        )
    if (
        expected_identity["schema_version"] != SUPPORTED_MANIFEST_SCHEMA
        or expected_identity["canonical_schema_version"]
        != SUPPORTED_CANONICAL_SCHEMA
    ):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "phase4 不支持配置中的 manifest/canonical schema",
        )
    layout = manifest.get("layout")
    if not isinstance(layout, dict):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "manifest 缺少 layout",
        )
    for name in ("canonical_schema", "validation", "statistics"):
        if not isinstance(layout.get(name), str):
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                f"manifest layout 缺少 {name}",
            )
    schema = read_json(
        _regular_file(
            root,
            layout["canonical_schema"],
            location="layout.canonical_schema",
        )
    )
    if (
        schema.get("$id") != config.data.expected_canonical_schema
        or not isinstance(schema.get("properties"), dict)
        or schema["properties"].get("schema_version", {}).get("const")
        != config.data.expected_canonical_schema
    ):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "canonical schema 文件 identity 不一致",
        )
    validation = read_json(
        _regular_file(
            root,
            layout["validation"],
            location="layout.validation",
        )
    )
    if (
        validation.get("schema_version")
        != "oa_landslidedesc.validation.v3"
        or validation.get("deep") is not True
        or validation.get("errors") != []
        or validation.get("warnings") != []
        or validation.get("payload_root_sha256")
        != config.data.expected_payload_sha256
    ):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "saved validation identity/status 不满足已发布输入合同",
        )
    statistics = read_json(
        _regular_file(
            root,
            layout["statistics"],
            location="layout.statistics",
        )
    )
    if statistics.get("schema_version") != "oa_landslidedesc.statistics.v3":
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "statistics schema identity 不一致",
        )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "manifest 缺少 counts",
        )
    if (
        isinstance(counts.get("records"), bool)
        or not isinstance(counts.get("records"), int)
        or isinstance(counts.get("parents"), bool)
        or not isinstance(counts.get("parents"), int)
        or statistics.get("record_count") != counts.get("records")
        or statistics.get("parent_count") != counts.get("parents")
    ):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "manifest/statistics count identity 不一致",
        )
    blockers_value = manifest.get("formal_acceptance_blockers")
    if not isinstance(blockers_value, list) or not all(
        isinstance(item, str) for item in blockers_value
    ):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "formal_acceptance_blockers 合同非法",
        )
    if (
        manifest.get("source_roots_embedded") is not False
        or manifest.get("scope_validation_complete") is not True
        or manifest.get("formal_acceptance_eligible") is not False
        or manifest.get("official_upstream_evaluation_reproducible")
        is not False
    ):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "已发布 External Benchmark 的 scope/formal flags 不匹配",
        )
    return BenchmarkIdentity(
        root=root.resolve(),
        manifest_schema=str(manifest["schema_version"]),
        canonical_schema=str(manifest["canonical_schema_version"]),
        build_id=str(manifest["build_id"]),
        payload_sha256=str(manifest["payload_root_sha256"]),
        benchmark_scope=str(manifest.get("benchmark_scope")),
        source_roots_embedded=bool(manifest.get("source_roots_embedded")),
        deep_validation_saved=True,
        formal_acceptance_eligible=bool(
            manifest.get("formal_acceptance_eligible")
        ),
        formal_acceptance_blockers=tuple(blockers_value),
        record_count=int(statistics["record_count"]),
        parent_count=int(statistics["parent_count"]),
    )


def run_preflight(
    config: Phase4Config,
    *,
    require_new_output: bool = True,
) -> dict[str, Any]:
    identity = inspect_benchmark_identity(config)
    if identity.source_roots_embedded:
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "phase4 只接受 source-hidden Benchmark",
        )
    if require_new_output and (
        config.run.output_root.exists() or config.run.output_root.is_symlink()
    ):
        raise PreflightError(
            ReasonCode.OUTPUT_EXISTS,
            f"fresh run 拒绝已有 output_root：{config.run.output_root}",
        )
    linked_output = first_symlink_component(config.run.output_root)
    if linked_output is not None:
        raise PreflightError(
            ReasonCode.OUTPUT_LINK,
            f"output_root 含链接组件：{linked_output}",
        )
    if config.run.resume_checkpoint is not None:
        if require_new_output:
            raise PreflightError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "resume 必须由专用恢复路径执行，不能冒充 fresh run",
            )
        checkpoint = config.run.resume_checkpoint
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise PreflightError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                f"resume checkpoint 不存在或是链接：{checkpoint}",
            )
    for path, label in (
        (config.model.path, "model.path"),
        (config.model.processor_path, "model.processor_path"),
    ):
        linked = first_symlink_component(path)
        if linked is not None:
            raise PreflightError(
                ReasonCode.OUTPUT_LINK,
                f"{label} 含链接组件：{linked}",
            )
        if path.is_symlink() or not path.is_dir():
            raise PreflightError(
                ReasonCode.ASSET_MISSING,
                f"{label} 必须是本地普通目录：{path}",
            )
    if config.run.mode is DataMode.EXTERNAL_GENERIC:
        if identity.benchmark_scope != "external_train_val":
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                "External run 要求 benchmark_scope=external_train_val",
            )
        if set(config.data.roles) - {"external_train", "external_val"}:
            raise PreflightError(
                ReasonCode.OA_ROLE_FORBIDDEN,
                "External run roles 非法",
            )
    else:
        if (
            "oa_component_disabled" in identity.formal_acceptance_blockers
            or identity.benchmark_scope == "external_train_val"
        ):
            raise PreflightError(
                ReasonCode.OA_COMPONENT_DISABLED,
                "当前 Benchmark 不含 OA mask-grounded split，OA preflight 失败",
            )
        if "oa_test" in config.data.roles:
            raise PreflightError(
                ReasonCode.OA_TEST_SEALED,
                "oa_test 保持封存",
            )
    return {
        "status": "ok",
        "mode": config.run.mode.value,
        "mask_mode": config.run.mask_mode.value,
        "benchmark": identity.to_dict(),
        "config_semantic_sha256": config.semantic_sha256,
        "formal_acceptance": False,
    }
