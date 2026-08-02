"""RS-VLM 的 native Benchmark/model/output 轻量 preflight。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    portable_relative_path,
    read_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.contracts import (
    BENCHMARK_SCOPE,
    BUILD_CONFIG_VERSION,
    STATISTICS_SCHEMA_VERSION,
    VALIDATION_VERSION,
)
from oa_groundrag.phase3.errors import RSGeneralDescError
from oa_groundrag.phase3.hash_ledger import HashLedgerVerifier

from .config import Phase4Config
from .contracts import SUPPORTED_CANONICAL_SCHEMA, SUPPORTED_MANIFEST_SCHEMA
from .errors import PreflightError, ReasonCode


@dataclass(frozen=True)
class BenchmarkIdentity:
    root: Path
    manifest_schema: str
    canonical_schema: str
    build_id: str
    semantic_config_sha256: str
    payload_sha256: str
    hash_manifest_sha256: str
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
            "semantic_config_sha256": self.semantic_config_sha256,
            "payload_sha256": self.payload_sha256,
            "hash_manifest_sha256": self.hash_manifest_sha256,
            "benchmark_scope": self.benchmark_scope,
            "source_roots_embedded": self.source_roots_embedded,
            "deep_validation_saved": self.deep_validation_saved,
            "formal_acceptance_eligible": self.formal_acceptance_eligible,
            "formal_acceptance_blockers": list(self.formal_acceptance_blockers),
            "record_count": self.record_count,
            "parent_count": self.parent_count,
        }

    def training_identity_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class BenchmarkAccess:
    """一次打开的 native Benchmark 身份及其按需 verifier。"""

    identity: BenchmarkIdentity
    manifest: Mapping[str, Any]
    validation: Mapping[str, Any]
    statistics: Mapping[str, Any]
    record_shards: tuple[Mapping[str, Any], ...]
    verifier: HashLedgerVerifier

    @property
    def root(self) -> Path:
        return self.identity.root

    def verify_file(self, relative: str) -> Path:
        try:
            return self.verifier.verify_file(relative)
        except RSGeneralDescError as error:
            details = dict(error.details)
            details.setdefault("path", relative)
            details["ledger_reason_code"] = error.code.value
            _fail("Benchmark 文件与 hash ledger 不一致", details=details)

    def assert_declared_identity(
        self,
        relative: str,
        *,
        size_bytes: object,
        sha256: object,
    ) -> None:
        try:
            self.verifier.assert_declared_identity(
                relative,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        except RSGeneralDescError as error:
            details = dict(error.details)
            details.setdefault("path", relative)
            details["ledger_reason_code"] = error.code.value
            _fail("运行时 media 声明与 hash ledger 不一致", details=details)


def _fail(message: str, *, details: Mapping[str, Any] | None = None) -> None:
    raise PreflightError(
        ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
        message,
        details=details,
    )


def _regular_file(root: Path, relative: str, *, location: str) -> Path:
    try:
        pure = portable_relative_path(relative, location=location)
    except RSGeneralDescError as error:
        _fail(
            f"{location}: 路径合同非法",
            details={"path": relative, "ledger_reason_code": error.code.value},
        )
    path = root.joinpath(*pure.parts)
    linked = first_symlink_component(path)
    if linked is not None:
        raise PreflightError(
            ReasonCode.OUTPUT_LINK,
            f"{location}: 路径含链接组件 {linked}",
            details={"path": relative},
        )
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise PreflightError(
            ReasonCode.PATH_ESCAPE,
            f"{location}: 路径逃逸",
            details={"path": relative},
        ) from error
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PreflightError(
            ReasonCode.ASSET_MISSING,
            f"{location}: 必须是普通单链接文件",
            details={"path": relative},
        )
    return path


def _read_mapping(path: Path, *, location: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except RSGeneralDescError as error:
        _fail(
            f"{location}: 无法严格读取 JSON",
            details={"path": str(path), "ledger_reason_code": error.code.value},
        )
    if not isinstance(value, dict):
        _fail(f"{location}: 必须是对象", details={"path": str(path)})
    return value


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{location}: 必须是对象")
    return value


def _layout_record_shards(
    layout: Mapping[str, Any],
    *,
    record_count: int,
    verifier: HashLedgerVerifier,
) -> tuple[Mapping[str, Any], ...]:
    shards = layout.get("record_shards")
    if not isinstance(shards, list) or not shards:
        _fail("manifest.layout.record_shards 必须是非空列表")
    output: list[Mapping[str, Any]] = []
    seen_paths: set[str] = set()
    total = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {"path", "record_count"}:
            _fail(f"record_shards[{index}] 字段非法")
        relative = shard.get("path")
        count = shard.get("record_count")
        if (
            not isinstance(relative, str)
            or relative in seen_paths
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            _fail(f"record_shards[{index}] identity/count 非法")
        try:
            verifier.assert_file_metadata(relative)
        except RSGeneralDescError as error:
            details = dict(error.details)
            details.setdefault("path", relative)
            details["ledger_reason_code"] = error.code.value
            _fail("manifest record shard 未绑定 hash ledger", details=details)
        seen_paths.add(relative)
        total += count
        output.append(dict(shard))
    if total != record_count:
        _fail(
            "record shard counts 与 manifest record_count 不一致",
            details={"expected": record_count, "actual": total},
        )

    role_map = layout.get("role_to_record_shards")
    if not isinstance(role_map, dict) or set(role_map) != {
        "external_train",
        "external_val",
    }:
        _fail("role_to_record_shards 必须精确包含 train/val")
    referenced: set[str] = set()
    path_order = {
        str(shard["path"]): index for index, shard in enumerate(output)
    }
    for role in ("external_train", "external_val"):
        values = role_map.get(role)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or any(value not in seen_paths for value in values)
        ):
            _fail(f"role_to_record_shards.{role} 非法")
        indices = [path_order[value] for value in values]
        if indices != sorted(indices):
            _fail(f"role_to_record_shards.{role} 顺序与 shard layout 不一致")
        referenced.update(values)
    if referenced != seen_paths:
        _fail("role_to_record_shards 未完整覆盖 record shards")
    return tuple(output)


def open_benchmark_access(config: Phase4Config) -> BenchmarkAccess:
    """打开并绑定 metadata identity；不读取任何 record/asset bytes。"""

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

    manifest_path = _regular_file(root, "manifest.json", location="manifest")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != config.data.expected_manifest_sha256:
        _fail(
            "manifest 文件 SHA-256 与配置不一致",
            details={
                "path": "manifest.json",
                "expected_sha256": config.data.expected_manifest_sha256,
                "actual_sha256": actual_manifest_sha256,
            },
        )
    manifest = _read_mapping(manifest_path, location="manifest")
    expected_identity = {
        "schema_version": config.data.expected_manifest_schema,
        "canonical_schema_version": config.data.expected_canonical_schema,
        "build_id": config.data.expected_build_id,
        "payload_root_sha256": config.data.expected_payload_sha256,
        "hash_manifest_sha256": config.data.expected_hash_manifest_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_identity.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        _fail("Benchmark identity 与配置不一致", details=mismatches)
    if (
        config.data.expected_manifest_schema != SUPPORTED_MANIFEST_SCHEMA
        or config.data.expected_canonical_schema != SUPPORTED_CANONICAL_SCHEMA
    ):
        _fail("RS-VLM 不支持配置中的 manifest/canonical schema")

    layout = _mapping(manifest.get("layout"), location="manifest.layout")
    for name in (
        "canonical_schema",
        "validation",
        "statistics",
        "hashes",
        "build_config",
    ):
        if not isinstance(layout.get(name), str):
            _fail(f"manifest.layout 缺少 {name}")

    try:
        verifier = HashLedgerVerifier(
            root,
            ledger_path=layout["hashes"],
            expected_ledger_sha256=config.data.expected_hash_manifest_sha256,
            expected_root_sha256=config.data.expected_payload_sha256,
        )
    except RSGeneralDescError as error:
        details = dict(error.details)
        details.setdefault("path", str(layout.get("hashes")))
        details["ledger_reason_code"] = error.code.value
        _fail("hash ledger identity/结构不满足合同", details=details)

    def verified_mapping(relative: str, *, location: str) -> dict[str, Any]:
        try:
            path = verifier.verify_file(relative)
        except RSGeneralDescError as error:
            details = dict(error.details)
            details.setdefault("path", relative)
            details["ledger_reason_code"] = error.code.value
            _fail(f"{location}: 与 hash ledger 不一致", details=details)
        return _read_mapping(path, location=location)

    schema = verified_mapping(
        layout["canonical_schema"],
        location="canonical schema",
    )
    properties = schema.get("properties")
    if (
        schema.get("$id") != SUPPORTED_CANONICAL_SCHEMA
        or not isinstance(properties, dict)
        or properties.get("schema_version", {}).get("const")
        != SUPPORTED_CANONICAL_SCHEMA
    ):
        _fail(
            "canonical schema 文件 identity 不一致",
            details={"path": layout["canonical_schema"]},
        )

    build_config = verified_mapping(
        layout["build_config"],
        location="build_config",
    )
    semantic_config_sha = sha256_text(canonical_json(build_config))
    derived_build_id = "build_" + sha256_text(
        canonical_json([semantic_config_sha, config.data.expected_payload_sha256])
    )
    if (
        build_config.get("schema_version") != BUILD_CONFIG_VERSION
        or manifest.get("semantic_config_sha256") != semantic_config_sha
        or manifest.get("build_id") != derived_build_id
    ):
        _fail(
            "build config semantic SHA/build ID 推导不一致",
            details={"path": layout["build_config"]},
        )

    validation_path = _regular_file(
        root,
        layout["validation"],
        location="layout.validation",
    )
    actual_validation_sha256 = sha256_file(validation_path)
    if actual_validation_sha256 != config.data.expected_validation_sha256:
        _fail(
            "saved validation 文件 SHA-256 与配置不一致",
            details={
                "path": layout["validation"],
                "expected_sha256": config.data.expected_validation_sha256,
                "actual_sha256": actual_validation_sha256,
            },
        )
    validation = _read_mapping(validation_path, location="saved validation")
    statistics = verified_mapping(
        layout["statistics"],
        location="statistics",
    )
    counts = _mapping(manifest.get("counts"), location="manifest.counts")
    record_count = counts.get("records")
    parent_count = counts.get("parents")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
        or isinstance(parent_count, bool)
        or not isinstance(parent_count, int)
        or parent_count <= 0
    ):
        _fail("manifest records/parents count 非法")
    if (
        validation.get("schema_version") != VALIDATION_VERSION
        or validation.get("deep") is not True
        or validation.get("errors") != []
        or validation.get("warnings") != []
        or validation.get("payload_root_sha256")
        != config.data.expected_payload_sha256
        or validation.get("hash_manifest_sha256")
        != config.data.expected_hash_manifest_sha256
        or validation.get("record_count") != record_count
        or validation.get("parent_count") != parent_count
        or validation.get("checked_records") != record_count
    ):
        _fail(
            "saved deep validation identity/status 不满足 native v1 合同",
            details={"path": layout["validation"]},
        )
    role_counts = statistics.get("role_counts")
    parent_role_counts = statistics.get("parent_role_counts")
    if (
        statistics.get("schema_version") != STATISTICS_SCHEMA_VERSION
        or statistics.get("record_count") != record_count
        or statistics.get("parent_count") != parent_count
        or not isinstance(role_counts, dict)
        or set(role_counts) != {"external_train", "external_val"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in role_counts.values()
        )
        or sum(role_counts.values()) != record_count
        or not isinstance(parent_role_counts, dict)
        or set(parent_role_counts) != {"external_train", "external_val"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in parent_role_counts.values()
        )
        or sum(parent_role_counts.values()) != parent_count
    ):
        _fail(
            "statistics native role/count identity 不一致",
            details={"path": layout["statistics"]},
        )
    record_shards = _layout_record_shards(
        layout,
        record_count=record_count,
        verifier=verifier,
    )

    blockers = manifest.get("formal_acceptance_blockers")
    if (
        manifest.get("build_profile") != "full"
        or manifest.get("benchmark_scope") != BENCHMARK_SCOPE
        or manifest.get("scope_validation_complete") is not True
        or manifest.get("formal_acceptance_eligible") is not True
        or blockers != []
        or manifest.get("source_roots_embedded") is not False
        or manifest.get("external_test_retained") is not False
        or manifest.get("official_upstream_evaluation_reproducible") is not False
    ):
        _fail("native RS-GeneralDesc acceptance flags 不满足合同")

    identity = BenchmarkIdentity(
        root=root.resolve(),
        manifest_schema=SUPPORTED_MANIFEST_SCHEMA,
        canonical_schema=SUPPORTED_CANONICAL_SCHEMA,
        build_id=config.data.expected_build_id,
        semantic_config_sha256=semantic_config_sha,
        payload_sha256=config.data.expected_payload_sha256,
        hash_manifest_sha256=config.data.expected_hash_manifest_sha256,
        benchmark_scope=BENCHMARK_SCOPE,
        source_roots_embedded=False,
        deep_validation_saved=True,
        formal_acceptance_eligible=True,
        formal_acceptance_blockers=(),
        record_count=record_count,
        parent_count=parent_count,
    )
    return BenchmarkAccess(
        identity=identity,
        manifest=manifest,
        validation=validation,
        statistics=statistics,
        record_shards=record_shards,
        verifier=verifier,
    )


def inspect_benchmark_identity(
    config: Phase4Config,
    *,
    access: BenchmarkAccess | None = None,
) -> BenchmarkIdentity:
    """返回已绑定的 native Benchmark identity。"""

    opened = access or open_benchmark_access(config)
    if opened.root != config.data.benchmark_root.resolve():
        _fail("BenchmarkAccess root 与配置不一致")
    return opened.identity


def run_preflight(
    config: Phase4Config,
    *,
    require_new_output: bool = True,
    access: BenchmarkAccess | None = None,
) -> dict[str, Any]:
    opened = access or open_benchmark_access(config)
    identity = inspect_benchmark_identity(config, access=opened)
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
    if set(config.data.roles) - {"external_train", "external_val"}:
        raise PreflightError(
            ReasonCode.ROLE_FORBIDDEN,
            "RS-GeneralDesc roles 非法",
        )
    return {
        "status": "ok",
        "mode": config.run.mode.value,
        "mask_mode": config.run.mask_mode.value,
        "benchmark": identity.to_dict(),
        "config_semantic_sha256": config.semantic_sha256,
        "formal_acceptance": False,
    }
