"""Unified Inference v2 严格配置与 lazy provider factory。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from oa_groundrag.artifacts.io import (
    PortablePathError,
    first_symlink_component,
    resolve_config_reference,
    validate_bundle_root,
)
from oa_groundrag.data.rs_general.io import (
    require_bool,
    require_exact_keys,
    require_mapping,
    require_string,
)
from oa_groundrag.vlm.config import _load_yaml

from .contracts import (
    UnifiedInferenceError,
    UnifiedReasonCode,
    reject_test_or_sealed_path,
)


UNIFIED_CONFIG_SCHEMA = "oa_groundrag.unified_inference.config.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SpatialRuntimeBinding:
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class SemanticCoreRuntimeBinding:
    config_path: Path
    best_pointer_sha256: str
    checkpoint_manifest_sha256: str
    adapter_sha256: str
    workflow_state_sha256: str


@dataclass(frozen=True)
class RetrievalRuntimeBinding:
    config_path: Path


@dataclass(frozen=True)
class UnifiedRuntimeOptions:
    device: str
    release_spatial_before_shared: bool
    release_shared_before_rag: bool
    release_retriever_after_query: bool
    release_unused_between_requests: bool
    reject_test_paths: bool


@dataclass(frozen=True)
class UnifiedInferenceConfig:
    config_path: Path
    repository_root: Path
    spatial: SpatialRuntimeBinding
    semantic_core: SemanticCoreRuntimeBinding
    retrieval: RetrievalRuntimeBinding
    runtime: UnifiedRuntimeOptions


def _regular_path(path: Path, *, label: str, directory: bool = False) -> None:
    linked = first_symlink_component(path)
    exists = path.is_dir() if directory else path.is_file()
    if linked is not None or not exists or path.is_symlink():
        kind = "目录" if directory else "文件"
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            f"{label} 必须是普通{kind}：{path}",
        )


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            f"{label} 必须是小写 SHA-256",
        )
    return value


def load_unified_config(path: Path | str) -> UnifiedInferenceConfig:
    """读取统一配置；不构造 provider，不哈希或打开模型/Bank/checkpoint。"""

    config_path = Path(os.path.abspath(Path(path)))
    reject_test_or_sealed_path(config_path, label="unified config")
    _regular_path(config_path, label="Unified config")
    try:
        row = _load_yaml(config_path)
        require_exact_keys(
            row,
            required=(
                "schema_version",
                "repository_root",
                "spatial",
                "semantic_core",
                "retrieval",
                "runtime",
            ),
            location="$",
        )
        if row["schema_version"] != UNIFIED_CONFIG_SCHEMA:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                f"只支持 unified config schema {UNIFIED_CONFIG_SCHEMA}",
            )
        repository_root = resolve_config_reference(
            config_path,
            require_string(
                row["repository_root"], location="$.repository_root"
            ),
        )
        spatial_row = require_mapping(row["spatial"], location="$.spatial")
        require_exact_keys(
            spatial_row,
            required=("config", "checkpoint", "checkpoint_sha256"),
            location="$.spatial",
        )
        semantic_row = require_mapping(
            row["semantic_core"], location="$.semantic_core"
        )
        semantic_fields = (
            "config",
            "best_pointer_sha256",
            "checkpoint_manifest_sha256",
            "adapter_sha256",
            "workflow_state_sha256",
        )
        require_exact_keys(
            semantic_row,
            required=semantic_fields,
            location="$.semantic_core",
        )
        retrieval_row = require_mapping(row["retrieval"], location="$.retrieval")
        require_exact_keys(
            retrieval_row,
            required=("config",),
            location="$.retrieval",
        )
        runtime_row = require_mapping(row["runtime"], location="$.runtime")
        option_names = (
            "device",
            "release_spatial_before_shared",
            "release_shared_before_rag",
            "release_retriever_after_query",
            "release_unused_between_requests",
            "reject_test_paths",
        )
        require_exact_keys(runtime_row, required=option_names, location="$.runtime")
        device = require_string(runtime_row["device"], location="$.runtime.device")
        if device not in {"cpu", "cuda"}:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "$.runtime.device 只允许 cpu/cuda",
            )
        booleans = {
            name: require_bool(runtime_row[name], location=f"$.runtime.{name}")
            for name in option_names[1:]
        }
    except UnifiedInferenceError:
        raise
    except Exception as error:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            str(error),
            cause_type=type(error).__name__,
        ) from error

    # P0 生命周期和路径安全合同不可由配置关闭。
    disabled = [name for name, enabled in booleans.items() if not enabled]
    if disabled:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            f"Unified v2 安全/释放开关必须为 true：{disabled}",
        )
    try:
        spatial_config = resolve_config_reference(
            config_path,
            require_string(spatial_row["config"], location="$.spatial.config"),
        )
        checkpoint = resolve_config_reference(
            config_path,
            require_string(
                spatial_row["checkpoint"], location="$.spatial.checkpoint"
            ),
        )
        semantic_config = resolve_config_reference(
            config_path,
            require_string(
                semantic_row["config"], location="$.semantic_core.config"
            ),
        )
        retrieval_config = resolve_config_reference(
            config_path,
            require_string(retrieval_row["config"], location="$.retrieval.config"),
        )
    except PortablePathError as error:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            f"Unified 配置路径非法：{error}",
            cause_type=type(error).__name__,
        ) from error
    for label, value in (
        ("repository_root", repository_root),
        ("spatial.config", spatial_config),
        ("spatial.checkpoint", checkpoint),
        ("semantic_core.config", semantic_config),
        ("retrieval.config", retrieval_config),
    ):
        reject_test_or_sealed_path(value, label=label)
    _regular_path(repository_root, label="repository_root", directory=True)
    return UnifiedInferenceConfig(
        config_path=config_path,
        repository_root=repository_root,
        spatial=SpatialRuntimeBinding(
            config_path=spatial_config,
            checkpoint_path=checkpoint,
            checkpoint_sha256=_sha256(
                spatial_row["checkpoint_sha256"],
                label="$.spatial.checkpoint_sha256",
            ),
        ),
        semantic_core=SemanticCoreRuntimeBinding(
            config_path=semantic_config,
            best_pointer_sha256=_sha256(
                semantic_row["best_pointer_sha256"],
                label="$.semantic_core.best_pointer_sha256",
            ),
            checkpoint_manifest_sha256=_sha256(
                semantic_row["checkpoint_manifest_sha256"],
                label="$.semantic_core.checkpoint_manifest_sha256",
            ),
            adapter_sha256=_sha256(
                semantic_row["adapter_sha256"],
                label="$.semantic_core.adapter_sha256",
            ),
            workflow_state_sha256=_sha256(
                semantic_row["workflow_state_sha256"],
                label="$.semantic_core.workflow_state_sha256",
            ),
        ),
        retrieval=RetrievalRuntimeBinding(config_path=retrieval_config),
        runtime=UnifiedRuntimeOptions(device=device, **booleans),
    )


def validate_provider_paths(config: UnifiedInferenceConfig) -> None:
    """provider 构造前检查其将读取的所有配置/资产根；不读取 payload。"""

    from oa_groundrag.segmentation.config import load_runtime_config
    from oa_groundrag.vlm.grounded_runtime import load_grounded_runtime_config
    from oa_groundrag.retrieval.runtime_config import load_text_rag_runtime_config

    _regular_path(config.spatial.config_path, label="spatial.config")
    _regular_path(config.spatial.checkpoint_path, label="spatial.checkpoint")
    _regular_path(config.semantic_core.config_path, label="semantic_core.config")
    _regular_path(config.retrieval.config_path, label="retrieval.config")
    spatial = load_runtime_config(config.spatial.config_path)
    benchmark_root = spatial.resolve_path(spatial.benchmark_root, config.repository_root)
    output_root = spatial.resolve_path(spatial.output_dir, config.repository_root)
    backbone_weights = spatial.resolve_path(spatial.backbone_weights, config.repository_root)
    semantic_core = load_grounded_runtime_config(config.semantic_core.config_path)
    retrieval = load_text_rag_runtime_config(config.retrieval.config_path)
    try:
        bundle_root = validate_bundle_root(config.repository_root.parent)
    except PortablePathError as error:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            f"Unified bundle 布局非法：{error}",
        ) from error
    if (
        config.repository_root != bundle_root / "paper7_VLM"
        or semantic_core.bundle_root != bundle_root
        or retrieval.bundle_root != bundle_root
    ):
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            "Unified、Grounded 与 Text RAG 的 bundle_root 不一致",
        )
    first_paths = {
        "spatial.benchmark_root": benchmark_root,
        "spatial.output_root": output_root,
        "spatial.backbone_weights": backbone_weights,
        "semantic_core.workflow_root": semantic_core.workflow_root,
        "semantic_core.output_root": semantic_core.training_root,
        "semantic_core.model": semantic_core.model.path,
        "semantic_core.processor": semantic_core.model.processor_path,
        "retrieval.source_registry": retrieval.source_registry_path,
        "retrieval.bank_root": retrieval.bank_root,
        "retrieval.dense_model_root": retrieval.dense.model_root,
    }
    for label, value in first_paths.items():
        reject_test_or_sealed_path(value, label=label)
    if retrieval.dense.device != config.runtime.device:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            "Unified runtime device 与 Stage 6 dense.device 必须一致",
        )


def build_unified_runtime(config: UnifiedInferenceConfig):
    """仅构造 lazy providers；权重在对应 capability 首次调用时加载。"""

    from .providers import (
        OAAuxSegSpatialProvider,
        EvidenceConstrainedRAGProvider,
        GroundedEvidenceInterface,
        GroundedVLMProvider,
    )
    from .inference import UnifiedInferenceRuntime

    return UnifiedInferenceRuntime(
        spatial=OAAuxSegSpatialProvider(
            config_path=config.spatial.config_path,
            checkpoint_path=config.spatial.checkpoint_path,
            checkpoint_sha256=config.spatial.checkpoint_sha256,
            repo_root=config.repository_root,
        ),
        shared_mllm=GroundedVLMProvider(
            binding=config.semantic_core,
            device=config.runtime.device,
        ),
        evidence=GroundedEvidenceInterface(),
        text_rag=EvidenceConstrainedRAGProvider(
            config_path=config.retrieval.config_path,
        ),
    )
