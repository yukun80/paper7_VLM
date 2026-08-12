"""Unified Inference v1 严格配置与 lazy provider factory。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from oa_groundrag.phase3.common import (
    first_symlink_component,
    require_bool,
    require_exact_keys,
    require_mapping,
    require_string,
    resolve_config_path,
)
from oa_groundrag.phase4.config import _load_yaml

from .contracts import (
    UnifiedInferenceError,
    UnifiedReasonCode,
    reject_test_or_sealed_path,
)


UNIFIED_CONFIG_SCHEMA = "oa_groundrag.unified_inference.config.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SpatialRuntimeBinding:
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str


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
    stage6_config_path: Path
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
            required=("schema_version", "repository_root", "spatial", "stage6", "runtime"),
            location="$",
        )
        if row["schema_version"] != UNIFIED_CONFIG_SCHEMA:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                f"只支持 unified config schema {UNIFIED_CONFIG_SCHEMA}",
            )
        base = config_path.parent
        repository_root = resolve_config_path(
            base,
            row["repository_root"],
            location="$.repository_root",
        )
        spatial_row = require_mapping(row["spatial"], location="$.spatial")
        require_exact_keys(
            spatial_row,
            required=("config", "checkpoint", "checkpoint_sha256"),
            location="$.spatial",
        )
        stage6_row = require_mapping(row["stage6"], location="$.stage6")
        require_exact_keys(stage6_row, required=("config",), location="$.stage6")
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
            f"Unified v1 安全/释放开关必须为 true：{disabled}",
        )
    spatial_config = resolve_config_path(
        base,
        spatial_row["config"],
        location="$.spatial.config",
    )
    checkpoint = resolve_config_path(
        base,
        spatial_row["checkpoint"],
        location="$.spatial.checkpoint",
    )
    stage6_config = resolve_config_path(
        base,
        stage6_row["config"],
        location="$.stage6.config",
    )
    for label, value in (
        ("repository_root", repository_root),
        ("spatial.config", spatial_config),
        ("spatial.checkpoint", checkpoint),
        ("stage6.config", stage6_config),
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
        stage6_config_path=stage6_config,
        runtime=UnifiedRuntimeOptions(device=device, **booleans),
    )


def validate_provider_paths(config: UnifiedInferenceConfig) -> None:
    """provider 构造前检查其将读取的所有配置/资产根；不读取 payload。"""

    from oa_groundrag.phase2.engine import load_runtime_config
    from oa_groundrag.phase4.stage5_config import load_stage5_config
    from oa_groundrag.text_rag.contracts import load_stage6_config

    _regular_path(config.spatial.config_path, label="spatial.config")
    _regular_path(config.spatial.checkpoint_path, label="spatial.checkpoint")
    _regular_path(config.stage6_config_path, label="stage6.config")
    phase2 = load_runtime_config(config.spatial.config_path)
    benchmark_root = phase2.resolve_path(phase2.benchmark_root, config.repository_root)
    output_root = phase2.resolve_path(phase2.output_dir, config.repository_root)
    backbone_weights = phase2.resolve_path(phase2.backbone_weights, config.repository_root)
    stage6 = load_stage6_config(config.stage6_config_path)
    first_paths = {
        "phase2.benchmark_root": benchmark_root,
        "phase2.output_root": output_root,
        "phase2.backbone_weights": backbone_weights,
        "stage6.source_registry": stage6.source_registry_path,
        "stage6.bank_root": stage6.bank_root,
        "stage6.retrieval_root": stage6.retrieval_root,
        "stage6.generation_root": stage6.generation_root,
        "stage6.dense_model_root": stage6.dense.model_root,
        "stage5.config": stage6.stage5.config_path,
    }
    for label, value in first_paths.items():
        reject_test_or_sealed_path(value, label=label)
    stage5 = load_stage5_config(stage6.stage5.config_path)
    stage5_paths = {
        "stage5.workflow_root": stage5.workflow_root,
        "stage5.output_root": stage5.run.output_root,
        "stage5.model": stage5.model.path,
        "stage5.processor": stage5.model.processor_path,
    }
    for label, value in stage5_paths.items():
        reject_test_or_sealed_path(value, label=label)
    if stage6.dense.device != config.runtime.device:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            "Unified runtime device 与 Stage 6 dense.device 必须一致",
        )


def build_unified_runtime(config: UnifiedInferenceConfig):
    """仅构造 lazy providers；权重在对应 capability 首次调用时加载。"""

    from .providers import (
        OAAuxSegSpatialProvider,
        Phase4GroundedEvidenceProvider,
        Stage5SharedMLLMProvider,
        Stage6TextRAGProvider,
    )
    from .runtime import UnifiedInferenceRuntime

    return UnifiedInferenceRuntime(
        spatial=OAAuxSegSpatialProvider(
            config_path=config.spatial.config_path,
            checkpoint_path=config.spatial.checkpoint_path,
            checkpoint_sha256=config.spatial.checkpoint_sha256,
            repo_root=config.repository_root,
        ),
        shared_mllm=Stage5SharedMLLMProvider(
            stage6_config_path=config.stage6_config_path,
            device=config.runtime.device,
        ),
        evidence=Phase4GroundedEvidenceProvider(),
        text_rag=Stage6TextRAGProvider(
            stage6_config_path=config.stage6_config_path,
        ),
    )
