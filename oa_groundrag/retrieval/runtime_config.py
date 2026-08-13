"""已发布 Text Evidence Bank 的只读 runtime 配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from oa_groundrag.artifacts.identity import canonical_json, sha256_text
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    resolve_config_path,
)
from oa_groundrag.vlm.config import _load_yaml
from oa_groundrag.vlm.errors import ConfigError, ReasonCode

from .contracts import DenseConfig, RetrievalConfig


TEXT_RAG_RUNTIME_CONFIG_SCHEMA = "oa_groundrag.text_rag.runtime_config.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TextBankRuntimeBinding:
    root: Path
    bank_id: str
    manifest_sha256: str
    ledger_sha256: str
    build_config_semantic_sha256: str
    registry_semantic_sha256: str


@dataclass(frozen=True)
class TextRAGRuntimeConfig:
    config_path: Path
    source_registry_path: Path
    bank: TextBankRuntimeBinding
    dense: DenseConfig
    retrieval: RetrievalConfig
    semantic_sha256: str

    @property
    def bank_root(self) -> Path:
        return self.bank.root


def _sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是小写 SHA-256")
    return value


def load_text_rag_runtime_config(path: Path | str) -> TextRAGRuntimeConfig:
    """读取不含 development evaluation/output 绑定的检索配置。"""

    config_path = Path(os.path.abspath(Path(path)))
    if (
        first_symlink_component(config_path) is not None
        or not config_path.is_file()
        or config_path.is_symlink()
        or config_path.stat().st_nlink != 1
    ):
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"Text RAG runtime 配置必须是普通文件：{config_path}")
    row = _load_yaml(config_path)
    require_exact_keys(
        row,
        required=("schema_version", "source_registry", "bank", "dense", "retrieval"),
        location="$",
    )
    if row["schema_version"] != TEXT_RAG_RUNTIME_CONFIG_SCHEMA:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 {TEXT_RAG_RUNTIME_CONFIG_SCHEMA}",
        )
    base = config_path.parent
    source_registry_path = resolve_config_path(
        base,
        row["source_registry"],
        location="$.source_registry",
    )
    bank_row = require_mapping(row["bank"], location="$.bank")
    require_exact_keys(
        bank_row,
        required=(
            "root",
            "bank_id",
            "manifest_sha256",
            "ledger_sha256",
            "build_config_semantic_sha256",
            "registry_semantic_sha256",
        ),
        location="$.bank",
    )
    bank = TextBankRuntimeBinding(
        root=resolve_config_path(base, bank_row["root"], location="$.bank.root"),
        bank_id=_sha256(bank_row["bank_id"], location="$.bank.bank_id"),
        manifest_sha256=_sha256(
            bank_row["manifest_sha256"],
            location="$.bank.manifest_sha256",
        ),
        ledger_sha256=_sha256(
            bank_row["ledger_sha256"],
            location="$.bank.ledger_sha256",
        ),
        build_config_semantic_sha256=_sha256(
            bank_row["build_config_semantic_sha256"],
            location="$.bank.build_config_semantic_sha256",
        ),
        registry_semantic_sha256=_sha256(
            bank_row["registry_semantic_sha256"],
            location="$.bank.registry_semantic_sha256",
        ),
    )
    dense_row = require_mapping(row["dense"], location="$.dense")
    require_exact_keys(
        dense_row,
        required=("repo_id", "revision", "model_root", "device", "batch_size", "max_tokens"),
        location="$.dense",
    )
    revision = require_string(dense_row["revision"], location="$.dense.revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "$.dense.revision 必须是完整 Hub commit")
    device = require_string(dense_row["device"], location="$.dense.device")
    if device not in {"cpu", "cuda"}:
        raise ConfigError(ReasonCode.INVALID_ENUM, "$.dense.device 只允许 cpu/cuda")
    dense = DenseConfig(
        repo_id=require_string(dense_row["repo_id"], location="$.dense.repo_id"),
        revision=revision,
        model_root=resolve_config_path(
            base,
            dense_row["model_root"],
            location="$.dense.model_root",
        ),
        device=device,
        batch_size=require_int(dense_row["batch_size"], location="$.dense.batch_size", minimum=1),
        max_tokens=require_int(dense_row["max_tokens"], location="$.dense.max_tokens", minimum=64),
    )
    retrieval_row = require_mapping(row["retrieval"], location="$.retrieval")
    retrieval_fields = (
        "lexical_k",
        "dense_k",
        "rrf_k",
        "interpretation_quota",
        "confounder_quota",
        "limitation_quota",
    )
    require_exact_keys(retrieval_row, required=retrieval_fields, location="$.retrieval")
    retrieval = RetrievalConfig(**{
        key: require_int(retrieval_row[key], location=f"$.retrieval.{key}", minimum=1)
        for key in retrieval_fields
    })
    semantic_payload = {
        "schema_version": TEXT_RAG_RUNTIME_CONFIG_SCHEMA,
        "source_registry": str(source_registry_path),
        "bank": {
            **bank.__dict__,
            "root": str(bank.root),
        },
        "dense": {
            **dense.__dict__,
            "model_root": str(dense.model_root),
        },
        "retrieval": retrieval.__dict__,
    }
    return TextRAGRuntimeConfig(
        config_path=config_path,
        source_registry_path=source_registry_path,
        bank=bank,
        dense=dense,
        retrieval=retrieval,
        semantic_sha256=sha256_text(canonical_json(semantic_payload)),
    )
