"""Stage 6 的确定性 query、FTS5 + BGE-M3 + RRF 检索。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.vlm.errors import ContractError, ModelError, ReasonCode

from .contracts import (
    KnowledgeType,
    PACKET_SCHEMA,
    QUERY_SCHEMA,
    QueryIntent,
    RetrievalConfig,
    packet_identity,
    query_identity,
)


_LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_PLACEHOLDERS = frozenset({
    "", "无法判断", "不能判断", "无法确定", "不能确定", "不可见", "未知",
    "not_applicable", "not applicable", "unknown", "cannot determine", "not visible",
})
_BGE_RUNTIME_FILES = frozenset({
    "1_Pooling/config.json", "README.md", "config.json",
    "config_sentence_transformers.json", "modules.json", "pytorch_model.bin",
    "sentence_bert_config.json", "sentencepiece.bpe.model",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
})
_BGE_ALLOWED_FILES = _BGE_RUNTIME_FILES | {"MODEL_IDENTITY.json"}


class DenseEmbedder(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def lexical_tokens(value: str) -> tuple[str, ...]:
    """生成 FTS unicode61 可稳定消费的英文 token 与中文 uni/bi-gram。"""

    normalized = " ".join(value.lower().split())
    tokens = [match.group(0) for match in _LATIN_TOKEN.finditer(normalized)]
    for match in _CJK_RUN.finditer(normalized):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[index:index + 2] for index in range(max(0, len(run) - 1)))
    return tuple(token for token in tokens if token)


def analyzed_text(value: str) -> str:
    return " ".join(lexical_tokens(value))


def _fts_query(value: str) -> str:
    tokens = tuple(dict.fromkeys(lexical_tokens(value)))
    if not tokens:
        raise ContractError(ReasonCode.TYPE_MISMATCH, "lexical query 没有可检索 token")
    return " OR ".join(f'"{token}"' for token in tokens)


def build_lexical_index(path: Path, units: Sequence[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"lexical index 已存在：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE evidence ("
            "unit_id TEXT PRIMARY KEY, knowledge_type TEXT NOT NULL, modalities TEXT NOT NULL, "
            "source_id TEXT NOT NULL, pdf_page INTEGER NOT NULL, section TEXT, content TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE evidence_fts USING fts5("
            "unit_id UNINDEXED, analyzed, tokenize='unicode61 remove_diacritics 2')"
        )
        for unit in units:
            if not unit.get("indexed"):
                continue
            modalities = canonical_json(list(unit["applicable_modalities"]))
            cursor = connection.execute(
                "INSERT INTO evidence(unit_id, knowledge_type, modalities, source_id, pdf_page, section, content) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    unit["unit_id"], unit["knowledge_type"], modalities, unit["source_id"],
                    unit["pdf_page"], unit["section"], unit["content"],
                ),
            )
            connection.execute(
                "INSERT INTO evidence_fts(rowid, unit_id, analyzed) VALUES (?, ?, ?)",
                (cursor.lastrowid, unit["unit_id"], analyzed_text(unit["content"])),
            )
        connection.commit()
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()


def lexical_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    knowledge_types: Sequence[str],
    available_modalities: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not knowledge_types or limit < 1:
        return []
    placeholders = ",".join("?" for _ in knowledge_types)
    rows = connection.execute(
        "SELECT e.unit_id, e.modalities, -bm25(evidence_fts) AS score "
        "FROM evidence_fts JOIN evidence e ON e.rowid=evidence_fts.rowid "
        f"WHERE evidence_fts MATCH ? AND e.knowledge_type IN ({placeholders}) "
        "ORDER BY score DESC, e.unit_id ASC LIMIT ?",
        (_fts_query(query), *knowledge_types, max(limit * 20, 100)),
    ).fetchall()
    available = set(available_modalities)
    output: list[dict[str, Any]] = []
    for unit_id, modalities_json, score in rows:
        modalities = set(json.loads(modalities_json))
        if "general" not in modalities and not modalities & available:
            continue
        output.append({"unit_id": str(unit_id), "score": round(float(score), 8)})
        if len(output) >= limit:
            break
    return output


def dense_model_identity(model_root: Path, *, repo_id: str, revision: str) -> dict[str, Any]:
    root = Path(model_root)
    linked = first_symlink_component(root)
    if linked is not None or not root.is_dir() or root.is_symlink():
        raise ModelError(ReasonCode.ASSET_MISSING, f"BGE-M3 root 非普通目录：{root}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".cache/"):
            continue
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ModelError(ReasonCode.OUTPUT_LINK, f"BGE-M3 文件必须普通单链接：{path}")
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    required = {
        "config.json", "pytorch_model.bin", "sentencepiece.bpe.model",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    }
    missing = sorted(required - set(files))
    unexpected = sorted(set(files) - _BGE_ALLOWED_FILES)
    if missing or unexpected or set(files) != _BGE_ALLOWED_FILES:
        raise ModelError(
            ReasonCode.ASSET_MISSING,
            "BGE-M3 文件集不符合固定 allowlist",
            details={"missing": missing, "unexpected": unexpected},
        )
    metadata = read_json(root / "MODEL_IDENTITY.json")
    if (
        set(metadata) != {
            "schema_version", "repo_id", "revision", "download_client",
            "linked_weight_sha256", "runtime_files",
        }
        or metadata.get("schema_version") != "oa_groundrag.text_embedding_model_identity.v1"
        or metadata.get("repo_id") != repo_id
        or metadata.get("revision") != revision
        or metadata.get("linked_weight_sha256") != files["pytorch_model.bin"]["sha256"]
        or metadata.get("runtime_files") != {
            relative: files[relative] for relative in sorted(_BGE_RUNTIME_FILES)
        }
    ):
        raise ModelError(ReasonCode.MODEL_IDENTITY_MISMATCH, "BGE-M3 MODEL_IDENTITY.json 无法重算")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "model_root": str(root.resolve()),
        "files": files,
        "files_identity_sha256": sha256_text(canonical_json(files)),
    }


class BGEM3DenseEmbedder:
    """只使用 BGE-M3 dense CLS；lexical 路径由 FTS5 独立承担。"""

    def __init__(
        self,
        model_root: Path,
        *,
        repo_id: str,
        revision: str,
        device: str,
        batch_size: int,
        max_tokens: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ModelError(ReasonCode.ASSET_MISSING, "BGE-M3 需要 torch/transformers") from error
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelError(ReasonCode.CUDA_REQUIRED, "BGE-M3 配置要求 CUDA，但当前不可见")
        self._torch = torch
        self._device = torch.device(device)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.model_root = Path(model_root)
        identity = dense_model_identity(self.model_root, repo_id=repo_id, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        ).to(self._device)
        self.model.eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if str(getattr(self.model.config, "model_type", "")) != "xlm-roberta" or hidden_size != 1024:
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "BGE-M3 config/model topology 不符合固定 dense backend",
                details={"model_type": getattr(self.model.config, "model_type", None), "hidden_size": hidden_size},
            )
        self._identity = {
            **identity,
            "backend": "transformers_cls_l2.v1",
            "model_type": "xlm-roberta",
            "embedding_dimension": hidden_size,
            "max_tokens": max_tokens,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "device_type": self._device.type,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, int(self._identity["embedding_dimension"])), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ContractError(ReasonCode.TYPE_MISMATCH, "dense texts 必须是非空字符串")
        torch = self._torch
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(texts), self.batch_size):
                batch = list(texts[offset:offset + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                output = self.model(**encoded)
                hidden = output.last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
                chunks.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        values = np.concatenate(chunks, axis=0)
        if not np.isfinite(values).all():
            raise ModelError(ReasonCode.MODEL_IDENTITY_MISMATCH, "BGE-M3 生成非有限向量")
        return values


def _flatten(value: Any) -> tuple[str, ...]:
    output: list[str] = []
    if isinstance(value, str):
        text = " ".join(value.split())
        if text.lower() not in _PLACEHOLDERS:
            output.append(text)
    elif isinstance(value, Mapping):
        for key in sorted(value):
            output.extend(_flatten(value[key]))
    elif isinstance(value, (list, tuple)):
        for child in value:
            output.extend(_flatten(child))
    return tuple(output)


def _query_text(*, question: str, fields: Sequence[tuple[str, Any]], modalities: Sequence[str]) -> str:
    parts = [f"用户问题：{' '.join(question.split())}"]
    for label, value in fields:
        flattened = tuple(dict.fromkeys(_flatten(value)))
        if flattened:
            parts.append(f"{label}：{'；'.join(flattened)}")
    parts.append(f"本次输入模态：{','.join(sorted(modalities))}")
    return "\n".join(parts)


def build_interpretation_query(
    *,
    question: str,
    observation: Mapping[str, Any],
    available_modalities: Sequence[str],
) -> str:
    return _query_text(
        question=question,
        fields=(
            ("目标外观", observation.get("target_appearance")),
            ("目标形态", observation.get("target_morphology")),
            ("区域与环境对比", observation.get("region_context_contrast")),
        ),
        modalities=available_modalities,
    )


def build_counter_limitation_query(
    *,
    question: str,
    observation: Mapping[str, Any],
    available_modalities: Sequence[str],
) -> str:
    return _query_text(
        question=question,
        fields=(
            ("可能混淆对象", observation.get("possible_confusers")),
            ("周围环境", observation.get("surrounding_environment")),
            ("证据充分性", observation.get("evidence_sufficiency")),
            ("视觉限制", observation.get("limitations")),
        ),
        modalities=available_modalities,
    )


def make_query_row(
    *,
    record_id: str,
    intent: QueryIntent,
    text: str,
    knowledge_types: Sequence[KnowledgeType],
    available_modalities: Sequence[str],
    observation_sha256: str,
    bank_id: str,
) -> dict[str, Any]:
    row = {
        "schema_version": QUERY_SCHEMA,
        "query_id": "",
        "record_id": record_id,
        "intent": intent.value,
        "text": text,
        "knowledge_types": [value.value for value in knowledge_types],
        "available_modalities": sorted(set(available_modalities)),
        "observation_sha256": observation_sha256,
        "bank_id": bank_id,
    }
    row["query_id"] = query_identity(row)
    return row


@dataclass(frozen=True)
class RankedCandidate:
    unit: Mapping[str, Any]
    lexical_rank: int | None
    lexical_score: float | None
    dense_rank: int | None
    dense_score: float | None
    rrf_score: float


class HybridRetriever:
    def __init__(
        self,
        *,
        sqlite_path: Path,
        units: Sequence[Mapping[str, Any]],
        embeddings: np.ndarray,
        dense_unit_ids: Sequence[str],
        config: RetrievalConfig,
    ) -> None:
        self.connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        self.units = {str(unit["unit_id"]): unit for unit in units if unit.get("indexed")}
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.dense_unit_ids = tuple(dense_unit_ids)
        self.config = config
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.dense_unit_ids):
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "dense matrix/order 不一致")
        if set(self.dense_unit_ids) != set(self.units):
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "dense unit set 与 Bank 不一致")

    def close(self) -> None:
        self.connection.close()

    def retrieve(self, query: Mapping[str, Any], query_vector: np.ndarray) -> list[RankedCandidate]:
        types = tuple(str(value) for value in query["knowledge_types"])
        modalities = tuple(str(value) for value in query["available_modalities"])
        lexical = lexical_search(
            self.connection,
            str(query["text"]),
            knowledge_types=types,
            available_modalities=modalities,
            limit=self.config.lexical_k,
        )
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.embeddings.shape[1] or not np.isfinite(vector).all():
            raise ContractError(ReasonCode.TYPE_MISMATCH, "query vector shape/value 非法")
        eligible: list[int] = []
        modality_set = set(modalities)
        for index, unit_id in enumerate(self.dense_unit_ids):
            unit = self.units[unit_id]
            unit_modalities = set(unit["applicable_modalities"])
            if unit["knowledge_type"] in types and (
                "general" in unit_modalities or unit_modalities & modality_set
            ):
                eligible.append(index)
        dense_rows: list[dict[str, Any]] = []
        if eligible:
            scores = self.embeddings[eligible] @ vector
            order = sorted(
                range(len(eligible)),
                key=lambda local: (-float(scores[local]), self.dense_unit_ids[eligible[local]]),
            )[:self.config.dense_k]
            dense_rows = [
                {
                    "unit_id": self.dense_unit_ids[eligible[local]],
                    "score": round(float(scores[local]), 8),
                }
                for local in order
            ]
        lexical_by_id = {row["unit_id"]: (index + 1, row["score"]) for index, row in enumerate(lexical)}
        dense_by_id = {row["unit_id"]: (index + 1, row["score"]) for index, row in enumerate(dense_rows)}
        candidates: list[RankedCandidate] = []
        for unit_id in set(lexical_by_id) | set(dense_by_id):
            lexical_value = lexical_by_id.get(unit_id)
            dense_value = dense_by_id.get(unit_id)
            score = 0.0
            if lexical_value is not None:
                score += 1.0 / (self.config.rrf_k + lexical_value[0])
            if dense_value is not None:
                score += 1.0 / (self.config.rrf_k + dense_value[0])
            candidates.append(RankedCandidate(
                unit=self.units[unit_id],
                lexical_rank=None if lexical_value is None else lexical_value[0],
                lexical_score=None if lexical_value is None else lexical_value[1],
                dense_rank=None if dense_value is None else dense_value[0],
                dense_score=None if dense_value is None else dense_value[1],
                rrf_score=round(score, 10),
            ))
        return sorted(
            candidates,
            key=lambda value: (-value.rrf_score, value.unit["authority_class"], value.unit["unit_id"]),
        )


def _packet_item(candidate: RankedCandidate) -> dict[str, Any]:
    unit = candidate.unit
    return {
        "evidence_id": unit["unit_id"],
        "unit_id": unit["unit_id"],
        "knowledge_type": unit["knowledge_type"],
        "content": unit["content"],
        "applicable_modalities": list(unit["applicable_modalities"]),
        "conditions": list(unit["conditions"]),
        "source_id": unit["source_id"],
        "source_title": unit["source_title"],
        "pdf_page": unit["pdf_page"],
        "section": unit["section"],
        "authority_class": unit["authority_class"],
        "source_status": unit["source_status"],
        "lexical_rank": candidate.lexical_rank,
        "lexical_score": candidate.lexical_score,
        "dense_rank": candidate.dense_rank,
        "dense_score": candidate.dense_score,
        "rrf_score": candidate.rrf_score,
    }


def build_balanced_packet(
    *,
    record_id: str,
    bank_id: str,
    interpretation_query_id: str,
    counter_query_id: str,
    interpretation_candidates: Sequence[RankedCandidate],
    counter_candidates: Sequence[RankedCandidate],
    config: RetrievalConfig,
) -> dict[str, Any]:
    quotas = {
        KnowledgeType.INTERPRETATION.value: config.interpretation_quota,
        KnowledgeType.CONFOUNDER.value: config.confounder_quota,
        KnowledgeType.LIMITATION.value: config.limitation_quota,
    }
    grouped: dict[str, list[RankedCandidate]] = defaultdict(list)
    for candidate in (*interpretation_candidates, *counter_candidates):
        grouped[str(candidate.unit["knowledge_type"])].append(candidate)
    selected: list[dict[str, Any]] = []
    used_pages: set[tuple[str, int]] = set()
    for knowledge_type in (
        KnowledgeType.INTERPRETATION.value,
        KnowledgeType.CONFOUNDER.value,
        KnowledgeType.LIMITATION.value,
    ):
        count = 0
        seen_ids: set[str] = set()
        for candidate in grouped.get(knowledge_type, []):
            unit_id = str(candidate.unit["unit_id"])
            page_key = (str(candidate.unit["source_id"]), int(candidate.unit["pdf_page"]))
            if unit_id in seen_ids or page_key in used_pages:
                continue
            seen_ids.add(unit_id)
            used_pages.add(page_key)
            selected.append(_packet_item(candidate))
            count += 1
            if count >= quotas[knowledge_type]:
                break
    row = {
        "schema_version": PACKET_SCHEMA,
        "packet_id": "",
        "record_id": record_id,
        "bank_id": bank_id,
        "query_ids": [interpretation_query_id, counter_query_id],
        "quotas": quotas,
        "items": selected,
    }
    row["packet_id"] = packet_identity(row)
    return row


def quota_counts(packet: Mapping[str, Any]) -> dict[str, int]:
    counts = defaultdict(int)
    for item in packet["items"]:
        counts[str(item["knowledge_type"])] += 1
    return {
        knowledge_type.value: counts[knowledge_type.value]
        for knowledge_type in (
            KnowledgeType.INTERPRETATION,
            KnowledgeType.CONFOUNDER,
            KnowledgeType.LIMITATION,
        )
    }
