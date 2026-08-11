"""既有 Text Evidence Bank 的公共只读 runtime retrieval API。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oa_groundrag.phase3.common import canonical_json, read_json, read_jsonl, sha256_text
from oa_groundrag.phase4.errors import ContractError, ReasonCode

from .bank import validate_bank
from .contracts import (
    KnowledgeType,
    QueryIntent,
    Stage6Config,
    TextRagTask,
    load_stage6_config,
    route_text_rag,
)
from .retrieval import (
    BGEM3DenseEmbedder,
    HybridRetriever,
    build_balanced_packet,
    build_counter_limitation_query,
    build_interpretation_query,
    make_query_row,
)


@dataclass(frozen=True)
class RuntimeRetrievalResult:
    task: TextRagTask
    queries: tuple[Mapping[str, Any], Mapping[str, Any]]
    packet: Mapping[str, Any]
    bank_identity: Mapping[str, Any]


def load_runtime_bank_payload(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, list[str]]:
    """读取已验证 Bank payload；不读取源 PDF，也不写 Bank。"""

    manifest = read_json(root / "manifest.json")
    units = read_jsonl(root / "evidence_units.jsonl")
    with (root / "dense_embeddings.npy").open("rb") as handle:
        embeddings = np.load(handle, allow_pickle=False)
    dense_index = read_json(root / "dense_index.json")
    return manifest, units, embeddings, list(dense_index["unit_ids"])


class RuntimeTextRetriever:
    """一次请求惰性加载 BGE-M3，查询后可显式释放。"""

    def __init__(self, config: Stage6Config) -> None:
        self.config = config
        self._embedder: BGEM3DenseEmbedder | None = None
        self._bank_validation: Mapping[str, Any] | None = None

    @classmethod
    def from_config_path(cls, path: Path | str) -> "RuntimeTextRetriever":
        return cls(load_stage6_config(path))

    def _load(self) -> BGEM3DenseEmbedder:
        if self._bank_validation is None:
            self._bank_validation = validate_bank(
                self.config.bank_root,
                config=self.config,
                verify_sources=False,
            )
        if self._embedder is None:
            self._embedder = BGEM3DenseEmbedder(
                self.config.dense.model_root,
                repo_id=self.config.dense.repo_id,
                revision=self.config.dense.revision,
                device=self.config.dense.device,
                batch_size=self.config.dense.batch_size,
                max_tokens=self.config.dense.max_tokens,
            )
        return self._embedder

    def retrieve(
        self,
        *,
        task: TextRagTask,
        record_id: str,
        question: str,
        observation: Mapping[str, Any],
        program_facts: Mapping[str, Any],
        target_status: str,
        candidate_count: int,
        available_modalities: Sequence[str],
    ) -> RuntimeRetrievalResult:
        if not route_text_rag(task):
            raise ContractError(
                ReasonCode.RAG_FORBIDDEN,
                f"runtime task 不启用 Text RAG：{task.value}",
            )
        if not record_id or not question.strip():
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "runtime retrieval 要求 record_id/question",
            )
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 0:
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "runtime candidate_count 必须是 >= 0 的整数",
            )
        observation_before = canonical_json(dict(observation))
        facts_before = canonical_json(dict(program_facts))
        query_question = question
        if target_status == "no_target":
            mask = program_facts.get("mask", {})
            area = mask.get("area_pixels", 0) if isinstance(mask, Mapping) else 0
            query_question = (
                f"{question}\n程序状态：no_target=true；"
                f"mask_area_pixels={area}；candidate_count={candidate_count}"
            )
        modalities = tuple(sorted(set(available_modalities))) or ("general",)
        embedder = self._load()
        assert self._bank_validation is not None
        manifest, units, embeddings, dense_ids = load_runtime_bank_payload(
            self.config.bank_root
        )
        observation_sha256 = sha256_text(observation_before)
        interpretation = make_query_row(
            record_id=record_id,
            intent=QueryIntent.INTERPRETATION,
            text=build_interpretation_query(
                question=query_question,
                observation=observation,
                available_modalities=modalities,
            ),
            knowledge_types=(KnowledgeType.INTERPRETATION,),
            available_modalities=modalities,
            observation_sha256=observation_sha256,
            bank_id=str(manifest["bank_id"]),
        )
        counter = make_query_row(
            record_id=record_id,
            intent=QueryIntent.COUNTER_LIMITATION,
            text=build_counter_limitation_query(
                question=query_question,
                observation=observation,
                available_modalities=modalities,
            ),
            knowledge_types=(KnowledgeType.CONFOUNDER, KnowledgeType.LIMITATION),
            available_modalities=modalities,
            observation_sha256=observation_sha256,
            bank_id=str(manifest["bank_id"]),
        )
        vectors = embedder.encode([interpretation["text"], counter["text"]])
        retriever = HybridRetriever(
            sqlite_path=self.config.bank_root / "lexical.sqlite3",
            units=units,
            embeddings=embeddings,
            dense_unit_ids=dense_ids,
            config=self.config.retrieval,
        )
        try:
            interpretation_candidates = retriever.retrieve(interpretation, vectors[0])
            counter_candidates = retriever.retrieve(counter, vectors[1])
        finally:
            retriever.close()
        packet = build_balanced_packet(
            record_id=record_id,
            bank_id=str(manifest["bank_id"]),
            interpretation_query_id=str(interpretation["query_id"]),
            counter_query_id=str(counter["query_id"]),
            interpretation_candidates=interpretation_candidates,
            counter_candidates=counter_candidates,
            config=self.config.retrieval,
        )
        if canonical_json(dict(observation)) != observation_before:
            raise ContractError(
                ReasonCode.PREDICTION_IDENTITY_MISMATCH,
                "runtime retrieval 修改了 Pass-1 observation",
            )
        if canonical_json(dict(program_facts)) != facts_before:
            raise ContractError(
                ReasonCode.PREDICTION_IDENTITY_MISMATCH,
                "runtime retrieval 修改了 Programmatic Facts",
            )
        return RuntimeRetrievalResult(
            task=task,
            queries=(interpretation, counter),
            packet=packet,
            bank_identity={
                "bank_id": self._bank_validation["bank_id"],
                "manifest_sha256": self._bank_validation["manifest_sha256"],
                "ledger_sha256": self._bank_validation["ledger_sha256"],
                "dense_model_identity": dict(embedder.identity),
                "read_only": True,
                "verify_sources": False,
            },
        )

    def release(self) -> None:
        self._embedder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
