from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from oa_groundrag.retrieval.contracts import (
    KnowledgeType,
    QueryIntent,
    RetrievalConfig,
    packet_identity,
)
from oa_groundrag.retrieval.search import (
    HybridRetriever,
    RankedCandidate,
    build_balanced_packet,
    build_counter_limitation_query,
    build_interpretation_query,
    build_lexical_index,
    lexical_search,
    make_query_row,
    quota_counts,
)


def _unit(unit_id: str, kind: str, page: int, content: str, *, modality: str = "optical", source: str = "s") -> dict:
    return {
        "unit_id": unit_id,
        "knowledge_type": kind,
        "applicable_modalities": [modality],
        "source_id": source,
        "source_title": source,
        "pdf_page": page,
        "section": f"section-{page}",
        "content": content,
        "conditions": [],
        "authority_class": "peer_reviewed",
        "source_status": "published",
        "indexed": True,
    }


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.units = [
            _unit("i1", "interpretation", 1, "滑坡 后缘 陡坎 光学 纹理"),
            _unit("i2", "interpretation", 2, "landslide scarp morphology texture"),
            _unit("c1", "confounder", 3, "道路切坡 采石场 可能混淆"),
            _unit("c2", "confounder", 4, "quarry road cut alternative"),
            _unit("l1", "limitation", 5, "单时相 光学 分辨率 限制 无法确定活动性"),
            _unit("l2", "limitation", 6, "InSAR unwrapping limitation", modality="insar"),
        ]
        self.index = self.root / "lexical.sqlite3"
        build_lexical_index(self.index, self.units)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_chinese_lexical_retrieval(self) -> None:
        connection = sqlite3.connect(self.index)
        try:
            rows = lexical_search(connection, "道路切坡混淆", knowledge_types=["confounder"], available_modalities=["optical"], limit=3)
        finally:
            connection.close()
        self.assertEqual(rows[0]["unit_id"], "c1")

    def test_fake_dense_rrf_is_stable_and_filters_modality(self) -> None:
        embeddings = np.eye(6, dtype=np.float32)
        config = RetrievalConfig(4, 4, 60, 2, 2, 2)
        query = make_query_row(
            record_id="r", intent=QueryIntent.COUNTER_LIMITATION,
            text="单时相 光学 限制", knowledge_types=(KnowledgeType.CONFOUNDER, KnowledgeType.LIMITATION),
            available_modalities=("optical",), observation_sha256="a" * 64, bank_id="b" * 64,
        )
        retriever = HybridRetriever(sqlite_path=self.index, units=self.units, embeddings=embeddings, dense_unit_ids=[row["unit_id"] for row in self.units], config=config)
        try:
            vector = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
            first = retriever.retrieve(query, vector)
            second = retriever.retrieve(query, vector)
        finally:
            retriever.close()
        self.assertEqual(first, second)
        self.assertIn("l1", [row.unit["unit_id"] for row in first])
        self.assertNotIn("l2", [row.unit["unit_id"] for row in first])

    def test_balanced_quota_does_not_cross_fill_or_repeat_page(self) -> None:
        candidates = [
            RankedCandidate(self.units[index], index + 1, 1.0, index + 1, 1.0, 0.1 - index / 100)
            for index in range(5)
        ]
        duplicate_page = _unit("c3", "confounder", 3, "another", source="s")
        candidates.append(RankedCandidate(duplicate_page, 6, 0.1, 6, 0.1, 0.01))
        config = RetrievalConfig(8, 8, 60, 1, 2, 1)
        packet = build_balanced_packet(
            record_id="r", bank_id="b" * 64, interpretation_query_id="qi", counter_query_id="qc",
            interpretation_candidates=candidates[:2], counter_candidates=candidates[2:], config=config,
        )
        self.assertEqual(quota_counts(packet), {"interpretation": 1, "confounder": 2, "limitation": 1})
        self.assertEqual(packet["packet_id"], packet_identity(packet))
        pages = [(row["source_id"], row["pdf_page"]) for row in packet["items"]]
        self.assertEqual(len(pages), len(set(pages)))

    def test_queries_are_deterministic_and_do_not_invent_confirmation(self) -> None:
        observation = {
            "target_appearance": {"tone": "较亮", "texture": "粗糙"},
            "target_morphology": {"shape": "不规则"},
            "region_context_contrast": {"tone_contrast": "较明显"},
            "possible_confusers": ["道路切坡"],
            "surrounding_environment": {"land_cover": ["植被"]},
            "evidence_sufficiency": "limited",
            "limitations": ["单时相"],
        }
        interpretation = build_interpretation_query(question="如何解释？", observation=observation, available_modalities=["optical"])
        counter = build_counter_limitation_query(question="如何解释？", observation=observation, available_modalities=["optical"])
        self.assertEqual(interpretation, build_interpretation_query(question="如何解释？", observation=observation, available_modalities=["optical"]))
        self.assertIn("较亮", interpretation)
        self.assertIn("道路切坡", counter)
        self.assertNotIn("确认滑坡", interpretation)
        self.assertNotIn("典型滑坡", counter)


if __name__ == "__main__":
    unittest.main()
