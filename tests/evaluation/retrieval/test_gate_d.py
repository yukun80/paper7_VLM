from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from oa_groundrag.data.rs_general.io import canonical_json, sha256_text
from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import ContractError, VLMError
from oa_groundrag.retrieval.contracts import PASS2_OUTPUT_SCHEMA, RagMode
from oa_groundrag.evaluation.retrieval.gate_d import (
    GATE_D_CONFIG_SCHEMA,
    audit_selected_prompts,
    build_automatic_evaluation,
    load_gate_d_config,
    prepare_gate_d_protocol,
    select_gate_d_records,
    _selected_from_protocol,
    _validate_upstream,
)


SOURCES = (
    "gdcld",
    "landslide4sense",
    "landslidebench_agent",
    "lmhld",
    "multimodal_landslide",
)


def _selection_records(per_source: int = 7) -> list[dict]:
    rows = []
    for slot in range(per_source):
        for source in SOURCES:
            record_id = f"{source}_{slot}"
            observation = {"target_appearance": {"tone": "bright"}, "limitations": []}
            facts = {"mask": {"target_status": "target_present"}}
            rows.append({
                "record_id": record_id,
                "source": source,
                "split": "val",
                "target_status": "target_present",
                "available_modalities": ["optical"],
                "program_facts": facts,
                "program_facts_sha256": sha256_text(canonical_json(facts)),
                "observation": observation,
                "observation_sha256": sha256_text(canonical_json(observation)),
            })
    return rows


def _packet(record_id: str) -> dict:
    items = []
    for suffix, kind in (("i", "interpretation"), ("c", "confounder"), ("l", "limitation")):
        items.append({
            "evidence_id": f"ev_{suffix}",
            "knowledge_type": kind,
            "content": f"{kind} evidence",
            "conditions": [],
            "applicable_modalities": ["optical"],
            "source_id": "source",
            "source_title": "Source",
            "pdf_page": 3,
            "section": "3.1",
            "authority_class": "paper",
            "source_status": "published",
        })
    return {"record_id": record_id, "packet_id": f"pkt_{record_id}", "items": items}


def _output(*, citations: bool, rag_text: str = "RAG explanation") -> dict:
    ids = {
        "supporting_interpretations": ["ev_i"],
        "alternative_explanations": ["ev_c"],
        "limitations": ["ev_l"],
        "recommended_verification": ["ev_l"],
        "summary": ["ev_i"],
    }
    return {
        "schema_version": PASS2_OUTPUT_SCHEMA,
        "supporting_interpretations": [{"text": rag_text, "evidence_ids": ids["supporting_interpretations"] if citations else []}],
        "alternative_explanations": [{"text": "alternative", "evidence_ids": ids["alternative_explanations"] if citations else []}],
        "limitations": [{"text": "limited evidence", "evidence_ids": ids["limitations"] if citations else []}],
        "recommended_verification": [{"text": "verify", "evidence_ids": ids["recommended_verification"] if citations else []}],
        "summary": {"text": "candidate only", "evidence_ids": ids["summary"] if citations else []},
    }


class _FakeProcessor:
    def __init__(self, *, rag_tokens: int = 120, image_count: int = 0) -> None:
        self.rag_tokens = rag_tokens
        self.image_count = image_count

    def identity(self):
        return {"processor": "fake"}

    def encode_text_inference(self, messages):
        serialized = json.dumps(messages, ensure_ascii=False)
        count = self.rag_tokens if "retrieved_evidence" in serialized and '"retrieved_evidence": []' not in serialized else 80
        return SimpleNamespace(input_token_count=count, image_count=self.image_count)


class GateDTest(unittest.TestCase):
    def test_selection_excludes_smoke_and_is_source_round_robin(self) -> None:
        records = _selection_records()
        smoke = [f"{source}_0" for source in SOURCES]
        selected = select_gate_d_records(
            records,
            smoke_record_ids=smoke,
            source_order=SOURCES,
            per_source=5,
        )
        self.assertEqual(len(selected), 25)
        self.assertEqual([row["source"] for row in selected[:5]], list(SOURCES))
        self.assertEqual([row["record_id"] for row in selected[:5]], [f"{source}_1" for source in SOURCES])
        self.assertFalse(set(smoke) & {row["record_id"] for row in selected})
        self.assertEqual({source: 5 for source in SOURCES}, {
            source: sum(row["source"] == source for row in selected) for source in SOURCES
        })
        repeated = select_gate_d_records(records, smoke_record_ids=smoke, source_order=SOURCES, per_source=5)
        self.assertEqual([row["record_id"] for row in selected], [row["record_id"] for row in repeated])

    def test_selection_rejects_wrong_smoke_and_split(self) -> None:
        records = _selection_records()
        with self.assertRaises((VLMError, RSGeneralDescError)):
            select_gate_d_records(records, smoke_record_ids=["gdcld_0"], source_order=SOURCES, per_source=5)
        records[5]["split"] = "test"
        with self.assertRaises((VLMError, RSGeneralDescError)):
            select_gate_d_records(
                records,
                smoke_record_ids=[f"{source}_0" for source in SOURCES],
                source_order=SOURCES,
                per_source=5,
            )

    def test_prompt_audit_is_text_only_and_enforces_limit(self) -> None:
        selected = _selection_records(per_source=1)[:1]
        packet = _packet(selected[0]["record_id"])
        stage6 = SimpleNamespace(dev=SimpleNamespace(question="How to interpret?"))
        rows, identity = audit_selected_prompts(
            stage6,
            selected,
            {packet["record_id"]: packet},
            processor=_FakeProcessor(),
            max_input_tokens=200,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["mode"] for row in rows}, {mode.value for mode in RagMode})
        self.assertTrue(all(row["image_count"] == 0 for row in rows))
        self.assertEqual(len(identity), 64)
        with self.assertRaises((VLMError, RSGeneralDescError)):
            audit_selected_prompts(
                stage6,
                selected,
                {packet["record_id"]: packet},
                processor=_FakeProcessor(rag_tokens=220),
                max_input_tokens=200,
            )
        with self.assertRaises((VLMError, RSGeneralDescError)):
            audit_selected_prompts(
                stage6,
                selected,
                {packet["record_id"]: packet},
                processor=_FakeProcessor(image_count=1),
                max_input_tokens=200,
            )

    def test_explicit_protocol_records_do_not_fall_back_to_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = _selection_records(per_source=2)
            (root / "selection.json").write_text(json.dumps({"records": records}), encoding="utf-8")
            chosen = records[5:10]
            protocol = {"records": [{
                "record_id": row["record_id"],
                "selection_record_sha256": sha256_text(canonical_json(row)),
                "observation_sha256": row["observation_sha256"],
                "program_facts_sha256": row["program_facts_sha256"],
            } for row in chosen]}
            stage6 = SimpleNamespace(retrieval_root=root)
            observed = _selected_from_protocol(stage6, protocol)
            self.assertEqual([row["record_id"] for row in observed], [row["record_id"] for row in chosen])
            protocol["records"][0]["selection_record_sha256"] = "0" * 64
            with self.assertRaises((VLMError, RSGeneralDescError)):
                _selected_from_protocol(stage6, protocol)

    def test_automatic_evaluation_has_no_scientific_metrics(self) -> None:
        record_id = "gdcld_1"
        protocol = {
            "protocol_id": "p" * 64,
            "records": [{
                "evaluation_ordinal": 0,
                "record_id": record_id,
                "source": "gdcld",
                "observation_sha256": "o" * 64,
                "program_facts_sha256": "f" * 64,
            }],
        }
        predictions = [
            {
                "record_id": record_id, "mode": "no_rag", "generator_identity_sha256": "g" * 64,
                "packet_id": None, "prediction_id": "n" * 64, "output": _output(citations=False, rag_text="plain"),
            },
            {
                "record_id": record_id, "mode": "text_rag", "generator_identity_sha256": "g" * 64,
                "packet_id": f"pkt_{record_id}", "prediction_id": "r" * 64, "output": _output(citations=True),
            },
        ]
        packet = _packet(record_id)
        cases, report = build_automatic_evaluation(
            protocol=protocol,
            run_manifest={"run_id": "u" * 64},
            predictions=predictions,
            packets={record_id: packet},
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(report["pair_complete_count"], 1)
        self.assertEqual(report["traceable_citation_count"], 5)
        self.assertIsNone(report["unsupported_claim_rate"])
        self.assertIsNone(report["mrr"])
        self.assertIsNone(report["gate_d_pass"])
        self.assertFalse(report["formal_acceptance"])

    def test_automatic_evaluation_rejects_unknown_citation_and_pair_drift(self) -> None:
        record_id = "gdcld_1"
        protocol = {
            "protocol_id": "p" * 64,
            "records": [{
                "evaluation_ordinal": 0, "record_id": record_id, "source": "gdcld",
                "observation_sha256": "o" * 64, "program_facts_sha256": "f" * 64,
            }],
        }
        no_row = {
            "record_id": record_id, "mode": "no_rag", "generator_identity_sha256": "g" * 64,
            "packet_id": None, "prediction_id": "n" * 64, "output": _output(citations=False),
        }
        rag_row = {
            "record_id": record_id, "mode": "text_rag", "generator_identity_sha256": "g" * 64,
            "packet_id": f"pkt_{record_id}", "prediction_id": "r" * 64, "output": _output(citations=True),
        }
        rag_row["output"]["summary"]["evidence_ids"] = ["ev_unknown"]
        with self.assertRaises((VLMError, RSGeneralDescError)):
            build_automatic_evaluation(
                protocol=protocol,
                run_manifest={"run_id": "u" * 64},
                predictions=[no_row, rag_row],
                packets={record_id: _packet(record_id)},
            )
        with self.assertRaises((VLMError, RSGeneralDescError)):
            build_automatic_evaluation(
                protocol=protocol,
                run_manifest={"run_id": "u" * 64},
                predictions=[no_row],
                packets={record_id: _packet(record_id)},
            )

    def test_config_is_strict_and_rejects_sealed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "gate.yaml"
            base = f"""schema_version: {GATE_D_CONFIG_SCHEMA}
stage6:
  config: stage6.yaml
  expected_semantic_sha256: {'a' * 64}
upstream:
  selection_id: {'b' * 64}
  retrieval_id: {'c' * 64}
  retrieval_manifest_sha256: {'d' * 64}
smoke_run:
  root: smoke
  run_id: {'e' * 64}
  manifest_sha256: {'f' * 64}
selection:
  source_order: [gdcld, landslide4sense, landslidebench_agent, lmhld, multimodal_landslide]
  per_source: 5
outputs:
  protocol_root: protocol
  run_root: run
  evaluation_root: evaluation
gpu:
  minimum_total_memory_bytes: 1
  minimum_free_memory_bytes: 1
reference_authority: automatic_contract_only
"""
            config.write_text(base, encoding="utf-8")
            observed = load_gate_d_config(config)
            self.assertEqual(observed.per_source, 5)
            config.write_text(base.replace("protocol_root: protocol", "protocol_root: sealed_test"), encoding="utf-8")
            with self.assertRaises((VLMError, RSGeneralDescError)):
                load_gate_d_config(config)
            config.write_text(base + "unknown: true\n", encoding="utf-8")
            with self.assertRaises((VLMError, RSGeneralDescError)):
                load_gate_d_config(config)

    def test_upstream_identity_drift_is_rejected(self) -> None:
        config = SimpleNamespace(
            stage6_config_path=Path("stage6.yaml"),
            expected_stage6_semantic_sha256="a" * 64,
            expected_retrieval_id="r" * 64,
            expected_selection_id="s" * 64,
            expected_retrieval_manifest_sha256="m" * 64,
        )
        stage6 = SimpleNamespace(semantic_sha256="a" * 64, bank_root=Path("bank"), retrieval_root=Path("retrieval"))
        with (
            patch("oa_groundrag.evaluation.retrieval.gate_d.load_stage6_config", return_value=stage6),
            patch("oa_groundrag.evaluation.retrieval.gate_d.validate_bank"),
            patch("oa_groundrag.evaluation.retrieval.gate_d.validate_retrieval", return_value={
                "retrieval_id": "wrong", "selection_id": "s" * 64, "manifest_sha256": "m" * 64,
            }),
        ):
            with self.assertRaises((VLMError, RSGeneralDescError)):
                _validate_upstream(config)

    def test_protocol_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "gate.yaml"
            config_path.write_text(f"""schema_version: {GATE_D_CONFIG_SCHEMA}
stage6:
  config: stage6.yaml
  expected_semantic_sha256: {'a' * 64}
upstream:
  selection_id: {'b' * 64}
  retrieval_id: {'c' * 64}
  retrieval_manifest_sha256: {'d' * 64}
smoke_run:
  root: smoke
  run_id: {'e' * 64}
  manifest_sha256: {'f' * 64}
selection:
  source_order: [gdcld, landslide4sense, landslidebench_agent, lmhld, multimodal_landslide]
  per_source: 5
outputs:
  protocol_root: protocol
  run_root: run
  evaluation_root: evaluation
gpu:
  minimum_total_memory_bytes: 1
  minimum_free_memory_bytes: 1
reference_authority: automatic_contract_only
""", encoding="utf-8")
            protocol = {
                "protocol_id": "p" * 64,
                "stage6_config_semantic_sha256": "a" * 64,
                "record_count": 25,
                "prompt_count": 50,
                "prompt_token_min": 1,
                "prompt_token_max": 2,
            }
            audit = [{"record_id": "r"}]
            environment = {"python": "test"}
            with patch("oa_groundrag.evaluation.retrieval.gate_d._build_protocol", return_value=(protocol, audit, environment)):
                prepare_gate_d_protocol(config_path)
                with self.assertRaises((VLMError, RSGeneralDescError)):
                    prepare_gate_d_protocol(config_path)


if __name__ == "__main__":
    unittest.main()
