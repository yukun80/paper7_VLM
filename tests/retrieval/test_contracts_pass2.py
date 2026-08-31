from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from oa_groundrag.vlm.errors import ContractError, ProcessingError
from oa_groundrag.vlm.backends.qwen3_vl.processing import (
    Qwen3VLProcessorAdapter,
)
from oa_groundrag.retrieval.contracts import (
    PASS2_OUTPUT_SCHEMA,
    RagMode,
    TextRagTask,
    route_text_rag,
)
from oa_groundrag.retrieval.pass2 import (
    build_pass2_logits_processor,
    build_pass2_messages,
    parse_pass2_output,
    pass2_constraint_identity,
    validate_prompt_fairness,
)


def _packet() -> dict:
    return {
        "items": [
            {"evidence_id": "ev_i", "knowledge_type": "interpretation"},
            {"evidence_id": "ev_c", "knowledge_type": "confounder"},
            {"evidence_id": "ev_l", "knowledge_type": "limitation"},
        ]
    }


def _output(*, citations: bool = True) -> dict:
    return {
        "schema_version": PASS2_OUTPUT_SCHEMA,
        "supporting_interpretations": [{"text": "这些纹理可能与坡体扰动相容。", "evidence_ids": ["ev_i"] if citations else []}],
        "alternative_explanations": [{"text": "还可能是道路切坡。", "evidence_ids": ["ev_c"] if citations else []}],
        "limitations": [{"text": "单时相光学证据不足。", "evidence_ids": ["ev_l"] if citations else []}],
        "recommended_verification": [{"text": "建议补充多时相核查。", "evidence_ids": ["ev_l"] if citations else []}],
        "summary": {"text": "当前仅能保守解释候选区域，不能确认滑坡。", "evidence_ids": ["ev_i", "ev_l"] if citations else []},
    }


class _FakeTokenizer:
    pad_token_id = 0


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _CharTokenizer:
    all_special_ids = [0]
    eos_token_id = 0

    def __len__(self):
        return 128

    def encode(self, value, add_special_tokens=False):
        return [ord(character) for character in value]

    def decode(self, values, **kwargs):
        return "".join(chr(value) for value in values)


class ContractsPass2Test(unittest.TestCase):
    def test_task_router_is_explicit_and_unknown_rejected(self) -> None:
        self.assertFalse(route_text_rag(TextRagTask.SCENE_DESCRIPTION))
        self.assertFalse(route_text_rag(TextRagTask.MASK_GROUNDED_REGION_DESCRIPTION))
        self.assertTrue(route_text_rag(TextRagTask.CANDIDATE_INTERPRETATION))
        self.assertTrue(route_text_rag(TextRagTask.PROFESSIONAL_QA))
        with self.assertRaises(ContractError):
            route_text_rag("guess_from_free_text")

    def test_strict_json_duplicate_nan_and_unknown_field_rejected(self) -> None:
        valid = _output()
        text = json.dumps(valid, ensure_ascii=False)
        duplicate = text.replace('{"schema_version"', '{"schema_version":"x","schema_version"', 1)
        with self.assertRaises(ContractError):
            parse_pass2_output(duplicate, mode=RagMode.TEXT_RAG, packet=_packet())
        with self.assertRaises(ContractError):
            parse_pass2_output(text.replace('"schema_version"', '"extra":NaN,"schema_version"', 1), mode=RagMode.TEXT_RAG, packet=_packet())
        invalid = dict(valid, extra=True)
        with self.assertRaises(ContractError):
            parse_pass2_output(invalid, mode=RagMode.TEXT_RAG, packet=_packet())

    def test_evidence_binding_and_type_are_strict(self) -> None:
        self.assertEqual(
            parse_pass2_output(_output(), mode=RagMode.TEXT_RAG, packet=_packet()).summary.evidence_ids,
            ("ev_i", "ev_l"),
        )
        unknown = _output()
        unknown["supporting_interpretations"][0]["evidence_ids"] = ["ev_unknown"]
        with self.assertRaises(ContractError):
            parse_pass2_output(unknown, mode=RagMode.TEXT_RAG, packet=_packet())
        wrong_type = _output()
        wrong_type["limitations"][0]["evidence_ids"] = ["ev_i"]
        with self.assertRaises(ContractError):
            parse_pass2_output(wrong_type, mode=RagMode.TEXT_RAG, packet=_packet())

    def test_no_rag_requires_empty_citations(self) -> None:
        parse_pass2_output(_output(citations=False), mode=RagMode.NO_RAG, packet={"items": []})
        with self.assertRaises(ContractError):
            parse_pass2_output(_output(), mode=RagMode.NO_RAG, packet={"items": []})

    def test_geometry_and_candidate_upgrade_are_rejected(self) -> None:
        geometry = _output()
        geometry["summary"] = {"text": "bbox 为 [1,2,3,4]。", "evidence_ids": ["ev_i"]}
        with self.assertRaises(ContractError):
            parse_pass2_output(geometry, mode=RagMode.TEXT_RAG, packet=_packet())
        upgraded = _output()
        upgraded["summary"] = {"text": "该区域确定为滑坡。", "evidence_ids": ["ev_i"]}
        with self.assertRaises(ContractError):
            parse_pass2_output(upgraded, mode=RagMode.TEXT_RAG, packet=_packet())
        negated = _output()
        negated["limitations"] = [{
            "text": "未明确区分该区域为滑坡体，缺乏滑坡体特征的直接证据。",
            "evidence_ids": ["ev_l"],
        }]
        parse_pass2_output(negated, mode=RagMode.TEXT_RAG, packet=_packet())

    def test_confounder_wording_is_not_misclassified_as_forbidden_claim(self) -> None:
        confounder = _output()
        confounder["alternative_explanations"] = [{
            "text": (
                "混淆因素可能包括植被差异或人类工程活动导致的局部地表扰动，"
                "这些现象可能被误认为目标区域为滑坡或裸露区域。"
            ),
            "evidence_ids": ["ev_c"],
        }]
        parse_pass2_output(confounder, mode=RagMode.TEXT_RAG, packet=_packet())

        no_rag = _output(citations=False)
        no_rag["alternative_explanations"] = [{
            "text": "植被亮度差异可能被误认为目标区域为滑坡或裸露区域。",
            "evidence_ids": [],
        }]
        parse_pass2_output(no_rag, mode=RagMode.NO_RAG, packet={"items": []})

    def test_true_trigger_and_post_qualifier_upgrade_remain_rejected(self) -> None:
        trigger = _output()
        trigger["summary"] = {"text": "暴雨导致该区域发生滑坡。", "evidence_ids": ["ev_i"]}
        with self.assertRaises(ContractError):
            parse_pass2_output(trigger, mode=RagMode.TEXT_RAG, packet=_packet())

        upgraded = _output()
        upgraded["summary"] = {
            "text": "该区域可能被误认为滑坡，但该区域确定为滑坡。",
            "evidence_ids": ["ev_i"],
        }
        with self.assertRaises(ContractError):
            parse_pass2_output(upgraded, mode=RagMode.TEXT_RAG, packet=_packet())

    def test_prompt_fairness_only_allows_evidence_difference(self) -> None:
        common = {
            "question": "如何解释？",
            "target_status": "target_present",
            "program_facts": {"mask": {"area_pixels": 4}},
            "observation": {"target_appearance": {"tone": "较亮"}},
        }
        no_rag = build_pass2_messages(**common, packet=None)
        packet = {
            "items": [{
                "evidence_id": "ev_i", "knowledge_type": "interpretation", "content": "规则",
                "applicable_modalities": ["optical"], "conditions": [], "source_id": "s",
                "source_title": "t", "pdf_page": 1, "section": None,
                "authority_class": "paper", "source_status": "published",
            }]
        }
        text_rag = build_pass2_messages(**common, packet=packet)
        self.assertEqual(len(validate_prompt_fairness(no_rag, text_rag)), 64)
        changed = build_pass2_messages(**{**common, "question": "另一个问题"}, packet=packet)
        with self.assertRaises(ContractError):
            validate_prompt_fairness(no_rag, changed)

    def test_text_only_processor_keeps_visual_contract_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("transformers.AutoProcessor.from_pretrained", return_value=_FakeProcessor()):
                processor = Qwen3VLProcessorAdapter(
                    processor_path=root,
                    local_files_only=True,
                    trust_remote_code=False,
                    min_pixels=1,
                    max_pixels=4,
                    max_images=3,
                    max_input_tokens=16,
                )
            messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
            encoded = processor.encode_text_inference(messages)
            self.assertEqual(encoded.image_count, 0)
            self.assertEqual(encoded.input_token_count, 3)
            with self.assertRaises(ProcessingError):
                processor.encode_inference(messages)
            prefilled = processor.encode_text_inference([
                *messages,
                {"role": "assistant", "content": [{"type": "text", "text": "{"}]},
            ])
            self.assertEqual(prefilled.image_count, 0)

    def test_json_constraint_binds_only_current_packet_ids(self) -> None:
        constraint = build_pass2_logits_processor(
            tokenizer=_CharTokenizer(),
            prompt_length=10,
            mode=RagMode.TEXT_RAG,
            packet=_packet(),
        )
        self.assertEqual(
            constraint.identity["citation_ids"],
            {
                "supporting_interpretations": ["ev_i"],
                "alternative_explanations": ["ev_c"],
                "limitations": ["ev_l"],
                "recommended_verification": ["ev_l"],
                "summary": ["ev_i"],
            },
        )
        self.assertEqual(
            constraint.identity,
            pass2_constraint_identity(mode=RagMode.TEXT_RAG, packet=_packet()),
        )
        self.assertEqual(
            pass2_constraint_identity(mode=RagMode.NO_RAG, packet=_packet())["citation_ids"],
            {field: [] for field in (
                "supporting_interpretations",
                "alternative_explanations",
                "limitations",
                "recommended_verification",
                "summary",
            )},
        )


if __name__ == "__main__":
    unittest.main()
