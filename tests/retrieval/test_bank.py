from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from oa_groundrag.phase3.common import sha256_file, sha256_text
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.errors import ContractError
from oa_groundrag.text_rag.bank import (
    RapidOCRAdapter,
    audit_sources,
    build_evidence_units,
    classify_unit,
    page_quality,
    split_page_units,
)
from oa_groundrag.text_rag.contracts import (
    DevBinding,
    DenseConfig,
    ExtractionConfig,
    KnowledgeType,
    RetrievalConfig,
    SourceEntry,
    Stage5Binding,
    Stage6Config,
    TextRagTask,
    load_source_registry,
)


def _registry_text(source_sha: str, *, relative: str = "one.pdf") -> str:
    return f"""schema_version: oa_groundrag.text_rag.source_registry.v1
source_root: sources
sources:
  - source_id: source_one
    title: One
    source_kind: paper
    authority_class: peer_reviewed
    source_status: published
    publication_year: 2020
    standard_number: null
    language: zh_en
    relative_path: {relative}
    expected_sha256: {source_sha}
    parser_profile: paper
    default_modalities: [general]
    retrieval_enabled: true
"""


def _config(root: Path, registry: Path) -> Stage6Config:
    return Stage6Config(
        config_path=root / "config.yaml",
        source_registry_path=registry,
        bank_root=root / "bank",
        retrieval_root=root / "retrieval",
        generation_root=root / "run",
        extraction=ExtractionConfig(20, 10, 0.05, 100, 10, 128, False),
        dense=DenseConfig("fake", "0" * 40, root / "model", "cpu", 2, 128),
        retrieval=RetrievalConfig(4, 4, 60, 2, 2, 2),
        stage5=Stage5Binding(root / "stage5.yaml", "0" * 64, "1" * 64, "2" * 64, "3" * 64),
        dev=DevBinding(root / "dev", "4" * 64, root / "pred", "5" * 64, "6" * 64, "q", TextRagTask.CANDIDATE_INTERPRETATION, 1, 1),
        semantic_sha256="7" * 64,
    )


class BankTest(unittest.TestCase):
    def test_registry_identity_sha_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            pdf = sources / "one.pdf"
            pdf.write_bytes(b"pdf")
            registry = root / "sources.yaml"
            registry.write_text(_registry_text(sha256_file(pdf)), encoding="utf-8")
            loaded = load_source_registry(registry)
            self.assertEqual(loaded.sources[0].expected_sha256, sha256_file(pdf))
            self.assertEqual(len(loaded.semantic_sha256), 64)
            registry.write_text(_registry_text(sha256_file(pdf), relative="../escape.pdf"), encoding="utf-8")
            with self.assertRaises(Exception):
                load_source_registry(registry)

    def test_registry_duplicate_key_and_symlink_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sources").mkdir()
            pdf = root / "sources" / "one.pdf"
            pdf.write_bytes(b"pdf")
            registry = root / "sources.yaml"
            registry.write_text(_registry_text(sha256_file(pdf)).replace("source_root: sources", "source_root: sources\nsource_root: sources"), encoding="utf-8")
            with self.assertRaises(Exception):
                load_source_registry(registry)
            target = root / "target.pdf"
            target.write_bytes(b"pdf")
            pdf.unlink()
            pdf.symlink_to(target)
            registry.write_text(_registry_text(sha256_file(target)), encoding="utf-8")
            with self.assertRaises(Exception):
                load_source_registry(registry)

    def test_pdf_page_traceability_and_quality_status(self) -> None:
        import pymupdf as fitz

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            pdf = sources / "one.pdf"
            document = fitz.open()
            first = document.new_page()
            first.insert_text((72, 72), "Landslide remote sensing morphology scarp texture and vegetation evidence." * 2)
            document.new_page()
            document.save(pdf)
            document.close()
            registry = root / "sources.yaml"
            registry.write_text(_registry_text(sha256_file(pdf)), encoding="utf-8")
            pages, environment = audit_sources(_config(root, registry))
            self.assertEqual([row["pdf_page"] for row in pages], [1, 2])
            self.assertEqual(pages[0]["extraction_method"], "text_layer")
            self.assertEqual(pages[0]["quality_status"], "usable")
            self.assertEqual(pages[1]["quality_status"], "ocr_required")
            self.assertEqual(environment["page_counts"], {"source_one": 2})

    def test_section_aware_split_and_long_natural_boundary(self) -> None:
        text = "1 General\nLandslide scarp texture is visible in optical imagery.\n\n2 Limits\nSingle image resolution cannot determine activity."
        units = split_page_units(
            text,
            parser_profile="standard",
            token_count=lambda value: len(value.split()),
            min_chars=5,
            max_tokens=8,
        )
        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(any(value["clause"] == "1" for value in units))
        self.assertTrue(all(len(value["content"].split()) <= 8 for value in units))

    def test_classification_priority_and_unclassified(self) -> None:
        self.assertEqual(classify_unit("光学纹理可作为滑坡识别标志")[0], KnowledgeType.INTERPRETATION)
        self.assertEqual(classify_unit("采石场可能是混淆对象")[0], KnowledgeType.CONFOUNDER)
        self.assertEqual(classify_unit("单时相分辨率限制导致无法确定活动性")[0], KnowledgeType.LIMITATION)
        self.assertEqual(classify_unit("普通说明文字")[0], KnowledgeType.UNCLASSIFIED)

    def test_exact_duplicate_has_single_canonical_index(self) -> None:
        source = SourceEntry(
            "source_one", "One", "paper", "peer_reviewed", "published", 2020, None,
            "zh_en", "one.pdf", "a" * 64, "paper", ("general",), True, Path("one.pdf"),
        )
        text = "Landslide remote sensing texture and scarp morphology interpretation evidence."
        pages = [
            {"quality_status": "usable", "source_id": "source_one", "pdf_page": page, "text": text, "extraction_method": "text_layer"}
            for page in (1, 2)
        ]
        units = build_evidence_units(pages=pages, sources=(source,), token_count=lambda value: len(value.split()), min_chars=10, max_tokens=128)
        self.assertEqual(len(units), 2)
        self.assertEqual(sum(bool(row["indexed"]) for row in units), 1)
        self.assertEqual(sum(row["duplicate_of"] is not None for row in units), 1)

    def test_ocr_result_adapter_and_page_quality(self) -> None:
        adapter = RapidOCRAdapter.__new__(RapidOCRAdapter)
        adapter.engine = lambda image: type("Result", (), {"txts": ["滑坡", "识别"]})()
        self.assertEqual(adapter.extract(object()), "滑坡\n识别")
        self.assertGreater(page_quality("滑坡识别")["useful_char_count"], 0)

    def test_atomic_publish_rejects_overwrite_and_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "artifact"
            with AtomicArtifactDirectory(target) as writer:
                writer.write_json("value.json", {"ok": True})
                writer.publish()
            with self.assertRaises(ContractError):
                with AtomicArtifactDirectory(target):
                    pass
            linked = root / "linked"
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ContractError):
                with AtomicArtifactDirectory(linked / "child"):
                    pass


if __name__ == "__main__":
    unittest.main()
