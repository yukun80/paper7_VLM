from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oa_groundrag.data.rs_general.io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.data.grounded.contracts import (
    EXPECTED_IDENTITY_FIELDS,
    LandslideEvidenceError,
)
from oa_groundrag.data.grounded.region_validation import (
    _validate_files,
    validate_eval_dev,
    validate_region_corpus,
)


def minimal_ledger_root(root: Path) -> None:
    payload = root / "payload.txt"
    payload.write_text("stable", encoding="utf-8")
    ledger = [{"path": "payload.txt", "size_bytes": payload.stat().st_size, "sha256": sha256_file(payload)}]
    atomic_write_jsonl(root / "SHA256SUMS.jsonl", ledger)
    atomic_write_json(root / "manifest.json", {
        "ledger": {
            "path": "SHA256SUMS.jsonl",
            "entry_count": 1,
            "size_bytes": (root / "SHA256SUMS.jsonl").stat().st_size,
            "file_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
            "root_sha256": sha256_text(canonical_json(ledger)),
        }
    })


class RegionValidationTest(unittest.TestCase):
    @staticmethod
    def _benchmark_identity() -> dict[str, str]:
        return {
            field: (
                "oa_auxseg_hdf5_v1"
                if field == "schema_version"
                else sha256_text(field)
            )
            for field in EXPECTED_IDENTITY_FIELDS
        }

    def test_ledger_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            minimal_ledger_root(root)
            (root / "payload.txt").write_text("tampered", encoding="utf-8")
            with self.assertRaises(LandslideEvidenceError):
                _validate_files(root)

    def test_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            minimal_ledger_root(root)
            (root / "link.txt").symlink_to(root / "payload.txt")
            with self.assertRaises(LandslideEvidenceError):
                _validate_files(root)

    def test_hardlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            minimal_ledger_root(root)
            os.link(root / "payload.txt", root / "alias.txt")
            with self.assertRaises(LandslideEvidenceError):
                _validate_files(root)

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / "outside-stage4-test.txt"
            outside.write_text("x", encoding="utf-8")
            ledger = [{"path": "../outside-stage4-test.txt", "size_bytes": 1, "sha256": sha256_file(outside)}]
            atomic_write_jsonl(root / "SHA256SUMS.jsonl", ledger)
            atomic_write_json(root / "manifest.json", {
                "ledger": {
                    "path": "SHA256SUMS.jsonl", "entry_count": 1,
                    "size_bytes": (root / "SHA256SUMS.jsonl").stat().st_size,
                    "file_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
                    "root_sha256": sha256_text(canonical_json(ledger)),
                }
            })
            try:
                with self.assertRaises(Exception):
                    _validate_files(root)
            finally:
                outside.unlink()

    def test_counterfactual_group_completeness_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, train = Path(temporary) / "eval", Path(temporary) / "train"
            root.mkdir(); train.mkdir()
            atomic_write_jsonl(train / "records.jsonl", [])
            baselines = [
                {"record_id": f"b{i}", "sample_id": f"s{i}", "parent_id": f"p{i}",
                 "target_status": "target_present" if i == 0 else "no_target"}
                for i in range(100)
            ]
            atomic_write_jsonl(root / "counterfactual_groups.jsonl", [{
                "schema_version": "oa_groundrag.oa_grounded_eval_dev.counterfactual_group.v1",
                "variants": {"baseline_correct_mask": "b0"},
            }])
            manifest = {
                "development_only": True, "sealed_test_accessed": False,
                "selection": {"baseline_record_ids": [row["record_id"] for row in baselines]},
            }
            with patch(
                "oa_groundrag.data.grounded.region_validation.validate_region_corpus",
                return_value={"manifest_sha256": "a" * 64},
            ), patch(
                "oa_groundrag.data.grounded.region_validation._validate_common",
                return_value=(manifest, [{"path": "counterfactual_groups.jsonl"}], baselines, set(), None),
            ):
                with self.assertRaises(LandslideEvidenceError):
                    validate_eval_dev(root, train_corpus_root=train, verify_source=False)

    def test_corpus_id_and_eval_train_binding_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            corpus_root = base / "corpus"
            corpus_root.mkdir()
            sample_ids = [f"sample-{index}" for index in range(500)]
            corpus_manifest = {
                "corpus_name": "mask_grounded_region_corpus_train_v1_500",
                "corpus_id": "0" * 64,
                "silver_generated": False,
                "benchmark": {"root": str(base / "benchmark"), **self._benchmark_identity()},
                "selection": {
                    "config_semantic_sha256": "1" * 64,
                    "ordered_sample_ids": sample_ids,
                    "ordered_sample_ids_sha256": sha256_text(canonical_json(sample_ids)),
                },
            }
            records = [{"sample_id": sample_id} for sample_id in sample_ids]
            with patch(
                "oa_groundrag.data.grounded.region_validation._validate_common",
                return_value=(corpus_manifest, [], records, set(), None),
            ):
                with self.assertRaises(LandslideEvidenceError) as raised:
                    validate_region_corpus(corpus_root, verify_source=False)
            self.assertEqual(raised.exception.code, "SELECTION_IDENTITY_MISMATCH")

            train_root = base / "train"
            eval_root = base / "eval"
            train_root.mkdir()
            eval_root.mkdir()
            atomic_write_jsonl(train_root / "records.jsonl", [])
            baselines = [
                {
                    "record_id": f"baseline-{index}",
                    "sample_id": f"val-{index}",
                    "parent_id": f"parent-{index}",
                    "target_status": "no_target",
                }
                for index in range(100)
            ]
            train_report = {
                "manifest_sha256": "a" * 64,
                "records_sha256": "b" * 64,
                "ledger_sha256": "c" * 64,
            }
            eval_manifest = {
                "development_only": True,
                "sealed_test_accessed": False,
                "eval_id": "d" * 64,
                "benchmark": {"root": str(base / "benchmark"), **self._benchmark_identity()},
                "train_corpus": {
                    "root": str(base / "wrong-train"),
                    **train_report,
                    "parent_count": 0,
                },
                "selection": {
                    "config_semantic_sha256": "2" * 64,
                    "sample_count": 100,
                    "baseline_record_ids": [row["record_id"] for row in baselines],
                    "ordered_sample_ids": [row["sample_id"] for row in baselines],
                    "ordered_parent_ids": [row["parent_id"] for row in baselines],
                },
            }
            with patch(
                "oa_groundrag.data.grounded.region_validation.validate_region_corpus",
                return_value=train_report,
            ), patch(
                "oa_groundrag.data.grounded.region_validation._validate_common",
                return_value=(eval_manifest, [], baselines, set(), None),
            ):
                with self.assertRaises(LandslideEvidenceError) as raised:
                    validate_eval_dev(
                        eval_root,
                        train_corpus_root=train_root,
                        verify_source=False,
                    )
            self.assertEqual(raised.exception.code, "FACT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
