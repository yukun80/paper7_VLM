from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oa_groundrag.phase3.common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.region_validation import _validate_files, validate_eval_dev


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
                "oa_groundrag.landslide_evidence.region_validation.validate_region_corpus",
                return_value={"manifest_sha256": "a" * 64},
            ), patch(
                "oa_groundrag.landslide_evidence.region_validation._validate_common",
                return_value=(manifest, [{"path": "counterfactual_groups.jsonl"}], baselines, set(), None),
            ):
                with self.assertRaises(LandslideEvidenceError):
                    validate_eval_dev(root, train_corpus_root=train, verify_source=False)


if __name__ == "__main__":
    unittest.main()
