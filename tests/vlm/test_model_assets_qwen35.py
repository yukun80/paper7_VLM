"""Qwen3.5 固定 Hub revision 资产 ledger 的离线合同。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from oa_groundrag.vlm.backends.assets import (
    ModelAssetLedgerError,
    create_model_asset_ledger,
    verify_model_asset_ledger,
)
from oa_groundrag.vlm.errors import ReasonCode


REVISION = "1" * 40
REPO_ID = "Qwen/Qwen3.5-4B"
FILENAMES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _write_fixture(root: Path) -> None:
    root.mkdir()
    for name in FILENAMES:
        (root / name).write_bytes(f"fixture:{name}".encode("utf-8"))
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({"processor_class": "Qwen3VLProcessor"}),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    "model.a": "model.safetensors-00001-of-00002.safetensors",
                    "model.b": "model.safetensors-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )


class ModelAssetLedgerTests(unittest.TestCase):
    def test_create_and_verify_bind_every_file_and_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "model"
            ledger = base / "ledger.json"
            _write_fixture(root)
            created = create_model_asset_ledger(
                model_root=root,
                ledger_path=ledger,
                backend="qwen3_5",
                repo_id=REPO_ID,
                revision=REVISION,
            )
            verified = verify_model_asset_ledger(
                ledger,
                expected_backend="qwen3_5",
                expected_repo_id=REPO_ID,
                expected_revision=REVISION,
                expected_model_root=root,
            )
            self.assertEqual(created, verified)
            self.assertEqual(verified.file_count, 14)
            self.assertEqual(
                verified.weight_shards,
                (
                    "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors",
                ),
            )
            row = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["path"] for item in row["files"]],
                sorted(FILENAMES),
            )
            self.assertEqual(len(row["ledger_payload_sha256"]), 64)

    def test_tamper_missing_extra_and_wrong_revision_fail_closed(self) -> None:
        for mutation in ("tamper", "missing", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                root = base / "model"
                ledger = base / "ledger.json"
                _write_fixture(root)
                create_model_asset_ledger(
                    model_root=root,
                    ledger_path=ledger,
                    backend="qwen3_5",
                    repo_id=REPO_ID,
                    revision=REVISION,
                )
                if mutation == "tamper":
                    (root / "config.json").write_text("{}", encoding="utf-8")
                elif mutation == "missing":
                    (root / "README.md").unlink()
                else:
                    (root / "extra.bin").write_bytes(b"extra")
                with self.assertRaises(ModelAssetLedgerError) as caught:
                    verify_model_asset_ledger(ledger)
                self.assertEqual(
                    caught.exception.code,
                    ReasonCode.ASSET_LEDGER_INVALID,
                )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "model"
            ledger = base / "ledger.json"
            _write_fixture(root)
            create_model_asset_ledger(
                model_root=root,
                ledger_path=ledger,
                backend="qwen3_5",
                repo_id=REPO_ID,
                revision=REVISION,
            )
            with self.assertRaises(ModelAssetLedgerError):
                verify_model_asset_ledger(
                    ledger,
                    expected_revision="2" * 40,
                )

    def test_symlink_bad_index_duplicate_json_and_overwrite_are_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "model"
            ledger = base / "ledger.json"
            _write_fixture(root)
            (root / "README.md").unlink()
            target = base / "README-target.md"
            target.write_text("target", encoding="utf-8")
            (root / "README.md").symlink_to(target)
            with self.assertRaises(ModelAssetLedgerError):
                create_model_asset_ledger(
                    model_root=root,
                    ledger_path=ledger,
                    backend="qwen3_5",
                    repo_id=REPO_ID,
                    revision=REVISION,
                )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "model"
            ledger = base / "ledger.json"
            _write_fixture(root)
            (root / "model.safetensors.index.json").write_text(
                '{"metadata":{},"weight_map":{"a":"missing.safetensors"}}',
                encoding="utf-8",
            )
            with self.assertRaises(ModelAssetLedgerError):
                create_model_asset_ledger(
                    model_root=root,
                    ledger_path=ledger,
                    backend="qwen3_5",
                    repo_id=REPO_ID,
                    revision=REVISION,
                )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "model"
            ledger = base / "ledger.json"
            _write_fixture(root)
            create_model_asset_ledger(
                model_root=root,
                ledger_path=ledger,
                backend="qwen3_5",
                repo_id=REPO_ID,
                revision=REVISION,
            )
            with self.assertRaises(ModelAssetLedgerError):
                create_model_asset_ledger(
                    model_root=root,
                    ledger_path=ledger,
                    backend="qwen3_5",
                    repo_id=REPO_ID,
                    revision=REVISION,
                )
            text = ledger.read_text(encoding="utf-8")
            ledger.write_text(
                text.replace(
                    '"schema_version":',
                    '"schema_version":"duplicate","schema_version":',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ModelAssetLedgerError):
                verify_model_asset_ledger(ledger)


if __name__ == "__main__":
    unittest.main()
