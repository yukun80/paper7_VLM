from __future__ import annotations

import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from fixture_helpers import make_all_sources, write_build_config
from oa_groundrag.phase3.builder import _payload_hashes, build_benchmark
from oa_groundrag.phase3.cli import main
from oa_groundrag.phase3.common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.config import load_build_config
from oa_groundrag.phase3.contracts import (
    BENCHMARK_SCOPE,
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_VERSION,
    RELEASE_EQUIVALENCE_SCHEMA_VERSION,
)
from oa_groundrag.phase3.dataset import RSGeneralDescDataset
from oa_groundrag.phase3.errors import BuildError, RSGeneralDescError, ReasonCode
from oa_groundrag.phase3.hash_ledger import HashLedgerVerifier
from oa_groundrag.phase3.validator import validate_benchmark


class RepackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        roots = make_all_sources(self.base / "sources")
        native = build_benchmark(
            load_build_config(
                write_build_config(
                    self.base / "build.yaml",
                    roots,
                    self.base / "native-fixture",
                    profile="full",
                    max_assets=None,
                )
            )
        )
        self.source = self.base / "predecessor"
        shutil.copytree(native, self.source)
        build_config_path = self.source / "metadata/build_config.json"
        build_config = read_json(build_config_path)
        build_config["retired_component"] = {"enabled": False}
        build_config["asset_policy"]["retired_format"] = "png"
        build_config["asset_policy"]["retired_preview"] = [-3.0, 3.0]
        atomic_write_json(build_config_path, build_config)

        hashes = _payload_hashes(
            self.source,
            {
                "manifest.json",
                "metadata/hashes.json",
                "metadata/validation.json",
            },
        )
        atomic_write_json(self.source / "metadata/hashes.json", hashes)
        semantic_sha = sha256_text(canonical_json(build_config))
        manifest = read_json(self.source / "manifest.json")
        manifest["semantic_config_sha256"] = semantic_sha
        manifest["payload_root_sha256"] = hashes["root_sha256"]
        manifest["hash_manifest_sha256"] = sha256_file(
            self.source / "metadata/hashes.json"
        )
        manifest["build_id"] = "build_" + sha256_text(
            canonical_json([semantic_sha, hashes["root_sha256"]])
        )
        atomic_write_json(self.source / "manifest.json", manifest)
        saved = read_json(self.source / "metadata/validation.json")
        saved["payload_root_sha256"] = hashes["root_sha256"]
        atomic_write_json(self.source / "metadata/validation.json", saved)
        self.manifest = manifest
        self.manifest_sha = sha256_file(self.source / "manifest.json")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _argv(self, target: Path) -> list[str]:
        return [
            "repackage",
            "--source-root",
            str(self.source),
            "--target-root",
            str(target),
            "--expected-manifest-sha256",
            self.manifest_sha,
            "--expected-build-id",
            self.manifest["build_id"],
            "--expected-payload-sha256",
            self.manifest["payload_root_sha256"],
            "--expected-hash-manifest-sha256",
            self.manifest["hash_manifest_sha256"],
        ]

    def test_repackage_cli_preserves_content_roles_and_asset_bytes(self) -> None:
        target = self.base / "repackaged"
        self.assertEqual(main(self._argv(target)), 0)
        report = validate_benchmark(target, deep=True)
        self.assertEqual(report["errors"], [])
        manifest = read_json(target / "manifest.json")
        self.assertEqual(manifest["schema_version"], MANIFEST_VERSION)
        self.assertEqual(
            manifest["canonical_schema_version"],
            CANONICAL_SCHEMA_VERSION,
        )
        self.assertEqual(manifest["benchmark_scope"], BENCHMARK_SCOPE)
        self.assertTrue(manifest["formal_acceptance_eligible"])
        self.assertEqual(manifest["formal_acceptance_blockers"], [])
        equivalence = read_json(
            target / manifest["layout"]["release_equivalence"]
        )
        self.assertEqual(
            equivalence["schema_version"],
            RELEASE_EQUIVALENCE_SCHEMA_VERSION,
        )
        self.assertTrue(equivalence["record_content_equivalent"])
        before = RSGeneralDescDataset(self.source, load_assets=False)
        after = RSGeneralDescDataset(target, load_assets=False)
        self.assertEqual(len(before), len(after))
        self.assertEqual(
            Counter(row["logical_role"] for row in before.records),
            Counter(row["logical_role"] for row in after.records),
        )
        before_assets = read_json(
            self.source / self.manifest["layout"]["assets"]
        )
        after_assets = read_json(target / manifest["layout"]["assets"])
        self.assertEqual(before_assets, after_assets)
        for asset in after_assets:
            path = target / asset["path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_nlink, 1)

    def test_repackage_rejects_identity_drift_and_existing_target(self) -> None:
        with self.assertRaises(BuildError) as caught:
            main(
                [
                    *self._argv(self.base / "identity-mismatch"),
                ][:-8]
                + [
                    "--expected-manifest-sha256",
                    "0" * 64,
                    "--expected-build-id",
                    self.manifest["build_id"],
                    "--expected-payload-sha256",
                    self.manifest["payload_root_sha256"],
                    "--expected-hash-manifest-sha256",
                    self.manifest["hash_manifest_sha256"],
                ]
            )
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        target = self.base / "existing"
        target.mkdir()
        with self.assertRaises(BuildError) as caught:
            main(self._argv(target))
        self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_EXISTS)

        real_parent = self.base / "real-parent"
        real_parent.mkdir()
        linked_parent = self.base / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaises(BuildError) as caught:
            main(self._argv(linked_parent / "target"))
        self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_LINK)

    def test_repackage_validation_failure_never_publishes(self) -> None:
        target = self.base / "rejected"
        with mock.patch(
            "oa_groundrag.phase3.validator.validate_benchmark",
            return_value={"errors": ["injected"]},
        ):
            with self.assertRaises(BuildError) as caught:
                main(self._argv(target))
        self.assertEqual(caught.exception.code, ReasonCode.STAGING_FAILURE)
        self.assertFalse(target.exists())
        self.assertEqual(
            list(target.parent.glob(f".{target.name}.staging-*")),
            [],
        )

    def test_repackage_rejects_predecessor_record_drift_before_equivalence(
        self,
    ) -> None:
        relative = self.manifest["layout"]["record_shards"][0]["path"]
        path = self.source / relative
        rows = read_jsonl(path)
        rows[0]["instruction"] += " drift"
        atomic_write_jsonl(path, rows)
        target = self.base / "record-drift"
        with mock.patch(
            "oa_groundrag.phase3.repackage.atomic_write_json",
            wraps=atomic_write_json,
        ) as writer:
            with self.assertRaises(BuildError) as caught:
                main(self._argv(target))
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        self.assertEqual(caught.exception.details["path"], relative)
        self.assertFalse(target.exists())
        self.assertFalse(
            any(
                str(call.args[0]).endswith("metadata/release_equivalence.json")
                for call in writer.call_args_list
            )
        )

    def test_repackage_rejects_predecessor_metadata_drift(self) -> None:
        relative = self.manifest["layout"]["build_config"]
        path = self.source / relative
        value = read_json(path)
        value["workers"] += 1
        atomic_write_json(path, value)
        target = self.base / "metadata-drift"
        with self.assertRaises(BuildError) as caught:
            main(self._argv(target))
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        self.assertEqual(caught.exception.details["path"], relative)
        self.assertFalse(target.exists())

    def test_repackage_rejects_predecessor_asset_drift(self) -> None:
        inventory = read_json(
            self.source / self.manifest["layout"]["assets"]
        )
        relative = inventory[0]["path"]
        path = self.source / relative
        payload = bytearray(path.read_bytes())
        payload[len(payload) // 2] ^= 1
        path.write_bytes(payload)
        target = self.base / "asset-drift"
        with self.assertRaises(BuildError) as caught:
            main(self._argv(target))
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        self.assertEqual(caught.exception.details["path"], relative)
        self.assertFalse(target.exists())


class HashLedgerVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "payload").mkdir()
        (self.root / "metadata").mkdir()
        (self.root / "payload/a.bin").write_bytes(b"alpha")
        (self.root / "payload/b.bin").write_bytes(b"bravo")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _row(
        self,
        relative: str,
        *,
        size_bytes: int | None = None,
        digest: str | None = None,
    ) -> dict[str, object]:
        path = self.root / relative
        return {
            "path": relative,
            "size_bytes": path.stat().st_size if size_bytes is None else size_bytes,
            "sha256": sha256_file(path) if digest is None else digest,
        }

    def _verifier(
        self,
        rows: list[dict[str, object]],
        *,
        root_sha256: str | None = None,
    ) -> HashLedgerVerifier:
        declared_root = root_sha256 or sha256_bytes(
            canonical_json(rows).encode("utf-8")
        )
        atomic_write_json(
            self.root / "metadata/hashes.json",
            {
                "schema_version": "rs_generaldesc.hashes.v1",
                "files": rows,
                "root_sha256": declared_root,
            },
        )
        return HashLedgerVerifier(
            self.root,
            ledger_path="metadata/hashes.json",
            expected_ledger_sha256=sha256_file(
                self.root / "metadata/hashes.json"
            ),
            expected_root_sha256=declared_root,
        )

    def test_strict_paths_order_duplicates_and_root_identity(self) -> None:
        a = self._row("payload/a.bin")
        b = self._row("payload/b.bin")
        for rows in ([a, a], [b, a]):
            with self.subTest(rows=rows):
                with self.assertRaises(RSGeneralDescError) as caught:
                    self._verifier(rows)
                self.assertEqual(caught.exception.code, ReasonCode.SCHEMA_MISMATCH)

        escaping = {
            "path": "../escape.bin",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
        with self.assertRaises(RSGeneralDescError) as caught:
            self._verifier([escaping])
        self.assertEqual(caught.exception.code, ReasonCode.PATH_ESCAPE)
        self.assertEqual(caught.exception.details["path"], "../escape.bin")

        with self.assertRaises(RSGeneralDescError) as caught:
            self._verifier([a, b], root_sha256="f" * 64)
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)

    def test_size_hash_missing_extra_and_nonregular_are_rejected(self) -> None:
        rows = [self._row("payload/a.bin"), self._row("payload/b.bin")]
        verifier = self._verifier(rows)
        verifier.assert_entry_set({"payload/a.bin", "payload/b.bin"})
        with self.assertRaises(RSGeneralDescError) as caught:
            verifier.assert_entry_set({"payload/a.bin"})
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)

        (self.root / "payload/extra.bin").write_bytes(b"extra")
        with self.assertRaises(RSGeneralDescError) as caught:
            verifier.assert_actual_file_set()
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        (self.root / "payload/extra.bin").unlink()

        (self.root / ".unexpected").write_bytes(b"hidden-extra")
        with self.assertRaises(RSGeneralDescError) as caught:
            verifier.assert_actual_file_set()
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        self.assertEqual(caught.exception.details["path"], ".unexpected")
        (self.root / ".unexpected").unlink()

        dangling = self.root / "payload/dangling.bin"
        dangling.symlink_to("missing.bin")
        with self.assertRaises(RSGeneralDescError) as caught:
            verifier.assert_actual_file_set()
        self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_LINK)
        self.assertEqual(caught.exception.details["path"], "payload/dangling.bin")
        dangling.unlink()

        (self.root / "payload/b.bin").unlink()
        with self.assertRaises(RSGeneralDescError) as caught:
            verifier.assert_actual_file_set()
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)

        (self.root / "payload/b.bin").write_bytes(b"bravo")
        wrong_size = self._verifier(
            [
                self._row("payload/a.bin", size_bytes=4),
                self._row("payload/b.bin"),
            ]
        )
        with self.assertRaises(RSGeneralDescError) as caught:
            wrong_size.verify_file("payload/a.bin")
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)
        self.assertEqual(caught.exception.details["path"], "payload/a.bin")

        wrong_hash = self._verifier(
            [
                self._row("payload/a.bin", digest="0" * 64),
                self._row("payload/b.bin"),
            ]
        )
        with self.assertRaises(RSGeneralDescError) as caught:
            wrong_hash.verify_file("payload/a.bin")
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)

        linked = self.root / "payload/a-link.bin"
        linked.hardlink_to(self.root / "payload/a.bin")
        hardlink_verifier = self._verifier(
            [
                self._row("payload/a-link.bin"),
                self._row("payload/a.bin"),
                self._row("payload/b.bin"),
            ]
        )
        with self.assertRaises(RSGeneralDescError) as caught:
            hardlink_verifier.verify_file("payload/a.bin")
        self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_LINK)

    def test_hash_cache_is_per_file_and_invalidated_by_stat_change(self) -> None:
        rows = [self._row("payload/a.bin"), self._row("payload/b.bin")]
        verifier = self._verifier(rows)
        verifier.verify_file("payload/a.bin")
        verifier.verify_file("payload/a.bin")
        verifier.verify_file("payload/b.bin")
        self.assertEqual(verifier.hash_count("payload/a.bin"), 1)
        self.assertEqual(verifier.hash_count("payload/b.bin"), 1)
        self.assertEqual(
            verifier.verified_paths,
            frozenset({"payload/a.bin", "payload/b.bin"}),
        )
        verifier.assert_all_verified()

        path = self.root / "payload/a.bin"
        path.write_bytes(b"ALPHA")
        with self.assertRaises(RSGeneralDescError) as caught:
            verifier.verify_file("payload/a.bin")
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)


if __name__ == "__main__":
    unittest.main()
