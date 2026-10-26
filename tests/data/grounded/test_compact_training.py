"""Stage 5 compact 训练资产的 raw 脱离与篡改回归。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tests.data.grounded.fixture_helpers import target_output, target_record

from oa_groundrag.data.grounded.supervision.compact_training import (
    CompactTrainingMessageDataset,
    load_compact_training_messages,
    publish_compact_training_messages,
)
from oa_groundrag.data.grounded.supervision.model_assisted import (
    EXPERT_AUTHORITY,
    MODEL_AUTHORITY,
    ModelAssistedCollectionContext,
    ModelAssistedCollectionEntry,
    ModelAssistedTrainingArtifact,
    TRAINING_MESSAGE_SCHEMA,
    TRAINING_MANIFEST_SCHEMA,
)
from oa_groundrag.data.grounded.region import region_asset_identity
from oa_groundrag.data.rs_general.io import atomic_write_json, atomic_write_jsonl, canonical_json, sha256_text
from oa_groundrag.grounding.messages import build_mask_grounded_region_messages


class CompactTrainingTests(unittest.TestCase):
    def _fixture(self, root: Path):
        member = root / "member"
        member.mkdir()
        entries = []
        source_rows = []
        for index, authority in enumerate((EXPERT_AUTHORITY, MODEL_AUTHORITY)):
            record = target_record(member, record_id=f"record-{index}")
            record["split"] = "train"
            record["parent_id"] = f"parent-{index}"
            record["source"] = "synthetic"
            asset = region_asset_identity(member, record["assets"])
            entry = ModelAssistedCollectionEntry(
                ordinal=index,
                member="fixture",
                member_root=member,
                member_manifest_sha256="a" * 64,
                record=record,
                queue={
                    "record_id": record["record_id"],
                    "split": "train",
                    "asset_identity_sha256": asset,
                },
            )
            entries.append(entry)
            target = canonical_json(target_output())
            source_rows.append({
                "schema_version": TRAINING_MESSAGE_SCHEMA,
                "record_id": record["record_id"],
                "parent_id": record["parent_id"],
                "source": record["source"],
                "logical_role": "train",
                "task_family": "mask_grounded_region_description",
                "messages": build_mask_grounded_region_messages(
                    record, asset_root=member, assistant_target=target
                ),
                "supervision_identity_sha256": sha256_text(f"supervision-{index}"),
                "asset_identity_sha256": asset,
                "supervision_authority": authority,
            })
        collection_root = root / "collection"
        collection_root.mkdir()
        collection = ModelAssistedCollectionContext(
            root=collection_root,
            manifest={"schema_version": "fixture"},
            manifest_sha256="b" * 64,
            entries=tuple(entries),
        )
        source_root = root / "raw_messages"
        source_root.mkdir()
        atomic_write_json(source_root / "manifest.json", {"fixture": True})
        atomic_write_jsonl(source_root / "messages.jsonl", source_rows)
        source = ModelAssistedTrainingArtifact(
            root=source_root,
            manifest={
                "schema_version": TRAINING_MANIFEST_SCHEMA,
                "source_collection_root": str(collection_root),
                "supervision_package_root": str(root / "raw_package"),
            },
            rows=tuple(source_rows),
        )
        history = {
            "training_manifest_sha256": "1" * 64,
            "messages_sha256": "2" * 64,
            "supervision_package_manifest_sha256": "3" * 64,
            "source_record_count": 4,
            "eligible_count": 2,
            "excluded_count": 2,
            "authority_counts": {EXPERT_AUTHORITY: 1, MODEL_AUTHORITY: 1},
            "exclusion_counts": {"parse_invalid": 2},
            "draft_provenance": [{"draft_run_id": "fixture", "prompt_sha256": "4" * 64}],
        }
        return source, collection, history

    def test_compact_loads_after_raw_dependency_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, collection, history = self._fixture(root)
            compact = root / "compact"
            with (
                patch("oa_groundrag.data.grounded.supervision.compact_training.EXPECTED_COMPACT_COUNT", 2),
                patch(
                    "oa_groundrag.data.grounded.supervision.compact_training._load_v2_source_lightweight",
                    return_value=source,
                ),
                patch(
                    "oa_groundrag.data.grounded.supervision.compact_training._load_collection_context",
                    return_value=collection,
                ) as load_collection,
                patch(
                    "oa_groundrag.data.grounded.supervision.compact_training._historical_source",
                    return_value=history,
                ),
            ):
                result = publish_compact_training_messages(
                    source_training_root=source.root,
                    output_root=compact,
                )
                self.assertEqual(result["record_count"], 2)
                # 模拟 raw package/work/messages 已删除；compact loader 不得尝试访问它们。
                for child in source.root.iterdir():
                    child.unlink()
                source.root.rmdir()
                artifact = load_compact_training_messages(compact)
                self.assertEqual(len(artifact.rows), 2)
                dataset = CompactTrainingMessageDataset(compact)
                self.assertEqual(dataset[0].logical_role, "mask_grounded_train")
                self.assertEqual(dataset[0].reference_responses[0], canonical_json(target_output()))
                self.assertTrue(load_collection.call_args_list)
                for call in load_collection.call_args_list:
                    self.assertFalse(call.kwargs["verify_members"])
                    self.assertEqual(
                        call.kwargs["eval_exclusion_policy"],
                        "retired_identity_only",
                    )

    def test_compact_rejects_assistant_and_asset_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, collection, history = self._fixture(root)
            compact = root / "compact"
            with (
                patch("oa_groundrag.data.grounded.supervision.compact_training.EXPECTED_COMPACT_COUNT", 2),
                patch("oa_groundrag.data.grounded.supervision.compact_training._load_v2_source_lightweight", return_value=source),
                patch("oa_groundrag.data.grounded.supervision.compact_training._load_collection_context", return_value=collection),
                patch("oa_groundrag.data.grounded.supervision.compact_training._historical_source", return_value=history),
            ):
                publish_compact_training_messages(source_training_root=source.root, output_root=compact)
                rows = list(__import__("oa_groundrag.data.rs_general.io", fromlist=["read_jsonl"]).read_jsonl(compact / "messages.jsonl"))
                rows[0]["assistant_target_sha256"] = "0" * 64
                atomic_write_jsonl(compact / "messages.jsonl", rows)
                with self.assertRaises(Exception):
                    load_compact_training_messages(compact)


if __name__ == "__main__":
    unittest.main()
