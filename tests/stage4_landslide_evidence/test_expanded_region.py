from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.expanded_region import (
    COLLECTION_COUNT,
    EXTENSION_CONFIG_PATH,
    EXTENSION_COUNT,
    FINAL_PER_SOURCE,
    SELECTION_SEED,
    TRAIN_COLLECTION_MEMBER_SCHEMA,
    ArtifactBinding,
    RegionBenchmarkAccess,
    _validate_inputs,
    _ordinary_root,
    _resolve_collection_entries,
    load_expanded_region_config,
    select_extension_rows,
)
from oa_groundrag.phase3.common import atomic_write_jsonl, canonical_json, sha256_text


def candidate(
    sample_id: str,
    source: str,
    ratio: float,
    *,
    parent: str | None = None,
    split: str = "train",
    component_count: int = 1,
    location: str = "center",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "foreground_ratio": ratio,
        "source_group_id": parent,
        "group_status": "known" if parent is not None else "sample_only",
        "component_count": component_count,
        "location_3x3": location,
    }


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        mask = np.zeros((1, 4, 4), dtype=np.uint8)
        mask[0, 1:3, 1:3] = 1
        return {
            "sample_id": row["sample_id"],
            "metadata": {"split": "train", "source": row["source"]},
            "mask": FakeTensor(mask),
        }


class ExpandedRegionTest(unittest.TestCase):
    def test_frozen_config_contract(self) -> None:
        config = load_expanded_region_config(EXTENSION_CONFIG_PATH)
        self.assertEqual(config.seed, SELECTION_SEED)
        self.assertEqual(config.extension_per_source * len(config.source_order), EXTENSION_COUNT)
        self.assertEqual(config.final_per_source, FINAL_PER_SOURCE)
        self.assertEqual(config.final_per_source * len(config.source_order), COLLECTION_COUNT)
        self.assertEqual(config.no_target_max_ratio, 0.20)

    def test_selection_is_deterministic_stratified_and_excludes_identities(self) -> None:
        rows: list[dict[str, object]] = []
        base_records: list[dict[str, object]] = []
        eval_parents: set[str] = set()
        for source in ("alpha", "beta"):
            base_records.append({
                "sample_id": f"{source}-base",
                "parent_id": f"{source}-base-parent",
                "source": source,
                "split": "train",
                "target_status": "target_present",
            })
            rows.append(candidate(
                f"{source}-base", source, 0.02, parent=f"{source}-base-parent"
            ))
            eval_parent = f"{source}-eval-parent"
            eval_parents.add(eval_parent)
            rows.append(candidate(f"{source}-excluded", source, 0.02, parent=eval_parent))
            for index in range(2):
                rows.append(candidate(f"{source}-empty-{index}", source, 0.0))
            for name, ratio in (("small", 0.005), ("medium", 0.05), ("large", 0.20)):
                for index in range(2):
                    rows.append(candidate(
                        f"{source}-{name}-{index}",
                        source,
                        ratio,
                        component_count=1 if index == 0 else 2,
                        location=("center", "top_left")[index],
                    ))

        def facts(row: dict[str, object]) -> dict[str, object]:
            return {
                "component_count": row["component_count"],
                "location_3x3": row["location_3x3"],
            }

        arguments = {
            "base_records": base_records,
            "eval_parent_ids": eval_parents,
            "source_order": ("alpha", "beta"),
            "per_source_count": 6,
            "final_per_source_count": 7,
            "seed": 17,
            "no_target_max_ratio": 0.34,
            "candidate_pool_multiplier": 2,
            "candidate_pool_min": 2,
            "small_upper_exclusive": 0.01,
            "medium_upper_exclusive": 0.10,
            "facts_provider": facts,
        }
        first, report = select_extension_rows(rows, **arguments)
        second, second_report = select_extension_rows(list(reversed(rows)), **arguments)
        self.assertEqual(first, second)
        self.assertEqual(report, second_report)
        self.assertEqual(len(first), 12)
        self.assertFalse(any(sample_id.endswith("-base") for sample_id in first))
        self.assertFalse(any(sample_id.endswith("-excluded") for sample_id in first))
        for source in ("alpha", "beta"):
            self.assertEqual(report[source]["selected_count"], 6)
            self.assertEqual(report[source]["no_target_count"], 2)
            self.assertEqual(report[source]["size_counts"], {"small": 2, "medium": 1, "large": 1})

    def test_selection_rejects_val_or_test_rows(self) -> None:
        with self.assertRaises(LandslideEvidenceError) as raised:
            select_extension_rows(
                [candidate("bad", "alpha", 0.1, split="test")],
                base_records=[],
                eval_parent_ids=set(),
                source_order=("alpha",),
                per_source_count=1,
                final_per_source_count=1,
                seed=1,
                no_target_max_ratio=0.2,
                candidate_pool_multiplier=1,
                candidate_pool_min=1,
                small_upper_exclusive=0.01,
                medium_upper_exclusive=0.10,
                facts_provider=lambda row: {"component_count": 1, "location_3x3": "center"},
            )
        self.assertEqual(raised.exception.code, "SPLIT_FORBIDDEN")

    def test_no_target_cap_is_applied_to_final_base_plus_extension(self) -> None:
        rows = [
            candidate("empty-1", "alpha", 0.0),
            candidate("empty-2", "alpha", 0.0),
            candidate("small", "alpha", 0.005),
            candidate("medium", "alpha", 0.05),
            candidate("large", "alpha", 0.20),
        ]
        selected, report = select_extension_rows(
            rows,
            base_records=[{
                "sample_id": "base-empty",
                "parent_id": "base-empty",
                "source": "alpha",
                "split": "train",
                "target_status": "no_target",
            }],
            eval_parent_ids=set(),
            source_order=("alpha",),
            per_source_count=3,
            final_per_source_count=4,
            seed=9,
            no_target_max_ratio=0.5,
            candidate_pool_multiplier=1,
            candidate_pool_min=1,
            small_upper_exclusive=0.01,
            medium_upper_exclusive=0.10,
            facts_provider=lambda row: {"component_count": 1, "location_3x3": "center"},
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(report["alpha"]["base_no_target_count"], 1)
        self.assertEqual(report["alpha"]["no_target_count"], 1)
        self.assertEqual(report["alpha"]["final_no_target_count"], 2)
        self.assertEqual(report["alpha"]["final_no_target_cap"], 2)

    def test_benchmark_access_uses_bounded_lru(self) -> None:
        rows = [
            {
                "sample_id": f"sample-{index}",
                "source": "synthetic",
                "foreground_ratio": 0.25,
                "record_sha256": str(index) * 64,
                "source_record_sha256": str(index) * 64,
                "storage": {"shard": "data/synthetic/train/a.h5", "row": index},
            }
            for index in range(3)
        ]
        access = RegionBenchmarkAccess.__new__(RegionBenchmarkAccess)
        access.allowed_split = "train"
        access.rows_by_id = {str(row["sample_id"]): row for row in rows}
        access.split_ids = {"train": set(access.rows_by_id), "val": set(), "test": set()}
        access.dataset = FakeDataset(rows)
        access.dataset_indices = {str(row["sample_id"]): index for index, row in enumerate(rows)}
        access._cache_size = 2
        access._cache = OrderedDict()
        access._verify_shard = lambda relative: "0" * 64
        for row in rows:
            access.load(str(row["sample_id"]))
        self.assertEqual(list(access._cache), ["sample-1", "sample-2"])
        access.load("sample-1")
        self.assertEqual(list(access._cache), ["sample-2", "sample-1"])

    def test_train_extension_validation_never_opens_val_source(self) -> None:
        config = load_expanded_region_config(EXTENSION_CONFIG_PATH)
        eval_records = [
            {"record_id": f"eval-{index}", "parent_id": f"eval-parent-{index}"}
            for index in range(config.eval_baseline_parent_count)
        ]
        with (
            patch(
                "oa_groundrag.landslide_evidence.region_validation.validate_region_corpus",
                return_value={"valid": True},
            ) as validate_train,
            patch(
                "oa_groundrag.landslide_evidence.region_validation.validate_eval_dev",
                return_value={"valid": True},
            ) as validate_eval,
            patch(
                "oa_groundrag.landslide_evidence.expanded_region._check_binding",
            ),
            patch(
                "oa_groundrag.landslide_evidence.expanded_region.read_json",
                return_value={
                    "selection": {
                        "baseline_record_ids": [row["record_id"] for row in eval_records],
                    },
                },
            ),
            patch(
                "oa_groundrag.landslide_evidence.expanded_region.read_jsonl",
                side_effect=lambda path: eval_records if path.name == "records.jsonl"
                and path.parent == config.eval_dev.root else [],
            ),
        ):
            _, _, _, parents = _validate_inputs(config, verify_source=True)
        validate_train.assert_called_once_with(config.base_corpus.root, verify_source=True)
        validate_eval.assert_called_once_with(
            config.eval_dev.root,
            train_corpus_root=config.base_corpus.root,
            verify_source=False,
        )
        self.assertEqual(parents, {row["parent_id"] for row in eval_records})

    def test_collection_index_resolves_member_root_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = {"base": base / "base", "extension": base / "extension"}
            manifest_members: dict[str, object] = {}
            source_rows: list[dict[str, object]] = []
            for name, root in roots.items():
                root.mkdir()
                record = {
                    "record_id": f"record-{name}",
                    "sample_id": f"sample-{name}",
                    "parent_id": f"parent-{name}",
                    "source": "synthetic",
                    "target_status": "target_present",
                }
                queue = {
                    "record_id": record["record_id"],
                    "asset_identity_sha256": sha256_text(name),
                }
                atomic_write_jsonl(root / "records.jsonl", [record])
                atomic_write_jsonl(root / "annotation_queue.jsonl", [queue])
                binding = ArtifactBinding(
                    root=root,
                    schema_version="schema",
                    manifest_sha256=sha256_text(f"{name}-manifest"),
                    records_sha256=sha256_text(f"{name}-records"),
                    ledger_sha256=sha256_text(f"{name}-ledger"),
                    record_count=1,
                )
                manifest_members[name] = binding.to_dict()
                source_rows.append({
                    "schema_version": TRAIN_COLLECTION_MEMBER_SCHEMA,
                    "ordinal": len(source_rows),
                    "member": name,
                    **{field: record[field] for field in (
                        "record_id", "sample_id", "parent_id", "source", "target_status"
                    )},
                    "record_sha256": sha256_text(canonical_json(record)),
                    "asset_identity_sha256": queue["asset_identity_sha256"],
                })
            entries = _resolve_collection_entries(
                base,
                {"members": manifest_members},
                source_rows,
            )
            self.assertEqual(entries[0].member_root, roots["base"])
            self.assertEqual(entries[1].member_root, roots["extension"])
            tampered = [dict(row) for row in source_rows]
            tampered[0]["record_sha256"] = "f" * 64
            with self.assertRaises(LandslideEvidenceError):
                _resolve_collection_entries(base, {"members": manifest_members}, tampered)

    def test_symlink_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir()
            link = base / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(LandslideEvidenceError):
                _ordinary_root(link, location="fixture")


if __name__ == "__main__":
    unittest.main()
