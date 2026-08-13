from __future__ import annotations

from dataclasses import replace
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.data.oa_auxseg.dataset import BenchmarkDataset
from oa_groundrag.runtime.contracts import UnifiedTask
from oa_groundrag.runtime.demo.access import DemoAccessError, DemoTestAccessController
from oa_groundrag.runtime.demo.catalog import (
    BenchmarkCatalog,
    BenchmarkFilter,
    DemoCatalogError,
)
from oa_groundrag.runtime.demo.gallery import DemoGalleryError, DemoGalleryStore

from tests.runtime.demo.helpers import DatasetReadCounter, build_benchmark


class BenchmarkCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binding = build_benchmark(self.root / "benchmark")
        self.catalog = BenchmarkCatalog(self.binding)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_filters_use_index_metadata_and_existing_size_bins(self) -> None:
        self.assertEqual(
            [row.sample_id for row in self.catalog.filtered(BenchmarkFilter(split="train"))],
            ["train-small", "train-medium"],
        )
        rows = self.catalog.filtered(BenchmarkFilter(
            split="train",
            source="source_b",
            target_status="target",
            size="medium",
            modalities=("dem",),
        ))
        self.assertEqual([row.sample_id for row in rows], ["train-medium"])
        self.assertEqual(
            [row.sample_id for row in self.catalog.filtered(BenchmarkFilter(
                split="val", target_status="no_target", size="empty", modalities=("slope",),
            ))],
            ["val-empty"],
        )

    def test_navigation_is_canonical_and_does_not_wrap(self) -> None:
        filters = BenchmarkFilter(split="train")
        first, position, total = self.catalog.navigate(
            filters, current_sample_id="train-small", delta=-1
        )
        self.assertEqual((first.sample_id, position, total), ("train-small", 0, 2))
        last, position, total = self.catalog.navigate(
            filters, current_sample_id="train-medium", delta=1
        )
        self.assertEqual((last.sample_id, position, total), ("train-medium", 1, 2))

    def test_random_is_limited_to_manual_filter_and_exact_lookup_is_unique(self) -> None:
        filters = BenchmarkFilter(split="val", source="source_b")
        self.assertEqual(
            self.catalog.random_record(filters, chooser=random.Random(1)).sample_id,
            "val-large",
        )
        located = self.catalog.locate("test-small")
        self.assertEqual((located.split, located.source), ("test", "source_c"))

    def test_test_lock_refuses_before_dataset_getitem(self) -> None:
        controller = DemoTestAccessController(
            demo_root=self.root / "demo",
            allow_test_demo=False,
            benchmark_identity=self.binding.identity,
            config_sha256="c" * 64,
        )
        catalog = BenchmarkCatalog(self.binding, access_controller=controller)
        with patch.object(BenchmarkDataset, "__getitem__", autospec=True) as getitem:
            with self.assertRaises(DemoAccessError):
                catalog.load(catalog.locate("test-small"), action="BROWSE")
            getitem.assert_not_called()

    def test_test_receipt_is_published_before_hdf5_payload_open(self) -> None:
        demo_root = self.root / "demo"
        controller = DemoTestAccessController(
            demo_root=demo_root,
            allow_test_demo=True,
            benchmark_identity=self.binding.identity,
            config_sha256="c" * 64,
        )
        catalog = BenchmarkCatalog(self.binding, access_controller=controller)
        original = BenchmarkDataset.__getitem__
        observed: list[bool] = []

        def guarded(dataset: BenchmarkDataset, index: int):
            observed.append(any((demo_root / "test_access_receipts").glob("dta_*")))
            return original(dataset, index)

        with patch.object(BenchmarkDataset, "__getitem__", new=guarded):
            loaded = catalog.load(catalog.locate("test-small"), action="BROWSE")
        self.assertTrue(observed and all(observed))
        self.assertIsNotNone(loaded.test_receipt)
        self.assertTrue((loaded.test_receipt.receipt_root / "receipt.json").is_file())

    def test_raw_preview_uses_one_payload_read_and_current_sample_auxiliary_values(self) -> None:
        counter = DatasetReadCounter()
        catalog = BenchmarkCatalog(
            self.binding,
            dataset_factory=counter.wrap(BenchmarkDataset),
        )
        loaded = catalog.load(
            catalog.locate("val-large"),
            action="BROWSE",
            model_normalization="none",
        )
        self.assertEqual(counter.calls, [("val", "val-large", "none")])
        self.assertEqual(
            [value.channel_name for value in loaded.optical_channel_previews],
            ["Red", "Green", "Blue"],
        )
        self.assertEqual(
            [value.modality for value in loaded.auxiliary_channel_previews],
            ["dem", "slope"],
        )
        self.assertEqual(
            loaded.auxiliary_channel_previews[0].raw_min,
            3000.0,
        )
        self.assertEqual(
            loaded.auxiliary_channel_previews[1].raw_min,
            3100.0,
        )
        for preview in loaded.auxiliary_channel_previews:
            self.assertIn("Spatial Expert Input Preview", preview.caption)
            self.assertIn(
                "Not formal MLLM grounded input in current P0",
                preview.caption,
            )
            self.assertFalse(preview.to_dict()["formal_mllm_grounded_input"])


class GalleryRevisionTest(unittest.TestCase):
    def test_revision_snapshots_update_and_logical_remove_preserve_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DemoGalleryStore(root / "demo")
            identity = {"manifest_sha256": "a" * 64, "index_sha256": "b" * 64}
            first = store.upsert(
                benchmark_identity=identity,
                sample_id="sample",
                split="val",
                source="source",
                demo_tags=("paper",),
                note="first",
                selected_tasks=(UnifiedTask.VLM_ONLY,),
            )
            first_entries = store.revisions_root / first / "entries.jsonl"
            first_sha = sha256_file(first_entries)
            second = store.upsert(
                benchmark_identity=identity,
                sample_id="sample",
                split="val",
                source="source",
                demo_tags=("defense",),
                note="updated",
                selected_tasks=(UnifiedTask.SEGMENT_ONLY,),
            )
            self.assertNotEqual(first, second)
            self.assertEqual(store.list_current()[0].note, "updated")
            third = store.remove(
                benchmark_identity=identity,
                split="val",
                sample_id="sample",
            )
            self.assertEqual(store.list_current(), ())
            self.assertEqual(len(store.history()), 3)
            self.assertEqual(sha256_file(first_entries), first_sha)
            self.assertTrue((store.revisions_root / third / "entries.jsonl").is_file())

    def test_test_gallery_requires_separate_gallery_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"manifest_sha256": "a" * 64, "index_sha256": "b" * 64}
            store = DemoGalleryStore(root / "demo")
            with self.assertRaises(DemoGalleryError):
                store.upsert(
                    benchmark_identity=identity,
                    sample_id="test-sample",
                    split="test",
                    source="source",
                    demo_tags=("demo",),
                    note="",
                    selected_tasks=(UnifiedTask.VLM_ONLY,),
                )
            controller = DemoTestAccessController(
                demo_root=root / "demo",
                allow_test_demo=True,
                benchmark_identity=identity,
                config_sha256="c" * 64,
            )
            receipt = controller.issue(sample_id="test-sample", action="GALLERY")
            forged = replace(receipt, receipt_root=root / "unbound-receipt")
            with self.assertRaises(DemoGalleryError):
                store.upsert(
                    benchmark_identity=identity,
                    sample_id="test-sample",
                    split="test",
                    source="source",
                    demo_tags=("demo",),
                    note="",
                    selected_tasks=(UnifiedTask.VLM_ONLY,),
                    test_receipt=forged,
                )
            store.upsert(
                benchmark_identity=identity,
                sample_id="test-sample",
                split="test",
                source="source",
                demo_tags=("demo",),
                note="",
                selected_tasks=(UnifiedTask.VLM_ONLY,),
                test_receipt=receipt,
            )
            self.assertEqual(store.list_current()[0].test_access_receipt_id, receipt.receipt_id)

if __name__ == "__main__":
    unittest.main()
