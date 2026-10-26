from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from torch.utils.data import DataLoader

from tests.data.rs_general.fixture_helpers import (
    make_all_sources,
    write_build_config,
    write_export_config,
)
from oa_groundrag.data.rs_general.assets import AssetStore
from oa_groundrag.data.rs_general.builder import audit_sources, build_benchmark
from oa_groundrag.data.rs_general.io import read_json, read_jsonl
from oa_groundrag.data.rs_general.config import load_build_config, load_export_config
from oa_groundrag.data.rs_general.dataset import (
    CanonicalRecordLocation,
    RSGeneralDescDataset,
    ParentBalancedSampler,
    collate_canonical_samples,
)
from oa_groundrag.data.rs_general.contracts import validate_canonical_record
from oa_groundrag.data.rs_general.errors import (
    BuildError,
    ConfigError,
    ExportError,
    ReasonCode,
    SchemaError,
)
from oa_groundrag.data.rs_general.exporter import export_qwen, render_canonical_messages
from oa_groundrag.data.rs_general.validator import validate_benchmark


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.roots = make_all_sources(cls.base / "sources")
        cls.config_path = write_build_config(
            cls.base / "build.yaml",
            cls.roots,
            cls.base / "benchmark",
        )
        cls.config = load_build_config(cls.config_path)
        cls.benchmark = build_benchmark(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_synthetic_build_validate_provenance_statistics_and_hash(self) -> None:
        shallow = validate_benchmark(self.benchmark, deep=False)
        deep = validate_benchmark(self.benchmark, deep=True)
        self.assertEqual(shallow["errors"], [])
        self.assertEqual(deep["errors"], [])
        manifest = read_json(self.benchmark / "manifest.json")
        statistics = read_json(
            self.benchmark / manifest["layout"]["statistics"]
        )
        provenance = read_jsonl(
            self.benchmark / manifest["layout"]["provenance"]
        )
        self.assertEqual(statistics["record_count"], deep["record_count"])
        self.assertLess(statistics["parent_count"], statistics["record_count"])
        self.assertGreater(len(provenance), 0)
        self.assertFalse(manifest["formal_acceptance_eligible"])
        self.assertIn("bounded_smoke_profile", manifest["formal_acceptance_blockers"])
        self.assertEqual(
            manifest["benchmark_scope"],
            "rs_generaldesc_external_train_val",
        )

    def test_canonical_nested_unknown_and_identity_tamper_are_rejected(self) -> None:
        dataset = RSGeneralDescDataset(self.benchmark, load_assets=False)
        nested_unknown = copy.deepcopy(dataset.records[0])
        nested_unknown["media"][0]["unknown"] = True
        with self.assertRaises(SchemaError) as caught:
            validate_canonical_record(nested_unknown)
        self.assertEqual(caught.exception.code, ReasonCode.UNKNOWN_FIELD)

        identity_tamper = copy.deepcopy(dataset.records[0])
        identity_tamper["instruction"] += " changed"
        with self.assertRaises(SchemaError) as caught:
            validate_canonical_record(identity_tamper)
        self.assertEqual(caught.exception.code, ReasonCode.HASH_MISMATCH)

        task_tamper = copy.deepcopy(
            next(
                record
                for record in dataset.records
                if record["task_family"] == "global_caption"
            )
        )
        task_tamper["supervision_kind"] = "numeric_qa"
        with self.assertRaises(SchemaError) as caught:
            validate_canonical_record(task_tamper)
        self.assertEqual(caught.exception.code, ReasonCode.SCHEMA_MISMATCH)

    def test_canonical_dataset_sampler_and_qwen_are_decoupled(self) -> None:
        dataset = RSGeneralDescDataset(self.benchmark)
        self.assertGreater(len(dataset), 0)
        first = dataset[0]
        self.assertNotIn("messages", first["record"])
        sampler = ParentBalancedSampler(dataset, seed=17)
        order_a = list(iter(sampler))
        state = sampler.state_dict()
        other = ParentBalancedSampler(dataset, seed=17)
        other.load_state_dict(state)
        self.assertEqual(order_a, list(iter(other)))
        incompatible = ParentBalancedSampler(
            dataset,
            seed=17,
            source_weights={"rsgpt": 2.0},
        )
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(state)
        loader = DataLoader(
            dataset,
            batch_size=3,
            sampler=sampler,
            collate_fn=collate_canonical_samples,
            num_workers=0,
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch["record_id"]), 3)
        export_config = load_export_config(
            write_export_config(
                self.base / "export.yaml",
                self.benchmark,
                self.base / "qwen",
            )
        )
        target = export_qwen(export_config)
        rows = read_jsonl(target / "qwen_messages.jsonl")
        self.assertGreater(len(rows), 0)
        self.assertIn("messages", rows[0])
        self.assertNotIn("messages", dataset.records[0])

    def test_dataset_rejects_non_native_roles_tasks_and_sources(self) -> None:
        for kwargs in (
            {"roles": ["unsupported_role"]},
            {"task_families": ["unsupported_task"]},
            {"sources": ["unsupported_source"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(SchemaError):
                RSGeneralDescDataset(
                    self.benchmark,
                    load_assets=False,
                    **kwargs,
                )

    def test_bounded_locations_and_public_renderer_preserve_contract(self) -> None:
        manifest = read_json(self.benchmark / "manifest.json")
        first_shard = manifest["layout"]["record_shards"][0]["path"]
        bounded = RSGeneralDescDataset.from_locations(
            self.benchmark,
            (
                CanonicalRecordLocation(first_shard, 0),
                CanonicalRecordLocation(first_shard, 1),
            ),
            load_assets=False,
            seed=19,
        )
        full = RSGeneralDescDataset(
            self.benchmark,
            load_assets=False,
            seed=19,
        )
        self.assertEqual(
            [row["record_id"] for row in bounded.records],
            [row["record_id"] for row in full.records[:2]],
        )
        export_config = load_export_config(
            write_export_config(
                self.base / "bounded-render-export.yaml",
                self.benchmark,
                self.base / "bounded-render-output",
            )
        )
        public, public_assets = render_canonical_messages(
            bounded.records[0],
            purpose=export_config.purpose,
            seed=export_config.seed,
            dataset=bounded,
            derived_root=self.base / "bounded-public-render",
        )
        repeated, repeated_assets = render_canonical_messages(
            bounded.records[0],
            purpose=export_config.purpose,
            seed=export_config.seed,
            dataset=bounded,
            derived_root=self.base / "bounded-repeated-render",
        )
        self.assertEqual(public, repeated)
        self.assertEqual(public_assets, repeated_assets)
        with self.assertRaises(SchemaError):
            RSGeneralDescDataset.from_locations(
                self.benchmark,
                (CanonicalRecordLocation("records/not-in-manifest.jsonl", 0),),
                load_assets=False,
            )

    def test_multireference_epoch_rotation_and_validation_contract(self) -> None:
        dataset = RSGeneralDescDataset(
            self.benchmark,
            load_assets=False,
            seed=31,
        )
        candidate_indices = [
            index
            for index, record in enumerate(dataset.records)
            if record["source"] == "mmrs1m"
            and len(record["training_responses"]) > 1
            and record["logical_role"] == "external_train"
        ]
        self.assertTrue(candidate_indices)
        index = candidate_indices[0]
        selected: set[str] = set()
        for epoch in range(32):
            dataset.set_epoch(epoch)
            item = dataset[index]
            self.assertEqual(
                item["reference_responses"],
                tuple(item["record"]["reference_responses"]),
            )
            self.assertIsNotNone(item["training_response"])
            selected.add(str(item["training_response"]))
        self.assertEqual(
            selected,
            set(dataset.records[index]["training_responses"]),
        )
        validation_index = next(
            index
            for index, record in enumerate(dataset.records)
            if record["logical_role"] == "external_val"
        )
        validation_item = dataset[validation_index]
        self.assertIsNone(validation_item["training_response"])
        self.assertEqual(
            validation_item["reference_responses"],
            tuple(validation_item["record"]["reference_responses"]),
        )

    def test_task_first_then_parent_balanced_sampler(self) -> None:
        dataset = RSGeneralDescDataset(
            self.benchmark,
            roles=["external_train"],
            load_assets=False,
        )
        task_count = len({record["task_family"] for record in dataset.records})
        sampler = ParentBalancedSampler(
            dataset,
            seed=41,
            num_samples=task_count * 20,
        )
        counts = Counter(
            dataset.records[index]["task_family"] for index in sampler
        )
        self.assertEqual(set(counts.values()), {20})
        sampler.set_epoch(3)
        order_a = list(sampler)
        sampler.set_epoch(3)
        self.assertEqual(order_a, list(sampler))
        self.assertEqual(dataset.epoch, 3)
        weighted_task = sorted(counts)[0]
        weighted = ParentBalancedSampler(
            dataset,
            seed=41,
            num_samples=(task_count + 1) * 20,
            task_weights={weighted_task: 2.0},
        )
        weighted_counts = Counter(
            dataset.records[index]["task_family"] for index in weighted
        )
        self.assertEqual(weighted_counts[weighted_task], 40)
        self.assertEqual(
            {
                count
                for task, count in weighted_counts.items()
                if task != weighted_task
            },
            {20},
        )

    def test_task_aware_qwen_renderers_and_fail_fast(self) -> None:
        export_config = load_export_config(
            write_export_config(
                self.base / "task-aware-export.yaml",
                self.benchmark,
                self.base / "task-aware-export",
            )
        )
        target = export_qwen(export_config)
        rows = read_jsonl(target / "qwen_messages.jsonl")
        required_layouts = {
            "single_image",
            "bbox_region",
            "boxed_image",
            "pre_post",
        }
        direct_dataset = RSGeneralDescDataset(
            self.benchmark,
            load_assets=False,
        )
        direct_stage = self.base / "direct-task-render"
        for layout in sorted(
            required_layouts - {row["input_layout"] for row in rows}
        ):
            record = next(
                row
                for row in direct_dataset.records
                if row["input_layout"] == layout
            )
            rendered, _ = render_canonical_messages(
                record,
                purpose=export_config.purpose,
                seed=export_config.seed,
                dataset=direct_dataset,
                derived_root=direct_stage,
            )
            rows.append(rendered)
        actual_layouts = {row["input_layout"] for row in rows}
        self.assertTrue(required_layouts <= actual_layouts, actual_layouts)
        by_layout = {
            layout: next(row for row in rows if row["input_layout"] == layout)
            for layout in required_layouts
        }
        bbox_content = by_layout["bbox_region"]["messages"][0]["content"]
        derived = [
            item["export_asset"]
            for item in bbox_content
            if item.get("type") == "image" and "export_asset" in item
        ]
        self.assertEqual(len(derived), 2)
        self.assertTrue(
            all(
                (target / path).is_file()
                or (direct_stage / path).is_file()
                for path in derived
            )
        )
        self.assertIn(
            "Canonical normalized xyxy",
            bbox_content[-1]["text"],
        )
        boxed_text = by_layout["boxed_image"]["messages"][0]["content"][-1]["text"]
        self.assertIn("red box", boxed_text)
        self.assertIn("blue box", boxed_text)
        pre_post = by_layout["pre_post"]["messages"][0]["content"]
        self.assertEqual(
            [item.get("asset_role") for item in pre_post if item["type"] == "image"],
            ["pre_image", "post_image"],
        )
        manifest = read_json(target / "manifest.json")
        direct_derived_count = sum(
            1 for path in direct_stage.rglob("*.png")
        )
        self.assertGreaterEqual(
            manifest["derived_asset_count"] + direct_derived_count,
            2,
        )

        dataset = RSGeneralDescDataset(self.benchmark, load_assets=False)
        record = copy.deepcopy(dataset.records[0])
        record["output_modality"] = "pixel_mask"
        with self.assertRaises(ExportError):
            render_canonical_messages(
                record,
                purpose=export_config.purpose,
                seed=export_config.seed,
                dataset=dataset,
                derived_root=self.base / "unused-render-stage",
            )
        bbox_record = copy.deepcopy(
            next(
                row
                for row in dataset.records
                if row["input_layout"] == "bbox_region"
            )
        )
        bbox_record["target"] = {"type": "none"}
        with self.assertRaises(ExportError):
            render_canonical_messages(
                bbox_record,
                purpose=export_config.purpose,
                seed=export_config.seed,
                dataset=dataset,
                derived_root=self.base / "unused-render-stage",
            )

    def test_source_hidden_load_uses_only_benchmark_root(self) -> None:
        dataset = RSGeneralDescDataset(self.benchmark, load_assets=True)
        for index in range(len(dataset)):
            dataset[index]
        report = validate_benchmark(self.benchmark, deep=True)
        self.assertEqual(report["errors"], [])
        manifest_text = (self.benchmark / "manifest.json").read_text()
        for root in self.roots.values():
            self.assertNotIn(str(root), manifest_text)

    def test_output_exists_is_rejected(self) -> None:
        with self.assertRaises(BuildError) as caught:
            build_benchmark(self.config)
        self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_EXISTS)
    def test_external_full_exports_train_val_separately(
        self,
    ) -> None:
        target = self.base / "external-full"
        config = load_build_config(
            write_build_config(
                self.base / "external-full.yaml",
                self.roots,
                target,
                profile="full",
                max_assets=None,
            )
        )
        build_benchmark(config)
        report = validate_benchmark(target, deep=True)
        self.assertEqual(report["errors"], [])
        manifest = read_json(target / "manifest.json")
        self.assertEqual(
            manifest["benchmark_scope"],
            "rs_generaldesc_external_train_val",
        )
        self.assertTrue(manifest["scope_validation_complete"])
        self.assertTrue(manifest["formal_acceptance_eligible"])
        self.assertEqual(
            manifest["formal_acceptance_blockers"],
            [],
        )
        dataset = RSGeneralDescDataset(target, load_assets=False)
        self.assertEqual(
            {row["source"] for row in dataset.records},
            {"rsgpt", "mmrs1m", "disasterm3"},
        )
        self.assertEqual(
            {row["logical_role"] for row in dataset.records},
            {"external_train", "external_val"},
        )
        for source in ("rsgpt", "mmrs1m", "disasterm3"):
            self.assertEqual(
                {
                    row["logical_role"]
                    for row in dataset.records
                    if row["source"] == source
                },
                {"external_train", "external_val"},
            )

        train_target = export_qwen(
            load_export_config(
                write_export_config(
                    self.base / "external-train-export.yaml",
                    target,
                    self.base / "external-train-export",
                    roles=["external_train"],
                )
            )
        )
        val_target = export_qwen(
            load_export_config(
                write_export_config(
                    self.base / "external-val-export.yaml",
                    target,
                    self.base / "external-val-export",
                    purpose="validation",
                    roles=["external_val"],
                )
            )
        )
        self.assertEqual(
            {
                row["logical_role"]
                for row in read_jsonl(train_target / "qwen_messages.jsonl")
            },
            {"external_train"},
        )
        self.assertEqual(
            {
                row["logical_role"]
                for row in read_jsonl(val_target / "qwen_messages.jsonl")
            },
            {"external_val"},
        )
        self.assertTrue(
            all(
                row["reference_responses"]
                for row in read_jsonl(val_target / "qwen_messages.jsonl")
            )
        )
        self.assertFalse(manifest["official_upstream_evaluation_reproducible"])

    def test_staging_failure_does_not_publish(self) -> None:
        target = self.base / "failed-output"
        config = load_build_config(
            write_build_config(
                self.base / "failed.yaml",
                self.roots,
                target,
            )
        )
        with mock.patch.object(
            AssetStore,
            "add_image",
            side_effect=RuntimeError("injected failure"),
        ):
            with self.assertRaises(RuntimeError):
                build_benchmark(config)
        self.assertFalse(target.exists())
        self.assertEqual(
            list(target.parent.glob(f".{target.name}.staging-*")),
            [],
        )

    def test_tamper_and_hardlink_are_detected(self) -> None:
        copied = self.base / "tampered"
        shutil.copytree(self.benchmark, copied)
        manifest = read_json(copied / "manifest.json")
        shard = copied / manifest["layout"]["record_shards"][0]["path"]
        with shard.open("ab") as handle:
            handle.write(b" ")
        report = validate_benchmark(copied, deep=False)
        self.assertTrue(any("hash" in error for error in report["errors"]))

        linked = self.base / "linked"
        shutil.copytree(self.benchmark, linked)
        manifest = read_json(linked / "manifest.json")
        asset = read_json(linked / manifest["layout"]["assets"])[0]["path"]
        asset_path = linked / asset
        hardlink = asset_path.with_name(f"hardlink-{asset_path.name}")
        hardlink.hardlink_to(asset_path)
        report = validate_benchmark(linked, deep=False)
        self.assertTrue(any("单链接" in error for error in report["errors"]))

    def test_deterministic_rebuild_and_worker_independent_records(self) -> None:
        target_a = self.base / "det-a"
        target_b = self.base / "det-b"
        config_a = load_build_config(
            write_build_config(self.base / "det-a.yaml", self.roots, target_a)
        )
        config_b = load_build_config(
            write_build_config(
                self.base / "det-b.yaml",
                self.roots,
                target_b,
                workers=2,
            )
        )
        build_benchmark(config_a)
        build_benchmark(config_b)
        manifest_a = read_json(target_a / "manifest.json")
        manifest_b = read_json(target_b / "manifest.json")
        rows_a = [
            row
            for shard in manifest_a["layout"]["record_shards"]
            for row in read_jsonl(target_a / shard["path"])
        ]
        rows_b = [
            row
            for shard in manifest_b["layout"]["record_shards"]
            for row in read_jsonl(target_b / shard["path"])
        ]
        self.assertEqual(rows_a, rows_b)
        assets_a = read_json(target_a / manifest_a["layout"]["assets"])
        assets_b = read_json(target_b / manifest_b["layout"]["assets"])
        self.assertEqual(assets_a, assets_b)

    def test_external_split_seed_changes_assignment(self) -> None:
        base_config = load_build_config(
            write_build_config(
                self.base / "seed-7.yaml",
                self.roots,
                self.base / "unused-seed-output",
                seed=7,
            )
        )
        base_assignment = audit_sources(
            base_config,
            deep=False,
        )["external_split"]["assignment_sha256"]
        alternatives: set[str] = set()
        for seed in range(8, 24):
            config = load_build_config(
                write_build_config(
                    self.base / f"seed-{seed}.yaml",
                    self.roots,
                    self.base / "unused-seed-output",
                    seed=seed,
                )
            )
            alternatives.add(
                audit_sources(
                    config,
                    deep=False,
                )["external_split"]["assignment_sha256"]
            )
        self.assertTrue(any(value != base_assignment for value in alternatives))

    def test_duplicate_image_component_cannot_cross_external_roles(self) -> None:
        duplicate_base = self.base / "duplicate-sources"
        roots = make_all_sources(duplicate_base)
        shutil.copyfile(
            roots["rsgpt"] / "RSICap/images/cap.png",
            roots["mmrs1m"] / "caption/demo/images/a.png",
        )
        target = self.base / "duplicate-benchmark"
        config = load_build_config(
            write_build_config(
                self.base / "duplicate-build.yaml",
                roots,
                target,
            )
        )
        build_benchmark(config)
        report = validate_benchmark(target, deep=True)
        self.assertEqual(report["errors"], [])
        manifest = read_json(target / "manifest.json")
        statistics = read_json(target / manifest["layout"]["statistics"])
        self.assertGreaterEqual(
            statistics["external_split"][
                "duplicate_content_component_count"
            ],
            1,
        )
        dataset = RSGeneralDescDataset(target, load_assets=False)
        image_roles: dict[str, set[str]] = {}
        image_parents: dict[str, set[str]] = {}
        for record in dataset.records:
            for media in record["media"]:
                if media["media_type"] != "image":
                    continue
                image_roles.setdefault(media["sha256"], set()).add(
                    record["logical_role"]
                )
                image_parents.setdefault(media["sha256"], set()).add(
                    record["parent_id"]
                )
        duplicate_hashes = [
            digest
            for digest, parents in image_parents.items()
            if len(parents) > 1
        ]
        self.assertTrue(duplicate_hashes)
        self.assertTrue(
            all(len(image_roles[digest]) == 1 for digest in duplicate_hashes)
        )

    def test_qwen_scope_rejects_unknown_and_role_mixing(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load_export_config(
                write_export_config(
                    self.base / "sealed.yaml",
                    self.benchmark,
                    self.base / "sealed-export",
                    purpose="training",
                    roles=["unsupported_role"],
                )
            )
        self.assertEqual(caught.exception.code, ReasonCode.INVALID_ENUM)

        external_val = load_export_config(
            write_export_config(
                self.base / "external-val-training.yaml",
                self.benchmark,
                self.base / "external-val-training",
                purpose="training",
                roles=["external_val"],
            )
        )
        with self.assertRaises(ExportError):
            export_qwen(external_val)

        with self.assertRaises(ConfigError):
            load_export_config(
                write_export_config(
                    self.base / "mixed-validation.yaml",
                    self.benchmark,
                    self.base / "mixed-validation",
                    purpose="validation",
                    roles=["external_val", "unsupported_role"],
                )
            )
