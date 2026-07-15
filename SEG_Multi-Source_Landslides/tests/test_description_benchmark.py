#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Description Benchmark M0/M1 协议测试。

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest SEG_Multi-Source_Landslides/tests/test_description_benchmark.py -v
写入行为：只在临时目录生成合成图片，不修改 benchmark、datasets 或 outputs。
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION_SCRIPTS = REPO_ROOT / "scripts/3-description"
sys.path.insert(0, str(DESCRIPTION_SCRIPTS))

from description_common import (  # noqa: E402
    bbox_pixel_half_open,
    caption_quality,
    mmrs_data_path,
    perceptual_rgb_mae,
)
from qpsalm_seg.data import build_single_image_modality_instance  # noqa: E402
from qpsalm_seg.description.vision_cache import (  # noqa: E402
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DescriptionVisionFeatureBank,
    description_cache_key,
)
from qpsalm_seg.description.output_protocol import parse_description_output  # noqa: E402


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DESCRIPTION_SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGION = load_script("qpsalm_description_region", "3-3_build_region_alignment_index.py")
SPLIT = load_script("qpsalm_description_split", "3-4_deduplicate_and_split.py")
MATERIALIZE = load_script("qpsalm_description_materialize", "3-5_materialize_description_images.py")


class DescriptionBenchmarkProtocolTest(unittest.TestCase):
    def test_schema_and_ontology_are_parseable(self) -> None:
        for name in ("qpsalm_description_record_v2.schema.json", "qpsalm_description_output_v1.schema.json"):
            payload = json.loads((REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
        ontology = yaml.safe_load((REPO_ROOT / "configs/description_ontology_v1.yaml").read_text(encoding="utf-8"))
        self.assertEqual(ontology["version"], "description_ontology_v1")
        self.assertIn("deformation_support", ontology["fields"])

    def test_description_cache_is_parent_level_and_task_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = description_cache_key("single_image", "parent_001")
            view_hash = "c" * 64
            fingerprint = hashlib.sha256("|".join([
                DESCRIPTION_CACHE_PROTOCOL, key, "a" * 64, "hash-smoke", "hash-smoke", view_hash,
            ]).encode()).hexdigest()
            record = {
                "lookup_key": key, "component": "single_image",
                "parent_sample_id": "parent_001", "source_ref": "benchmark/image.png",
                "source_content_hash": "a" * 64, "source_cache": None,
                "cache_fingerprint": fingerprint,
                "views": [{
                    "content_hash": view_hash,
                    "spatial_features": [
                        torch.zeros(1024, size, size, dtype=torch.float16)
                        for size in (16, 8, 6, 4)
                    ],
                    "view_tokens": torch.zeros(8, 2048, dtype=torch.float16),
                    "valid_mask": torch.ones(1, 16, 16, dtype=torch.float16),
                }],
            }
            torch_payload = {"format": DESCRIPTION_CACHE_FORMAT, "records": [record]}
            torch.save(torch_payload, root / "shard_00000.pt")
            manifest = {
                "format": DESCRIPTION_CACHE_FORMAT,
                "protocol": DESCRIPTION_CACHE_PROTOCOL,
                "builder_version": "synthetic",
                "model_revision": "hash-smoke",
                "processor_revision": "hash-smoke",
                "layers": [5, 11, 17, 23],
                "spatial_sizes": [16, 8, 6, 4],
                "view_tokens_per_view": 8,
                "spatial_channels": 1024,
                "token_dim": 2048,
                "backend": "hash-smoke",
                "input_fingerprints": {},
                "num_samples": 1,
                "components": ["single_image"],
                "lookup": {key: {
                    "shard": 0, "index": 0, "component": "single_image",
                    "parent_sample_id": "parent_001",
                }},
                "shards": ["shard_00000.pt"],
                "shard_size": 256,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded = DescriptionVisionFeatureBank(root).record("single_image", "parent_001")
            self.assertEqual(loaded["lookup_key"], key)
            self.assertNotIn("instruction", loaded)
            self.assertNotIn("region_geometry", loaded)

    def test_raw_schema_metric_is_not_replaced_by_deterministic_repair(self) -> None:
        invalid = parse_description_output('{"target_status":"present","summary":"partial"}')
        self.assertFalse(invalid.schema_valid)
        self.assertTrue(invalid.parse_errors)
        self.assertEqual(invalid.parsed["summary"], "partial")
        self.assertEqual(invalid.repaired["schema_version"], "qpsalm_description_output_v1")
        self.assertEqual(invalid.repaired["region"]["location"], "unavailable")

    def test_valid_structured_output_passes_without_metric_repair(self) -> None:
        payload = {
            "schema_version": "qpsalm_description_output_v1",
            "target_status": "absent",
            "region": {
                "location": "unavailable", "size_class": "unavailable",
                "shape": "unavailable", "elongation": "unavailable",
                "compactness": "unavailable", "fragmentation": "unavailable",
            },
            "evidence": {
                "surface_observation": "unavailable", "terrain_support": "unavailable",
                "sar_support": "unavailable", "deformation_support": "unavailable",
                "surrounding_context": "unavailable", "evidence_sufficiency": "unavailable",
            },
            "summary": "No target is present.",
        }
        parsed = parse_description_output(json.dumps(payload))
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.parse_errors, ())

    def test_single_rgb_image_builds_physical_unknown_modality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(np.full((7, 11, 3), 127, dtype=np.uint8)).save(path)
            instance = build_single_image_modality_instance({
                "type": "single_image", "path": str(path), "width": 11, "height": 7,
                "modality_instance": {"sensor": "generic_aerial_rgb", "quality": 0.8},
            })
            self.assertEqual(instance.family, "optical")
            self.assertEqual(instance.product_type, "rgb")
            self.assertEqual(tuple(instance.image.shape), (3, 7, 11))
            self.assertEqual(tuple(instance.valid_mask.shape), (1, 7, 11))
            self.assertTrue(bool((instance.valid_mask == 1).all()))
            self.assertIsNone(instance.native_gsd_m)
            self.assertIsNone(instance.aligned_gsd_m)

    def test_grayscale_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.png"
            Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "RGB/RGBA"):
                build_single_image_modality_instance({"type": "single_image", "path": str(path)})

    def test_mmrs_logical_path_and_bbox_conversion(self) -> None:
        resolved = mmrs_data_path("data/RSVG/DIOR_RSVG/images/11207.jpg")
        self.assertTrue(str(resolved).endswith("datasets/MMRS-1M/RSVG/DIOR_RSVG/images/11207.jpg"))
        self.assertEqual(bbox_pixel_half_open((0.1, 0.2, 0.3, 0.4), 100, 50), [10, 10, 30, 20])

    def test_dior_bidirectional_turns_form_one_region_pair(self) -> None:
        records = [
            (0, {"conversations": [
                {"from": "human", "value": "Please provide a short description for this region in this remote sensing image :[0.1,0.2,0.3,0.4]"},
                {"from": "gpt", "value": "The tiny vehicle"},
            ]}),
            (1, {"conversations": [
                {"from": "human", "value": "Please provide the horizontal bounding box coordinate of the region which is described as:The tiny vehicle in this remote sensing image"},
                {"from": "gpt", "value": "[0.1,0.2,0.3,0.4]"},
            ]}),
        ]
        regions, errors, warnings = REGION.parse_parent_records("synthetic.jpg", records)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["phrase"], "The tiny vehicle")

    def test_dior_zero_area_source_box_is_audited_and_excluded(self) -> None:
        records = [
            (0, {"conversations": [
                {"from": "human", "value": "Please provide a short description for this region in this remote sensing image :[0.1,0.2,0.1,0.4]"},
                {"from": "gpt", "value": "The tiny vehicle"},
            ]}),
            (1, {"conversations": [
                {"from": "human", "value": "Please provide the horizontal bounding box coordinate of the region which is described as:The tiny vehicle in this remote sensing image"},
                {"from": "gpt", "value": "[0.1,0.2,0.1,0.4]"},
            ]}),
        ]
        regions, errors, warnings = REGION.parse_parent_records("invalid.jpg", records)
        self.assertEqual(regions, [])
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all("excluded_invalid_source_bbox" in warning for warning in warnings))

    def test_exact_duplicate_inherits_test_priority(self) -> None:
        base = {
            "width": 10, "height": 10, "sha256": "a" * 64, "dhash64": "0" * 16,
            "source_scene_group": None, "source_scene_group_status": "unavailable",
            "task_count": 1, "stratum": {},
        }
        parents = [
            {**base, "parent_sample_id": "train_candidate", "source_dataset": "MMRS-RSICD", "source_split": None},
            {**base, "parent_sample_id": "held_out", "source_dataset": "RSIEval", "source_split": "test"},
        ]
        assignments = SPLIT.connected_assignments(parents, seed=42)
        self.assertEqual(assignments["train_candidate"]["split"], "test")
        self.assertEqual(assignments["held_out"]["split"], "test")

    def test_perceptual_rgb_mae_verifies_reencoded_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[:, :16] = [20, 120, 220]
            pixels[:, 16:] = [180, 60, 30]
            png = root / "source.png"
            jpeg = root / "reencoded.jpg"
            Image.fromarray(pixels).save(png)
            Image.fromarray(pixels).save(jpeg, quality=95)
            self.assertLessEqual(perceptual_rgb_mae(png, jpeg), 3.0)

    def test_verified_cluster_uses_held_out_canonical_parent(self) -> None:
        base = {
            "width": 64, "height": 64, "dhash64": "1" * 16,
            "source_scene_group": None, "source_scene_group_status": "unavailable",
            "task_count": 1, "stratum": {},
        }
        parents = [
            {
                **base, "parent_sample_id": "train_parent", "source_dataset": "MMRS-RSICD",
                "source_split": None, "source_image_path": "datasets/train.png", "sha256": "a" * 64,
            },
            {
                **base, "parent_sample_id": "test_parent", "source_dataset": "MMRS-RSITMD",
                "source_split": "test", "source_image_path": "datasets/test.jpg", "sha256": "b" * 64,
            },
        ]
        canonical, mapping, clusters = SPLIT.build_canonical_parents(
            parents, [("train_parent", "test_parent")]
        )
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["parent_sample_id"], "test_parent")
        self.assertEqual(canonical[0]["source_split"], "test")
        self.assertEqual(mapping, {"train_parent": "test_parent", "test_parent": "test_parent"})
        self.assertEqual(clusters[0]["merge_kind"], "verified_near_duplicate")

    def test_canonical_caption_merge_preserves_every_source_answer(self) -> None:
        parent = {
            "parent_sample_id": "canonical", "perceptual_cluster_id": "cluster_1",
        }
        rows = []
        for parent_id, texts in (
            ("canonical", ["A shared caption.", "A canonical detail."]),
            ("duplicate", ["A shared caption.", "A duplicate detail."]),
        ):
            rows.append({
                "component_benchmark": "rs_global_caption_v1",
                "parent_sample_id": parent_id,
                "sample_id": f"{parent_id}__global_caption",
                "source_dataset": f"source_{parent_id}",
                "answers": [{
                    "text": text, "quality": 1.0, "caption_quality_weight": 1.0,
                    "annotation_origin": "human", "language": "en",
                } for text in texts],
                "answer_type": "multi_reference_caption",
                "quality_flags": [],
                "provenance": {
                    "annotation_path": f"datasets/{parent_id}.json",
                    "source_image_path": f"datasets/{parent_id}.png",
                    "original_record_id": parent_id,
                },
            })
        merged = SPLIT.merge_caption_records(
            rows,
            [parent],
            {"canonical": "canonical", "duplicate": "canonical"},
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["answers"]), 3)
        provenance = [
            source
            for answer in merged[0]["answers"]
            for source in answer["source_provenance"]
        ]
        self.assertEqual(len(provenance), 4)
        self.assertEqual(
            {(value["source_sample_id"], value["source_answer_index"]) for value in provenance},
            {
                ("canonical__global_caption", 0), ("canonical__global_caption", 1),
                ("duplicate__global_caption", 0), ("duplicate__global_caption", 1),
            },
        )
        self.assertTrue(all(len(value["source_text_sha256"]) == 64 for value in provenance))

    def test_caption_quality_preserves_but_downweights_weak_claims(self) -> None:
        weight, flags = caption_quality("This image shows a sunny summer day.", "RSICap")
        self.assertEqual(weight, 0.5)
        self.assertIn("low_verifiability", flags)
        weight, flags = caption_quality("Three aircraft are parked beside a runway.", "RSICap")
        self.assertEqual(weight, 1.0)
        self.assertEqual(flags, [])

    def test_parent_image_is_copied_byte_exact_and_then_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.fromarray(np.full((7, 11, 3), 83, dtype=np.uint8)).save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            parent = {
                "parent_sample_id": "parent_001", "source_dataset": "Synthetic Source",
                "split": "train", "source_image_path": str(source), "sha256": digest,
                "width": 11, "height": 7,
            }
            first = MATERIALIZE.copy_parent_image(root / "benchmark", parent)
            target = Path(first["image_path"])
            self.assertEqual(first["materialization_status"], "copied")
            self.assertEqual(target.read_bytes(), source.read_bytes())
            target.write_bytes(b"corrupted")
            repaired = MATERIALIZE.copy_parent_image(root / "benchmark", parent)
            self.assertEqual(repaired["materialization_status"], "copied")
            self.assertEqual(target.read_bytes(), source.read_bytes())
            reused = MATERIALIZE.copy_parent_image(root / "benchmark", parent)
            self.assertEqual(reused["materialization_status"], "reused")
            self.assertEqual(reused["image_path"], first["image_path"])

    def test_missing_source_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = {
                "parent_sample_id": "parent_missing", "source_dataset": "Synthetic",
                "split": "test", "source_image_path": str(root / "missing.png"),
                "sha256": "0" * 64, "width": 5, "height": 5,
            }
            with self.assertRaisesRegex(FileNotFoundError, "源图片不存在"):
                MATERIALIZE.copy_parent_image(root / "benchmark", parent)

    def test_hash_mismatch_does_not_publish_or_leave_part_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(source)
            parent = {
                "parent_sample_id": "parent_bad_hash", "source_dataset": "Synthetic",
                "split": "dev", "source_image_path": str(source), "sha256": "0" * 64,
                "width": 5, "height": 5,
            }
            target = MATERIALIZE.destination_for_parent(root / "benchmark", parent)
            with self.assertRaisesRegex(ValueError, "hash"):
                MATERIALIZE.copy_parent_image(root / "benchmark", parent)
            self.assertFalse(target.exists())
            self.assertEqual(list((root / "benchmark").rglob("*.part")), [])

    def test_all_parent_task_views_share_materialized_path(self) -> None:
        source_ref = "datasets/source/image.png"
        rows = [
            {
                "sample_id": f"sample_{index}", "parent_sample_id": "parent_001",
                "split": "train", "source_dataset": "Synthetic",
                "visual_ref": {"path": source_ref},
                "provenance": {"source_image_path": source_ref},
            }
            for index in range(3)
        ]
        materialized = {
            "parent_001": {
                "image_path": "benchmark/qpsalm_description_v2_small/data/train/synthetic/parent_001.png",
                "source_image_path": source_ref,
            }
        }
        rewritten = MATERIALIZE.rewrite_final_records(rows, materialized)
        self.assertEqual({row["visual_ref"]["path"] for row in rewritten}, {materialized["parent_001"]["image_path"]})
        self.assertTrue(all(row["visual_ref"]["storage_mode"] == "materialized_copy" for row in rewritten))
        self.assertTrue(all(row["provenance"]["source_image_path"] == source_ref for row in rewritten))
        self.assertNotIn("storage_mode", rows[0]["visual_ref"])

    def test_train_eligible_filters_zero_weight_answers_without_mutating_audit_rows(self) -> None:
        rows = [{
            "sample_id": "caption_1", "split": "train",
            "answers": [
                {"text": "usable", "caption_quality_weight": 1.0},
                {"text": "?", "caption_quality_weight": 0.0},
            ],
        }]
        eligible = MATERIALIZE.training_eligible_rows(rows)
        self.assertEqual([answer["text"] for answer in eligible[0]["answers"]], ["usable"])
        self.assertEqual(len(rows[0]["answers"]), 2)

    def test_stale_and_part_files_are_removed_inside_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            expected = data_root / "train/source/expected.png"
            stale = data_root / "dev/source/stale.png"
            part = data_root / "train/source/.copy.part"
            outside = Path(directory) / "keep.txt"
            for path in (expected, stale, part, outside):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            count, _ = MATERIALIZE.remove_stale_files(data_root, {expected.resolve()})
            self.assertEqual(count, 2)
            self.assertTrue(expected.exists())
            self.assertTrue(outside.exists())
            self.assertFalse(stale.exists())
            self.assertFalse(part.exists())


if __name__ == "__main__":
    unittest.main()
