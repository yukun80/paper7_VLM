#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict benchmark-v2 Dataset -> train -> checkpoint -> eval integration test.

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest
SEG_Multi-Source_Landslides/tests/test_v2_integration.py -v
写入行为：仅使用系统临时目录，不读取或改写正式 benchmark。
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from qpsalm_seg.cli.cache_qwen_vision_features import main as cache_vision_main
from qpsalm_seg.cli.integration_check import (
    main as integration_check_main,
    select_representative_batch_indices,
)
from qpsalm_seg.config import QPSalmConfig, save_config
from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.engine.checkpoint import load_checkpoint
from qpsalm_seg.engine.common import build_model
from qpsalm_seg.engine.evaluator import evaluate
from qpsalm_seg.engine.trainer import train
from qpsalm_seg.models.vision_cache import QwenVisionFeatureBank


SCHEMA = "multisource_landslide_schema_v2"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def modality(path: Path, valid_path: Path, h: int, w: int) -> dict:
    return {
        "path": str(path), "format": "npy", "internal_key": None,
        "family": "optical", "sensor": "generic_rgb", "product_type": "rgb",
        "band_names": ["R", "G", "B"],
        "band_metadata": [
            {
                "name": name, "native_gsd_m": 0.5, "center_wavelength_nm": None,
                "bandwidth_nm": None, "polarization": None, "units": "reflectance",
                "signed": False, "measurement_geometry": None, "sign_convention": None,
            }
            for name in ("R", "G", "B")
        ],
        "shape": [3, h, w], "dtype": "uint8", "native_gsd_m": 0.5,
        "units": "reflectance", "signed": False, "orbit": "unknown", "quality": 1.0,
        "available": True, "normalization": {"method": "preserve_rgb_values", "scope": "none", "parameters": {}},
        "valid_mask": {
            "path": str(valid_path), "format": "npy", "shape": [1, h, w],
            "dtype": "uint8", "status": "materialized_before_normalization", "nodata_value": None,
        },
    }


def mask_entry(path: Path, array: np.ndarray) -> dict:
    ys, xs = np.where(array[0] > 0)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None
    return {
        "path": str(path), "format": "npy", "internal_key": None,
        "label_type": "binary_landslide", "shape": list(array.shape), "dtype": "uint8",
        "positive_pixels": int(array.sum()), "empty_mask": not bool(array.any()),
        "bbox_xyxy": bbox, "bbox_status": "derived" if bbox else "empty_mask", "binarize_rule": "mask > 0",
    }


class BenchmarkV2IntegrationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> QPSalmConfig:
        h, w = 24, 32
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        image_path = data_dir / "optical.npy"
        valid_path = data_dir / "optical_valid.npy"
        image = np.zeros((3, h, w), dtype=np.uint8)
        image[:, 4:20, 5:27] = 180
        valid = np.ones((1, h, w), dtype=np.uint8)
        valid[:, :2] = 0
        np.save(image_path, image)
        np.save(valid_path, valid)
        parent_mask = np.zeros((1, h, w), dtype=np.uint8)
        parent_mask[:, 5:12, 5:13] = 1
        parent_mask[:, 14:20, 20:28] = 1
        referring_mask = np.zeros_like(parent_mask)
        referring_mask[:, 5:12, 5:13] = 1
        referring_right_mask = np.zeros_like(parent_mask)
        referring_right_mask[:, 14:20, 20:28] = 1
        empty_mask = np.zeros_like(parent_mask)
        masks = {
            "global": parent_mask,
            "referring": referring_mask,
            "referring_right": referring_right_mask,
            "no_target": empty_mask,
        }
        mask_entries = {}
        for name, values in masks.items():
            path = data_dir / f"{name}_mask.npy"
            np.save(path, values)
            mask_entries[name] = mask_entry(path, values)

        rows_by_split = {}
        for split in ("train", "val"):
            parent = f"synthetic-parent-{split}"
            common = {
                "schema_version": SCHEMA, "dataset_name": "synthetic-v2", "split": split,
                "source_level": "patch", "modalities": {"optical_rgb": modality(image_path, valid_path, h, w)},
                "spatial": {"original_size": [h, w], "bucket_size": 64, "gsd_m": 0.5},
                "quality_flags": [], "supervision": "mask", "answer_format": "binary_mask",
                "parent_sample_id": parent,
            }
            rows = []
            task_specs = [
                ("global", "global_landslide_segmentation", "generic_landslide_v2", "Segment all landslide regions."),
                ("referring", "referring_landslide_segmentation", "referring_position_upper_left_v2", "Segment the landslide region in the upper-left part of the image."),
                ("referring_right", "referring_landslide_segmentation", "referring_position_lower_right_v2", "Segment the landslide region in the lower-right part of the image."),
                ("no_target", "no_target_segmentation", "no_target_position_upper_right_v2", "Segment landslides in the upper-right part; output an empty mask if absent."),
            ]
            for name, family, template, text in task_specs:
                row = {
                    **common,
                    "sample_id": f"{parent}-{name}", "mask": mask_entries[name],
                    "template_id": template, "task_family": family,
                    "instruction": {
                        "language": "en", "template_id": template, "task_family": family,
                        "text": text, "text_zh": text, "answer_format": "binary_mask",
                    },
                }
                if name != "global":
                    category = "no_target" if name == "no_target" else "position"
                    row["referring_target"] = {
                        "category": category,
                        "subtype": (
                            "upper-right" if name == "no_target"
                            else "lower-right" if name == "referring_right"
                            else "upper-left"
                        ),
                        "target_mask": mask_entries[name], "grounding": {}, "confidence": "synthetic",
                    }
                rows.append(row)
            rows_by_split[split] = rows
            write_jsonl(root / "indexes" / f"instruction_{split}.jsonl", rows)
        write_jsonl(root / "indexes" / "instruction_test.jsonl", rows_by_split["val"])
        return replace(
            QPSalmConfig(),
            benchmark_dir=str(root), output_dir=str(root / "run"),
            controller="text_probe", preset="raw_sane_qmef_pmrd",
            target_size=32, use_size_buckets=False, max_native_size=32,
            decoder_dim=32, num_heads=4, num_mask_tokens=4, num_decoder_layers=1,
            qwen_view_tokens_per_view=2,
            batch_size=1, grad_accum_steps=1, num_workers=0,
            max_steps=2, val_interval=2, save_interval=2, visualize_interval=2,
            max_train_samples=None, max_val_samples=None, max_val_batches=0,
            num_visualizations=0, log_interval=2, modality_dropout=0.5,
            missing_modality_consistency_weight=0.0,
        )

    def test_representative_gate_selects_one_sampler_shaped_batch(self) -> None:
        def row(
            sample_id: str,
            parent: str,
            bucket: int,
            load: int,
            task: str = "global_landslide_segmentation",
        ) -> dict:
            return {
                "sample_id": sample_id,
                "parent_sample_id": parent,
                "bucket": bucket,
                "load": load,
                "task_family": task,
                "mask": {"empty_mask": False},
                "modalities": {
                    f"modality_{index}": {"family": family, "available": True}
                    for index, family in enumerate(("multispectral", "terrain", "sar"))
                },
            }

        rows = [row(f"high-{index}", f"parent-{index}", 256, 320) for index in range(8)]
        rows += [row(f"low-{index}", f"low-parent-{index}", 128, 192) for index in range(8)]
        dataset = type("Rows", (), {
            "rows": rows,
            "bucket_size": lambda self, index: self.rows[index]["bucket"],
            "sequence_load_bucket": lambda self, index: self.rows[index]["load"],
        })()
        selected = select_representative_batch_indices(dataset, 6)
        self.assertEqual(selected, list(range(6)))

    def test_strict_v2_train_reload_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            result = train(config, device_name="cpu")
            run_dir = Path(result["output_dir"])
            self.assertTrue((run_dir / "checkpoint_last.pt").exists())
            self.assertTrue((run_dir / "validation_latest.json").exists())
            self.assertTrue((run_dir / "train_history.jsonl").exists())
            self.assertTrue((run_dir / "monitor_val_manifest.json").exists())
            dataset = MultiSourceLandslideDataset(config, "val")
            batch = qpsalm_collate([dataset[0]])
            self.assertEqual(batch.metadata[0]["valid_coverage"], 22 / 32)
            self.assertNotIn("synthetic-v2", batch.proposal_context_text[0].lower())
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=qpsalm_collate)
            model = build_model(config, torch.device("cpu"))
            self.assertEqual(load_checkpoint(run_dir / "checkpoint_last.pt", model), 2)
            report = evaluate(model, loader, torch.device("cpu"), threshold=0.5)
            self.assertEqual(int(report["coverage"]["num_samples"]), 4)
            self.assertEqual(int(report["instruction_sensitivity"]["num_no_target"]), 1)
            self.assertEqual(int(report["instruction_sensitivity"]["num_paired_parents"]), 1)
            self.assertEqual(float(report["instruction_sensitivity"]["mean_paired_target_iou_16"]), 0.0)
            proposal = report["proposal_diagnostics"]["records"][0]
            self.assertIn("proposal_union_dice", proposal)
            self.assertIn("coverage_mode", proposal)
            self.assertIn("selected_mask_area", proposal)
            self.assertIn("oracle_matched_query", proposal)
            self.assertIn("oracle_relevance_logit", proposal)
            self.assertIn("oracle_mask_area", proposal)
            self.assertTrue(torch.isfinite(torch.tensor(report["loss"])))

    def test_hash_vision_cache_v3_roundtrip_and_pretrained_sane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            config_path = root / "config.yaml"
            cache_dir = root / "vision_cache"
            save_config(config_path, config)
            argv = [
                "cache_qwen_vision_features", "--config", str(config_path),
                "--output-dir", str(cache_dir),
                "--backend", "hash-smoke", "--render-size", "32",
                "--spatial-sizes", "2,2,2,2", "--view-tokens", "2",
                "--shard-size", "1", "--max-samples", "2",
            ]
            original_getitem = MultiSourceLandslideDataset.__getitem__
            loaded_indices = []

            def counted_getitem(dataset, index):
                loaded_indices.append((dataset.split, index))
                return original_getitem(dataset, index)

            with patch.object(MultiSourceLandslideDataset, "__getitem__", counted_getitem):
                with patch("sys.argv", argv):
                    cache_vision_main()
            manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_samples"], 2)
            self.assertEqual(len(manifest["shards"]), 2)
            self.assertEqual(manifest["shard_size"], 1)
            self.assertEqual(manifest["peak_buffer_records"], 1)
            self.assertEqual(
                set(manifest["input_protocol"]["index_fingerprints"]),
                {"train", "val", "test"},
            )
            self.assertEqual(len(loaded_indices), 2)
            self.assertFalse(list(cache_dir.glob(".*.tmp")))
            with patch("sys.argv", [
                "cache_qwen_vision_features", "--config", str(config_path),
                "--output-dir", str(cache_dir), "--verify-only",
            ]):
                cache_vision_main()
            bank = QwenVisionFeatureBank(cache_dir, decoder_dim=32)
            dataset = MultiSourceLandslideDataset(config, "val", max_samples=1)
            item = dataset[0]
            _, _, counts, family_ids, segments = bank.tokens_for(
                [item["visual_evidence_key"]], [item["active_subset"]], torch.device("cpu"), 2
            )
            self.assertEqual(counts, [2])
            self.assertEqual(family_ids[0, :2].tolist(), [1, 1])
            self.assertTrue(segments[0])
            pretrained = replace(config, use_pretrained_sane=True, vision_feature_cache=str(cache_dir))
            model = build_model(pretrained, torch.device("cpu"))
            output = model(qpsalm_collate([item]))
            self.assertTrue(torch.isfinite(output["loss"]))
            val_index = root / "indexes" / "instruction_val.jsonl"
            val_index.write_text(val_index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input protocol"):
                with patch("sys.argv", [
                    "cache_qwen_vision_features", "--config", str(config_path),
                    "--output-dir", str(cache_dir), "--verify-only",
                ]):
                    cache_vision_main()

    def test_real_integration_cli_contract_runs_raw_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            config_path = root / "config.yaml"
            report_path = root / "integration_report.json"
            save_config(config_path, config)
            with patch("sys.argv", [
                "integration_check",
                "--config", str(config_path),
                "--mode", "raw",
                "--device", "cpu",
                "--output", str(report_path),
            ]):
                integration_check_main()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["format"], "qpsalm_real_integration_v2")
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(set(report["checks"]["raw"]["selected_indices"]), {
                "global", "referring", "no_target",
            })
            self.assertGreater(
                report["checks"]["raw"]["gradients"]["gradient_norm_sum"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
