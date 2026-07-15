#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmentation-grounded description M5-M7 协议测试。

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest
SEG_Multi-Source_Landslides/tests/test_segdesc_protocol.py -v
写入行为：仅使用合成张量和临时目录，不加载 Qwen、benchmark 或 checkpoint。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from qpsalm_seg.controllers import QwenMaskQueryController
from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.description.counterfactuals import counterfactual_region_masks
from qpsalm_seg.description.data import end_to_end_region_support
from qpsalm_seg.description.metrics import DescriptionMetricAccumulator, structured_disagreement
from qpsalm_seg.description.predicted_regions import export_predicted_regions
from qpsalm_seg.description.expert_factuality import aggregate_expert_factuality
from qpsalm_seg.description.evaluator import EndToEndTargetResolver, _same_image_retrieval
from qpsalm_seg.description.checkpoint import (
    initialize_segdesc_checkpoint,
    load_segdesc_checkpoint,
    save_segdesc_checkpoint,
)
from qpsalm_seg.description.oof import build_oof_fold_indexes, load_oof_manifest


def valid_target(status: str = "absent") -> dict:
    return {
        "schema_version": "qpsalm_description_output_v1",
        "target_status": status,
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


class FakePeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ParameterDict({
            "default": nn.Parameter(torch.ones(2, 2)),
            "desc_adapter": nn.Parameter(torch.ones(2, 2)),
        })
        self.active_adapters = ("default",)
        self.peft_config = {"default": object(), "desc_adapter": object()}

    def set_adapter(self, name: str) -> None:
        self.active_adapters = (name,)
        for key, parameter in self.lora_A.items():
            parameter.requires_grad_(key == name)


class AdapterScopeHarness:
    adapter_scope = QwenMaskQueryController.adapter_scope

    def __init__(self) -> None:
        self.model = FakePeftModel()

    def ensure_named_adapter(self, _name: str) -> None:
        return None


class FakeSegDescCheckpointModel(nn.Module):
    def __init__(self, region_encoder: str) -> None:
        super().__init__()
        self.region_encoder_name = region_encoder
        self.shared = nn.Linear(4, 4)
        self.mgrr = nn.Linear(4, 4) if region_encoder == "mgrr" else nn.Sequential(
            nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4)
        )
        self.segmentation = SimpleNamespace(config=SimpleNamespace(decoder_dim=4))
        self.controller = SimpleNamespace(
            model=SimpleNamespace(peft_config={"default": object(), "desc_adapter": object()})
        )


class SegDescProtocolTest(unittest.TestCase):
    def test_adapter_scope_restores_optimizer_trainability(self) -> None:
        controller = AdapterScopeHarness()
        controller.model.lora_A["default"].requires_grad_(False)
        controller.model.lora_A["desc_adapter"].requires_grad_(True)
        with controller.adapter_scope("desc_adapter"):
            self.assertEqual(controller.model.active_adapters, ("desc_adapter",))
            self.assertTrue(controller.model.lora_A["desc_adapter"].requires_grad)
        self.assertEqual(controller.model.active_adapters, ("default",))
        self.assertFalse(controller.model.lora_A["default"].requires_grad)
        self.assertTrue(controller.model.lora_A["desc_adapter"].requires_grad)

    def test_invalid_raw_json_scores_zero_even_when_repair_is_valid(self) -> None:
        target = json.dumps(valid_target())
        metric = DescriptionMetricAccumulator()
        row = metric.update(
            prediction='{"target_status":"absent","summary":"partial"}',
            target_text=target,
            references=[target],
            structured=True,
            metadata={"sample_id": "synthetic", "task_family": "no_target_response"},
        )
        report = metric.compute()
        self.assertFalse(row["raw_schema_valid"])
        self.assertEqual(report["raw_schema_valid_rate"], 0.0)
        self.assertEqual(report["repair_schema_valid_rate"], 1.0)
        self.assertLess(report["structured_field_macro_f1"], 1.0)

    def test_region_counterfactuals_preserve_canvas(self) -> None:
        mask = torch.zeros(2, 1, 8, 12)
        mask[0, :, 2:5, 3:7] = 1
        mask[1, :, 1:4, 8:10] = 1
        for mode in ("full_mask", "zero_mask", "shuffled_mask", "region_swap"):
            changed = counterfactual_region_masks(mask, mode)
            self.assertEqual(changed.shape, mask.shape)
            self.assertTrue(bool(torch.isfinite(changed).all()))
        self.assertEqual(counterfactual_region_masks(mask, "shuffled_mask").sum(), mask.sum())

    def test_structured_disagreement_detects_region_change(self) -> None:
        first = valid_target("present")
        second = valid_target("present")
        second["region"]["shape"] = "elongated"
        self.assertGreater(structured_disagreement(first, second), 0.0)

    def test_same_image_retrieval_reports_parent_level_scores(self) -> None:
        region = [torch.eye(4)]
        text = [torch.eye(4)]
        report = _same_image_retrieval(
            region, text, ["parent_a", "parent_a", "parent_b", "parent_b"]
        )
        self.assertEqual(report["mean_r1"], 1.0)
        self.assertEqual(report["per_parent_mean_r1"], {"parent_a": 1.0, "parent_b": 1.0})

    def test_end_to_end_region_targets_never_fall_back_to_wrong_global_mask(self) -> None:
        rows = [
            {
                "sample_id": "global_positive",
                "parent_sample_id": "parent_positive",
                "task_family": "global_landslide_segmentation",
                "mask": {"positive_pixels": 50, "empty_mask": False},
            },
            {
                "sample_id": "referring_positive_instruction",
                "parent_sample_id": "parent_positive",
                "parent_referring_target_sample_id": "ref_target_1",
                "task_family": "referring_landslide_segmentation",
                "mask": {"positive_pixels": 10, "empty_mask": False},
            },
            {
                "sample_id": "referring_absent_instruction",
                "parent_sample_id": "parent_positive",
                "parent_referring_target_sample_id": "ref_target_absent",
                "task_family": "no_target_segmentation",
                "mask": {"positive_pixels": 0, "empty_mask": True},
            },
            {
                "sample_id": "global_empty",
                "parent_sample_id": "parent_empty",
                "task_family": "global_landslide_segmentation",
                "mask": {"positive_pixels": 0, "empty_mask": True},
            },
        ]
        resolver = EndToEndTargetResolver(rows)
        referring = resolver.resolve({
            "parent_sample_id": "parent_positive",
            "region_id": "referring_region",
            "region_source": "gt_referring_mask",
            "source_region_aliases": [{"sample_id": "ref_target_1"}],
        })
        self.assertEqual(referring["segmentation_sample_id"], "referring_positive_instruction")
        absent = resolver.resolve({
            "parent_sample_id": "parent_positive",
            "region_id": "absent_region",
            "region_source": "no_target",
            "source_region_aliases": [{"sample_id": "ref_target_absent"}],
        })
        self.assertEqual(absent["segmentation_task_family"], "no_target_segmentation")
        empty_global = resolver.resolve({
            "parent_sample_id": "parent_empty",
            "region_id": "no_target",
            "region_source": "no_target",
            "source_region_aliases": [],
        })
        self.assertEqual(empty_global["mapping_kind"], "empty_global_instruction")
        supported, reason = end_to_end_region_support({
            "region_source": "pseudo_instance_component",
            "source_region_aliases": [],
        })
        self.assertFalse(supported)
        self.assertEqual(reason, "component_without_language_target")
        with self.assertRaisesRegex(KeyError, "referring alias"):
            resolver.resolve({
                "parent_sample_id": "parent_positive",
                "region_id": "component_001",
                "region_source": "pseudo_instance_component",
                "source_region_aliases": [],
            })
        with self.assertRaisesRegex(KeyError, "global target.*非空"):
            resolver.resolve({
                "parent_sample_id": "parent_positive",
                "region_id": "no_target",
                "region_source": "no_target",
                "source_region_aliases": [],
            })

    def test_train_prediction_requires_out_of_fold_protocol(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-fold|fold"):
            export_predicted_regions(
                segmentation_config=None,
                checkpoint="missing.pt",
                source_index="missing.jsonl",
                split="train",
                output_dir="unused",
                device=torch.device("cpu"),
                threshold=0.5,
            )

    def test_oof_fold_indexes_are_parent_isolated_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmentation = root / "instruction_train.jsonl"
            bridge = root / "auto_train.jsonl"
            parents = [f"parent_{index}" for index in range(9)]
            segmentation_rows = [
                {
                    "sample_id": f"instruction_{parent}_{view}",
                    "parent_sample_id": parent,
                    "split": "train",
                }
                for parent in parents
                for view in range(2)
            ]
            bridge_rows = [
                {
                    "bridge_record_id": f"bridge_{parent}",
                    "parent_sample_id": parent,
                    "split": "train",
                    "region_source": "gt_global_mask",
                    "dataset_name": "dataset_a" if index < 6 else "dataset_b",
                    "modality_family_combo": "optical" if index % 2 else "multispectral+terrain",
                    "expert_target": {"summary": "reviewed target"},
                }
                for index, parent in enumerate(parents)
            ]
            segmentation.write_text(
                "".join(json.dumps(row) + "\n" for row in segmentation_rows), encoding="utf-8"
            )
            bridge.write_text(
                "".join(json.dumps(row) + "\n" for row in bridge_rows), encoding="utf-8"
            )
            first = build_oof_fold_indexes(
                segmentation_index=segmentation,
                bridge_index=bridge,
                output_dir=root / "folds_a",
                num_folds=3,
                seed=42,
            )
            second = build_oof_fold_indexes(
                segmentation_index=segmentation,
                bridge_index=bridge,
                output_dir=root / "folds_b",
                num_folds=3,
                seed=42,
            )
            self.assertEqual(first["parent_to_fold"], second["parent_to_fold"])
            loaded = load_oof_manifest(root / "folds_a/fold_manifest.json")
            for fold, metadata in loaded["folds"].items():
                train_rows = [
                    json.loads(line)
                    for line in Path(metadata["train_index"]).read_text(encoding="utf-8").splitlines()
                ]
                train_parents = {row["parent_sample_id"] for row in train_rows}
                held_out = {
                    parent for parent, assigned in loaded["parent_to_fold"].items()
                    if assigned == fold
                }
                self.assertTrue(held_out)
                self.assertFalse(train_parents & held_out)

    def test_segdesc_config_rejects_unknown_region_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml")
            payload = __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
            payload["region_encoder"] = "union_bbox"
            path = Path(directory) / "invalid.yaml"
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "region_encoder"):
                load_segdesc_config(path)

    def test_joint_default_pattern_is_fifty_twenty_five_twenty_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml")
            payload = __import__("yaml").safe_load(source.read_text(encoding="utf-8"))
            payload.pop("joint_task_pattern", None)
            path = Path(directory) / "default_joint_pattern.yaml"
            path.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
            config = load_segdesc_config(path)
            self.assertEqual(
                config.resolved_joint_task_pattern(),
                ("segmentation", "global_caption", "segmentation", "region_description"),
            )

    def test_initialize_can_replace_only_region_encoder_while_resume_stays_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = FakeSegDescCheckpointModel("mgrr")
            target = FakeSegDescCheckpointModel("crop_only")
            target_region_before = {
                key: value.detach().clone() for key, value in target.mgrr.state_dict().items()
            }
            save_segdesc_checkpoint(
                path,
                source,
                step=7,
                segmentation_migration={"source": "synthetic"},
            )
            step, report = initialize_segdesc_checkpoint(path, target)
            self.assertEqual(step, 7)
            self.assertTrue(report["initialization"]["region_encoder_reinitialized"])
            self.assertTrue(torch.equal(source.shared.weight, target.shared.weight))
            for key, value in target.mgrr.state_dict().items():
                self.assertTrue(torch.equal(value, target_region_before[key]))
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                load_segdesc_checkpoint(path, target)

    def test_expert_factuality_is_aggregated_by_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generations = [
                {
                    "sample_id": sample,
                    "parent_sample_id": "parent_1",
                    "raw_metrics": {"raw_schema_valid": True},
                }
                for sample in ("sample_a", "sample_b")
            ]
            (root / "raw_generations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
            )
            review_paths = []
            for reviewer in ("reviewer_1", "reviewer_2"):
                path = root / f"{reviewer}.jsonl"
                rows = [{
                    "sample_id": sample,
                    "reviewer_id": reviewer,
                    "family_scores": {
                        "target_status": 1.0,
                        "region_geometry": 1.0,
                        "surface": 1.0,
                        "terrain": 1.0,
                        "sar": 0.5,
                        "deformation": 0.5,
                        "surrounding_context": 0.5,
                        "summary": 0.5,
                    },
                    "claims": [{"claim_id": "c1", "support": "supported"}],
                } for sample in ("sample_a", "sample_b")]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                review_paths.append(path)
            report = aggregate_expert_factuality(root, review_paths, seed=42)
            self.assertEqual(report["num_parents"], 1)
            self.assertAlmostEqual(report["expert_region_factuality_score"], 0.75)
            self.assertEqual(report["expert_unsupported_claim_rate"], 0.0)
            self.assertAlmostEqual(
                report["field_agreement"]["target_status"]["exact_agreement"], 1.0
            )
            self.assertAlmostEqual(
                report["field_agreement"]["summary"]["krippendorff_alpha_nominal"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
