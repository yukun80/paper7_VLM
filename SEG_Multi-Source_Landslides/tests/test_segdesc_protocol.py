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
from qpsalm_seg.engine.evaluator import SAMPLE_IDENTITY_FIELDS
from qpsalm_seg.description.config import load_segdesc_config
from qpsalm_seg.description.counterfactuals import counterfactual_region_masks
from qpsalm_seg.description.data import (
    DescriptionTaskDataset,
    end_to_end_region_support,
    same_parent_region_swap_candidates,
)
from qpsalm_seg.description.metrics import DescriptionMetricAccumulator, structured_disagreement
from qpsalm_seg.description.predicted_regions import export_predicted_regions
from qpsalm_seg.description.expert_factuality import (
    aggregate_expert_factuality,
    build_expert_review_template,
)
from qpsalm_seg.description.evaluator import EndToEndTargetResolver, _same_image_retrieval
from qpsalm_seg.description.model import (
    SegmentationGroundedDescriptionModel,
    alignment_positive_mask,
    multi_positive_alignment_loss,
)
from qpsalm_seg.description.runtime import (
    description_parameter_groups,
    description_trainable_parameter_manifest,
)
from qpsalm_seg.description.backbone import DescriptionCacheBackboneEncoder
from qpsalm_seg.description.common import ParentGroupedRegionBatchSampler
from qpsalm_seg.description.joint_trainer import (
    build_joint_optimizer,
    joint_optimizer_manifest,
    monitor_baseline_identity,
    monitor_retention_gate,
    restore_joint_progress,
    validate_joint_task_gradients,
)
from qpsalm_seg.description.checkpoint import (
    initialize_segdesc_checkpoint,
    load_segdesc_checkpoint,
    save_segdesc_checkpoint,
)
from qpsalm_seg.description.oof import build_oof_fold_indexes, load_oof_manifest
from qpsalm_seg.description.comparison import _counterfactual_gate, _rows
from qpsalm_seg.cli.eval_segdesc_retention import build_retention_gate


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


class FakeTokenizer:
    eos_token_id = 3

    def __call__(self, _text: str, *, add_special_tokens: bool = False) -> dict:
        del add_special_tokens
        return {"input_ids": [1, 2]}


class SequenceProtocolHarness(nn.Module):
    _token_ids = SegmentationGroundedDescriptionModel._token_ids
    _instruction_prompt = staticmethod(SegmentationGroundedDescriptionModel._instruction_prompt)
    _visual_tokens_for_sample = SegmentationGroundedDescriptionModel._visual_tokens_for_sample
    _build_sequences = SegmentationGroundedDescriptionModel._build_sequences

    def __init__(self) -> None:
        super().__init__()
        language_model = nn.Module()
        language_model.embedding = nn.Embedding(8, 4)
        language_model.get_input_embeddings = lambda: language_model.embedding
        self.controller = SimpleNamespace(model=language_model, tokenizer=FakeTokenizer())
        self.description_view_to_hidden = nn.Linear(3, 4, bias=False)
        self.region_to_hidden = nn.Linear(2, 4, bias=False)
        self.instruction_type = nn.Parameter(torch.zeros(4))
        self.visual_type = nn.Parameter(torch.zeros(4))
        self.region_type = nn.Parameter(torch.zeros(4))


class StageParameterHarness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segmentation = nn.Module()
        self.segmentation.controller = nn.Module()
        self.segmentation.controller.lora_A = nn.Module()
        self.segmentation.controller.lora_A.default = nn.Linear(2, 2, bias=False)
        self.segmentation.controller.lora_A.desc_adapter = nn.Linear(2, 2, bias=False)
        self.description_backbone = nn.Linear(2, 2)
        self.mgrr = nn.Linear(2, 2)
        self.region_to_hidden = nn.Linear(2, 2)
        self.description_view_to_hidden = nn.Linear(2, 2)
        self.alignment_text_projection = nn.Linear(2, 2)
        self.region_type = nn.Parameter(torch.zeros(2))
        self.instruction_type = nn.Parameter(torch.zeros(2))
        self.visual_type = nn.Parameter(torch.zeros(2))
        self.alignment_temperature = nn.Parameter(torch.tensor(0.07))


class RegionBypassHarness:
    _description_region_state = SegmentationGroundedDescriptionModel._description_region_state

    def __init__(self) -> None:
        self.segmentation = SimpleNamespace(config=SimpleNamespace(decoder_dim=2))

    def build_region_state(self, *_args, **_kwargs):
        raise RuntimeError("region replay executed")


class SegDescProtocolTest(unittest.TestCase):
    @staticmethod
    def _joint_gradient_role(nonzero: int) -> dict:
        return {
            "num_parameters": 2,
            "num_with_grad": int(nonzero > 0),
            "num_nonzero": nonzero,
            "norm_sum": float(nonzero),
            "all_finite": True,
        }

    def test_joint_gradient_gate_distinguishes_global_and_region_description(self) -> None:
        global_report = {
            "segmentation_adapter": self._joint_gradient_role(0),
            "description_adapter": self._joint_gradient_role(1),
            "mgrr": self._joint_gradient_role(0),
            "description_projection": self._joint_gradient_role(1),
        }
        gate = validate_joint_task_gradients(
            "global_caption", global_report,
            train_shared_segmentation_dense=False,
        )
        self.assertTrue(gate["passed"])
        region_gate = validate_joint_task_gradients(
            "region_description", global_report,
            train_shared_segmentation_dense=False,
        )
        self.assertFalse(region_gate["passed"])
        self.assertEqual(region_gate["missing_or_zero"], ["mgrr"])

        region_report = dict(global_report)
        region_report["mgrr"] = self._joint_gradient_role(1)
        self.assertTrue(validate_joint_task_gradients(
            "region_description", region_report,
            train_shared_segmentation_dense=False,
        )["passed"])
        leaked = dict(global_report)
        leaked["segmentation_adapter"] = self._joint_gradient_role(1)
        leak_gate = validate_joint_task_gradients(
            "global_caption", leaked,
            train_shared_segmentation_dense=False,
        )
        self.assertFalse(leak_gate["passed"])
        self.assertEqual(leak_gate["leaked_nonzero_roles"], ["segmentation_adapter"])

    def test_joint_optimizer_manifest_covers_exact_trainable_parameters(self) -> None:
        model = StageParameterHarness()
        config = SimpleNamespace(
            joint_train_shared_segmentation_dense=False,
            desc_adapter_lr_scale=0.2,
            learning_rate=1.0e-4,
            weight_decay=0.01,
            warmup_steps=2,
            max_steps=10,
        )
        optimizer, _scheduler = build_joint_optimizer(model, config)
        manifest = joint_optimizer_manifest(model, optimizer)
        self.assertEqual(manifest["protocol"], "qpsalm_segdesc_joint_optimizer_v1")
        self.assertEqual(
            {group["role"] for group in manifest["groups"]},
            {"segmentation_adapter", "description_adapter", "mgrr", "description_projection"},
        )
        listed = {
            name for group in manifest["groups"] for name in group["parameter_names"]
        }
        self.assertEqual(
            listed,
            {name for name, value in model.named_parameters() if value.requires_grad},
        )

    @staticmethod
    def _monitor_report(population_hash: str, dice: float) -> dict:
        return {
            "threshold": 0.5,
            "coverage": {
                "num_samples": 2,
                "sample_population": {
                    "protocol": "qpsalm_segmentation_eval_population_v1",
                    "fields": list(SAMPLE_IDENTITY_FIELDS),
                    "sha256": population_hash,
                    "num_records": 2,
                    "num_unique_sample_ids": 2,
                    "complete": True,
                    "unique": True,
                    "incomplete_record_indices": [],
                    "duplicate_sample_ids": [],
                },
            },
            "metrics": {"positive_only": {"dice": dice}},
        }

    def test_joint_monitor_retention_freezes_population_identity(self) -> None:
        baseline = monitor_baseline_identity(self._monitor_report("population-a", 0.60))
        passed = monitor_retention_gate(
            baseline,
            self._monitor_report("population-a", 0.595),
            maximum_allowed_drop=0.01,
        )
        self.assertTrue(passed["passed"])
        changed = monitor_retention_gate(
            baseline,
            self._monitor_report("population-b", 0.80),
            maximum_allowed_drop=0.01,
        )
        self.assertFalse(changed["passed"])
        self.assertFalse(changed["same_sample_population"])

    def test_joint_progress_resume_rejects_changed_population(self) -> None:
        populations = {
            "segmentation": {"s1", "s2"},
            "global_caption": {"g1"},
            "region_description": {"r1", "r2"},
        }
        from qpsalm_seg.description.joint_trainer import _joint_progress_payload
        progress = _joint_progress_payload(
            step=3,
            task_steps={"segmentation": 1, "global_caption": 1, "region_description": 1},
            task_samples={"segmentation": 2, "global_caption": 1, "region_description": 2},
            parent_coverage={
                "segmentation": {"s1"},
                "global_caption": {"g1"},
                "region_description": {"r1"},
            },
            parent_populations=populations,
        )
        steps, samples, coverage = restore_joint_progress(
            progress, populations, checkpoint_step=3, required=True,
        )
        self.assertEqual(sum(steps.values()), 3)
        self.assertEqual(samples["segmentation"], 2)
        self.assertEqual(coverage["region_description"], {"r1"})
        changed = dict(populations)
        changed["region_description"] = {"r1", "r3"}
        with self.assertRaisesRegex(RuntimeError, "population"):
            restore_joint_progress(
                progress, changed, checkpoint_step=3, required=True,
            )
        with self.assertRaisesRegex(RuntimeError, "step"):
            restore_joint_progress(
                progress, populations, checkpoint_step=4, required=True,
            )

    def test_dior_sampler_places_same_parent_candidates_in_batches(self) -> None:
        class SizedDataset:
            rows = [
                {"parent_sample_id": parent, "sample_id": f"{parent}_{index}"}
                for parent, count in (
                    ("p1", 3), ("p2", 3), ("p3", 2),
                    ("single_1", 1), ("single_2", 1), ("single_3", 1), ("single_4", 1),
                )
                for index in range(count)
            ]
            epoch = 0

            def __len__(self) -> int:
                return len(self.rows)

        source = SizedDataset()
        sampler = ParentGroupedRegionBatchSampler(
            source, 4, seed=42, drop_last=True
        )
        first = list(iter(sampler))
        self.assertTrue(first)
        self.assertEqual(len(first), len(sampler))
        self.assertTrue(all(len(batch) == 4 for batch in first))
        for batch in first:
            parents = [source.rows[index]["parent_sample_id"] for index in batch]
            self.assertLess(len(set(parents)), len(parents))
        self.assertEqual(first, list(iter(sampler)))

    def test_global_caption_cache_path_skips_spatial_feature_projection(self) -> None:
        class FakeDescriptionBank:
            manifest = {
                "spatial_channels": 2,
                "render_size": 8,
                "token_dim": 3,
                "format": "synthetic_description_cache",
            }

            @staticmethod
            def record(_component: str, _parent: str) -> dict:
                return {
                    "lookup_key": "synthetic-key",
                    "source_ref": {"kind": "single_image"},
                    "views": [{
                        "name": "rgb",
                        "source_families": ["optical"],
                        "source_modalities": ["rgb"],
                        "quality_flags": [],
                        "render_transform": {},
                        "content_hash": "a" * 64,
                        "valid_mask": torch.ones(1, 8, 8),
                        "view_tokens": torch.ones(2, 3),
                        "description": "RGB view",
                        "spatial_features": [
                            torch.ones(2, size, size) for size in (8, 4, 2, 1)
                        ],
                    }],
                }

        encoder = DescriptionCacheBackboneEncoder(FakeDescriptionBank(), dim=4)
        global_state = encoder(
            [("single_image", "parent_001")], include_spatial=False
        )
        self.assertEqual(global_state.features.samples, [[]])
        self.assertFalse(global_state.metadata[0]["spatial_features_loaded"])
        self.assertEqual(global_state.visual_evidence.token_counts, (2,))
        region_state = encoder(
            [("single_image", "parent_001")], include_spatial=True
        )
        self.assertEqual(len(region_state.features.samples[0]), 1)
        self.assertTrue(region_state.metadata[0]["spatial_features_loaded"])

    def test_global_caption_sequence_excludes_region_replay_tokens(self) -> None:
        model = SequenceProtocolHarness()
        backbone = SimpleNamespace(visual_evidence=SimpleNamespace(
            tokens=torch.ones(1, 2, 3),
            token_mask=torch.ones(1, 2, dtype=torch.bool),
        ))
        region_state = SimpleNamespace(
            backbone=backbone,
            region_sequence_tokens=torch.ones(1, 1, 3, 2),
            region_sequence_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            region_tokens=None,
        )
        global_sequence, _, _, global_lengths = model._build_sequences(
            region_state, ["Describe the image."], None, [False]
        )
        region_sequence, _, _, region_lengths = model._build_sequences(
            region_state, ["Describe the selected region."], None, [True]
        )
        self.assertEqual(global_sequence.shape[1], 4)
        self.assertEqual(global_lengths, (4,))
        self.assertEqual(region_sequence.shape[1], 7)
        self.assertEqual(region_lengths, (7,))

    def test_description_trainable_modules_follow_curriculum_stage(self) -> None:
        def trainable(stage: str) -> set[str]:
            model = StageParameterHarness()
            config = SimpleNamespace(
                stage=stage,
                learning_rate=1.0e-4,
                desc_adapter_lr_scale=0.2,
                weight_decay=0.01,
            )
            description_parameter_groups(model, config)
            return {name for name, value in model.named_parameters() if value.requires_grad}

        global_names = trainable("mmrs_caption")
        self.assertTrue(any("desc_adapter" in name for name in global_names))
        self.assertTrue(any(name.startswith("description_view_to_hidden.") for name in global_names))
        self.assertIn("instruction_type", global_names)
        self.assertIn("visual_type", global_names)
        self.assertFalse(any(name.startswith("mgrr.") for name in global_names))
        self.assertFalse(any(name.startswith("description_backbone.") for name in global_names))
        self.assertNotIn("region_type", global_names)

        alignment_names = trainable("dior_alignment")
        self.assertTrue(any(name.startswith("mgrr.") for name in alignment_names))
        self.assertTrue(any(name.startswith("description_backbone.") for name in alignment_names))
        self.assertTrue(any(name.startswith("alignment_text_projection.") for name in alignment_names))
        self.assertIn("instruction_type", alignment_names)
        self.assertFalse(any(name.startswith("region_to_hidden.") for name in alignment_names))
        self.assertNotIn("region_type", alignment_names)
        self.assertNotIn("visual_type", alignment_names)

        bridge_names = trainable("bridge_auto")
        for prefix in (
            "description_backbone.", "mgrr.", "region_to_hidden.",
            "description_view_to_hidden.", "alignment_text_projection.",
        ):
            self.assertTrue(any(name.startswith(prefix) for name in bridge_names))
        self.assertIn("region_type", bridge_names)
        self.assertIn("instruction_type", bridge_names)
        self.assertIn("visual_type", bridge_names)

    def test_global_caption_state_skips_region_replay(self) -> None:
        model = RegionBypassHarness()
        mask = torch.zeros(2, 1, 8, 8)
        backbone = SimpleNamespace(valid_mask=torch.ones_like(mask))
        state = model._description_region_state(
            backbone,
            mask,
            region_valid_mask=None,
            protocol="vision_only",
            structured_outputs=[False, False],
        )
        self.assertEqual(state.region_tokens.shape, (2, 1, 2))
        self.assertTrue(bool(state.diagnostics["global_caption_region_replay_skipped"].all()))
        with self.assertRaisesRegex(RuntimeError, "region replay executed"):
            model._description_region_state(
                backbone,
                mask,
                region_valid_mask=None,
                protocol="vision_only",
                structured_outputs=[False, True],
            )

    def test_trainable_parameter_manifest_matches_optimizer_groups(self) -> None:
        model = StageParameterHarness()
        config = SimpleNamespace(
            stage="mmrs_caption",
            learning_rate=1.0e-4,
            desc_adapter_lr_scale=0.2,
            weight_decay=0.01,
        )
        groups = description_parameter_groups(model, config)
        manifest = description_trainable_parameter_manifest(
            model, groups, stage=config.stage
        )
        self.assertEqual(
            manifest["protocol"], "qpsalm_description_trainable_parameters_v1"
        )
        self.assertEqual(manifest["stage"], "mmrs_caption")
        flattened = {
            name
            for group in manifest["groups"]
            for name in group["parameter_names"]
        }
        self.assertEqual(
            flattened,
            {name for name, value in model.named_parameters() if value.requires_grad},
        )

    def test_retention_requires_exact_sample_population_identity(self) -> None:
        population = {
            "protocol": "qpsalm_segmentation_eval_population_v1",
            "fields": list(SAMPLE_IDENTITY_FIELDS),
            "sha256": "a" * 64,
            "complete": True,
            "unique": True,
            "num_records": 10,
            "num_unique_sample_ids": 10,
        }
        baseline = {
            "threshold": 0.5,
            "coverage": {"num_samples": 10, "sample_population": population},
            "metrics": {"positive_only": {"dice": 0.50}},
        }
        candidate = {
            "threshold": 0.5,
            "coverage": {"num_samples": 10, "sample_population": dict(population)},
            "metrics": {"positive_only": {"dice": 0.495}},
        }
        gate = build_retention_gate(
            baseline, candidate, split="val", max_samples=0,
            checkpoint="joint.pt", checkpoint_step=100,
            checkpoint_metadata={"metadata": {"stage": "joint"}},
            maximum_allowed_drop=0.01,
        )
        self.assertTrue(gate["passed"])
        candidate["coverage"]["sample_population"]["sha256"] = "b" * 64
        gate = build_retention_gate(
            baseline, candidate, split="val", max_samples=0,
            checkpoint="joint.pt", checkpoint_step=100,
            checkpoint_metadata={"metadata": {"stage": "joint"}},
            maximum_allowed_drop=0.01,
        )
        self.assertFalse(gate["scientific_gate_eligible"])
        self.assertFalse(gate["passed"])

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

    def test_target_status_macro_uses_only_labels_present_in_evaluation(self) -> None:
        metric = DescriptionMetricAccumulator()
        for status in ("present", "absent"):
            target = json.dumps(valid_target(status))
            metric.update(
                prediction=target,
                target_text=target,
                references=[target],
                structured=True,
                metadata={"sample_id": status, "task_family": "bridge"},
            )
        status = metric.compute()["target_status"]
        self.assertEqual(status["active_labels"], ["present", "absent"])
        self.assertEqual(status["macro_f1"], 1.0)
        self.assertEqual(status["balanced_accuracy"], 1.0)

    def test_region_counterfactuals_preserve_canvas(self) -> None:
        mask = torch.zeros(2, 1, 8, 12)
        mask[0, :, 2:5, 3:7] = 1
        mask[1, :, 1:4, 8:10] = 1
        for mode in ("full_mask", "zero_mask", "shuffled_mask"):
            changed = counterfactual_region_masks(mask, mode)
            self.assertEqual(changed.shape, mask.shape)
            self.assertTrue(bool(torch.isfinite(changed).all()))
        self.assertEqual(counterfactual_region_masks(mask, "shuffled_mask").sum(), mask.sum())
        with self.assertRaisesRegex(ValueError, "同一 parent"):
            counterfactual_region_masks(mask, "region_swap")

    def test_region_swap_candidates_are_real_regions_from_same_parent(self) -> None:
        rows = [{
            "bridge_record_id": "current", "parent_sample_id": "p1",
            "region_id": "global", "region_source": "gt_global_mask",
            "target_status": "present", "region_mask": {"path": "global.npy"},
        }]
        catalog = [
            *rows,
            {
                "bridge_record_id": "same-parent", "parent_sample_id": "p1",
                "region_id": "component-1", "region_source": "pseudo_instance_component",
                "target_status": "present", "region_mask": {"path": "component.npy"},
            },
            {
                "bridge_record_id": "null", "parent_sample_id": "p1",
                "region_id": "no-target", "region_source": "no_target",
                "target_status": "absent", "region_mask": None,
            },
            {
                "bridge_record_id": "other-parent", "parent_sample_id": "p2",
                "region_id": "component-2", "region_source": "pseudo_instance_component",
                "target_status": "present", "region_mask": {"path": "other.npy"},
            },
        ]
        selected = same_parent_region_swap_candidates(
            rows, "current", catalog=catalog
        )
        self.assertEqual(
            [description_row["bridge_record_id"] for description_row in selected],
            ["same-parent"],
        )
        self.assertEqual(
            same_parent_region_swap_candidates(rows, "missing", catalog=catalog),
            [],
        )

    def test_cross_parent_donor_resolution_works_with_batch_size_one(self) -> None:
        class FakeBank:
            def record(self, _component: str, parent: str) -> dict:
                family = "optical" if parent != "p3" else "terrain"
                return {"views": [{"source_families": [family]}]}

        rows = [
            {"bridge_record_id": "s1", "parent_sample_id": "p1"},
            {"bridge_record_id": "s1-view", "parent_sample_id": "p1"},
            {"bridge_record_id": "s2", "parent_sample_id": "p2"},
            {"bridge_record_id": "s3", "parent_sample_id": "p3"},
        ]
        dataset = DescriptionTaskDataset.__new__(DescriptionTaskDataset)
        dataset.stage = "bridge_expert"
        dataset.rows = rows
        dataset._rows_by_sample_id = {
            row["bridge_record_id"]: row for row in rows
        }
        dataset._request_family_cache = {}
        dataset.vision_bank = FakeBank()
        donor = dataset.cross_parent_modality_swap_request("s1")
        self.assertIsNotNone(donor)
        request, audit = donor
        self.assertEqual(request, ("multisource_parent", "p2"))
        self.assertEqual(audit["target_parent_sample_id"], "p1")
        self.assertEqual(audit["donor_parent_sample_id"], "p2")
        self.assertEqual(audit["common_modality_families"], ["optical"])

    def test_formal_counterfactual_gate_requires_negative_paired_ci(self) -> None:
        report = {"counterfactual_sensitivity": {
            "shuffled_mask": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.1},
            },
            "region_swap": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.2},
            },
            "cross_parent_modality_swap": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_target_score_delta_ci": {"high": -0.05}
            },
            "modality_removal": {
                "requested": 8, "n": 8, "coverage_complete": True,
                "paired_factual_claim_count_delta_ci": {"high": -0.1}
            },
        }}
        self.assertTrue(_counterfactual_gate(report)["passed"])
        report["counterfactual_sensitivity"]["region_swap"][
            "paired_target_score_delta_ci"
        ]["high"] = 0.01
        self.assertFalse(_counterfactual_gate(report)["passed"])
        report["counterfactual_sensitivity"]["region_swap"][
            "paired_target_score_delta_ci"
        ]["high"] = -0.2
        report["counterfactual_sensitivity"]["region_swap"]["n"] = 7
        self.assertFalse(_counterfactual_gate(report)["passed"])

    def test_formal_comparison_rejects_pre_v3_evaluation_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw_generations.jsonl").write_text("", encoding="utf-8")
            report = {
                "protocol": "qpsalm_description_evaluation_v2",
                "generation_coverage": {"complete": True},
            }
            (root / "eval_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "evaluation_v3"):
                _rows(root, require_complete_generation=True)
            report["protocol"] = "qpsalm_description_evaluation_v3"
            (root / "eval_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            rows, observed = _rows(root, require_complete_generation=True)
            self.assertEqual(rows, {})
            self.assertEqual(observed["protocol"], report["protocol"])

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

    def test_same_image_retrieval_treats_duplicate_phrases_as_multi_positive(self) -> None:
        region = [torch.eye(3)]
        text = [torch.eye(3)[torch.tensor([1, 0, 2])]]
        report = _same_image_retrieval(
            region,
            text,
            ["parent_a", "parent_a", "parent_a"],
            ["landslide scar", "landslide scar", "road"],
        )
        self.assertEqual(report["num_ambiguous_phrase_queries"], 2)
        self.assertEqual(report["mean_r1"], 1.0)

    def test_alignment_loss_does_not_treat_same_parent_duplicate_phrase_as_negative(self) -> None:
        logits = torch.tensor([
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        positives = alignment_positive_mask(
            ["landslide scar", "landslide scar", "road"],
            ["parent_a", "parent_a", "parent_a"],
            device=logits.device,
        )
        loss = multi_positive_alignment_loss(logits, positives)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertLess(float(loss), 0.05)

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
            preview = root / "synthetic_preview.png"
            preview.write_bytes(b"synthetic-review-asset")
            generations = [
                {
                    "sample_id": sample,
                    "parent_sample_id": "parent_1",
                    "raw_metrics": {"raw_schema_valid": True},
                    "raw_generation": json.dumps(valid_target("present")),
                    "instruction": "Describe the selected landslide region.",
                    "visual_preview_path": str(preview),
                }
                for sample in ("sample_a", "sample_b")
            ]
            (root / "raw_generations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
            )
            (root / "eval_report.json").write_text(json.dumps({
                "generation_coverage": {"complete": True},
                "evaluation_mode": "gt_mask",
            }), encoding="utf-8")
            templates = build_expert_review_template(root)
            review_paths = []
            for reviewer in ("reviewer_1", "reviewer_2"):
                path = root / f"{reviewer}.jsonl"
                rows = [{
                    **template,
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
                    "claims": [
                        {**claim, "support": "supported"}
                        for claim in template["claims"]
                    ],
                } for template in templates]
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
