#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SANE/QMEF/PMRD 核心正确性测试。

用途：验证 valid mask、负样本指标、SANE/QMEF/PMRD、matching 和 checkpoint 行为。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m unittest
SEG_Multi-Source_Landslides/tests/test_refactor_core.py -v
主要输入：代码内构造的合成张量和临时 checkpoint。
主要输出：unittest 终端结果。
写入行为：仅使用系统临时目录，不写 benchmark 或正式 outputs。
所属流程：算法重构后的单元回归测试。
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data._utils.pin_memory import pin_memory as torch_pin_memory

from qpsalm_seg.config import QPSalmConfig, load_config
from qpsalm_seg.controllers import (
    QwenMaskQueryController,
    configure_qwen_gradient_checkpointing,
    local_model_revision,
    pad_qwen_sequences,
    qwen_gradient_checkpointing_kwargs,
)
from qpsalm_seg.cli.cache_qwen_vision_features import restore_qwen_patch_grid
from qpsalm_seg.cli.ablation_report import build_ablation_report
from qpsalm_seg.cli.compare_runs import compare_seed_series
from qpsalm_seg.cli.integration_check import (
    lora_gradient_report,
    lora_parameter_update_summary,
    runtime_library_report,
    select_representative_batch_indices,
    snapshot_lora_parameters,
)
from qpsalm_seg.cli.summarize_run import train_history_summary
from qpsalm_seg.data import build_prompt_triplet, modality_valid_mask, resize_pad_tensor, swap_padding_after_flip
from qpsalm_seg.data.prompts import PROMPT_VERSION, transform_spatial_instruction
from qpsalm_seg.data.dataset import (
    MultiSourceLandslideDataset,
    effective_canvas_gsd,
    scaled_band_metadata,
    select_monitor_rows,
)
from qpsalm_seg.data.samplers import TaskBalancedSizeBucketBatchSampler, task_group
from qpsalm_seg.description import (
    MultiGranularityRegionReplay,
    SingleVectorRegionPooling,
    rasterize_region_geometry,
    retarget_region_mask_between_cache_views,
    transform_region_mask_to_cache,
)
import qpsalm_seg.description.mgrr as mgrr_module
from qpsalm_seg.description.counterfactuals import counterfactual_backbone
from qpsalm_seg.matching import assign_proposals
from qpsalm_seg.metrics import batch_binary_metric_tensors, batch_binary_metrics
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.models.sane import SensorAwareNativeScaleEncoder
from qpsalm_seg.models.pmrd import ProposalSetMaskRefinementDecoder
from qpsalm_seg.models.qmef import (
    QwenGuidedEvidenceFusion,
    ScaleAwareDeformableAggregator,
    valid_weighted_pool,
)
from qpsalm_seg.models.vision_cache import (
    CACHE_FORMAT,
    QwenVisionFeatureBank,
    view_fingerprint_fragment,
)
from qpsalm_seg.presets import apply_preset
from qpsalm_seg.rendering import RENDERER_VERSION
from qpsalm_seg.schema import (
    ActiveModalitySubset,
    ModalityBatch,
    ModalityInstance,
    MultiScaleFeatures,
    MultisourceBackboneState,
)
from qpsalm_seg.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_training_schedule,
)
from qpsalm_seg.engine.diagnostics import collect_proposal_records, metric_metadata_with_scale
from qpsalm_seg.engine.optimizer import (
    apply_optimizer_schedule,
    build_optimizer,
    qwen_lora_gradient_summary,
    qwen_lora_update_summary,
    qwen_training_stage,
    snapshot_qwen_lora,
)
from qpsalm_seg.engine.threshold import restored_original_space_metrics
from qpsalm_seg.engine.trainer import validation_selection_score
from qpsalm_seg.visualize import save_visualizations


def instance(name: str, family: str, channels: int, size: int, sensor: str = "unknown") -> ModalityInstance:
    generator = torch.Generator().manual_seed(channels * 100 + size)
    image = torch.rand((channels, size, size), generator=generator)
    if family == "optical":
        product_type = "rgb"
        band_names = ("R", "G", "B")
        sensor = "generic_rgb"
    elif family == "multispectral":
        product_type = "surface_reflectance"
        band_names = tuple(f"B{index + 1:02d}" for index in range(channels))
    elif family == "terrain":
        product_type = "elevation"
        band_names = ("DEM",)
        sensor = "generic_dem"
    else:
        raise ValueError(f"unsupported test family={family}")
    return ModalityInstance(
        name=name,
        family=family,
        sensor=sensor,
        product_type=product_type,
        band_names=band_names,
        band_metadata=tuple({"native_gsd_m": 10.0, "signed": False} for _ in range(channels)),
        orbit="unknown",
        units="reflectance" if family != "terrain" else "m",
        signed=False,
        image=image,
        valid_mask=torch.ones((1, size, size)),
        native_gsd_m=10.0 if family != "optical" else 0.5,
        aligned_gsd_m=10.0,
        quality=1.0,
    )


def synthetic_batch(instances: list[ModalityInstance], components: int = 2, size: int = 64) -> ModalityBatch:
    mask = torch.zeros((1, 1, size, size))
    for index in range(components):
        y = 2 + (index // 8) * 6
        x = 2 + (index % 8) * 7
        mask[:, :, y : y + 3, x : x + 3] = 1.0
    metadata = [{
        "sample_id": "synthetic",
        "parent_sample_id": "synthetic",
        "dataset_name": "synthetic",
        "raw_combo": "+".join(item.name for item in instances),
        "family_combo": "+".join(sorted({item.family for item in instances})),
        "sensor_combo": "synthetic",
        "product_combo": "+".join(sorted({item.product_type for item in instances})),
        "gsd_token": "meter_5_10",
        "raw_modalities": [{"name": item.name} for item in instances],
    }]
    active = ActiveModalitySubset(
        active_names=tuple(sorted(item.name for item in instances)),
        dropped_names=(),
        signature="synthetic-full",
        is_full=True,
    )
    return ModalityBatch(
        instances=[instances],
        full_instances=[instances],
        active_subsets=[active],
        mask=mask,
        valid_mask=torch.ones_like(mask),
        metadata=metadata,
        proposal_context_text=["Segment all landslide regions using available sensors."],
        condition_prompt_text=["Condition prompt: landslide."],
        evidence_reasoning_text=["Use optical, multispectral and terrain evidence."],
        full_proposal_context_text=["Segment all landslide regions using available sensors."],
        full_condition_prompt_text=["Condition prompt: landslide."],
        full_evidence_reasoning_text=["Use optical, multispectral and terrain evidence."],
        visual_evidence_key=["qmv3-parent:synthetic"],
    )


def repeat_batch(batch: ModalityBatch, count: int) -> ModalityBatch:
    def repeat(values):
        return [values[0] for _ in range(count)]

    return ModalityBatch(
        instances=repeat(batch.instances),
        full_instances=repeat(batch.full_instances),
        active_subsets=repeat(batch.active_subsets),
        mask=batch.mask.repeat(count, 1, 1, 1),
        valid_mask=batch.valid_mask.repeat(count, 1, 1, 1),
        metadata=[{**batch.metadata[0], "sample_id": f"synthetic-{index}", "parent_sample_id": f"parent-{index}"} for index in range(count)],
        proposal_context_text=repeat(batch.proposal_context_text),
        condition_prompt_text=repeat(batch.condition_prompt_text),
        evidence_reasoning_text=repeat(batch.evidence_reasoning_text),
        full_proposal_context_text=repeat(batch.full_proposal_context_text),
        full_condition_prompt_text=repeat(batch.full_condition_prompt_text),
        full_evidence_reasoning_text=repeat(batch.full_evidence_reasoning_text),
        visual_evidence_key=[f"qmv3-parent:synthetic-{index}" for index in range(count)],
        component_masks=None,
    )


class ValidMetricTest(unittest.TestCase):
    def test_throughput_summary_weights_windows_and_preserves_steady_state(self) -> None:
        summary = train_history_summary([
            {
                "step_start": 0, "step_end": 0, "loss": 3.0,
                "samples_per_sec": 1.0, "qwen_tokens_per_sec": 100.0,
                "peak_reserved_gib": 10.0,
            },
            {
                "step_start": 1, "step_end": 9, "loss": 2.0,
                "samples_per_sec": 5.0, "qwen_tokens_per_sec": 500.0,
                "peak_reserved_gib": 20.0,
            },
        ])
        performance = summary["performance"]
        self.assertAlmostEqual(performance["weighted_mean"]["samples_per_sec"], 4.6)
        self.assertEqual(performance["steady_state_last_window"]["samples_per_sec"], 5.0)
        self.assertEqual(performance["peak_reserved_gib"], 20.0)

    def test_main_yaml_owns_single_gpu_training_budget(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs"
        small = apply_preset(load_config(root / "qpsalm_v2_small.yaml"), None)
        full = apply_preset(load_config(root / "qpsalm_v2_full.yaml"), None)
        for config in (small, full):
            self.assertEqual(config.batch_size, 4)
            self.assertEqual(config.grad_accum_steps, 1)
            self.assertEqual(config.query_chunk_size, 16)
            self.assertEqual(config.amp_dtype, "bf16")
            self.assertEqual(config.qwen_gradient_checkpointing, "disabled")
        self.assertEqual(small.batch_size * small.max_steps, 24000)
        self.assertEqual(full.batch_size * full.max_steps, 80004)

    def test_gpu_train_metrics_match_evaluation_semantics(self) -> None:
        logits = torch.tensor([[[[-8.0, 8.0]]], [[[-8.0, -8.0]]]])
        target = torch.tensor([[[[0.0, 1.0]]], [[[0.0, 0.0]]]])
        valid = torch.ones_like(target)
        expected = batch_binary_metrics(logits, target, valid_mask=valid)
        observed = batch_binary_metric_tensors(logits, target, valid_mask=valid)
        self.assertAlmostEqual(float(observed["dice"]), sum(row["dice"] for row in expected) / 2, places=6)
        self.assertAlmostEqual(float(observed["iou"]), sum(row["iou"] for row in expected) / 2, places=6)

    def test_monitor_selection_is_parent_aware_and_deterministic(self) -> None:
        rows = []
        for dataset in ("a", "b"):
            for parent_index in range(8):
                parent = f"{dataset}-{parent_index}"
                for group, family in (
                    ("global", "global_landslide_segmentation"),
                    ("referring", "referring_landslide_segmentation"),
                    ("no_target", "no_target_segmentation"),
                ):
                    rows.append({
                        "sample_id": f"{parent}-{group}",
                        "parent_sample_id": parent,
                        "dataset_name": dataset,
                        "task_family": family,
                        "modalities": {"rgb": {"family": "optical", "available": True}},
                        "referring_target": {"category": "position" if group == "referring" else "no_target"},
                    })
        first = select_monitor_rows(rows, 24, 42)
        second = select_monitor_rows(rows, 24, 42)
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        self.assertEqual(len(first), 24)
        self.assertEqual({row["dataset_name"] for row in first}, {"a", "b"})
        self.assertEqual({task_group(row) for row in first}, {"global", "referring", "no_target"})

    @staticmethod
    def _ablation_bundle(instruction: str, visual: str, score: float) -> dict:
        records = {
            sample_id: {
                "sample_id": sample_id,
                "family_combo": family_combo,
                "final_dice": score,
                "final_iou": score,
                "selected_is_matched": score,
                "matched_relevance_rank_score": score,
                "component_recall": score,
                "proposal_union_dice": score,
            }
            for sample_id, family_combo in (
                ("terrain-sample", "optical+terrain"),
                ("optical-sample", "optical"),
            )
        }
        return {
            "path": f"{instruction}-{visual}/eval_report.json",
            "manifest": {
                "checkpoint": "checkpoint.pt", "checkpoint_step": 100,
                "split": "val", "preset": "qwen_psalm_full",
                "resolved_config": {
                    "instruction_ablation": instruction,
                    "visual_ablation": visual,
                },
            },
            "report": {
                "instruction_sensitivity": {
                    "instruction_contrast_ratio_16": score,
                    "paired_prediction_difference_rate": score,
                    "no_target_empty_prediction_rate": score,
                    "no_target_mean_unmatched_rejection": score,
                }
            },
            "records": records,
        }

    def test_ablation_report_requires_paired_instruction_and_visual_degradation(self) -> None:
        normal = self._ablation_bundle("normal", "normal", 0.8)
        instruction = {
            name: self._ablation_bundle(name, "normal", 0.4)
            for name in ("shuffled", "fixed-generic", "no-semantic")
        }
        visual = {
            "shuffled": (self._ablation_bundle("normal", "shuffled", 0.5), None),
            "text-only": (self._ablation_bundle("normal", "text-only", 0.5), None),
            "remove:terrain": (
                self._ablation_bundle("normal", "remove:terrain", 0.3), "terrain"
            ),
        }
        report = build_ablation_report(normal, instruction, visual, min_delta=0.0)
        self.assertTrue(report["acceptance"]["passed"])
        self.assertEqual(report["acceptance"]["num_checks"], 6)
        self.assertEqual(report["visual"]["remove:terrain"]["n"], 1)
        equal_visual = dict(visual)
        equal_visual["text-only"] = (
            self._ablation_bundle("normal", "text-only", 0.8), None
        )
        failed = build_ablation_report(normal, instruction, equal_visual, min_delta=0.0)
        self.assertFalse(failed["acceptance"]["passed"])
        mismatched_instruction = dict(instruction)
        mismatched = self._ablation_bundle("shuffled", "normal", 0.4)
        mismatched["manifest"] = {**mismatched["manifest"], "checkpoint": "other.pt"}
        mismatched_instruction["shuffled"] = mismatched
        with self.assertRaisesRegex(ValueError, "同一模型"):
            build_ablation_report(normal, mismatched_instruction, visual, min_delta=0.0)

    def test_local_qwen_revision_hashes_weight_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            weight = root / "model.safetensors"
            weight.write_bytes(b"first-weight-content")
            first = local_model_revision(root)
            local_model_revision.cache_clear()
            weight.write_bytes(b"other-weight-content")
            second = local_model_revision(root)
            self.assertNotEqual(first, second)

    def test_qwen_checkpoint_protocol_is_explicit(self) -> None:
        self.assertEqual(
            qwen_gradient_checkpointing_kwargs("reentrant"),
            {"use_reentrant": True, "preserve_rng_state": True},
        )
        with self.assertRaisesRegex(ValueError, "qwen_gradient_checkpointing"):
            qwen_gradient_checkpointing_kwargs("non_reentrant")

    def test_qwen_sequence_padding_preserves_variable_length_gradients(self) -> None:
        sequences = [
            torch.randn(length, 8, requires_grad=True)
            for length in (3, 5, 7)
        ]
        inputs, attention, lengths = pad_qwen_sequences(sequences)
        self.assertEqual(tuple(inputs.shape), (3, 7, 8))
        self.assertEqual(lengths.tolist(), [3, 5, 7])
        self.assertEqual(attention.sum(1).tolist(), [3, 5, 7])
        (inputs * attention[..., None]).sum().backward()
        self.assertTrue(all(value.grad is not None for value in sequences))
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in sequences))

    def test_qwen_checkpoint_configuration_enables_exactly_once(self) -> None:
        model = MagicMock()
        resolved = configure_qwen_gradient_checkpointing(model, "reentrant")
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={
                "use_reentrant": True,
                "preserve_rng_state": True,
            }
        )
        model.enable_input_require_grads.assert_called_once_with()
        model.gradient_checkpointing_disable.assert_not_called()
        self.assertEqual(resolved["use_reentrant"], True)

    def test_qwen_checkpoint_configuration_can_be_disabled(self) -> None:
        model = MagicMock()
        self.assertIsNone(configure_qwen_gradient_checkpointing(model, "disabled"))
        model.gradient_checkpointing_disable.assert_called_once_with()
        model.gradient_checkpointing_enable.assert_not_called()
        model.enable_input_require_grads.assert_not_called()

    def test_instruction_shuffle_never_silently_reuses_original_prompt(self) -> None:
        dataset = MultiSourceLandslideDataset.__new__(MultiSourceLandslideDataset)
        dataset.config = SimpleNamespace(instruction_ablation="shuffled")
        row = {
            "sample_id": "a", "parent_sample_id": "parent-a",
            "instruction": {"text": "Segment all landslides."},
        }
        dataset.rows = [row]
        with self.assertRaisesRegex(RuntimeError, "至少需要两个"):
            dataset._prompt_row(0, row)
    def test_referring_position_prompt_follows_spatial_flip(self) -> None:
        row = {
            "instruction": {"text": "Segment upper-left.", "text_zh": "分割左上。"},
            "referring_target": {
                "category": "position", "subtype": "upper-left",
                "grounding": {"grid": "upper-left"},
            },
        }
        transformed = transform_spatial_instruction(row, hflip=True, vflip=False)
        self.assertEqual(transformed["referring_target"]["grounding"]["grid"], "upper-right")
        self.assertIn("upper right", transformed["instruction"]["text"])
        self.assertEqual(row["referring_target"]["grounding"]["grid"], "upper-left")

    def test_two_of_three_seed_gate_uses_positive_and_component_metrics(self) -> None:
        def summary(value: float) -> dict:
            return {
                "acceptance": {"research_pipeline_ready": True},
                "eval": {
                    "overall": {"dice": value, "iou": value},
                    "positive_only": {"dice": value, "iou": value},
                    "instruction_sensitivity": {"instruction_contrast_ratio_16": value},
                    "proposal_diagnostics": {
                        "summary": {"overall": {"mean_component_recall": value}}
                    },
                },
            }

        report = compare_seed_series(
            [summary(0.2), summary(0.2), summary(0.2)],
            [summary(0.3), summary(0.25), summary(0.1)],
            baseline_name="base",
            candidate_name="candidate",
            min_delta=0.0,
        )
        self.assertEqual(report["successful_seeds"], 2)
        self.assertTrue(report["passed_2_of_3_gate"])

    def test_dynamic_resize_updates_canvas_and_encoder_gsd(self) -> None:
        row = {"spatial": {"original_size": [100, 200], "gsd_m": 10.0}}
        self.assertAlmostEqual(float(effective_canvas_gsd(row, 100)), 20.0)
        metadata = scaled_band_metadata(({"name": "B02", "native_gsd_m": 10.0},), 4.0)
        self.assertEqual(metadata[0]["source_native_gsd_m"], 10.0)
        self.assertEqual(metadata[0]["native_gsd_m"], 40.0)

    def test_modality_valid_mask_respects_nan_and_nodata(self) -> None:
        array = torch.ones((2, 4, 4)).numpy()
        array[:, 0, 0] = float("nan")
        array[:, 1, 1] = -9999.0
        valid = modality_valid_mask(array, {"valid_mask": {"nodata_value": -9999.0}})
        self.assertEqual(float(valid[0, 0, 0]), 0.0)
        self.assertEqual(float(valid[0, 1, 1]), 0.0)
        self.assertEqual(float(valid[0, 2, 2]), 1.0)

    def test_padding_does_not_change_metrics(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        target[:, :, 2:4, 2:4] = 1.0
        logits = torch.full_like(target, -8.0)
        logits[:, :, 2:4, 2:4] = 8.0
        logits[:, :, 6:, :] = 8.0
        valid = torch.ones_like(target)
        valid[:, :, 6:, :] = 0.0
        masked = batch_binary_metrics(logits, target, valid_mask=valid)[0]
        cropped = batch_binary_metrics(logits[:, :, :6], target[:, :, :6])[0]
        self.assertAlmostEqual(masked["iou"], cropped["iou"], places=7)
        self.assertAlmostEqual(masked["dice"], cropped["dice"], places=7)

    def test_empty_metrics_are_reported_separately(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        logits = torch.full_like(target, -8.0)
        result = batch_binary_metrics(logits, target)[0]
        self.assertEqual(result["negative_accuracy"], 1.0)
        self.assertEqual(result["empty_false_positive_rate"], 0.0)

    def test_proposal_signatures_ignore_invalid_pixels(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        target[:, :, 6:, :] = 1.0
        valid = torch.ones_like(target)
        valid[:, :, 6:, :] = 0.0
        final = torch.full_like(target, -8.0)
        final[:, :, 6:, :] = 8.0
        outputs = {
            "final_mask_logits": final,
            "proposal_mask_logits": final,
            "proposal_relevance_logits": torch.zeros((1, 1)),
        }
        records = collect_proposal_records(outputs, target, valid, [{}], [{}])
        self.assertEqual(records[0]["target_area"], 0.0)
        self.assertEqual(records[0]["final_mask_area"], 0.0)

    def test_matched_relevance_rank_uses_assignment_targets(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        valid = torch.ones_like(target)
        outputs = {
            "final_mask_logits": torch.full_like(target, -8.0),
            "proposal_mask_logits": torch.zeros((1, 4, 8, 8)),
            "proposal_relevance_logits": torch.tensor([[0.2, 2.0, 1.0, -1.0]]),
            "proposal_relevance_targets": torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        }
        record = collect_proposal_records(outputs, target, valid, [{}], [{}])[0]
        self.assertEqual(record["matched_relevance_mean_rank"], 1.5)
        self.assertAlmostEqual(record["matched_relevance_rank_score"], 5.0 / 6.0)

    def test_restored_original_size_metric_matches_perfect_canvas(self) -> None:
        source = torch.zeros((1, 20, 40))
        source[:, 4:14, 9:27] = 1.0
        canvas, transform = resize_pad_tensor(source, 64, mode="nearest")
        logits = torch.where(canvas[None] > 0.5, torch.full_like(canvas[None], 20.0), torch.full_like(canvas[None], -20.0))
        metrics = restored_original_space_metrics(logits, canvas[None], [{"resize_transform": transform}], 0.5)
        self.assertAlmostEqual(metrics[0]["iou"], 1.0, places=6)

    def test_original_size_metric_rejects_invalid_transform(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        malformed = [
            {"source_hw": [8, 8]},
            {
                "source_hw": [8, 8], "resized_hw": [8, 8], "target_hw": [8, 8],
                "pad_top": 0, "pad_left": 1,
            },
        ]
        for transform in malformed:
            with self.subTest(transform=transform):
                with self.assertRaisesRegex(ValueError, "cannot restore original-space metrics"):
                    restored_original_space_metrics(
                        torch.full_like(target, -8.0),
                        target,
                        [{"sample_id": "broken", "resize_transform": transform}],
                        0.5,
                        torch.ones_like(target),
                    )

    def test_metric_strata_and_original_metrics_ignore_invalid_pixels(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        target[:, :, 2:4, 2:4] = 1
        valid = torch.zeros_like(target)
        valid[:, :, :4] = 1
        metadata = [{
            "gsd_m": 1.0,
            "resize_transform": {
                "source_hw": [8, 8], "resized_hw": [8, 8], "target_hw": [8, 8],
                "pad_top": 0, "pad_left": 0, "scale": 1.0,
            },
        }]
        enriched = metric_metadata_with_scale(metadata, target, valid)
        self.assertAlmostEqual(enriched[0]["target_area_fraction"], 4 / 32)
        logits = torch.full_like(target, -20.0)
        logits[target > 0] = 20.0
        logits[:, :, 4:] = 20.0
        records = restored_original_space_metrics(logits, target, enriched, 0.5, valid)
        self.assertAlmostEqual(float(records[0]["dice"]), 1.0, places=5)

    def test_checkpoint_selection_uses_positive_only_dice(self) -> None:
        report = {"metrics": {"overall": {"dice": 0.9}, "positive_only": {"dice": 0.2}}}
        self.assertEqual(validation_selection_score(report, "positive_only_dice"), 0.2)

    def test_deformable_alignment_preserves_aspect_ratio_padding(self) -> None:
        module = ScaleAwareDeformableAggregator(dim=4, num_points=1)
        torch.nn.init.zeros_(module.offset_head.weight)
        torch.nn.init.zeros_(module.offset_head.bias)
        torch.nn.init.zeros_(module.weight_head.weight)
        torch.nn.init.zeros_(module.weight_head.bias)
        feature = torch.ones((4, 8, 16))
        valid = torch.ones((1, 8, 16))
        _, aligned_valid = module(
            feature, valid, (16, 16), torch.zeros(4), torch.zeros(4), 1.0
        )
        self.assertLess(float(aligned_valid[:, :3].mean().detach()), 0.1)
        self.assertGreater(float(aligned_valid[:, 5:11].mean().detach()), 0.9)

    def test_deformable_alignment_zeroes_fully_invalid_features(self) -> None:
        module = ScaleAwareDeformableAggregator(dim=4, num_points=1)
        feature = torch.ones((4, 4, 4))
        valid = torch.zeros((1, 4, 4))
        aligned, aligned_valid = module(
            feature, valid, (4, 4), torch.zeros(4), torch.zeros(4), 1.0
        )
        self.assertEqual(float(aligned_valid.sum().detach()), 0.0)
        self.assertEqual(float(aligned.abs().sum().detach()), 0.0)

    def test_flip_swaps_uneven_canvas_padding(self) -> None:
        source = torch.ones((1, 2, 5))
        canvas, transform = resize_pad_tensor(source, 8, "nearest")
        self.assertNotEqual(transform["pad_top"], transform["pad_bottom"])
        shifted, updated = swap_padding_after_flip(canvas, transform, hflip=False, vflip=True)
        self.assertEqual(updated["pad_top"], transform["pad_bottom"])
        self.assertEqual(updated["pad_bottom"], transform["pad_top"])
        self.assertEqual(float(shifted.sum()), float(canvas.sum()))


class ThreeModuleModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            QPSalmConfig(),
            controller="text_probe",
            preset="raw_sane_qmef_pmrd",
            decoder_dim=32,
            num_heads=4,
            num_mask_tokens=4,
            num_decoder_layers=1,
            modality_dropout=0.0,
            max_native_size=64,
            missing_modality_consistency_weight=0.0,
        )
        self.model = MultiSourceQwenPSALMSeg(self.config, torch.device("cpu")).eval()

    def test_variable_channels_and_modality_order(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 64)
        s2 = instance("multispectral", "multispectral", 12, 32, "sentinel2")
        terrain = instance("dem", "terrain", 1, 64, "dem")
        first = self.model(synthetic_batch([optical, s2, terrain]))
        second = self.model(synthetic_batch([terrain, optical, s2]))
        self.assertEqual(tuple(first.final_mask_logits.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(first["loss"]))
        self.assertTrue(torch.allclose(first.final_mask_logits, second.final_mask_logits, atol=2.0e-5, rtol=2.0e-5))

    def test_explicit_backbone_and_segmentation_state_match_forward(self) -> None:
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 64),
            instance("dem", "terrain", 1, 32),
        ])
        with torch.no_grad():
            backbone = self.model.encode_multisource(
                batch, use_full=False, include_visual_tokens=False
            )
            state = self.model.build_segmentation_state(
                batch, use_full=False, backbone=backbone
            )
            explicit = self.model.segment_from_state(state)
            regular = self.model(batch)
        self.assertTrue(torch.allclose(
            explicit.final_mask_logits, regular.final_mask_logits, atol=1.0e-6, rtol=1.0e-6
        ))
        self.assertEqual(backbone.reference_hw, batch.reference_hw)
        self.assertIsNone(backbone.visual_evidence)

    def test_backbone_metadata_does_not_capture_task_text(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        batch.metadata[0]["instruction"] = "private task text"
        batch.metadata[0]["condition"] = "private condition"
        backbone = self.model.encode_multisource(
            batch, include_visual_tokens=False
        )
        self.assertNotIn("instruction", backbone.metadata[0])
        self.assertNotIn("condition", backbone.metadata[0])

    def test_mgrr_region_tokens_backpropagate_to_sane_features(self) -> None:
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], size=32)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        region = torch.zeros((1, 1, 32, 32))
        region[:, :, 4:12, 5:13] = 1
        region[:, :, 20:25, 21:27] = 1
        mgrr = MultiGranularityRegionReplay(32, max_components=8)
        state = mgrr(backbone, region, protocol="assisted")
        self.assertEqual(tuple(state.region_tokens.shape), (1, 1, 32))
        self.assertIsNotNone(state.region_sequence_tokens)
        self.assertIsNotNone(state.region_sequence_mask)
        self.assertGreater(int(state.region_sequence_mask.sum()), 6)
        self.assertGreaterEqual(float(state.diagnostics["component_count"][0, 0]), 2.0)
        self.assertEqual(float(state.diagnostics["roi_grid_sample_count"]), 118.0)
        self.assertEqual(float(state.diagnostics["roi_query_count"]), 2.0)
        selected = state.region_sequence_tokens[state.region_sequence_mask]
        probe = torch.linspace(0.1, 1.0, selected.shape[-1])
        (selected * probe).sum().backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in self.model.sane.parameters()
        ))

    def test_mgrr_roi_grid_uses_reference_canvas_coordinates(self) -> None:
        feature = torch.ones((2, 4, 4), dtype=torch.float32)
        valid = torch.ones((1, 4, 4), dtype=torch.float32)
        reference_mask = torch.ones((1, 8, 8), dtype=torch.float32)
        tokens, token_valid = MultiGranularityRegionReplay._roi_grid_tokens(
            feature, valid, reference_mask, (2, 3)
        )
        self.assertEqual(tuple(tokens.shape), (6, 2))
        self.assertTrue(bool(token_valid.all()))
        self.assertTrue(torch.allclose(tokens, torch.ones_like(tokens)))

    def test_mgrr_view_transform_retargets_components_for_native_shapes_and_batch(self) -> None:
        source_transform = {
            "source_h": 16, "source_w": 32,
            "resized_h": 16, "resized_w": 32,
            "pad_top": 8, "pad_left": 0, "size": 32,
        }
        target_transform = {
            "source_h": 32, "source_w": 16,
            "resized_h": 32, "resized_w": 16,
            "pad_top": 0, "pad_left": 8, "size": 32,
        }
        source_region = torch.zeros((1, 16, 32))
        source_region[:, 1:5, 1:5] = 1
        source_region[:, 11:15, 27:31] = 1
        reference_region = transform_region_mask_to_cache(
            source_region, source_transform
        )
        target_region = retarget_region_mask_between_cache_views(
            reference_region, source_transform, target_transform
        )
        expected_target = transform_region_mask_to_cache(
            torch.nn.functional.interpolate(
                source_region[None], size=(32, 16), mode="nearest"
            )[0],
            target_transform,
        )
        self.assertTrue(torch.equal(target_region, expected_target))
        self.assertFalse(torch.equal(reference_region, target_region))

        batch = repeat_batch(synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], size=32), 2)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        for sample_index, pyramids in enumerate(backbone.features.samples):
            backbone.metadata[sample_index]["render_transforms"] = [
                source_transform, target_transform,
            ]
            pyramids[0].instance.metadata["render_transform"] = source_transform
            pyramids[1].instance.metadata["render_transform"] = target_transform
            for name in ("detail_valid", "high_valid", "mid_valid", "low_valid"):
                current = getattr(pyramids[1], name)
                setattr(pyramids[1], name, torch.nn.functional.interpolate(
                    target_region[None], size=current.shape[-2:], mode="nearest"
                )[0])
        target_detail = backbone.features.samples[0][1].detail
        target_detail.retain_grad()
        regions = reference_region[None].repeat(2, 1, 1, 1)
        mgrr = MultiGranularityRegionReplay(32)
        state = mgrr(backbone, regions, protocol="vision_only")
        self.assertEqual(tuple(state.region_tokens.shape), (2, 1, 32))
        self.assertTrue(bool(
            state.diagnostics["view_transform_retargeted"][:, :, 1].all()
        ))
        self.assertFalse(bool(
            state.diagnostics["view_transform_retargeted"][:, :, 0].any()
        ))
        self.assertTrue(bool(
            (state.diagnostics["modality_coverage"][:, :, 1] > 0).all()
        ))
        self.assertTrue(bool(
            (state.diagnostics["component_count"] == 2).all()
        ))
        selected = state.region_sequence_tokens[state.region_sequence_mask]
        probe = torch.linspace(0.1, 1.0, selected.shape[-1])
        (selected * probe).sum().backward()
        self.assertIsNotNone(target_detail.grad)
        self.assertGreater(float(target_detail.grad.abs().sum()), 0.0)

        for mode in ("crop_only", "masked_pooling", "full_image_box"):
            baseline = SingleVectorRegionPooling(32, mode)(backbone, regions)
            self.assertEqual(tuple(baseline.region_tokens.shape), (2, 1, 32))
            self.assertTrue(bool(
                baseline.diagnostics["view_transform_retargeted"][:, :, 1].all()
            ))

    def test_mgrr_reports_component_truncation_without_residual_union_roi(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 64)], size=64)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        region = torch.zeros((1, 1, 64, 64))
        for offset in range(6):
            y = 2 + offset * 9
            region[:, :, y:y + 3, 4 + offset * 8:7 + offset * 8] = 1
        mgrr = MultiGranularityRegionReplay(
            32, max_components=2, component_coverage=0.99
        )
        state = mgrr(backbone, region, protocol="vision_only")
        self.assertEqual(float(state.diagnostics["component_count"][0, 0]), 6.0)
        self.assertEqual(float(state.diagnostics["selected_component_count"][0, 0]), 2.0)
        self.assertGreater(float(state.diagnostics["residual_area_ratio"][0, 0]), 0.0)
        self.assertLess(float(state.diagnostics["component_coverage"][0, 0]), 1.0)

    def test_mgrr_component_inventory_is_shared_across_modalities(self) -> None:
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], size=32)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        region = torch.zeros((1, 1, 32, 32))
        region[:, :, 3:10, 4:12] = 1
        original = mgrr_module._component_masks
        with patch.object(mgrr_module, "_component_masks", wraps=original) as mocked:
            MultiGranularityRegionReplay(32)(backbone, region)
        self.assertEqual(mocked.call_count, 1)

    def test_mgrr_null_region_uses_null_evidence(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        mgrr = MultiGranularityRegionReplay(32)
        state = mgrr(backbone, torch.zeros((1, 1, 32, 32)), protocol="vision_only")
        self.assertAlmostEqual(
            float(state.diagnostics["null_reliability"][0, 0].detach()),
            1.0,
            places=5,
        )
        self.assertEqual(float(state.geometry_tokens.abs().sum()), 0.0)

    def test_mgrr_zero_valid_coverage_is_replaced_by_null_evidence(self) -> None:
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], size=32)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        valid_detail = backbone.features.samples[0][0].detail
        invalid = backbone.features.samples[0][1]
        valid_detail.retain_grad()
        invalid.detail.retain_grad()
        invalid.detail_valid.zero_()
        invalid.high_valid.zero_()
        invalid.mid_valid.zero_()
        invalid.low_valid.zero_()
        region = torch.zeros((1, 1, 32, 32))
        region[:, :, 5:20, 7:22] = 1
        mgrr = MultiGranularityRegionReplay(32)
        state = mgrr(backbone, region, protocol="vision_only")
        coverage = state.diagnostics["modality_coverage"][0, 0]
        reliability = state.diagnostics["modality_reliability"][0, 0]
        self.assertGreater(float(coverage[0]), 0.0)
        self.assertEqual(float(coverage[1]), 0.0)
        self.assertEqual(float(reliability[1].detach()), 0.0)
        self.assertTrue(torch.equal(
            state.diagnostics["modality_coverage_active"][0, 0],
            torch.tensor([True, False]),
        ))
        self.assertTrue(torch.allclose(
            state.modality_tokens[0, 0, 1], mgrr.null_evidence
        ))
        selected = state.region_sequence_tokens[state.region_sequence_mask]
        probe = torch.linspace(0.1, 1.0, selected.shape[-1])
        (selected * probe).sum().backward()
        self.assertIsNotNone(valid_detail.grad)
        self.assertGreater(float(valid_detail.grad.abs().sum()), 0.0)
        self.assertTrue(
            invalid.detail.grad is None
            or float(invalid.detail.grad.abs().sum()) == 0.0
        )

    def test_region_encoder_ablation_modes_share_output_contract(self) -> None:
        region = torch.zeros((1, 1, 32, 32))
        region[:, :, 5:20, 7:22] = 1
        encoder_factories = [
            lambda: SingleVectorRegionPooling(32, "crop_only"),
            lambda: SingleVectorRegionPooling(32, "masked_pooling"),
            lambda: SingleVectorRegionPooling(32, "full_image_box"),
            lambda: MultiGranularityRegionReplay(32, ablation="roi_replay_only"),
            lambda: MultiGranularityRegionReplay(32, ablation="no_context"),
            lambda: MultiGranularityRegionReplay(32, ablation="full"),
        ]
        for factory in encoder_factories:
            self.model.zero_grad(set_to_none=True)
            backbone = self.model.encode_multisource(
                synthetic_batch([
                    instance("optical_rgb", "optical", 3, 32),
                    instance("dem", "terrain", 1, 32),
                ], size=32),
                include_visual_tokens=False,
            )
            detail = backbone.features.samples[0][0].detail
            detail.retain_grad()
            encoder = factory()
            state = encoder(backbone, region, protocol="vision_only")
            self.assertEqual(tuple(state.region_tokens.shape), (1, 1, 32))
            self.assertEqual(tuple(state.region_sequence_mask.shape[:2]), (1, 1))
            self.assertTrue(bool(torch.isfinite(state.region_sequence_tokens).all()))
            selected = state.region_sequence_tokens[state.region_sequence_mask]
            probe = torch.linspace(0.1, 1.0, selected.shape[-1])
            (selected * probe).sum().backward()
            self.assertIsNotNone(detail.grad)
            self.assertGreater(float(detail.grad.abs().sum()), 0.0)
            self.assertTrue(any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in encoder.parameters()
            ))

    def test_region_encoder_protocols_do_not_leak_assisted_geometry(self) -> None:
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], size=32)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        region = torch.zeros((1, 1, 32, 32))
        region[:, :, 5:20, 7:22] = 1

        mgrr = MultiGranularityRegionReplay(32)
        vision_mgrr = mgrr(backbone, region, protocol="vision_only")
        assisted_mgrr = mgrr(backbone, region, protocol="assisted")
        self.assertEqual(float(vision_mgrr.geometry_tokens.abs().sum()), 0.0)
        self.assertGreater(
            float(assisted_mgrr.geometry_tokens.detach().abs().sum()), 0.0
        )
        self.assertEqual(float(vision_mgrr.diagnostics["protocol_assisted"]), 0.0)
        self.assertEqual(float(assisted_mgrr.diagnostics["protocol_assisted"]), 1.0)

        full_image_box = SingleVectorRegionPooling(32, "full_image_box")
        vision_box = full_image_box(backbone, region, protocol="vision_only")
        assisted_box = full_image_box(backbone, region, protocol="assisted")
        vision_values = vision_box.diagnostics["geometry_input_values"][0, 0]
        assisted_values = assisted_box.diagnostics["geometry_input_values"][0, 0]
        self.assertEqual(float(vision_values[0]), 0.0)
        self.assertGreater(float(vision_values[1:5].abs().sum()), 0.0)
        self.assertEqual(float(vision_values[5:9].abs().sum()), 0.0)
        self.assertEqual(float(vision_values[9]), 1.0)
        self.assertGreater(float(assisted_values[[0, 5, 6, 7, 8]].abs().sum()), 0.0)
        self.assertFalse(torch.equal(vision_box.geometry_tokens, assisted_box.geometry_tokens))
        self.assertEqual(float(vision_box.diagnostics["protocol_assisted"]), 0.0)
        self.assertEqual(float(assisted_box.diagnostics["protocol_assisted"]), 1.0)

        for mode in ("crop_only", "masked_pooling"):
            baseline = SingleVectorRegionPooling(32, mode)
            vision_state = baseline(backbone, region, protocol="vision_only")
            assisted_state = baseline(backbone, region, protocol="assisted")
            self.assertEqual(float(vision_state.geometry_tokens.abs().sum()), 0.0)
            self.assertGreater(
                float(assisted_state.geometry_tokens.detach().abs().sum()), 0.0
            )
            self.assertFalse(torch.equal(
                vision_state.region_tokens, assisted_state.region_tokens
            ))
            self.assertEqual(
                float(vision_state.diagnostics["protocol_assisted"]), 0.0
            )
            self.assertEqual(
                float(assisted_state.diagnostics["protocol_assisted"]), 1.0
            )

    def test_full_image_box_keeps_visual_evidence_for_null_region(self) -> None:
        backbone = self.model.encode_multisource(
            synthetic_batch([
                instance("optical_rgb", "optical", 3, 32),
                instance("dem", "terrain", 1, 32),
            ], size=32),
            include_visual_tokens=False,
        )
        detail = backbone.features.samples[0][0].detail
        detail.retain_grad()
        regions = torch.zeros((1, 2, 32, 32))
        regions[:, 1, 5:20, 7:22] = 1
        encoder = SingleVectorRegionPooling(32, "full_image_box")
        state = encoder(backbone, regions, protocol="vision_only")
        self.assertEqual(
            state.diagnostics["region_present"].tolist(), [[False, True]]
        )
        self.assertTrue(bool(state.diagnostics["visual_evidence_active"].all()))
        self.assertEqual(
            float(state.diagnostics["full_image_visual_for_null_region"]), 1.0
        )
        self.assertTrue(torch.allclose(
            state.modality_tokens[:, 0], state.modality_tokens[:, 1]
        ))
        self.assertEqual(
            float(state.diagnostics["geometry_input_values"][0, 0].abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(state.diagnostics["geometry_input_values"][0, 1].abs().sum()),
            0.0,
        )
        probe = torch.linspace(
            0.1, 1.0, state.region_sequence_tokens.shape[-1]
        )
        (state.region_sequence_tokens[0, 0] * probe).sum().backward()
        self.assertIsNotNone(detail.grad)
        self.assertGreater(float(detail.grad.abs().sum()), 0.0)

    def test_single_modality_removal_removes_dense_and_visual_evidence(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        backbone = self.model.encode_multisource(batch, include_visual_tokens=False)
        removed = counterfactual_backbone(backbone, "modality_removal")
        self.assertEqual(removed.features.samples[0], [])
        self.assertEqual(removed.active_subsets[0].active_names, ())
        mgrr = MultiGranularityRegionReplay(32)
        region = torch.ones((1, 1, 32, 32))
        state = mgrr(removed, region)
        self.assertAlmostEqual(float(state.diagnostics["null_reliability"][0, 0]), 1.0)

    def test_cross_parent_modality_swap_never_uses_same_parent_task_view(self) -> None:
        first = synthetic_batch([instance("optical_a", "optical", 3, 32)], size=32)
        second = synthetic_batch([instance("optical_b", "optical", 3, 32)], size=32)
        third = synthetic_batch([instance("optical_c", "optical", 3, 32)], size=32)
        states = [
            self.model.encode_multisource(batch, include_visual_tokens=False)
            for batch in (first, second, third)
        ]
        backbone = MultisourceBackboneState(
            features=MultiScaleFeatures(
                samples=[state.features.samples[0] for state in states],
                reference_hw=states[0].reference_hw,
            ),
            valid_mask=torch.cat([state.valid_mask for state in states]),
            active_subsets=tuple(state.active_subsets[0] for state in states),
            metadata=(
                {"parent_sample_id": "same"},
                {"parent_sample_id": "same"},
                {"parent_sample_id": "different"},
            ),
            reference_hw=states[0].reference_hw,
            use_full_evidence=True,
            visual_evidence=None,
        )
        swapped = counterfactual_backbone(backbone, "cross_parent_modality_swap")
        self.assertEqual(
            swapped.metadata[0]["counterfactual_modality_swap"]["donor_parent_sample_id"],
            "different",
        )
        self.assertEqual(
            swapped.metadata[1]["counterfactual_modality_swap"]["donor_parent_sample_id"],
            "different",
        )
        self.assertEqual(
            swapped.metadata[2]["counterfactual_modality_swap"]["donor_parent_sample_id"],
            "same",
        )

    def test_region_geometry_unifies_full_box_mask_and_null(self) -> None:
        valid = torch.ones((1, 10, 20))
        full = rasterize_region_geometry({"type": "full_image"}, valid)
        box = rasterize_region_geometry({
            "type": "box", "bbox_xyxy_pixel_half_open": [2, 3, 8, 7]
        }, valid)
        explicit = torch.zeros((1, 5, 10))
        explicit[:, 1:3, 2:5] = 1
        mask = rasterize_region_geometry({"type": "mask"}, valid, explicit_mask=explicit)
        null = rasterize_region_geometry({"type": "null"}, valid)
        self.assertEqual(int(full.sum()), 200)
        self.assertEqual(int(box.sum()), 24)
        self.assertGreater(int(mask.sum()), 0)
        self.assertEqual(int(null.sum()), 0)

    def test_torch_pin_memory_preserves_typed_modality_batch(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        with patch.object(ModalityBatch, "pin_memory", autospec=True, return_value=batch) as mocked:
            pinned = torch_pin_memory(batch)
        self.assertIs(pinned, batch)
        mocked.assert_called_once_with(batch)

    def test_switching_preset_replaces_algorithm_controls(self) -> None:
        qwen_config = replace(QPSalmConfig(), preset="qwen_psalm_full", controller="qwen_mask_query")
        baseline = apply_preset(qwen_config, "raw_sane_baseline")
        self.assertEqual(baseline.controller, "text_probe")
        self.assertEqual(baseline.num_mask_tokens, 1)
        self.assertFalse(baseline.use_qmef)
        self.assertEqual(baseline.size_buckets, [64, 128, 256, 384])
        qwen = apply_preset(QPSalmConfig(), "qwen_psalm_full")
        self.assertEqual(qwen.size_buckets, [64, 128, 256])
        self.assertEqual(qwen.max_native_size, 256)
        self.assertEqual(qwen.qwen_gradient_checkpointing, "disabled")
        frozen = apply_preset(QPSalmConfig(), "qwen_mask_query_frozen")
        self.assertEqual(frozen.controller, "qwen_mask_query")
        self.assertFalse(frozen.qwen_lora_trainable)

    def test_modality_batch_select_preserves_order_and_payload(self) -> None:
        batch = repeat_batch(
            synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32),
            3,
        )
        selected = batch.select([2, 0])
        self.assertEqual(selected.batch_size, 2)
        self.assertEqual([row["parent_sample_id"] for row in selected.metadata], ["parent-2", "parent-0"])
        self.assertEqual(selected.visual_evidence_key, ["qmv3-parent:synthetic-2", "qmv3-parent:synthetic-0"])

    def test_student_graph_is_built_before_consistency_teacher(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 32)
        terrain = instance("dem", "terrain", 1, 32)
        batch = synthetic_batch([optical, terrain], components=1, size=32)
        batch.instances[0] = [optical]
        batch.active_subsets[0] = ActiveModalitySubset(
            active_names=(optical.name,),
            dropped_names=(terrain.name,),
            signature="synthetic-dropped",
            is_full=False,
        )
        model = MultiSourceQwenPSALMSeg(
            replace(
                self.config,
                max_native_size=32,
                missing_modality_consistency_weight=0.1,
            ),
            torch.device("cpu"),
        ).train()
        calls = []
        original = model.controller.encode_batch

        def record(*args, **kwargs):
            calls.append(bool(kwargs.get("use_full", False)))
            return original(*args, **kwargs)

        with patch.object(model.controller, "encode_batch", side_effect=record):
            output = model(batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(calls, [False, True])

    def test_sane_baseline_uses_uniform_real_evidence_without_null_gate(self) -> None:
        baseline_config = replace(
            apply_preset(QPSalmConfig(), "raw_sane_baseline"),
            decoder_dim=32, num_heads=4, num_decoder_layers=1, max_native_size=32,
        )
        baseline = MultiSourceQwenPSALMSeg(baseline_config, torch.device("cpu")).eval()
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], components=1, size=32)
        output = baseline(batch)
        self.assertTrue(torch.allclose(output["modality_reliability_weights"], torch.full((1, 2), 0.5)))
        self.assertEqual(float(output["null_evidence_weight"][0]), 0.0)

    def test_task_sampler_uses_auditable_40_40_20_epoch_quota(self) -> None:
        class Dataset:
            rows = (
                [{"task_family": "global_landslide_segmentation"}] * 40
                + [{"task_family": "referring_landslide_segmentation"}] * 40
                + [{"task_family": "no_target_segmentation"}] * 20
            )

            def __len__(self):
                return len(self.rows)

            def bucket_size(self, _index):
                return 64

        dataset = Dataset()
        sampler = TaskBalancedSizeBucketBatchSampler(dataset, 1, shuffle=False, seed=42)
        groups = [task_group(dataset.rows[batch[0]]) for batch in sampler]
        counts = __import__("collections").Counter(groups)
        self.assertEqual(counts, {"global": 40, "referring": 40, "no_target": 20})

    def test_task_sampler_keeps_global_quota_across_disjoint_size_buckets(self) -> None:
        class Dataset:
            rows = (
                [{"task_family": "global_landslide_segmentation", "bucket": 64}] * 90
                + [{"task_family": "referring_landslide_segmentation", "bucket": 128}] * 5
                + [{"task_family": "no_target_segmentation", "bucket": 256}] * 5
            )

            def __len__(self):
                return len(self.rows)

            def bucket_size(self, index):
                return self.rows[index]["bucket"]

        dataset = Dataset()
        sampler = TaskBalancedSizeBucketBatchSampler(dataset, 2, shuffle=False, seed=42)
        batches = list(sampler)
        self.assertEqual(len(batches), 50)
        self.assertTrue(all(len({dataset.bucket_size(index) for index in batch}) == 1 for batch in batches))
        counts = __import__("collections").Counter(
            task_group(dataset.rows[index]) for batch in batches for index in batch
        )
        self.assertEqual(counts, {"global": 40, "referring": 40, "no_target": 20})

    def test_sampler_groups_qwen_load_and_avoids_parent_duplicates(self) -> None:
        class Dataset:
            rows = [
                {
                    "sample_id": f"sample-{index}",
                    "parent_sample_id": f"parent-{index}",
                    "task_family": "global_landslide_segmentation",
                    "load": 128 if index < 3 else 256,
                }
                for index in range(6)
            ]

            def __len__(self):
                return len(self.rows)

            def bucket_size(self, _index):
                return 64

            def sequence_load_bucket(self, index):
                return self.rows[index]["load"]

        dataset = Dataset()
        sampler = TaskBalancedSizeBucketBatchSampler(
            dataset, 3, shuffle=False, seed=42, balance_tasks=False
        )
        for batch in sampler:
            self.assertEqual(len({dataset.rows[index]["load"] for index in batch}), 1)
            self.assertEqual(len({dataset.rows[index]["parent_sample_id"] for index in batch}), len(batch))

    def test_integration_representative_batch_matches_training_stratum(self) -> None:
        class Dataset:
            rows = [
                {
                    "sample_id": f"sample-{index}",
                    "parent_sample_id": f"parent-{index}",
                    "task_family": "global_landslide_segmentation",
                    "load": 224,
                    "mask": {"empty_mask": False},
                    "modalities": {
                        "s2": {"available": True, "family": "multispectral"},
                        "s1": {"available": True, "family": "sar"},
                        "dem": {"available": True, "family": "terrain"},
                    },
                }
                for index in range(8)
            ]

            def bucket_size(self, _index):
                return 256

            def sequence_load_bucket(self, index):
                return self.rows[index]["load"]

        dataset = Dataset()
        selected = select_representative_batch_indices(dataset, 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({dataset.rows[index]["parent_sample_id"] for index in selected}), 6)
        self.assertEqual(len({dataset.bucket_size(index) for index in selected}), 1)
        self.assertEqual(len({dataset.sequence_load_bucket(index) for index in selected}), 1)
        self.assertEqual(len({task_group(dataset.rows[index]) for index in selected}), 1)

    def test_lora_update_summary_detects_optimizer_change(self) -> None:
        class LoRAModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.ones(2, 2))

        module = LoRAModule()
        wrapper = SimpleNamespace(controller=SimpleNamespace(model=module))
        before = snapshot_lora_parameters(wrapper)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        module.lora_A.square().sum().backward()
        optimizer.step()
        update = lora_parameter_update_summary(wrapper, before)
        self.assertEqual(update["num_parameters"], 1)
        self.assertEqual(update["num_changed"], 1)
        self.assertGreater(update["norm_sum"], 0.0)
        self.assertTrue(update["all_finite"])

    def test_two_stage_qwen_optimizer_schedule(self) -> None:
        class FakeController(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.lora_A = torch.nn.Parameter(torch.ones(2, 2))
                self.model.lora_B = torch.nn.Parameter(torch.zeros(2, 2))
                self.mask_embeddings = torch.nn.Parameter(torch.ones(2, 2))
                self.output_projection = torch.nn.Linear(2, 2)

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.controller = FakeController()
                self.sane = torch.nn.Linear(2, 2)

        model = FakeModel()
        config = QPSalmConfig(
            controller="qwen_mask_query",
            qwen_lora_start_step=2,
            qwen_lora_lr_scale=0.2,
            controller_lr_scale=0.5,
            lr=1.0e-4,
        )
        optimizer, _ = build_optimizer(model, config)
        stage = apply_optimizer_schedule(model, optimizer, config, step=0, lr_multiplier=1.0)
        self.assertEqual(stage, "decoder_warmup")
        self.assertFalse(model.controller.model.lora_A.requires_grad)
        lora_group = next(group for group in optimizer.param_groups if group["group_role"] == "qwen_lora")
        self.assertEqual(float(lora_group["lr"]), 0.0)
        stage = apply_optimizer_schedule(model, optimizer, config, step=2, lr_multiplier=1.0)
        self.assertEqual(stage, "qlora_active")
        self.assertTrue(model.controller.model.lora_A.requires_grad)
        self.assertAlmostEqual(float(lora_group["lr"]), 2.0e-5)
        self.assertEqual(qwen_training_stage(replace(config, qwen_lora_trainable=False), 100), "qwen_frozen")

    def test_trainer_lora_update_diagnostics_require_real_parameter_change(self) -> None:
        class FakeController(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.lora_A = torch.nn.Parameter(torch.ones(2, 2))
                self.model.lora_B = torch.nn.Parameter(torch.ones(2, 2))

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.controller = FakeController()

        model = FakeModel()
        before = snapshot_qwen_lora(model)
        loss = sum(parameter.square().sum() for parameter in model.parameters())
        loss.backward()
        gradients = qwen_lora_gradient_summary(model)
        self.assertEqual(gradients["num_nonzero"], 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        update = qwen_lora_update_summary(model, before)
        self.assertEqual(update["num_changed"], 2)
        self.assertGreater(update["norm_sum"], 0.0)

    def test_lora_gradient_report_separates_a_and_b(self) -> None:
        class Adapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})

        adapter = Adapter()
        adapter.lora_A["default"].weight.grad = torch.ones_like(adapter.lora_A["default"].weight)
        adapter.lora_B["default"].weight.grad = torch.ones_like(adapter.lora_B["default"].weight) * 2
        wrapper = SimpleNamespace(controller=SimpleNamespace(model=adapter))
        report = lora_gradient_report(wrapper)
        self.assertEqual(report["by_matrix"]["lora_A"]["num_nonzero"], 1)
        self.assertEqual(report["by_matrix"]["lora_B"]["num_nonzero"], 1)
        self.assertTrue(report["all_finite"])

    def test_trainability_report_records_runtime_versions(self) -> None:
        report = runtime_library_report(torch.device("cpu"))
        self.assertIn("torch", report)
        self.assertIn("transformers", report)
        self.assertNotIn("device_name", report)

    def test_active_subset_controls_prompt_and_null_reliability(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 64)
        terrain = instance("dem", "terrain", 1, 64)
        row = {
            "instruction": {"text": "Segment all landslide regions."},
            "task_family": "global_landslide_segmentation",
            "spatial": {"gsd_m": 1.0},
        }
        proposal, _, reasoning = build_prompt_triplet(row, [optical], subset_signature="optical")
        self.assertIn("optical", proposal)
        self.assertNotIn("terrain", proposal)
        self.assertNotIn("terrain", reasoning)
        output = self.model(synthetic_batch([optical, terrain]))
        total = output["modality_reliability_weights"].sum(1) + output["null_evidence_weight"]
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1.0e-5))
        self.assertEqual(tuple(output["query_embeddings"].shape), (1, 4, 32))

    def test_qmef_maps_family_specific_qwen_anchors_to_modalities(self) -> None:
        batch = synthetic_batch([
            instance("optical_rgb", "optical", 3, 32),
            instance("dem", "terrain", 1, 32),
        ], size=32)
        semantic = self.model.controller.encode_batch(batch)
        pyramids = self.model.sane(batch)
        evidence = self.model.qmef(pyramids, semantic)
        self.assertTrue(torch.allclose(
            evidence.modality_semantic_anchors[0, 0], semantic.evidence_anchors[0, 1]
        ))
        self.assertTrue(torch.allclose(
            evidence.modality_semantic_anchors[0, 1], semantic.evidence_anchors[0, 4]
        ))

    def test_inactive_modality_is_absent_from_student_model_channels(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 32)
        terrain = instance("dem", "terrain", 1, 32)
        batch = synthetic_batch([optical], components=1, size=32)
        batch.full_instances = [[optical, terrain]]
        batch.active_subsets = [ActiveModalitySubset(
            active_names=("optical_rgb",), dropped_names=("dem",),
            signature="optical-only", is_full=False,
        )]
        batch.full_proposal_context_text = ["Use optical and terrain evidence."]
        batch.full_evidence_reasoning_text = ["Terrain evidence is available to the teacher."]
        self.assertEqual(float(batch.availability[0, 3]), 0.0)
        student_semantic = self.model.controller.encode_batch(batch, use_full=False)
        teacher_semantic = self.model.controller.encode_batch(batch, use_full=True)
        self.assertEqual(tuple(student_semantic.evidence_anchors.shape), (1, 6, 32))
        self.assertFalse(torch.allclose(
            student_semantic.evidence_anchors[:, 4], teacher_semantic.evidence_anchors[:, 4]
        ))
        output = self.model(batch)
        self.assertEqual(tuple(output["modality_reliability_weights"].shape), (1, 1))
        self.assertEqual(tuple(output["modality_active"].shape), (1, 1))

    def test_valid_weighted_pool_excludes_nodata_and_reports_coverage(self) -> None:
        feature = torch.ones((2, 4, 4))
        feature[:, 2:] = 1000.0
        valid = torch.zeros((1, 4, 4))
        valid[:, :2] = 1.0
        pooled, coverage = valid_weighted_pool(feature, valid)
        self.assertTrue(torch.allclose(pooled, torch.ones_like(pooled)))
        self.assertAlmostEqual(float(coverage), 0.5)

    def test_sane_valid_pyramid_preserves_tiny_valid_regions(self) -> None:
        valid = torch.zeros((1, 17, 17))
        valid[:, 1, 1] = 1
        pooled = SensorAwareNativeScaleEncoder._valid_at_scale(valid[None], (2, 2))
        self.assertGreater(float(pooled.sum()), 0.0)

    def test_sane_masks_nodata_before_band_convolution(self) -> None:
        encoder = SensorAwareNativeScaleEncoder(16).eval()
        first = instance("optical_rgb", "optical", 3, 32)
        valid = torch.zeros((1, 32, 32))
        valid[:, :, :16] = 1.0
        first.valid_mask = valid
        second = instance("optical_rgb", "optical", 3, 32)
        second.valid_mask = valid.clone()
        second.image = first.image.clone()
        first.image[:, :, 16:] = 1000.0
        second.image[:, :, 16:] = -1000.0
        first_features = encoder._encode_instance(first, torch.device("cpu"))
        second_features = encoder._encode_instance(second, torch.device("cpu"))
        for name in ("detail", "high", "mid", "low"):
            self.assertTrue(torch.allclose(
                getattr(first_features, name), getattr(second_features, name), atol=1.0e-6
            ))

    def test_pretrained_sane_has_four_near_zero_raw_residuals(self) -> None:
        fake_bank = SimpleNamespace(spatial_channels=48)
        encoder = SensorAwareNativeScaleEncoder(32, pretrained_bank=fake_bank)
        self.assertEqual(tuple(encoder.raw_residual_scale.shape), (4,))
        self.assertLess(float(torch.sigmoid(encoder.raw_residual_scale).max().detach()), 0.02)
        self.assertEqual(encoder.pretrained_adapters[0][0].in_channels, 48)

    def test_qmef_uses_four_query_points_per_scale(self) -> None:
        self.assertFalse(hasattr(self.model.pmrd, "query_position"))
        self.assertEqual(self.model.qmef.num_points, 4)
        self.assertEqual(tuple(self.model.qmef.point_pattern.shape), (4, 2))
        self.assertTrue(torch.equal(self.model.qmef.point_pattern[0], torch.zeros(2)))
        self.assertEqual(self.model.qmef.query_offsets.out_features, 3 * 4 * 2)
        output = self.model(synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32))
        self.assertEqual(tuple(output["query_sampling_grid"].shape), (1, 4, 3, 4, 2))
        self.assertEqual(tuple(output["query_scale_attention"].shape), (1, 4, 3))
        self.assertTrue(torch.allclose(output["query_scale_attention"].sum(-1), torch.ones((1, 4))))

    def test_qmef_joint_attention_can_select_scale(self) -> None:
        fusion = QwenGuidedEvidenceFusion(1, deformable_points=1)
        with torch.no_grad():
            fusion.query_proj.weight.fill_(1.0)
            fusion.query_proj.bias.zero_()
            fusion.family_anchor_proj.weight.zero_()
            fusion.family_anchor_proj.bias.zero_()
            fusion.scale_embedding.zero_()
            fusion.query_offsets.weight.zero_()
            fusion.query_offsets.bias.zero_()
            fusion.query_point_bias.weight.zero_()
            fusion.query_point_bias.bias.zero_()
            for scale in ("high", "mid", "low"):
                fusion.key_proj[scale].weight.fill_(1.0)
                fusion.key_proj[scale].bias.zero_()
                fusion.value_proj[scale].weight.fill_(1.0)
                fusion.value_proj[scale].bias.zero_()
        evidence = SimpleNamespace(
            modality_high=torch.full((1, 1, 1, 4, 4), 10.0),
            modality_mid=torch.zeros((1, 1, 1, 2, 2)),
            modality_low=torch.zeros((1, 1, 1, 1, 1)),
            modality_valid_high=torch.ones((1, 1, 1, 4, 4)),
            modality_valid_mid=torch.ones((1, 1, 1, 2, 2)),
            modality_valid_low=torch.ones((1, 1, 1, 1, 1)),
            modality_semantic_anchors=torch.zeros((1, 1, 1)),
            reliability_weights=torch.ones((1, 1)),
            real_reliability_mass=torch.ones(1),
        )
        context, modality_mass, scale_mass, _, _, _ = fusion.attend_queries(
            torch.ones((1, 1, 1)), evidence, torch.zeros((1, 1, 8, 8))
        )
        self.assertGreater(float(scale_mass[0, 0, 0].detach()), 0.99)
        self.assertGreater(float(context[0, 0, 0].detach()), 9.9)
        self.assertTrue(torch.allclose(modality_mass, torch.ones_like(modality_mass), atol=1.0e-5))

    def test_qmef_null_evidence_suppresses_query_context(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        semantic = self.model.controller.encode_batch(batch)
        evidence = self.model.qmef(self.model.sane(batch), semantic)
        queries, coarse = self.model.pmrd.propose(evidence, semantic, batch.reference_hw)
        evidence.real_reliability_mass.zero_()
        context, _, _, _, _, _ = self.model.qmef.attend_queries(queries, evidence, coarse)
        self.assertEqual(float(context.abs().sum().detach()), 0.0)

    def test_qmef_film_cannot_recreate_invalid_pixels(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 32)
        optical.valid_mask[:, :16] = 0.0
        batch = synthetic_batch([optical], size=32)
        semantic = self.model.controller.encode_batch(batch)
        film_linear = self.model.qmef.semantic_film[-1]
        torch.nn.init.zeros_(film_linear.weight)
        with torch.no_grad():
            film_linear.bias[: self.config.decoder_dim].zero_()
            film_linear.bias[self.config.decoder_dim :].fill_(10.0)
        evidence = self.model.qmef(self.model.sane(batch), semantic)
        invalid = evidence.modality_valid_high[0, 0, 0] <= 1.0e-4
        self.assertTrue(bool(invalid.any()))
        self.assertEqual(float(evidence.modality_high[0, 0, :, invalid].detach().abs().max()), 0.0)

    def test_qmef_geometric_validity_is_independent_of_null_reliability(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        semantic = self.model.controller.encode_batch(batch)
        null_linear = self.model.qmef.null_head[-1]
        torch.nn.init.zeros_(null_linear.weight)
        torch.nn.init.constant_(null_linear.bias, 40.0)
        evidence = self.model.qmef(self.model.sane(batch), semantic)
        self.assertGreater(float(evidence.null_reliability[0].detach()), 0.999)
        self.assertLess(float(evidence.real_reliability_mass[0].detach()), 1.0e-6)
        self.assertGreater(float(evidence.fused_high_valid.detach().max()), 0.99)

    def test_proposal_assignment_reports_set_metrics(self) -> None:
        logits = torch.full((4, 16, 16), -8.0)
        logits[0, 2:6, 2:6] = 8.0
        logits[1, 9:13, 9:13] = 8.0
        target = torch.zeros((16, 16))
        target[2:6, 2:6] = 1.0
        target[9:13, 9:13] = 1.0
        assignment = assign_proposals(logits, torch.tensor([5.0, 5.0, -5.0, -5.0]), target, torch.ones_like(target))
        self.assertAlmostEqual(float(assignment.component_recall), 1.0, places=5)
        self.assertAlmostEqual(float(assignment.component_precision), 1.0, places=5)
        self.assertGreater(float(assignment.proposal_union_dice), 0.95)

    def test_verifier_positive_weight_tracks_query_imbalance(self) -> None:
        output = self.model(synthetic_batch([
            instance("optical_rgb", "optical", 3, 32)
        ], components=1, size=32))
        self.assertEqual(float(output["proposal_target_positive_count"][0]), 1.0)
        self.assertEqual(float(output["proposal_verifier_pos_weight"][0]), 3.0)
        self.assertGreaterEqual(int(output["proposal_oracle_matched_query"][0]), 0)
        self.assertTrue(torch.isfinite(output["proposal_oracle_matched_dice"]).all())

    def test_empty_target_has_no_oracle_query(self) -> None:
        output = self.model(synthetic_batch([
            instance("optical_rgb", "optical", 3, 32)
        ], components=0, size=32))
        self.assertEqual(int(output["proposal_oracle_matched_query"][0]), -1)
        self.assertEqual(float(output["proposal_oracle_matched_dice"][0]), 0.0)

    def test_visualization_exports_selected_and_gt_oracle_proposals(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], components=1, size=32)
        output = self.model(batch)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_visualizations(
                batch, output, Path(directory), max_items=1, prefix="unit", threshold=0.5
            )
            self.assertEqual(len(paths), 1)
            manifest_path = Path(directory) / "visualization_manifest.jsonl"
            record = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(record["oracle_is_gt_diagnostic_only"])
            self.assertGreaterEqual(record["oracle_matched_query"], 0)
            self.assertIn("selected_proposal", record["mask_paths"])
            self.assertIn("oracle_matched_proposal", record["mask_paths"])
            self.assertTrue(Path(record["mask_paths"]["oracle_matched_proposal"]).exists())

    def test_vision_cache_obeys_active_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = []
            for name in ("optical_rgb", "dem"):
                views.append({
                    "name": name,
                    "description": f"Physical view for {name}.",
                    "source_modalities": [name],
                    "source_families": ["optical" if name == "optical_rgb" else "terrain"],
                    "content_hash": name,
                    "quality_flags": [],
                    "spatial_features": torch.ones((4, 1024, 2, 2)),
                    "view_tokens": torch.ones((3, 2048)),
                    "valid_mask": torch.ones((1, 2, 2)),
                    "render_transform": {"size": 2, "pad_top": 0, "pad_left": 0, "resized_h": 2, "resized_w": 2},
                    "vision_grid_thw": [1, 2, 2],
                    "merged_grid_hw": [1, 3],
                })
            full_names = sorted(view["source_modalities"][0] for view in views)
            full_signature = "subset:" + "+".join(full_names) + ":" + __import__("hashlib").sha256(
                "\n".join(full_names).encode()
            ).hexdigest()[:12]
            torch.save({
                "format": CACHE_FORMAT,
                "records": [{"lookup_key": "parent", "full_subset_signature": full_signature, "views": views}],
            }, root / "shard_00000.pt")
            manifest = {
                "format": CACHE_FORMAT, "renderer_version": RENDERER_VERSION, "model_revision": "test",
                "processor_revision": "test", "prompt_version": PROMPT_VERSION,
                "pooling_method": "spatial_layers_plus_adaptive_view_tokens",
                "layers": [5, 11, 17, 23], "spatial_channels": 1024,
                "token_dim": 2048, "spatial_sizes": [2, 2, 2, 2],
                "render_size": 2, "view_tokens_per_view": 3,
                "subset_policy": "dynamic_by_source_modality",
                "input_protocol": {
                    "preset": "test", "use_size_buckets": False, "size_buckets": [],
                    "target_size": 32, "max_native_size": 32,
                    "index_fingerprints": {
                        split: {
                            "reference": f"instruction_{split}.jsonl", "status": "present",
                            "size": 1, "sha256": "0" * 64,
                        }
                        for split in ("train", "val", "test")
                    },
                },
                "backend": "hash-smoke",
                "num_samples": 1, "shard_size": 1, "peak_buffer_records": 1,
                "lookup": {"parent": {
                    "shard": 0, "index": 0,
                    "source_modalities": ["dem", "optical_rgb"],
                    "source_families": ["optical", "terrain"],
                    "modality_families": {"dem": "terrain", "optical_rgb": "optical"},
                }}, "shards": ["shard_00000.pt"],
            }
            fingerprint_payload = "|".join(
                [RENDERER_VERSION, "test", "test", PROMPT_VERSION, manifest["pooling_method"], full_signature]
                + sorted(view_fingerprint_fragment(view) for view in views)
            )
            records = torch.load(root / "shard_00000.pt", weights_only=False)
            records["records"][0]["cache_fingerprint"] = __import__("hashlib").sha256(
                fingerprint_payload.encode()
            ).hexdigest()
            torch.save(records, root / "shard_00000.pt")
            (root / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
            bank = QwenVisionFeatureBank(root, decoder_dim=32)
            subset = ActiveModalitySubset(("optical_rgb",), ("dem",), "optical-only", False)
            selected_views = bank.selected_views_for("parent", subset)
            self.assertEqual(
                {source for view in selected_views for source in view["source_modalities"]},
                {"optical_rgb"},
            )
            _, _, counts, family_ids, segments = bank.tokens_for(
                ["parent"], [subset], torch.device("cpu"), max_tokens_per_view=2
            )
            self.assertEqual(counts, [2])
            self.assertEqual(family_ids[0, :2].tolist(), [1, 1])
            self.assertEqual(segments, [[("Physical view for optical_rgb.", 2)]])
            optical = instance("optical_rgb", "optical", 3, 32)
            optical.metadata["vision_cache_key"] = "parent"
            normal_spatial = bank.features_for(optical, torch.device("cpu"))
            bank.set_visual_ablation("text-only")
            _, _, counts, _, _ = bank.tokens_for(
                ["parent"], [subset], torch.device("cpu"), max_tokens_per_view=2
            )
            self.assertEqual(counts, [0])
            text_only_spatial = bank.features_for(optical, torch.device("cpu"))
            self.assertTrue(torch.equal(normal_spatial[0], text_only_spatial[0]))
            bank.visual_ablation = "remove:optical"
            _, _, counts, _, _ = bank.tokens_for(
                ["parent"], [subset], torch.device("cpu"), max_tokens_per_view=2
            )
            self.assertEqual(counts, [0])
            removed_view_spatial = bank.features_for(optical, torch.device("cpu"))
            self.assertTrue(torch.equal(normal_spatial[0], removed_view_spatial[0]))

    def test_visual_shuffle_preserves_modality_semantics(self) -> None:
        bank = QwenVisionFeatureBank.__new__(QwenVisionFeatureBank)
        torch.nn.Module.__init__(bank)
        bank.lookup = {
            "a": {"source_modalities": ["dem", "multispectral"], "source_families": ["multispectral", "terrain"]},
            "b": {"source_modalities": ["slope", "s2"], "source_families": ["multispectral", "terrain"]},
            "c": {"source_modalities": ["dem", "multispectral"], "source_families": ["multispectral", "terrain"]},
        }
        self.assertEqual(bank._shuffled_key("a"), "c")
        del bank.lookup["c"]
        self.assertEqual(bank._shuffled_key("a"), "b")
        del bank.lookup["b"]
        with self.assertRaisesRegex(RuntimeError, "至少两个 parent"):
            bank._shuffled_key("a")

    def test_qwen_patch_tokens_restore_true_spatial_order(self) -> None:
        t, h, w, merge = 1, 4, 6, 2
        natural = torch.arange(t * h * w).view(t, h, w, 1).float()
        permuted = natural.view(t, h // merge, merge, w // merge, merge, 1)
        permuted = permuted.permute(0, 1, 3, 2, 4, 5).reshape(t * h * w, 1)
        restored = restore_qwen_patch_grid(permuted, (t, h, w), merge)
        self.assertTrue(torch.equal(restored[0], natural[0, :, :, 0]))

    def test_unknown_sensor_and_band_use_fallback_ids(self) -> None:
        unknown = instance("future_sensor", "multispectral", 2, 32, sensor="future_satellite")
        unknown.sensor = "future_satellite"
        unknown.band_names = ("X_RED_EDGE", "X_SWIR")
        output = self.model(synthetic_batch([unknown], components=1, size=32))
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_qwen_lora_selects_only_last_language_blocks(self) -> None:
        class Attention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(self, name, torch.nn.Linear(2, 2))
        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = Attention()
        class Fake(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList([Block() for _ in range(8)])
        fake = Fake()
        selected = QwenMaskQueryController._last_language_layer_indices(fake, 4)
        self.assertEqual(selected, (4, 5, 6, 7))

    def test_qwen_view_projection_preserves_pretrained_hidden_space(self) -> None:
        projection = torch.nn.Linear(8, 8)
        QwenMaskQueryController._initialize_view_projection(projection)
        self.assertTrue(torch.equal(projection.weight, torch.eye(8)))
        self.assertTrue(torch.equal(projection.bias, torch.zeros(8)))

    def test_qwen_view_pooling_ablation_contract(self) -> None:
        controller = QwenMaskQueryController.__new__(QwenMaskQueryController)
        torch.nn.Module.__init__(controller)
        controller.view_attention_query = torch.nn.Parameter(torch.ones(2))
        chunk = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        controller.view_pooling = "tokens"
        self.assertTrue(torch.equal(controller._pool_view_chunk(chunk), chunk))
        controller.view_pooling = "image-end"
        self.assertTrue(torch.equal(controller._pool_view_chunk(chunk), chunk[-1:]))
        controller.view_pooling = "attention"
        pooled = controller._pool_view_chunk(chunk)
        self.assertEqual(tuple(pooled.shape), (1, 2))
        self.assertGreater(float(pooled[0, 1].detach()), float(pooled[0, 0].detach()))

    def test_mask_query_states_are_updated_by_language_context(self) -> None:
        class Tokenizer:
            eos_token_id = 0

            def __call__(self, text, **_kwargs):
                ids = [1 + (ord(value) % 15) for value in text[:8]] or [0]
                return {"input_ids": torch.tensor([ids])}

        class Language(torch.nn.Module):
            def forward(self, inputs_embeds, **_kwargs):
                self.last_inputs = inputs_embeds.detach().clone()
                return SimpleNamespace(last_hidden_state=torch.cumsum(inputs_embeds, dim=1))

        class FakeQwen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)
                self.model = torch.nn.Module()
                self.model.language_model = Language()
                self.wrapper_called = False

            def forward(self, inputs_embeds, **kwargs):
                self.wrapper_called = True
                output = self.model.language_model(inputs_embeds=inputs_embeds, **kwargs)
                return SimpleNamespace(hidden_states=(output.last_hidden_state,))

            def get_input_embeddings(self):
                return self.embedding

            def get_base_model(self):
                return self

        class VisionBank:
            token_dim = 8

            def tokens_for(self, keys, subsets, device, max_tokens_per_view):
                del keys, subsets, max_tokens_per_view
                return (
                    torch.ones((1, 1, 8), device=device),
                    torch.ones((1, 1), dtype=torch.bool, device=device),
                    [1], torch.ones((1, 1), dtype=torch.long, device=device),
                    [[("Optical true-color evidence.", 1)]],
                )

        controller = QwenMaskQueryController.__new__(QwenMaskQueryController)
        torch.nn.Module.__init__(controller)
        controller.model = FakeQwen()
        controller.tokenizer = Tokenizer()
        controller.hidden_size = controller.dim = 8
        controller.num_queries = 2
        controller.max_text_tokens = 16
        controller.view_tokens_per_view = 1
        controller.view_pooling = "tokens"
        controller.visual_ablation = "normal"
        controller.gradient_checkpointing_mode = "disabled"
        controller.gradient_checkpointing_kwargs = None
        controller.vision_bank = VisionBank()
        controller.vision_start_token_id = 30
        controller.vision_end_token_id = 31
        controller.text_type = torch.nn.Parameter(torch.zeros(3, 8))
        controller.view_description_type = torch.nn.Parameter(torch.zeros(8))
        controller.view_attention_query = torch.nn.Parameter(torch.zeros(8))
        controller.evidence_anchors = torch.nn.Parameter(torch.randn(6, 8) * 0.02)
        controller.anchor_availability = torch.nn.Embedding(2, 8)
        controller.mask_embeddings = torch.nn.Parameter(torch.randn(2, 8) * 0.02)
        controller.view_to_hidden = torch.nn.Linear(8, 8, bias=False)
        controller.visual_family_embedding = torch.nn.Embedding(6, 8)
        with torch.no_grad():
            controller.view_to_hidden.weight.copy_(torch.eye(8))
        controller.output_projection = torch.nn.Identity()
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], components=1, size=32)
        evidence = controller.encode_batch(batch)
        self.assertEqual(tuple(evidence.mask_query_states.shape), (1, 2, 8))
        self.assertTrue(controller.model.wrapper_called)
        self.assertFalse(torch.allclose(evidence.mask_query_states[0], controller.mask_embeddings))
        language_inputs = controller.model.model.language_model.last_inputs[0]
        embedding = controller.model.get_input_embeddings().weight.detach()
        self.assertTrue(torch.any(torch.all(language_inputs == embedding[30], dim=-1)))
        self.assertTrue(torch.any(torch.all(language_inputs == embedding[31], dim=-1)))
        evidence.mask_query_states.sum().backward()
        self.assertIsNotNone(controller.mask_embeddings.grad)
        self.assertGreater(float(controller.mask_embeddings.grad.abs().sum()), 0.0)
        controller.visual_ablation = "image-text-delta"
        delta_evidence = controller.encode_batch(batch)
        self.assertIsNotNone(delta_evidence.visual_delta_norm)
        self.assertGreater(float(delta_evidence.visual_delta_norm[0].detach()), 0.0)
        self.assertTrue(torch.allclose(delta_evidence.mask_query_states, evidence.mask_query_states))

    def test_real_tiny_qwen_peft_wrapper_trains_lora_b_then_a(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            self.skipTest(f"Qwen/PEFT production dependencies unavailable: {exc}")

        config = Qwen3VLConfig(
            text_config={
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "max_position_embeddings": 64,
                "pad_token_id": 0,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 4,
                "out_hidden_size": 16,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [],
            },
            image_token_id=29,
            video_token_id=30,
            vision_start_token_id=27,
            vision_end_token_id=28,
        )
        base = Qwen3VLForConditionalGeneration(config)
        base.model.visual = None
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                layers_to_transform=[1],
                layers_pattern="layers",
                task_type="CAUSAL_LM",
            ),
        )
        controller = QwenMaskQueryController.__new__(QwenMaskQueryController)
        torch.nn.Module.__init__(controller)
        controller.model = model
        controller.lora_layer_indices = (1,)
        controller.lora_module_names = controller._validate_lora_injection((1,))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1.0e-2,
            weight_decay=0.0,
        )
        inputs = torch.randn(1, 8, 16, requires_grad=True)
        target = torch.linspace(-0.5, 0.5, 2 * 16).reshape(1, 2, 16)
        reports = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with controller.trace_lora_execution() as execution:
                output = model(
                    inputs_embeds=inputs,
                    attention_mask=torch.ones(1, 8, dtype=torch.long),
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
            loss = torch.nn.functional.mse_loss(output.hidden_states[-1][:, -2:].float(), target)
            loss.backward()
            reports.append({
                matrix: sum(
                    float(parameter.grad.detach().float().norm())
                    for name, parameter in model.named_parameters()
                    if matrix in name and parameter.grad is not None
                )
                for matrix in ("lora_A", "lora_B")
            } | {"executed": sum(value > 0 for value in execution.values())})
            optimizer.step()
        lora_modules = [
            module
            for _, module in model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        ]
        self.assertEqual(len(lora_modules), 4)
        self.assertEqual(reports[0]["executed"], 4)
        self.assertGreater(reports[0]["lora_B"], 0.0)
        self.assertEqual(reports[0]["lora_A"], 0.0)
        self.assertGreater(reports[1]["lora_A"], 0.0)

        # Mirror the production ordering: build the student segmentation
        # graph first, then run a no-grad consistency teacher under outer
        # BF16 autocast. Both LoRA matrices must remain connected.
        mask_head = torch.nn.Linear(16, 1)
        model.zero_grad(set_to_none=True)
        mask_head.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            student = model(
                inputs_embeds=inputs,
                attention_mask=torch.ones(1, 8, dtype=torch.long),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=1,
            )
            student_logits = mask_head(student.hidden_states[-1][:, -2:].float()).squeeze(-1)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            teacher = model(
                inputs_embeds=inputs.detach(),
                attention_mask=torch.ones(1, 8, dtype=torch.long),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=1,
            )
            teacher_logits = mask_head(teacher.hidden_states[-1][:, -2:].float()).squeeze(-1)
        segmentation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            student_logits.float(), torch.ones_like(student_logits).float()
        )
        consistency_loss = (
            torch.sigmoid(student_logits.float()) - torch.sigmoid(teacher_logits.float())
        ).square().mean()
        (segmentation_loss + 0.1 * consistency_loss).backward()
        end_to_end = {
            matrix: sum(
                float(parameter.grad.detach().float().norm())
                for name, parameter in model.named_parameters()
                if matrix in name and parameter.grad is not None
            )
            for matrix in ("lora_A", "lora_B")
        }
        self.assertGreater(end_to_end["lora_A"], 0.0)
        self.assertGreater(end_to_end["lora_B"], 0.0)

    def test_coverage_mode_when_components_exceed_queries(self) -> None:
        output = self.model(synthetic_batch([instance("optical_rgb", "optical", 3, 64)], components=10))
        self.assertEqual(float(output["proposal_matching_coverage_mode"][0]), 1.0)
        self.assertGreater(float(output["loss_proposal_coverage"]), 0.0)
        self.assertGreater(float(output["loss_proposal_coverage_coarse"]), 0.0)
        self.assertEqual(float(output["loss_proposal_set"]), 0.0)

    def test_coverage_assignment_marks_only_queries_that_win_components(self) -> None:
        target = torch.zeros((32, 32))
        for top, left in ((2, 2), (12, 12), (22, 22)):
            target[top:top + 4, left:left + 4] = 1.0
        proposals = torch.full((2, 32, 32), -8.0)
        proposals[0][target > 0] = 8.0
        assignment = assign_proposals(
            proposals, torch.tensor([2.0, -2.0]), target, torch.ones_like(target)
        )
        self.assertTrue(assignment.coverage_mode)
        self.assertEqual(assignment.matched_queries.tolist(), [0, 0, 0])
        self.assertEqual(assignment.matched_components.tolist(), [0, 1, 2])
        self.assertEqual(assignment.relevance_targets.tolist(), [1.0, 0.0])

    def test_relevance_union_is_calibrated_for_query_count(self) -> None:
        masks = torch.zeros((1, 16, 8, 8))
        relevance = torch.zeros((1, 16))
        final = ProposalSetMaskRefinementDecoder.compose_final_mask(masks, relevance)
        self.assertLess(float(torch.sigmoid(final).mean()), 0.5)

    def test_pmrd_query_detail_masks_invalid_modality_pixels(self) -> None:
        decoder = ProposalSetMaskRefinementDecoder(4, num_queries=1, num_layers=1, num_heads=1)
        evidence = SimpleNamespace(
            modality_active=torch.tensor([[True, True]]),
            real_reliability_mass=torch.ones(1),
            modality_detail=torch.tensor([[[[[10.0, 10.0]]], [[[2.0, 2.0]]]]]),
            modality_valid_detail=torch.tensor([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]]),
        )
        detail = decoder._query_detail(
            evidence,
            torch.tensor([[[0.5, 0.5]]]),
            torch.zeros((1, 1, 1, 2)),
            0,
            1,
        )
        self.assertTrue(torch.allclose(detail[0, 0, 0, 0], torch.tensor([5.0, 1.0])))

    def test_pmrd_coarse_gate_and_mask_bias_suppress_background_detail(self) -> None:
        decoder = ProposalSetMaskRefinementDecoder(4, num_queries=1, num_layers=1, num_heads=1)
        evidence = SimpleNamespace(
            modality_active=torch.tensor([[True]]),
            real_reliability_mass=torch.ones(1),
            modality_detail=torch.ones((1, 1, 4, 2, 2)),
            modality_valid_detail=torch.ones((1, 1, 1, 2, 2)),
        )
        weights = torch.ones((1, 1, 1))
        outside = decoder._query_detail(
            evidence, weights, torch.full((1, 1, 2, 2), -10.0), 0, 1
        )
        inside = decoder._query_detail(
            evidence, weights, torch.full((1, 1, 2, 2), 10.0), 0, 1
        )
        self.assertLess(float(outside.mean().detach()), 1.0e-3)
        self.assertGreater(float(inside.mean().detach()), 0.99)
        self.assertEqual(float(decoder.coarse_mask_bias.bias.detach()), -2.0)
        self.assertEqual(float(decoder.final_mask_bias.bias.detach()), -2.0)

    def test_missing_modality_consistency_is_finite(self) -> None:
        config = replace(self.config, modality_dropout=1.0, missing_modality_consistency_weight=0.1)
        model = MultiSourceQwenPSALMSeg(config, torch.device("cpu")).train()
        output = model(
            synthetic_batch(
                [
                    instance("multispectral", "multispectral", 8, 32, "sentinel2"),
                    instance("dem", "terrain", 1, 64, "dem"),
                ]
            )
        )
        self.assertTrue(torch.isfinite(output["loss_missing_modality_consistency"]))

    def test_missing_modality_teacher_is_deterministic_and_restores_training_state(self) -> None:
        batch = synthetic_batch([instance("optical_rgb", "optical", 3, 32)], size=32)
        self.model.train()
        first = self.model._teacher_mask_logits(batch)
        second = self.model._teacher_mask_logits(batch)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.controller.training)

    def test_missing_modality_teacher_only_decodes_dropped_samples(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 32)
        terrain = instance("dem", "terrain", 1, 32)
        batch = repeat_batch(synthetic_batch([optical, terrain], size=32), 2)
        batch.instances[1] = [optical]
        batch.active_subsets[1] = ActiveModalitySubset(
            active_names=("optical_rgb",),
            dropped_names=("dem",),
            signature="synthetic-dropped",
            is_full=False,
        )
        config = replace(self.config, missing_modality_consistency_weight=0.1)
        model = MultiSourceQwenPSALMSeg(config, torch.device("cpu")).train()
        with patch.object(model, "_teacher_mask_logits", wraps=model._teacher_mask_logits) as teacher:
            output = model(batch)
        self.assertEqual(teacher.call_count, 1)
        self.assertEqual(teacher.call_args.args[0].batch_size, 1)
        self.assertEqual(float(output["teacher_sample_count"]), 1.0)

    def test_v2_forward_backward_is_finite(self) -> None:
        model = MultiSourceQwenPSALMSeg(self.config, torch.device("cpu")).train()
        output = model(synthetic_batch([instance("optical_rgb", "optical", 3, 48)], size=48))
        output["loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_multimodal_anomaly_backward_is_finite(self) -> None:
        model = MultiSourceQwenPSALMSeg(self.config, torch.device("cpu")).train()
        batch = synthetic_batch(
            [
                instance("optical_rgb", "optical", 3, 64),
                instance("multispectral", "multispectral", 8, 32, "sentinel2"),
                instance("dem", "terrain", 1, 48, "dem"),
            ],
            components=2,
            size=64,
        )
        with torch.autograd.detect_anomaly(check_nan=True):
            output = model(batch)
            output["loss"].backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        alignment_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "qmef.align" in name and parameter.grad is not None
        ]
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(gradients)
        self.assertTrue(alignment_gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_new_checkpoint_format_reloads(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_last.pt"
            save_checkpoint(path, self.model, optimizer, 3, self.config, update_last=False)
            restored = MultiSourceQwenPSALMSeg(self.config, torch.device("cpu")).eval()
            self.assertEqual(load_checkpoint(path, restored), 3)
            source = next(self.model.parameters()).detach()
            target = next(restored.parameters()).detach()
            self.assertTrue(torch.equal(source, target))
            incompatible_config = replace(self.config, use_qmef=False)
            incompatible = MultiSourceQwenPSALMSeg(incompatible_config, torch.device("cpu")).eval()
            with self.assertRaisesRegex(RuntimeError, "architecture spec"):
                load_checkpoint(path, incompatible)

    def test_checkpoint_roundtrips_enabled_grad_scaler(self) -> None:
        class FakeScaler:
            def __init__(self, state=None):
                self.state = state or {"scale": 1024.0}

            def is_enabled(self):
                return True

            def state_dict(self):
                return dict(self.state)

            def load_state_dict(self, state):
                self.state = dict(state)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_last.pt"
            save_checkpoint(
                path, self.model, optimizer, 4, self.config,
                update_last=False, scaler=FakeScaler({"scale": 2048.0}),
            )
            restored = MultiSourceQwenPSALMSeg(self.config, torch.device("cpu")).eval()
            restored_scaler = FakeScaler()
            self.assertEqual(load_checkpoint(path, restored, scaler=restored_scaler), 4)
            self.assertEqual(restored_scaler.state, {"scale": 2048.0})

    def test_qwen_checkpoint_schedule_validation(self) -> None:
        config = QPSalmConfig(
            controller="qwen_mask_query",
            qwen_lora_start_step=300,
            qwen_lora_lr_scale=0.2,
            controller_lr_scale=0.5,
        )
        checkpoint = {
            "step": 300,
            "resume_training_stage": "qlora_active",
            "runtime_spec": {
                "qwen_lora_start_step": 300,
                "qwen_lora_lr_scale": 0.2,
                "controller_lr_scale": 0.5,
            },
        }
        validate_checkpoint_training_schedule(checkpoint, config)
        with self.assertRaisesRegex(RuntimeError, "training schedule"):
            validate_checkpoint_training_schedule(
                checkpoint,
                replace(config, qwen_lora_start_step=400),
            )
        with self.assertRaisesRegex(RuntimeError, "training stage"):
            validate_checkpoint_training_schedule(
                {**checkpoint, "resume_training_stage": "decoder_warmup"},
                config,
            )


if __name__ == "__main__":
    unittest.main()
