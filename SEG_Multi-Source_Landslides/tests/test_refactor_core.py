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
from pathlib import Path
import tempfile
import unittest

import torch

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.controllers import TextProbeController
from qpsalm_seg.data import modality_valid_mask, resize_pad_tensor
from qpsalm_seg.metrics import batch_binary_metrics
from qpsalm_seg.models import MultiSourceQwenPSALMSeg
from qpsalm_seg.models.pmrd import ProposalSetMaskRefinementDecoder
from qpsalm_seg.schema import ModalityBatch, ModalityInstance
from qpsalm_seg.train_eval import load_checkpoint, restored_original_space_metrics, save_checkpoint, validation_selection_score


def instance(name: str, family: str, channels: int, size: int, sensor: str = "unknown") -> ModalityInstance:
    generator = torch.Generator().manual_seed(channels * 100 + size)
    image = torch.rand((channels, size, size), generator=generator)
    return ModalityInstance(
        name=name,
        family=family,
        sensor=sensor,
        band_names=tuple(f"B{index}" for index in range(channels)),
        orbit="unknown",
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
        "canonical_combo": "synthetic",
        "sensor_combo": "synthetic",
        "normalization_combo": "synthetic",
        "gsd_token": "meter_5_10",
        "raw_modalities": [{"name": item.name} for item in instances],
    }]
    return ModalityBatch(
        instances=[instances],
        mask=mask,
        valid_mask=torch.ones_like(mask),
        metadata=metadata,
        proposal_context_text=["Segment all landslide regions using available sensors."],
        condition_prompt_text=["Condition prompt: landslide."],
        evidence_reasoning_text=["Use optical, multispectral and terrain evidence."],
        visual_evidence_key=["qmv-parent:synthetic"],
        visual_preview=torch.zeros((1, 3, size, size)),
    )


class ValidMetricTest(unittest.TestCase):
    def test_modality_valid_mask_respects_nan_and_nodata(self) -> None:
        array = torch.ones((2, 4, 4)).numpy()
        array[:, 0, 0] = float("nan")
        array[:, 1, 1] = -9999.0
        valid = modality_valid_mask(array, {"nodata_value": -9999.0})
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

    def test_restored_original_size_metric_matches_perfect_canvas(self) -> None:
        source = torch.zeros((1, 20, 40))
        source[:, 4:14, 9:27] = 1.0
        canvas, transform = resize_pad_tensor(source, 64, mode="nearest")
        logits = torch.where(canvas[None] > 0.5, torch.full_like(canvas[None], 20.0), torch.full_like(canvas[None], -20.0))
        metrics = restored_original_space_metrics(logits, canvas[None], [{"resize_transform": transform}], 0.5)
        self.assertAlmostEqual(metrics[0]["iou"], 1.0, places=6)

    def test_checkpoint_selection_uses_positive_only_dice(self) -> None:
        report = {"metrics": {"overall": {"dice": 0.9}, "positive_only": {"dice": 0.2}}}
        self.assertEqual(validation_selection_score(report, "positive_only_dice"), 0.2)


class ThreeModuleModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            QPSalmConfig(),
            decoder_dim=32,
            num_heads=4,
            num_mask_tokens=4,
            num_decoder_layers=1,
            modality_dropout=0.0,
            max_native_size=64,
            missing_modality_consistency_weight=0.0,
        )
        self.model = MultiSourceQwenPSALMSeg(self.config, TextProbeController(32)).eval()

    def test_variable_channels_and_modality_order(self) -> None:
        optical = instance("optical_rgb", "optical", 3, 64)
        s2 = instance("multispectral", "multispectral", 12, 32, "sentinel2")
        terrain = instance("dem", "terrain", 1, 64, "dem")
        first = self.model(synthetic_batch([optical, s2, terrain]))
        second = self.model(synthetic_batch([terrain, optical, s2]))
        self.assertEqual(tuple(first.final_mask_logits.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(first["loss"]))
        self.assertTrue(torch.allclose(first.final_mask_logits, second.final_mask_logits, atol=2.0e-5, rtol=2.0e-5))

    def test_coverage_mode_when_components_exceed_queries(self) -> None:
        output = self.model(synthetic_batch([instance("optical_rgb", "optical", 3, 64)], components=10))
        self.assertEqual(float(output["proposal_matching_coverage_mode"][0]), 1.0)
        self.assertGreater(float(output["loss_proposal_coverage"]), 0.0)

    def test_relevance_union_is_calibrated_for_query_count(self) -> None:
        masks = torch.zeros((1, 16, 8, 8))
        relevance = torch.zeros((1, 16))
        final = ProposalSetMaskRefinementDecoder.compose_final_mask(masks, relevance)
        self.assertLess(float(torch.sigmoid(final).mean()), 0.5)

    def test_missing_modality_consistency_is_finite(self) -> None:
        config = replace(self.config, modality_dropout=1.0, missing_modality_consistency_weight=0.1)
        model = MultiSourceQwenPSALMSeg(config, TextProbeController(32)).train()
        output = model(
            synthetic_batch(
                [
                    instance("multispectral", "multispectral", 8, 32, "sentinel2"),
                    instance("dem", "terrain", 1, 64, "dem"),
                ]
            )
        )
        self.assertTrue(torch.isfinite(output["loss_missing_modality_consistency"]))

    def test_new_checkpoint_format_reloads(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_last.pt"
            save_checkpoint(path, self.model, optimizer, 3, self.config, update_last=False)
            restored = MultiSourceQwenPSALMSeg(self.config, TextProbeController(32)).eval()
            self.assertEqual(load_checkpoint(path, restored), 3)
            source = next(self.model.parameters()).detach()
            target = next(restored.parameters()).detach()
            self.assertTrue(torch.equal(source, target))


if __name__ == "__main__":
    unittest.main()
