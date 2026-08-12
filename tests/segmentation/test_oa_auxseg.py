"""OA-AuxSeg 多模态合同、梯度、重载与区域测试。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from torchvision.models import convnext_small

from oa_groundrag.segmentation.checkpoint import (
    model_from_checkpoint,
    read_checkpoint,
    save_training_checkpoint,
)
from oa_groundrag.segmentation.contracts import (
    CONFIG_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    INFERENCE_SCHEMA_VERSION,
    ModelRegistry,
    OAAuxSegBatch,
    OAAuxSegConfig,
    PackedAuxiliary,
    RuntimeConfig,
    SUPPORTED_AUXILIARY_ORDER,
    TRAINING_REPORT_SCHEMA_VERSION,
    VARIANTS,
    ordered_auxiliary_names,
)
from oa_groundrag.segmentation.data import (
    AuxiliarySubsetSampler,
    PreparedBatch,
    StatefulTrainingBatcher,
    available_auxiliaries_by_sample,
    benchmark_contract_from_root,
    filter_auxiliaries,
    prepare_collated_batch,
    registry_from_benchmark,
)
from oa_groundrag.segmentation.config import load_runtime_config
from oa_groundrag.segmentation.policy import (
    autocast_context as _autocast,
    prepare_policy_batch,
)
from oa_groundrag.evaluation.segmentation import evaluate_model
from oa_groundrag.segmentation.inference import run_inference
from oa_groundrag.training.segmentation.engine import (
    _acceptance_collated,
    _capacity_acceptance,
    _training_state,
    _validation_is_better,
    effective_training_config,
    make_optimizer,
    make_scaler,
    make_scheduler,
    optimizer_learning_rates,
    require_local_backbone,
)
from oa_groundrag.training.segmentation.finalization import (
    _metric_replay_delta,
    _read_finalization_log,
    _validate_finalization_evidence,
    finalize_training_run,
)
from oa_groundrag.segmentation.inference import SpatialInferenceSession
from oa_groundrag.segmentation.losses import bce_dice_loss
from oa_groundrag.segmentation.metrics import SegmentationMetrics
from oa_groundrag.segmentation.model import (
    BACKBONE_STAGE_DEPTHS,
    EXPECTED_BACKBONE_SHA256,
    OAAuxSegModel,
    optical_rgb_indices,
)
from oa_groundrag.segmentation.fusion import (
    CMNeXtSpatialSelector,
    FullChannelCrossAttention,
    SparseStageFeature,
)
from oa_groundrag.segmentation.regions import extract_regions_and_features
from oa_groundrag.training.segmentation.progress import (
    TrainingProgress,
    format_compact_finalization_report,
    format_compact_training_report,
)
from oa_groundrag.segmentation.validity import (
    resample_validity,
    validity_from_channels,
)
from oa_groundrag.data.oa_auxseg.dataset import (
    BenchmarkDataset,
    collate_benchmark_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_ROOT = (
    REPO_ROOT.parent / "benchmark" / "oa_auxseg_hdf5_v1" / "small"
)


def synthetic_registry() -> ModelRegistry:
    rgb = ("Red", "Green", "Blue")
    rgb_short = ("R", "G", "B")
    nir = ("Blue", "Green", "Red", "NIR")
    sentinel = tuple(f"B{index:02d}" for index in range(1, 13))
    return ModelRegistry(
        optical_signatures=tuple(
            sorted((rgb_short, rgb, nir, sentinel))
        ),
        auxiliary_channels={
            "dem": ("DEM",),
            "insar_velocity": ("InSAR_mean_LOS_velocity_encoded",),
            "slope": ("slope",),
        },
        available_auxiliaries={
            rgb_short: (),
            rgb: ("dem", "insar_velocity"),
            nir: (),
            sentinel: ("dem", "slope"),
        },
    )


def synthetic_batch(size: int = 32) -> OAAuxSegBatch:
    generator = torch.Generator().manual_seed(11)
    names = (
        ("R", "G", "B"),
        ("Blue", "Green", "Red", "NIR"),
        tuple(f"B{index:02d}" for index in range(1, 13)),
        ("Red", "Green", "Blue"),
        tuple(f"B{index:02d}" for index in range(1, 13)),
        ("Red", "Green", "Blue"),
    )
    optical = tuple(
        torch.randn(len(signature), size, size, generator=generator)
        for signature in names
    )
    optical_valid = tuple(torch.ones_like(value, dtype=torch.uint8) for value in optical)
    channel_valid = tuple(
        torch.ones(value.shape[0], dtype=torch.uint8) for value in optical
    )

    def packed(
        indices: list[int], channel_names: tuple[str, ...]
    ) -> PackedAuxiliary:
        values = torch.randn(
            len(indices), len(channel_names), size, size, generator=generator
        )
        return PackedAuxiliary(
            sample_indices=torch.tensor(indices, dtype=torch.int64),
            values=values,
            pixel_valid=torch.ones_like(values, dtype=torch.uint8),
            channel_valid=torch.ones(
                (len(indices), len(channel_names)), dtype=torch.uint8
            ),
            channel_names=channel_names,
        )

    return OAAuxSegBatch(
        optical=optical,
        optical_pixel_valid=optical_valid,
        optical_channel_valid=channel_valid,
        optical_channel_names=names,
        auxiliaries={
            "dem": packed([2, 3, 4, 5], ("DEM",)),
            "insar_velocity": packed(
                [3, 5], ("InSAR_mean_LOS_velocity_encoded",)
            ),
            "slope": packed([2, 4], ("slope",)),
        },
    )


def replace_auxiliary(
    batch: OAAuxSegBatch, name: str, replacement: PackedAuxiliary
) -> OAAuxSegBatch:
    auxiliaries = dict(batch.auxiliaries)
    auxiliaries[name] = replacement
    return OAAuxSegBatch(
        optical=batch.optical,
        optical_pixel_valid=batch.optical_pixel_valid,
        optical_channel_valid=batch.optical_channel_valid,
        optical_channel_names=batch.optical_channel_names,
        auxiliaries=auxiliaries,
    )


class OAAuxSegTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.registry = synthetic_registry()
        cls.batch = synthetic_batch()
        cls.model = OAAuxSegModel(
            OAAuxSegConfig(variant="proposed_dropout"),
            cls.registry,
            backbone_weights=None,
        )

    def setUp(self) -> None:
        self.model.variant = "proposed_dropout"
        self.model.eval()

    def test_single_batch_inference_helper_matches_existing_math(self) -> None:
        prepared = PreparedBatch(
            model=self.batch,
            mask=torch.zeros((self.batch.batch_size, 1, 32, 32)),
            sample_ids=[f"sample-{index}" for index in range(self.batch.batch_size)],
            metadata=[{} for _ in range(self.batch.batch_size)],
        )
        session = SpatialInferenceSession.__new__(SpatialInferenceSession)
        session.config = SimpleNamespace(use_bf16=False)
        session.model = self.model
        session.device = torch.device("cpu")
        helper = session.infer_prepared(prepared)

        direct_prepared = prepared.to(torch.device("cpu"), non_blocking=False)
        direct_batch, available, active = prepare_policy_batch(
            direct_prepared.model,
            variant=self.model.variant,
            subset_sampler=None,
        )
        with torch.inference_mode(), _autocast(session.config, session.device):
            direct = self.model(direct_batch, return_regions=True)
        torch.testing.assert_close(helper.output.mask_logits, direct.mask_logits)
        torch.testing.assert_close(helper.output.mask_probability, direct.mask_probability)
        torch.testing.assert_close(helper.output.no_target_score, direct.no_target_score)
        torch.testing.assert_close(helper.output.modality_weights, direct.modality_weights)
        self.assertEqual(helper.available_modalities, tuple(tuple(row) for row in available))
        self.assertEqual(helper.active_modalities, tuple(tuple(row) for row in active))
        self.assertIsNotNone(helper.output.candidate_regions)
        self.assertIsNotNone(direct.candidate_regions)
        assert helper.output.candidate_regions is not None
        assert direct.candidate_regions is not None
        self.assertEqual(
            [[region.region_id for region in rows] for rows in helper.output.candidate_regions],
            [[region.region_id for region in rows] for rows in direct.candidate_regions],
        )
        for helper_rows, direct_rows in zip(
            helper.output.candidate_regions,
            direct.candidate_regions,
            strict=True,
        ):
            for helper_region, direct_region in zip(helper_rows, direct_rows, strict=True):
                torch.testing.assert_close(helper_region.mask, direct_region.mask)
                self.assertEqual(helper_region.bbox_xyxy, direct_region.bbox_xyxy)
                self.assertEqual(helper_region.area_pixels, direct_region.area_pixels)


    def test_all_variants_forward_variable_channels_and_outputs(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.model.variant = variant
                output = self.model(self.batch, return_regions=True)
                self.assertEqual(tuple(output.mask_logits.shape), (6, 1, 32, 32))
                self.assertEqual(
                    tuple(output.mask_probability.shape), (6, 1, 32, 32)
                )
                self.assertEqual(tuple(output.no_target_score.shape), (6,))
                self.assertEqual(tuple(output.modality_weights.shape), (6, 4))
                self.assertEqual(
                    output.modality_weight_map_strides,
                    (4, 8, 16, 32),
                )
                self.assertEqual(
                    tuple(
                        tuple(weight_map.shape)
                        for weight_map in output.modality_weight_maps
                    ),
                    (
                        (6, 4, 8, 8),
                        (6, 4, 4, 4),
                        (6, 4, 2, 2),
                        (6, 4, 1, 1),
                    ),
                )
                for weight_map in output.modality_weight_maps:
                    torch.testing.assert_close(
                        weight_map.sum(dim=1),
                        torch.ones_like(weight_map[:, 0]),
                        rtol=0,
                        atol=1e-6,
                    )
                torch.testing.assert_close(
                    output.modality_weights.sum(dim=1),
                    torch.ones(6),
                    rtol=0,
                    atol=1e-6,
                )
                self.assertEqual(
                    output.modality_names,
                    self.registry.modality_weight_order,
                )
                self.assertEqual(len(output.candidate_regions or []), 6)
                self.assertEqual(len(output.region_features or []), 6)
                for feature in output.region_features or []:
                    self.assertEqual(feature.shape[1:], (268,))

    def test_absent_auxiliary_is_exact_optical_path(self) -> None:
        no_aux = filter_auxiliaries(
            self.batch, [tuple() for _ in range(self.batch.batch_size)]
        )
        with torch.no_grad():
            self.model.variant = "optical_only"
            optical = self.model(no_aux)
            for variant in VARIANTS[1:]:
                with self.subTest(variant=variant):
                    self.model.variant = variant
                    multimodal = self.model(no_aux)
                    self.assertTrue(
                        torch.equal(
                            multimodal.mask_logits,
                            optical.mask_logits,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            multimodal.mask_probability,
                            optical.mask_probability,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            multimodal.modality_weights,
                            optical.modality_weights,
                        )
                    )
                    self.assertTrue(
                        all(
                            torch.equal(first, second)
                            for first, second in zip(
                                multimodal.modality_weight_maps,
                                optical.modality_weight_maps,
                                strict=True,
                            )
                        )
                    )
                    self.assertTrue(
                        bool(
                            (
                                multimodal.modality_weights[:, -1]
                                == 1
                            ).all()
                        )
                    )
        self.model.variant = "proposed_dropout"

    def test_direct_concat_cannot_expand_optical_support(self) -> None:
        optical_validity = list(self.batch.optical_pixel_valid)
        changed = optical_validity[3].clone()
        changed[:, :, :16] = 0
        optical_validity[3] = changed
        batch = OAAuxSegBatch(
            optical=self.batch.optical,
            optical_pixel_valid=tuple(optical_validity),
            optical_channel_valid=self.batch.optical_channel_valid,
            optical_channel_names=self.batch.optical_channel_names,
            auxiliaries=self.batch.auxiliaries,
        )
        self.model.variant = "direct_concat"
        with torch.no_grad():
            output = self.model(batch)
        self.assertEqual(
            float(output.mask_probability[3, :, :, :16].max().item()),
            0.0,
        )

    def test_fractional_validity_area_downsampling(self) -> None:
        pixel = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
        pixel[:, :, 0, 0] = 1
        validity = validity_from_channels(
            pixel,
            torch.ones(1, 1, dtype=torch.uint8),
        )
        coarse = resample_validity(validity, (1, 1))
        self.assertTrue(bool(coarse.support.item()))
        self.assertAlmostEqual(
            float(coarse.coverage.item()),
            1.0 / 64.0,
            places=7,
        )
        channel_fraction = validity_from_channels(
            torch.ones(1, 2, 4, 4, dtype=torch.uint8),
            torch.tensor([[1, 0]], dtype=torch.uint8),
        )
        torch.testing.assert_close(
            channel_fraction.coverage,
            torch.full((1, 1, 4, 4), 0.5),
        )
        none_valid = validity_from_channels(
            torch.zeros(1, 2, 4, 4, dtype=torch.uint8),
            torch.ones(1, 2, dtype=torch.uint8),
        )
        self.assertFalse(bool(none_valid.support.any()))
        self.assertEqual(float(none_valid.coverage.max().item()), 0.0)

    def test_ffm_matches_deliver_context_formula(self) -> None:
        attention = FullChannelCrossAttention(channels=4, heads=2)
        values = torch.randn(
            2, 5, 4, generator=torch.Generator().manual_seed(37)
        )
        coverage = torch.ones(2, 5)
        key, value = attention.optical_kv(values).chunk(2, dim=-1)
        key = attention._heads(key)
        value = attention._heads(value)
        expected = torch.softmax(
            (key.transpose(-2, -1) @ value) * attention.scale,
            dim=-2,
        )
        actual = attention._context(
            values, coverage, attention.optical_kv
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        fractional = torch.tensor(
            [[1.0, 0.5, 0.25, 0.0, 1.0]] * 2
        )
        expected_fractional = torch.softmax(
            (
                key.transpose(-2, -1)
                @ (value * fractional[:, None, :, None])
            )
            * attention.scale,
            dim=-2,
        )
        actual_fractional = attention._context(
            values, fractional, attention.optical_kv
        )
        torch.testing.assert_close(
            actual_fractional,
            expected_fractional,
            rtol=0,
            atol=0,
        )

    def test_cmnext_selector_uses_masked_channelwise_max(self) -> None:
        selector = CMNeXtSpatialSelector(
            2,
            ("first", "second"),
        )
        for parameter in selector.parameters():
            torch.nn.init.zeros_(parameter)
        validity = validity_from_channels(
            torch.ones(1, 1, 2, 2, dtype=torch.uint8),
            torch.ones(1, 1, dtype=torch.uint8),
        )
        first = torch.stack(
            (
                torch.ones(2, 2),
                torch.full((2, 2), 4.0),
            )
        ).unsqueeze(0)
        second = torch.stack(
            (
                torch.full((2, 2), 3.0),
                torch.full((2, 2), 2.0),
            )
        ).unsqueeze(0)
        streams = {
            "first": SparseStageFeature(
                torch.tensor([0]),
                first,
                validity,
            ),
            "second": SparseStageFeature(
                torch.tensor([0]),
                second,
                validity,
            ),
        }
        result = selector(
            streams,
            torch.zeros(1, 2, 2, 2),
            validity,
        )
        expected = torch.stack(
            (
                torch.full((2, 2), 4.5),
                torch.full((2, 2), 6.0),
            )
        ).unsqueeze(0)
        torch.testing.assert_close(result.feature, expected)
        torch.testing.assert_close(
            result.weight_map[:, :2],
            torch.full((1, 2, 2, 2), 0.5),
        )
        self.assertEqual(
            float(result.weight_map[:, -1].detach().max().item()),
            0.0,
        )

    def test_stride4_fusion_propagates_into_stride8(self) -> None:
        no_aux = filter_auxiliaries(
            self.batch, [tuple() for _ in range(self.batch.batch_size)]
        )
        captured: list[torch.Tensor] = []

        def capture_input(
            _module: torch.nn.Module,
            arguments: tuple[torch.Tensor, ...],
        ) -> None:
            captured.append(arguments[0].detach().clone())

        handle = self.model.backbone[1].register_forward_pre_hook(
            capture_input
        )
        try:
            with torch.no_grad():
                self.model.variant = "proposed_dropout"
                self.model(self.batch)
                self.model(no_aux)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        self.assertGreater(
            float((captured[0] - captured[1]).abs().max().item()),
            0.0,
        )

    def test_modality_order_does_not_change_identity(self) -> None:
        reordered: OrderedDict[str, PackedAuxiliary] = OrderedDict()
        for name, packed in reversed(tuple(self.batch.auxiliaries.items())):
            positions = torch.arange(
                packed.sample_indices.shape[0] - 1, -1, -1
            )
            reordered[name] = PackedAuxiliary(
                sample_indices=packed.sample_indices.index_select(0, positions),
                values=packed.values.index_select(0, positions),
                pixel_valid=packed.pixel_valid.index_select(0, positions),
                channel_valid=packed.channel_valid.index_select(0, positions),
                channel_names=packed.channel_names,
            )
        reversed_batch = OAAuxSegBatch(
            optical=self.batch.optical,
            optical_pixel_valid=self.batch.optical_pixel_valid,
            optical_channel_valid=self.batch.optical_channel_valid,
            optical_channel_names=self.batch.optical_channel_names,
            auxiliaries=reordered,
        )
        with torch.no_grad():
            first = self.model(self.batch)
            second = self.model(reversed_batch)
        torch.testing.assert_close(first.mask_logits, second.mask_logits)
        torch.testing.assert_close(
            first.modality_weights, second.modality_weights
        )
        for first_map, second_map in zip(
            first.modality_weight_maps,
            second.modality_weight_maps,
            strict=True,
        ):
            torch.testing.assert_close(first_map, second_map)

    def test_current_auxiliary_combinations_are_sparse_and_independent(
        self,
    ) -> None:
        available = available_auxiliaries_by_sample(self.batch)
        self.assertEqual(available[2], ("dem", "slope"))
        self.assertEqual(available[3], ("dem", "insar_velocity"))
        self.model.variant = "mean_auxiliary_fusion"
        with torch.no_grad():
            output = self.model(self.batch)
        slope = output.modality_names.index("slope")
        insar = output.modality_names.index("insar_velocity")
        self.assertGreater(
            float(output.modality_weights[2, slope].item()), 0.0
        )
        self.assertEqual(
            float(output.modality_weights[2, insar].item()), 0.0
        )
        self.assertEqual(
            float(output.modality_weights[3, slope].item()), 0.0
        )
        self.assertGreater(
            float(output.modality_weights[3, insar].item()), 0.0
        )

    def test_invalid_values_cannot_affect_features(self) -> None:
        insar = self.batch.auxiliaries["insar_velocity"]
        valid = insar.pixel_valid.clone()
        valid[:, :, :, :16] = 0
        zeros = insar.values.clone()
        zeros[:, :, :, :16] = 0
        huge = insar.values.clone()
        huge[:, :, :, :16] = 1e30
        zero_batch = replace_auxiliary(
            self.batch,
            "insar_velocity",
            PackedAuxiliary(
                insar.sample_indices,
                zeros,
                valid,
                insar.channel_valid,
                insar.channel_names,
            ),
        )
        huge_batch = replace_auxiliary(
            self.batch,
            "insar_velocity",
            PackedAuxiliary(
                insar.sample_indices,
                huge,
                valid,
                insar.channel_valid,
                insar.channel_names,
            ),
        )
        with torch.no_grad():
            first = self.model(zero_batch)
            second = self.model(huge_batch)
        torch.testing.assert_close(first.mask_logits, second.mask_logits)
        torch.testing.assert_close(
            first.modality_weights, second.modality_weights
        )
        for first_map, second_map in zip(
            first.modality_weight_maps,
            second.modality_weight_maps,
            strict=True,
        ):
            torch.testing.assert_close(first_map, second_map)

    def test_quality_selector_masks_zero_coverage(self) -> None:
        for name in ("insar_velocity", "slope"):
            with self.subTest(modality=name):
                auxiliary = self.batch.auxiliaries[name]
                zero_coverage = replace_auxiliary(
                    self.batch,
                    name,
                    PackedAuxiliary(
                        auxiliary.sample_indices,
                        torch.full_like(auxiliary.values, 1e30),
                        torch.zeros_like(auxiliary.pixel_valid),
                        torch.zeros_like(auxiliary.channel_valid),
                        auxiliary.channel_names,
                    ),
                )
                with torch.no_grad():
                    output = self.model(zero_coverage)
                sample_index = int(auxiliary.sample_indices[0])
                self.assertEqual(
                    float(
                        output.modality_weights[
                            sample_index,
                            self.registry.modality_weight_order.index(name),
                        ].item()
                    ),
                    0.0,
                )
                self.assertAlmostEqual(
                    float(
                        output.modality_weights[sample_index].sum().item()
                    ),
                    1.0,
                    places=6,
                )

    def test_all_zero_coverage_auxiliary_is_optical_path(self) -> None:
        zero_auxiliaries = {
            name: PackedAuxiliary(
                packed.sample_indices,
                torch.full_like(packed.values, 1e30),
                torch.zeros_like(packed.pixel_valid),
                torch.zeros_like(packed.channel_valid),
                packed.channel_names,
            )
            for name, packed in self.batch.auxiliaries.items()
        }
        zero_coverage = OAAuxSegBatch(
            optical=self.batch.optical,
            optical_pixel_valid=self.batch.optical_pixel_valid,
            optical_channel_valid=self.batch.optical_channel_valid,
            optical_channel_names=self.batch.optical_channel_names,
            auxiliaries=zero_auxiliaries,
        )
        with torch.no_grad():
            self.model.variant = "optical_only"
            optical = self.model(zero_coverage)
            for variant in VARIANTS[1:]:
                with self.subTest(variant=variant):
                    self.model.variant = variant
                    output = self.model(zero_coverage)
                    self.assertTrue(
                        torch.equal(output.mask_logits, optical.mask_logits)
                    )
                    self.assertTrue(
                        torch.equal(
                            output.mask_probability,
                            optical.mask_probability,
                        )
                    )
                    torch.testing.assert_close(
                        output.modality_weights,
                        optical.modality_weights,
                        rtol=0,
                        atol=0,
                    )

    def test_auxiliary_spatial_quality_is_local(self) -> None:
        insar = self.batch.auxiliaries["insar_velocity"]
        local_valid = insar.pixel_valid.clone()
        local_valid[:, :, :, :16] = 0
        localized = replace_auxiliary(
            self.batch,
            "insar_velocity",
            PackedAuxiliary(
                insar.sample_indices,
                insar.values,
                local_valid,
                insar.channel_valid,
                insar.channel_names,
            ),
        )
        with torch.no_grad():
            output = self.model(localized)
        column = output.modality_names.index("insar_velocity")
        sample_index = int(insar.sample_indices[0])
        stride4 = output.modality_weight_maps[0][
            sample_index,
            column,
        ]
        self.assertEqual(float(stride4[:, :4].max().item()), 0.0)
        self.assertGreater(float(stride4[:, 4:].max().item()), 0.0)

    def test_multimodal_loss_backward_and_optimizer_step(self) -> None:
        self.model.train()
        self.model.variant = "proposed_dropout"
        self.model.set_gradient_checkpointing(True)
        self.addCleanup(self.model.set_gradient_checkpointing, False)
        config = RuntimeConfig(
            schema_version=CONFIG_SCHEMA_VERSION,
            benchmark_root="unused",
            output_dir="unused",
            backbone="convnext_small",
            backbone_weights="unused",
            variant="proposed_dropout",
            device="cpu",
            max_steps=2,
        )
        optimizer = make_optimizer(self.model, config)
        optimizer.zero_grad(set_to_none=True)
        target = torch.zeros(6, 1, 32, 32)
        target[1:, :, 8:24, 8:24] = 1
        output = self.model(self.batch)
        loss, components = bce_dice_loss(output.mask_logits, target)
        loss.backward()
        self.assertGreater(float(loss.item()), 0)
        self.assertGreater(float(components["bce"].item()), 0)
        for name, module in self.model.auxiliary_adapters.items():
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and float(parameter.grad.abs().sum().item()) > 0
                    for parameter in module.parameters()
                ),
                name,
            )
        for name, module in (
            ("auxiliary_encoder", self.model.auxiliary_stages),
            ("progressive_fusion", self.model.progressive_fusions),
            ("quality_selector", self.model.quality_selectors),
        ):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and float(parameter.grad.abs().sum().item()) > 0
                    for parameter in module.parameters()
                ),
                name,
            )
        for stride, fusion in zip(
            (4, 8, 16, 32),
            self.model.progressive_fusions,
            strict=True,
        ):
            for name, module in (
                (f"frm_stride{stride}", fusion.rectification),
                (f"ffm_stride{stride}", fusion.fusion),
            ):
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and float(parameter.grad.abs().sum().item()) > 0
                        for parameter in module.parameters()
                    ),
                    name,
                )
        for stride, stage in zip(
            (4, 8, 16, 32),
            self.model.auxiliary_stages,
            strict=True,
        ):
            for block_index, block in enumerate(stage.blocks):
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and float(parameter.grad.abs().sum().item()) > 0
                        for parameter in block.parameters()
                    ),
                    f"mspa_stride{stride}_block{block_index}",
                )
        before_auxiliary = {
            name: self.model.auxiliary_adapters[name]
            .projection[0]
            .weight.detach()
            .clone()
            for name in SUPPORTED_AUXILIARY_ORDER
        }
        optimizer.step()
        for name, before in before_auxiliary.items():
            self.assertFalse(
                torch.equal(
                    before,
                    self.model.auxiliary_adapters[name].projection[0].weight,
                ),
                name,
            )

    def test_subset_sampler_is_stateful_and_sparse(self) -> None:
        available = [
            (),
            ("dem",),
            ("dem", "slope"),
            ("dem", "insar_velocity"),
            ("dem", "insar_velocity", "slope"),
        ]
        sampler = AuxiliarySubsetSampler(123, 0.2)
        first = sampler.sample(available)
        state = sampler.state_dict()
        expected_next = sampler.sample(available)
        restored = AuxiliarySubsetSampler(999, 0.2)
        restored.load_state_dict(state)
        self.assertEqual(expected_next, restored.sample(available))
        self.assertEqual(sampler.state_dict(), restored.state_dict())
        self.assertEqual(first[0], ())
        for selected, names in zip(first, available, strict=True):
            self.assertTrue(set(selected).issubset(names))
            if names:
                self.assertTrue(selected)
        self.assertEqual(
            sum(sampler.reason_counts.values()),
            len(available) * 2,
        )
        self.assertNotIn("sampled_null", sampler.reason_counts)
        self.assertNotIn("dropped_to_none", sampler.reason_counts)
        for seed in (0, 1, 97):
            for dropout in (0.0, 0.2, 0.99):
                candidate = AuxiliarySubsetSampler(seed, dropout)
                for _ in range(20):
                    active = candidate.sample(available)
                    self.assertTrue(
                        all(
                            bool(selected) == bool(present)
                            for selected, present in zip(
                                active, available, strict=True
                            )
                        )
                    )



    def test_optimizer_groups_extra_bands_at_new_lr_and_decay_rules(
        self,
    ) -> None:
        config = RuntimeConfig(
            schema_version=CONFIG_SCHEMA_VERSION,
            benchmark_root="unused",
            output_dir="unused",
            backbone="convnext_small",
            backbone_weights="unused",
            variant="proposed_dropout",
            device="cpu",
        )
        optimizer = make_optimizer(self.model, config)
        parameter_group = {
            id(parameter): group
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        rgb_group = parameter_group[id(self.model.optical_rgb_stem.weight)]
        signature = tuple(f"B{index:02d}" for index in range(1, 13))
        extra = self.model.optical_stems[
            self.model._optical_stem_keys[signature]
        ].extra_projection
        self.assertIsNotNone(extra)
        extra_group = parameter_group[id(extra.weight)]
        self.assertEqual(rgb_group["group_name"], "backbone_decay")
        self.assertEqual(rgb_group["lr"], config.backbone_lr)
        self.assertEqual(extra_group["group_name"], "new_decay")
        self.assertEqual(extra_group["lr"], config.new_lr)
        classifier_bias_group = parameter_group[
            id(self.model.decoder.classifier.bias)
        ]
        self.assertEqual(classifier_bias_group["weight_decay"], 0.0)
        layer_scale = self.model.auxiliary_stages[
            0
        ].blocks[0].layer_scale_attention
        self.assertEqual(
            parameter_group[id(layer_scale)]["weight_decay"], 0.0
        )

    def test_scheduler_respects_lr_floor(self) -> None:
        config = RuntimeConfig(
            schema_version=CONFIG_SCHEMA_VERSION,
            benchmark_root="unused",
            output_dir="unused",
            backbone="convnext_small",
            backbone_weights="unused",
            variant="proposed_dropout",
            device="cpu",
            max_steps=20,
            warmup_ratio=0.05,
            min_lr_ratio=0.10,
        )
        optimizer = make_optimizer(self.model, config)
        scheduler = make_scheduler(
            optimizer,
            total_steps=config.max_steps,
            warmup_ratio=config.warmup_ratio,
            min_lr_ratio=config.min_lr_ratio,
        )
        for _ in range(config.max_steps):
            optimizer.step()
            scheduler.step()
        backbone_lr, new_lr = optimizer_learning_rates(optimizer)
        self.assertAlmostEqual(
            backbone_lr,
            config.backbone_lr * config.min_lr_ratio,
            places=12,
        )
        self.assertAlmostEqual(
            new_lr,
            config.new_lr * config.min_lr_ratio,
            places=12,
        )

    def test_overfit_regime_and_best_checkpoint_order(self) -> None:
        production = load_runtime_config(
            REPO_ROOT
            / "configs"
            / "segmentation"
            / "small_proposed_dropout.json"
        )
        self.assertIs(
            effective_training_config(
                production, capacity_overfit=False
            ),
            production,
        )
        overfit = effective_training_config(
            production, capacity_overfit=True
        )
        self.assertEqual(overfit.weight_decay, 0.0)
        self.assertEqual(overfit.min_lr_ratio, 0.10)
        self.assertEqual(overfit.grad_clip, 5.0)
        self.assertEqual(overfit.optical_stochastic_depth, 0.0)
        self.assertEqual(overfit.auxiliary_drop_path, 0.0)
        self.assertEqual(overfit.decoder_dropout, 0.0)
        self.assertFalse(overfit.use_bf16)
        self.assertFalse(overfit.gradient_checkpointing)
        self.assertEqual(self.model.decoder.fusion[-1].p, 0.1)
        self.assertAlmostEqual(
            max(
                module.probability
                for stage in self.model.auxiliary_stages
                for block in stage.blocks
                for module in (block.drop_path,)
            ),
            0.1,
        )
        self.assertAlmostEqual(
            max(
                module.p
                for module in self.model.backbone.modules()
                if type(module).__name__ == "StochasticDepth"
            ),
            0.1,
        )

        best = {
            "dice": 0.5,
            "loss": 0.8,
            "no_target_false_positive_rate": 0.3,
        }
        self.assertTrue(
            _validation_is_better(
                {
                    "dice": 0.51,
                    "loss": 1.0,
                    "no_target_false_positive_rate": 0.5,
                },
                best,
            )
        )
        self.assertTrue(
            _validation_is_better(
                {
                    "dice": 0.5,
                    "loss": 0.7,
                    "no_target_false_positive_rate": 0.5,
                },
                best,
            )
        )
        self.assertTrue(
            _validation_is_better(
                {
                    "dice": 0.5,
                    "loss": 0.8,
                    "no_target_false_positive_rate": 0.2,
                },
                best,
            )
        )
        state = _training_state(
            config=overfit,
            loss_history=[1.0, 0.5],
            ema_loss=0.5,
            initial_ema_loss=1.0,
            gradient_max={"mspa_stride4": 1.0},
            parameter_change_max={"mspa_stride4": 0.1},
            training_step_seconds=2.0,
            training_samples_seen=16,
            training_steps_timed=2,
            clipped_steps=1,
            clip_scale_sum=1.5,
            clip_scale_min=0.5,
            validation_trajectory=[],
            best_selection=None,
            training_subset_counts={"none": 1, "single": 1},
            training_subset_reason_counts={
                "native_none": 1,
                "active": 1,
            },
            training_native_auxiliary_samples=1,
            training_active_auxiliary_samples=1,
        )
        self.assertEqual(
            state["parameter_change_max"]["mspa_stride4"], 0.1
        )
        capacity = _capacity_acceptance(
            loss_history=[1.0] * 10 + [0.01] * 10,
            metrics={
                "overall": {
                    "dice": 0.99,
                    "positive_only_dice": 0.99,
                    "no_target_false_positive_rate": 0.0,
                    "empty_mean_foreground_probability": 0.0,
                }
            },
            gradient_max={"module": 1.0},
            parameter_updates={"module": 0.1},
            required_modules=("module",),
        )
        self.assertTrue(all(capacity.values()))

    def test_six_variants_synthetic_optimizer_step(self) -> None:
        target = torch.zeros(6, 1, 32, 32)
        target[1:, :, 6:26, 7:25] = 1
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.model.variant = variant
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                output = self.model(self.batch)
                loss, _ = bce_dice_loss(output.mask_logits, target)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        for parameter in self.model.parameters()
                    )
                )
                optimizer.step()


    def test_regions_are_deterministic_and_registry_dimensional(self) -> None:
        probability = torch.zeros(1, 1, 32, 32)
        probability[:, :, 2:8, 3:10] = 0.8
        probability[:, :, 20:22, 20:22] = 0.9
        probability = probability.to(torch.bfloat16)
        optical = torch.randn(1, 128, 8, 8)
        fused = torch.randn(1, 128, 8, 8)
        weights = torch.tensor([[0.2, 0.1, 0.1, 0.6]])
        regions, features = extract_regions_and_features(
            probability=probability,
            optical_feature=optical,
            fused_feature=fused,
            modality_weights=weights,
            expected_feature_dim=268,
            threshold=0.5,
            min_area=16,
        )
        self.assertEqual(len(regions[0]), 1)
        self.assertEqual(regions[0][0].bbox_xyxy, (3, 2, 10, 8))
        self.assertEqual(regions[0][0].area_pixels, 42)
        self.assertEqual(tuple(features[0].shape), (1, 268))
        torch.testing.assert_close(features[0][0, -4:], weights[0])

    def test_metrics_include_empty_mask_contract(self) -> None:
        target = torch.zeros(2, 1, 4, 4)
        target[0, :, 1:3, 1:3] = 1
        probability = target.clone()
        metrics = SegmentationMetrics()
        metrics.update(probability, target)
        result = metrics.compute()
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["positive_only_dice"], 1.0)
        self.assertEqual(result["no_target_false_positive_rate"], 0.0)
        self.assertEqual(result["empty_mean_foreground_probability"], 0.0)

    def test_evaluation_reports_conditional_modality_weights(self) -> None:
        collated = {
            "optical": list(self.batch.optical),
            "optical_pixel_valid": list(
                self.batch.optical_pixel_valid
            ),
            "optical_channel_valid": list(
                self.batch.optical_channel_valid
            ),
            "optical_channel_names": list(
                self.batch.optical_channel_names
            ),
            "auxiliaries": {
                name: {
                    "sample_indices": packed.sample_indices,
                    "values": packed.values,
                    "pixel_valid": packed.pixel_valid,
                    "channel_valid": packed.channel_valid,
                    "channel_names": packed.channel_names,
                }
                for name, packed in self.batch.auxiliaries.items()
            },
            "mask": torch.zeros(6, 1, 32, 32),
            "sample_id": [f"sample-{index}" for index in range(6)],
            "metadata": [{"source": "synthetic"} for _ in range(6)],
        }
        config = RuntimeConfig(
            schema_version=CONFIG_SCHEMA_VERSION,
            benchmark_root="unused",
            output_dir="unused",
            backbone="convnext_small",
            backbone_weights="unused",
            variant="proposed_dropout",
            device="cpu",
            batch_size=6,
        )
        result = evaluate_model(
            self.model,
            [collated],
            device=torch.device("cpu"),
            config=config,
        )
        self.assertEqual(
            result["conditional_weight_counts"]["dem"], 4
        )
        self.assertEqual(
            result["conditional_weight_counts"]["insar_velocity"], 2
        )
        self.assertEqual(
            result["conditional_weight_counts"]["slope"], 2
        )
        self.assertEqual(
            result["conditional_weight_counts"]["__null__"], 4
        )
        self.assertEqual(
            set(result["conditional_stage_modality_weights"]),
            {"stride4", "stride8", "stride16", "stride32"},
        )
        self.assertIn(
            "synthetic", result["conditional_modality_weights_by_source"]
        )

    def test_runtime_config_is_strict_and_weight_gate_is_explicit(self) -> None:
        invalid_model_config = self.model.config.to_dict()
        invalid_model_config["unexpected"] = True
        with self.assertRaises(ValueError):
            OAAuxSegConfig.from_dict(invalid_model_config)
        with self.assertRaises(ValueError):
            RuntimeConfig.from_dict(
                {
                    "schema_version": CONFIG_SCHEMA_VERSION,
                    "benchmark_root": "benchmark",
                    "output_dir": "output",
                    "backbone": "convnext_small",
                    "backbone_weights": "weight",
                    "variant": "proposed_dropout",
                    "unexpected": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "schema"):
            RuntimeConfig.from_dict(
                {
                    "schema_version": "oa_auxseg_runtime_config_v4",
                    "benchmark_root": "benchmark",
                    "output_dir": "output",
                    "backbone": "convnext_small",
                    "backbone_weights": "weight",
                    "variant": "proposed_dropout",
                }
            )
        production = load_runtime_config(
            REPO_ROOT
            / "configs"
            / "segmentation"
            / "small_proposed_dropout.json"
        )
        overfit = load_runtime_config(
            REPO_ROOT
            / "configs"
            / "segmentation"
            / "small_overfit.json"
        )
        self.assertEqual(production.optical_stochastic_depth, 0.1)
        self.assertEqual(production.auxiliary_drop_path, 0.1)
        self.assertEqual(production.decoder_dropout, 0.1)
        self.assertEqual(overfit.weight_decay, 0.0)
        self.assertEqual(overfit.optical_stochastic_depth, 0.0)
        self.assertEqual(overfit.auxiliary_drop_path, 0.0)
        self.assertEqual(overfit.decoder_dropout, 0.0)
        self.assertFalse(overfit.use_bf16)
        self.assertFalse(overfit.gradient_checkpointing)
        self.assertEqual(overfit.grad_clip, 5.0)
        with self.assertRaises(FileNotFoundError):
            require_local_backbone(
                REPO_ROOT / "models_zoo" / "definitely_missing.pth"
            )

    def test_multispectral_extra_band_stem_receives_gradient(self) -> None:
        signature = tuple(f"B{index:02d}" for index in range(1, 13))
        stem = self.model.optical_stems[
            self.model._optical_stem_keys[signature]
        ]
        stem.zero_grad(set_to_none=True)
        values = torch.randn(2, 12, 32, 32)
        pixel_valid = torch.ones_like(values, dtype=torch.uint8)
        channel_valid = torch.ones(2, 12, dtype=torch.uint8)
        feature, _, _ = stem(
            values,
            pixel_valid,
            channel_valid,
            rgb_projection=self.model.optical_rgb_stem,
            normalization=self.model.optical_stem_norm,
        )
        generator = torch.Generator().manual_seed(97)
        probe = torch.randn(
            feature.shape,
            generator=generator,
            dtype=feature.dtype,
        )
        (feature * probe).sum().backward()
        self.assertIsNotNone(stem.extra_projection)
        gradient = stem.extra_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(
            float(gradient.abs().sum().item()),
            0.0,
        )
        self.model.zero_grad(set_to_none=True)

    def test_convnext_small_official_weight_strict_load(self) -> None:
        path = (
            REPO_ROOT
            / "models_zoo"
            / "ConvNeXt"
            / "convnext_small-0c510722.pth"
        )
        self.assertTrue(path.is_file())
        model = OAAuxSegModel(
            OAAuxSegConfig(variant="optical_only"),
            self.registry,
            backbone_weights=path,
        )
        reference = convnext_small(weights=None)
        reference.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True),
            strict=True,
        )
        reference.eval()
        model.eval()
        values = torch.randn(2, 3, 32, 32)
        signature = ("R", "G", "B")
        with torch.no_grad():
            expected = reference.features[0](values)
            actual, _, _ = model.optical_stems[
                model._optical_stem_keys[signature]
            ](
                values,
                torch.ones_like(values, dtype=torch.uint8),
                torch.ones(2, 3, dtype=torch.uint8),
                rgb_projection=model.optical_rgb_stem,
                normalization=model.optical_stem_norm,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.backbone_sha256, EXPECTED_BACKBONE_SHA256)
        self.assertEqual(
            tuple(len(model.backbone[index]) for index in (0, 2, 4, 6)),
            BACKBONE_STAGE_DEPTHS,
        )

    def test_registry_is_dynamic_and_rejects_sar_or_ten_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "未注册辅助模态"):
            ModelRegistry(
                optical_signatures=self.registry.optical_signatures,
                auxiliary_channels={
                    **self.registry.auxiliary_channels,
                    "sar_ascending": ("VV", "VH"),
                },
                available_auxiliaries=self.registry.available_auxiliaries,
            )
        with self.assertRaisesRegex(ValueError, "没有审计过"):
            optical_rgb_indices(tuple(f"B{index:02d}" for index in range(1, 11)))

        dem_only = ModelRegistry(
            optical_signatures=self.registry.optical_signatures,
            auxiliary_channels={"dem": ("DEM",)},
            available_auxiliaries={
                signature: (
                    ("dem",)
                    if "dem"
                    in self.registry.available_auxiliaries.get(signature, ())
                    else ()
                )
                for signature in self.registry.optical_signatures
            },
        )
        self.assertEqual(dem_only.auxiliary_order, ("dem",))
        self.assertEqual(
            dem_only.modality_weight_order,
            ("dem", "__null__"),
        )
        self.assertEqual(dem_only.region_feature_dim(128), 266)
        optical_only_registry = ModelRegistry(
            optical_signatures=self.registry.optical_signatures,
            auxiliary_channels={},
            available_auxiliaries={
                signature: ()
                for signature in self.registry.optical_signatures
            },
        )
        self.assertEqual(
            optical_only_registry.modality_weight_order,
            ("__null__",),
        )
        self.assertEqual(
            optical_only_registry.region_feature_dim(128),
            265,
        )
        dem_model = OAAuxSegModel(
            OAAuxSegConfig(variant="injection_quality"),
            dem_only,
            backbone_weights=None,
        )
        dem_batch = filter_auxiliaries(
            self.batch,
            [
                tuple(),
                tuple(),
                ("dem",),
                ("dem",),
                ("dem",),
                ("dem",),
            ],
        )
        dem_model.eval()
        with torch.no_grad():
            output = dem_model(dem_batch, return_regions=True)
        self.assertEqual(tuple(output.modality_weights.shape), (6, 2))
        self.assertTrue(
            all(
                weight_map.shape[1] == 2
                for weight_map in output.modality_weight_maps
            )
        )
        self.assertTrue(
            all(
                features.shape[1:] == (266,)
                for features in output.region_features or ()
            )
        )


class _PseudoTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class TrainingProgressTest(unittest.TestCase):
    @staticmethod
    def _metrics() -> dict[str, object]:
        return {
            "loss": 0.75,
            "duration_seconds": 3.0,
            "overall": {
                "iou": 0.5,
                "dice": 2.0 / 3.0,
                "precision": 0.7,
                "recall": 0.64,
                "f1": 2.0 / 3.0,
                "positive_only_dice": 0.61,
                "no_target_false_positive_rate": 0.1,
            },
        }

    @classmethod
    def _report(cls) -> dict[str, object]:
        metrics = cls._metrics()
        return {
            "command": "train",
            "variant": "proposed_dropout",
            "steps": 300,
            "wall_elapsed_seconds": 125.0,
            "peak_cuda_memory_gib": 3.8,
            "loss_drop_fraction": 0.35,
            "ema_drop_fraction": 0.53,
            "train_metrics": metrics,
            "validation_metrics": metrics,
            "acceptance_enforced": True,
            "acceptance_passed": True,
            "checkpoint": "/tmp/checkpoint.pt",
            "training_report": "/tmp/training_report.json",
        }

    def test_non_tty_progress_is_rate_limited_and_plain(self) -> None:
        stream = io.StringIO()
        progress = TrainingProgress(
            log_interval=2, stream=stream, force_tty=False
        )
        progress.start_training(
            variant="proposed_dropout", total_steps=5, start_step=0
        )
        for step in range(1, 6):
            progress.update_training(
                step=step,
                loss=1.0 / step,
                ema_loss=0.8,
                bce=0.2,
                dice_loss=0.6,
                learning_rates=(3e-5, 3e-4),
                samples_per_second=2.5,
                estimated_remaining_seconds=float(5 - step),
                peak_cuda_memory_gib=3.5,
            )
        progress.start_evaluation(label="val@step5", total_batches=2)
        progress.update_evaluation(
            batch=1,
            running_loss=0.8,
            metrics=self._metrics()["overall"],  # type: ignore[arg-type]
        )
        progress.finish_evaluation(
            label="val@step5",
            result=self._metrics(),
            duration_seconds=3.0,
        )
        progress.close()
        output = stream.getvalue()
        train_lines = [
            line for line in output.splitlines() if line.startswith("[train]")
        ]
        self.assertEqual(len(train_lines), 4)
        for step in (1, 2, 4, 5):
            self.assertTrue(
                any(f"step={step}/5" in line for line in train_lines)
            )
        self.assertIn("[eval] phase=val@step5 batches=2 start", output)
        self.assertIn("dice=0.6667", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\x1b", output)
        self.assertTrue(progress.closed)

    def test_tty_progress_resume_evaluation_and_exception_close(self) -> None:
        stream = _PseudoTTY()
        progress = TrainingProgress(
            log_interval=10, stream=stream, force_tty=True
        )
        with self.assertRaises(KeyboardInterrupt):
            with progress:
                progress.start_training(
                    variant="proposed_dropout",
                    total_steps=3,
                    start_step=1,
                )
                progress.update_training(
                    step=2,
                    loss=0.9,
                    ema_loss=1.0,
                    bce=0.2,
                    dice_loss=0.7,
                    learning_rates=(3e-5, 3e-4),
                    samples_per_second=2.0,
                    estimated_remaining_seconds=1.0,
                    peak_cuda_memory_gib=4.0,
                )
                progress.start_evaluation(
                    label="val@step2", total_batches=1
                )
                progress.update_evaluation(
                    batch=1,
                    running_loss=0.75,
                    metrics=self._metrics()["overall"],  # type: ignore[arg-type]
                )
                progress.finish_evaluation(
                    label="val@step2",
                    result=self._metrics(),
                    duration_seconds=3.0,
                )
                raise KeyboardInterrupt
        output = stream.getvalue()
        self.assertIn("train proposed_dropout", output)
        self.assertIn("val@step2", output)
        self.assertIn("loss=0.9000", output)
        self.assertIn("\r", output)
        self.assertTrue(progress.closed)

    def test_training_cli_defaults_to_compact_and_can_emit_full_json(
        self,
    ) -> None:
        from oa_groundrag.training.segmentation import cli as cli_module

        report = self._report()
        compact = format_compact_training_report(report)
        self.assertIn("acceptance=PASS", compact)
        self.assertIn("val_dice=0.6667", compact)
        with (
            patch.object(cli_module, "load_runtime_config", return_value=object()),
            patch.object(cli_module, "run_training", return_value=report),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_module.main(
                    ["train", "--config", "unused.json"]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue().strip(), compact)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_module.main(
                    [
                        "train",
                        "--config",
                        "unused.json",
                        "--full-report-json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), report)


class FinalizationTest(unittest.TestCase):
    @staticmethod
    def _config() -> RuntimeConfig:
        return load_runtime_config(
            REPO_ROOT
            / "configs/segmentation/"
            "full_proposed_dropout_b16_nockpt_e100.json"
        )

    @staticmethod
    def _metrics(
        *,
        dice: float = 0.6,
        loss: float = 0.4,
        false_positive_rate: float = 0.2,
        sample_count: int = 2,
    ) -> dict[str, object]:
        overall = {
            "sample_count": sample_count,
            "iou": 0.5,
            "dice": dice,
            "precision": 0.7,
            "recall": 0.65,
            "f1": dice,
            "positive_only_iou": 0.52,
            "positive_only_dice": 0.61,
            "no_target_count": 1,
            "no_target_false_positive_rate": false_positive_rate,
            "empty_mean_foreground_probability": 0.03,
        }
        return {
            "loss": loss,
            "duration_seconds": 1.0,
            "overall": overall,
            "by_source": {"synthetic": dict(overall)},
            "by_available_modality_signature": {"none": dict(overall)},
            "by_active_subset": {"none": dict(overall)},
            "mean_modality_weights": {"__null__": 1.0},
            "conditional_mean_modality_weights": {"__null__": None},
            "conditional_weight_counts": {"__null__": 0},
            "conditional_stage_modality_weights": {
                "stride4": {"__null__": None}
            },
            "conditional_modality_weights_by_source": {},
        }

    @classmethod
    def _evidence(cls) -> tuple[
        RuntimeConfig,
        list[dict[str, object]],
        dict[str, object],
        dict[str, object],
    ]:
        config = cls._config()
        first = {
            "step": 1,
            "validation": cls._metrics(dice=0.5, loss=0.5),
            "is_new_best": True,
        }
        selected = {
            "step": 2,
            "validation": cls._metrics(dice=0.6, loss=0.4),
            "is_new_best": True,
        }
        later = {
            "step": 3,
            "validation": cls._metrics(dice=0.59, loss=0.39),
            "is_new_best": False,
        }
        rows: list[dict[str, object]] = [
            first,
            selected,
            {"step": 2, "checkpoint_best": {"path": "best"}},
            later,
            {"step": 4, "loss": 0.2},
        ]
        best_selection = {
            "step": 2,
            "dice": 0.6,
            "loss": 0.4,
            "no_target_false_positive_rate": 0.2,
        }

        def state(
            step: int, trajectory: list[dict[str, object]]
        ) -> dict[str, object]:
            return {
                "runtime_config": config.to_dict(),
                "loss_history": [1.0 / index for index in range(1, step + 1)],
                "ema_loss": 0.2,
                "initial_ema_loss": 1.0,
                "gradient_max": {"decoder": 0.5},
                "parameter_change_max": {"decoder": 0.1},
                "training_step_seconds": float(step),
                "training_samples_seen": step * config.batch_size,
                "training_steps_timed": step,
                "clipped_steps": 0,
                "clip_scale_sum": float(step),
                "clip_scale_min": 1.0,
                "validation_trajectory": trajectory,
                "best_selection": dict(best_selection),
                "training_subset_counts": {"none": step * config.batch_size},
                "training_subset_reason_counts": {
                    "native_none": step * config.batch_size
                },
                "training_native_auxiliary_samples": 0,
                "training_active_auxiliary_samples": 0,
            }

        contract = {"architecture": "synthetic"}
        selected_payload = {
            "step": 2,
            "training_state": state(2, [first, selected]),
            "model_contract": contract,
        }
        last_payload = {
            "step": 3,
            "training_state": state(3, [first, selected, later]),
            "model_contract": contract,
        }
        return config, rows, selected_payload, last_payload

    def test_manual_stop_evidence_keeps_log_tail_without_resume(self) -> None:
        config, rows, selected_payload, last_payload = self._evidence()
        evidence = _validate_finalization_evidence(
            config=config,
            rows=rows,
            selected_payload=selected_payload,
            last_payload=last_payload,
        )
        self.assertEqual(evidence["selected_checkpoint_step"], 2)
        self.assertEqual(evidence["last_checkpoint_step"], 3)
        self.assertEqual(evidence["last_logged_step"], 4)
        self.assertEqual(
            evidence["logged_but_not_checkpointed"],
            {
                "after_step": 3,
                "through_step": 4,
                "step_count": 1,
                "included_in_final_weights": False,
            },
        )

    def test_finalization_rejects_best_selection_mismatch(self) -> None:
        config, rows, selected_payload, last_payload = self._evidence()
        last_payload["training_state"]["best_selection"][  # type: ignore[index]
            "dice"
        ] = 0.7
        with self.assertRaisesRegex(
            ValueError, "best/last checkpoint 的 best_selection"
        ):
            _validate_finalization_evidence(
                config=config,
                rows=rows,
                selected_payload=selected_payload,
                last_payload=last_payload,
            )

    def test_finalization_log_accepts_same_step_and_rejects_regression(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_log.jsonl"
            path.write_text(
                '{"step":1}\n{"step":1,"checkpoint_last":{}}\n'
                '{"step":2}\n',
                encoding="utf-8",
            )
            self.assertEqual(len(_read_finalization_log(path)), 3)
            path.write_text(
                '{"step":2}\n{"step":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "单调不下降"):
                _read_finalization_log(path)

    def test_metric_replay_delta_records_values_without_gate(self) -> None:
        stored = self._metrics(dice=0.6, loss=0.4)
        fresh = self._metrics(dice=0.61, loss=0.39)
        delta = _metric_replay_delta(stored, fresh)
        self.assertAlmostEqual(delta["loss"], -0.01)
        self.assertAlmostEqual(delta["overall"]["dice"], 0.01)

    def test_finalize_writes_atomic_report_without_training_calls(self) -> None:
        config, rows, selected_payload, last_payload = self._evidence()
        train_metrics = self._metrics(sample_count=3)
        validation_metrics = self._metrics(sample_count=2)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            best_path = output_dir / "checkpoint_best.pt"
            last_path = output_dir / "checkpoint_last.pt"
            log_path = output_dir / "train_log.jsonl"
            best_path.touch()
            last_path.touch()
            log_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            model = SimpleNamespace(
                backbone_sha256="b" * 64,
                modality_weight_order=("__null__",),
                region_feature_dim=10,
                model_contract=lambda: {"architecture": "synthetic"},
            )
            loaders = [
                SimpleNamespace(dataset=[0, 1, 2]),
                SimpleNamespace(dataset=[0, 1]),
            ]
            with (
                patch(
                    "oa_groundrag.training.segmentation.finalization.resolve_runtime",
                    return_value=(
                        output_dir / "benchmark",
                        output_dir,
                        output_dir / "backbone.pt",
                        torch.device("cpu"),
                    ),
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.benchmark_contract_from_root",
                    return_value={"index_sha256": "i" * 64},
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.read_checkpoint",
                    side_effect=[selected_payload, last_payload],
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.model_from_checkpoint",
                    return_value=model,
                ),
                patch("oa_groundrag.training.segmentation.finalization._validate_inference_config"),
                patch("oa_groundrag.training.segmentation.finalization._validate_benchmark_registry"),
                patch(
                    "oa_groundrag.training.segmentation.finalization.make_dataloader",
                    side_effect=loaders,
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.evaluate_model",
                    side_effect=[train_metrics, validation_metrics],
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.checkpoint_reload_difference",
                    return_value={"mask_probability": 0.0},
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization._acceptance_collated",
                    return_value={},
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.sha256_file",
                    return_value="f" * 64,
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.make_optimizer",
                    side_effect=AssertionError("finalize 不能创建 optimizer"),
                ),
                patch(
                    "oa_groundrag.training.segmentation.finalization.save_training_checkpoint",
                    side_effect=AssertionError("finalize 不能保存 checkpoint"),
                ),
            ):
                report = finalize_training_run(
                    config,
                    repo_root=REPO_ROOT,
                    checkpoint_path=best_path,
                    termination_reason="project_owner_manual_stop",
                )
            self.assertEqual(
                report["schema_version"], TRAINING_REPORT_SCHEMA_VERSION
            )
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["resume_required"])
            self.assertFalse(report["gate_a_evaluated"])
            self.assertFalse(report["formal_acceptance"])
            self.assertFalse(report["test_evaluated"])
            self.assertTrue(report["engineering_checks_passed"])
            self.assertTrue((output_dir / "training_report.json").is_file())
            self.assertEqual(best_path.stat().st_size, 0)
            self.assertEqual(last_path.stat().st_size, 0)

    def test_finalize_refuses_existing_report(self) -> None:
        config = self._config()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            best_path = output_dir / "checkpoint_best.pt"
            (output_dir / "checkpoint_last.pt").touch()
            (output_dir / "train_log.jsonl").touch()
            (output_dir / "training_report.json").write_text(
                "{}\n", encoding="utf-8"
            )
            best_path.touch()
            with patch(
                "oa_groundrag.training.segmentation.finalization.resolve_runtime",
                return_value=(
                    output_dir / "benchmark",
                    output_dir,
                    output_dir / "backbone.pt",
                    torch.device("cpu"),
                ),
            ):
                with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                    finalize_training_run(
                        config,
                        repo_root=REPO_ROOT,
                        checkpoint_path=best_path,
                        termination_reason="project_owner_manual_stop",
                    )

    def test_finalize_cli_defaults_to_compact_report(self) -> None:
        from oa_groundrag.training.segmentation import cli as cli_module

        report = {
            "status": "completed",
            "completion_mode": "project_owner_manual_stop",
            "selected_checkpoint_step": 2,
            "last_checkpoint_step": 3,
            "last_logged_step": 4,
            "train_metrics": self._metrics(sample_count=3),
            "validation_metrics": self._metrics(sample_count=2),
            "peak_cuda_memory_gib": None,
            "engineering_checks_passed": True,
            "checkpoint": "/tmp/checkpoint_best.pt",
            "training_report": "/tmp/training_report.json",
        }
        compact = format_compact_finalization_report(report)
        with (
            patch.object(cli_module, "load_runtime_config", return_value=object()),
            patch.object(
                cli_module,
                "finalize_training_run",
                return_value=report,
            ) as finalize,
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = cli_module.main(
                    [
                        "finalize",
                        "--config",
                        "unused.json",
                        "--checkpoint",
                        "best.pt",
                        "--termination-reason",
                        "project_owner_manual_stop",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue().strip(), compact)
        self.assertEqual(
            finalize.call_args.kwargs["termination_reason"],
            "project_owner_manual_stop",
        )


if __name__ == "__main__":
    unittest.main()
