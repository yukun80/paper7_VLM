"""Phase 2 OA-AuxSeg 多模态合同、梯度、重载与区域测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np
from torchvision.models import convnext_small

from oa_groundrag.phase2.checkpoint import (
    model_from_checkpoint,
    read_checkpoint,
    save_training_checkpoint,
)
from oa_groundrag.phase2.contracts import (
    CONFIG_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    ModelRegistry,
    OAAuxSegBatch,
    OAAuxSegConfig,
    PackedAuxiliary,
    RuntimeConfig,
    SUPPORTED_AUXILIARY_ORDER,
    VARIANTS,
)
from oa_groundrag.phase2.data import (
    AuxiliarySubsetSampler,
    available_auxiliaries_by_sample,
    filter_auxiliaries,
    prepare_collated_batch,
    registry_from_benchmark,
)
from oa_groundrag.phase2.engine import (
    make_optimizer,
    make_scaler,
    make_scheduler,
    require_local_backbone,
    run_inference,
)
from oa_groundrag.phase2.losses import bce_dice_loss
from oa_groundrag.phase2.metrics import SegmentationMetrics
from oa_groundrag.phase2.model import (
    BACKBONE_STAGE_DEPTHS,
    EXPECTED_BACKBONE_SHA256,
    OAAuxSegModel,
)
from oa_groundrag.phase2.fusion import (
    CMNeXtSpatialSelector,
    SparseStageFeature,
)
from oa_groundrag.phase2.regions import extract_regions_and_features
from oa_groundrag.phase2.validity import (
    resample_validity,
    validity_from_channels,
)
from scripts.phase1_benchmark_build.benchmark_common import (
    BenchmarkDataset,
    collate_benchmark_samples,
    sha256_file,
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
    sen12 = (
        "s2_b02",
        "s2_b03",
        "s2_b04",
        "s2_b05",
        "s2_b06",
        "s2_b07",
        "s2_b08",
        "s2_b8a",
        "s2_b11",
        "s2_b12",
    )
    return ModelRegistry(
        optical_signatures=tuple(
            sorted((rgb_short, rgb, nir, sen12, sentinel))
        ),
        auxiliary_channels={
            "dem": ("DEM",),
            "insar_velocity": ("InSAR_mean_LOS_velocity_encoded",),
            "sar_ascending": ("s1asc_vv", "s1asc_vh"),
            "sar_descending": ("s1dsc_vv", "s1dsc_vh"),
            "slope": ("slope",),
        },
        available_auxiliaries={
            rgb_short: (),
            rgb: ("dem", "insar_velocity"),
            nir: (),
            sen12: ("dem", "sar_ascending", "sar_descending"),
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
        (
            "s2_b02",
            "s2_b03",
            "s2_b04",
            "s2_b05",
            "s2_b06",
            "s2_b07",
            "s2_b08",
            "s2_b8a",
            "s2_b11",
            "s2_b12",
        ),
        (
            "s2_b02",
            "s2_b03",
            "s2_b04",
            "s2_b05",
            "s2_b06",
            "s2_b07",
            "s2_b08",
            "s2_b8a",
            "s2_b11",
            "s2_b12",
        ),
        (
            "s2_b02",
            "s2_b03",
            "s2_b04",
            "s2_b05",
            "s2_b06",
            "s2_b07",
            "s2_b08",
            "s2_b8a",
            "s2_b11",
            "s2_b12",
        ),
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
            "dem": packed([2, 3, 4, 5, 6], ("DEM",)),
            "insar_velocity": packed(
                [3], ("InSAR_mean_LOS_velocity_encoded",)
            ),
            "sar_ascending": packed(
                [4, 6], ("s1asc_vv", "s1asc_vh")
            ),
            "sar_descending": packed(
                [5, 6], ("s1dsc_vv", "s1dsc_vh")
            ),
            "slope": packed([2], ("slope",)),
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

    def test_real_small_sparse_contract_includes_sen12_sar(self) -> None:
        self.assertTrue((SMALL_ROOT / "index.jsonl").is_file())
        dataset = BenchmarkDataset(
            SMALL_ROOT,
            split="train",
            auxiliary_policy="all",
            normalization="zscore",
        )
        self.assertEqual(len(dataset), 204)
        self.assertEqual(
            sum(bool(row["auxiliaries"]) for row in dataset.rows), 114
        )
        self.assertEqual(
            {str(row["source"]) for row in dataset.rows},
            {
                "gdcld",
                "lmhld",
                "landslidebench_agent",
                "landslide4sense",
                "multimodal_landslide",
                "sen12landslides",
            },
        )
        self.assertEqual(
            {
                len(row["optical"]["channel_names"])
                for row in dataset.rows
            },
            {3, 4, 10, 12},
        )
        registry = registry_from_benchmark(SMALL_ROOT)
        self.assertEqual(registry, self.registry)
        self.assertEqual(
            registry.auxiliary_order, SUPPORTED_AUXILIARY_ORDER
        )
        self.assertEqual(
            registry.modality_weight_order,
            (*SUPPORTED_AUXILIARY_ORDER, "__null__"),
        )
        selected_sources = (
            "gdcld",
            "lmhld",
            "landslidebench_agent",
            "landslide4sense",
            "multimodal_landslide",
            "sen12landslides",
        )
        selected = [
            next(
                dataset[index]
                for index, row in enumerate(dataset.rows)
                if row["source"] == source
            )
            for source in selected_sources
        ]
        prepared = prepare_collated_batch(
            collate_benchmark_samples(selected)
        )
        self.assertEqual(tuple(prepared.mask.shape), (6, 1, 224, 224))
        self.assertEqual(
            {len(names) for names in prepared.model.optical_channel_names},
            {3, 4, 10, 12},
        )
        self.assertEqual(
            set(prepared.model.auxiliaries),
            set(SUPPORTED_AUXILIARY_ORDER),
        )
        self.assertTrue(
            any(
                packed.values.shape[0] < prepared.model.batch_size
                for packed in prepared.model.auxiliaries.values()
            )
        )
        for signature in prepared.model.optical_channel_names:
            self.assertIn(signature, registry.optical_signatures)
        for name, packed in prepared.model.auxiliaries.items():
            self.assertEqual(
                packed.channel_names,
                registry.auxiliary_channels[name],
            )

    def test_all_variants_forward_variable_channels_and_outputs(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.model.variant = variant
                output = self.model(self.batch, return_regions=True)
                self.assertEqual(tuple(output.mask_logits.shape), (7, 1, 32, 32))
                self.assertEqual(
                    tuple(output.mask_probability.shape), (7, 1, 32, 32)
                )
                self.assertEqual(tuple(output.no_target_score.shape), (7,))
                self.assertEqual(tuple(output.modality_weights.shape), (7, 6))
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
                        (7, 6, 8, 8),
                        (7, 6, 4, 4),
                        (7, 6, 2, 2),
                        (7, 6, 1, 1),
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
                    torch.ones(7),
                    rtol=0,
                    atol=1e-6,
                )
                self.assertEqual(
                    output.modality_names,
                    self.registry.modality_weight_order,
                )
                self.assertEqual(len(output.candidate_regions or []), 7)
                self.assertEqual(len(output.region_features or []), 7)
                for feature in output.region_features or []:
                    self.assertEqual(feature.shape[1:], (270,))

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

    def test_sar_orbits_remain_sparse_and_independent(self) -> None:
        available = available_auxiliaries_by_sample(self.batch)
        self.assertEqual(available[4], ("dem", "sar_ascending"))
        self.assertEqual(available[5], ("dem", "sar_descending"))
        self.assertEqual(
            available[6],
            ("dem", "sar_ascending", "sar_descending"),
        )
        self.model.variant = "mean_auxiliary_fusion"
        with torch.no_grad():
            output = self.model(self.batch)
        ascending = output.modality_names.index("sar_ascending")
        descending = output.modality_names.index("sar_descending")
        self.assertGreater(
            float(output.modality_weights[4, ascending].item()), 0.0
        )
        self.assertEqual(
            float(output.modality_weights[4, descending].item()), 0.0
        )
        self.assertEqual(
            float(output.modality_weights[5, ascending].item()), 0.0
        )
        self.assertGreater(
            float(output.modality_weights[5, descending].item()), 0.0
        )
        self.assertGreater(
            float(output.modality_weights[6, ascending].item()), 0.0
        )
        self.assertGreater(
            float(output.modality_weights[6, descending].item()), 0.0
        )

    def test_invalid_values_cannot_affect_features(self) -> None:
        sar = self.batch.auxiliaries["sar_ascending"]
        valid = sar.pixel_valid.clone()
        valid[:, :, :, :16] = 0
        zeros = sar.values.clone()
        zeros[:, :, :, :16] = 0
        huge = sar.values.clone()
        huge[:, :, :, :16] = 1e30
        zero_batch = replace_auxiliary(
            self.batch,
            "sar_ascending",
            PackedAuxiliary(
                sar.sample_indices,
                zeros,
                valid,
                sar.channel_valid,
                sar.channel_names,
            ),
        )
        huge_batch = replace_auxiliary(
            self.batch,
            "sar_ascending",
            PackedAuxiliary(
                sar.sample_indices,
                huge,
                valid,
                sar.channel_valid,
                sar.channel_names,
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
        for name in ("sar_ascending", "sar_descending"):
            with self.subTest(modality=name):
                sar = self.batch.auxiliaries[name]
                zero_coverage = replace_auxiliary(
                    self.batch,
                    name,
                    PackedAuxiliary(
                        sar.sample_indices,
                        torch.full_like(sar.values, 1e30),
                        torch.zeros_like(sar.pixel_valid),
                        torch.zeros_like(sar.channel_valid),
                        sar.channel_names,
                    ),
                )
                with torch.no_grad():
                    output = self.model(zero_coverage)
                sample_index = int(sar.sample_indices[0])
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

    def test_sar_spatial_quality_is_local(self) -> None:
        sar = self.batch.auxiliaries["sar_ascending"]
        local_valid = sar.pixel_valid.clone()
        local_valid[:, :, :, :16] = 0
        localized = replace_auxiliary(
            self.batch,
            "sar_ascending",
            PackedAuxiliary(
                sar.sample_indices,
                sar.values,
                local_valid,
                sar.channel_valid,
                sar.channel_names,
            ),
        )
        with torch.no_grad():
            output = self.model(localized)
        column = output.modality_names.index("sar_ascending")
        sample_index = int(sar.sample_indices[0])
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
        target = torch.zeros(7, 1, 32, 32)
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
        before_sar = {
            name: self.model.auxiliary_adapters[name]
            .projection[0]
            .weight.detach()
            .clone()
            for name in ("sar_ascending", "sar_descending")
        }
        optimizer.step()
        for name, before in before_sar.items():
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
            ("dem", "sar_ascending", "sar_descending"),
        ]
        sampler = AuxiliarySubsetSampler(123, 0.2)
        first = sampler.sample(available)
        state = sampler.state_dict()
        expected_next = sampler.sample(available)
        restored = AuxiliarySubsetSampler(999, 0.2)
        restored.load_state_dict(state)
        self.assertEqual(expected_next, restored.sample(available))
        self.assertEqual(first[0], ())
        for selected, names in zip(first, available, strict=True):
            self.assertTrue(set(selected).issubset(names))

    def test_six_variants_synthetic_optimizer_step(self) -> None:
        target = torch.zeros(7, 1, 32, 32)
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

    def test_checkpoint_strict_reload_is_identical(self) -> None:
        self.model.eval()
        self.model.variant = self.model.config.variant
        self.model.backbone_sha256 = "a" * 64
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
        scheduler = make_scheduler(
            optimizer, total_steps=2, warmup_ratio=config.warmup_ratio
        )
        scaler = make_scaler(torch.device("cpu"))
        benchmark_hash = sha256_file(SMALL_ROOT / "index.jsonl")
        with torch.no_grad():
            before = self.model(self.batch)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=1,
                benchmark_index_sha256=benchmark_hash,
                subset_sampler_state=None,
                dataloader_generator_state=torch.Generator()
                .manual_seed(8)
                .get_state(),
                training_state={
                    "runtime_config": config.to_dict(),
                    "unit_test": True,
                },
            )
            payload = read_checkpoint(
                path, expected_benchmark_index_sha256=benchmark_hash
            )
            self.assertEqual(
                payload["schema_version"], CHECKPOINT_SCHEMA_VERSION
            )
            legacy_path = Path(temporary) / "legacy_checkpoint.pt"
            legacy_payload = dict(payload)
            legacy_payload["schema_version"] = "oa_auxseg_checkpoint_v3"
            torch.save(legacy_payload, legacy_path)
            with self.assertRaisesRegex(ValueError, "只支持"):
                read_checkpoint(legacy_path)
            reloaded = model_from_checkpoint(
                payload, device=torch.device("cpu")
            )
            reloaded.eval()
            with torch.no_grad():
                after = reloaded(self.batch)
            inference_config = RuntimeConfig(
                schema_version=CONFIG_SCHEMA_VERSION,
                benchmark_root=str(SMALL_ROOT),
                output_dir="unused",
                backbone="convnext_small",
                backbone_weights="unused",
                variant="proposed_dropout",
                device="cpu",
                batch_size=1,
                max_steps=1,
            )
            for source in (
                "landslide4sense",
                "multimodal_landslide",
                "sen12landslides",
            ):
                export_dir = Path(temporary) / source
                report = run_inference(
                    inference_config,
                    repo_root=REPO_ROOT,
                    checkpoint_path=path,
                    split="val",
                    source=source,
                    limit=1,
                    output_dir=export_dir,
                )
                self.assertEqual(report["sample_count"], 1)
                rows = [
                    json.loads(line)
                    for line in (export_dir / "predictions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(len(rows), 1)
                with np.load(export_dir / "predictions.npz") as arrays:
                    feature_key = rows[0]["array_keys"]["region_features"]
                    self.assertEqual(arrays[feature_key].shape[1:], (270,))
                    self.assertEqual(
                        arrays[
                            rows[0]["array_keys"]["modality_weights"]
                        ].shape,
                        (6,),
                    )
                    for stride, shape in zip(
                        (4, 8, 16, 32),
                        ((6, 56, 56), (6, 28, 28), (6, 14, 14), (6, 7, 7)),
                        strict=True,
                    ):
                        map_key = rows[0]["array_keys"][
                            "modality_weight_maps"
                        ][str(stride)]
                        self.assertEqual(arrays[map_key].shape, shape)
                manifest = json.loads(
                    (export_dir / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    manifest["schema_version"], "oa_auxseg_inference_v4"
                )
                self.assertEqual(manifest["region_feature_dim"], 270)
                self.assertEqual(
                    tuple(manifest["modality_weight_order"]),
                    self.registry.modality_weight_order,
                )
                self.assertEqual(
                    tuple(manifest["modality_weight_map_strides"]),
                    (4, 8, 16, 32),
                )
                with self.assertRaises(FileExistsError):
                    run_inference(
                        inference_config,
                        repo_root=REPO_ROOT,
                        checkpoint_path=path,
                        split="val",
                        source=source,
                        limit=1,
                        output_dir=export_dir,
                    )
        torch.testing.assert_close(
            before.mask_logits, after.mask_logits, rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            before.mask_probability,
            after.mask_probability,
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            before.modality_weights,
            after.modality_weights,
            rtol=0,
            atol=1e-6,
        )
        for before_map, after_map in zip(
            before.modality_weight_maps,
            after.modality_weight_maps,
            strict=True,
        ):
            torch.testing.assert_close(
                before_map,
                after_map,
                rtol=0,
                atol=1e-6,
            )

    def test_regions_are_deterministic_and_registry_dimensional(self) -> None:
        probability = torch.zeros(1, 1, 32, 32)
        probability[:, :, 2:8, 3:10] = 0.8
        probability[:, :, 20:22, 20:22] = 0.9
        probability = probability.to(torch.bfloat16)
        optical = torch.randn(1, 128, 8, 8)
        fused = torch.randn(1, 128, 8, 8)
        weights = torch.tensor([[0.2, 0.0, 0.1, 0.1, 0.1, 0.5]])
        regions, features = extract_regions_and_features(
            probability=probability,
            optical_feature=optical,
            fused_feature=fused,
            modality_weights=weights,
            expected_feature_dim=270,
            threshold=0.5,
            min_area=16,
        )
        self.assertEqual(len(regions[0]), 1)
        self.assertEqual(regions[0][0].bbox_xyxy, (3, 2, 10, 8))
        self.assertEqual(regions[0][0].area_pixels, 42)
        self.assertEqual(tuple(features[0].shape), (1, 270))
        torch.testing.assert_close(features[0][0, -6:], weights[0])

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
            normalization=self.model.optical_stem_norm,
        )
        generator = torch.Generator().manual_seed(97)
        probe = torch.randn(
            feature.shape,
            generator=generator,
            dtype=feature.dtype,
        )
        (feature * probe).sum().backward()
        gradient = stem.projection.weight.grad
        self.assertIsNotNone(gradient)
        rgb_indices = {3, 2, 1}
        extra_indices = [
            index for index in range(12) if index not in rgb_indices
        ]
        self.assertGreater(
            float(gradient[:, extra_indices].abs().sum().item()),
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
                normalization=model.optical_stem_norm,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.backbone_sha256, EXPECTED_BACKBONE_SHA256)
        self.assertEqual(
            tuple(len(model.backbone[index]) for index in (0, 2, 4, 6)),
            BACKBONE_STAGE_DEPTHS,
        )

    def test_registry_rejects_unknown_sensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "未注册辅助模态"):
            ModelRegistry(
                optical_signatures=self.registry.optical_signatures,
                auxiliary_channels={
                    **self.registry.auxiliary_channels,
                    "unknown_sensor": ("value",),
                },
                available_auxiliaries=self.registry.available_auxiliaries,
            )
        with self.assertRaisesRegex(ValueError, "缺少固定辅助模态列"):
            ModelRegistry(
                optical_signatures=self.registry.optical_signatures,
                auxiliary_channels={
                    name: channels
                    for name, channels in self.registry.auxiliary_channels.items()
                    if name != "slope"
                },
                available_auxiliaries=self.registry.available_auxiliaries,
            )


if __name__ == "__main__":
    unittest.main()
