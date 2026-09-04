"""需要本地 OA-AuxSeg small Benchmark 的集成回归。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from oa_groundrag.data.oa_auxseg.dataset import (
    BenchmarkDataset,
    collate_benchmark_samples,
)
from oa_groundrag.segmentation.checkpoint import (
    INFERENCE_CHECKPOINT_SCHEMA_VERSION,
    model_from_checkpoint,
    read_checkpoint,
    save_training_checkpoint,
)
from oa_groundrag.segmentation.contracts import (
    CONFIG_SCHEMA_VERSION,
    INFERENCE_SCHEMA_VERSION,
    OAAuxSegConfig,
    RuntimeConfig,
    SUPPORTED_AUXILIARY_ORDER,
    ordered_auxiliary_names,
)
from oa_groundrag.segmentation.data import (
    AuxiliarySubsetSampler,
    StatefulTrainingBatcher,
    available_auxiliaries_by_sample,
    benchmark_contract_from_root,
    prepare_collated_batch,
    registry_from_benchmark,
)
from oa_groundrag.segmentation.inference import run_inference
from oa_groundrag.segmentation.model import OAAuxSegModel
from oa_groundrag.training.segmentation.engine import (
    _acceptance_collated,
)
from tests.segmentation.test_oa_auxseg import synthetic_batch, synthetic_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_ROOT = REPO_ROOT.parent / "benchmark" / "oa_auxseg_hdf5_v1" / "small"


class OAAuxSegSmallAssetIntegrationTest(unittest.TestCase):
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

    @unittest.skipUnless(
        (SMALL_ROOT / "index.jsonl").is_file(),
        "requires local oa_auxseg_hdf5_v1/small",
    )
    def test_real_small_sparse_contract_matches_five_sources(self) -> None:
        self.assertTrue((SMALL_ROOT / "index.jsonl").is_file())
        dataset = BenchmarkDataset(
            SMALL_ROOT,
            split="train",
            auxiliary_policy="all",
            normalization="zscore",
        )
        self.assertEqual(len(dataset), 158)
        self.assertEqual(
            sum(bool(row["auxiliaries"]) for row in dataset.rows), 68
        )
        self.assertEqual(
            {str(row["source"]) for row in dataset.rows},
            {
                "gdcld",
                "lmhld",
                "landslidebench_agent",
                "landslide4sense",
                "multimodal_landslide",
            },
        )
        self.assertEqual(
            {
                len(row["optical"]["channel_names"])
                for row in dataset.rows
            },
            {3, 4, 12},
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
        benchmark_contract = benchmark_contract_from_root(SMALL_ROOT)
        self.assertEqual(
            benchmark_contract["index_sha256"],
            "85c86a8d09dc2f602f04dad890ca65dab59e16235415e818427673dc1483f2df",
        )
        self.assertEqual(
            benchmark_contract["included_sources"],
            [
                "gdcld",
                "lmhld",
                "landslidebench_agent",
                "landslide4sense",
                "multimodal_landslide",
            ],
        )
        self.assertEqual(
            benchmark_contract["excluded_sources"],
            ["sen12landslides"],
        )
        selected_sources = (
            "gdcld",
            "lmhld",
            "landslidebench_agent",
            "landslide4sense",
            "multimodal_landslide",
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
        self.assertEqual(tuple(prepared.mask.shape), (5, 1, 224, 224))
        self.assertEqual(
            {len(names) for names in prepared.model.optical_channel_names},
            {3, 4, 12},
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
        acceptance = _acceptance_collated(
            SMALL_ROOT,
            normalization="zscore",
            batch_size=8,
        )
        repeated = _acceptance_collated(
            SMALL_ROOT,
            normalization="zscore",
            batch_size=8,
        )
        self.assertEqual(
            acceptance["sample_id"],
            repeated["sample_id"],
        )
        accepted = prepare_collated_batch(acceptance)
        self.assertEqual(
            {metadata["source"] for metadata in accepted.metadata},
            set(benchmark_contract["included_sources"]),
        )
        self.assertEqual(
            {len(names) for names in accepted.model.optical_channel_names},
            {3, 4, 12},
        )
        accepted_combinations = set(
            available_auxiliaries_by_sample(accepted.model)
        )
        self.assertIn(tuple(), accepted_combinations)
        self.assertIn(("dem", "slope"), accepted_combinations)
        self.assertIn(
            ("dem", "insar_velocity"),
            accepted_combinations,
        )

    @unittest.skipUnless(
        (SMALL_ROOT / "index.jsonl").is_file(),
        "requires local oa_auxseg_hdf5_v1/small",
    )
    def test_training_batcher_has_reproducible_sequence(self) -> None:
        batcher = StatefulTrainingBatcher(
            SMALL_ROOT,
            batch_size=8,
            normalization="zscore",
            seed=123,
            policy="uniform",
        )
        first = [batcher.next_indices() for _ in range(27)]
        self.assertTrue(all(len(indices) == 8 for indices in first))
        self.assertEqual(batcher.samples_emitted, 216)
        state = batcher.state_dict()
        expected = [batcher.next_indices() for _ in range(5)]
        restored = StatefulTrainingBatcher(
            SMALL_ROOT,
            batch_size=8,
            normalization="zscore",
            seed=999,
            policy="uniform",
        )
        restored.load_state_dict(state)
        self.assertEqual(
            expected,
            [restored.next_indices() for _ in range(5)],
        )

        balanced = StatefulTrainingBatcher(
            SMALL_ROOT,
            batch_size=8,
            normalization="zscore",
            seed=123,
            policy="balanced_target_presence",
        )
        indices = balanced.next_indices()
        positive = sum(
            float(balanced.dataset.rows[index]["foreground_ratio"]) > 0
            for index in indices
        )
        self.assertEqual(positive, 4)
        self.assertEqual(len(indices) - positive, 4)

    @unittest.skipUnless(
        (SMALL_ROOT / "index.jsonl").is_file(),
        "requires local oa_auxseg_hdf5_v1/small",
    )
    def test_proposed_sampling_preserves_native_auxiliary_exposure(self) -> None:
        batcher = StatefulTrainingBatcher(
            SMALL_ROOT,
            batch_size=8,
            normalization="zscore",
            seed=20260725,
            policy="uniform",
        )
        sampler = AuxiliarySubsetSampler(20260726, 0.2)
        native_auxiliary_samples = 0
        active_auxiliary_samples = 0
        for _ in range(300):
            available = [
                ordered_auxiliary_names(
                    tuple(batcher.dataset.rows[index]["auxiliaries"])
                )
                for index in batcher.next_indices()
            ]
            active = sampler.sample(available)
            native_auxiliary_samples += sum(bool(item) for item in available)
            active_auxiliary_samples += sum(bool(item) for item in active)
            self.assertTrue(
                all(
                    bool(selected) == bool(present)
                    for selected, present in zip(
                        active, available, strict=True
                    )
                )
            )
        self.assertEqual(sum(sampler.counts.values()), 2400)
        self.assertEqual(sum(sampler.reason_counts.values()), 2400)
        active_count = sum(
            sampler.counts[name]
            for name in ("single", "multi", "all")
        )
        self.assertEqual(active_count, native_auxiliary_samples)
        self.assertEqual(
            active_auxiliary_samples,
            native_auxiliary_samples,
        )
        self.assertGreater(sampler.reason_counts["native_none"], 0)
        self.assertGreater(sampler.reason_counts["active"], 0)
        self.assertGreater(
            sampler.reason_counts["dropout_restored"],
            0,
        )
        self.assertGreater(sampler.counts["single"], 0)
        self.assertGreater(
            sampler.counts["multi"] + sampler.counts["all"],
            0,
        )

    @unittest.skipUnless(
        (SMALL_ROOT / "manifest.json").is_file(),
        "requires local oa_auxseg_hdf5_v1/small",
    )
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
        benchmark_contract = benchmark_contract_from_root(SMALL_ROOT)
        with torch.no_grad():
            before = self.model(self.batch)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=self.model,
                step=1,
                benchmark_contract=benchmark_contract,
                selection_metrics={"validation_dice": 0.5},
            )
            payload = read_checkpoint(
                path,
                expected_benchmark_contract=benchmark_contract,
            )
            self.assertEqual(
                payload["schema_version"], INFERENCE_CHECKPOINT_SCHEMA_VERSION
            )
            self.assertEqual(
                payload["benchmark_contract"],
                benchmark_contract,
            )
            mismatched_contract = dict(benchmark_contract)
            mismatched_contract["manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "Benchmark 合同不一致"):
                read_checkpoint(
                    path,
                    expected_benchmark_contract=mismatched_contract,
                )
            legacy_path = Path(temporary) / "legacy_checkpoint.pt"
            legacy_payload = dict(payload)
            legacy_payload["schema_version"] = "oa_auxseg_checkpoint_v5"
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
                    self.assertEqual(arrays[feature_key].shape[1:], (268,))
                    self.assertEqual(
                        arrays[
                            rows[0]["array_keys"]["modality_weights"]
                        ].shape,
                        (4,),
                    )
                    for stride, shape in zip(
                        (4, 8, 16, 32),
                        (
                            (4, 56, 56),
                            (4, 28, 28),
                            (4, 14, 14),
                            (4, 7, 7),
                        ),
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
                    manifest["schema_version"], INFERENCE_SCHEMA_VERSION
                )
                self.assertEqual(manifest["region_feature_dim"], 268)
                self.assertEqual(
                    manifest["benchmark_contract"],
                    benchmark_contract,
                )
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


if __name__ == "__main__":
    unittest.main()
