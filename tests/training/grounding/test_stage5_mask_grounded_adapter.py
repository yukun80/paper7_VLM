"""Stage 5 split、90/10 replay、warm-start 与非正式报告回归。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from oa_groundrag.data.rs_general.io import atomic_write_json, sha256_file
from oa_groundrag.training.grounding.config import (
    STAGE5_CONFIG_SCHEMA_V3,
    WarmStartContract,
    load_stage5_config,
    verify_retention_reference_files,
    verify_warm_start_files,
)
from oa_groundrag.training.grounding.data import (
    REGION_TRAIN_ROLE,
    Stage5MixedDataset,
    Stage5MixedSampler,
    split_compact_by_parent,
)
from oa_groundrag.training.grounding.workflow import _load_warm_start
from oa_groundrag.training.grounding.workflow import (
    STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS,
    _prepare_stage5_training_memory,
    _stage5_gate_b_protocol_path,
    _stage5_gate_b_selection_path,
    _stage5_rs_general_predictions_path,
    run_stage5_workflow,
)
from oa_groundrag.evaluation.rs_general.contracts import load_gate_b_protocol
from oa_groundrag.evaluation.rs_general.selection import (
    load_gate_b_selection,
    load_gate_b_selection_for_stage5_retention,
)
from oa_groundrag.evaluation.grounding.adapter import (
    _counterfactual_kind,
    evaluate_stage5_dev,
)
from oa_groundrag.vlm.errors import (
    ConfigError,
    ContractError,
    ModelError,
    PredictionError,
)
from oa_groundrag.training.vlm.trainer import training_layout_identity
from oa_groundrag.training.vlm.validation import ValidationResult, validation_is_better


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/vlm/grounded/train_v2.yaml"
QWEN35_CONFIG = REPO / "configs/vlm/grounded/train_qwen35_4b_v3.yaml"
QWEN35_SMOKE_CONFIG = (
    REPO / "configs/vlm/grounded/qwen35_4b_bounded_train_smoke_v3.yaml"
)


class FakeDataset:
    def __init__(self, records, payload="fixture"):
        self.records = tuple(records)
        self.manifest = {"payload_root_sha256": payload, "compact_id": payload}
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return index


class Stage5DataTests(unittest.TestCase):
    def test_parent_split_is_deterministic_disjoint_and_forces_experts_train(self) -> None:
        records = []
        for index in range(200):
            records.append({
                "record_id": f"r-{index}",
                "parent_id": f"p-{index}",
                "source": f"source-{index % 5}",
                "supervision_authority": "expert_verified" if index < 5 else "model_generated_unreviewed",
            })
        dataset = FakeDataset(records)
        first = split_compact_by_parent(dataset)
        second = split_compact_by_parent(dataset)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_parents) & set(first.monitor_parents))
        self.assertEqual(len(first.monitor_parents), 20)
        self.assertTrue(set(first.expert_parents) <= set(first.train_parents))

    def test_sampler_is_exact_90_10_parent_aware_and_resume_stable(self) -> None:
        region = FakeDataset([
            {
                "record_id": f"region-{index}",
                "parent_id": f"region-parent-{index % 300}",
                "source": f"region-source-{index % 5}",
                "logical_role": REGION_TRAIN_ROLE,
                "task_family": "mask_grounded_region_description",
            }
            for index in range(600)
        ], payload="region")
        tasks = (
            "global_caption", "visual_qa", "bbox_region_caption", "scene_understanding",
            "object_count", "spatial_relation", "visible_change_report",
        )
        replay = FakeDataset([
            {
                "record_id": f"replay-{index}",
                "parent_id": f"replay-parent-{index}",
                "source": f"replay-source-{index % 3}",
                "logical_role": "external_train",
                "task_family": tasks[index % len(tasks)],
            }
            for index in range(2_100)
        ], payload="replay")
        mixed = Stage5MixedDataset(region, replay)
        sampler = Stage5MixedSampler(mixed)
        first = list(sampler)
        self.assertEqual(len(first), 8_000)
        self.assertEqual(sum(index < mixed.region_count for index in first), 7_200)
        self.assertEqual(sum(index >= mixed.region_count for index in first), 800)
        replay_indices = [index - mixed.region_count for index in first if index >= mixed.region_count]
        self.assertEqual(len({replay.records[index]["parent_id"] for index in replay_indices}), 800)
        self.assertEqual({replay.records[index]["task_family"] for index in replay_indices}, set(tasks))
        state = sampler.state_dict()
        restored = Stage5MixedSampler(mixed)
        restored.load_state_dict(state)
        self.assertEqual(first, list(restored))
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)

    def test_config_freezes_confirmed_hyperparameters_and_warm_identity(self) -> None:
        config = load_stage5_config(CONFIG)
        self.assertEqual(config.data_contract.split_seed, 20260808)
        self.assertEqual(config.data_contract.region_micro_ratio, 0.9)
        self.assertEqual(config.training.learning_rate, 5e-5)
        self.assertEqual(config.training.warmup_ratio, 0.05)
        self.assertEqual(config.limits.max_input_tokens, 4096)
        identity = verify_warm_start_files(config)
        self.assertEqual(identity["checkpoint_step"], 1000)

    def test_v3_binds_qwen35_gate_b_and_preserves_v2_semantic_identity(self) -> None:
        legacy = load_stage5_config(CONFIG)
        self.assertEqual(
            legacy.semantic_sha256,
            "0d334601208a60af28ab32922c6684673bb9687504e6d028e517cf14441afcf5",
        )
        formal = load_stage5_config(QWEN35_CONFIG)
        smoke = load_stage5_config(QWEN35_SMOKE_CONFIG)
        self.assertEqual(formal.schema_version, STAGE5_CONFIG_SCHEMA_V3)
        self.assertEqual(formal.model.backend, "qwen3_5")
        self.assertEqual(formal.training.learning_rate, 5e-5)
        self.assertEqual(formal.training.max_steps, 1000)
        self.assertEqual(formal.adaptation.rank, 8)
        self.assertEqual(formal.adaptation.target_modules, (
            "q_proj", "k_proj", "v_proj", "o_proj",
        ))
        self.assertNotEqual(formal.workflow_root, smoke.workflow_root)
        self.assertNotEqual(formal.run.name, smoke.run.name)
        self.assertNotEqual(formal.semantic_sha256, smoke.semantic_sha256)
        warm = verify_warm_start_files(formal)
        self.assertEqual(warm["checkpoint_step"], 1000)
        retention = verify_retention_reference_files(formal)
        self.assertTrue(retention["explicit_identity_binding"])
        self.assertEqual(retention["max_new_tokens"], 768)

    def test_v3_retention_identity_tamper_fails_closed(self) -> None:
        config = load_stage5_config(QWEN35_CONFIG)
        assert config.retention_contract is not None
        tampered = replace(
            config,
            retention_contract=replace(
                config.retention_contract,
                gate_b_report_file_sha256="0" * 64,
            ),
        )
        with self.assertRaises(ConfigError):
            verify_retention_reference_files(tampered)

    def test_bounded_smoke_accepts_only_one_or_twenty_steps(self) -> None:
        for value in (True, 0, 2, 19, 21, 1000):
            with self.subTest(value=value), self.assertRaises(ModelError):
                run_stage5_workflow(QWEN35_CONFIG, stop_after_steps=value)

    def test_stage5_cache_policy_extends_only_stage5_checkpoint_layout(self) -> None:
        config = load_stage5_config(CONFIG)
        generic = training_layout_identity(config)
        stage5 = training_layout_identity(
            config,
            cuda_cache_cleanup_interval_steps=(
                STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS
            ),
        )
        self.assertNotIn("cuda_cache_cleanup_interval_steps", generic)
        self.assertEqual(stage5["cuda_cache_cleanup_interval_steps"], 1)
        with self.assertRaises(ModelError):
            training_layout_identity(
                config,
                cuda_cache_cleanup_interval_steps=0,
            )

    def test_stage5_pretraining_memory_cleanup_runs_gc_then_cuda_cleanup(self) -> None:
        calls = []
        with patch(
            "oa_groundrag.training.grounding.workflow.gc.collect",
            side_effect=lambda: calls.append("gc"),
        ), patch(
            "oa_groundrag.training.grounding.workflow.clear_cuda_cache",
            side_effect=lambda device: calls.append(f"cache:{device.type}"),
        ):
            _prepare_stage5_training_memory(torch.device("cuda"))
        self.assertEqual(calls, ["gc", "cache:cuda"])

    def test_retention_uses_static_gate_b_yaml_not_frozen_snapshot(self) -> None:
        config = load_stage5_config(CONFIG)
        protocol_path = _stage5_gate_b_protocol_path(config)
        self.assertEqual(
            protocol_path,
            REPO / "configs/vlm/rs_general/rs_generaldesc_gate_b_qwen3vl_2b.yaml",
        )
        self.assertEqual(protocol_path.suffix, ".yaml")
        self.assertEqual(
            load_gate_b_protocol(protocol_path).raw["protocol_id"],
            "rs_generaldesc_gate_b_qwen3vl_2b_v1",
        )
        self.assertEqual(
            _stage5_gate_b_selection_path(config),
            REPO
            / "outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1/selection/gate_b_selection.json",
        )

    def test_qwen35_retention_paths_are_config_driven(self) -> None:
        config = load_stage5_config(QWEN35_CONFIG)
        self.assertEqual(
            _stage5_gate_b_protocol_path(config),
            REPO / "configs/vlm/rs_general/rs_generaldesc_gate_b_qwen35_4b_v2.yaml",
        )
        self.assertEqual(
            _stage5_gate_b_selection_path(config),
            REPO
            / "outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen35_4b_r851bf6e8_v1/selection/gate_b_selection.json",
        )
        self.assertEqual(
            _stage5_rs_general_predictions_path(config),
            REPO
            / "outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen35_4b_r851bf6e8_v1/adapter/predictions.jsonl",
        )

    def test_retention_loader_preserves_frozen_selection_with_code_drift(self) -> None:
        config = load_stage5_config(CONFIG)
        selection_root = (
            REPO
            / "outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1/selection"
        )
        with self.assertRaises(ContractError):
            load_gate_b_selection(
                _stage5_gate_b_protocol_path(config),
                selection_root / "gate_b_selection.json",
            )
        context = load_gate_b_selection_for_stage5_retention(
            _stage5_gate_b_protocol_path(config),
            selection_root / "gate_b_selection.json",
        )
        self.assertEqual(context.selection["sample_count"], 256)
        self.assertFalse(context.historical_implementation_match)

    def test_monitor_best_tie_uses_earlier_step(self) -> None:
        earlier = ValidationResult(100, 1.0, 1.0, {"region": 1.0}, {"region": 1}, 1, 1, 1, 1, 0.1)
        later = ValidationResult(200, 1.0, 1.0, {"region": 1.0}, {"region": 1}, 1, 1, 1, 1, 0.1)
        self.assertFalse(validation_is_better(later, earlier))


class WarmStartTests(unittest.TestCase):
    def test_warm_start_loads_only_lora_and_declares_fresh_training_state(self) -> None:
        config = load_stage5_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            checkpoint = run / "checkpoints" / "step-00001000"
            adapter = checkpoint / "adapter" / "adapter_model.safetensors"
            adapter.parent.mkdir(parents=True)
            tensors = {"lora.weight": torch.ones(2, 2)}
            save_file(tensors, str(adapter))
            processor_identity = {"fixture": True}
            model_identity = {"model_type": "fixture"}
            atomic_write_json(run / "best_checkpoint.json", {"step": 1000})
            atomic_write_json(checkpoint / "manifest.json", {
                "cursor": {"global_step": 1000},
                "model_identity": model_identity,
                "processor_identity": processor_identity,
                "trainable_parameter_names": ["lora.weight"],
            })
            warm = WarmStartContract(
                checkpoint_root=checkpoint,
                best_pointer_sha256=sha256_file(run / "best_checkpoint.json"),
                checkpoint_step=1000,
                checkpoint_manifest_sha256=sha256_file(checkpoint / "manifest.json"),
                adapter_sha256=sha256_file(adapter),
            )
            local = replace(config, warm_start=warm)

            class Identity:
                def to_dict(self):
                    return model_identity

            class Model:
                identity = Identity()
                trainable_names = ("lora.weight",)

                def load_trainable_state_dict(self, state):
                    self.loaded = dict(state)

            model = Model()
            result = _load_warm_start(model, processor_identity, local)
            self.assertEqual(set(model.loaded), {"lora.weight"})
            self.assertEqual(
                result["fresh_components"],
                ["optimizer", "scheduler", "rng", "sampler"],
            )
            self.assertNotIn("optimizer", result["loaded_components"])

    def test_qwen35_rejects_qwen3_vl_warm_start_before_tensor_load(self) -> None:
        qwen35 = load_stage5_config(QWEN35_CONFIG)
        qwen3_vl = load_stage5_config(CONFIG)
        cross_family = replace(qwen35, warm_start=qwen3_vl.warm_start)

        class Identity:
            def to_dict(self):
                return {"backend": "qwen3_5"}

        class Model:
            identity = Identity()
            trainable_names = ()

        with self.assertRaises(ModelError):
            _load_warm_start(Model(), {"backend": "qwen3_5"}, cross_family)


class Stage5ReportTests(unittest.TestCase):
    def test_baseline_counterfactual_null_and_variant_kind(self) -> None:
        self.assertIsNone(_counterfactual_kind({"counterfactual": None}))
        self.assertEqual(
            _counterfactual_kind({"counterfactual": {"kind": "empty_mask"}}),
            "empty_mask",
        )
        with self.assertRaises(PredictionError):
            _counterfactual_kind({"counterfactual": {}})

    def test_automatic_report_can_never_claim_formal_or_retention_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root = root / "eval"
            prediction_root = root / "predictions"
            eval_root.mkdir()
            prediction_root.mkdir()
            atomic_write_json(eval_root / "manifest.json", {"fixture": True})
            atomic_write_json(prediction_root / "manifest.json", {"fixture": True})
            (prediction_root / "predictions.jsonl").write_text("", encoding="utf-8")

            def fake_evaluate_dev(**kwargs):
                output = Path(kwargs["output_root"])
                output.mkdir(parents=True)
                atomic_write_json(output / "report.json", {"automatic_metrics": {"schema_validity": 0.5}})
                return {"root": str(output)}

            with patch(
                "oa_groundrag.evaluation.grounding.adapter.evaluate_dev",
                side_effect=fake_evaluate_dev,
            ):
                result = evaluate_stage5_dev(
                    eval_root=eval_root,
                    prediction_root=prediction_root,
                    output_root=root / "report",
                    model_role="fixture",
                )
            report = __import__("oa_groundrag.data.rs_general.io", fromlist=["read_json"]).read_json(
                Path(result["root"]) / "report.json"
            )
            self.assertEqual(report["reference_authority"], "automatic_contract_only")
            self.assertFalse(report["expert_metrics_available"])
            self.assertFalse(report["retention_gate_frozen"])
            self.assertFalse(report["formal_acceptance"])
            self.assertFalse(report["scientific_acceptance"])
            self.assertFalse(report["sealed_test_evaluated"])


if __name__ == "__main__":
    unittest.main()
