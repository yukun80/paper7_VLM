from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from PIL import Image

from oa_groundrag.phase4.cli import entrypoint
from oa_groundrag.phase4.config import (
    CONFIG_SCHEMA_VERSION,
    apply_runtime_overrides,
    load_config,
)
from oa_groundrag.phase4.contracts import MaskMode
from oa_groundrag.phase4.data import (
    DescriptionSample,
    ExternalDescriptionDataset,
    inventory_from_auxseg_inference,
    inventory_from_canonical_oa_item,
)
from oa_groundrag.phase4.errors import (
    ConfigError,
    ContractError,
    ModelError,
    PreflightError,
    ProcessingError,
    ReasonCode,
)
from oa_groundrag.phase4.preflight import (
    inspect_benchmark_identity,
    run_preflight,
)
from oa_groundrag.phase4.processing import (
    DescriptionCollator,
    EncodedSample,
    assistant_only_labels,
)
from oa_groundrag.phase4.model import (
    EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS,
    ModelIdentity,
    Qwen3VLModelAdapter,
    assistant_sample_mean_causal_loss,
    local_model_identity,
)
from oa_groundrag.phase4.reference import MAIN_REFERENCE


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs/phase4_mask_grounded_description"


class ConfigAndPreflightTests(unittest.TestCase):
    def test_three_configs_strictly_parse(self) -> None:
        configs = {
            path.name: load_config(path)
            for path in CONFIG_ROOT.glob("*.yaml")
        }
        self.assertEqual(
            set(configs),
            {
                "bounded_smoke.yaml",
                "external_lora_qwen3vl_2b.yaml",
                "prompt_only_qwen3vl_2b.yaml",
            },
        )
        for config in configs.values():
            self.assertEqual(config.schema_version, CONFIG_SCHEMA_VERSION)
            self.assertEqual(len(config.semantic_sha256), 64)
            self.assertTrue(config.model.local_files_only)
            self.assertTrue(config.adaptation.freeze_vision)
            self.assertTrue(config.adaptation.freeze_merger)
        self.assertEqual(
            configs[
                "external_lora_qwen3vl_2b.yaml"
            ].adaptation.target_modules,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        external = configs["external_lora_qwen3vl_2b.yaml"]
        self.assertEqual(external.run.name, "external_lora_qwen3vl_2b_bs4")
        self.assertEqual(external.training.batch_size, 4)
        self.assertEqual(external.training.gradient_accumulation_steps, 4)
        self.assertEqual(
            external.training.batch_size
            * external.training.gradient_accumulation_steps,
            16,
        )
        self.assertEqual(external.training.log_interval, 10)
        self.assertEqual(external.training.validation_interval, 100)
        self.assertEqual(external.training.validation_max_parents, 128)
        self.assertEqual(external.training.num_workers, 4)
        self.assertEqual(external.training.prefetch_factor, 2)
        self.assertTrue(external.training.pin_memory)
        self.assertTrue(
            str(external.run.output_root).startswith(str(REPO / "outputs"))
        )

    def test_training_layout_accepts_only_b1a16_or_b4a4(self) -> None:
        source = yaml.safe_load(
            (
                CONFIG_ROOT / "external_lora_qwen3vl_2b.yaml"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for batch_size, accumulation in ((1, 16), (4, 4)):
                candidate = yaml.safe_load(yaml.safe_dump(source))
                candidate["training"]["batch_size"] = batch_size
                candidate["training"][
                    "gradient_accumulation_steps"
                ] = accumulation
                path = root / f"valid-{batch_size}-{accumulation}.yaml"
                path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False),
                    encoding="utf-8",
                )
                parsed = load_config(path)
                self.assertEqual(parsed.training.batch_size, batch_size)
                self.assertEqual(
                    parsed.training.gradient_accumulation_steps,
                    accumulation,
                )

            for batch_size, accumulation in ((2, 8), (4, 16), (1, 4)):
                candidate = yaml.safe_load(yaml.safe_dump(source))
                candidate["training"]["batch_size"] = batch_size
                candidate["training"][
                    "gradient_accumulation_steps"
                ] = accumulation
                path = root / f"invalid-{batch_size}-{accumulation}.yaml"
                path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError) as caught:
                    load_config(path)
                self.assertEqual(
                    caught.exception.code,
                    ReasonCode.TYPE_MISMATCH,
                )

    def test_input_pipeline_layout_is_strict(self) -> None:
        source = yaml.safe_load(
            (
                CONFIG_ROOT / "external_lora_qwen3vl_2b.yaml"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_sync = yaml.safe_load(yaml.safe_dump(source))
            valid_sync["training"].update(
                {
                    "num_workers": 0,
                    "prefetch_factor": 0,
                    "pin_memory": False,
                }
            )
            valid_path = root / "valid-sync.yaml"
            valid_path.write_text(
                yaml.safe_dump(valid_sync, sort_keys=False),
                encoding="utf-8",
            )
            parsed = load_config(valid_path)
            self.assertEqual(parsed.training.num_workers, 0)
            self.assertEqual(parsed.training.prefetch_factor, 0)
            self.assertFalse(parsed.training.pin_memory)

            invalid_rows = (
                {"num_workers": 0, "prefetch_factor": 2, "pin_memory": False},
                {"num_workers": 2, "prefetch_factor": 0, "pin_memory": True},
                {"num_workers": 2, "prefetch_factor": 2, "pin_memory": False},
                {"num_workers": 5, "prefetch_factor": 2, "pin_memory": True},
            )
            for index, values in enumerate(invalid_rows):
                candidate = yaml.safe_load(yaml.safe_dump(source))
                candidate["training"].update(values)
                path = root / f"invalid-input-{index}.yaml"
                path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError):
                    load_config(path)

    def test_unknown_bool_as_int_nonfinite_and_duplicate_are_rejected(self) -> None:
        source = yaml.safe_load(
            (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            unknown = dict(source)
            unknown["unknown"] = True
            unknown_path = base / "unknown.yaml"
            unknown_path.write_text(
                yaml.safe_dump(unknown, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(unknown_path)
            self.assertEqual(caught.exception.code, ReasonCode.UNKNOWN_FIELD)

            bool_seed = yaml.safe_load(
                (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
            )
            bool_seed["run"]["seed"] = True
            bool_path = base / "bool.yaml"
            bool_path.write_text(
                yaml.safe_dump(bool_seed, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(bool_path)
            self.assertEqual(caught.exception.code, ReasonCode.TYPE_MISMATCH)

            nonfinite = yaml.safe_load(
                (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
            )
            nonfinite["training"]["learning_rate"] = float("nan")
            nonfinite_path = base / "nan.yaml"
            nonfinite_path.write_text(
                yaml.safe_dump(nonfinite, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(nonfinite_path)
            self.assertEqual(
                caught.exception.code,
                ReasonCode.NONFINITE_NUMBER,
            )

            duplicate_path = base / "duplicate.yaml"
            duplicate_path.write_text(
                "schema_version: one\nschema_version: two\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(duplicate_path)

            linked_path = base / "linked.yaml"
            linked_path.symlink_to(unknown_path)
            with self.assertRaises(ConfigError) as caught:
                load_config(linked_path)
            self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_LINK)

    def test_light_external_identity_and_oa_component_gate(self) -> None:
        external = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        identity = inspect_benchmark_identity(external)
        self.assertEqual(
            identity.build_id,
            "build_8adb325c14ed7a8419b7d0e95ab2871ee277c3eac7d0409b3dbf64a9f831f96e",
        )
        self.assertFalse(identity.source_roots_embedded)
        oa = load_config(CONFIG_ROOT / "prompt_only_qwen3vl_2b.yaml")
        with self.assertRaises(PreflightError) as caught:
            run_preflight(oa, require_new_output=False)
        self.assertEqual(
            caught.exception.code,
            ReasonCode.OA_COMPONENT_DISABLED,
        )

    def test_resume_semantic_hash_excludes_output_pointer_only(self) -> None:
        config = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        changed_output = replace(
            config,
            run=replace(
                config.run,
                output_root=Path("/tmp/another-output"),
                resume_checkpoint=Path("/tmp/explicit-checkpoint"),
            ),
        )
        self.assertEqual(
            config.semantic_dict(),
            changed_output.semantic_dict(),
        )
        overridden = apply_runtime_overrides(
            config,
            output_root=Path("/tmp/override-output"),
        )
        self.assertEqual(config.semantic_sha256, overridden.semantic_sha256)
        changed_log = apply_runtime_overrides(config, log_interval=7)
        self.assertNotEqual(
            config.semantic_sha256,
            changed_log.semantic_sha256,
        )

    def test_train_cli_output_override_and_cuda_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "oa_groundrag.phase4.cli.torch.cuda.is_available",
            return_value=False,
        ):
            output = Path(temporary) / "repo-local-run"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = entrypoint(
                    [
                        "train",
                        "--config",
                        str(
                            CONFIG_ROOT
                            / "external_lora_qwen3vl_2b.yaml"
                        ),
                        "--output-root",
                        str(output),
                        "--stop-after-steps",
                        "1",
                        "--log-interval",
                        "5",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("CUDA_REQUIRED", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_main_reference_and_local_model_identity_are_locked(self) -> None:
        self.assertEqual(
            MAIN_REFERENCE.revision,
            "96588727e44c78b25ba03ea03b8e12f7e64fd0da",
        )
        self.assertEqual(MAIN_REFERENCE.license, "Apache-2.0")
        identity = local_model_identity(
            REPO / "models_zoo/Qwen3-VL-2B-Instruct",
            base_parameter_count=EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS,
            model_type="qwen3_vl",
        )
        self.assertEqual(
            identity.base_parameter_count,
            EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS,
        )
        self.assertEqual(len(identity.config_sha256), 64)
        self.assertEqual(len(identity.weights_metadata_sha256), 64)
        self.assertEqual(len(identity.processor_config_sha256), 64)


class FakeProcessor:
    pad_token_id = 0

    def encode_training(self, messages):
        assistant = messages[-1]["content"][0]["text"]
        target = torch.tensor(
            [20 + index for index, _ in enumerate(assistant[:3], 1)],
            dtype=torch.long,
        )
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        full = torch.cat((prompt, target))
        attention = torch.ones_like(full)
        return EncodedSample(
            tensors={
                "input_ids": full,
                "attention_mask": attention,
                "mm_token_type_ids": torch.zeros_like(full),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=assistant_only_labels(
                full,
                prompt,
                attention_mask=attention,
            ),
            input_token_count=int(full.numel()),
            image_count=1,
        )

    def encode_inference(self, messages):
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        return EncodedSample(
            tensors={
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.zeros_like(ids),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=None,
            input_token_count=3,
            image_count=1,
        )


def sample(record_id: str, answer: str) -> DescriptionSample:
    return DescriptionSample(
        record_id=record_id,
        parent_id=f"parent-{record_id}",
        logical_role="external_train",
        task_family="visual_qa",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/fixture.png"},
                    {"type": "text", "text": "Question?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ),
        reference_responses=(answer,),
        mask_mode=MaskMode.EXTERNAL_GENERIC,
        evidence_ids=(),
        provenance={"fixture": True},
    )


class ProcessingTests(unittest.TestCase):
    def test_assistant_only_mask_has_no_hardcoded_token(self) -> None:
        full = torch.tensor([100, 101, 555, 777])
        prompt = torch.tensor([100, 101])
        labels = assistant_only_labels(full, prompt)
        self.assertEqual(labels.tolist(), [-100, -100, 555, 777])
        with self.assertRaises(ProcessingError) as caught:
            assistant_only_labels(full, torch.tensor([100, 999]))
        self.assertEqual(caught.exception.code, ReasonCode.LOSS_MASK_INVALID)

    def test_collator_padding_and_loss_mask(self) -> None:
        collator = DescriptionCollator(FakeProcessor(), training=True)
        batch = collator(
            [
                sample("a", "yes"),
                sample("b", "long answer"),
            ]
        )
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["pixel_values"].shape, (2, 4))
        self.assertEqual(batch["image_grid_thw"].shape, (2, 3))
        self.assertTrue(torch.all(batch["labels"][:, :3] == -100))
        self.assertTrue(torch.all(batch["labels"][:, 3:6] != -100))

    def test_batch4_collator_preserves_variable_tokens_and_images(
        self,
    ) -> None:
        class VariableProcessor:
            pad_token_id = 0

            def encode_training(self, messages):
                answer = messages[-1]["content"][0]["text"]
                index = int(answer)
                prompt = torch.tensor([1, 2], dtype=torch.long)
                target = torch.arange(3, 4 + index, dtype=torch.long)
                full = torch.cat((prompt, target))
                image_count = index + 1
                return EncodedSample(
                    tensors={
                        "input_ids": full,
                        "attention_mask": torch.ones_like(full),
                        "pixel_values": torch.full(
                            (image_count, 4),
                            float(index),
                        ),
                        "image_grid_thw": torch.ones(
                            (image_count, 3),
                            dtype=torch.long,
                        ),
                    },
                    labels=assistant_only_labels(full, prompt),
                    input_token_count=int(full.numel()),
                    image_count=image_count,
                )

            def encode_inference(self, messages):
                raise AssertionError("training fixture only")

        batch = DescriptionCollator(
            VariableProcessor(),
            training=True,
        )(
            [
                sample(f"record-{index}", str(index))
                for index in range(4)
            ]
        )
        self.assertEqual(batch["input_token_counts"], [3, 4, 5, 6])
        self.assertEqual(batch["image_counts"], [1, 2, 3, 4])
        self.assertEqual(batch["input_ids"].shape, (4, 6))
        self.assertEqual(batch["pixel_values"].shape, (10, 4))
        self.assertEqual(batch["image_grid_thw"].shape, (10, 3))
        self.assertTrue(torch.all(batch["labels"][0, 3:] == -100))

    def test_inference_collator_has_no_labels(self) -> None:
        batch = DescriptionCollator(
            FakeProcessor(),
            training=False,
        )([sample("a", "yes")])
        self.assertNotIn("labels", batch)
        self.assertEqual(batch["input_token_counts"], [3])

    def test_qwen_training_forward_explicitly_disables_cache(self) -> None:
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.use_cache = None

            def forward(self, input_ids, use_cache):
                self.use_cache = use_cache
                logits = (
                    torch.zeros(
                        (*input_ids.shape, 64),
                        dtype=self.weight.dtype,
                        device=input_ids.device,
                    )
                    + self.weight
                )
                return type(
                    "Output",
                    (),
                    {
                        "logits": logits,
                    },
                )()

        config = load_config(
            CONFIG_ROOT / "external_lora_qwen3vl_2b.yaml"
        )
        core = CaptureModel()
        adapter = Qwen3VLModelAdapter(
            core,
            identity=ModelIdentity(
                model_path="fixture",
                model_type="fixture",
                config_sha256="a" * 64,
                weights_metadata_sha256="b" * 64,
                processor_config_sha256="c" * 64,
                base_parameter_count=1,
            ),
            adaptation=config.adaptation,
        )
        adapter.forward(
            {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "labels": torch.ones((1, 2), dtype=torch.long),
            }
        )
        self.assertIs(core.use_cache, False)

    def test_causal_loss_is_equal_weight_per_sample(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn((4, 6, 32), dtype=torch.float32)
        labels = torch.tensor(
            [
                [-100, 1, 2, 3, 4, 5],
                [-100, -100, -100, 6, 7, -100],
                [-100, 8, 9, -100, -100, -100],
                [-100, -100, 10, 11, 12, -100],
            ],
            dtype=torch.long,
        )
        batched = assistant_sample_mean_causal_loss(logits, labels)
        separate = torch.stack(
            [
                assistant_sample_mean_causal_loss(
                    logits[index : index + 1],
                    labels[index : index + 1],
                )
                for index in range(4)
            ]
        ).mean()
        torch.testing.assert_close(batched, separate, rtol=0, atol=0)

        invalid = labels.clone()
        invalid[2] = -100
        with self.assertRaises(ModelError) as caught:
            assistant_sample_mean_causal_loss(logits, invalid)
        self.assertEqual(caught.exception.code, ReasonCode.LOSS_MASK_INVALID)

    def test_external_dataset_uses_phase2_epoch_multireference_contract(
        self,
    ) -> None:
        class CanonicalFixture:
            def __init__(self):
                self.epoch = 0
                self.records = [
                    {
                        "record_id": "record",
                        "parent_id": "parent",
                        "logical_role": "external_train",
                        "task_family": "global_caption",
                        "reference_responses": ["first", "second"],
                    }
                ]
                self.manifest = {
                    "build_id": "build",
                    "payload_root_sha256": "payload",
                }

            def set_epoch(self, epoch):
                self.epoch = epoch

            def training_response(self, record):
                return ("first", "second")[self.epoch % 2]

        rendered = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe."}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "placeholder"}],
                },
            ]
        }
        canonical = CanonicalFixture()
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "oa_groundrag.phase4.data.render_canonical_messages",
            return_value=(rendered, []),
        ) as renderer:
            dataset = ExternalDescriptionDataset(
                canonical,
                derived_root=Path(temporary) / "derived",
                seed=9,
            )
            dataset.set_epoch(0)
            first = dataset[0].messages[-1]["content"][0]["text"]
            dataset.set_epoch(1)
            second = dataset[0].messages[-1]["content"][0]["text"]
        self.assertEqual((first, second), ("first", "second"))
        renderer.assert_called_once()


class OAInputAdapterTests(unittest.TestCase):
    def test_canonical_oa_mask_is_gt_only_and_tensor_backed(self) -> None:
        mask = torch.zeros((1, 6, 7), dtype=torch.uint8)
        mask[:, 1:4, 2:6] = 1
        item = {
            "record": {
                "record_id": "oa-record",
                "record_sha256": "a" * 64,
                "logical_role": "oa_train",
                "input_layout": "mask_grounded",
                "target": {"type": "mask", "mask_role": "landslide"},
            },
            "masks": {"landslide": mask},
        }
        value = inventory_from_canonical_oa_item(item)
        self.assertEqual(value.mask_mode, MaskMode.GT_MASK)
        self.assertEqual(value.global_mask.shape, (6, 7))
        self.assertEqual(int(value.global_mask.sum()), 12)
        with self.assertRaises(ContractError) as caught:
            inventory_from_canonical_oa_item(
                item,
                mask_mode=MaskMode.FIXED_PREDICTED_MASK,
            )
        self.assertEqual(caught.exception.code, ReasonCode.INVALID_ENUM)

    def _write_auxseg_fixture(
        self,
        root: Path,
        *,
        duplicate: bool = False,
    ) -> None:
        root.mkdir()
        manifest = {
            "schema_version": "oa_auxseg_inference_v6",
            "checkpoint": "/read-only/checkpoint.pt",
            "checkpoint_step": 12,
            "benchmark_index_sha256": "b" * 64,
            "benchmark_contract": {"schema_version": "fixture"},
            "backbone_sha256": "c" * 64,
            "backbone": "convnext_small",
            "architecture": {"schema_version": "fixture"},
            "split": "val",
            "source": None,
            "sample_count": 2 if duplicate else 1,
            "modality_weight_order": ["dem", "__null__"],
            "modality_weight_map_strides": [4, 8, 16, 32],
            "modality_weight_map_summary": (
                "coverage_pool_each_stage_then_equal_stage_mean"
            ),
            "region_feature_dim": 266,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        array_keys = {
            "mask_probability": "probability",
            "global_mask": "global",
            "region_masks": "regions",
            "region_features": "features",
            "modality_weights": "weights",
            "modality_weight_maps": {
                "4": "weights_stride4",
                "8": "weights_stride8",
                "16": "weights_stride16",
                "32": "weights_stride32",
            },
        }
        row = {
            "sample_id": "sample-a",
            "source": "fixture",
            "split": "val",
            "available_modalities": ["dem"],
            "active_modalities": ["dem"],
            "no_target_score": 0.1,
            "modality_names": ["dem", "__null__"],
            "modality_weights": [0.8, 0.2],
            "array_keys": array_keys,
            "regions": [
                {
                    "region_id": "region_001",
                    "bbox_xyxy": [0, 0, 1, 1],
                    "centroid_xy": [0.0, 0.0],
                    "area_pixels": 999,
                    "confidence": 0.75,
                }
            ],
        }
        lines = [json.dumps(row)]
        if duplicate:
            lines.append(json.dumps(row))
        (root / "predictions.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        global_mask = np.zeros((6, 7), dtype=np.uint8)
        global_mask[1:5, 2:6] = 1
        region_mask = np.zeros_like(global_mask)
        region_mask[2:4, 3:5] = 1
        np.savez(
            root / "predictions.npz",
            **{
                "global": global_mask,
                "regions": region_mask[None, ...],
            },
        )

    def test_auxseg_v6_inventory_uses_masks_not_feature_or_geometry_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inference"
            self._write_auxseg_fixture(root)
            value = inventory_from_auxseg_inference(
                root,
                sample_id="sample-a",
                mask_mode=MaskMode.FIXED_PREDICTED_MASK,
                max_rows=4,
            )
        self.assertTrue(
            value.source_identity.startswith("oa_auxseg_inference_v6:")
        )
        self.assertTrue(value.source_identity.endswith(":sample-a"))
        self.assertEqual(len(value.candidates), 1)
        self.assertEqual(int(value.candidates[0].mask.sum()), 4)
        self.assertEqual(
            value.candidates[0].provenance,
            "oa_auxseg_inference_v6",
        )

    def test_auxseg_duplicate_sample_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inference"
            self._write_auxseg_fixture(root, duplicate=True)
            with self.assertRaises(ContractError) as caught:
                inventory_from_auxseg_inference(
                    root,
                    sample_id="sample-a",
                    mask_mode=MaskMode.END_TO_END_PREDICTED_MASK,
                    max_rows=4,
                )
        self.assertEqual(caught.exception.code, ReasonCode.TYPE_MISMATCH)


if __name__ == "__main__":
    unittest.main()
