from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from oa_groundrag.data.rs_general.io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.vlm.checkpoint import CheckpointManager
from oa_groundrag.vlm.config import load_config
from oa_groundrag.grounding.contracts import (
    MODEL_OUTPUT_SCHEMA_VERSION,
    Claim,
    EvidenceSufficiency,
    MaskMode,
    RegionCandidate,
    RegionInventory,
    SelectionMode,
    SelectionRequest,
    StructuredModelOutput,
    TargetStatus,
)
from oa_groundrag.vlm.data import (
    DescriptionSample,
)
from oa_groundrag.evaluation.vlm import evaluate_predictions
from oa_groundrag.grounding.evidence import EvidenceBuilder
from oa_groundrag.vlm.errors import (
    CheckpointError,
    ContractError,
    EvaluationError,
    ModelError,
    PredictionError,
    ReasonCode,
)
from oa_groundrag.vlm.inference import run_inference
from oa_groundrag.grounding.messages import build_mask_grounded_messages
from oa_groundrag.vlm.model import assistant_sample_mean_causal_loss
from oa_groundrag.vlm.outputs import prediction_row, serialize_model_output
from oa_groundrag.vlm.preflight import BenchmarkIdentity
from oa_groundrag.vlm.processing import (
    DescriptionCollator,
    EncodedSample,
    assistant_only_labels,
)
from oa_groundrag.training.vlm.progress import TrainingProgress
from oa_groundrag.grounding.regions import RegionSelector
from oa_groundrag.training.vlm.trainer import (
    DescriptionTrainer,
    training_layout_identity,
)
from oa_groundrag.training.vlm.validation import (
    select_bounded_external_validation,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs/vlm/rs_general"


class TinyProcessor:
    pad_token_id = 0

    def clone_for_worker(self):
        return TinyProcessor()

    def encode_training(self, messages):
        answer = messages[-1]["content"][0]["text"]
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        target = torch.tensor(
            [10 + (ord(character) % 40) for character in answer[:6]],
            dtype=torch.long,
        )
        if not target.numel():
            target = torch.tensor([10], dtype=torch.long)
        ids = torch.cat((prompt, target))
        attention = torch.ones_like(ids)
        return EncodedSample(
            tensors={
                "input_ids": ids,
                "attention_mask": attention,
                "mm_token_type_ids": torch.zeros_like(ids),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=assistant_only_labels(ids, prompt),
            input_token_count=int(ids.numel()),
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


class TinyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.output = nn.Linear(8, 64)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


class TinyIdentity:
    def to_dict(self):
        return {
            "model_type": "tiny_fixture",
            "revision": "deterministic-v1",
        }


class TinyAdapter:
    def __init__(self, generated_text: str | None = None) -> None:
        self.model = TinyCore()
        self.identity = TinyIdentity()
        self.trainable_names = tuple(
            name for name, _ in self.model.named_parameters()
        )
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.generated_text = generated_text

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, batch):
        logits = self.model(batch["input_ids"])
        loss = assistant_sample_mean_causal_loss(
            logits,
            batch["labels"],
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def trainable_state_dict(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def load_trainable_state_dict(self, state):
        self.model.load_state_dict(dict(state), strict=True)

    def generate_text(
        self,
        batch,
        *,
        processor,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
    ):
        if self.generated_text is not None:
            return [self.generated_text for _ in batch["record_ids"]]
        return [
            str(references[0])
            for references in batch["reference_responses"]
        ]


def external_sample(
    record_id: str,
    parent_id: str,
    *,
    answer: str,
    task: str,
    logical_role: str = "external_train",
) -> DescriptionSample:
    return DescriptionSample(
        record_id=record_id,
        parent_id=parent_id,
        logical_role=logical_role,
        task_family=task,
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/tiny.png"},
                    {"type": "text", "text": "Describe."},
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


class TinyDataset:
    def __init__(self, *, logical_role: str = "external_train") -> None:
        self.epoch = 0
        self.manifest = {"payload_root_sha256": "fixture-payload"}
        tasks = (
            "global_caption",
            "visual_qa",
            "bbox_region_caption",
            "scene_understanding",
            "object_count",
            "spatial_relation",
            "visible_change_report",
        )
        count = 4 if logical_role == "external_train" else len(tasks)
        self.records = [
            {
                "record_id": f"record-{index}",
                "parent_id": f"parent-{index}",
                "logical_role": logical_role,
                "source": (
                    ("rsgpt", "mmrs1m", "disasterm3")[index % 3]
                    if logical_role == "external_val"
                    else "fixture"
                ),
                "task_family": (
                    tasks[index]
                    if logical_role == "external_val"
                    else (
                        "global_caption"
                        if index % 2 == 0
                        else "visual_qa"
                    )
                ),
            }
            for index in range(count)
        ]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return external_sample(
            row["record_id"],
            row["parent_id"],
            answer=f"answer-{index}-epoch-{self.epoch % 2}",
            task=row["task_family"],
            logical_role=row["logical_role"],
        )


def benchmark_identity(base: Path) -> BenchmarkIdentity:
    return BenchmarkIdentity(
        root=base,
        manifest_schema="fixture.manifest",
        canonical_schema="fixture.canonical",
        build_id="fixture-build",
        semantic_config_sha256="fixture-semantic",
        payload_sha256="fixture-payload",
        hash_manifest_sha256="fixture-hash-manifest",
        benchmark_scope="rs_generaldesc_external_train_val",
        source_roots_embedded=False,
        deep_validation_saved=True,
        formal_acceptance_eligible=True,
        formal_acceptance_blockers=(),
        record_count=4,
        parent_count=4,
    )


class TrainerCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        base_config = load_config(
            CONFIG_ROOT / "rs_generaldesc_lora_qwen3vl_2b.yaml"
        )
        self.config = replace(
            base_config,
            training=replace(
                base_config.training,
                max_steps=2,
                epochs=8,
                checkpoint_interval=1,
                log_interval=1,
                validation_interval=1,
                validation_max_parents=7,
            ),
        )
        self.processor_identity = {"processor": "tiny-v1"}
        self.validation_dataset = TinyDataset(logical_role="external_val")
        self.validation_selection = select_bounded_external_validation(
            self.validation_dataset,
            benchmark_build_id="fixture-build",
            benchmark_payload_sha256="fixture-payload",
            seed=self.config.run.seed,
            max_parents=7,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def trainer(
        self,
        config,
        model,
        *,
        cuda_cache_cleanup_interval_steps=None,
    ):
        return DescriptionTrainer(
            config=config,
            model=model,
            collator=DescriptionCollator(TinyProcessor(), training=True),
            validation_dataset=self.validation_dataset,
            validation_collator=DescriptionCollator(
                TinyProcessor(),
                training=True,
            ),
            validation_selection=self.validation_selection,
            benchmark_identity=benchmark_identity(self.base),
            processor_identity=self.processor_identity,
            device=torch.device("cpu"),
            progress=TrainingProgress(
                log_interval=1,
                stream=io.StringIO(),
                force_tty=False,
            ),
            cuda_cache_cleanup_interval_steps=(
                cuda_cache_cleanup_interval_steps
            ),
        )

    def test_cuda_cache_cleanup_is_opt_in_and_checkpoint_bound(self) -> None:
        generic_config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "generic-cache-policy",
            ),
        )
        with patch("oa_groundrag.training.vlm.trainer.clear_cuda_cache") as cleanup:
            self.trainer(generic_config, TinyAdapter()).fit(
                TinyDataset(),
                stop_after_steps=1,
            )
        cleanup.assert_not_called()
        self.assertNotIn(
            "cuda_cache_cleanup_interval_steps",
            training_layout_identity(generic_config),
        )

        stage5_config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "stage5-cache-policy",
            ),
        )
        with patch("oa_groundrag.training.vlm.trainer.clear_cuda_cache") as cleanup:
            first = self.trainer(
                stage5_config,
                TinyAdapter(),
                cuda_cache_cleanup_interval_steps=1,
            ).fit(TinyDataset(), stop_after_steps=1)
        cleanup.assert_called_once_with(torch.device("cpu"))
        expected = training_layout_identity(
            stage5_config,
            cuda_cache_cleanup_interval_steps=1,
        )
        manifest = read_json(first.checkpoint / "manifest.json")
        self.assertEqual(manifest["training_layout"], expected)

        with self.assertRaises(CheckpointError):
            self.trainer(stage5_config, TinyAdapter()).fit(
                TinyDataset(),
                resume_checkpoint=first.checkpoint,
            )
        with patch("oa_groundrag.training.vlm.trainer.clear_cuda_cache") as cleanup:
            resumed = self.trainer(
                stage5_config,
                TinyAdapter(),
                cuda_cache_cleanup_interval_steps=1,
            ).fit(TinyDataset(), resume_checkpoint=first.checkpoint)
        self.assertEqual(resumed.cursor.global_step, 2)
        cleanup.assert_called_once_with(torch.device("cpu"))

    def test_interrupted_resume_matches_uninterrupted_next_sample_and_weights(
        self,
    ) -> None:
        uninterrupted_config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "uninterrupted",
            ),
        )
        torch.manual_seed(91)
        uninterrupted_model = TinyAdapter()
        uninterrupted_dataset = TinyDataset()
        uninterrupted = self.trainer(
            uninterrupted_config,
            uninterrupted_model,
        ).fit(uninterrupted_dataset)
        self.assertGreater(uninterrupted_dataset.epoch, 0)

        resumed_config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "resumed",
            ),
        )
        torch.manual_seed(91)
        first_model = TinyAdapter()
        first = self.trainer(resumed_config, first_model).fit(
            TinyDataset(),
            stop_after_steps=1,
        )
        torch.manual_seed(999)
        second_model = TinyAdapter()
        second = self.trainer(resumed_config, second_model).fit(
            TinyDataset(),
            resume_checkpoint=first.checkpoint,
        )
        self.assertEqual(
            first.sample_trace + second.sample_trace,
            uninterrupted.sample_trace,
        )
        for name, value in uninterrupted_model.model.state_dict().items():
            torch.testing.assert_close(
                value,
                second_model.model.state_dict()[name],
                rtol=0,
                atol=0,
            )
        self.assertEqual(second.cursor.global_step, 2)
        self.assertEqual(second.cursor.micro_step, 0)
        manifest = read_json(resumed_config.run.output_root / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "rs_vlm.run_manifest.v1",
        )
        self.assertEqual(manifest["cursor"], second.cursor.to_dict())
        self.assertTrue(second.best_checkpoint.is_dir())
        best = read_json(
            resumed_config.run.output_root / "best_checkpoint.json"
        )
        self.assertEqual(best["selection_metric"], "macro_task_loss")
        self.assertIn(best["step"], {1, 2})
        selection = read_json(
            resumed_config.run.output_root / "validation_selection.json"
        )
        self.assertEqual(selection["selected_parents"], 7)
        self.assertEqual(len(selection["source_counts"]), 3)
        self.assertEqual(len(selection["task_counts"]), 7)
        log_rows = read_jsonl(
            resumed_config.run.output_root / "train_log.jsonl"
        )
        self.assertEqual([row["step"] for row in log_rows], [1, 2])
        trace = read_jsonl(
            resumed_config.run.output_root / "sample_trace.jsonl"
        )
        self.assertEqual(len(trace), 32)
        self.assertTrue(
            all(
                row["schema_version"]
                == "rs_vlm.sample_trace.v1"
                for row in trace
            )
        )
        self.assertEqual(
            [row["batch_slot"] for row in trace[:16]],
            list(range(resumed_config.training.batch_size))
            * resumed_config.training.gradient_accumulation_steps,
        )
        self.assertEqual(
            [row["micro_step"] for row in trace[:16]],
            [
                micro_step
                for micro_step in range(
                    1,
                    resumed_config.training.gradient_accumulation_steps + 1,
                )
                for _ in range(resumed_config.training.batch_size)
            ],
        )
        report = read_json(
            resumed_config.run.output_root / "training_report.json"
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["validation_history"]), 2)
        self.assertGreater(report["throughput"]["samples_per_second"], 0)
        self.assertGreater(report["throughput"]["tokens_per_second"], 0)
        self.assertTrue(np.isfinite(report["last_gradient_norm"]))
        expected_layout = training_layout_identity(resumed_config)
        self.assertEqual(report["training_layout"], expected_layout)
        self.assertEqual(manifest["training_layout"], expected_layout)

    def test_batch4_matches_batch1_first_effective_batch_and_update(
        self,
    ) -> None:
        batch4_config = replace(
            self.config,
            run=replace(
                self.config.run,
                name="fixture-batch4",
                output_root=self.base / "batch4",
            ),
            training=replace(
                self.config.training,
                batch_size=4,
                gradient_accumulation_steps=4,
            ),
        )
        batch1_config = replace(
            self.config,
            run=replace(
                self.config.run,
                name="fixture-batch1",
                output_root=self.base / "batch1",
            ),
        )
        torch.manual_seed(29)
        batch4_model = TinyAdapter()
        batch4 = self.trainer(batch4_config, batch4_model).fit(
            TinyDataset(),
            stop_after_steps=1,
        )
        torch.manual_seed(29)
        batch1_model = TinyAdapter()
        batch1 = self.trainer(batch1_config, batch1_model).fit(
            TinyDataset(),
            stop_after_steps=1,
        )

        self.assertEqual(batch4.sample_trace, batch1.sample_trace)
        self.assertEqual(len(batch4.sample_trace), 16)
        for name, value in batch1_model.model.state_dict().items():
            torch.testing.assert_close(
                value,
                batch4_model.model.state_dict()[name],
                rtol=1e-6,
                atol=1e-7,
            )

        with self.assertRaises(CheckpointError) as caught:
            CheckpointManager().load(
                batch1.checkpoint,
                expected_config_semantic_sha256=(
                    batch1_config.semantic_sha256
                ),
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity=(
                    self.validation_selection.identity_dict()
                ),
                expected_model_identity=batch1_model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(
                    batch4_config
                ),
                expected_trainable_names=batch1_model.trainable_names,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
        )

    def test_resume_recovers_validation_pending_at_checkpoint(self) -> None:
        config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "pending-validation",
            ),
        )
        torch.manual_seed(17)
        first = self.trainer(config, TinyAdapter()).fit(
            TinyDataset(),
            stop_after_steps=1,
        )
        (config.run.output_root / "validation_results.jsonl").unlink()
        (config.run.output_root / "best_checkpoint.json").unlink()

        resumed = self.trainer(config, TinyAdapter()).fit(
            TinyDataset(),
            resume_checkpoint=first.checkpoint,
        )

        self.assertEqual(resumed.cursor.global_step, 2)
        results = read_jsonl(
            config.run.output_root / "validation_results.jsonl"
        )
        self.assertEqual([row["step"] for row in results], [1, 2])
        self.assertTrue(
            (config.run.output_root / "best_checkpoint.json").is_file()
        )

    def test_resume_rejects_bool_in_structured_train_log(self) -> None:
        config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "bool-log",
            ),
        )
        first = self.trainer(config, TinyAdapter()).fit(
            TinyDataset(),
            stop_after_steps=1,
        )
        log_path = config.run.output_root / "train_log.jsonl"
        rows = read_jsonl(log_path)
        rows[-1]["loss"] = True
        atomic_write_jsonl(log_path, rows)

        with self.assertRaises(ModelError) as caught:
            self.trainer(config, TinyAdapter()).fit(
                TinyDataset(),
                resume_checkpoint=first.checkpoint,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
        )

    def test_resume_archives_rows_after_explicit_checkpoint(self) -> None:
        config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "orphaned-rows",
            ),
        )
        first = self.trainer(config, TinyAdapter()).fit(
            TinyDataset(),
            stop_after_steps=1,
        )
        log_path = config.run.output_root / "train_log.jsonl"
        log_rows = read_jsonl(log_path)
        orphaned_log = dict(log_rows[-1])
        orphaned_log["step"] = 2
        orphaned_log["samples"] = 32
        atomic_write_jsonl(log_path, (*log_rows, orphaned_log))

        trace_path = config.run.output_root / "sample_trace.jsonl"
        trace_rows = read_jsonl(trace_path)
        orphaned_trace = []
        batch_size = config.training.batch_size
        for index, row in enumerate(trace_rows):
            orphaned = dict(row)
            orphaned["sequence_index"] = len(trace_rows) + index
            orphaned["optimizer_step"] = 2
            orphaned["micro_step"] = index // batch_size + 1
            orphaned["batch_slot"] = index % batch_size
            orphaned["record_id"] = f"{row['record_id']}-orphaned"
            orphaned_trace.append(orphaned)
        atomic_write_jsonl(
            trace_path,
            (*trace_rows, *orphaned_trace),
        )

        resumed = self.trainer(config, TinyAdapter()).fit(
            TinyDataset(),
            resume_checkpoint=first.checkpoint,
        )

        recoveries = sorted(
            (
                config.run.output_root / "resume_recoveries"
            ).glob("*.jsonl")
        )
        self.assertEqual(len(recoveries), 2)
        self.assertEqual(
            [row["step"] for row in read_jsonl(log_path)],
            [1, 2],
        )
        self.assertEqual(len(read_jsonl(trace_path)), 32)
        report = read_json(resumed.training_report)
        self.assertEqual(
            len(report["artifacts"]["resume_recoveries"]),
            2,
        )

    def test_checkpoint_rejects_semantic_mismatch_and_tamper(self) -> None:
        config = replace(
            self.config,
            run=replace(
                self.config.run,
                output_root=self.base / "checkpoint",
            ),
        )
        torch.manual_seed(10)
        model = TinyAdapter()
        result = self.trainer(config, model).fit(
            TinyDataset(),
            stop_after_steps=1,
        )
        manager = CheckpointManager()
        with self.assertRaises(CheckpointError) as caught:
            manager.load(
                result.checkpoint,
                expected_config_semantic_sha256="0" * 64,
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity=(
                    self.validation_selection.identity_dict()
                ),
                expected_model_identity=model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(config),
                expected_trainable_names=model.trainable_names,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
        )
        with self.assertRaises(CheckpointError) as caught:
            manager.load(
                result.checkpoint,
                expected_config_semantic_sha256=config.semantic_sha256,
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity={
                    "selection_sha256": "0" * 64
                },
                expected_model_identity=model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(config),
                expected_trainable_names=model.trainable_names,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
        )
        manifest_path = result.checkpoint / "manifest.json"
        manifest = read_json(manifest_path)
        incomplete_manifest = dict(manifest)
        del incomplete_manifest["training_layout"]
        atomic_write_json(manifest_path, incomplete_manifest)
        with self.assertRaises(CheckpointError) as caught:
            manager.load(
                result.checkpoint,
                expected_config_semantic_sha256=config.semantic_sha256,
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity=(
                    self.validation_selection.identity_dict()
                ),
                expected_model_identity=model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(config),
                expected_trainable_names=model.trainable_names,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.CHECKPOINT_CORRUPT,
        )
        atomic_write_json(manifest_path, manifest)
        manifest["unknown"] = True
        atomic_write_json(manifest_path, manifest)
        with self.assertRaises(CheckpointError) as caught:
            manager.load(
                result.checkpoint,
                expected_config_semantic_sha256=config.semantic_sha256,
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity=(
                    self.validation_selection.identity_dict()
                ),
                expected_model_identity=model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(config),
                expected_trainable_names=model.trainable_names,
            )
        self.assertEqual(caught.exception.code, ReasonCode.CHECKPOINT_CORRUPT)
        del manifest["unknown"]
        atomic_write_json(manifest_path, manifest)
        unexpected_path = result.checkpoint / "unexpected.bin"
        unexpected_path.write_bytes(b"unexpected")
        with self.assertRaises(CheckpointError) as caught:
            manager.load(
                result.checkpoint,
                expected_config_semantic_sha256=config.semantic_sha256,
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity=(
                    self.validation_selection.identity_dict()
                ),
                expected_model_identity=model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(config),
                expected_trainable_names=model.trainable_names,
            )
        self.assertEqual(caught.exception.code, ReasonCode.CHECKPOINT_CORRUPT)
        unexpected_path.unlink()
        optimizer_path = result.checkpoint / "optimizer.pt"
        optimizer_path.write_bytes(optimizer_path.read_bytes() + b"tamper")
        with self.assertRaises(CheckpointError) as caught:
            manager.load(
                result.checkpoint,
                expected_config_semantic_sha256=config.semantic_sha256,
                expected_benchmark_identity=benchmark_identity(
                    self.base
                ).training_identity_dict(),
                expected_validation_selection_identity=(
                    self.validation_selection.identity_dict()
                ),
                expected_model_identity=model.identity.to_dict(),
                expected_processor_identity=self.processor_identity,
                expected_training_layout=training_layout_identity(config),
                expected_trainable_names=model.trainable_names,
            )
        self.assertEqual(caught.exception.code, ReasonCode.CHECKPOINT_CORRUPT)

    def test_artifact_output_rejects_symlink_ancestor(self) -> None:
        real = self.base / "real"
        real.mkdir()
        linked = self.base / "linked"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ContractError) as caught:
            with AtomicArtifactDirectory(linked / "output"):
                self.fail("symlink ancestor must fail before staging")
        self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_LINK)


class InferenceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_external_tiny_inference_and_per_task_evaluation(self) -> None:
        config = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        samples = [
            external_sample(
                "external-1",
                "parent-1",
                answer="A road is visible.",
                task="global_caption",
            ),
            external_sample(
                "external-2",
                "parent-2",
                answer="Yes.",
                task="visual_qa",
            ),
        ]
        inference = run_inference(
            config=config,
            samples=samples,
            collator=DescriptionCollator(TinyProcessor(), training=False),
            model=TinyAdapter(),
            processor=None,
            output_root=self.base / "external-inference",
        )
        evaluation = evaluate_predictions(
            inference / "predictions.jsonl",
            output_root=self.base / "external-evaluation",
            expected_mask_mode=MaskMode.EXTERNAL_GENERIC,
        )
        metrics = read_json(evaluation / "metrics.json")
        self.assertEqual(
            metrics["per_task"]["global_caption"][
                "normalized_exact_match"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["per_task"]["visual_qa"][
                "best_reference_token_f1"
            ],
            1.0,
        )
        self.assertFalse(metrics["formal_acceptance"])

    def test_mask_modes_cannot_be_mixed_in_one_run(self) -> None:
        config = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        generic = external_sample(
            "external",
            "parent-external",
            answer="Visible.",
            task="global_caption",
        )
        mixed = replace(generic, mask_mode=MaskMode.GT_MASK)
        with self.assertRaises(PredictionError) as caught:
            run_inference(
                config=config,
                samples=[generic, mixed],
                collator=DescriptionCollator(
                    TinyProcessor(),
                    training=False,
                ),
                model=TinyAdapter(),
                processor=None,
                output_root=self.base / "mixed",
            )
        self.assertEqual(caught.exception.code, ReasonCode.MASK_MODE_MIXED)

    def test_synthetic_mask_grounded_dataset_to_inference_and_evaluation(
        self,
    ) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        selected = RegionSelector().select(
            RegionInventory(
                "grounded-record",
                MaskMode.GT_MASK,
                mask,
                (RegionCandidate("r1", mask),),
                "fixture",
            ),
            SelectionRequest(SelectionMode.REGION_ID, region_id="r1"),
        )
        evidence = EvidenceBuilder().build(
            optical_image=Image.new("RGB", (8, 8), (40, 80, 120)),
            selected=selected,
            mask_mode=MaskMode.GT_MASK,
            output_root=self.base / "evidence",
        )
        output = StructuredModelOutput(
            schema_version=MODEL_OUTPUT_SCHEMA_VERSION,
            target_status=TargetStatus.TARGET_PRESENT,
            description="The selected region is visible.",
            answer=None,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            claims=(
                Claim(
                    "The selected region is visible.",
                    ("ev_mask_overlay",),
                ),
            ),
            limitations=(),
        )
        target = serialize_model_output(output)
        messages = build_mask_grounded_messages(
            evidence.bundle,
            evidence_root=evidence.root,
            instruction="Describe the selected region.",
            assistant_target=target,
        )
        sample = DescriptionSample(
            record_id="grounded-record",
            parent_id="grounded-parent",
            logical_role="grounded_core_fixture",
            task_family="mask_description",
            messages=tuple(messages),
            reference_responses=(target,),
            mask_mode=MaskMode.GT_MASK,
            evidence_ids=tuple(sorted(evidence.bundle.evidence_ids)),
            provenance={
                "evidence_schema_version": evidence.bundle.schema_version,
                "expected_target_status": evidence.bundle.target_status.value,
                "known_limitations": list(evidence.bundle.limitations),
                "selection": dict(evidence.bundle.selection),
            },
        )
        base_config = load_config(
            CONFIG_ROOT / "rs_generaldesc_prompt_only_qwen3vl_2b.yaml"
        )
        config = replace(
            base_config,
            run=replace(
                base_config.run,
                mask_mode=MaskMode.GT_MASK,
                output_root=self.base / "grounded-core",
            ),
        )
        trained_model = TinyAdapter()
        inference = run_inference(
            config=config,
            samples=[sample],
            collator=DescriptionCollator(TinyProcessor(), training=False),
            model=trained_model,
            processor=None,
            output_root=self.base / "grounded-inference",
        )
        evaluation = evaluate_predictions(
            inference / "predictions.jsonl",
            output_root=self.base / "grounded-evaluation",
            expected_mask_mode=MaskMode.GT_MASK,
        )
        metrics = read_json(evaluation / "metrics.json")
        self.assertEqual(metrics["evidence_constraint_violations"], 0)
        self.assertEqual(metrics["target_status_mismatches"], 0)
        self.assertEqual(metrics["no_target_hallucinations"], 0)

    def test_counterfactual_mask_swap_wrong_empty_and_modality_removal(self) -> None:
        present = StructuredModelOutput(
            MODEL_OUTPUT_SCHEMA_VERSION,
            TargetStatus.TARGET_PRESENT,
            "Baseline region.",
            None,
            EvidenceSufficiency.SUFFICIENT,
            (Claim("Visible.", ("ev_overlay",)),),
            (),
        )
        swapped = replace(present, description="Swapped region.")
        wrong = replace(present, description="Different selected region.")
        empty = StructuredModelOutput(
            MODEL_OUTPUT_SCHEMA_VERSION,
            TargetStatus.NO_TARGET,
            "No target is visible in the supplied evidence.",
            None,
            EvidenceSufficiency.LIMITED,
            (Claim("No target is visible.", ("ev_full",)),),
            ("NO_TARGET_NO_REGION_GEOMETRY",),
        )
        removed = replace(
            present,
            description="Only optical evidence remains.",
            evidence_sufficiency=EvidenceSufficiency.LIMITED,
            claims=(Claim("Optical evidence remains.", ("ev_overlay",)),),
        )
        variants = (
            ("baseline", present, ["ev_overlay"], {}),
            ("mask_swap", swapped, ["ev_overlay"], {}),
            ("wrong_region", wrong, ["ev_overlay"], {}),
            ("empty_mask", empty, ["ev_full"], {}),
            (
                "modality_removal",
                removed,
                ["ev_overlay"],
                {"removed_evidence_ids": ["ev_aux"]},
            ),
        )
        rows = []
        for variant, output, evidence_ids, extra in variants:
            rows.append(
                prediction_row(
                    record_id=f"record-{variant}",
                    parent_id="counterfactual-parent",
                    logical_role="grounded_val",
                    task_family="mask_description",
                    mask_mode=MaskMode.GT_MASK,
                    model_output=output,
                    reference_responses=(output.description,),
                    evidence_ids=evidence_ids,
                    provenance={
                        "expected_target_status": output.target_status.value
                    },
                    counterfactual={
                        "group_id": "group-1",
                        "variant": variant,
                        **extra,
                    },
                )
            )
        predictions = self.base / "counterfactuals.jsonl"
        atomic_write_jsonl(predictions, rows)
        evaluation = evaluate_predictions(
            predictions,
            output_root=self.base / "counterfactual-evaluation",
            expected_mask_mode=MaskMode.GT_MASK,
        )
        metrics = read_json(evaluation / "metrics.json")
        self.assertEqual(
            metrics["counterfactual"]["checks"],
            {
                "empty_mask": 1,
                "mask_swap": 1,
                "modality_removal": 1,
                "wrong_region": 1,
            },
        )
        self.assertEqual(metrics["counterfactual"]["failures"], {})

    def test_counterfactual_group_rejects_cross_parent_leakage(self) -> None:
        output = StructuredModelOutput(
            MODEL_OUTPUT_SCHEMA_VERSION,
            TargetStatus.TARGET_PRESENT,
            "Visible selected region.",
            None,
            EvidenceSufficiency.SUFFICIENT,
            (Claim("Visible.", ("ev_overlay",)),),
            (),
        )
        rows = [
            prediction_row(
                record_id="baseline",
                parent_id="parent-a",
                logical_role="grounded_val",
                task_family="mask_description",
                mask_mode=MaskMode.GT_MASK,
                model_output=output,
                reference_responses=(output.description,),
                evidence_ids=("ev_overlay",),
                provenance={
                    "expected_target_status": TargetStatus.TARGET_PRESENT.value
                },
                counterfactual={
                    "group_id": "leaked-group",
                    "variant": "baseline",
                },
            ),
            prediction_row(
                record_id="swap",
                parent_id="parent-b",
                logical_role="grounded_val",
                task_family="mask_description",
                mask_mode=MaskMode.GT_MASK,
                model_output=replace(output, description="Changed region."),
                reference_responses=("Changed region.",),
                evidence_ids=("ev_overlay",),
                provenance={
                    "expected_target_status": TargetStatus.TARGET_PRESENT.value
                },
                counterfactual={
                    "group_id": "leaked-group",
                    "variant": "mask_swap",
                },
            ),
        ]
        predictions = self.base / "cross-parent.jsonl"
        atomic_write_jsonl(predictions, rows)
        with self.assertRaises(EvaluationError) as caught:
            evaluate_predictions(
                predictions,
                output_root=self.base / "cross-parent-evaluation",
                expected_mask_mode=MaskMode.GT_MASK,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.COUNTERFACTUAL_INVALID,
        )


if __name__ == "__main__":
    unittest.main()
