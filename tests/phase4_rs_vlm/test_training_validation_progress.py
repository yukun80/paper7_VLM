from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from oa_groundrag.phase4.contracts import MaskMode
from oa_groundrag.phase4.data import DescriptionSample
from oa_groundrag.phase4.processing import (
    DescriptionCollator,
    EncodedSample,
    assistant_only_labels,
)
from oa_groundrag.phase4.progress import TrainingProgress
from oa_groundrag.phase4.validation import (
    ValidationResult,
    evaluate_teacher_forced_loss,
    select_bounded_external_validation,
    validation_is_better,
)


TASKS = (
    "global_caption",
    "visual_qa",
    "bbox_region_caption",
    "scene_understanding",
    "object_count",
    "spatial_relation",
    "visible_change_report",
)
SOURCES = ("rsgpt", "mmrs1m", "disasterm3")


class ValidationFixture:
    def __init__(self) -> None:
        self.records = [
            {
                "record_id": f"val-record-{index:02d}",
                "parent_id": f"val-parent-{index:02d}",
                "logical_role": "external_val",
                "source": SOURCES[index % len(SOURCES)],
                "task_family": TASKS[index % len(TASKS)],
            }
            for index in range(14)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return DescriptionSample(
            record_id=row["record_id"],
            parent_id=row["parent_id"],
            logical_role="external_val",
            task_family=row["task_family"],
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/tmp/fixture.png"},
                        {"type": "text", "text": "Describe."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
            ),
            reference_responses=("answer",),
            mask_mode=MaskMode.EXTERNAL_GENERIC,
            evidence_ids=(),
            provenance={"fixture": True},
        )


class ValidationProcessor:
    pad_token_id = 0

    def encode_training(self, messages):
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        full = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        return EncodedSample(
            tensors={
                "input_ids": full,
                "attention_mask": torch.ones_like(full),
                "mm_token_type_ids": torch.zeros_like(full),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=assistant_only_labels(full, prompt),
            input_token_count=5,
            image_count=1,
        )

    def encode_inference(self, messages):
        raise AssertionError("validation uses training labels")


class TaskLossAdapter:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, batch):
        task = batch["task_families"][0]
        loss = float(TASKS.index(task) + 1)
        return SimpleNamespace(loss=torch.tensor(loss), logits=None)


class ValidationAndProgressTests(unittest.TestCase):
    def test_bounded_selection_is_deterministic_and_covers_contract(self):
        dataset = ValidationFixture()
        first = select_bounded_external_validation(
            dataset,
            benchmark_build_id="build",
            benchmark_payload_sha256="a" * 64,
            seed=20260730,
            max_parents=14,
        )
        second = select_bounded_external_validation(
            dataset,
            benchmark_build_id="build",
            benchmark_payload_sha256="a" * 64,
            seed=20260730,
            max_parents=14,
        )
        changed = select_bounded_external_validation(
            dataset,
            benchmark_build_id="build",
            benchmark_payload_sha256="a" * 64,
            seed=20260731,
            max_parents=14,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.selection_sha256, changed.selection_sha256)
        self.assertEqual(len(first.items), 14)
        self.assertEqual(len({item.parent_id for item in first.items}), 14)
        self.assertEqual(set(first.source_counts), set(SOURCES))
        self.assertEqual(set(first.task_counts), set(TASKS))

    def test_teacher_forced_macro_loss_preserves_rng_and_training_mode(self):
        dataset = ValidationFixture()
        selection = select_bounded_external_validation(
            dataset,
            benchmark_build_id="build",
            benchmark_payload_sha256="a" * 64,
            seed=11,
            max_parents=14,
        )
        model = TaskLossAdapter()
        model.train()
        torch.manual_seed(99)
        rng_before = torch.get_rng_state().clone()
        result = evaluate_teacher_forced_loss(
            model=model,
            collator=DescriptionCollator(
                ValidationProcessor(),
                training=True,
            ),
            dataset=dataset,
            selection=selection,
            device=torch.device("cpu"),
            step=100,
        )
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        self.assertTrue(model.model.training)
        self.assertAlmostEqual(result.macro_task_loss, 4.0)
        self.assertAlmostEqual(result.overall_loss, 4.0)
        self.assertEqual(result.sample_count, 14)
        self.assertEqual(result.parent_count, 14)
        self.assertEqual(result.input_tokens, 70)
        self.assertEqual(result.images, 14)

    def test_best_tie_break_and_terminal_modes(self):
        def result(step, macro, overall):
            return ValidationResult(
                step=step,
                macro_task_loss=macro,
                overall_loss=overall,
                task_losses={task: macro for task in TASKS},
                task_counts={task: 1 for task in TASKS},
                sample_count=7,
                parent_count=7,
                input_tokens=35,
                images=7,
                duration_seconds=0.1,
            )

        incumbent = result(100, 2.0, 2.5)
        self.assertTrue(validation_is_better(result(200, 1.9, 3.0), incumbent))
        self.assertTrue(validation_is_better(result(200, 2.0, 2.4), incumbent))
        self.assertFalse(validation_is_better(result(200, 2.0, 2.5), incumbent))

        stream = io.StringIO()
        progress = TrainingProgress(
            log_interval=10,
            stream=stream,
            force_tty=False,
        )
        progress.start_training(start_step=0, stop_step=10)
        progress.update_training(
            {
                "step": 1,
                "loss": 2.0,
                "ema_loss": 2.0,
                "learning_rate": 2e-4,
                "gradient_norm": 0.5,
                "gradient_clipped": False,
                "samples": 16,
                "input_tokens": 100,
                "images": 16,
                "samples_per_second": 1.5,
                "tokens_per_second": 10.0,
                "step_duration_seconds": 2.0,
                "data_wait_seconds": 0.25,
                "data_wait_fraction": 0.125,
                "run_elapsed_seconds": 2.0,
                "eta_seconds": 18.0,
                "cuda_allocated_gib": None,
                "cuda_reserved_gib": None,
                "cuda_peak_gib": None,
            }
        )
        progress.start_validation(step=100, total_samples=14)
        progress.finish_validation(
            step=100,
            macro_task_loss=1.0,
            overall_loss=1.1,
            duration_seconds=3.0,
        )
        progress.close()
        output = stream.getvalue()
        self.assertIn("[train] step=1/10", output)
        self.assertIn("samples=16", output)
        self.assertIn("tokens=100", output)
        self.assertIn("data_wait=0.250s", output)
        self.assertIn("[val] step=100", output)

        tty_stream = io.StringIO()
        tty = TrainingProgress(
            log_interval=10,
            stream=tty_stream,
            force_tty=True,
        )
        tty.start_training(start_step=0, stop_step=1)
        tty.update_training(
            {
                "step": 1,
                "loss": 1.0,
                "ema_loss": 1.0,
                "learning_rate": 1e-4,
                "gradient_norm": 0.1,
                "gradient_clipped": False,
                "samples": 16,
                "input_tokens": 100,
                "images": 16,
                "step_duration_seconds": 2.0,
                "data_wait_seconds": 0.25,
                "data_wait_fraction": 0.125,
                "samples_per_second": 1.0,
                "tokens_per_second": 5.0,
                "cuda_allocated_gib": None,
                "cuda_reserved_gib": None,
                "cuda_peak_gib": None,
            }
        )
        tty.close()
        tty_output = tty_stream.getvalue()
        self.assertIn("phase4 train", tty_output)
        self.assertIn("img=16", tty_output)
        self.assertIn("tok=100", tty_output)


if __name__ == "__main__":
    unittest.main()
