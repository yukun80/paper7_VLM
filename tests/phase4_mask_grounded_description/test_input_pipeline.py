from __future__ import annotations

import time
import unittest

import torch

from oa_groundrag.phase4.contracts import MaskMode
from oa_groundrag.phase4.data import DescriptionSample
from oa_groundrag.phase4.input_pipeline import (
    DeterministicBatchPlanner,
    OrderedBatchPrefetcher,
    sampler_state_at_epoch,
)
from oa_groundrag.phase4.processing import (
    DescriptionCollator,
    EncodedSample,
    assistant_only_labels,
)


class PipelineDataset:
    def __init__(self, size: int = 4) -> None:
        self.epoch = 0
        self.records = [
            {
                "record_id": f"record-{index}",
                "parent_id": f"parent-{index}",
                "logical_role": "external_train",
                "source": "fixture",
                "task_family": "global_caption",
            }
            for index in range(size)
        ]
        self.manifest = {"payload_root_sha256": "fixture-payload"}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> DescriptionSample:
        record_id = str(self.records[index]["record_id"])
        answer = f"{record_id}-epoch-{self.epoch}"
        return DescriptionSample(
            record_id=record_id,
            parent_id=str(self.records[index]["parent_id"]),
            logical_role="external_train",
            task_family="global_caption",
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
                    "content": [{"type": "text", "text": answer}],
                },
            ),
            reference_responses=(answer,),
            mask_mode=MaskMode.EXTERNAL_GENERIC,
            evidence_ids=(),
            provenance={"fixture": True},
        )


class PipelineSampler:
    def __init__(self, dataset: PipelineDataset) -> None:
        self.dataset = dataset
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def __iter__(self):
        return iter(range(len(self.dataset.records)))

    def state_dict(self):
        return {
            "schema_version": "fixture.sampler.v1",
            "epoch": self.epoch,
            "payload": "fixture-payload",
        }


class SlowProcessor:
    pad_token_id = 0

    def __init__(
        self,
        *,
        completion: list[str],
        fail_record: str | None = None,
    ) -> None:
        self.completion = completion
        self.fail_record = fail_record

    def clone_for_worker(self):
        return SlowProcessor(
            completion=self.completion,
            fail_record=self.fail_record,
        )

    def encode_training(self, messages):
        answer = str(messages[-1]["content"][0]["text"])
        record_id = answer.split("-epoch-", maxsplit=1)[0]
        if record_id == self.fail_record:
            raise RuntimeError(f"fixture failure: {record_id}")
        time.sleep(0.08 if record_id == "record-0" else 0.002)
        self.completion.append(record_id)
        prompt = torch.tensor([1, 2], dtype=torch.long)
        full = torch.tensor([1, 2, 3], dtype=torch.long)
        return EncodedSample(
            tensors={
                "input_ids": full,
                "attention_mask": torch.ones_like(full),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=assistant_only_labels(full, prompt),
            input_token_count=3,
            image_count=1,
        )

    def encode_inference(self, messages):
        raise AssertionError("training pipeline must use training encoding")


class InputPipelineTests(unittest.TestCase):
    def build_planner(
        self,
        dataset: PipelineDataset,
        sampler: PipelineSampler,
        *,
        batch_size: int,
        max_batches: int,
    ) -> DeterministicBatchPlanner:
        return DeterministicBatchPlanner(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            max_epochs=3,
            start_epoch=0,
            start_sample_offset=0,
            max_batches=max_batches,
        )

    def test_thread_completion_cannot_reorder_batches(self) -> None:
        dataset = PipelineDataset()
        sampler = PipelineSampler(dataset)
        completion: list[str] = []
        collator = DescriptionCollator(
            SlowProcessor(completion=completion),
            training=True,
        )
        planner = self.build_planner(
            dataset,
            sampler,
            batch_size=1,
            max_batches=4,
        )
        with OrderedBatchPrefetcher(
            planner=planner,
            collator=collator,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=False,
        ) as pipeline:
            time.sleep(0.03)
            self.assertTrue(completion)
            self.assertEqual(completion[0], "record-1")
            observed = [
                prepared.batch["record_ids"][0] for prepared in pipeline
            ]
        self.assertEqual(
            observed,
            ["record-0", "record-1", "record-2", "record-3"],
        )
        self.assertEqual(planner.planned_batches, 4)

    def test_cross_epoch_cursor_and_checkpoint_state_are_committed(self) -> None:
        dataset = PipelineDataset()
        sampler = PipelineSampler(dataset)
        planner = self.build_planner(
            dataset,
            sampler,
            batch_size=2,
            max_batches=3,
        )
        collator = DescriptionCollator(
            SlowProcessor(completion=[]),
            training=True,
        )
        with OrderedBatchPrefetcher(
            planner=planner,
            collator=collator,
            num_workers=0,
            prefetch_factor=0,
            pin_memory=False,
        ) as pipeline:
            prepared = list(pipeline)
        self.assertEqual(
            [item.plan.sample_epochs for item in prepared],
            [(0, 0), (0, 0), (1, 1)],
        )
        self.assertEqual(prepared[-1].plan.next_epoch, 1)
        self.assertEqual(prepared[-1].plan.next_sample_offset, 2)
        dataset_epoch_before = dataset.epoch
        state = sampler_state_at_epoch(sampler, epoch=0)
        self.assertEqual(state["epoch"], 0)
        self.assertEqual(dataset.epoch, dataset_epoch_before)

    def test_worker_error_is_not_silently_swallowed(self) -> None:
        dataset = PipelineDataset()
        sampler = PipelineSampler(dataset)
        collator = DescriptionCollator(
            SlowProcessor(completion=[], fail_record="record-1"),
            training=True,
        )
        planner = self.build_planner(
            dataset,
            sampler,
            batch_size=1,
            max_batches=2,
        )
        with OrderedBatchPrefetcher(
            planner=planner,
            collator=collator,
            num_workers=2,
            prefetch_factor=1,
            pin_memory=False,
        ) as pipeline:
            self.assertEqual(
                next(pipeline).batch["record_ids"],
                ["record-0"],
            )
            with self.assertRaisesRegex(RuntimeError, "record-1"):
                next(pipeline)


if __name__ == "__main__":
    unittest.main()
