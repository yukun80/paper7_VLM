"""M7 CUDA allocator identity、同步遥测、resume 与失败证据测试。"""

from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from oa_groundrag.data.rs_general.io import read_json, read_jsonl
from oa_groundrag.training.vlm.cuda_telemetry import (
    ALLOCATOR_EXPANDABLE,
    ALLOCATOR_NATIVE,
    CUDA_TELEMETRY_SCHEMA,
    MICRO_CACHE_NONE,
    CudaMicrobatchTelemetry,
    CudaTelemetryPolicy,
    allocator_environment_identity,
)
from oa_groundrag.vlm.errors import ModelError


def _policy() -> CudaTelemetryPolicy:
    return CudaTelemetryPolicy(
        schema_version=CUDA_TELEMETRY_SCHEMA,
        allocator_profile=ALLOCATOR_NATIVE,
        microbatch_cache_policy=MICRO_CACHE_NONE,
        min_cuda_free_bytes=2**31,
        max_microbatches=16,
        synchronize_cuda=True,
        resource_profile_identity={"selection_sha256": "fixture"},
    )


class FakeCuda:
    def __init__(self, *, free_bytes=3 * 1024**3, allocated_bytes=1024) -> None:
        self.free_bytes = free_bytes
        self.allocated_bytes = allocated_bytes

    def patches(self):
        stack = ExitStack()
        stack.enter_context(
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(
                    name="fixture-gpu",
                    total_memory=24 * 1024**3,
                ),
            )
        )
        stack.enter_context(
            patch("torch.cuda.memory.get_allocator_backend", return_value="native")
        )
        stack.enter_context(patch("torch.cuda.synchronize"))
        stack.enter_context(
            patch(
                "torch.cuda.memory_stats",
                return_value={
                    "active_bytes.all.current": self.allocated_bytes,
                    "inactive_split_bytes.all.current": 0,
                    "num_alloc_retries": 0,
                    "num_ooms": 0,
                },
            )
        )
        stack.enter_context(
            patch(
                "torch.cuda.mem_get_info",
                return_value=(self.free_bytes, 24 * 1024**3),
            )
        )
        stack.enter_context(
            patch("torch.cuda.memory_allocated", return_value=self.allocated_bytes)
        )
        stack.enter_context(
            patch("torch.cuda.memory_reserved", return_value=self.allocated_bytes * 2)
        )
        stack.enter_context(patch("torch.cuda.empty_cache"))
        return stack


class CudaTelemetryTests(unittest.TestCase):
    def test_allocator_environment_is_strict(self) -> None:
        clean = {
            "PYTORCH_ALLOC_CONF": "",
            "PYTORCH_CUDA_ALLOC_CONF": "",
        }
        with patch.dict(os.environ, clean, clear=False):
            self.assertEqual(
                allocator_environment_identity(ALLOCATOR_NATIVE)["profile"],
                ALLOCATOR_NATIVE,
            )
            with self.assertRaises(ModelError):
                allocator_environment_identity(ALLOCATOR_EXPANDABLE)
        expandable = {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "",
        }
        with patch.dict(os.environ, expandable, clear=False):
            self.assertEqual(
                allocator_environment_identity(ALLOCATOR_EXPANDABLE)["profile"],
                ALLOCATOR_EXPANDABLE,
            )
            with self.assertRaises(ModelError):
                allocator_environment_identity(ALLOCATOR_NATIVE)

    def test_lifecycle_writes_durable_rows_and_resumes_exact_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"PYTORCH_ALLOC_CONF": "", "PYTORCH_CUDA_ALLOC_CONF": ""},
            clear=False,
        ), FakeCuda().patches():
            root = Path(temporary)
            telemetry = CudaMicrobatchTelemetry(
                policy=_policy(),
                device=torch.device("cuda:0"),
            )
            telemetry.start(root, completed_microbatches=0)
            telemetry.begin(
                {
                    "optimizer_step": 1,
                    "micro_step": 1,
                    "record_id": "record",
                }
            )
            telemetry.after_forward()
            telemetry.after_backward()
            telemetry.complete_after_release()
            telemetry.record_post_zero_grad(optimizer_step=1)

            rows = read_jsonl(root / "cuda_resource_telemetry.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["pre_forward"]["device_free_bytes"], 3 * 1024**3)
            self.assertEqual(
                read_json(root / "cuda_active_microbatch.json")["status"],
                "completed",
            )
            resumed = CudaMicrobatchTelemetry(
                policy=_policy(),
                device=torch.device("cuda:0"),
            )
            resumed.start(root, completed_microbatches=1)
            self.assertEqual(resumed.completed_microbatches, 1)

    def test_low_free_memory_preserves_last_planned_sample_without_cuda_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"PYTORCH_ALLOC_CONF": "", "PYTORCH_CUDA_ALLOC_CONF": ""},
            clear=False,
        ), FakeCuda(free_bytes=1024).patches():
            root = Path(temporary)
            telemetry = CudaMicrobatchTelemetry(
                policy=_policy(),
                device=torch.device("cuda:0"),
            )
            telemetry.start(root, completed_microbatches=0)
            with self.assertRaises(ModelError) as caught:
                telemetry.begin(
                    {
                        "optimizer_step": 1,
                        "micro_step": 1,
                        "record_id": "last-planned",
                    }
                )
            telemetry.persist_failure(
                caught.exception,
                last_completed_optimizer_step=0,
                last_completed_microbatches=0,
            )
            failure = read_json(root / "cuda_resource_failure.json")
            self.assertEqual(
                failure["last_planned_microbatch"]["record_id"],
                "last-planned",
            )
            self.assertEqual(failure["last_completed_microbatches"], 0)


if __name__ == "__main__":
    unittest.main()

