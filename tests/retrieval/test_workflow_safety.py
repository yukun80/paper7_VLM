from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from oa_groundrag.vlm.checkpoint import CheckpointManager, TrainingCursor
from oa_groundrag.vlm.errors import ContractError
from oa_groundrag.retrieval.workflow import _reject_sealed_path


class WorkflowSafetyTest(unittest.TestCase):
    def test_sealed_and_test_paths_rejected_but_dev_allowed(self) -> None:
        _reject_sealed_path(Path("/tmp/oa_grounded_eval_dev"), label="dev")
        with self.assertRaises(ContractError):
            _reject_sealed_path(Path("/tmp/sealed_test"), label="sealed")
        with self.assertRaises(ContractError):
            _reject_sealed_path(Path("/tmp/eval-test"), label="test")

    def test_checkpoint_inspect_and_trainable_load_skip_training_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            manager = CheckpointManager()
            layout = {
                "physical_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "effective_batch_size": 1,
                "input_pipeline_backend": "synchronous.v1",
                "num_workers": 0,
                "prefetch_factor": 0,
                "pin_memory": False,
                "sample_trace_schema_version": "rs_vlm.sample_trace.v1",
            }
            common = {
                "expected_config_semantic_sha256": "a" * 64,
                "expected_benchmark_identity": {"id": "b"},
                "expected_validation_selection_identity": {"id": "v"},
                "expected_model_identity": {"id": "m"},
                "expected_processor_identity": {"id": "p"},
                "expected_training_layout": layout,
                "expected_trainable_names": ("adapter.weight",),
            }
            manager.save(
                root,
                trainable_state={"adapter.weight": torch.ones(2)},
                cursor=TrainingCursor(epoch=0, sample_offset=0, global_step=1, micro_step=0),
                config_semantic_sha256="a" * 64,
                benchmark_identity={"id": "b"}, validation_selection_identity={"id": "v"},
                model_identity={"id": "m"}, processor_identity={"id": "p"},
                training_layout=layout, trainable_names=("adapter.weight",),
                selection_metrics={"loss": 0.5},
            )
            with patch("torch.load", side_effect=AssertionError("training state must not load")):
                manifest = manager.inspect_trainable(root, **common)
                loaded_manifest, tensors = manager.load_trainable(root, **common)
            self.assertEqual(manifest, loaded_manifest)
            self.assertTrue(torch.equal(tensors["adapter.weight"], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
