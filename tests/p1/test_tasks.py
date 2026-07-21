"""Synthetic T1--T4 expansion and modality-condition tests."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sami_gsd.contracts.canonical import ArtifactRef, CanonicalParentV3
from sami_gsd.data.materialize import materialize_spatial_parent
from sami_gsd.data.tasks import (
    FixedRegionPrediction,
    RegionAnswer,
    TaskExpansionError,
    expand_task_views,
)
from tests.p1.test_materialization import spatial_input


SHA = "c" * 64


def freeze_split(parent: CanonicalParentV3, split: str = "train") -> CanonicalParentV3:
    """Freeze a synthetic audit parent before task expansion."""

    payload = parent.model_dump(mode="json")
    payload["split"] = split
    return CanonicalParentV3.model_validate(payload)


class TaskExpansionTests(unittest.TestCase):
    """Verify T1--T4 are derived only from frozen parents and real inputs."""

    def test_positive_parent_expands_t1_t2_t3_t4_without_modality_copies(self) -> None:
        """T3/T4 require answer/prediction bindings; conditions remain separate."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = freeze_split(
                materialize_spatial_parent(spatial_input(), benchmark_root=root, canvas_hw=(6, 6)).parent
            )
            key = (parent.parent_id, "region-001")
            expansion = expand_task_views(
                (parent,),
                region_answers={
                    key: RegionAnswer(
                        answer_ref=ArtifactRef(path="answers/synthetic-parent-001/region-001.json", sha256=SHA),
                        annotation_origin="expert",
                    )
                },
                fixed_predictions={
                    key: FixedRegionPrediction(
                        mask_ref=ArtifactRef(path="predictions/synthetic-parent-001/region-001.npy", sha256=SHA),
                        bbox_half_open=(3, 3, 6, 5),
                        checkpoint_fingerprint=SHA,
                    )
                },
            )
            self.assertEqual(
                {task.task_type for task in expansion.tasks},
                {"t1_global", "t2_referring", "t3_gt_region", "t4_predicted_region"},
            )
            self.assertEqual(len(expansion.tasks), 4)
            self.assertTrue(expansion.evaluation_conditions["task_rows_are_not_duplicated_by_modality"])
            self.assertEqual(len(expansion.evaluation_conditions["parents"]), 1)
            self.assertEqual(expansion.evaluation_conditions["mask_modes"], ["gt_mask", "fixed_prediction", "end_to_end"])

    def test_no_target_parent_emits_one_empty_aware_t1(self) -> None:
        """A valid empty global mask becomes a derived no-target view."""

        item = spatial_input()
        item = replace(item, global_mask=np.zeros_like(item.global_mask), referring_regions=())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = freeze_split(
                materialize_spatial_parent(item, benchmark_root=root, canvas_hw=(6, 6)).parent,
                "val",
            )
            expansion = expand_task_views((parent,))
            self.assertEqual(len(expansion.tasks), 1)
            task = expansion.tasks[0]
            self.assertEqual(task.task_type, "t1_global")
            self.assertEqual(task.target_status, "no_target")
            self.assertEqual(task.annotation_origin, "derived_no_target")

    def test_audit_parent_and_unanswered_prediction_are_rejected(self) -> None:
        """Task order and T4 provenance cannot be bypassed."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_parent = materialize_spatial_parent(spatial_input(), benchmark_root=root, canvas_hw=(6, 6)).parent
            with self.assertRaisesRegex(TaskExpansionError, "frozen parent splits"):
                expand_task_views((audit_parent,))
            parent = freeze_split(audit_parent)
            key = (parent.parent_id, "region-001")
            with self.assertRaisesRegex(TaskExpansionError, "matching reviewed answer"):
                expand_task_views(
                    (parent,),
                    fixed_predictions={
                        key: FixedRegionPrediction(
                            mask_ref=ArtifactRef(path="predictions/fixed.npy", sha256=SHA),
                            bbox_half_open=(1, 1, 3, 3),
                            checkpoint_fingerprint=SHA,
                        )
                    },
                )


if __name__ == "__main__":
    unittest.main()
