#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互推理、图库选择与 presentation 导出的轻量回归测试。

运行：PYTHONPATH=SEG_Multi-Source_Landslides python -m unittest discover
-s SEG_Multi-Source_Landslides/tests -p 'test_inference_gallery.py' -v
"""

from __future__ import annotations

from pathlib import Path
import gc
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import torch

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.demo_app import build_demo
from qpsalm_seg.gallery import select_gallery_records
from qpsalm_seg.inference import CatalogEntry, PredictionResult, override_inference_item
from qpsalm_seg.presentation import save_presentation_result, write_gallery_html
from qpsalm_seg.schema import ActiveModalitySubset, ModalityBatch, ModalityInstance


def modality(name: str, family: str) -> ModalityInstance:
    product = "rgb" if family == "optical" else "elevation"
    return ModalityInstance(
        name=name,
        family=family,
        sensor=f"sensor_{family}",
        product_type=product,
        band_names=("R", "G", "B") if family == "optical" else ("elevation",),
        band_metadata=(),
        orbit="unknown",
        units="unknown",
        signed=False,
        image=torch.rand(3 if family == "optical" else 1, 16, 16),
        valid_mask=torch.ones(1, 16, 16),
        native_gsd_m=10.0,
        aligned_gsd_m=10.0,
    )


def fixture_item():
    optical, terrain = modality("optical_rgb", "optical"), modality("dem", "terrain")
    transform = {
        "source_hw": [16, 16], "target_hw": [16, 16], "resized_hw": [16, 16],
        "scale": 1.0, "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0,
    }
    row = {
        "sample_id": "sample-1", "parent_sample_id": "parent-1", "dataset_name": "dataset",
        "task_family": "global_landslide_segmentation", "template_id": "generic_landslide_v2",
        "instruction": {"text": "Segment all landslide regions."}, "spatial": {"gsd_m": 10.0},
    }
    item = {
        "instances": [optical, terrain], "full_instances": [optical, terrain],
        "active_subset": ActiveModalitySubset(("dem", "optical_rgb"), (), "full", True),
        "mask": torch.zeros(1, 16, 16), "valid_mask": torch.ones(1, 16, 16),
        "metadata": {
            "sample_id": "sample-1", "parent_sample_id": "parent-1", "dataset_name": "dataset",
            "task_family": "global_landslide_segmentation", "template_id": "generic_landslide_v2",
            "instruction": "Segment all landslide regions.", "target_size": 16,
            "resize_transform": transform, "active_modalities": ["dem", "optical_rgb"],
        },
        "proposal_context_text": "old", "condition_prompt_text": "old", "evidence_reasoning_text": "old",
        "full_proposal_context_text": "old", "full_condition_prompt_text": "old",
        "full_evidence_reasoning_text": "old", "visual_evidence_key": "qmv3-parent:parent-1",
        "component_masks": torch.zeros(0, 16, 16),
    }
    return row, item


class InferenceGalleryTest(unittest.TestCase):
    def test_prompt_and_modality_override_do_not_mutate_source(self) -> None:
        row, item = fixture_item()
        output = override_inference_item(
            item,
            row,
            QPSalmConfig(min_component_area_pixels=1),
            active_modalities=["optical_rgb"],
            instruction_override="Segment the requested scar.",
        )
        self.assertEqual([value.name for value in output["instances"]], ["optical_rgb"])
        self.assertIn("Segment the requested scar.", output["proposal_context_text"])
        self.assertIn("Segment the requested scar.", output["condition_prompt_text"])
        self.assertNotIn("terrain products", output["evidence_reasoning_text"])
        self.assertTrue(output["metadata"]["gt_is_reference_only"])
        self.assertEqual(item["proposal_context_text"], "old")
        self.assertEqual(item["metadata"]["active_modalities"], ["dem", "optical_rgb"])
        with self.assertRaisesRegex(ValueError, "至少需要一个"):
            override_inference_item(item, row, QPSalmConfig(), active_modalities=[])

    def test_gallery_selection_is_deterministic_and_balanced(self) -> None:
        records = []
        for dataset in ("A", "B", "Sen12Landslides"):
            for index, score in enumerate((0.0, 0.5, 0.9, 1.0)):
                records.append({
                    "sample_id": f"{dataset}-{index}", "parent_sample_id": f"{dataset}-p{index}",
                    "dataset_name": dataset, "family_combo": "multispectral+terrain",
                    "task_family": "global_landslide_segmentation", "target_area": 10.0,
                    "target_area_px_bin": "tiny_le_16px" if index == 0 else "medium_65_256px",
                    "final_dice": score,
                })
        first = select_gallery_records(records, max_items=30, seed=7)
        second = select_gallery_records(records, max_items=30, seed=7)
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        categories = {tag for row in first for tag in row.get("gallery_tags", [row["gallery_category"]])}
        self.assertTrue({"strong", "typical", "failure", "weak_modality_sen12", "small_target"} <= categories)
        self.assertEqual(len(first), len({row["sample_id"] for row in first}))

    def test_presentation_export_contains_no_oracle_output(self) -> None:
        row, item = fixture_item()
        batch = ModalityBatch(
            instances=[item["instances"]], full_instances=[item["full_instances"]],
            active_subsets=[item["active_subset"]], mask=item["mask"][None],
            valid_mask=item["valid_mask"][None], metadata=[item["metadata"]],
            proposal_context_text=["task"], condition_prompt_text=["condition"],
            evidence_reasoning_text=["reason"], full_proposal_context_text=["task"],
            full_condition_prompt_text=["condition"], full_evidence_reasoning_text=["reason"],
            visual_evidence_key=[item["visual_evidence_key"]], component_masks=[item["component_masks"]],
        )
        result = PredictionResult(
            sample_id="sample-1", checkpoint_step=1, batch=batch,
            probability=np.zeros((16, 16), dtype=np.float32), final_mask=np.zeros((16, 16), dtype=np.uint8),
            selected_proposal=np.zeros((16, 16), dtype=np.uint8), selected_query=0,
            ground_truth=np.zeros((16, 16), dtype=np.uint8), valid_mask=np.ones((16, 16), dtype=np.uint8),
            restored_final_mask=np.zeros((16, 16), dtype=np.uint8),
            metrics={"dice": 1.0, "iou": 1.0}, metrics_are_reference_only=False,
            latency_seconds=0.1, diagnostics={"mask_area": 0},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = save_presentation_result(result, root, category="strong", stratum="test")
            self.assertFalse(exported["contains_oracle_output"])
            self.assertNotIn("oracle", str(exported["mask_paths"]).lower())
            self.assertTrue(Path(exported["overview_path"]).exists())
            write_gallery_html([exported], root / "gallery_index.html")
            self.assertNotIn("oracle", (root / "gallery_index.html").read_text().lower())

    def test_gradio_blocks_build_with_catalog(self) -> None:
        entry = CatalogEntry(
            "sample", "parent", "dataset", "task", "optical", "optical_rgb",
            "Segment all landslide regions.", ("optical_rgb",),
        )

        class FakeSession:
            split = "val"
            config = SimpleNamespace(eval_threshold=0.5)
            catalog = (entry,)

            def sample_defaults(self, _sample_id):
                return {**entry.as_dict(), "condition": "all landslide regions"}

            def filter_catalog(self, **_kwargs):
                return [entry]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            demo = build_demo(FakeSession())
            self.assertGreater(len(demo.blocks), 10)
            demo.close()
            del demo
            gc.collect()


if __name__ == "__main__":
    unittest.main()
