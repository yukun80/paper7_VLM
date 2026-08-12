from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tests.data.rs_general.fixture_helpers import make_all_sources, write_build_config
from oa_groundrag.data.rs_general.adapters import (
    DisasterM3Adapter,
    MMRS1MAdapter,
    RSGPTAdapter,
)
from oa_groundrag.data.rs_general.config import load_build_config
from oa_groundrag.data.rs_general.builder import _enabled_adapters
from oa_groundrag.data.rs_general.contracts import TaskFamily
from oa_groundrag.data.rs_general.errors import ReasonCode


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.roots = make_all_sources(self.base / "sources")
        self.config = load_build_config(
            write_build_config(
                self.base / "config.yaml",
                self.roots,
                self.base / "out",
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builder_enables_only_three_external_adapters(self) -> None:
        self.assertEqual(
            [type(adapter).__name__ for adapter in _enabled_adapters(self.config)],
            ["RSGPTAdapter", "MMRS1MAdapter", "DisasterM3Adapter"],
        )

    def test_rsgpt_valid_reject_and_duplicate(self) -> None:
        result = RSGPTAdapter(self.config).scan()
        tasks = Counter(row.task_family for row in result.examples)
        reasons = Counter(row.reason_code for row in result.skips)
        self.assertGreaterEqual(tasks[TaskFamily.GLOBAL_CAPTION], 2)
        self.assertEqual(tasks[TaskFamily.VISUAL_QA], 4)
        self.assertEqual(tasks[TaskFamily.OBJECT_COUNT], 1)
        self.assertEqual(tasks[TaskFamily.SCENE_UNDERSTANDING], 1)
        self.assertEqual(reasons[ReasonCode.DUPLICATE_RECORD], 1)
        self.assertGreaterEqual(reasons[ReasonCode.UNSUPPORTED_CLAIM], 2)
        self.assertEqual(reasons[ReasonCode.UNUSED_ASSET], 1)
        self.assertEqual(reasons[ReasonCode.UNSUPPORTED_TASK], 1)
        self.assertEqual(result.audit["unreferenced_image_count"], 1)
        quantity = next(
            row
            for row in result.examples
            if row.deterministic_facts.get("qa_type") == "quantity"
        )
        self.assertEqual(quantity.task_family, TaskFamily.OBJECT_COUNT)
        self.assertEqual(quantity.supervision_kind.value, "numeric_qa")
        color = next(
            row
            for row in result.examples
            if row.deterministic_facts.get("qa_type") == "color"
        )
        self.assertEqual(color.supervision_kind.value, "short_qa")

    def test_mmrs_multireference_duplicate_reverse_and_zero_bbox(self) -> None:
        result = MMRS1MAdapter(self.config).scan()
        caption = next(
            row
            for row in result.examples
            if row.task_family is TaskFamily.GLOBAL_CAPTION
        )
        self.assertEqual(len(caption.reference_responses), 2)
        reasons = Counter(row.reason_code for row in result.skips)
        self.assertEqual(reasons[ReasonCode.DUPLICATE_REFERENCE], 1)
        self.assertEqual(reasons[ReasonCode.DUPLICATE_RECORD], 1)
        self.assertEqual(reasons[ReasonCode.REVERSE_GROUNDING_DIRECTION], 1)
        self.assertEqual(reasons[ReasonCode.BBOX_ZERO_AREA], 1)
        bbox = next(
            row
            for row in result.examples
            if row.task_family is TaskFamily.BBOX_REGION_CAPTION
        )
        self.assertEqual(
            bbox.target["source_convention"], "xyxy_normalized_top_left"
        )
        self.assertEqual(bbox.input_layout.value, "bbox_region")
        self.assertEqual(bbox.supervision_kind.value, "region_description")

    def test_disaster_text_task_matrix_and_refseg_rejection(self) -> None:
        full_config = load_build_config(
            write_build_config(
                self.base / "disaster-full.yaml",
                self.roots,
                self.base / "unused-full-output",
                profile="full",
                max_assets=None,
            )
        )
        result = DisasterM3Adapter(full_config).scan(deep=True)
        tasks = Counter(row.task_family for row in result.examples)
        reasons = Counter(row.reason_code for row in result.skips)
        for task in (
            TaskFamily.SCENE_UNDERSTANDING,
            TaskFamily.OBJECT_COUNT,
            TaskFamily.SPATIAL_RELATION,
            TaskFamily.VISIBLE_CHANGE_REPORT,
        ):
            self.assertGreater(tasks[task], 0)
        self.assertEqual(reasons[ReasonCode.UNSUPPORTED_MODALITY], 1)
        self.assertEqual(reasons[ReasonCode.UNSUPPORTED_TASK], 3)
        self.assertEqual(result.audit["refseg_excluded_count"], 2)
        self.assertFalse(result.audit["mask_archive_read"])
        self.assertEqual(result.audit["path_rewrite_count"], 1)
        relation = next(
            row
            for row in result.examples
            if row.task_family is TaskFamily.SPATIAL_RELATION
        )
        self.assertEqual(len(relation.target["boxes"]), 2)
        self.assertEqual(relation.input_layout.value, "boxed_image")
        self.assertEqual(
            relation.deterministic_facts["source_object_candidate_count"],
            4,
        )
        self.assertEqual(
            relation.deterministic_facts["boxed_object_roles"],
            [
                {"object_key": "2", "visual_role": "red_box"},
                {"object_key": "4", "visual_role": "blue_box"},
            ],
        )
        report = next(
            row
            for row in result.examples
            if row.task_family is TaskFamily.VISIBLE_CHANGE_REPORT
        )
        self.assertEqual(
            [asset.role for asset in report.assets],
            ["pre_image", "post_image"],
        )
        single = next(
            row
            for row in result.examples
            if row.task_family is TaskFamily.OBJECT_COUNT
        )
        self.assertEqual([asset.role for asset in single.assets], ["image"])
