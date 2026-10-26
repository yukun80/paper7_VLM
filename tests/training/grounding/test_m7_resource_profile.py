"""M7 v4 资源画像、严格配置与最坏样本选择测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from oa_groundrag.training.grounding.config import (
    ALLOCATOR_PROFILE_NATIVE,
    CUDA_TELEMETRY_SCHEMA,
    QWEN35_LOSS_PROJECTION,
    STAGE5_CONFIG_SCHEMA_V4,
    load_stage5_config,
)
from oa_groundrag.training.grounding.resource_profile import (
    _profile_row,
    _worst_case_selection,
    verify_stage5_resource_profile,
)
from oa_groundrag.vlm.errors import ConfigError, ModelError


REPO = Path(__file__).resolve().parents[3]
NATIVE_CONFIG = (
    REPO
    / "configs/vlm/grounded/qwen35_4b_resource_gate_native_v4.yaml"
)
PROFILE_ROOT = REPO / "outputs/smoke/qwen35_4b_m7_schedule_profile_v4"


class M7ResourceProfileTests(unittest.TestCase):
    def test_v4_config_freezes_loss_allocator_and_telemetry(self) -> None:
        config = load_stage5_config(NATIVE_CONFIG)
        self.assertEqual(config.schema_version, STAGE5_CONFIG_SCHEMA_V4)
        self.assertIsNotNone(config.resource_contract)
        assert config.resource_contract is not None
        self.assertEqual(
            config.resource_contract.loss_projection,
            QWEN35_LOSS_PROJECTION,
        )
        self.assertEqual(
            config.resource_contract.allocator_profile,
            ALLOCATOR_PROFILE_NATIVE,
        )
        self.assertEqual(
            config.resource_contract.telemetry_schema,
            CUDA_TELEMETRY_SCHEMA,
        )
        self.assertEqual(config.resource_contract.min_cuda_free_bytes, 2**31)
        self.assertEqual(config.resource_contract.telemetry_max_microbatches, 16_000)
        self.assertEqual(config.resource_contract.profile_root, PROFILE_ROOT)

    def test_v4_unknown_resource_field_and_allocator_mismatch_fail_closed(self) -> None:
        row = yaml.safe_load(NATIVE_CONFIG.read_text(encoding="utf-8"))
        unknown = {**row, "resource": {**row["resource"], "unknown": True}}
        with patch(
            "oa_groundrag.training.grounding.config._load_yaml",
            return_value=unknown,
        ), self.assertRaises(ConfigError):
            load_stage5_config(NATIVE_CONFIG)
        mismatch = {
            **row,
            "resource": {
                **row["resource"],
                "allocator_profile": "expandable_segments",
                "microbatch_cache_policy": "post_backward_empty_cache",
            },
        }
        with patch(
            "oa_groundrag.training.grounding.config._load_yaml",
            return_value=mismatch,
        ), self.assertRaises(ConfigError):
            load_stage5_config(NATIVE_CONFIG)

    def test_profile_row_uses_causal_shift_and_fixed_footprint_formula(self) -> None:
        sample = SimpleNamespace(
            record_id="record",
            parent_id="parent",
            logical_role="mask_grounded_train",
            task_family="mask_grounded_region_description",
        )
        row = _profile_row(
            sequence_index=17,
            sample=sample,
            epoch=0,
            batch={
                "input_token_counts": [4],
                "image_counts": [1],
                "labels": torch.tensor([[-100, -100, 3, 4]]),
                "pixel_values": torch.zeros((4, 2), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            },
        )
        self.assertEqual(row["optimizer_step"], 2)
        self.assertEqual(row["micro_step"], 2)
        self.assertEqual(row["supervised_tokens"], 2)
        self.assertEqual(row["vision_grid_volume"], 16)
        self.assertEqual(row["vision_tokens"], 4)
        self.assertEqual(row["pixel_bytes"], 32)
        self.assertGreater(
            row["full_logits_fp32_ce_footprint_bytes"],
            row["projected_logits_fp32_ce_footprint_bytes"],
        )

    def test_profile_row_rejects_empty_shifted_supervision(self) -> None:
        sample = SimpleNamespace(
            record_id="record",
            parent_id="parent",
            logical_role="external_train",
            task_family="global_caption",
        )
        with self.assertRaises(ModelError):
            _profile_row(
                sequence_index=0,
                sample=sample,
                epoch=0,
                batch={
                    "input_token_counts": [2],
                    "image_counts": [1],
                    "labels": torch.tensor([[7, -100]]),
                    "pixel_values": torch.zeros((4, 2)),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                },
            )

    def test_worst_case_tie_breaks_to_earliest_sequence_and_coalesces(self) -> None:
        rows = []
        for index in range(2):
            rows.append(
                {
                    "sequence_index": index,
                    "record_id": f"record-{index}",
                    "parent_id": f"parent-{index}",
                    "logical_role": "mask_grounded_train",
                    "task_family": "mask_grounded_region_description",
                    "epoch": 0,
                    "input_tokens": 10,
                    "supervised_tokens": 5 + index,
                    "pixel_numel": 20,
                    "vision_grid_volume": 30,
                    "projected_composite_footprint_bytes": 40,
                }
            )
        selection = _worst_case_selection(rows)
        selected = selection["selected"]
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["sequence_index"], 0)
        self.assertEqual(
            selected[0]["reasons"],
            [
                "input_tokens",
                "pixel_numel",
                "projected_composite_footprint_bytes",
                "vision_grid_volume",
            ],
        )
        self.assertEqual(selected[1]["reasons"], ["supervised_tokens"])

    @unittest.skipUnless(PROFILE_ROOT.is_dir(), "需要本轮已发布的 16,000 条画像")
    def test_live_profile_verifies_strictly(self) -> None:
        identity = verify_stage5_resource_profile(load_stage5_config(NATIVE_CONFIG))
        self.assertEqual(identity["schedule_row_count"], 16_000)
        self.assertEqual(identity["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()

