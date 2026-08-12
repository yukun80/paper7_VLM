from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fixture_helpers import target_record

from oa_groundrag.landslide_evidence.region_contracts import validate_region_record
from oa_groundrag.phase4.errors import ContractError, ReasonCode
from oa_groundrag.phase4.messages import build_mask_grounded_region_messages
from oa_groundrag.phase4.outputs import (
    REGION_OUTPUT_SCHEMA_VERSION,
)


class RegionMessagesTest(unittest.TestCase):
    def test_formal_message_image_order_and_overlay_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = target_record(root)
            validate_region_record(record, expected_split="val")
            messages = build_mask_grounded_region_messages(record, asset_root=root)
        content = messages[0]["content"]
        image_paths = [item["image"] for item in content if item["type"] == "image"]
        self.assertEqual(
            [Path(path).parent.name for path in image_paths],
            ["optical_full", "binary_mask", "context_crop"],
        )
        self.assertNotIn("audit_overlay", " ".join(image_paths))
        self.assertIn("白色仅表示关注位置", content[2]["text"])

    def test_formal_and_audit_roles_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = target_record(root)
            record["formal_model_input_roles"].append("audit_overlay")
            with self.assertRaises(Exception):
                validate_region_record(record)

    def test_overlay_baseline_requires_explicit_audit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = target_record(root)
            with self.assertRaises(ContractError) as raised:
                build_mask_grounded_region_messages(
                    record,
                    asset_root=root,
                    representation_mode="overlay_audit_baseline",
                )
            self.assertEqual(raised.exception.code, ReasonCode.ASSET_ROLE_LEAKAGE)
            messages = build_mask_grounded_region_messages(
                record,
                asset_root=root,
                representation_mode="overlay_audit_baseline",
                allow_audit_only=True,
            )
        images = [item for item in messages[0]["content"] if item["type"] == "image"]
        self.assertEqual(len(images), 1)
        self.assertIn("audit_overlay", images[0]["image"])

    def test_full_plus_mask_uses_two_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = target_record(root)
            messages = build_mask_grounded_region_messages(
                record,
                asset_root=root,
                representation_mode="full_plus_mask",
            )
        images = [item for item in messages[0]["content"] if item["type"] == "image"]
        self.assertEqual(len(images), 2)

    def test_v2_message_embeds_complete_strict_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            messages = build_mask_grounded_region_messages(
                target_record(root),
                asset_root=root,
            )
        instruction = messages[0]["content"][-1]["text"]
        contract = json.loads(instruction.split("\nContract: ", maxsplit=1)[1])
        output_contract = contract["strict_output_contract"]
        self.assertFalse(output_contract["additional_properties"])
        self.assertEqual(
            output_contract["properties"]["schema_version"]["const"],
            REGION_OUTPUT_SCHEMA_VERSION,
        )
        self.assertEqual(
            output_contract["properties"]["target_status"]["const"],
            "target_present",
        )
        self.assertEqual(
            set(output_contract["properties"]["target_appearance"]["properties"]),
            {
                "tone", "texture", "vegetation_or_exposure", "homogeneity",
                "boundary_visibility",
            },
        )
        self.assertEqual(
            output_contract["properties"]["surrounding_environment"]
            ["properties"]["land_cover"]["type"],
            "array",
        )
        self.assertEqual(
            output_contract["properties"]["region_context_contrast"]
            ["properties"]["adjacency"]["type"],
            "array",
        )
        self.assertEqual(
            output_contract["properties"]["evidence_sufficiency"]["enum"],
            ["sufficient", "limited", "insufficient", "not_applicable"],
        )
        self.assertNotIn("json_template", output_contract)
        self.assertNotIn("mask 指定区域尚待依据当前影像核验", instruction)
        self.assertIn("不要复制统一的保守答案", instruction)
        self.assertIn("英文 ASCII 枚举，禁止翻译", instruction)


if __name__ == "__main__":
    unittest.main()
