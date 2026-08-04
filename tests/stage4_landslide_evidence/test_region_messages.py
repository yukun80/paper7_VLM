from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixture_helpers import target_record

from oa_groundrag.landslide_evidence.region_contracts import validate_region_record
from oa_groundrag.phase4.errors import ContractError, ReasonCode
from oa_groundrag.phase4.messages import build_mask_grounded_region_messages


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


if __name__ == "__main__":
    unittest.main()
