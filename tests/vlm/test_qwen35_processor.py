"""Qwen3.5 Jinja、thinking 与多图 processor 合同。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from oa_groundrag.vlm.backends.qwen3_5.processing import (
    QWEN3_5_PROCESSOR_IDENTITY_FILES,
    Qwen3_5ProcessorAdapter,
)
from oa_groundrag.vlm.errors import ProcessingError, ReasonCode
from oa_groundrag.vlm.processing import single_inference_tensor_batch


def _asset_identity(root: Path) -> dict[str, object]:
    return {
        "backend": "qwen3_5",
        "hub_repo_id": "Qwen/Qwen3.5-4B",
        "hub_revision": "1" * 40,
        "model_root": str(root),
        "asset_ledger_path": str(root.parent / "ledger.json"),
        "asset_ledger_file_sha256": "a" * 64,
        "asset_ledger_payload_sha256": "b" * 64,
        "asset_total_size_bytes": 1,
        "asset_file_count": 14,
        "weight_shards": [
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
        ],
    }


def _messages(image_count: int, *, assistant: bool) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "image", "image": object()} for _ in range(image_count)
    ]
    content.append({"type": "text", "text": "Describe."})
    messages: list[dict[str, object]] = [
        {"role": "user", "content": content}
    ]
    if assistant:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer."}],
            }
        )
    return messages


class _Tokenizer:
    pad_token_id = 0


class _FakeProcessor:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(dict(kwargs))
        thinking = kwargs.get("enable_thinking", "missing")
        prefix = "P-NT-" if thinking is False else "P-T-"
        if kwargs.get("add_generation_prompt"):
            return prefix
        if messages and messages[-1].get("role") == "assistant":
            return prefix + "ANSWER"
        return prefix + "USER"

    def __call__(self, *, text, images=None, videos=None, **kwargs):
        del videos, kwargs
        ids = torch.tensor([[ord(value) for value in text[0]]], dtype=torch.long)
        output = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.zeros_like(ids),
        }
        if images is not None:
            count = len(images)
            output.update(
                {
                    "pixel_values": torch.ones((count, 8)),
                    "image_grid_thw": torch.tensor(
                        [[1, 1, 1]] * count,
                        dtype=torch.long,
                    ),
                }
            )
        return output


def _vision_info(messages):
    images = [
        item["image"]
        for message in messages
        for item in message["content"]
        if item["type"] == "image"
    ]
    return images, None


class Qwen35ProcessorTests(unittest.TestCase):
    def _adapter(
        self,
        root: Path,
        fake: _FakeProcessor,
        *,
        thinking: bool,
    ) -> Qwen3_5ProcessorAdapter:
        for name in QWEN3_5_PROCESSOR_IDENTITY_FILES:
            (root / name).write_text(f"fixture:{name}", encoding="utf-8")
        with patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=fake,
        ):
            return Qwen3_5ProcessorAdapter(
                processor_path=root,
                local_files_only=True,
                trust_remote_code=False,
                min_pixels=28 * 28 * 16,
                max_pixels=28 * 28 * 256,
                max_images=5,
                max_input_tokens=2048,
                enable_thinking=thinking,
                asset_identity=_asset_identity(root),
            )

    def test_nonthinking_training_has_assistant_only_labels_and_jinja_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = _FakeProcessor()
            adapter = self._adapter(root, fake, thinking=False)
            with patch("qwen_vl_utils.process_vision_info", side_effect=_vision_info):
                encoded = adapter.encode_training(
                    _messages(1, assistant=True)
                )
            self.assertEqual(encoded.image_count, 1)
            self.assertIsNotNone(encoded.labels)
            self.assertTrue(bool((encoded.labels == -100).any()))
            self.assertTrue(bool((encoded.labels != -100).any()))
            self.assertTrue(
                all(
                    call.get("enable_thinking") is False
                    for call in fake.template_calls
                )
            )
            identity = adapter.identity()
            self.assertEqual(identity["backend"], "qwen3_5")
            self.assertEqual(identity["chat_template_format"], "jinja")
            self.assertIn("chat_template.jinja", identity["files"])
            self.assertNotIn("chat_template.json", identity["files"])

    def test_one_and_three_image_tensor_keys_are_stable(self) -> None:
        for count in (1, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fake = _FakeProcessor()
                adapter = self._adapter(root, fake, thinking=False)
                with patch(
                    "qwen_vl_utils.process_vision_info",
                    side_effect=_vision_info,
                ):
                    encoded = adapter.encode_inference(
                        _messages(count, assistant=False)
                    )
                self.assertEqual(encoded.image_count, count)
                self.assertEqual(
                    set(encoded.tensors),
                    {
                        "input_ids",
                        "attention_mask",
                        "mm_token_type_ids",
                        "pixel_values",
                        "image_grid_thw",
                    },
                )
                batch = single_inference_tensor_batch(encoded)
                self.assertEqual(batch["input_ids"].shape[0], 1)
                self.assertEqual(batch["image_grid_thw"].shape[0], count)

    def test_text_only_and_thinking_switch_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = _FakeProcessor()
            nonthinking = self._adapter(root, fake, thinking=False)
            text = nonthinking.encode_text_inference(
                [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]
            )
            self.assertEqual(text.image_count, 0)
            with patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=fake,
            ):
                thinking = nonthinking.with_thinking(True)
            with patch("qwen_vl_utils.process_vision_info", side_effect=_vision_info):
                thinking.encode_inference(_messages(1, assistant=False))
            self.assertTrue(thinking.thinking_enabled)
            self.assertIs(fake.template_calls[-1]["enable_thinking"], True)
            with self.assertRaises(ProcessingError) as caught:
                thinking.encode_training(_messages(1, assistant=True))
            self.assertEqual(
                caught.exception.code,
                ReasonCode.MODEL_IDENTITY_MISMATCH,
            )


if __name__ == "__main__":
    unittest.main()
