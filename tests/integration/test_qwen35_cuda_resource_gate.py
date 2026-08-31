"""显式启用的 Qwen3.5-4B RTX 4090 D 资源门，不纳入默认 CPU 回归。"""

from __future__ import annotations

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import unittest

from PIL import Image
import torch

from oa_groundrag.vlm.backends.assets import verify_model_asset_ledger
from oa_groundrag.vlm.backends.qwen3_5.model import Qwen35ModelAdapter
from oa_groundrag.vlm.backends.qwen3_5.processing import Qwen3_5ProcessorAdapter
from oa_groundrag.vlm.config import CONFIG_SCHEMA_VERSION_V3, load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPOSITORY_ROOT / "models_zoo/Qwen3.5-4B-r851bf6e8"
LEDGER_PATH = (
    REPOSITORY_ROOT
    / "configs/vlm/assets/qwen3_5_4b_r851bf6e8.asset_ledger.json"
)
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _tensor_batch(encoded, device: torch.device, *, training: bool):
    output = {}
    for key, value in encoded.tensors.items():
        if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
            value = value.unsqueeze(0)
        output[key] = value.to(device)
    if training:
        assert encoded.labels is not None
        output["labels"] = encoded.labels.unsqueeze(0).to(device)
    return output


@unittest.skipUnless(
    os.environ.get("OA_GROUNDRAG_RUN_QWEN35_CUDA_GATE") == "1",
    "需要负责人显式授权 Qwen3.5 CUDA 资源门",
)
class Qwen35CudaResourceGate(unittest.TestCase):
    def test_real_inference_one_step_and_twenty_steps(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        self.assertGreaterEqual(properties.total_memory, 24_000_000_000)
        assets = verify_model_asset_ledger(
            LEDGER_PATH,
            expected_backend="qwen3_5",
            expected_repo_id="Qwen/Qwen3.5-4B",
            expected_revision=REVISION,
            expected_model_root=MODEL_ROOT,
        )
        base = load_config(
            REPOSITORY_ROOT / "configs/vlm/rs_general/bounded_smoke.yaml"
        )
        model_section = replace(
            base.model,
            path=MODEL_ROOT,
            processor_path=MODEL_ROOT,
            backend="qwen3_5",
            hub_repo_id="Qwen/Qwen3.5-4B",
            hub_revision=REVISION,
            asset_ledger_path=LEDGER_PATH,
        )
        prompt_config = replace(
            base,
            schema_version=CONFIG_SCHEMA_VERSION_V3,
            model=model_section,
        )
        lora_adaptation = load_config(
            REPOSITORY_ROOT
            / "configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml"
        ).adaptation
        processor = Qwen3_5ProcessorAdapter(
            processor_path=MODEL_ROOT,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=prompt_config.limits.min_pixels,
            max_pixels=prompt_config.limits.max_pixels,
            max_images=prompt_config.limits.max_images,
            max_input_tokens=prompt_config.limits.max_input_tokens,
            enable_thinking=False,
            asset_identity=assets.identity_dict(),
        )
        image = Image.new("RGB", (64, 64), "white")
        inference_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ]
        encoded_inference = processor.encode_inference(inference_messages)
        inference_batch = _tensor_batch(
            encoded_inference,
            device,
            training=False,
        )
        torch.cuda.reset_peak_memory_stats(device)
        prompt_model = Qwen35ModelAdapter.load(
            model_section,
            prompt_config.adaptation,
            assets=assets,
            device=device,
            gradient_checkpointing=False,
        )
        generated = prompt_model.generate_text(
            inference_batch,
            processor=processor.processor,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        torch.cuda.synchronize(device)
        inference_peak = torch.cuda.max_memory_allocated(device)
        self.assertEqual(len(generated), 1)
        del prompt_model, inference_batch
        gc.collect()
        torch.cuda.empty_cache()

        training_messages = [
            *inference_messages,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "A plain white square is visible."}
                ],
            },
        ]
        encoded_training = processor.encode_training(training_messages)
        training_batch = _tensor_batch(
            encoded_training,
            device,
            training=True,
        )
        lora_config = replace(prompt_config, adaptation=lora_adaptation)
        torch.cuda.reset_peak_memory_stats(device)
        lora_model = Qwen35ModelAdapter.load(
            model_section,
            lora_config.adaptation,
            assets=assets,
            device=device,
            gradient_checkpointing=True,
        )
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in lora_model.model.parameters()
                if parameter.requires_grad
            ],
            lr=2e-4,
            weight_decay=0.01,
        )
        losses = []
        first_step_peak = None
        for step in range(1, 21):
            result = lora_model.forward(training_batch)
            self.assertTrue(bool(torch.isfinite(result.loss)))
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in lora_model.model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(result.loss.detach().cpu()))
            del result
            torch.cuda.synchronize(device)
            if step == 1:
                first_step_peak = torch.cuda.max_memory_allocated(device)
        training_peak = torch.cuda.max_memory_allocated(device)
        self.assertIsNotNone(first_step_peak)
        self.assertTrue(all(value == value for value in losses))
        self.assertLess(training_peak, properties.total_memory)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "gpu": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "inference_peak_bytes": inference_peak,
                    "step_1_peak_bytes": first_step_peak,
                    "step_20_peak_bytes": training_peak,
                    "trainable_parameters": lora_model.trainable_parameter_count,
                    "steps": 20,
                    "loss_step_1": losses[0],
                    "loss_step_20": losses[-1],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
