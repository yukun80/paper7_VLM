"""显式启用的 Qwen3.5-4B Gate B 单记录 CUDA 生成预检。"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from oa_groundrag.evaluation.rs_general.generation import (
    _configure_determinism,
    _move_batch,
    _verify_selected_dataset,
)
from oa_groundrag.evaluation.rs_general.selection import (
    load_gate_b_selection,
    prepare_gate_b,
)
from oa_groundrag.training.vlm.trainer import training_layout_identity
from oa_groundrag.vlm.backends import (
    build_model_adapter,
    build_processor_adapter,
)
from oa_groundrag.vlm.checkpoint import CheckpointManager
from oa_groundrag.vlm.processing import DescriptionCollator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "configs/vlm/rs_general/rs_generaldesc_gate_b_qwen35_4b_v2.yaml"
)
TRAINING_ROOT = (
    REPOSITORY_ROOT
    / "outputs/phase4_rs_vlm/rs_vlm_lora_qwen35_4b_r851bf6e8_b1a16_v1"
)


@unittest.skipUnless(
    os.environ.get("OA_GROUNDRAG_RUN_QWEN35_GATE_B_CUDA_SMOKE") == "1",
    "需要负责人显式授权 Qwen3.5 Gate B CUDA 预检",
)
class Qwen35GateBCudaSmoke(unittest.TestCase):
    def test_base_and_best_adapter_generate_the_same_frozen_record(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        self.assertGreaterEqual(properties.total_memory, 24_000_000_000)
        _configure_determinism()

        with tempfile.TemporaryDirectory(
            prefix="qwen35_gate_b_cuda_smoke_"
        ) as temporary:
            temporary_root = Path(temporary)
            selection_root = prepare_gate_b(
                PROTOCOL_PATH,
                training_root=TRAINING_ROOT,
                output_root=temporary_root / "selection",
            )
            context = load_gate_b_selection(
                PROTOCOL_PATH,
                selection_root / "gate_b_selection.json",
            )
            dataset = _verify_selected_dataset(
                context,
                derived_root=temporary_root / "derived",
            )
            sample = dataset[0]
            results: dict[str, dict[str, object]] = {}

            for model_role in ("base", "adapter"):
                config = (
                    context.protocol_source.base_config
                    if model_role == "base"
                    else context.protocol_source.adapter_config
                )
                processor = build_processor_adapter(config)
                self.assertEqual(
                    processor.identity(),
                    context.frozen_protocol["static_protocol"][
                        "processor_identity"
                    ],
                )
                model = build_model_adapter(
                    config,
                    device=device,
                    gradient_checkpointing=False,
                )
                model_identity = model.identity.to_dict()
                self.assertEqual(
                    model_identity,
                    context.frozen_protocol["static_protocol"][
                        "model_identity"
                    ],
                )
                if model_role == "adapter":
                    best = context.frozen_protocol["training_run"][
                        "best_checkpoint"
                    ]
                    payload = CheckpointManager().load(
                        TRAINING_ROOT / best["relative_path"],
                        expected_config_semantic_sha256=config.semantic_sha256,
                        expected_benchmark_identity=(
                            context.access.identity.training_identity_dict()
                        ),
                        expected_validation_selection_identity=None,
                        expected_model_identity=model_identity,
                        expected_processor_identity=processor.identity(),
                        expected_training_layout=training_layout_identity(config),
                        expected_trainable_names=model.trainable_names,
                    )
                    model.load_trainable_state_dict(payload.trainable_state)

                batch = DescriptionCollator(processor, training=False)([sample])
                torch.cuda.reset_peak_memory_stats(device)
                generated = model.generate_text(
                    _move_batch(batch, device),
                    processor=processor.processor,
                    max_new_tokens=config.generation.max_new_tokens,
                    do_sample=config.generation.do_sample,
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                )
                torch.cuda.synchronize(device)
                self.assertEqual(len(generated), 1)
                self.assertTrue(generated[0].strip())
                results[model_role] = {
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "generated_character_count": len(generated[0]),
                    "input_token_count": int(batch["input_token_counts"][0]),
                    "image_count": int(batch["image_counts"][0]),
                }
                del batch, generated, model, processor
                gc.collect()
                torch.cuda.empty_cache()

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "gpu": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "record_id": sample.record_id,
                    "results": results,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
