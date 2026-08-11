from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from oa_groundrag.phase4.evidence import EvidenceBuilder
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.errors import EvidenceError
from oa_groundrag.phase4.messages import build_mask_grounded_region_messages
from oa_groundrag.phase4.processing import EncodedSample, single_inference_tensor_batch
from oa_groundrag.text_rag.contracts import TextRagTask
from oa_groundrag.text_rag.pass2 import build_pass2_messages
from oa_groundrag.text_rag.runtime import RuntimeTextRetriever
from oa_groundrag.unified.config import build_unified_runtime, load_unified_config
from oa_groundrag.unified.contracts import UnifiedInferenceError
from scripts.unified.run_oa_groundrag import entrypoint


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_CONFIG = REPO_ROOT / "configs/unified/inference_v1.yaml"


class RuntimeEvidenceContractTest(unittest.TestCase):
    @staticmethod
    def _assets(root: Path) -> tuple[Path, Path]:
        image = root / "image.png"
        mask = root / "mask.png"
        Image.new("RGB", (20, 12), color=(40, 80, 120)).save(image)
        values = np.zeros((12, 20), dtype=np.uint8)
        values[3:8, 5:14] = 255
        Image.fromarray(values).save(mask)
        return image, mask

    def test_nonempty_runtime_region_is_full_mask_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, mask = self._assets(root)
            built = EvidenceBuilder(max_auxiliary_views=2).build_runtime_region(
                optical_image=image,
                mask=mask,
                output_root=root / "grounded",
                sample_id="sample",
                source="user",
                split="runtime",
                mask_source="user_mask",
                source_identity={"request_id": "r"},
            )
            self.assertEqual(
                built.record["formal_model_input_roles"],
                ["optical_full", "binary_mask", "context_crop"],
            )
            self.assertTrue((built.root / "context_crop.png").is_file())
            original = build_mask_grounded_region_messages(
                built.record,
                asset_root=built.root,
            )
            instructed = build_mask_grounded_region_messages(
                built.record,
                asset_root=built.root,
                instruction="Describe this masked region.",
            )
            self.assertNotIn("用户任务：", original[0]["content"][-1]["text"])
            self.assertIn("用户任务：Describe this masked region.", instructed[0]["content"][-1]["text"])

    def test_no_target_has_no_fabricated_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, _mask = self._assets(root)
            built = EvidenceBuilder(max_auxiliary_views=2).build_runtime_region(
                optical_image=image,
                mask=np.zeros((12, 20), dtype=bool),
                output_root=root / "grounded",
                sample_id="sample",
                source="oa",
                split="val",
                mask_source="oa_auxseg_global",
                source_identity={"request_id": "r"},
            )
            self.assertEqual(built.record["target_status"], "no_target")
            self.assertEqual(
                built.record["formal_model_input_roles"],
                ["optical_full", "binary_mask"],
            )
            self.assertFalse((built.root / "context_crop.png").exists())

    def test_empty_or_linked_user_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, mask = self._assets(root)
            Image.fromarray(np.zeros((12, 20), dtype=np.uint8)).save(mask)
            builder = EvidenceBuilder(max_auxiliary_views=2)
            with self.assertRaises(EvidenceError):
                builder.build_runtime_region(
                    optical_image=image,
                    mask=mask,
                    output_root=root / "empty",
                    sample_id="sample",
                    source="user",
                    split="runtime",
                    mask_source="user_mask",
                    source_identity={},
                )
            link = root / "linked-mask.png"
            link.symlink_to(mask)
            with self.assertRaises(EvidenceError):
                builder.build_runtime_region(
                    optical_image=image,
                    mask=link,
                    output_root=root / "linked",
                    sample_id="sample",
                    source="user",
                    split="runtime",
                    mask_source="user_mask",
                    source_identity={},
                )


class Pass2CompatibilityTest(unittest.TestCase):
    def test_single_visual_batch_keeps_qwen_patch_and_grid_dimensions(self) -> None:
        encoded = EncodedSample(
            tensors={
                "input_ids": torch.arange(5),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "pixel_values": torch.zeros((12, 1176)),
                "image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 5]]),
            },
            labels=None,
            input_token_count=5,
            image_count=2,
        )
        batch = single_inference_tensor_batch(encoded)
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(batch["pixel_values"].shape), (12, 1176))
        self.assertEqual(tuple(batch["image_grid_thw"].shape), (2, 3))

    def test_candidate_default_prompt_is_unchanged_by_explicit_task(self) -> None:
        kwargs = {
            "question": "question",
            "target_status": "target_present",
            "program_facts": {"mask": {"area_pixels": 10}},
            "observation": {"short_summary": "observation"},
            "packet": {"items": []},
        }
        self.assertEqual(
            build_pass2_messages(**kwargs),
            build_pass2_messages(
                **kwargs,
                task=TextRagTask.CANDIDATE_INTERPRETATION,
            ),
        )
        professional = build_pass2_messages(
            **kwargs,
            task=TextRagTask.PROFESSIONAL_QA,
        )
        self.assertIn("专业问答", professional[0]["content"][0]["text"])


class FakeEmbedder:
    texts: list[str] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.identity = {"backend": "fake"}

    def encode(self, texts: list[str]) -> np.ndarray:
        type(self).texts = list(texts)
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class FakeHybridRetriever:
    def __init__(self, **kwargs: object) -> None:
        pass

    def retrieve(self, query: object, vector: object) -> list[dict[str, object]]:
        return []

    def close(self) -> None:
        pass


class RuntimeRetrieverContractTest(unittest.TestCase):
    def test_no_target_query_is_explicit_and_does_not_use_dev_workflow(self) -> None:
        config = SimpleNamespace(
            bank_root=Path("/unused/bank"),
            dense=SimpleNamespace(
                model_root=Path("/unused/model"),
                repo_id="repo",
                revision="0" * 40,
                device="cpu",
                batch_size=2,
                max_tokens=32,
            ),
            retrieval=SimpleNamespace(),
        )
        retriever = RuntimeTextRetriever(config)
        observation: dict[str, object] = {}
        facts = {"mask": {"area_pixels": 0}}
        packet = {
            "packet_id": "packet",
            "record_id": "record",
            "bank_id": "bank",
            "items": [],
        }
        with (
            patch("oa_groundrag.text_rag.runtime.validate_bank", return_value={
                "bank_id": "bank",
                "manifest_sha256": "a" * 64,
                "ledger_sha256": "b" * 64,
            }),
            patch("oa_groundrag.text_rag.runtime.BGEM3DenseEmbedder", FakeEmbedder),
            patch("oa_groundrag.text_rag.runtime.HybridRetriever", FakeHybridRetriever),
            patch("oa_groundrag.text_rag.runtime.load_runtime_bank_payload", return_value=(
                {"bank_id": "bank"}, [], np.empty((0, 2), dtype=np.float32), [],
            )),
            patch("oa_groundrag.text_rag.runtime.build_balanced_packet", return_value=packet),
        ):
            result = retriever.retrieve(
                task=TextRagTask.CANDIDATE_INTERPRETATION,
                record_id="record",
                question="interpret",
                observation=observation,
                program_facts=facts,
                target_status="no_target",
                candidate_count=0,
                available_modalities=("optical",),
            )
        self.assertEqual(result.packet, packet)
        self.assertTrue(all("no_target=true" in text for text in FakeEmbedder.texts))
        self.assertTrue(all("mask_area_pixels=0" in text for text in FakeEmbedder.texts))
        self.assertTrue(all("candidate_count=0" in text for text in FakeEmbedder.texts))
        self.assertEqual(observation, {})
        self.assertEqual(facts, {"mask": {"area_pixels": 0}})


class ConfigAndCliContractTest(unittest.TestCase):
    def test_existing_and_unified_cli_help_remain_independent(self) -> None:
        scripts = (
            "scripts/phase2_oa_auxseg/run_oa_auxseg.py",
            "scripts/phase4_rs_vlm/run_rs_vlm.py",
            "scripts/phase4_rs_vlm/run_mask_grounded_adapter.py",
            "scripts/stage6_text_rag/run_text_rag.py",
            "scripts/stage6_text_rag/run_gate_d_dev.py",
            "scripts/unified/run_oa_groundrag.py",
        )
        for relative in scripts:
            process = subprocess.run(
                [sys.executable, str(REPO_ROOT / relative), "--help"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, msg=f"{relative}: {process.stderr}")

    def test_live_unified_config_resolves_from_its_own_directory(self) -> None:
        config = load_unified_config(UNIFIED_CONFIG)
        self.assertEqual(config.repository_root, REPO_ROOT)
        self.assertEqual(config.runtime.device, "cuda")

    def test_unknown_config_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "schema_version: oa_groundrag.unified_inference.config.v1\n"
                "repository_root: /tmp\nspatial: {}\nstage6: {}\nruntime: {}\n"
                "unknown: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(UnifiedInferenceError):
                load_unified_config(path)

    def test_dry_config_does_not_require_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "schema_version: oa_groundrag.unified_inference.config.v1\n"
                f"repository_root: {REPO_ROOT}\n"
                "spatial:\n"
                f"  config: {REPO_ROOT / 'configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json'}\n"
                "  checkpoint: /tmp/nonexistent-unified-checkpoint.pt\n"
                f"  checkpoint_sha256: \"{'0' * 64}\"\n"
                "stage6:\n"
                f"  config: {REPO_ROOT / 'configs/stage6_text_rag/dev_v1.yaml'}\n"
                "runtime:\n"
                "  device: cuda\n"
                "  release_spatial_before_shared: true\n"
                "  release_shared_before_rag: true\n"
                "  release_retriever_after_query: true\n"
                "  release_unused_between_requests: true\n"
                "  reject_test_paths: true\n",
                encoding="utf-8",
            )
            config = load_unified_config(path)
            self.assertFalse(config.spatial.checkpoint_path.exists())

    def test_runtime_factory_constructs_only_unloaded_adapters(self) -> None:
        runtime = build_unified_runtime(load_unified_config(UNIFIED_CONFIG))
        self.assertIsNone(runtime.spatial._session)
        self.assertIsNone(runtime.shared_mllm._bundle)
        self.assertIsNone(runtime.text_rag._retriever)

    def test_artifact_relative_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with AtomicArtifactDirectory(Path(directory) / "result") as writer:
                with self.assertRaises(Exception):
                    writer.path("../escape.json")

    def test_dry_run_constructs_no_provider_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps({
                "schema_version": "oa_groundrag.unified_request.v1",
                "request_id": "dry-run",
                "task": "KNOWLEDGE_QA",
                "instruction": "Why is LOS not full 3-D displacement?",
                "images": [],
                "user_mask": None,
                "spatial_input": None,
                "region_source": "NONE",
                "candidate_region_id": None,
                "auxiliary_views": [],
                "include_audit": True,
            }), encoding="utf-8")
            output = io.StringIO()
            with (
                patch(
                    "scripts.unified.run_oa_groundrag.build_unified_runtime",
                    side_effect=AssertionError("provider must not be constructed"),
                ),
                patch("sys.stdout", output),
            ):
                code = entrypoint([
                    "--config", str(UNIFIED_CONFIG),
                    "--request", str(request_path),
                    "--dry-run",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["providers_constructed"])
            self.assertFalse(payload["output_written"])


if __name__ == "__main__":
    unittest.main()
