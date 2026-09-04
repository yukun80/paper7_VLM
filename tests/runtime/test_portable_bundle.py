"""2B Unified Runtime 的可迁移路径与已发布资产身份回归。"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from safetensors.torch import load_file as load_safetensors

from oa_groundrag.artifacts.io import (
    PortablePathError,
    bundle_relative_reference,
    resolve_bundle_reference,
    resolve_config_reference,
    validate_bundle_root,
)
from oa_groundrag.retrieval.bank import validate_runtime_bank
from oa_groundrag.retrieval.runtime_config import load_text_rag_runtime_config
from oa_groundrag.runtime.config import load_unified_config, validate_provider_paths
from oa_groundrag.vlm.grounded_runtime import (
    load_grounded_runtime_config,
    resolve_grounded_runtime_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REGION_WORKFLOW = (
    REPO_ROOT
    / "outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1"
)
REGION_ADAPTER = (
    REGION_WORKFLOW
    / "training/checkpoints/step-00000900/adapter/adapter_model.safetensors"
)
TEXT_BANK = REPO_ROOT / "outputs/stage6_text_rag/text_evidence_bank_v1"


class PortablePathTest(unittest.TestCase):
    def test_rejects_absolute_escape_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "renamed-delivery-root"
            repository = root / "paper7_VLM"
            benchmark = root / "benchmark"
            repository.mkdir(parents=True)
            benchmark.mkdir()
            asset = repository / "asset.bin"
            asset.write_bytes(b"portable")
            config = repository / "configs/runtime.yaml"
            config.parent.mkdir()
            config.write_text("fixture", encoding="utf-8")

            self.assertEqual(validate_bundle_root(root), root.resolve())
            self.assertEqual(
                resolve_bundle_reference(
                    root,
                    "paper7_VLM/asset.bin",
                    expected="file",
                ),
                asset.resolve(),
            )
            self.assertEqual(
                bundle_relative_reference(root, asset),
                "paper7_VLM/asset.bin",
            )
            self.assertEqual(
                resolve_config_reference(config, "../asset.bin"),
                asset.resolve(),
            )
            for reference in (
                str(asset),
                r"C:\absolute\asset.bin",
                "../paper7_VLM/asset.bin",
                "paper7_VLM/../../escape.bin",
            ):
                with self.subTest(reference=reference):
                    with self.assertRaises(PortablePathError):
                        resolve_bundle_reference(root, reference)
            outside = Path(directory) / "outside"
            outside.mkdir()
            (repository / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(PortablePathError):
                resolve_bundle_reference(root, "paper7_VLM/linked/value.bin")
            with self.assertRaises(PortablePathError):
                resolve_config_reference(config, str(asset))
            with self.assertRaises(PortablePathError):
                resolve_config_reference(config, r"C:\absolute\asset.bin")


@unittest.skipUnless(
    REGION_ADAPTER.is_file() and (TEXT_BANK / "manifest.json").is_file(),
    "requires published 2B Adapter and Text Bank",
)
class PublishedBundleRelocationTest(unittest.TestCase):
    def test_runtime_metadata_relocates_under_renamed_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_root = Path(directory) / "another-server-delivery"
            repository = bundle_root / "paper7_VLM"
            benchmark = bundle_root / "benchmark"
            benchmark.mkdir(parents=True)
            for relative in (
                "configs/runtime/inference_v2.yaml",
                "configs/vlm/grounded/runtime_v1.yaml",
                "configs/retrieval/runtime_v1.yaml",
                "configs/retrieval/sources_v1.yaml",
                "configs/segmentation/full_proposed_dropout_b16_nockpt_e100.json",
            ):
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / relative, target)
            for relative in (
                "models_zoo/Qwen3-VL-2B-Instruct",
                "models_zoo/text_embeddings/"
                "bge-m3-5617a9f61b028005a4858fdac845db406aefb181",
                "models_zoo/ConvNeXt",
                "docs/RAG_knowledge",
            ):
                (repository / relative).mkdir(parents=True, exist_ok=True)
            spatial_checkpoint = (
                repository
                / "outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/"
                "checkpoint_best.pt"
            )
            spatial_checkpoint.parent.mkdir(parents=True)
            spatial_checkpoint.touch()
            relocated_workflow = repository / REGION_WORKFLOW.relative_to(REPO_ROOT)
            for relative in (
                "workflow_state.json",
                "training/best_checkpoint.json",
                "training/checkpoints/step-00000900/manifest.json",
                "training/checkpoints/step-00000900/adapter/adapter_model.safetensors",
            ):
                target = relocated_workflow / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REGION_WORKFLOW / relative, target)
            shutil.copytree(
                TEXT_BANK,
                repository / TEXT_BANK.relative_to(REPO_ROOT),
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(Path(directory))
                unified = load_unified_config(
                    repository / "configs/runtime/inference_v2.yaml"
                )
                validate_provider_paths(unified)
                grounded = load_grounded_runtime_config(
                    unified.semantic_core.config_path
                )
                reference = resolve_grounded_runtime_checkpoint(
                    grounded,
                    unified.semantic_core,
                )
                tensors = load_safetensors(
                    str(
                        reference.checkpoint
                        / "adapter/adapter_model.safetensors"
                    ),
                    device="cpu",
                )
                retrieval = load_text_rag_runtime_config(
                    unified.retrieval.config_path
                )
                validated_bank = validate_runtime_bank(
                    retrieval.bank_root,
                    config=retrieval,
                    verify_sources=False,
                )
            finally:
                os.chdir(original_cwd)
            self.assertEqual(unified.repository_root, repository.resolve())
            self.assertEqual(len(tensors), 224)
            self.assertEqual(validated_bank["bank_id"], retrieval.bank.bank_id)


if __name__ == "__main__":
    unittest.main()
