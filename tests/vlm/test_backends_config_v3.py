"""VLM backend registry/factory 与 rs_vlm.config.v3 严格合同。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from oa_groundrag.vlm.backends import (
    BackendRegistryError,
    VLMBackendRegistry,
    VLMBackendSpec,
    build_model_adapter,
    build_processor_adapter,
    resolve_vlm_backend,
)
from oa_groundrag.vlm.config import (
    CONFIG_SCHEMA_VERSION_V3,
    load_config,
)
from oa_groundrag.vlm.errors import ConfigError, ReasonCode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs/vlm/rs_general"


def _v3_row(*, backend: str = "qwen3_5") -> dict[str, object]:
    row = yaml.safe_load(
        (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
    )
    row["schema_version"] = CONFIG_SCHEMA_VERSION_V3
    row["model"].update(
        {
            "backend": backend,
            "hub_repo_id": (
                "Qwen/Qwen3.5-4B"
                if backend == "qwen3_5"
                else "Qwen/Qwen3-VL-2B-Instruct"
            ),
            "hub_revision": "1" * 40,
            "asset_ledger_path": "model_asset_ledger.json",
        }
    )
    return row


def _write_yaml(path: Path, row: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(row, sort_keys=False),
        encoding="utf-8",
    )


class ConfigV3Tests(unittest.TestCase):
    def test_v2_semantic_identity_is_byte_for_byte_stable(self) -> None:
        expected = {
            "bounded_smoke.yaml": (
                "829b5548dfe7b83b002f71241029a5bb73d14c64dd018c6452ef2ebe07415fa3"
            ),
            "rs_generaldesc_lora_qwen3vl_2b.yaml": (
                "a4b30e9a4f654cd85ee3d9e19fe49b1b4b1e53c88e7b2c38aeffffc9834fdd85"
            ),
            "rs_generaldesc_prompt_only_qwen3vl_2b.yaml": (
                "37c772d70b2fed6bb349d0404cbe5d8931e52a18c64253948eb6f8f62a992a7e"
            ),
        }
        for name, semantic_sha256 in expected.items():
            with self.subTest(name=name):
                config = load_config(CONFIG_ROOT / name)
                self.assertEqual(config.semantic_sha256, semantic_sha256)
                self.assertEqual(config.model.backend, "qwen3_vl")
                self.assertNotIn("backend", config.semantic_dict()["model"])

    def test_v3_requires_and_hashes_explicit_backend_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen35.yaml"
            row = _v3_row()
            _write_yaml(path, row)
            config = load_config(path)
            self.assertEqual(config.schema_version, CONFIG_SCHEMA_VERSION_V3)
            self.assertEqual(config.model.backend, "qwen3_5")
            self.assertEqual(config.model.hub_repo_id, "Qwen/Qwen3.5-4B")
            self.assertEqual(config.model.hub_revision, "1" * 40)
            self.assertEqual(
                config.model.asset_ledger_path,
                path.parent / "model_asset_ledger.json",
            )
            self.assertEqual(
                config.semantic_dict()["model"]["backend"],
                "qwen3_5",
            )

            changed = _v3_row()
            changed["model"]["hub_revision"] = "2" * 40
            changed_path = path.parent / "changed.yaml"
            _write_yaml(changed_path, changed)
            self.assertNotEqual(
                config.semantic_sha256,
                load_config(changed_path).semantic_sha256,
            )

    def test_v3_unknown_missing_extra_and_bad_revision_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases: list[tuple[str, dict[str, object], ReasonCode]] = []

            unknown = _v3_row()
            unknown["model"]["backend"] = "unknown"
            cases.append(("unknown", unknown, ReasonCode.BACKEND_UNKNOWN))

            missing = _v3_row()
            del missing["model"]["backend"]
            cases.append(("missing", missing, ReasonCode.TYPE_MISMATCH))

            extra = _v3_row()
            extra["model"]["legacy_family"] = "qwen"
            cases.append(("extra", extra, ReasonCode.UNKNOWN_FIELD))

            bad_revision = _v3_row()
            bad_revision["model"]["hub_revision"] = "main"
            cases.append(
                ("bad-revision", bad_revision, ReasonCode.TYPE_MISMATCH)
            )

            wrong_repo = _v3_row()
            wrong_repo["model"]["hub_repo_id"] = "Qwen/Qwen3.5-4B-Base"
            cases.append(
                (
                    "wrong-repo",
                    wrong_repo,
                    ReasonCode.MODEL_IDENTITY_MISMATCH,
                )
            )

            for name, row, code in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.yaml"
                    _write_yaml(path, row)
                    with self.assertRaises(ConfigError) as caught:
                        load_config(path)
                    self.assertEqual(caught.exception.code, code)

    def test_qwen35_rejects_four_by_four_training_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            row = _v3_row()
            row["training"]["batch_size"] = 4
            row["training"]["gradient_accumulation_steps"] = 4
            path = Path(temporary) / "qwen35-4x4.yaml"
            _write_yaml(path, row)
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
        self.assertEqual(caught.exception.code, ReasonCode.TYPE_MISMATCH)


class BackendRegistryFactoryTests(unittest.TestCase):
    @staticmethod
    def _spec(name: str = "fixture") -> VLMBackendSpec:
        return VLMBackendSpec(
            name=name,
            model_types=("fixture_model",),
            architectures=("FixtureForConditionalGeneration",),
            processor_classes=("FixtureProcessor",),
            supports_thinking=False,
            processor_builder=lambda config: (
                "processor",
                config.model.backend,
            ),
            model_builder=lambda config, device, gradient_checkpointing: (
                "model",
                config.model.backend,
                device,
                gradient_checkpointing,
            ),
        )

    def test_unknown_and_duplicate_registry_entries_fail_closed(self) -> None:
        registry = VLMBackendRegistry()
        with self.assertRaises(BackendRegistryError) as unknown:
            registry.resolve("missing")
        self.assertEqual(unknown.exception.code, ReasonCode.BACKEND_UNKNOWN)

        spec = self._spec()
        registry.register(spec)
        with self.assertRaises(BackendRegistryError) as duplicate:
            registry.register(spec)
        self.assertEqual(
            duplicate.exception.code,
            ReasonCode.BACKEND_DUPLICATE,
        )

    def test_factory_dispatches_only_the_configured_backend(self) -> None:
        registry = VLMBackendRegistry()
        registry.register(self._spec())
        base = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        config = replace(
            base,
            model=replace(base.model, backend="fixture"),
        )
        self.assertEqual(
            resolve_vlm_backend(config, registry=registry).name,
            "fixture",
        )
        self.assertEqual(
            build_processor_adapter(config, registry=registry),
            ("processor", "fixture"),
        )
        self.assertEqual(
            build_model_adapter(
                config,
                device="cpu",
                gradient_checkpointing=False,
                registry=registry,
            ),
            ("model", "fixture", "cpu", False),
        )

    def test_builtin_specs_are_separate_and_thinking_is_explicit(self) -> None:
        qwen3_vl = resolve_vlm_backend("qwen3_vl")
        qwen3_5 = resolve_vlm_backend("qwen3_5")
        self.assertFalse(qwen3_vl.supports_thinking)
        self.assertTrue(qwen3_5.supports_thinking)
        self.assertEqual(qwen3_vl.model_types, ("qwen3_vl",))
        self.assertEqual(qwen3_5.model_types, ("qwen3_5",))
        self.assertIsNot(qwen3_vl.processor_builder, qwen3_5.processor_builder)
        self.assertIsNot(qwen3_vl.model_builder, qwen3_5.model_builder)

    def test_qwen35_model_factory_binds_verified_assets(self) -> None:
        base = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        model = replace(
            base.model,
            backend="qwen3_5",
            hub_repo_id="Qwen/Qwen3.5-4B",
            hub_revision="1" * 40,
            asset_ledger_path=Path("/tmp/qwen35-ledger.json"),
        )
        config = replace(
            base,
            schema_version=CONFIG_SCHEMA_VERSION_V3,
            model=model,
        )
        verified = object()
        with (
            patch(
                "oa_groundrag.vlm.backends.qwen3_5.verify_model_asset_ledger",
                return_value=verified,
            ) as verify,
            patch(
                "oa_groundrag.vlm.backends.qwen3_5.model."
                "Qwen35ModelAdapter.load",
                return_value="qwen35-model",
            ) as load,
        ):
            result = build_model_adapter(
                config,
                device="cuda:0",
                gradient_checkpointing=True,
            )
        self.assertEqual(result, "qwen35-model")
        verify.assert_called_once_with(
            Path("/tmp/qwen35-ledger.json"),
            expected_backend="qwen3_5",
            expected_repo_id="Qwen/Qwen3.5-4B",
            expected_revision="1" * 40,
            expected_model_root=model.path,
        )
        load.assert_called_once_with(
            model,
            config.adaptation,
            assets=verified,
            device="cuda:0",
            gradient_checkpointing=True,
        )

    def test_common_backend_modules_do_not_import_concrete_backends(self) -> None:
        root = REPOSITORY_ROOT / "oa_groundrag/vlm/backends"
        for name in ("contracts.py", "registry.py", "factory.py"):
            path = root / name
            imports: list[str] = []
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(
                any("qwen3_vl" in value or "qwen3_5" in value for value in imports),
                msg=str(path),
            )


if __name__ == "__main__":
    unittest.main()
