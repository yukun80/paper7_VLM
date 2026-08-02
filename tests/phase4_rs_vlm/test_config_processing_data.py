from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from PIL import Image

import oa_groundrag.phase4 as phase4_public
from oa_groundrag.phase4.cli import entrypoint
from oa_groundrag.phase3.builder import build_benchmark
from oa_groundrag.phase3.common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.contracts import (
    BENCHMARK_SCOPE,
    BUILD_CONFIG_VERSION,
    CANONICAL_SCHEMA_VERSION,
    HASH_SCHEMA_VERSION,
    MANIFEST_VERSION,
    STATISTICS_SCHEMA_VERSION,
    VALIDATION_VERSION,
)
from oa_groundrag.phase3.config import load_build_config
from oa_groundrag.phase3.dataset import RSGeneralDescDataset
from oa_groundrag.phase3.errors import RSGeneralDescError
from oa_groundrag.phase3.hash_ledger import HashLedgerVerifier
from oa_groundrag.phase4.config import (
    CONFIG_SCHEMA_VERSION,
    apply_runtime_overrides,
    load_config,
)
from oa_groundrag.phase4.contracts import MaskMode
from oa_groundrag.phase4.data import (
    DescriptionSample,
    ExternalDescriptionDataset,
    inventory_from_auxseg_inference,
)
from oa_groundrag.phase4.errors import (
    ConfigError,
    ContractError,
    ModelError,
    PreflightError,
    ProcessingError,
    ReasonCode,
)
from oa_groundrag.phase4.preflight import (
    inspect_benchmark_identity,
    run_preflight,
)
from oa_groundrag.phase4.processing import (
    DescriptionCollator,
    EncodedSample,
    assistant_only_labels,
)
from oa_groundrag.phase4.model import (
    EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS,
    ModelIdentity,
    Qwen3VLModelAdapter,
    assistant_sample_mean_causal_loss,
    local_model_identity,
)
from oa_groundrag.phase4.reference import MAIN_REFERENCE
from tests.phase3_rs_generaldesc.fixture_helpers import (
    make_all_sources,
    write_build_config,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs/phase4_rs_vlm"


def native_preflight_fixture(root: Path) -> dict[str, str]:
    build_config = {"schema_version": BUILD_CONFIG_VERSION}
    atomic_write_json(root / "metadata/build_config.json", build_config)
    semantic_sha = sha256_text(canonical_json(build_config))
    atomic_write_json(
        root / "schemas/canonical.schema.json",
        {
            "$id": CANONICAL_SCHEMA_VERSION,
            "properties": {
                "schema_version": {"const": CANONICAL_SCHEMA_VERSION}
            },
        },
    )
    atomic_write_json(
        root / "metadata/statistics.json",
        {
            "schema_version": STATISTICS_SCHEMA_VERSION,
            "record_count": 10,
            "parent_count": 4,
            "role_counts": {"external_train": 8, "external_val": 2},
            "parent_role_counts": {"external_train": 3, "external_val": 1},
        },
    )
    atomic_write_jsonl(
        root / "records/part-00000.jsonl",
        ({"fixture": index} for index in range(10)),
    )
    ledger_paths = (
        "metadata/build_config.json",
        "metadata/statistics.json",
        "records/part-00000.jsonl",
        "schemas/canonical.schema.json",
    )
    files = [
        {
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in ledger_paths
    ]
    payload_sha = sha256_text(canonical_json(files))
    hashes = {
        "schema_version": HASH_SCHEMA_VERSION,
        "files": files,
        "root_sha256": payload_sha,
    }
    atomic_write_json(root / "metadata/hashes.json", hashes)
    hash_manifest_sha = sha256_file(root / "metadata/hashes.json")
    build_id = "build_" + sha256_text(
        canonical_json([semantic_sha, payload_sha])
    )
    atomic_write_json(
        root / "metadata/validation.json",
        {
            "schema_version": VALIDATION_VERSION,
            "deep": True,
            "errors": [],
            "warnings": [],
            "payload_root_sha256": payload_sha,
            "hash_manifest_sha256": hash_manifest_sha,
            "record_count": 10,
            "parent_count": 4,
            "checked_records": 10,
        },
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": MANIFEST_VERSION,
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "build_profile": "full",
            "benchmark_scope": BENCHMARK_SCOPE,
            "scope_validation_complete": True,
            "build_id": build_id,
            "semantic_config_sha256": semantic_sha,
            "payload_root_sha256": payload_sha,
            "hash_manifest_sha256": hash_manifest_sha,
            "layout": {
                "canonical_schema": "schemas/canonical.schema.json",
                "record_shards": [
                    {
                        "path": "records/part-00000.jsonl",
                        "record_count": 10,
                    }
                ],
                "role_to_record_shards": {
                    "external_train": ["records/part-00000.jsonl"],
                    "external_val": ["records/part-00000.jsonl"],
                },
                "validation": "metadata/validation.json",
                "statistics": "metadata/statistics.json",
                "hashes": "metadata/hashes.json",
                "build_config": "metadata/build_config.json",
            },
            "counts": {"records": 10, "parents": 4},
            "formal_acceptance_eligible": True,
            "formal_acceptance_blockers": [],
            "source_roots_embedded": False,
            "external_test_retained": False,
            "official_upstream_evaluation_reproducible": False,
        },
    )
    manifest_sha = sha256_file(root / "manifest.json")
    validation_sha = sha256_file(root / "metadata/validation.json")
    return {
        "build_id": build_id,
        "payload_sha256": payload_sha,
        "hash_manifest_sha256": hash_manifest_sha,
        "manifest_sha256": manifest_sha,
        "validation_sha256": validation_sha,
    }


class ConfigAndPreflightTests(unittest.TestCase):
    def test_inactive_mask_grounded_dataset_contract_is_not_public(self) -> None:
        self.assertFalse(hasattr(phase4_public, "MaskGroundedExample"))
        self.assertFalse(
            hasattr(phase4_public, "MaskGroundedDescriptionDataset")
        )
        from oa_groundrag.phase4 import data

        self.assertFalse(hasattr(data, "MaskGroundedExample"))
        self.assertFalse(hasattr(data, "MaskGroundedDescriptionDataset"))

    def test_three_configs_strictly_parse(self) -> None:
        configs = {
            path.name: load_config(path)
            for path in CONFIG_ROOT.glob("*.yaml")
        }
        self.assertEqual(
            set(configs),
            {
                "bounded_smoke.yaml",
                "rs_generaldesc_lora_qwen3vl_2b.yaml",
                "rs_generaldesc_prompt_only_qwen3vl_2b.yaml",
            },
        )
        for config in configs.values():
            self.assertEqual(config.schema_version, CONFIG_SCHEMA_VERSION)
            self.assertEqual(len(config.semantic_sha256), 64)
            self.assertFalse(hasattr(config, "evidence"))
            self.assertFalse(hasattr(config, "evaluation"))
            self.assertNotIn("evidence", config.semantic_dict())
            self.assertNotIn("evaluation", config.semantic_dict())
            self.assertTrue(config.model.local_files_only)
            self.assertTrue(config.adaptation.freeze_vision)
            self.assertTrue(config.adaptation.freeze_merger)
        self.assertEqual(
            configs[
                "rs_generaldesc_lora_qwen3vl_2b.yaml"
            ].adaptation.target_modules,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        external = configs["rs_generaldesc_lora_qwen3vl_2b.yaml"]
        self.assertEqual(external.run.name, "rs_vlm_lora_qwen3vl_2b_bs4")
        self.assertEqual(external.training.batch_size, 4)
        self.assertEqual(external.training.gradient_accumulation_steps, 4)
        self.assertEqual(
            external.training.batch_size
            * external.training.gradient_accumulation_steps,
            16,
        )
        self.assertEqual(external.training.log_interval, 10)
        self.assertEqual(external.training.validation_interval, 100)
        self.assertEqual(external.training.validation_max_parents, 128)
        self.assertEqual(external.training.num_workers, 4)
        self.assertEqual(external.training.prefetch_factor, 2)
        self.assertTrue(external.training.pin_memory)
        self.assertTrue(
            str(external.run.output_root).startswith(str(REPO / "outputs"))
        )
        base = configs["rs_generaldesc_prompt_only_qwen3vl_2b.yaml"]
        self.assertEqual(base.adaptation.strategy, "prompt_only")
        self.assertEqual(base.data.roles, ("external_val",))

    def test_training_layout_accepts_only_b1a16_or_b4a4(self) -> None:
        source = yaml.safe_load(
            (
                CONFIG_ROOT / "rs_generaldesc_lora_qwen3vl_2b.yaml"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for batch_size, accumulation in ((1, 16), (4, 4)):
                candidate = yaml.safe_load(yaml.safe_dump(source))
                candidate["training"]["batch_size"] = batch_size
                candidate["training"][
                    "gradient_accumulation_steps"
                ] = accumulation
                path = root / f"valid-{batch_size}-{accumulation}.yaml"
                path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False),
                    encoding="utf-8",
                )
                parsed = load_config(path)
                self.assertEqual(parsed.training.batch_size, batch_size)
                self.assertEqual(
                    parsed.training.gradient_accumulation_steps,
                    accumulation,
                )

            for batch_size, accumulation in ((2, 8), (4, 16), (1, 4)):
                candidate = yaml.safe_load(yaml.safe_dump(source))
                candidate["training"]["batch_size"] = batch_size
                candidate["training"][
                    "gradient_accumulation_steps"
                ] = accumulation
                path = root / f"invalid-{batch_size}-{accumulation}.yaml"
                path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError) as caught:
                    load_config(path)
                self.assertEqual(
                    caught.exception.code,
                    ReasonCode.TYPE_MISMATCH,
                )

    def test_input_pipeline_layout_is_strict(self) -> None:
        source = yaml.safe_load(
            (
                CONFIG_ROOT / "rs_generaldesc_lora_qwen3vl_2b.yaml"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_sync = yaml.safe_load(yaml.safe_dump(source))
            valid_sync["training"].update(
                {
                    "num_workers": 0,
                    "prefetch_factor": 0,
                    "pin_memory": False,
                }
            )
            valid_path = root / "valid-sync.yaml"
            valid_path.write_text(
                yaml.safe_dump(valid_sync, sort_keys=False),
                encoding="utf-8",
            )
            parsed = load_config(valid_path)
            self.assertEqual(parsed.training.num_workers, 0)
            self.assertEqual(parsed.training.prefetch_factor, 0)
            self.assertFalse(parsed.training.pin_memory)

            invalid_rows = (
                {"num_workers": 0, "prefetch_factor": 2, "pin_memory": False},
                {"num_workers": 2, "prefetch_factor": 0, "pin_memory": True},
                {"num_workers": 2, "prefetch_factor": 2, "pin_memory": False},
                {"num_workers": 5, "prefetch_factor": 2, "pin_memory": True},
            )
            for index, values in enumerate(invalid_rows):
                candidate = yaml.safe_load(yaml.safe_dump(source))
                candidate["training"].update(values)
                path = root / f"invalid-input-{index}.yaml"
                path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError):
                    load_config(path)

    def test_unknown_bool_as_int_nonfinite_and_duplicate_are_rejected(self) -> None:
        source = yaml.safe_load(
            (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            version_one = yaml.safe_load(yaml.safe_dump(source))
            version_one["schema_version"] = "rs_vlm.config.v1"
            version_one_path = base / "v1.yaml"
            version_one_path.write_text(
                yaml.safe_dump(version_one, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(version_one_path)
            self.assertEqual(caught.exception.code, ReasonCode.INVALID_ENUM)

            legacy_section = yaml.safe_load(yaml.safe_dump(source))
            legacy_section["evidence"] = {"rag_enabled": False}
            legacy_path = base / "legacy-section.yaml"
            legacy_path.write_text(
                yaml.safe_dump(legacy_section, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(legacy_path)
            self.assertEqual(caught.exception.code, ReasonCode.UNKNOWN_FIELD)

            unknown = dict(source)
            unknown["unknown"] = True
            unknown_path = base / "unknown.yaml"
            unknown_path.write_text(
                yaml.safe_dump(unknown, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(unknown_path)
            self.assertEqual(caught.exception.code, ReasonCode.UNKNOWN_FIELD)

            bool_seed = yaml.safe_load(
                (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
            )
            bool_seed["run"]["seed"] = True
            bool_path = base / "bool.yaml"
            bool_path.write_text(
                yaml.safe_dump(bool_seed, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(bool_path)
            self.assertEqual(caught.exception.code, ReasonCode.TYPE_MISMATCH)

            nonfinite = yaml.safe_load(
                (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
            )
            nonfinite["training"]["learning_rate"] = float("nan")
            nonfinite_path = base / "nan.yaml"
            nonfinite_path.write_text(
                yaml.safe_dump(nonfinite, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(nonfinite_path)
            self.assertEqual(
                caught.exception.code,
                ReasonCode.NONFINITE_NUMBER,
            )

            duplicate_path = base / "duplicate.yaml"
            duplicate_path.write_text(
                "schema_version: one\nschema_version: two\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(duplicate_path)

            linked_path = base / "linked.yaml"
            linked_path.symlink_to(unknown_path)
            with self.assertRaises(ConfigError) as caught:
                load_config(linked_path)
            self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_LINK)

    def test_all_active_configs_pass_native_preflight(self) -> None:
        external = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        identity = inspect_benchmark_identity(external)
        self.assertEqual(identity.build_id, external.data.expected_build_id)
        self.assertEqual(
            identity.hash_manifest_sha256,
            external.data.expected_hash_manifest_sha256,
        )
        self.assertFalse(identity.source_roots_embedded)
        self.assertTrue(identity.formal_acceptance_eligible)
        self.assertEqual(identity.formal_acceptance_blockers, ())
        for path in sorted(CONFIG_ROOT.glob("*.yaml")):
            with self.subTest(config=path.name):
                preflight = run_preflight(
                    load_config(path),
                    require_new_output=False,
                )
                self.assertEqual(preflight["status"], "ok")
                self.assertTrue(
                    preflight["benchmark"]["formal_acceptance_eligible"]
                )

    def test_native_preflight_rejects_missing_tamper_and_identity(self) -> None:
        external = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "native"
            native_identity = native_preflight_fixture(metadata_root)
            local = replace(
                external,
                data=replace(
                    external.data,
                    benchmark_root=metadata_root,
                    expected_manifest_sha256=native_identity[
                        "manifest_sha256"
                    ],
                    expected_validation_sha256=native_identity[
                        "validation_sha256"
                    ],
                    expected_build_id=native_identity["build_id"],
                    expected_payload_sha256=native_identity["payload_sha256"],
                    expected_hash_manifest_sha256=(
                        native_identity["hash_manifest_sha256"]
                    ),
                ),
            )
            missing = replace(
                local,
                data=replace(
                    local.data,
                    benchmark_root=root / "missing",
                ),
            )
            with self.assertRaises(PreflightError) as caught:
                inspect_benchmark_identity(missing)
            self.assertEqual(
                caught.exception.code,
                ReasonCode.ASSET_MISSING,
            )

            bad_hash_identity = replace(
                local,
                data=replace(
                    local.data,
                    expected_hash_manifest_sha256="0" * 64,
                ),
            )
            with self.assertRaises(PreflightError) as caught:
                inspect_benchmark_identity(bad_hash_identity)
            self.assertEqual(
                caught.exception.code,
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            )

            mutations = {
                "not-eligible": (
                    "manifest.json",
                    lambda value: value.update(
                        {
                            "formal_acceptance_eligible": False,
                            "formal_acceptance_blockers": ["fixture_blocker"],
                        }
                    ),
                ),
                "validation-error": (
                    "metadata/validation.json",
                    lambda value: value.update({"errors": ["fixture"]}),
                ),
                "hash-file": (
                    "metadata/hashes.json",
                    lambda value: value.update({"files": [{"tampered": True}]}),
                ),
                "statistics-count": (
                    "metadata/statistics.json",
                    lambda value: value.update(
                        {"record_count": value["record_count"] + 1}
                    ),
                ),
            }
            for name, (relative, mutation) in mutations.items():
                with self.subTest(name=name):
                    path = metadata_root / relative
                    original = path.read_bytes()
                    candidate = read_json(path)
                    mutation(candidate)
                    atomic_write_json(path, candidate)
                    with self.assertRaises(PreflightError) as caught:
                        inspect_benchmark_identity(local)
                    self.assertEqual(
                        caught.exception.code,
                        ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                    )
                    path.write_bytes(original)

    def test_native_identity_is_semantic_and_training_identity_is_native(self) -> None:
        expected_semantic_hashes = {
            "bounded_smoke.yaml": (
                "829b5548dfe7b83b002f71241029a5bb73d14c64dd018c6452ef2ebe07415fa3"
            ),
            "rs_generaldesc_lora_qwen3vl_2b.yaml": (
                "48bf9949f4f15eeab51149868a1a373c95444003ad286d04ada0ec1e072c1d72"
            ),
            "rs_generaldesc_prompt_only_qwen3vl_2b.yaml": (
                "37c772d70b2fed6bb349d0404cbe5d8931e52a18c64253948eb6f8f62a992a7e"
            ),
        }
        for name, expected in expected_semantic_hashes.items():
            with self.subTest(config=name):
                self.assertEqual(
                    load_config(CONFIG_ROOT / name).semantic_sha256,
                    expected,
                )
        base = load_config(
            CONFIG_ROOT / "rs_generaldesc_prompt_only_qwen3vl_2b.yaml"
        )
        self.assertEqual(len(base.semantic_sha256), 64)
        config = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        changed = replace(
            config,
            data=replace(
                config.data,
                expected_hash_manifest_sha256="f" * 64,
            ),
        )
        self.assertNotEqual(config.semantic_dict(), changed.semantic_dict())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "native"
            native_identity = native_preflight_fixture(root)
            native_config = replace(
                config,
                data=replace(
                    config.data,
                    benchmark_root=root,
                    expected_manifest_sha256=native_identity[
                        "manifest_sha256"
                    ],
                    expected_validation_sha256=native_identity[
                        "validation_sha256"
                    ],
                    expected_build_id=native_identity["build_id"],
                    expected_payload_sha256=native_identity["payload_sha256"],
                    expected_hash_manifest_sha256=(
                        native_identity["hash_manifest_sha256"]
                    ),
                ),
            )
            identity = inspect_benchmark_identity(native_config)
        training_identity = identity.training_identity_dict()
        self.assertEqual(
            set(training_identity),
            {
                "root",
                "manifest_schema",
                "canonical_schema",
                "build_id",
                "semantic_config_sha256",
                "payload_sha256",
                "hash_manifest_sha256",
                "benchmark_scope",
                "source_roots_embedded",
                "deep_validation_saved",
                "formal_acceptance_eligible",
                "formal_acceptance_blockers",
                "record_count",
                "parent_count",
            },
        )
        self.assertEqual(
            training_identity["hash_manifest_sha256"],
            native_config.data.expected_hash_manifest_sha256,
        )

    def test_native_identity_and_mode_are_strict(self) -> None:
        external_source = yaml.safe_load(
            (CONFIG_ROOT / "bounded_smoke.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = []
            missing_manifest = json.loads(json.dumps(external_source))
            del missing_manifest["data"]["expected_manifest_sha256"]
            cases.append(("missing-manifest", missing_manifest))
            invalid_validation = json.loads(json.dumps(external_source))
            invalid_validation["data"]["expected_validation_sha256"] = "bad"
            cases.append(("invalid-validation", invalid_validation))
            missing_hash = json.loads(json.dumps(external_source))
            del missing_hash["data"]["expected_hash_manifest_sha256"]
            cases.append(("missing-hash", missing_hash))
            invalid_hash = json.loads(json.dumps(external_source))
            invalid_hash["data"]["expected_hash_manifest_sha256"] = "bad"
            cases.append(("invalid-hash", invalid_hash))
            retired_mode = json.loads(json.dumps(external_source))
            retired_mode["run"]["mode"] = "unsupported_mode"
            retired_mode["run"]["mask_mode"] = "gt_mask"
            cases.append(("retired-mode", retired_mode))
            inert_path = json.loads(json.dumps(external_source))
            inert_path["data"]["retired_inference_root"] = "/tmp/retired"
            cases.append(("retired-path", inert_path))
            retired_flag = json.loads(json.dumps(external_source))
            retired_flag["run"]["retired_test_flag"] = True
            cases.append(("retired-test-flag", retired_flag))
            for name, value in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.yaml"
                    path.write_text(
                        yaml.safe_dump(value, sort_keys=False),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ConfigError):
                        load_config(path)

    def test_resume_semantic_hash_excludes_output_pointer_only(self) -> None:
        config = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        changed_output = replace(
            config,
            run=replace(
                config.run,
                output_root=Path("/tmp/another-output"),
                resume_checkpoint=Path("/tmp/explicit-checkpoint"),
            ),
        )
        self.assertEqual(
            config.semantic_dict(),
            changed_output.semantic_dict(),
        )
        overridden = apply_runtime_overrides(
            config,
            output_root=Path("/tmp/override-output"),
        )
        self.assertEqual(config.semantic_sha256, overridden.semantic_sha256)
        changed_log = apply_runtime_overrides(config, log_interval=7)
        self.assertNotEqual(
            config.semantic_sha256,
            changed_log.semantic_sha256,
        )

    def test_train_cli_output_override_and_cuda_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "oa_groundrag.phase4.cli.torch.cuda.is_available",
            return_value=False,
        ):
            output = Path(temporary) / "repo-local-run"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = entrypoint(
                    [
                        "train",
                        "--config",
                        str(
                            CONFIG_ROOT
                            / "rs_generaldesc_lora_qwen3vl_2b.yaml"
                        ),
                        "--output-root",
                        str(output),
                        "--stop-after-steps",
                        "1",
                        "--log-interval",
                        "5",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("CUDA_REQUIRED", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_main_reference_and_local_model_identity_are_locked(self) -> None:
        self.assertEqual(
            MAIN_REFERENCE.revision,
            "96588727e44c78b25ba03ea03b8e12f7e64fd0da",
        )
        self.assertEqual(MAIN_REFERENCE.license, "Apache-2.0")
        identity = local_model_identity(
            REPO / "models_zoo/Qwen3-VL-2B-Instruct",
            base_parameter_count=EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS,
            model_type="qwen3_vl",
        )
        self.assertEqual(
            identity.base_parameter_count,
            EXPECTED_QWEN3VL_2B_UNIQUE_PARAMETERS,
        )
        self.assertEqual(len(identity.config_sha256), 64)
        self.assertEqual(len(identity.weights_metadata_sha256), 64)
        self.assertEqual(len(identity.processor_config_sha256), 64)


class RuntimeIdentityBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        roots = make_all_sources(cls.base / "sources")
        cls.template = build_benchmark(
            load_build_config(
                write_build_config(
                    cls.base / "build.yaml",
                    roots,
                    cls.base / "template",
                    profile="full",
                    max_assets=None,
                )
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.root = self.base / self._testMethodName
        shutil.copytree(self.template, self.root)
        base_config = load_config(CONFIG_ROOT / "bounded_smoke.yaml")
        manifest = read_json(self.root / "manifest.json")
        self.config = replace(
            base_config,
            data=replace(
                base_config.data,
                benchmark_root=self.root,
                expected_manifest_sha256=sha256_file(
                    self.root / "manifest.json"
                ),
                expected_validation_sha256=sha256_file(
                    self.root / "metadata/validation.json"
                ),
                expected_build_id=manifest["build_id"],
                expected_payload_sha256=manifest["payload_root_sha256"],
                expected_hash_manifest_sha256=manifest[
                    "hash_manifest_sha256"
                ],
            ),
            run=replace(
                base_config.run,
                output_root=self.base / f"output-{self._testMethodName}",
            ),
        )

    def test_preflight_binds_metadata_manifest_validation_and_layout(self) -> None:
        inspect_benchmark_identity(self.config)
        statistics_path = self.root / "metadata/statistics.json"
        statistics = read_json(statistics_path)
        statistics["record_count"] += 1
        atomic_write_json(statistics_path, statistics)
        with self.assertRaises(PreflightError) as caught:
            inspect_benchmark_identity(self.config)
        self.assertEqual(
            caught.exception.code,
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
        )
        self.assertEqual(
            caught.exception.details["path"],
            "metadata/statistics.json",
        )

        shutil.rmtree(self.root)
        shutil.copytree(self.template, self.root)
        manifest_path = self.root / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["layout"]["record_shards"][0]["path"] = (
            "records/not-in-ledger.jsonl"
        )
        atomic_write_json(manifest_path, manifest)
        layout_config = replace(
            self.config,
            data=replace(
                self.config.data,
                expected_manifest_sha256=sha256_file(manifest_path),
            ),
        )
        with self.assertRaises(PreflightError) as caught:
            inspect_benchmark_identity(layout_config)
        self.assertEqual(
            caught.exception.code,
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
        )
        self.assertEqual(
            caught.exception.details["path"],
            "records/not-in-ledger.jsonl",
        )

        shutil.rmtree(self.root)
        shutil.copytree(self.template, self.root)
        validation_path = self.root / "metadata/validation.json"
        validation = read_json(validation_path)
        validation["warnings"] = ["drift"]
        atomic_write_json(validation_path, validation)
        with self.assertRaises(PreflightError) as caught:
            inspect_benchmark_identity(self.config)
        self.assertEqual(
            caught.exception.details["path"],
            "metadata/validation.json",
        )

    def test_preflight_does_not_hash_record_or_asset_content(self) -> None:
        verified: list[str] = []
        original = HashLedgerVerifier.verify_file

        def capture(verifier, relative):
            verified.append(relative)
            return original(verifier, relative)

        with mock.patch.object(HashLedgerVerifier, "verify_file", new=capture):
            inspect_benchmark_identity(self.config)
        self.assertTrue(
            {
                "schemas/canonical.schema.json",
                "metadata/build_config.json",
                "metadata/statistics.json",
            }
            <= set(verified)
        )
        self.assertFalse(
            any(
                relative.startswith("records/")
                or relative.startswith("assets/")
                for relative in verified
            )
        )

    def test_dataset_rejects_shard_drift_before_jsonl_consumption(self) -> None:
        manifest = read_json(self.root / "manifest.json")
        relative = manifest["layout"]["record_shards"][0]["path"]
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("\n", " \n", 1), encoding="utf-8")
        with mock.patch(
            "oa_groundrag.phase3.dataset.read_jsonl",
            side_effect=AssertionError("drifted JSONL must not be parsed"),
        ) as parser:
            with self.assertRaises(RSGeneralDescError) as caught:
                RSGeneralDescDataset(
                    self.root,
                    load_assets=False,
                    expected_manifest_sha256=self.config.data.expected_manifest_sha256,
                )
        self.assertEqual(caught.exception.code.value, "HASH_MISMATCH")
        self.assertEqual(caught.exception.details["path"], relative)
        parser.assert_not_called()

    def test_dataset_rejects_asset_drift_before_decode(self) -> None:
        dataset = RSGeneralDescDataset(
            self.root,
            load_assets=False,
            expected_manifest_sha256=self.config.data.expected_manifest_sha256,
        )
        media = dataset.records[0]["media"][0]
        asset_path = self.root / media["path"]
        Image.new("RGB", (8, 6), (1, 2, 3)).save(asset_path)
        with mock.patch(
            "oa_groundrag.phase3.dataset.Image.open",
            side_effect=AssertionError("drifted asset must not be decoded"),
        ) as decoder:
            with self.assertRaises(RSGeneralDescError) as caught:
                dataset.load_image(media)
        self.assertEqual(caught.exception.code.value, "HASH_MISMATCH")
        self.assertEqual(caught.exception.details["path"], media["path"])
        decoder.assert_not_called()


class FakeProcessor:
    pad_token_id = 0

    def encode_training(self, messages):
        assistant = messages[-1]["content"][0]["text"]
        target = torch.tensor(
            [20 + index for index, _ in enumerate(assistant[:3], 1)],
            dtype=torch.long,
        )
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        full = torch.cat((prompt, target))
        attention = torch.ones_like(full)
        return EncodedSample(
            tensors={
                "input_ids": full,
                "attention_mask": attention,
                "mm_token_type_ids": torch.zeros_like(full),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=assistant_only_labels(
                full,
                prompt,
                attention_mask=attention,
            ),
            input_token_count=int(full.numel()),
            image_count=1,
        )

    def encode_inference(self, messages):
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        return EncodedSample(
            tensors={
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.zeros_like(ids),
                "pixel_values": torch.ones((1, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            labels=None,
            input_token_count=3,
            image_count=1,
        )


def sample(record_id: str, answer: str) -> DescriptionSample:
    return DescriptionSample(
        record_id=record_id,
        parent_id=f"parent-{record_id}",
        logical_role="external_train",
        task_family="visual_qa",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/fixture.png"},
                    {"type": "text", "text": "Question?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ),
        reference_responses=(answer,),
        mask_mode=MaskMode.EXTERNAL_GENERIC,
        evidence_ids=(),
        provenance={"fixture": True},
    )


class ProcessingTests(unittest.TestCase):
    def test_assistant_only_mask_has_no_hardcoded_token(self) -> None:
        full = torch.tensor([100, 101, 555, 777])
        prompt = torch.tensor([100, 101])
        labels = assistant_only_labels(full, prompt)
        self.assertEqual(labels.tolist(), [-100, -100, 555, 777])
        with self.assertRaises(ProcessingError) as caught:
            assistant_only_labels(full, torch.tensor([100, 999]))
        self.assertEqual(caught.exception.code, ReasonCode.LOSS_MASK_INVALID)

    def test_collator_padding_and_loss_mask(self) -> None:
        collator = DescriptionCollator(FakeProcessor(), training=True)
        batch = collator(
            [
                sample("a", "yes"),
                sample("b", "long answer"),
            ]
        )
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["pixel_values"].shape, (2, 4))
        self.assertEqual(batch["image_grid_thw"].shape, (2, 3))
        self.assertTrue(torch.all(batch["labels"][:, :3] == -100))
        self.assertTrue(torch.all(batch["labels"][:, 3:6] != -100))

    def test_batch4_collator_preserves_variable_tokens_and_images(
        self,
    ) -> None:
        class VariableProcessor:
            pad_token_id = 0

            def encode_training(self, messages):
                answer = messages[-1]["content"][0]["text"]
                index = int(answer)
                prompt = torch.tensor([1, 2], dtype=torch.long)
                target = torch.arange(3, 4 + index, dtype=torch.long)
                full = torch.cat((prompt, target))
                image_count = index + 1
                return EncodedSample(
                    tensors={
                        "input_ids": full,
                        "attention_mask": torch.ones_like(full),
                        "pixel_values": torch.full(
                            (image_count, 4),
                            float(index),
                        ),
                        "image_grid_thw": torch.ones(
                            (image_count, 3),
                            dtype=torch.long,
                        ),
                    },
                    labels=assistant_only_labels(full, prompt),
                    input_token_count=int(full.numel()),
                    image_count=image_count,
                )

            def encode_inference(self, messages):
                raise AssertionError("training fixture only")

        batch = DescriptionCollator(
            VariableProcessor(),
            training=True,
        )(
            [
                sample(f"record-{index}", str(index))
                for index in range(4)
            ]
        )
        self.assertEqual(batch["input_token_counts"], [3, 4, 5, 6])
        self.assertEqual(batch["image_counts"], [1, 2, 3, 4])
        self.assertEqual(batch["input_ids"].shape, (4, 6))
        self.assertEqual(batch["pixel_values"].shape, (10, 4))
        self.assertEqual(batch["image_grid_thw"].shape, (10, 3))
        self.assertTrue(torch.all(batch["labels"][0, 3:] == -100))

    def test_inference_collator_has_no_labels(self) -> None:
        batch = DescriptionCollator(
            FakeProcessor(),
            training=False,
        )([sample("a", "yes")])
        self.assertNotIn("labels", batch)
        self.assertEqual(batch["input_token_counts"], [3])

    def test_qwen_training_forward_explicitly_disables_cache(self) -> None:
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.use_cache = None

            def forward(self, input_ids, use_cache):
                self.use_cache = use_cache
                logits = (
                    torch.zeros(
                        (*input_ids.shape, 64),
                        dtype=self.weight.dtype,
                        device=input_ids.device,
                    )
                    + self.weight
                )
                return type(
                    "Output",
                    (),
                    {
                        "logits": logits,
                    },
                )()

        config = load_config(
            CONFIG_ROOT / "rs_generaldesc_lora_qwen3vl_2b.yaml"
        )
        core = CaptureModel()
        adapter = Qwen3VLModelAdapter(
            core,
            identity=ModelIdentity(
                model_path="fixture",
                model_type="fixture",
                config_sha256="a" * 64,
                weights_metadata_sha256="b" * 64,
                processor_config_sha256="c" * 64,
                base_parameter_count=1,
            ),
            adaptation=config.adaptation,
        )
        adapter.forward(
            {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "labels": torch.ones((1, 2), dtype=torch.long),
            }
        )
        self.assertIs(core.use_cache, False)

    def test_causal_loss_is_equal_weight_per_sample(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn((4, 6, 32), dtype=torch.float32)
        labels = torch.tensor(
            [
                [-100, 1, 2, 3, 4, 5],
                [-100, -100, -100, 6, 7, -100],
                [-100, 8, 9, -100, -100, -100],
                [-100, -100, 10, 11, 12, -100],
            ],
            dtype=torch.long,
        )
        batched = assistant_sample_mean_causal_loss(logits, labels)
        separate = torch.stack(
            [
                assistant_sample_mean_causal_loss(
                    logits[index : index + 1],
                    labels[index : index + 1],
                )
                for index in range(4)
            ]
        ).mean()
        torch.testing.assert_close(batched, separate, rtol=0, atol=0)

        invalid = labels.clone()
        invalid[2] = -100
        with self.assertRaises(ModelError) as caught:
            assistant_sample_mean_causal_loss(logits, invalid)
        self.assertEqual(caught.exception.code, ReasonCode.LOSS_MASK_INVALID)

    def test_external_dataset_uses_phase2_epoch_multireference_contract(
        self,
    ) -> None:
        class CanonicalFixture:
            def __init__(self):
                self.epoch = 0
                self.records = [
                    {
                        "record_id": "record",
                        "parent_id": "parent",
                        "logical_role": "external_train",
                        "task_family": "global_caption",
                        "reference_responses": ["first", "second"],
                    }
                ]
                self.manifest = {
                    "build_id": "build",
                    "payload_root_sha256": "payload",
                }

            def set_epoch(self, epoch):
                self.epoch = epoch

            def training_response(self, record):
                return ("first", "second")[self.epoch % 2]

        rendered = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe."}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "placeholder"}],
                },
            ]
        }
        canonical = CanonicalFixture()
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "oa_groundrag.phase4.data.render_canonical_messages",
            return_value=(rendered, []),
        ) as renderer:
            dataset = ExternalDescriptionDataset(
                canonical,
                derived_root=Path(temporary) / "derived",
                seed=9,
            )
            dataset.set_epoch(0)
            first = dataset[0].messages[-1]["content"][0]["text"]
            dataset.set_epoch(1)
            second = dataset[0].messages[-1]["content"][0]["text"]
        self.assertEqual((first, second), ("first", "second"))
        renderer.assert_called_once()


class MaskInputAdapterTests(unittest.TestCase):
    def _write_auxseg_fixture(
        self,
        root: Path,
        *,
        duplicate: bool = False,
    ) -> None:
        root.mkdir()
        manifest = {
            "schema_version": "oa_auxseg_inference_v6",
            "checkpoint": "/read-only/checkpoint.pt",
            "checkpoint_step": 12,
            "benchmark_index_sha256": "b" * 64,
            "benchmark_contract": {"schema_version": "fixture"},
            "backbone_sha256": "c" * 64,
            "backbone": "convnext_small",
            "architecture": {"schema_version": "fixture"},
            "split": "val",
            "source": None,
            "sample_count": 2 if duplicate else 1,
            "modality_weight_order": ["dem", "__null__"],
            "modality_weight_map_strides": [4, 8, 16, 32],
            "modality_weight_map_summary": (
                "coverage_pool_each_stage_then_equal_stage_mean"
            ),
            "region_feature_dim": 266,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        array_keys = {
            "mask_probability": "probability",
            "global_mask": "global",
            "region_masks": "regions",
            "region_features": "features",
            "modality_weights": "weights",
            "modality_weight_maps": {
                "4": "weights_stride4",
                "8": "weights_stride8",
                "16": "weights_stride16",
                "32": "weights_stride32",
            },
        }
        row = {
            "sample_id": "sample-a",
            "source": "fixture",
            "split": "val",
            "available_modalities": ["dem"],
            "active_modalities": ["dem"],
            "no_target_score": 0.1,
            "modality_names": ["dem", "__null__"],
            "modality_weights": [0.8, 0.2],
            "array_keys": array_keys,
            "regions": [
                {
                    "region_id": "region_001",
                    "bbox_xyxy": [0, 0, 1, 1],
                    "centroid_xy": [0.0, 0.0],
                    "area_pixels": 999,
                    "confidence": 0.75,
                }
            ],
        }
        lines = [json.dumps(row)]
        if duplicate:
            lines.append(json.dumps(row))
        (root / "predictions.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        global_mask = np.zeros((6, 7), dtype=np.uint8)
        global_mask[1:5, 2:6] = 1
        region_mask = np.zeros_like(global_mask)
        region_mask[2:4, 3:5] = 1
        np.savez(
            root / "predictions.npz",
            **{
                "global": global_mask,
                "regions": region_mask[None, ...],
            },
        )

    def test_auxseg_v6_inventory_uses_masks_not_feature_or_geometry_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inference"
            self._write_auxseg_fixture(root)
            value = inventory_from_auxseg_inference(
                root,
                sample_id="sample-a",
                mask_mode=MaskMode.FIXED_PREDICTED_MASK,
                max_rows=4,
            )
        self.assertTrue(
            value.source_identity.startswith("oa_auxseg_inference_v6:")
        )
        self.assertTrue(value.source_identity.endswith(":sample-a"))
        self.assertEqual(len(value.candidates), 1)
        self.assertEqual(int(value.candidates[0].mask.sum()), 4)
        self.assertEqual(
            value.candidates[0].provenance,
            "oa_auxseg_inference_v6",
        )

    def test_auxseg_duplicate_sample_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inference"
            self._write_auxseg_fixture(root, duplicate=True)
            with self.assertRaises(ContractError) as caught:
                inventory_from_auxseg_inference(
                    root,
                    sample_id="sample-a",
                    mask_mode=MaskMode.END_TO_END_PREDICTED_MASK,
                    max_rows=4,
                )
        self.assertEqual(caught.exception.code, ReasonCode.TYPE_MISMATCH)


if __name__ == "__main__":
    unittest.main()
