#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SegDesc v2 package boundaries, StageSpec and strict cache migration tests."""

from __future__ import annotations

import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

import torch

from qpsalm_seg.description.protocols.config import (
    SEGDESC_CONFIG_PROTOCOL,
    load_segdesc_config,
    require_serialized_segdesc_config,
)
from qpsalm_seg.description.data.vision_cache import (
    DESCRIPTION_CACHE_BUILDER_VERSION,
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL,
    description_cache_key,
    source_cache_snapshot,
)
from qpsalm_seg.description.protocols.stages import (
    DESCRIPTION_STAGES,
    DESCRIPTION_STREAM_SEED_OFFSETS,
    get_stage_spec,
)
from qpsalm_seg.description.protocols.launch import (
    D0_TRAINING_LAUNCH_PROTOCOL,
)
from qpsalm_seg.description.protocols.io import (
    nested_file_bindings_current,
)
from qpsalm_seg.description.protocols.versions import (
    DESCRIPTION_COLLATOR_AUDIT_PROTOCOL,
    D0_CONSTRUCTION_CONTRACT_PROTOCOL,
    D0_PREFLIGHT_ACCEPTANCE_PROTOCOL,
)
from qpsalm_seg.description.data.cache_migration import (
    CACHE_MIGRATION_PROTOCOL,
    hardlink_shard,
    load_legacy_shard,
    revalidate_published_cache_origin,
    revalidate_published_cache_migration,
    validate_migration_record,
)
from qpsalm_seg.description.data.artifact_readiness import (
    ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL,
    validate_artifact_readiness_report,
)
from qpsalm_seg.description.data.datasets import collate_description
from qpsalm_seg.description.data.loaders import (
    DMinusOneTaskPathBatchSampler,
    description_collator_audit,
)
from qpsalm_seg.description.data.unified_artifact import (
    UNIFIED_BUILDER_VERSION,
    UNIFIED_STATISTICS_PROTOCOL,
    UNIFIED_VALIDATION_PROTOCOL,
)
from qpsalm_seg.description.workflows.d0_acceptance import (
    validate_d0_preflight_for_launch,
)
from qpsalm_seg.description.workflows.d0_preflight import run_d0_preflight
from qpsalm_seg.description.workflows.artifact_readiness import (
    ARTIFACT_READINESS_PROTOCOL,
    run_artifact_readiness,
)
from qpsalm_seg.description.workflows.d_minus_one import (
    d_minus_one_train_arguments,
    d_minus_one_zero_shot_arguments,
)
from qpsalm_seg.description.workflows.train import (
    DescriptionLaunchError,
    run_description_training,
)
from qpsalm_seg.description.workflows.zero_shot import (
    run_zero_shot_evaluation,
)
from qpsalm_seg.description.training.gradient_gates import (
    DescriptionGradientGateTracker,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION_ROOT = (
    REPOSITORY_ROOT / "SEG_Multi-Source_Landslides/qpsalm_seg/description"
)


def _cache_fixture(*, source_cache: str | None = None):
    component = "multisource_parent" if source_cache else "single_image"
    parent = "parent-1"
    key = description_cache_key(component, parent)
    source_content_hash = "a" * 64
    view = {
        "name": "view",
        "description": "synthetic",
        "source_modalities": ["rgb"],
        "source_families": ["optical"],
        "quality_flags": [],
        "content_hash": "view-content",
        "render_transform": {},
        "vision_grid_thw": [1, 2, 2],
        "merged_grid_hw": [2, 2],
        "valid_mask": torch.ones(1, 2, 2),
        "spatial_features": [
            torch.ones(2, 2, 2),
            torch.ones(2, 1, 1),
            torch.ones(2, 1, 1),
            torch.ones(2, 1, 1),
        ],
        "view_tokens": torch.ones(2, 3),
    }
    fingerprint = hashlib.sha256("|".join([
        DESCRIPTION_CACHE_PROTOCOL,
        key,
        source_content_hash,
        "model-revision",
        "processor-revision",
        "view-content",
    ]).encode()).hexdigest()
    row = {
        "lookup_key": key,
        "component": component,
        "parent_sample_id": parent,
        "source_ref": "benchmark/source",
        "source_content_hash": source_content_hash,
        "source_cache": source_cache,
        "cache_fingerprint": fingerprint,
        "views": [view],
    }
    manifest = {
        "format": DESCRIPTION_CACHE_FORMAT,
        "protocol": DESCRIPTION_CACHE_PROTOCOL,
        "model_revision": "model-revision",
        "processor_revision": "processor-revision",
        "spatial_sizes": [2, 1, 1, 1],
        "spatial_channels": 2,
        "token_dim": 3,
        "view_tokens_per_view": 2,
        "lookup": {
            key: {
                "shard": 0,
                "index": 0,
                "component": component,
                "parent_sample_id": parent,
            }
        },
    }
    return manifest, row, {key: source_content_hash}


class SegDescArchitectureTest(unittest.TestCase):
    def test_artifact_acceptance_cannot_inherit_smoke_or_dry_run(self) -> None:
        script = (
            REPOSITORY_ROOT / "scripts/run_segdesc_artifact_acceptance.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('ARTIFACT_SEED="${ARTIFACT_SEED:-42}"', script)
        self.assertGreaterEqual(script.count("MAX_SAMPLES=0 \\\n"), 2)
        self.assertIn("DRY_RUN= \\\n", script)

    def test_zero_shot_failure_removes_partial_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "zero-shot"

            def fail_after_raw(**_: object) -> dict[str, object]:
                (output / "raw_generations.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                raise RuntimeError("synthetic publication failure")

            with mock.patch(
                "qpsalm_seg.description.evaluation.zero_shot."
                "evaluate_zero_shot_global_caption",
                side_effect=fail_after_raw,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "synthetic publication failure"
                ):
                    run_zero_shot_evaluation(
                        model_path="models_zoo/Qwen3-VL-2B-Instruct",
                        benchmark="benchmark/qpsalm_description_v2_small",
                        split="dev",
                        output_dir=str(output),
                        device_name="cpu",
                        max_samples=64,
                        max_new_tokens=8,
                        seed=42,
                        load_4bit=False,
                        overwrite_output=False,
                    )
            self.assertTrue((output / "failure_report.json").is_file())
            self.assertFalse((output / "eval_report.json").exists())
            self.assertFalse((output / "raw_generations.jsonl").exists())

    def test_d_minus_one_sampler_keeps_gradient_windows_path_homogeneous(self) -> None:
        class Dataset:
            rows = [
                {"_d_minus_one_category": category}
                for category in ("global", "box", "mask", "null")
                for _ in range(16)
            ]

        dataset = Dataset()
        sampler = DMinusOneTaskPathBatchSampler(
            dataset,
            batch_size=2,
            gradient_window_batches=4,
            seed=42,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        observed = set()
        for start in range(0, len(batches), 4):
            window = batches[start:start + 4]
            categories = {
                dataset.rows[index]["_d_minus_one_category"]
                for batch in window
                for index in batch
            }
            paths = {
                "global_caption"
                if dataset.rows[index]["_d_minus_one_category"] == "global"
                else "region_description"
                for batch in window
                for index in batch
            }
            self.assertEqual(len(paths), 1)
            if paths == {"region_description"}:
                self.assertTrue(categories & {"box", "mask"})
            observed.update(paths)
        self.assertEqual(
            observed, {"global_caption", "region_description"}
        )

    def test_collator_audit_consumes_the_real_training_contract(self) -> None:
        batch = collate_description([{
            "request": ("single_image", "parent-1"),
            "region_mask": torch.ones(1, 3, 4),
            "instruction": "Describe the image.",
            "target_text": "A synthetic scene.",
            "reference_texts": ["A synthetic scene."],
            "structured_output": False,
            "use_region_tokens": False,
            "weight": 1.0,
            "sample_id": "sample-1",
            "parent_sample_id": "parent-1",
            "task_family": "global_caption",
        }])
        self.assertEqual(
            batch["requests"], [["single_image", "parent-1"]]
        )
        audit = description_collator_audit(batch)
        self.assertEqual(
            audit["protocol"],
            DESCRIPTION_COLLATOR_AUDIT_PROTOCOL,
        )
        self.assertEqual(audit["batch_size"], 1)
        self.assertEqual(audit["row_tasks"], ["global_caption"])
        self.assertEqual(audit["region_grounded_samples"], 0)
        self.assertNotIn("rows", audit["contract_fields"])
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            description_collator_audit({
                "rows": [{"task_family": "global_caption"}],
                "region_masks": torch.ones(1, 1, 3, 4),
                "weights": torch.ones(1),
            })

    def test_d_minus_one_workflow_owns_fixed_engineering_budget(self) -> None:
        train = d_minus_one_train_arguments([])
        self.assertEqual(train[train.index("--stage") + 1], "overfit")
        self.assertEqual(train[train.index("--max-steps") + 1], "100")
        self.assertEqual(
            train[train.index("--max-train-samples") + 1], "64"
        )
        self.assertEqual(train[train.index("--batch-size") + 1], "2")
        zero = d_minus_one_zero_shot_arguments([])
        self.assertEqual(zero, ["--max-samples", "64"])
        with self.assertRaisesRegex(ValueError, "固定预算禁止覆盖"):
            d_minus_one_train_arguments(["--max-steps", "101"])
        with self.assertRaisesRegex(ValueError, "batch-size >= 2"):
            d_minus_one_train_arguments(["--batch-size=1"])
        with self.assertRaisesRegex(ValueError, "出现冲突值"):
            d_minus_one_train_arguments([
                "--batch-size", "2", "--batch-size=4",
            ])
        with self.assertRaisesRegex(ValueError, "固定预算禁止覆盖"):
            d_minus_one_zero_shot_arguments(["--max-samples", "32"])
        with self.assertRaisesRegex(ValueError, "max_steps=100"):
            load_segdesc_config(
                "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
                {"stage": "overfit", "batch_size": 2},
            )
        fixed = load_segdesc_config(
            "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
            {
                "stage": "overfit",
                "batch_size": 2,
                "max_steps": 100,
                "max_train_samples": 64,
            },
        )
        self.assertEqual(fixed.training.max_steps, 100)
        self.assertEqual(fixed.data.max_train_samples, 64)

    def test_d_minus_one_gradient_tracker_requires_both_real_paths(self) -> None:
        tracker = DescriptionGradientGateTracker(
            ["main"], run_stage="overfit"
        )
        with mock.patch(
            "qpsalm_seg.description.training.gradient_gates."
            "description_step_gradient_gate",
            return_value={"passed": True, "checks": {"synthetic": True}},
        ):
            tracker.audit_window(
                object(),
                object(),
                stream_name="main",
                stream_stage="overfit",
                observed_task_paths={"global_caption"},
            )
            self.assertFalse(tracker.complete)
            self.assertFalse(tracker.payload()["passed"])
            tracker.audit_window(
                object(),
                object(),
                stream_name="main",
                stream_stage="overfit",
                observed_task_paths={"region_description"},
            )
        self.assertTrue(tracker.complete)
        self.assertEqual(
            tracker.payload()["streams"]["main"]["observed_task_paths"],
            ["global_caption", "region_description"],
        )

    def test_config_v2_is_composed_and_stage_spec_is_centralized(self) -> None:
        config = load_segdesc_config(
            "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml"
        )
        self.assertEqual(config.protocol, SEGDESC_CONFIG_PROTOCOL)
        self.assertEqual(config.model.region_encoder, "mgrr")
        self.assertEqual(config.data.num_workers, 0)
        self.assertEqual(config.training.stage, "bridge_auto")
        self.assertEqual(config.evaluation.evaluation_mode, "gt_mask")
        self.assertIn("model", config.to_dict())
        self.assertFalse(hasattr(config, "stage"))
        self.assertEqual(
            require_serialized_segdesc_config(config.to_dict()),
            config.to_dict(),
        )
        with self.assertRaisesRegex(ValueError, "不是.*config_v2"):
            require_serialized_segdesc_config({"stage": "bridge_auto"})

        d0 = get_stage_spec("mmrs_caption")
        self.assertEqual(d0.milestone, "D0")
        self.assertFalse(d0.uses_region_tokens)
        self.assertEqual(d0.region_token_policy, "forbidden")
        self.assertEqual(
            d0.data_sources,
            ("description_v4:mmrs_global_caption_train",),
        )
        self.assertTrue(d0.trains_desc_adapter)
        self.assertEqual(d0.initialization_kind, "segmentation_checkpoint")
        self.assertIsNone(d0.initialize_from_stage)
        self.assertTrue(d0.requires_d_minus_one_gate)
        d_minus_one = get_stage_spec("overfit")
        self.assertTrue(d_minus_one.uses_region_tokens)
        self.assertEqual(
            d_minus_one.region_token_policy, "mixed_explicit"
        )
        d3b = get_stage_spec("bridge_expert")
        self.assertTrue(d3b.requires_expert_bridge)
        self.assertEqual(d3b.initialize_from_stage, "bridge_auto")
        self.assertEqual(d3b.initialize_from_checkpoint_role, "terminal_last")
        self.assertIn("bridge_m2_expert_pilot_frozen", d3b.gate_requirements)
        self.assertIn("alignment_text_projection.", d3b.trainable_prefixes)
        for name in DESCRIPTION_STAGES:
            spec = get_stage_spec(name)
            self.assertTrue(spec.data_sources)
            self.assertTrue(spec.gate_requirements)
            if spec.initialize_from_stage is not None:
                self.assertEqual(
                    spec.initialization_kind, "previous_stage_checkpoint"
                )
                predecessor = get_stage_spec(spec.initialize_from_stage)
                self.assertEqual(
                    spec.initialize_from_checkpoint_role,
                    predecessor.checkpoint_role,
                )

    def test_installed_package_discovers_subpackages_and_unified_entrypoint(
        self,
    ) -> None:
        """Keep the refactor usable after installation, not only via PYTHONPATH."""
        pyproject_path = (
            REPOSITORY_ROOT / "SEG_Multi-Source_Landslides/pyproject.toml"
        )
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertIn("qpsalm_seg*", package_find["include"])
        self.assertEqual(
            pyproject["project"]["scripts"]["qpsalm-segdesc"],
            "qpsalm_seg.cli.segdesc:main",
        )
        for package in (
            "modeling", "data", "training", "evaluation", "protocols",
            "workflows",
        ):
            self.assertTrue((DESCRIPTION_ROOT / package / "__init__.py").is_file())

    def test_nested_cache_bindings_resolve_relative_to_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            manifest = cache / "manifest.json"
            validation = cache / "validation_report.json"
            manifest.write_bytes(b"manifest")
            validation.write_bytes(b"validation")
            binding = {
                "cache_dir": str(cache),
                "manifest": {
                    "path": "manifest.json",
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "validation_report": {
                    "path": "validation_report.json",
                    "sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
                },
            }
            self.assertTrue(nested_file_bindings_current(binding))
            self.assertTrue(nested_file_bindings_current({
                "origin": {"path": None, "sha256": None},
            }))
            self.assertFalse(nested_file_bindings_current({
                "origin": {"path": None, "sha256": "0" * 64},
            }))
            validation.write_bytes(b"drifted")
            self.assertFalse(nested_file_bindings_current(binding))

    def test_description_module_graph_is_acyclic(self) -> None:
        prefix = "qpsalm_seg.description"
        modules = {}
        for path in DESCRIPTION_ROOT.rglob("*.py"):
            parts = list(path.relative_to(DESCRIPTION_ROOT).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            name = ".".join((prefix, *parts)) if parts else prefix
            modules[name] = path
        edges = {name: set() for name in modules}
        for name, path in modules.items():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            package = name if path.name == "__init__.py" else name.rpartition(".")[0]
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Import):
                    targets.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        package_parts = package.split(".")
                        base = ".".join(
                            package_parts[:len(package_parts) - node.level + 1]
                        )
                        targets.append(
                            ".".join(value for value in (base, node.module or "") if value)
                        )
                    else:
                        targets.append(node.module or "")
                for target in targets:
                    candidates = [
                        candidate for candidate in modules
                        if target == candidate or target.startswith(candidate + ".")
                    ]
                    if candidates:
                        dependency = max(candidates, key=len)
                        if dependency != name:
                            edges[name].add(dependency)

        visiting = set()
        visited = set()

        def visit(name: str, trail: tuple[str, ...]) -> None:
            if name in visiting:
                raise AssertionError(" -> ".join((*trail, name)))
            if name in visited:
                return
            visiting.add(name)
            for dependency in edges[name]:
                visit(dependency, (*trail, name))
            visiting.remove(name)
            visited.add(name)

        for name in sorted(modules):
            visit(name, ())

    def test_local_from_imports_resolve_to_public_symbols(self) -> None:
        """Catch stale imports left behind by module moves before runtime."""
        package_root = DESCRIPTION_ROOT.parent
        prefix = "qpsalm_seg"
        modules = {}
        packages = {}
        symbols = {}
        for path in package_root.rglob("*.py"):
            parts = list(
                path.relative_to(package_root).with_suffix("").parts
            )
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            name = ".".join((prefix, *parts)) if parts else prefix
            tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
            modules[name] = (path, tree)
            packages[name] = is_package
            public = set()
            for node in tree.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    public.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign) else [node.target]
                    )
                    public.update(
                        target.id
                        for target in targets
                        if isinstance(target, ast.Name)
                    )
                elif isinstance(node, ast.Import):
                    public.update(
                        alias.asname or alias.name.split(".")[0]
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    public.update(
                        alias.asname or alias.name for alias in node.names
                    )
            symbols[name] = public

        unresolved = []
        for name, (path, tree) in modules.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    package = name if packages[name] else name.rpartition(".")[0]
                    parts = package.split(".")
                    target = ".".join(
                        parts[:len(parts) - node.level + 1]
                        + ((node.module or "").split(".") if node.module else [])
                    )
                else:
                    target = node.module or ""
                if target not in modules or "__getattr__" in symbols[target]:
                    continue
                for imported in node.names:
                    child_module = f"{target}.{imported.name}"
                    if (
                        imported.name != "*"
                        and imported.name not in symbols[target]
                        and child_module not in modules
                    ):
                        unresolved.append((
                            str(path.relative_to(package_root)),
                            node.lineno,
                            target,
                            imported.name,
                        ))
        self.assertEqual(unresolved, [])

    def test_dependency_layers_and_private_imports_are_static(self) -> None:
        layers = {
            "protocols": 0,
            "modeling": 1,
            "data": 1,
            "training": 2,
            "evaluation": 2,
            "workflows": 3,
        }
        violations = []
        private_imports = []
        for path in DESCRIPTION_ROOT.rglob("*.py"):
            relative = path.relative_to(DESCRIPTION_ROOT)
            source = relative.parts[0] if len(relative.parts) > 1 else "root"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_parts = [
                "qpsalm_seg", "description", *relative.with_suffix("").parts
            ]
            if module_parts[-1] == "__init__":
                module_parts.pop()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                imported_private = [
                    name.name for name in node.names if name.name.startswith("_")
                ]
                if imported_private:
                    private_imports.append(
                        (str(relative), node.lineno, imported_private)
                    )
                if source not in layers:
                    continue
                if node.level:
                    target = module_parts[:-node.level] + (
                        (node.module or "").split(".") if node.module else []
                    )
                else:
                    target = (node.module or "").split(".")
                if (
                    len(target) >= 3
                    and target[:2] == ["qpsalm_seg", "description"]
                    and target[2] in layers
                    and layers[source] < layers[target[2]]
                ):
                    violations.append(
                        (str(relative), node.lineno, source, target[2])
                    )
        self.assertEqual(private_imports, [])
        self.assertEqual(violations, [])

    def test_typed_segdesc_config_has_no_flat_runtime_access(self) -> None:
        """Keep v2 consumers on explicit model/data/train/eval sections."""
        stable_attributes = {
            "protocol", "model", "data", "training", "evaluation", "joint",
            "validate", "to_dict", "with_overrides",
            "resolved_joint_task_pattern",
        }
        violations = []
        for path in DESCRIPTION_ROOT.rglob("*.py"):
            tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
            for function in (
                node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                typed_names = {
                    argument.arg
                    for argument in (
                        *function.args.posonlyargs,
                        *function.args.args,
                        *function.args.kwonlyargs,
                    )
                    if (
                        isinstance(argument.annotation, ast.Name)
                        and argument.annotation.id == "SegDescConfig"
                    )
                }
                for node in ast.walk(function):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in typed_names
                        and node.attr not in stable_attributes
                    ):
                        violations.append((
                            str(path.relative_to(DESCRIPTION_ROOT)),
                            node.lineno,
                            node.value.id,
                            node.attr,
                        ))
        self.assertEqual(violations, [])

    def test_module_and_entrypoint_sizes_remain_bounded(self) -> None:
        """Prevent orchestration and algorithm contracts from collapsing into monoliths."""
        oversized_core = {
            str(path.relative_to(DESCRIPTION_ROOT)): len(
                path.read_text(encoding="utf-8").splitlines()
            )
            for path in DESCRIPTION_ROOT.rglob("*.py")
            if len(path.read_text(encoding="utf-8").splitlines()) > 900
        }
        self.assertEqual(oversized_core, {})

        cli_root = DESCRIPTION_ROOT.parent / "cli"
        entrypoints = (
            "segdesc.py",
            "cache_description_vision_features.py",
            "train_description.py",
            "eval_description.py",
            "eval_description_zero_shot.py",
            "train_segdesc_joint.py",
            "eval_segdesc_retention.py",
            "validate_d_minus_one.py",
            "validate_m4_region_encoder_suite.py",
            "validate_d4_curriculum.py",
            "validate_m6_acceptance.py",
        )
        oversized_cli = {
            name: len((cli_root / name).read_text(encoding="utf-8").splitlines())
            for name in entrypoints
            if len((cli_root / name).read_text(encoding="utf-8").splitlines()) > 250
        }
        self.assertEqual(oversized_cli, {})

        oversized_workflows = {
            str(path.relative_to(DESCRIPTION_ROOT)): len(
                path.read_text(encoding="utf-8").splitlines()
            )
            for path in (DESCRIPTION_ROOT / "workflows").glob("*.py")
            if len(path.read_text(encoding="utf-8").splitlines()) > 350
        }
        self.assertEqual(oversized_workflows, {})

    def test_segdesc_entrypoints_use_only_orchestration_and_config_boundaries(
        self,
    ) -> None:
        """Keep reusable data/model/training/evaluation code out of CLI modules."""
        cli_root = DESCRIPTION_ROOT.parent / "cli"
        entrypoints = (
            "segdesc.py",
            "cache_description_vision_features.py",
            "train_description.py",
            "eval_description.py",
            "eval_description_zero_shot.py",
            "train_segdesc_joint.py",
            "eval_segdesc_retention.py",
            "demo_description.py",
            "build_oof_folds.py",
            "export_predicted_regions.py",
            "merge_oof_predictions.py",
            "compare_description_runs.py",
            "compare_segdesc_retention.py",
            "score_caption_metrics.py",
            "score_caption_human_review.py",
            "score_expert_factuality.py",
            "validate_d_minus_one.py",
            "validate_m4_region_encoder_suite.py",
            "validate_d4_curriculum.py",
            "validate_m6_acceptance.py",
        )
        violations = []
        for name in entrypoints:
            path = cli_root / name
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = str(node.module or "")
                prefix = "qpsalm_seg.description."
                if not module.startswith(prefix):
                    continue
                boundary = module[len(prefix):].split(".", 1)[0]
                if (
                    boundary != "workflows"
                    and module != "qpsalm_seg.description.protocols.config"
                ):
                    violations.append((name, node.lineno, module))
        self.assertEqual(violations, [])

    def test_argument_parsing_does_not_leak_into_algorithm_layers(self) -> None:
        """Only CLI/workflow orchestration may own argparse or import CLI code."""
        violations = []
        algorithm_layers = (
            "protocols", "modeling", "data", "training", "evaluation",
        )
        for layer in algorithm_layers:
            for path in (DESCRIPTION_ROOT / layer).rglob("*.py"):
                tree = ast.parse(
                    path.read_text(encoding="utf-8"), filename=str(path)
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [str(node.module or "")]
                    else:
                        continue
                    for module in modules:
                        if module == "argparse" or module.startswith(
                            "qpsalm_seg.cli"
                        ):
                            violations.append((
                                str(path.relative_to(DESCRIPTION_ROOT)),
                                node.lineno,
                                module,
                            ))
        self.assertEqual(violations, [])

    def test_cache_migration_rejects_content_and_lookup_drift(self) -> None:
        manifest, row, current = _cache_fixture()
        self.assertFalse(validate_migration_record(
            row,
            legacy_manifest=manifest,
            shard_index=0,
            local_index=0,
            current_content=current,
            source_record_fingerprint=lambda _: "unused",
        ))
        drifted = dict(current)
        drifted[row["lookup_key"]] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "source_content_hash"):
            validate_migration_record(
                row,
                legacy_manifest=manifest,
                shard_index=0,
                local_index=0,
                current_content=drifted,
                source_record_fingerprint=lambda _: "unused",
            )
        with self.assertRaisesRegex(ValueError, "位置不一致"):
            validate_migration_record(
                row,
                legacy_manifest=manifest,
                shard_index=0,
                local_index=1,
                current_content=current,
                source_record_fingerprint=lambda _: "unused",
            )
        manifest, nonfinite, current = _cache_fixture()
        nonfinite["views"][0]["view_tokens"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "非 finite"):
            validate_migration_record(
                nonfinite,
                legacy_manifest=manifest,
                shard_index=0,
                local_index=0,
                current_content=current,
                source_record_fingerprint=lambda _: "unused",
            )

    def test_cache_migration_replays_source_record_and_hardlinks(self) -> None:
        manifest, row, current = _cache_fixture(source_cache="qmv3-parent:parent-1")
        self.assertTrue(validate_migration_record(
            row,
            legacy_manifest=manifest,
            shard_index=0,
            local_index=0,
            current_content=current,
            source_record_fingerprint=lambda _: row["source_content_hash"],
        ))
        with self.assertRaisesRegex(RuntimeError, "segmentation cache record"):
            validate_migration_record(
                row,
                legacy_manifest=manifest,
                shard_index=0,
                local_index=0,
                current_content=current,
                source_record_fingerprint=lambda _: "c" * 64,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.pt"
            target = root / "current.pt"
            source.write_bytes(b"immutable-shard")
            audit = hardlink_shard(source, target)
            self.assertTrue(audit["same_inode"])
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_cache_migration_rejects_corrupted_shard_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard_00001.pt"
            shard.write_bytes(b"not-a-torch-shard")
            with self.assertRaisesRegex(ValueError, "shard 无法读取"):
                load_legacy_shard(shard)

    def test_d0_preflight_publishes_ready_without_optimizer_step(self) -> None:
        config = load_segdesc_config(
            "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
            {
                "stage": "mmrs_caption",
                "d_minus_one_gate": "outputs/synthetic/d_minus_one_gate.json",
            },
        )

        class Bank:
            snapshot = {"file_count": 3, "metadata_sha256": "c" * 64}

            def verify_all_shards(self):
                return {
                    "all_verified": True,
                    "verified_shards": 1,
                    "metadata_snapshot": self.snapshot,
                }

            def artifact_binding(self):
                return {"protocol": "synthetic-cache-binding"}

            def file_metadata_snapshot(self):
                return self.snapshot

        class Loader:
            dataset = object()
            batch_sampler = object()

            def __len__(self):
                return 1

            def __iter__(self):
                yield collate_description([{
                    "request": ("single_image", "parent-1"),
                    "region_mask": torch.zeros(1, 2, 2),
                    "instruction": "Describe the image.",
                    "target_text": "A synthetic scene.",
                    "reference_texts": ["A synthetic scene."],
                    "structured_output": False,
                    "use_region_tokens": False,
                    "weight": 1.0,
                    "sample_id": "sample-1",
                    "parent_sample_id": "parent-1",
                    "task_family": "global_caption",
                }])

        class Optimizer:
            state = {}
            param_groups = [{"name": "desc_adapter"}]

        parameter_manifest = {
            "groups": [{
                "name": "desc_adapter",
                "numel": 4,
                "parameter_names": ["controller.desc_adapter.lora_A.weight"],
            }]
        }
        module = "qpsalm_seg.description.workflows.d0_preflight"
        acceptance_module = (
            "qpsalm_seg.description.workflows.d0_acceptance"
        )
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            enter = stack.enter_context
            enter(mock.patch(
                f"{module}.validate_d_minus_one_gate",
                return_value={"protocol": "accepted", "passed": True},
            ))
            enter(mock.patch(
                f"{module}.DescriptionVisionFeatureBank", return_value=Bank()
            ))
            enter(mock.patch(
                f"{module}.revalidate_published_cache_origin",
                return_value={
                    "origin": "synthetic",
                    "checks": {"source_cache_current": True},
                },
            ))
            enter(mock.patch(
                f"{module}.require_engineering_description",
                return_value={"protocol": "description-audit"},
            ))
            enter(mock.patch(
                f"{module}.require_engineering_bridge",
                return_value={
                    "protocol": "bridge-audit",
                    "status": "awaiting_expert_review",
                    "expert_truth_used": False,
                },
            ))
            enter(mock.patch(
                f"{module}.description_device",
                return_value=torch.device("cpu"),
            ))
            enter(mock.patch(
                f"{module}.build_segdesc_model",
                return_value=(object(), {
                    "protocol": "synthetic-migration",
                }),
            ))
            enter(mock.patch(
                f"{module}.build_description_dataset",
                return_value=object(),
            ))
            enter(mock.patch(
                f"{module}.dataset_data_audit",
                return_value={
                    "num_samples": 1,
                    "population_sha256": "a" * 64,
                },
            ))
            preflight_loader = enter(mock.patch(
                f"{module}.build_description_loader",
                return_value=Loader(),
            ))
            enter(mock.patch(
                f"{module}.description_stream_binding",
                return_value={"protocol": "synthetic-stream-binding"},
            ))
            enter(mock.patch(
                f"{module}.build_description_optimizer",
                return_value=(Optimizer(), object()),
            ))
            enter(mock.patch(
                f"{module}.description_trainable_parameter_manifest",
                return_value=parameter_manifest,
            ))
            output_safety = enter(mock.patch(
                f"{module}.validate_output_replacement_safety"
            ))
            enter(mock.patch(
                f"{acceptance_module}.validate_d_minus_one_gate",
                return_value={"protocol": "accepted", "passed": True},
            ))
            enter(mock.patch(
                f"{acceptance_module}.DescriptionVisionFeatureBank",
                return_value=Bank(),
            ))
            enter(mock.patch(
                f"{acceptance_module}.revalidate_published_cache_origin",
                return_value={
                    "origin": "synthetic",
                    "checks": {"source_cache_current": True},
                },
            ))
            enter(mock.patch(
                f"{acceptance_module}.require_engineering_description",
                return_value={"protocol": "description-audit"},
            ))
            enter(mock.patch(
                f"{acceptance_module}.require_engineering_bridge",
                return_value={
                    "protocol": "bridge-audit",
                    "status": "awaiting_expert_review",
                    "expert_truth_used": False,
                },
            ))
            enter(mock.patch(
                f"{acceptance_module}.build_description_dataset",
                return_value=object(),
            ))
            enter(mock.patch(
                f"{acceptance_module}.dataset_data_audit",
                return_value={
                    "num_samples": 1,
                    "population_sha256": "a" * 64,
                },
            ))
            acceptance_loader = enter(mock.patch(
                f"{acceptance_module}.build_description_loader",
                return_value=Loader(),
            ))
            enter(mock.patch(
                f"{acceptance_module}.description_stream_binding",
                return_value={"protocol": "synthetic-stream-binding"},
            ))
            root = Path(directory)
            report = run_d0_preflight(
                config,
                device_name="cpu",
                output_dir=root / "preflight",
                formal_output_dir=root / "formal",
            )
            formal_config = load_segdesc_config(
                report["formal_training_launch"]["resolved_config"],
                {"output_dir": str(root / "formal")},
            )
            acceptance = validate_d0_preflight_for_launch(
                formal_config,
                config_reference=report[
                    "formal_training_launch"
                ]["resolved_config"],
                report_reference=root / "preflight/preflight_report.json",
                device_name="cpu",
            )
            safety_inputs = output_safety.call_args.args[1]
            self.assertEqual(
                safety_inputs["unified-benchmark"],
                config.data.unified_benchmark,
            )
            expected_loader_seed = (
                config.training.seed
                + DESCRIPTION_STREAM_SEED_OFFSETS["main"]
            )
            self.assertEqual(
                preflight_loader.call_args.kwargs["sampler_seed"],
                expected_loader_seed,
            )
            self.assertEqual(
                acceptance_loader.call_args.kwargs["sampler_seed"],
                expected_loader_seed,
            )
            self.assertEqual(
                report["formal_training_launch"]["protocol"],
                D0_TRAINING_LAUNCH_PROTOCOL,
            )
            self.assertEqual(
                report["formal_training_launch"]["argv"][3:5],
                ["qpsalm_seg.cli.segdesc", "train"],
            )
            report_path = root / "preflight/preflight_report.json"
            tampered = json.loads(json.dumps(report))
            tampered["resolved_config_sha256"] = "0" * 64
            report_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical config SHA"):
                validate_d0_preflight_for_launch(
                    formal_config,
                    config_reference=report[
                        "formal_training_launch"
                    ]["resolved_config"],
                    report_reference=report_path,
                    device_name="cpu",
                )
            tampered = json.loads(json.dumps(report))
            tampered["formal_training_launch"]["command"] += " --learning-rate 1e-9"
            report_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "唯一正式命令"):
                validate_d0_preflight_for_launch(
                    formal_config,
                    config_reference=report[
                        "formal_training_launch"
                    ]["resolved_config"],
                    report_reference=report_path,
                    device_name="cpu",
                )
        self.assertTrue(report["ready"])
        self.assertEqual(report["status"], "engineering-valid")
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(report["checks"]["no_optimizer_step"])
        self.assertTrue(report["formal_training_launch"]["unique"])
        self.assertIn("qpsalm_seg.cli.segdesc train", report[
            "formal_training_launch"
        ]["command"])
        self.assertNotIn(
            "--overwrite-output", report["formal_training_launch"]["command"]
        )
        self.assertIn(
            "--d0-preflight-report", report["formal_training_launch"]["command"]
        )
        self.assertEqual(acceptance["status"], "engineering-valid")
        self.assertEqual(
            acceptance["protocol"], D0_PREFLIGHT_ACCEPTANCE_PROTOCOL
        )
        self.assertEqual(
            acceptance["construction_contract"]["protocol"],
            D0_CONSTRUCTION_CONTRACT_PROTOCOL,
        )
        self.assertEqual(
            acceptance["construction_contract"]["optimizer"],
            report["optimizer"],
        )

    def test_formal_d0_cannot_bypass_preflight_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_segdesc_config(
                "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
                {
                    "stage": "mmrs_caption",
                    "d_minus_one_gate": "outputs/synthetic/d_minus_one_gate.json",
                    "output_dir": str(Path(directory) / "formal"),
                },
            )
            with self.assertRaisesRegex(
                DescriptionLaunchError, "--d0-preflight-report"
            ):
                run_description_training(
                    config,
                    config_reference=(
                        "SEG_Multi-Source_Landslides/configs/"
                        "qpsalm_segdesc_small.yaml"
                    ),
                    device_name="cpu",
                )

    def test_d0_preflight_stops_before_model_on_cache_origin_failure(self) -> None:
        config = load_segdesc_config(
            "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
            {
                "stage": "mmrs_caption",
                "d_minus_one_gate": "outputs/synthetic/d_minus_one_gate.json",
            },
        )

        class Bank:
            def verify_all_shards(self):
                return {"all_verified": True}

            def artifact_binding(self):
                return {"protocol": "synthetic-cache-binding"}

        module = "qpsalm_seg.description.workflows.d0_preflight"
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                f"{module}.validate_d_minus_one_gate",
                return_value={"protocol": "accepted"},
            ),
            mock.patch(
                f"{module}.DescriptionVisionFeatureBank",
                return_value=Bank(),
            ),
            mock.patch(
                f"{module}.revalidate_published_cache_origin",
                return_value={
                    "origin": "synthetic",
                    "checks": {"source_cache_current": False},
                },
            ),
            mock.patch(f"{module}.build_segdesc_model") as build_model,
        ):
            report = run_d0_preflight(
                config,
                device_name="cpu",
                output_dir=Path(directory) / "preflight",
                formal_output_dir=Path(directory) / "formal",
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["errors"][0]["type"], "RuntimeError")
        self.assertIn("cache provenance", report["errors"][0]["message"])
        build_model.assert_not_called()

    def test_formal_d0_rejects_unsafe_flags_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_segdesc_config(
                "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
                {
                    "stage": "mmrs_caption",
                    "d_minus_one_gate": "outputs/synthetic/d_minus_one_gate.json",
                    "output_dir": str(Path(directory) / "formal"),
                },
            )
            with self.assertRaisesRegex(
                DescriptionLaunchError, "禁止追加 --overwrite-output"
            ):
                run_description_training(
                    config,
                    config_reference=(
                        "SEG_Multi-Source_Landslides/configs/"
                        "qpsalm_segdesc_small.yaml"
                    ),
                    device_name="cpu",
                    overwrite_output=True,
                    d0_preflight_report="outputs/synthetic/preflight.json",
                )
            with self.assertRaisesRegex(
                DescriptionLaunchError, "禁止追加 --initialize-from"
            ):
                run_description_training(
                    config,
                    config_reference=(
                        "SEG_Multi-Source_Landslides/configs/"
                        "qpsalm_segdesc_small.yaml"
                    ),
                    device_name="cpu",
                    initialize_from="outputs/synthetic/foreign.pt",
                    d0_preflight_report="outputs/synthetic/preflight.json",
                )

    def test_d_minus_one_cannot_bypass_artifact_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_segdesc_config(
                "SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml",
                {
                    "stage": "overfit",
                    "batch_size": 2,
                    "max_steps": 100,
                    "max_train_samples": 64,
                    "output_dir": str(Path(directory) / "overfit"),
                },
            )
            with self.assertRaisesRegex(
                DescriptionLaunchError, "artifact-readiness-report"
            ):
                run_description_training(
                    config,
                    config_reference=(
                        "SEG_Multi-Source_Landslides/configs/"
                        "qpsalm_segdesc_small.yaml"
                    ),
                    device_name="cpu",
                )

    def test_artifact_readiness_binds_auto_only_unified_to_live_cache(self) -> None:
        class Bank:
            shards = ("shard-00000.pt", "shard-00001.pt")
            manifest = {
                "num_samples": 10,
                "migration": {
                    "protocol": CACHE_MIGRATION_PROTOCOL,
                    "reuse_method": "hardlink",
                    "all_same_inode": True,
                    "reused_records": 10,
                    "reused_shards": 2,
                    "reused_bytes": 2048,
                },
            }

            def verify_all_shards(self):
                return {"all_verified": True, "verified_shards": 2}

            def artifact_binding(self):
                return {"protocol": "synthetic-m3-v3"}

        module = "qpsalm_seg.description.data.artifact_readiness"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            description = root / "description"
            bridge = root / "bridge"
            segmentation = root / "segmentation"
            unified = root / "unified"
            cache = root / "cache"
            for path in (
                description / "reports",
                bridge / "reports",
                segmentation / "reports",
                unified / "reports",
                cache,
            ):
                path.mkdir(parents=True)
            (cache / "migration_report.json").write_text(
                __import__("json").dumps({
                    "protocol": CACHE_MIGRATION_PROTOCOL,
                    "status": "engineering-valid",
                    "output_dir": str(cache.resolve(strict=False)),
                    "records": 10,
                    "shards": 2,
                    "reused_bytes": 2048,
                    "errors": [],
                }),
                encoding="utf-8",
            )
            description_validation_path = (
                description / "reports/validation_report.json"
            )
            bridge_validation_path = bridge / "reports/validation_report.json"
            description_validation_path.write_text(
                '{"mode":"small"}', encoding="utf-8"
            )
            bridge_validation_path.write_text(
                '{"mode":"small","expert":{"expert_records":0,'
                '"gate_frozen":false}}',
                encoding="utf-8",
            )
            segmentation_validation_path = (
                segmentation / "reports/validation_report.json"
            )
            segmentation_validation_path.write_text("{}", encoding="utf-8")
            segmentation_instruction_path = (
                segmentation / "reports/instruction_validation_report.json"
            )
            segmentation_instruction_path.write_text("{}", encoding="utf-8")
            component_reports = {
                "segmentation": {
                    "path": str(segmentation_validation_path),
                    "sha256": hashlib.sha256(
                        segmentation_validation_path.read_bytes()
                    ).hexdigest(),
                    "instruction_validation": {
                        "path": str(segmentation_instruction_path),
                        "sha256": hashlib.sha256(
                            segmentation_instruction_path.read_bytes()
                        ).hexdigest(),
                    },
                },
                "description": {
                    "path": str(description_validation_path),
                    "sha256": hashlib.sha256(
                        description_validation_path.read_bytes()
                    ).hexdigest(),
                },
                "bridge": {
                    "path": str(bridge_validation_path),
                    "sha256": hashlib.sha256(
                        bridge_validation_path.read_bytes()
                    ).hexdigest(),
                },
            }
            validation_path = unified / "reports/validation_report.json"
            statistics_path = unified / "reports/statistics.json"
            manifest_path = unified / "manifests/component_manifest.json"
            build_path = unified / "reports/build_report.json"
            (unified / "manifests").mkdir()
            (unified / "indexes").mkdir()
            rows = [
                {
                    "unified_record_id": f"record-{index}",
                    "split": "train",
                    "task_group": "segmentation",
                    "expert_supervision": False,
                }
                for index in range(10)
            ]
            (unified / "indexes/all.jsonl").write_text(
                "".join(
                    __import__("json").dumps(row) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            for split in ("train", "dev", "val", "test"):
                selected = rows if split == "train" else []
                (unified / f"indexes/{split}.jsonl").write_text(
                    "".join(
                        __import__("json").dumps(row) + "\n"
                        for row in selected
                    ),
                    encoding="utf-8",
                )
            manifest = {
                "builder_version": UNIFIED_BUILDER_VERSION,
                "schema_version": "qpsalm_segdesc_index_v1",
                "mode": "small",
                "storage_mode": "component_references_only",
                "components": {
                    "segmentation": str(segmentation),
                    "description": str(description),
                    "bridge": str(bridge),
                },
                "num_records": 10,
                "by_split": {"train": 10},
                "by_task_group": {"segmentation": 10},
                "component_validation_reports": component_reports,
                "bridge_status": "awaiting_expert_review",
                "expert_index_published": False,
                "contains_expert_bridge": False,
            }
            manifest_path.write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            build_path.write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            validation = {
                "protocol": UNIFIED_VALIDATION_PROTOCOL,
                "builder_version": UNIFIED_BUILDER_VERSION,
                "mode": "small",
                "status": "valid",
                "errors": [],
                "component_contracts_verified": True,
                "num_records": 10,
                "by_split": {"train": 10},
                "by_task_group": {"segmentation": 10},
                "component_validation_reports": component_reports,
                "bridge_status": "awaiting_expert_review",
                "expert_index_published": False,
            }
            statistics = {
                "protocol": UNIFIED_STATISTICS_PROTOCOL,
                "builder_version": UNIFIED_BUILDER_VERSION,
                "mode": "small",
                "num_records": 10,
                "by_split": {"train": 10},
                "by_task_group": {"segmentation": 10},
                "component_validation_reports": component_reports,
                "bridge_status": "awaiting_expert_review",
                "expert_index_published": False,
                "expert_records": 0,
            }
            validation_path.write_text(
                __import__("json").dumps(validation), encoding="utf-8"
            )
            statistics_path.write_text(
                __import__("json").dumps(statistics), encoding="utf-8"
            )
            with (
                mock.patch(
                    f"{module}.DescriptionVisionFeatureBank",
                    return_value=Bank(),
                ),
                mock.patch(
                    f"{module}.require_engineering_description",
                    return_value={
                        "builder_version": "description_benchmark_m1_v4_answer_trace",
                        "validation_report": str(description_validation_path),
                    },
                ),
                mock.patch(
                    f"{module}.require_engineering_bridge",
                    return_value={
                        "builder_version": (
                            "landslide_bridge_m2_v7_expert_review_replay_bound"
                        ),
                        "status": "awaiting_expert_review",
                        "expert_truth_used": False,
                        "validation_report": str(bridge_validation_path),
                    },
                ),
                mock.patch(
                    f"{module}.revalidate_published_cache_origin",
                    return_value={
                        "checks": {
                            "report_current": True,
                            "population_current": True,
                            "legacy_reports_current": True,
                            "hardlinks_current": True,
                        },
                    },
                ),
            ):
                report = run_artifact_readiness(
                    mode="small",
                    description_benchmark=description,
                    bridge_benchmark=bridge,
                    unified_benchmark=unified,
                    description_cache=cache,
                    output=root / "readiness.json",
                )
                readiness_acceptance = validate_artifact_readiness_report(
                    root / "readiness.json",
                    expected_description_benchmark=description,
                    expected_bridge_benchmark=bridge,
                    expected_unified_benchmark=unified,
                    expected_description_cache=cache,
                )
                validation["expert_index_published"] = True
                validation_path.write_text(
                    __import__("json").dumps(validation), encoding="utf-8"
                )
                drifted = run_artifact_readiness(
                    mode="small",
                    description_benchmark=description,
                    bridge_benchmark=bridge,
                    unified_benchmark=unified,
                    description_cache=cache,
                    output=root / "drifted.json",
                )
                validation["expert_index_published"] = False
                validation_path.write_text(
                    __import__("json").dumps(validation), encoding="utf-8"
                )
                (unified / "indexes/train.jsonl").write_text(
                    "", encoding="utf-8"
                )
                (unified / "indexes/dev.jsonl").write_text(
                    "".join(
                        __import__("json").dumps(row) + "\n" for row in rows
                    ),
                    encoding="utf-8",
                )
                split_drifted = run_artifact_readiness(
                    mode="small",
                    description_benchmark=description,
                    bridge_benchmark=bridge,
                    unified_benchmark=unified,
                    description_cache=cache,
                    output=root / "split_drifted.json",
                )
        self.assertEqual(report["protocol"], ARTIFACT_READINESS_PROTOCOL)
        self.assertEqual(
            readiness_acceptance["protocol"],
            ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL,
        )
        self.assertTrue(report["ready"])
        self.assertEqual(report["status"], "engineering-valid")
        self.assertTrue(report["checks"]["m3_v3_origin_bound"])
        self.assertFalse(drifted["ready"])
        self.assertFalse(
            drifted["unified"]["checks"]["expert_publication_consistent"]
        )
        self.assertFalse(split_drifted["ready"])
        self.assertFalse(
            split_drifted["unified"]["checks"][
                "split_indexes_exact_partition"
            ]
        )

    def test_published_cache_migration_replays_legacy_inodes(self) -> None:
        class Bank:
            shards = ("shard-00000.pt",)
            manifest: dict = {}

            def file_metadata_snapshot(self):
                return {"synthetic": True}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            cache = root / "cache"
            legacy.mkdir()
            cache.mkdir()
            (legacy / "manifest.json").write_text("{}", encoding="utf-8")
            (legacy / "validation_report.json").write_text(
                "{}", encoding="utf-8"
            )
            source = legacy / "shard-00000.pt"
            target = cache / source.name
            source.write_bytes(b"strict-reuse")
            target.hardlink_to(source)
            source_stat = source.stat()
            Bank.manifest = {
                "num_samples": 1,
                "migration": {
                    "protocol": CACHE_MIGRATION_PROTOCOL,
                    "source_cache_dir": str(legacy),
                    "source_manifest_sha256": hashlib.sha256(
                        (legacy / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "source_validation_report_sha256": hashlib.sha256(
                        (legacy / "validation_report.json").read_bytes()
                    ).hexdigest(),
                    "reuse_method": "hardlink",
                    "reused_shards": 1,
                    "reused_records": 1,
                    "reused_bytes": len(b"strict-reuse"),
                    "all_same_inode": True,
                    "hardlinks": [{
                        "source": str(source),
                        "target": target.name,
                        "device": source_stat.st_dev,
                        "source_inode": source_stat.st_ino,
                        "target_inode": source_stat.st_ino,
                        "same_inode": True,
                    }],
                },
            }
            (cache / "migration_report.json").write_text(
                __import__("json").dumps({
                    "protocol": CACHE_MIGRATION_PROTOCOL,
                    "status": "engineering-valid",
                    "output_dir": str(cache),
                    "records": 1,
                    "shards": 1,
                    "reused_bytes": len(b"strict-reuse"),
                    "published_replay": {
                        "protocol": DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL,
                        "all_verified": True,
                        "verified_shards": 1,
                        "verified_bytes": len(b"strict-reuse"),
                        "metadata_snapshot": {"synthetic": True},
                    },
                    "errors": [],
                }),
                encoding="utf-8",
            )
            accepted = revalidate_published_cache_migration(cache, Bank())
            target.unlink()
            target.write_bytes(source.read_bytes())
            detached = revalidate_published_cache_migration(cache, Bank())
        self.assertTrue(all(accepted["checks"].values()))
        self.assertFalse(detached["checks"]["hardlinks_current"])
        self.assertTrue(detached["hardlink_errors"])

    def test_published_cache_origin_accepts_native_current_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "segmentation-cache"
            source.mkdir()
            (source / "shard-00000.pt").write_bytes(b"source-shard")
            (source / "manifest.json").write_text(
                __import__("json").dumps({
                    "shards": ["shard-00000.pt"],
                }),
                encoding="utf-8",
            )

            class Bank:
                shards = ("shard-00000.pt",)
                manifest = {
                    "builder_version": DESCRIPTION_CACHE_BUILDER_VERSION,
                    "components": ["single_image", "multisource_parent"],
                    "source_cache_provenance": {
                        "provided": True,
                        "path": str(source),
                        **source_cache_snapshot(source),
                        "isolation_unchanged": True,
                    },
                }
                validation_report = {"status": "valid", "errors": []}

            audit = revalidate_published_cache_origin(
                root, Bank()
            )
            (source / "shard-00000.pt").write_bytes(b"drifted-source")
            drifted = revalidate_published_cache_origin(root, Bank())
        self.assertEqual(audit["origin"], "native_m3_v3_build")
        self.assertTrue(all(audit["checks"].values()))
        self.assertFalse(
            drifted["checks"]["segmentation_source_cache_current"]
        )


if __name__ == "__main__":
    unittest.main()
