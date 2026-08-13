"""能力化目录、公共入口与导入方向的架构回归。"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from oa_groundrag.data.rs_general.config import load_build_config, load_export_config
from oa_groundrag.data.grounded.annotation.cli import TRAIN_WORKFLOW_PATHS
from oa_groundrag.retrieval.runtime_config import load_text_rag_runtime_config
from oa_groundrag.runtime.config import load_unified_config
from oa_groundrag.runtime.demo.config import load_demo_config
from oa_groundrag.segmentation.config import load_runtime_config
from oa_groundrag.training.grounding.config import load_stage5_config
from oa_groundrag.vlm.config import load_config
from oa_groundrag.vlm.grounded_runtime import load_grounded_runtime_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "oa_groundrag"
FORBIDDEN_PACKAGES = {
    "oa_groundrag.phase2",
    "oa_groundrag.phase3",
    "oa_groundrag.phase4",
    "oa_groundrag.text_rag",
    "oa_groundrag.unified",
}


def _module_paths() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        module = "oa_groundrag." + ".".join(relative.parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        modules[module] = path
    return modules


def _import_edges(modules: dict[str, Path]) -> dict[str, set[str]]:
    graph = {module: set() for module in modules}
    for module, path in modules.items():
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                graph[module].update(
                    item.name for item in node.names if item.name in modules
                )
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = (
                    f"{base}.{node.module}" if node.module else base
                ).rstrip(".")
            else:
                target = node.module or ""
            for item in node.names:
                candidate = f"{target}.{item.name}"
                if candidate in modules:
                    graph[module].add(candidate)
                elif target in modules:
                    graph[module].add(target)
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    result: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        indices[module] = lowlinks[module] = len(indices)
        stack.append(module)
        active.add(module)
        for dependency in graph[module]:
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in active:
                lowlinks[module] = min(lowlinks[module], indices[dependency])
        if lowlinks[module] != indices[module]:
            return
        component: list[str] = []
        while True:
            child = stack.pop()
            active.remove(child)
            component.append(child)
            if child == module:
                break
        if len(component) > 1:
            result.append(tuple(sorted(component)))

    for module in graph:
        if module not in indices:
            visit(module)
    return result


class ArchitectureTest(unittest.TestCase):
    def test_long_term_roots_are_capability_driven(self) -> None:
        forbidden = {"phase2", "phase3", "phase4", "text_rag", "unified"}
        for root in (PACKAGE_ROOT, REPOSITORY_ROOT / "configs", REPOSITORY_ROOT / "scripts", REPOSITORY_ROOT / "tests"):
            self.assertFalse(
                forbidden & {path.name for path in root.iterdir() if path.is_dir()},
                msg=str(root),
            )

    def test_production_imports_neither_legacy_packages_nor_scripts(self) -> None:
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(item.name for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            for name in imports:
                self.assertFalse(name == "scripts" or name.startswith("scripts."), msg=str(path))
                self.assertFalse(
                    any(name == old or name.startswith(f"{old}.") for old in FORBIDDEN_PACKAGES),
                    msg=f"{path}: {name}",
                )

    def test_package_import_graph_is_acyclic(self) -> None:
        modules = _module_paths()
        self.assertEqual(_cycles(_import_edges(modules)), [])

    def test_segmentation_engine_contains_only_training_runtime(self) -> None:
        path = PACKAGE_ROOT / "training/segmentation/engine.py"
        definitions = {
            node.name
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertFalse(
            definitions
            & {
                "evaluate_model",
                "finalize_training_run",
                "run_evaluation",
                "run_inference",
                "run_smoke",
            }
        )

    def test_public_capability_imports_load(self) -> None:
        for name in (
            "oa_groundrag.segmentation",
            "oa_groundrag.vlm",
            "oa_groundrag.grounding",
            "oa_groundrag.retrieval",
            "oa_groundrag.runtime",
        ):
            self.assertIsNotNone(importlib.import_module(name))

    def test_active_configs_parse_at_new_locations(self) -> None:
        configs = REPOSITORY_ROOT / "configs"
        load_runtime_config(configs / "segmentation/full_proposed_dropout_b16_nockpt_e100.json")
        load_build_config(configs / "data/rs_general/full.yaml")
        load_export_config(configs / "data/rs_general/qwen_train.yaml")
        load_config(configs / "vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml")
        load_stage5_config(configs / "vlm/grounded/train_v2.yaml")
        grounded = load_grounded_runtime_config(configs / "vlm/grounded/runtime_v1.yaml")
        retrieval = load_text_rag_runtime_config(configs / "retrieval/runtime_v1.yaml")
        bank_manifest = json.loads(
            (
                REPOSITORY_ROOT
                / "outputs/stage6_text_rag/text_evidence_bank_v1/manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            retrieval.bank.build_config_semantic_sha256,
            bank_manifest["config_semantic_sha256"],
        )
        self.assertEqual(
            retrieval.source_registry_path,
            configs / "retrieval/sources_v1.yaml",
        )
        self.assertEqual(
            grounded.published_training_config_semantic_sha256,
            "8ea33e0e058a75a9d27ce248684bd6734fc10e3d09ffe22db16cb5b904ca943d",
        )
        load_unified_config(configs / "runtime/inference_v2.yaml")
        load_demo_config(configs / "runtime/demo_v1.yaml")

    def test_fixed_workflow_config_references_exist(self) -> None:
        for path in (
            TRAIN_WORKFLOW_PATHS.prompt_path,
            TRAIN_WORKFLOW_PATHS.draft_config_path,
            REPOSITORY_ROOT / "configs/vlm/grounded/train_v2.yaml",
            REPOSITORY_ROOT / "configs/vlm/grounded/runtime_v1.yaml",
            REPOSITORY_ROOT / "configs/retrieval/runtime_v1.yaml",
        ):
            self.assertTrue(path.is_file(), msg=str(path))

    def test_all_task_scripts_expose_help(self) -> None:
        for path in sorted((REPOSITORY_ROOT / "scripts").glob("*/*.py")):
            process = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, msg=f"{path}: {process.stderr}")


if __name__ == "__main__":
    unittest.main()
