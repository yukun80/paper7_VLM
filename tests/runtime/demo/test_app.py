from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from oa_groundrag.runtime.demo.access import DemoTestAccessController
from oa_groundrag.runtime.demo.app import (
    DEMO_MODE,
    DemoWorkbenchServices,
    create_demo_app,
    ensure_demo_loopback_proxy_bypass,
    serve_demo,
)
from oa_groundrag.runtime.demo.catalog import BenchmarkCatalog
from oa_groundrag.runtime.demo.config import load_demo_config
from oa_groundrag.runtime.demo.gallery import DemoGalleryStore
from oa_groundrag.runtime.demo.i18n import (
    EVIDENCE_HEADERS,
    GALLERY_HEADERS,
    RUN_MODE_SINGLE,
    RUN_MODE_SUITE,
    SOURCE_FILTER_ALL,
    MessageSpec,
)
from oa_groundrag.runtime.demo.runner import UnifiedDemoRunner

from tests.runtime.demo.helpers import REPO_ROOT, build_benchmark, fake_runtime


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio optional dependency is unavailable")
class DemoAppTest(unittest.TestCase):
    def test_blocks_build_with_private_callbacks_and_single_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = load_demo_config(REPO_ROOT / "configs/runtime/demo_v1.yaml")
            binding = build_benchmark(root / "benchmark")
            config = replace(
                real,
                benchmark=binding,
                demo_root=root / "demo",
                allow_test_demo=False,
            )
            access = DemoTestAccessController(
                demo_root=config.demo_root,
                allow_test_demo=False,
                benchmark_identity=binding.identity,
                config_sha256=config.config_sha256,
            )
            catalog = BenchmarkCatalog(binding, access_controller=access)
            gallery = DemoGalleryStore(config.demo_root)
            runtime, calls = fake_runtime()
            runner = UnifiedDemoRunner(config, catalog, runtime=runtime)
            services = DemoWorkbenchServices(
                config=config,
                catalog=catalog,
                gallery=gallery,
                runner=runner,
                test_access=access,
            )
            app = create_demo_app(config, services=services)
            try:
                self.assertTrue(app.enable_queue)
                dependencies = app.config["dependencies"]
                self.assertTrue(dependencies)
                self.assertTrue(all(
                    dependency.get("api_visibility") in {"private", "undocumented"}
                    for dependency in dependencies
                ))
                self.assertEqual(app._queue.default_concurrency_limit, 1)
                ui_config = json.dumps(app.config, ensure_ascii=False, default=str)
                self.assertNotIn("Frozen Evaluation", ui_config)
                self.assertIn("B. Demo Gallery", ui_config)
                self.assertIn("A. 基准数据浏览", ui_config)
                self.assertIn("C. 任务运行 / 结果查看 / 执行轨迹", ui_config)
                self.assertIn("界面语言 / Interface Language", ui_config)
                self.assertIn("模型原始输出", ui_config)
                execute = next(
                    block.fn
                    for block in app.fns.values()
                    if getattr(block.fn, "__name__", "") == "execute"
                )
                result = execute(
                    "zh",
                    {
                        "mode": DEMO_MODE,
                        "sample_id": "test-small",
                        "split": "test",
                        "source": "source_c",
                    },
                    RUN_MODE_SINGLE,
                    "KNOWLEDGE_QA",
                    [],
                    "",
                    "Why is one sensor insufficient?",
                    None,
                    None,
                    "OA_AUXSEG_CANDIDATE",
                )
                self.assertEqual(result[0]["input_scope"], "KNOWLEDGE_ONLY")
                self.assertFalse(result[0]["benchmark_payload_consumed"])
                self.assertFalse(result[0]["sealed_test_accessed"])
                self.assertFalse(any(value.startswith("spatial:") for value in calls))
                receipt_root = config.demo_root / "test_access_receipts"
                self.assertTrue(
                    not receipt_root.exists() or not any(receipt_root.iterdir())
                )
                apply_filter = next(
                    block.fn
                    for block in app.fns.values()
                    if getattr(block.fn, "__name__", "") == "apply_filter"
                )
                with self.assertRaisesRegex(Exception, "No samples match"):
                    apply_filter(
                        "en",
                        "val",
                        SOURCE_FILTER_ALL,
                        "does-not-exist",
                        "all",
                        "all",
                        [],
                    )
            finally:
                app.close()

    def test_language_switch_is_presentation_only_and_preserves_stable_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = load_demo_config(REPO_ROOT / "configs/runtime/demo_v1.yaml")
            binding = build_benchmark(root / "benchmark")
            config = replace(
                real,
                benchmark=binding,
                demo_root=root / "demo",
                allow_test_demo=False,
            )
            access = DemoTestAccessController(
                demo_root=config.demo_root,
                allow_test_demo=False,
                benchmark_identity=binding.identity,
                config_sha256=config.config_sha256,
            )
            catalog = BenchmarkCatalog(binding, access_controller=access)
            gallery = DemoGalleryStore(config.demo_root)
            runtime, _ = fake_runtime()
            runner = UnifiedDemoRunner(config, catalog, runtime=runtime)
            services = DemoWorkbenchServices(
                config=config,
                catalog=catalog,
                gallery=gallery,
                runner=runner,
                test_access=access,
            )
            app = create_demo_app(config, services=services)
            try:
                switch = next(
                    block.fn
                    for block in app.fns.values()
                    if getattr(block.fn, "__name__", "") == "switch_language"
                )
                load_dependency = next(
                    value
                    for value in app.config["dependencies"]
                    if value["targets"][0][1] == "load"
                )
                loaded_values = app.fns[load_dependency["id"]].fn("zh")
                loaded_preview_state = loaded_values[12]
                candidate_payload = {
                    "snapshot": {"snapshot_id": "dss_fixed"},
                    "options": [{
                        "kind": "CANDIDATE",
                        "token": "dss_fixed::CANDIDATE::7",
                        "candidate_id": 7,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "area_pixels": 23,
                        "confidence": 0.75,
                        "mask_path": "/tmp/mask.png",
                        "overlay_path": "/tmp/overlay.png",
                    }],
                    "gallery_tokens": ["dss_fixed::CANDIDATE::7"],
                }
                run_state = {
                    "run_id": "demo_stable",
                    "response": {"text": "unchanged raw model output"},
                    "tasks": [{"task": "SEGMENT_ONLY", "status": "SUCCESS"}],
                }
                selection = {
                    "mode": DEMO_MODE,
                    "sample_id": "val-large",
                    "split": "val",
                    "source": "source_b",
                }
                original_candidate = deepcopy(candidate_payload)
                original_run = deepcopy(run_state)
                original_selection = deepcopy(selection)
                before_runs = set((config.demo_root / "runs").glob("*")) if (
                    config.demo_root / "runs"
                ).exists() else set()
                with (
                    patch(
                        "oa_groundrag.data.oa_auxseg.dataset.BenchmarkDataset.__getitem__"
                    ) as getitem,
                    patch.object(catalog, "load") as load,
                    patch.object(runner, "run") as run,
                    patch.object(runner, "load_viewer") as load_viewer,
                    patch.object(gallery, "list_current") as list_current,
                    patch.object(access, "issue") as issue,
                ):
                    result = switch(
                        "en",
                        loaded_preview_state,
                        candidate_payload,
                        "dss_fixed::CANDIDATE::7",
                        run_state,
                        "SEGMENT_ONLY",
                        selection,
                        RUN_MODE_SUITE,
                        "VLM_ONLY",
                        ["SEGMENT_ONLY", "REGION_INTERPRETATION"],
                        "val",
                        SOURCE_FILTER_ALL,
                        "all",
                        "all",
                        ["optical"],
                        ["VLM_ONLY", "SEGMENT_ONLY"],
                        "OA_AUXSEG_CANDIDATE",
                        (MessageSpec.create("status.browser.loaded"),),
                        (),
                        (),
                        (MessageSpec.create("status.candidate.selected", candidate_id=7),),
                    )
                    load.assert_not_called()
                    getitem.assert_not_called()
                    run.assert_not_called()
                    load_viewer.assert_not_called()
                    list_current.assert_not_called()
                    issue.assert_not_called()
                self.assertEqual(result[0], "en")
                self.assertEqual(result[1], "OA-GroundRAG Unified Demo Workbench")
                updates = [value for value in result if isinstance(value, dict)]
                self.assertTrue(any(value.get("label") == "Run Mode" for value in updates))
                self.assertTrue(any(value.get("headers") == GALLERY_HEADERS["en"] for value in updates))
                self.assertTrue(any(value.get("headers") == EVIDENCE_HEADERS["en"] for value in updates))
                optical_preview_update = next(
                    value
                    for value in updates
                    if value.get("label", "").startswith("Optical / Multispectral")
                )
                self.assertTrue(any(
                    "Spatial Expert Input Preview" in caption
                    for _, caption in optical_preview_update["value"]
                ))
                candidate_update = next(
                    value
                    for value in updates
                    if value.get("value") == "dss_fixed::CANDIDATE::7"
                )
                self.assertEqual(
                    candidate_update["choices"][0][1],
                    "dss_fixed::CANDIDATE::7",
                )
                task_update = next(
                    value
                    for value in updates
                    if value.get("label") == "Single Task"
                )
                self.assertEqual(task_update["value"], "VLM_ONLY")
                self.assertIn(("Knowledge QA (KNOWLEDGE_QA)", "KNOWLEDGE_QA"), task_update["choices"])
                self.assertEqual(candidate_payload, original_candidate)
                self.assertEqual(run_state, original_run)
                self.assertEqual(selection, original_selection)
                after_runs = set((config.demo_root / "runs").glob("*")) if (
                    config.demo_root / "runs"
                ).exists() else set()
                self.assertEqual(after_runs, before_runs)

                dependency = next(
                    value
                    for value in app.config["dependencies"]
                    if value.get("api_name") == "switch_language"
                )
                self.assertFalse(dependency["queue"])
                self.assertEqual(len(result), len(dependency["outputs"]))
                execute_dependency = next(
                    value
                    for value in app.config["dependencies"]
                    if value.get("api_name") == "execute"
                )
                viewer_after_run = next(
                    value
                    for value in app.config["dependencies"]
                    if value.get("api_name") == "viewer_values"
                )
                self.assertNotIn(load_dependency["outputs"][0], dependency["outputs"])
                self.assertNotIn(execute_dependency["outputs"][0], dependency["outputs"])
                self.assertNotIn(execute_dependency["outputs"][7], dependency["outputs"])
                self.assertNotIn(execute_dependency["outputs"][8], dependency["outputs"])
                self.assertIn(viewer_after_run["outputs"][14], dependency["outputs"])
                raw_output_update = next(
                    value
                    for value in updates
                    if value.get("label") == "Raw Model Output"
                )
                self.assertNotIn("value", raw_output_update)
                gallery_header_update = next(
                    value
                    for value in updates
                    if value.get("headers") == GALLERY_HEADERS["en"]
                )
                self.assertNotIn("value", gallery_header_update)
                candidate_dependencies = [
                    value
                    for value in app.config["dependencies"]
                    if value.get("api_name") == "candidate_preview_values"
                ]
                self.assertEqual(candidate_dependencies[0]["targets"][0][1], "input")
                viewer_dependency = next(
                    value
                    for value in app.config["dependencies"]
                    if value.get("api_name") == "viewer_values_2"
                )
                self.assertEqual(viewer_dependency["targets"][0][1], "input")
            finally:
                app.close()

    def test_loopback_proxy_bypass_is_exact_and_preserves_existing_values(self) -> None:
        environ = {"NO_PROXY": "example.com", "no_proxy": ""}
        updated = ensure_demo_loopback_proxy_bypass(environ)
        self.assertIn("example.com", updated["NO_PROXY"])
        for value in ("127.0.0.1", "localhost", "::1"):
            self.assertIn(value, updated["NO_PROXY"])
            self.assertIn(value, updated["no_proxy"])

    def test_config_allows_empty_frozen_bindings(self) -> None:
        config = load_demo_config(REPO_ROOT / "configs/runtime/demo_v1.yaml")
        self.assertEqual(config.frozen_evaluations, ())

    def test_serve_is_loopback_only_and_exposes_only_demo_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = load_demo_config(REPO_ROOT / "configs/runtime/demo_v1.yaml")
            binding = build_benchmark(root / "benchmark")
            config = replace(real, benchmark=binding, demo_root=root / "demo")
            config.demo_root.mkdir(parents=True)
            app = MagicMock()

            with (
                patch(
                    "oa_groundrag.runtime.demo.app.build_demo_services",
                    return_value=MagicMock(),
                ),
                patch(
                    "oa_groundrag.runtime.demo.app.create_demo_app",
                    return_value=app,
                ),
            ):
                returned = serve_demo(
                    config,
                    port=7861,
                    prevent_thread_lock=True,
                )

            self.assertIs(returned, app)
            launch = app.launch.call_args.kwargs
            self.assertEqual(launch["server_name"], "127.0.0.1")
            self.assertEqual(launch["server_port"], 7861)
            self.assertFalse(launch["share"])
            self.assertFalse(launch["enable_monitoring"])
            self.assertFalse(launch["mcp_server"])
            self.assertEqual(launch["max_file_size"], "64mb")
            self.assertEqual(
                launch["allowed_paths"],
                [str(config.demo_root)],
            )
            self.assertIn(str(binding.root), launch["blocked_paths"])
            self.assertNotIn(str(binding.root), launch["allowed_paths"])
            self.assertNotIn(
                str(config.unified.repository_root / "models_zoo"),
                launch["allowed_paths"],
            )


if __name__ == "__main__":
    unittest.main()
