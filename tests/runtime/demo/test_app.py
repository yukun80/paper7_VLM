from __future__ import annotations

from dataclasses import replace
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from oa_groundrag.runtime.demo.access import DemoTestAccessController
from oa_groundrag.runtime.demo.app import (
    DemoWorkbenchServices,
    create_demo_app,
    ensure_demo_loopback_proxy_bypass,
    serve_demo,
)
from oa_groundrag.runtime.demo.catalog import BenchmarkCatalog, FrozenEvaluationCatalog
from oa_groundrag.runtime.demo.config import load_demo_config
from oa_groundrag.runtime.demo.gallery import DemoGalleryStore
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
            frozen = {
                item.name: FrozenEvaluationCatalog(item, verify_payloads=False)
                for item in config.frozen_evaluations
            }
            gallery = DemoGalleryStore(config.demo_root)
            runtime, _calls = fake_runtime()
            runner = UnifiedDemoRunner(config, catalog, runtime=runtime)
            services = DemoWorkbenchServices(
                config=config,
                catalog=catalog,
                frozen=frozen,
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
            finally:
                app.close()

    def test_loopback_proxy_bypass_is_exact_and_preserves_existing_values(self) -> None:
        environ = {"NO_PROXY": "example.com", "no_proxy": ""}
        updated = ensure_demo_loopback_proxy_bypass(environ)
        self.assertIn("example.com", updated["NO_PROXY"])
        for value in ("127.0.0.1", "localhost", "::1"):
            self.assertIn(value, updated["NO_PROXY"])
            self.assertIn(value, updated["no_proxy"])

    def test_serve_is_loopback_only_and_exposes_only_demo_and_frozen_roots(self) -> None:
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
                [
                    str(config.demo_root),
                    *(str(item.root) for item in config.frozen_evaluations),
                ],
            )
            self.assertIn(str(binding.root), launch["blocked_paths"])
            self.assertNotIn(str(binding.root), launch["allowed_paths"])
            self.assertNotIn(
                str(config.unified.repository_root / "models_zoo"),
                launch["allowed_paths"],
            )


if __name__ == "__main__":
    unittest.main()
