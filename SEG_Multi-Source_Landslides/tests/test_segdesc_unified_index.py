#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SegDesc 统一索引发布门禁的合成协议测试。

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest
SEG_Multi-Source_Landslides/tests/test_segdesc_unified_index.py -v
写入行为：仅写临时目录，不读取或修改真实 benchmark。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGDESC_SCRIPTS = REPO_ROOT / "scripts/5-segdesc"
sys.path.insert(0, str(SEGDESC_SCRIPTS))

from segdesc_common import (  # noqa: E402
    BRIDGE_AWAITING_STATUS,
    BRIDGE_FROZEN_STATUS,
    BUILDER_VERSION,
    INDEX_SCHEMA,
    TASK_INDEX_NAMES,
    bridge_publication_policy,
    sha256_file,
)


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SEGDESC_SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = load_script("qpsalm_segdesc_validate", "5-2_validate_unified_index.py")


class SegDescUnifiedIndexProtocolTest(unittest.TestCase):
    def test_task_groups_bind_to_explicit_component_indexes(self) -> None:
        self.assertEqual(TASK_INDEX_NAMES["region_description_auto"], {"auto_train.jsonl"})
        self.assertEqual(TASK_INDEX_NAMES["region_description_expert"], {"expert_all.jsonl"})
        self.assertNotIn("expert_all.jsonl", TASK_INDEX_NAMES["region_description_auto"])

    def test_bridge_publication_policy_never_infers_expert_from_stale_file(self) -> None:
        awaiting = bridge_publication_policy(
            BRIDGE_AWAITING_STATUS,
            expert_index_present=True,
            gate_present=True,
        )
        self.assertFalse(awaiting["expert_index_published"])
        self.assertFalse(awaiting["bridge_gate_published"])
        self.assertTrue(awaiting["stale_expert_index_ignored"])
        self.assertTrue(awaiting["stale_bridge_gate_ignored"])

        with self.assertRaisesRegex(ValueError, "expert_all"):
            bridge_publication_policy(
                BRIDGE_FROZEN_STATUS,
                expert_index_present=False,
                gate_present=True,
            )
        frozen = bridge_publication_policy(
            BRIDGE_FROZEN_STATUS,
            expert_index_present=True,
            gate_present=True,
        )
        self.assertTrue(frozen["expert_index_published"])
        self.assertTrue(frozen["bridge_gate_published"])

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _manifest(self, root: Path, bridge_status: str, *, stale: bool) -> dict:
        roots = {
            "segmentation": root / "segmentation",
            "description": root / "description",
            "bridge": root / "bridge",
        }
        reports = {}
        for name, component_root in roots.items():
            status = bridge_status if name == "bridge" else "complete"
            report_path = component_root / "reports/validation_report.json"
            self._write_json(report_path, {"status": status, "errors": []})
            reports[name] = {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "status": status,
                "errors": 0,
            }
        expert_path = roots["bridge"] / "indexes/expert_all.jsonl"
        gate_path = roots["bridge"] / "manifests/evaluation_gate_manifest.json"
        if stale or bridge_status == BRIDGE_FROZEN_STATUS:
            expert_path.parent.mkdir(parents=True, exist_ok=True)
            expert_path.write_text("{}\n", encoding="utf-8")
            self._write_json(gate_path, {
                "protocol": "landslide_bridge_evaluation_gate_v2",
                "frozen": True,
                "status": "frozen_after_pilot",
            })
        frozen = bridge_status == BRIDGE_FROZEN_STATUS
        return {
            "builder_version": BUILDER_VERSION,
            "schema_version": INDEX_SCHEMA,
            "mode": "small",
            "storage_mode": "component_references_only",
            "components": {name: str(path) for name, path in roots.items()},
            "component_validation_reports": reports,
            "bridge_status": bridge_status,
            "bridge_gate": (
                {"path": str(gate_path), "sha256": sha256_file(gate_path)}
                if frozen else None
            ),
            "expert_index_present": stale or frozen,
            "expert_index_published": frozen,
            "stale_expert_index_ignored": stale and not frozen,
            "stale_bridge_gate_ignored": stale and not frozen,
        }

    def test_awaiting_review_ignores_stale_expert_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, BRIDGE_AWAITING_STATUS, stale=True)
            errors: list[str] = []
            warnings: list[str] = []
            _roots, status = VALIDATE._validate_publication_manifest(
                root / "unified", manifest, errors, warnings, expected_mode="small"
            )
            self.assertEqual(status, BRIDGE_AWAITING_STATUS)
            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 2)

            manifest["expert_index_published"] = True
            errors = []
            VALIDATE._validate_publication_manifest(
                root / "unified", manifest, errors, [], expected_mode="small"
            )
            self.assertTrue(any("publication policy" in value for value in errors))
            self.assertTrue(any("不得发布 expert" in value for value in errors))

    def test_frozen_bridge_binds_expert_gate_to_current_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, BRIDGE_FROZEN_STATUS, stale=False)
            errors: list[str] = []
            VALIDATE._validate_publication_manifest(
                root / "unified", manifest, errors, [], expected_mode="small"
            )
            self.assertEqual(errors, [])

            foreign_gate = root / "foreign/evaluation_gate_manifest.json"
            self._write_json(foreign_gate, {
                "protocol": "landslide_bridge_evaluation_gate_v2",
                "frozen": True,
                "status": "frozen_after_pilot",
            })
            manifest["bridge_gate"] = {
                "path": str(foreign_gate),
                "sha256": sha256_file(foreign_gate),
            }
            errors = []
            VALIDATE._validate_publication_manifest(
                root / "unified", manifest, errors, [], expected_mode="small"
            )
            self.assertTrue(any("gate 路径越出" in value for value in errors))


if __name__ == "__main__":
    unittest.main()
