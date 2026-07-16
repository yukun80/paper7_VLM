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
    BRIDGE_BUILDER_VERSION,
    BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
    BRIDGE_EXPERT_REPLAY_PROTOCOL,
    DESCRIPTION_BUILDER_VERSION,
    INDEX_SCHEMA,
    SEGMENTATION_INSTRUCTION_REPORT,
    TASK_INDEX_NAMES,
    bridge_publication_policy,
    component_validation_contract,
    segmentation_instruction_validation_contract,
    sha256_file,
)


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SEGDESC_SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = load_script("qpsalm_segdesc_validate", "5-2_validate_unified_index.py")
BUILD = load_script("qpsalm_segdesc_build", "5-1_build_unified_index.py")


class SegDescUnifiedIndexProtocolTest(unittest.TestCase):
    def test_component_index_is_hashed_once_per_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "instruction_train.jsonl"
            index.write_text(
                "".join(
                    json.dumps({
                        "sample_id": f"sample_{number}",
                        "split": "train",
                        "task_family": "landslide_segmentation",
                    }) + "\n"
                    for number in range(2)
                ),
                encoding="utf-8",
            )
            calls = 0

            def counted_hash(path: Path) -> str:
                nonlocal calls
                calls += 1
                return sha256_file(path)

            original = BUILD.sha256_file
            BUILD.sha256_file = counted_hash
            try:
                rows: list[dict] = []
                BUILD._append(
                    rows, index, "landslide_segmentation_v2", lambda _row: "segmentation"
                )
            finally:
                BUILD.sha256_file = original
            self.assertEqual(calls, 1)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["component_index_sha256"] for row in rows},
                {sha256_file(index)},
            )

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

    def _write_bound_frozen_bridge(self, bridge: Path) -> dict:
        (bridge / "indexes").mkdir(parents=True, exist_ok=True)
        (bridge / "manifests").mkdir(exist_ok=True)
        (bridge / "reports").mkdir(exist_ok=True)
        source_dir = bridge / "review_sources"
        source_dir.mkdir(exist_ok=True)

        def write_jsonl(path: Path, rows: list[dict]) -> None:
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

        def artifact(path: Path, records: int | None = None) -> dict:
            result = {
                "path": str(path.resolve(strict=False)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            if records is not None:
                result["records"] = records
            return result

        rows = [
            {
                "bridge_record_id": f"expert-{split}",
                "parent_sample_id": f"parent-{split}",
                "split": split,
            }
            for split in ("test", "train", "val")
        ]
        write_jsonl(bridge / "manifests/review_selection.jsonl", [
            {
                "review_item_id": f"review-{index}",
                "bridge_record_id": row["bridge_record_id"],
            }
            for index, row in enumerate(rows)
        ])
        candidate_path = bridge / "indexes/candidate_all.jsonl"
        write_jsonl(candidate_path, rows)
        expert_all = bridge / "indexes/expert_all.jsonl"
        pending = bridge / "indexes/pending_arbitration.jsonl"
        write_jsonl(expert_all, rows)
        write_jsonl(pending, [])
        split_paths = {}
        for split in ("train", "val", "test"):
            path = bridge / f"indexes/expert_{split}.jsonl"
            write_jsonl(path, [row for row in rows if row["split"] == split])
            split_paths[split] = path
        reviewer_paths = {}
        for reviewer in ("reviewer_1", "reviewer_2"):
            path = source_dir / f"{reviewer}.jsonl"
            write_jsonl(path, [
                {
                    "review_item_id": f"review-{index}",
                    "reviewer_id": reviewer,
                    "decision": "accept",
                }
                for index in range(len(rows))
            ])
            reviewer_paths[reviewer] = path
        gate_source = source_dir / "evaluation_gate_frozen.json"
        self._write_json(gate_source, {
            "protocol": "landslide_bridge_evaluation_gate_v2",
            "builder_version": BRIDGE_BUILDER_VERSION,
            "frozen": True,
            "status": "frozen_after_pilot",
        })
        gate_output = bridge / "manifests/evaluation_gate_manifest.json"
        gate = json.loads(gate_source.read_text(encoding="utf-8"))
        gate["source_file"] = str(gate_source.resolve(strict=False))
        self._write_json(gate_output, gate)
        merge_binding = {
            "protocol": BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
            "builder_version": BRIDGE_BUILDER_VERSION,
            "sources": {
                "reviewer_1": artifact(reviewer_paths["reviewer_1"], len(rows)),
                "reviewer_2": artifact(reviewer_paths["reviewer_2"], len(rows)),
                "arbitration": None,
                "evaluation_gate_source": artifact(gate_source),
            },
            "outputs": {
                "expert_all": artifact(expert_all, len(rows)),
                "expert_train": artifact(split_paths["train"], 1),
                "expert_val": artifact(split_paths["val"], 1),
                "expert_test": artifact(split_paths["test"], 1),
                "pending_arbitration": artifact(pending, 0),
                "evaluation_gate": artifact(gate_output),
            },
        }
        review_report = bridge / "reports/expert_review_report.json"
        self._write_json(review_report, {
            "builder_version": BRIDGE_BUILDER_VERSION,
            "status": "complete",
            "frozen_evaluation_gate": True,
            "errors": [],
            "review_items": len(rows),
            "expert_records": len(rows),
            "pending_arbitration": 0,
            "final_decisions": {"accept": len(rows)},
            "expert_artifact_binding": merge_binding,
        })
        return {
            "protocol": BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
            "builder_version": BRIDGE_BUILDER_VERSION,
            "review_report": artifact(review_report),
            "merge_artifacts": merge_binding,
            "semantic_replay": {
                "protocol": BRIDGE_EXPERT_REPLAY_PROTOCOL,
                "candidate_index": artifact(candidate_path, len(rows)),
                "review_selection": artifact(
                    bridge / "manifests/review_selection.jsonl", len(rows)
                ),
                "review_items": len(rows),
                "disputed_review_items": 0,
                "expert_records": len(rows),
                "pending_arbitration": 0,
                "final_decisions": {"accept": len(rows)},
                "review_report_statistics_verified": True,
            },
        }

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
            contract = component_validation_contract(
                name, mode="small", root=component_root,
            )
            self._write_json(
                report_path,
                {"status": status, "errors": [], **contract},
            )
            reports[name] = {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "status": status,
                "errors": 0,
                "contract": contract,
            }
        instruction_path = roots["segmentation"] / SEGMENTATION_INSTRUCTION_REPORT
        instruction_contract = segmentation_instruction_validation_contract(
            roots["segmentation"]
        )
        instruction_report = {
            "errors": [],
            "parent_split_isolation": {"num_parents": 2, "num_leaking": 0},
            **{
                key: value
                for key, value in instruction_contract.items()
                if "." not in key
            },
        }
        self._write_json(instruction_path, instruction_report)
        reports["segmentation"]["instruction_validation"] = {
            "path": str(instruction_path),
            "sha256": sha256_file(instruction_path),
            "errors": 0,
            "contract": instruction_contract,
        }
        expert_path = roots["bridge"] / "indexes/expert_all.jsonl"
        gate_path = roots["bridge"] / "manifests/evaluation_gate_manifest.json"
        if bridge_status == BRIDGE_FROZEN_STATUS:
            expert_binding = self._write_bound_frozen_bridge(roots["bridge"])
            bridge_report_path = roots["bridge"] / "reports/validation_report.json"
            bridge_report = json.loads(bridge_report_path.read_text(encoding="utf-8"))
            bridge_report["expert_artifact_binding"] = expert_binding
            self._write_json(bridge_report_path, bridge_report)
            reports["bridge"]["sha256"] = sha256_file(bridge_report_path)
        elif stale:
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

    def test_build_rejects_stale_bridge_component_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory) / "bridge"
            report_path = bridge / "reports/validation_report.json"
            self._write_json(report_path, {
                "builder_version": "landslide_bridge_m2_v4_parent_schema_adapter",
                "mode": "small",
                "status": BRIDGE_AWAITING_STATUS,
                "errors": [],
            })
            with self.assertRaisesRegex(ValueError, "contract 过期"):
                BUILD._validated(bridge, component="bridge", mode="small")

    def test_validator_rechecks_component_contract_from_actual_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, BRIDGE_AWAITING_STATUS, stale=False)
            bridge_report_path = root / "bridge/reports/validation_report.json"
            stale_report = json.loads(bridge_report_path.read_text(encoding="utf-8"))
            stale_report["builder_version"] = "landslide_bridge_m2_v4_parent_schema_adapter"
            self._write_json(bridge_report_path, stale_report)
            manifest["component_validation_reports"]["bridge"]["sha256"] = sha256_file(
                bridge_report_path
            )
            errors: list[str] = []
            VALIDATE._validate_publication_manifest(
                root / "unified", manifest, errors, [], expected_mode="small"
            )
            self.assertTrue(any("contract 过期: bridge" in value for value in errors))

    def test_current_component_versions_are_explicit(self) -> None:
        self.assertEqual(
            DESCRIPTION_BUILDER_VERSION,
            "description_benchmark_m1_v4_answer_trace",
        )
        self.assertEqual(
            BRIDGE_BUILDER_VERSION,
            "landslide_bridge_m2_v7_expert_review_replay_bound",
        )

    def test_validator_binds_segmentation_instruction_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, BRIDGE_AWAITING_STATUS, stale=False)
            instruction_path = root / "segmentation" / SEGMENTATION_INSTRUCTION_REPORT
            instruction_report = json.loads(
                instruction_path.read_text(encoding="utf-8")
            )
            instruction_report["parent_split_isolation"]["num_leaking"] = 1
            self._write_json(instruction_path, instruction_report)
            manifest["component_validation_reports"]["segmentation"][
                "instruction_validation"
            ]["sha256"] = sha256_file(instruction_path)
            errors: list[str] = []
            VALIDATE._validate_publication_manifest(
                root / "unified", manifest, errors, [], expected_mode="small"
            )
            self.assertTrue(any(
                "segmentation instruction validation contract 过期" in value
                for value in errors
            ))


if __name__ == "__main__":
    unittest.main()
