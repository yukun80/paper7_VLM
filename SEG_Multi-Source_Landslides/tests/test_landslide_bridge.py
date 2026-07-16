#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Landslide Bridge M2 协议与合成事实测试。

推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest SEG_Multi-Source_Landslides/tests/test_landslide_bridge.py -v
写入行为：只在临时目录创建合成 npy，不修改 benchmark、datasets 或 outputs。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPTS = REPO_ROOT / "scripts/4-landslide-bridge"
sys.path.insert(0, str(BRIDGE_SCRIPTS))

from landslide_bridge_common import (  # noqa: E402
    BUILDER_VERSION,
    EVALUATION_GATE_THRESHOLDS,
    EXPERT_ARTIFACT_BINDING_PROTOCOL,
    EXPERT_REVIEW_REPLAY_PROTOCOL,
    bridge_parent_from_landslide_v2,
    cohen_kappa,
    connected_components,
    disputed_review_item_ids,
    evaluation_gate_scientific_template,
    expert_modification_statistics,
    expert_review_report_statistics,
    file_artifact_binding,
    geometry_from_mask,
    krippendorff_alpha_nominal,
    levenshtein_distance,
    load_config,
    replay_expert_review_merge,
    review_revisions_match,
    sha256_file,
    to_project_ref,
    validate_bridge_structured_target,
    validate_arbitration_usage,
    write_json,
    write_jsonl,
)


def frozen_gate_payload(*, threshold_overrides: dict | None = None) -> dict:
    thresholds = {key: 0.5 for key in EVALUATION_GATE_THRESHOLDS}
    thresholds.update(threshold_overrides or {})
    scientific = evaluation_gate_scientific_template()
    scientific["counterfactual_minimum_effective_parents"] = {
        key: 2 for key in scientific["counterfactual_minimum_effective_parents"]
    }
    return {
        "protocol": "landslide_bridge_evaluation_gate_v2",
        "builder_version": MERGE.BUILDER_VERSION,
        "status": "frozen_after_pilot",
        "frozen": True,
        "thresholds": thresholds,
        "scientific_protocol": scientific,
    }


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BRIDGE_SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INVENTORY = load_script("qpsalm_bridge_inventory", "4-1_inventory_regions.py")
FACTS = load_script("qpsalm_bridge_facts", "4-2_extract_region_facts.py")
CANDIDATES = load_script("qpsalm_bridge_candidates", "4-3_build_candidate_descriptions.py")
MERGE = load_script("qpsalm_bridge_merge", "4-5_merge_expert_reviews.py")
VALIDATOR = load_script("qpsalm_bridge_validator", "4-6_validate_landslide_bridge.py")


class LandslideBridgeProtocolTest(unittest.TestCase):
    @staticmethod
    def _valid_structured_target(status: str = "present") -> dict:
        if status == "absent":
            return {
                "target_status": "absent",
                "region": {
                    "location": "unavailable", "size_class": "unavailable",
                    "shape": "unavailable", "elongation": "unavailable",
                    "compactness": "unavailable", "fragmentation": "unavailable",
                },
                "evidence": {
                    "surface_observation": "unavailable",
                    "terrain_support": "unavailable",
                    "sar_support": "insufficient_evidence",
                    "deformation_support": "unavailable",
                    "surrounding_context": "unavailable",
                    "evidence_sufficiency": "insufficient",
                },
            }
        return {
            "target_status": status,
            "region": {
                "location": "center", "size_class": "small", "shape": "irregular",
                "elongation": "moderate", "compactness": "moderate",
                "fragmentation": "single",
            },
            "evidence": {
                "surface_observation": "A visible surface anomaly is present.",
                "terrain_support": "insufficient_evidence",
                "sar_support": "unavailable",
                "deformation_support": "unavailable",
                "surrounding_context": "Context is available.",
                "evidence_sufficiency": "partial",
            },
        }

    def test_expert_summary_and_claim_modification_statistics(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        candidate = self._valid_structured_target("present")
        accepted = {
            "candidate": {
                "structured_output": candidate,
                "summary": "Original summary.",
            },
            "expert_target": {
                "structured_output": candidate,
                "summary": "Original summary.",
            },
            "review": {"final_decision": "accept"},
        }
        revised_target = json.loads(json.dumps(candidate))
        revised_target["region"]["shape"] = "elongated"
        revised_target["evidence"]["surface_observation"] = "Revised evidence."
        revised = {
            "candidate": {
                "structured_output": candidate,
                "summary": "Original summary.",
            },
            "expert_target": {
                "structured_output": revised_target,
                "summary": "Revised summary.",
            },
            "review": {"final_decision": "revise"},
        }
        report = expert_modification_statistics([accepted, revised])
        self.assertEqual(report["num_expert_records"], 2)
        self.assertEqual(report["structured_claim_fields_changed"], 2)
        self.assertGreater(report["factual_claim_modification_rate"], 0.0)
        self.assertGreater(report["summary_mean_edit_distance_characters"], 0.0)
        self.assertEqual(
            report["expert_records_by_final_decision"],
            {"accept": 1, "revise": 1},
        )

    def test_schema_and_config_parse(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "configs/qpsalm_landslide_region_description_v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(load_config()["version"], "landslide_bridge_v1")

    @staticmethod
    def _landslide_v2_parent() -> dict:
        return {
            "schema_version": "multisource_landslide_schema_v2",
            "sample_id": "parent_001",
            "source_level": "patch",
            "supervision": "mask",
            "split": "train",
            "dataset_name": "synthetic",
            "mask": {"path": "benchmark/example/mask.npy"},
            "modalities": {
                "optical_rgb": {
                    "available": True,
                    "path": "benchmark/example/optical.npy",
                }
            },
            "spatial": {"original_size": [16, 16]},
        }

    def test_landslide_v2_parent_is_adapted_without_mutating_source(self) -> None:
        source = self._landslide_v2_parent()
        parent = bridge_parent_from_landslide_v2(source)
        self.assertEqual(parent["parent_sample_id"], source["sample_id"])
        self.assertNotIn("parent_sample_id", source)
        parent["modalities"]["optical_rgb"]["available"] = False
        self.assertTrue(source["modalities"]["optical_rgb"]["available"])

    def test_landslide_v2_parent_rejects_conflicting_or_missing_identity(self) -> None:
        conflict = self._landslide_v2_parent()
        conflict["parent_sample_id"] = "different_parent"
        with self.assertRaisesRegex(ValueError, "冲突"):
            bridge_parent_from_landslide_v2(conflict)
        missing = self._landslide_v2_parent()
        missing.pop("sample_id")
        with self.assertRaisesRegex(ValueError, "sample_id"):
            bridge_parent_from_landslide_v2(missing)

    def test_eight_connected_components_and_area_filter(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1, 1] = 1
        mask[2, 2] = 1
        mask[6, 6] = 1
        components = connected_components(mask, np.ones_like(mask), min_pixels=2, min_fraction=0.0)
        self.assertEqual(len(components), 1)
        self.assertEqual(int(components[0].sum()), 2)

    def test_absent_geometry_is_explicitly_unavailable(self) -> None:
        geometry = geometry_from_mask(None, np.ones((6, 7), dtype=np.uint8))
        self.assertEqual(geometry["area_pixels"], 0)
        self.assertEqual(geometry["location"], "unavailable")
        self.assertIsNone(geometry["bbox_xyxy_pixel_half_open"])

    def _evidence_item(
        self, root: Path, values: np.ndarray, valid: np.ndarray, *,
        family: str, units: str, normalization: str,
    ) -> dict:
        value_path = root / f"{family}_values.npy"
        valid_path = root / f"{family}_valid.npy"
        np.save(value_path, values.astype(np.float32))
        np.save(valid_path, valid.astype(np.uint8))
        return {
            "path": str(value_path), "available": True, "family": family,
            "sensor": "synthetic", "product_type": "synthetic", "band_names": ["band_0"],
            "units": units, "normalization": {"method": normalization},
            "valid_mask": {"path": str(valid_path)},
        }

    def test_evidence_levels_a_b_and_c(self) -> None:
        config = load_config()
        region = np.zeros((16, 16), dtype=np.uint8)
        region[5:10, 5:10] = 1
        values = np.linspace(0, 10, 256, dtype=np.float32).reshape(1, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = self._evidence_item(
                root, values, np.ones((16, 16)), family="terrain", units="m",
                normalization="preserve_physical_values",
            )
            relative = self._evidence_item(
                root, values, np.ones((16, 16)), family="optical", units="digital_number",
                normalization="preserve_rgb_values",
            )
            unavailable = self._evidence_item(
                root, values, np.zeros((16, 16)), family="sar", units="normalized",
                normalization="linear_clip_scale",
            )
            self.assertEqual(FACTS.modality_evidence(physical, region, config)["evidence_level"], "A_physical")
            self.assertEqual(FACTS.modality_evidence(relative, region, config)["evidence_level"], "B_normalized_relative")
            self.assertEqual(FACTS.modality_evidence(unavailable, region, config)["evidence_level"], "C_unavailable")

    def test_pilot_quota_rescaling_preserves_total(self) -> None:
        quotas = INVENTORY._split_quotas(30, load_config())
        self.assertEqual(sum(quotas.values()), 30)
        self.assertEqual(set(quotas), {"train", "val", "test"})

    def test_candidate_never_claims_expert_truth(self) -> None:
        record = {
            "target_status": "absent",
            "structured_targets": {"target_status": "absent", "region": {}, "evidence": {}},
            "candidate": {}, "provenance": {},
        }
        candidate = CANDIDATES.build_candidate(record)
        self.assertFalse(candidate["candidate"]["is_expert_truth"])
        self.assertIn("absent", candidate["candidate"]["summary"].casefold())

    def test_revision_requires_exact_double_review_agreement(self) -> None:
        left = {
            "decision": "revise", "corrected_structured_targets": {"target_status": "present"},
            "revised_summary": "A reviewed summary.",
        }
        self.assertTrue(review_revisions_match(left, dict(left)))
        changed = dict(left, revised_summary="A different summary.")
        self.assertFalse(review_revisions_match(left, changed))

    def test_review_identity_and_arbitration_usage_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.jsonl"
            path.write_text(json.dumps({
                "review_item_id": "item-1",
                "reviewer_id": "",
                "decision": "accept",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewer_id 不能为空"):
                MERGE.read_review_file(str(path), "reviewer_1")

        left = {
            "item-1": {
                "reviewer_id": "reviewer_1", "decision": "accept",
            }
        }
        right = {
            "item-1": {
                "reviewer_id": "reviewer_2", "decision": "accept",
            }
        }
        arbitration = {
            "item-1": {"reviewer_id": "arbitrator", "decision": "accept"}
        }
        disputed = disputed_review_item_ids(left, right)
        self.assertEqual(disputed, set())
        with self.assertRaisesRegex(ValueError, "未使用记录"):
            validate_arbitration_usage(
                arbitration,
                selected_ids={"item-1"},
                disputed_ids=disputed,
                reviewer_ids={"reviewer_1", "reviewer_2"},
            )
        right["item-1"]["decision"] = "reject"
        disputed = disputed_review_item_ids(left, right)
        validate_arbitration_usage(
            arbitration,
            selected_ids={"item-1"},
            disputed_ids=disputed,
            reviewer_ids={"reviewer_1", "reviewer_2"},
        )
        arbitration["item-1"]["reviewer_id"] = "reviewer_1"
        with self.assertRaisesRegex(ValueError, "仲裁者必须独立"):
            validate_arbitration_usage(
                arbitration,
                selected_ids={"item-1"},
                disputed_ids=disputed,
                reviewer_ids={"reviewer_1", "reviewer_2"},
            )

    def test_expert_review_replay_covers_decisions_arbitration_and_pending(self) -> None:
        decisions = ("accept", "revise", "reject", "arbitrated", "pending")
        candidates = [
            {
                "bridge_record_id": f"record-{decision}",
                "parent_sample_id": f"parent-{decision}",
                "split": "train" if decision in {"accept", "revise"} else "val",
                "target_status": "present",
                "candidate": {
                    "structured_output": self._valid_structured_target("present"),
                    "summary": f"Candidate {decision}.",
                },
                "provenance": {"candidate_builder": BUILDER_VERSION},
            }
            for decision in decisions
        ]
        selection = [
            {
                "review_item_id": f"item-{decision}",
                "bridge_record_id": f"record-{decision}",
                "parent_sample_id": f"parent-{decision}",
                "split": "train" if decision in {"accept", "revise"} else "val",
            }
            for decision in decisions
        ]
        revised_target = self._valid_structured_target("present")
        revised_target["region"]["shape"] = "elongated"

        def response(reviewer: str, decision: str, item: str) -> dict:
            row = {
                "review_item_id": f"item-{item}",
                "reviewer_id": reviewer,
                "decision": decision,
            }
            if decision == "revise":
                row["corrected_structured_targets"] = revised_target
                row["revised_summary"] = "Exactly revised by both reviewers."
            return row

        left = {
            "item-accept": response("reviewer_1", "accept", "accept"),
            "item-revise": response("reviewer_1", "revise", "revise"),
            "item-reject": response("reviewer_1", "reject", "reject"),
            "item-arbitrated": response("reviewer_1", "accept", "arbitrated"),
            "item-pending": response("reviewer_1", "accept", "pending"),
        }
        right = {
            "item-accept": response("reviewer_2", "accept", "accept"),
            "item-revise": response("reviewer_2", "revise", "revise"),
            "item-reject": response("reviewer_2", "reject", "reject"),
            "item-arbitrated": response("reviewer_2", "reject", "arbitrated"),
            "item-pending": response("reviewer_2", "reject", "pending"),
        }
        replay = replay_expert_review_merge(
            candidates=candidates,
            selection=selection,
            reviewer_1=left,
            reviewer_2=right,
            arbitration={
                "item-arbitrated": response(
                    "independent_arbitrator", "accept", "arbitrated"
                )
            },
        )
        by_id = {row["bridge_record_id"]: row for row in replay["expert"]}
        self.assertEqual(
            set(by_id),
            {"record-accept", "record-revise", "record-arbitrated"},
        )
        self.assertEqual(
            by_id["record-revise"]["expert_target"]["summary"],
            "Exactly revised by both reviewers.",
        )
        self.assertEqual(
            by_id["record-arbitrated"]["review"]["status"], "arbitrated"
        )
        self.assertEqual(
            replay["final_decisions"],
            {"accept": 2, "reject": 1, "revise": 1},
        )
        self.assertEqual(
            [row["review_item_id"] for row in replay["pending"]],
            ["item-pending"],
        )

    def test_validator_replays_review_sources_and_exact_expert_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("indexes", "manifests", "reports", "review_sources"):
                (root / name).mkdir(parents=True, exist_ok=True)
            candidates = [
                {
                    "bridge_record_id": f"expert-{split}",
                    "parent_sample_id": f"parent-{split}",
                    "split": split,
                    "target_status": "present",
                    "candidate": {
                        "structured_output": self._valid_structured_target("present"),
                        "summary": "Reviewed target.",
                    },
                    "provenance": {"candidate_builder": BUILDER_VERSION},
                }
                for split in ("test", "train", "val")
            ]
            selection = [
                {
                    "review_item_id": f"review-{row['bridge_record_id']}",
                    "bridge_record_id": row["bridge_record_id"],
                }
                for row in candidates
            ]
            pilot = root / "manifests/pilot_parent_manifest.jsonl"
            selection_path = root / "manifests/review_selection.jsonl"
            candidate = root / "indexes/candidate_all.jsonl"
            write_jsonl(pilot, [{"parent_sample_id": "pilot"}])
            write_jsonl(selection_path, selection)
            write_jsonl(candidate, candidates)

            reviewer_paths = {}
            for reviewer in ("reviewer_1", "reviewer_2"):
                path = root / f"review_sources/{reviewer}.jsonl"
                write_jsonl(path, [
                    {
                        **item,
                        "reviewer_id": reviewer,
                        "decision": "accept",
                    }
                    for item in selection
                ])
                reviewer_paths[reviewer] = path
            left = MERGE._unique_by_item(
                MERGE.read_review_file(
                    str(reviewer_paths["reviewer_1"]), "reviewer_1"
                ),
                "reviewer_1",
            )
            right = MERGE._unique_by_item(
                MERGE.read_review_file(
                    str(reviewer_paths["reviewer_2"]), "reviewer_2"
                ),
                "reviewer_2",
            )
            replay = replay_expert_review_merge(
                candidates=candidates,
                selection=selection,
                reviewer_1=left,
                reviewer_2=right,
                arbitration={},
            )
            expert = replay["expert"]
            review_statistics = expert_review_report_statistics(
                candidates=candidates,
                selection=selection,
                reviewer_1=left,
                reviewer_2=right,
                replay=replay,
            )
            gate_source = root / "review_sources/evaluation_gate_frozen.json"
            gate = frozen_gate_payload()
            gate["bindings"] = {
                "pilot_parent_manifest_sha256": sha256_file(pilot),
                "review_selection_sha256": sha256_file(selection_path),
                "candidate_index_sha256": sha256_file(candidate),
            }
            write_json(gate_source, gate)
            gate_output = root / "manifests/evaluation_gate_manifest.json"
            published_gate = json.loads(json.dumps(gate))
            published_gate["source_file"] = to_project_ref(gate_source)
            write_json(gate_output, published_gate)

            expert_all = root / "indexes/expert_all.jsonl"
            pending = root / "indexes/pending_arbitration.jsonl"
            write_jsonl(expert_all, expert)
            write_jsonl(pending, [])
            split_rows = {}
            for split in ("train", "val", "test"):
                split_rows[split] = [row for row in expert if row["split"] == split]
                write_jsonl(root / f"indexes/expert_{split}.jsonl", split_rows[split])
            merge_binding = {
                "protocol": EXPERT_ARTIFACT_BINDING_PROTOCOL,
                "builder_version": BUILDER_VERSION,
                "sources": {
                    "reviewer_1": file_artifact_binding(
                        reviewer_paths["reviewer_1"], records=len(selection)
                    ),
                    "reviewer_2": file_artifact_binding(
                        reviewer_paths["reviewer_2"], records=len(selection)
                    ),
                    "arbitration": None,
                    "evaluation_gate_source": file_artifact_binding(gate_source),
                },
                "outputs": {
                    "expert_all": file_artifact_binding(expert_all, records=len(expert)),
                    "expert_train": file_artifact_binding(
                        root / "indexes/expert_train.jsonl", records=1
                    ),
                    "expert_val": file_artifact_binding(
                        root / "indexes/expert_val.jsonl", records=1
                    ),
                    "expert_test": file_artifact_binding(
                        root / "indexes/expert_test.jsonl", records=1
                    ),
                    "pending_arbitration": file_artifact_binding(pending, records=0),
                    "evaluation_gate": file_artifact_binding(gate_output),
                },
            }
            report = {
                "builder_version": BUILDER_VERSION,
                "status": "complete",
                "frozen_evaluation_gate": True,
                "errors": [],
                **review_statistics,
                "expert_artifact_binding": merge_binding,
            }
            write_json(root / "reports/expert_review_report.json", report)
            errors: list[str] = []
            result = VALIDATOR._validate_expert(root, errors)
            self.assertEqual(errors, [])
            self.assertEqual(
                result["artifact_binding"]["protocol"],
                EXPERT_ARTIFACT_BINDING_PROTOCOL,
            )
            self.assertEqual(
                result["artifact_binding"]["semantic_replay"]["protocol"],
                EXPERT_REVIEW_REPLAY_PROTOCOL,
            )

            original_reviewer = reviewer_paths["reviewer_1"].read_text(encoding="utf-8")
            reviewer_paths["reviewer_1"].write_text(
                original_reviewer + "{}\n", encoding="utf-8"
            )
            errors = []
            VALIDATOR._validate_expert(root, errors)
            self.assertTrue(any("reviewer_1 artifact hash 漂移" in error for error in errors))
            reviewer_paths["reviewer_1"].write_text(original_reviewer, encoding="utf-8")

            report["acceptance_rate"] = 0.5
            write_json(root / "reports/expert_review_report.json", report)
            errors = []
            VALIDATOR._validate_expert(root, errors)
            self.assertTrue(any("审核统计" in error for error in errors))
            report["acceptance_rate"] = review_statistics["acceptance_rate"]
            write_json(root / "reports/expert_review_report.json", report)

            tampered = json.loads(json.dumps(expert))
            tampered[0]["expert_target"]["summary"] = "Self-consistent forged target."
            write_jsonl(expert_all, tampered)
            for split in ("train", "val", "test"):
                write_jsonl(
                    root / f"indexes/expert_{split}.jsonl",
                    [row for row in tampered if row["split"] == split],
                )
            for name, path in (
                ("expert_all", expert_all),
                ("expert_train", root / "indexes/expert_train.jsonl"),
                ("expert_val", root / "indexes/expert_val.jsonl"),
                ("expert_test", root / "indexes/expert_test.jsonl"),
            ):
                merge_binding["outputs"][name] = file_artifact_binding(
                    path,
                    records=len(
                        tampered
                        if name == "expert_all"
                        else [
                            row for row in tampered
                            if row["split"] == name.removeprefix("expert_")
                        ]
                    ),
                )
            report["expert_artifact_binding"] = merge_binding
            write_json(root / "reports/expert_review_report.json", report)
            errors = []
            VALIDATOR._validate_expert(root, errors)
            self.assertTrue(any(
                "精确语义重放结果" in error for error in errors
            ))

            write_jsonl(expert_all, expert)
            for split in ("train", "val", "test"):
                write_jsonl(root / f"indexes/expert_{split}.jsonl", split_rows[split])
            for name, path in (
                ("expert_all", expert_all),
                ("expert_train", root / "indexes/expert_train.jsonl"),
                ("expert_val", root / "indexes/expert_val.jsonl"),
                ("expert_test", root / "indexes/expert_test.jsonl"),
            ):
                merge_binding["outputs"][name] = file_artifact_binding(
                    path,
                    records=(len(expert) if name == "expert_all" else 1),
                )
            report["expert_artifact_binding"] = merge_binding
            write_json(root / "reports/expert_review_report.json", report)

            write_jsonl(root / "indexes/expert_val.jsonl", split_rows["val"] * 2)
            errors = []
            VALIDATOR._validate_expert(root, errors)
            self.assertTrue(any("精确 split 投影" in error for error in errors))
    def test_expert_structured_target_is_schema_and_gt_status_constrained(self) -> None:
        target = self._valid_structured_target()
        self.assertEqual(
            validate_bridge_structured_target(target, expected_target_status="present"), []
        )
        invalid = json.loads(json.dumps(target))
        invalid["region"]["size_class"] = "huge"
        self.assertTrue(validate_bridge_structured_target(invalid))
        self.assertTrue(
            validate_bridge_structured_target(target, expected_target_status="absent")
        )

    def test_absent_structured_target_rejects_region_or_support_claims(self) -> None:
        target = self._valid_structured_target("absent")
        self.assertEqual(
            validate_bridge_structured_target(
                target, expected_target_status="absent"
            ),
            [],
        )
        claimed_region = json.loads(json.dumps(target))
        claimed_region["region"]["location"] = "center"
        self.assertTrue(any(
            "region.location=unavailable" in error
            for error in validate_bridge_structured_target(claimed_region)
        ))
        claimed_support = json.loads(json.dumps(target))
        claimed_support["evidence"]["terrain_support"] = "supports"
        self.assertTrue(any(
            "evidence.terrain_support" in error
            for error in validate_bridge_structured_target(claimed_support)
        ))

    def test_absent_fact_builder_forces_null_region_evidence(self) -> None:
        record = {
            "target_status": "absent",
            "region_geometry": {
                "location": "center", "size_class": "large",
                "shape": "compact", "elongation": "low",
                "compactness": "compact", "fragmentation": "single",
            },
        }
        evidence = {
            "terrain": {
                "family": "terrain", "evidence_level": "A_physical",
            }
        }
        target = FACTS.structured_targets(record, evidence)
        self.assertEqual(target["region"]["location"], "unavailable")
        self.assertEqual(target["evidence"]["terrain_support"], "unavailable")
        self.assertEqual(target["evidence"]["evidence_sufficiency"], "unavailable")

    def test_frozen_gate_rejects_out_of_range_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(frozen_gate_payload(
                threshold_overrides={"no_target_rejection": 1.2}
            )), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
                MERGE._load_frozen_gate(str(path), Path(directory))

    def test_frozen_gate_is_bound_to_current_pilot_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding_paths = {
                "pilot_parent_manifest_sha256": root / "manifests/pilot_parent_manifest.jsonl",
                "review_selection_sha256": root / "manifests/review_selection.jsonl",
                "candidate_index_sha256": root / "indexes/candidate_all.jsonl",
            }
            for path in binding_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            gate = root / "gate.json"
            payload = frozen_gate_payload()
            payload["bindings"] = {
                name: sha256_file(path) for name, path in binding_paths.items()
            }
            gate.write_text(json.dumps(payload), encoding="utf-8")
            loaded = MERGE._load_frozen_gate(str(gate), root)
            self.assertEqual(loaded["bindings"], {
                name: sha256_file(path) for name, path in binding_paths.items()
            })
            binding_paths["review_selection_sha256"].write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "当前 Pilot/candidate 不匹配"):
                MERGE._load_frozen_gate(str(gate), root)

    def test_agreement_statistics(self) -> None:
        self.assertEqual(cohen_kappa(["accept", "reject"], ["accept", "reject"]), 1.0)
        alpha = krippendorff_alpha_nominal([["accept", "accept"], ["reject", "reject"]])
        self.assertEqual(alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
    evaluation_gate_scientific_template,
