from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.data.rs_general.io import atomic_write_json
from oa_groundrag.evaluation.rs_general.comparison import (
    FAMILY_COMPARISON_SCHEMA_VERSION,
    _rename_metric_roles,
)
from oa_groundrag.evaluation.rs_general.contracts import (
    GATE_B_QWEN35_PROTOCOL_ID,
    gate_b_protocol_profile,
    load_gate_b_protocol,
)
from oa_groundrag.evaluation.rs_general.metrics import _completed_metrics
from oa_groundrag.evaluation.rs_general.selection import (
    _reference_selection_identity,
    _selection_document,
    _strict_selection,
)
from oa_groundrag.grounding.contracts import (
    GATE_B_PROTOCOL_SCHEMA_VERSION_V2,
    GATE_B_REPORT_SCHEMA_VERSION_V2,
)
from oa_groundrag.vlm.errors import ContractError
from oa_groundrag.vlm.outputs import generic_prediction_row

from tests.evaluation.rs_general.test_gate_b import (
    _Identity,
    selected_fixture,
    synthetic_candidates,
)


REPO = Path(__file__).resolve().parents[3]
PROTOCOL = (
    REPO
    / "configs/vlm/rs_general/rs_generaldesc_gate_b_qwen35_4b_v2.yaml"
)
PROFILE = gate_b_protocol_profile(
    GATE_B_PROTOCOL_SCHEMA_VERSION_V2,
    GATE_B_QWEN35_PROTOCOL_ID,
)


def _reference_identity(path: str = "/tmp/reference-selection.json"):
    return {
        "path": path,
        "file_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "ordered_record_ids_sha256": "3" * 64,
        "sample_count": 256,
        "exact_item_identity_match": True,
    }


class GateBQwen35ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_gate_b_protocol(PROTOCOL)

    def test_v2_protocol_binds_family_revision_and_reference_selection(self) -> None:
        source = self.source
        self.assertEqual(source.profile, PROFILE)
        self.assertEqual(source.base_config.model.backend, "qwen3_5")
        self.assertEqual(source.adapter_config.model.backend, "qwen3_5")
        self.assertEqual(source.raw["generation"]["enable_thinking"], False)
        self.assertEqual(source.raw["generation"]["max_new_tokens"], 768)
        self.assertEqual(
            source.raw["reference_selection"]["ordered_record_ids_sha256"],
            "00109c005c9f06d16d2cce58be76fee24f49abb6949517fd913a61c28e0928f7",
        )

    def test_v2_protocol_rejects_thinking_and_unknown_fields(self) -> None:
        row = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        row["generation"]["enable_thinking"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.yaml"
            path.write_text(yaml.safe_dump(row), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_gate_b_protocol(path)
        row = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        row["unexpected"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.yaml"
            path.write_text(yaml.safe_dump(row), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_gate_b_protocol(path)


class GateBQwen35SelectionTests(unittest.TestCase):
    def test_v2_selection_requires_reference_identity(self) -> None:
        selected, quotas, capacities, cells = selected_fixture()
        monitoring = tuple(f"monitoring-{index:03d}" for index in range(128))
        document = _selection_document(
            frozen_protocol={"protocol_sha256": "a" * 64},
            access=SimpleNamespace(identity=_Identity()),
            selected=selected,
            quotas=quotas,
            capacities=capacities,
            cells=cells,
            probed_shards=tuple(
                sorted({item.shard_path for item in synthetic_candidates()})
            ),
            monitoring_file_sha256="b" * 64,
            monitoring_selection_sha256="c" * 64,
            monitoring_parents=monitoring,
            profile=PROFILE,
            reference_selection_identity=_reference_identity(),
        )
        _strict_selection(document, PROFILE)
        changed = copy.deepcopy(document)
        changed["reference_selection_identity"][
            "exact_item_identity_match"
        ] = False
        body = {
            key: value
            for key, value in changed.items()
            if key != "selection_sha256"
        }
        changed["selection_sha256"] = sha256_text(canonical_json(body))
        with self.assertRaises(ContractError):
            _strict_selection(changed, PROFILE)

    def test_reference_selection_compares_every_item_identity(self) -> None:
        selected, _, _, _ = selected_fixture()
        items = [
            {
                "ordinal": ordinal,
                "record_id": candidate.record_id,
                "parent_id": candidate.parent_id,
                "source": candidate.source,
                "task_family": candidate.task_family,
                "shard_path": candidate.shard_path,
                "line_index": candidate.line_index,
            }
            for ordinal, candidate in enumerate(selected)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            atomic_write_json(path, {"items": items})
            source = SimpleNamespace(
                profile=PROFILE,
                raw={
                    "reference_selection": {
                        "path": str(path),
                        "file_sha256": sha256_file(path),
                        "selection_sha256": "2" * 64,
                        "ordered_record_ids_sha256": "3" * 64,
                        "sample_count": 256,
                    }
                },
            )
            identity = _reference_selection_identity(source, selected)
            self.assertTrue(identity["exact_item_identity_match"])
            items[0]["record_id"] = "different"
            atomic_write_json(path, {"items": items})
            source.raw["reference_selection"]["file_sha256"] = sha256_file(path)
            with self.assertRaises(ContractError):
                _reference_selection_identity(source, selected)


class GateBQwen35MetricsTests(unittest.TestCase):
    def test_v2_paired_scores_use_new_schema_and_comparison_is_report_only(self) -> None:
        selected, quotas, capacities, cells = selected_fixture()
        items = [
            {
                "ordinal": ordinal,
                "record_id": candidate.record_id,
                "parent_id": candidate.parent_id,
                "source": candidate.source,
                "task_family": candidate.task_family,
                "shard_path": candidate.shard_path,
                "line_index": candidate.line_index,
            }
            for ordinal, candidate in enumerate(selected)
        ]
        rows = []
        for item in items:
            rows.append(
                generic_prediction_row(
                    record_id=item["record_id"],
                    parent_id=item["parent_id"],
                    logical_role="external_val",
                    task_family=item["task_family"],
                    generated_text="reference answer",
                    reference_responses=("reference answer",),
                    provenance={},
                )
            )
        context = SimpleNamespace(
            protocol_source=SimpleNamespace(profile=PROFILE),
            selection={"items": items},
        )
        paired, metrics, bootstrap, criteria, passed = _completed_metrics(
            context,
            rows,
            rows,
        )
        self.assertEqual(len(paired), 256)
        self.assertTrue(
            all(
                row["schema_version"] == GATE_B_REPORT_SCHEMA_VERSION_V2
                for row in paired
            )
        )
        self.assertFalse(passed)
        self.assertEqual(len(criteria), 6)
        renamed = _rename_metric_roles(metrics)
        self.assertIn("reference_2b", renamed["overall_task_macro"])
        self.assertIn("candidate_4b", renamed["overall_task_macro"])
        self.assertEqual(bootstrap["lower"], 0.0)
        self.assertEqual(
            FAMILY_COMPARISON_SCHEMA_VERSION,
            "rs_vlm.gate_b_family_comparison.v1",
        )


if __name__ == "__main__":
    unittest.main()
