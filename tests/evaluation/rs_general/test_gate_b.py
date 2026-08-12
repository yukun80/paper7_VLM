from __future__ import annotations

import copy
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from oa_groundrag.data.rs_general.io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from oa_groundrag.vlm.cli import _parser, entrypoint
from oa_groundrag.vlm.errors import (
    ContractError,
    EvaluationError,
    PredictionError,
    ReasonCode,
)
from oa_groundrag.evaluation.rs_general.contracts import (
    GATE_B_PROTOCOL_ID,
    GATE_B_SAMPLE_COUNT,
    GATE_B_TASK_ORDER,
    build_frozen_protocol,
    load_gate_b_protocol,
    static_protocol_snapshot,
    validate_frozen_protocol,
)
from oa_groundrag.evaluation.rs_general.acceptance import (
    GateBAcceptanceVerification,
    _recompute_selection,
    verify_gate_b_acceptance,
)
from oa_groundrag.evaluation.rs_general.metrics import (
    GateBEvaluationOutcome,
    _load_generation_run,
    _validate_prediction_rows,
    evaluate_gate_b,
    evaluate_gate_b_criteria,
    gate_b_rouge_l_f1,
    gate_b_token_f1,
    normalize_gate_b_text,
    score_gate_b_text,
    task_stratified_paired_bootstrap,
)
from oa_groundrag.evaluation.rs_general.generation import (
    GateBGenerationOutcome,
    generate_gate_b,
)
from oa_groundrag.evaluation.rs_general.selection import (
    GateBCandidate,
    _select_candidates,
    _selection_document,
    _strict_selection,
)
from oa_groundrag.vlm.model import Qwen3VLModelAdapter
from oa_groundrag.vlm.outputs import generic_prediction_row


REPO = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO / "configs/vlm/rs_general"
PROTOCOL = CONFIG_ROOT / "rs_generaldesc_gate_b_qwen3vl_2b.yaml"
TRAINING_ROOT = (
    REPO
    / "outputs/phase4_rs_vlm/rs_vlm_lora_qwen3vl_2b_b1a16"
)


CELL_SOURCES = {
    "bbox_region_caption": ("mmrs1m",),
    "global_caption": ("mmrs1m", "rsgpt"),
    "object_count": ("disasterm3", "rsgpt"),
    "scene_understanding": ("disasterm3", "rsgpt"),
    "spatial_relation": ("disasterm3",),
    "visible_change_report": ("disasterm3",),
    "visual_qa": ("mmrs1m", "rsgpt"),
}


def synthetic_candidates() -> list[GateBCandidate]:
    values: list[GateBCandidate] = []
    line_index = 0
    for task in GATE_B_TASK_ORDER:
        count = 11 if task == "spatial_relation" else 60
        sources = CELL_SOURCES[task]
        for index in range(count):
            source = sources[index % len(sources)]
            parent = f"{source}-{task}-{index:03d}"
            if task in {"object_count", "spatial_relation"} and index == 0:
                parent = "disasterm3-shared-cross-task-parent"
            values.append(
                GateBCandidate(
                    record_id=f"record-{task}-{index:03d}",
                    parent_id=parent,
                    source=source,
                    task_family=task,
                    shard_path=f"records/{source}-{task}.jsonl",
                    line_index=line_index,
                )
            )
            line_index += 1
    return values


def selected_fixture():
    candidates = synthetic_candidates()
    selected, quotas, capacities, cells = _select_candidates(
        candidates,
        excluded_parents={"monitoring-parent-never-eligible"},
    )
    return selected, quotas, capacities, cells


class _Identity:
    build_id = "build_fixture"
    payload_sha256 = "1" * 64

    def to_dict(self):
        return {
            "manifest_schema": "rs_generaldesc.manifest.v1",
            "canonical_schema": "rs_generaldesc.canonical.v1",
            "build_id": self.build_id,
            "payload_sha256": self.payload_sha256,
            "hash_manifest_sha256": "2" * 64,
        }


class GateBSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_parent_disjoint(self) -> None:
        baseline, _, _, _ = _select_candidates(
            synthetic_candidates(),
            excluded_parents=set(),
        )
        excluded_parent = baseline[0].parent_id
        first = _select_candidates(
            synthetic_candidates(),
            excluded_parents={excluded_parent},
        )
        second = _select_candidates(
            list(reversed(synthetic_candidates())),
            excluded_parents={excluded_parent},
        )
        self.assertEqual(first, second)
        selected, quotas, capacities, cells = first
        self.assertNotIn(excluded_parent, {item.parent_id for item in selected})
        self.assertEqual(len(selected), GATE_B_SAMPLE_COUNT)
        self.assertEqual(len({item.parent_id for item in selected}), 256)
        self.assertEqual(
            [quotas[task] for task in GATE_B_TASK_ORDER],
            [42, 41, 41, 41, 11, 40, 40],
        )
        self.assertEqual(capacities["spatial_relation"], 11)
        self.assertEqual(len(cells), 11)
        self.assertEqual(
            {item.cell for item in selected},
            set(cells),
        )
        self.assertEqual(
            Counter(item.task_family for item in selected),
            Counter(quotas),
        )

    def test_selection_contract_rejects_duplicate_order_and_hash_tamper(self) -> None:
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
        )
        _strict_selection(document)

        changed = copy.deepcopy(document)
        changed["items"][1] = copy.deepcopy(changed["items"][0])
        changed["items"][1]["ordinal"] = 1
        body = {
            key: value
            for key, value in changed.items()
            if key != "selection_sha256"
        }
        changed["selection_sha256"] = sha256_text(canonical_json(body))
        with self.assertRaises(ContractError):
            _strict_selection(changed)

        changed = copy.deepcopy(document)
        changed["source_task_counts"][next(iter(changed["source_task_counts"]))][
            next(iter(next(iter(changed["source_task_counts"].values()))))
        ] += 1
        with self.assertRaises(ContractError):
            _strict_selection(changed)


class GateBProtocolTests(unittest.TestCase):
    def test_protocol_is_distinct_strict_contract_and_configs_are_pinned(self) -> None:
        source = load_gate_b_protocol(PROTOCOL)
        self.assertEqual(source.raw["schema_version"], "rs_vlm.gate_b_protocol.v1")
        self.assertEqual(
            source.base_config.semantic_sha256,
            "37c772d70b2fed6bb349d0404cbe5d8931e52a18c64253948eb6f8f62a992a7e",
        )
        self.assertEqual(
            source.adapter_config.semantic_sha256,
            "a4b30e9a4f654cd85ee3d9e19fe49b1b4b1e53c88e7b2c38aeffffc9834fdd85",
        )
        row = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        row["base"]["config"] = str(
            CONFIG_ROOT / "rs_generaldesc_prompt_only_qwen3vl_2b.yaml"
        )
        row["adapter"]["config"] = str(
            CONFIG_ROOT / "rs_generaldesc_lora_qwen3vl_2b.yaml"
        )
        row["unexpected"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.yaml"
            path.write_text(yaml.safe_dump(row), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_gate_b_protocol(path)

    def test_frozen_protocol_rejects_implementation_tamper(self) -> None:
        source = load_gate_b_protocol(PROTOCOL)
        body = {
            "schema_version": "rs_vlm.gate_b_protocol.v1",
            "protocol_id": GATE_B_PROTOCOL_ID,
            "static_protocol": static_protocol_snapshot(source),
            "training_run": {},
        }
        frozen = {
            **body,
            "protocol_sha256": sha256_text(canonical_json(body)),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate_b_protocol.json"
            atomic_write_json(path, frozen)
            validate_frozen_protocol(PROTOCOL, path)
            changed = copy.deepcopy(frozen)
            changed["static_protocol"]["implementation_files"][
                "phase4/gate_b_evaluation.py"
            ] = "f" * 64
            changed_body = {
                key: value
                for key, value in changed.items()
                if key != "protocol_sha256"
            }
            changed["protocol_sha256"] = sha256_text(
                canonical_json(changed_body)
            )
            atomic_write_json(path, changed)
            with self.assertRaises(ContractError):
                validate_frozen_protocol(PROTOCOL, path)

    def test_training_report_and_best_pointer_tamper_stop_prepare(self) -> None:
        files = {
            "training_report.json": "training_report_sha256",
            "manifest.json": "training_manifest_sha256",
            "config_snapshot.json": "config_snapshot_sha256",
            "best_checkpoint.json": "best_checkpoint_sha256",
            "validation_selection.json": (
                "monitoring_selection_file_sha256"
            ),
            "validation_results.jsonl": "validation_results_sha256",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            root.mkdir()
            for index, relative in enumerate(files):
                (root / relative).write_text(
                    f'{{"fixture": {index}}}\n',
                    encoding="utf-8",
                )
            expected = {
                key: sha256_file(root / relative)
                for relative, key in files.items()
            }
            source = SimpleNamespace(
                raw={"training": expected},
                base_config=SimpleNamespace(),
            )
            access = SimpleNamespace()
            patches = (
                mock.patch(
                    "oa_groundrag.evaluation.rs_general.contracts.load_gate_b_protocol",
                    return_value=source,
                ),
                mock.patch(
                    "oa_groundrag.evaluation.rs_general.contracts.open_benchmark_access",
                    return_value=access,
                ),
            )
            for target in ("training_report.json", "best_checkpoint.json"):
                with self.subTest(target=target):
                    original = (root / target).read_bytes()
                    (root / target).write_bytes(original + b" ")
                    with patches[0], patches[1]:
                        with self.assertRaises(ContractError):
                            build_frozen_protocol(
                                Path(temporary) / "protocol.yaml",
                                training_root=root,
                            )
                    (root / target).write_bytes(original)


class GateBMetricTests(unittest.TestCase):
    def test_text_metrics_and_task_primary_definitions(self) -> None:
        self.assertEqual(normalize_gate_b_text("滑坡，AREA_1!"), "滑坡 area_1")
        self.assertAlmostEqual(gate_b_token_f1("a a b", "a b b"), 2.0 / 3.0)
        self.assertAlmostEqual(gate_b_rouge_l_f1("a x b", "a b"), 0.8)
        short = score_gate_b_text(
            "Two",
            ("two", "2"),
            task_family="object_count",
        )
        self.assertEqual(short["normalized_exact_match"], 1.0)
        self.assertEqual(short["primary"], 1.0)
        opened = score_gate_b_text(
            "a x b",
            ("a b",),
            task_family="global_caption",
        )
        self.assertEqual(
            opened["primary"],
            (
                opened["best_reference_token_f1"]
                + opened["best_reference_rouge_l_f1"]
            )
            / 2.0,
        )

    def test_bootstrap_is_exactly_reproducible(self) -> None:
        values = {
            task: (0.1, -0.02, 0.04, 0.08)
            for task in GATE_B_TASK_ORDER
        }
        first = task_stratified_paired_bootstrap(values)
        second = task_stratified_paired_bootstrap(values)
        self.assertEqual(first, second)
        self.assertEqual(first["iterations"], 10_000)
        self.assertEqual(first["seed"], 20260802)

    def test_all_six_gate_criteria_use_registered_boundaries(self) -> None:
        task_deltas = {
            task: (0.01 if index < 4 else -0.02)
            for index, task in enumerate(GATE_B_TASK_ORDER)
        }
        kwargs = {
            "per_task_deltas": task_deltas,
            "per_source_deltas": {
                "disasterm3": -0.02,
                "mmrs1m": 0.0,
                "rsgpt": 0.01,
            },
            "open_rouge_l_delta": 0.0,
            "short_exact_delta": 0.0,
        }
        criteria, passed = evaluate_gate_b_criteria(
            bootstrap_lower=0.0,
            **kwargs,
        )
        self.assertFalse(passed)
        self.assertEqual(criteria[0]["status"], "FAIL")
        self.assertTrue(all(row["status"] == "PASS" for row in criteria[1:]))
        criteria, passed = evaluate_gate_b_criteria(
            bootstrap_lower=1e-12,
            **kwargs,
        )
        self.assertTrue(passed)
        self.assertTrue(all(row["status"] == "PASS" for row in criteria))


class GateBPairingTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        identity = _Identity()
        items = []
        records = []
        rows = []
        for ordinal in range(GATE_B_SAMPLE_COUNT):
            task = GATE_B_TASK_ORDER[ordinal % len(GATE_B_TASK_ORDER)]
            source = ("disasterm3", "mmrs1m", "rsgpt")[ordinal % 3]
            item = {
                "ordinal": ordinal,
                "record_id": f"record-{ordinal:03d}",
                "parent_id": f"parent-{ordinal:03d}",
                "source": source,
                "task_family": task,
                "shard_path": "records/fixture.jsonl",
                "line_index": ordinal,
            }
            references = ["reference answer"]
            provenance = {
                "canonical_build_id": identity.build_id,
                "canonical_payload_sha256": identity.payload_sha256,
                "renderer": "phase3.render_canonical_messages",
                "gate_b": {
                    "protocol_id": GATE_B_PROTOCOL_ID,
                    "protocol_sha256": "a" * 64,
                    "selection_sha256": "b" * 64,
                    "ordinal": ordinal,
                    "model_role": "base",
                    "source": source,
                    "shard_path": item["shard_path"],
                    "line_index": ordinal,
                    "template_version": "qwen3vl_messages.v2",
                },
            }
            items.append(item)
            records.append({"reference_responses": references})
            rows.append(
                generic_prediction_row(
                    record_id=item["record_id"],
                    parent_id=item["parent_id"],
                    logical_role="external_val",
                    task_family=task,
                    generated_text="prediction",
                    reference_responses=references,
                    provenance=provenance,
                )
            )
        context = SimpleNamespace(
            selection={"items": items, "selection_sha256": "b" * 64},
            frozen_protocol={"protocol_sha256": "a" * 64},
            access=SimpleNamespace(identity=identity),
        )
        return context, records, rows

    def test_pairing_rejects_missing_duplicate_wrong_order_and_role(self) -> None:
        context, records, rows = self._fixture()
        _validate_prediction_rows(
            rows,
            model_role="base",
            context=context,
            canonical_records=records,
        )
        mutations = []
        mutations.append(rows[:-1])
        duplicated = copy.deepcopy(rows)
        duplicated[1] = copy.deepcopy(duplicated[0])
        mutations.append(duplicated)
        swapped = copy.deepcopy(rows)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        mutations.append(swapped)
        wrong_role = copy.deepcopy(rows)
        wrong_role[0]["provenance"]["gate_b"]["model_role"] = "adapter"
        mutations.append(wrong_role)
        for changed in mutations:
            with self.subTest(length=len(changed)):
                with self.assertRaises(EvaluationError):
                    _validate_prediction_rows(
                        changed,
                        model_role="base",
                        context=context,
                        canonical_records=records,
                    )

    def test_generation_manifest_role_interchange_is_rejected(self) -> None:
        context, _, rows = self._fixture()
        task_counts = dict(
            sorted(Counter(item["task_family"] for item in context.selection["items"]).items())
        )
        context.selection["task_counts"] = task_counts
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            root.mkdir()
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text("fixture: true\n", encoding="utf-8")
            config = SimpleNamespace(
                config_path=config_path,
                semantic_sha256="c" * 64,
                adaptation=SimpleNamespace(strategy="prompt_only"),
            )
            context.protocol_source = SimpleNamespace(base_config=config)
            context.frozen_protocol["static_protocol"] = {
                "model_identity": {"model": "fixture"},
                "processor_identity": {"processor": "fixture"},
                "generation": {
                    "seed": 20260802,
                    "max_new_tokens": 384,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "template_version": "qwen3vl_messages.v2",
                },
            }
            atomic_write_jsonl(root / "predictions.jsonl", rows)
            atomic_write_jsonl(root / "failures.jsonl", [])
            manifest = {
                "schema_version": "rs_vlm.gate_b_generation.v1",
                "status": "completed",
                "model_role": "base",
                "protocol_id": GATE_B_PROTOCOL_ID,
                "protocol_sha256": "a" * 64,
                "selection_sha256": "b" * 64,
                "selection_file_sha256": "d" * 64,
                "benchmark_identity": context.access.identity.to_dict(),
                "config_identity": {
                    "path": str(config_path.resolve()),
                    "file_sha256": sha256_file(config_path),
                    "semantic_sha256": "c" * 64,
                    "adaptation": "prompt_only",
                },
                "model_identity": {"model": "fixture"},
                "processor_identity": {"processor": "fixture"},
                "checkpoint_identity": None,
                "generation": context.frozen_protocol["static_protocol"]["generation"],
                "ordered_record_ids_sha256": sha256_text(
                    canonical_json(
                        [item["record_id"] for item in context.selection["items"]]
                    )
                ),
                "predictions": {
                    "path": "predictions.jsonl",
                    "count": 256,
                    "sha256": sha256_file(root / "predictions.jsonl"),
                },
                "failures": {
                    "path": "failures.jsonl",
                    "count": 0,
                    "sha256": sha256_file(root / "failures.jsonl"),
                },
                "task_counts": task_counts,
                "input_token_count": 256,
                "image_count": 256,
                "valid_for_evaluation": True,
            }
            atomic_write_json(root / "generation_manifest.json", manifest)
            loaded, loaded_rows, _ = _load_generation_run(
                root,
                model_role="base",
                context=context,
            )
            self.assertEqual(loaded["model_role"], "base")
            self.assertEqual(len(loaded_rows), 256)
            manifest["model_role"] = "adapter"
            atomic_write_json(root / "generation_manifest.json", manifest)
            with self.assertRaises(EvaluationError):
                _load_generation_run(root, model_role="base", context=context)

    def test_missing_inputs_publish_invalid_not_scientific_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            outcome = evaluate_gate_b(
                PROTOCOL,
                Path(temporary) / "missing_selection.json",
                base_run=Path(temporary) / "missing_base",
                adapter_run=Path(temporary) / "missing_adapter",
                output_root=output,
            )
            self.assertEqual(outcome.status, "invalid")
            self.assertFalse(outcome.gate_b_passed)
            report = read_json(output / "gate_b_report.json")
            self.assertFalse(report["gate_b_evaluated"])
            self.assertFalse(report["formal_acceptance"])
            self.assertEqual(report["adapter_status"], "not_evaluated")
            self.assertTrue(report["invalid_reasons"])

    def test_public_completed_evaluation_pass_and_fail_contract(self) -> None:
        context, records, template_rows = self._fixture()
        context.selection.update(
            schema_version="rs_vlm.gate_b_selection.v1",
            sample_count=GATE_B_SAMPLE_COUNT,
            parent_count=GATE_B_SAMPLE_COUNT,
            monitoring_exclusion={"intersection_count": 0},
        )
        base_rows = copy.deepcopy(template_rows)
        for row in base_rows:
            row["generated_text"] = "unrelated"
        adapter_rows = copy.deepcopy(template_rows)
        for row in adapter_rows:
            row["generated_text"] = "reference answer"
            row["provenance"]["gate_b"]["model_role"] = "adapter"
        selection_file_sha = "d" * 64

        def manifest(role: str) -> dict:
            return {
                "selection_file_sha256": selection_file_sha,
                "predictions": {"count": 256, "sha256": role[0] * 64},
                "failures": {"count": 0},
                "config_identity": {"role": role},
                "model_identity": {"model": "fixture"},
                "processor_identity": {"processor": "fixture"},
                "checkpoint_identity": (
                    None if role == "base" else {"step": 1}
                ),
            }

        context.frozen_protocol["training_run"] = {
            "best_checkpoint": {"step": 1}
        }
        context.protocol_source = SimpleNamespace(
            base_config=SimpleNamespace(
                data=SimpleNamespace(
                    benchmark_root=Path("/fixture"),
                    expected_manifest_sha256="e" * 64,
                )
            )
        )
        context.access.verifier = SimpleNamespace()
        for expected_passed in (True, False):
            with self.subTest(expected_passed=expected_passed):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selection_path = root / "selection.json"
                    selection_path.write_text("fixture", encoding="utf-8")
                    actual_selection_sha = sha256_file(selection_path)
                    base_manifest = manifest("base")
                    adapter_manifest = manifest("adapter")
                    base_manifest["selection_file_sha256"] = actual_selection_sha
                    adapter_manifest["selection_file_sha256"] = actual_selection_sha
                    compared_adapter = copy.deepcopy(adapter_rows)
                    if not expected_passed:
                        for base_row, adapter_row in zip(
                            base_rows,
                            compared_adapter,
                            strict=True,
                        ):
                            adapter_row["generated_text"] = base_row[
                                "generated_text"
                            ]
                    with (
                        mock.patch(
                            "oa_groundrag.evaluation.rs_general.metrics.load_gate_b_selection",
                            return_value=context,
                        ),
                        mock.patch(
                            "oa_groundrag.evaluation.rs_general.metrics._load_generation_run",
                            side_effect=(
                                (base_manifest, base_rows, "1" * 64),
                                (
                                    adapter_manifest,
                                    compared_adapter,
                                    "2" * 64,
                                ),
                            ),
                        ),
                        mock.patch(
                            "oa_groundrag.evaluation.rs_general.metrics.RSGeneralDescDataset.from_locations",
                            return_value=SimpleNamespace(records=records),
                        ),
                    ):
                        outcome = evaluate_gate_b(
                            PROTOCOL,
                            selection_path,
                            base_run=root / "base",
                            adapter_run=root / "adapter",
                            output_root=root / "evaluation",
                        )
                    report = read_json(
                        root / "evaluation" / "gate_b_report.json"
                    )
                    self.assertEqual(outcome.status, "completed")
                    self.assertEqual(outcome.gate_b_passed, expected_passed)
                    self.assertEqual(
                        report["adapter_status"],
                        "accepted" if expected_passed else "not_accepted",
                    )
                    self.assertEqual(report["pairing"]["paired_count"], 256)


class GateBAcceptanceTests(unittest.TestCase):
    @staticmethod
    def _publish_fixture(root: Path):
        context, records, template_rows = GateBPairingTests._fixture()
        context.selection.update(
            schema_version="rs_vlm.gate_b_selection.v1",
            sample_count=GATE_B_SAMPLE_COUNT,
            parent_count=GATE_B_SAMPLE_COUNT,
            monitoring_exclusion={"intersection_count": 0},
        )
        context.frozen_protocol["training_run"] = {
            "best_checkpoint": {"step": 1},
            "monitoring_selection": {
                "file_sha256": "c" * 64,
                "selection_sha256": "d" * 64,
            },
        }
        context.protocol_source = SimpleNamespace(
            base_config=SimpleNamespace(
                data=SimpleNamespace(
                    benchmark_root=Path("/fixture"),
                    expected_manifest_sha256="e" * 64,
                )
            )
        )
        context.access.verifier = SimpleNamespace()

        selection_root = root / "selection"
        training_root = root / "training"
        base_root = root / "base"
        adapter_root = root / "adapter"
        for directory in (
            selection_root,
            training_root,
            base_root,
            adapter_root,
        ):
            directory.mkdir()
        protocol_file = selection_root / "gate_b_protocol.json"
        selection_path = selection_root / "gate_b_selection.json"
        atomic_write_json(protocol_file, {"fixture": "frozen-protocol"})
        atomic_write_json(selection_path, {"fixture": "selection"})
        selection_file_sha = sha256_file(selection_path)

        base_rows = copy.deepcopy(template_rows)
        for row in base_rows:
            row["generated_text"] = "unrelated"
        adapter_rows = copy.deepcopy(template_rows)
        for row in adapter_rows:
            row["generated_text"] = "reference answer"
            row["provenance"]["gate_b"]["model_role"] = "adapter"
        atomic_write_jsonl(base_root / "predictions.jsonl", base_rows)
        atomic_write_jsonl(adapter_root / "predictions.jsonl", adapter_rows)
        atomic_write_jsonl(base_root / "failures.jsonl", [])
        atomic_write_jsonl(adapter_root / "failures.jsonl", [])

        def manifest(role: str, run_root: Path) -> dict:
            return {
                "selection_file_sha256": selection_file_sha,
                "predictions": {
                    "count": GATE_B_SAMPLE_COUNT,
                    "sha256": sha256_file(run_root / "predictions.jsonl"),
                },
                "failures": {"count": 0},
                "input_token_count": 256,
                "image_count": 256,
                "config_identity": {"role": role},
                "model_identity": {"model": "fixture"},
                "processor_identity": {"processor": "fixture"},
                "checkpoint_identity": (
                    None if role == "base" else {"step": 1}
                ),
            }

        base_manifest = manifest("base", base_root)
        adapter_manifest = manifest("adapter", adapter_root)
        atomic_write_json(base_root / "generation_manifest.json", base_manifest)
        atomic_write_json(
            adapter_root / "generation_manifest.json",
            adapter_manifest,
        )
        base_manifest_sha = sha256_file(base_root / "generation_manifest.json")
        adapter_manifest_sha = sha256_file(
            adapter_root / "generation_manifest.json"
        )
        load_runs = (
            (base_manifest, base_rows, base_manifest_sha),
            (adapter_manifest, adapter_rows, adapter_manifest_sha),
        )
        with (
            mock.patch(
                "oa_groundrag.evaluation.rs_general.metrics.load_gate_b_selection",
                return_value=context,
            ),
            mock.patch(
                "oa_groundrag.evaluation.rs_general.metrics._load_generation_run",
                side_effect=load_runs,
            ),
            mock.patch(
                "oa_groundrag.evaluation.rs_general.metrics.RSGeneralDescDataset.from_locations",
                return_value=SimpleNamespace(records=records),
            ),
        ):
            outcome = evaluate_gate_b(
                PROTOCOL,
                selection_path,
                base_run=base_root,
                adapter_run=adapter_root,
                output_root=root / "evaluation",
            )
        if outcome.status != "completed" or not outcome.gate_b_passed:
            raise AssertionError("fixture Gate B evaluation must pass")
        return SimpleNamespace(
            context=context,
            records=records,
            base_rows=base_rows,
            adapter_rows=adapter_rows,
            base_manifest=base_manifest,
            adapter_manifest=adapter_manifest,
            base_manifest_sha=base_manifest_sha,
            adapter_manifest_sha=adapter_manifest_sha,
            protocol_file=protocol_file,
            selection_path=selection_path,
            training_root=training_root,
            base_root=base_root,
            adapter_root=adapter_root,
            evaluation_root=root / "evaluation",
            expected_report_sha=sha256_file(
                root / "evaluation/gate_b_report.json"
            ),
        )

    @staticmethod
    def _verify(fixture, *, training_error: Exception | None = None):
        with (
            mock.patch(
                "oa_groundrag.evaluation.rs_general.acceptance.load_gate_b_selection",
                return_value=fixture.context,
            ),
            mock.patch(
                "oa_groundrag.evaluation.rs_general.acceptance.validate_frozen_training_root",
                side_effect=training_error,
            ),
            mock.patch(
                "oa_groundrag.evaluation.rs_general.acceptance._recompute_selection"
            ),
            mock.patch(
                "oa_groundrag.evaluation.rs_general.acceptance._load_generation_run",
                side_effect=(
                    (
                        fixture.base_manifest,
                        fixture.base_rows,
                        fixture.base_manifest_sha,
                    ),
                    (
                        fixture.adapter_manifest,
                        fixture.adapter_rows,
                        fixture.adapter_manifest_sha,
                    ),
                ),
            ),
            mock.patch(
                "oa_groundrag.evaluation.rs_general.acceptance.RSGeneralDescDataset.from_locations",
                return_value=SimpleNamespace(records=fixture.records),
            ),
        ):
            return verify_gate_b_acceptance(
                PROTOCOL,
                fixture.selection_path,
                training_root=fixture.training_root,
                base_run=fixture.base_root,
                adapter_run=fixture.adapter_root,
                evaluation_root=fixture.evaluation_root,
                expected_protocol_file_sha256=sha256_file(
                    fixture.protocol_file
                ),
                expected_report_sha256=fixture.expected_report_sha,
            )

    def test_read_only_verifier_accepts_and_rejects_evidence_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._publish_fixture(Path(temporary))
            verification = self._verify(fixture)
            self.assertEqual(verification.status, "accepted")
            self.assertEqual(verification.paired_count, GATE_B_SAMPLE_COUNT)

            report_path = fixture.evaluation_root / "gate_b_report.json"
            report_bytes = report_path.read_bytes()
            report_path.write_bytes(report_bytes + b" ")
            with self.assertRaises(EvaluationError):
                self._verify(fixture)
            report_path.write_bytes(report_bytes)

            paired_path = fixture.evaluation_root / "paired_scores.jsonl"
            paired_rows = read_jsonl(paired_path)
            paired_rows[0]["adapter_primary"] = 0.0
            atomic_write_jsonl(paired_path, paired_rows)
            with self.assertRaises(EvaluationError):
                self._verify(fixture)

    def test_verifier_rejects_training_identity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._publish_fixture(Path(temporary))
            with self.assertRaises(ContractError):
                self._verify(
                    fixture,
                    training_error=ContractError(
                        ReasonCode.CHECKPOINT_INCOMPATIBLE,
                        "fixture training tamper",
                    ),
                )

    def test_selection_recomputation_rejects_structurally_valid_substitute(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            training_root = Path(temporary)
            monitoring_path = training_root / "validation_selection.json"
            atomic_write_json(monitoring_path, {"items": []})
            context = SimpleNamespace(
                frozen_protocol={
                    "training_run": {
                        "monitoring_selection": {
                            "file_sha256": sha256_file(monitoring_path),
                            "selection_sha256": "a" * 64,
                        }
                    }
                },
                access=SimpleNamespace(),
                selection={"items": [{"record_id": "substitute"}]},
            )
            with (
                mock.patch(
                    "oa_groundrag.evaluation.rs_general.acceptance._scan_candidates",
                    return_value=([], ("records/fixture.jsonl",)),
                ),
                mock.patch(
                    "oa_groundrag.evaluation.rs_general.acceptance._select_candidates",
                    return_value=([], {}, {}, ()),
                ),
                mock.patch(
                    "oa_groundrag.evaluation.rs_general.acceptance._selection_document",
                    return_value={"items": [{"record_id": "registered"}]},
                ),
            ):
                with self.assertRaises(EvaluationError):
                    _recompute_selection(context, training_root)


class GateBGenerationAndCliTests(unittest.TestCase):
    def test_generation_rejects_output_before_loading_and_requires_best_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            with mock.patch(
                "oa_groundrag.evaluation.rs_general.generation.load_gate_b_selection"
            ) as load_selection:
                with self.assertRaises(PredictionError) as caught:
                    generate_gate_b(
                        PROTOCOL,
                        root / "missing-selection.json",
                        model_role="base",
                        output_root=existing,
                    )
                self.assertEqual(caught.exception.code, ReasonCode.OUTPUT_EXISTS)
                load_selection.assert_not_called()
            with self.assertRaises(PredictionError) as caught:
                generate_gate_b(
                    PROTOCOL,
                    root / "missing-selection.json",
                    model_role="adapter",
                    output_root=root / "adapter",
                )
            self.assertEqual(
                caught.exception.code,
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
            )
            with self.assertRaises(PredictionError):
                generate_gate_b(
                    PROTOCOL,
                    root / "missing-selection.json",
                    model_role="base",
                    training_root=TRAINING_ROOT,
                    output_root=root / "base",
                )
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                [
                    "gate-b-generate",
                    "--protocol",
                    str(PROTOCOL),
                    "--selection",
                    "selection.json",
                    "--model-role",
                    "adapter",
                    "--checkpoint",
                    "arbitrary",
                    "--output-root",
                    "output",
                ]
            )

    def test_greedy_generation_omits_sampling_kwargs(self) -> None:
        class DummyModel:
            def __init__(self):
                self.arguments = None

            def eval(self):
                return None

            def named_modules(self):
                return []

            def generate(self, **kwargs):
                self.arguments = kwargs
                return torch.tensor([[10, 11, 12]])

        class DummyProcessor:
            @staticmethod
            def batch_decode(*args, **kwargs):
                return ["answer"]

        adapter = object.__new__(Qwen3VLModelAdapter)
        adapter.model = DummyModel()
        output = adapter.generate_text(
            {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            },
            processor=DummyProcessor(),
            max_new_tokens=384,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        self.assertEqual(output, ["answer"])
        self.assertNotIn("temperature", adapter.model.arguments)
        self.assertNotIn("top_p", adapter.model.arguments)

    def test_cli_exit_codes_distinguish_pass_fail_and_invalid(self) -> None:
        with mock.patch(
            "oa_groundrag.vlm.cli.prepare_gate_b",
            return_value=Path("/tmp/selection"),
        ) as prepare:
            with redirect_stdout(StringIO()):
                code = entrypoint(
                    [
                        "gate-b-prepare",
                        "--protocol",
                        str(PROTOCOL),
                        "--training-root",
                        str(TRAINING_ROOT),
                        "--output-root",
                        "/tmp/selection",
                    ]
                )
            self.assertEqual(code, 0)
            prepare.assert_called_once()
        outcomes = (
            ("invalid", False, 2),
            ("completed", False, 1),
            ("completed", True, 0),
        )
        for status, passed, expected in outcomes:
            with self.subTest(status=status, passed=passed):
                with mock.patch(
                    "oa_groundrag.vlm.cli.evaluate_gate_b",
                    return_value=GateBEvaluationOutcome(
                        root=Path("/tmp/evaluation"),
                        status=status,
                        gate_b_passed=passed,
                    ),
                ):
                    with redirect_stdout(StringIO()):
                        code = entrypoint(
                            [
                                "gate-b-evaluate",
                                "--protocol",
                                str(PROTOCOL),
                                "--selection",
                                "selection.json",
                                "--base-run",
                                "base",
                                "--adapter-run",
                                "adapter",
                                "--output-root",
                                "evaluation",
                            ]
                        )
                self.assertEqual(code, expected)
        with mock.patch(
            "oa_groundrag.vlm.cli.generate_gate_b",
            return_value=GateBGenerationOutcome(
                root=Path("/tmp/base"),
                prediction_count=255,
                failure_count=1,
            ),
        ):
            with redirect_stdout(StringIO()):
                code = entrypoint(
                    [
                        "gate-b-generate",
                        "--protocol",
                        str(PROTOCOL),
                        "--selection",
                        "selection.json",
                        "--model-role",
                        "base",
                        "--output-root",
                        "base",
                    ]
                )
        self.assertEqual(code, 2)

    def test_verify_cli_and_narrow_runtime_error_mapping(self) -> None:
        verification = GateBAcceptanceVerification(
            status="accepted",
            gate_b_evaluated=True,
            gate_b_passed=True,
            formal_acceptance=True,
            paired_count=256,
            protocol_sha256="a" * 64,
            selection_sha256="b" * 64,
            artifact_sha256={
                "protocol_file": "1" * 64,
                "selection_file": "2" * 64,
                "base_manifest": "3" * 64,
                "base_predictions": "4" * 64,
                "adapter_manifest": "5" * 64,
                "adapter_predictions": "6" * 64,
                "paired_scores": "7" * 64,
                "report": "8" * 64,
            },
        )
        verify_argv = [
            "gate-b-verify",
            "--protocol",
            "protocol.yaml",
            "--selection",
            "selection.json",
            "--training-root",
            "training",
            "--base-run",
            "base",
            "--adapter-run",
            "adapter",
            "--evaluation-root",
            "evaluation",
            "--expected-protocol-file-sha256",
            "1" * 64,
            "--expected-report-sha256",
            "8" * 64,
        ]
        with mock.patch(
            "oa_groundrag.vlm.cli.verify_gate_b_acceptance",
            return_value=verification,
        ) as verify:
            with redirect_stdout(StringIO()):
                self.assertEqual(entrypoint(verify_argv), 0)
            verify.assert_called_once()

        generate_argv = [
            "gate-b-generate",
            "--protocol",
            "protocol.yaml",
            "--selection",
            "selection.json",
            "--model-role",
            "base",
            "--output-root",
            "base",
        ]
        with mock.patch(
            "oa_groundrag.vlm.cli.generate_gate_b",
            side_effect=RuntimeError("CUDA error: driver unavailable"),
        ):
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(entrypoint(generate_argv), 2)
            self.assertIn("GATE_B_RUN_INVALID", stderr.getvalue())
        with mock.patch(
            "oa_groundrag.vlm.cli.generate_gate_b",
            side_effect=RuntimeError("programming bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "programming bug"):
                entrypoint(generate_argv)


if __name__ == "__main__":
    unittest.main()
