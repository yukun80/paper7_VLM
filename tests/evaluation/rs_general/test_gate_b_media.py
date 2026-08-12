from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

from oa_groundrag.data.rs_general.io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.data.rs_general.contracts import (
    BENCHMARK_SCOPE,
    CANONICAL_SCHEMA_VERSION,
    HASH_SCHEMA_VERSION,
    MANIFEST_VERSION,
    parent_id,
    record_hash,
    record_id,
    validate_canonical_record,
)
from oa_groundrag.data.rs_general.errors import (
    RSGeneralDescError,
    ReasonCode as Phase3ReasonCode,
)
from oa_groundrag.vlm.cli import entrypoint
from oa_groundrag.vlm.errors import PredictionError, ReasonCode
from oa_groundrag.evaluation.rs_general.contracts import (
    GATE_B_PROTOCOL_ID,
    QWEN_TEMPLATE_VERSION,
)
from oa_groundrag.evaluation.rs_general.media import locate_gate_b_media
from oa_groundrag.vlm.outputs import generic_prediction_row


class _GateBMediaFixture:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "benchmark"
        self.root.mkdir()
        self.shard_relative = "records/part-00000.jsonl"
        self.assets: dict[str, list[tuple[str, Path]]] = {}
        self.records = [
            self._record(
                name="single",
                source="rsgpt",
                task_family="global_caption",
                supervision_kind="long_description",
                input_layout="single_image",
                roles=("image",),
            ),
            self._record(
                name="prepost",
                source="disasterm3",
                task_family="visible_change_report",
                supervision_kind="structured_report",
                input_layout="pre_post",
                roles=("pre_image", "post_image"),
            ),
            self._record(
                name="bbox",
                source="mmrs1m",
                task_family="bbox_region_caption",
                supervision_kind="region_description",
                input_layout="bbox_region",
                roles=("image",),
                bbox=True,
            ),
        ]
        atomic_write_jsonl(
            self.root / self.shard_relative,
            self.records,
        )
        self.build_id = "build_" + "a" * 64
        self.payload_sha256 = self._write_ledger_and_manifest()
        self.prediction_rows = [
            self._prediction(record, ordinal)
            for ordinal, record in enumerate(self.records)
        ]
        self.predictions = base / "predictions.jsonl"
        atomic_write_jsonl(self.predictions, self.prediction_rows)

    def _image_media(
        self,
        *,
        name: str,
        role: str,
        color: tuple[int, int, int],
    ) -> dict[str, object]:
        temporary = self.base / f"{name}-{role}.png"
        Image.new("RGB", (4, 3), color).save(temporary)
        digest = sha256_file(temporary)
        relative = f"assets/image/{digest[:2]}/{digest}.png"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
        self.assets.setdefault(name, []).append((role, target))
        return {
            "asset_id": f"asset_{digest}",
            "path": relative,
            "media_type": "image",
            "sha256": digest,
            "size_bytes": target.stat().st_size,
            "extension": "png",
            "role": role,
            "source_sha256": digest,
            "width": 4,
            "height": 3,
            "mode": "RGB",
        }

    @staticmethod
    def _transforms(role: str) -> list[dict[str, object]]:
        size = [4, 3]
        return [
            {
                "asset_role": role,
                "type": "exif_orientation",
                "orientation": 1,
                "input_size": size,
                "output_size": size,
            },
            {
                "asset_role": role,
                "type": "color_conversion",
                "output_mode": "RGB",
            },
            {
                "asset_role": role,
                "type": "resize",
                "interpolation": "none",
                "input_size": size,
                "output_size": size,
            },
        ]

    def _record(
        self,
        *,
        name: str,
        source: str,
        task_family: str,
        supervision_kind: str,
        input_layout: str,
        roles: tuple[str, ...],
        bbox: bool = False,
    ) -> dict[str, object]:
        media = [
            self._image_media(
                name=name,
                role=role,
                color=(40 + index * 20, 80 + len(name), 120 + index),
            )
            for index, role in enumerate(roles)
        ]
        target: dict[str, object]
        if bbox:
            target = {
                "type": "bbox",
                "image_role": "image",
                "source_convention": "xyxy_normalized_top_left",
                "canonical_convention": "xyxy_normalized_top_left",
                "boxes": [
                    {
                        "source": [0.1, 0.1, 0.8, 0.8],
                        "source_convention": "xyxy_normalized_top_left",
                        "source_image_size": [4, 3],
                        "canonical_xyxy_norm": [0.1, 0.1, 0.8, 0.8],
                        "label": "region",
                    }
                ],
            }
        else:
            target = {"type": "none"}
        parent = parent_id(source, f"fixture-{name}")
        training_responses = [f"training response {name}"]
        reference_responses = [f"reference response {name}"]
        instruction = f"describe fixture {name}"
        identity = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "parent_id": parent,
            "source": source,
            "logical_role": "external_val",
            "task_family": task_family,
            "supervision_kind": supervision_kind,
            "input_layout": input_layout,
            "output_modality": "text",
            "media_asset_ids": sorted(str(value["asset_id"]) for value in media),
            "target": target,
            "instruction": instruction,
            "training_responses": training_responses,
            "reference_responses": reference_responses,
            "annotation_layer": "external_source",
        }
        record: dict[str, object] = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "record_id": record_id(identity),
            "parent_id": parent,
            "source": source,
            "source_record_id": f"fixture-{name}",
            "source_split": "val",
            "logical_role": "external_val",
            "task_family": task_family,
            "supervision_kind": supervision_kind,
            "input_layout": input_layout,
            "output_modality": "text",
            "media": media,
            "target": target,
            "instruction": instruction,
            "training_responses": training_responses,
            "reference_responses": reference_responses,
            "deterministic_facts": {},
            "annotation": {
                "layer": "external_source",
                "review_status": "not_required",
            },
            "coordinate_convention": {
                "image_origin": "top_left",
                "bbox_canonical": "xyxy_normalized",
            },
            "transforms": [
                transform
                for role in roles
                for transform in self._transforms(role)
            ],
            "quality_flags": [],
            "provenance_ids": ["provenance_" + "b" * 64],
        }
        record["record_sha256"] = record_hash(record)
        validate_canonical_record(record)
        return record

    def _write_ledger_and_manifest(self) -> str:
        paths = [self.shard_relative]
        paths.extend(
            str(path.relative_to(self.root).as_posix())
            for values in self.assets.values()
            for _, path in values
        )
        files = [
            {
                "path": relative,
                "size_bytes": (self.root / relative).stat().st_size,
                "sha256": sha256_file(self.root / relative),
            }
            for relative in sorted(paths)
        ]
        payload_sha256 = sha256_text(canonical_json(files))
        ledger = {
            "schema_version": HASH_SCHEMA_VERSION,
            "files": files,
            "root_sha256": payload_sha256,
        }
        ledger_relative = "metadata/hashes.json"
        atomic_write_json(self.root / ledger_relative, ledger)
        atomic_write_json(
            self.root / "manifest.json",
            {
                "schema_version": MANIFEST_VERSION,
                "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
                "benchmark_scope": BENCHMARK_SCOPE,
                "build_id": self.build_id,
                "payload_root_sha256": payload_sha256,
                "hash_manifest_sha256": sha256_file(
                    self.root / ledger_relative
                ),
                "layout": {
                    "record_shards": [
                        {
                            "path": self.shard_relative,
                            "record_count": len(self.records),
                        }
                    ],
                    "role_to_record_shards": {
                        "external_train": [self.shard_relative],
                        "external_val": [self.shard_relative],
                    },
                    "hashes": ledger_relative,
                },
            },
        )
        return payload_sha256

    def _prediction(
        self,
        record: dict[str, object],
        ordinal: int,
    ) -> dict[str, object]:
        provenance = {
            "canonical_build_id": self.build_id,
            "canonical_payload_sha256": self.payload_sha256,
            "renderer": "phase3.render_canonical_messages",
            "gate_b": {
                "protocol_id": GATE_B_PROTOCOL_ID,
                "protocol_sha256": "c" * 64,
                "selection_sha256": "d" * 64,
                "ordinal": ordinal,
                "model_role": "adapter",
                "source": record["source"],
                "shard_path": self.shard_relative,
                "line_index": ordinal,
                "template_version": QWEN_TEMPLATE_VERSION,
            },
        }
        return generic_prediction_row(
            record_id=str(record["record_id"]),
            parent_id=str(record["parent_id"]),
            logical_role=str(record["logical_role"]),
            task_family=str(record["task_family"]),
            generated_text=f"prediction {ordinal}",
            reference_responses=record["reference_responses"],
            provenance=provenance,
        )


class GateBMediaLocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.fixture = _GateBMediaFixture(self.base)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _locate(self, line_number: int):
        return locate_gate_b_media(
            self.fixture.predictions,
            line_number=line_number,
            benchmark_root=self.fixture.root,
        )

    def test_single_prepost_and_bbox_return_only_persistent_media(self) -> None:
        for line_number, name in enumerate(
            ("single", "prepost", "bbox"),
            1,
        ):
            with self.subTest(name=name):
                actual = self._locate(line_number)
                expected = self.fixture.assets[name]
                self.assertEqual(
                    [(value.role, value.path) for value in actual],
                    expected,
                )
                self.assertTrue(all(value.path.is_absolute() for value in actual))
        bbox = self._locate(3)
        self.assertEqual([value.role for value in bbox], ["image"])
        self.assertNotIn("overlay", str(bbox[0].path))
        self.assertNotIn("crop", str(bbox[0].path))

    def test_invalid_line_numbers_and_malformed_jsonl_are_rejected(self) -> None:
        for line_number in (0, -1, 4):
            with self.subTest(line_number=line_number):
                with self.assertRaises(PredictionError):
                    self._locate(line_number)

        for name, payload in (
            ("empty.jsonl", "\n"),
            ("invalid.jsonl", "{not-json}\n"),
        ):
            path = self.base / name
            path.write_text(payload, encoding="utf-8")
            with self.subTest(name=name):
                with self.assertRaises(PredictionError):
                    locate_gate_b_media(
                        path,
                        line_number=1,
                        benchmark_root=self.fixture.root,
                    )

    def test_non_gate_b_prediction_and_record_mismatch_are_rejected(self) -> None:
        non_gate = self.base / "non-gate.jsonl"
        atomic_write_jsonl(non_gate, ({"record_id": "not-gate-b"},))
        with self.assertRaises(PredictionError):
            locate_gate_b_media(
                non_gate,
                line_number=1,
                benchmark_root=self.fixture.root,
            )

        changed = copy.deepcopy(self.fixture.prediction_rows[0])
        changed["record_id"] = "record_" + "f" * 64
        mismatch = self.base / "record-mismatch.jsonl"
        atomic_write_jsonl(mismatch, (changed,))
        with self.assertRaises(PredictionError) as caught:
            locate_gate_b_media(
                mismatch,
                line_number=1,
                benchmark_root=self.fixture.root,
            )
        self.assertEqual(caught.exception.code, ReasonCode.PREDICTION_INVALID)

    def test_manifest_identity_drift_is_rejected_before_payload_lookup(self) -> None:
        manifest_path = self.fixture.root / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["build_id"] = "build_" + "e" * 64
        atomic_write_json(manifest_path, manifest)
        with self.assertRaises(PredictionError) as caught:
            self._locate(1)
        self.assertEqual(
            caught.exception.code,
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
        )

    def test_shard_byte_tamper_is_rejected(self) -> None:
        shard = self.fixture.root / self.fixture.shard_relative
        with shard.open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaises(RSGeneralDescError) as caught:
            self._locate(1)
        self.assertEqual(caught.exception.code, Phase3ReasonCode.HASH_MISMATCH)

    def test_asset_byte_tamper_is_rejected(self) -> None:
        asset = self.fixture.assets["single"][0][1]
        asset.write_bytes(b"tampered image bytes")
        with self.assertRaises(RSGeneralDescError) as caught:
            self._locate(1)
        self.assertEqual(caught.exception.code, Phase3ReasonCode.HASH_MISMATCH)

    def test_missing_asset_is_rejected(self) -> None:
        asset = self.fixture.assets["single"][0][1]
        asset.unlink()
        with self.assertRaises(RSGeneralDescError) as caught:
            self._locate(1)
        self.assertEqual(caught.exception.code, Phase3ReasonCode.ASSET_MISSING)

    def test_symlink_asset_is_rejected(self) -> None:
        asset = self.fixture.assets["single"][0][1]
        replacement = self.base / "replacement.png"
        replacement.write_bytes(asset.read_bytes())
        asset.unlink()
        asset.symlink_to(replacement)
        with self.assertRaises(RSGeneralDescError) as caught:
            self._locate(1)
        self.assertEqual(caught.exception.code, Phase3ReasonCode.OUTPUT_LINK)

    def test_shard_path_escape_is_rejected(self) -> None:
        changed = copy.deepcopy(self.fixture.prediction_rows[0])
        changed["provenance"]["gate_b"]["shard_path"] = "../escape.jsonl"
        escaped = self.base / "escaped.jsonl"
        atomic_write_jsonl(escaped, (changed,))
        with self.assertRaises(RSGeneralDescError) as caught:
            locate_gate_b_media(
                escaped,
                line_number=1,
                benchmark_root=self.fixture.root,
            )
        self.assertEqual(caught.exception.code, Phase3ReasonCode.PATH_ESCAPE)

    def test_cli_stdout_stderr_and_exit_codes_follow_contract(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = entrypoint(
                (
                    "gate-b-locate-media",
                    "--predictions",
                    str(self.fixture.predictions),
                    "--line-number",
                    "1",
                    "--benchmark-root",
                    str(self.fixture.root),
                )
            )
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            f"image\t{self.fixture.assets['single'][0][1]}\n",
        )

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = entrypoint(
                (
                    "gate-b-locate-media",
                    "--predictions",
                    str(self.fixture.predictions),
                    "--line-number",
                    "0",
                    "--benchmark-root",
                    str(self.fixture.root),
                )
            )
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        error = json.loads(stderr.getvalue())
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["reason_code"], ReasonCode.TYPE_MISMATCH.value)


if __name__ == "__main__":
    unittest.main()
