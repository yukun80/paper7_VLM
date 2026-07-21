"""Canonical language-parent materialization and split-isolation tests."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sami_gsd.contracts.canonical import LicenseRecord
from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.contracts.language import (
    DescriptionSourceRecord,
    LanguageAnswer,
    LanguageImageRef,
)
from sami_gsd.data.adapters.formats import read_image_header
from sami_gsd.data.builder import build_canonical_benchmark
from sami_gsd.data.validation import validate_benchmark_payload, validate_published_benchmark
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from tests.p1.test_builder_validation import synthetic_build_config
from tests.p1.test_materialization import spatial_input
from tests.p1.test_source_adapters import write_png


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INDEX_SHA = "9" * 64


def reviewed_license(source_key: str) -> LicenseRecord:
    """Return a permissive reviewed language-source license for fixtures."""

    return LicenseRecord(
        source_key=source_key,
        license_status="verified",
        license_name="CC-BY-4.0",
        license_url_or_document=f"licenses/{source_key}.txt",
        allowed_for_training=True,
        allowed_for_evaluation=True,
        allowed_for_redistribution=False,
        academic_only=True,
        attribution=f"Synthetic {source_key} language fixture.",
        reviewed_by="test-suite",
        review_date="2026-07-21",
    )


def component_license(source_key: str, component: str) -> LicenseRecord:
    """Return the exact fixture policy, keeping RSIEval test-only."""

    license_record = reviewed_license(source_key)
    if component == "rsieval":
        return license_record.model_copy(update={"allowed_for_training": False})
    return license_record


def language_build_config() -> BenchmarkAuditConfig:
    """Extend the spatial synthetic config with licensed language sources."""

    payload = synthetic_build_config().model_dump(mode="json")
    component_sets = {
        "mmrs_1m": ("rsicd", "ucm", "sydney", "nwpu", "rsitmd", "dior_rsvg"),
        "rsgpt": ("rsicap", "rsieval"),
    }
    for source_key, local_path in (("mmrs_1m", "MMRS-1M"), ("rsgpt", "RSGPT")):
        aggregate = reviewed_license(source_key).model_copy(
            update={
                "allowed_for_training": False,
                "allowed_for_evaluation": False,
                "allowed_for_redistribution": False,
            }
        )
        payload["sources"].append(
            {
                "source_key": source_key,
                "display_name": f"Synthetic {source_key}",
                "local_path": local_path,
                "enabled": True,
                "allowed_task_roles": ["inventory"],
                "license": aggregate.model_dump(mode="json"),
                "language_components": [
                    {
                        "component": component,
                        "component_key": f"{source_key}:{component}",
                        "allowed_task_roles": [
                            "language_region" if component == "dior_rsvg" else "language_global"
                        ],
                        "split_policy": (
                            "permanent_test_only" if component == "rsieval" else "train_candidate"
                        ),
                        "license": component_license(source_key, component).model_dump(mode="json"),
                    }
                    for component in component_sets[source_key]
                ],
            }
        )
    return BenchmarkAuditConfig.model_validate(payload)


def add_png_text_chunk(source: Path, destination: Path) -> None:
    """Re-encode identical PNG pixels by adding one harmless ancillary chunk."""

    raw = source.read_bytes()
    chunk_type = b"tEXt"
    data = b"fixture=perceptual-duplicate"
    chunk = struct.pack(">I", len(data)) + chunk_type + data
    chunk += struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw[:-12] + chunk + raw[-12:])


def description_record(
    *,
    record_id: str,
    source_key: str,
    component: str,
    role: str,
    image_path: Path,
    logical_path: str,
    text: str,
    split_policy: str = "train_candidate",
    training_eligible: bool = True,
    box: tuple[float, float, float, float] | None = None,
) -> DescriptionSourceRecord:
    """Build one strict source row from a real synthetic image."""

    header = read_image_header(image_path)
    return DescriptionSourceRecord.model_validate(
        {
            "schema_version": "sami_description_source_v2_component_license_bound",
            "record_id": record_id,
            "source_key": source_key,
            "component": component,
            "component_license_key": f"{source_key}:{component}",
            "source_group_id": f"group/{record_id}",
            "role": role,
            "split_policy": split_policy,
            "image": LanguageImageRef(
                logical_path=logical_path,
                sha256=sha256_file(image_path),
                native_hw=(header.height, header.width),
            ).model_dump(mode="json"),
            "answers": [
                LanguageAnswer(
                    answer_id=f"answer/{record_id}",
                    text=text,
                    annotation_origin="source_expression" if role == "region_short_phrase" else "source_caption",
                    index_logical_path=f"datasets/indexes/{record_id}.json",
                    index_sha256=INDEX_SHA,
                ).model_dump(mode="json")
            ],
            "normalized_box_xyxy": box,
            "license": component_license(source_key, component).model_dump(mode="json"),
            "training_eligible": training_eligible,
        }
    )


class CanonicalLanguageBuildTests(unittest.TestCase):
    """Verify one visual parent, test-priority clusters and runtime-only paths."""

    def test_language_images_materialize_once_and_rsieval_forces_duplicate_cluster_test(self) -> None:
        """Exact rows share a parent; perceptual test duplicates cannot leak to train."""

        config = language_build_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            shared = datasets / "MMRS-1M/shared.png"
            write_png(shared, height=3, width=4)
            rsieval = datasets / "RSGPT/rsieval.png"
            add_png_text_chunk(shared, rsieval)
            noise = datasets / "MMRS-1M/noise.png"
            noise.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.random.default_rng(7).integers(0, 256, size=(5, 6, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(noise, format="PNG")

            records = (
                description_record(
                    record_id="mmrs/shared-a",
                    source_key="mmrs_1m",
                    component="rsicd",
                    role="global_caption",
                    image_path=shared,
                    logical_path="datasets/MMRS-1M/shared.png",
                    text="First exact-image caption.",
                ),
                description_record(
                    record_id="mmrs/shared-b",
                    source_key="mmrs_1m",
                    component="rsitmd",
                    role="global_caption",
                    image_path=shared,
                    logical_path="datasets/MMRS-1M/shared.png",
                    text="Second exact-image caption.",
                ),
                description_record(
                    record_id="rsgpt/rsieval",
                    source_key="rsgpt",
                    component="rsieval",
                    role="global_caption",
                    image_path=rsieval,
                    logical_path="datasets/RSGPT/rsieval.png",
                    text="Permanent test caption.",
                    split_policy="permanent_test_only",
                    training_eligible=False,
                ),
                description_record(
                    record_id="mmrs/noise-caption",
                    source_key="mmrs_1m",
                    component="nwpu",
                    role="global_caption",
                    image_path=noise,
                    logical_path="datasets/MMRS-1M/noise.png",
                    text="A separate training image.",
                ),
                description_record(
                    record_id="mmrs/noise-region",
                    source_key="mmrs_1m",
                    component="dior_rsvg",
                    role="region_short_phrase",
                    image_path=noise,
                    logical_path="datasets/MMRS-1M/noise.png",
                    text="the short target phrase",
                    box=(0.1, 0.2, 0.8, 0.9),
                ),
            )
            language_license_sha256 = sha256_bytes(
                canonical_json_bytes(component_license("mmrs_1m", "nwpu").model_dump(mode="json"))
            )
            noise_parent = (
                f"language-mmrs_1m-{sha256_file(noise)[:16]}-{language_license_sha256[:8]}"
            )
            first_root = root / "build-one"
            second_root = root / "build-two"
            first = build_canonical_benchmark(
                config,
                parent_inputs=(spatial_input(),),
                description_records=records,
                output_dir=first_root,
                schemas_root=REPOSITORY_ROOT / "schemas",
                datasets_root=datasets,
                forced_splits={noise_parent: "train"},
            )
            second = build_canonical_benchmark(
                config,
                parent_inputs=(spatial_input(),),
                description_records=tuple(reversed(records)),
                output_dir=second_root,
                schemas_root=REPOSITORY_ROOT / "schemas",
                datasets_root=datasets,
                forced_splits={noise_parent: "train"},
            )
            self.assertEqual(first["aggregate_sha256"], second["aggregate_sha256"])
            self.assertEqual(first["output_sha256"], second["output_sha256"])
            replay = validate_published_benchmark(first_root, schemas_root=REPOSITORY_ROOT / "schemas")
            self.assertEqual(replay["errors"], [])
            self.assertEqual(replay["canonical_description_count"], 5)

            rows = [json.loads(line) for line in (first_root / "descriptions/all.jsonl").read_text().splitlines()]
            by_id = {row["record_id"]: row for row in rows}
            self.assertEqual(by_id["mmrs/shared-a"]["parent_id"], by_id["mmrs/shared-b"]["parent_id"])
            self.assertEqual(by_id["mmrs/shared-a"]["split"], "test")
            self.assertEqual(by_id["rsgpt/rsieval"]["split"], "test")
            self.assertEqual(by_id["mmrs/noise-caption"]["split"], "train")
            self.assertIsNotNone(by_id["mmrs/noise-region"]["region_box_half_open"])
            self.assertTrue(all(not row["image_ref"]["path"].startswith("datasets/") for row in rows))
            self.assertEqual(len(list(first_root.glob("assets/language-*"))), 3)

            train_rows = [
                json.loads(line)
                for line in (first_root / "descriptions/train_eligible.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["record_id"] for row in train_rows}, {"mmrs/noise-caption", "mmrs/noise-region"})
            t2_rows = sum(
                len((first_root / f"tasks/t2_referring/{split}.jsonl").read_text().splitlines())
                for split in ("train", "val", "test")
            )
            self.assertEqual(t2_rows, 1)

            registry_path = first_root / "manifests/source_registry.yaml"
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            mmrs = next(entry for entry in registry["entries"] if entry["source_key"] == "mmrs_1m")
            rsicd = next(
                component
                for component in mmrs["language_components"]
                if component["component"] == "rsicd"
            )
            rsicd["attribution"] = "Tampered after publication."
            registry_path.write_text(yaml.safe_dump(registry, sort_keys=True), encoding="utf-8")
            tampered = validate_benchmark_payload(first_root, schemas_root=REPOSITORY_ROOT / "schemas")
            self.assertTrue(
                any(
                    error == "description_component_license_mismatch:mmrs/shared-a"
                    for error in tampered["errors"]
                )
            )

    def test_unapproved_language_rows_remain_audit_only_without_raw_decode(self) -> None:
        """A licensed spatial build may retain, but never materialize, denied language rows."""

        payload = language_build_config().model_dump(mode="json")
        mmrs_source = next(source for source in payload["sources"] if source["source_key"] == "mmrs_1m")
        rsicd_policy = next(
            component
            for component in mmrs_source["language_components"]
            if component["component"] == "rsicd"
        )
        rsicd_policy["license"]["allowed_for_training"] = False
        rsicd_policy["license"]["allowed_for_evaluation"] = False
        config = BenchmarkAuditConfig.model_validate(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_after_selection = root / "selected.png"
            write_png(missing_after_selection)
            approved = description_record(
                record_id="mmrs/audit-only",
                source_key="mmrs_1m",
                component="rsicd",
                role="global_caption",
                image_path=missing_after_selection,
                logical_path="datasets/MMRS-1M/missing-after-selection.png",
                text="Audit-only caption.",
            )
            denied_payload = approved.model_dump(mode="json")
            denied_payload["license"] = rsicd_policy["license"]
            denied_payload["training_eligible"] = False
            denied = DescriptionSourceRecord.model_validate(denied_payload)
            missing_after_selection.unlink()

            output = root / "build"
            build_canonical_benchmark(
                config,
                parent_inputs=(spatial_input(),),
                description_records=(denied,),
                output_dir=output,
                schemas_root=REPOSITORY_ROOT / "schemas",
            )
            self.assertEqual((output / "descriptions/all.jsonl").read_text(), "")
            self.assertEqual(len((output / "manifests/description_source_subset.jsonl").read_text().splitlines()), 1)
            replay = validate_published_benchmark(output, schemas_root=REPOSITORY_ROOT / "schemas")
            self.assertEqual(replay["errors"], [])


if __name__ == "__main__":
    unittest.main()
