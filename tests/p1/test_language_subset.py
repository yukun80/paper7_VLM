"""Frozen MMRS/RSGPT language-subset tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from sami_gsd.contracts.config import BenchmarkAuditConfig, load_audit_config
from sami_gsd.contracts.language import DescriptionSourceRecord
from sami_gsd.data.language_subset import build_description_subset
from tests.p1.test_source_adapters import write_png


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_language_fixture(root: Path) -> Path:
    """Create one record for each frozen component and an invalid excluded index."""

    datasets = root / "datasets"
    mmrs = datasets / "MMRS-1M"
    definitions = (
        ("rsicd", "caption_rsicd.json"),
        ("ucm", "caption_ucm.json"),
        ("sydney", "caption_syndney.json"),
        ("nwpu", "caption_nwpu.json"),
        ("rsitmd", "caption_rsitmd.json"),
    )
    for component, filename in definitions:
        image_relative = Path("caption") / component / "images" / f"{component}.png"
        write_png(mmrs / image_relative)
        index_path = mmrs / "json/caption" / filename
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(
                [
                    {
                        "image": f"data/{image_relative.as_posix()}",
                        "conversations": [
                            {"from": "human", "value": "Describe."},
                            {"from": "gpt", "value": f"A {component} scene."},
                        ],
                    }
                ],
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    dior_image = Path("RSVG/images/dior.png")
    write_png(mmrs / dior_image)
    rsvg_path = mmrs / "json/RSVG/rsvg_trainval.json"
    rsvg_path.parent.mkdir(parents=True, exist_ok=True)
    rsvg_path.write_text(
        json.dumps(
            [
                {
                    "image": f"data/{dior_image.as_posix()}",
                    "conversations": [
                        {"from": "human", "value": "Find region: [0.1, 0.2, 0.8, 0.9]"},
                        {"from": "gpt", "value": "the short target phrase"},
                    ],
                }
            ],
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (mmrs / "json/total.json").write_text("THIS EXCLUDED FILE IS NOT JSON", encoding="utf-8")

    rsgpt = datasets / "RSGPT/dataset"
    for directory, filename, field in (
        ("RSICap", "cap.png", "text_output"),
        ("RSIEval", "eval.png", "caption"),
    ):
        write_png(rsgpt / directory / "images" / filename)
        index_name = "captions.json" if directory == "RSICap" else "annotations.json"
        index_path = rsgpt / directory / index_name
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(
                {"annotations": [{"filename": filename, field: f"A {directory} scene."}]},
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return datasets


class LanguageSubsetTests(unittest.TestCase):
    """Verify exact source selection, provenance and fail-closed eligibility."""

    def test_frozen_eight_components_are_repeatable_and_exclusions_are_unread(self) -> None:
        """Five captions, DIOR phrase, RSICap and RSIEval are the only inputs."""

        with tempfile.TemporaryDirectory() as directory:
            datasets = build_language_fixture(Path(directory))
            config = load_audit_config(REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml")
            first = build_description_subset(config, datasets_root=datasets, limit_per_component=1)
            second = build_description_subset(config, datasets_root=datasets, limit_per_component=1)
            self.assertEqual(first["aggregate_sha256"], second["aggregate_sha256"])
            self.assertEqual(first["record_count"], 8)
            self.assertEqual(set(first["components"].values()), {1})
            self.assertEqual(first["training_eligible_count"], 0)
            self.assertEqual(first["permanent_test_only_count"], 1)
            self.assertEqual(first["excluded_inputs_read"], [])
            records = [DescriptionSourceRecord.model_validate(record) for record in first["records"]]
            dior = next(record for record in records if record.component == "dior_rsvg")
            rsieval = next(record for record in records if record.component == "rsieval")
            self.assertEqual(dior.role, "region_short_phrase")
            self.assertEqual(len(dior.answers), 1)
            self.assertEqual(rsieval.split_policy, "permanent_test_only")
            self.assertTrue(all(not Path(record.image.logical_path).is_absolute() for record in records))

    def test_unlicensed_record_cannot_be_promoted_and_dior_cannot_be_global_caption(self) -> None:
        """License and scientific-role constraints are enforced by the public contract."""

        with tempfile.TemporaryDirectory() as directory:
            datasets = build_language_fixture(Path(directory))
            config = load_audit_config(REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml")
            report = build_description_subset(config, datasets_root=datasets, limit_per_component=1)
            dior = next(record for record in report["records"] if record["component"] == "dior_rsvg")
            promoted = dict(dior)
            promoted["training_eligible"] = True
            with self.assertRaisesRegex(ValidationError, "approved non-test source"):
                DescriptionSourceRecord.model_validate(promoted)
            wrong_role = dict(dior)
            wrong_role["role"] = "global_caption"
            wrong_role["normalized_box_xyxy"] = None
            with self.assertRaisesRegex(ValidationError, "sole region-short-phrase"):
                DescriptionSourceRecord.model_validate(wrong_role)

    def test_one_component_approval_promotes_only_that_component(self) -> None:
        """Partial owner approval cannot leak through the aggregate MMRS container."""

        with tempfile.TemporaryDirectory() as directory:
            datasets = build_language_fixture(Path(directory))
            payload = yaml.safe_load(
                (REPOSITORY_ROOT / "configs/benchmark_v3_small.yaml").read_text(encoding="utf-8")
            )
            mmrs = next(source for source in payload["sources"] if source["source_key"] == "mmrs_1m")
            rsicd = next(
                component
                for component in mmrs["language_components"]
                if component["component"] == "rsicd"
            )
            rsicd["license"].update(
                {
                    "license_status": "verified",
                    "license_name": "synthetic-rsicd-approval",
                    "license_url_or_document": "licenses/rsicd.txt",
                    "allowed_for_training": True,
                    "allowed_for_evaluation": True,
                    "reviewed_by": "test-suite",
                    "review_date": "2026-07-21",
                }
            )
            config = BenchmarkAuditConfig.model_validate(payload)
            report = build_description_subset(config, datasets_root=datasets, limit_per_component=1)
            promoted = [record for record in report["records"] if record["training_eligible"]]
            self.assertEqual([record["component"] for record in promoted], ["rsicd"])
            self.assertNotIn("license_not_approved:mmrs_1m:rsicd", report["warnings"])
            self.assertIn("license_not_approved:mmrs_1m:ucm", report["warnings"])


if __name__ == "__main__":
    unittest.main()
