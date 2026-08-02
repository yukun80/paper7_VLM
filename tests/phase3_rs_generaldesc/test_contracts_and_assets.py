from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from fixture_helpers import build_config_dict, make_all_sources, make_image
from oa_groundrag.phase3.assets import (
    AssetStore,
    normalize_bbox_target,
    normalized_image_sha256,
)
from oa_groundrag.phase3.common import portable_relative_path, read_json
from oa_groundrag.phase3.config import load_build_config, load_export_config
from oa_groundrag.phase3.contracts import (
    AdapterResult,
    AnnotationLayer,
    InputLayout,
    LogicalRole,
    MediaType,
    OutputModality,
    PendingAsset,
    ReviewStatus,
    SourceExample,
    SupervisionKind,
    TaskFamily,
    parent_id,
    record_id,
    validate_role_layer,
)
from oa_groundrag.phase3.errors import AssetError, ConfigError, ReasonCode, SchemaError
from oa_groundrag.phase3.splitter import assign_external_splits


REPO = Path(__file__).resolve().parents[2]


class ConfigAndIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.roots = make_all_sources(self.base / "sources")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, value: dict) -> Path:
        path = self.base / "config.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        return path

    def test_strict_config_and_unknown_field(self) -> None:
        value = build_config_dict(self.roots, self.base / "out")
        config = load_build_config(self._write(value))
        self.assertEqual(config.profile, "smoke")
        self.assertEqual(len(config.semantic_hash), 64)
        value["unknown"] = 1
        with self.assertRaises(ConfigError) as caught:
            load_build_config(self._write(value))
        self.assertEqual(caught.exception.code, ReasonCode.UNKNOWN_FIELD)

    def test_wrong_type_bool_as_int_and_nonfinite(self) -> None:
        value = build_config_dict(self.roots, self.base / "out")
        value["seed"] = True
        with self.assertRaises(ConfigError):
            load_build_config(self._write(value))
        value = build_config_dict(self.roots, self.base / "out")
        value["asset_policy"]["image_quality"] = float("nan")
        self._write(value)
        with self.assertRaises(SchemaError) as caught:
            load_build_config(self.base / "config.yaml")
        self.assertEqual(caught.exception.code, ReasonCode.NONFINITE_NUMBER)

    def test_duplicate_yaml_json_keys_and_invalid_regex_are_rejected(self) -> None:
        duplicate_yaml = self.base / "duplicate.yaml"
        duplicate_yaml.write_text(
            "schema_version: first\nschema_version: second\n",
            encoding="utf-8",
        )
        with self.assertRaises(ConfigError):
            load_build_config(duplicate_yaml)

        duplicate_json = self.base / "duplicate.json"
        duplicate_json.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
        with self.assertRaises(SchemaError):
            read_json(duplicate_json)

        value = build_config_dict(self.roots, self.base / "out")
        value["text_policy"]["forbidden_patterns"] = ["("]
        with self.assertRaises(ConfigError):
            load_build_config(self._write(value))

    def test_export_config_requires_typed_nonempty_roles(self) -> None:
        value = {
            "schema_version": "rs_generaldesc.qwen_export.v1",
            "profile": "description_multitask.v1",
            "benchmark_root": str(self.base / "benchmark"),
            "output_root": str(self.base / "export"),
            "seed": 7,
            "purpose": "training",
            "roles": [],
            "task_families": [],
            "template_version": "qwen3vl_messages.v2",
        }
        with self.assertRaises(ConfigError):
            load_export_config(self._write(value))
        value["roles"] = ["external_train"]
        with self.assertRaises(ConfigError):
            load_export_config(self._write(value))
        value["roles"] = ["not_a_role"]
        value["task_families"] = ["global_caption"]
        with self.assertRaises(ConfigError) as caught:
            load_export_config(self._write(value))
        self.assertEqual(caught.exception.code, ReasonCode.INVALID_ENUM)

    def test_external_split_and_native_config_are_strict(self) -> None:
        value = build_config_dict(
            self.roots,
            self.base / "external",
        )
        config = load_build_config(self._write(value))
        self.assertNotIn("retired_component", config.sanitized())
        self.assertEqual(config.external_split.validation_percent, 5)
        value["retired_component"] = {"enabled": False}
        with self.assertRaises(ConfigError):
            load_build_config(self._write(value))

        value = build_config_dict(
            self.roots,
            self.base / "bad-percent",
        )
        value["external_split"]["validation_percent"] = True
        with self.assertRaises(ConfigError):
            load_build_config(self._write(value))

    def test_native_full_semantic_identity_is_stable(self) -> None:
        config = load_build_config(
            REPO / "configs/phase3_rs_generaldesc/full.yaml"
        )
        self.assertEqual(
            config.semantic_hash,
            "bb9b00ea44fb1c79e9efdfa00fbf73e8d8e9e5c416b8e9ff48d0c0b550e0d162",
        )

    def test_stable_record_and_parent_ids(self) -> None:
        first = parent_id("rsgpt", "RSICap/a.png")
        second = parent_id("rsgpt", "RSICap/a.png")
        self.assertEqual(first, second)
        self.assertNotEqual(first, parent_id("rsgpt", "RSICap/b.png"))
        identity = {"a": 1, "b": [2, 3]}
        self.assertEqual(record_id(identity), record_id({"b": [2, 3], "a": 1}))

    def test_exact_parent_95_5_split_and_task_coverage(self) -> None:
        config = load_build_config(
            self._write(
                build_config_dict(
                    self.roots,
                    self.base / "split-output",
                    validation_percent=5,
                )
            )
        )
        examples = [
            SourceExample(
                source="rsgpt",
                source_record_id=f"record-{index:03d}",
                source_split="upstream",
                parent_key=f"parent-{index:03d}",
                logical_role=LogicalRole.EXTERNAL_TRAIN,
                task_family=(
                    TaskFamily.GLOBAL_CAPTION
                    if index < 80
                    else TaskFamily.VISUAL_QA
                ),
                supervision_kind=(
                    SupervisionKind.LONG_DESCRIPTION
                    if index < 80
                    else SupervisionKind.SHORT_QA
                ),
                input_layout=InputLayout.SINGLE_IMAGE,
                output_modality=OutputModality.TEXT,
                assets=(),
                target={"type": "none"},
                instruction="Describe.",
                training_responses=("Visible content.",),
                reference_responses=("Visible content.",),
                deterministic_facts={},
                annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                review_status=ReviewStatus.NOT_REQUIRED,
                provenance=({"type": "fixture"},),
            )
            for index in range(100)
        ]
        result = AdapterResult(source="rsgpt", examples=examples)
        summary = assign_external_splits(
            [result],
            config,
            check_content=False,
        )
        self.assertEqual(
            summary["parent_role_counts"],
            {"external_train": 95, "external_val": 5},
        )
        for task in (
            TaskFamily.GLOBAL_CAPTION.value,
            TaskFamily.VISUAL_QA.value,
        ):
            self.assertEqual(
                set(summary["task_role_counts"][task]),
                {"external_train", "external_val"},
            )

    def test_deep_asset_rejection_precedes_external_split(self) -> None:
        config = load_build_config(
            self._write(
                build_config_dict(
                    self.roots,
                    self.base / "deep-filter-output",
                )
            )
        )
        valid_path = self.base / "valid.png"
        corrupt_path = self.base / "corrupt.png"
        make_image(valid_path)
        corrupt_path.write_bytes(b"not-an-image")

        def example(source_id: str, path: Path) -> SourceExample:
            return SourceExample(
                source="rsgpt",
                source_record_id=source_id,
                source_split="upstream",
                parent_key=source_id,
                logical_role=LogicalRole.EXTERNAL_TRAIN,
                task_family=TaskFamily.GLOBAL_CAPTION,
                supervision_kind=SupervisionKind.LONG_DESCRIPTION,
                input_layout=InputLayout.SINGLE_IMAGE,
                output_modality=OutputModality.TEXT,
                assets=(
                    PendingAsset(
                        role="image",
                        media_type=MediaType.IMAGE,
                        extension="png",
                        source_ref=f"fixture/{path.name}",
                        source_path=path,
                    ),
                ),
                target={"type": "none"},
                instruction="Describe.",
                training_responses=("Visible content.",),
                reference_responses=("Visible content.",),
                deterministic_facts={},
                annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                review_status=ReviewStatus.NOT_REQUIRED,
                provenance=({"type": "fixture"},),
            )

        result = AdapterResult(
            source="rsgpt",
            examples=[
                example("valid-parent", valid_path),
                example("corrupt-parent", corrupt_path),
            ],
        )
        summary = assign_external_splits(
            [result],
            config,
            check_content=True,
        )
        self.assertEqual(
            [row.source_record_id for row in result.examples],
            ["valid-parent"],
        )
        self.assertEqual(result.examples[0].logical_role.value, "external_train")
        self.assertEqual(result.audit["deep_rejected_examples"], 1)
        self.assertEqual(result.audit["selected_examples"], 1)
        self.assertEqual(result.skips[0].reason_code, ReasonCode.ASSET_CORRUPT)
        self.assertEqual(
            summary["record_role_counts"],
            {"external_train": 1},
        )
        self.assertEqual(summary["content_fingerprint_error_count"], 1)

    def test_native_role_and_annotation_contract_is_external_only(self) -> None:
        validate_role_layer(
            LogicalRole.EXTERNAL_VAL,
            AnnotationLayer.EXTERNAL_SOURCE,
            ReviewStatus.NOT_REQUIRED,
        )
        with self.assertRaises(ValueError):
            LogicalRole("unsupported_role")

    def test_portable_paths_reject_absolute_and_escape(self) -> None:
        self.assertEqual(
            portable_relative_path("assets/a.png", location="test").as_posix(),
            "assets/a.png",
        )
        for value in ("/tmp/a.png", "../a.png", "a/../../b.png"):
            with self.assertRaises(SchemaError):
                portable_relative_path(value, location="test")


class AssetGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        roots = make_all_sources(self.base / "sources")
        path = self.base / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                build_config_dict(roots, self.base / "out"),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.config = load_build_config(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _store(self) -> AssetStore:
        return AssetStore(
            self.base / "stage",
            policy=self.config.asset_policy,
            limits=self.config.limits,
        )

    def test_exif_bbox_transform_and_normalized_hash_are_synchronized(self) -> None:
        image = self.base / "oriented.jpg"
        make_image(image, size=(3, 2), orientation=6)
        assets = (
            PendingAsset(
                role="image",
                media_type=MediaType.IMAGE,
                extension="jpg",
                source_ref="fixture/image.jpg",
                source_path=image,
            ),
        )
        media, geometry, _ = self._store().materialize(assets)
        image_media = next(row for row in media if row["role"] == "image")
        self.assertEqual((image_media["width"], image_media["height"]), (2, 3))
        target = normalize_bbox_target(
            {
                "type": "bbox",
                "image_role": "image",
                "source_convention": "xywh_pixel_top_left",
                "boxes": [{"source": [0, 0, 1, 1], "label": "x"}],
            },
            geometry,
        )
        box = target["boxes"][0]["canonical_xyxy_norm"]
        self.assertEqual(box, [0.5, 0.0, 1.0, 0.3333333333])
        self.assertEqual(
            normalized_image_sha256(
                assets[0],
                policy=self.config.asset_policy,
            ),
            image_media["sha256"],
        )

    def test_zero_bbox_and_non_image_media_are_rejected(self) -> None:
        image = self.base / "image.png"
        make_image(image)
        store = self._store()
        image_asset = PendingAsset(
            role="image",
            media_type=MediaType.IMAGE,
            extension="png",
            source_ref="fixture/image.png",
            source_path=image,
        )
        _, geometry, _ = store.materialize((image_asset,))
        with self.assertRaises(AssetError) as caught:
            normalize_bbox_target(
                {
                    "type": "bbox",
                    "image_role": "image",
                    "source_convention": "xywh_pixel_top_left",
                    "boxes": [{"source": [1, 1, 0, 2], "label": "bad"}],
                },
                geometry,
            )
        self.assertEqual(caught.exception.code, ReasonCode.BBOX_ZERO_AREA)
        with self.assertRaises(ValueError):
            MediaType("unsupported_media")

    def test_asset_count_limit_is_enforced(self) -> None:
        limited = AssetStore(
            self.base / "limited-stage",
            policy=self.config.asset_policy,
            limits=type(self.config.limits)(
                max_parents=self.config.limits.max_parents,
                max_records=self.config.limits.max_records,
                max_assets=1,
                max_copied_bytes=self.config.limits.max_copied_bytes,
                max_validation_candidates_per_task=10,
                per_task={},
            ),
        )
        first = self.base / "first.png"
        second = self.base / "second.png"
        make_image(first, color=(1, 2, 3))
        make_image(second, color=(4, 5, 6))
        limited.add_image(
            PendingAsset(
                role="first",
                media_type=MediaType.IMAGE,
                extension="png",
                source_ref="fixture/first.png",
                source_path=first,
            )
        )
        with self.assertRaises(AssetError) as caught:
            limited.add_image(
                PendingAsset(
                    role="second",
                    media_type=MediaType.IMAGE,
                    extension="png",
                    source_ref="fixture/second.png",
                    source_path=second,
                )
            )
        self.assertEqual(caught.exception.code, ReasonCode.ASSET_LIMIT_EXCEEDED)

    def test_corrupt_missing_and_symlink_assets(self) -> None:
        corrupt = self.base / "corrupt.png"
        corrupt.write_bytes(b"not-an-image")
        missing = self.base / "missing.png"
        link = self.base / "link.png"
        link.symlink_to(corrupt)
        for path, expected in (
            (corrupt, ReasonCode.ASSET_CORRUPT),
            (missing, ReasonCode.ASSET_MISSING),
            (link, ReasonCode.SOURCE_SYMLINK),
        ):
            with self.assertRaises(AssetError) as caught:
                self._store().add_image(
                    PendingAsset(
                        role="image",
                        media_type=MediaType.IMAGE,
                        extension="png",
                        source_ref=f"fixture/{path.name}",
                        source_path=path,
                    )
                )
            self.assertEqual(caught.exception.code, expected)
