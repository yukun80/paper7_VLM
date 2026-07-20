"""Synthetic P1.2 reference-canvas and spatial primitive tests."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from sami_gsd.contracts.canonical import CanonicalParentV3, ReferenceCanvas, TransformStep
from sami_gsd.contracts.spatial import ReferenceCanvasCandidate
from sami_gsd.data.reference_canvas import (
    ReferenceSelectionError,
    require_spatial_task_eligibility,
    select_reference_canvas,
)
from sami_gsd.data.transforms import (
    SpatialTransformError,
    apply_binary_transform,
    apply_image_transform,
    build_transform_chain,
    crop_step,
    deserialize_qwen1000_box,
    forward_box,
    forward_point,
    inverse_box,
    inverse_point,
    pad_step,
    qwen_round_trip_error,
    resize_step,
    serialize_qwen1000_box,
    transform_mask_and_valid,
)
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes
from tests.p1.conftest import canonical_parent_payload


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def candidate(
    modality_id: str,
    *,
    mask_grid: str,
    coverage: float = 1.0,
    gsd: float | None = 10.0,
    language: bool = False,
    inverse: bool = True,
) -> ReferenceCanvasCandidate:
    """Build one strict reference candidate for a synthetic parent."""

    return ReferenceCanvasCandidate(
        modality_id=modality_id,
        original_hw=(32, 48),
        mask_grid=mask_grid,
        annotation_origin=None if mask_grid == "none" else "official",
        valid_coverage=coverage,
        native_gsd_m=gsd,
        single_image_language=language,
        coordinate_inverse_available=inverse,
    )


def spatial_trace_payload() -> dict[str, object]:
    """Return the deterministic synthetic trace bound into the P1.2 report."""

    candidates = [
        candidate("sar-20m", mask_grid="registered", gsd=20.0),
        candidate("optical-10m", mask_grid="registered", gsd=10.0),
        candidate("partial-1m", mask_grid="registered", coverage=0.75, gsd=1.0),
    ]
    decision = select_reference_canvas(list(reversed(candidates)))
    chain = build_transform_chain(
        (
            crop_step((6, 8), top=1, left=2, height=4, width=4),
            resize_step((4, 4), (8, 8)),
            pad_step((8, 8), top=1, bottom=1, left=2, right=2),
        )
    )
    source_box = (2, 1, 6, 5)
    transformed_box = forward_box(source_box, chain)
    qwen_box = serialize_qwen1000_box((17, 33, 4031, 2000), (2048, 4096))
    recovered_qwen_box = deserialize_qwen1000_box(qwen_box, (2048, 4096))
    mask_valid = transform_mask_and_valid(
        ((1, 1), (1, 1)),
        ((1, 0), (1, 1)),
        (
            resize_step((2, 2), (4, 4)),
            pad_step((4, 4), top=1, bottom=1, left=1, right=1),
        ),
    )
    return {
        "decision": decision.model_dump(mode="json"),
        "transform_chain": [step.model_dump(mode="json") for step in chain],
        "source_box": source_box,
        "transformed_box": transformed_box,
        "inverse_box": inverse_box(transformed_box, chain),
        "qwen_box": qwen_box,
        "qwen_recovered_box": recovered_qwen_box,
        "qwen_error": qwen_round_trip_error((17, 33, 4031, 2000), (2048, 4096)),
        "mask_valid": asdict(mask_valid),
    }


class ReferenceCanvasSelectionTests(unittest.TestCase):
    """Verify the frozen reference priority and explicit eligibility gate."""

    def test_native_mask_has_priority_and_order_does_not_matter(self) -> None:
        """A native official/human mask wins over a finer registered grid."""

        values = [
            candidate("registered-1m", mask_grid="registered", gsd=1.0),
            candidate("native-20m", mask_grid="native", gsd=20.0),
        ]
        first = select_reference_canvas(values)
        second = select_reference_canvas(list(reversed(values)))
        self.assertEqual(first, second)
        self.assertEqual(first.reference_modality_id, "native-20m")
        self.assertEqual(first.selection_rule, "authoritative_native_mask")

    def test_registered_grid_prefers_complete_coverage_then_finest_gsd(self) -> None:
        """A partial finer grid cannot displace a complete registered grid."""

        values = [
            candidate("partial-1m", mask_grid="registered", coverage=0.9, gsd=1.0),
            candidate("full-20m", mask_grid="registered", gsd=20.0),
            candidate("full-10m", mask_grid="registered", gsd=10.0),
        ]
        decision = select_reference_canvas(values)
        self.assertEqual(decision.reference_modality_id, "full-10m")
        self.assertEqual(decision.selection_rule, "registered_mask_complete_finest_gsd")
        self.assertTrue(decision.spatial_tasks_eligible)

    def test_mask_ambiguity_fails_closed(self) -> None:
        """Multiple incomplete or incomparable registered grids are unresolved."""

        with self.assertRaisesRegex(ReferenceSelectionError, "complete valid coverage"):
            select_reference_canvas(
                [
                    candidate("a", mask_grid="registered", coverage=0.8, gsd=1.0),
                    candidate("b", mask_grid="registered", coverage=0.9, gsd=2.0),
                ]
            )
        with self.assertRaisesRegex(ReferenceSelectionError, "comparable native GSD"):
            select_reference_canvas(
                [
                    candidate("a", mask_grid="registered", gsd=None),
                    candidate("b", mask_grid="registered", gsd=2.0),
                ]
            )

    def test_single_language_image_and_spatial_ineligibility_are_explicit(self) -> None:
        """Language-only selection never silently enters a T1--T4 task."""

        decision = select_reference_canvas(
            [candidate("language-image", mask_grid="none", gsd=None, language=True, inverse=False)]
        )
        self.assertEqual(decision.selection_rule, "single_image_original")
        self.assertFalse(decision.spatial_tasks_eligible)
        self.assertEqual(decision.spatial_exclusion_reason, "coordinate_inverse_unavailable")
        with self.assertRaisesRegex(ReferenceSelectionError, "T1--T4 are forbidden"):
            require_spatial_task_eligibility(decision)

    def test_unresolved_and_duplicate_candidate_sets_are_rejected(self) -> None:
        """Reference selection does not invent a grid or accept duplicate IDs."""

        with self.assertRaisesRegex(ReferenceSelectionError, "no authoritative"):
            select_reference_canvas([candidate("plain", mask_grid="none", gsd=None)])
        duplicate = candidate("same", mask_grid="native")
        with self.assertRaisesRegex(ReferenceSelectionError, "must be unique"):
            select_reference_canvas([duplicate, duplicate])


class TransformChainTests(unittest.TestCase):
    """Verify coordinate, raster and public-record transform contracts."""

    def setUp(self) -> None:
        """Create one crop-resize-pad chain with an exact coordinate inverse."""

        self.chain = build_transform_chain(
            (
                crop_step((6, 8), top=1, left=2, height=4, width=4),
                resize_step((4, 4), (8, 8)),
                pad_step((8, 8), top=1, bottom=1, left=2, right=2),
            )
        )

    def test_point_and_half_open_box_round_trip(self) -> None:
        """Pixel-edge geometry round-trips exactly on retained valid content."""

        point = (3.0, 2.0)
        transformed_point = forward_point(point, self.chain)
        self.assertEqual(transformed_point, (4.0, 3.0))
        inverse = inverse_point(transformed_point, self.chain)
        self.assertTrue(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(inverse, point, strict=True)))

        box = (2.0, 1.0, 6.0, 5.0)
        transformed_box = forward_box(box, self.chain)
        self.assertEqual(transformed_box, (2.0, 1.0, 10.0, 9.0))
        recovered_box = inverse_box(transformed_box, self.chain)
        self.assertTrue(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(recovered_box, box, strict=True)))

    def test_padding_has_no_inverse_outside_valid_content(self) -> None:
        """A padded coordinate is not allowed to fabricate a source location."""

        with self.assertRaisesRegex(SpatialTransformError, "padded coordinate"):
            inverse_point((0.5, 0.5), self.chain)

    def test_chain_continuity_and_step_policy_are_strict(self) -> None:
        """Discontinuous grids and ambiguous resize interpolation are rejected."""

        with self.assertRaisesRegex(SpatialTransformError, "discontinuous"):
            build_transform_chain((resize_step((2, 2), (4, 4)), resize_step((5, 5), (10, 10))))
        with self.assertRaises(ValidationError):
            TransformStep(
                operation="resize",
                input_hw=(2, 2),
                output_hw=(4, 4),
                interpolation="not_applicable",
                invertible=True,
                parameters={"coordinate_mapping": "pixel_edges", "raster_sampling": "half_pixel_centers"},
            )

    def test_reference_canvas_binds_chain_endpoints_and_inverse_flag(self) -> None:
        """The canonical canvas cannot disagree with its transform trace."""

        canvas = ReferenceCanvas(
            reference_modality_id="optical",
            coordinate_space="reference_pixel_half_open",
            original_hw=(6, 8),
            canvas_hw=(10, 12),
            valid_mask_path="assets/parent/valid.npy",
            transform_chain=self.chain,
            inverse_transform_available=True,
            crs=None,
            geotransform=None,
        )
        self.assertTrue(canvas.inverse_transform_available)
        with self.assertRaisesRegex(ValidationError, "endpoints"):
            ReferenceCanvas(
                reference_modality_id="optical",
                coordinate_space="reference_pixel_half_open",
                original_hw=(7, 8),
                canvas_hw=(10, 12),
                valid_mask_path="assets/parent/valid.npy",
                transform_chain=self.chain,
                inverse_transform_available=True,
                crs=None,
                geotransform=None,
            )

    def test_image_bilinear_and_binary_nearest_are_distinct(self) -> None:
        """Images use fixed bilinear sampling while mask/valid remain binary."""

        step = (resize_step((2, 2), (4, 4)),)
        image = apply_image_transform((((0.0,), (10.0,)), ((20.0,), (30.0,))), step)
        mask = apply_binary_transform(((0, 1), (1, 0)), step, kind="mask")
        self.assertEqual((len(image), len(image[0]), len(image[0][0])), (4, 4, 1))
        self.assertAlmostEqual(image[1][1][0], 7.5)
        self.assertEqual(mask, ((0, 0, 1, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 0)))
        self.assertEqual({value for row in mask for value in row}, {0, 1})
        with self.assertRaisesRegex(SpatialTransformError, "binary"):
            apply_binary_transform(((0.0, 1.0), (1.0, 0.0)), step, kind="valid")  # type: ignore[arg-type]
        with self.assertRaisesRegex(SpatialTransformError, "explicit int or float"):
            apply_image_transform((((False,), (1.0,)), ((2.0,), (3.0,))), step)

    def test_padding_and_nodata_are_excluded_from_evidence(self) -> None:
        """Nearest propagation retains nodata and zero pad never becomes valid."""

        steps = (
            resize_step((2, 2), (4, 4)),
            pad_step((4, 4), top=1, bottom=1, left=1, right=1),
        )
        result = transform_mask_and_valid(((1, 1), (1, 1)), ((1, 0), (1, 1)), steps)
        self.assertEqual((len(result.mask), len(result.mask[0])), (6, 6))
        self.assertEqual(result.total_pixel_count, 36)
        self.assertEqual(result.valid_pixel_count, 12)
        self.assertEqual(result.excluded_pixel_count, 24)
        self.assertEqual(result.positive_valid_pixel_count, 12)
        self.assertTrue(all(value == 0 for value in result.valid[0]))
        self.assertTrue(all(value == 0 for value in result.valid[-1]))
        self.assertEqual(sum(value for row in result.effective_mask for value in row), 12)


class BboxBoundaryTests(unittest.TestCase):
    """Verify half-open reference and integer Qwen-1000 boundary conversion."""

    def test_qwen_round_trip_preserves_coverage_with_bounded_error(self) -> None:
        """Quantized recovery covers the source box within one declared bin."""

        canvas_hw = (2048, 4096)
        source = (17, 33, 4031, 2000)
        serialized = serialize_qwen1000_box(source, canvas_hw)
        recovered = deserialize_qwen1000_box(serialized, canvas_hw)
        self.assertTrue(0 <= serialized[0] < serialized[2] <= 1000)
        self.assertTrue(0 <= serialized[1] < serialized[3] <= 1000)
        self.assertLessEqual(recovered[0], source[0])
        self.assertLessEqual(recovered[1], source[1])
        self.assertGreaterEqual(recovered[2], source[2])
        self.assertGreaterEqual(recovered[3], source[3])
        error = qwen_round_trip_error(source, canvas_hw)
        self.assertTrue(all(value >= 0 for value in error))
        self.assertEqual(
            deserialize_qwen1000_box(serialize_qwen1000_box((0, 0, 4096, 2048), canvas_hw), canvas_hw),
            (0, 0, 4096, 2048),
        )

    def test_qwen_boundary_rejects_non_integer_or_invalid_boxes(self) -> None:
        """The serialization boundary never guesses dtype or coordinate order."""

        with self.assertRaisesRegex(SpatialTransformError, "integers"):
            deserialize_qwen1000_box((False, 0, 100, 100), (32, 48))
        with self.assertRaisesRegex(SpatialTransformError, "outside canvas"):
            serialize_qwen1000_box((0, 0, 49, 32), (32, 48))
        with self.assertRaisesRegex(SpatialTransformError, "must be integers"):
            serialize_qwen1000_box((0.0, 0, 48, 32), (32, 48))  # type: ignore[arg-type]
        with self.assertRaisesRegex(SpatialTransformError, "positive integer"):
            serialize_qwen1000_box((0, 0, 1, 1), (0, 48))

    def test_qwen_round_trip_bound_holds_across_canvas_scales(self) -> None:
        """Representative sub- and super-1000 grids satisfy the same contract."""

        sizes = (1, 2, 7, 999, 1000, 1001, 2048, 4096)
        for height in sizes:
            for width in sizes:
                boxes = {
                    (0, 0, width, height),
                    (0, 0, 1, 1),
                    (width - 1, height - 1, width, height),
                    (width // 2, height // 2, min(width, width // 2 + 1), min(height, height // 2 + 1)),
                }
                for box in boxes:
                    with self.subTest(canvas=(height, width), box=box):
                        qwen_round_trip_error(box, (height, width))


class SpatialSchemaAndDeterminismTests(unittest.TestCase):
    """Bind static schemas and deterministic traces to the P1.2 behavior."""

    def test_static_and_python_contracts_accept_audited_transform_chain(self) -> None:
        """The canonical schema records crop/resize/pad without ambiguity."""

        payload = canonical_parent_payload()
        chain = (
            crop_step((32, 48), top=2, left=4, height=28, width=40),
            resize_step((28, 40), (56, 80)),
            pad_step((56, 80), top=4, bottom=4, left=8, right=8),
        )
        payload["reference_canvas"]["canvas_hw"] = [64, 96]
        payload["reference_canvas"]["transform_chain"] = [step.model_dump(mode="json") for step in chain]
        CanonicalParentV3.model_validate(payload)
        schema = json.loads(
            (REPOSITORY_ROOT / "schemas" / "canonical_parent_v3.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)

        ambiguous = json.loads(json.dumps(payload))
        ambiguous["reference_canvas"]["transform_chain"][1]["interpolation"] = "bilinear"
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(ambiguous)

    def test_synthetic_trace_hash_is_repeatable(self) -> None:
        """Independent construction produces byte-identical canonical evidence."""

        first = sha256_bytes(canonical_json_bytes(spatial_trace_payload()))
        second = sha256_bytes(canonical_json_bytes(spatial_trace_payload()))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
