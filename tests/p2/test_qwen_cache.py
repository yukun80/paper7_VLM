"""Official native multi-image wrapper, spatial-state and strict-cache tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sami_gsd.model.cache import QwenBackboneCache, compare_backbone_states
from sami_gsd.model.qwen_backbone import QwenBackboneWrapper
from sami_gsd.model.sensor_adapter import SensorAwareMultiImageAdapter
from tests.p2.conftest import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    loaded_parent,
    model_config,
)


class QwenCacheTests(unittest.TestCase):
    """Prove one complete forward, reconstructable grids and transparent memoization."""

    def _wrapper(self) -> tuple[QwenBackboneWrapper, Qwen3VLForConditionalGeneration]:
        model = Qwen3VLForConditionalGeneration()
        wrapper = QwenBackboneWrapper(
            model_config(),
            model=model,
            processor=Qwen3VLProcessor(),
            device=torch.device("cpu"),
            model_fingerprint="1" * 64,
            processor_fingerprint="2" * 64,
            qwen_code_revision="3" * 64,
        )
        return wrapper, model

    def _batch(self):
        parent = loaded_parent(
            (
                ("opt-ref", "optical", "present_valid"),
                ("sar-a", "sar", "present_partial_valid"),
                ("dem-z", "dem", "present_valid"),
            ),
            reference_id="opt-ref",
        )
        adapter = SensorAwareMultiImageAdapter(model_config())
        return adapter.prepare((parent,), (("dem-z", "sar-a", "opt-ref"),))

    def test_native_multi_image_uses_exactly_one_full_model_forward(self) -> None:
        """All effective views enter one official model call, not per-view controller calls."""

        wrapper, model = self._wrapper()
        state = wrapper.encode(self._batch(), return_spatial_features=True)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(state.view_order, (("opt-ref", "sar-a", "dem-z"),))
        self.assertEqual(len(state.views), 3)
        for view in state.views:
            self.assertEqual(tuple(view.language_aligned_visual_tokens.shape), (4, 8))
            self.assertEqual(view.transform.processor_grid_thw, (1, 4, 4))
            self.assertEqual(view.transform.merged_grid_hw, (2, 2))
            self.assertEqual(tuple(view.valid_mask.shape), (2, 2))
            self.assertEqual(
                [tuple(level.features.shape) for level in view.spatial_features],
                [(8, 2, 2), (8, 2, 2), (8, 2, 2)],
            )

    def test_optical_sar_and_terrain_only_all_forward(self) -> None:
        """The official input contract forwards all required single-family cases."""

        parents = tuple(
            loaded_parent(
                ((f"{family}-ref", family, "present_valid"),),
                reference_id=f"{family}-ref",
                parent_id=f"synthetic-{family}-parent",
            )
            for family in ("optical", "sar", "dem")
        )
        active = tuple((parent.record.reference_canvas.reference_modality_id,) for parent in parents)
        batch = SensorAwareMultiImageAdapter(model_config()).prepare(parents, active)
        wrapper, model = self._wrapper()
        state = wrapper.encode(batch, return_spatial_features=True)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(state.parent_ids, tuple(parent.record.parent_id for parent in parents))
        self.assertEqual(state.view_order, (("optical-ref",), ("sar-ref",), ("dem-ref",)))
        self.assertEqual([view.sensor_card.family for view in state.views], ["optical", "sar", "dem"])

    def test_optional_cache_hit_skips_forward_and_is_strictly_equivalent(self) -> None:
        """The second identical request reopens bytes and never calls Qwen again."""

        wrapper, model = self._wrapper()
        batch = self._batch()
        with tempfile.TemporaryDirectory() as directory:
            cache = QwenBackboneCache(
                Path(directory), schema_version="sami_qwen_backbone_cache_v1"
            )
            online = wrapper.encode(batch, return_spatial_features=True, cache_store=cache)
            cached = wrapper.encode(batch, return_spatial_features=True, cache_store=cache)
            self.assertEqual(model.forward_count, 1)
            self.assertFalse(online.from_cache)
            self.assertTrue(cached.from_cache)
            result = compare_backbone_states(online, cached, model_config().cache.equivalence)
            self.assertTrue(result["passed"])
            self.assertEqual(result["maximum_abs_difference"], 0.0)

    def test_cache_key_changes_with_active_subset_and_spatial_flag(self) -> None:
        """View identity and requested state surface are bound into memoization identity."""

        wrapper, model = self._wrapper()
        full = self._batch()
        parent = loaded_parent(
            (
                ("opt-ref", "optical", "present_valid"),
                ("sar-a", "sar", "present_partial_valid"),
                ("dem-z", "dem", "present_valid"),
            ),
            reference_id="opt-ref",
        )
        subset = SensorAwareMultiImageAdapter(model_config()).prepare(
            (parent,), (("opt-ref", "sar-a"),)
        )
        full_state = wrapper.encode(full, return_spatial_features=True)
        subset_state = wrapper.encode(subset, return_spatial_features=True)
        no_spatial = wrapper.encode(subset, return_spatial_features=False)
        self.assertEqual(model.forward_count, 3)
        self.assertEqual(len({full_state.cache_key, subset_state.cache_key, no_spatial.cache_key}), 3)
        self.assertEqual(no_spatial.views[0].spatial_features, ())

    def test_cache_byte_corruption_is_rejected(self) -> None:
        """A write-once entry with changed tensor bytes cannot become a silent hit."""

        wrapper, _ = self._wrapper()
        batch = self._batch()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = QwenBackboneCache(root, schema_version="sami_qwen_backbone_cache_v1")
            online = wrapper.encode(batch, return_spatial_features=True, cache_store=cache)
            state_path = root / online.cache_key[:2] / online.cache_key / "state.pt"
            with state_path.open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "state bytes"):
                wrapper.encode(batch, return_spatial_features=True, cache_store=cache)


if __name__ == "__main__":
    unittest.main()
