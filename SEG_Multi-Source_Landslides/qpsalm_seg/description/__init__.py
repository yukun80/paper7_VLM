"""Small lazy public surface for segmentation-grounded description.

Importing a low-level submodule must not initialize MGRR, Qwen, training or evaluation.
The stable convenience names below are therefore resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_EXPORTS = {
    "MultisourceBackboneState": (
        "qpsalm_seg.schema", "MultisourceBackboneState",
    ),
    "SegmentationState": ("qpsalm_seg.schema", "SegmentationState"),
    "RegionEvidenceState": ("qpsalm_seg.schema", "RegionEvidenceState"),
    "MultiGranularityRegionReplay": (".modeling.mgrr", "MultiGranularityRegionReplay"),
    "SingleVectorRegionPooling": (".modeling.region_baselines", "SingleVectorRegionPooling"),
    "rasterize_region_geometry": (".modeling.mgrr", "rasterize_region_geometry"),
    "retarget_region_mask_between_cache_views": (
        ".protocols.region_geometry",
        "retarget_region_mask_between_cache_views",
    ),
    "transform_region_mask_to_cache": (
        ".protocols.region_geometry",
        "transform_region_mask_to_cache",
    ),
}


def __getattr__(name: str) -> Any:
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    *_PUBLIC_EXPORTS,
]
