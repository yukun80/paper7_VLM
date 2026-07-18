"""Stable, dependency-light contracts shared by SegDesc subsystems."""

from .io import (
    NonFiniteJSONError,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    strict_json_loads,
)
from .stages import DESCRIPTION_STAGES, StageSpec, get_stage_spec
from .cache import description_cache_key
from .region_geometry import (
    restore_region_mask_from_cache,
    retarget_region_mask_between_cache_views,
    transform_region_mask_to_cache,
)

__all__ = [
    "DESCRIPTION_STAGES",
    "NonFiniteJSONError",
    "StageSpec",
    "append_jsonl",
    "atomic_write_json",
    "atomic_write_jsonl",
    "canonical_sha256",
    "description_cache_key",
    "get_stage_spec",
    "read_json",
    "read_jsonl",
    "restore_region_mask_from_cache",
    "retarget_region_mask_between_cache_views",
    "sha256_file",
    "strict_json_loads",
    "transform_region_mask_to_cache",
]
