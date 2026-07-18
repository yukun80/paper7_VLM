#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable public surface for Description Vision Cache v1."""

from __future__ import annotations

from ..protocols.cache import description_cache_key
from .vision_cache_bank import (
    DescriptionVisionFeatureBank,
    revalidate_description_cache_artifact,
)
from .vision_cache_contracts import (
    DESCRIPTION_CACHE_ARTIFACT_BINDING_PROTOCOL,
    DESCRIPTION_CACHE_ARTIFACT_REVALIDATION_PROTOCOL,
    DESCRIPTION_CACHE_BUILDER_VERSION,
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL,
    DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
    sha256_file,
    source_cache_snapshot,
    validate_description_cache_record,
    validate_source_cache_snapshot,
)


__all__ = [
    "DESCRIPTION_CACHE_ARTIFACT_BINDING_PROTOCOL",
    "DESCRIPTION_CACHE_ARTIFACT_REVALIDATION_PROTOCOL",
    "DESCRIPTION_CACHE_BUILDER_VERSION",
    "DESCRIPTION_CACHE_FORMAT",
    "DESCRIPTION_CACHE_PROTOCOL",
    "DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL",
    "DESCRIPTION_CACHE_VALIDATION_PROTOCOL",
    "DescriptionVisionFeatureBank",
    "description_cache_key",
    "revalidate_description_cache_artifact",
    "sha256_file",
    "source_cache_snapshot",
    "validate_description_cache_record",
    "validate_source_cache_snapshot",
]

