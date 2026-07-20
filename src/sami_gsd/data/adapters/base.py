"""Public protocol for source-specific, audit-only P1.3 adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from sami_gsd.contracts.canonical import StrictModel
from sami_gsd.contracts.config import SourceConfig
from sami_gsd.contracts.sources import SourceSampleProjection


class SourceAdapterError(ValueError):
    """Raised when one source cannot be projected without guessing."""


class AdapterDescriptor(StrictModel):
    """Stable registry metadata for exactly one configured source."""

    source_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    adapter_version: str = Field(min_length=1)
    implementation_status: Literal["implemented", "blocked"]
    supported_record_types: tuple[
        Literal["spatial_mask", "derived_spatial_mask", "global_language", "region_language"], ...
    ]
    blockers: tuple[str, ...]


@runtime_checkable
class SourceAdapter(Protocol):
    """Unique extraction boundary from a raw source to strict audit records."""

    @property
    def descriptor(self) -> AdapterDescriptor:
        """Return immutable registry metadata without touching the source root."""

    def extract_samples(
        self,
        source_root: Path,
        source_config: SourceConfig,
        *,
        limit: int,
    ) -> tuple[SourceSampleProjection, ...]:
        """Read at most ``limit`` deterministic records and never modify raw bytes."""


__all__ = ["AdapterDescriptor", "SourceAdapter", "SourceAdapterError"]
