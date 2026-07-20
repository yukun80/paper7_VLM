"""The sole deterministic registry for all configured P1.3 sources."""

from __future__ import annotations

from pathlib import Path

from sami_gsd.contracts.config import SourceConfig
from sami_gsd.contracts.sources import SourceSampleProjection
from sami_gsd.data.adapters.base import AdapterDescriptor, SourceAdapter, SourceAdapterError
from sami_gsd.data.adapters.implemented import (
    GDCLDAdapter,
    LMHLDAdapter,
    LandslideBenchAdapter,
    MMRSAdapter,
    RSGPTAdapter,
)


class BlockedSourceAdapter:
    """Registered source whose live structure cannot yet be interpreted uniquely."""

    def __init__(self, descriptor: AdapterDescriptor) -> None:
        self.descriptor = descriptor

    def extract_samples(
        self,
        source_root: Path,
        source_config: SourceConfig,
        *,
        limit: int,
    ) -> tuple[SourceSampleProjection, ...]:
        """Refuse extraction instead of falling back to a legacy reader."""

        del source_root, source_config, limit
        raise SourceAdapterError(
            f"source adapter {self.descriptor.source_key!r} is blocked: {', '.join(self.descriptor.blockers)}"
        )


class SourceAdapterRegistry:
    """Unique source-key registry with duplicate and fallback rejection."""

    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}

    def register(self, adapter: SourceAdapter) -> None:
        """Register one adapter and reject a duplicate key immediately."""

        key = adapter.descriptor.source_key
        if key in self._adapters:
            raise ValueError(f"duplicate source adapter registration: {key}")
        self._adapters[key] = adapter

    def get(self, source_key: str) -> SourceAdapter:
        """Return the exact adapter; unknown keys have no generic fallback."""

        try:
            return self._adapters[source_key]
        except KeyError as error:
            raise KeyError(f"no source adapter registered for {source_key!r}") from error

    def keys(self) -> tuple[str, ...]:
        """Return stable lexical registry keys."""

        return tuple(sorted(self._adapters))

    def descriptors(self) -> tuple[AdapterDescriptor, ...]:
        """Return descriptors in the same stable key order."""

        return tuple(self._adapters[key].descriptor for key in self.keys())


def _blocked(source_key: str, *blockers: str) -> BlockedSourceAdapter:
    """Create one explicit blocked descriptor with no extraction fallback."""

    return BlockedSourceAdapter(
        AdapterDescriptor(
            source_key=source_key,
            adapter_version="sami_source_adapter_p1_3_blocked_v1",
            implementation_status="blocked",
            supported_record_types=(),
            blockers=tuple(blockers),
        )
    )


def build_source_adapter_registry() -> SourceAdapterRegistry:
    """Build the frozen nine-source registry from independent greenfield code."""

    registry = SourceAdapterRegistry()
    for adapter in (
        GDCLDAdapter(),
        LMHLDAdapter(),
        _blocked(
            "sen12_landslides",
            "fifteen_step_temporal_slice_policy_unresolved",
            "sample_annotated_false",
            "pre_post_metadata_present_but_frozen_task_is_single_time",
        ),
        _blocked(
            "landslide4sense",
            "hdf5_metadata_reader_not_declared_in_greenfield_runtime",
            "official_source_license_unresolved",
        ),
        _blocked(
            "multimodal_landslide",
            "geotiff_metadata_reader_not_declared_in_greenfield_runtime",
            "insar_units_and_sign_convention_unresolved",
            "official_source_license_unresolved",
        ),
        LandslideBenchAdapter(),
        MMRSAdapter(),
        RSGPTAdapter(),
        _blocked(
            "disasterm3",
            "pre_post_change_task_excluded_by_frozen_scope",
            "inventory_only_no_canonical_candidate",
        ),
    ):
        registry.register(adapter)
    return registry


__all__ = ["BlockedSourceAdapter", "SourceAdapterRegistry", "build_source_adapter_registry"]
