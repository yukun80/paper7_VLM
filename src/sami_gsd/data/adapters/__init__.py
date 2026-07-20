"""P1.3 source adapters and the unique fail-closed registry."""

from sami_gsd.data.adapters.audit import audit_source_samples
from sami_gsd.data.adapters.base import AdapterDescriptor, SourceAdapter, SourceAdapterError
from sami_gsd.data.adapters.registry import build_source_adapter_registry

__all__ = [
    "AdapterDescriptor",
    "SourceAdapter",
    "SourceAdapterError",
    "audit_source_samples",
    "build_source_adapter_registry",
]
