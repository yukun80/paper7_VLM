"""OA-GroundRAG Instruction-Routed Unified Inference Core。"""

from .contracts import (
    BenchmarkSampleRef,
    ExecutionPlan,
    InMemorySpatialInput,
    RegionSelection,
    RegionSource,
    ResponseKind,
    RuntimeAccessContext,
    UnifiedInferenceError,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedTask,
)
from .router import CapabilityRouter
from .inference import UnifiedInferenceRuntime

__all__ = [
    "BenchmarkSampleRef",
    "CapabilityRouter",
    "ExecutionPlan",
    "InMemorySpatialInput",
    "RegionSelection",
    "RegionSource",
    "ResponseKind",
    "RuntimeAccessContext",
    "UnifiedInferenceError",
    "UnifiedInferenceRuntime",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedTask",
]
