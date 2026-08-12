"""OA-GroundRAG Instruction-Routed Unified Inference Core。"""

from .contracts import (
    BenchmarkSampleRef,
    ExecutionPlan,
    InMemorySpatialInput,
    RegionSelection,
    RegionSource,
    ResponseKind,
    UnifiedInferenceError,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedTask,
)
from .router import CapabilityRouter

__all__ = [
    "BenchmarkSampleRef",
    "CapabilityRouter",
    "ExecutionPlan",
    "InMemorySpatialInput",
    "RegionSelection",
    "RegionSource",
    "ResponseKind",
    "UnifiedInferenceError",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedTask",
]
