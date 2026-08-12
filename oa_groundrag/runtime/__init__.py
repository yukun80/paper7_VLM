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
from .inference import UnifiedInferenceRuntime

__all__ = [
    "BenchmarkSampleRef",
    "CapabilityRouter",
    "ExecutionPlan",
    "InMemorySpatialInput",
    "RegionSelection",
    "RegionSource",
    "ResponseKind",
    "UnifiedInferenceError",
    "UnifiedInferenceRuntime",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedTask",
]
