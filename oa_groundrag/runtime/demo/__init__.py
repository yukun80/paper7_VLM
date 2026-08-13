"""OA-GroundRAG 只读 Unified Demo Workbench。"""

from .access import (
    DemoAuthorizedSpatialInput,
    DemoInferenceAccess,
    DemoTestAccessController,
    DemoTestAccessReceipt,
)
from .app import create_demo_app, serve_demo
from .catalog import (
    BenchmarkCatalog,
    BenchmarkFilter,
    BenchmarkRecord,
    FrozenEvaluationCatalog,
    FrozenEvaluationItem,
)
from .config import DemoConfig, load_demo_config
from .gallery import DemoGalleryEntry, DemoGalleryStore
from .i18n import DEFAULT_LOCALE, DemoI18nError, MessageSpec, tr
from .previews import InputChannelPreview
from .runner import (
    DemoCandidateKind,
    DemoCandidateSelection,
    DemoRunSummary,
    DemoSpatialSnapshot,
    DemoTaskResult,
    UnifiedDemoRunner,
)

__all__ = [
    "BenchmarkCatalog",
    "BenchmarkFilter",
    "BenchmarkRecord",
    "DemoAuthorizedSpatialInput",
    "DemoConfig",
    "DemoCandidateKind",
    "DemoCandidateSelection",
    "DemoGalleryEntry",
    "DemoGalleryStore",
    "DemoI18nError",
    "DemoInferenceAccess",
    "DemoRunSummary",
    "DemoSpatialSnapshot",
    "DemoTaskResult",
    "DemoTestAccessController",
    "DemoTestAccessReceipt",
    "FrozenEvaluationCatalog",
    "FrozenEvaluationItem",
    "InputChannelPreview",
    "MessageSpec",
    "UnifiedDemoRunner",
    "create_demo_app",
    "DEFAULT_LOCALE",
    "load_demo_config",
    "serve_demo",
    "tr",
]
