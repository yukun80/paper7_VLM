"""Stage 4A Landslide Evidence Corpus 算法核心。"""

from .contracts import (
    CONFIG_SCHEMA, MANIFEST_SCHEMA, RECORD_SCHEMA, LandslideEvidenceError,
)
from .pipeline import BuildResult, build_auto
from .grounded_eval import build_eval_dev
from .region_contracts import (
    EVAL_MANIFEST_SCHEMA,
    REGION_MANIFEST_SCHEMA,
    REGION_RECORD_SCHEMA,
    RepresentationMode,
)
from .region_pipeline import RegionBuildResult, build_region_corpus
from .region_validation import validate_eval_dev, validate_region_corpus
from .validation import validate_corpus

__all__ = [
    "BuildResult", "CONFIG_SCHEMA", "EVAL_MANIFEST_SCHEMA", "LandslideEvidenceError",
    "MANIFEST_SCHEMA", "RECORD_SCHEMA", "REGION_MANIFEST_SCHEMA", "REGION_RECORD_SCHEMA",
    "RegionBuildResult", "RepresentationMode", "build_auto", "build_eval_dev",
    "build_region_corpus", "validate_corpus", "validate_eval_dev", "validate_region_corpus",
]
