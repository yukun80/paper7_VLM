"""Stage 4A Landslide Evidence Corpus 算法核心。"""

from .contracts import (
    CONFIG_SCHEMA, MANIFEST_SCHEMA, RECORD_SCHEMA, LandslideEvidenceError,
)
from .pipeline import BuildResult, build_auto
from .validation import validate_corpus

__all__ = [
    "BuildResult", "CONFIG_SCHEMA", "LandslideEvidenceError", "MANIFEST_SCHEMA", "RECORD_SCHEMA",
    "build_auto", "validate_corpus",
]
