"""算法 Phase 3 / 仓库 phase4：Mask-Grounded VLM Description。"""

from .checkpoint import CheckpointManager, TrainingCursor
from .contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    FAILURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    AlignmentStatus,
    AuxiliaryView,
    Claim,
    DataMode,
    EvidenceBundle,
    EvidenceSufficiency,
    MaskMode,
    RegionCandidate,
    RegionInventory,
    SelectedRegion,
    SelectionMode,
    SelectionRequest,
    StructuredModelOutput,
    TargetStatus,
)
from .evidence import EvidenceBuilder, deterministic_mask_facts
from .config import Phase4Config, apply_runtime_overrides, load_config
from .data import (
    DescriptionSample,
    ExternalDescriptionDataset,
    MaskGroundedDescriptionDataset,
    MaskGroundedExample,
    inventory_from_auxseg_inference,
    inventory_from_canonical_oa_item,
    locate_bounded_external_records,
)
from .evaluation import evaluate_predictions
from .inference import run_inference
from .messages import build_mask_grounded_messages
from .model import Qwen3VLModelAdapter
from .outputs import parse_model_output, serialize_model_output
from .preflight import inspect_benchmark_identity, run_preflight
from .processing import DescriptionCollator, Qwen3VLProcessorAdapter
from .reference import MAIN_REFERENCE, ReferenceProject
from .regions import RegionSelector
from .smoke import run_bounded_external_smoke
from .trainer import DescriptionTrainer
from .validation import (
    ValidationResult,
    ValidationSelection,
    evaluate_teacher_forced_loss,
    select_bounded_external_validation,
)

__all__ = [
    "AlignmentStatus",
    "AuxiliaryView",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManager",
    "Claim",
    "CONFIG_SCHEMA_VERSION",
    "DataMode",
    "DescriptionCollator",
    "DescriptionSample",
    "DescriptionTrainer",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceBuilder",
    "EvidenceBundle",
    "EvidenceSufficiency",
    "ExternalDescriptionDataset",
    "FAILURE_SCHEMA_VERSION",
    "MODEL_OUTPUT_SCHEMA_VERSION",
    "MaskMode",
    "MaskGroundedDescriptionDataset",
    "MaskGroundedExample",
    "MAIN_REFERENCE",
    "Phase4Config",
    "PREDICTION_SCHEMA_VERSION",
    "Qwen3VLProcessorAdapter",
    "Qwen3VLModelAdapter",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RegionCandidate",
    "RegionInventory",
    "ReferenceProject",
    "RegionSelector",
    "SelectedRegion",
    "SelectionMode",
    "SelectionRequest",
    "StructuredModelOutput",
    "TargetStatus",
    "TrainingCursor",
    "ValidationResult",
    "ValidationSelection",
    "apply_runtime_overrides",
    "build_mask_grounded_messages",
    "deterministic_mask_facts",
    "evaluate_predictions",
    "evaluate_teacher_forced_loss",
    "inspect_benchmark_identity",
    "inventory_from_auxseg_inference",
    "inventory_from_canonical_oa_item",
    "load_config",
    "locate_bounded_external_records",
    "parse_model_output",
    "run_bounded_external_smoke",
    "run_inference",
    "run_preflight",
    "select_bounded_external_validation",
    "serialize_model_output",
]
