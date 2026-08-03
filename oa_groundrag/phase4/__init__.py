"""算法 Phase 3 / 仓库 phase4：RS-VLM。"""

from .checkpoint import CheckpointManager, TrainingCursor
from .contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    FAILURE_SCHEMA_VERSION,
    GATE_B_GENERATION_SCHEMA_VERSION,
    GATE_B_PROTOCOL_SCHEMA_VERSION,
    GATE_B_REPORT_SCHEMA_VERSION,
    GATE_B_SELECTION_SCHEMA_VERSION,
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
    inventory_from_auxseg_inference,
    locate_bounded_external_records,
)
from .evaluation import evaluate_predictions
from .gate_b_acceptance import (
    GateBAcceptanceVerification,
    verify_gate_b_acceptance,
)
from .gate_b_evaluation import evaluate_gate_b
from .gate_b_generation import generate_gate_b
from .gate_b_media import GateBMediaPath, locate_gate_b_media
from .gate_b_selection import prepare_gate_b
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
    "GATE_B_GENERATION_SCHEMA_VERSION",
    "GATE_B_PROTOCOL_SCHEMA_VERSION",
    "GATE_B_REPORT_SCHEMA_VERSION",
    "GATE_B_SELECTION_SCHEMA_VERSION",
    "GateBAcceptanceVerification",
    "GateBMediaPath",
    "MODEL_OUTPUT_SCHEMA_VERSION",
    "MaskMode",
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
    "evaluate_gate_b",
    "evaluate_teacher_forced_loss",
    "inspect_benchmark_identity",
    "inventory_from_auxseg_inference",
    "generate_gate_b",
    "load_config",
    "locate_gate_b_media",
    "locate_bounded_external_records",
    "parse_model_output",
    "prepare_gate_b",
    "run_bounded_external_smoke",
    "run_inference",
    "run_preflight",
    "select_bounded_external_validation",
    "serialize_model_output",
    "verify_gate_b_acceptance",
]
