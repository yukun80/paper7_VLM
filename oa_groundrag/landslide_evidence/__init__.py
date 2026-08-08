"""Stage 4A Landslide Evidence Corpus 算法核心。"""

from .contracts import (
    CONFIG_SCHEMA, MANIFEST_SCHEMA, RECORD_SCHEMA, LandslideEvidenceError,
)
from .pipeline import BuildResult, build_auto
from .grounded_eval import build_eval_dev
from .expanded_region import (
    build_region_extension,
    build_train_collection,
    prepare_expanded_region_assets,
    validate_expanded_region_collection,
    validate_region_extension,
)
from .region_contracts import (
    EVAL_MANIFEST_SCHEMA,
    REGION_MANIFEST_SCHEMA,
    REGION_RECORD_SCHEMA,
    RepresentationMode,
)
from .region_pipeline import RegionBuildResult, build_region_corpus
from .region_validation import (
    validate_eval_dev,
    validate_region_asset_files,
    validate_region_corpus,
)
from .single_expert import (
    DRAFT_CONFIG_SCHEMA,
    MODEL_DRAFT_FAILURE_SCHEMA,
    VERIFIED_ANNOTATION_SCHEMA,
    create_annotation_project,
)
from .single_expert_package import (
    VERIFIED_PACKAGE_SCHEMA,
    export_verified_annotations,
    validate_verified_annotation_package,
)
from .single_expert_training import (
    MaskGroundedTrainingMessageDataset,
    TRAINING_MESSAGE_SCHEMA,
    export_training_messages,
    load_training_message_artifact,
)
from .single_expert_workflow import (
    TRAIN_WORKFLOW_SCHEMA,
    TRAIN_WORKFLOW_STATE_SCHEMA,
    TrainWorkflowPaths,
    run_train_annotation_workflow,
)
from .model_assisted_workflow import (
    ModelAssistedWorkflowPaths,
    prepare_expanded_corpus,
    run_model_assisted_train_workflow,
)
from .validation import validate_corpus

__all__ = [
    "BuildResult", "CONFIG_SCHEMA", "EVAL_MANIFEST_SCHEMA", "LandslideEvidenceError",
    "MANIFEST_SCHEMA", "RECORD_SCHEMA", "REGION_MANIFEST_SCHEMA", "REGION_RECORD_SCHEMA",
    "DRAFT_CONFIG_SCHEMA", "MODEL_DRAFT_FAILURE_SCHEMA",
    "MaskGroundedTrainingMessageDataset", "ModelAssistedWorkflowPaths",
    "RegionBuildResult", "RepresentationMode",
    "TRAINING_MESSAGE_SCHEMA", "VERIFIED_ANNOTATION_SCHEMA", "VERIFIED_PACKAGE_SCHEMA",
    "TRAIN_WORKFLOW_SCHEMA", "TRAIN_WORKFLOW_STATE_SCHEMA", "TrainWorkflowPaths",
    "build_auto", "build_eval_dev", "build_region_corpus", "build_region_extension",
    "build_train_collection",
    "create_annotation_project", "export_training_messages", "export_verified_annotations",
    "load_training_message_artifact",
    "prepare_expanded_corpus", "prepare_expanded_region_assets",
    "run_model_assisted_train_workflow", "run_train_annotation_workflow",
    "validate_corpus", "validate_eval_dev", "validate_region_asset_files",
    "validate_expanded_region_collection", "validate_region_corpus",
    "validate_region_extension",
    "validate_verified_annotation_package",
]
