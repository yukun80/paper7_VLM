#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmentation-grounded region description modules."""

from .vision_cache import (
    DESCRIPTION_CACHE_FORMAT,
    DescriptionVisionFeatureBank,
    description_cache_key,
)
from .mgrr import MultiGranularityRegionReplay, rasterize_region_geometry
from .region_baselines import SingleVectorRegionPooling
from .model import (
    DESCRIPTION_ADAPTER_NAME,
    DESCRIPTION_SEQUENCE_PROTOCOL,
    DescriptionForwardOutput,
    SegmentationGroundedDescriptionModel,
    SegmentThenDescribeOutput,
)
from .output_protocol import ParsedDescription, deterministic_repair, parse_description_output
from .checkpoint import (
    SEGDESC_CHECKPOINT_FORMAT,
    initialize_segdesc_checkpoint,
    load_segdesc_checkpoint,
    migrate_segmentation_checkpoint,
    save_segdesc_checkpoint,
)
from .backbone import DescriptionCacheBackboneEncoder, transform_region_mask_to_cache
from .data import DescriptionTaskDataset, collate_description
from .config import (
    DESCRIPTION_EVAL_MODES,
    DESCRIPTION_STAGES,
    SegDescConfig,
    load_segdesc_config,
)
from .runtime import (
    build_description_optimizer,
    build_segdesc_model,
    description_parameter_groups,
    description_trainable_parameter_manifest,
)
from .metrics import DescriptionMetricAccumulator, retrieval_metrics
from .counterfactuals import COUNTERFACTUAL_MODES
from .expert_factuality import aggregate_expert_factuality, build_expert_review_template
from .oof import OOF_FOLD_FORMAT, build_oof_fold_indexes, load_oof_manifest

__all__ = [
    "DESCRIPTION_CACHE_FORMAT",
    "DescriptionVisionFeatureBank",
    "description_cache_key",
    "MultiGranularityRegionReplay",
    "rasterize_region_geometry",
    "SingleVectorRegionPooling",
    "DESCRIPTION_ADAPTER_NAME",
    "DESCRIPTION_SEQUENCE_PROTOCOL",
    "DescriptionForwardOutput",
    "SegmentationGroundedDescriptionModel",
    "SegmentThenDescribeOutput",
    "ParsedDescription",
    "deterministic_repair",
    "parse_description_output",
    "SEGDESC_CHECKPOINT_FORMAT",
    "initialize_segdesc_checkpoint",
    "load_segdesc_checkpoint",
    "migrate_segmentation_checkpoint",
    "save_segdesc_checkpoint",
    "DescriptionCacheBackboneEncoder",
    "transform_region_mask_to_cache",
    "DescriptionTaskDataset",
    "collate_description",
    "DESCRIPTION_STAGES",
    "DESCRIPTION_EVAL_MODES",
    "SegDescConfig",
    "load_segdesc_config",
    "build_description_optimizer",
    "build_segdesc_model",
    "description_parameter_groups",
    "description_trainable_parameter_manifest",
    "DescriptionMetricAccumulator",
    "retrieval_metrics",
    "COUNTERFACTUAL_MODES",
    "aggregate_expert_factuality",
    "build_expert_review_template",
    "OOF_FOLD_FORMAT",
    "build_oof_fold_indexes",
    "load_oof_manifest",
]
