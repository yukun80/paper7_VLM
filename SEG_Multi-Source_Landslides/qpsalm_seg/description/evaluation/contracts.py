"""Versioned contracts shared by description evaluation subsystems."""

DESCRIPTION_EVALUATION_PROTOCOL = (
    "qpsalm_description_evaluation_v17_structured_decoder_bound"
)
EVALUATION_PUBLICATION_PROTOCOL = (
    "qpsalm_description_evaluation_publication_v1_artifact_bound"
)
EVALUATION_MASK_ARTIFACT_PROTOCOL = (
    "qpsalm_description_evaluation_mask_artifact_v1_binary_npy"
)
EVALUATION_MASK_INVENTORY_PROTOCOL = (
    "qpsalm_description_evaluation_mask_inventory_v1_role_bound"
)
EVALUATION_CHECKPOINT_BINDING_PROTOCOL = (
    "qpsalm_description_evaluation_checkpoint_binding_v5_run_completion_bound"
)
SAME_IMAGE_RETRIEVAL_PROTOCOL = "qpsalm_same_image_region_retrieval_v2_parent_ranked"
COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL = (
    "qpsalm_counterfactual_input_change_v1_state_fingerprinted"
)
EVALUATION_POPULATION_FIELDS = (
    "sample_id",
    "parent_sample_id",
    "task_family",
    "target_status",
    "source_dataset",
    "visual_image_path",
    "region_pair_id",
    "region_id",
    "region_source",
    "source_region_aliases",
    "region_mask_path",
    "split",
    "evaluation_mode",
    "instruction",
    "target_text",
    "reference_texts",
    "has_unavailable_modality",
    "end_to_end_segmentation_target",
    "region_input_mask_artifact",
    "region_input_source_binding",
)
