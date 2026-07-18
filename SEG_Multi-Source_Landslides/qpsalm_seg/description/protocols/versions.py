"""Cross-layer protocol identifiers with more than one owning consumer."""

DESCRIPTION_TRAINING_COMPLETION_PROTOCOL = (
    "qpsalm_description_training_completion_v3_checkpoint_replayed"
)
JOINT_TRAINING_COMPLETION_PROTOCOL = (
    "qpsalm_segdesc_joint_training_completion_v3_checkpoint_replayed"
)
CHECKPOINT_RUN_COMPLETION_PROTOCOL = (
    "qpsalm_segdesc_checkpoint_run_completion_v1_selection_role_bound"
)
D0_PREFLIGHT_PROTOCOL = (
    "qpsalm_d0_preflight_v6_region_route_bound"
)
D0_PREFLIGHT_ACCEPTANCE_PROTOCOL = (
    "qpsalm_d0_preflight_acceptance_v6_region_route_consumed"
)
D0_CONSTRUCTION_CONTRACT_PROTOCOL = (
    "qpsalm_d0_construction_contract_v2_region_route_replayed"
)
STRICT_RELOAD_PROBE_PROTOCOL = (
    "qpsalm_segdesc_strict_reload_probe_v2_stateful"
)
DESCRIPTION_GRADIENT_GATE_PROTOCOL = (
    "qpsalm_description_gradient_gate_v4_window_homogeneous"
)
DESCRIPTION_COLLATOR_AUDIT_PROTOCOL = (
    "qpsalm_description_collator_audit_v3_output_format_region_route_separated"
)
STRUCTURED_GENERATION_PROTOCOL = (
    "qpsalm_description_structured_generation_v2_token_stream_bound"
)
DESCRIPTION_SEQUENCE_PROTOCOL = (
    "qpsalm_description_causal_v5_stage_separated_schema_ordered"
)

# D-1 training and evaluation both bind these immutable protocol assets. Keep
# their inventory below both layers so neither side imports the other's module.
DESCRIPTION_PROTOCOL_ASSETS = (
    "configs/description_ontology_v1.yaml",
    "configs/qpsalm_description_record_v2.schema.json",
    "configs/qpsalm_description_output_v1.schema.json",
)
D_MINUS_ONE_ACCEPTANCE_PROTOCOL = (
    "qpsalm_d_minus_one_acceptance_v11_structured_decoder_bound"
)
D_MINUS_ONE_GATE_PROTOCOL = (
    "qpsalm_d_minus_one_engineering_gate_v13_structured_decoder_bound"
)
D_MINUS_ONE_OVERFIT_PROTOCOL = (
    "qpsalm_d_minus_one_overfit_validation_v10_structured_decoder_bound"
)
D_MINUS_ONE_OVERFIT_PROTOCOL_ASSET_SOURCES = {
    "description_ontology": DESCRIPTION_PROTOCOL_ASSETS[0],
    "description_record_schema": DESCRIPTION_PROTOCOL_ASSETS[1],
    "description_output_schema": DESCRIPTION_PROTOCOL_ASSETS[2],
}
D_MINUS_ONE_OVERFIT_SOURCE_NAMES = frozenset({
    "artifact_readiness_report", "checkpoint", "dataset_summary",
    "gradient_gate", "raw_generations", "resolved_config", "train_history",
    "trainable_manifest", "validation_report",
    *D_MINUS_ONE_OVERFIT_PROTOCOL_ASSET_SOURCES,
})
