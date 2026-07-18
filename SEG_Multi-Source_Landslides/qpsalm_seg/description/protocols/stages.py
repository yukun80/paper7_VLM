"""Single source of truth for the D-1 and D0-D4 curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .versions import D_MINUS_ONE_GATE_PROTOCOL


DESCRIPTION_STREAM_SEED_OFFSETS = {
    "main": 11_003,
    "bridge": 11_003,
    "dior": 21_013,
    "global_caption": 31_019,
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    milestone: str
    task_family: str
    data_sources: tuple[str, ...]
    validation_split: str | None
    uses_region_tokens: bool
    region_token_policy: str
    trains_region_modules: bool
    trains_desc_adapter: bool
    initialization_kind: str
    initialize_from_stage: str | None
    initialize_from_checkpoint_role: str | None
    requires_d_minus_one_gate: bool
    requires_expert_bridge: bool
    gate_requirements: tuple[str, ...]
    checkpoint_role: str
    trainable_prefixes: tuple[str, ...]
    trainable_direct_parameters: tuple[str, ...]

    @property
    def requires_initialize_from(self) -> bool:
        return self.initialize_from_stage is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_GLOBAL_TRAINABLE_PREFIXES = ("description_view_to_hidden.",)
_GLOBAL_DIRECT_PARAMETERS = ("instruction_type", "visual_type")
_ALIGNMENT_TRAINABLE_PREFIXES = (
    "description_backbone.",
    "mgrr.",
    "alignment_text_projection.",
)
_ALIGNMENT_DIRECT_PARAMETERS = ("instruction_type", "alignment_temperature")
_REGION_TRAINABLE_PREFIXES = (
    "description_backbone.",
    "mgrr.",
    "region_to_hidden.",
    "description_view_to_hidden.",
)
_REGION_DIRECT_PARAMETERS = ("region_type", "instruction_type", "visual_type")


_STAGE_SPECS = {
    "overfit": StageSpec(
        name="overfit",
        milestone="D-1",
        task_family="mixed_protocol_overfit",
        data_sources=(
            "description_v4:train_eligible",
            "bridge_v7:candidate_all",
        ),
        validation_split="train",
        uses_region_tokens=True,
        region_token_policy="mixed_explicit",
        trains_region_modules=True,
        trains_desc_adapter=True,
        initialization_kind="segmentation_checkpoint",
        initialize_from_stage=None,
        initialize_from_checkpoint_role=None,
        requires_d_minus_one_gate=False,
        requires_expert_bridge=False,
        gate_requirements=(
            "description_m1_1_engineering_valid",
            "bridge_m2_engineering_valid_candidate_only",
            "segdesc_artifact_readiness_v2",
        ),
        checkpoint_role="terminal_last",
        trainable_prefixes=_REGION_TRAINABLE_PREFIXES,
        trainable_direct_parameters=_REGION_DIRECT_PARAMETERS,
    ),
    "mmrs_caption": StageSpec(
        name="mmrs_caption",
        milestone="D0",
        task_family="global_caption",
        data_sources=("description_v4:mmrs_global_caption_train",),
        validation_split="dev",
        uses_region_tokens=False,
        region_token_policy="forbidden",
        trains_region_modules=False,
        trains_desc_adapter=True,
        initialization_kind="segmentation_checkpoint",
        initialize_from_stage=None,
        initialize_from_checkpoint_role=None,
        requires_d_minus_one_gate=True,
        requires_expert_bridge=False,
        gate_requirements=(D_MINUS_ONE_GATE_PROTOCOL,),
        checkpoint_role="validation_best",
        trainable_prefixes=_GLOBAL_TRAINABLE_PREFIXES,
        trainable_direct_parameters=_GLOBAL_DIRECT_PARAMETERS,
    ),
    "rsicap_caption": StageSpec(
        name="rsicap_caption",
        milestone="D1",
        task_family="global_caption",
        data_sources=(
            "description_v4:rsicap_global_caption_train",
            "description_v4:mmrs_caption_replay",
        ),
        validation_split="dev",
        uses_region_tokens=False,
        region_token_policy="forbidden",
        trains_region_modules=False,
        trains_desc_adapter=True,
        initialization_kind="previous_stage_checkpoint",
        initialize_from_stage="mmrs_caption",
        initialize_from_checkpoint_role="validation_best",
        requires_d_minus_one_gate=True,
        requires_expert_bridge=False,
        gate_requirements=("inherited_d_minus_one_acceptance",),
        checkpoint_role="validation_best",
        trainable_prefixes=_GLOBAL_TRAINABLE_PREFIXES,
        trainable_direct_parameters=_GLOBAL_DIRECT_PARAMETERS,
    ),
    "dior_alignment": StageSpec(
        name="dior_alignment",
        milestone="D2",
        task_family="region_alignment",
        data_sources=("description_v4:dior_region_alignment_train",),
        validation_split="dev",
        uses_region_tokens=True,
        region_token_policy="required",
        trains_region_modules=True,
        trains_desc_adapter=False,
        initialization_kind="previous_stage_checkpoint",
        initialize_from_stage="rsicap_caption",
        initialize_from_checkpoint_role="validation_best",
        requires_d_minus_one_gate=True,
        requires_expert_bridge=False,
        gate_requirements=("inherited_d_minus_one_acceptance",),
        checkpoint_role="validation_best",
        trainable_prefixes=_ALIGNMENT_TRAINABLE_PREFIXES,
        trainable_direct_parameters=_ALIGNMENT_DIRECT_PARAMETERS,
    ),
    "bridge_auto": StageSpec(
        name="bridge_auto",
        milestone="D3a",
        task_family="region_description_auto",
        data_sources=("bridge_v7:auto_train",),
        validation_split=None,
        uses_region_tokens=True,
        region_token_policy="required",
        trains_region_modules=True,
        trains_desc_adapter=True,
        initialization_kind="previous_stage_checkpoint",
        initialize_from_stage="dior_alignment",
        initialize_from_checkpoint_role="validation_best",
        requires_d_minus_one_gate=True,
        requires_expert_bridge=False,
        gate_requirements=(
            "bridge_m2_engineering_valid_candidate_only",
            "inherited_d_minus_one_acceptance",
        ),
        checkpoint_role="terminal_last",
        trainable_prefixes=_REGION_TRAINABLE_PREFIXES,
        trainable_direct_parameters=_REGION_DIRECT_PARAMETERS,
    ),
    "bridge_expert": StageSpec(
        name="bridge_expert",
        milestone="D3b",
        task_family="region_description_expert",
        data_sources=(
            "bridge_v7:expert_train",
            "description_v4:dior_alignment_replay",
            "description_v4:global_caption_replay",
        ),
        validation_split="val",
        uses_region_tokens=True,
        region_token_policy="required",
        trains_region_modules=True,
        trains_desc_adapter=True,
        initialization_kind="previous_stage_checkpoint",
        initialize_from_stage="bridge_auto",
        initialize_from_checkpoint_role="terminal_last",
        requires_d_minus_one_gate=True,
        requires_expert_bridge=True,
        gate_requirements=(
            "bridge_m2_expert_pilot_frozen",
            "inherited_d_minus_one_acceptance",
        ),
        checkpoint_role="validation_best",
        trainable_prefixes=(
            *_REGION_TRAINABLE_PREFIXES,
            "alignment_text_projection.",
        ),
        trainable_direct_parameters=(
            *_REGION_DIRECT_PARAMETERS,
            "alignment_temperature",
        ),
    ),
    "predicted_mask": StageSpec(
        name="predicted_mask",
        milestone="D4",
        task_family="region_description_expert",
        data_sources=(
            "bridge_v7:expert_train",
            "predicted_regions:oof_train",
            "predicted_regions:fixed_expert_val",
        ),
        validation_split="val",
        uses_region_tokens=True,
        region_token_policy="required",
        trains_region_modules=True,
        trains_desc_adapter=True,
        initialization_kind="previous_stage_checkpoint",
        initialize_from_stage="bridge_expert",
        initialize_from_checkpoint_role="validation_best",
        requires_d_minus_one_gate=True,
        requires_expert_bridge=True,
        gate_requirements=(
            "bridge_m2_expert_pilot_frozen",
            "d4_curriculum_transition",
            "inherited_d_minus_one_acceptance",
        ),
        checkpoint_role="validation_best",
        trainable_prefixes=_REGION_TRAINABLE_PREFIXES,
        trainable_direct_parameters=_REGION_DIRECT_PARAMETERS,
    ),
}

DESCRIPTION_STAGES = tuple(_STAGE_SPECS)


def validate_stage_registry() -> None:
    """Reject an internally inconsistent curriculum at import time."""
    allowed_region_policies = {"forbidden", "required", "mixed_explicit"}
    allowed_checkpoint_roles = {"terminal_last", "validation_best"}
    allowed_initialization_kinds = {
        "segmentation_checkpoint", "previous_stage_checkpoint",
    }
    for name, spec in _STAGE_SPECS.items():
        if name != spec.name or not spec.data_sources or not spec.gate_requirements:
            raise RuntimeError(f"StageSpec inventory incomplete: {name}")
        if spec.region_token_policy not in allowed_region_policies:
            raise RuntimeError(
                f"StageSpec region token policy invalid: {name}"
            )
        if (
            (spec.region_token_policy == "forbidden" and spec.uses_region_tokens)
            or (
                spec.region_token_policy in {"required", "mixed_explicit"}
                and not spec.uses_region_tokens
            )
        ):
            raise RuntimeError(
                f"StageSpec region-token boolean/policy mismatch: {name}"
            )
        if spec.checkpoint_role not in allowed_checkpoint_roles:
            raise RuntimeError(f"StageSpec checkpoint role invalid: {name}")
        if spec.initialization_kind not in allowed_initialization_kinds:
            raise RuntimeError(f"StageSpec initialization kind invalid: {name}")
        if spec.initialize_from_stage is None:
            if spec.initialization_kind != "segmentation_checkpoint":
                raise RuntimeError(
                    f"StageSpec root initialization kind invalid: {name}"
                )
            if spec.initialize_from_checkpoint_role is not None:
                raise RuntimeError(
                    f"StageSpec root stage declares initialization role: {name}"
                )
            continue
        if spec.initialization_kind != "previous_stage_checkpoint":
            raise RuntimeError(
                f"StageSpec staged initialization kind invalid: {name}"
            )
        predecessor = _STAGE_SPECS.get(spec.initialize_from_stage)
        if predecessor is None:
            raise RuntimeError(f"StageSpec predecessor missing: {name}")
        if spec.initialize_from_checkpoint_role != predecessor.checkpoint_role:
            raise RuntimeError(
                f"StageSpec predecessor role mismatch: {name}"
            )


def get_stage_spec(stage: str) -> StageSpec:
    try:
        return _STAGE_SPECS[str(stage)]
    except KeyError as exc:
        raise ValueError(f"未知 description stage={stage!r}") from exc


validate_stage_registry()
