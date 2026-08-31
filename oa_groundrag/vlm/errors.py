"""RS-VLM 的稳定错误合同。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class ReasonCode(StrEnum):
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    INVALID_ENUM = "INVALID_ENUM"
    NONFINITE_NUMBER = "NONFINITE_NUMBER"
    PATH_ESCAPE = "PATH_ESCAPE"
    OUTPUT_LINK = "OUTPUT_LINK"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    ASSET_MISSING = "ASSET_MISSING"
    ASSET_LIMIT_EXCEEDED = "ASSET_LIMIT_EXCEEDED"
    BENCHMARK_IDENTITY_MISMATCH = "BENCHMARK_IDENTITY_MISMATCH"
    ROLE_FORBIDDEN = "ROLE_FORBIDDEN"
    FORMAL_EVALUATION_FORBIDDEN = "FORMAL_EVALUATION_FORBIDDEN"
    EXTERNAL_MASK_FORBIDDEN = "EXTERNAL_MASK_FORBIDDEN"
    MASK_INVALID = "MASK_INVALID"
    MASK_SHAPE_MISMATCH = "MASK_SHAPE_MISMATCH"
    REGION_NOT_FOUND = "REGION_NOT_FOUND"
    REGION_ID_INVALID = "REGION_ID_INVALID"
    AMBIGUOUS_REGION = "AMBIGUOUS_REGION"
    AUXILIARY_LIMIT_EXCEEDED = "AUXILIARY_LIMIT_EXCEEDED"
    RAG_FORBIDDEN = "RAG_FORBIDDEN"
    EVIDENCE_REFERENCE_INVALID = "EVIDENCE_REFERENCE_INVALID"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    FORBIDDEN_MODEL_FACT = "FORBIDDEN_MODEL_FACT"
    IMAGE_LIMIT_EXCEEDED = "IMAGE_LIMIT_EXCEEDED"
    TOKEN_LIMIT_EXCEEDED = "TOKEN_LIMIT_EXCEEDED"
    LOSS_MASK_INVALID = "LOSS_MASK_INVALID"
    VALIDATION_SELECTION_INVALID = "VALIDATION_SELECTION_INVALID"
    CUDA_REQUIRED = "CUDA_REQUIRED"
    MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
    PREDICTION_INVALID = "PREDICTION_INVALID"
    MASK_MODE_MIXED = "MASK_MODE_MIXED"
    COUNTERFACTUAL_INVALID = "COUNTERFACTUAL_INVALID"
    SPLIT_FORBIDDEN = "SPLIT_FORBIDDEN"
    PARENT_OVERLAP = "PARENT_OVERLAP"
    ASSET_ROLE_LEAKAGE = "ASSET_ROLE_LEAKAGE"
    ANNOTATION_INVALID = "ANNOTATION_INVALID"
    FORBIDDEN_CLAIM = "FORBIDDEN_CLAIM"
    PREDICTION_IDENTITY_MISMATCH = "PREDICTION_IDENTITY_MISMATCH"
    GATE_B_PROTOCOL_INVALID = "GATE_B_PROTOCOL_INVALID"
    GATE_B_SELECTION_INVALID = "GATE_B_SELECTION_INVALID"
    GATE_B_RUN_INVALID = "GATE_B_RUN_INVALID"
    BACKEND_UNKNOWN = "BACKEND_UNKNOWN"
    BACKEND_DUPLICATE = "BACKEND_DUPLICATE"
    BACKEND_CONTRACT_INVALID = "BACKEND_CONTRACT_INVALID"
    ASSET_LEDGER_INVALID = "ASSET_LEDGER_INVALID"


@dataclass(frozen=True)
class ErrorContext:
    code: ReasonCode
    message: str
    details: Mapping[str, Any]


class VLMError(RuntimeError):
    """带稳定 reason code 的 VLM 基类异常。"""

    def __init__(
        self,
        code: ReasonCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.context = ErrorContext(code, message, dict(details or {}))
        super().__init__(f"{code.value}: {message}")

    @property
    def code(self) -> ReasonCode:
        return self.context.code

    @property
    def details(self) -> Mapping[str, Any]:
        return self.context.details


class ConfigError(VLMError):
    pass


class ContractError(VLMError):
    pass


class PreflightError(VLMError):
    pass


class SelectionError(VLMError):
    pass


class EvidenceError(VLMError):
    pass


class ProcessingError(VLMError):
    pass


class ModelError(VLMError):
    pass


class CheckpointError(VLMError):
    pass


class PredictionError(VLMError):
    pass


class EvaluationError(VLMError):
    pass
