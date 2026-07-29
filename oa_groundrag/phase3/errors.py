"""OA-LandslideDesc Phase 2 的公共异常与稳定错误码。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class ReasonCode(StrEnum):
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    INVALID_ENUM = "INVALID_ENUM"
    NONFINITE_NUMBER = "NONFINITE_NUMBER"
    PATH_ABSOLUTE = "PATH_ABSOLUTE"
    PATH_ESCAPE = "PATH_ESCAPE"
    SOURCE_SYMLINK = "SOURCE_SYMLINK"
    OUTPUT_LINK = "OUTPUT_LINK"
    ASSET_MISSING = "ASSET_MISSING"
    ASSET_CORRUPT = "ASSET_CORRUPT"
    ASSET_ZERO_BYTES = "ASSET_ZERO_BYTES"
    ASSET_LIMIT_EXCEEDED = "ASSET_LIMIT_EXCEEDED"
    MASK_MISSING = "MASK_MISSING"
    MASK_EMPTY = "MASK_EMPTY"
    MASK_INVALID_VALUES = "MASK_INVALID_VALUES"
    MASK_GEOMETRY_MISMATCH = "MASK_GEOMETRY_MISMATCH"
    BBOX_INVALID = "BBOX_INVALID"
    BBOX_ZERO_AREA = "BBOX_ZERO_AREA"
    DUPLICATE_RECORD = "DUPLICATE_RECORD"
    DUPLICATE_REFERENCE = "DUPLICATE_REFERENCE"
    UNUSED_ASSET = "UNUSED_ASSET"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    REVERSE_GROUNDING_DIRECTION = "REVERSE_GROUNDING_DIRECTION"
    INVALID_CONVERSATION = "INVALID_CONVERSATION"
    EXTERNAL_SPLIT_LEAKAGE = "EXTERNAL_SPLIT_LEAKAGE"
    OA_SPLIT_LEAKAGE = "OA_SPLIT_LEAKAGE"
    OA_REVIEW_INCOMPLETE = "OA_REVIEW_INCOMPLETE"
    OA_ROLE_CONTAMINATION = "OA_ROLE_CONTAMINATION"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    STAGING_FAILURE = "STAGING_FAILURE"
    HASH_MISMATCH = "HASH_MISMATCH"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"


@dataclass(frozen=True)
class ErrorContext:
    code: ReasonCode
    message: str
    details: Mapping[str, Any]


class LandslideDescError(RuntimeError):
    """带稳定 reason code 的 Phase 2 基类异常。"""

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


class ConfigError(LandslideDescError):
    pass


class SchemaError(LandslideDescError):
    pass


class SourceAuditError(LandslideDescError):
    pass


class AssetError(LandslideDescError):
    pass


class BuildError(LandslideDescError):
    pass


class ValidationError(LandslideDescError):
    pass


class ExportError(LandslideDescError):
    pass
