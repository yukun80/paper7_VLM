"""OA-LandslideDesc 的版本化 canonical 与 adapter 中间合同。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    canonical_json,
    portable_relative_path,
    reject_nonfinite,
    sha256_text,
)
from .errors import ReasonCode, SchemaError


CANONICAL_SCHEMA_VERSION = "oa_landslidedesc.canonical.v3"
BUILD_CONFIG_VERSION = "oa_landslidedesc.build_config.v3"
EXPORT_CONFIG_VERSION = "oa_landslidedesc.qwen_export.v3"
MANIFEST_VERSION = "oa_landslidedesc.manifest.v3"
VALIDATION_VERSION = "oa_landslidedesc.validation.v3"
STATISTICS_SCHEMA_VERSION = "oa_landslidedesc.statistics.v3"
PROVENANCE_SCHEMA_VERSION = "oa_landslidedesc.provenance.v3"
EXTERNAL_SPLIT_SCHEMA_VERSION = "oa_landslidedesc.external_split.v1"
QWEN_EXPORT_MANIFEST_VERSION = "oa_landslidedesc.qwen_export_manifest.v3"
QWEN_TEMPLATE_VERSION = "qwen3vl_messages.v2"
DESCRIPTION_MULTITASK_PROFILE = "description_multitask.v1"
EXTERNAL_SPLIT_VERSION = "external_parent_split.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID_PATTERN = re.compile(r"^record_[0-9a-f]{64}$")
_PARENT_ID_PATTERN = re.compile(r"^parent_[0-9a-f]{64}$")
_ASSET_ID_PATTERN = re.compile(r"^asset_[0-9a-f]{64}$")
_PROVENANCE_ID_PATTERN = re.compile(r"^provenance_[0-9a-f]{64}$")

HF_DATASETS_REFERENCE = {
    "repository_url": "https://github.com/huggingface/datasets",
    "release_tag": "4.8.5",
    "commit_sha": "a015b2fa5c1a6cda677fa46f20a54773258553ac",
    "inspected_date": "2026-07-27",
}


class LogicalRole(StrEnum):
    EXTERNAL_TRAIN = "external_train"
    EXTERNAL_VAL = "external_val"
    OA_TRAIN = "oa_train"
    OA_VAL = "oa_val"
    OA_TEST = "oa_test"


class AnnotationLayer(StrEnum):
    EXTERNAL_SOURCE = "external_source"
    OA_GOLD = "oa_gold"
    OA_AUTO = "oa_auto"
    OA_SILVER = "oa_silver"


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskFamily(StrEnum):
    GLOBAL_CAPTION = "global_caption"
    VISUAL_QA = "visual_qa"
    BBOX_REGION_CAPTION = "bbox_region_caption"
    SCENE_UNDERSTANDING = "scene_understanding"
    OBJECT_COUNT = "object_count"
    SPATIAL_RELATION = "spatial_relation"
    VISIBLE_CHANGE_REPORT = "visible_change_report"
    OA_MASK_FACTS = "oa_mask_facts"
    OA_MASK_DESCRIPTION = "oa_mask_description"


class SupervisionKind(StrEnum):
    LONG_DESCRIPTION = "long_description"
    SHORT_QA = "short_qa"
    NUMERIC_QA = "numeric_qa"
    REGION_DESCRIPTION = "region_description"
    SPATIAL_DESCRIPTION = "spatial_description"
    STRUCTURED_REPORT = "structured_report"


class InputLayout(StrEnum):
    SINGLE_IMAGE = "single_image"
    BBOX_REGION = "bbox_region"
    BOXED_IMAGE = "boxed_image"
    PRE_POST = "pre_post"
    MASK_GROUNDED = "mask_grounded"


class OutputModality(StrEnum):
    TEXT = "text"


TASK_SUPERVISION_KINDS: Mapping[TaskFamily, frozenset[SupervisionKind]] = {
    TaskFamily.GLOBAL_CAPTION: frozenset({SupervisionKind.LONG_DESCRIPTION}),
    TaskFamily.VISUAL_QA: frozenset({SupervisionKind.SHORT_QA}),
    TaskFamily.BBOX_REGION_CAPTION: frozenset(
        {SupervisionKind.REGION_DESCRIPTION}
    ),
    TaskFamily.SCENE_UNDERSTANDING: frozenset({SupervisionKind.SHORT_QA}),
    TaskFamily.OBJECT_COUNT: frozenset({SupervisionKind.NUMERIC_QA}),
    TaskFamily.SPATIAL_RELATION: frozenset(
        {SupervisionKind.SPATIAL_DESCRIPTION}
    ),
    TaskFamily.VISIBLE_CHANGE_REPORT: frozenset(
        {SupervisionKind.STRUCTURED_REPORT}
    ),
    TaskFamily.OA_MASK_FACTS: frozenset({SupervisionKind.STRUCTURED_REPORT}),
    TaskFamily.OA_MASK_DESCRIPTION: frozenset(
        {SupervisionKind.LONG_DESCRIPTION}
    ),
}

TASK_INPUT_LAYOUTS: Mapping[TaskFamily, frozenset[InputLayout]] = {
    TaskFamily.GLOBAL_CAPTION: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.VISUAL_QA: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.BBOX_REGION_CAPTION: frozenset({InputLayout.BBOX_REGION}),
    TaskFamily.SCENE_UNDERSTANDING: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.OBJECT_COUNT: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.SPATIAL_RELATION: frozenset({InputLayout.BOXED_IMAGE}),
    TaskFamily.VISIBLE_CHANGE_REPORT: frozenset({InputLayout.PRE_POST}),
    TaskFamily.OA_MASK_FACTS: frozenset({InputLayout.MASK_GROUNDED}),
    TaskFamily.OA_MASK_DESCRIPTION: frozenset({InputLayout.MASK_GROUNDED}),
}


class MediaType(StrEnum):
    IMAGE = "image"
    MASK = "mask"
    ARRAY = "array"


@dataclass(frozen=True)
class PendingAsset:
    role: str
    media_type: MediaType
    extension: str
    source_ref: str
    source_path: Path | None = None
    archive_path: Path | None = None
    archive_member: str | None = None
    payload: bytes | None = None
    align_to_role: str | None = None
    allow_empty: bool = False

    def validate(self) -> None:
        variants = sum(
            (
                self.source_path is not None,
                self.archive_path is not None or self.archive_member is not None,
                self.payload is not None,
            )
        )
        if variants != 1:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{self.role}: asset 必须且只能指定 file、archive 或 payload",
            )
        if (self.archive_path is None) != (self.archive_member is None):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{self.role}: archive_path/member 必须同时存在",
            )
        portable_relative_path(self.source_ref, location=f"asset[{self.role}].source_ref")


@dataclass(frozen=True)
class SourceExample:
    source: str
    source_record_id: str
    source_split: str
    parent_key: str
    logical_role: LogicalRole
    task_family: TaskFamily
    supervision_kind: SupervisionKind
    input_layout: InputLayout
    output_modality: OutputModality
    assets: tuple[PendingAsset, ...]
    target: Mapping[str, Any]
    instruction: str
    training_responses: tuple[str, ...]
    reference_responses: tuple[str, ...]
    deterministic_facts: Mapping[str, Any]
    annotation_layer: AnnotationLayer
    review_status: ReviewStatus
    provenance: tuple[Mapping[str, Any], ...]
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkipRecord:
    source: str
    source_record_id: str
    source_split: str
    task_family: str
    reason_code: ReasonCode
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_record_id": self.source_record_id,
            "source_split": self.source_split,
            "task_family": self.task_family,
            "reason_code": self.reason_code.value,
            "evidence": dict(self.evidence),
        }


@dataclass
class AdapterResult:
    source: str
    examples: list[SourceExample] = field(default_factory=list)
    skips: list[SkipRecord] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    def add_skip(
        self,
        *,
        source_record_id: str,
        source_split: str,
        task_family: str,
        reason_code: ReasonCode,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.skips.append(
            SkipRecord(
                source=self.source,
                source_record_id=source_record_id,
                source_split=source_split,
                task_family=task_family,
                reason_code=reason_code,
                evidence=dict(evidence or {}),
            )
        )


def parent_id(source: str, parent_key: str) -> str:
    return f"parent_{sha256_text(canonical_json([CANONICAL_SCHEMA_VERSION, source, parent_key]))}"


def record_id(identity: Mapping[str, Any]) -> str:
    return f"record_{sha256_text(canonical_json(identity))}"


def record_hash(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return sha256_text(canonical_json(payload))


def validate_role_layer(
    role: LogicalRole,
    layer: AnnotationLayer,
    review: ReviewStatus,
) -> None:
    if role in {LogicalRole.EXTERNAL_TRAIN, LogicalRole.EXTERNAL_VAL}:
        if (
            layer is not AnnotationLayer.EXTERNAL_SOURCE
            or review is not ReviewStatus.NOT_REQUIRED
        ):
            raise SchemaError(
                ReasonCode.OA_ROLE_CONTAMINATION,
                "external_train/external_val 只能使用 not_required external_source",
            )
        return
    if layer is AnnotationLayer.EXTERNAL_SOURCE:
        raise SchemaError(
            ReasonCode.OA_ROLE_CONTAMINATION,
            "OA 逻辑角色不能使用 external_source",
        )
    if layer in {AnnotationLayer.OA_AUTO, AnnotationLayer.OA_SILVER}:
        if (
            role is not LogicalRole.OA_TRAIN
            or review is not ReviewStatus.NOT_REQUIRED
        ):
            raise SchemaError(
                ReasonCode.OA_ROLE_CONTAMINATION,
                f"{layer.value} 只能以 not_required 进入 oa_train",
            )
    if layer is AnnotationLayer.OA_GOLD and review is not ReviewStatus.APPROVED:
        raise SchemaError(
            ReasonCode.OA_ROLE_CONTAMINATION,
            "进入 canonical 的 oa_gold 必须 approved",
        )
    if role in {LogicalRole.OA_VAL, LogicalRole.OA_TEST}:
        if layer is not AnnotationLayer.OA_GOLD or review is not ReviewStatus.APPROVED:
            raise SchemaError(
                ReasonCode.OA_ROLE_CONTAMINATION,
                f"{role.value} 只接受 approved oa_gold",
            )


def _exact_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    location: str,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        raise SchemaError(
            ReasonCode.UNKNOWN_FIELD if unknown else ReasonCode.SCHEMA_MISMATCH,
            f"{location}: missing={missing} unknown={unknown}",
        )


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是非空字符串",
        )
    return value


def _integer(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是 >= {minimum} 的整数",
        )
    return value


def _number(value: Any, *, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是有限数值",
        )
    return float(value)


def _size(value: Any, *, location: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是 [width,height]",
        )
    return (
        _integer(value[0], location=f"{location}[0]", minimum=1),
        _integer(value[1], location=f"{location}[1]", minimum=1),
    )


def _string_list(
    value: Any,
    *,
    location: str,
    allow_empty: bool,
    pattern: re.Pattern[str] | None = None,
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是{'可空' if allow_empty else '非空'}字符串列表",
        )
    output: list[str] = []
    for index, item in enumerate(value):
        text = _string(item, location=f"{location}[{index}]")
        if pattern is not None and pattern.fullmatch(text) is None:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{location}[{index}]: 格式非法",
            )
        output.append(text)
    if len(set(output)) != len(output):
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{location}: 不允许重复值",
        )
    return output


def validate_canonical_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "record_id",
        "parent_id",
        "source",
        "source_record_id",
        "source_split",
        "logical_role",
        "task_family",
        "supervision_kind",
        "input_layout",
        "output_modality",
        "media",
        "target",
        "instruction",
        "training_responses",
        "reference_responses",
        "deterministic_facts",
        "annotation",
        "coordinate_convention",
        "transforms",
        "quality_flags",
        "provenance_ids",
        "record_sha256",
    }
    if not isinstance(record, dict):
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "canonical record 必须是对象")
    _exact_fields(record, required=required, location="$")
    reject_nonfinite(record, location="$")
    if record["schema_version"] != CANONICAL_SCHEMA_VERSION:
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "canonical schema_version 错误")
    record_identifier = _string(record["record_id"], location="$.record_id")
    parent_identifier = _string(record["parent_id"], location="$.parent_id")
    if _RECORD_ID_PATTERN.fullmatch(record_identifier) is None:
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "record_id 格式非法")
    if _PARENT_ID_PATTERN.fullmatch(parent_identifier) is None:
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "parent_id 格式非法")
    source = _string(record["source"], location="$.source")
    _string(record["source_record_id"], location="$.source_record_id")
    source_split = _string(record["source_split"], location="$.source_split")
    instruction = _string(record["instruction"], location="$.instruction")
    if not isinstance(record["annotation"], dict):
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "annotation 必须是对象")
    _exact_fields(
        record["annotation"],
        required={"layer", "review_status"},
        location="$.annotation",
    )
    try:
        role = LogicalRole(record["logical_role"])
        layer = AnnotationLayer(record["annotation"]["layer"])
        review = ReviewStatus(record["annotation"]["review_status"])
        task = TaskFamily(record["task_family"])
        supervision = SupervisionKind(record["supervision_kind"])
        input_layout = InputLayout(record["input_layout"])
        output_modality = OutputModality(record["output_modality"])
    except (KeyError, TypeError, ValueError) as error:
        raise SchemaError(
            ReasonCode.INVALID_ENUM,
            "canonical role/task/supervision/layout/output/annotation 枚举非法",
        ) from error
    if supervision not in TASK_SUPERVISION_KINDS[task]:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{task.value} 不接受 supervision_kind={supervision.value}",
        )
    if input_layout not in TASK_INPUT_LAYOUTS[task]:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{task.value} 不接受 input_layout={input_layout.value}",
        )
    if output_modality is not OutputModality.TEXT:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            "description_multitask canonical 只接受 text 输出",
        )
    validate_role_layer(role, layer, review)
    external_roles = {LogicalRole.EXTERNAL_TRAIN, LogicalRole.EXTERNAL_VAL}
    if role in external_roles and source == "oa":
        raise SchemaError(
            ReasonCode.OA_ROLE_CONTAMINATION,
            "source=oa 不能进入 external_train/external_val",
        )
    if role not in external_roles:
        expected_split = role.value.removeprefix("oa_")
        if source != "oa" or source_split != expected_split:
            raise SchemaError(
                ReasonCode.OA_ROLE_CONTAMINATION,
                "OA role 的 source/source_split 不匹配",
            )
    if task in {TaskFamily.OA_MASK_FACTS, TaskFamily.OA_MASK_DESCRIPTION}:
        if source != "oa":
            raise SchemaError(
                ReasonCode.OA_ROLE_CONTAMINATION,
                "OA task family 只能来自 source=oa",
            )
    if not isinstance(record["media"], list) or not record["media"]:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "canonical media 必须为非空列表")
    media_roles: set[str] = set()
    media_asset_ids: list[str] = []
    media_types_by_role: dict[str, MediaType] = {}
    for index, media in enumerate(record["media"]):
        if not isinstance(media, dict):
            raise SchemaError(ReasonCode.TYPE_MISMATCH, f"media[{index}] 必须为对象")
        media_type_raw = media.get("media_type")
        try:
            media_type = MediaType(media_type_raw)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                ReasonCode.INVALID_ENUM,
                f"media[{index}].media_type 非法",
            ) from error
        common = {
            "asset_id",
            "path",
            "media_type",
            "sha256",
            "size_bytes",
            "extension",
            "role",
            "source_sha256",
        }
        geometry = {"width", "height", "mode"}
        _exact_fields(
            media,
            required=common | (geometry if media_type is not MediaType.ARRAY else set()),
            location=f"$.media[{index}]",
        )
        asset_identifier = _string(
            media["asset_id"], location=f"$.media[{index}].asset_id"
        )
        digest = _string(media["sha256"], location=f"$.media[{index}].sha256")
        source_digest = _string(
            media["source_sha256"],
            location=f"$.media[{index}].source_sha256",
        )
        if (
            _ASSET_ID_PATTERN.fullmatch(asset_identifier) is None
            or _SHA256_PATTERN.fullmatch(digest) is None
            or _SHA256_PATTERN.fullmatch(source_digest) is None
            or asset_identifier != f"asset_{digest}"
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"media[{index}] hash/asset_id 格式非法",
            )
        path = _string(media["path"], location=f"$.media[{index}].path")
        portable_relative_path(path, location=f"media[{index}].path")
        role_name = _string(media["role"], location=f"$.media[{index}].role")
        if role_name in media_roles:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"media role 重复：{role_name}",
            )
        media_roles.add(role_name)
        media_types_by_role[role_name] = media_type
        media_asset_ids.append(asset_identifier)
        _string(media["extension"], location=f"$.media[{index}].extension")
        _integer(media["size_bytes"], location=f"$.media[{index}].size_bytes", minimum=1)
        if media_type is not MediaType.ARRAY:
            _integer(media["width"], location=f"$.media[{index}].width", minimum=1)
            _integer(media["height"], location=f"$.media[{index}].height", minimum=1)
            expected_mode = "RGB" if media_type is MediaType.IMAGE else "L"
            if media["mode"] != expected_mode:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"media[{index}].mode 应为 {expected_mode}",
                )
    target = record["target"]
    if not isinstance(target, dict) or target.get("type") not in {"none", "mask", "bbox"}:
        raise SchemaError(ReasonCode.INVALID_ENUM, "target.type 必须是 none/mask/bbox")
    if target["type"] == "none":
        _exact_fields(target, required={"type"}, location="$.target")
    elif target["type"] == "mask":
        _exact_fields(
            target,
            required={
                "type",
                "image_role",
                "mask_role",
                "source_convention",
                "canonical_convention",
                "mask_asset_id",
                "image_asset_id",
                "size",
            },
            optional={"empty_allowed"},
            location="$.target",
        )
        image_role = _string(target["image_role"], location="$.target.image_role")
        mask_role = _string(target["mask_role"], location="$.target.mask_role")
        if image_role not in media_roles or mask_role not in media_roles:
            raise SchemaError(
                ReasonCode.MASK_GEOMETRY_MISMATCH,
                "mask target role 未引用 media",
            )
        if target["source_convention"] != "binary_raster_top_left" or target[
            "canonical_convention"
        ] != "binary_raster_top_left":
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "mask 坐标/栅格约定非法",
            )
        if target.get("empty_allowed", False) not in {True, False} or not isinstance(
            target.get("empty_allowed", False), bool
        ):
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH,
                "target.empty_allowed 必须是 bool",
            )
        target_size = _size(target["size"], location="$.target.size")
        by_role = {str(media["role"]): media for media in record["media"]}
        if (
            target["image_asset_id"] != by_role[image_role]["asset_id"]
            or target["mask_asset_id"] != by_role[mask_role]["asset_id"]
            or media_types_by_role[image_role] is not MediaType.IMAGE
            or media_types_by_role[mask_role] is not MediaType.MASK
            or target_size
            != (int(by_role[image_role]["width"]), int(by_role[image_role]["height"]))
            or target_size
            != (int(by_role[mask_role]["width"]), int(by_role[mask_role]["height"]))
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "mask target asset_id 与 media 不一致",
            )
    else:
        _exact_fields(
            target,
            required={
                "type",
                "image_role",
                "source_convention",
                "canonical_convention",
                "boxes",
            },
            location="$.target",
        )
        if target["source_convention"] not in {
            "xyxy_normalized_top_left",
            "xywh_pixel_top_left",
        } or target["canonical_convention"] != "xyxy_normalized_top_left":
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "bbox 坐标约定非法",
            )
        bbox_image_role = _string(
            target["image_role"],
            location="$.target.image_role",
        )
        if (
            bbox_image_role not in media_roles
            or media_types_by_role.get(bbox_image_role) is not MediaType.IMAGE
        ):
            raise SchemaError(
                ReasonCode.BBOX_INVALID,
                "bbox image_role 未引用 media",
            )
        boxes = target["boxes"]
        if not isinstance(boxes, list) or not boxes:
            raise SchemaError(ReasonCode.BBOX_INVALID, "bbox boxes 必须非空")
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise SchemaError(
                    ReasonCode.TYPE_MISMATCH,
                    f"bbox[{index}] 必须是对象",
                )
            _exact_fields(
                box,
                required={
                    "source",
                    "source_convention",
                    "source_image_size",
                    "canonical_xyxy_norm",
                },
                optional={"label", "object_key"},
                location=f"$.target.boxes[{index}]",
            )
            if box["source_convention"] != target["source_convention"]:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"bbox[{index}] source_convention 不一致",
                )
            source_values = box["source"]
            canonical_values = box["canonical_xyxy_norm"]
            if (
                not isinstance(source_values, list)
                or len(source_values) != 4
                or not isinstance(canonical_values, list)
                or len(canonical_values) != 4
            ):
                raise SchemaError(
                    ReasonCode.BBOX_INVALID,
                    f"bbox[{index}] 必须包含四元数值",
                )
            for value_index, value in enumerate(source_values):
                _number(value, location=f"$.target.boxes[{index}].source[{value_index}]")
            canonical = [
                _number(
                    value,
                    location=(
                        f"$.target.boxes[{index}].canonical_xyxy_norm[{value_index}]"
                    ),
                )
                for value_index, value in enumerate(canonical_values)
            ]
            if (
                min(canonical) < 0
                or max(canonical) > 1
                or canonical[2] <= canonical[0]
                or canonical[3] <= canonical[1]
            ):
                raise SchemaError(
                    ReasonCode.BBOX_INVALID,
                    f"bbox[{index}] canonical 坐标非法",
                )
            _size(
                box["source_image_size"],
                location=f"$.target.boxes[{index}].source_image_size",
            )
            for optional_key in ("label", "object_key"):
                if optional_key in box:
                    _string(
                        box[optional_key],
                        location=f"$.target.boxes[{index}].{optional_key}",
                    )
    image_roles = {
        name
        for name, media_type in media_types_by_role.items()
        if media_type is MediaType.IMAGE
    }
    mask_roles = {
        name
        for name, media_type in media_types_by_role.items()
        if media_type is MediaType.MASK
    }
    if input_layout is InputLayout.SINGLE_IMAGE:
        if target["type"] != "none" or len(image_roles) != 1 or mask_roles:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "single_image 必须是单张 image、无 mask 且 target=none",
            )
    elif input_layout in {InputLayout.BBOX_REGION, InputLayout.BOXED_IMAGE}:
        if target["type"] != "bbox" or len(image_roles) != 1 or mask_roles:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{input_layout.value} 必须是单张 image、bbox context 且无 mask",
            )
    elif input_layout is InputLayout.PRE_POST:
        if (
            target["type"] != "none"
            or image_roles != {"pre_image", "post_image"}
            or mask_roles
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "pre_post 必须包含且只包含 pre_image/post_image，target=none",
            )
    elif input_layout is InputLayout.MASK_GROUNDED:
        if (
            target["type"] != "mask"
            or len(image_roles) != 1
            or len(mask_roles) != 1
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "mask_grounded 必须包含一张 image、一张 mask 和 mask context",
            )
    training_responses = _string_list(
        record["training_responses"],
        location="$.training_responses",
        allow_empty=role in {LogicalRole.OA_VAL, LogicalRole.OA_TEST},
    )
    _string_list(
        record["reference_responses"],
        location="$.reference_responses",
        allow_empty=False,
    )
    if role not in {LogicalRole.OA_VAL, LogicalRole.OA_TEST} and not training_responses:
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            "训练角色必须包含 training_responses",
        )
    if not isinstance(record["deterministic_facts"], dict):
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            "deterministic_facts 必须是对象",
        )
    coordinate = record["coordinate_convention"]
    if not isinstance(coordinate, dict):
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            "coordinate_convention 必须是对象",
        )
    _exact_fields(
        coordinate,
        required={"image_origin", "bbox_canonical", "mask_layout"},
        location="$.coordinate_convention",
    )
    if coordinate != {
        "image_origin": "top_left",
        "bbox_canonical": "xyxy_normalized",
        "mask_layout": "height_width",
    }:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            "coordinate_convention 非 canonical v1 固定约定",
        )
    transforms = record["transforms"]
    if not isinstance(transforms, list) or not transforms:
        raise SchemaError(
            ReasonCode.TYPE_MISMATCH,
            "transforms 必须是非空列表",
        )
    transform_fields = {
        "exif_orientation": {
            "asset_role",
            "type",
            "orientation",
            "input_size",
            "output_size",
        },
        "color_conversion": {"asset_role", "type", "output_mode"},
        "resize": {
            "asset_role",
            "type",
            "interpolation",
            "input_size",
            "output_size",
        },
        "mask_exif_orientation": {
            "asset_role",
            "type",
            "orientation",
            "interpolation",
        },
        "mask_resize": {
            "asset_role",
            "type",
            "input_size",
            "output_size",
            "interpolation",
        },
    }
    for index, transform in enumerate(transforms):
        if not isinstance(transform, dict) or transform.get("type") not in transform_fields:
            raise SchemaError(
                ReasonCode.INVALID_ENUM,
                f"transforms[{index}].type 非法",
            )
        transform_type = str(transform["type"])
        _exact_fields(
            transform,
            required=transform_fields[transform_type],
            location=f"$.transforms[{index}]",
        )
        role_name = _string(
            transform["asset_role"],
            location=f"$.transforms[{index}].asset_role",
        )
        if role_name not in media_roles:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"transforms[{index}] 引用未知 asset_role",
            )
        expected_media_type = (
            MediaType.MASK
            if transform_type.startswith("mask_")
            else MediaType.IMAGE
        )
        if media_types_by_role[role_name] is not expected_media_type:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"transforms[{index}] 与 media_type 不匹配",
            )
        if "orientation" in transform:
            orientation = _integer(
                transform["orientation"],
                location=f"$.transforms[{index}].orientation",
                minimum=1,
            )
            if orientation > 8:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"transforms[{index}].orientation 必须 <= 8",
                )
        for size_key in ("input_size", "output_size"):
            if size_key in transform:
                _size(
                    transform[size_key],
                    location=f"$.transforms[{index}].{size_key}",
                )
        if transform_type == "color_conversion" and transform["output_mode"] != "RGB":
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "color_conversion output_mode 必须为 RGB",
            )
        if "interpolation" in transform and transform["interpolation"] not in {
            "none",
            "bilinear",
            "nearest",
        }:
            raise SchemaError(
                ReasonCode.INVALID_ENUM,
                f"transforms[{index}].interpolation 非法",
            )
    _string_list(
        record["quality_flags"],
        location="$.quality_flags",
        allow_empty=True,
    )
    _string_list(
        record["provenance_ids"],
        location="$.provenance_ids",
        allow_empty=False,
        pattern=_PROVENANCE_ID_PATTERN,
    )
    record_digest = _string(record["record_sha256"], location="$.record_sha256")
    if _SHA256_PATTERN.fullmatch(record_digest) is None:
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "record_sha256 格式非法")
    identity = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "parent_id": parent_identifier,
        "source": source,
        "logical_role": role.value,
        "task_family": task.value,
        "supervision_kind": supervision.value,
        "input_layout": input_layout.value,
        "output_modality": output_modality.value,
        "media_asset_ids": sorted(media_asset_ids),
        "target": target,
        "instruction": instruction,
        "training_responses": list(record["training_responses"]),
        "reference_responses": list(record["reference_responses"]),
        "annotation_layer": layer.value,
    }
    if record_identifier != record_id(identity):
        raise SchemaError(ReasonCode.HASH_MISMATCH, "record_id 与 canonical identity 不一致")
    actual_hash = record_hash(record)
    if record_digest != actual_hash:
        raise SchemaError(ReasonCode.HASH_MISMATCH, "record_sha256 不一致")


def canonical_schema_document() -> dict[str, Any]:
    """供 manifest 固定的机器可读 typed-union schema。"""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CANONICAL_SCHEMA_VERSION,
        "$defs": {
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "size": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer", "minimum": 1},
                    {"type": "integer", "minimum": 1},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "mediaBase": {
                "type": "object",
                "required": [
                    "asset_id",
                    "path",
                    "media_type",
                    "sha256",
                    "size_bytes",
                    "extension",
                    "role",
                    "source_sha256",
                ],
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "pattern": "^asset_[0-9a-f]{64}$",
                    },
                    "path": {"type": "string", "minLength": 1},
                    "media_type": {"enum": [item.value for item in MediaType]},
                    "sha256": {"$ref": "#/$defs/sha256"},
                    "size_bytes": {"type": "integer", "minimum": 1},
                    "extension": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "minLength": 1},
                    "source_sha256": {"$ref": "#/$defs/sha256"},
                },
            },
            "rasterMedia": {
                "allOf": [
                    {"$ref": "#/$defs/mediaBase"},
                    {
                        "type": "object",
                        "required": ["width", "height", "mode"],
                        "properties": {
                            "media_type": {"enum": ["image", "mask"]},
                            "width": {"type": "integer", "minimum": 1},
                            "height": {"type": "integer", "minimum": 1},
                            "mode": {"enum": ["RGB", "L"]},
                        },
                    },
                ],
                "unevaluatedProperties": False,
            },
            "arrayMedia": {
                "allOf": [
                    {"$ref": "#/$defs/mediaBase"},
                    {
                        "type": "object",
                        "properties": {"media_type": {"const": "array"}},
                    },
                ],
                "unevaluatedProperties": False,
            },
            "noneTarget": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type"],
                "properties": {"type": {"const": "none"}},
            },
            "maskTarget": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "type",
                    "image_role",
                    "mask_role",
                    "source_convention",
                    "canonical_convention",
                    "mask_asset_id",
                    "image_asset_id",
                    "size",
                ],
                "properties": {
                    "type": {"const": "mask"},
                    "image_role": {"type": "string", "minLength": 1},
                    "mask_role": {"type": "string", "minLength": 1},
                    "source_convention": {"const": "binary_raster_top_left"},
                    "canonical_convention": {"const": "binary_raster_top_left"},
                    "mask_asset_id": {
                        "type": "string",
                        "pattern": "^asset_[0-9a-f]{64}$",
                    },
                    "image_asset_id": {
                        "type": "string",
                        "pattern": "^asset_[0-9a-f]{64}$",
                    },
                    "size": {"$ref": "#/$defs/size"},
                    "empty_allowed": {"type": "boolean"},
                },
            },
            "bboxItem": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source",
                    "source_convention",
                    "source_image_size",
                    "canonical_xyxy_norm",
                ],
                "properties": {
                    "source": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "source_convention": {
                        "enum": [
                            "xyxy_normalized_top_left",
                            "xywh_pixel_top_left",
                        ]
                    },
                    "source_image_size": {"$ref": "#/$defs/size"},
                    "canonical_xyxy_norm": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string", "minLength": 1},
                    "object_key": {"type": "string", "minLength": 1},
                },
            },
            "bboxTarget": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "type",
                    "image_role",
                    "source_convention",
                    "canonical_convention",
                    "boxes",
                ],
                "properties": {
                    "type": {"const": "bbox"},
                    "image_role": {"type": "string", "minLength": 1},
                    "source_convention": {
                        "enum": [
                            "xyxy_normalized_top_left",
                            "xywh_pixel_top_left",
                        ]
                    },
                    "canonical_convention": {
                        "const": "xyxy_normalized_top_left"
                    },
                    "boxes": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/bboxItem"},
                        "minItems": 1,
                    },
                },
            },
        },
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "record_id",
            "parent_id",
            "source",
            "source_record_id",
            "source_split",
            "logical_role",
            "task_family",
            "supervision_kind",
            "input_layout",
            "output_modality",
            "media",
            "target",
            "instruction",
            "training_responses",
            "reference_responses",
            "deterministic_facts",
            "annotation",
            "coordinate_convention",
            "transforms",
            "quality_flags",
            "provenance_ids",
            "record_sha256",
        ],
        "properties": {
            "schema_version": {"const": CANONICAL_SCHEMA_VERSION},
            "record_id": {"type": "string", "pattern": "^record_[0-9a-f]{64}$"},
            "parent_id": {"type": "string", "pattern": "^parent_[0-9a-f]{64}$"},
            "source": {"type": "string", "minLength": 1},
            "source_record_id": {"type": "string", "minLength": 1},
            "source_split": {"type": "string", "minLength": 1},
            "logical_role": {"enum": [item.value for item in LogicalRole]},
            "task_family": {"enum": [item.value for item in TaskFamily]},
            "supervision_kind": {
                "enum": [item.value for item in SupervisionKind]
            },
            "input_layout": {"enum": [item.value for item in InputLayout]},
            "output_modality": {"const": OutputModality.TEXT.value},
            "media": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"$ref": "#/$defs/rasterMedia"},
                        {"$ref": "#/$defs/arrayMedia"},
                    ]
                },
                "minItems": 1,
            },
            "target": {
                "oneOf": [
                    {"$ref": "#/$defs/noneTarget"},
                    {"$ref": "#/$defs/maskTarget"},
                    {"$ref": "#/$defs/bboxTarget"},
                ]
            },
            "instruction": {"type": "string", "minLength": 1},
            "training_responses": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "reference_responses": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            },
            "deterministic_facts": {"type": "object"},
            "annotation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["layer", "review_status"],
                "properties": {
                    "layer": {"enum": [item.value for item in AnnotationLayer]},
                    "review_status": {
                        "enum": [item.value for item in ReviewStatus]
                    },
                },
            },
            "coordinate_convention": {
                "type": "object",
                "additionalProperties": False,
                "required": ["image_origin", "bbox_canonical", "mask_layout"],
                "properties": {
                    "image_origin": {"const": "top_left"},
                    "bbox_canonical": {"const": "xyxy_normalized"},
                    "mask_layout": {"const": "height_width"},
                },
            },
            "transforms": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "asset_role",
                                "type",
                                "orientation",
                                "input_size",
                                "output_size",
                            ],
                            "properties": {
                                "asset_role": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "type": {"const": "exif_orientation"},
                                "orientation": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 8,
                                },
                                "input_size": {"$ref": "#/$defs/size"},
                                "output_size": {"$ref": "#/$defs/size"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["asset_role", "type", "output_mode"],
                            "properties": {
                                "asset_role": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "type": {"const": "color_conversion"},
                                "output_mode": {"const": "RGB"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "asset_role",
                                "type",
                                "interpolation",
                                "input_size",
                                "output_size",
                            ],
                            "properties": {
                                "asset_role": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "type": {"const": "resize"},
                                "interpolation": {
                                    "enum": ["none", "bilinear"]
                                },
                                "input_size": {"$ref": "#/$defs/size"},
                                "output_size": {"$ref": "#/$defs/size"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "asset_role",
                                "type",
                                "orientation",
                                "interpolation",
                            ],
                            "properties": {
                                "asset_role": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "type": {"const": "mask_exif_orientation"},
                                "orientation": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 8,
                                },
                                "interpolation": {"const": "nearest"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "asset_role",
                                "type",
                                "input_size",
                                "output_size",
                                "interpolation",
                            ],
                            "properties": {
                                "asset_role": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "type": {"const": "mask_resize"},
                                "input_size": {"$ref": "#/$defs/size"},
                                "output_size": {"$ref": "#/$defs/size"},
                                "interpolation": {"const": "nearest"},
                            },
                        },
                    ]
                },
                "minItems": 1,
            },
            "quality_flags": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "provenance_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": "^provenance_[0-9a-f]{64}$",
                },
                "minItems": 1,
                "uniqueItems": True,
            },
            "record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }
