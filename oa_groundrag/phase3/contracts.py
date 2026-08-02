"""RS-GeneralDesc 的原生 v1 canonical 与 adapter 中间合同。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .common import canonical_json, portable_relative_path, reject_nonfinite, sha256_text
from .errors import ReasonCode, SchemaError


CANONICAL_SCHEMA_VERSION = "rs_generaldesc.canonical.v1"
BUILD_CONFIG_VERSION = "rs_generaldesc.build_config.v1"
EXPORT_CONFIG_VERSION = "rs_generaldesc.qwen_export.v1"
MANIFEST_VERSION = "rs_generaldesc.manifest.v1"
VALIDATION_VERSION = "rs_generaldesc.validation.v1"
STATISTICS_SCHEMA_VERSION = "rs_generaldesc.statistics.v1"
PROVENANCE_SCHEMA_VERSION = "rs_generaldesc.provenance.v1"
EXTERNAL_SPLIT_SCHEMA_VERSION = "rs_generaldesc.external_split.v1"
QWEN_EXPORT_MANIFEST_VERSION = "rs_generaldesc.qwen_export_manifest.v1"
RELEASE_EQUIVALENCE_SCHEMA_VERSION = "rs_generaldesc.release_equivalence.v1"
HASH_SCHEMA_VERSION = "rs_generaldesc.hashes.v1"
QWEN_TEMPLATE_VERSION = "qwen3vl_messages.v2"
DESCRIPTION_MULTITASK_PROFILE = "description_multitask.v1"
EXTERNAL_SPLIT_VERSION = "external_parent_split.v1"
BENCHMARK_SCOPE = "rs_generaldesc_external_train_val"
RS_GENERALDESC_SOURCES = frozenset({"rsgpt", "mmrs1m", "disasterm3"})

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


class AnnotationLayer(StrEnum):
    EXTERNAL_SOURCE = "external_source"


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "not_required"


class TaskFamily(StrEnum):
    GLOBAL_CAPTION = "global_caption"
    VISUAL_QA = "visual_qa"
    BBOX_REGION_CAPTION = "bbox_region_caption"
    SCENE_UNDERSTANDING = "scene_understanding"
    OBJECT_COUNT = "object_count"
    SPATIAL_RELATION = "spatial_relation"
    VISIBLE_CHANGE_REPORT = "visible_change_report"


RS_GENERALDESC_ROLES = frozenset(LogicalRole)
RS_GENERALDESC_TASK_FAMILIES = frozenset(TaskFamily)


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


class OutputModality(StrEnum):
    TEXT = "text"


TASK_SUPERVISION_KINDS: Mapping[TaskFamily, frozenset[SupervisionKind]] = {
    TaskFamily.GLOBAL_CAPTION: frozenset({SupervisionKind.LONG_DESCRIPTION}),
    TaskFamily.VISUAL_QA: frozenset({SupervisionKind.SHORT_QA}),
    TaskFamily.BBOX_REGION_CAPTION: frozenset({SupervisionKind.REGION_DESCRIPTION}),
    TaskFamily.SCENE_UNDERSTANDING: frozenset({SupervisionKind.SHORT_QA}),
    TaskFamily.OBJECT_COUNT: frozenset({SupervisionKind.NUMERIC_QA}),
    TaskFamily.SPATIAL_RELATION: frozenset({SupervisionKind.SPATIAL_DESCRIPTION}),
    TaskFamily.VISIBLE_CHANGE_REPORT: frozenset({SupervisionKind.STRUCTURED_REPORT}),
}

TASK_INPUT_LAYOUTS: Mapping[TaskFamily, frozenset[InputLayout]] = {
    TaskFamily.GLOBAL_CAPTION: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.VISUAL_QA: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.BBOX_REGION_CAPTION: frozenset({InputLayout.BBOX_REGION}),
    TaskFamily.SCENE_UNDERSTANDING: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.OBJECT_COUNT: frozenset({InputLayout.SINGLE_IMAGE}),
    TaskFamily.SPATIAL_RELATION: frozenset({InputLayout.BOXED_IMAGE}),
    TaskFamily.VISIBLE_CHANGE_REPORT: frozenset({InputLayout.PRE_POST}),
}


class MediaType(StrEnum):
    IMAGE = "image"


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
        if self.media_type is not MediaType.IMAGE:
            raise SchemaError(
                ReasonCode.INVALID_ENUM,
                f"{self.role}: RS-GeneralDesc 只接受 image media",
            )
        portable_relative_path(
            self.source_ref,
            location=f"asset[{self.role}].source_ref",
        )


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
    identity = ["rs_generaldesc.parent.v1", source, parent_key]
    return f"parent_{sha256_text(canonical_json(identity))}"


def parent_content_fingerprint(records: list[Mapping[str, Any]]) -> str:
    """对一个已发布 parent 的非身份内容生成稳定 fingerprint。"""

    normalized: list[dict[str, Any]] = []
    omitted = {
        "schema_version",
        "record_id",
        "parent_id",
        "logical_role",
        "provenance_ids",
        "record_sha256",
    }
    for record in records:
        projected = {
            key: value for key, value in record.items() if key not in omitted
        }
        coordinate = projected.get("coordinate_convention")
        if isinstance(coordinate, dict):
            projected["coordinate_convention"] = {
                key: coordinate[key]
                for key in ("image_origin", "bbox_canonical")
                if key in coordinate
            }
        normalized.append(projected)
    normalized.sort(key=canonical_json)
    return sha256_text(canonical_json(normalized))


def parent_id_from_fingerprint(fingerprint: str) -> str:
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise SchemaError(ReasonCode.HASH_MISMATCH, "parent fingerprint 格式非法")
    identity = ["rs_generaldesc.parent.v1", fingerprint]
    return f"parent_{sha256_text(canonical_json(identity))}"


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
    if (
        role not in RS_GENERALDESC_ROLES
        or layer is not AnnotationLayer.EXTERNAL_SOURCE
        or review is not ReviewStatus.NOT_REQUIRED
    ):
        raise SchemaError(
            ReasonCode.ROLE_CONTAMINATION,
            "RS-GeneralDesc 只接受 external train/val 与 external_source",
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
        raise SchemaError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是非空字符串")
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
        raise SchemaError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是有限数值")
    return float(value)


def _size(value: Any, *, location: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是 [width,height]")
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
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"{location}[{index}]: 格式非法")
        output.append(text)
    if len(set(output)) != len(output):
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"{location}: 不允许重复值")
    return output


def _validate_bbox_target(
    target: Mapping[str, Any],
    media_by_role: Mapping[str, Mapping[str, Any]],
) -> None:
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
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "bbox 坐标约定非法")
    image_role = _string(target["image_role"], location="$.target.image_role")
    if image_role not in media_by_role:
        raise SchemaError(ReasonCode.BBOX_INVALID, "bbox image_role 未引用 media")
    boxes = target["boxes"]
    if not isinstance(boxes, list) or not boxes:
        raise SchemaError(ReasonCode.BBOX_INVALID, "bbox boxes 必须非空")
    for index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise SchemaError(ReasonCode.TYPE_MISMATCH, f"bbox[{index}] 必须是对象")
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
            raise SchemaError(ReasonCode.BBOX_INVALID, f"bbox[{index}] 必须包含四元数值")
        for value_index, value in enumerate(source_values):
            _number(value, location=f"$.target.boxes[{index}].source[{value_index}]")
        canonical = [
            _number(
                value,
                location=f"$.target.boxes[{index}].canonical_xyxy_norm[{value_index}]",
            )
            for value_index, value in enumerate(canonical_values)
        ]
        if (
            min(canonical) < 0
            or max(canonical) > 1
            or canonical[2] <= canonical[0]
            or canonical[3] <= canonical[1]
        ):
            raise SchemaError(ReasonCode.BBOX_INVALID, f"bbox[{index}] canonical 坐标非法")
        _size(box["source_image_size"], location=f"$.target.boxes[{index}].source_image_size")
        for optional_key in ("label", "object_key"):
            if optional_key in box:
                _string(box[optional_key], location=f"$.target.boxes[{index}].{optional_key}")


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
    if source not in RS_GENERALDESC_SOURCES:
        raise SchemaError(ReasonCode.INVALID_ENUM, f"不支持 source={source}")
    _string(record["source_record_id"], location="$.source_record_id")
    _string(record["source_split"], location="$.source_split")
    instruction = _string(record["instruction"], location="$.instruction")
    annotation = record["annotation"]
    if not isinstance(annotation, dict):
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "annotation 必须是对象")
    _exact_fields(annotation, required={"layer", "review_status"}, location="$.annotation")
    try:
        role = LogicalRole(record["logical_role"])
        layer = AnnotationLayer(annotation["layer"])
        review = ReviewStatus(annotation["review_status"])
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
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "canonical 只接受 text 输出")
    validate_role_layer(role, layer, review)

    media_rows = record["media"]
    if not isinstance(media_rows, list) or not media_rows:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "canonical media 必须为非空列表")
    media_by_role: dict[str, Mapping[str, Any]] = {}
    media_asset_ids: list[str] = []
    for index, media in enumerate(media_rows):
        if not isinstance(media, dict):
            raise SchemaError(ReasonCode.TYPE_MISMATCH, f"media[{index}] 必须为对象")
        _exact_fields(
            media,
            required={
                "asset_id",
                "path",
                "media_type",
                "sha256",
                "size_bytes",
                "extension",
                "role",
                "source_sha256",
                "width",
                "height",
                "mode",
            },
            location=f"$.media[{index}]",
        )
        try:
            media_type = MediaType(media["media_type"])
        except (TypeError, ValueError) as error:
            raise SchemaError(ReasonCode.INVALID_ENUM, f"media[{index}].media_type 非法") from error
        if media_type is not MediaType.IMAGE or media["mode"] != "RGB":
            raise SchemaError(ReasonCode.INVALID_ENUM, "RS-GeneralDesc media 必须是 RGB image")
        asset_identifier = _string(media["asset_id"], location=f"$.media[{index}].asset_id")
        digest = _string(media["sha256"], location=f"$.media[{index}].sha256")
        source_digest = _string(media["source_sha256"], location=f"$.media[{index}].source_sha256")
        if (
            _ASSET_ID_PATTERN.fullmatch(asset_identifier) is None
            or _SHA256_PATTERN.fullmatch(digest) is None
            or _SHA256_PATTERN.fullmatch(source_digest) is None
            or asset_identifier != f"asset_{digest}"
        ):
            raise SchemaError(ReasonCode.HASH_MISMATCH, f"media[{index}] hash/asset_id 非法")
        path = _string(media["path"], location=f"$.media[{index}].path")
        portable_relative_path(path, location=f"media[{index}].path")
        if not path.startswith("assets/image/"):
            raise SchemaError(ReasonCode.PATH_ESCAPE, "media path 必须位于 assets/image")
        role_name = _string(media["role"], location=f"$.media[{index}].role")
        if role_name in media_by_role:
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"media role 重复：{role_name}")
        media_by_role[role_name] = media
        media_asset_ids.append(asset_identifier)
        _string(media["extension"], location=f"$.media[{index}].extension")
        _integer(media["size_bytes"], location=f"$.media[{index}].size_bytes", minimum=1)
        _integer(media["width"], location=f"$.media[{index}].width", minimum=1)
        _integer(media["height"], location=f"$.media[{index}].height", minimum=1)

    target = record["target"]
    if not isinstance(target, dict) or target.get("type") not in {"none", "bbox"}:
        raise SchemaError(ReasonCode.INVALID_ENUM, "target.type 必须是 none/bbox")
    if target["type"] == "none":
        _exact_fields(target, required={"type"}, location="$.target")
    else:
        _validate_bbox_target(target, media_by_role)
    image_roles = set(media_by_role)
    if input_layout is InputLayout.SINGLE_IMAGE:
        if target["type"] != "none" or len(image_roles) != 1:
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "single_image 必须是单图且 target=none")
    elif input_layout in {InputLayout.BBOX_REGION, InputLayout.BOXED_IMAGE}:
        if target["type"] != "bbox" or len(image_roles) != 1:
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"{input_layout.value} 必须是单图 bbox")
    elif input_layout is InputLayout.PRE_POST:
        if target["type"] != "none" or image_roles != {"pre_image", "post_image"}:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "pre_post 必须只含 pre_image/post_image 且 target=none",
            )

    _string_list(record["training_responses"], location="$.training_responses", allow_empty=False)
    _string_list(record["reference_responses"], location="$.reference_responses", allow_empty=False)
    if not isinstance(record["deterministic_facts"], dict):
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "deterministic_facts 必须是对象")
    coordinate = record["coordinate_convention"]
    if coordinate != {"image_origin": "top_left", "bbox_canonical": "xyxy_normalized"}:
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "coordinate_convention 非 canonical v1 约定")
    transforms = record["transforms"]
    if not isinstance(transforms, list) or not transforms:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, "transforms 必须是非空列表")
    transform_fields = {
        "exif_orientation": {"asset_role", "type", "orientation", "input_size", "output_size"},
        "color_conversion": {"asset_role", "type", "output_mode"},
        "resize": {"asset_role", "type", "interpolation", "input_size", "output_size"},
    }
    transform_types_by_role: dict[str, set[str]] = {role_name: set() for role_name in media_by_role}
    for index, transform in enumerate(transforms):
        if not isinstance(transform, dict) or transform.get("type") not in transform_fields:
            raise SchemaError(ReasonCode.INVALID_ENUM, f"transforms[{index}].type 非法")
        transform_type = str(transform["type"])
        _exact_fields(transform, required=transform_fields[transform_type], location=f"$.transforms[{index}]")
        role_name = _string(transform["asset_role"], location=f"$.transforms[{index}].asset_role")
        if role_name not in media_by_role:
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"transforms[{index}] 引用未知 asset_role")
        if transform_type in transform_types_by_role[role_name]:
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"{role_name}: transform 类型重复")
        transform_types_by_role[role_name].add(transform_type)
        if "orientation" in transform:
            orientation = _integer(transform["orientation"], location=f"$.transforms[{index}].orientation", minimum=1)
            if orientation > 8:
                raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "EXIF orientation 必须 <= 8")
        for size_key in ("input_size", "output_size"):
            if size_key in transform:
                _size(transform[size_key], location=f"$.transforms[{index}].{size_key}")
        if transform_type == "color_conversion" and transform["output_mode"] != "RGB":
            raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "color_conversion output_mode 必须为 RGB")
        if "interpolation" in transform and transform["interpolation"] not in {"none", "bilinear"}:
            raise SchemaError(ReasonCode.INVALID_ENUM, "resize interpolation 非法")
    expected_transforms = {"exif_orientation", "color_conversion", "resize"}
    if any(value != expected_transforms for value in transform_types_by_role.values()):
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, "每张 image 必须有完整的三项规范化 transform")

    _string_list(record["quality_flags"], location="$.quality_flags", allow_empty=True)
    _string_list(
        record["provenance_ids"],
        location="$.provenance_ids",
        allow_empty=False,
        pattern=_PROVENANCE_ID_PATTERN,
    )
    record_digest = _string(record["record_sha256"], location="$.record_sha256")
    if _SHA256_PATTERN.fullmatch(record_digest) is None:
        raise SchemaError(ReasonCode.HASH_MISMATCH, "record_sha256 格式非法")
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
    if record_digest != record_hash(record):
        raise SchemaError(ReasonCode.HASH_MISMATCH, "record_sha256 不一致")


def canonical_schema_document() -> dict[str, Any]:
    """返回与运行时 validator 同步的严格 JSON Schema。"""

    size = {
        "type": "array",
        "prefixItems": [
            {"type": "integer", "minimum": 1},
            {"type": "integer", "minimum": 1},
        ],
        "minItems": 2,
        "maxItems": 2,
    }
    bbox_item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source",
            "source_convention",
            "source_image_size",
            "canonical_xyxy_norm",
        ],
        "properties": {
            "source": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "source_convention": {"enum": ["xyxy_normalized_top_left", "xywh_pixel_top_left"]},
            "source_image_size": size,
            "canonical_xyxy_norm": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 4,
                "maxItems": 4,
            },
            "label": {"type": "string", "minLength": 1},
            "object_key": {"type": "string", "minLength": 1},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CANONICAL_SCHEMA_VERSION,
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
            "source": {"enum": sorted(RS_GENERALDESC_SOURCES)},
            "source_record_id": {"type": "string", "minLength": 1},
            "source_split": {"type": "string", "minLength": 1},
            "logical_role": {"enum": [item.value for item in LogicalRole]},
            "task_family": {"enum": [item.value for item in TaskFamily]},
            "supervision_kind": {"enum": [item.value for item in SupervisionKind]},
            "input_layout": {"enum": [item.value for item in InputLayout]},
            "output_modality": {"const": OutputModality.TEXT.value},
            "media": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "asset_id",
                        "path",
                        "media_type",
                        "sha256",
                        "size_bytes",
                        "extension",
                        "role",
                        "source_sha256",
                        "width",
                        "height",
                        "mode",
                    ],
                    "properties": {
                        "asset_id": {"type": "string", "pattern": "^asset_[0-9a-f]{64}$"},
                        "path": {"type": "string", "pattern": "^assets/image/"},
                        "media_type": {"const": "image"},
                        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "size_bytes": {"type": "integer", "minimum": 1},
                        "extension": {"type": "string", "minLength": 1},
                        "role": {"type": "string", "minLength": 1},
                        "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                        "mode": {"const": "RGB"},
                    },
                },
            },
            "target": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type"],
                        "properties": {"type": {"const": "none"}},
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "image_role", "source_convention", "canonical_convention", "boxes"],
                        "properties": {
                            "type": {"const": "bbox"},
                            "image_role": {"type": "string", "minLength": 1},
                            "source_convention": {"enum": ["xyxy_normalized_top_left", "xywh_pixel_top_left"]},
                            "canonical_convention": {"const": "xyxy_normalized_top_left"},
                            "boxes": {"type": "array", "items": bbox_item, "minItems": 1},
                        },
                    },
                ]
            },
            "instruction": {"type": "string", "minLength": 1},
            "training_responses": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "uniqueItems": True},
            "reference_responses": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "uniqueItems": True},
            "deterministic_facts": {"type": "object"},
            "annotation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["layer", "review_status"],
                "properties": {
                    "layer": {"const": AnnotationLayer.EXTERNAL_SOURCE.value},
                    "review_status": {"const": ReviewStatus.NOT_REQUIRED.value},
                },
            },
            "coordinate_convention": {
                "type": "object",
                "additionalProperties": False,
                "required": ["image_origin", "bbox_canonical"],
                "properties": {
                    "image_origin": {"const": "top_left"},
                    "bbox_canonical": {"const": "xyxy_normalized"},
                },
            },
            "transforms": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["asset_role", "type", "orientation", "input_size", "output_size"],
                            "properties": {
                                "asset_role": {"type": "string", "minLength": 1},
                                "type": {"const": "exif_orientation"},
                                "orientation": {"type": "integer", "minimum": 1, "maximum": 8},
                                "input_size": size,
                                "output_size": size,
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["asset_role", "type", "output_mode"],
                            "properties": {
                                "asset_role": {"type": "string", "minLength": 1},
                                "type": {"const": "color_conversion"},
                                "output_mode": {"const": "RGB"},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["asset_role", "type", "interpolation", "input_size", "output_size"],
                            "properties": {
                                "asset_role": {"type": "string", "minLength": 1},
                                "type": {"const": "resize"},
                                "interpolation": {"enum": ["none", "bilinear"]},
                                "input_size": size,
                                "output_size": size,
                            },
                        },
                    ]
                },
            },
            "quality_flags": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True},
            "provenance_ids": {"type": "array", "items": {"type": "string", "pattern": "^provenance_[0-9a-f]{64}$"}, "minItems": 1, "uniqueItems": True},
            "record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }
