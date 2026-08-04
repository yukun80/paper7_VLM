"""Stage 4 v2 Region Corpus、OA-GroundedEval-dev 与人工标注严格合同。"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.config import _load_yaml

from .contracts import EXPECTED_IDENTITY_FIELDS, fail


REGION_CONFIG_SCHEMA = "oa_groundrag.mask_grounded_region.config.v1"
REGION_MANIFEST_SCHEMA = "oa_groundrag.mask_grounded_region.manifest.v1"
REGION_RECORD_SCHEMA = "oa_groundrag.mask_grounded_region.record.v1"
ANNOTATION_QUEUE_SCHEMA = "oa_groundrag.mask_grounded_region.annotation_queue.v1"
ANNOTATION_SCHEMA = "oa_groundrag.mask_grounded_region.annotation.v1"
ANNOTATION_PACKAGE_SCHEMA = "oa_groundrag.mask_grounded_region.annotation_package.v1"
EVAL_CONFIG_SCHEMA = "oa_groundrag.oa_grounded_eval_dev.config.v1"
EVAL_MANIFEST_SCHEMA = "oa_groundrag.oa_grounded_eval_dev.manifest.v1"
COUNTERFACTUAL_GROUP_SCHEMA = "oa_groundrag.oa_grounded_eval_dev.counterfactual_group.v1"

REGION_SELECTION_ALGORITHM = "frozen_stage4a_ordered_ids.v1"
EVAL_SELECTION_ALGORITHM = "oa_grounded_eval_source_size_coverage.v1"
RGB_RENDERER_VERSION = "oa_benchmark_rgb_percentile_2_98.v1"
MASK_RENDERER_VERSION = "binary_mask_png_l_0_255.v1"
COORDINATE_BASIS = "top_left_pixel_xy_half_open"


class RepresentationMode(StrEnum):
    FULL_ONLY = "full_only"
    CROP_ONLY = "crop_only"
    FULL_PLUS_MASK = "full_plus_mask"
    FULL_PLUS_MASK_PLUS_CROP = "full_plus_mask_plus_crop"
    OVERLAY_AUDIT_BASELINE = "overlay_audit_baseline"


class RegionTargetStatus(StrEnum):
    TARGET_PRESENT = "target_present"
    NO_TARGET = "no_target"


class ParentIdentityStatus(StrEnum):
    KNOWN = "known"
    SAMPLE_ONLY = "sample_only"
    UNKNOWN = "unknown"


class AnnotationStatus(StrEnum):
    QUEUED = "queued"
    ANNOTATED = "annotated"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"


class AdjudicationStatus(StrEnum):
    NOT_STARTED = "not_started"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    RESOLVED = "resolved"


class CounterfactualKind(StrEnum):
    BASELINE = "baseline_correct_mask"
    EMPTY_MASK = "empty_mask"
    MASK_SHIFT = "deterministic_mask_shift"
    CONTEXT_REMOVAL = "context_removal"
    MASK_SWAP = "mask_swap"


FORMAL_ROLES: Mapping[RepresentationMode, tuple[str, ...]] = {
    RepresentationMode.FULL_ONLY: ("optical_full",),
    RepresentationMode.CROP_ONLY: ("context_crop",),
    RepresentationMode.FULL_PLUS_MASK: ("optical_full", "binary_mask"),
    RepresentationMode.FULL_PLUS_MASK_PLUS_CROP: (
        "optical_full",
        "binary_mask",
        "context_crop",
    ),
    RepresentationMode.OVERLAY_AUDIT_BASELINE: (),
}


TARGET_APPEARANCE_FIELDS = (
    "tone",
    "texture",
    "vegetation_or_exposure",
    "homogeneity",
    "boundary_visibility",
)
TARGET_MORPHOLOGY_FIELDS = ("shape", "fragmentation", "qualitative_orientation")
SURROUNDING_FIELDS = (
    "land_cover",
    "nearby_objects",
    "visible_terrain_context",
    "human_disturbance",
)
CONTRAST_FIELDS = (
    "tone_contrast",
    "texture_contrast",
    "vegetation_contrast",
    "boundary_transition",
    "adjacency",
)
DESCRIPTION_FIELDS = (
    "target_appearance",
    "target_morphology",
    "surrounding_environment",
    "region_context_contrast",
    "possible_confusers",
    "evidence_sufficiency",
    "short_summary",
    "limitations",
)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("TYPE_MISMATCH", f"{location} 必须是字符串键对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: Sequence[str], location: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        fail(
            "SCHEMA_MISMATCH",
            f"{location} 字段不匹配",
            details={
                "missing": sorted(expected - set(value)),
                "unknown": sorted(set(value) - expected),
            },
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("TYPE_MISMATCH", f"{location} 必须是非空字符串")
    return value.strip()


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail("TYPE_MISMATCH", f"{location} 必须是 >= {minimum} 的整数")
    return value


def _ratio(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("TYPE_MISMATCH", f"{location} 必须是数值")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        fail("TYPE_MISMATCH", f"{location} 必须位于 [0,1]")
    return result


def _sha256(value: Any, location: str) -> str:
    text = _string(value, location)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        fail("TYPE_MISMATCH", f"{location} 必须是小写 SHA-256")
    return text


def _absolute(base: Path, value: Any, location: str) -> Path:
    path = Path(_string(value, location))
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def _config_path(path: Path | str) -> Path:
    result = Path(os.path.abspath(Path(path)))
    linked = first_symlink_component(result)
    if linked is not None:
        fail("OUTPUT_LINK", f"配置路径含 symlink：{linked}")
    return result


def _benchmark(
    value: Any,
    *,
    base: Path,
    location: str,
) -> tuple[Path, dict[str, str]]:
    row = _mapping(value, location)
    _exact(row, ("root", *EXPECTED_IDENTITY_FIELDS), location)
    root = _absolute(base, row.pop("root"), f"{location}.root")
    expected = {
        field: _string(row[field], f"{location}.{field}")
        if field == "schema_version"
        else _sha256(row[field], f"{location}.{field}")
        for field in EXPECTED_IDENTITY_FIELDS
    }
    return root, expected


@dataclass(frozen=True)
class FrozenStage4ASelection:
    root: Path
    manifest_sha256: str
    records_sha256: str
    ledger_sha256: str
    ordered_sample_ids_sha256: str
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_sha256": self.manifest_sha256,
            "records_sha256": self.records_sha256,
            "ledger_sha256": self.ledger_sha256,
            "ordered_sample_ids_sha256": self.ordered_sample_ids_sha256,
            "sample_count": self.sample_count,
        }


def _stage4a_selection(value: Any, *, base: Path, location: str) -> FrozenStage4ASelection:
    row = _mapping(value, location)
    _exact(
        row,
        (
            "root",
            "manifest_sha256",
            "records_sha256",
            "ledger_sha256",
            "ordered_sample_ids_sha256",
            "sample_count",
        ),
        location,
    )
    return FrozenStage4ASelection(
        root=_absolute(base, row["root"], f"{location}.root"),
        manifest_sha256=_sha256(row["manifest_sha256"], f"{location}.manifest_sha256"),
        records_sha256=_sha256(row["records_sha256"], f"{location}.records_sha256"),
        ledger_sha256=_sha256(row["ledger_sha256"], f"{location}.ledger_sha256"),
        ordered_sample_ids_sha256=_sha256(
            row["ordered_sample_ids_sha256"], f"{location}.ordered_sample_ids_sha256"
        ),
        sample_count=_integer(row["sample_count"], f"{location}.sample_count", 1),
    )


@dataclass(frozen=True)
class RegionCorpusConfig:
    config_path: Path
    config_file_sha256: str
    semantic_sha256: str
    benchmark_root: Path
    benchmark_expected: Mapping[str, str]
    stage4a_selection: FrozenStage4ASelection
    output_root: Path
    context_margin_ratio: float
    overlay_alpha: float

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGION_CONFIG_SCHEMA,
            "benchmark": {"root": str(self.benchmark_root), **dict(self.benchmark_expected)},
            "stage4a_selection": self.stage4a_selection.to_dict(),
            "output_root": str(self.output_root),
            "rendering": {
                "context_margin_ratio": self.context_margin_ratio,
                "overlay_alpha": self.overlay_alpha,
                "rgb_renderer": RGB_RENDERER_VERSION,
                "mask_renderer": MASK_RENDERER_VERSION,
            },
        }


def load_region_corpus_config(path: Path | str) -> RegionCorpusConfig:
    config_path = _config_path(path)
    row = _load_yaml(config_path)
    _exact(row, ("schema_version", "benchmark", "stage4a_selection", "output_root", "rendering"), "$")
    if row["schema_version"] != REGION_CONFIG_SCHEMA:
        fail("CONFIG_INVALID", f"仅支持 {REGION_CONFIG_SCHEMA}")
    benchmark_root, expected = _benchmark(row["benchmark"], base=config_path.parent, location="$.benchmark")
    selection = _stage4a_selection(row["stage4a_selection"], base=config_path.parent, location="$.stage4a_selection")
    if selection.sample_count != 500:
        fail("CONFIG_INVALID", "首版 Region Corpus 必须精确复用 500 个 Stage 4A IDs")
    rendering = _mapping(row["rendering"], "$.rendering")
    _exact(rendering, ("context_margin_ratio", "overlay_alpha"), "$.rendering")
    config = RegionCorpusConfig(
        config_path=config_path,
        config_file_sha256=sha256_file(config_path),
        semantic_sha256="",
        benchmark_root=benchmark_root,
        benchmark_expected=expected,
        stage4a_selection=selection,
        output_root=_absolute(config_path.parent, row["output_root"], "$.output_root"),
        context_margin_ratio=_ratio(rendering["context_margin_ratio"], "$.rendering.context_margin_ratio"),
        overlay_alpha=_ratio(rendering["overlay_alpha"], "$.rendering.overlay_alpha"),
    )
    if config.context_margin_ratio != 0.15 or config.overlay_alpha != 0.45:
        fail("CONFIG_INVALID", "v1 rendering 必须冻结为 margin=0.15、overlay_alpha=0.45")
    return replace(config, semantic_sha256=sha256_text(canonical_json(config.semantic_dict())))


@dataclass(frozen=True)
class EvalQuota:
    total: int
    target: int
    no_target: int
    small: int
    medium: int
    large: int


@dataclass(frozen=True)
class GroundedEvalConfig:
    config_path: Path
    config_file_sha256: str
    semantic_sha256: str
    benchmark_root: Path
    benchmark_expected: Mapping[str, str]
    train_corpus_root: Path
    output_root: Path
    sample_count: int
    seed: int
    algorithm_version: str
    source_order: tuple[str, ...]
    per_source: EvalQuota
    candidate_pool_min: int
    small_upper_exclusive: float
    medium_upper_exclusive: float
    context_margin_ratio: float
    overlay_alpha: float

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVAL_CONFIG_SCHEMA,
            "benchmark": {"root": str(self.benchmark_root), **dict(self.benchmark_expected)},
            "train_corpus": {"root": str(self.train_corpus_root), "schema_version": REGION_MANIFEST_SCHEMA},
            "output_root": str(self.output_root),
            "selection": {
                "split": "val",
                "sample_count": self.sample_count,
                "seed": self.seed,
                "algorithm_version": self.algorithm_version,
                "source_order": list(self.source_order),
                "per_source": self.per_source.__dict__,
                "candidate_pool_min": self.candidate_pool_min,
                "size_bins": {
                    "small_upper_exclusive": self.small_upper_exclusive,
                    "medium_upper_exclusive": self.medium_upper_exclusive,
                },
            },
            "rendering": {
                "context_margin_ratio": self.context_margin_ratio,
                "overlay_alpha": self.overlay_alpha,
                "rgb_renderer": RGB_RENDERER_VERSION,
                "mask_renderer": MASK_RENDERER_VERSION,
            },
        }


def load_grounded_eval_config(path: Path | str) -> GroundedEvalConfig:
    config_path = _config_path(path)
    row = _load_yaml(config_path)
    _exact(row, ("schema_version", "benchmark", "train_corpus", "output_root", "selection", "rendering"), "$")
    if row["schema_version"] != EVAL_CONFIG_SCHEMA:
        fail("CONFIG_INVALID", f"仅支持 {EVAL_CONFIG_SCHEMA}")
    benchmark_root, expected = _benchmark(row["benchmark"], base=config_path.parent, location="$.benchmark")
    train = _mapping(row["train_corpus"], "$.train_corpus")
    _exact(train, ("root", "schema_version"), "$.train_corpus")
    if train["schema_version"] != REGION_MANIFEST_SCHEMA:
        fail("CONFIG_INVALID", "train_corpus schema_version 非法")
    selection = _mapping(row["selection"], "$.selection")
    _exact(
        selection,
        (
            "split", "sample_count", "seed", "algorithm_version", "source_order",
            "per_source", "candidate_pool_min", "size_bins",
        ),
        "$.selection",
    )
    if selection["split"] != "val":
        fail("SPLIT_FORBIDDEN", "OA-GroundedEval-dev 配置只允许 val；test 被代码级拒绝")
    if selection["algorithm_version"] != EVAL_SELECTION_ALGORITHM:
        fail("CONFIG_INVALID", f"algorithm_version 必须是 {EVAL_SELECTION_ALGORITHM}")
    sources = selection["source_order"]
    if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
        fail("CONFIG_INVALID", "source_order 必须是唯一非空列表")
    source_order = tuple(_string(value, "$.selection.source_order[]") for value in sources)
    quota_row = _mapping(selection["per_source"], "$.selection.per_source")
    _exact(quota_row, ("total", "target", "no_target", "small", "medium", "large"), "$.selection.per_source")
    quota = EvalQuota(*(_integer(quota_row[field], f"$.selection.per_source.{field}") for field in (
        "total", "target", "no_target", "small", "medium", "large"
    )))
    if quota.total != quota.target + quota.no_target or quota.target != quota.small + quota.medium + quota.large:
        fail("CONFIG_INVALID", "Eval per_source quota 加总不一致")
    size_bins = _mapping(selection["size_bins"], "$.selection.size_bins")
    _exact(size_bins, ("small_upper_exclusive", "medium_upper_exclusive"), "$.selection.size_bins")
    small = _ratio(size_bins["small_upper_exclusive"], "$.selection.size_bins.small_upper_exclusive")
    medium = _ratio(size_bins["medium_upper_exclusive"], "$.selection.size_bins.medium_upper_exclusive")
    rendering = _mapping(row["rendering"], "$.rendering")
    _exact(rendering, ("context_margin_ratio", "overlay_alpha"), "$.rendering")
    config = GroundedEvalConfig(
        config_path=config_path,
        config_file_sha256=sha256_file(config_path),
        semantic_sha256="",
        benchmark_root=benchmark_root,
        benchmark_expected=expected,
        train_corpus_root=_absolute(config_path.parent, train["root"], "$.train_corpus.root"),
        output_root=_absolute(config_path.parent, row["output_root"], "$.output_root"),
        sample_count=_integer(selection["sample_count"], "$.selection.sample_count", 1),
        seed=_integer(selection["seed"], "$.selection.seed"),
        algorithm_version=str(selection["algorithm_version"]),
        source_order=source_order,
        per_source=quota,
        candidate_pool_min=_integer(selection["candidate_pool_min"], "$.selection.candidate_pool_min", 1),
        small_upper_exclusive=small,
        medium_upper_exclusive=medium,
        context_margin_ratio=_ratio(rendering["context_margin_ratio"], "$.rendering.context_margin_ratio"),
        overlay_alpha=_ratio(rendering["overlay_alpha"], "$.rendering.overlay_alpha"),
    )
    if (
        config.sample_count != 100
        or quota != EvalQuota(20, 16, 4, 5, 6, 5)
        or small != 0.01
        or medium != 0.10
        or config.context_margin_ratio != 0.15
        or config.overlay_alpha != 0.45
    ):
        fail("CONFIG_INVALID", "OA-GroundedEval-dev v1 的 100-parent quota/bin/rendering 已冻结")
    return replace(config, semantic_sha256=sha256_text(canonical_json(config.semantic_dict())))


def parent_identity(row: Mapping[str, Any]) -> tuple[str, ParentIdentityStatus]:
    """未知 group 必须保持未知，不从文件名补造 parent。"""

    status_text = str(row.get("group_status", "unknown"))
    group_id = row.get("source_group_id")
    if status_text == ParentIdentityStatus.KNOWN.value and isinstance(group_id, str) and group_id:
        return group_id, ParentIdentityStatus.KNOWN
    status = ParentIdentityStatus.SAMPLE_ONLY if status_text == "sample_only" else ParentIdentityStatus.UNKNOWN
    return _string(row.get("sample_id"), "row.sample_id"), status


def empty_description_template() -> dict[str, Any]:
    return {
        "target_appearance": {field: None for field in TARGET_APPEARANCE_FIELDS},
        "target_morphology": {field: None for field in TARGET_MORPHOLOGY_FIELDS},
        "surrounding_environment": {field: [] for field in SURROUNDING_FIELDS},
        "region_context_contrast": {
            **{field: None for field in CONTRAST_FIELDS if field != "adjacency"},
            "adjacency": [],
        },
        "possible_confusers": [],
        "evidence_sufficiency": None,
        "short_summary": None,
        "limitations": [],
    }


def annotation_state_template() -> dict[str, Any]:
    return {
        "annotation_status": AnnotationStatus.QUEUED.value,
        "annotator_id": None,
        "reviewer_id": None,
        "adjudicator_id": None,
        "adjudication_status": AdjudicationStatus.NOT_STARTED.value,
        "annotation_version": "region_annotation.v1",
    }


REGION_RECORD_FIELDS = (
    "schema_version", "record_id", "parent_id", "parent_identity_status", "sample_id",
    "source", "split", "source_identity", "mask_source", "target_status",
    "representation_mode", "assets", "formal_model_input_roles", "audit_only_roles",
    "program_facts", "description", "annotation",
)


def _unique_strings(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        fail("TYPE_MISMATCH", f"{location} 必须是非空字符串列表")
    result = [item.strip() for item in value]
    if len(result) != len(set(result)):
        fail("SCHEMA_MISMATCH", f"{location} 不得重复")
    return result


def validate_empty_description(value: Any, *, location: str) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, DESCRIPTION_FIELDS, location)
    appearance = _mapping(row["target_appearance"], f"{location}.target_appearance")
    morphology = _mapping(row["target_morphology"], f"{location}.target_morphology")
    surrounding = _mapping(row["surrounding_environment"], f"{location}.surrounding_environment")
    contrast = _mapping(row["region_context_contrast"], f"{location}.region_context_contrast")
    _exact(appearance, TARGET_APPEARANCE_FIELDS, f"{location}.target_appearance")
    _exact(morphology, TARGET_MORPHOLOGY_FIELDS, f"{location}.target_morphology")
    _exact(surrounding, SURROUNDING_FIELDS, f"{location}.surrounding_environment")
    _exact(contrast, CONTRAST_FIELDS, f"{location}.region_context_contrast")
    if any(value is not None for value in appearance.values()) or any(value is not None for value in morphology.values()):
        fail("ANNOTATION_INVALID", f"{location} 初始 target 字段必须为 null")
    if any(value is not None for key, value in contrast.items() if key != "adjacency"):
        fail("ANNOTATION_INVALID", f"{location} 初始 contrast 字段必须为 null")
    if row["evidence_sufficiency"] is not None or row["short_summary"] is not None:
        fail("ANNOTATION_INVALID", f"{location} 初始 summary 字段必须为 null")
    for field in SURROUNDING_FIELDS:
        if surrounding[field] != []:
            fail("ANNOTATION_INVALID", f"{location}.surrounding_environment.{field} 必须为空")
    for field in ("adjacency",):
        if contrast[field] != []:
            fail("ANNOTATION_INVALID", f"{location}.region_context_contrast.{field} 必须为空")
    if row["possible_confusers"] != [] or row["limitations"] != []:
        fail("ANNOTATION_INVALID", f"{location} 初始列表必须为空")
    return row


def validate_region_record(value: Any, *, expected_split: str | None = None, location: str = "record") -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, REGION_RECORD_FIELDS, location)
    if row["schema_version"] != REGION_RECORD_SCHEMA:
        fail("SCHEMA_MISMATCH", f"{location}.schema_version 非法")
    for field in ("record_id", "parent_id", "sample_id", "source", "split", "mask_source"):
        _string(row[field], f"{location}.{field}")
    if expected_split is not None and row["split"] != expected_split:
        fail("SPLIT_FORBIDDEN", f"{location} 只允许 {expected_split}")
    try:
        target = RegionTargetStatus(row["target_status"])
        parent_status = ParentIdentityStatus(row["parent_identity_status"])
        mode = RepresentationMode(row["representation_mode"])
    except (TypeError, ValueError) as error:
        fail("INVALID_ENUM", f"{location} enum 非法", details={"error": str(error)})
    del parent_status
    assets = _mapping(row["assets"], f"{location}.assets")
    _exact(assets, ("optical_full", "binary_mask", "context_crop", "audit_overlay"), f"{location}.assets")
    if not isinstance(assets["optical_full"], str) or not isinstance(assets["binary_mask"], str):
        fail("ASSET_REFERENCE_MISMATCH", f"{location} 必须引用 full/mask")
    formal = _unique_strings(row["formal_model_input_roles"], f"{location}.formal_model_input_roles")
    audit = _unique_strings(row["audit_only_roles"], f"{location}.audit_only_roles")
    if tuple(formal) != FORMAL_ROLES[mode]:
        fail("ASSET_ROLE_LEAKAGE", f"{location} formal roles 与 representation mode 不一致")
    if set(formal) & set(audit) or "audit_overlay" in formal:
        fail("ASSET_ROLE_LEAKAGE", f"{location} formal/audit role 泄漏")
    if mode is RepresentationMode.OVERLAY_AUDIT_BASELINE:
        if assets["audit_overlay"] is None or audit != ["audit_overlay"]:
            fail("ASSET_ROLE_LEAKAGE", f"{location} overlay baseline 必须仅引用 audit overlay")
    elif assets["audit_overlay"] is not None and audit != ["audit_overlay"]:
        fail("ASSET_ROLE_LEAKAGE", f"{location} audit overlay 必须显式标为 audit-only")
    if target is RegionTargetStatus.NO_TARGET and mode is RepresentationMode.FULL_PLUS_MASK_PLUS_CROP:
        fail("MASK_INVALID", f"{location} no-target 不得要求 crop")
    _mapping(row["source_identity"], f"{location}.source_identity")
    _mapping(row["program_facts"], f"{location}.program_facts")
    validate_empty_description(row["description"], location=f"{location}.description")
    annotation = _mapping(row["annotation"], f"{location}.annotation")
    _exact(
        annotation,
        (
            "annotation_status", "annotator_id", "reviewer_id", "adjudicator_id",
            "adjudication_status", "annotation_version",
        ),
        f"{location}.annotation",
    )
    if annotation != annotation_state_template():
        fail("ANNOTATION_INVALID", f"{location} 发布记录必须保持 queued/空审核身份")
    return row


def size_bin(area_ratio: float, *, small_upper: float = 0.01, medium_upper: float = 0.10) -> str:
    if area_ratio <= 0:
        return "empty"
    if area_ratio < small_upper:
        return "small"
    if area_ratio < medium_upper:
        return "medium"
    return "large"
