"""与 canonical 真值解耦的 task-aware Qwen3-VL messages exporter。"""

from __future__ import annotations

import math
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_file,
    stable_hash,
)
from .config import ExportConfig
from .contracts import (
    DESCRIPTION_MULTITASK_PROFILE,
    QWEN_EXPORT_MANIFEST_VERSION,
    QWEN_TEMPLATE_VERSION,
    InputLayout,
    OutputModality,
)
from .dataset import RSGeneralDescDataset
from .errors import ExportError, ReasonCode


EXPORT_STAGING_SENTINEL = ".rs_generaldesc_export_staging"


def _response(
    record: Mapping[str, Any],
    *,
    purpose: str,
    seed: int,
) -> str:
    values = (
        record["training_responses"]
        if purpose == "training"
        else record["reference_responses"]
    )
    if not isinstance(values, list) or not values:
        raise ExportError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{record['record_id']}: export purpose={purpose} 缺少响应",
        )
    rank = int(
        stable_hash(
            seed,
            record["record_id"],
            purpose,
            "response",
        ),
        16,
    )
    return str(values[rank % len(values)])


def _benchmark_image(media: Mapping[str, Any]) -> dict[str, Any]:
    if media.get("media_type") != "image":
        raise ExportError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{media.get('asset_id')}: RS-GeneralDesc 只能引用 image",
        )
    return {
        "type": "image",
        "benchmark_asset": str(media["path"]),
        "asset_role": str(media["role"]),
    }


def _media_by_role(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(media["role"]): media for media in record["media"]}


def _normalized_boxes(record: Mapping[str, Any]) -> list[list[float]]:
    target = record["target"]
    if target.get("type") != "bbox":
        raise ExportError(
            ReasonCode.BBOX_INVALID,
            f"{record['record_id']}: renderer 需要 bbox context",
        )
    boxes = target.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ExportError(
            ReasonCode.BBOX_INVALID,
            f"{record['record_id']}: bbox context 为空",
        )
    output: list[list[float]] = []
    for box in boxes:
        values = box.get("canonical_xyxy_norm") if isinstance(box, dict) else None
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            )
        ):
            raise ExportError(
                ReasonCode.BBOX_INVALID,
                f"{record['record_id']}: canonical bbox context 非法",
            )
        normalized = [float(value) for value in values]
        if (
            min(normalized) < 0
            or max(normalized) > 1
            or normalized[2] <= normalized[0]
            or normalized[3] <= normalized[1]
        ):
            raise ExportError(
                ReasonCode.BBOX_INVALID,
                f"{record['record_id']}: canonical bbox context 越界或零面积",
            )
        output.append(normalized)
    return output


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _bbox_pixels(
    box: list[float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = min(width - 1, max(0, math.floor(box[0] * width)))
    top = min(height - 1, max(0, math.floor(box[1] * height)))
    right = min(width, max(left + 1, math.ceil(box[2] * width)))
    bottom = min(height, max(top + 1, math.ceil(box[3] * height)))
    return left, top, right, bottom


def _render_bbox_region(
    record: Mapping[str, Any],
    *,
    dataset: RSGeneralDescDataset,
    staging: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    boxes = _normalized_boxes(record)
    by_role = _media_by_role(record)
    image_role = str(record["target"].get("image_role"))
    media = by_role.get(image_role)
    if media is None or media.get("media_type") != "image":
        raise ExportError(
            ReasonCode.BBOX_INVALID,
            f"{record['record_id']}: bbox image_role 缺失",
        )
    path = dataset._resolve_asset(media)
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    stroke = max(2, min(width, height) // 200)
    pixels = [
        _bbox_pixels(box, width=width, height=height) for box in boxes
    ]
    colors = (
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 0),
        (255, 165, 0),
    )
    for index, (left, top, right, bottom) in enumerate(pixels):
        draw.rectangle(
            (left, top, right - 1, bottom - 1),
            outline=colors[index % len(colors)],
            width=stroke,
        )
    union = (
        min(value[0] for value in pixels),
        min(value[1] for value in pixels),
        max(value[2] for value in pixels),
        max(value[3] for value in pixels),
    )
    crop = image.crop(union)
    base = Path("rendered") / str(record["record_id"])
    overlay_relative = (base / "bbox_overlay.png").as_posix()
    crop_relative = (base / "region_crop.png").as_posix()
    _save_png(overlay, staging / overlay_relative)
    _save_png(crop, staging / crop_relative)
    context = (
        "The first image is the full image with the requested region outlined; "
        "the second image is the corresponding region crop. "
        "Canonical normalized xyxy coordinates (top-left origin): "
        f"{canonical_json(boxes)}\n{record['instruction']}"
    )
    return (
        [
            {
                "type": "image",
                "export_asset": overlay_relative,
                "asset_role": "bbox_overlay",
            },
            {
                "type": "image",
                "export_asset": crop_relative,
                "asset_role": "region_crop",
            },
            {"type": "text", "text": context},
        ],
        [overlay_relative, crop_relative],
    )


def _render_single_image(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    images = [
        media for media in record["media"] if media["media_type"] == "image"
    ]
    if len(images) != 1 or record["target"]["type"] != "none":
        raise ExportError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{record['record_id']}: single_image context 非法",
        )
    return [
        _benchmark_image(images[0]),
        {"type": "text", "text": str(record["instruction"])},
    ]


def _render_boxed_image(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    boxes = _normalized_boxes(record)
    images = [
        media for media in record["media"] if media["media_type"] == "image"
    ]
    roles = record["deterministic_facts"].get("boxed_object_roles")
    if (
        len(images) != 1
        or not isinstance(roles, list)
        or [item.get("visual_role") for item in roles if isinstance(item, dict)]
        != ["red_box", "blue_box"]
    ):
        raise ExportError(
            ReasonCode.BBOX_INVALID,
            f"{record['record_id']}: red/blue boxed object context 缺失",
        )
    context = (
        "The red box marks the first object and the blue box marks the second "
        "object. Describe the requested spatial relation between those roles. "
        "Canonical normalized xyxy contexts: "
        f"{canonical_json(boxes)}\n{record['instruction']}"
    )
    return [_benchmark_image(images[0]), {"type": "text", "text": context}]


def _render_pre_post(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_role = _media_by_role(record)
    pre = by_role.get("pre_image")
    post = by_role.get("post_image")
    if (
        pre is None
        or post is None
        or pre.get("media_type") != "image"
        or post.get("media_type") != "image"
        or record["target"]["type"] != "none"
    ):
        raise ExportError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{record['record_id']}: pre/post context 缺失",
        )
    return [
        {"type": "text", "text": "Image 1 is the pre-disaster image."},
        _benchmark_image(pre),
        {"type": "text", "text": "Image 2 is the post-disaster image."},
        _benchmark_image(post),
        {"type": "text", "text": str(record["instruction"])},
    ]


def render_canonical_messages(
    record: Mapping[str, Any],
    *,
    purpose: str,
    seed: int,
    dataset: RSGeneralDescDataset,
    derived_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """使用唯一的 RS-GeneralDesc task-aware 模板渲染一条 canonical record。"""

    if purpose not in {"training", "validation"}:
        raise ExportError(
            ReasonCode.INVALID_ENUM,
            f"不支持 renderer purpose={purpose!r}",
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExportError(
            ReasonCode.TYPE_MISMATCH,
            "renderer seed 必须是 >= 0 的整数",
        )
    if record.get("output_modality") != OutputModality.TEXT.value:
        raise ExportError(
            ReasonCode.UNSUPPORTED_TASK,
            f"{record['record_id']}: description_multitask 只支持 text 输出",
        )
    try:
        layout = InputLayout(record["input_layout"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExportError(
            ReasonCode.INVALID_ENUM,
            f"{record.get('record_id')}: input_layout 非法",
        ) from error
    derived: list[str] = []
    if layout is InputLayout.SINGLE_IMAGE:
        user_content = _render_single_image(record)
    elif layout is InputLayout.BBOX_REGION:
        user_content, derived = _render_bbox_region(
            record,
            dataset=dataset,
            staging=derived_root,
        )
    elif layout is InputLayout.BOXED_IMAGE:
        user_content = _render_boxed_image(record)
    elif layout is InputLayout.PRE_POST:
        user_content = _render_pre_post(record)
    else:
        raise ExportError(
            ReasonCode.UNSUPPORTED_TASK,
            f"{record['record_id']}: renderer 不支持 {layout.value}",
        )
    row: dict[str, Any] = {
        "record_id": record["record_id"],
        "parent_id": record["parent_id"],
        "logical_role": record["logical_role"],
        "task_family": record["task_family"],
        "supervision_kind": record["supervision_kind"],
        "input_layout": record["input_layout"],
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": _response(
                            record,
                            purpose=purpose,
                            seed=seed,
                        ),
                    }
                ],
            },
        ],
    }
    if purpose != "training":
        row["reference_responses"] = list(record["reference_responses"])
    return row, derived


def _validate_export_scope(config: ExportConfig) -> None:
    if config.profile != DESCRIPTION_MULTITASK_PROFILE:
        raise ExportError(
            ReasonCode.INVALID_ENUM,
            f"仅支持 profile={DESCRIPTION_MULTITASK_PROFILE}",
        )
    if not config.task_families:
        raise ExportError(
            ReasonCode.SCHEMA_MISMATCH,
            "task_families 必须显式列出，禁止空列表隐式导出全部",
        )
    roles = set(config.roles)
    if config.purpose == "training" and roles != {"external_train"}:
        raise ExportError(
            ReasonCode.ROLE_CONTAMINATION,
            "training export 的 roles 必须且只能是 external_train",
        )
    if config.purpose == "validation" and roles != {"external_val"}:
        raise ExportError(
            ReasonCode.ROLE_CONTAMINATION,
            "validation export 的 roles 必须且只能是 external_val",
        )


def export_qwen(config: ExportConfig) -> Path:
    _validate_export_scope(config)
    if config.output_root.exists() or config.output_root.is_symlink():
        raise ExportError(
            ReasonCode.OUTPUT_EXISTS,
            f"拒绝覆盖已有 export output_root：{config.output_root}",
        )
    if not config.benchmark_root.is_dir():
        raise ExportError(
            ReasonCode.ASSET_MISSING,
            f"canonical benchmark 不存在：{config.benchmark_root}",
        )
    from .validator import validate_benchmark

    validation = validate_benchmark(config.benchmark_root, deep=False)
    if validation["errors"]:
        raise ExportError(
            ReasonCode.HASH_MISMATCH,
            "canonical benchmark 浅验证失败，拒绝导出",
            details={"errors": validation["errors"][:20]},
        )
    try:
        config.output_root.resolve().relative_to(config.benchmark_root.resolve())
    except ValueError:
        pass
    else:
        raise ExportError(
            ReasonCode.PATH_ESCAPE,
            "export output_root 不能位于 canonical benchmark 内",
        )
    manifest = read_json(config.benchmark_root / "manifest.json")
    dataset = RSGeneralDescDataset(
        config.benchmark_root,
        roles=config.roles,
        task_families=config.task_families,
        load_assets=False,
        seed=config.seed,
    )
    if len(dataset) == 0:
        raise ExportError(ReasonCode.SCHEMA_MISMATCH, "export 过滤后无记录")
    target = config.output_root
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    (staging / EXPORT_STAGING_SENTINEL).write_text(
        "owned-by-rs-generaldesc-exporter\n", encoding="utf-8"
    )
    try:
        rows: list[dict[str, Any]] = []
        derived_assets: list[str] = []
        for record in dataset.records:
            row, derived = render_canonical_messages(
                record,
                purpose=config.purpose,
                seed=config.seed,
                dataset=dataset,
                derived_root=staging,
            )
            rows.append(row)
            derived_assets.extend(derived)
        rows.sort(
            key=lambda row: (
                row["logical_role"],
                row["task_family"],
                row["parent_id"],
                row["record_id"],
            )
        )
        data_path = staging / "qwen_messages.jsonl"
        atomic_write_jsonl(data_path, rows)
        payload_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file()
            and path.name != EXPORT_STAGING_SENTINEL
            and path.name != "manifest.json"
        )
        files = {
            path.relative_to(staging).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in payload_paths
        }
        export_manifest = {
            "schema_version": QWEN_EXPORT_MANIFEST_VERSION,
            "profile": config.profile,
            "template_version": QWEN_TEMPLATE_VERSION,
            "purpose": config.purpose,
            "seed": config.seed,
            "roles": list(config.roles),
            "task_families": list(config.task_families),
            "record_count": len(rows),
            "parent_count": len({row["parent_id"] for row in rows}),
            "role_counts": dict(
                sorted(Counter(row["logical_role"] for row in rows).items())
            ),
            "task_counts": dict(
                sorted(Counter(row["task_family"] for row in rows).items())
            ),
            "supervision_kind_counts": dict(
                sorted(
                    Counter(row["supervision_kind"] for row in rows).items()
                )
            ),
            "input_layout_counts": dict(
                sorted(Counter(row["input_layout"] for row in rows).items())
            ),
            "canonical_payload_root_sha256": manifest["payload_root_sha256"],
            "canonical_assets_copied": False,
            "canonical_asset_reference_type": "benchmark_relative_path",
            "derived_assets_created": bool(derived_assets),
            "derived_asset_count": len(set(derived_assets)),
            "derived_asset_reference_type": "export_relative_path",
            "files": files,
            "content_sha256": sha256_bytes(
                canonical_json(files).encode("utf-8")
            ),
        }
        atomic_write_json(staging / "manifest.json", export_manifest)
        if target.exists() or target.is_symlink():
            raise ExportError(
                ReasonCode.OUTPUT_EXISTS,
                f"发布前发现 export output_root 已存在：{target}",
            )
        (staging / EXPORT_STAGING_SENTINEL).unlink()
        staging.replace(target)
    except BaseException:
        if staging.exists() and (staging / EXPORT_STAGING_SENTINEL).is_file():
            shutil.rmtree(staging)
        raise
    return target
