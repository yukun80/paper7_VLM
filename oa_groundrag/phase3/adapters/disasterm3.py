"""DisasterM3 光学任务 adapter。"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..common import (
    normalized_multiline_text,
    read_json,
    safe_join,
    sha256_file,
)
from ..contracts import (
    AdapterResult,
    AnnotationLayer,
    InputLayout,
    LogicalRole,
    MediaType,
    OutputModality,
    PendingAsset,
    ReviewStatus,
    SourceExample,
    SupervisionKind,
    TaskFamily,
)
from ..errors import ReasonCode, SchemaError
from .base import (
    SourceAdapter,
    safe_text,
    select_examples,
    summarize_result,
    text_has_pattern,
)


TASK_MAPPING = {
    "Disaster Bearing Bodies Recognition": TaskFamily.SCENE_UNDERSTANDING,
    "Building Damage Counting": TaskFamily.OBJECT_COUNT,
    "Road Damage Counting": TaskFamily.OBJECT_COUNT,
    "disaster caption": TaskFamily.VISIBLE_CHANGE_REPORT,
    "Disaster Report": TaskFamily.VISIBLE_CHANGE_REPORT,
    "Disaster Scene Recognition": TaskFamily.SCENE_UNDERSTANDING,
    "relational reasoning": TaskFamily.SPATIAL_RELATION,
    "Relational Reasoning": TaskFamily.SPATIAL_RELATION,
}

EXCLUDED_TASKS = {
    "Disaster Type Recognition",
    "disaster restoration advice",
    "Disaster Restoration Advice",
    "Referring Expression Segmentation",
}

TASK_SUPERVISION = {
    TaskFamily.SCENE_UNDERSTANDING: SupervisionKind.SHORT_QA,
    TaskFamily.OBJECT_COUNT: SupervisionKind.NUMERIC_QA,
    TaskFamily.SPATIAL_RELATION: SupervisionKind.SPATIAL_DESCRIPTION,
    TaskFamily.VISIBLE_CHANGE_REPORT: SupervisionKind.STRUCTURED_REPORT,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SchemaError(ReasonCode.SCHEMA_MISMATCH, f"{path}: 预期对象数组")
    return value


def _path_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, f"{location}: 路径必须是字符串")
    return value.replace("\\", "/")


def _response(row: dict[str, Any]) -> Any:
    return row.get("training_answer", row.get("ground_truth"))


def _prompt(row: dict[str, Any]) -> Any:
    value = row.get("prompts")
    if isinstance(value, list):
        if len(value) != 1:
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH, "DisasterM3 prompts 列表必须仅一项"
            )
        return value[0]
    return value


def _visible_report(value: Any, patterns: tuple[str, ...]) -> tuple[str, list[str]]:
    text = normalized_multiline_text(value, location="DisasterM3 report")
    kept: list[str] = []
    removed: list[str] = []
    for line in text.splitlines():
        section = line.split(":", 1)[0].strip().upper()
        if section in {"DISASTER", "CONCLUSION"}:
            removed.append(section)
            continue
        matched = text_has_pattern(line, patterns)
        if matched:
            removed.append(f"CLAIM:{matched}")
            continue
        kept.append(line)
    if not kept:
        return "", removed
    return "\n".join(kept), removed


def _is_optical(row: dict[str, Any]) -> bool:
    value = row.get("post_image_type", row.get("image_type"))
    return isinstance(value, str) and value.lower() == "optical"


def _resolve_image(root: Path, value: Any, *, relation: bool) -> tuple[str, Path, str | None]:
    ref = _path_text(value, location="DisasterM3 image path")
    rewrite: str | None = None
    if relation and ref.startswith("train_images/box_train_images/"):
        rewritten = ref[len("train_images/") :]
        rewrite = f"{ref}->{rewritten}"
        ref = rewritten
    return ref, safe_join(root, ref, location="DisasterM3 image"), rewrite


def _boxed_pair(
    image_path: Path,
    objects: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]] | None:
    """从已绘制 boxed image 的像素证据识别红框与蓝框对应的任意对象 key。"""

    if not isinstance(objects, dict) or len(objects) < 2:
        return None
    try:
        with Image.open(image_path) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    height, width = array.shape[:2]
    candidates: list[tuple[str, list[float], int, int]] = []
    for key, raw_box in sorted(objects.items(), key=lambda item: str(item[0])):
        if (
            not isinstance(raw_box, list)
            or len(raw_box) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_box
            )
        ):
            return None
        x, y, box_width, box_height = (float(value) for value in raw_box)
        if (
            x < 0
            or y < 0
            or box_width <= 0
            or box_height <= 0
            or x + box_width > width
            or y + box_height > height
        ):
            return None
        left = max(0, min(width - 1, int(math.floor(x))))
        top = max(0, min(height - 1, int(math.floor(y))))
        right = max(left + 1, min(width, int(math.ceil(x + box_width))))
        bottom = max(top + 1, min(height, int(math.ceil(y + box_height))))
        thickness = min(6, max(1, (right - left) // 4, (bottom - top) // 4))
        border = np.concatenate(
            (
                array[top : min(bottom, top + thickness), left:right].reshape(-1, 3),
                array[max(top, bottom - thickness) : bottom, left:right].reshape(-1, 3),
                array[top:bottom, left : min(right, left + thickness)].reshape(-1, 3),
                array[top:bottom, max(left, right - thickness) : right].reshape(-1, 3),
            ),
            axis=0,
        ).astype(np.int16, copy=False)
        red = int(
            np.count_nonzero(
                (border[:, 0] >= 180)
                & (border[:, 0] >= border[:, 1] + 60)
                & (border[:, 0] >= border[:, 2] + 60)
            )
        )
        blue = int(
            np.count_nonzero(
                (border[:, 2] >= 150)
                & (border[:, 2] >= border[:, 0] + 50)
                & (border[:, 2] >= border[:, 1] + 30)
            )
        )
        candidates.append((str(key), [x, y, box_width, box_height], red, blue))
    red_candidate = max(candidates, key=lambda item: (item[2], item[0]))
    blue_candidate = max(candidates, key=lambda item: (item[3], item[0]))
    if (
        red_candidate[0] == blue_candidate[0]
        or red_candidate[2] < 4
        or blue_candidate[3] < 4
    ):
        return None
    boxes = [
        {
            "source": candidate[1],
            "label": f"{visual_role}_object",
            "object_key": candidate[0],
        }
        for candidate, visual_role in (
            (red_candidate, "red_box"),
            (blue_candidate, "blue_box"),
        )
    ]
    roles = [
        {
            "object_key": candidate[0],
            "visual_role": visual_role,
        }
        for candidate, visual_role in (
            (red_candidate, "red_box"),
            (blue_candidate, "blue_box"),
        )
    ]
    return boxes, roles


class DisasterM3Adapter(SourceAdapter):
    source_name = "disasterm3"

    def scan(
        self,
        *,
        deep: bool = False,
        for_build: bool = True,
    ) -> AdapterResult:
        root = self.config.sources["disasterm3"].root
        result = AdapterResult(source=self.source_name)
        partitions = (
            (
                "train_release",
                root / "DisasterM3_Instruct",
                root / "DisasterM3_Instruct/train_release.json",
            ),
            (
                "benchmark_release",
                root / "DisasterM3_Bench",
                root / "DisasterM3_Bench/benchmark_release.json",
            ),
        )
        task_counts: Counter[str] = Counter()
        optical_candidate_counts: Counter[str] = Counter()
        path_rewrite_count = 0
        for split, asset_root, metadata in partitions:
            metadata_rows = _rows(metadata)
            metadata_hash = sha256_file(metadata)
            metadata_ref = metadata.relative_to(root).as_posix()
            for row_index, row in enumerate(metadata_rows):
                task = str(row.get("task", ""))
                task_counts[task] += 1
                source_id = f"{split}:{row_index}"
                if task in EXCLUDED_TASKS or task not in TASK_MAPPING:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split=split,
                        task_family=task or "unknown",
                        reason_code=ReasonCode.UNSUPPORTED_TASK,
                        evidence={
                            "task": task,
                            "reason": (
                                "pixel_mask_output_not_text_description"
                                if task == "Referring Expression Segmentation"
                                else "outside_description_multitask_scope"
                            ),
                        },
                    )
                    continue
                family = TASK_MAPPING[task]
                if not _is_optical(row):
                    result.add_skip(
                        source_record_id=source_id,
                        source_split=split,
                        task_family=family.value,
                        reason_code=ReasonCode.UNSUPPORTED_MODALITY,
                        evidence={
                            "image_type": row.get(
                                "post_image_type", row.get("image_type")
                            )
                        },
                    )
                    continue
                optical_candidate_counts[family.value] += 1
                prompt, bad_prompt = safe_text(_prompt(row), self.config)
                if bad_prompt:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split=split,
                        task_family=family.value,
                        reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                        evidence={"matched_policy": bad_prompt},
                    )
                    continue

                assets: list[PendingAsset] = []
                target: dict[str, Any] = {"type": "none"}
                facts: dict[str, Any] = {"source_task": task}
                quality_flags: list[str] = []
                provenance_extra: dict[str, Any] = {}
                if family is TaskFamily.SPATIAL_RELATION:
                    input_layout = InputLayout.BOXED_IMAGE
                    image_ref, image_path, rewrite = _resolve_image(
                        asset_root, row.get("image_path"), relation=True
                    )
                    if rewrite:
                        path_rewrite_count += 1
                        provenance_extra["path_rewrite"] = rewrite
                    if not image_path.is_file() or image_path.stat().st_size == 0:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split=split,
                            task_family=family.value,
                            reason_code=(
                                ReasonCode.ASSET_ZERO_BYTES
                                if image_path.is_file()
                                else ReasonCode.ASSET_MISSING
                            ),
                            evidence={"asset": image_ref},
                        )
                        continue
                    objects = row.get("objects")
                    boxed_pair = _boxed_pair(image_path, objects)
                    if boxed_pair is None:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split=split,
                            task_family=family.value,
                            reason_code=ReasonCode.BBOX_INVALID,
                            evidence={
                                "object_count": (
                                    len(objects)
                                    if isinstance(objects, dict)
                                    else None
                                ),
                                "reason": "red_blue_box_pixel_evidence_missing",
                            },
                        )
                        continue
                    boxes, boxed_roles = boxed_pair
                    target = {
                        "type": "bbox",
                        "image_role": "image",
                        "source_convention": "xywh_pixel_top_left",
                        "boxes": boxes,
                    }
                    facts["boxed_object_roles"] = boxed_roles
                    facts["source_object_candidate_count"] = len(objects)
                    assets.append(
                        PendingAsset(
                            role="image",
                            media_type=MediaType.IMAGE,
                            extension=image_path.suffix.lower().lstrip("."),
                            source_ref=image_ref,
                            source_path=image_path,
                        )
                    )
                    parent_key = f"image:{image_ref}"
                else:
                    pre_ref, pre_path, _ = _resolve_image(
                        asset_root, row.get("pre_image_path"), relation=False
                    )
                    post_ref, post_path, _ = _resolve_image(
                        asset_root, row.get("post_image_path"), relation=False
                    )
                    parent_key = f"pair:{pre_ref}|{post_ref}"
                    if family is TaskFamily.VISIBLE_CHANGE_REPORT:
                        input_layout = InputLayout.PRE_POST
                        selected_assets = (
                            ("pre_image", pre_ref, pre_path),
                            ("post_image", post_ref, post_path),
                        )
                        facts["image_semantic_roles"] = ["pre_disaster", "post_disaster"]
                    elif task == "Disaster Scene Recognition":
                        input_layout = InputLayout.SINGLE_IMAGE
                        selected_assets = (("image", pre_ref, pre_path),)
                        facts["image_semantic_role"] = "pre_disaster"
                    else:
                        input_layout = InputLayout.SINGLE_IMAGE
                        selected_assets = (("image", post_ref, post_path),)
                        facts["image_semantic_role"] = "post_disaster"
                    missing = [
                        ref for _, ref, path in selected_assets if not path.is_file()
                    ]
                    zero = [
                        ref
                        for _, ref, path in selected_assets
                        if path.is_file() and path.stat().st_size == 0
                    ]
                    if missing or zero:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split=split,
                            task_family=family.value,
                            reason_code=(
                                ReasonCode.ASSET_ZERO_BYTES
                                if zero
                                else ReasonCode.ASSET_MISSING
                            ),
                            evidence={"missing": missing, "zero_bytes": zero},
                        )
                        continue
                    for role, ref, path in selected_assets:
                        assets.append(
                            PendingAsset(
                                role=role,
                                media_type=MediaType.IMAGE,
                                extension=path.suffix.lower().lstrip("."),
                                source_ref=ref,
                                source_path=path,
                            )
                        )

                response_raw = _response(row)
                removed_sections: list[str] = []
                if response_raw is None or (
                    isinstance(response_raw, str) and not response_raw.strip()
                ):
                    result.add_skip(
                        source_record_id=source_id,
                        source_split=split,
                        task_family=family.value,
                        reason_code=ReasonCode.SCHEMA_MISMATCH,
                        evidence={"field": "training_answer/ground_truth"},
                    )
                    continue
                if family is TaskFamily.VISIBLE_CHANGE_REPORT:
                    response, removed_sections = _visible_report(
                        response_raw,
                        self.config.text_policy.forbidden_patterns,
                    )
                    if not response:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split=split,
                            task_family=family.value,
                            reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                            evidence={"removed_sections": removed_sections},
                        )
                        continue
                    quality_flags.append("visible_report_sections_filtered")
                else:
                    response, bad_response = safe_text(
                        str(response_raw), self.config
                    )
                    if bad_response:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split=split,
                            task_family=family.value,
                            reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                            evidence={"matched_policy": bad_response},
                        )
                        continue
                facts["removed_report_sections"] = removed_sections
                result.examples.append(
                    SourceExample(
                        source=self.source_name,
                        source_record_id=source_id,
                        source_split=split,
                        parent_key=parent_key,
                        logical_role=LogicalRole.EXTERNAL_TRAIN,
                        task_family=family,
                        supervision_kind=TASK_SUPERVISION[family],
                        input_layout=input_layout,
                        output_modality=OutputModality.TEXT,
                        assets=tuple(assets),
                        target=target,
                        instruction=prompt,
                        training_responses=(response,),
                        reference_responses=(response,),
                        deterministic_facts=facts,
                        annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                        review_status=ReviewStatus.NOT_REQUIRED,
                        provenance=(
                            {
                                "source_item_id": source_id,
                                "metadata": metadata_ref,
                                "row_index": row_index,
                                "metadata_sha256": metadata_hash,
                                **provenance_extra,
                            },
                        ),
                        quality_flags=tuple(quality_flags),
                    )
                )

        pre_deep_candidate_count = len(result.examples)

        result.audit.update(
            {
                "task_source_counts": dict(sorted(task_counts.items())),
                "optical_candidate_task_counts": dict(
                    sorted(optical_candidate_counts.items())
                ),
                "refseg_excluded_count": task_counts[
                    "Referring Expression Segmentation"
                ],
                "mask_archive_read": False,
                "path_rewrite_count": path_rewrite_count,
            }
        )
        summarize_result(result)
        zero_byte_skips = [
            skip
            for skip in result.skips
            if skip.reason_code is ReasonCode.ASSET_ZERO_BYTES
        ]
        result.audit["zero_byte_asset_count"] = len(
            {
                asset
                for skip in zero_byte_skips
                for asset in (
                    [
                        skip.evidence["asset"]
                    ]
                    if isinstance(skip.evidence.get("asset"), str)
                    else []
                )
                + list(skip.evidence.get("zero_bytes", []))
                if isinstance(asset, str)
            }
        )
        result.audit["zero_byte_reference_count"] = len(zero_byte_skips)
        result.audit["full_candidate_examples"] = pre_deep_candidate_count
        result.examples = select_examples(result.examples, config=self.config)
        result.audit["selected_examples"] = len(result.examples)
        return result
