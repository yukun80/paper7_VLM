"""MMRS-1M caption/VQA/RSVG adapter。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..common import read_json, safe_join, sha256_file, sha256_text
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
    deduplicate_responses,
    safe_text,
    select_examples,
    summarize_result,
)


_BOX_PATTERN = re.compile(r":\s*(\[[^\]]+\])\s*$")
_ANY_BOX_PATTERN = re.compile(r"\[[^\]]+\]")


def _task_rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{path}: 预期对象数组",
        )
    return value


def _normalize_image_ref(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, f"{location}: image 必须是字符串")
    normalized = value.replace("\\", "/")
    if normalized.startswith("data/"):
        normalized = normalized[5:]
    return normalized


def _pairs(conversations: Any, *, location: str) -> list[tuple[str, str, int]]:
    if not isinstance(conversations, list) or not conversations or len(conversations) % 2:
        raise SchemaError(
            ReasonCode.INVALID_CONVERSATION,
            f"{location}: conversations 必须是非空偶数 turn",
        )
    pairs: list[tuple[str, str, int]] = []
    for index in range(0, len(conversations), 2):
        human = conversations[index]
        assistant = conversations[index + 1]
        if (
            not isinstance(human, dict)
            or set(human) != {"from", "value"}
            or human["from"] != "human"
            or not isinstance(assistant, dict)
            or set(assistant) != {"from", "value"}
            or assistant["from"] != "gpt"
        ):
            raise SchemaError(
                ReasonCode.INVALID_CONVERSATION,
                f"{location}: turn {index}/{index + 1} 角色非法",
            )
        pairs.append((str(human["value"]), str(assistant["value"]), index // 2))
    return pairs


class MMRS1MAdapter(SourceAdapter):
    source_name = "mmrs1m"

    def scan(
        self,
        *,
        deep: bool = False,
        for_build: bool = True,
    ) -> AdapterResult:
        root = self.config.sources["mmrs1m"].root
        result = AdapterResult(source=self.source_name)
        metadata_root = root / "json"
        caption_files = sorted((metadata_root / "caption").glob("*.json"))
        vqa_files = sorted((metadata_root / "VQA").glob("*.json"))
        rsvg_path = metadata_root / "RSVG/rsvg_trainval.json"
        excluded_metadata = sorted(
            [
                path
                for directory in ("classification", "detection")
                for path in (metadata_root / directory).glob("*.json")
            ]
            + (
                [metadata_root / "total.json"]
                if (metadata_root / "total.json").is_file()
                else []
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in excluded_metadata:
            relative = path.relative_to(root).as_posix()
            result.add_skip(
                source_record_id=f"excluded_metadata:{relative}",
                source_split="source_inventory",
                task_family="excluded_metadata",
                reason_code=ReasonCode.UNSUPPORTED_TASK,
                evidence={
                    "metadata": relative,
                    "size_bytes": path.stat().st_size,
                    "reason": "outside_phase2_adoption_matrix",
                },
            )
        caption_parents = 0
        caption_pairs = 0
        vqa_parents = 0
        vqa_pairs = 0
        rsvg_pairs = 0
        seen_vqa: set[tuple[str, str, str]] = set()

        for metadata in caption_files:
            rows = _task_rows(metadata)
            metadata_ref = metadata.relative_to(root).as_posix()
            metadata_hash = sha256_file(metadata)
            for row_index, row in enumerate(rows):
                if set(row) != {"image", "conversations"}:
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"{metadata}[{row_index}] 字段非法",
                    )
                caption_parents += 1
                image_ref = _normalize_image_ref(
                    row["image"], location=f"{metadata}[{row_index}].image"
                )
                image_path = safe_join(root, image_ref, location="MMRS image")
                try:
                    pairs = _pairs(
                        row["conversations"],
                        location=f"{metadata}[{row_index}].conversations",
                    )
                except SchemaError:
                    result.add_skip(
                        source_record_id=f"{metadata.stem}:{row_index}",
                        source_split="source_mixed",
                        task_family=TaskFamily.GLOBAL_CAPTION.value,
                        reason_code=ReasonCode.INVALID_CONVERSATION,
                        evidence={"metadata": metadata_ref},
                    )
                    continue
                caption_pairs += len(pairs)
                grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
                for question_raw, answer_raw, pair_index in pairs:
                    question, bad_question = safe_text(question_raw, self.config)
                    answer, bad_answer = safe_text(answer_raw, self.config)
                    if bad_question or bad_answer:
                        result.add_skip(
                            source_record_id=f"{metadata.stem}:{row_index}:{pair_index}",
                            source_split="source_mixed",
                            task_family=TaskFamily.GLOBAL_CAPTION.value,
                            reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                            evidence={
                                "matched_policy": bad_question or bad_answer,
                                "metadata": metadata_ref,
                            },
                        )
                        continue
                    grouped[question].append((answer, pair_index))
                for question, answer_rows in sorted(grouped.items()):
                    responses, duplicates = deduplicate_responses(
                        answer for answer, _ in answer_rows
                    )
                    for duplicate_index in range(duplicates):
                        result.add_skip(
                            source_record_id=(
                                f"{metadata.stem}:{row_index}:duplicate:{duplicate_index}"
                            ),
                            source_split="source_mixed",
                            task_family=TaskFamily.GLOBAL_CAPTION.value,
                            reason_code=ReasonCode.DUPLICATE_REFERENCE,
                            evidence={"metadata": metadata_ref},
                        )
                    source_id = (
                        f"caption:{metadata.stem}:{row_index}:"
                        f"{sha256_text(question)[:16]}"
                    )
                    if not responses:
                        continue
                    if not image_path.is_file():
                        result.add_skip(
                            source_record_id=source_id,
                            source_split="source_mixed",
                            task_family=TaskFamily.GLOBAL_CAPTION.value,
                            reason_code=ReasonCode.ASSET_MISSING,
                            evidence={"asset": image_ref, "metadata": metadata_ref},
                        )
                        continue
                    result.examples.append(
                        SourceExample(
                            source=self.source_name,
                            source_record_id=source_id,
                            source_split="source_mixed",
                            parent_key=image_ref,
                            logical_role=LogicalRole.EXTERNAL_TRAIN,
                            task_family=TaskFamily.GLOBAL_CAPTION,
                            supervision_kind=SupervisionKind.LONG_DESCRIPTION,
                            input_layout=InputLayout.SINGLE_IMAGE,
                            output_modality=OutputModality.TEXT,
                            assets=(
                                PendingAsset(
                                    role="image",
                                    media_type=MediaType.IMAGE,
                                    extension=image_path.suffix.lower().lstrip("."),
                                    source_ref=image_ref,
                                    source_path=image_path,
                                ),
                            ),
                            target={"type": "none"},
                            instruction=question,
                            training_responses=responses,
                            reference_responses=responses,
                            deterministic_facts={},
                            annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                            review_status=ReviewStatus.NOT_REQUIRED,
                            provenance=tuple(
                                {
                                    "source_item_id": (
                                        f"{metadata.stem}:{row_index}:{pair_index}"
                                    ),
                                    "metadata": metadata_ref,
                                    "row_index": row_index,
                                    "pair_index": pair_index,
                                    "metadata_sha256": metadata_hash,
                                }
                                for _, pair_index in answer_rows
                            ),
                        )
                    )

        for metadata in vqa_files:
            rows = _task_rows(metadata)
            metadata_ref = metadata.relative_to(root).as_posix()
            metadata_hash = sha256_file(metadata)
            for row_index, row in enumerate(rows):
                if set(row) != {"image", "conversations"}:
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"{metadata}[{row_index}] 字段非法",
                    )
                vqa_parents += 1
                image_ref = _normalize_image_ref(
                    row["image"], location=f"{metadata}[{row_index}].image"
                )
                image_path = safe_join(root, image_ref, location="MMRS image")
                try:
                    pairs = _pairs(
                        row["conversations"],
                        location=f"{metadata}[{row_index}].conversations",
                    )
                except SchemaError:
                    result.add_skip(
                        source_record_id=f"vqa:{metadata.stem}:{row_index}",
                        source_split="source_mixed",
                        task_family=TaskFamily.VISUAL_QA.value,
                        reason_code=ReasonCode.INVALID_CONVERSATION,
                        evidence={"metadata": metadata_ref},
                    )
                    continue
                vqa_pairs += len(pairs)
                for question_raw, answer_raw, pair_index in pairs:
                    source_id = f"vqa:{metadata.stem}:{row_index}:{pair_index}"
                    question, bad_question = safe_text(question_raw, self.config)
                    answer, bad_answer = safe_text(answer_raw, self.config)
                    if bad_question or bad_answer:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split="source_mixed",
                            task_family=TaskFamily.VISUAL_QA.value,
                            reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                            evidence={"matched_policy": bad_question or bad_answer},
                        )
                        continue
                    duplicate_key = (image_ref, question, answer)
                    if duplicate_key in seen_vqa:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split="source_mixed",
                            task_family=TaskFamily.VISUAL_QA.value,
                            reason_code=ReasonCode.DUPLICATE_RECORD,
                            evidence={"metadata": metadata_ref},
                        )
                        continue
                    seen_vqa.add(duplicate_key)
                    if not image_path.is_file():
                        result.add_skip(
                            source_record_id=source_id,
                            source_split="source_mixed",
                            task_family=TaskFamily.VISUAL_QA.value,
                            reason_code=ReasonCode.ASSET_MISSING,
                            evidence={"asset": image_ref},
                        )
                        continue
                    result.examples.append(
                        SourceExample(
                            source=self.source_name,
                            source_record_id=source_id,
                            source_split="source_mixed",
                            parent_key=image_ref,
                            logical_role=LogicalRole.EXTERNAL_TRAIN,
                            task_family=TaskFamily.VISUAL_QA,
                            supervision_kind=SupervisionKind.SHORT_QA,
                            input_layout=InputLayout.SINGLE_IMAGE,
                            output_modality=OutputModality.TEXT,
                            assets=(
                                PendingAsset(
                                    role="image",
                                    media_type=MediaType.IMAGE,
                                    extension=image_path.suffix.lower().lstrip("."),
                                    source_ref=image_ref,
                                    source_path=image_path,
                                ),
                            ),
                            target={"type": "none"},
                            instruction=question,
                            training_responses=(answer,),
                            reference_responses=(answer,),
                            deterministic_facts={"task_dataset": metadata.stem},
                            annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                            review_status=ReviewStatus.NOT_REQUIRED,
                            provenance=(
                                {
                                    "source_item_id": source_id,
                                    "metadata": metadata_ref,
                                    "row_index": row_index,
                                    "pair_index": pair_index,
                                    "metadata_sha256": metadata_hash,
                                },
                            ),
                        )
                    )

        rsvg_rows = _task_rows(rsvg_path)
        rsvg_hash = sha256_file(rsvg_path)
        rsvg_seen: set[tuple[str, tuple[float, ...], str]] = set()
        for row_index, row in enumerate(rsvg_rows):
            if set(row) != {"image", "conversations"}:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"{rsvg_path}[{row_index}] 字段非法",
                )
            image_ref = _normalize_image_ref(
                row["image"], location=f"RSVG[{row_index}].image"
            )
            image_path = safe_join(root, image_ref, location="MMRS RSVG image")
            try:
                pairs = _pairs(
                    row["conversations"],
                    location=f"RSVG[{row_index}].conversations",
                )
            except SchemaError:
                result.add_skip(
                    source_record_id=f"rsvg:{row_index}",
                    source_split="trainval",
                    task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                    reason_code=ReasonCode.INVALID_CONVERSATION,
                )
                continue
            rsvg_pairs += len(pairs)
            for question_raw, answer_raw, pair_index in pairs:
                source_id = f"rsvg:{row_index}:{pair_index}"
                match = _BOX_PATTERN.search(question_raw)
                if match is None:
                    if _ANY_BOX_PATTERN.search(answer_raw):
                        result.add_skip(
                            source_record_id=source_id,
                            source_split="trainval",
                            task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                            reason_code=ReasonCode.REVERSE_GROUNDING_DIRECTION,
                        )
                    else:
                        result.add_skip(
                            source_record_id=source_id,
                            source_split="trainval",
                            task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                            reason_code=ReasonCode.INVALID_CONVERSATION,
                        )
                    continue
                try:
                    raw_box = json.loads(match.group(1))
                except json.JSONDecodeError:
                    raw_box = None
                if (
                    not isinstance(raw_box, list)
                    or len(raw_box) != 4
                    or any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        for value in raw_box
                    )
                ):
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="trainval",
                        task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                        reason_code=ReasonCode.BBOX_INVALID,
                    )
                    continue
                box = tuple(float(value) for value in raw_box)
                if (
                    min(box) < 0
                    or max(box) > 1
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="trainval",
                        task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                        reason_code=(
                            ReasonCode.BBOX_ZERO_AREA
                            if box[2] <= box[0] or box[3] <= box[1]
                            else ReasonCode.BBOX_INVALID
                        ),
                        evidence={"bbox": list(box)},
                    )
                    continue
                question, bad_question = safe_text(question_raw, self.config)
                answer, bad_answer = safe_text(answer_raw, self.config)
                if bad_question or bad_answer:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="trainval",
                        task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                        reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                        evidence={"matched_policy": bad_question or bad_answer},
                    )
                    continue
                key = (image_ref, box, answer)
                if key in rsvg_seen:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="trainval",
                        task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                        reason_code=ReasonCode.DUPLICATE_RECORD,
                    )
                    continue
                rsvg_seen.add(key)
                if not image_path.is_file():
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="trainval",
                        task_family=TaskFamily.BBOX_REGION_CAPTION.value,
                        reason_code=ReasonCode.ASSET_MISSING,
                        evidence={"asset": image_ref},
                    )
                    continue
                result.examples.append(
                    SourceExample(
                        source=self.source_name,
                        source_record_id=source_id,
                        source_split="trainval",
                        parent_key=image_ref,
                        logical_role=LogicalRole.EXTERNAL_TRAIN,
                        task_family=TaskFamily.BBOX_REGION_CAPTION,
                        supervision_kind=SupervisionKind.REGION_DESCRIPTION,
                        input_layout=InputLayout.BBOX_REGION,
                        output_modality=OutputModality.TEXT,
                        assets=(
                            PendingAsset(
                                role="image",
                                media_type=MediaType.IMAGE,
                                extension=image_path.suffix.lower().lstrip("."),
                                source_ref=image_ref,
                                source_path=image_path,
                            ),
                        ),
                        target={
                            "type": "bbox",
                            "image_role": "image",
                            "source_convention": "xyxy_normalized_top_left",
                            "boxes": [{"source": list(box), "label": answer}],
                        },
                        instruction=question,
                        training_responses=(answer,),
                        reference_responses=(answer,),
                        deterministic_facts={},
                        annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                        review_status=ReviewStatus.NOT_REQUIRED,
                        provenance=(
                            {
                                "source_item_id": source_id,
                                "metadata": "json/RSVG/rsvg_trainval.json",
                                "row_index": row_index,
                                "pair_index": pair_index,
                                "metadata_sha256": rsvg_hash,
                            },
                        ),
                    )
                )

        result.audit.update(
            {
                "caption_metadata_files": len(caption_files),
                "caption_parent_count": caption_parents,
                "caption_pair_count": caption_pairs,
                "vqa_metadata_files": len(vqa_files),
                "vqa_parent_count": vqa_parents,
                "vqa_pair_count": vqa_pairs,
                "rsvg_row_count": len(rsvg_rows),
                "rsvg_pair_count": rsvg_pairs,
                "excluded_total_json": (metadata_root / "total.json").is_file(),
                "excluded_classification_metadata_files": sum(
                    path.parent.name == "classification"
                    for path in excluded_metadata
                ),
                "excluded_detection_metadata_files": sum(
                    path.parent.name == "detection"
                    for path in excluded_metadata
                ),
            }
        )
        summarize_result(result)
        result.audit["full_candidate_examples"] = result.audit["candidate_examples"]
        result.examples = select_examples(result.examples, config=self.config)
        result.audit["selected_examples"] = len(result.examples)
        return result
