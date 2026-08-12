"""RSGPT RSICap/RSIEval adapter。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..common import read_json, safe_join, sha256_file
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
)


QA_TASK_CONTRACTS = {
    "presence": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
    "quantity": (TaskFamily.OBJECT_COUNT, SupervisionKind.NUMERIC_QA),
    "ab_position": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
    "re_position": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
    "color": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
    "image": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
    "area_comp": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
    "scene": (TaskFamily.SCENE_UNDERSTANDING, SupervisionKind.SHORT_QA),
    "reasoning": (TaskFamily.VISUAL_QA, SupervisionKind.SHORT_QA),
}


def _annotations(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"annotations"}
        or not isinstance(value["annotations"], list)
        or not all(isinstance(row, dict) for row in value["annotations"])
    ):
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{path}: 预期仅含 annotations 数组",
        )
    return value["annotations"]


class RSGPTAdapter(SourceAdapter):
    source_name = "rsgpt"

    def scan(
        self,
        *,
        deep: bool = False,
        for_build: bool = True,
    ) -> AdapterResult:
        root = self.config.sources["rsgpt"].root
        result = AdapterResult(source=self.source_name)
        cap_metadata = root / "RSICap/captions.json"
        eval_metadata = root / "RSIEval/annotations.json"
        cap_images = root / "RSICap/images"
        eval_images = root / "RSIEval/images"
        captions = _annotations(cap_metadata)
        evaluations = _annotations(eval_metadata)
        cap_metadata_hash = sha256_file(cap_metadata)
        eval_metadata_hash = sha256_file(eval_metadata)
        referenced_cap: set[str] = set()
        referenced_eval: set[str] = set()

        for index, row in enumerate(captions):
            required = {"image_id", "filename", "text_input", "text_output"}
            if set(row) != required:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"RSICap annotations[{index}] 字段必须为 {sorted(required)}",
                )
            filename = str(row["filename"])
            referenced_cap.add(filename)
            source_id = f"rsicap:{row['image_id']}"
            image_path = safe_join(
                cap_images, filename, location=f"RSICap[{index}].filename"
            )
            if not image_path.is_file():
                result.add_skip(
                    source_record_id=source_id,
                    source_split="train",
                    task_family=TaskFamily.GLOBAL_CAPTION.value,
                    reason_code=ReasonCode.ASSET_MISSING,
                    evidence={"asset": f"RSICap/images/{filename}"},
                )
                continue
            instruction, bad_instruction = safe_text(row["text_input"], self.config)
            response, bad_response = safe_text(row["text_output"], self.config)
            bad = bad_instruction or bad_response
            if bad:
                result.add_skip(
                    source_record_id=source_id,
                    source_split="train",
                    task_family=TaskFamily.GLOBAL_CAPTION.value,
                    reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                    evidence={"matched_policy": bad},
                )
                continue
            result.examples.append(
                SourceExample(
                    source=self.source_name,
                    source_record_id=source_id,
                    source_split="train",
                    parent_key=f"RSICap/{filename}",
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
                            source_ref=f"RSICap/images/{filename}",
                            source_path=image_path,
                        ),
                    ),
                    target={"type": "none"},
                    instruction=instruction,
                    training_responses=(response,),
                    reference_responses=(response,),
                    deterministic_facts={"source_collection": "RSICap"},
                    annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                    review_status=ReviewStatus.NOT_REQUIRED,
                    provenance=(
                        {
                            "source_item_id": source_id,
                            "metadata": "RSICap/captions.json",
                            "annotation_index": index,
                            "metadata_sha256": cap_metadata_hash,
                        },
                    ),
                )
            )

        seen_qa: set[tuple[str, str, str, str]] = set()
        qa_type_counts: Counter[str] = Counter()
        for index, row in enumerate(evaluations):
            required = {"filename", "caption", "qa_pairs"}
            if set(row) != required or not isinstance(row["qa_pairs"], list):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"RSIEval annotations[{index}] 合同非法",
                )
            filename = str(row["filename"])
            referenced_eval.add(filename)
            image_path = safe_join(
                eval_images, filename, location=f"RSIEval[{index}].filename"
            )
            caption_id = f"rsieval:{filename}:caption"
            if image_path.is_file():
                instruction = "Describe the visible content of this remote sensing image."
                response, bad = safe_text(row["caption"], self.config)
                if bad:
                    result.add_skip(
                        source_record_id=caption_id,
                        source_split="eval",
                        task_family=TaskFamily.GLOBAL_CAPTION.value,
                        reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                        evidence={"matched_policy": bad},
                    )
                else:
                    result.examples.append(
                        SourceExample(
                            source=self.source_name,
                            source_record_id=caption_id,
                            source_split="eval",
                            parent_key=f"RSIEval/{filename}",
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
                                    source_ref=f"RSIEval/images/{filename}",
                                    source_path=image_path,
                                ),
                            ),
                            target={"type": "none"},
                            instruction=instruction,
                            training_responses=(response,),
                            reference_responses=(response,),
                            deterministic_facts={"source_collection": "RSIEval"},
                            annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                            review_status=ReviewStatus.NOT_REQUIRED,
                            provenance=(
                                {
                                    "source_item_id": caption_id,
                                    "metadata": "RSIEval/annotations.json",
                                    "annotation_index": index,
                                    "metadata_sha256": eval_metadata_hash,
                                },
                            ),
                        )
                    )
            else:
                result.add_skip(
                    source_record_id=caption_id,
                    source_split="eval",
                    task_family=TaskFamily.GLOBAL_CAPTION.value,
                    reason_code=ReasonCode.ASSET_MISSING,
                    evidence={"asset": f"RSIEval/images/{filename}"},
                )
            for qa_index, qa in enumerate(row["qa_pairs"]):
                if not isinstance(qa, dict) or set(qa) != {
                    "question",
                    "answer",
                    "type",
                }:
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"RSIEval[{index}].qa_pairs[{qa_index}] 合同非法",
                    )
                qa_type = str(qa["type"])
                qa_type_counts[qa_type] += 1
                source_id = f"rsieval:{filename}:qa:{qa_index:03d}"
                if qa_type not in QA_TASK_CONTRACTS:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="eval",
                        task_family=TaskFamily.VISUAL_QA.value,
                        reason_code=ReasonCode.UNSUPPORTED_TASK,
                        evidence={"qa_type": qa_type},
                    )
                    continue
                family, supervision = QA_TASK_CONTRACTS[qa_type]
                question, bad_question = safe_text(
                    qa["question"],
                    self.config,
                    reasoning=qa_type == "reasoning",
                )
                answer, bad_answer = safe_text(
                    qa["answer"],
                    self.config,
                    reasoning=qa_type == "reasoning",
                )
                bad = bad_question or bad_answer
                if bad:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="eval",
                        task_family=family.value,
                        reason_code=ReasonCode.UNSUPPORTED_CLAIM,
                        evidence={"qa_type": qa_type, "matched_policy": bad},
                    )
                    continue
                key = (filename, question, answer, qa_type)
                if key in seen_qa:
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="eval",
                        task_family=family.value,
                        reason_code=ReasonCode.DUPLICATE_RECORD,
                        evidence={"qa_type": qa_type},
                    )
                    continue
                seen_qa.add(key)
                if not image_path.is_file():
                    result.add_skip(
                        source_record_id=source_id,
                        source_split="eval",
                        task_family=family.value,
                        reason_code=ReasonCode.ASSET_MISSING,
                        evidence={"asset": f"RSIEval/images/{filename}"},
                    )
                    continue
                result.examples.append(
                    SourceExample(
                        source=self.source_name,
                        source_record_id=source_id,
                        source_split="eval",
                        parent_key=f"RSIEval/{filename}",
                        logical_role=LogicalRole.EXTERNAL_TRAIN,
                        task_family=family,
                        supervision_kind=supervision,
                        input_layout=InputLayout.SINGLE_IMAGE,
                        output_modality=OutputModality.TEXT,
                        assets=(
                            PendingAsset(
                                role="image",
                                media_type=MediaType.IMAGE,
                                extension=image_path.suffix.lower().lstrip("."),
                                source_ref=f"RSIEval/images/{filename}",
                                source_path=image_path,
                            ),
                        ),
                        target={"type": "none"},
                        instruction=question,
                        training_responses=(answer,),
                        reference_responses=(answer,),
                        deterministic_facts={"qa_type": qa_type},
                        annotation_layer=AnnotationLayer.EXTERNAL_SOURCE,
                        review_status=ReviewStatus.NOT_REQUIRED,
                        provenance=(
                            {
                                "source_item_id": source_id,
                                "metadata": "RSIEval/annotations.json",
                                "annotation_index": index,
                                "qa_index": qa_index,
                                "metadata_sha256": eval_metadata_hash,
                            },
                        ),
                    )
                )

        all_cap_images = {
            path.name
            for path in cap_images.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        all_eval_images = {
            path.name
            for path in eval_images.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        unused_assets = [
            ("RSICap", cap_images, filename)
            for filename in sorted(all_cap_images - referenced_cap)
        ] + [
            ("RSIEval", eval_images, filename)
            for filename in sorted(all_eval_images - referenced_eval)
        ]
        for collection, image_root, filename in unused_assets:
            path = image_root / filename
            result.add_skip(
                source_record_id=f"unused:{collection}:{filename}",
                source_split="source_inventory",
                task_family="unreferenced_asset",
                reason_code=ReasonCode.UNUSED_ASSET,
                evidence={
                    "asset": f"{collection}/images/{filename}",
                    "size_bytes": path.stat().st_size,
                    "reason": "not_referenced_by_source_metadata",
                },
            )
        system_files = sorted(
            {
                path
                for pattern in (".DS_Store", "Thumbs.db")
                for path in root.rglob(pattern)
                if path.is_file()
            },
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in system_files:
            relative = path.relative_to(root).as_posix()
            result.add_skip(
                source_record_id=f"system_file:{relative}",
                source_split="source_inventory",
                task_family="system_file",
                reason_code=ReasonCode.UNUSED_ASSET,
                evidence={
                    "asset": relative,
                    "size_bytes": path.stat().st_size,
                    "reason": "system_metadata_file",
                },
            )
        result.audit.update(
            {
                "rsicap_annotation_count": len(captions),
                "rsieval_parent_count": len(evaluations),
                "rsieval_qa_count": sum(qa_type_counts.values()),
                "rsieval_qa_type_counts": dict(sorted(qa_type_counts.items())),
                "unreferenced_image_count": len(unused_assets),
                "system_file_count": len(system_files),
            }
        )
        summarize_result(result)
        result.audit["full_candidate_examples"] = result.audit["candidate_examples"]
        result.examples = select_examples(result.examples, config=self.config)
        result.audit["selected_examples"] = len(result.examples)
        return result
