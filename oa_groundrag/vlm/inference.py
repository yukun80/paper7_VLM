"""独立推理：逐条 prediction/failure JSONL 与严格 mask-mode 隔离。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from torch import Tensor

from .artifacts import AtomicArtifactDirectory
from .config import Phase4Config
from .contracts import (
    RUN_MANIFEST_SCHEMA_VERSION,
    EvidenceSufficiency,
    MaskMode,
)
from .data import DescriptionSample
from .errors import Phase4Error, PredictionError, ReasonCode
from .outputs import (
    failure_row,
    generic_prediction_row,
    parse_model_output,
    prediction_row,
)
from .processing import DescriptionCollator


class GenerativeAdapter(Protocol):
    def generate_text(
        self,
        batch: Mapping[str, Any],
        *,
        processor: Any,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        ...


def _single_mask_mode(
    samples: Sequence[DescriptionSample],
    *,
    expected: MaskMode,
) -> None:
    actual = {sample.mask_mode for sample in samples}
    if actual != {expected}:
        raise PredictionError(
            ReasonCode.MASK_MODE_MIXED,
            "inference Dataset mask_mode 与 run 不一致或发生混合",
            details={
                "expected": expected.value,
                "actual": sorted(value.value for value in actual),
            },
        )


def run_inference(
    *,
    config: Phase4Config,
    samples: Sequence[DescriptionSample],
    collator: DescriptionCollator,
    model: GenerativeAdapter,
    processor: Any,
    output_root: Path,
) -> Path:
    if not samples:
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "inference samples 不能为空",
        )
    _single_mask_mode(samples, expected=config.run.mask_mode)
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    token_total = 0
    image_total = 0
    for sample in samples:
        try:
            batch = collator([sample])
            token_total += int(batch["input_token_counts"][0])
            image_total += int(batch["image_counts"][0])
            if token_total > config.limits.max_total_tokens:
                raise PredictionError(
                    ReasonCode.TOKEN_LIMIT_EXCEEDED,
                    "inference total token cap exceeded",
                )
            outputs = model.generate_text(
                batch,
                processor=processor,
                max_new_tokens=config.generation.max_new_tokens,
                do_sample=config.generation.do_sample,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
            )
            if len(outputs) != 1:
                raise PredictionError(
                    ReasonCode.PREDICTION_INVALID,
                    "单样本 generate 必须返回一个文本",
                )
            generated = outputs[0]
            if sample.mask_mode is MaskMode.EXTERNAL_GENERIC:
                predictions.append(
                    generic_prediction_row(
                        record_id=sample.record_id,
                        parent_id=sample.parent_id,
                        logical_role=sample.logical_role,
                        task_family=sample.task_family,
                        generated_text=generated,
                        reference_responses=sample.reference_responses,
                        provenance=sample.provenance,
                    )
                )
            else:
                structured = parse_model_output(
                    generated,
                    valid_evidence_ids=sample.evidence_ids,
                )
                expected_status = sample.provenance.get(
                    "expected_target_status"
                )
                if (
                    expected_status is not None
                    and structured.target_status.value != expected_status
                ):
                    raise PredictionError(
                        ReasonCode.PREDICTION_INVALID,
                        "模型 target_status 与程序 evidence 不一致",
                    )
                known_limitations = sample.provenance.get(
                    "known_limitations",
                    [],
                )
                if (
                    not isinstance(known_limitations, list)
                    or not all(
                        isinstance(value, str)
                        for value in known_limitations
                    )
                ):
                    raise PredictionError(
                        ReasonCode.PREDICTION_INVALID,
                        "sample known_limitations provenance 非法",
                    )
                missing_limitations = sorted(
                    set(known_limitations) - set(structured.limitations)
                )
                if missing_limitations:
                    raise PredictionError(
                        ReasonCode.PREDICTION_INVALID,
                        "模型遗漏程序确定的 evidence limitations",
                        details={"missing": missing_limitations},
                    )
                if (
                    known_limitations
                    and structured.evidence_sufficiency
                    is EvidenceSufficiency.SUFFICIENT
                ):
                    raise PredictionError(
                        ReasonCode.PREDICTION_INVALID,
                        "存在已知 evidence limitation 时不得声明 sufficient",
                    )
                predictions.append(
                    prediction_row(
                        record_id=sample.record_id,
                        parent_id=sample.parent_id,
                        logical_role=sample.logical_role,
                        task_family=sample.task_family,
                        mask_mode=sample.mask_mode,
                        model_output=structured,
                        reference_responses=sample.reference_responses,
                        evidence_ids=sample.evidence_ids,
                        provenance=sample.provenance,
                        counterfactual=sample.counterfactual,
                    )
                )
        except Phase4Error as error:
            if error.code in {
                ReasonCode.TOKEN_LIMIT_EXCEEDED,
                ReasonCode.IMAGE_LIMIT_EXCEEDED,
            }:
                raise
            failures.append(
                failure_row(
                    record_id=sample.record_id,
                    parent_id=sample.parent_id,
                    stage="inference",
                    code=error.code,
                    message=str(error),
                    details=error.details,
                )
            )
    if not predictions:
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "所有 inference 样本失败，拒绝发布空 prediction",
            details={"failures": failures[:10]},
        )
    with AtomicArtifactDirectory(Path(output_root)) as writer:
        writer.write_jsonl("predictions.jsonl", predictions)
        writer.write_jsonl("failures.jsonl", failures)
        writer.write_json(
            "manifest.json",
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_kind": "inference",
                "config_semantic_sha256": config.semantic_sha256,
                "mask_mode": config.run.mask_mode.value,
                "prediction_count": len(predictions),
                "failure_count": len(failures),
                "task_counts": dict(
                    sorted(
                        Counter(
                            row["task_family"] for row in predictions
                        ).items()
                    )
                ),
                "input_token_count": token_total,
                "image_count": image_total,
                "formal_acceptance": False,
            },
        )
        return writer.publish()
