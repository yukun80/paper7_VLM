"""Stage 5 的 340 条 GT-mask 自动开发评价；不产生人工 reference 或科学 Gate。"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from torch import Tensor

from oa_groundrag.data.grounded.region import ledger_rows
from oa_groundrag.data.grounded.region_validation import validate_eval_dev
from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
)

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.grounding.contracts import MaskMode
from oa_groundrag.vlm.data import DescriptionSample
from oa_groundrag.vlm.errors import VLMError, PredictionError, ReasonCode
from .observations import evaluate_dev
from oa_groundrag.evaluation.rs_general.metrics import score_gate_b_text
from oa_groundrag.evaluation.rs_general.generation import _verify_selected_dataset
from oa_groundrag.evaluation.rs_general.selection import load_gate_b_selection_for_stage5_retention
from oa_groundrag.grounding.messages import build_mask_grounded_region_messages
from oa_groundrag.vlm.outputs import generic_prediction_row
from oa_groundrag.grounding.outputs import (
    parse_region_model_output,
    region_failure_row,
    region_prediction_row,
    region_provenance_row,
)


STAGE5_PREDICTION_MANIFEST_SCHEMA = "rs_vlm.mask_grounded_stage5_predictions.v1"
STAGE5_EVAL_REPORT_SCHEMA = "rs_vlm.mask_grounded_stage5_automatic_report.v1"


def _counterfactual_kind(program_facts: Mapping[str, Any]) -> str | None:
    """baseline 的 counterfactual 明确为 null；仅反事实变体读取 kind。"""

    value = program_facts.get("counterfactual")
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "Eval-dev counterfactual 程序事实非法",
        )
    return str(value["kind"])


def load_stage5_eval_samples(eval_root: Path | str) -> tuple[DescriptionSample, ...]:
    """严格装载既有 100 baseline + 240 反事实，不读取 sealed test。"""

    root = Path(os.path.abspath(Path(eval_root)))
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise PredictionError(ReasonCode.PREDICTION_INVALID, "Eval-dev manifest 非法")
    train_root = Path(str(manifest.get("train_corpus", {}).get("root", "")))
    report = validate_eval_dev(root, train_corpus_root=train_root, verify_source=True)
    if report["record_count"] != 340 or report["baseline_count"] != 100:
        raise PredictionError(ReasonCode.PREDICTION_INVALID, "Stage 5 Eval-dev 必须精确为 340/100")
    records = read_jsonl(root / "records.jsonl")
    queue = {str(row["record_id"]): row for row in read_jsonl(root / "annotation_queue.jsonl")}
    group_by_record: dict[str, str] = {}
    for group in read_jsonl(root / "counterfactual_groups.jsonl"):
        group_id = str(group["group_id"])
        for record_id in group["variants"].values():
            group_by_record[str(record_id)] = group_id
    manifest_sha = sha256_file(root / "manifest.json")
    output = []
    for record in records:
        record_id = str(record["record_id"])
        queue_row = queue.get(record_id)
        if queue_row is None:
            raise PredictionError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "Eval queue 缺失 record")
        messages = build_mask_grounded_region_messages(record, asset_root=root)
        output.append(DescriptionSample(
            record_id=record_id,
            parent_id=str(record["parent_id"]),
            logical_role="oa_grounded_eval_dev",
            task_family="mask_grounded_region_description",
            messages=tuple(messages),
            reference_responses=(),
            mask_mode=MaskMode.GT_MASK,
            evidence_ids=(str(queue_row["asset_identity_sha256"]),),
            provenance={
                "eval_manifest_sha256": manifest_sha,
                "asset_identity_sha256": queue_row["asset_identity_sha256"],
                "representation_mode": record["representation_mode"],
                "formal_model_input_roles": record["formal_model_input_roles"],
                "expected_target_status": record["target_status"],
                "prompt_sha256": sha256_text(canonical_json(messages)),
            },
            counterfactual={
                "group_id": group_by_record.get(record_id),
                "kind": _counterfactual_kind(record["program_facts"]),
            },
        ))
    return tuple(output)


def run_stage5_region_inference(
    *,
    config: Any,
    samples: Sequence[DescriptionSample],
    collator: Any,
    model: Any,
    processor: Any,
    output_root: Path,
    model_role: str,
    device: Any,
) -> Path:
    """逐条生成严格 Region prediction；非法 JSON 作为 failure 保存。"""

    if len(samples) != 340:
        raise PredictionError(ReasonCode.PREDICTION_INVALID, "Stage 5 inference 必须消费 340 条")
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    token_total = 0
    image_total = 0
    generation = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
    }
    for sample in samples:
        try:
            batch = collator([sample])
            token_total += int(batch["input_token_counts"][0])
            image_total += int(batch["image_counts"][0])
            moved = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            outputs = model.generate_text(
                moved,
                processor=processor,
                **generation,
            )
            if len(outputs) != 1:
                raise PredictionError(ReasonCode.PREDICTION_INVALID, "逐条生成必须返回一个文本")
            parsed = parse_region_model_output(outputs[0])
            provenance = region_provenance_row(
                record_id=sample.record_id,
                asset_manifest_sha256=str(sample.provenance["eval_manifest_sha256"]),
                asset_identity_sha256=str(sample.provenance["asset_identity_sha256"]),
                representation_mode=str(sample.provenance["representation_mode"]),
                formal_model_input_roles=sample.provenance["formal_model_input_roles"],
                prompt_sha256=str(sample.provenance["prompt_sha256"]),
                generation=generation,
            )
            predictions.append(region_prediction_row(
                record_id=sample.record_id,
                parent_id=sample.parent_id,
                output=parsed,
                provenance=provenance,
                counterfactual_group_id=(
                    None if sample.counterfactual is None
                    else sample.counterfactual.get("group_id")
                ),
            ))
        except VLMError as error:
            failures.append(region_failure_row(
                record_id=sample.record_id,
                parent_id=sample.parent_id,
                stage="stage5_inference",
                code=error.code,
                message=str(error),
                details=error.details,
            ))
        except Exception as error:
            failures.append(region_failure_row(
                record_id=sample.record_id,
                parent_id=sample.parent_id,
                stage="stage5_inference",
                code=ReasonCode.PREDICTION_INVALID,
                message=str(error),
            ))
    output = Path(output_root)
    with AtomicArtifactDirectory(output) as writer:
        writer.write_jsonl("predictions.jsonl", predictions)
        writer.write_jsonl("failures.jsonl", failures)
        writer.write_json("manifest.json", {
            "schema_version": STAGE5_PREDICTION_MANIFEST_SCHEMA,
            "model_role": model_role,
            "config_semantic_sha256": config.semantic_sha256,
            "input_count": len(samples),
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "input_token_count": token_total,
            "image_count": image_total,
            "reference_authority": "automatic_contract_only",
            "expert_metrics_available": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        })
        return writer.publish()


def evaluate_stage5_dev(
    *,
    eval_root: Path | str,
    prediction_root: Path | str,
    output_root: Path | str,
    model_role: str,
) -> dict[str, Any]:
    """复用 Stage 4 自动检查并发布 Stage 5 明确的 automatic-only 报告。"""

    prediction_root = Path(prediction_root)
    with tempfile.TemporaryDirectory(prefix="stage5_grounded_eval_") as temporary:
        legacy = evaluate_dev(
            eval_root=eval_root,
            predictions_path=prediction_root / "predictions.jsonl",
            output_root=Path(temporary) / "automatic",
            annotations_root=None,
        )
        legacy_report = read_json(Path(legacy["root"]) / "report.json")
    report = {
        "schema_version": STAGE5_EVAL_REPORT_SCHEMA,
        "model_role": model_role,
        "eval_root": str(Path(os.path.abspath(Path(eval_root)))),
        "eval_manifest_sha256": sha256_file(Path(eval_root) / "manifest.json"),
        "prediction_root": str(Path(os.path.abspath(prediction_root))),
        "prediction_manifest_sha256": sha256_file(prediction_root / "manifest.json"),
        "predictions_sha256": sha256_file(prediction_root / "predictions.jsonl"),
        "automatic_metrics": legacy_report["automatic_metrics"],
        "reference_authority": "automatic_contract_only",
        "expert_metrics_available": False,
        "retention_gate_frozen": False,
        "thresholds_frozen": False,
        "development_only": True,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_evaluated": False,
    }
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_json("report.json", report)
        ledger = ledger_rows(writer.staging, ("report.json",))
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        writer.write_json("manifest.json", {
            "schema_version": "rs_vlm.mask_grounded_stage5_report_artifact.v1",
            "model_role": model_role,
            "report_sha256": sha256_file(writer.path("report.json")),
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "reference_authority": "automatic_contract_only",
            "expert_metrics_available": False,
            "retention_gate_frozen": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        })
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "report_sha256": sha256_file(root / "report.json"),
        "automatic_metrics": report["automatic_metrics"],
        "formal_acceptance": False,
    }


def run_rs_general_retention_report(
    *,
    protocol_path: Path | str,
    selection_path: Path | str,
    frozen_rs_predictions_path: Path | str,
    model: Any,
    processor_adapter: Any,
    config: Any,
    device: Any,
    output_root: Path | str,
) -> dict[str, Any]:
    """在冻结 Gate B selection 上只报告相对 RS-General Adapter 的变化。"""

    context = load_gate_b_selection_for_stage5_retention(
        protocol_path,
        selection_path,
    )
    selection = context.selection
    frozen_rows = read_jsonl(Path(frozen_rs_predictions_path))
    frozen_by_id = {str(row.get("record_id")): row for row in frozen_rows}
    if len(frozen_by_id) != 256 or len(selection["items"]) != 256:
        raise PredictionError(ReasonCode.GATE_B_RUN_INVALID, "冻结 retention 输入必须精确为 256 条")
    predictions: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    from .processing import DescriptionCollator

    with tempfile.TemporaryDirectory(prefix="stage5_retention_derived_") as temporary:
        dataset = _verify_selected_dataset(context, derived_root=Path(temporary) / "derived")
        collator = DescriptionCollator(processor_adapter, training=False)
        for ordinal, item in enumerate(selection["items"]):
            sample = dataset[ordinal]
            batch = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in collator([sample]).items()
            }
            generated = model.generate_text(
                batch,
                processor=processor_adapter.processor,
                max_new_tokens=384,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            if len(generated) != 1 or not generated[0].strip():
                raise PredictionError(ReasonCode.PREDICTION_INVALID, "retention 逐条生成为空")
            provenance = {
                "selection_sha256": selection["selection_sha256"],
                "ordinal": ordinal,
                "model_role": "mask_grounded_region_adapter",
                "reference_authority": "frozen_gate_b_selection",
            }
            predictions.append(generic_prediction_row(
                record_id=sample.record_id,
                parent_id=sample.parent_id,
                logical_role=sample.logical_role,
                task_family=sample.task_family,
                generated_text=generated[0],
                reference_responses=sample.reference_responses,
                provenance=provenance,
            ))
            frozen = frozen_by_id.get(sample.record_id)
            if frozen is None or frozen.get("task_family") != sample.task_family:
                raise PredictionError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "冻结 RS prediction 配对失败")
            new_score = score_gate_b_text(
                generated[0], sample.reference_responses, task_family=sample.task_family
            )
            old_score = score_gate_b_text(
                str(frozen["generated_text"]),
                sample.reference_responses,
                task_family=sample.task_family,
            )
            paired.append({
                "record_id": sample.record_id,
                "parent_id": sample.parent_id,
                "task_family": sample.task_family,
                "source": item["source"],
                "region_adapter_primary": new_score["primary"],
                "rs_general_adapter_primary": old_score["primary"],
                "delta": new_score["primary"] - old_score["primary"],
            })
    task_values: dict[str, list[float]] = {}
    for row in paired:
        task_values.setdefault(str(row["task_family"]), []).append(float(row["delta"]))
    report = {
        "schema_version": "rs_vlm.mask_grounded_stage5_retention_report.v1",
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(Path(selection_path)),
        "frozen_rs_predictions_sha256": sha256_file(Path(frozen_rs_predictions_path)),
        "sample_count": len(paired),
        "selection_authority": "frozen_gate_b_selection_only",
        "historical_gate_b_implementation_match": (
            context.historical_implementation_match
        ),
        "historical_gate_b_acceptance_reused": False,
        "mean_primary_delta": sum(float(row["delta"]) for row in paired) / len(paired),
        "task_primary_deltas": {
            task: sum(values) / len(values) for task, values in sorted(task_values.items())
        },
        "retention_gate_frozen": False,
        "thresholds": None,
        "checkpoint_blocked_by_retention": False,
        "gate_f_passed": False,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_evaluated": False,
    }
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("predictions.jsonl", predictions)
        writer.write_jsonl("paired_scores.jsonl", paired)
        writer.write_json("report.json", report)
        ledger = ledger_rows(
            writer.staging,
            ("predictions.jsonl", "paired_scores.jsonl", "report.json"),
        )
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        writer.write_json("manifest.json", {
            "schema_version": "rs_vlm.mask_grounded_stage5_retention_artifact.v1",
            "selection_sha256": selection["selection_sha256"],
            "prediction_count": len(predictions),
            "selection_authority": "frozen_gate_b_selection_only",
            "historical_gate_b_implementation_match": (
                context.historical_implementation_match
            ),
            "historical_gate_b_acceptance_reused": False,
            "report_sha256": sha256_file(writer.path("report.json")),
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "retention_gate_frozen": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
        })
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "report_sha256": sha256_file(root / "report.json"),
        "mean_primary_delta": report["mean_primary_delta"],
        "formal_acceptance": False,
    }
