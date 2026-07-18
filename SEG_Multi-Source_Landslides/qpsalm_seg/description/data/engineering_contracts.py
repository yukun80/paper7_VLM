"""Engineering-valid Description/Bridge and predicted-index bindings."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import read_jsonl, sha256_file, strict_json_loads
from .expert_contracts import BRIDGE_BUILDER_VERSION
from .vision_cache import DescriptionVisionFeatureBank


BRIDGE_ENGINEERING_AUDIT_PROTOCOL = (
    "landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound"
)
DESCRIPTION_BUILDER_VERSION = "description_benchmark_m1_v4_answer_trace"
DESCRIPTION_ENGINEERING_AUDIT_PROTOCOL = (
    "qpsalm_description_engineering_audit_v1_cache_partition_bound"
)
REGION_TRAINING_DATA_PROTOCOL = (
    "qpsalm_region_training_data_binding_v2_cache_candidate_bound"
)
REGION_INPUT_SOURCE_PROTOCOL = (
    "qpsalm_description_region_input_source_v2_native_cache_projection_bound"
)

def require_engineering_bridge(
    bridge_dir: Path,
    vision_bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Revalidate Bridge rows and the cache input used by region stages."""

    report_path = bridge_dir / "reports/validation_report.json"
    candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
    auto_path = bridge_dir / "indexes/auto_train.jsonl"
    for path in (report_path, candidate_path, auto_path):
        if not path.is_file():
            raise FileNotFoundError(f"engineering Bridge 缺少 artifact: {path}")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or report.get("status") not in {
            "awaiting_expert_review", "expert_pilot_frozen",
        }
        or report.get("pilot_protocol_complete") is not True
        or (report.get("errors") or [])
    ):
        raise RuntimeError(
            "D-1/D3a 要求当前 M2 Bridge engineering-valid 且 Pilot 完整；"
            f"当前 status={report.get('status')!r}"
        )
    candidates = read_jsonl(candidate_path, label="Bridge candidate index")
    auto_rows = read_jsonl(auto_path, label="Bridge auto-train index")
    candidate_ids = [str(row.get("bridge_record_id") or "") for row in candidates]
    auto_ids = [str(row.get("bridge_record_id") or "") for row in auto_rows]
    if (
        any(not value for value in candidate_ids + auto_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or len(auto_ids) != len(set(auto_ids))
    ):
        raise RuntimeError("engineering Bridge candidate/auto ID 缺失或重复")
    candidate_by_id = {
        str(row["bridge_record_id"]): row for row in candidates
    }
    expected_train = {
        record_id for record_id, row in candidate_by_id.items()
        if str(row.get("split") or "") == "train"
    }
    if set(auto_ids) != expected_train:
        raise RuntimeError(
            "engineering Bridge auto_train 不是 candidate train 的精确 ID 投影"
        )
    for row in auto_rows:
        record_id = str(row["bridge_record_id"])
        if row != candidate_by_id[record_id]:
            raise RuntimeError(
                f"engineering Bridge auto_train row 已偏离 candidate: {record_id}"
            )
    invalid_authority = [
        record_id for record_id, row in candidate_by_id.items()
        if not isinstance(row.get("candidate"), dict)
        or row["candidate"].get("protocol")
        != "landslide_bridge_rule_candidate_v1"
        or row["candidate"].get("is_expert_truth") is not False
        or "expert_target" in row
    ]
    if invalid_authority:
        raise RuntimeError(
            "engineering Bridge candidate authority 非法，禁止冒充 expert truth: "
            f"{invalid_authority[:10]}"
        )
    cache_inputs = dict(vision_bank.manifest.get("input_fingerprints") or {})
    multisource_parent = dict(cache_inputs.get("multisource_parent") or {})
    cache_root = resolve_project_path(
        str(multisource_parent.get("benchmark") or "")
    )
    candidate_resolved = candidate_path.resolve(strict=False)
    if (
        cache_root is None
        or cache_root.resolve(strict=False) != bridge_dir.resolve(strict=False)
        or multisource_parent.get("index") != "indexes/candidate_all.jsonl"
        or int(multisource_parent.get("size", -1))
        != candidate_resolved.stat().st_size
        or multisource_parent.get("sha256") != sha256_file(candidate_resolved)
        or multisource_parent.get("validation_report")
        != "reports/validation_report.json"
        or int(multisource_parent.get("validation_report_size", -1))
        != report_path.stat().st_size
        or multisource_parent.get("validation_report_sha256")
        != sha256_file(report_path)
        or multisource_parent.get("validation_builder_version")
        != BRIDGE_BUILDER_VERSION
        or multisource_parent.get("validation_status")
        != str(report.get("status") or "")
    ):
        raise RuntimeError(
            "Bridge live candidate index 与 Description Vision Cache binding 不一致"
        )
    observed_by_source = dict(sorted(Counter(
        str(row.get("region_source") or "") for row in candidates
    ).items()))
    observed_parents = len({
        str(row.get("parent_sample_id") or "") for row in candidates
    })
    if (
        int(report.get("records", -1)) != len(candidates)
        or int(report.get("parents", -1)) != observed_parents
        or report.get("records_by_region_source") != observed_by_source
    ):
        raise RuntimeError(
            "engineering Bridge validation summary 与 live candidate population 不一致"
        )
    population_payload = [
        {
            "bridge_record_id": str(row["bridge_record_id"]),
            "parent_sample_id": str(row.get("parent_sample_id") or ""),
            "split": str(row.get("split") or ""),
            "region_source": str(row.get("region_source") or ""),
            "candidate_is_expert_truth": row["candidate"]["is_expert_truth"],
        }
        for row in sorted(candidates, key=lambda value: str(value["bridge_record_id"]))
    ]
    population_sha256 = hashlib.sha256(json.dumps(
        population_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return {
        "protocol": BRIDGE_ENGINEERING_AUDIT_PROTOCOL,
        "status": str(report["status"]),
        "builder_version": BRIDGE_BUILDER_VERSION,
        "expert_truth_used": False,
        "validation_report": str(report_path.resolve(strict=False)),
        "validation_report_sha256": sha256_file(report_path),
        "cache_input_fingerprint": multisource_parent,
        "candidate_index": str(candidate_path.resolve(strict=False)),
        "candidate_index_sha256": sha256_file(candidate_path),
        "auto_train_index": str(auto_path.resolve(strict=False)),
        "auto_train_index_sha256": sha256_file(auto_path),
        "candidate_records": len(candidates),
        "auto_train_records": len(auto_rows),
        "population_sha256": population_sha256,
    }


def require_engineering_description(
    description_dir: Path,
    vision_bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Bind live M1.1 partitions to the all-index used by Description Cache v1."""

    report_path = description_dir / "reports/validation_report.json"
    index_paths = {
        name: description_dir / f"indexes/{name}.jsonl"
        for name in ("all", "train", "dev", "test", "train_eligible")
    }
    missing = [str(path) for path in (report_path, *index_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"engineering Description 缺少 artifact: {missing}"
        )
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("builder_version") != DESCRIPTION_BUILDER_VERSION
        or (report.get("errors") or [])
        or int(
            report.get(
                "verified_perceptual_duplicate_cross_split_groups", -1
            )
        ) != 0
    ):
        raise RuntimeError(
            "D-1/D0-D2 要求 engineering-valid Description M1.1 v4，"
            "且 verified cross-split cluster 必须为零"
        )
    cache_inputs = dict(vision_bank.manifest.get("input_fingerprints") or {})
    single_image = dict(cache_inputs.get("single_image") or {})
    cache_root = resolve_project_path(str(single_image.get("benchmark") or ""))
    all_path = index_paths["all"].resolve(strict=False)
    if (
        cache_root is None
        or cache_root.resolve(strict=False) != description_dir.resolve(strict=False)
        or single_image.get("index") != "indexes/all.jsonl"
        or int(single_image.get("size", -1)) != all_path.stat().st_size
        or single_image.get("sha256") != sha256_file(all_path)
        or single_image.get("validation_report")
        != "reports/validation_report.json"
        or int(single_image.get("validation_report_size", -1))
        != report_path.stat().st_size
        or single_image.get("validation_report_sha256")
        != sha256_file(report_path)
        or single_image.get("validation_builder_version")
        != DESCRIPTION_BUILDER_VERSION
        or single_image.get("validation_status") != "engineering-valid"
    ):
        raise RuntimeError(
            "Description M1.1 live all index 与 Description Vision Cache binding 不一致"
        )
    rows_by_name = {
        name: read_jsonl(path, label=f"Description {name} index")
        for name, path in index_paths.items()
    }
    all_rows = rows_by_name["all"]
    all_ids = [str(row.get("sample_id") or "") for row in all_rows]
    if any(not value for value in all_ids) or len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Description M1.1 all index sample_id 缺失或重复")
    all_by_id = {
        str(row["sample_id"]): row for row in all_rows
    }
    partition_ids: list[str] = []
    for split in ("train", "dev", "test"):
        split_rows = rows_by_name[split]
        split_ids = [str(row.get("sample_id") or "") for row in split_rows]
        if (
            any(str(row.get("split") or "") != split for row in split_rows)
            or any(
                sample_id not in all_by_id
                or row != all_by_id[sample_id]
                for sample_id, row in zip(split_ids, split_rows, strict=True)
            )
        ):
            raise RuntimeError(
                f"Description M1.1 {split} 不是 all index 的精确投影"
            )
        partition_ids.extend(split_ids)
    if Counter(partition_ids) != Counter(all_ids):
        raise RuntimeError("Description M1.1 train/dev/test 未精确分区 all index")
    train_by_id = {
        str(row["sample_id"]): row for row in rows_by_name["train"]
    }
    expected_eligible: dict[str, dict[str, Any]] = {}
    for sample_id, source_row in train_by_id.items():
        positive_answers = [
            answer for answer in source_row.get("answers", [])
            if float(answer.get("caption_quality_weight", 0.0)) > 0.0
        ]
        if positive_answers:
            expected = dict(source_row)
            expected["answers"] = positive_answers
            expected_eligible[sample_id] = expected
    observed_eligible = {
        str(row.get("sample_id") or ""): row
        for row in rows_by_name["train_eligible"]
    }
    if (
        len(observed_eligible) != len(rows_by_name["train_eligible"])
        or observed_eligible != expected_eligible
    ):
        raise RuntimeError(
            "Description M1.1 train_eligible 不是正权重 train 的精确投影"
        )
    observed_parents = len({
        str(row.get("parent_sample_id") or "") for row in all_rows
    })
    if (
        int(report.get("num_records", -1)) != len(all_rows)
        or int(report.get("deep_checked_records", -1)) != len(all_rows)
        or int(report.get("num_parents", -1)) != observed_parents
        or int(report.get("decoded_unique_images", -1)) != observed_parents
        or int(report.get("materialized_files", -1)) != observed_parents
        or int(report.get("train_eligible_records", -1))
        != len(observed_eligible)
    ):
        raise RuntimeError(
            "Description M1.1 validation summary 与 live index population 不一致"
        )
    index_bindings = {
        name: {
            "path": str(path.resolve(strict=False)),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "records": len(rows_by_name[name]),
        }
        for name, path in index_paths.items()
    }
    return {
        "protocol": DESCRIPTION_ENGINEERING_AUDIT_PROTOCOL,
        "builder_version": DESCRIPTION_BUILDER_VERSION,
        "validation_report": str(report_path.resolve(strict=False)),
        "validation_report_sha256": sha256_file(report_path),
        "cache_input_fingerprint": single_image,
        "indexes": index_bindings,
        "num_records": len(all_rows),
        "num_parents": observed_parents,
        "verified_perceptual_duplicate_cross_split_groups": 0,
    }


def validate_predicted_index(
    index_path: Path, *, split: str, expert_gate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Bind fixed/OOF masks to the report that published their exact index."""
    if split == "train":
        # 延迟导入避免 data -> predicted_regions -> data 的模块级环依赖。
        from .predicted_regions import revalidate_oof_merged_index

        replay = revalidate_oof_merged_index(
            index_path,
            expected_expert_gate_audit=expert_gate_audit,
        )
        return {**replay, "split": "train"}
    # val/test 同样逐行重放 checkpoint、专家源记录与每个 mask，而非只看顶层 report。
    from .predicted_regions import revalidate_fixed_predicted_index

    return revalidate_fixed_predicted_index(
        index_path,
        split=split,
        expected_expert_gate_audit=expert_gate_audit,
    )


def revalidate_predicted_index_audit(
    audit: Any,
    *,
    expected_split: str,
    expert_gate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a saved predicted-index audit from its bound live artifacts."""
    if (
        not isinstance(audit, dict)
        or str(audit.get("split") or "") != expected_split
        or not str(audit.get("index") or "").strip()
    ):
        raise ValueError(
            f"predicted index audit 缺少 split={expected_split} 的可重放 index"
        )
    index_path = resolve_project_path(str(audit["index"])) or Path(str(audit["index"]))
    current = validate_predicted_index(
        index_path,
        split=expected_split,
        expert_gate_audit=expert_gate_audit,
    )
    if current != audit:
        raise ValueError("predicted index audit 与当前深度重放结果不一致")
    return current
