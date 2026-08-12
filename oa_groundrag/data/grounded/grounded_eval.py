"""val-only OA-GroundedEval-dev 的确定性选择、反事实构建与原子发布。"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oa_groundrag.phase3.common import canonical_json, read_json, read_jsonl, sha256_file, sha256_text
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.evidence import boundary_contrast_proxy, deterministic_shift_mask
from oa_groundrag.phase4.errors import EvidenceError

from .contracts import fail
from .region_contracts import (
    COUNTERFACTUAL_GROUP_SCHEMA,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SELECTION_ALGORITHM,
    MASK_RENDERER_VERSION,
    RGB_RENDERER_VERSION,
    CounterfactualKind,
    GroundedEvalConfig,
    RegionTargetStatus,
    RepresentationMode,
    load_grounded_eval_config,
    parent_identity,
    size_bin,
)
from .region_pipeline import (
    RegionBenchmarkAccess,
    RegionBuildResult,
    RegionLoadedSample,
    annotation_guideline,
    annotation_queue_rows,
    ledger_rows,
    materialize_region_record,
)
from .region_validation import validate_region_corpus
from .pipeline import render_optical


EVAL_DEV_NAME = "oa_grounded_eval_dev_v1_100"


@dataclass(frozen=True)
class EvalCandidate:
    row: Mapping[str, Any]
    loaded: RegionLoadedSample
    parent_id: str
    size: str
    component_class: str
    location_3x3: str
    contrast_score: float
    contrast_median: float
    difficulty_proxy: str
    stable_rank: str


def _rank(config: GroundedEvalConfig, *parts: object) -> str:
    return sha256_text("\0".join((str(config.seed), config.algorithm_version, *(str(part) for part in parts))))


def _candidate_pool(
    config: GroundedEvalConfig,
    access: RegionBenchmarkAccess,
    rows: Sequence[Mapping[str, Any]],
    *,
    quota: int,
) -> list[EvalCandidate]:
    """只读取固定 hash pool，并把难度明确标为可复算 proxy。"""

    limit = max(config.candidate_pool_min, quota * 8)
    ranked = sorted(rows, key=lambda row: (_rank(config, row["sample_id"]), str(row["sample_id"])))[:limit]
    intermediate: list[tuple[Mapping[str, Any], RegionLoadedSample, str, float, str]] = []
    for row in ranked:
        loaded = access.load(str(row["sample_id"]))
        if not bool(loaded.mask.any()) or loaded.mask_facts is None:
            fail("SELECTION_FAILED", "target candidate 意外为空 mask")
        try:
            deterministic_shift_mask(loaded.mask)
        except EvidenceError:
            continue
        optical = render_optical(loaded.sample)
        score = boundary_contrast_proxy(optical, loaded.mask)
        parent_id, _ = parent_identity(row)
        intermediate.append((row, loaded, parent_id, score, _rank(config, row["sample_id"])))
    if len(intermediate) < quota:
        fail("SELECTION_CAPACITY_INSUFFICIENT", "Eval target candidate pool 容量不足")
    ordered_scores = sorted((item[3], item[4]) for item in intermediate)
    median = float(statistics.median(score for score, _ in ordered_scores))
    difficulty_by_id: dict[str, str] = {}
    score_order = sorted(intermediate, key=lambda item: (item[3], item[4], str(item[0]["sample_id"])))
    difficult_count = (len(score_order) + 1) // 2
    for index, item in enumerate(score_order):
        difficulty_by_id[str(item[0]["sample_id"])] = (
            "difficult_proxy" if index < difficult_count else "clear_proxy"
        )
    result = []
    for row, loaded, parent_id, score, stable in intermediate:
        facts = loaded.mask_facts
        assert facts is not None
        result.append(EvalCandidate(
            row=row,
            loaded=loaded,
            parent_id=parent_id,
            size=size_bin(
                float(row["foreground_ratio"]),
                small_upper=config.small_upper_exclusive,
                medium_upper=config.medium_upper_exclusive,
            ),
            component_class="single" if facts["fragment_count"] == 1 else "multiple",
            location_3x3=str(facts["location_3x3"]),
            contrast_score=float(score),
            contrast_median=median,
            difficulty_proxy=difficulty_by_id[str(row["sample_id"])],
            stable_rank=stable,
        ))
    return result


def _greedy_cover(candidates: Sequence[EvalCandidate], quota: int) -> list[EvalCandidate]:
    chosen: list[EvalCandidate] = []
    remaining = list(candidates)
    component_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    while len(chosen) < quota and remaining:
        best = min(
            remaining,
            key=lambda candidate: (
                component_counts[candidate.component_class],
                difficulty_counts[candidate.difficulty_proxy],
                location_counts[candidate.location_3x3],
                candidate.stable_rank,
                str(candidate.row["sample_id"]),
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        component_counts[best.component_class] += 1
        difficulty_counts[best.difficulty_proxy] += 1
        location_counts[best.location_3x3] += 1
    if len(chosen) != quota:
        fail("SELECTION_CAPACITY_INSUFFICIENT", "Eval coverage greedy 无法达到 quota")
    return chosen


def select_eval_baselines(
    config: GroundedEvalConfig,
    access: RegionBenchmarkAccess,
    *,
    excluded_parents: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if access.allowed_split != "val":
        fail("SPLIT_FORBIDDEN", "Eval selector 只接受 val access")
    if config.source_order != access.source_order:
        fail("CONFIG_INVALID", "Eval source_order 必须等于 OA manifest included_sources")
    rows_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    excluded_overlap = 0
    for row in access.rows:
        parent_id, _ = parent_identity(row)
        if parent_id in excluded_parents:
            excluded_overlap += 1
            continue
        ratio = float(row["foreground_ratio"])
        cell = size_bin(
            ratio,
            small_upper=config.small_upper_exclusive,
            medium_upper=config.medium_upper_exclusive,
        )
        rows_by_cell[(str(row["source"]), cell)].append(row)
    selected: list[dict[str, Any]] = []
    used_parents: set[str] = set()
    reports: list[dict[str, Any]] = []
    for source in config.source_order:
        source_selected: list[dict[str, Any]] = []
        bin_report: dict[str, Any] = {}
        for bin_name, quota in (
            ("small", config.per_source.small),
            ("medium", config.per_source.medium),
            ("large", config.per_source.large),
        ):
            unique_rows = []
            seen_parents: set[str] = set()
            for row in sorted(
                rows_by_cell[(source, bin_name)],
                key=lambda item: (_rank(config, item["sample_id"]), str(item["sample_id"])),
            ):
                parent_id, _ = parent_identity(row)
                if parent_id in used_parents or parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                unique_rows.append(row)
            pool = _candidate_pool(config, access, unique_rows, quota=quota)
            chosen = _greedy_cover(pool, quota)
            for candidate in chosen:
                used_parents.add(candidate.parent_id)
                source_selected.append({
                    "row": candidate.row,
                    "loaded": candidate.loaded,
                    "size_bin": candidate.size,
                    "difficulty_proxy": {
                        "schema_version": "boundary_contrast_proxy.v1",
                        "label": candidate.difficulty_proxy,
                        "score": candidate.contrast_score,
                        "cell_median": candidate.contrast_median,
                        "split_rule": "score_then_stable_rank_median_halves.v1",
                        "expert_label": None,
                    },
                    "stable_rank": candidate.stable_rank,
                })
            bin_report[bin_name] = {
                "requested": quota,
                "achieved": len(chosen),
                "component_counts": dict(sorted(Counter(item.component_class for item in chosen).items())),
                "location_counts": dict(sorted(Counter(item.location_3x3 for item in chosen).items())),
                "difficulty_proxy_counts": dict(sorted(Counter(item.difficulty_proxy for item in chosen).items())),
            }
        no_target_rows = []
        for row in sorted(
            rows_by_cell[(source, "empty")],
            key=lambda item: (_rank(config, item["sample_id"]), str(item["sample_id"])),
        ):
            parent_id, _ = parent_identity(row)
            if parent_id in used_parents:
                continue
            used_parents.add(parent_id)
            no_target_rows.append(row)
            if len(no_target_rows) == config.per_source.no_target:
                break
        if len(no_target_rows) != config.per_source.no_target:
            fail("SELECTION_CAPACITY_INSUFFICIENT", f"{source} no-target parent 容量不足")
        for row in no_target_rows:
            source_selected.append({
                "row": row,
                "loaded": access.load(str(row["sample_id"])),
                "size_bin": "empty",
                "difficulty_proxy": None,
                "stable_rank": _rank(config, row["sample_id"]),
            })
        if len(source_selected) != config.per_source.total:
            fail("SELECTION_FAILED", f"{source} baseline 总数错误")
        source_selected.sort(key=lambda item: (
            {"small": 0, "medium": 1, "large": 2, "empty": 3}[item["size_bin"]],
            item["stable_rank"],
        ))
        selected.extend(source_selected)
        reports.append({
            "source": source,
            "requested": config.per_source.__dict__,
            "achieved": {
                "total": len(source_selected),
                "target": sum(item["size_bin"] != "empty" for item in source_selected),
                "no_target": sum(item["size_bin"] == "empty" for item in source_selected),
            },
            "target_bins": bin_report,
        })
    if len(selected) != config.sample_count or len(used_parents) != config.sample_count:
        fail("SELECTION_FAILED", "Eval baseline 必须是 100 个唯一 parent")
    return selected, {
        "sources": reports,
        "excluded_train_parent_row_count": excluded_overlap,
        "unique_parent_count": len(used_parents),
    }


def _eligible_swap(
    baseline: RegionLoadedSample,
    access: RegionBenchmarkAccess,
    rows_by_parent: Mapping[str, Sequence[Mapping[str, Any]]],
) -> RegionLoadedSample | None:
    parent_id, _ = parent_identity(baseline.row)
    base_provenance = baseline.row.get("provenance") or {}
    base_image_sha = base_provenance.get("source_image_sha256")
    if not isinstance(base_image_sha, str) or not base_image_sha:
        return None
    for row in sorted(rows_by_parent.get(parent_id, ()), key=lambda item: str(item["sample_id"])):
        if row["sample_id"] == baseline.row["sample_id"] or float(row["foreground_ratio"]) <= 0:
            continue
        provenance = row.get("provenance") or {}
        if provenance.get("source_image_sha256") != base_image_sha:
            continue
        donor = access.load(str(row["sample_id"]))
        if donor.mask.shape == baseline.mask.shape and not np.array_equal(donor.mask, baseline.mask):
            from .pipeline import render_optical

            if np.array_equal(np.asarray(render_optical(donor.sample)), np.asarray(render_optical(baseline.sample))):
                return donor
    return None


def build_eval_dev(config_path: Path | str) -> RegionBuildResult:
    config = load_grounded_eval_config(config_path)
    if config.output_root.exists() or config.output_root.is_symlink():
        fail("OUTPUT_EXISTS", f"Eval-dev output 已存在，拒绝覆盖：{config.output_root}")
    train_report = validate_region_corpus(config.train_corpus_root, verify_source=True)
    train_records = read_jsonl(config.train_corpus_root / "records.jsonl")
    train_parents = {str(record["parent_id"]) for record in train_records}
    train_samples = {str(record["sample_id"]) for record in train_records}
    access = RegionBenchmarkAccess(config.benchmark_root, config.benchmark_expected, allowed_split="val")
    selected, selection_report = select_eval_baselines(config, access, excluded_parents=train_parents)
    if train_samples & {str(item["row"]["sample_id"]) for item in selected}:
        fail("SPLIT_LEAKAGE", "Eval selected sample 与 train Corpus 相交")
    rows_by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in access.rows:
        parent_id, _ = parent_identity(row)
        rows_by_parent[parent_id].append(row)
    with AtomicArtifactDirectory(config.output_root) as writer:
        assert writer.staging is not None
        records: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        baseline_record_ids: list[str] = []
        swap_count = 0
        for index, item in enumerate(selected):
            loaded: RegionLoadedSample = item["loaded"]
            target = bool(loaded.mask.any())
            baseline = materialize_region_record(
                record_key=f"eval_{index + 1:04d}_baseline",
                loaded=loaded,
                access=access,
                writer=writer,
                context_margin_ratio=config.context_margin_ratio,
                overlay_alpha=config.overlay_alpha,
                counterfactual={
                    "kind": CounterfactualKind.BASELINE.value,
                    "evaluation_only": True,
                } if target else None,
                difficulty_proxy=item["difficulty_proxy"],
            )
            records.append(baseline)
            baseline_record_ids.append(baseline["record_id"])
            if not target:
                continue
            variants = {CounterfactualKind.BASELINE.value: baseline["record_id"]}
            shared_optical = baseline["assets"]["optical_full"]
            empty = materialize_region_record(
                record_key=f"eval_{index + 1:04d}_empty",
                loaded=loaded,
                access=access,
                writer=writer,
                effective_mask=np.zeros_like(loaded.mask),
                mask_source="counterfactual_empty_from_gt",
                representation_mode=RepresentationMode.FULL_PLUS_MASK,
                include_audit_overlay=False,
                shared_optical_path=shared_optical,
                counterfactual={"kind": CounterfactualKind.EMPTY_MASK.value, "evaluation_only": True},
                difficulty_proxy=item["difficulty_proxy"],
            )
            records.append(empty)
            variants[CounterfactualKind.EMPTY_MASK.value] = empty["record_id"]
            shifted_mask, shift = deterministic_shift_mask(loaded.mask)
            shifted = materialize_region_record(
                record_key=f"eval_{index + 1:04d}_shift",
                loaded=loaded,
                access=access,
                writer=writer,
                effective_mask=shifted_mask,
                mask_source="counterfactual_shift_from_gt",
                representation_mode=RepresentationMode.FULL_PLUS_MASK_PLUS_CROP,
                include_audit_overlay=False,
                shared_optical_path=shared_optical,
                context_margin_ratio=config.context_margin_ratio,
                counterfactual={"kind": CounterfactualKind.MASK_SHIFT.value, **shift},
                difficulty_proxy=item["difficulty_proxy"],
            )
            records.append(shifted)
            variants[CounterfactualKind.MASK_SHIFT.value] = shifted["record_id"]
            context_removed = materialize_region_record(
                record_key=f"eval_{index + 1:04d}_context_removed",
                loaded=loaded,
                access=access,
                writer=writer,
                mask_source="oa_auxseg_benchmark_gt",
                representation_mode=RepresentationMode.CROP_ONLY,
                include_audit_overlay=False,
                shared_optical_path=shared_optical,
                crop_margin_ratio=0.0,
                counterfactual={"kind": CounterfactualKind.CONTEXT_REMOVAL.value, "evaluation_only": True},
                difficulty_proxy=item["difficulty_proxy"],
            )
            records.append(context_removed)
            variants[CounterfactualKind.CONTEXT_REMOVAL.value] = context_removed["record_id"]
            donor = _eligible_swap(loaded, access, rows_by_parent)
            swap_eligible = donor is not None
            if donor is not None:
                swapped = materialize_region_record(
                    record_key=f"eval_{index + 1:04d}_swap",
                    loaded=loaded,
                    access=access,
                    writer=writer,
                    effective_mask=donor.mask,
                    mask_source="counterfactual_mask_swap_from_gt",
                    representation_mode=RepresentationMode.FULL_PLUS_MASK_PLUS_CROP,
                    include_audit_overlay=False,
                    shared_optical_path=shared_optical,
                    context_margin_ratio=config.context_margin_ratio,
                    counterfactual={
                        "kind": CounterfactualKind.MASK_SWAP.value,
                        "evaluation_only": True,
                        "donor_sample_id": donor.row["sample_id"],
                        "donor_record_sha256": donor.row["record_sha256"],
                    },
                    difficulty_proxy=item["difficulty_proxy"],
                )
                records.append(swapped)
                variants[CounterfactualKind.MASK_SWAP.value] = swapped["record_id"]
                swap_count += 1
            groups.append({
                "schema_version": COUNTERFACTUAL_GROUP_SCHEMA,
                "group_id": f"cfg_{sha256_text(baseline['record_id'])[:24]}",
                "parent_id": baseline["parent_id"],
                "baseline_record_id": baseline["record_id"],
                "variants": variants,
                "mask_swap_eligible": swap_eligible,
            })
        writer.write_jsonl("records.jsonl", records)
        writer.write_jsonl("counterfactual_groups.jsonl", groups)
        queue = annotation_queue_rows(records, writer.staging)
        writer.write_jsonl("annotation_queue.jsonl", queue)
        writer.write_json("annotation_guideline.json", annotation_guideline())
        assets = sorted({
            path for record in records for path in record["assets"].values() if isinstance(path, str)
        })
        ledger = ledger_rows(
            writer.staging,
            [
                "records.jsonl", "counterfactual_groups.jsonl", "annotation_queue.jsonl",
                "annotation_guideline.json", *assets,
            ],
        )
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        baseline_records = [record for record in records if record["record_id"] in baseline_record_ids]
        manifest = {
            "schema_version": EVAL_MANIFEST_SCHEMA,
            "eval_name": EVAL_DEV_NAME,
            "eval_id": sha256_text(canonical_json({
                "benchmark": config.benchmark_expected,
                "config": config.semantic_sha256,
                "baseline_samples": [record["sample_id"] for record in baseline_records],
            })),
            "benchmark": {"root": str(config.benchmark_root), **dict(config.benchmark_expected)},
            "train_corpus": {
                "root": str(config.train_corpus_root),
                "manifest_sha256": train_report["manifest_sha256"],
                "records_sha256": train_report["records_sha256"],
                "ledger_sha256": train_report["ledger_sha256"],
                "parent_count": len(train_parents),
            },
            "selection": {
                "split": "val",
                "algorithm_version": EVAL_SELECTION_ALGORITHM,
                "seed": config.seed,
                "config_file_sha256": config.config_file_sha256,
                "config_semantic_sha256": config.semantic_sha256,
                "sample_count": len(baseline_records),
                "baseline_record_ids": baseline_record_ids,
                "ordered_sample_ids": [record["sample_id"] for record in baseline_records],
                "ordered_parent_ids": [record["parent_id"] for record in baseline_records],
                "report": selection_report,
            },
            "rendering": {
                "context_margin_ratio": config.context_margin_ratio,
                "overlay_alpha": config.overlay_alpha,
                "rgb_renderer": RGB_RENDERER_VERSION,
                "mask_renderer": MASK_RENDERER_VERSION,
            },
            "counterfactuals": {
                "group_count": len(groups),
                "mask_swap_count": swap_count,
                "required_kinds": [
                    CounterfactualKind.BASELINE.value,
                    CounterfactualKind.EMPTY_MASK.value,
                    CounterfactualKind.MASK_SHIFT.value,
                    CounterfactualKind.CONTEXT_REMOVAL.value,
                ],
            },
            "records": {
                **next(item for item in ledger if item["path"] == "records.jsonl"),
                "count": len(records),
            },
            "annotation_queue": {
                **next(item for item in ledger if item["path"] == "annotation_queue.jsonl"),
                "count": len(queue),
            },
            "assets": {
                "count": len(assets),
                "total_bytes": sum(item["size_bytes"] for item in ledger if item["path"] in assets),
                "role_counts": dict(sorted(Counter(path.split("/")[1] for path in assets).items())),
            },
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "size_bytes": writer.path("SHA256SUMS.jsonl").stat().st_size,
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "development_only": True,
            "sealed_test_accessed": False,
            "expert_annotation_completed": False,
            "formal_acceptance": False,
        }
        writer.write_json("manifest.json", manifest)
        from .region_validation import validate_eval_dev

        validate_eval_dev(
            writer.staging,
            train_corpus_root=config.train_corpus_root,
            verify_source=True,
        )
        root = writer.publish()
    return RegionBuildResult(
        root=root,
        manifest_sha256=sha256_file(root / "manifest.json"),
        records_sha256=sha256_file(root / "records.jsonl"),
        ledger_sha256=sha256_file(root / "SHA256SUMS.jsonl"),
        record_count=len(records),
        asset_count=len(assets),
        asset_bytes=sum(item["size_bytes"] for item in ledger if item["path"] in assets),
    )
