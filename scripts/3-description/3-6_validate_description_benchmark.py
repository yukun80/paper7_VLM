#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-6：验证自包含 qpsalm_description_v2 benchmark。

用途：检查 schema、路径/解码/hash、bbox、双向 region pair、partition 和 split 泄漏约束。
推荐运行命令：python scripts/3-description/3-6_validate_description_benchmark.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：3-5 生成的物化图片、最终 indexes、parent/split manifests 与 schema。
主要输出：reports/validation_report.json；errors 非空时返回非零状态。
写入行为：不修改索引和源数据；--dry-run 不写报告。
所属流程：Description Benchmark M1 硬质量门。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from description_common import (
    BUILDER_VERSION, REPO_ROOT, bbox_pixel_half_open, description_dir_for_mode,
    ensure_writable, read_json, read_jsonl, resolve_project_path, sha256_file,
    sha256_jsonl_rows, source_slug, write_json,
)


REQUIRED = {
    "schema_version", "sample_id", "parent_sample_id", "source_dataset", "component_benchmark",
    "split", "task_family", "visual_ref", "region_geometry", "target_status", "region_source",
    "instruction", "answer_type", "answers", "structured_targets", "provenance", "quality_flags",
    "canonical_parent_id", "merged_parent_ids", "source_datasets", "perceptual_cluster_id",
}
PROVENANCE_REQUIRED = {
    "builder_version", "annotation_path", "original_record_id", "license_status",
    "license_source", "annotation_origin", "source_image_path", "sources",
}
FORBIDDEN_REF_PARTS = ("/json/total.json", "/classification/", "/detection/", "/vqa/", "/infrared/")
MATERIALIZATION_PROTOCOL = "qpsalm_description_materialization_v2"
SPLIT_PROTOCOL = "qpsalm_description_parent_split_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 Description Benchmark M1")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0, help="smoke 时限制逐行深度检查数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_schema_files(errors: list[str]) -> None:
    for name in ("qpsalm_description_record_v2.schema.json", "qpsalm_description_output_v1.schema.json"):
        path = REPO_ROOT / "configs" / name
        try:
            payload = read_json(path)
            if payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                errors.append(f"{name}: $schema 必须使用 draft 2020-12")
        except Exception as exc:
            errors.append(f"{name}: 无法解析: {exc}")
    ontology = REPO_ROOT / "configs/description_ontology_v1.yaml"
    if not ontology.exists() or "description_ontology_v1" not in ontology.read_text(encoding="utf-8"):
        errors.append("description_ontology_v1.yaml 缺失或版本不正确")


def validate_row(
    row: dict[str, Any], image_cache: dict[str, dict[str, Any]], data_root: Path,
    errors: list[str], warnings: list[str]
) -> None:
    sample_id = str(row.get("sample_id") or "<missing>")
    missing = sorted(REQUIRED - set(row))
    if missing:
        errors.append(f"{sample_id}: 缺少字段 {missing}")
        return
    if row["schema_version"] != "qpsalm_description_v2":
        errors.append(f"{sample_id}: schema_version 非法")
    if row["split"] not in {"train", "dev", "test"}:
        errors.append(f"{sample_id}: split 非法 {row['split']!r}")
    if row["target_status"] not in {"present", "absent", "uncertain"}:
        errors.append(f"{sample_id}: target_status 非法")
    if not isinstance(row["instruction"], str) or not row["instruction"].strip():
        errors.append(f"{sample_id}: instruction 为空")
    if not isinstance(row["answers"], list) or not row["answers"]:
        errors.append(f"{sample_id}: answers 为空")
    for answer in row.get("answers", []):
        if not str(answer.get("text", "")).strip():
            errors.append(f"{sample_id}: answer text 为空")
        if answer.get("language") != "en":
            errors.append(f"{sample_id}: 首版语言必须为 en")
        weight = answer.get("caption_quality_weight")
        if not isinstance(weight, (int, float)) or not 0 <= float(weight) <= 1:
            errors.append(f"{sample_id}: caption_quality_weight 非法")
        sources = answer.get("source_provenance")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{sample_id}: answer 缺少 source_provenance")
        else:
            for source in sources:
                required = {
                    "source_dataset", "source_parent_sample_id", "source_sample_id",
                    "source_answer_index", "source_text_sha256", "annotation_origin",
                }
                missing_source = sorted(required - set(source))
                if missing_source:
                    errors.append(
                        f"{sample_id}: answer source_provenance 缺少 {missing_source}"
                    )
                digest = str(source.get("source_text_sha256") or "")
                if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                    errors.append(f"{sample_id}: source_text_sha256 非法")

    source_datasets = row.get("source_datasets")
    if not isinstance(source_datasets, list) or not source_datasets:
        errors.append(f"{sample_id}: source_datasets 缺失或为空")
    if row.get("canonical_parent_id") != row.get("parent_sample_id"):
        errors.append(f"{sample_id}: final record 必须引用 canonical parent")
    merged_ids = row.get("merged_parent_ids")
    if not isinstance(merged_ids, list) or not merged_ids:
        errors.append(f"{sample_id}: merged_parent_ids 缺失或为空")

    provenance = row["provenance"]
    if not isinstance(provenance, dict):
        errors.append(f"{sample_id}: provenance 必须是 object")
    else:
        missing_provenance = sorted(PROVENANCE_REQUIRED - set(provenance))
        if missing_provenance:
            errors.append(f"{sample_id}: provenance 缺少 {missing_provenance}")
        refs = json.dumps(provenance, ensure_ascii=False).casefold()
        if any(part in refs for part in FORBIDDEN_REF_PARTS):
            errors.append(f"{sample_id}: provenance 引用了禁止数据源")
        if provenance.get("builder_version") != BUILDER_VERSION:
            errors.append(f"{sample_id}: builder_version 不是当前物化协议")
        if not str(provenance.get("source_image_path", "")).startswith("datasets/"):
            errors.append(f"{sample_id}: provenance.source_image_path 必须保留 datasets 逻辑路径")
        provenance_sources = provenance.get("sources")
        if not isinstance(provenance_sources, list) or not provenance_sources:
            errors.append(f"{sample_id}: provenance.sources 缺失或为空")
        else:
            for source in provenance_sources:
                if not str(source.get("source_image_path", "")).startswith("datasets/"):
                    errors.append(f"{sample_id}: provenance.sources 存在非 datasets 源路径")

    visual = row["visual_ref"]
    if not isinstance(visual, dict) or visual.get("type") != "single_image":
        errors.append(f"{sample_id}: visual_ref 必须是 single_image")
        return
    path_ref = str(visual.get("path", ""))
    path = resolve_project_path(path_ref)
    if path_ref.startswith("datasets/"):
        errors.append(f"{sample_id}: 最终 visual_ref 不得指向 datasets")
    try:
        path.resolve(strict=False).relative_to(data_root)
    except ValueError:
        errors.append(f"{sample_id}: 最终 visual_ref 不在 benchmark/data 内: {path_ref}")
    if visual.get("storage_mode") != "materialized_copy":
        errors.append(f"{sample_id}: visual_ref.storage_mode 必须是 materialized_copy")
    if path_ref not in image_cache:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image.load()
                if image.mode not in {"RGB", "RGBA"}:
                    raise ValueError(f"mode={image.mode}")
                width, height = image.size
            image_cache[path_ref] = {"width": width, "height": height, "sha256": sha256_file(path)}
        except Exception as exc:
            errors.append(f"{sample_id}: 图像无法验证 {path_ref}: {exc}")
            return
    decoded = image_cache[path_ref]
    if [visual.get("width"), visual.get("height")] != [decoded["width"], decoded["height"]]:
        errors.append(f"{sample_id}: visual_ref 尺寸与解码尺寸不一致")
    if visual.get("sha256") != decoded["sha256"]:
        errors.append(f"{sample_id}: visual_ref sha256 与文件不一致")
    modality = visual.get("modality_instance") or {}
    if modality.get("family") != "optical" or modality.get("product_type") != "rgb":
        errors.append(f"{sample_id}: single-image modality 必须是 optical/rgb")
    if modality.get("native_gsd_m") is not None or modality.get("aligned_gsd_m") is not None:
        errors.append(f"{sample_id}: single-image 不得伪造 GSD")

    geometry = row["region_geometry"]
    if not isinstance(geometry, dict) or geometry.get("type") not in {"full_image", "box", "mask", "null"}:
        errors.append(f"{sample_id}: region_geometry 非法")
        return
    if geometry["type"] == "box":
        bbox = geometry.get("bbox_xyxy_normalized")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{sample_id}: box 缺少 normalized bbox")
        else:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                errors.append(f"{sample_id}: normalized bbox 越界")
            expected = bbox_pixel_half_open(bbox, decoded["width"], decoded["height"])
            if geometry.get("bbox_xyxy_pixel_half_open") != expected:
                errors.append(f"{sample_id}: pixel half-open bbox 不一致")
    elif geometry["type"] == "full_image":
        if geometry.get("bbox_xyxy_normalized") is not None:
            errors.append(f"{sample_id}: full_image 不应保存 bbox")

    if row["source_dataset"] == "RSIEval" and row["split"] != "test":
        errors.append(f"{sample_id}: RSIEval 必须仅位于 test")
    if row["task_family"] == "region_grounding" and row["answer_type"] != "candidate_region_id":
        errors.append(f"{sample_id}: region_grounding 必须使用 candidate_region_id，不得回归自由坐标")
    if row["source_dataset"].startswith("MMRS-") and provenance.get("license_status") == "redistributable":
        warnings.append(f"{sample_id}: MMRS 许可被声明为 redistributable，需要人工复核")


def validate_global_invariants(
    rows: list[dict[str, Any]], parents: list[dict[str, Any]], output_dir: Path,
    errors: list[str], warnings: list[str],
) -> None:
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample_id 不唯一")
    parent_splits: dict[str, set[str]] = defaultdict(set)
    pair_tasks: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        parent_splits[str(row["parent_sample_id"])].add(str(row["split"]))
        if row.get("region_pair_id"):
            pair_tasks[str(row["region_pair_id"])].add(str(row["task_family"]))
            pair_splits[str(row["region_pair_id"])].add(str(row["split"]))
    for parent_id, splits in parent_splits.items():
        if len(splits) != 1:
            errors.append(f"parent 跨 split: {parent_id} -> {sorted(splits)}")
    expected_pair_tasks = {"region_referring_expression", "region_grounding"}
    for pair_id, tasks in pair_tasks.items():
        if tasks != expected_pair_tasks:
            errors.append(f"region pair 视图不完整: {pair_id} -> {sorted(tasks)}")
        if len(pair_splits[pair_id]) != 1:
            errors.append(f"region pair 跨 split: {pair_id}")

    for field in ("sha256", "connected_group_id", "exact_cluster_id"):
        grouped: dict[str, set[str]] = defaultdict(set)
        for parent in parents:
            grouped[str(parent[field])].add(str(parent["split"]))
        for key, splits in grouped.items():
            if len(splits) != 1:
                errors.append(f"{field} 跨 split: {key} -> {sorted(splits)}")

    original_parent_owner: dict[str, str] = {}
    perceptual_clusters: dict[str, list[str]] = defaultdict(list)
    for parent in parents:
        parent_id = str(parent["parent_sample_id"])
        if parent.get("canonical_parent_id") != parent_id:
            errors.append(f"parent manifest 非 canonical parent: {parent_id}")
        source_datasets = parent.get("source_datasets")
        if not isinstance(source_datasets, list) or not source_datasets:
            errors.append(f"parent 缺少 source_datasets: {parent_id}")
        merged_ids = parent.get("merged_parent_ids")
        if not isinstance(merged_ids, list) or not merged_ids:
            errors.append(f"parent 缺少 merged_parent_ids: {parent_id}")
            continue
        for original_id in merged_ids:
            previous = original_parent_owner.setdefault(str(original_id), parent_id)
            if previous != parent_id:
                errors.append(
                    f"原始 parent 被多个 canonical parent 持有: {original_id} -> {previous}, {parent_id}"
                )
        cluster_id = str(parent.get("perceptual_cluster_id") or "")
        if not cluster_id:
            errors.append(f"parent 缺少 perceptual_cluster_id: {parent_id}")
        else:
            perceptual_clusters[cluster_id].append(parent_id)
    for cluster_id, parent_ids in perceptual_clusters.items():
        if len(parent_ids) != 1:
            errors.append(f"perceptual cluster 发布了多个 canonical parent: {cluster_id} -> {parent_ids}")
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for parent in parents:
        if parent["source_dataset"] in {"RSICap", "RSIEval"} and parent.get("source_scene_group"):
            scene_splits[str(parent["source_scene_group"])].add(str(parent["split"]))
    for scene, splits in scene_splits.items():
        if len(splits) != 1:
            errors.append(f"RSGPT scene group 跨 split: {scene} -> {sorted(splits)}")

    partition_ids: list[str] = []
    for split in ("train", "dev", "test"):
        split_rows = read_jsonl(output_dir / f"indexes/{split}.jsonl")
        if any(row.get("split") != split for row in split_rows):
            errors.append(f"indexes/{split}.jsonl 包含其他 split")
        partition_ids.extend(str(row["sample_id"]) for row in split_rows)
    if Counter(partition_ids) != Counter(sample_ids):
        errors.append("train/dev/test partition 与 indexes/all.jsonl 不一致")

    state = read_json(output_dir / "manifests/materialization_state.json")
    split_protocol = read_json(output_dir / "manifests/split_protocol.json")
    verified_duplicates = read_jsonl(output_dir / "manifests/verified_perceptual_duplicates.jsonl")
    materialized = read_jsonl(output_dir / "manifests/materialization_manifest.jsonl")
    materialization_report = read_json(output_dir / "reports/materialization_report.json")
    if state.get("protocol") != MATERIALIZATION_PROTOCOL or state.get("status") != "complete":
        errors.append("materialization_state 不是当前 complete 协议")
    if state.get("builder_version") != BUILDER_VERSION:
        errors.append("materialization_state builder_version 过期")
    if materialization_report.get("errors"):
        errors.append("materialization_report 仍包含 errors")
    if state.get("materialization_manifest_hash") != sha256_jsonl_rows(materialized):
        errors.append("materialization manifest hash 与 state 不一致")
    if state.get("final_index_hash") != sha256_jsonl_rows(rows):
        errors.append("final index hash 与 state 不一致")
    if split_protocol.get("protocol") != SPLIT_PROTOCOL:
        errors.append("split_protocol 不是当前 canonical split 协议")
    if split_protocol.get("verified_duplicate_manifest_hash") != sha256_jsonl_rows(verified_duplicates):
        errors.append("verified duplicate manifest hash 与 split protocol 不一致")

    selected_parent_ids = {str(parent["parent_sample_id"]) for parent in parents}
    for cluster in verified_duplicates:
        canonical_id = str(cluster.get("canonical_parent_id") or "")
        selected_for_mode = bool(cluster.get("selected_for_mode"))
        if selected_for_mode != (canonical_id in selected_parent_ids):
            errors.append(f"verified cluster selected 状态不一致: {cluster.get('perceptual_cluster_id')}")
        members = cluster.get("merged_parent_ids")
        if not isinstance(members, list) or len(members) < 2:
            errors.append(f"verified cluster 成员不足: {cluster.get('perceptual_cluster_id')}")
            continue
        final_split = str(cluster.get("assigned_final_split") or "")
        details = cluster.get("member_details")
        if final_split not in {"train", "dev", "test"}:
            errors.append(f"verified cluster final split 非法: {cluster.get('perceptual_cluster_id')}")
        if not isinstance(details, list) or {
            str(value.get("parent_sample_id")) for value in details
        } != {str(value) for value in members}:
            errors.append(f"verified cluster member_details 不完整: {cluster.get('perceptual_cluster_id')}")
        else:
            if any(str(value.get("assigned_final_split")) != final_split for value in details):
                errors.append(f"verified cluster member split 不一致: {cluster.get('perceptual_cluster_id')}")
            canonical_members = [
                str(value.get("parent_sample_id")) for value in details if value.get("is_canonical")
            ]
            if canonical_members != [canonical_id]:
                errors.append(f"verified cluster canonical 标记不唯一: {cluster.get('perceptual_cluster_id')}")
            if any(value.get("source_declared_split") == "test" for value in details) and final_split != "test":
                errors.append(f"verified cluster 未继承 official test: {cluster.get('perceptual_cluster_id')}")
        edges = cluster.get("verified_edges")
        threshold = float(cluster.get("mae_threshold", -1.0))
        if not isinstance(edges, list) or not edges:
            errors.append(f"verified cluster 缺少 MAE edge: {cluster.get('perceptual_cluster_id')}")
        elif any(float(edge.get("rgb64_mae", 256.0)) > threshold for edge in edges):
            errors.append(f"verified cluster 包含未通过 MAE 的 edge: {cluster.get('perceptual_cluster_id')}")
        if selected_for_mode:
            owner_ids = {original_parent_owner.get(str(member)) for member in members}
            if owner_ids != {canonical_id}:
                errors.append(
                    f"verified cluster 未完全合并到 canonical parent: {cluster.get('perceptual_cluster_id')}"
                )

    source_global_rows = read_jsonl(output_dir / "indexes/global_caption_source.jsonl")
    source_global_by_parent = {
        str(row["parent_sample_id"]): row for row in source_global_rows
    }
    for row in rows:
        if row.get("task_family") != "global_caption":
            continue
        expected_answers: set[tuple[str, int, str]] = set()
        for original_parent_id in row.get("merged_parent_ids", []):
            source_row = source_global_by_parent.get(str(original_parent_id))
            if source_row is None:
                errors.append(
                    f"{row['sample_id']}: merged caption parent 无源记录 {original_parent_id}"
                )
                continue
            for answer_index, answer in enumerate(source_row.get("answers", [])):
                expected_answers.add((
                    str(source_row["sample_id"]),
                    answer_index,
                    hashlib.sha256(str(answer.get("text", "")).encode("utf-8")).hexdigest(),
                ))
        actual_answers = {
            (
                str(source.get("source_sample_id")),
                (
                    int(source["source_answer_index"])
                    if isinstance(source.get("source_answer_index"), int) else -1
                ),
                str(source.get("source_text_sha256") or ""),
            )
            for answer in row.get("answers", [])
            for source in answer.get("source_provenance", [])
        }
        if actual_answers != expected_answers:
            errors.append(
                f"{row['sample_id']}: canonical caption 未完整追溯 source answers "
                f"expected={len(expected_answers)} actual={len(actual_answers)}"
            )

    materialized_ids = [str(row["parent_sample_id"]) for row in materialized]
    parent_ids = [str(row["parent_sample_id"]) for row in parents]
    if len(materialized_ids) != len(set(materialized_ids)):
        errors.append("materialization manifest parent_sample_id 不唯一")
    if Counter(materialized_ids) != Counter(parent_ids):
        errors.append("materialization manifest 与 parent manifest 不一一对应")
    materialized_by_parent = {str(row["parent_sample_id"]): row for row in materialized}
    parent_by_id = {str(row["parent_sample_id"]): row for row in parents}
    row_paths_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row_paths_by_parent[str(row["parent_sample_id"])].add(str(row["visual_ref"]["path"]))
    expected_files: set[Path] = set()
    data_root = (output_dir / "data").resolve(strict=False)
    for parent_id, item in materialized_by_parent.items():
        parent = parent_by_id[parent_id]
        image_ref = str(item.get("image_path", ""))
        image_path = resolve_project_path(image_ref).resolve(strict=False)
        expected_files.add(image_path)
        try:
            relative = image_path.relative_to(data_root)
        except ValueError:
            errors.append(f"materialized parent 不在 benchmark/data 内: {parent_id}")
            continue
        parts = relative.parts
        if len(parts) != 3:
            errors.append(f"materialized 路径层级非法: {parent_id} -> {relative}")
        else:
            if parts[0] != str(item.get("split")):
                errors.append(f"materialized 路径 split 不一致: {parent_id}")
            if parts[1] != source_slug(str(item.get("source_dataset"))):
                errors.append(f"materialized 路径 source slug 不一致: {parent_id}")
        if item.get("storage_mode") != "materialized_copy":
            errors.append(f"materialization storage_mode 非法: {parent_id}")
        if parent.get("image_path") != image_ref or parent.get("source_image_path") != item.get("source_image_path"):
            errors.append(f"parent/materialization 路径不一致: {parent_id}")
        if parent.get("storage_mode") != "materialized_copy":
            errors.append(f"parent storage_mode 非法: {parent_id}")
        if row_paths_by_parent.get(parent_id) != {image_ref}:
            errors.append(f"同一 parent 最终任务路径不唯一: {parent_id}")
        if not image_path.is_file():
            errors.append(f"materialized 图片不存在: {parent_id} -> {image_ref}")
        elif image_path.stat().st_size != int(item.get("file_size_bytes", -1)):
            errors.append(f"materialized 图片字节数不一致: {parent_id}")

    actual_files = {
        path.resolve(strict=False)
        for path in (output_dir / "data").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    part_files = sorted(str(path) for path in actual_files if path.name.endswith(".part"))
    if part_files:
        errors.append(f"benchmark/data 存在未完成 .part 文件，首个为 {part_files[0]}")
    extra_files = actual_files - expected_files
    missing_files = expected_files - actual_files
    if extra_files:
        errors.append(f"benchmark/data 存在 {len(extra_files)} 个未登记文件")
    if missing_files:
        errors.append(f"benchmark/data 缺少 {len(missing_files)} 个登记文件")

    for filename in ("global_caption_source.jsonl", "region_alignment_source.jsonl", "selected_source_all.jsonl"):
        source_rows = read_jsonl(output_dir / "indexes" / filename)
        for source_row in source_rows:
            if not str(source_row["visual_ref"]["path"]).startswith("datasets/"):
                errors.append(f"{filename}: source visual_ref 不再指向 datasets")
                break

    train_rows = read_jsonl(output_dir / "indexes/train.jsonl")
    train_eligible = read_jsonl(output_dir / "indexes/train_eligible.jsonl")
    train_by_id = {str(row["sample_id"]): row for row in train_rows}
    expected_eligible: dict[str, list[str]] = {}
    for row in train_rows:
        positive_answers = [
            str(answer["text"])
            for answer in row.get("answers", [])
            if float(answer.get("caption_quality_weight", 0.0)) > 0.0
        ]
        if positive_answers:
            expected_eligible[str(row["sample_id"])] = positive_answers
    if {str(row["sample_id"]) for row in train_eligible} != set(expected_eligible):
        errors.append("train_eligible 与正权重训练记录集合不一致")
    for row in train_eligible:
        sample_id = str(row["sample_id"])
        if sample_id not in train_by_id or row.get("split") != "train":
            errors.append(f"train_eligible 包含非法记录: {sample_id}")
            continue
        if any(float(answer.get("caption_quality_weight", 0.0)) <= 0.0 for answer in row.get("answers", [])):
            errors.append(f"train_eligible 仍包含零权重答案: {sample_id}")
        if [str(answer["text"]) for answer in row.get("answers", [])] != expected_eligible[sample_id]:
            errors.append(f"train_eligible 答案过滤不一致: {sample_id}")

    audit = read_json(output_dir / "reports/source_audit.json")
    rsieval = audit.get("sources", {}).get("RSIEval", {})
    if rsieval.get("local_qa_pairs") != 943:
        warnings.append(f"RSIEval 本地 QA 数量不是预期 943: {rsieval.get('local_qa_pairs')}")
    elif rsieval.get("official_readme_qa_pairs") == 936:
        warnings.append("RSIEval 本地 943 QA 与官方 README 936 条存在已记录差异")


def main() -> None:
    args = parse_args()
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    report_path = output_dir / "reports/validation_report.json"
    ensure_writable(report_path, args.overwrite, args.dry_run)
    rows = read_jsonl(output_dir / "indexes/all.jsonl")
    parents = read_jsonl(output_dir / "manifests/parent_manifest.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    validate_schema_files(errors)
    checked = rows[:args.max_samples] if args.max_samples > 0 else rows
    image_cache: dict[str, dict[str, Any]] = {}
    data_root = (output_dir / "data").resolve(strict=False)
    for row in checked:
        validate_row(row, image_cache, data_root, errors, warnings)
    validate_global_invariants(rows, parents, output_dir, errors, warnings)
    verified_clusters = read_jsonl(output_dir / "manifests/verified_perceptual_duplicates.jsonl")
    verified_cross_split_groups = sum(
        len({
            str(member.get("assigned_final_split"))
            for member in cluster.get("member_details", [])
        }) > 1
        for cluster in verified_clusters
        if cluster.get("selected_for_mode")
    )
    if verified_cross_split_groups:
        errors.append(f"verified perceptual cluster 跨 split: {verified_cross_split_groups}")
    report = {
        "builder_version": BUILDER_VERSION, "mode": args.mode,
        "num_records": len(rows), "num_parents": len(parents), "deep_checked_records": len(checked),
        "decoded_unique_images": len(image_cache),
        "materialized_files": len(read_jsonl(output_dir / "manifests/materialization_manifest.jsonl")),
        "verified_perceptual_clusters": len(verified_clusters),
        "verified_perceptual_duplicate_cross_split_groups": verified_cross_split_groups,
        "train_eligible_records": len(read_jsonl(output_dir / "indexes/train_eligible.jsonl")),
        "errors": errors, "warnings": sorted(set(warnings)),
    }
    print(f"[VALIDATE] records={len(rows)} parents={len(parents)} errors={len(errors)} warnings={len(set(warnings))}")
    if not args.dry_run:
        write_json(report_path, report)
        print(f"[VALIDATE] report={report_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
