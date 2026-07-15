#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-5：复制已选 parent 图片并发布模型侧最终索引。

用途：将 3-4 冻结的 datasets 源引用物化到 benchmark/data，并只在全部图片
校验成功后生成 all/train/dev/test 与 component final indexes。
推荐运行命令：python scripts/3-description/3-5_materialize_description_images.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：indexes/selected_source_all.jsonl、selected parent source manifest。
主要输出：data/、最终 indexes、parent/materialization manifests 和物化报告。
写入行为：复制入选图片，不修改原始 datasets；--dry-run 只统计计划。
所属流程：Description Benchmark M1 自包含数据发布阶段。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from description_common import (
    BUILDER_VERSION,
    description_dir_for_mode,
    ensure_writable,
    read_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    sha256_jsonl_rows,
    source_slug,
    to_project_ref,
    write_json,
    write_jsonl,
)


MATERIALIZATION_PROTOCOL = "qpsalm_description_materialization_v2"
COPY_CHUNK_SIZE = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="物化 Description Benchmark 图片并发布最终索引")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int, default=8, help="并行复制 worker 数")
    parser.add_argument("--max-samples", type=int, default=0, help="协议兼容参数；抽样已在 3-4 完成")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def destination_for_parent(output_dir: Path, parent: dict[str, Any]) -> Path:
    source = resolve_project_path(parent["source_image_path"])
    suffix = source.suffix.casefold()
    if not suffix or not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        raise ValueError(f"源图片扩展名不适合物化: {source}")
    return (
        output_dir
        / "data"
        / str(parent["split"])
        / source_slug(str(parent["source_dataset"]))
        / f"{parent['parent_sample_id']}{suffix}"
    )


def _decode_materialized(path: Path, expected_width: int, expected_height: int) -> None:
    with Image.open(path) as image:
        image.load()
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"物化图片必须是 RGB/RGBA: mode={image.mode} path={path}")
        if image.size != (expected_width, expected_height):
            raise ValueError(
                f"物化图片尺寸不一致: expected=({expected_width},{expected_height}) "
                f"actual={image.size} path={path}"
            )


def copy_parent_image(output_dir: Path, parent: dict[str, Any]) -> dict[str, Any]:
    source_ref = str(parent["source_image_path"])
    source = resolve_project_path(source_ref)
    destination = destination_for_parent(output_dir, parent)
    expected_hash = str(parent["sha256"])
    if not source.is_file():
        raise FileNotFoundError(f"源图片不存在: {source_ref} -> {source}")
    if destination.is_symlink():
        raise ValueError(f"物化目标不得是符号链接: {destination}")

    status = "copied"
    if destination.is_file() and sha256_file(destination) == expected_hash:
        status = "reused"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            digest = hashlib.sha256()
            copied_bytes = 0
            with source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as target_handle:
                temp_path = Path(target_handle.name)
                while chunk := source_handle.read(COPY_CHUNK_SIZE):
                    target_handle.write(chunk)
                    digest.update(chunk)
                    copied_bytes += len(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"源图片 hash 与索引不一致: parent={parent['parent_sample_id']} "
                    f"expected={expected_hash} actual={actual_hash}"
                )
            if copied_bytes != source.stat().st_size:
                raise ValueError(f"图片复制字节数不一致: {source}")
            os.replace(temp_path, destination)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    if sha256_file(destination) != expected_hash:
        raise ValueError(f"物化目标 hash 校验失败: {destination}")
    _decode_materialized(destination, int(parent["width"]), int(parent["height"]))
    return {
        "parent_sample_id": str(parent["parent_sample_id"]),
        "canonical_parent_id": str(parent.get("canonical_parent_id") or parent["parent_sample_id"]),
        "merged_parent_ids": list(parent.get("merged_parent_ids") or [parent["parent_sample_id"]]),
        "perceptual_cluster_id": parent.get("perceptual_cluster_id"),
        "source_dataset": str(parent["source_dataset"]),
        "source_datasets": list(parent.get("source_datasets") or [parent["source_dataset"]]),
        "split": str(parent["split"]),
        "source_image_path": source_ref,
        "image_path": to_project_ref(destination),
        "storage_mode": "materialized_copy",
        "sha256": expected_hash,
        "file_size_bytes": destination.stat().st_size,
        "source_suffix": source.suffix,
        "materialization_status": status,
    }


def validate_source_contract(
    rows: list[dict[str, Any]], parents: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    parent_by_id: dict[str, dict[str, Any]] = {}
    for parent in parents:
        parent_id = str(parent["parent_sample_id"])
        if parent_id in parent_by_id:
            raise ValueError(f"selected parent 重复: {parent_id}")
        if not str(parent.get("source_image_path", "")).startswith("datasets/"):
            raise ValueError(f"selected parent 源路径必须位于 datasets: {parent_id}")
        parent_by_id[parent_id] = parent

    row_paths: dict[str, set[str]] = {parent_id: set() for parent_id in parent_by_id}
    for row in rows:
        parent_id = str(row["parent_sample_id"])
        if parent_id not in parent_by_id:
            raise ValueError(f"record 引用了未选择 parent: {row['sample_id']} -> {parent_id}")
        source_path = str(row.get("provenance", {}).get("source_image_path") or row["visual_ref"]["path"])
        row_paths[parent_id].add(source_path)
        if row["visual_ref"]["path"] != source_path:
            raise ValueError(f"source record visual_ref 与 provenance 路径不一致: {row['sample_id']}")
    for parent_id, paths in row_paths.items():
        expected = str(parent_by_id[parent_id]["source_image_path"])
        if paths != {expected}:
            raise ValueError(f"同一 parent 源路径不唯一: {parent_id} -> {sorted(paths)}")
    return parent_by_id


def rewrite_final_records(
    rows: list[dict[str, Any]], materialized_by_parent: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    for source_row in rows:
        row = copy.deepcopy(source_row)
        materialized = materialized_by_parent[str(row["parent_sample_id"])]
        row["visual_ref"]["path"] = materialized["image_path"]
        row["visual_ref"]["storage_mode"] = "materialized_copy"
        row["provenance"]["source_image_path"] = materialized["source_image_path"]
        final_rows.append(row)
    final_rows.sort(
        key=lambda row: (row["split"], row["source_dataset"], row["parent_sample_id"], row["sample_id"])
    )
    return final_rows


def training_eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成训练入口，保留审计索引但排除零权重 caption。"""
    eligible: list[dict[str, Any]] = []
    for source_row in rows:
        if source_row.get("split") != "train":
            continue
        row = copy.deepcopy(source_row)
        row["answers"] = [
            answer
            for answer in row.get("answers", [])
            if float(answer.get("caption_quality_weight", 0.0)) > 0.0
        ]
        if row["answers"]:
            eligible.append(row)
    return eligible


def remove_stale_files(data_root: Path, expected_paths: set[Path]) -> tuple[int, int]:
    if not data_root.exists():
        return 0, 0
    removed_count = 0
    removed_bytes = 0
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        resolved = path.resolve(strict=False)
        if resolved in expected_paths and not path.name.endswith(".part"):
            continue
        if path.is_file() and not path.is_symlink():
            removed_bytes += path.stat().st_size
        path.unlink()
        removed_count += 1
    directories = sorted((path for path in data_root.rglob("*") if path.is_dir()), reverse=True)
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed_count, removed_bytes


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须大于 0")
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    selected_path = output_dir / "indexes/selected_source_all.jsonl"
    selected_parent_path = output_dir / "manifests/selected_parent_source_manifest.jsonl"
    state_path = output_dir / "manifests/materialization_state.json"
    rows = read_jsonl(selected_path)
    parents = read_jsonl(selected_parent_path)
    parent_by_id = validate_source_contract(rows, parents)

    state = read_json(state_path)
    if state.get("status") != "pending" or state.get("protocol") != MATERIALIZATION_PROTOCOL:
        raise ValueError("materialization_state 必须由当前 3-4 阶段写为 pending")
    if state.get("selected_source_index_hash") != sha256_jsonl_rows(rows):
        raise ValueError("selected source index hash 与 materialization state 不一致")
    if state.get("selected_parent_manifest_hash") != sha256_jsonl_rows(parents):
        raise ValueError("selected parent manifest hash 与 materialization state 不一致")

    destinations = {parent_id: destination_for_parent(output_dir, parent) for parent_id, parent in parent_by_id.items()}
    if len({path.resolve(strict=False) for path in destinations.values()}) != len(destinations):
        raise ValueError("不同 parent 产生了相同物化目标路径")
    source_bytes = 0
    missing: list[str] = []
    for parent in parents:
        source = resolve_project_path(parent["source_image_path"])
        if source.is_file():
            source_bytes += source.stat().st_size
        else:
            missing.append(str(parent["source_image_path"]))
    print(
        f"[MATERIALIZE] mode={args.mode} parents={len(parents)} records={len(rows)} "
        f"bytes={source_bytes} workers={args.workers} dry_run={args.dry_run}"
    )
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 个源图片，首个为 {missing[0]}")
    if args.dry_run:
        return

    final_paths = {
        output_dir / "indexes/global_caption_all.jsonl",
        output_dir / "indexes/region_alignment_all.jsonl",
        output_dir / "indexes/all.jsonl",
        output_dir / "indexes/train.jsonl",
        output_dir / "indexes/dev.jsonl",
        output_dir / "indexes/test.jsonl",
        output_dir / "indexes/train_eligible.jsonl",
        output_dir / "manifests/parent_manifest.jsonl",
        output_dir / "manifests/materialization_manifest.jsonl",
        output_dir / "reports/materialization_report.json",
    }
    for path in final_paths:
        ensure_writable(path, args.overwrite, args.dry_run)

    materialized: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="description-copy") as pool:
        futures = {pool.submit(copy_parent_image, output_dir, parent): parent for parent in parents}
        for future in as_completed(futures):
            materialized.append(future.result())
    materialized.sort(key=lambda row: row["parent_sample_id"])
    materialized_by_parent = {row["parent_sample_id"]: row for row in materialized}

    final_rows = rewrite_final_records(rows, materialized_by_parent)
    split_rows = {split: [row for row in final_rows if row["split"] == split] for split in ("train", "dev", "test")}
    train_eligible = training_eligible_rows(final_rows)
    global_rows = [row for row in final_rows if row["component_benchmark"] == "rs_global_caption_v1"]
    region_rows = [row for row in final_rows if row["component_benchmark"] == "rs_region_alignment_v1"]
    final_parents = []
    for parent in parents:
        item = materialized_by_parent[str(parent["parent_sample_id"])]
        final_parents.append({
            **parent,
            "image_path": item["image_path"],
            "source_image_path": item["source_image_path"],
            "storage_mode": item["storage_mode"],
            "file_size_bytes": item["file_size_bytes"],
        })
    final_parents.sort(key=lambda row: row["parent_sample_id"])

    expected_paths = {path.resolve(strict=False) for path in destinations.values()}
    stale_count, stale_bytes = remove_stale_files(output_dir / "data", expected_paths)
    outputs = {
        output_dir / "indexes/global_caption_all.jsonl": global_rows,
        output_dir / "indexes/region_alignment_all.jsonl": region_rows,
        output_dir / "indexes/all.jsonl": final_rows,
        output_dir / "indexes/train.jsonl": split_rows["train"],
        output_dir / "indexes/dev.jsonl": split_rows["dev"],
        output_dir / "indexes/test.jsonl": split_rows["test"],
        output_dir / "indexes/train_eligible.jsonl": train_eligible,
        output_dir / "manifests/parent_manifest.jsonl": final_parents,
        output_dir / "manifests/materialization_manifest.jsonl": materialized,
    }
    report = {
        "protocol": MATERIALIZATION_PROTOCOL,
        "builder_version": BUILDER_VERSION,
        "mode": args.mode,
        "num_parents": len(final_parents),
        "num_records": len(final_rows),
        "num_train_eligible_records": len(train_eligible),
        "num_zero_weight_answers_excluded_from_training": sum(
            float(answer.get("caption_quality_weight", 0.0)) <= 0.0
            for row in final_rows if row.get("split") == "train"
            for answer in row.get("answers", [])
        ),
        "num_files": len(materialized),
        "total_bytes": sum(int(row["file_size_bytes"]) for row in materialized),
        "status_counts": dict(sorted(Counter(row["materialization_status"] for row in materialized).items())),
        "files_by_split": dict(sorted(Counter(row["split"] for row in materialized).items())),
        "bytes_by_split": {
            split: sum(int(row["file_size_bytes"]) for row in materialized if row["split"] == split)
            for split in sorted({row["split"] for row in materialized})
        },
        "files_by_source": dict(sorted(Counter(row["source_dataset"] for row in materialized).items())),
        "stale_files_removed": stale_count,
        "stale_bytes_removed": stale_bytes,
        "errors": [],
    }
    complete_state = {
        **state,
        "status": "complete",
        "materialization_manifest_hash": sha256_jsonl_rows(materialized),
        "final_index_hash": sha256_jsonl_rows(final_rows),
        "num_files": len(materialized),
        "total_bytes": report["total_bytes"],
    }
    for path, payload in outputs.items():
        write_jsonl(path, payload)
    write_json(output_dir / "reports/materialization_report.json", report)
    write_json(state_path, complete_state)
    print(
        f"[MATERIALIZE] complete files={len(materialized)} copied={report['status_counts'].get('copied', 0)} "
        f"reused={report['status_counts'].get('reused', 0)} stale_removed={stale_count}"
    )


if __name__ == "__main__":
    main()
