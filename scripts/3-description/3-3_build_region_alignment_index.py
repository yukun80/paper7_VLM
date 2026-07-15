#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-3：构建 DIOR-RSVG 双向区域对齐源索引。

用途：把多轮 box->phrase 与 phrase->box 对齐成稳定 region pair，并生成
region_referring_expression 和 region_grounding 两种候选区域任务视图。
推荐运行命令：python scripts/3-description/3-3_build_region_alignment_index.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：datasets/MMRS-1M/json/RSVG/rsvg_trainval.json 与 DIOR-RSVG 图像。
主要输出：indexes/region_alignment_source.jsonl、parent manifest 和构建报告。
写入行为：只写指向原始 datasets 的 source index；正式图片复制由 3-5 执行。
所属流程：Description Benchmark M1，抽样与 split 在 3-4 执行。
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from typing import Any

from description_common import (
    BUILDER_VERSION, MMRS_ROOT, InvalidBoundingBoxError, answer_record, base_record, bbox_pixel_half_open,
    description_dir_for_mode, ensure_writable, iter_turn_pairs, mmrs_data_path,
    normalize_phrase, parse_bbox, probe_image, read_json, single_image_visual_ref,
    stable_id, to_project_ref, write_json, write_jsonl,
)


BOX_TO_PHRASE = "box_to_phrase"
PHRASE_TO_REGION = "phrase_to_candidate_region"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 DIOR-RSVG region alignment source index")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0, help="smoke 时最多解析的 parent 图像数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def phrase_from_grounding_prompt(prompt: str) -> str:
    match = re.search(
        r"described\s+as\s*:\s*(.*?)\s+in\s+this\s+remote\s+sensing\s+image\s*$",
        prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"无法从 grounding prompt 提取 phrase: {prompt!r}")
    phrase = " ".join(match.group(1).strip().split())
    if not phrase:
        raise ValueError("grounding phrase 为空")
    return phrase


def pair_key(bbox: tuple[float, float, float, float], phrase: str) -> tuple[tuple[float, ...], str]:
    return tuple(round(value, 6) for value in bbox), normalize_phrase(phrase)


def modifiers(phrase: str) -> dict[str, list[str]]:
    words = set(re.findall(r"[a-z]+", phrase.casefold()))
    size = sorted(words & {"tiny", "small", "large", "huge", "long", "short"})
    position = sorted(words & {"top", "bottom", "left", "right", "middle", "center", "central", "upper", "lower"})
    return {"size": size, "position": position}


def region_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 8:
        return "4-8"
    return "9+"


def parse_parent_records(
    image_ref: str, records: list[tuple[int, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    paired: dict[tuple[tuple[float, ...], str], dict[str, Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for record_index, source in records:
        try:
            turns = list(iter_turn_pairs(source.get("conversations", [])))
        except ValueError as exc:
            errors.append(f"record={record_index}: {exc}")
            continue
        for turn_index, prompt, response in turns:
            try:
                if "short description" in prompt.casefold():
                    bbox = parse_bbox(prompt)
                    phrase = " ".join(response.strip().split())
                    direction = BOX_TO_PHRASE
                elif "bounding box coordinate" in prompt.casefold():
                    bbox = parse_bbox(response)
                    phrase = phrase_from_grounding_prompt(prompt)
                    direction = PHRASE_TO_REGION
                else:
                    raise ValueError(f"未知 DIOR task prompt: {prompt[:100]!r}")
                if not phrase:
                    raise ValueError("region phrase 为空")
            except InvalidBoundingBoxError as exc:
                warnings.append(f"record={record_index} turn={turn_index}: excluded_invalid_source_bbox: {exc}")
                continue
            except (ValueError, SyntaxError) as exc:
                errors.append(f"record={record_index} turn={turn_index}: {exc}")
                continue
            key = pair_key(bbox, phrase)
            entry = paired.setdefault(key, {"bbox": bbox, "phrase": phrase, "directions": defaultdict(list)})
            entry["directions"][direction].append({
                "record_index": record_index, "turn_index": turn_index,
                "source_prompt": prompt, "source_response": response,
            })

    regions: list[dict[str, Any]] = []
    for key, value in paired.items():
        forward = value["directions"].get(BOX_TO_PHRASE, [])
        reverse = value["directions"].get(PHRASE_TO_REGION, [])
        if len(forward) != 1 or len(reverse) != 1:
            errors.append(
                f"image={image_ref} pair={key}: 双向记录数量应为 1/1，当前 {len(forward)}/{len(reverse)}"
            )
            continue
        regions.append({
            "bbox": value["bbox"], "phrase": value["phrase"],
            "forward": forward[0], "reverse": reverse[0],
        })
    regions.sort(key=lambda item: (item["bbox"], normalize_phrase(item["phrase"])))
    return regions, errors, warnings


def build_rows(max_parents: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    annotation_path = MMRS_ROOT / "json/RSVG/rsvg_trainval.json"
    source_rows = read_json(annotation_path)
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(source_rows):
        grouped[str(row["image"])].append((index, row))
    image_refs = sorted(grouped)
    if max_parents > 0:
        image_refs = image_refs[:max_parents]

    records: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for image_ref in image_refs:
        image_path = mmrs_data_path(image_ref)
        meta = probe_image(image_path)
        parent_id = stable_id("dior_rsvg", image_ref)
        regions, pair_errors, pair_warnings = parse_parent_records(image_ref, grouped[image_ref])
        errors.extend(pair_errors)
        warnings.extend(pair_warnings)
        phrase_to_boxes: dict[str, set[tuple[float, ...]]] = defaultdict(set)
        box_to_phrases: dict[tuple[float, ...], set[str]] = defaultdict(set)
        for region in regions:
            phrase_to_boxes[normalize_phrase(region["phrase"])].add(tuple(region["bbox"]))
            box_to_phrases[tuple(region["bbox"])].add(normalize_phrase(region["phrase"]))

        visual_ref = single_image_visual_ref(image_path, "DIOR-RSVG", meta)
        for region_index, region in enumerate(regions):
            bbox = tuple(float(value) for value in region["bbox"])
            phrase = str(region["phrase"])
            pair_id = stable_id("dior_region", parent_id, *bbox, normalize_phrase(phrase))
            flags: list[str] = []
            if len(phrase_to_boxes[normalize_phrase(phrase)]) > 1:
                flags.append("ambiguous_phrase_with_multiple_boxes")
            if len(box_to_phrases[bbox]) > 1:
                flags.append("duplicate_box_with_multiple_phrases")
            geometry = {
                "type": "box", "mask_path": None,
                "bbox_xyxy_normalized": list(bbox),
                "bbox_xyxy_pixel_half_open": bbox_pixel_half_open(bbox, meta["width"], meta["height"]),
                "coordinate_space": "original_image",
            }
            phrase_answer, _ = answer_record(phrase, "human_referring_expression", "DIOR-RSVG", alignment=True)
            common_provenance = {
                "annotation_path": to_project_ref(annotation_path),
                "source_image_path": to_project_ref(image_path),
                "original_record_id": f"{region['forward']['record_index']}:{region['forward']['turn_index']}|{region['reverse']['record_index']}:{region['reverse']['turn_index']}",
                "source_image_ref": image_ref, "source_scene_group": None,
                "source_scene_group_status": "unavailable", "source_split": None,
                "license_status": "source_specific_review_required", "license_source": "DIOR-RSVG",
                "annotation_origin": "human_referring_expression",
            }
            ref_row = base_record(
                sample_id=f"{pair_id}__box_to_phrase", parent_sample_id=parent_id, region_pair_id=pair_id,
                source_dataset="DIOR-RSVG", task_family="region_referring_expression",
                visual_ref=visual_ref, region_geometry=geometry, target_status="present",
                region_source="source_box", instruction="Describe the annotated region with a short referring expression.",
                answer_type="referring_expression", answers=[phrase_answer], provenance={
                    **common_provenance,
                    "source_instruction": region["forward"]["source_prompt"],
                    "source_answer": region["forward"]["source_response"],
                }, quality_flags=flags,
            )
            ref_row["alignment_metadata"] = {"direction": BOX_TO_PHRASE, "modifiers": modifiers(phrase), "region_index": region_index}
            candidate_answer, _ = answer_record(pair_id, "derived_candidate_region_id", "DIOR-RSVG", alignment=True)
            ground_row = base_record(
                sample_id=f"{pair_id}__phrase_to_region", parent_sample_id=parent_id, region_pair_id=pair_id,
                source_dataset="DIOR-RSVG", task_family="region_grounding",
                visual_ref=visual_ref, region_geometry=geometry, target_status="present",
                region_source="source_box", instruction=f"Select the annotated candidate region matching: {phrase}",
                answer_type="candidate_region_id", answers=[candidate_answer], provenance={
                    **common_provenance,
                    "source_instruction": region["reverse"]["source_prompt"],
                    "source_answer": region["reverse"]["source_response"],
                }, quality_flags=flags,
            )
            ground_row["alignment_metadata"] = {"direction": PHRASE_TO_REGION, "modifiers": modifiers(phrase), "region_index": region_index}
            records.extend((ref_row, ground_row))

        parents.append({
            "parent_sample_id": parent_id, "source_dataset": "DIOR-RSVG",
            "source_image_path": visual_ref["path"],
            "width": meta["width"], "height": meta["height"], "sha256": meta["sha256"],
            "dhash64": meta["dhash64"], "source_scene_group": None,
            "source_scene_group_status": "unavailable", "source_split": None,
            "task_count": len(regions) * 2, "region_count": len(regions),
            "stratum": {"source_group": "DIOR-RSVG", "region_count_bin": region_count_bin(len(regions))},
        })
    return records, parents, errors, warnings


def main() -> None:
    args = parse_args()
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    index_path = output_dir / "indexes/region_alignment_source.jsonl"
    parent_path = output_dir / "manifests/region_alignment_parents.jsonl"
    report_path = output_dir / "reports/region_alignment_build.json"
    for path in (index_path, parent_path, report_path):
        ensure_writable(path, args.overwrite, args.dry_run)
    rows, parents, errors, warnings = build_rows(args.max_samples)
    counts = Counter(row["task_family"] for row in rows)
    flags = Counter(flag for row in rows for flag in row["quality_flags"])
    report = {
        "builder_version": BUILDER_VERSION, "mode": args.mode,
        "num_parents": len(parents), "num_region_pairs": len(rows) // 2, "num_task_views": len(rows),
        "task_counts": dict(sorted(counts.items())), "quality_flag_counts": dict(sorted(flags.items())),
        "excluded_invalid_source_bbox_task_turns": len(warnings),
        "excluded_invalid_source_bbox_region_pairs": len(warnings) // 2,
        "warnings": warnings, "errors": errors,
    }
    print(
        f"[REGION] parents={len(parents)} pairs={len(rows) // 2} "
        f"excluded_invalid_pairs={len(warnings) // 2} errors={len(errors)}"
    )
    if errors:
        raise ValueError(f"DIOR region pair 解析失败，首条错误: {errors[0]}")
    if not args.dry_run:
        write_jsonl(index_path, rows)
        write_jsonl(parent_path, parents)
        write_json(report_path, report)
        print(f"[REGION] index={index_path}")


if __name__ == "__main__":
    main()
