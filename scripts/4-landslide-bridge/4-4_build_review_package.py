#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 4-4：生成双人专家审核包，不生成或推断专家标签。

用途：为 Pilot region 生成 mask/overlay/多模态面板及两份独立审核模板。
推荐运行命令：python scripts/4-landslide-bridge/4-4_build_review_package.py --mode small --overwrite
主要输入：candidate_all、review_selection 与 Landslide V2 preview。
主要输出：review_package、review templates、manifest 和 pending gate 模板。
写入行为：只写 Bridge benchmark；模板中的 decision 为空；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from landslide_bridge_common import (
    BUILDER_VERSION,
    atomic_write_text,
    binary_mask,
    bridge_dir,
    ensure_writable,
    evaluation_gate_scientific_template,
    preview_image,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    to_project_ref,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Landslide Bridge 专家审核包")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/landslide_bridge_v1.yaml")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _placeholder(width: int, height: int, text: str) -> Image.Image:
    image = Image.new("RGB", (width, height), "#222222")
    ImageDraw.Draw(image).text((12, 12), text, fill="white")
    return image


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.convert("RGB")
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#161616")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def _label(image: Image.Image, title: str) -> Image.Image:
    header = 28
    canvas = Image.new("RGB", (image.width, image.height + header), "#111111")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((8, 7), title, fill="white")
    return canvas


def _region_views(record: dict[str, Any], visual: Image.Image) -> tuple[Image.Image, Image.Image]:
    if not record.get("region_mask"):
        blank = Image.new("RGB", visual.size, "black")
        return blank, visual.copy()
    mask = binary_mask(str(record["region_mask"]["path"]))
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize(visual.size, Image.Resampling.NEAREST)
    mask_rgb = Image.merge("RGB", (mask_image, mask_image, mask_image))
    overlay = visual.convert("RGBA")
    red = Image.new("RGBA", visual.size, (255, 48, 48, 0))
    red.putalpha(mask_image.point(lambda value: 125 if value > 0 else 0))
    overlay.alpha_composite(red)
    return mask_rgb, overlay.convert("RGB")


def build_panel(record: dict[str, Any], destination: Path) -> None:
    paths = record.get("visual_ref", {}).get("preview_paths", {})
    visual_ref = paths.get("visual")
    modalities_ref = paths.get("modalities")
    visual = preview_image(str(visual_ref), 640) if visual_ref else _placeholder(512, 512, "visual unavailable")
    modalities = (
        preview_image(str(modalities_ref), 1200)
        if modalities_ref and resolve_project_path(str(modalities_ref)).is_file()
        else _placeholder(1024, 320, "multimodal preview unavailable")
    )
    mask, overlay = _region_views(record, visual)
    cell_width, cell_height = 400, 400
    top = Image.new("RGB", (cell_width * 3, cell_height + 28), "#111111")
    for index, (title, image) in enumerate((
        ("REFERENCE", visual), ("REGION MASK", mask), ("REGION OVERLAY", overlay),
    )):
        top.paste(_label(_fit(image, cell_width, cell_height), title), (index * cell_width, 0))
    lower = _label(_fit(modalities, cell_width * 3, 420), "MULTISOURCE EVIDENCE")
    canvas = Image.new("RGB", (cell_width * 3, top.height + lower.height), "#111111")
    canvas.paste(top, (0, 0))
    canvas.paste(lower, (0, top.height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    canvas.save(temporary, format="PNG", optimize=True)
    temporary.replace(destination)


def review_template(selection: dict[str, Any], record: dict[str, Any], reviewer_id: str, panel_ref: str) -> dict[str, Any]:
    return {
        "review_item_id": selection["review_item_id"],
        "bridge_record_id": record["bridge_record_id"],
        "parent_sample_id": record["parent_sample_id"],
        "split": record["split"],
        "reviewer_id": reviewer_id,
        "panel_path": panel_ref,
        "target_status": record["target_status"],
        "region_source": record["region_source"],
        "candidate_structured_targets": record["candidate"]["structured_output"],
        "candidate_summary": record["candidate"]["summary"],
        "decision": "",
        "corrected_structured_targets": None,
        "revised_summary": "",
        "notes": "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "review_item_id", "bridge_record_id", "parent_sample_id", "split", "reviewer_id",
        "panel_path", "target_status", "region_source", "candidate_structured_targets",
        "candidate_summary", "decision", "corrected_structured_targets", "revised_summary", "notes",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        for field in ("candidate_structured_targets", "corrected_structured_targets"):
            serialized[field] = (
                "" if serialized[field] is None else
                json.dumps(serialized[field], ensure_ascii=False, sort_keys=True)
            )
        writer.writerow(serialized)
    atomic_write_text(path, buffer.getvalue())


def main() -> None:
    args = parse_args()
    output_dir = bridge_dir(args.mode, args.output_dir)
    records = {row["bridge_record_id"]: row for row in read_jsonl(output_dir / "indexes/candidate_all.jsonl")}
    selection = read_jsonl(output_dir / "manifests/review_selection.jsonl")
    if args.max_samples > 0:
        selection = selection[:args.max_samples]

    manifest_path = output_dir / "manifests/review_package_manifest.jsonl"
    report_path = output_dir / "reports/review_package_report.json"
    gate_path = output_dir / "manifests/evaluation_gate_manifest.template.json"
    templates: dict[str, list[dict[str, Any]]] = {"reviewer_1": [], "reviewer_2": []}
    for path in (manifest_path, report_path, gate_path):
        ensure_writable(path, args.overwrite, args.dry_run)
    for reviewer_id in templates:
        for suffix in ("jsonl", "csv"):
            ensure_writable(
                output_dir / f"review_package/{reviewer_id}_template.{suffix}",
                args.overwrite, args.dry_run,
            )

    manifest: list[dict[str, Any]] = []
    for item in selection:
        record = records.get(item["bridge_record_id"])
        if record is None:
            raise KeyError(f"review selection 缺少 candidate record: {item['bridge_record_id']}")
        panel_path = output_dir / "review_package/panels" / record["split"] / f"{item['review_item_id']}.png"
        panel_ref = to_project_ref(panel_path)
        if not args.dry_run:
            if panel_path.exists() and not args.overwrite:
                raise FileExistsError(f"panel 已存在，请使用 --overwrite: {panel_path}")
            build_panel(record, panel_path)
        manifest.append({
            **item,
            "panel_path": panel_ref,
            "candidate_summary": record["candidate"]["summary"],
            "candidate_is_expert_truth": False,
        })
        for reviewer_id in templates:
            templates[reviewer_id].append(review_template(item, record, reviewer_id, panel_ref))

    gate_template = {
        "protocol": "landslide_bridge_evaluation_gate_v2",
        "builder_version": BUILDER_VERSION,
        "status": "pending_pilot_review",
        "frozen": False,
        "pilot_parent_manifest": to_project_ref(output_dir / "manifests/pilot_parent_manifest.jsonl"),
        "bindings": {
            "pilot_parent_manifest_sha256": sha256_file(
                output_dir / "manifests/pilot_parent_manifest.jsonl"
            ),
            "review_selection_sha256": sha256_file(
                output_dir / "manifests/review_selection.jsonl"
            ),
            "candidate_index_sha256": sha256_file(
                output_dir / "indexes/candidate_all.jsonl"
            ),
        },
        "thresholds": {
            "no_target_rejection": None,
            "unsupported_claim_rate": None,
            "unavailable_unsupported_claim_rate": None,
            "unsupported_claim_rate_noninferiority": None,
            "expert_fact_score": None,
            "target_status_macro_f1": None,
            "present_recall": None,
            "absent_recall": None,
            "false_description_rate": None,
            "false_rejection_rate": None,
        },
        "scientific_protocol": evaluation_gate_scientific_template(),
        "note": "Thresholds must be filled and frozen only after completed expert review and Pilot analysis.",
    }
    report = {
        "builder_version": BUILDER_VERSION,
        "review_items": len(manifest),
        "parents": len({row["parent_sample_id"] for row in manifest}),
        "by_split": dict(sorted(Counter(row["split"] for row in manifest).items())),
        "reviewers": sorted(templates),
        "expert_labels_created": 0,
        "status": "awaiting_expert_review",
        "errors": [],
    }
    print(f"[BRIDGE:REVIEW] items={len(manifest)} reviewers=2 expert_labels=0")
    if not args.dry_run:
        write_jsonl(manifest_path, manifest)
        for reviewer_id, rows in templates.items():
            write_jsonl(output_dir / f"review_package/{reviewer_id}_template.jsonl", rows)
            _write_csv(output_dir / f"review_package/{reviewer_id}_template.csv", rows)
        write_json(gate_path, gate_template)
        write_json(report_path, report)


if __name__ == "__main__":
    main()
