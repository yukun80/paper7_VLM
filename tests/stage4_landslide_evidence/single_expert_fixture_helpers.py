"""单专家工作流永久测试的轻量、CPU-only 合成资产。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from fixture_helpers import no_target_output, target_output

from oa_groundrag.landslide_evidence.region_contracts import (
    ANNOTATION_QUEUE_SCHEMA,
    EVAL_MANIFEST_SCHEMA,
    REGION_MANIFEST_SCHEMA,
    REGION_RECORD_SCHEMA,
    annotation_state_template,
    empty_description_template,
)
from oa_groundrag.landslide_evidence.region_pipeline import ledger_rows, region_asset_identity
from oa_groundrag.landslide_evidence.single_expert import (
    DRAFT_MODEL_REVISION,
    MODEL_DRAFT_RUN_SCHEMA,
    MODEL_DRAFT_SCHEMA,
    VERIFIED_ANNOTATION_SCHEMA,
    VERIFICATION_STATUS,
    build_annotation_draft_messages,
    load_annotation_asset,
    load_annotation_project,
    write_draft_results,
)
from oa_groundrag.phase3.common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.evidence import render_binary_mask, render_mask_overlay


SOURCES = (
    "gdcld",
    "landslide4sense",
    "landslidebench_agent",
    "lmhld",
    "multimodal_landslide",
)
LOCATIONS = (
    "top_left", "top_center", "top_right", "middle_left", "center",
    "middle_right", "bottom_left", "bottom_center", "bottom_right",
)


def _write_assets(root: Path) -> dict[str, dict[str, str | None]]:
    width, height = 24, 20
    optical = Image.fromarray(
        np.arange(width * height * 3, dtype=np.uint16).reshape(height, width, 3).astype(np.uint8)
    )
    definitions = {
        "small": (slice(2, 3), slice(3, 4)),
        "medium": (slice(5, 9), slice(7, 12)),
        "large": (slice(4, 16), slice(5, 17)),
    }
    result: dict[str, dict[str, str | None]] = {}
    for name, (ys, xs) in definitions.items():
        mask = np.zeros((height, width), dtype=bool)
        mask[ys, xs] = True
        x0, x1 = int(xs.start), int(xs.stop)
        y0, y1 = int(ys.start), int(ys.stop)
        paths = {
            "optical_full": f"assets/optical_full/{name}.png",
            "binary_mask": f"assets/binary_mask/{name}.png",
            "context_crop": f"assets/context_crop/{name}.png",
            "audit_overlay": f"assets/audit_overlay/{name}.png",
        }
        for relative in paths.values():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
        optical.save(root / paths["optical_full"], format="PNG")
        render_binary_mask(mask).save(root / paths["binary_mask"], format="PNG")
        optical.crop((max(0, x0 - 1), max(0, y0 - 1), min(width, x1 + 1), min(height, y1 + 1))).save(
            root / paths["context_crop"], format="PNG"
        )
        render_mask_overlay(optical, mask, alpha=0.45).save(
            root / paths["audit_overlay"], format="PNG"
        )
        result[name] = paths
    empty_paths: dict[str, str | None] = {
        "optical_full": "assets/optical_full/empty.png",
        "binary_mask": "assets/binary_mask/empty.png",
        "context_crop": None,
        "audit_overlay": None,
    }
    (root / str(empty_paths["optical_full"])).parent.mkdir(parents=True, exist_ok=True)
    (root / str(empty_paths["binary_mask"])).parent.mkdir(parents=True, exist_ok=True)
    optical.save(root / str(empty_paths["optical_full"]), format="PNG")
    render_binary_mask(np.zeros((height, width), dtype=bool)).save(
        root / str(empty_paths["binary_mask"]), format="PNG"
    )
    result["empty"] = empty_paths
    return result


def _record(
    *,
    record_id: str,
    source: str,
    split: str,
    target_status: str,
    size: str,
    fragment_count: int,
    location: str,
    assets: Mapping[str, str | None],
    eval_baseline: bool,
) -> dict[str, Any]:
    area_ratio = {"small": 0.002, "medium": 0.04, "large": 0.30, "empty": 0.0}[size]
    target = target_status == "target_present"
    program_facts = {
        "image": {"width_pixels": 24, "height_pixels": 20},
        "mask": {
            "bbox_xyxy_pixel_half_open": [3, 2, 4, 3] if target else None,
            "centroid_xy_pixel": [3.0, 2.0] if target else None,
            "area_pixels": int(round(480 * area_ratio)),
            "area_ratio": area_ratio,
            "location_3x3": location if target else "not_applicable",
            "fragment_count": fragment_count if target else 0,
            "perimeter_pixels": 4.0 if target else 0.0,
            "elongation": 1.0 if target else None,
            "compactness": 1.0 if target else None,
            "crop_window_xyxy_pixel_half_open": [2, 1, 5, 4] if target else None,
        },
        "coordinate_basis": "top_left_pixel_xy_half_open",
        "mask_renderer": "binary_mask_png_l_0_255.v1",
        "difficulty_proxy": None,
        "counterfactual": (
            {"kind": "baseline_correct_mask", "evaluation_only": True}
            if eval_baseline else None
        ),
    }
    return {
        "schema_version": REGION_RECORD_SCHEMA,
        "record_id": record_id,
        "parent_id": f"parent_{record_id}",
        "parent_identity_status": "known",
        "sample_id": f"sample_{record_id}",
        "source": source,
        "split": split,
        "source_identity": {"record_sha256": sha256_text(record_id)},
        "mask_source": "oa_auxseg_benchmark_gt",
        "target_status": target_status,
        "representation_mode": "full_plus_mask_plus_crop" if target else "full_plus_mask",
        "assets": dict(assets),
        "formal_model_input_roles": (
            ["optical_full", "binary_mask", "context_crop"]
            if target else ["optical_full", "binary_mask"]
        ),
        "audit_only_roles": ["audit_overlay"] if target else [],
        "program_facts": program_facts,
        "description": empty_description_template(),
        "annotation": annotation_state_template(),
    }


def build_annotation_asset(root: Path, *, kind: str, split: str | None = None) -> Path:
    """生成 500 train-like 或 100 val-baseline 资产，不创建 counterfactual payload。"""

    root.mkdir(parents=True, exist_ok=False)
    assets = _write_assets(root)
    records: list[dict[str, Any]] = []
    if kind == "train":
        active_split = split or "train"
        for source in SOURCES:
            for index in range(100):
                no_target = source != "multimodal_landslide" and index < 25
                size = "empty" if no_target else ("small", "medium", "large")[index % 3]
                records.append(_record(
                    record_id=f"train_{source}_{index:03d}",
                    source=source,
                    split=active_split,
                    target_status="no_target" if no_target else "target_present",
                    size=size,
                    fragment_count=1 if index % 2 == 0 else 2,
                    location=LOCATIONS[index % len(LOCATIONS)],
                    assets=assets[size],
                    eval_baseline=False,
                ))
        schema = REGION_MANIFEST_SCHEMA
    elif kind == "eval":
        active_split = split or "val"
        for source in SOURCES:
            for index in range(20):
                no_target = index < 4
                size = "empty" if no_target else ("small", "medium", "large")[(index - 4) % 3]
                records.append(_record(
                    record_id=f"eval_{source}_{index:03d}",
                    source=source,
                    split=active_split,
                    target_status="no_target" if no_target else "target_present",
                    size=size,
                    fragment_count=1 if index % 2 == 0 else 2,
                    location=LOCATIONS[index % len(LOCATIONS)],
                    assets=assets[size],
                    eval_baseline=True,
                ))
        schema = EVAL_MANIFEST_SCHEMA
    else:
        raise ValueError(f"unknown kind: {kind}")
    queue = []
    for record in records:
        queue.append({
            "schema_version": ANNOTATION_QUEUE_SCHEMA,
            "record_id": record["record_id"],
            "split": record["split"],
            "target_status": record["target_status"],
            "assets": deepcopy(record["assets"]),
            "program_facts": deepcopy(record["program_facts"]),
            "asset_identity_sha256": region_asset_identity(root, record["assets"]),
        })
    atomic_write_jsonl(root / "records.jsonl", records)
    atomic_write_jsonl(root / "annotation_queue.jsonl", queue)
    manifest: dict[str, Any] = {
        "schema_version": schema,
        "records": {"path": "records.jsonl", "sha256": sha256_file(root / "records.jsonl")},
        "annotation_queue": {
            "path": "annotation_queue.jsonl",
            "sha256": sha256_file(root / "annotation_queue.jsonl"),
        },
        "record_count": len(records),
    }
    if kind == "eval":
        manifest["selection"] = {"baseline_record_ids": [row["record_id"] for row in records]}
        manifest["train_corpus"] = {"root": str(root.parent / "train")}
    relatives = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    ]
    ledger = ledger_rows(root, relatives)
    atomic_write_jsonl(root / "SHA256SUMS.jsonl", ledger)
    manifest["ledger"] = {
        "path": "SHA256SUMS.jsonl",
        "entry_count": len(ledger),
        "size_bytes": (root / "SHA256SUMS.jsonl").stat().st_size,
        "file_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
        "root_sha256": sha256_text(canonical_json(ledger)),
    }
    atomic_write_json(root / "manifest.json", manifest)
    return root


def draft_run(
    record_ids: Sequence[str],
    *,
    partition: str,
    suffix: str,
    prompt_text: str = "fixture prompt",
) -> dict[str, Any]:
    ids = list(record_ids)
    config = {
        "schema_version": "oa_groundrag.mask_grounded_region.draft_config.v1",
        "model": {
            "path": "/fixture/Qwen3-VL-8B-Instruct",
            "processor_path": "/fixture/Qwen3-VL-8B-Instruct",
            "repository": "Qwen/Qwen3-VL-8B-Instruct",
            "revision": DRAFT_MODEL_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "processor": {
            "min_pixels": 12544,
            "max_pixels": 200704,
            "max_images": 3,
            "max_input_tokens": 4096,
        },
        "generation": {
            "max_new_tokens": 768,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260804,
        },
    }
    return {
        "schema_version": MODEL_DRAFT_RUN_SCHEMA,
        "draft_run_id": f"draft_run_{suffix}",
        "config_sha256": "1" * 64,
        "config_semantic_sha256": sha256_text(canonical_json(config)),
        "config": config,
        "model_repository": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": DRAFT_MODEL_REVISION,
        "model_identity": {"fixture": True},
        "processor_identity": {"fixture": True},
        "prompt_text": prompt_text,
        "prompt_sha256": sha256_text(prompt_text),
        "generation": {
            "max_new_tokens": 768,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260804,
            "single_attempt": True,
        },
        "partition": partition,
        "record_ids": ids,
        "record_ids_sha256": sha256_text(canonical_json(ids)),
        "formal_acceptance": False,
    }


def populate_project(project_root: Path, *, edited_record_id: str | None = None) -> None:
    """为 package/export 测试批量写入一次草稿和单专家核验结果。"""

    project, loaded_assignments = load_annotation_project(project_root)
    assignments = list(loaded_assignments)
    prompt_text = (project_root / "prompt.txt").read_text(encoding="utf-8")
    context = load_annotation_asset(project["asset_root"])
    records = {row["record_id"]: row for row in context.records}
    by_partition: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        by_partition.setdefault(assignment["partition"], []).append(assignment)
    all_drafts: dict[str, dict[str, Any]] = {}
    for partition in ("calibration", "remaining", "all"):
        rows = by_partition.get(partition, [])
        if not rows:
            continue
        run = draft_run(
            [row["record_id"] for row in rows],
            partition=partition,
            suffix=partition,
            prompt_text=prompt_text,
        )
        drafts = []
        for assignment in rows:
            description = (
                no_target_output()
                if assignment["target_status"] == "no_target"
                else target_output()
            )
            value = {
                "schema_version": MODEL_DRAFT_SCHEMA,
                "draft_id": f"draft_{assignment['record_id']}",
                "draft_run_id": run["draft_run_id"],
                "record_id": assignment["record_id"],
                "asset_identity_sha256": assignment["asset_identity_sha256"],
                "messages_sha256": sha256_text(canonical_json(
                    build_annotation_draft_messages(
                        records[assignment["record_id"]],
                        asset_root=context.root,
                        prompt_text=prompt_text,
                    )
                )),
                "raw_output": canonical_json(description),
                "parse_status": "valid",
                "description": description,
                "failure": None,
            }
            drafts.append(value)
            all_drafts[assignment["record_id"]] = value
        write_draft_results(
            project_root,
            draft_run=run,
            new_drafts=drafts,
            freeze_prompt=partition in {"remaining", "all"},
        )
    for assignment in assignments:
        description = deepcopy(all_drafts[assignment["record_id"]]["description"])
        if assignment["record_id"] == edited_record_id:
            description["short_summary"] = "专家核验后保留的当前影像可见区域描述。"
        atomic_write_json(project_root / "verified" / f"{assignment['record_id']}.json", {
            "schema_version": VERIFIED_ANNOTATION_SCHEMA,
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "draft_id": all_drafts[assignment["record_id"]]["draft_id"],
            "annotator": "expert",
            "verification_status": VERIFICATION_STATUS,
            "description": description,
        })


class FakeDraftRuntime:
    model_identity = {"fixture_model": "qwen3-vl-8b"}
    processor_identity = {"fixture_processor": "qwen3-vl"}

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> str:
        text = canonical_json(messages)
        output = no_target_output() if '"target_status":"no_target"' in text else target_output()
        return canonical_json(output)
