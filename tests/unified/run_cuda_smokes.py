#!/usr/bin/env python3
"""六任务真实 CUDA engineering smoke；不被 unittest discover 自动执行。"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.landslide_evidence.pipeline import render_optical
from oa_groundrag.phase3.common import atomic_write_json, atomic_write_text, read_json
from scripts.phase1_benchmark_build.benchmark_common import BenchmarkDataset


CONFIG = REPO_ROOT / "configs/unified/inference_v1.yaml"
CLI = REPO_ROOT / "scripts/unified/run_oa_groundrag.py"
BENCHMARK = REPO_ROOT.parent / "benchmark/oa_auxseg_hdf5_v1/full"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Runtime 六任务真实 CUDA smoke")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/tmp") / (
            "oa_groundrag_unified_cuda_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=(
            "vlm_only",
            "segment_only",
            "region_understanding",
            "segment_and_understand",
            "knowledge_qa",
            "region_interpretation",
        ),
        help="只运行指定 smoke；可重复给出",
    )
    return parser


def _request(
    *,
    request_id: str,
    task: str,
    instruction: str | None,
    images: list[str] | None = None,
    user_mask: str | None = None,
    spatial: dict[str, str] | None = None,
    region_source: str = "NONE",
) -> dict[str, Any]:
    return {
        "schema_version": "oa_groundrag.unified_request.v1",
        "request_id": request_id,
        "task": task,
        "instruction": instruction,
        "images": images or [],
        "user_mask": user_mask,
        "spatial_input": spatial,
        "region_source": region_source,
        "candidate_region_id": None,
        "auxiliary_views": [],
        "include_audit": True,
    }


def _prepare(root: Path) -> tuple[str, Path, Path]:
    assets = root / "assets"
    requests = root / "requests"
    logs = root / "logs"
    assets.mkdir(parents=True)
    requests.mkdir()
    logs.mkdir()
    dataset = BenchmarkDataset(
        BENCHMARK,
        split="train",
        auxiliary_policy="all",
        normalization="none",
    )
    row = dataset.rows[0]
    sample_id = str(row["sample_id"])
    image = render_optical(dataset[0])
    image_path = assets / "benchmark_train_optical.png"
    image.save(image_path, format="PNG")
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    x0, x1 = image.width // 4, 3 * image.width // 4
    y0, y1 = image.height // 4, 3 * image.height // 4
    mask[y0:y1, x0:x1] = 255
    mask_path = assets / "synthetic_user_mask.png"
    Image.fromarray(mask).save(mask_path, format="PNG")
    del dataset
    gc.collect()
    return sample_id, image_path, mask_path


def _requests(sample_id: str, image: Path, mask: Path) -> list[tuple[str, dict[str, Any]]]:
    spatial = {"kind": "benchmark_sample", "split": "train", "sample_id": sample_id}
    return [
        ("vlm_only", _request(
            request_id="cuda-vlm-only",
            task="VLM_ONLY",
            instruction="Describe this remote sensing image.",
            images=[str(image)],
        )),
        ("segment_only", _request(
            request_id="cuda-segment-only",
            task="SEGMENT_ONLY",
            instruction=None,
            spatial=spatial,
        )),
        ("region_understanding", _request(
            request_id="cuda-region-understanding",
            task="REGION_UNDERSTANDING",
            instruction="Describe this masked region using only visible evidence.",
            images=[str(image)],
            user_mask=str(mask),
            region_source="USER_MASK",
        )),
        ("segment_and_understand", _request(
            request_id="cuda-segment-and-understand",
            task="SEGMENT_AND_UNDERSTAND",
            instruction="Describe the region selected by the predicted mask using only visible evidence.",
            spatial=spatial,
            region_source="OA_AUXSEG_GLOBAL",
        )),
        ("knowledge_qa", _request(
            request_id="cuda-knowledge-qa",
            task="KNOWLEDGE_QA",
            instruction="Why can InSAR LOS measurements not directly determine full 3-D displacement?",
        )),
        ("region_interpretation", _request(
            request_id="cuda-region-interpretation",
            task="REGION_INTERPRETATION",
            instruction=(
                "Interpret the candidate region professionally, including confounders, limitations, "
                "and recommended verification, without confirming a landslide."
            ),
            spatial=spatial,
            region_source="OA_AUXSEG_CANDIDATE",
        )),
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _validate_output(name: str, root: Path, expected_task: str) -> dict[str, Any]:
    response = read_json(root / "response.json")
    manifest = read_json(root / "manifest.json")
    errors: list[str] = []
    if response.get("schema_version") != "oa_groundrag.unified_response.v1":
        errors.append("response schema mismatch")
    if response.get("task") != expected_task:
        errors.append("task mismatch")
    if response.get("status") != "SUCCESS":
        errors.append("status is not SUCCESS")
    if manifest.get("scientific_acceptance") is not False:
        errors.append("scientific_acceptance boundary missing")
    if manifest.get("sealed_test_accessed") is not False:
        errors.append("sealed test boundary missing")
    trace = response.get("trace", [])
    completed = [
        row for row in trace
        if row.get("event") == "provider_call_completed"
    ]
    metadata = [
        row for row in trace
        if row.get("event") == "provider_metadata" and row.get("identity")
    ]
    heavy = {
        row.get("provider")
        for row in completed
        if row.get("operation") in {"infer", "describe", "visual_observation", "retrieve", "knowledge_generation"}
    }
    released = {
        row.get("provider")
        for row in completed
        if row.get("operation") == "release"
    }
    if not heavy <= released:
        errors.append(f"provider not released: {sorted(heavy - released)}")
    if any(
        int(row.get("cuda_peak_allocated_bytes", 0)) <= 0
        for row in completed
        if row.get("operation") in {"infer", "describe", "visual_observation", "retrieve", "knowledge_generation"}
    ):
        errors.append("heavy provider missing positive CUDA peak")
    if not metadata:
        errors.append("provider identity metadata missing")
    for key in ("mask_reference", "mask_probability_reference"):
        relative = response.get(key)
        if relative is not None and not (root / relative).is_file():
            errors.append(f"unreadable {key}: {relative}")
    if name == "region_interpretation":
        selection = response.get("region_selection") or {}
        expected_reason = (
            "NO_CANDIDATES"
            if len(response.get("candidate_regions", [])) == 0
            else "CANDIDATE_ID_MISSING"
        )
        expected = {
            "requested_source": "OA_AUXSEG_CANDIDATE",
            "effective_source": "OA_AUXSEG_GLOBAL",
            "status": "FALLBACK_GLOBAL",
            "reason": expected_reason,
            "requested_candidate_id": None,
            "selected_candidate_id": None,
        }
        for key, value in expected.items():
            if selection.get(key) != value:
                errors.append(f"candidate fallback {key} mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "task": expected_task,
        "response_kind": response.get("response_kind"),
        "no_target": response.get("no_target"),
        "candidate_count": len(response.get("candidate_regions", [])),
        "citation_count": len(response.get("citations", [])),
        "heavy_providers": sorted(str(value) for value in heavy),
        "released_providers": sorted(str(value) for value in released),
        "provider_metadata_events": len(metadata),
        "max_cuda_peak_allocated_bytes": max(
            (int(row.get("cuda_peak_allocated_bytes", 0)) for row in completed),
            default=0,
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"smoke root 已存在：{root}")
    root.mkdir(parents=True)
    sample_id, image, mask = _prepare(root)
    rows: list[dict[str, Any]] = []
    selected = [
        item for item in _requests(sample_id, image, mask)
        if args.only is None or item[0] in set(args.only)
    ]
    for name, request in selected:
        request_path = root / "requests" / f"{name}.json"
        atomic_write_json(request_path, request)
        dry = _run([
            sys.executable,
            str(CLI),
            "--config", str(CONFIG),
            "--request", str(request_path),
            "--dry-run",
        ])
        atomic_write_text(root / "logs" / f"{name}.dry.stdout.log", dry.stdout)
        atomic_write_text(root / "logs" / f"{name}.dry.stderr.log", dry.stderr)
        output = root / name
        process = _run([
            sys.executable,
            str(CLI),
            "--config", str(CONFIG),
            "--request", str(request_path),
            "--output-root", str(output),
        ])
        atomic_write_text(root / "logs" / f"{name}.stdout.log", process.stdout)
        atomic_write_text(root / "logs" / f"{name}.stderr.log", process.stderr)
        validation: dict[str, Any]
        if process.returncode == 0 and (output / "response.json").is_file():
            validation = _validate_output(name, output, str(request["task"]))
        else:
            failure = read_json(output / "failure.json") if (output / "failure.json").is_file() else None
            validation = {
                "ok": False,
                "errors": ["CLI failed"],
                "failure": failure,
            }
        row = {
            "name": name,
            "task": request["task"],
            "dry_run_exit_code": dry.returncode,
            "cli_exit_code": process.returncode,
            "validation": validation,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {
        "schema_version": "oa_groundrag.unified_cuda_smoke_summary.v1",
        "root": str(root),
        "python": sys.executable,
        "sample_id": sample_id,
        "results": rows,
        "ok": all(
            row["dry_run_exit_code"] == 0
            and row["cli_exit_code"] == 0
            and row["validation"].get("ok") is True
            for row in rows
        ),
        "engineering_runtime_only": True,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
        "training_performed": False,
    }
    atomic_write_json(root / "smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
