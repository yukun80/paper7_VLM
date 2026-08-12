from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.runtime.config import load_unified_config
from oa_groundrag.runtime.demo.config import BenchmarkBinding, DemoConfig, DemoDefaults
from oa_groundrag.runtime.inference import UnifiedInferenceRuntime
from oa_groundrag.runtime.providers import (
    GroundedEvidenceResult,
    SpatialCandidate,
    SpatialResult,
    TextRAGResult,
)
from oa_groundrag.runtime.contracts import UnifiedRequest, UnifiedTask


REPO_ROOT = Path(__file__).resolve().parents[3]


def build_benchmark(root: Path) -> BenchmarkBinding:
    specs = (
        ("train-small", "train", "source_a", 0.005, ()),
        ("train-medium", "train", "source_b", 0.05, ("dem",)),
        ("val-empty", "val", "source_a", 0.0, ("slope",)),
        ("val-large", "val", "source_b", 0.20, ("dem", "slope")),
        ("test-small", "test", "source_c", 0.005, ("insar_velocity",)),
    )
    rows: list[dict[str, Any]] = []
    for ordinal, (sample_id, split, source, ratio, auxiliaries) in enumerate(specs):
        shard_relative = f"data/{source}/{split}/shard-00000.h5"
        shard = root / shard_relative
        shard.parent.mkdir(parents=True, exist_ok=True)
        optical = np.stack([
            np.full((16, 16), 20 + ordinal, dtype=np.float32),
            np.full((16, 16), 40 + ordinal, dtype=np.float32),
            np.full((16, 16), 60 + ordinal, dtype=np.float32),
        ])
        mask = np.zeros((1, 16, 16), dtype=np.uint8)
        pixels = max(1, int(round(ratio * 256))) if ratio > 0 else 0
        if pixels:
            mask.reshape(-1)[:pixels] = 1
        actual_ratio = float(mask.mean())
        with h5py.File(shard, "w") as handle:
            handle.create_dataset("optical", data=optical[None])
            handle.create_dataset("optical_pixel_valid", data=np.ones_like(optical[None], dtype=np.uint8))
            handle.create_dataset("optical_channel_valid", data=np.ones((1, 3), dtype=np.uint8))
            handle.create_dataset("mask", data=mask[None])
            for name in auxiliaries:
                group = handle.create_group(f"auxiliary/{name}")
                group.create_dataset("values", data=np.ones((1, 1, 16, 16), dtype=np.float32))
                group.create_dataset("pixel_valid", data=np.ones((1, 1, 16, 16), dtype=np.uint8))
                group.create_dataset("channel_valid", data=np.ones((1, 1), dtype=np.uint8))
        rows.append({
            "schema_version": "oa_auxseg_hdf5_v1",
            "sample_id": sample_id,
            "source": source,
            "split": split,
            "foreground_ratio": actual_ratio,
            "optical": {
                "shape": [3, 16, 16],
                "channel_names": ["Red", "Green", "Blue"],
                "pixel_validity": True,
                "channel_validity": True,
            },
            "auxiliaries": {
                name: {"channel_names": [name], "shape": [1, 16, 16]}
                for name in auxiliaries
            },
            "mask": {"shape": [1, 16, 16], "values": [0, 1]},
            "resize": {"original_size": [16, 16], "target_size": [16, 16]},
            "storage": {"shard": shard_relative, "row": 0},
            "record_sha256": f"{ordinal + 1:064x}",
        })
    root.mkdir(parents=True, exist_ok=True)
    index = root / "index.jsonl"
    index.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "source_statistics.json").write_text(
        json.dumps({"sources": {}}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "oa_auxseg_hdf5_v1",
        "sample_count": len(rows),
        "split_counts": {"train": 2, "val": 2, "test": 1},
        "included_sources": ["source_a", "source_b", "source_c"],
        "index_sha256": sha256_file(index),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return BenchmarkBinding(
        root=root,
        schema_version="oa_auxseg_hdf5_v1",
        manifest_sha256=sha256_file(manifest_path),
        index_sha256=sha256_file(index),
    )


def make_demo_config(
    root: Path,
    binding: BenchmarkBinding,
    *,
    allow_test_demo: bool = False,
) -> DemoConfig:
    unified = load_unified_config(REPO_ROOT / "configs/runtime/inference_v2.yaml")
    prompts = {task: f"prompt for {task.value}" for task in UnifiedTask}
    return DemoConfig(
        config_path=root / "synthetic_demo.yaml",
        config_sha256="d" * 64,
        unified=unified,
        benchmark=binding,
        frozen_evaluations=(),
        demo_root=root / "demo",
        allow_test_demo=allow_test_demo,
        defaults=DemoDefaults(
            mode="Demo / Qualitative Exploration",
            split="val",
            suite_tasks=(
                UnifiedTask.VLM_ONLY,
                UnifiedTask.SEGMENT_ONLY,
                UnifiedTask.SEGMENT_AND_UNDERSTAND,
                UnifiedTask.REGION_INTERPRETATION,
            ),
            prompts=prompts,
        ),
    )


class FakeSpatial:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult:
        self.calls.append(f"spatial:{request.task.value}")
        output_dir.mkdir(parents=True)
        width, height = request.spatial_input.optical_image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[2 : height - 2, 2 : width - 2] = True
        probability = mask.astype(np.float32) * 0.9
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "global_mask.png")
        with (output_dir / "mask_probability.npy").open("wb") as handle:
            np.save(handle, probability, allow_pickle=False)
        candidate_root = output_dir / "candidates"
        candidate_root.mkdir()
        Image.fromarray(mask.astype(np.uint8) * 255).save(candidate_root / "region_5.png")
        return SpatialResult(
            sample_id=request.spatial_input.sample_id,
            source=request.spatial_input.source,
            split=request.spatial_input.split,
            optical_image=request.spatial_input.optical_image,
            global_mask=mask,
            mask_probability=probability,
            no_target=False,
            no_target_score=0.1,
            candidates=(SpatialCandidate(
                5,
                mask,
                (2, 2, width - 2, height - 2),
                ((width - 1) / 2, (height - 1) / 2),
                int(mask.sum()),
                0.9,
                "spatial/candidates/region_5.png",
            ),),
            active_modalities=("dem",),
            mask_reference="spatial/global_mask.png",
            mask_probability_reference="spatial/mask_probability.npy",
            identity={"fake": True},
        )

    def release(self) -> None:
        self.calls.append("release:spatial")


class FakeShared:
    def __init__(self, calls: list[str], *, fail_describe_once: bool = False) -> None:
        self.calls = calls
        self.fail_describe_once = fail_describe_once

    def describe(self, request: UnifiedRequest) -> str:
        self.calls.append("shared:describe")
        if self.fail_describe_once:
            self.fail_describe_once = False
            raise RuntimeError("synthetic VLM failure")
        return "scene description"

    def generate_visual(self, messages: Sequence[Mapping[str, Any]]) -> str:
        self.calls.append("shared:visual")
        return "raw visual"

    def generate_text(self, messages: Sequence[Mapping[str, Any]], *, packet: Mapping[str, Any]) -> str:
        self.calls.append("shared:text")
        return "raw knowledge"

    def release(self) -> None:
        self.calls.append("release:shared")

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {"fake": True}


class FakeEvidence:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def build(self, request: UnifiedRequest, *, optical_image: Any, mask: Any, selection: Any,
              output_dir: Path, sample_id: str, source: str, split: str, no_target: bool) -> GroundedEvidenceResult:
        self.calls.append(f"evidence:{request.task.value}")
        output_dir.mkdir(parents=True)
        image = optical_image if isinstance(optical_image, Image.Image) else Image.open(optical_image).convert("RGB")
        image.save(output_dir / "optical_full.png")
        if isinstance(mask, (str, Path)):
            mask_image = Image.open(mask)
            mask_values = np.asarray(mask_image, dtype=np.uint8) > 0
        else:
            mask_values = np.asarray(mask, dtype=bool)
        Image.fromarray(mask_values.astype(np.uint8) * 255).save(output_dir / "binary_mask.png")
        image.crop((1, 1, image.width - 1, image.height - 1)).save(output_dir / "context_crop.png")
        return GroundedEvidenceResult(
            messages=({"role": "user", "content": []},),
            program_facts={"mask": {"area_pixels": int(mask_values.sum())}},
            target_status="target_present",
            mask_reference="grounded/binary_mask.png",
            limitations=(),
            metadata={"formal_model_input_roles": ["optical_full", "binary_mask", "context_crop"]},
        )

    def parse_observation(self, raw_output: str, *, evidence: GroundedEvidenceResult) -> Mapping[str, Any]:
        self.calls.append("evidence:parse")
        return {"target_status": "target_present", "short_summary": "observation", "limitations": []}


class FakeRAG:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def retrieve(self, request: UnifiedRequest, *, observation: Mapping[str, Any],
                 program_facts: Mapping[str, Any], target_status: str,
                 available_modalities: Sequence[str], candidate_count: int) -> TextRAGResult:
        self.calls.append(f"rag:{request.task.value}")
        item = {
            "knowledge_type": "interpretation",
            "evidence_id": "ev-1",
            "source_id": "source-1",
            "source_title": "Synthetic Source",
            "pdf_page": 3,
            "section": "S1",
            "text": "bounded evidence",
        }
        return TextRAGResult(
            packet={"packet_id": "packet-1", "items": [item]},
            messages=({"role": "user", "content": []},),
            citations=(item,),
            metadata={"fake": True},
        )

    def parse_generation(self, raw_output: str, *, result: TextRAGResult) -> Mapping[str, Any]:
        self.calls.append("rag:parse")
        return {
            "summary": {"text": "interpretation", "evidence_ids": ["ev-1"]},
            "limitations": [],
        }

    def release(self) -> None:
        self.calls.append("release:rag")


def fake_runtime(*, fail_describe_once: bool = False) -> tuple[UnifiedInferenceRuntime, list[str]]:
    calls: list[str] = []
    return UnifiedInferenceRuntime(
        spatial=FakeSpatial(calls),
        shared_mllm=FakeShared(calls, fail_describe_once=fail_describe_once),
        evidence=FakeEvidence(calls),
        text_rag=FakeRAG(calls),
    ), calls
