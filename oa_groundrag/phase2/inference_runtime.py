"""OA-AuxSeg 可复用只读 single-batch / single-sample inference。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.phase1_benchmark_build.benchmark_common import (
    BenchmarkDataset,
    collate_benchmark_samples,
)

from .checkpoint import model_from_checkpoint, read_checkpoint
from .contracts import OAAuxSegBatch, OAAuxSegOutput, RuntimeConfig
from .data import PreparedBatch, benchmark_contract_from_root, prepare_collated_batch
from .engine import (
    _autocast,
    _validate_benchmark_registry,
    _validate_inference_config,
    prepare_policy_batch,
    resolve_runtime,
)


@dataclass(frozen=True)
class BatchInferenceResult:
    prepared: PreparedBatch
    output: OAAuxSegOutput
    available_modalities: tuple[tuple[str, ...], ...]
    active_modalities: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SingleSampleInference:
    sample_id: str
    source: str
    split: str
    optical_image: Any
    result: BatchInferenceResult


class SpatialInferenceSession:
    """严格绑定当前 RuntimeConfig、Benchmark 与 checkpoint 的只读会话。"""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        repo_root: Path,
        checkpoint_path: Path,
    ) -> None:
        benchmark_root, _, _, device = resolve_runtime(config, repo_root)
        benchmark_contract = benchmark_contract_from_root(benchmark_root)
        payload = read_checkpoint(
            checkpoint_path,
            expected_benchmark_contract=benchmark_contract,
        )
        model = model_from_checkpoint(payload, device=device)
        _validate_inference_config(payload, model, config)
        _validate_benchmark_registry(model, benchmark_root)
        model.eval()
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_step = int(payload["step"])
        self.benchmark_root = benchmark_root.resolve()
        self.benchmark_contract = dict(benchmark_contract)
        self.model = model
        self.device = device
        del payload
        gc.collect()

    @property
    def identity(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("SpatialInferenceSession 已释放")
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_step": self.checkpoint_step,
            "benchmark_contract": dict(self.benchmark_contract),
            "backbone_sha256": self.model.backbone_sha256,
            "architecture": self.model.model_contract()["architecture"],
            "region_threshold": self.config.region_threshold,
            "min_region_area": self.config.min_region_area,
            "loaded_components": ["oa_auxseg_model_state"],
            "excluded_components": ["optimizer", "scheduler", "rng", "sampler"],
        }

    def infer_batch(self, batch: OAAuxSegBatch) -> BatchInferenceResult:
        prepared = PreparedBatch(
            model=batch,
            mask=torch.zeros(
                (batch.batch_size, 1, *batch.spatial_size),
                dtype=torch.float32,
            ),
            sample_ids=[f"in_memory_{index}" for index in range(batch.batch_size)],
            metadata=[{} for _ in range(batch.batch_size)],
        )
        return self.infer_prepared(prepared)

    def infer_prepared(self, prepared: PreparedBatch) -> BatchInferenceResult:
        if self.model is None:
            raise RuntimeError("SpatialInferenceSession 已释放")
        device_prepared = prepared.to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )
        model_batch, available, active = prepare_policy_batch(
            device_prepared.model,
            variant=self.model.variant,
            subset_sampler=None,
        )
        with torch.inference_mode(), _autocast(self.config, self.device):
            output = self.model(model_batch, return_regions=True)
        if output.candidate_regions is None or output.region_features is None:
            raise RuntimeError("OA-AuxSeg inference 未返回区域合同")
        return BatchInferenceResult(
            prepared=device_prepared,
            output=output,
            available_modalities=tuple(tuple(value) for value in available),
            active_modalities=tuple(tuple(value) for value in active),
        )

    def infer_benchmark_sample(self, *, split: str, sample_id: str) -> SingleSampleInference:
        from oa_groundrag.landslide_evidence.pipeline import render_optical

        if split not in {"train", "val"}:
            raise ValueError("Unified OA-AuxSeg sample inference 只允许 train/val")
        model_dataset = BenchmarkDataset(
            self.benchmark_root,
            split=split,
            auxiliary_policy="all",
            normalization=self.config.normalization,
        )
        indices = [
            index
            for index, row in enumerate(model_dataset.rows)
            if str(row["sample_id"]) == sample_id
        ]
        if len(indices) != 1:
            raise ValueError(
                f"{split} sample_id 必须唯一存在，实际匹配 {len(indices)}：{sample_id}"
            )
        index = indices[0]
        model_sample = model_dataset[index]
        raw_dataset = BenchmarkDataset(
            self.benchmark_root,
            split=split,
            auxiliary_policy="all",
            normalization="none",
        )
        raw_sample = raw_dataset[index]
        if raw_sample["sample_id"] != model_sample["sample_id"]:
            raise RuntimeError("normalized/raw Benchmark sample 顺序漂移")
        prepared = prepare_collated_batch(collate_benchmark_samples([model_sample]))
        result = self.infer_prepared(prepared)
        return SingleSampleInference(
            sample_id=str(model_sample["sample_id"]),
            source=str(model_sample["metadata"]["source"]),
            split=split,
            optical_image=render_optical(raw_sample),
            result=result,
        )

    def release(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
