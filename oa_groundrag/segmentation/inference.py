"""OA-AuxSeg 可复用只读 single-batch / single-sample inference。"""

from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from oa_groundrag.data.oa_auxseg.dataset import (
    BenchmarkDataset,
    atomic_write_json,
    atomic_write_jsonl,
    collate_benchmark_samples,
)

from .checkpoint import model_from_checkpoint, read_checkpoint
from .contracts import (
    INFERENCE_SCHEMA_VERSION,
    OAAuxSegBatch,
    OAAuxSegOutput,
    RuntimeConfig,
    SUPPORTED_BACKBONE,
)
from .data import (
    PreparedBatch,
    benchmark_contract_from_root,
    make_dataloader,
    prepare_collated_batch,
)
from .config import resolve_runtime
from .policy import (
    autocast_context,
    prepare_policy_batch,
    validate_benchmark_registry,
    validate_inference_config,
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
        validate_inference_config(payload, model, config)
        validate_benchmark_registry(model, benchmark_root)
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
        with torch.inference_mode(), autocast_context(self.config, self.device):
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
        from oa_groundrag.data.grounded.pilot import render_optical

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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
@torch.no_grad()
def run_inference(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    split: str,
    source: str | None,
    limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    # 局部导入避免 inference_runtime 复用本模块 validation helper 时形成导入环。
    from oa_groundrag.segmentation.inference import SpatialInferenceSession

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"推理输出目录已存在，拒绝覆盖：{output_dir}")
    session = SpatialInferenceSession(
        config,
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
    )
    benchmark_root = session.benchmark_root
    device = session.device
    benchmark_contract = session.benchmark_contract
    benchmark_hash = str(benchmark_contract["index_sha256"])
    model = session.model
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        loader = make_dataloader(
            benchmark_root,
            split=split,
            source=source,
            batch_size=config.batch_size,
            normalization=config.normalization,
            shuffle=False,
            num_workers=config.num_workers,
        )
        rows: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        seen = 0
        for collated in loader:
            inference = session.infer_prepared(prepare_collated_batch(collated))
            prepared = inference.prepared
            output = inference.output
            available = inference.available_modalities
            active = inference.active_modalities
            for index, sample_id in enumerate(prepared.sample_ids):
                if limit is not None and seen >= limit:
                    break
                prefix = f"sample_{seen:06d}"
                regions = output.candidate_regions[index]
                region_masks = np.stack(
                    [region.mask.numpy().astype(np.uint8) for region in regions],
                    axis=0,
                ) if regions else np.empty(
                    (0, *prepared.model.spatial_size), dtype=np.uint8
                )
                arrays[f"{prefix}__mask_probability"] = (
                    output.mask_probability[index, 0].float().cpu().numpy()
                )
                arrays[f"{prefix}__global_mask"] = (
                    output.mask_probability[index, 0].float().cpu().numpy() >= 0.5
                ).astype(np.uint8)
                arrays[f"{prefix}__region_masks"] = region_masks
                arrays[f"{prefix}__region_features"] = (
                    output.region_features[index].float().cpu().numpy()
                )
                arrays[f"{prefix}__modality_weights"] = (
                    output.modality_weights[index].float().cpu().numpy()
                )
                weight_map_keys: dict[str, str] = {}
                for stride, weight_map in zip(
                    output.modality_weight_map_strides,
                    output.modality_weight_maps,
                    strict=True,
                ):
                    key = (
                        f"{prefix}__modality_weight_map_stride{stride}"
                    )
                    arrays[key] = (
                        weight_map[index].float().cpu().numpy()
                    )
                    weight_map_keys[str(stride)] = key
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source": str(prepared.metadata[index]["source"]),
                        "split": split,
                        "available_modalities": list(available[index]),
                        "active_modalities": list(active[index]),
                        "no_target_score": float(
                            output.no_target_score[index].float().item()
                        ),
                        "modality_names": list(output.modality_names),
                        "modality_weights": [
                            float(value)
                            for value in output.modality_weights[index]
                            .float()
                            .cpu()
                            .tolist()
                        ],
                        "array_keys": {
                            "mask_probability": f"{prefix}__mask_probability",
                            "global_mask": f"{prefix}__global_mask",
                            "region_masks": f"{prefix}__region_masks",
                            "region_features": f"{prefix}__region_features",
                            "modality_weights": f"{prefix}__modality_weights",
                            "modality_weight_maps": weight_map_keys,
                        },
                        "regions": [
                            {
                                "region_id": region.region_id,
                                "bbox_xyxy": list(region.bbox_xyxy),
                                "centroid_xy": list(region.centroid_xy),
                                "area_pixels": region.area_pixels,
                                "confidence": region.confidence,
                            }
                            for region in regions
                        ],
                    }
                )
                seen += 1
            if limit is not None and seen >= limit:
                break
        if not rows:
            raise ValueError("推理没有选择任何样本")
        atomic_write_jsonl(temporary_dir / "predictions.jsonl", rows)
        _atomic_npz(temporary_dir / "predictions.npz", arrays)
        atomic_write_json(
            temporary_dir / "manifest.json",
            {
                "schema_version": INFERENCE_SCHEMA_VERSION,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_step": session.checkpoint_step,
                "benchmark_index_sha256": benchmark_hash,
                "benchmark_contract": dict(benchmark_contract),
                "backbone_sha256": model.backbone_sha256,
                "backbone": SUPPORTED_BACKBONE,
                "architecture": model.model_contract()["architecture"],
                "split": split,
                "source": source,
                "sample_count": len(rows),
                "modality_weight_order": list(
                    model.modality_weight_order
                ),
                "modality_weight_map_strides": list(
                    model.modality_weight_map_strides
                ),
                "modality_weight_map_summary": (
                    "coverage_pool_each_stage_then_equal_stage_mean"
                ),
                "region_feature_dim": model.region_feature_dim,
            },
        )
        temporary_dir.replace(output_dir)
    except BaseException:
        for child in temporary_dir.iterdir():
            child.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise
    finally:
        session.release()
    return {
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "jsonl": str(output_dir / "predictions.jsonl"),
        "npz": str(output_dir / "predictions.npz"),
    }
