"""Qwen3.5 Mask-Grounded 两 epoch 资源画像与最坏样本冻结。"""

from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from torch import Tensor

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.grounded.supervision.compact_training import (
    CompactTrainingMessageDataset,
)
from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset
from oa_groundrag.data.rs_general.io import read_json, read_jsonl
from oa_groundrag.training.vlm.input_pipeline import (
    DeterministicBatchPlanner,
    OrderedBatchPrefetcher,
)
from oa_groundrag.vlm.backends import build_processor_adapter
from oa_groundrag.vlm.data import ExternalDescriptionDataset
from oa_groundrag.vlm.errors import ConfigError, ModelError, ReasonCode
from oa_groundrag.vlm.preflight import open_benchmark_access
from oa_groundrag.vlm.processing import DescriptionCollator

from .config import STAGE5_CONFIG_SCHEMA_V4, Stage5Config, load_stage5_config
from .data import (
    REGION_TRAIN_ROLE,
    RegionSubsetDataset,
    Stage5MixedDataset,
    Stage5MixedSampler,
    split_compact_by_parent,
)


RESOURCE_PROFILE_SCHEMA = "rs_vlm.mask_grounded_resource_profile.v1"
RESOURCE_PROFILE_ROW_SCHEMA = "rs_vlm.mask_grounded_resource_profile_row.v1"
WORST_CASE_SELECTION_SCHEMA = "rs_vlm.mask_grounded_worst_case_selection.v1"
RESOURCE_PROFILE_LEDGER_SCHEMA = "rs_vlm.mask_grounded_resource_profile_ledger.v1"
EXPECTED_SCHEDULE_ROWS = 16_000
QWEN35_VOCAB_SIZE = 248_320
QWEN35_HIDDEN_SIZE = 2_560
QWEN35_VISION_MERGE_AREA = 4

_PROFILE_FILES = (
    "schedule_profile.jsonl",
    "worst_case_selection.json",
    "SHA256SUMS.jsonl",
)
_SELECTION_METRICS = (
    "input_tokens",
    "supervised_tokens",
    "pixel_numel",
    "vision_grid_volume",
    "projected_composite_footprint_bytes",
)


def _ordinary_file(path: Path, *, label: str) -> Path:
    linked = first_symlink_component(path)
    if (
        linked is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        raise ConfigError(
            ReasonCode.CHECKPOINT_CORRUPT,
            f"{label} 必须是普通单链接文件：{path}",
        )
    return path


def _profile_row(
    *,
    sequence_index: int,
    sample: Any,
    epoch: int,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    if len(batch.get("input_token_counts", ())) != 1:
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "M7 资源画像只允许 physical batch=1",
        )
    labels = batch.get("labels")
    pixels = batch.get("pixel_values")
    grid = batch.get("image_grid_thw")
    if not all(isinstance(value, Tensor) for value in (labels, pixels, grid)):
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "M7 资源画像缺少 labels/pixel_values/image_grid_thw",
        )
    assert isinstance(labels, Tensor)
    assert isinstance(pixels, Tensor)
    assert isinstance(grid, Tensor)
    if labels.ndim != 2 or labels.shape[0] != 1 or grid.ndim != 2 or grid.shape[1] != 3:
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "M7 资源画像 tensor shape 非法",
        )
    input_tokens = int(batch["input_token_counts"][0])
    supervised_tokens = int(labels[:, 1:].ne(-100).sum())
    if supervised_tokens <= 0:
        raise ModelError(
            ReasonCode.LOSS_MASK_INVALID,
            "M7 资源画像样本没有 causal-shift 后监督位置",
        )
    grid_rows = [[int(value) for value in row] for row in grid.tolist()]
    grid_volumes = [math.prod(row) for row in grid_rows]
    if any(value % QWEN35_VISION_MERGE_AREA != 0 for value in grid_volumes):
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5 vision grid 不能按固定 merge area 折叠",
        )
    vision_grid_volume = sum(grid_volumes)
    vision_tokens = sum(
        value // QWEN35_VISION_MERGE_AREA for value in grid_volumes
    )
    pixel_numel = int(pixels.numel())
    pixel_bytes = pixel_numel * int(pixels.element_size())
    full_logits_bytes = input_tokens * QWEN35_VOCAB_SIZE * 6
    projected_logits_bytes = supervised_tokens * QWEN35_VOCAB_SIZE * 6
    language_hidden_bytes = input_tokens * QWEN35_HIDDEN_SIZE * 2
    vision_hidden_bytes = vision_tokens * QWEN35_HIDDEN_SIZE * 2
    composite = (
        projected_logits_bytes
        + language_hidden_bytes
        + vision_hidden_bytes
        + pixel_bytes
    )
    return {
        "schema_version": RESOURCE_PROFILE_ROW_SCHEMA,
        "sequence_index": sequence_index,
        "optimizer_step": sequence_index // 16 + 1,
        "micro_step": sequence_index % 16 + 1,
        "epoch": epoch,
        "record_id": sample.record_id,
        "parent_id": sample.parent_id,
        "logical_role": sample.logical_role,
        "task_family": sample.task_family,
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "image_count": int(batch["image_counts"][0]),
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
        "pixel_numel": pixel_numel,
        "pixel_bytes": pixel_bytes,
        "image_grid_thw": grid_rows,
        "vision_grid_volume": vision_grid_volume,
        "vision_tokens": vision_tokens,
        "full_logits_fp32_ce_footprint_bytes": full_logits_bytes,
        "projected_logits_fp32_ce_footprint_bytes": projected_logits_bytes,
        "projected_composite_footprint_bytes": composite,
    }


def _worst_case_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_by_metric = {
        metric: max(
            rows,
            key=lambda row: (int(row[metric]), -int(row["sequence_index"])),
        )
        for metric in _SELECTION_METRICS
    }
    reasons: dict[int, list[str]] = defaultdict(list)
    for metric, row in selected_by_metric.items():
        reasons[int(row["sequence_index"])].append(metric)
    selected = []
    for sequence_index in sorted(reasons):
        row = rows[sequence_index]
        selected.append(
            {
                "sequence_index": sequence_index,
                "record_id": row["record_id"],
                "parent_id": row["parent_id"],
                "logical_role": row["logical_role"],
                "task_family": row["task_family"],
                "epoch": row["epoch"],
                "reasons": sorted(reasons[sequence_index]),
                "metrics": {
                    metric: row[metric] for metric in _SELECTION_METRICS
                },
            }
        )
    payload = {
        "schema_version": WORST_CASE_SELECTION_SCHEMA,
        "selection_algorithm": "metric_max_then_earliest_sequence.v1",
        "selection_metrics": list(_SELECTION_METRICS),
        "schedule_row_count": len(rows),
        "selected": selected,
    }
    return {
        **payload,
        "selection_sha256": sha256_text(canonical_json(payload)),
    }


def profile_stage5_resource_schedule(
    config_path: Path | str,
    *,
    output_root: Path | str | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """CPU 回放两 epoch 的固定顺序并原子发布资源画像；不读 test。"""

    config = load_stage5_config(config_path)
    if (
        config.schema_version != STAGE5_CONFIG_SCHEMA_V4
        or config.resource_contract is None
        or config.model.backend != "qwen3_5"
        or config.training.batch_size != 1
        or config.training.gradient_accumulation_steps != 16
        or config.training.epochs != 2
        or config.training.max_steps != 1000
    ):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "资源画像只接受冻结的 Qwen3.5 M7 v4 配置",
        )
    target = Path(
        os.path.abspath(
            Path(output_root)
            if output_root is not None
            else config.resource_contract.profile_root
        )
    )
    if target != config.resource_contract.profile_root:
        raise ConfigError(
            ReasonCode.PATH_ESCAPE,
            "资源画像 output_root 必须等于 v4 配置绑定路径",
        )
    compact = CompactTrainingMessageDataset(
        config.data_contract.compact_training_root
    )
    split = split_compact_by_parent(
        compact,
        seed=config.data_contract.split_seed,
    )
    access = open_benchmark_access(config.base)
    processor = build_processor_adapter(config.base)
    collator = DescriptionCollator(processor, training=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="m7_resource_profile_rs_general_") as temporary:
        canonical = RSGeneralDescDataset(
            config.data.benchmark_root,
            roles=("external_train",),
            task_families=config.data.task_families,
            load_assets=False,
            seed=config.run.seed,
            expected_manifest_sha256=config.data.expected_manifest_sha256,
            verifier=access.verifier,
        )
        replay = ExternalDescriptionDataset(
            canonical,
            derived_root=Path(temporary) / "replay",
            seed=config.run.seed,
            roles=("external_train",),
        )
        region = RegionSubsetDataset(
            compact,
            split.train_indices,
            logical_role=REGION_TRAIN_ROLE,
        )
        mixed = Stage5MixedDataset(region, replay)
        sampler = Stage5MixedSampler(mixed, seed=config.run.seed)
        planner = DeterministicBatchPlanner(
            dataset=mixed,
            sampler=sampler,
            batch_size=1,
            max_epochs=2,
            start_epoch=0,
            start_sample_offset=0,
            max_batches=EXPECTED_SCHEDULE_ROWS,
        )
        pipeline = OrderedBatchPrefetcher(
            planner=planner,
            collator=collator,
            num_workers=config.training.num_workers,
            prefetch_factor=config.training.prefetch_factor,
            pin_memory=False,
            worker_collators=tuple(
                collator.clone_for_worker()
                for _ in range(config.training.num_workers)
            ),
        )
        try:
            for sequence_index in range(EXPECTED_SCHEDULE_ROWS):
                prepared = next(pipeline)
                rows.append(
                    _profile_row(
                        sequence_index=sequence_index,
                        sample=prepared.plan.samples[0],
                        epoch=prepared.plan.sample_epochs[0],
                        batch=prepared.batch,
                    )
                )
                if progress_callback is not None and (
                    sequence_index == 0 or (sequence_index + 1) % 100 == 0
                ):
                    progress_callback(
                        {
                            "schema_version": RESOURCE_PROFILE_SCHEMA,
                            "event": "profile_progress",
                            "completed": sequence_index + 1,
                            "total": EXPECTED_SCHEDULE_ROWS,
                        }
                    )
            try:
                next(pipeline)
            except StopIteration:
                pass
            else:
                raise AssertionError("M7 两 epoch 画像未在 16,000 条后耗尽")
        finally:
            pipeline.close()
    selection = _worst_case_selection(rows)
    with AtomicArtifactDirectory(target) as artifact:
        artifact.write_jsonl("schedule_profile.jsonl", rows)
        artifact.write_json("worst_case_selection.json", selection)
        ledger_rows = [
            {
                "schema_version": RESOURCE_PROFILE_LEDGER_SCHEMA,
                "path": relative,
                "size_bytes": artifact.path(relative).stat().st_size,
                "sha256": sha256_file(artifact.path(relative)),
            }
            for relative in _PROFILE_FILES[:2]
        ]
        artifact.write_jsonl("SHA256SUMS.jsonl", ledger_rows)
        manifest = {
            "schema_version": RESOURCE_PROFILE_SCHEMA,
            "config_semantic_sha256": config.semantic_sha256,
            "compact_manifest_sha256": sha256_file(compact.root / "manifest.json"),
            "region_split_identity_sha256": split.identity_sha256,
            "schedule_algorithm": "stage5_mixed_sampler_two_epochs.v1",
            "schedule_row_count": len(rows),
            "optimizer_steps": 1000,
            "gradient_accumulation_steps": 16,
            "selection_sha256": selection["selection_sha256"],
            "schedule_profile_sha256": sha256_file(
                artifact.path("schedule_profile.jsonl")
            ),
            "worst_case_selection_file_sha256": sha256_file(
                artifact.path("worst_case_selection.json")
            ),
            "ledger_sha256": sha256_file(artifact.path("SHA256SUMS.jsonl")),
            "vocab_size": QWEN35_VOCAB_SIZE,
            "hidden_size": QWEN35_HIDDEN_SIZE,
            "full_logits_byte_formula": "input_tokens*vocab_size*(bf16_2+fp32_4)",
            "projected_composite_formula": (
                "supervised_tokens*vocab_size*6+input_tokens*hidden_size*2+"
                "vision_tokens*hidden_size*2+pixel_bytes"
            ),
            "split_scope": "train_only",
            "test_or_sealed_accessed": False,
        }
        artifact.write_json("manifest.json", manifest)
        artifact.publish()
    return verify_stage5_resource_profile(config)


def verify_stage5_resource_profile(config: Stage5Config) -> dict[str, Any]:
    """严格验证 v4 配置绑定的画像、ledger 与最坏样本选择。"""

    resource = config.resource_contract
    if resource is None:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            "非 v4 配置没有 M7 resource profile",
        )
    root = resource.profile_root
    linked = first_symlink_component(root)
    if linked is not None or not root.is_dir() or root.is_symlink():
        raise ConfigError(
            ReasonCode.CHECKPOINT_CORRUPT,
            f"M7 resource profile 必须是普通目录：{root}",
        )
    manifest_path = _ordinary_file(root / "manifest.json", label="profile manifest")
    schedule_path = _ordinary_file(
        root / "schedule_profile.jsonl", label="schedule profile"
    )
    selection_path = _ordinary_file(
        root / "worst_case_selection.json", label="worst-case selection"
    )
    ledger_path = _ordinary_file(root / "SHA256SUMS.jsonl", label="profile ledger")
    if {path.name for path in root.iterdir()} != {"manifest.json", *_PROFILE_FILES}:
        raise ConfigError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "M7 resource profile 出现缺失或额外文件",
        )
    manifest = read_json(manifest_path)
    selection = read_json(selection_path)
    ledger = read_jsonl(ledger_path)
    expected_ledger = [
        {
            "schema_version": RESOURCE_PROFILE_LEDGER_SCHEMA,
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in _PROFILE_FILES[:2]
    ]
    if ledger != expected_ledger:
        raise ConfigError(ReasonCode.CHECKPOINT_CORRUPT, "M7 resource profile ledger 漂移")
    if (
        manifest.get("schema_version") != RESOURCE_PROFILE_SCHEMA
        or manifest.get("schedule_row_count") != EXPECTED_SCHEDULE_ROWS
        or manifest.get("optimizer_steps") != 1000
        or manifest.get("gradient_accumulation_steps") != 16
        or manifest.get("schedule_profile_sha256") != sha256_file(schedule_path)
        or manifest.get("worst_case_selection_file_sha256")
        != sha256_file(selection_path)
        or manifest.get("ledger_sha256") != sha256_file(ledger_path)
        or manifest.get("selection_sha256") != selection.get("selection_sha256")
        or manifest.get("split_scope") != "train_only"
        or manifest.get("test_or_sealed_accessed") is not False
    ):
        raise ConfigError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "M7 resource profile manifest/selection 身份漂移",
        )
    selection_payload = dict(selection)
    selection_sha = selection_payload.pop("selection_sha256", None)
    if selection_sha != sha256_text(canonical_json(selection_payload)):
        raise ConfigError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "M7 worst-case selection canonical SHA 漂移",
        )
    rows = read_jsonl(schedule_path)
    if len(rows) != EXPECTED_SCHEDULE_ROWS:
        raise ConfigError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "M7 resource profile 行数不是 16,000",
        )
    for item in selection.get("selected", ()):
        if not isinstance(item, dict):
            raise ConfigError(ReasonCode.CHECKPOINT_CORRUPT, "最坏样本选择项非法")
        index = item.get("sequence_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(rows)
            or rows[index].get("record_id") != item.get("record_id")
            or rows[index].get("parent_id") != item.get("parent_id")
            or rows[index].get("logical_role") != item.get("logical_role")
            or rows[index].get("task_family") != item.get("task_family")
            or rows[index].get("epoch") != item.get("epoch")
            or item.get("metrics")
            != {metric: rows[index].get(metric) for metric in _SELECTION_METRICS}
        ):
            raise ConfigError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "最坏样本与 schedule profile 不一致",
            )
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "schedule_profile_sha256": sha256_file(schedule_path),
        "selection_file_sha256": sha256_file(selection_path),
        "selection_sha256": str(selection_sha),
        "schedule_row_count": len(rows),
        "selected_count": len(selection.get("selected", ())),
    }
