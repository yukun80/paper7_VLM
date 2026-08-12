"""已发布 External Benchmark 的真实、有界、source-hidden processor smoke。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from oa_groundrag.data.rs_general.dataset import (
    RSGeneralDescDataset,
    ParentBalancedSampler,
)

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.vlm.config import VLMConfig
from oa_groundrag.grounding.contracts import RUN_MANIFEST_SCHEMA_VERSION, DataMode
from oa_groundrag.vlm.data import (
    REQUIRED_EXTERNAL_ROLES,
    REQUIRED_EXTERNAL_SOURCES,
    REQUIRED_EXTERNAL_TASKS,
    ExternalDescriptionDataset,
    locate_bounded_external_records,
)
from oa_groundrag.vlm.errors import PreflightError, ReasonCode
from oa_groundrag.vlm.preflight import open_benchmark_access
from oa_groundrag.vlm.processing import DescriptionCollator, Qwen3VLProcessorAdapter
from oa_groundrag.vlm.reference import MAIN_REFERENCE


def run_bounded_external_smoke(
    config: VLMConfig,
    *,
    output_root: Path | None = None,
) -> Path:
    if config.run.mode is not DataMode.EXTERNAL_GENERIC:
        raise PreflightError(
            ReasonCode.EXTERNAL_MASK_FORBIDDEN,
            "真实 bounded smoke 只接受 external_generic 配置",
        )
    target = Path(output_root or config.run.output_root)
    if target.exists() or target.is_symlink():
        raise PreflightError(
            ReasonCode.OUTPUT_EXISTS,
            f"smoke output_root 已存在：{target}",
        )
    access = open_benchmark_access(config)
    identity = access.identity
    selection = locate_bounded_external_records(config, access=access)
    with AtomicArtifactDirectory(target) as writer:
        assert writer.staging is not None
        canonical = RSGeneralDescDataset.from_locations(
            config.data.benchmark_root,
            selection.locations,
            roles=config.data.roles,
            task_families=config.data.task_families,
            load_assets=False,
            seed=config.run.seed,
            expected_manifest_sha256=config.data.expected_manifest_sha256,
            verifier=access.verifier,
        )
        expected_ids = [str(row["record_id"]) for row in selection.records]
        actual_ids = [str(row["record_id"]) for row in canonical.records]
        if actual_ids != expected_ids:
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                "from_locations 读取结果与 locator 不一致",
                details={"expected": expected_ids, "actual": actual_ids},
            )
        dataset = ExternalDescriptionDataset(
            canonical,
            derived_root=writer.path("derived"),
            seed=config.run.seed,
        )
        sampler = ParentBalancedSampler(
            dataset,
            seed=config.run.seed,
            num_samples=len(dataset),
            source_weights=config.data.source_weights,
            task_weights=config.data.task_weights,
        )
        processor = Qwen3VLProcessorAdapter(
            processor_path=config.model.processor_path,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
            min_pixels=config.limits.min_pixels,
            max_pixels=config.limits.max_pixels,
            max_images=config.limits.max_images,
            max_input_tokens=config.limits.max_input_tokens,
        )
        collator = DescriptionCollator(processor, training=True)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collator,
        )
        rows: list[dict[str, Any]] = []
        token_total = 0
        image_total = 0
        parent_ids: set[str] = set()
        for batch in loader:
            token_count = int(batch["input_token_counts"][0])
            image_count = int(batch["image_counts"][0])
            token_total += token_count
            image_total += image_count
            if token_total > config.limits.max_total_tokens:
                raise PreflightError(
                    ReasonCode.TOKEN_LIMIT_EXCEEDED,
                    "processor smoke total token cap exceeded",
                )
            labels = batch["labels"]
            supervised = int((labels != -100).sum().item())
            if supervised <= 0:
                raise PreflightError(
                    ReasonCode.LOSS_MASK_INVALID,
                    "processor smoke assistant-only labels 为空",
                )
            parent_ids.add(str(batch["parent_ids"][0]))
            rows.append(
                {
                    "record_id": str(batch["record_ids"][0]),
                    "parent_id": str(batch["parent_ids"][0]),
                    "task_family": str(batch["task_families"][0]),
                    "input_tokens": token_count,
                    "supervised_tokens": supervised,
                    "image_count": image_count,
                }
            )
        role_set = set(selection.role_counts)
        source_set = set(selection.source_counts)
        task_set = set(selection.task_counts)
        if (
            role_set != REQUIRED_EXTERNAL_ROLES
            or source_set != REQUIRED_EXTERNAL_SOURCES
            or task_set != REQUIRED_EXTERNAL_TASKS
        ):
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                "bounded smoke coverage 不完整",
                details={
                    "roles": sorted(role_set),
                    "sources": sorted(source_set),
                    "tasks": sorted(task_set),
                },
            )
        if len(parent_ids) > config.limits.max_parents:
            raise PreflightError(
                ReasonCode.ASSET_LIMIT_EXCEEDED,
                "smoke parent count 超限",
            )
        writer.write_jsonl("processor_records.jsonl", rows)
        report = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_kind": "bounded_external_processor_smoke",
            "seed": config.run.seed,
            "config_semantic_sha256": config.semantic_sha256,
            "benchmark": identity.to_dict(),
            "selection": selection.to_dict(),
            "dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "sampler_schema": sampler.state_dict()["schema_version"],
                "processed_records": len(rows),
                "processed_parents": len(parent_ids),
                "input_tokens": token_total,
                "images": image_total,
                "supervised_tokens": sum(
                    int(row["supervised_tokens"]) for row in rows
                ),
                "task_counts": dict(
                    sorted(
                        Counter(
                            str(row["task_family"]) for row in rows
                        ).items()
                    )
                ),
            },
            "processor_identity": processor.identity(),
            "main_reference": MAIN_REFERENCE.to_dict(),
            "asset_policy": {
                "canonical_assets_copied": False,
                "copied_asset_bytes": 0,
                "canonical_assets_modified": False,
            },
            "qwen_forward": {
                "run": False,
                "reason": (
                    "cuda_unavailable"
                    if not torch.cuda.is_available()
                    else "not_requested_after_process_safety_gate"
                ),
            },
            "formal_acceptance": False,
            "mask_grounded_data_used": False,
        }
        writer.write_json("report.json", report)
        writer.write_json(
            "manifest.json",
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_kind": "bounded_external_processor_smoke",
                "record_count": len(rows),
                "parent_count": len(parent_ids),
                "copied_asset_bytes": 0,
                "formal_acceptance": False,
            },
        )
        return writer.publish()
