#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-aware datasets for global, region-alignment and Landslide Bridge description."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import MODALITY_FAMILY_IDS

from ..protocols.region_geometry import (
    bridge_region_mask_digest,
    project_native_region_mask_to_cache,
    transform_region_mask_to_cache,
)
from ..protocols.io import sha256_file, strict_json_loads
from .vision_cache import DescriptionVisionFeatureBank, description_cache_key
from .expert_contracts import (
    BRIDGE_BUILDER_VERSION,
    require_frozen_expert_bridge,
    validate_expert_rows,
)
from .engineering_contracts import (
    DESCRIPTION_BUILDER_VERSION,
    REGION_INPUT_SOURCE_PROTOCOL,
    require_engineering_bridge,
    require_engineering_description,
    validate_predicted_index,
)
from .sampling import (
    caption_source_weights,
    cross_parent_region_swap_candidates,
    description_row_sample_id,
    end_to_end_region_support,
    read_jsonl_rows,
    same_parent_region_swap_candidates,
    select_d_minus_one_mixture,
    stable_subset,
    stable_weighted_index,
)
from .records import (
    append_fraction,
    bridge_region_metadata,
    filter_evaluation_region_source,
    filter_evaluation_source,
    structured_text,
)


DescriptionStage = Literal[
    "overfit", "mmrs_caption", "rsicap_caption", "dior_alignment",
    "bridge_auto", "bridge_expert", "predicted_mask",
]



class DescriptionTaskDataset(Dataset):
    """One task family per dataset instance; joint training uses separate DataLoaders."""

    def __init__(
        self,
        *,
        stage: DescriptionStage,
        split: str,
        vision_bank: DescriptionVisionFeatureBank,
        description_benchmark: str | Path,
        bridge_benchmark: str | Path,
        predicted_index: str | Path | None = None,
        seed: int = 42,
        max_samples: int = 0,
        training: bool = False,
        evaluation_mode: str = "gt_mask",
        evaluation_source_dataset: str | None = None,
        evaluation_region_source: str | None = None,
        rsicap_mmrs_fraction: float = 0.30,
        predicted_mask_fraction: float = 0.25,
        d4_curriculum_sampling_seed: int = 42,
    ) -> None:
        self.stage = stage
        self.split = split
        self.vision_bank = vision_bank
        self.seed = int(seed)
        self.d4_curriculum_sampling_seed = int(d4_curriculum_sampling_seed)
        self.epoch = 0
        self.training = bool(training)
        self.evaluation_mode = str(evaluation_mode)
        self.source_filter_audit: dict[str, Any] | None = None
        self.region_source_filter_audit: dict[str, Any] | None = None
        self.end_to_end_exclusion_counts: Counter[str] = Counter()
        self.end_to_end_source_count = 0
        self.end_to_end_eligible_count = 0
        self.predicted_index_audit: dict[str, Any] | None = None
        self.d_minus_one_sampling_audit: dict[str, Any] | None = None
        self.bridge_engineering_audit: dict[str, Any] | None = None
        self.description_engineering_audit: dict[str, Any] | None = None
        self._verified_mask_hashes: dict[
            tuple[str, str, str, int, int, int], tuple[str, str]
        ] = {}
        description_dir = resolve_project_path(description_benchmark)
        bridge_dir = resolve_project_path(bridge_benchmark)
        if description_dir is None or bridge_dir is None:
            raise ValueError("description/bridge benchmark 路径不能为空")
        if stage in {
            "bridge_auto", "bridge_expert", "predicted_mask", "overfit",
        }:
            self.bridge_engineering_audit = require_engineering_bridge(
                bridge_dir, vision_bank
            )
        self.expert_gate_audit = (
            require_frozen_expert_bridge(bridge_dir)
            if stage in {"bridge_expert", "predicted_mask"} else None
        )
        if stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}:
            self.description_engineering_audit = (
                require_engineering_description(description_dir, vision_bank)
            )
            index_name = "train_eligible.jsonl" if split == "train" else f"{split}.jsonl"
            rows = read_jsonl_rows(description_dir / f"indexes/{index_name}")
            if stage == "mmrs_caption":
                rows = [
                    row for row in rows
                    if row["task_family"] == "global_caption" and str(row["source_dataset"]).startswith("MMRS-")
                ]
            elif stage == "rsicap_caption":
                rsicap_rows = [
                    row for row in rows
                    if row["task_family"] == "global_caption" and row["source_dataset"] in {"RSICap", "RSIEval"}
                ]
                if self.training and split == "train":
                    mmrs_rows = [
                        row for row in read_jsonl_rows(description_dir / "indexes/train_eligible.jsonl")
                        if row["task_family"] == "global_caption"
                        and str(row["source_dataset"]).startswith("MMRS-")
                    ]
                    rows = append_fraction(
                        rsicap_rows, mmrs_rows, rsicap_mmrs_fraction,
                        seed=self.seed, namespace="d1_rsicap_mmrs",
                    )
                else:
                    rows = rsicap_rows
            else:
                rows = [row for row in rows if row["task_family"] == "region_referring_expression"]
        elif stage == "bridge_auto":
            rows = read_jsonl_rows(bridge_dir / "indexes/auto_train.jsonl")
            if split != "train":
                rows = []
        elif stage == "overfit":
            self.description_engineering_audit = (
                require_engineering_description(description_dir, vision_bank)
            )
            description_index = description_dir / "indexes/train_eligible.jsonl"
            bridge_index = bridge_dir / "indexes/candidate_all.jsonl"
            description_report_path = description_dir / "reports/validation_report.json"
            bridge_report_path = bridge_dir / "reports/validation_report.json"
            for report_path in (description_report_path, bridge_report_path):
                if not report_path.is_file():
                    raise FileNotFoundError(f"D-1 缺少 benchmark validation report: {report_path}")
            description_report = strict_json_loads(
                description_report_path.read_text(encoding="utf-8")
            )
            bridge_report = strict_json_loads(
                bridge_report_path.read_text(encoding="utf-8")
            )
            if (
                description_report.get("builder_version")
                != DESCRIPTION_BUILDER_VERSION
                or description_report.get("errors")
            ):
                raise RuntimeError(
                    "D-1 要求 engineering-valid Description M1.1 v4 benchmark"
                )
            if (
                bridge_report.get("builder_version") != BRIDGE_BUILDER_VERSION
                or bridge_report.get("status") not in {
                    "awaiting_expert_review", "expert_pilot_frozen",
                }
                or bridge_report.get("errors")
            ):
                raise RuntimeError(
                    "D-1 要求当前 M2 v7 Bridge prepare/frozen artifact；"
                    "awaiting_expert_review 可用于 candidate 工程过拟合，"
                    "但旧 builder 或有错误的 artifact 不可使用"
                )
            description_rows = read_jsonl_rows(description_index)
            bridge_rows = read_jsonl_rows(bridge_index)
            if int(max_samples) not in {0, 64}:
                raise ValueError(
                    "D-1 overfit 当前协议固定选择 64 条样本；"
                    f"observed max_samples={max_samples}"
                )
            requested = 64
            rows, self.d_minus_one_sampling_audit = select_d_minus_one_mixture(
                description_rows,
                bridge_rows,
                count=requested,
                seed=self.seed,
            )
            self.d_minus_one_sampling_audit.update({
                "bridge_engineering_audit": self.bridge_engineering_audit,
                "description_builder_version": description_report.get(
                    "builder_version"
                ),
                "description_index": str(description_index),
                "description_index_sha256": sha256_file(description_index),
                "description_validation_report": str(description_report_path),
                "description_validation_report_sha256": sha256_file(
                    description_report_path
                ),
                "bridge_builder_version": bridge_report.get("builder_version"),
                "bridge_status": bridge_report.get("status"),
                "bridge_index": str(bridge_index),
                "bridge_index_sha256": sha256_file(bridge_index),
                "bridge_validation_report": str(bridge_report_path),
                "bridge_validation_report_sha256": sha256_file(
                    bridge_report_path
                ),
            })
        elif stage == "bridge_expert":
            path = bridge_dir / f"indexes/expert_{split}.jsonl"
            rows = read_jsonl_rows(path)
        elif stage == "predicted_mask":
            if predicted_index is None:
                raise ValueError("predicted_mask stage 需要独立离线 --predicted-index")
            path = resolve_project_path(predicted_index)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"predicted index 不存在: {predicted_index}")
            self.predicted_index_audit = validate_predicted_index(
                path,
                split=split,
                expert_gate_audit=dict(self.expert_gate_audit or {}),
            )
            rows = read_jsonl_rows(path)
            rows = [row for row in rows if row.get("split") == split]
            if self.training and split == "train":
                expert_path = bridge_dir / "indexes/expert_train.jsonl"
                if not expert_path.is_file():
                    raise FileNotFoundError(
                        "D4 GT/predicted curriculum 需要已冻结 indexes/expert_train.jsonl"
                    )
                expert_rows = read_jsonl_rows(expert_path)
                requested_predicted = round(
                    len(expert_rows) * float(predicted_mask_fraction)
                    / max(1.0 - float(predicted_mask_fraction), 1.0e-8)
                )
                if requested_predicted > len(rows):
                    raise RuntimeError(
                        "D4 predicted index 不足以实现预注册 curriculum tier: "
                        f"requested={requested_predicted} available={len(rows)} "
                        f"fraction={predicted_mask_fraction}"
                    )
                rows = expert_rows + stable_subset(
                    rows,
                    requested_predicted,
                    self.d4_curriculum_sampling_seed,
                    "d4_predicted_masks",
                )
        else:
            raise ValueError(f"未知 description stage={stage!r}")
        rows, self.source_filter_audit = filter_evaluation_source(
            rows,
            stage=stage,
            split=split,
            source_dataset=evaluation_source_dataset,
        )
        rows, self.region_source_filter_audit = filter_evaluation_region_source(
            rows,
            stage=stage,
            split=split,
            training=self.training,
            evaluation_mode=self.evaluation_mode,
            region_source=evaluation_region_source,
        )
        if self.evaluation_mode == "end_to_end":
            if stage != "bridge_expert":
                raise ValueError("end_to_end evaluation 只支持 bridge_expert stage")
            self.end_to_end_source_count = len(rows)
            supported_rows = []
            for row in rows:
                supported, reason = end_to_end_region_support(row)
                if supported:
                    supported_rows.append(row)
                else:
                    self.end_to_end_exclusion_counts[reason] += 1
            rows = supported_rows
            self.end_to_end_eligible_count = len(rows)
        if stage in {"bridge_expert", "predicted_mask"}:
            validate_expert_rows(rows, stage=stage, split=split)
        if stage != "overfit":
            rows.sort(key=lambda row: str(
                row.get("sample_id") or row.get("bridge_record_id") or ""
            ))
        if max_samples > 0 and stage != "overfit":
            rows = stable_subset(rows, max_samples, self.seed, f"{stage}:{split}:limit")
            rows.sort(key=lambda row: str(
                row.get("sample_id") or row.get("bridge_record_id") or ""
            ))
        self.rows = rows
        self._rows_by_sample_id: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            sample_id = description_row_sample_id(row)
            if sample_id in self._rows_by_sample_id:
                raise ValueError(f"description dataset sample_id 重复: {sample_id}")
            self._rows_by_sample_id[sample_id] = row
        self._region_swap_catalog = self.rows
        if stage in {"bridge_auto", "bridge_expert", "predicted_mask", "overfit"}:
            candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
            if candidate_path.is_file():
                self._region_swap_catalog = [
                    row for row in read_jsonl_rows(candidate_path)
                    if str(row.get("split") or "") == self.split
                ]
        self._request_family_cache: dict[tuple[str, str], set[str]] = {}
        (
            self._caption_source_weight_by_sample,
            self.caption_sampling_audit,
        ) = caption_source_weights(
            self.rows,
            stage=self.stage,
            rsicap_mmrs_fraction=rsicap_mmrs_fraction,
        )
        predicted_rows = [
            row for row in self.rows
            if (
                str(row.get("region_source") or "") == "predicted_proposal"
                or str(row.get("schema_version") or "").startswith(
                    "qpsalm_predicted_region"
                )
            )
        ]
        self.curriculum_audit = (
            {
                "protocol": "qpsalm_d4_predicted_mask_curriculum_v1",
                "requested_predicted_fraction": float(predicted_mask_fraction),
                "selection_seed": self.d4_curriculum_sampling_seed,
                "num_total": len(self.rows),
                "num_predicted": len(predicted_rows),
                "num_gt": len(self.rows) - len(predicted_rows),
                "realized_predicted_fraction": (
                    len(predicted_rows) / len(self.rows) if self.rows else 0.0
                ),
                "training_mix": bool(self.training and self.split == "train"),
            }
            if self.stage == "predicted_mask" else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def same_parent_region_swap(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Load and identify another real region from the same image."""
        candidates = same_parent_region_swap_candidates(
            self.rows,
            sample_id,
            catalog=self._region_swap_catalog,
        )
        current = reference_mask.detach().cpu()
        for row in candidates:
            alternate = self._counterfactual_region_mask(row).detach().cpu()
            if alternate.shape != current.shape:
                continue
            if not torch.equal(alternate, current):
                return alternate, {
                    "protocol": "qpsalm_same_parent_region_swap_v1",
                    "parent_sample_id": str(row["parent_sample_id"]),
                    "alternate_sample_id": description_row_sample_id(row),
                    "alternate_region_id": str(row.get("region_id") or "unknown"),
                    "alternate_region_source": str(
                        row.get("region_source") or "region_geometry"
                    ),
                    "alternate_mask_path": (row.get("region_mask") or {}).get("path"),
                }
        return None

    def cross_parent_region_swap(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Load one real region from a deterministic, different parent."""
        current_row = self._rows_by_sample_id.get(str(sample_id))
        if current_row is None:
            return None
        target_parent = str(current_row.get("parent_sample_id") or "")
        candidates = cross_parent_region_swap_candidates(
            self.rows,
            sample_id,
            catalog=self._region_swap_catalog,
        )
        current = reference_mask.detach().cpu()
        for row in candidates:
            donor_parent = str(row.get("parent_sample_id") or "")
            if not donor_parent or donor_parent == target_parent:
                continue
            alternate = self._counterfactual_region_mask(row).detach().cpu()
            if alternate.shape != current.shape or torch.equal(alternate, current):
                continue
            return alternate, {
                "protocol": "qpsalm_cross_parent_region_swap_v1",
                "target_parent_sample_id": target_parent,
                "donor_parent_sample_id": donor_parent,
                "donor_sample_id": description_row_sample_id(row),
                "donor_region_id": str(row.get("region_id") or "unknown"),
                "donor_region_source": str(
                    row.get("region_source") or "region_geometry"
                ),
                "donor_mask_path": (row.get("region_mask") or {}).get("path"),
            }
        return None

    def same_parent_region_swap_mask(
        self,
        sample_id: str,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compatibility convenience returning only the resolved mask."""
        resolved = self.same_parent_region_swap(sample_id, reference_mask)
        return resolved[0] if resolved is not None else None

    def _counterfactual_region_mask(self, row: dict[str, Any]) -> torch.Tensor:
        """Load donor geometry without consuming or requiring its text target."""
        if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}:
            return self._description_item(row)["region_mask"]
        return self._bridge_region_mask(row)

    def _request_for_row(self, row: dict[str, Any]) -> tuple[str, str]:
        component = (
            "single_image"
            if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
            or row.get("_d_minus_one_item_kind") == "description"
            else "multisource_parent"
        )
        return component, str(row["parent_sample_id"])

    def _request_families(self, request: tuple[str, str]) -> set[str]:
        cached = self._request_family_cache.get(request)
        if cached is not None:
            return cached
        record = self.vision_bank.record(*request)
        families = set()
        for view in record["views"]:
            source = [str(value) for value in view.get("source_families") or []]
            family = (
                source[0]
                if source and len(set(source)) == 1 and source[0] in MODALITY_FAMILY_IDS
                else "unknown"
            )
            families.add(family)
        self._request_family_cache[request] = families
        return families

    def cross_parent_modality_swap_request(
        self,
        sample_id: str,
    ) -> tuple[tuple[str, str], dict[str, Any]] | None:
        """Select a deterministic donor parent sharing at least one view family."""
        current = self._rows_by_sample_id.get(str(sample_id))
        if current is None:
            return None
        current_parent = str(current["parent_sample_id"])
        current_request = self._request_for_row(current)
        current_families = self._request_families(current_request)
        seen_parents = {current_parent}
        for row in sorted(self.rows, key=description_row_sample_id):
            donor_parent = str(row.get("parent_sample_id") or "")
            if not donor_parent or donor_parent in seen_parents:
                continue
            seen_parents.add(donor_parent)
            donor_request = self._request_for_row(row)
            common = sorted(current_families & self._request_families(donor_request))
            if not common:
                continue
            return donor_request, {
                "protocol": "qpsalm_cross_parent_modality_donor_v1",
                "target_parent_sample_id": current_parent,
                "donor_parent_sample_id": donor_parent,
                "common_modality_families": common,
            }
        return None

    def _description_item(self, row: dict[str, Any]) -> dict[str, Any]:
        answers = [
            answer for answer in row.get("answers", [])
            if self.split != "train" or float(answer.get("caption_quality_weight", 1.0)) > 0
        ]
        if not answers:
            raise ValueError(f"description record 没有可训练 answer: {row['sample_id']}")
        answer = answers[stable_weighted_index(
            self.seed,
            self.epoch,
            str(row["sample_id"]),
            [float(value.get("caption_quality_weight", 1.0)) for value in answers],
        )]
        visual = row["visual_ref"]
        width, height = int(visual["width"]), int(visual["height"])
        geometry = row["region_geometry"]
        source_mask = torch.zeros((1, height, width), dtype=torch.float32)
        if geometry["type"] == "full_image":
            source_mask.fill_(1.0)
        elif geometry["type"] == "box":
            x1, y1, x2, y2 = [int(value) for value in geometry["bbox_xyxy_pixel_half_open"]]
            source_mask[:, y1:y2, x1:x2] = 1.0
        elif geometry["type"] not in {"null"}:
            raise ValueError(f"M1 record region type 暂不支持: {geometry['type']}")
        cache = self.vision_bank.record("single_image", str(row["parent_sample_id"]))
        region = transform_region_mask_to_cache(source_mask, cache["views"][0]["render_transform"])
        return {
            "request": ("single_image", str(row["parent_sample_id"])),
            "region_mask": region,
            # 输出格式与视觉路由是两个独立契约。DIOR box 目标仍是自由文本，
            # 但 D-1/D2 必须消费其真实区域，不能退化为全图 caption。
            "use_region_tokens": geometry["type"] != "full_image",
            "instruction": str(row["instruction"]),
            "target_text": str(answer["text"]),
            "reference_texts": [str(value["text"]) for value in answers],
            "structured_output": False,
            "weight": (
                float(answer.get("caption_quality_weight", 1.0))
                * float(self._caption_source_weight_by_sample.get(
                    str(row["sample_id"]), 1.0
                ))
            ),
            "sample_id": str(row["sample_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "task_family": str(row["task_family"]),
            "target_status": str(row.get("target_status") or "present"),
            "source_dataset": str(row.get("source_dataset") or "unknown"),
            "visual_image_path": str(visual.get("path") or ""),
            "region_pair_id": row.get("region_pair_id"),
        }

    def _bridge_region_mask_and_binding(
        self, row: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Materialize Bridge geometry and bind its deterministic cache projection."""
        cache = self.vision_bank.record("multisource_parent", str(row["parent_sample_id"]))
        transform = dict(cache["views"][0]["render_transform"])
        source_binding: dict[str, Any]
        if row.get("region_mask"):
            mask_ref = row["region_mask"]
            path = resolve_project_path(mask_ref["path"])
            if path is None or not path.is_file():
                raise FileNotFoundError(f"region mask 不存在: {mask_ref.get('path')}")
            expected_hash = str(mask_ref.get("sha256") or "")
            if self.stage == "predicted_mask" and len(expected_hash) != 64:
                raise ValueError("predicted-mask row 缺少 region_mask.sha256")
            values = np.load(path, allow_pickle=False)
            expected_shape = tuple(int(value) for value in (mask_ref.get("shape") or []))
            if expected_shape and tuple(values.shape) != expected_shape:
                raise ValueError(
                    f"region mask shape 不一致: observed={tuple(values.shape)} "
                    f"expected={expected_shape}"
                )
            if values.ndim != 2 or not np.isin(values, (0, 1)).all():
                raise ValueError("region mask 必须是二维 binary array")

            # Bridge v7 绑定解码后的 mask 像素；D4 prediction 绑定 NPY 文件本身。
            hash_kind = (
                "file_sha256"
                if self.stage == "predicted_mask"
                else "bridge_binary_content_sha256"
            )
            stat = path.stat()
            key = (
                str(path.resolve(strict=False)),
                hash_kind,
                expected_hash,
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
            cached_hashes = self._verified_mask_hashes.get(key)
            if cached_hashes is None:
                file_hash = sha256_file(path)
                observed_hash = (
                    file_hash
                    if self.stage == "predicted_mask"
                    else bridge_region_mask_digest(values)
                )
                cached_hashes = (observed_hash, file_hash)
                self._verified_mask_hashes[key] = cached_hashes
            observed_hash, file_hash = cached_hashes
            if expected_hash and observed_hash != expected_hash:
                raise ValueError(
                    f"region mask hash ({hash_kind}) 不一致: {path}"
                )

            values = values[None]
            source_mask = torch.from_numpy((values > 0).astype(np.float32))
            source_binding = {
                "kind": "binary_npy",
                "path": str(mask_ref["path"]),
                "file_sha256": file_hash,
                "bytes": int(path.stat().st_size),
                "shape": list(source_mask.shape[-2:]),
                "positive_pixels": int(source_mask.sum().item()),
            }
        else:
            source_h = int(transform["source_h"])
            source_w = int(transform["source_w"])
            source_mask = torch.zeros((1, source_h, source_w), dtype=torch.float32)
            source_binding = {
                "kind": "null",
                "path": None,
                "file_sha256": None,
                "bytes": 0,
                "shape": [source_h, source_w],
                "positive_pixels": 0,
            }
        projected, source_mapping = project_native_region_mask_to_cache(
            source_mask, transform
        )
        lookup_key = str(cache.get("lookup_key") or description_cache_key(
            "multisource_parent", str(row["parent_sample_id"])
        ))
        cache_fingerprint = str(cache.get("cache_fingerprint") or hashlib.sha256(
            json.dumps(
                {"lookup_key": lookup_key, "render_transform": transform},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest())
        return projected, {
            "protocol": REGION_INPUT_SOURCE_PROTOCOL,
            "sample_id": str(row["bridge_record_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "region_id": str(row.get("region_id") or "unknown"),
            "region_source": str(row.get("region_source") or "unknown"),
            "cache_lookup_key": lookup_key,
            "cache_fingerprint": cache_fingerprint,
            "render_transform": transform,
            "source_to_render_mapping": source_mapping,
            "source_mask": source_binding,
        }

    def _bridge_region_mask(self, row: dict[str, Any]) -> torch.Tensor:
        """Materialize only Bridge geometry; counterfactual donors need no text."""
        return self._bridge_region_mask_and_binding(row)[0]

    def _bridge_item(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.stage in {"bridge_expert", "predicted_mask"} and not isinstance(
            row.get("expert_target"), dict
        ):
            raise ValueError(
                "expert/predicted-mask row 缺少 expert_target；禁止回退到 candidate"
            )
        region, region_source_binding = self._bridge_region_mask_and_binding(row)
        # Predicted-mask rows inherit the reviewed target from their source row.
        # Falling back to the deterministic candidate here would make fixed-mask
        # evaluation measure a different target than the GT-mask expert run.
        expert = self.stage in {"bridge_expert", "predicted_mask"}
        return {
            "request": ("multisource_parent", str(row["parent_sample_id"])),
            "region_mask": region,
            "use_region_tokens": True,
            "instruction": str(row["instruction"]),
            "target_text": structured_text(row, expert=expert),
            "reference_texts": [structured_text(row, expert=expert)],
            "structured_output": True,
            "weight": 1.0,
            **bridge_region_metadata(row),
            "region_input_source_binding": region_source_binding,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.stage == "overfit":
            item = (
                self._description_item(row)
                if row.get("_d_minus_one_item_kind") == "description"
                else self._bridge_item(row)
            )
            item["d_minus_one_category"] = str(row["_d_minus_one_category"])
            item["d_minus_one_target_authority"] = str(
                row["_d_minus_one_target_authority"]
            )
            return item
        return (
            self._description_item(row)
            if self.stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}
            else self._bridge_item(row)
        )


def collate_description(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("description batch 不能为空")
    shapes = {tuple(item["region_mask"].shape) for item in items}
    if len(shapes) != 1:
        raise ValueError(f"description region canvas 必须一致: {sorted(shapes)}")
    return {
        # 普通 tuple 会被 CUDA DataLoader.pin_memory 递归转换为 list。
        # 在 collator 边界直接采用稳定的二元素 list，避免 preflight 与
        # 正式 trainer 对同一 cache request 观察到不同的容器类型。
        "requests": [list(item["request"]) for item in items],
        "region_masks": torch.stack([item["region_mask"] for item in items]),
        "instructions": [item["instruction"] for item in items],
        "target_texts": [item["target_text"] for item in items],
        "reference_texts": [item["reference_texts"] for item in items],
        "structured_outputs": [bool(item["structured_output"]) for item in items],
        "use_region_tokens": [bool(item["use_region_tokens"]) for item in items],
        "weights": torch.tensor([float(item["weight"]) for item in items], dtype=torch.float32),
        "metadata": [{
            key: item[key]
            for key in (
                "sample_id", "parent_sample_id", "task_family", "target_status",
                "source_dataset", "region_pair_id", "region_id", "region_source",
                "source_region_aliases", "region_mask_path", "expert_review_panel_path",
                "visual_preview_path", "multimodal_preview_path", "visual_image_path",
                "has_unavailable_modality",
                "region_input_source_binding",
                "d_minus_one_category", "d_minus_one_target_authority",
            )
            if key in item
        } for item in items],
    }
