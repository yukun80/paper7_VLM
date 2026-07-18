#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded reader and deep artifact replay for Description Vision Cache v1."""

from __future__ import annotations

from collections import Counter, OrderedDict
import json
from pathlib import Path
from typing import Any, Callable

import torch

from qpsalm_seg.paths import resolve_project_path

from ..protocols.cache import description_cache_key
from ..protocols.io import canonical_sha256, strict_json_loads
from .vision_cache_contracts import (
    DESCRIPTION_CACHE_ARTIFACT_BINDING_PROTOCOL,
    DESCRIPTION_CACHE_ARTIFACT_REVALIDATION_PROTOCOL,
    DESCRIPTION_CACHE_BUILDER_VERSION,
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL,
    DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
    sha256_file,
    validate_description_cache_record,
)


# 同一进程内只有目录元数据未变化时才复用全 shard 深验证结果。
_VERIFIED_DESCRIPTION_CACHE_ARTIFACTS: dict[str, dict[str, Any]] = {}


class DescriptionVisionFeatureBank:
    """Strict sharded reader that never stores instruction or region state."""

    def __init__(
        self,
        cache_dir: str | Path,
        max_open_shards: int = 8,
        *,
        require_validation_report: bool = True,
    ) -> None:
        path = resolve_project_path(cache_dir) or Path(cache_dir)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"description vision cache manifest 不存在: {manifest_path}")
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"description vision cache manifest 顶层必须为 object: {manifest_path}")
        self._validate_manifest(manifest, manifest_path)
        self.cache_dir = path
        self.manifest_path = manifest_path
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = manifest
        self.lookup = manifest["lookup"]
        self.shards = tuple(str(value) for value in manifest["shards"])
        self.shard_fingerprints = tuple(
            dict(value) for value in manifest["shard_fingerprints"]
        )
        missing = [name for name in self.shards if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"description vision cache 缺少 shards: {missing[:8]}")
        self.max_open_shards = max(1, int(max_open_shards))
        self._loaded: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._verified_shards: set[int] = set()
        self.validation_report_path = path / "validation_report.json"
        self.validation_report: dict[str, Any] | None = None
        if require_validation_report:
            if not self.validation_report_path.is_file():
                raise FileNotFoundError(
                    "description vision cache 缺少深度 validation report: "
                    f"{self.validation_report_path}"
                )
            try:
                validation_report = strict_json_loads(
                    self.validation_report_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "description vision cache validation report 不是合法 JSON: "
                    f"{self.validation_report_path}"
                ) from exc
            self._validate_validation_report(
                validation_report,
                self.validation_report_path,
                manifest,
                self.manifest_sha256,
            )
            self.validation_report = validation_report

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
        required = {
            "format", "protocol", "builder_version", "model_revision", "processor_revision",
            "layers", "spatial_sizes", "view_tokens_per_view", "spatial_channels", "token_dim",
            "backend", "input_fingerprints", "num_samples", "components", "lookup", "shards",
            "shard_size", "renderer_version", "render_size", "forbidden_state",
            "source_cache_provenance",
            "shard_fingerprints",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"description vision cache manifest 缺少 {missing}: {path}")
        if manifest["format"] != DESCRIPTION_CACHE_FORMAT:
            raise ValueError(f"只支持 {DESCRIPTION_CACHE_FORMAT}: {path}")
        if manifest["protocol"] != DESCRIPTION_CACHE_PROTOCOL:
            raise ValueError(f"description cache protocol 不匹配: {manifest['protocol']!r}")
        if manifest["builder_version"] != DESCRIPTION_CACHE_BUILDER_VERSION:
            raise ValueError(
                "description cache builder_version 过期: "
                f"{manifest['builder_version']!r} != {DESCRIPTION_CACHE_BUILDER_VERSION!r}"
            )
        if list(manifest["layers"]) != [5, 11, 17, 23]:
            raise ValueError("description cache layers 必须为 [5,11,17,23]")
        if int(manifest["num_samples"]) <= 0:
            raise ValueError("description cache num_samples 必须为正整数")
        if int(manifest["num_samples"]) != len(manifest["lookup"]):
            raise ValueError("description cache lookup/sample 数量不一致")
        if not str(manifest.get("renderer_version") or "").strip() or int(manifest["render_size"]) <= 0:
            raise ValueError("description cache renderer metadata 非法")
        components = list(manifest["components"])
        if not components or len(components) != len(set(components)):
            raise ValueError(f"description cache components 非法: {components}")
        if set(components) - {"single_image", "multisource_parent"}:
            raise ValueError(f"description cache component 非法: {manifest['components']}")
        inputs = manifest["input_fingerprints"]
        if not isinstance(inputs, dict) or set(inputs) != set(components):
            raise ValueError("description cache input_fingerprints 与 components 不一致")
        required_input_fields = {
            "benchmark", "index", "size", "sha256",
            "validation_report", "validation_report_size",
            "validation_report_sha256", "validation_builder_version",
            "validation_status",
        }
        for component, fingerprint in inputs.items():
            if (
                not isinstance(fingerprint, dict)
                or required_input_fields - set(fingerprint)
            ):
                raise ValueError(f"description cache 输入指纹不完整: {component}")
            if int(fingerprint["size"]) <= 0:
                raise ValueError(f"description cache 输入 size 非法: {component}")
            if not isinstance(fingerprint["sha256"], str) or len(fingerprint["sha256"]) != 64:
                raise ValueError(f"description cache 输入 sha256 非法: {component}")
            if int(fingerprint["validation_report_size"]) <= 0:
                raise ValueError(
                    f"description cache validation report size 非法: {component}"
                )
            if (
                not isinstance(fingerprint["validation_report_sha256"], str)
                or len(fingerprint["validation_report_sha256"]) != 64
                or not str(fingerprint["validation_report"] or "").strip()
                or not str(
                    fingerprint["validation_builder_version"] or ""
                ).strip()
                or not str(fingerprint["validation_status"] or "").strip()
            ):
                raise ValueError(
                    f"description cache validation report 指纹非法: {component}"
                )
        if set(manifest["forbidden_state"]) != {
            "instruction", "condition", "region_geometry", "segmentation_state",
        }:
            raise ValueError("description cache forbidden_state 协议不完整")
        source = manifest["source_cache_provenance"]
        if not isinstance(source, dict) or {
            "provided", "path", "manifest_sha256", "metadata_fingerprint", "file_count",
            "reused_records", "isolation_unchanged",
        } - set(source):
            raise ValueError("description cache source_cache_provenance 不完整")
        if bool(source["provided"]):
            if not bool(source["isolation_unchanged"]):
                raise ValueError("description cache 构建时修改了源 segmentation cache")
            if not str(source.get("path") or "").strip():
                raise ValueError("description cache 缺少源 segmentation cache path")
            for key in ("manifest_sha256", "metadata_fingerprint"):
                if not isinstance(source.get(key), str) or len(source[key]) != 64:
                    raise ValueError(f"description cache source {key} 非法")
            if int(source.get("file_count") or 0) <= 0:
                raise ValueError("description cache source file_count 非法")
        elif any(source.get(key) is not None for key in (
            "path", "manifest_sha256", "metadata_fingerprint", "file_count",
        )):
            raise ValueError("未使用源 segmentation cache 时 provenance 必须为空")
        if int(source.get("reused_records") or 0) < 0:
            raise ValueError("description cache source reused_records 非法")
        if not bool(source["provided"]) and int(source.get("reused_records") or 0) != 0:
            raise ValueError("未使用源 segmentation cache 时 reused_records 必须为 0")
        shard_size = int(manifest["shard_size"])
        if shard_size <= 0:
            raise ValueError("description cache shard_size 必须为正整数")
        expected_shards = (
            int(manifest["num_samples"]) + shard_size - 1
        ) // shard_size
        if len(manifest["shards"]) != expected_shards:
            raise ValueError(
                "description cache shard 数量不一致: "
                f"expected={expected_shards} actual={len(manifest['shards'])}"
            )
        if len(manifest["shards"]) != len(set(manifest["shards"])):
            raise ValueError("description cache shard 名称重复")
        if any(Path(str(name)).name != str(name) for name in manifest["shards"]):
            raise ValueError("description cache shard 必须是 cache 根目录内的文件名")
        fingerprints = manifest["shard_fingerprints"]
        if not isinstance(fingerprints, list) or len(fingerprints) != len(
            manifest["shards"]
        ):
            raise ValueError("description cache shard_fingerprints 数量不一致")
        fingerprint_paths = []
        total_fingerprint_records = 0
        for index, (name, fingerprint) in enumerate(
            zip(manifest["shards"], fingerprints)
        ):
            if not isinstance(fingerprint, dict) or {
                "path", "size", "records", "sha256",
            } - set(fingerprint):
                raise ValueError(f"description cache shard fingerprint 不完整: {name}")
            if fingerprint["path"] != name:
                raise ValueError(
                    f"description cache shard fingerprint 顺序/路径不一致: {name}"
                )
            fingerprint_paths.append(str(fingerprint["path"]))
            expected_records = min(
                shard_size,
                int(manifest["num_samples"]) - index * shard_size,
            )
            if int(fingerprint["records"]) != expected_records:
                raise ValueError(
                    "description cache shard record 数不一致: "
                    f"{name} expected={expected_records} "
                    f"actual={fingerprint['records']}"
                )
            total_fingerprint_records += int(fingerprint["records"])
            if int(fingerprint["size"]) <= 0:
                raise ValueError(f"description cache shard size 非法: {name}")
            if (
                not isinstance(fingerprint["sha256"], str)
                or len(fingerprint["sha256"]) != 64
            ):
                raise ValueError(f"description cache shard sha256 非法: {name}")
        if len(fingerprint_paths) != len(set(fingerprint_paths)):
            raise ValueError("description cache shard fingerprint 路径重复")
        if total_fingerprint_records != int(manifest["num_samples"]):
            raise ValueError("description cache shard fingerprint record 总数不一致")
        for key, location in manifest["lookup"].items():
            if not isinstance(location, dict):
                raise ValueError(f"description cache lookup 非法: {key}")
            if not {"shard", "index", "component", "parent_sample_id"} <= set(location):
                raise ValueError(f"description cache lookup 不完整: {key}")
            expected_key = description_cache_key(
                str(location.get("component")), str(location.get("parent_sample_id"))
            )
            if str(key) != expected_key:
                raise ValueError(f"description cache key 非法: {key}")
            shard_index = int(location["shard"])
            local_index = int(location["index"])
            if not 0 <= shard_index < len(manifest["shards"]):
                raise ValueError(f"description cache lookup shard 越界: {key}")
            if not 0 <= local_index < shard_size:
                raise ValueError(f"description cache lookup index 越界: {key}")

    @staticmethod
    def _validate_validation_report(
        report: Any,
        path: Path,
        manifest: dict[str, Any],
        manifest_sha256: str,
    ) -> None:
        """Require the exact successful deep report produced for this manifest."""
        if not isinstance(report, dict):
            raise ValueError(
                f"description cache validation report 顶层必须为 object: {path}"
            )
        expected = {
            "protocol": DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
            "format": DESCRIPTION_CACHE_FORMAT,
            "cache_protocol": DESCRIPTION_CACHE_PROTOCOL,
            "builder_version": DESCRIPTION_CACHE_BUILDER_VERSION,
            "manifest_sha256": manifest_sha256,
            "num_records": int(manifest["num_samples"]),
            "num_shards": len(manifest["shards"]),
            "input_fingerprints": manifest["input_fingerprints"],
        }
        for field, value in expected.items():
            if report.get(field) != value:
                raise ValueError(
                    "description cache validation report 与 manifest 不一致: "
                    f"field={field} expected={value!r} observed={report.get(field)!r}"
                )
        if (
            report.get("status") != "valid"
            or report.get("errors") != []
            or int(report.get("num_errors", -1)) != 0
        ):
            raise ValueError(
                "description cache validation report 未通过: "
                f"status={report.get('status')!r} errors={report.get('errors')!r}"
            )
        if not isinstance(report.get("warnings"), list):
            raise ValueError("description cache validation report warnings 必须为 list")
        if int(report.get("num_warnings", -1)) != len(report["warnings"]):
            raise ValueError("description cache validation report warning 数量不一致")

        integrity = report.get("shard_integrity")
        expected_shards = len(manifest["shards"])
        expected_bytes = sum(
            int(value["size"]) for value in manifest["shard_fingerprints"]
        )
        if not isinstance(integrity, dict) or (
            integrity.get("protocol") != "sha256_size_record_count_v1"
            or int(integrity.get("manifest_entries", -1)) != expected_shards
            or int(integrity.get("verified_shards", -1)) != expected_shards
            or int(integrity.get("verified_bytes", -1)) != expected_bytes
            or integrity.get("all_verified") is not True
        ):
            raise ValueError(
                "description cache validation report 未证明全部 shard 完整性"
            )
        records_by_component = report.get("records_by_component")
        if (
            not isinstance(records_by_component, dict)
            or set(records_by_component) != set(manifest["components"])
            or sum(int(value) for value in records_by_component.values())
            != int(manifest["num_samples"])
        ):
            raise ValueError(
                "description cache validation report component 统计与 manifest 不一致"
            )
        source = report.get("source_cache")
        provenance = manifest["source_cache_provenance"]
        reused = int(provenance.get("reused_records") or 0)
        if not isinstance(source, dict) or (
            source.get("provided") is not bool(provenance["provided"])
            or int(source.get("reused_records", -1)) != reused
            or int(source.get("validated_records", -1)) != reused
            or source.get("isolation_unchanged") is not True
        ):
            raise ValueError(
                "description cache validation report 未证明 source cache 隔离/record 绑定"
            )

    def artifact_binding(self) -> dict[str, Any]:
        """Return the exact cache identity embedded in checkpoints and formal reports."""
        if self.validation_report is None:
            raise RuntimeError(
                "未发布 validation_report 的 builder reader 不能生成 cache artifact binding"
            )
        current_manifest_sha256 = sha256_file(self.manifest_path)
        if current_manifest_sha256 != self.manifest_sha256:
            raise RuntimeError(
                "Description Vision Cache manifest 在 reader 初始化后发生变化"
            )
        try:
            current_report = strict_json_loads(
                self.validation_report_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Description Vision Cache validation report 在运行期间损坏"
            ) from exc
        self._validate_validation_report(
            current_report,
            self.validation_report_path,
            self.manifest,
            current_manifest_sha256,
        )
        manifest_stat = self.manifest_path.stat()
        report_stat = self.validation_report_path.stat()
        return {
            "protocol": DESCRIPTION_CACHE_ARTIFACT_BINDING_PROTOCOL,
            "cache_dir": str(self.cache_dir.resolve(strict=False)),
            "format": DESCRIPTION_CACHE_FORMAT,
            "cache_protocol": DESCRIPTION_CACHE_PROTOCOL,
            "builder_version": DESCRIPTION_CACHE_BUILDER_VERSION,
            "validation_protocol": DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
            "manifest": {
                "path": "manifest.json",
                "sha256": current_manifest_sha256,
                "bytes": int(manifest_stat.st_size),
            },
            "validation_report": {
                "path": "validation_report.json",
                "sha256": sha256_file(self.validation_report_path),
                "bytes": int(report_stat.st_size),
            },
            "num_records": int(self.manifest["num_samples"]),
            "num_shards": len(self.shards),
            "shard_inventory_sha256": canonical_sha256(
                list(self.shard_fingerprints)
            ),
            "input_fingerprints_sha256": canonical_sha256(
                self.manifest["input_fingerprints"]
            ),
            "source_cache_provenance_sha256": canonical_sha256(
                self.manifest["source_cache_provenance"]
            ),
        }

    def _file_metadata_snapshot(self) -> dict[str, Any]:
        entries = []
        for name in ("manifest.json", "validation_report.json", *self.shards):
            path = self.cache_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"description cache artifact 文件不存在: {path}")
            stat = path.stat()
            entries.append({
                "path": name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "inode": int(stat.st_ino),
            })
        return {
            "file_count": len(entries),
            "metadata_sha256": canonical_sha256(entries),
        }

    def file_metadata_snapshot(self) -> dict[str, Any]:
        """Return a cheap public identity for detecting post-preflight drift."""
        return self._file_metadata_snapshot()

    def verify_all_shards(self) -> dict[str, Any]:
        """Replay every manifest shard hash without loading all tensors into memory."""
        binding_before = self.artifact_binding()
        before = self._file_metadata_snapshot()
        self._verified_shards.clear()
        verified_bytes = 0
        for index, fingerprint in enumerate(self.shard_fingerprints):
            self._verify_shard_content(index)
            verified_bytes += int(fingerprint["size"])
        after = self._file_metadata_snapshot()
        binding_after = self.artifact_binding()
        if after != before or binding_after != binding_before:
            raise RuntimeError(
                "description cache 在全 shard provenance 重放期间发生变化"
            )
        return {
            "protocol": DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL,
            "all_verified": len(self._verified_shards) == len(self.shards),
            "verified_shards": len(self._verified_shards),
            "verified_bytes": verified_bytes,
            "metadata_snapshot": after,
        }

    def _verify_shard_content(self, index: int) -> None:
        if index in self._verified_shards:
            return
        expected = self.shard_fingerprints[index]
        path = self.cache_dir / self.shards[index]
        stat = path.stat()
        if int(stat.st_size) != int(expected["size"]):
            raise ValueError(
                "description cache shard size 不一致: "
                f"{path.name} expected={expected['size']} actual={stat.st_size}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != expected["sha256"]:
            raise ValueError(
                "description cache shard SHA-256 不一致: "
                f"{path.name} expected={expected['sha256']} actual={observed_hash}"
            )
        self._verified_shards.add(index)

    def _load_shard(
        self, index: int, *, verify_content: bool = True
    ) -> list[dict[str, Any]]:
        if verify_content:
            self._verify_shard_content(index)
        if index in self._loaded:
            rows = self._loaded.pop(index)
            self._loaded[index] = rows
            return rows
        path = self.cache_dir / self.shards[index]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != DESCRIPTION_CACHE_FORMAT:
            raise ValueError(f"description cache shard 损坏: {path}")
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError(f"description cache shard records 非法: {path}")
        self._loaded[index] = rows
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return rows

    def record(self, component: str, parent_sample_id: str) -> dict[str, Any]:
        key = description_cache_key(component, parent_sample_id)
        location = self.lookup.get(key)
        if location is None:
            raise KeyError(f"description vision cache 缺少 parent: {key}")
        row = self._load_shard(int(location["shard"]))[int(location["index"])]
        if row.get("lookup_key") != key:
            raise ValueError(f"description cache lookup/shard key 不一致: {key}")
        self._validate_record(row)
        return row

    def has(self, component: str, parent_sample_id: str) -> bool:
        return description_cache_key(component, parent_sample_id) in self.lookup

    def _validate_record(self, row: dict[str, Any]) -> None:
        validate_description_cache_record(row, self.manifest)

    def validate_all(
        self,
        *,
        expected_input_fingerprints: dict[str, dict[str, Any]] | None = None,
        source_record_fingerprint: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Deeply validate every shard position, record and optional cache-v3 binding."""
        errors: list[str] = []
        warnings: list[str] = []
        seen_keys: set[str] = set()
        seen_positions: set[tuple[int, int]] = set()
        by_component: Counter[str] = Counter()
        reused_records = 0
        validated_source_records = 0
        verified_shards = 0
        verified_shard_bytes = 0

        for shard_index, shard_name in enumerate(self.shards):
            try:
                self._verify_shard_content(shard_index)
            except (OSError, ValueError) as exc:
                errors.append(f"shard 内容完整性失败: {shard_name}: {exc}")
            else:
                verified_shards += 1
                verified_shard_bytes += int(
                    self.shard_fingerprints[shard_index]["size"]
                )
            try:
                # 即使内容 hash 失败仍读取结构，以便一次报告全部语义问题。
                rows = self._load_shard(shard_index, verify_content=False)
            except Exception as exc:
                errors.append(f"shard 无法读取: {shard_name}: {exc}")
                continue
            if len(rows) > int(self.manifest["shard_size"]):
                errors.append(f"shard 超过 shard_size: {shard_name} records={len(rows)}")
            for local_index, row in enumerate(rows):
                position = (shard_index, local_index)
                seen_positions.add(position)
                if not isinstance(row, dict):
                    errors.append(f"record 非 object: shard={shard_name} index={local_index}")
                    continue
                key = str(row.get("lookup_key") or "")
                if not key:
                    errors.append(f"record 缺少 lookup_key: shard={shard_name} index={local_index}")
                    continue
                if key in seen_keys:
                    errors.append(f"record lookup_key 重复: {key}")
                seen_keys.add(key)
                location = self.lookup.get(key)
                if location is None:
                    errors.append(f"shard record 未注册到 lookup: {key}")
                else:
                    actual = (int(location["shard"]), int(location["index"]))
                    if actual != position:
                        errors.append(
                            f"lookup/shard 位置不一致: {key} lookup={actual} actual={position}"
                        )
                    if str(location["component"]) != str(row.get("component")):
                        errors.append(f"lookup/record component 不一致: {key}")
                    if str(location["parent_sample_id"]) != str(row.get("parent_sample_id")):
                        errors.append(f"lookup/record parent_sample_id 不一致: {key}")
                expected_key = None
                try:
                    expected_key = description_cache_key(
                        str(row.get("component")), str(row.get("parent_sample_id"))
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                if expected_key is not None and key != expected_key:
                    errors.append(f"record lookup_key 与 component/parent 不一致: {key}")
                try:
                    self._validate_record(row)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"record 校验失败: {key}: {exc}")
                component = str(row.get("component") or "unknown")
                by_component[component] += 1
                source_key = row.get("source_cache")
                if source_key:
                    reused_records += 1
                    if source_record_fingerprint is not None:
                        try:
                            current = source_record_fingerprint(str(source_key))
                        except Exception as exc:
                            errors.append(f"源 cache record 无法读取: {source_key}: {exc}")
                        else:
                            validated_source_records += 1
                            if current != str(row.get("source_content_hash") or ""):
                                errors.append(
                                    f"源 cache record 指纹已变化: {source_key} "
                                    f"expected={row.get('source_content_hash')} current={current}"
                                )

        missing_keys = sorted(set(self.lookup) - seen_keys)
        if missing_keys:
            errors.append(
                "lookup 指向缺失 record: "
                f"count={len(missing_keys)} examples={missing_keys[:8]}"
            )
        orphan_positions = {
            (int(location["shard"]), int(location["index"]))
            for location in self.lookup.values()
        } - seen_positions
        if orphan_positions:
            errors.append(
                "lookup 指向不存在的 shard position: "
                f"count={len(orphan_positions)} examples={sorted(orphan_positions)[:8]}"
            )
        if len(seen_keys) != int(self.manifest["num_samples"]):
            errors.append(
                "description cache 实际 record 数与 manifest 不一致: "
                f"records={len(seen_keys)} manifest={self.manifest['num_samples']}"
            )
        if set(by_component) != set(self.manifest["components"]):
            errors.append(
                "description cache 实际 components 与 manifest 不一致: "
                f"records={sorted(by_component)} manifest={sorted(self.manifest['components'])}"
            )
        expected_reused = int(
            self.manifest["source_cache_provenance"].get("reused_records") or 0
        )
        if reused_records != expected_reused:
            errors.append(
                "源 cache 复用 record 数不一致: "
                f"records={reused_records} manifest={expected_reused}"
            )
        if reused_records and source_record_fingerprint is None:
            errors.append(
                "复用源 segmentation cache 的 records 未验证: "
                f"count={reused_records}"
            )

        if expected_input_fingerprints is not None:
            stored = self.manifest["input_fingerprints"]
            for component in self.manifest["components"]:
                expected = expected_input_fingerprints.get(component)
                actual = stored.get(component)
                if expected is None:
                    errors.append(f"当前输入缺少 component 指纹: {component}")
                    continue
                if not isinstance(actual, dict):
                    errors.append(f"manifest 缺少输入指纹: {component}")
                    continue
                for field in (
                    "benchmark", "index", "size", "sha256",
                    "validation_report", "validation_report_size",
                    "validation_report_sha256", "validation_builder_version",
                    "validation_status",
                ):
                    if actual.get(field) != expected.get(field):
                        errors.append(
                            f"输入索引指纹已变化: component={component} field={field} "
                            f"expected={actual.get(field)!r} current={expected.get(field)!r}"
                        )

        part_files = sorted(
            path.relative_to(self.cache_dir).as_posix()
            for path in self.cache_dir.rglob("*.part")
        )
        if part_files:
            errors.append(f"description cache 残留 .part 文件: {part_files[:8]}")
        num_errors = len(errors)
        errors_truncated = num_errors > 200
        if errors_truncated:
            errors = errors[:200]
        report = {
            "protocol": DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
            "format": DESCRIPTION_CACHE_FORMAT,
            "cache_protocol": DESCRIPTION_CACHE_PROTOCOL,
            "builder_version": self.manifest["builder_version"],
            "status": "valid" if not errors else "invalid",
            "num_records": len(seen_keys),
            "num_shards": len(self.shards),
            "manifest_sha256": sha256_file(self.cache_dir / "manifest.json"),
            "shard_integrity": {
                "protocol": "sha256_size_record_count_v1",
                "manifest_entries": len(self.shard_fingerprints),
                "verified_shards": verified_shards,
                "verified_bytes": verified_shard_bytes,
                "all_verified": verified_shards == len(self.shards),
            },
            "records_by_component": dict(sorted(by_component.items())),
            "input_fingerprints": self.manifest["input_fingerprints"],
            "source_cache": {
                "provided": bool(self.manifest["source_cache_provenance"]["provided"]),
                "reused_records": reused_records,
                "validated_records": validated_source_records,
                "isolation_unchanged": bool(
                    self.manifest["source_cache_provenance"]["isolation_unchanged"]
                ),
            },
            "num_errors": num_errors,
            "num_warnings": len(warnings),
            "errors_truncated": errors_truncated,
            "errors": errors,
            "warnings": warnings,
        }
        return report


def revalidate_description_cache_artifact(
    expected_binding: Any,
) -> dict[str, Any]:
    """Replay a checkpoint-bound cache identity and all shard content hashes."""
    if (
        not isinstance(expected_binding, dict)
        or expected_binding.get("protocol")
        != DESCRIPTION_CACHE_ARTIFACT_BINDING_PROTOCOL
    ):
        raise RuntimeError(
            "checkpoint 缺少当前 Description Vision Cache artifact binding"
        )
    cache_ref = expected_binding.get("cache_dir")
    if not isinstance(cache_ref, str) or not cache_ref.strip():
        raise RuntimeError("Description Vision Cache artifact binding 缺少 cache_dir")
    cache_dir = resolve_project_path(cache_ref) or Path(cache_ref)
    bank = DescriptionVisionFeatureBank(cache_dir, max_open_shards=1)
    observed_binding = bank.artifact_binding()
    if observed_binding != expected_binding:
        raise RuntimeError(
            "Description Vision Cache manifest/validation artifact 已漂移"
        )

    metadata_snapshot = bank._file_metadata_snapshot()
    binding_sha256 = canonical_sha256(observed_binding)
    verification_key = canonical_sha256({
        "binding_sha256": binding_sha256,
        "metadata_snapshot": metadata_snapshot,
    })
    memo_key = str(cache_dir.resolve(strict=False))
    memo = _VERIFIED_DESCRIPTION_CACHE_ARTIFACTS.get(memo_key)
    memoized = bool(memo and memo.get("verification_key") == verification_key)
    if memoized:
        shard_replay = dict(memo["shard_replay"])
    else:
        shard_replay = bank.verify_all_shards()
        if shard_replay["metadata_snapshot"] != metadata_snapshot:
            raise RuntimeError(
                "Description Vision Cache artifact metadata 在重放期间漂移"
            )
        _VERIFIED_DESCRIPTION_CACHE_ARTIFACTS[memo_key] = {
            "verification_key": verification_key,
            "shard_replay": dict(shard_replay),
        }
    if shard_replay.get("all_verified") is not True:
        raise RuntimeError("Description Vision Cache 未完成全部 shard provenance 重放")
    return {
        "protocol": DESCRIPTION_CACHE_ARTIFACT_REVALIDATION_PROTOCOL,
        "cache_dir": memo_key,
        "artifact_binding_sha256": binding_sha256,
        "manifest_sha256": observed_binding["manifest"]["sha256"],
        "validation_report_sha256": observed_binding["validation_report"]["sha256"],
        "shard_inventory_sha256": observed_binding["shard_inventory_sha256"],
        # 是否命中进程内 memo 只是性能细节，不进入可持久化科学证据。
        "shard_replay": shard_replay,
    }
