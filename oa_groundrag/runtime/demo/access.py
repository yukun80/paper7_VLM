"""Demo-only test 去盲化 receipt 与显式 runtime 访问上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.runtime.contracts import (
    InMemorySpatialInput,
    UnifiedInferenceError,
    UnifiedReasonCode,
    UnifiedRequest,
)


TEST_RECEIPT_SCHEMA = "oa_groundrag.demo_test_access_receipt.v1"
DEMO_INFERENCE_ACCESS_SCHEMA = "oa_groundrag.demo_inference_access.v1"
_ACTIONS = {"BROWSE", "GALLERY", "INFERENCE"}


class DemoAccessError(RuntimeError):
    """Demo test 授权、receipt 或路径合同失败。"""


@dataclass(frozen=True)
class DemoTestAccessReceipt:
    receipt_id: str
    sample_id: str
    action: str
    benchmark_identity: Mapping[str, Any]
    config_sha256: str
    created_at: str
    receipt_root: Path
    schema_version: str = TEST_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "created_at": self.created_at,
            "sample_id": self.sample_id,
            "split": "test",
            "action": self.action,
            "benchmark_identity": dict(self.benchmark_identity),
            "config_sha256": self.config_sha256,
            "qualitative_demo_only": True,
            "blind_or_sealed_evaluation_property": False,
            "formal_test_evaluation": False,
            "scientific_acceptance": False,
            "declassification_is_monotonic": True,
        }

    def verify_published(
        self,
        *,
        demo_root: Path,
        required_action: str,
        benchmark_identity: Mapping[str, Any] | None = None,
        config_sha256: str | None = None,
    ) -> None:
        """验证 receipt 是当前 Demo root 下已发布且身份完整的直接子资产。"""

        root = Path(os.path.abspath(self.receipt_root))
        expected_parent = Path(os.path.abspath(demo_root)) / "test_access_receipts"
        if (
            required_action not in _ACTIONS
            or self.action != required_action
            or root.parent != expected_parent
            or root.name != self.receipt_id
            or not root.is_dir()
            or root.is_symlink()
            or first_symlink_component(root) is not None
        ):
            raise DemoAccessError("test receipt 位置、action 或目录身份不匹配")
        if (
            benchmark_identity is not None
            and dict(self.benchmark_identity) != dict(benchmark_identity)
        ):
            raise DemoAccessError("test receipt Benchmark identity 不匹配")
        if config_sha256 is not None and self.config_sha256 != config_sha256:
            raise DemoAccessError("test receipt config SHA-256 不匹配")
        receipt_path = root / "receipt.json"
        manifest_path = root / "manifest.json"
        ledger_path = root / "SHA256SUMS.jsonl"
        for path in (receipt_path, manifest_path, ledger_path):
            if (
                not path.is_file()
                or path.is_symlink()
                or first_symlink_component(path) is not None
            ):
                raise DemoAccessError(f"test receipt 资产不完整或含链接：{path.name}")
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ledger = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DemoAccessError("test receipt JSON/ledger 无法严格读取") from error
        receipt_sha = sha256_file(receipt_path)
        expected_ledger = [{
            "path": "receipt.json",
            "size_bytes": receipt_path.stat().st_size,
            "sha256": receipt_sha,
        }]
        if (
            payload != self.to_dict()
            or ledger != expected_ledger
            or manifest.get("schema_version")
            != "oa_groundrag.demo_test_access_manifest.v1"
            or manifest.get("receipt_id") != self.receipt_id
            or manifest.get("receipt_sha256") != receipt_sha
            or manifest.get("ledger_root_sha256")
            != sha256_text(canonical_json(expected_ledger))
            or manifest.get("immutable_receipt") is not True
            or manifest.get("blind_or_sealed_evaluation_property") is not False
            or manifest.get("formal_test_evaluation") is not False
        ):
            raise DemoAccessError("test receipt payload、ledger 或 manifest 身份漂移")


@dataclass(frozen=True)
class DemoAuthorizedSpatialInput(InMemorySpatialInput):
    """仅能与匹配 receipt 的 DemoInferenceAccess 一起交给普通 runtime。"""

    receipt_id: str

    def __post_init__(self) -> None:
        if self.split != "test":
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "DemoAuthorizedSpatialInput 只允许 split=test",
            )
        if not self.sample_id or not self.source or not self.receipt_id:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "DemoAuthorizedSpatialInput 缺少 sample/source/receipt",
            )


class DemoTestAccessController:
    """先发布不可变 receipt，再允许调用方打开具体 test payload。"""

    def __init__(
        self,
        *,
        demo_root: Path,
        allow_test_demo: bool,
        benchmark_identity: Mapping[str, Any],
        config_sha256: str,
    ) -> None:
        self.demo_root = Path(os.path.abspath(demo_root))
        self.receipt_root = self.demo_root / "test_access_receipts"
        self.allow_test_demo = allow_test_demo
        self.benchmark_identity = dict(benchmark_identity)
        self.config_sha256 = config_sha256

    def issue(self, *, sample_id: str, action: str) -> DemoTestAccessReceipt:
        if not self.allow_test_demo:
            raise DemoAccessError(
                "test split 已锁定；只有配置 allow_test_demo=true 才能去盲化访问"
            )
        if not sample_id:
            raise DemoAccessError("test receipt 要求非空 sample_id")
        action = str(action).upper()
        if action not in _ACTIONS:
            raise DemoAccessError(f"未知 test Demo action：{action}")
        linked = first_symlink_component(self.receipt_root)
        if linked is not None:
            raise DemoAccessError(f"test receipt root 含 symlink：{linked}")
        now = datetime.now(timezone.utc)
        receipt_id = f"dta_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}"
        target = self.receipt_root / receipt_id
        receipt = DemoTestAccessReceipt(
            receipt_id=receipt_id,
            sample_id=sample_id,
            action=action,
            benchmark_identity=self.benchmark_identity,
            config_sha256=self.config_sha256,
            created_at=now.isoformat(),
            receipt_root=target,
        )
        try:
            with AtomicArtifactDirectory(target) as writer:
                writer.write_json("receipt.json", receipt.to_dict())
                assert writer.staging is not None
                ledger = [{
                    "path": "receipt.json",
                    "size_bytes": writer.path("receipt.json").stat().st_size,
                    "sha256": sha256_file(writer.path("receipt.json")),
                }]
                writer.write_jsonl("SHA256SUMS.jsonl", ledger)
                writer.write_json("manifest.json", {
                    "schema_version": "oa_groundrag.demo_test_access_manifest.v1",
                    "receipt_id": receipt_id,
                    "receipt_sha256": ledger[0]["sha256"],
                    "ledger_root_sha256": sha256_text(canonical_json(ledger)),
                    "immutable_receipt": True,
                    "blind_or_sealed_evaluation_property": False,
                    "formal_test_evaluation": False,
                })
                writer.publish()
        except Exception as error:
            raise DemoAccessError(
                "test receipt 未能在 payload 打开前原子发布；访问已拒绝"
            ) from error
        return receipt


@dataclass(frozen=True)
class DemoInferenceAccess:
    """把已发布的 INFERENCE receipt 显式绑定到一次 runtime 调用。"""

    receipt: DemoTestAccessReceipt
    demo_root: Path
    benchmark_identity: Mapping[str, Any]
    config_sha256: str
    schema_version: str = DEMO_INFERENCE_ACCESS_SCHEMA

    def _verify_receipt(self) -> None:
        try:
            self.receipt.verify_published(
                demo_root=self.demo_root,
                required_action="INFERENCE",
                benchmark_identity=self.benchmark_identity,
                config_sha256=self.config_sha256,
            )
        except DemoAccessError as error:
            raise UnifiedInferenceError(
                UnifiedReasonCode.ARTIFACT_IDENTITY_MISMATCH,
                "Demo inference receipt 身份漂移",
            ) from error

    def validate_runtime_access(
        self,
        request: UnifiedRequest,
        *,
        output_root: Path | None,
    ) -> Mapping[str, Any]:
        self._verify_receipt()
        spatial = request.spatial_input
        if spatial is not None and not (
            isinstance(spatial, DemoAuthorizedSpatialInput)
            and spatial.split == "test"
            and spatial.sample_id == self.receipt.sample_id
            and spatial.receipt_id == self.receipt.receipt_id
        ):
            raise UnifiedInferenceError(
                UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
                "Demo test spatial input 与 receipt 不匹配",
                task=request.task,
            )
        root = Path(os.path.abspath(self.demo_root))
        for path in request.images:
            try:
                Path(os.path.abspath(path)).relative_to(root)
            except ValueError as error:
                raise UnifiedInferenceError(
                    UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
                    "Demo test image 必须先 staging 到独立 Demo root",
                    task=request.task,
                ) from error
        if output_root is not None:
            try:
                Path(os.path.abspath(output_root)).relative_to(root)
            except ValueError as error:
                raise UnifiedInferenceError(
                    UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
                    "Demo test inference output 必须位于独立 Demo root",
                    task=request.task,
                ) from error
        return {
            "access_type": "DEMO_TEST_ACCESS",
            "access_receipt_id": self.receipt.receipt_id,
            "sealed_test_accessed": True,
            "blind_or_sealed_evaluation_property": False,
            "formal_test_evaluation": False,
        }
