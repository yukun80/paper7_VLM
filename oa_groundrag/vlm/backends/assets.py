"""固定 Hub revision 的本地 VLM 资产 ledger 与严格复算。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.vlm.errors import ReasonCode, VLMError


MODEL_ASSET_LEDGER_SCHEMA = "oa_groundrag.vlm_model_asset_ledger.v1"

_QWEN3_5_REQUIRED_FILES = frozenset(
    {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "model.safetensors-00001-of-00002.safetensors",
        "model.safetensors-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)


class ModelAssetLedgerError(VLMError):
    """资产树、ledger 或固定 revision 不一致。"""


@dataclass(frozen=True)
class VerifiedModelAssets:
    backend: str
    repo_id: str
    revision: str
    model_root: Path
    ledger_path: Path
    ledger_file_sha256: str
    ledger_payload_sha256: str
    total_size_bytes: int
    file_count: int
    weight_shards: tuple[str, ...]

    def identity_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "hub_repo_id": self.repo_id,
            "hub_revision": self.revision,
            "model_root": str(self.model_root),
            "asset_ledger_path": str(self.ledger_path),
            "asset_ledger_file_sha256": self.ledger_file_sha256,
            "asset_ledger_payload_sha256": self.ledger_payload_sha256,
            "asset_total_size_bytes": self.total_size_bytes,
            "asset_file_count": self.file_count,
            "weight_shards": list(self.weight_shards),
        }


def _fail(message: str, **details: Any) -> None:
    raise ModelAssetLedgerError(
        ReasonCode.ASSET_LEDGER_INVALID,
        message,
        details=details,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key: {key}")
        output[key] = value
    return output


def _read_strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        _fail("无法严格读取 JSON", path=str(path), error=str(error))


def _regular_file(path: Path, *, label: str) -> None:
    if (
        first_symlink_component(path) is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        _fail(f"{label} 必须是普通单链接文件", path=str(path))


def _model_root(path: Path) -> Path:
    root = Path(os.path.abspath(path))
    if (
        first_symlink_component(root) is not None
        or root.is_symlink()
        or not root.is_dir()
    ):
        _fail("model_root 必须是本地普通目录", path=str(root))
    return root


def _revision(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("Hub revision 必须是 40 位小写 commit SHA")
    return value


def _required_files(backend: str) -> frozenset[str]:
    if backend == "qwen3_5":
        return _QWEN3_5_REQUIRED_FILES
    _fail("asset ledger 不支持该 backend", backend=backend)


def _file_role(name: str) -> str:
    if name.startswith("model.safetensors-") and name.endswith(".safetensors"):
        return "weight_shard"
    return {
        ".gitattributes": "hub_metadata",
        "LICENSE": "license",
        "README.md": "model_card",
        "chat_template.jinja": "chat_template",
        "config.json": "model_config",
        "merges.txt": "tokenizer",
        "model.safetensors.index.json": "weight_index",
        "preprocessor_config.json": "image_processor",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer",
        "video_preprocessor_config.json": "video_processor",
        "vocab.json": "tokenizer",
    }.get(name, "unclassified")


def _inventory(root: Path, *, backend: str) -> tuple[dict[str, Any], ...]:
    children = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    directories = [child.name for child in children if child.is_dir()]
    if directories:
        _fail("model_root 不允许额外目录", directories=directories)
    actual = {child.name for child in children if child.is_file()}
    expected = _required_files(backend)
    if actual != expected:
        _fail(
            "model_root 文件树与固定 Hub revision 不一致",
            missing=sorted(expected - actual),
            extra=sorted(actual - expected),
        )
    rows: list[dict[str, Any]] = []
    for name in sorted(actual):
        path = root / name
        _regular_file(path, label=f"asset {name}")
        rows.append(
            {
                "path": name,
                "role": _file_role(name),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(rows)


def _weight_shards(root: Path, files: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    index_path = root / "model.safetensors.index.json"
    index = _read_strict_json(index_path)
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        _fail("weight index 顶层合同非法")
    weight_map = index["weight_map"]
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(name, str)
            and name
            and isinstance(shard, str)
            and shard
            for name, shard in weight_map.items()
        )
    ):
        _fail("weight_map 必须是非空 tensor→shard 对象")
    shards = tuple(sorted(set(weight_map.values())))
    ledger_shards = tuple(
        sorted(
            str(row["path"])
            for row in files
            if row.get("role") == "weight_shard"
        )
    )
    if shards != ledger_shards:
        _fail(
            "weight index 与实际分片不一致",
            index_shards=list(shards),
            actual_shards=list(ledger_shards),
        )
    for shard in shards:
        relative = PurePosixPath(shard)
        if relative.is_absolute() or len(relative.parts) != 1:
            _fail("weight index shard 路径非法", shard=shard)
    return shards


def _validate_backend_metadata(root: Path, *, backend: str, repo_id: str) -> None:
    config = _read_strict_json(root / "config.json")
    processor = _read_strict_json(root / "preprocessor_config.json")
    if backend == "qwen3_5" and (
        repo_id != "Qwen/Qwen3.5-4B"
        or not isinstance(config, dict)
        or config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or not isinstance(processor, dict)
        or processor.get("processor_class") != "Qwen3VLProcessor"
    ):
        _fail("Qwen3.5 Hub/model/processor metadata 不匹配")


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(os.path.abspath(path))
    if first_symlink_component(target) is not None or target.exists() or target.is_symlink():
        _fail("ledger 目标必须全新且路径不含 symlink", path=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        temporary.unlink()
    except FileExistsError as error:
        temporary.unlink(missing_ok=True)
        _fail("ledger 目标已存在，拒绝覆盖", path=str(target), error=str(error))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def create_model_asset_ledger(
    *,
    model_root: Path,
    ledger_path: Path,
    backend: str,
    repo_id: str,
    revision: str,
) -> VerifiedModelAssets:
    """哈希固定资产树并创建一个不可覆盖 ledger。"""

    root = _model_root(model_root)
    fixed_revision = _revision(revision)
    _validate_backend_metadata(root, backend=backend, repo_id=repo_id)
    files = _inventory(root, backend=backend)
    shards = _weight_shards(root, files)
    payload: dict[str, Any] = {
        "schema_version": MODEL_ASSET_LEDGER_SCHEMA,
        "backend": backend,
        "hub": {"repo_id": repo_id, "revision": fixed_revision},
        "model_root": str(root),
        "files": list(files),
        "weight_index": {
            "path": "model.safetensors.index.json",
            "shards": list(shards),
        },
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
    }
    payload_sha256 = sha256_text(canonical_json(payload))
    ledger = {**payload, "ledger_payload_sha256": payload_sha256}
    _atomic_write_new_json(ledger_path, ledger)
    return verify_model_asset_ledger(
        ledger_path,
        expected_backend=backend,
        expected_repo_id=repo_id,
        expected_revision=fixed_revision,
        expected_model_root=root,
    )


def verify_model_asset_ledger(
    ledger_path: Path,
    *,
    expected_backend: str | None = None,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
    expected_model_root: Path | None = None,
) -> VerifiedModelAssets:
    """严格复算 ledger、文件树、每个 SHA 与权重分片映射。"""

    path = Path(os.path.abspath(ledger_path))
    _regular_file(path, label="asset ledger")
    value = _read_strict_json(path)
    expected_keys = {
        "schema_version",
        "backend",
        "hub",
        "model_root",
        "files",
        "weight_index",
        "total_size_bytes",
        "ledger_payload_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        _fail("asset ledger 顶层字段不匹配")
    if value["schema_version"] != MODEL_ASSET_LEDGER_SCHEMA:
        _fail("asset ledger schema 不匹配")
    backend = value["backend"]
    hub = value["hub"]
    if (
        not isinstance(backend, str)
        or not isinstance(hub, dict)
        or set(hub) != {"repo_id", "revision"}
        or not isinstance(hub["repo_id"], str)
    ):
        _fail("asset ledger backend/hub 合同非法")
    revision = _revision(hub["revision"])
    if expected_backend is not None and backend != expected_backend:
        _fail("ledger backend 与配置不一致")
    if expected_repo_id is not None and hub["repo_id"] != expected_repo_id:
        _fail("ledger Hub repo 与配置不一致")
    if expected_revision is not None and revision != expected_revision:
        _fail("ledger Hub revision 与配置不一致")
    root = _model_root(Path(value["model_root"]))
    if (
        expected_model_root is not None
        and root != Path(os.path.abspath(expected_model_root))
    ):
        _fail("ledger model_root 与配置不一致")
    files = value["files"]
    if not isinstance(files, list) or not files:
        _fail("ledger files 必须是非空列表")
    parsed_files: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != {
            "path", "role", "size_bytes", "sha256"
        }:
            _fail("ledger file row 字段非法", index=index)
        name = row["path"]
        if (
            not isinstance(name, str)
            or PurePosixPath(name).is_absolute()
            or len(PurePosixPath(name).parts) != 1
            or not isinstance(row["role"], str)
            or isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            _fail("ledger file row 类型非法", index=index)
        parsed_files.append(dict(row))
    names = [str(row["path"]) for row in parsed_files]
    if names != sorted(names) or len(names) != len(set(names)):
        _fail("ledger files 必须按路径排序且不重复")
    actual_files = _inventory(root, backend=backend)
    if tuple(parsed_files) != actual_files:
        _fail("ledger 文件 SHA/size/role 与当前资产树不一致")
    shards = _weight_shards(root, parsed_files)
    weight_index = value["weight_index"]
    if weight_index != {
        "path": "model.safetensors.index.json",
        "shards": list(shards),
    }:
        _fail("ledger weight_index 合同不一致")
    total_size = sum(int(row["size_bytes"]) for row in parsed_files)
    if value["total_size_bytes"] != total_size:
        _fail("ledger total_size_bytes 不一致")
    payload = dict(value)
    payload_sha = payload.pop("ledger_payload_sha256")
    actual_payload_sha = sha256_text(canonical_json(payload))
    if payload_sha != actual_payload_sha:
        _fail("ledger payload SHA 不一致")
    _validate_backend_metadata(root, backend=backend, repo_id=hub["repo_id"])
    return VerifiedModelAssets(
        backend=backend,
        repo_id=hub["repo_id"],
        revision=revision,
        model_root=root,
        ledger_path=path,
        ledger_file_sha256=sha256_file(path),
        ledger_payload_sha256=actual_payload_sha,
        total_size_bytes=total_size,
        file_count=len(parsed_files),
        weight_shards=shards,
    )

