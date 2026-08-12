"""严格 JSON/JSONL、SHA-256、路径与原子 I/O 公共工具。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_text,
    stable_hash,
)
from oa_groundrag.artifacts.io import (
    PortablePathError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    first_symlink_component,
    portable_relative_path as _portable_relative_path,
)

from .errors import ConfigError, ReasonCode, SchemaError


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_unique_json_object)


def read_json(path: Path) -> Any:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"无法严格读取 JSON：{path}",
            details={"error": str(error)},
        ) from error
    reject_nonfinite(value, location=str(path))
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"{path}:{line_number}: JSONL 不允许空行",
                    )
                try:
                    value = _strict_json_loads(line)
                except (json.JSONDecodeError, ValueError) as error:
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"{path}:{line_number}: 非法 JSON",
                        details={"error": str(error)},
                    ) from error
                if not isinstance(value, dict):
                    raise SchemaError(
                        ReasonCode.TYPE_MISMATCH,
                        f"{path}:{line_number}: JSONL 行必须为对象",
                    )
                reject_nonfinite(value, location=f"{path}:{line_number}")
                rows.append(value)
    except OSError as error:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"无法读取 JSONL：{path}",
            details={"error": str(error)},
        ) from error
    return rows


def read_jsonl_indices(
    path: Path,
    indices: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """只读取 JSONL 中指定的零基行；扫描在最大目标行后立即停止。"""

    requested: set[int] = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH,
                f"{path}: JSONL 行号必须是 >= 0 的整数",
            )
        if index in requested:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{path}: JSONL 行号不能重复：{index}",
            )
        requested.add(index)
    if not requested:
        return {}
    last = max(requested)
    rows: dict[int, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle):
                if index > last:
                    break
                if not line.strip():
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"{path}:{index + 1}: JSONL 不允许空行",
                    )
                if index not in requested:
                    continue
                try:
                    value = _strict_json_loads(line)
                except (json.JSONDecodeError, ValueError) as error:
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"{path}:{index + 1}: 非法 JSON",
                        details={"error": str(error)},
                    ) from error
                if not isinstance(value, dict):
                    raise SchemaError(
                        ReasonCode.TYPE_MISMATCH,
                        f"{path}:{index + 1}: JSONL 行必须为对象",
                    )
                reject_nonfinite(value, location=f"{path}:{index + 1}")
                rows[index] = value
    except OSError as error:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"无法读取 JSONL：{path}",
            details={"error": str(error)},
        ) from error
    missing = sorted(requested - set(rows))
    if missing:
        raise SchemaError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{path}: 请求行超出 JSONL：{missing}",
        )
    return rows


def reject_nonfinite(value: Any, *, location: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(
                ReasonCode.NONFINITE_NUMBER,
                f"{location}: 不允许 NaN 或 Inf",
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite(child, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, location=f"{location}.{key}")
        return


def require_mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是字符串键对象",
        )
    return dict(value)


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    location: str,
) -> None:
    keys = set(value)
    required_set = set(required)
    allowed = required_set | set(optional)
    unknown = sorted(keys - allowed)
    missing = sorted(required_set - keys)
    if unknown:
        raise ConfigError(
            ReasonCode.UNKNOWN_FIELD,
            f"{location}: 未知字段 {unknown}",
        )
    if missing:
        raise ConfigError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{location}: 缺少字段 {missing}",
        )


def require_bool(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是 bool")
    return value


def require_int(
    value: Any,
    *,
    location: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是整数")
    if minimum is not None and value < minimum:
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须 >= {minimum}",
        )
    return value


def require_string(
    value: Any,
    *,
    location: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是非空字符串",
        )
    return value.strip() if not allow_empty else value


def require_enum(
    value: Any,
    *,
    choices: Sequence[str],
    location: str,
) -> str:
    text = require_string(value, location=location)
    if text not in choices:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"{location}: {text!r} 不在 {list(choices)} 中",
        )
    return text


def portable_relative_path(value: str, *, location: str) -> PurePosixPath:
    try:
        return _portable_relative_path(value)
    except PortablePathError as error:
        if error.reason == "absolute":
            raise SchemaError(
                ReasonCode.PATH_ABSOLUTE,
                f"{location}: 可移植合同不得使用绝对路径",
            ) from error
        raise SchemaError(
            ReasonCode.PATH_ESCAPE,
            f"{location}: 非法相对路径 {error.value!r}",
        ) from error


def resolve_config_path(base: Path, value: Any, *, location: str) -> Path:
    text = require_string(value, location=location)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def safe_join(root: Path, relative: str, *, location: str) -> Path:
    pure = portable_relative_path(relative, location=location)
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise SchemaError(
            ReasonCode.PATH_ESCAPE,
            f"{location}: 路径逃逸 {relative!r}",
        ) from error
    return candidate


def normalized_text(value: Any, *, location: str) -> str:
    text = require_string(value, location=location)
    return " ".join(text.split())


def normalized_multiline_text(value: Any, *, location: str) -> str:
    text = require_string(value, location=location)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
