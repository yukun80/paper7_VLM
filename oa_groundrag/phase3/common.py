"""严格 JSON/JSONL、SHA-256、路径与原子 I/O 公共工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .errors import ConfigError, ReasonCode, SchemaError


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, *parts: object) -> str:
    return sha256_text("\0".join((str(seed), *(str(part) for part in parts))))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(f"{canonical_json(dict(row))}\n" for row in rows)
    _atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


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
    if "\\" in value:
        value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise SchemaError(
            ReasonCode.PATH_ABSOLUTE,
            f"{location}: 可移植合同不得使用绝对路径",
        )
    if not value or value in {".", ".."} or any(part in {"", ".", ".."} for part in path.parts):
        raise SchemaError(
            ReasonCode.PATH_ESCAPE,
            f"{location}: 非法相对路径 {value!r}",
        )
    return path


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
