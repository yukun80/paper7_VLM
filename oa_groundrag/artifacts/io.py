"""原子字节写入与路径安全原语。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .identity import canonical_json


class PortablePathError(ValueError):
    """可移植相对路径不满足绝对路径/逃逸约束。"""

    def __init__(self, reason: str, value: str) -> None:
        super().__init__(f"{reason}: {value!r}")
        self.reason = reason
        self.value = value


def first_symlink_component(path: Path) -> Path | None:
    """返回词法绝对路径中第一个已存在的 symlink 组件。"""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
        if not current.exists():
            break
    return None


def portable_relative_path(value: str) -> PurePosixPath:
    """将跨平台路径规范为安全的 POSIX 相对路径。"""

    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise PortablePathError("absolute", normalized)
    if (
        not normalized
        or normalized in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PortablePathError("escape", normalized)
    return path


def atomic_write_bytes(path: Path, payload: bytes) -> None:
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
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(f"{canonical_json(dict(row))}\n" for row in rows)
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))
