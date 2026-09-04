"""原子字节写入与路径安全原语。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Literal, Mapping

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
    windows_path = PureWindowsPath(value)
    if path.is_absolute() or windows_path.drive or windows_path.root:
        raise PortablePathError("absolute", normalized)
    if (
        not normalized
        or normalized in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PortablePathError("escape", normalized)
    return path


BUNDLE_REPOSITORY_NAME = "paper7_VLM"
BUNDLE_BENCHMARK_NAME = "benchmark"


def validate_bundle_root(path: Path) -> Path:
    """验证可迁移包根及固定的 repository/benchmark 兄弟目录。"""

    root = Path(os.path.abspath(path))
    linked = first_symlink_component(root)
    if linked is not None:
        raise PortablePathError("symlink", str(linked))
    if root.is_symlink() or not root.is_dir():
        raise PortablePathError("missing_bundle_root", str(root))
    for name in (BUNDLE_REPOSITORY_NAME, BUNDLE_BENCHMARK_NAME):
        child = root / name
        if child.is_symlink() or not child.is_dir():
            raise PortablePathError("missing_bundle_child", child.as_posix())
    return root.resolve()


def resolve_config_reference(config_path: Path, reference: str) -> Path:
    """按配置文件所在目录解析相对路径，不依赖当前工作目录。"""

    if not isinstance(reference, str) or not reference.strip():
        raise PortablePathError("empty", str(reference))
    normalized = reference.replace("\\", "/")
    relative = Path(normalized)
    windows_path = PureWindowsPath(reference)
    if relative.is_absolute() or windows_path.drive or windows_path.root:
        raise PortablePathError("absolute", reference)
    config_file = Path(os.path.abspath(config_path))
    config_dir = config_file.parent
    linked = first_symlink_component(config_file)
    if linked is not None:
        raise PortablePathError("symlink", str(linked))
    candidate = Path(os.path.abspath(config_dir / relative))
    linked = first_symlink_component(candidate)
    if linked is not None:
        raise PortablePathError("symlink", str(linked))
    return candidate.resolve(strict=False)


def resolve_bundle_reference(
    bundle_root: Path,
    reference: str,
    *,
    expected: Literal["file", "directory"] | None = None,
) -> Path:
    """从包根解析安全的 POSIX 相对引用，并拒绝链接与目录逃逸。"""

    root = validate_bundle_root(bundle_root)
    relative = portable_relative_path(reference)
    path = root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root)
    except ValueError as error:
        raise PortablePathError("escape", reference) from error
    linked = first_symlink_component(path)
    if linked is not None:
        raise PortablePathError("symlink", str(linked))
    if expected == "file" and (path.is_symlink() or not path.is_file()):
        raise PortablePathError("missing_file", reference)
    if expected == "directory" and (path.is_symlink() or not path.is_dir()):
        raise PortablePathError("missing_directory", reference)
    return resolved


def bundle_relative_reference(bundle_root: Path, path: Path) -> str:
    """将现场路径规范为相对包根的 POSIX 身份引用。"""

    root = validate_bundle_root(bundle_root)
    target = Path(os.path.abspath(path))
    linked = first_symlink_component(target)
    if linked is not None:
        raise PortablePathError("symlink", str(linked))
    try:
        relative = target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise PortablePathError("outside_bundle", str(target)) from error
    return portable_relative_path(relative.as_posix()).as_posix()


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
