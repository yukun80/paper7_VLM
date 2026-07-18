"""Strict JSON and content-addressed artifact I/O for M3-M7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from qpsalm_seg.paths import resolve_project_path


class NonFiniteJSONError(json.JSONDecodeError):
    """Reject Python's non-standard NaN/Infinity JSON extension."""

    def __init__(self, token: str) -> None:
        super().__init__(
            f"non-standard JSON numeric constant is forbidden: {token}", token, 0
        )


def _reject_nonfinite_json_constant(token: str) -> None:
    raise NonFiniteJSONError(token)


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    return json.loads(payload, parse_constant=_reject_nonfinite_json_constant)


def _resolve(path: str | Path) -> Path:
    return resolve_project_path(path) or Path(path)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_raw_bytes(value: torch.Tensor) -> bytes:
    """Return logical dense tensor bytes, including scalar and BF16 tensors."""
    # Flattening is required before a dtype-changing view of a scalar tensor.
    tensor = value.detach().cpu().contiguous().reshape(-1)
    if tensor.numel() == 0:
        return b""
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape and raw values independently of torch serialization."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(
        list(tensor.shape), separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    digest.update(tensor_raw_bytes(tensor))
    return digest.hexdigest()


def nested_file_bindings_current(
    value: Any,
    *,
    relative_roots: Iterable[str | Path] = (),
) -> bool:
    """Replay nested ``path + sha256`` bindings with explicit root context.

    Most persisted bindings use absolute or project-logical paths. Description
    cache artifact bindings intentionally keep ``manifest.json`` and
    ``validation_report.json`` relative to their ``cache_dir`` so the cache can
    be moved as one artifact. Recursion therefore carries that directory into
    child bindings instead of accidentally resolving them from the repository
    root.
    """

    roots: tuple[Path, ...] = tuple(
        (_resolve(root).resolve(strict=False)) for root in relative_roots
    )

    def replay(item: Any, inherited_roots: tuple[Path, ...]) -> bool:
        if isinstance(item, list):
            return all(replay(child, inherited_roots) for child in item)
        if not isinstance(item, dict):
            return True

        local_roots = list(inherited_roots)
        cache_root = item.get("cache_dir")
        if isinstance(cache_root, str) and cache_root.strip():
            resolved_root = _resolve(cache_root).resolve(strict=False)
            if resolved_root not in local_roots:
                local_roots.append(resolved_root)
        roots_tuple = tuple(local_roots)

        if "path" in item and "sha256" in item:
            raw_path = item.get("path")
            raw_sha256 = item.get("sha256")
            if raw_path is None and raw_sha256 is None:
                return all(
                    replay(child, roots_tuple) for child in item.values()
                )
            if (
                not isinstance(raw_path, str)
                or not raw_path.strip()
                or not isinstance(raw_sha256, str)
                or len(raw_sha256) != 64
            ):
                return False
            path = Path(raw_path)
            candidates = []
            if path.is_absolute():
                candidates.append(path.resolve(strict=False))
            else:
                candidates.append(_resolve(path).resolve(strict=False))
                candidates.extend(
                    (root / path).resolve(strict=False) for root in roots_tuple
                )
            unique_candidates = tuple(dict.fromkeys(candidates))
            if not any(
                candidate.is_file()
                and sha256_file(candidate) == raw_sha256
                for candidate in unique_candidates
            ):
                return False
        return all(replay(child, roots_tuple) for child in item.values())

    return replay(value, roots)


def read_json(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    resolved = _resolve(path)
    value = strict_json_loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须是 object: {resolved}")
    return value


def read_jsonl(path: str | Path, *, label: str = "JSONL") -> list[dict[str, Any]]:
    resolved = _resolve(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{label} 第 {line_number} 行必须是 object: {resolved}"
                )
            rows.append(value)
    return rows


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = True,
) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=sort_keys,
        allow_nan=False,
    ) + "\n"
    _atomic_text(_resolve(path), encoded)


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    sort_keys: bool = True,
) -> None:
    encoded = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=sort_keys,
            allow_nan=False,
        ) + "\n"
        for row in rows
    )
    _atomic_text(_resolve(path), encoded)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Atomically replace one arbitrary byte artifact on the target filesystem."""
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically append one JSON object for single-writer run histories."""
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False
    ) + "\n"
    previous = resolved.read_text(encoding="utf-8") if resolved.is_file() else ""
    if previous and not previous.endswith("\n"):
        raise ValueError(
            f"拒绝追加到非完整 JSONL artifact: {resolved}"
        )
    _atomic_text(resolved, previous + encoded)
