"""Deterministic hashing and atomic artifact publication helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


def reject_non_finite(value: Any, *, location: str = "$") -> None:
    """Recursively reject non-finite floating-point values before publishing."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_non_finite(item, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            reject_non_finite(item, location=f"{location}[{index}]")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize strict UTF-8 JSON with stable ordering and a final newline."""

    reject_non_finite(payload)
    rendered = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{rendered}\n".encode("utf-8")


def canonical_yaml_bytes(payload: Any) -> bytes:
    """Serialize stable UTF-8 YAML after applying the JSON finiteness rule."""

    reject_non_finite(payload)
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    return rendered.encode("utf-8")


def atomic_write_bytes(path: Path, content: bytes, *, overwrite: bool = False) -> None:
    """Atomically publish bytes beside their target.

    Args:
        path: Final artifact path.
        content: Complete artifact bytes.
        overwrite: Whether an existing non-accepted target may be replaced.

    Raises:
        FileExistsError: ``path`` exists and overwrite is false.
        OSError: Temporary or final publication fails.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.part-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    """Atomically write canonical strict JSON."""

    atomic_write_bytes(path, canonical_json_bytes(payload), overwrite=overwrite)


def atomic_write_yaml(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    """Atomically write deterministic YAML."""

    atomic_write_bytes(path, canonical_yaml_bytes(payload), overwrite=overwrite)


def atomic_copy_file(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    """Byte-copy one immutable asset and verify it before atomic publication."""

    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing copied asset: {target}")
    if source.stat().st_size != expected_size_bytes:
        raise ValueError(f"source size changed before copy: {source}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.part-",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if temporary_path.stat().st_size != expected_size_bytes:
            raise ValueError(f"copied asset size mismatch: {target}")
        if sha256_file(temporary_path) != expected_sha256:
            raise ValueError(f"copied asset hash mismatch: {target}")
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of in-memory bytes."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into a lowercase SHA-256 digest without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_output_directory(target: Path) -> Iterator[Path]:
    """Build a new directory privately and rename it into place on success.

    Existing targets are never overwritten. Failure cleanup is limited to the
    uniquely named staging directory created by this function.
    """

    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.part-", dir=target.parent))
    try:
        yield staging
        os.rename(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
