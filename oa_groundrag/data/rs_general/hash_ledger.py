"""严格解析并按需验证 RS-GeneralDesc hash ledger。"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar

from .io import (
    canonical_json,
    first_symlink_component,
    portable_relative_path,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .contracts import HASH_SCHEMA_VERSION
from .errors import RSGeneralDescError, ReasonCode, SchemaError


_CHUNK_SIZE = 8 * 1024 * 1024
_SHA256_LENGTH = 64
_FORBIDDEN_LEDGER_PATHS = frozenset(
    {
        "manifest.json",
        "metadata/hashes.json",
        "metadata/validation.json",
    }
)

_ErrorT = TypeVar("_ErrorT", bound=RSGeneralDescError)
_StatFingerprint = tuple[int, int, int, int, int, int]
_CacheKey = tuple[str, str, int, str, _StatFingerprint]
_VERIFIED_FILE_CACHE: set[_CacheKey] = set()
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class HashLedgerEntry:
    """Ledger 中一个已冻结文件的内容身份。"""

    path: str
    size_bytes: int
    sha256: str

    def canonical_row(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


class HashLedgerVerifier:
    """解析一次 ledger，并在文件被实际消费时验证其 bytes。"""

    def __init__(
        self,
        root: Path | str,
        *,
        ledger_path: str,
        expected_ledger_sha256: str,
        expected_root_sha256: str,
        error_type: type[_ErrorT] = SchemaError,
    ) -> None:
        self._error_type = error_type
        self.root = Path(root).absolute()
        if first_symlink_component(self.root) is not None or not self.root.is_dir():
            self._raise(
                ReasonCode.OUTPUT_LINK,
                "Benchmark root 必须是无链接普通目录",
                path=".",
            )
        self._root_resolved = self.root.resolve()
        self.ledger_relative = self._relative(ledger_path, location="hash ledger path")
        self._excluded_paths = _FORBIDDEN_LEDGER_PATHS | {self.ledger_relative}
        if self.ledger_relative in _FORBIDDEN_LEDGER_PATHS - {"metadata/hashes.json"}:
            self._raise(
                ReasonCode.SCHEMA_MISMATCH,
                "hash ledger path 非法",
                path=self.ledger_relative,
            )
        ledger_file = self._ordinary_file(self.ledger_relative)
        actual_ledger_sha256 = sha256_file(ledger_file)
        if actual_ledger_sha256 != expected_ledger_sha256:
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "hash ledger 文件 SHA-256 不匹配",
                path=self.ledger_relative,
                expected_sha256=expected_ledger_sha256,
                actual_sha256=actual_ledger_sha256,
            )
        try:
            value = read_json(ledger_file)
        except RSGeneralDescError as error:
            self._raise(
                error.code,
                "hash ledger 不是严格 JSON",
                path=self.ledger_relative,
                cause=str(error),
            )
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "files",
            "root_sha256",
        }:
            self._raise(
                ReasonCode.SCHEMA_MISMATCH,
                "hash ledger 顶层字段非法",
                path=self.ledger_relative,
            )
        if value.get("schema_version") != HASH_SCHEMA_VERSION:
            self._raise(
                ReasonCode.SCHEMA_MISMATCH,
                "hash ledger schema 非法",
                path=self.ledger_relative,
            )
        rows = value.get("files")
        if not isinstance(rows, list):
            self._raise(
                ReasonCode.TYPE_MISMATCH,
                "hash ledger files 必须是列表",
                path=self.ledger_relative,
            )
        entries: list[HashLedgerEntry] = []
        seen: set[str] = set()
        previous: str | None = None
        for index, row in enumerate(rows):
            location = f"{self.ledger_relative}#files[{index}]"
            if not isinstance(row, dict) or set(row) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                self._raise(
                    ReasonCode.SCHEMA_MISMATCH,
                    "hash ledger entry 字段非法",
                    path=location,
                )
            relative_value = row.get("path")
            if not isinstance(relative_value, str):
                self._raise(
                    ReasonCode.TYPE_MISMATCH,
                    "hash ledger entry path 必须是字符串",
                    path=location,
                )
            relative = self._relative(relative_value, location=location)
            if relative in self._excluded_paths:
                self._raise(
                    ReasonCode.SCHEMA_MISMATCH,
                    "hash ledger 不得包含自引用或生成态文件",
                    path=relative,
                )
            if relative in seen:
                self._raise(
                    ReasonCode.SCHEMA_MISMATCH,
                    "hash ledger path 重复",
                    path=relative,
                )
            if previous is not None and relative <= previous:
                self._raise(
                    ReasonCode.SCHEMA_MISMATCH,
                    "hash ledger entries 必须按 path 严格排序",
                    path=relative,
                )
            size_bytes = row.get("size_bytes")
            digest = row.get("sha256")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                self._raise(
                    ReasonCode.TYPE_MISMATCH,
                    "hash ledger size_bytes 必须是非负整数",
                    path=relative,
                )
            if not self._is_sha256(digest):
                self._raise(
                    ReasonCode.TYPE_MISMATCH,
                    "hash ledger sha256 必须是小写 SHA-256",
                    path=relative,
                )
            entry = HashLedgerEntry(relative, size_bytes, digest)
            entries.append(entry)
            seen.add(relative)
            previous = relative
        canonical_rows = [entry.canonical_row() for entry in entries]
        actual_root_sha256 = sha256_bytes(canonical_json(canonical_rows).encode("utf-8"))
        declared_root_sha256 = value.get("root_sha256")
        if (
            not self._is_sha256(declared_root_sha256)
            or declared_root_sha256 != actual_root_sha256
            or declared_root_sha256 != expected_root_sha256
        ):
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "hash ledger root identity 不匹配",
                path=self.ledger_relative,
                expected_sha256=expected_root_sha256,
                actual_sha256=actual_root_sha256,
            )
        self.ledger_sha256 = actual_ledger_sha256
        self.root_sha256 = actual_root_sha256
        self.entries = tuple(entries)
        self._entries = {entry.path: entry for entry in entries}
        self._verified_paths: set[str] = set()
        self._hash_counts: dict[str, int] = {}

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == _SHA256_LENGTH
            and value == value.lower()
            and all(character in "0123456789abcdef" for character in value)
        )

    def _raise(self, code: ReasonCode, message: str, **details: object) -> None:
        raise self._error_type(code, message, details=details)

    def _relative(self, value: str, *, location: str) -> str:
        try:
            pure = portable_relative_path(value, location=location)
        except RSGeneralDescError as error:
            self._raise(
                error.code,
                "hash ledger path 非法",
                path=value,
                cause=str(error),
            )
        return pure.as_posix()

    @staticmethod
    def _fingerprint(value: os.stat_result) -> _StatFingerprint:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def _ordinary_file(self, relative: str) -> Path:
        path = self.root.joinpath(*Path(relative).parts)
        linked = first_symlink_component(path)
        if linked is not None:
            self._raise(
                ReasonCode.OUTPUT_LINK,
                "ledger 文件路径含 symlink",
                path=relative,
            )
        try:
            path.resolve(strict=False).relative_to(self._root_resolved)
        except ValueError:
            self._raise(
                ReasonCode.PATH_ESCAPE,
                "ledger 文件路径逃逸 Benchmark root",
                path=relative,
            )
        try:
            file_stat = path.stat(follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            self._raise(
                ReasonCode.ASSET_MISSING,
                "ledger 文件不存在",
                path=relative,
            )
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or path.is_symlink()
            or file_stat.st_nlink != 1
        ):
            self._raise(
                ReasonCode.OUTPUT_LINK,
                "ledger 文件必须是普通单链接文件",
                path=relative,
            )
        return path

    def entry(self, relative: str) -> HashLedgerEntry:
        normalized = self._relative(relative, location="ledger lookup")
        entry = self._entries.get(normalized)
        if entry is None:
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "运行时文件不在 hash ledger 中",
                path=normalized,
            )
        return entry

    def assert_declared_identity(
        self,
        relative: str,
        *,
        size_bytes: object,
        sha256: object,
    ) -> HashLedgerEntry:
        entry = self.entry(relative)
        if size_bytes != entry.size_bytes or sha256 != entry.sha256:
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "文件声明与 hash ledger 不一致",
                path=entry.path,
                expected_size_bytes=entry.size_bytes,
                actual_size_bytes=size_bytes,
                expected_sha256=entry.sha256,
                actual_sha256=sha256,
            )
        return entry

    def verify_file(self, relative: str) -> Path:
        entry = self.entry(relative)
        path = self._ordinary_file(entry.path)
        before = path.stat(follow_symlinks=False)
        fingerprint = self._fingerprint(before)
        cache_key: _CacheKey = (
            str(self._root_resolved),
            entry.path,
            entry.size_bytes,
            entry.sha256,
            fingerprint,
        )
        with _CACHE_LOCK:
            cached = cache_key in _VERIFIED_FILE_CACHE
        if not cached:
            actual_sha256 = sha256_file(path, chunk_size=_CHUNK_SIZE)
            after = path.stat(follow_symlinks=False)
            if self._fingerprint(after) != fingerprint:
                self._raise(
                    ReasonCode.HASH_MISMATCH,
                    "文件在 hash 验证期间发生变化",
                    path=entry.path,
                )
            if after.st_size != entry.size_bytes or actual_sha256 != entry.sha256:
                self._raise(
                    ReasonCode.HASH_MISMATCH,
                    "文件 bytes 与 hash ledger 不一致",
                    path=entry.path,
                    expected_size_bytes=entry.size_bytes,
                    actual_size_bytes=after.st_size,
                    expected_sha256=entry.sha256,
                    actual_sha256=actual_sha256,
                )
            with _CACHE_LOCK:
                _VERIFIED_FILE_CACHE.add(cache_key)
            self._hash_counts[entry.path] = self._hash_counts.get(entry.path, 0) + 1
        self._verified_paths.add(entry.path)
        return path

    def assert_file_metadata(self, relative: str) -> Path:
        """只核对普通文件属性与 size，不读取文件内容。"""

        entry = self.entry(relative)
        path = self._ordinary_file(entry.path)
        actual_size = path.stat(follow_symlinks=False).st_size
        if actual_size != entry.size_bytes:
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "文件 size 与 hash ledger 不一致",
                path=entry.path,
                expected_size_bytes=entry.size_bytes,
                actual_size_bytes=actual_size,
            )
        return path

    def copy_verified_file(self, relative: str, target: Path) -> HashLedgerEntry:
        """复制并在同一次流式读取中校验源 bytes。"""

        entry = self.entry(relative)
        source = self._ordinary_file(entry.path)
        before = source.stat(follow_symlinks=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with source.open("rb") as input_handle, os.fdopen(
                descriptor, "wb"
            ) as output_handle:
                while True:
                    chunk = input_handle.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    output_handle.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            after = source.stat(follow_symlinks=False)
            actual_sha256 = digest.hexdigest()
            if (
                self._fingerprint(after) != self._fingerprint(before)
                or size_bytes != entry.size_bytes
                or actual_sha256 != entry.sha256
            ):
                self._raise(
                    ReasonCode.HASH_MISMATCH,
                    "复制时源文件 bytes 与 hash ledger 不一致",
                    path=entry.path,
                    expected_size_bytes=entry.size_bytes,
                    actual_size_bytes=size_bytes,
                    expected_sha256=entry.sha256,
                    actual_sha256=actual_sha256,
                )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        self._verified_paths.add(entry.path)
        self._hash_counts[entry.path] = self._hash_counts.get(entry.path, 0) + 1
        return entry

    def assert_entry_set(self, expected_paths: Iterable[str]) -> None:
        expected = {
            self._relative(path, location="expected payload path")
            for path in expected_paths
        }
        actual = set(self._entries)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "manifest/inventory 与 hash ledger 文件集合不一致",
                path=(missing or extra or [self.ledger_relative])[0],
                missing=missing,
                extra=extra,
            )

    def assert_actual_file_set(self) -> None:
        actual: set[str] = set()
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root).as_posix()
            file_stat = path.lstat()
            if path.is_symlink() or (
                not stat.S_ISDIR(file_stat.st_mode)
                and not stat.S_ISREG(file_stat.st_mode)
            ):
                self._raise(
                    ReasonCode.OUTPUT_LINK,
                    "Benchmark tree 含非普通文件",
                    path=relative,
                )
            if stat.S_ISDIR(file_stat.st_mode):
                continue
            if file_stat.st_nlink != 1:
                self._raise(
                    ReasonCode.OUTPUT_LINK,
                    "Benchmark tree 文件必须为单链接",
                    path=relative,
                )
            if relative in self._excluded_paths:
                continue
            actual.add(relative)
        expected = set(self._entries)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "实际 payload 文件集合与 hash ledger 不一致",
                path=(missing or extra or [self.ledger_relative])[0],
                missing=missing,
                extra=extra,
            )

    def assert_all_verified(self) -> None:
        missing = sorted(set(self._entries) - self._verified_paths)
        if missing:
            self._raise(
                ReasonCode.HASH_MISMATCH,
                "仍有 ledger entry 未完成 bytes 验证",
                path=missing[0],
                missing=missing,
            )

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(self._entries)

    @property
    def verified_paths(self) -> frozenset[str]:
        return frozenset(self._verified_paths)

    def hash_count(self, relative: str) -> int:
        return self._hash_counts.get(relative, 0)
