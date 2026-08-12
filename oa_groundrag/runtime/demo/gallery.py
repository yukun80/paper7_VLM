"""独立于 Benchmark/Frozen Eval 的 qualitative Demo Gallery 审计存储。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import atomic_write_json, first_symlink_component
from oa_groundrag.runtime.contracts import UnifiedTask

from .access import DemoAccessError, DemoTestAccessReceipt


GALLERY_ENTRY_SCHEMA = "oa_groundrag.demo_gallery.entry.v1"
GALLERY_MANIFEST_SCHEMA = "oa_groundrag.demo_gallery.manifest.v1"


class DemoGalleryError(RuntimeError):
    """Gallery revision、身份或人工选择合同失败。"""


@dataclass(frozen=True)
class DemoGalleryEntry:
    key: str
    benchmark_identity: Mapping[str, Any]
    sample_id: str
    split: str
    source: str
    demo_tags: tuple[str, ...]
    note: str
    selected_tasks: tuple[UnifiedTask, ...]
    selection_method: str
    test_access_receipt_id: str | None
    created_at: str
    updated_at: str
    deleted: bool = False
    schema_version: str = GALLERY_ENTRY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key,
            "benchmark_identity": dict(self.benchmark_identity),
            "sample_id": self.sample_id,
            "split": self.split,
            "source": self.source,
            "demo_tags": list(self.demo_tags),
            "note": self.note,
            "selected_tasks": [task.value for task in self.selected_tasks],
            "selection_method": self.selection_method,
            "test_access_receipt_id": self.test_access_receipt_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "qualitative_demo_only": True,
            "formal_acceptance": False,
            "scientific_acceptance": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DemoGalleryEntry":
        row = dict(value)
        required = {
            "schema_version", "key", "benchmark_identity", "sample_id", "split",
            "source", "demo_tags", "note", "selected_tasks", "selection_method",
            "test_access_receipt_id", "created_at", "updated_at", "deleted",
            "qualitative_demo_only", "formal_acceptance", "scientific_acceptance",
        }
        if set(row) != required:
            raise DemoGalleryError("Gallery entry 字段不匹配")
        if (
            row["schema_version"] != GALLERY_ENTRY_SCHEMA
            or row["qualitative_demo_only"] is not True
            or row["formal_acceptance"] is not False
            or row["scientific_acceptance"] is not False
            or row["selection_method"] != "manual_qualitative_selection"
        ):
            raise DemoGalleryError("Gallery entry 科学边界不匹配")
        try:
            tasks = tuple(UnifiedTask(value) for value in row["selected_tasks"])
        except (TypeError, ValueError) as error:
            raise DemoGalleryError("Gallery selected_tasks 非法") from error
        return cls(
            key=str(row["key"]),
            benchmark_identity=dict(row["benchmark_identity"]),
            sample_id=str(row["sample_id"]),
            split=str(row["split"]),
            source=str(row["source"]),
            demo_tags=tuple(str(value) for value in row["demo_tags"]),
            note=str(row["note"]),
            selected_tasks=tasks,
            selection_method=str(row["selection_method"]),
            test_access_receipt_id=row["test_access_receipt_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            deleted=bool(row["deleted"]),
        )


class DemoGalleryStore:
    """每次修改发布完整不可变 snapshot；``current.json`` 是原子指针。"""

    def __init__(self, demo_root: Path) -> None:
        self.demo_root = Path(os.path.abspath(demo_root))
        self.root = self.demo_root / "gallery"
        linked = first_symlink_component(self.root)
        if linked is not None or self.root.is_symlink():
            raise DemoGalleryError(f"Gallery root 含 symlink：{linked or self.root}")
        self.revisions_root = self.root / "revisions"
        self.revisions_root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".lock"
        if self.lock_path.is_symlink():
            raise DemoGalleryError("Gallery lock 不得是 symlink")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def entry_key(
        benchmark_identity: Mapping[str, Any],
        *,
        split: str,
        sample_id: str,
    ) -> str:
        return "gallery_" + sha256_text(canonical_json({
            "benchmark_identity": dict(benchmark_identity),
            "split": split,
            "sample_id": sample_id,
        }))[:24]

    def _pointer(self) -> Mapping[str, Any] | None:
        path = self.root / "current.json"
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise DemoGalleryError("Gallery current pointer 非普通文件")
        row = json.loads(path.read_text(encoding="utf-8"))
        if set(row) != {"schema_version", "revision_id", "manifest_sha256"}:
            raise DemoGalleryError("Gallery current pointer 字段不匹配")
        if row["schema_version"] != "oa_groundrag.demo_gallery.current.v1":
            raise DemoGalleryError("Gallery current pointer schema 不匹配")
        return row

    def _load_snapshot(self) -> tuple[int, list[DemoGalleryEntry]]:
        pointer = self._pointer()
        if pointer is None:
            return 0, []
        root = self.revisions_root / str(pointer["revision_id"])
        manifest_path = root / "manifest.json"
        entries_path = root / "entries.jsonl"
        ledger_path = root / "SHA256SUMS.jsonl"
        if (
            not root.is_dir()
            or root.is_symlink()
            or first_symlink_component(root) is not None
            or sha256_file(manifest_path) != pointer["manifest_sha256"]
        ):
            raise DemoGalleryError("Gallery revision identity 漂移")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != GALLERY_MANIFEST_SCHEMA
            or manifest.get("qualitative_demo_only") is not True
            or manifest.get("formal_acceptance") is not False
            or manifest.get("scientific_acceptance") is not False
        ):
            raise DemoGalleryError("Gallery manifest 科学边界不匹配")
        ledger = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if sha256_text(canonical_json(ledger)) != manifest.get("ledger_root_sha256"):
            raise DemoGalleryError("Gallery ledger identity 漂移")
        by_path = {str(row["path"]): row for row in ledger}
        entry_identity = by_path.get("entries.jsonl")
        if (
            entry_identity is None
            or sha256_file(entries_path) != entry_identity.get("sha256")
            or entries_path.stat().st_size != entry_identity.get("size_bytes")
        ):
            raise DemoGalleryError("Gallery entries payload 漂移")
        entries = [
            DemoGalleryEntry.from_dict(json.loads(line))
            for line in entries_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(entries) != manifest.get("entry_count"):
            raise DemoGalleryError("Gallery entry count 不匹配")
        if len({entry.key for entry in entries}) != len(entries):
            raise DemoGalleryError("Gallery snapshot key 重复")
        if sha256_text(canonical_json([entry.to_dict() for entry in entries])) != manifest.get("state_sha256"):
            raise DemoGalleryError("Gallery snapshot state identity 漂移")
        if manifest.get("revision_id") != pointer["revision_id"]:
            raise DemoGalleryError("Gallery revision pointer/manifest 不匹配")
        return int(manifest["revision_number"]), entries

    @staticmethod
    def _normalize_tags(values: Sequence[str]) -> tuple[str, ...]:
        tags = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
        if any(len(value) > 80 for value in tags):
            raise DemoGalleryError("demo tag 最长 80 字符")
        return tags

    @staticmethod
    def _normalize_tasks(values: Sequence[UnifiedTask | str]) -> tuple[UnifiedTask, ...]:
        try:
            tasks = tuple(
                value if isinstance(value, UnifiedTask) else UnifiedTask(value)
                for value in values
            )
        except (TypeError, ValueError) as error:
            raise DemoGalleryError("selected_tasks 含未知 UnifiedTask") from error
        if not tasks or len(tasks) != len(set(tasks)):
            raise DemoGalleryError("selected_tasks 必须非空且不重复")
        order = {task: index for index, task in enumerate(UnifiedTask)}
        return tuple(sorted(tasks, key=order.__getitem__))

    def _publish(self, *, revision_number: int, entries: Sequence[DemoGalleryEntry], action: str) -> str:
        state_sha = sha256_text(canonical_json([entry.to_dict() for entry in entries]))
        revision_id = f"r{revision_number:08d}_{state_sha[:12]}"
        target = self.revisions_root / revision_id
        with AtomicArtifactDirectory(target) as writer:
            writer.write_jsonl("entries.jsonl", (entry.to_dict() for entry in entries))
            assert writer.staging is not None
            entries_path = writer.path("entries.jsonl")
            ledger = [{
                "path": "entries.jsonl",
                "size_bytes": entries_path.stat().st_size,
                "sha256": sha256_file(entries_path),
            }]
            writer.write_jsonl("SHA256SUMS.jsonl", ledger)
            writer.write_json("manifest.json", {
                "schema_version": GALLERY_MANIFEST_SCHEMA,
                "revision_id": revision_id,
                "revision_number": revision_number,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "entry_count": len(entries),
                "active_entry_count": sum(not entry.deleted for entry in entries),
                "state_sha256": state_sha,
                "ledger_root_sha256": sha256_text(canonical_json(ledger)),
                "selection_method": "manual_qualitative_selection",
                "automatic_top_k_selection": False,
                "qualitative_demo_only": True,
                "formal_acceptance": False,
                "scientific_acceptance": False,
            })
            writer.publish()
        atomic_write_json(self.root / "current.json", {
            "schema_version": "oa_groundrag.demo_gallery.current.v1",
            "revision_id": revision_id,
            "manifest_sha256": sha256_file(target / "manifest.json"),
        })
        return revision_id

    def list_current(self) -> tuple[DemoGalleryEntry, ...]:
        _revision, entries = self._load_snapshot()
        return tuple(entry for entry in entries if not entry.deleted)

    def history(self) -> tuple[str, ...]:
        return tuple(
            path.name
            for path in sorted(self.revisions_root.iterdir())
            if path.is_dir() and not path.is_symlink()
        )

    def upsert(
        self,
        *,
        benchmark_identity: Mapping[str, Any],
        sample_id: str,
        split: str,
        source: str,
        demo_tags: Sequence[str],
        note: str,
        selected_tasks: Sequence[UnifiedTask | str],
        test_receipt: DemoTestAccessReceipt | None = None,
    ) -> str:
        if split not in {"train", "val", "test"} or not sample_id or not source:
            raise DemoGalleryError("Gallery sample split/id/source 非法")
        if len(note) > 4000:
            raise DemoGalleryError("Gallery note 最长 4000 字符")
        receipt_id: str | None = None
        if split == "test":
            if (
                test_receipt is None
                or test_receipt.action != "GALLERY"
                or test_receipt.sample_id != sample_id
            ):
                raise DemoGalleryError("test Gallery 必须绑定已发布的 GALLERY receipt")
            try:
                test_receipt.verify_published(
                    demo_root=self.demo_root,
                    required_action="GALLERY",
                    benchmark_identity=benchmark_identity,
                )
            except DemoAccessError as error:
                raise DemoGalleryError(
                    "test Gallery receipt 身份、位置或 Benchmark binding 非法"
                ) from error
            receipt_id = test_receipt.receipt_id
        elif test_receipt is not None:
            raise DemoGalleryError("train/val Gallery 不得绑定 test receipt")
        key = self.entry_key(benchmark_identity, split=split, sample_id=sample_id)
        now = datetime.now(timezone.utc).isoformat()
        tags = self._normalize_tags(demo_tags)
        tasks = self._normalize_tasks(selected_tasks)
        with self._lock():
            revision, entries = self._load_snapshot()
            by_key = {entry.key: entry for entry in entries}
            previous = by_key.get(key)
            by_key[key] = DemoGalleryEntry(
                key=key,
                benchmark_identity=dict(benchmark_identity),
                sample_id=sample_id,
                split=split,
                source=source,
                demo_tags=tags,
                note=note,
                selected_tasks=tasks,
                selection_method="manual_qualitative_selection",
                test_access_receipt_id=receipt_id,
                created_at=now if previous is None else previous.created_at,
                updated_at=now,
                deleted=False,
            )
            ordered = tuple(by_key[value] for value in sorted(by_key))
            return self._publish(
                revision_number=revision + 1,
                entries=ordered,
                action="ADD" if previous is None or previous.deleted else "UPDATE",
            )

    def remove(
        self,
        *,
        benchmark_identity: Mapping[str, Any],
        split: str,
        sample_id: str,
    ) -> str:
        key = self.entry_key(benchmark_identity, split=split, sample_id=sample_id)
        with self._lock():
            revision, entries = self._load_snapshot()
            by_key = {entry.key: entry for entry in entries}
            previous = by_key.get(key)
            if previous is None or previous.deleted:
                raise DemoGalleryError("Gallery 当前视图不存在该样本")
            by_key[key] = replace(
                previous,
                updated_at=datetime.now(timezone.utc).isoformat(),
                deleted=True,
            )
            ordered = tuple(by_key[value] for value in sorted(by_key))
            return self._publish(
                revision_number=revision + 1,
                entries=ordered,
                action="REMOVE_TOMBSTONE",
            )
