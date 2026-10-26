"""新输出根的原子目录 writer；JSON/JSONL/二进制职责分离。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import (
    PortablePathError,
    atomic_write_json,
    atomic_write_jsonl,
    first_symlink_component,
    portable_relative_path,
)

from oa_groundrag.vlm.errors import ContractError, ReasonCode


class AtomicArtifactDirectory:
    def __init__(self, target: Path) -> None:
        self.target = Path(target)
        self.staging: Path | None = None
        self._published = False

    def __enter__(self) -> "AtomicArtifactDirectory":
        linked = first_symlink_component(self.target)
        if linked is not None:
            raise ContractError(
                ReasonCode.OUTPUT_LINK,
                f"artifact output_root 含链接组件：{linked}",
            )
        if self.target.exists() or self.target.is_symlink():
            raise ContractError(
                ReasonCode.OUTPUT_EXISTS,
                f"artifact output_root 已存在：{self.target}",
            )
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.target.name}.staging-",
                dir=self.target.parent,
            )
        )
        return self

    def path(self, relative: str) -> Path:
        if self.staging is None:
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "ArtifactDirectory 尚未进入 context",
            )
        try:
            pure = portable_relative_path(relative)
        except PortablePathError as error:
            raise ContractError(
                ReasonCode.PATH_ESCAPE,
                f"artifact 非法相对路径：{relative}",
            ) from error
        path = self.staging.joinpath(*pure.parts)
        try:
            path.resolve(strict=False).relative_to(self.staging.resolve())
        except ValueError as error:
            raise ContractError(
                ReasonCode.PATH_ESCAPE,
                f"artifact 路径逃逸：{relative}",
            ) from error
        return path

    def write_json(self, relative: str, value: Any) -> None:
        path = self.path(relative)
        if path.exists() or path.is_symlink():
            raise ContractError(
                ReasonCode.OUTPUT_EXISTS,
                f"artifact 已存在：{relative}",
            )
        atomic_write_json(path, value)

    def write_jsonl(
        self,
        relative: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> None:
        path = self.path(relative)
        if path.exists() or path.is_symlink():
            raise ContractError(
                ReasonCode.OUTPUT_EXISTS,
                f"artifact 已存在：{relative}",
            )
        atomic_write_jsonl(path, rows)

    def publish(self) -> Path:
        if self.staging is None:
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "ArtifactDirectory 尚未进入 context",
            )
        if self._published:
            raise ContractError(
                ReasonCode.OUTPUT_EXISTS,
                "ArtifactDirectory 已发布",
            )
        if self.target.exists() or self.target.is_symlink():
            raise ContractError(
                ReasonCode.OUTPUT_EXISTS,
                f"发布前 output_root 已存在：{self.target}",
            )
        self.staging.replace(self.target)
        self._published = True
        return self.target

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if (
            not self._published
            and self.staging is not None
            and self.staging.exists()
        ):
            shutil.rmtree(self.staging)
