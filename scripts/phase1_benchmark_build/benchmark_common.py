"""阶段 1B Benchmark 公共合同、重采样、原子 I/O 与 DataLoader。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset


SCHEMA_VERSION = "oa_auxseg_hdf5_v1"
DEFAULT_PATCH_SIZE = 224
DEFAULT_SEED = 20260724
SOURCE_ORDER = (
    "gdcld",
    "lmhld",
    "landslidebench_agent",
    "landslide4sense",
    "multimodal_landslide",
)
SPLIT_ORDER = ("train", "val", "test")


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return sha256_bytes(payload.encode("utf-8"))


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
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
    _atomic_replace_bytes(
        path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(f"{canonical_json(dict(row))}\n" for row in rows)
    _atomic_replace_bytes(path, payload.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: JSONL 不允许空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 行必须是对象")
            rows.append(value)
    return rows


def ensure_binary(name: str, array: np.ndarray) -> np.ndarray:
    values = np.unique(array)
    if not set(values.tolist()).issubset({0, 1}):
        raise ValueError(f"{name} 必须是二值数组，实际值包含 {values[:20].tolist()}")
    return array.astype(np.uint8, copy=False)


def _resize(
    array: np.ndarray, target_size: int, *, mode: str, output_dtype: np.dtype[Any]
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32)
    squeeze_channel = tensor.ndim == 2
    if squeeze_channel:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"resize 输入必须是 HW 或 CHW，实际 shape={tuple(array.shape)}")
    kwargs: dict[str, Any] = {"size": (target_size, target_size), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = functional.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)
    if squeeze_channel:
        resized = resized.squeeze(0)
    return resized.cpu().numpy().astype(output_dtype, copy=False)


def resize_continuous_with_validity(
    values: np.ndarray,
    pixel_valid: np.ndarray,
    channel_valid: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """无效值直接置零，连续值双线性，validity 最近邻。"""
    values = np.asarray(values, dtype=np.float32)
    pixel_valid = np.asarray(pixel_valid, dtype=np.uint8)
    channel_valid = np.asarray(channel_valid, dtype=np.uint8)
    if values.ndim != 3:
        raise ValueError(f"连续影像必须是 CHW，实际 shape={values.shape}")
    if pixel_valid.shape != values.shape:
        raise ValueError(
            f"pixel_valid shape={pixel_valid.shape} 与影像 shape={values.shape} 不一致"
        )
    if channel_valid.shape != (values.shape[0],):
        raise ValueError(
            f"channel_valid shape={channel_valid.shape}，预期 {(values.shape[0],)}"
        )
    ensure_binary("pixel_valid", pixel_valid)
    ensure_binary("channel_valid", channel_valid)
    finite = np.isfinite(values)
    valid = (
        pixel_valid.astype(bool)
        & finite
        & channel_valid[:, None, None].astype(bool)
    )
    clean = np.where(valid, values, 0.0).astype(np.float32, copy=False)
    resized_values = _resize(
        clean, target_size, mode="bilinear", output_dtype=np.dtype("float32")
    )
    resized_valid = _resize(
        valid.astype(np.uint8),
        target_size,
        mode="nearest",
        output_dtype=np.dtype("uint8"),
    )
    resized_valid = ensure_binary("resized pixel_valid", resized_valid)
    resized_values[resized_valid == 0] = 0.0
    if not np.isfinite(resized_values).all():
        raise ValueError("重采样后连续影像仍包含 NaN 或 Inf")
    return resized_values, resized_valid, channel_valid


def resize_binary_mask(
    mask: np.ndarray, source_valid: np.ndarray | None, target_size: int
) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是 HW 或 1HW，实际 shape={mask.shape}")
    finite = np.isfinite(mask)
    clean = np.where(finite, mask, 0)
    ensure_binary("source mask", clean)
    if source_valid is not None:
        valid = np.asarray(source_valid)
        if valid.ndim == 3 and valid.shape[0] == 1:
            valid = valid[0]
        if valid.shape != clean.shape:
            raise ValueError(
                f"标签 valid shape={valid.shape} 与 mask shape={clean.shape} 不一致"
            )
        ensure_binary("source label valid", valid)
        clean = np.where(valid.astype(bool), clean, 0)
    resized = _resize(
        clean.astype(np.uint8),
        target_size,
        mode="nearest",
        output_dtype=np.dtype("uint8"),
    )
    return ensure_binary("resized mask", resized)[None, ...]


@dataclass
class RunningChannelStats:
    count: np.ndarray
    total: np.ndarray
    total_square: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def create(cls, channels: int) -> "RunningChannelStats":
        return cls(
            count=np.zeros(channels, dtype=np.int64),
            total=np.zeros(channels, dtype=np.float64),
            total_square=np.zeros(channels, dtype=np.float64),
            minimum=np.full(channels, np.inf, dtype=np.float64),
            maximum=np.full(channels, -np.inf, dtype=np.float64),
        )

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        for index in range(values.shape[0]):
            selected = values[index][valid[index].astype(bool)].astype(
                np.float64, copy=False
            )
            if selected.size == 0:
                continue
            self.count[index] += selected.size
            self.total[index] += selected.sum(dtype=np.float64)
            self.total_square[index] += np.square(selected).sum(dtype=np.float64)
            self.minimum[index] = min(self.minimum[index], float(selected.min()))
            self.maximum[index] = max(self.maximum[index], float(selected.max()))

    def to_rows(self, names: Sequence[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            count = int(self.count[index])
            if count:
                mean = float(self.total[index] / count)
                variance = max(float(self.total_square[index] / count) - mean**2, 0.0)
                std = math.sqrt(variance)
                minimum: float | None = float(self.minimum[index])
                maximum: float | None = float(self.maximum[index])
            else:
                mean = std = 0.0
                minimum = maximum = None
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "valid_pixel_count": count,
                    "mean": mean,
                    "std": std,
                    "min": minimum,
                    "max": maximum,
                }
            )
        return rows


def load_index(index_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(index_path)


class BenchmarkDataset(Dataset[dict[str, Any]]):
    """统一 Benchmark 随机读取接口；单样本 mask shape 为 [1,H,W]。"""

    def __init__(
        self,
        root: Path | str,
        *,
        split: str | None = None,
        auxiliary_policy: str = "all",
        normalization: str = "none",
    ) -> None:
        self.root = Path(root).resolve()
        self.rows = load_index(self.root / "index.jsonl")
        if split is not None:
            self.rows = [row for row in self.rows if row["split"] == split]
        if auxiliary_policy not in {"none", "single", "all"}:
            raise ValueError("auxiliary_policy 必须是 none、single 或 all")
        if normalization not in {"none", "zscore"}:
            raise ValueError("normalization 必须是 none 或 zscore")
        self.auxiliary_policy = auxiliary_policy
        self.normalization = normalization
        self.statistics = read_json(self.root / "source_statistics.json")

    def __len__(self) -> int:
        return len(self.rows)

    def _normalization_rows(
        self, source: str, modality: str
    ) -> list[dict[str, Any]] | None:
        source_stats = self.statistics.get("sources", {}).get(source, {})
        return source_stats.get(modality)

    def _normalize(
        self, values: np.ndarray, valid: np.ndarray, rows: list[dict[str, Any]] | None
    ) -> np.ndarray:
        if self.normalization == "none" or rows is None:
            return values
        result = values.copy()
        for index, row in enumerate(rows):
            std = float(row["std"])
            if std <= 0:
                continue
            channel_valid = valid[index].astype(bool)
            result[index][channel_valid] = (
                result[index][channel_valid] - float(row["mean"])
            ) / std
            result[index][~channel_valid] = 0.0
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        shard_path = self.root / row["storage"]["shard"]
        row_index = int(row["storage"]["row"])
        with h5py.File(shard_path, "r") as handle:
            optical = np.asarray(handle["optical"][row_index], dtype=np.float32)
            optical_valid = np.asarray(
                handle["optical_pixel_valid"][row_index], dtype=np.uint8
            )
            optical_channel_valid = np.asarray(
                handle["optical_channel_valid"][row_index], dtype=np.uint8
            )
            mask = np.asarray(handle["mask"][row_index], dtype=np.uint8)
            optical = self._normalize(
                optical,
                optical_valid,
                self._normalization_rows(row["source"], "optical"),
            )
            auxiliaries: dict[str, dict[str, Any]] = {}
            names = sorted(row["auxiliaries"])
            if self.auxiliary_policy == "none":
                names = []
            elif self.auxiliary_policy == "single":
                names = names[:1]
            for name in names:
                group = handle[f"auxiliary/{name}"]
                values = np.asarray(group["values"][row_index], dtype=np.float32)
                valid = np.asarray(group["pixel_valid"][row_index], dtype=np.uint8)
                values = self._normalize(
                    values,
                    valid,
                    self._normalization_rows(row["source"], name),
                )
                auxiliaries[name] = {
                    "values": torch.from_numpy(values),
                    "pixel_valid": torch.from_numpy(valid),
                    "channel_valid": torch.from_numpy(
                        np.asarray(group["channel_valid"][row_index], dtype=np.uint8)
                    ),
                    "channel_names": list(row["auxiliaries"][name]["channel_names"]),
                }
        return {
            "sample_id": row["sample_id"],
            "optical": torch.from_numpy(optical),
            "optical_pixel_valid": torch.from_numpy(optical_valid),
            "optical_channel_valid": torch.from_numpy(optical_channel_valid),
            "optical_channel_names": list(row["optical"]["channel_names"]),
            "mask": torch.from_numpy(mask),
            "auxiliaries": auxiliaries,
            "metadata": {
                "source": row["source"],
                "split": row["split"],
                "original_size": row["resize"]["original_size"],
                "target_size": row["resize"]["target_size"],
            },
        }


def collate_benchmark_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("不能 collate 空样本列表")
    masks = torch.stack([sample["mask"] for sample in samples], dim=0)
    auxiliaries: dict[str, dict[str, Any]] = {}
    auxiliary_names = sorted(
        {
            name
            for sample in samples
            for name in sample["auxiliaries"]
        }
    )
    for name in auxiliary_names:
        selected = [
            (index, sample["auxiliaries"][name])
            for index, sample in enumerate(samples)
            if name in sample["auxiliaries"]
        ]
        channel_names = [item["channel_names"] for _, item in selected]
        if len({tuple(names) for names in channel_names}) != 1:
            raise ValueError(f"同名辅助模态 {name} 的通道合同不一致")
        auxiliaries[name] = {
            "sample_indices": torch.tensor(
                [index for index, _ in selected], dtype=torch.int64
            ),
            "values": torch.stack([item["values"] for _, item in selected], dim=0),
            "pixel_valid": torch.stack(
                [item["pixel_valid"] for _, item in selected], dim=0
            ),
            "channel_valid": torch.stack(
                [item["channel_valid"] for _, item in selected], dim=0
            ),
            "channel_names": channel_names[0],
        }
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "optical": [sample["optical"] for sample in samples],
        "optical_pixel_valid": [
            sample["optical_pixel_valid"] for sample in samples
        ],
        "optical_channel_valid": [
            sample["optical_channel_valid"] for sample in samples
        ],
        "optical_channel_names": [
            sample["optical_channel_names"] for sample in samples
        ],
        "mask": masks,
        "auxiliaries": auxiliaries,
        "metadata": [sample["metadata"] for sample in samples],
    }


def iter_chunks(values: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]
