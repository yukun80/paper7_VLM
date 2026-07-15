#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Description Benchmark M0/M1 公共协议与 I/O 工具。

用途：统一项目路径、源/物化索引记录、图像探测、hash、DIOR bbox、确定性抽样和原子写入。
主要输入：external/RSGPT、仓库同级 datasets/MMRS-1M 与阶段脚本参数。
主要输出：供 3-1 到 3-7 调用的结构化数据；本模块不作为独立入口。
写入行为：仅调用方显式写入派生 benchmark，不修改原始数据。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image, ImageChops, ImageStat


SCHEMA_VERSION = "qpsalm_description_v2"
BUILDER_VERSION = "description_benchmark_m1_v4_answer_trace"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _external_root(name: str) -> Path:
    sibling = REPO_ROOT.parent / name
    legacy = REPO_ROOT / name
    return sibling if sibling.exists() or not legacy.exists() else legacy


DATASETS_ROOT = Path(
    os.environ.get("PAPER7_DATASETS_ROOT")
    or os.environ.get("DATASETS_ROOT")
    or _external_root("datasets")
).expanduser().resolve(strict=False)
BENCHMARK_ROOT = Path(
    os.environ.get("PAPER7_BENCHMARK_ROOT") or _external_root("benchmark")
).expanduser().resolve(strict=False)
MMRS_ROOT = (DATASETS_ROOT / "MMRS-1M").resolve(strict=False)


class InvalidBoundingBoxError(ValueError):
    """源标注 bbox 可解析但几何无效；构建器应排除该区域并记录。"""


def _rsgpt_data_root() -> Path:
    override = os.environ.get("PAPER7_RSGPT_DATA_ROOT")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute() and path.parts and path.parts[0] == "datasets":
            path = DATASETS_ROOT.joinpath(*path.parts[1:])
        elif not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve(strict=False)
    candidates = [
        DATASETS_ROOT / "RSGPT/dataset",
        DATASETS_ROOT / "dataset",
        REPO_ROOT / "external/RSGPT/dataset",
    ]
    return next((path.resolve(strict=False) for path in candidates if path.exists()), candidates[0].resolve(strict=False))


RSGPT_ROOT = _rsgpt_data_root()


def resolve_project_path(ref: str | Path) -> Path:
    path = Path(ref).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if path.parts and path.parts[0] == "datasets":
        return DATASETS_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    if path.parts and path.parts[0] == "benchmark":
        return BENCHMARK_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def to_project_ref(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute():
        return source.as_posix()
    source = source.resolve(strict=False)
    for logical, root in (("datasets", DATASETS_ROOT), ("benchmark", BENCHMARK_ROOT)):
        try:
            return (Path(logical) / source.relative_to(root)).as_posix()
        except ValueError:
            pass
    try:
        return source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return source.as_posix()


def description_dir_for_mode(mode: str, output_dir: str | Path | None = None) -> Path:
    if output_dir:
        return resolve_project_path(output_dir)
    if mode not in {"small", "full"}:
        raise ValueError(f"mode 必须是 small/full，当前为 {mode!r}")
    return BENCHMARK_ROOT / f"qpsalm_description_v2_{mode}"


def mmrs_data_path(ref: str) -> Path:
    clean = str(ref).replace("\\", "/")
    if clean.startswith("data/"):
        clean = clean[5:]
    return (MMRS_ROOT / clean).resolve(strict=False)


def ensure_writable(path: Path, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"输出已存在，请使用 --overwrite: {path}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
    )
    return len(materialized)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: 非法 JSONL: {exc}") from exc
    return rows


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")
    return f"{safe}_{stable_hash(*parts)[:length]}"


def source_slug(value: str) -> str:
    """将数据源名称转换为稳定、可移植的目录名。"""
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug:
        raise ValueError(f"无法生成 source slug: {value!r}")
    return slug


def probe_image(path: Path, include_hash: bool = True) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"正式描述输入必须是 RGB/RGBA: mode={image.mode} path={path}")
        width, height = image.size
        gray = image.convert("RGB").resize((9, 8), Image.Resampling.BILINEAR).convert("L")
        pixels = list(gray.getdata())
        bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
        dhash = sum(int(bit) << index for index, bit in enumerate(bits))
    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "sha256": sha256_file(path) if include_hash else None,
        "dhash64": f"{dhash:016x}",
    }


def perceptual_rgb_mae(
    left: str | Path,
    right: str | Path,
    *,
    size: int = 64,
) -> float:
    """在统一 RGB 画布上计算逐通道 MAE，用于验证 dHash 候选。

    dHash 只负责召回外观近似项；这里保留颜色和亮度差异，用一个可审计的
    像素误差门槛判断是否属于同图重编码。该函数不依赖 NumPy，构建脚本可在
    仅安装 Pillow 的环境中运行。
    """
    if size < 8:
        raise ValueError("perceptual preview size 必须至少为 8")

    def load(path_ref: str | Path) -> Image.Image:
        path = resolve_project_path(path_ref)
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"近重复验证只支持 RGB/RGBA: mode={image.mode} path={path}")
            return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)

    difference = ImageChops.difference(load(left), load(right))
    channel_means = ImageStat.Stat(difference).mean
    return float(sum(channel_means) / max(len(channel_means), 1))


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def scene_prefix(filename: str) -> str | None:
    match = re.match(r"^(P\d+)_", Path(filename).name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def deduplicate_texts(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value).strip().split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def caption_quality(text: str, source: str) -> tuple[float, list[str]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    if not tokens:
        return 0.0, ["empty_caption"]
    flags: list[str] = []
    if len(tokens) < 4:
        flags.append("low_information_caption")
    if source == "RSICap" and re.search(
        r"\b(sunny|rainy|winter|summer|spring|autumn|morning|afternoon|weather|season)\b",
        text.casefold(),
    ):
        flags.append("low_verifiability")
    return (0.5 if flags else 1.0), flags


def single_image_visual_ref(path: Path, source_dataset: str, meta: dict[str, Any]) -> dict[str, Any]:
    sensor = "generic_aerial_rgb"
    if source_dataset in {"RSICap", "RSIEval"}:
        sensor = "dota_derived_aerial_rgb"
    elif source_dataset == "DIOR-RSVG":
        sensor = "dior_aerial_rgb"
    return {
        "type": "single_image",
        "path": to_project_ref(path),
        "width": meta["width"],
        "height": meta["height"],
        "sha256": meta["sha256"],
        "dhash64": meta["dhash64"],
        "modality_instance": {
            "family": "optical",
            "sensor": sensor,
            "product_type": "rgb",
            "band_names": ["R", "G", "B"],
            "units": "display_rgb",
            "native_gsd_m": None,
            "aligned_gsd_m": None,
            "valid_mask": "all_valid_decoded_pixels",
            "quality": 1.0,
        },
    }


def full_image_geometry() -> dict[str, Any]:
    return {
        "type": "full_image",
        "mask_path": None,
        "bbox_xyxy_normalized": None,
        "bbox_xyxy_pixel_half_open": None,
        "coordinate_space": "original_image",
    }


def base_record(
    *, sample_id: str, parent_sample_id: str, source_dataset: str, task_family: str,
    visual_ref: dict[str, Any], region_geometry: dict[str, Any], target_status: str,
    region_source: str, instruction: str, answer_type: str, answers: list[dict[str, Any]],
    provenance: dict[str, Any], quality_flags: Sequence[str] = (), split: str | None = None,
    region_pair_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "parent_sample_id": parent_sample_id,
        "region_pair_id": region_pair_id,
        "source_dataset": source_dataset,
        "component_benchmark": (
            "rs_global_caption_v1" if task_family == "global_caption" else "rs_region_alignment_v1"
        ),
        "split": split,
        "task_family": task_family,
        "visual_ref": visual_ref,
        "region_geometry": region_geometry,
        "target_status": target_status,
        "region_source": region_source,
        "instruction": instruction,
        "answer_type": answer_type,
        "answers": answers,
        "structured_targets": {},
        "provenance": {"builder_version": BUILDER_VERSION, **provenance},
        "quality_flags": sorted(set(quality_flags)),
    }


def answer_record(text: str, origin: str, source: str, *, alignment: bool = False) -> tuple[dict[str, Any], list[str]]:
    weight, flags = (1.0, []) if alignment else caption_quality(text, source)
    return {
        "text": text,
        "language": "en",
        "annotation_origin": origin,
        "quality": weight,
        "caption_quality_weight": weight,
    }, flags


def iter_turn_pairs(conversations: Sequence[dict[str, Any]]) -> Iterator[tuple[int, str, str]]:
    if len(conversations) % 2:
        raise ValueError("conversation turn 数必须为偶数")
    for index in range(0, len(conversations), 2):
        first, second = conversations[index], conversations[index + 1]
        if first.get("from") != "human" or second.get("from") != "gpt":
            raise ValueError(f"conversation role 非 human->gpt: {first.get('from')}->{second.get('from')}")
        yield index // 2, str(first.get("value", "")).strip(), str(second.get("value", "")).strip()


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    matches = re.findall(r"\[[^\[\]]+\]", text)
    if not matches:
        raise ValueError(f"未找到 bbox: {text[:120]!r}")
    value = ast.literal_eval(matches[-1])
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox 必须包含四个数: {value!r}")
    bbox = tuple(float(item) for item in value)
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(item) for item in bbox) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise InvalidBoundingBoxError(f"normalized bbox 非法: {bbox}")
    return bbox


def bbox_pixel_half_open(bbox: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, math.floor(x1 * width)))
    top = max(0, min(height - 1, math.floor(y1 * height)))
    right = max(left + 1, min(width, math.ceil(x2 * width)))
    bottom = max(top + 1, min(height, math.ceil(y2 * height)))
    return [left, top, right, bottom]


def normalize_phrase(value: str) -> str:
    return " ".join(value.strip().split()).casefold().rstrip(" .")


def deterministic_split(key: str, train_ratio: float = 0.9) -> str:
    ratio = int(stable_hash(key)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return "train" if ratio < train_ratio else "dev"


def select_parent_ids(
    parent_rows: Sequence[dict[str, Any]], limit: int, strata_fields: Sequence[str]
) -> set[str]:
    if limit <= 0 or len(parent_rows) <= limit:
        return {str(row["parent_sample_id"]) for row in parent_rows}
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        key = tuple(str(row.get(field, "unknown")) for field in strata_fields)
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_hash(row["parent_sample_id"]))
    selected: set[str] = set()
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.add(str(groups[key].pop()["parent_sample_id"]))
                progressed = True
        if not progressed:
            break
    return selected


def sha256_jsonl_rows(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
