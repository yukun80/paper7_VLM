"""只在已有 global/candidate mask 中进行区域选择。"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .contracts import (
    RegionCandidate,
    RegionInventory,
    SelectedRegion,
    SelectionMode,
    SelectionRequest,
    TargetStatus,
)
from .errors import ReasonCode, SelectionError


_LOCATIONS = (
    ("northwest", "north", "northeast"),
    ("west", "center", "east"),
    ("southwest", "south", "southeast"),
)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """返回半开区间 pixel xyxy。"""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise SelectionError(
            ReasonCode.MASK_INVALID,
            "不能为 empty mask 计算 bbox",
        )
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise SelectionError(
            ReasonCode.MASK_INVALID,
            "不能为 empty mask 计算 centroid",
        )
    return float(xs.mean()), float(ys.mean())


def _location(mask: np.ndarray) -> str:
    height, width = mask.shape
    x, y = mask_centroid(mask)
    column = min(2, int(3 * (x + 0.5) / width))
    row = min(2, int(3 * (y + 0.5) / height))
    return _LOCATIONS[row][column]


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = float(
        max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def parse_numbered_region_id(value: str) -> str:
    """冻结 selector 的唯一接受格式：严格 JSON ``{"region_id": "..."}``。"""

    if not isinstance(value, str):
        raise SelectionError(
            ReasonCode.TYPE_MISMATCH,
            "numbered selector 输出必须是字符串",
        )
    try:
        row = json.loads(
            value,
            object_pairs_hook=lambda pairs: _unique_object(pairs),
            parse_constant=lambda token: _invalid_constant(token),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SelectionError(
            ReasonCode.REGION_ID_INVALID,
            "numbered selector 输出不是严格 JSON",
            details={"error": str(error)},
        ) from error
    if (
        not isinstance(row, dict)
        or set(row) != {"region_id"}
        or not isinstance(row["region_id"], (str, int))
        or isinstance(row["region_id"], bool)
    ):
        raise SelectionError(
            ReasonCode.REGION_ID_INVALID,
            "numbered selector 只允许单一 region_id 字段",
        )
    region_id = str(row["region_id"]).strip()
    if not region_id:
        raise SelectionError(
            ReasonCode.REGION_ID_INVALID,
            "numbered selector region_id 不能为空",
        )
    return region_id


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-finite constant: {token}")


class RegionSelector:
    """确定性 selector；不生成、合并或修改像素 mask。"""

    def select(
        self,
        inventory: RegionInventory,
        request: SelectionRequest,
    ) -> SelectedRegion:
        if not bool(inventory.global_mask.any()):
            return SelectedRegion(
                sample_id=inventory.sample_id,
                target_status=TargetStatus.NO_TARGET,
                selection_mode=request.mode,
                region_id=None,
                mask=None,
                source_identity=inventory.source_identity,
            )
        if request.mode in {SelectionMode.GLOBAL, SelectionMode.ALL}:
            return self._present(
                inventory,
                request,
                region_id="global",
                mask=inventory.global_mask,
            )
        candidates = inventory.candidates
        if not candidates:
            raise SelectionError(
                ReasonCode.REGION_NOT_FOUND,
                f"{inventory.sample_id}: global 非空但没有 candidate regions",
            )
        if request.mode is SelectionMode.REGION_ID:
            assert request.region_id is not None
            return self._by_id(inventory, request, request.region_id)
        if request.mode is SelectionMode.NUMBERED_OVERLAY:
            assert request.numbered_response is not None
            return self._by_id(
                inventory,
                request,
                parse_numbered_region_id(request.numbered_response),
            )
        if request.mode is SelectionMode.BBOX:
            assert request.bbox_xyxy is not None
            query = self._validate_bbox(request.bbox_xyxy, inventory)
            scored = [
                (_bbox_iou(query, mask_bbox(candidate.mask)), candidate)
                for candidate in candidates
            ]
            maximum = max(score for score, _ in scored)
            if maximum <= 0:
                raise SelectionError(
                    ReasonCode.REGION_NOT_FOUND,
                    "bbox 与所有已有候选的 IoU 均为 0",
                )
            winners = [
                candidate
                for score, candidate in scored
                if math.isclose(score, maximum, rel_tol=0.0, abs_tol=1e-12)
            ]
            if len(winners) != 1:
                raise SelectionError(
                    ReasonCode.AMBIGUOUS_REGION,
                    "bbox 最大 IoU 并列，拒绝隐式破同分",
                    details={
                        "region_ids": sorted(
                            candidate.region_id for candidate in winners
                        ),
                        "iou": maximum,
                    },
                )
            return self._candidate(inventory, request, winners[0])
        if request.mode is SelectionMode.CLICK:
            assert request.click_xy is not None
            x, y = self._validate_click(request.click_xy, inventory)
            matches = [
                candidate
                for candidate in candidates
                if bool(candidate.mask[y, x])
            ]
            if not matches:
                raise SelectionError(
                    ReasonCode.REGION_NOT_FOUND,
                    "click 不在任何已有候选内",
                )
            candidate = min(
                matches,
                key=lambda item: (
                    int(item.mask.sum()),
                    item.region_id,
                ),
            )
            return self._candidate(inventory, request, candidate)
        if request.mode is SelectionMode.AREA_RANK:
            assert request.area_rank is not None
            ordered = sorted(
                candidates,
                key=lambda item: (-int(item.mask.sum()), item.region_id),
            )
            if request.area_rank > len(ordered):
                raise SelectionError(
                    ReasonCode.REGION_NOT_FOUND,
                    f"area_rank={request.area_rank} 超出候选数 {len(ordered)}",
                )
            return self._candidate(
                inventory,
                request,
                ordered[request.area_rank - 1],
            )
        if request.mode is SelectionMode.LOCATION:
            assert request.location is not None
            matches = [
                candidate
                for candidate in candidates
                if _location(candidate.mask) == request.location
            ]
            if not matches:
                raise SelectionError(
                    ReasonCode.REGION_NOT_FOUND,
                    f"3x3 location={request.location} 无已有候选",
                )
            candidate = min(
                matches,
                key=lambda item: (-int(item.mask.sum()), item.region_id),
            )
            return self._candidate(inventory, request, candidate)
        raise SelectionError(
            ReasonCode.INVALID_ENUM,
            f"未实现 selection mode={request.mode.value}",
        )

    def _by_id(
        self,
        inventory: RegionInventory,
        request: SelectionRequest,
        region_id: str,
    ) -> SelectedRegion:
        matches = [
            candidate
            for candidate in inventory.candidates
            if candidate.region_id == region_id
        ]
        if len(matches) != 1:
            raise SelectionError(
                ReasonCode.REGION_ID_INVALID,
                f"region_id 不在已有候选中：{region_id}",
            )
        return self._candidate(inventory, request, matches[0])

    def _candidate(
        self,
        inventory: RegionInventory,
        request: SelectionRequest,
        candidate: RegionCandidate,
    ) -> SelectedRegion:
        return self._present(
            inventory,
            request,
            region_id=candidate.region_id,
            mask=candidate.mask,
        )

    @staticmethod
    def _present(
        inventory: RegionInventory,
        request: SelectionRequest,
        *,
        region_id: str,
        mask: np.ndarray,
    ) -> SelectedRegion:
        return SelectedRegion(
            sample_id=inventory.sample_id,
            target_status=TargetStatus.TARGET_PRESENT,
            selection_mode=request.mode,
            region_id=region_id,
            mask=mask,
            source_identity=inventory.source_identity,
        )

    @staticmethod
    def _validate_bbox(
        values: tuple[float, float, float, float],
        inventory: RegionInventory,
    ) -> tuple[float, float, float, float]:
        if len(values) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise SelectionError(
                ReasonCode.TYPE_MISMATCH,
                "bbox_xyxy 必须是四个有限数",
            )
        result = tuple(float(value) for value in values)
        height, width = inventory.global_mask.shape
        if (
            result[0] < 0
            or result[1] < 0
            or result[2] > width
            or result[3] > height
            or result[2] <= result[0]
            or result[3] <= result[1]
        ):
            raise SelectionError(
                ReasonCode.TYPE_MISMATCH,
                "bbox_xyxy 越界或零面积",
            )
        return result

    @staticmethod
    def _validate_click(
        values: tuple[float, float],
        inventory: RegionInventory,
    ) -> tuple[int, int]:
        if len(values) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise SelectionError(
                ReasonCode.TYPE_MISMATCH,
                "click_xy 必须是两个有限数",
            )
        x_float, y_float = (float(value) for value in values)
        if not x_float.is_integer() or not y_float.is_integer():
            raise SelectionError(
                ReasonCode.TYPE_MISMATCH,
                "click_xy 必须指向整数像素",
            )
        x, y = int(x_float), int(y_float)
        height, width = inventory.global_mask.shape
        if x < 0 or y < 0 or x >= width or y >= height:
            raise SelectionError(
                ReasonCode.TYPE_MISMATCH,
                "click_xy 越界",
            )
        return x, y
