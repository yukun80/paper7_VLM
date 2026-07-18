"""Exact Bridge-region to segmentation-instruction target resolution."""

from __future__ import annotations

from typing import Any


END_TO_END_TARGET_PROTOCOL = "qpsalm_end_to_end_region_target_v3_source_bound"


class EndToEndTargetResolver:
    """Map one Bridge region to the exact segmentation instruction that names it."""

    PROTOCOL = END_TO_END_TARGET_PROTOCOL
    GLOBAL_FAMILY_PRIORITY = {
        "global_landslide_segmentation": 0,
        "negative_aware_segmentation": 1,
        "multisource_evidence_segmentation": 2,
    }

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        ranked_global: dict[str, tuple[int, int]] = {}
        self.referring: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            family = str(row.get("task_family") or "")
            if family in self.GLOBAL_FAMILY_PRIORITY:
                priority = self.GLOBAL_FAMILY_PRIORITY[family]
                if parent not in ranked_global or priority < ranked_global[parent][0]:
                    ranked_global[parent] = (priority, index)
            target_id = row.get("parent_referring_target_sample_id")
            if target_id:
                key = (parent, str(target_id))
                previous = self.referring.setdefault(key, index)
                if previous != index:
                    previous_row = rows[previous]
                    if str(previous_row.get("sample_id")) != str(row.get("sample_id")):
                        raise ValueError(f"重复 referring instruction identity: {key}")
        self.global_indices = {
            parent: index for parent, (_priority, index) in ranked_global.items()
        }

    @staticmethod
    def _empty_target(row: dict[str, Any]) -> bool:
        mask = row.get("mask") or {}
        if bool(mask.get("empty_mask")):
            return True
        positive = mask.get("positive_pixels")
        return positive is not None and int(positive) == 0

    @staticmethod
    def _aliases(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(
            (
                dict(value)
                for value in (metadata.get("source_region_aliases") or [])
                if isinstance(value, dict) and value.get("sample_id")
            ),
            key=lambda value: str(value["sample_id"]),
        )

    def _global(self, parent: str) -> tuple[int, str, str | None]:
        index = self.global_indices.get(parent)
        if index is None:
            raise KeyError(f"segmentation split 缺少 global instruction: parent={parent}")
        return index, "global_instruction", None

    def _referring(
        self,
        parent: str,
        aliases: list[dict[str, Any]],
        *,
        expected_family: str,
    ) -> tuple[int, str, str | None]:
        for alias in aliases:
            target_id = str(alias["sample_id"])
            index = self.referring.get((parent, target_id))
            if index is None:
                continue
            family = str(self.rows[index].get("task_family") or "")
            if family == expected_family:
                return index, "referring_alias", target_id
        raise KeyError(
            "segmentation split 缺少精确 referring instruction: "
            f"parent={parent} family={expected_family} "
            f"aliases={[value['sample_id'] for value in aliases[:8]]}"
        )

    def resolve(self, metadata: dict[str, Any]) -> dict[str, Any]:
        parent = str(metadata.get("parent_sample_id") or "")
        source = str(metadata.get("region_source") or "unknown")
        aliases = self._aliases(metadata)
        alias_id: str | None
        if source == "gt_global_mask":
            index, kind, alias_id = self._global(parent)
        elif source in {"gt_referring_mask", "pseudo_instance_component"}:
            if not aliases:
                raise KeyError(
                    f"{source} 没有可识别的 referring alias: "
                    f"parent={parent} region={metadata.get('region_id')}"
                )
            index, kind, alias_id = self._referring(
                parent,
                aliases,
                expected_family="referring_landslide_segmentation",
            )
        elif source == "no_target":
            if aliases:
                index, kind, alias_id = self._referring(
                    parent, aliases, expected_family="no_target_segmentation"
                )
            else:
                index, kind, alias_id = self._global(parent)
                if not self._empty_target(self.rows[index]):
                    raise KeyError(
                        "no_target region 既无 no-target alias，parent global target 也非空: "
                        f"parent={parent}"
                    )
                kind = "empty_global_instruction"
        else:
            raise KeyError(
                f"region_source={source!r} 没有端到端 segmentation target protocol"
            )
        row = self.rows[index]
        return {
            "protocol": self.PROTOCOL,
            "bridge_sample_id": str(metadata.get("sample_id") or ""),
            "dataset_index": int(index),
            "mapping_kind": kind,
            "alias_sample_id": alias_id,
            "segmentation_sample_id": str(row.get("sample_id")),
            "segmentation_task_family": str(row.get("task_family")),
            "parent_sample_id": parent,
            "bridge_region_id": str(metadata.get("region_id") or "unknown"),
            "bridge_region_source": source,
        }
