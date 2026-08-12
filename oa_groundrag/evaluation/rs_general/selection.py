"""Gate B 的 256-parent 固定集合选择与冻结读取。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    stable_hash,
)
from oa_groundrag.phase3.contracts import validate_canonical_record
from oa_groundrag.phase3.dataset import CanonicalRecordLocation
from oa_groundrag.phase3.errors import RSGeneralDescError

from .artifacts import AtomicArtifactDirectory
from .contracts import GATE_B_SELECTION_SCHEMA_VERSION
from .errors import ContractError, ReasonCode
from .gate_b_contracts import (
    GATE_B_PROTOCOL_ID,
    GATE_B_SAMPLE_COUNT,
    GATE_B_SEED,
    GATE_B_SELECTION_ALGORITHM,
    GATE_B_TASK_ORDER,
    GateBProtocolSource,
    build_frozen_protocol,
    load_gate_b_protocol,
    read_frozen_protocol,
    static_protocol_snapshot,
    validate_frozen_protocol,
)
from .preflight import BenchmarkAccess, open_benchmark_access
from .validation import VALIDATION_SELECTION_SCHEMA_VERSION


_SELECTION_FIELDS = {
    "schema_version",
    "protocol_id",
    "protocol_sha256",
    "selection_sha256",
    "benchmark_identity",
    "role",
    "algorithm",
    "seed",
    "sample_count",
    "parent_count",
    "task_order",
    "task_parent_capacities",
    "task_counts",
    "source_counts",
    "source_task_counts",
    "available_source_task_cells",
    "probed_shards",
    "monitoring_exclusion",
    "items",
}
_ITEM_FIELDS = {
    "ordinal",
    "record_id",
    "parent_id",
    "source",
    "task_family",
    "shard_path",
    "line_index",
}
_EXCLUSION_FIELDS = {
    "selection_schema",
    "selection_file_sha256",
    "selection_sha256",
    "parent_count",
    "parent_ids_sha256",
    "intersection_count",
}


@dataclass(frozen=True)
class GateBCandidate:
    record_id: str
    parent_id: str
    source: str
    task_family: str
    shard_path: str
    line_index: int

    @property
    def cell(self) -> tuple[str, str]:
        return self.task_family, self.source

    def location(self) -> CanonicalRecordLocation:
        return CanonicalRecordLocation(self.shard_path, self.line_index)


@dataclass(frozen=True)
class GateBSelectionContext:
    frozen_protocol: Mapping[str, Any]
    protocol_source: GateBProtocolSource
    selection: Mapping[str, Any]
    access: BenchmarkAccess
    historical_implementation_match: bool = True


def _fail(message: str, *, details: Mapping[str, Any] | None = None) -> None:
    raise ContractError(
        ReasonCode.GATE_B_SELECTION_INVALID,
        message,
        details=details,
    )


def _stable(*parts: object) -> str:
    return stable_hash(
        GATE_B_SEED,
        GATE_B_SELECTION_SCHEMA_VERSION,
        *parts,
    )


def _task_index(task: str) -> int:
    try:
        return GATE_B_TASK_ORDER.index(task)
    except ValueError as error:
        _fail(f"未知 Gate B task：{task}")
        raise AssertionError from error


def _candidate_key(candidate: GateBCandidate) -> tuple[str, str, str]:
    return (
        _stable(
            "parent",
            candidate.source,
            candidate.task_family,
            candidate.parent_id,
        ),
        _stable(
            "record",
            candidate.source,
            candidate.task_family,
            candidate.parent_id,
            candidate.record_id,
        ),
        candidate.record_id,
    )


def _source_key(task: str, source: str) -> tuple[str, str]:
    return _stable("source", task, source), source


def _scan_candidates(access: BenchmarkAccess) -> tuple[list[GateBCandidate], tuple[str, ...]]:
    layout = access.manifest.get("layout")
    if not isinstance(layout, dict):
        _fail("manifest layout 非法")
    role_map = layout.get("role_to_record_shards")
    if not isinstance(role_map, dict):
        _fail("manifest 缺少 role_to_record_shards")
    shard_values = role_map.get("external_val")
    if (
        not isinstance(shard_values, list)
        or not shard_values
        or not all(isinstance(value, str) and value for value in shard_values)
        or len(shard_values) != len(set(shard_values))
    ):
        _fail("external_val shard layout 非法")
    count_by_path = {
        str(shard["path"]): int(shard["record_count"])
        for shard in access.record_shards
    }
    output: list[GateBCandidate] = []
    seen_records: set[str] = set()
    parent_sources: dict[str, str] = {}
    for relative in shard_values:
        expected_count = count_by_path.get(relative)
        if expected_count is None:
            _fail("external_val shard 不在 manifest record_shards", details={"path": relative})
        path = access.verify_file(relative)
        try:
            rows = read_jsonl(path)
        except RSGeneralDescError as error:
            _fail("external_val shard 无法严格读取", details={"path": relative, "error": str(error)})
        if len(rows) != expected_count:
            _fail(
                "external_val shard 行数与 manifest 不一致",
                details={"path": relative, "expected": expected_count, "actual": len(rows)},
            )
        for line_index, row in enumerate(rows):
            validate_canonical_record(row)
            if row["logical_role"] != "external_val":
                continue
            task = str(row["task_family"])
            if task not in GATE_B_TASK_ORDER:
                _fail("external_val 含 Gate B 未注册 task", details={"task": task})
            record_id = str(row["record_id"])
            parent_id = str(row["parent_id"])
            source = str(row["source"])
            if record_id in seen_records:
                _fail("external_val record_id 重复", details={"record_id": record_id})
            seen_records.add(record_id)
            previous_source = parent_sources.setdefault(parent_id, source)
            if previous_source != source:
                _fail("同一 external_val parent 跨 source", details={"parent_id": parent_id})
            for media in row["media"]:
                access.assert_declared_identity(
                    str(media["path"]),
                    size_bytes=media["size_bytes"],
                    sha256=media["sha256"],
                )
            output.append(
                GateBCandidate(
                    record_id=record_id,
                    parent_id=parent_id,
                    source=source,
                    task_family=task,
                    shard_path=relative,
                    line_index=line_index,
                )
            )
    if not output:
        _fail("external_val 候选为空")
    return output, tuple(shard_values)


def _representatives(
    candidates: Sequence[GateBCandidate],
) -> dict[tuple[str, str], tuple[GateBCandidate, ...]]:
    grouped: dict[
        tuple[str, str],
        dict[str, list[GateBCandidate]],
    ] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        grouped[candidate.cell][candidate.parent_id].append(candidate)
    output: dict[tuple[str, str], tuple[GateBCandidate, ...]] = {}
    for cell, by_parent in grouped.items():
        values = [
            min(records, key=_candidate_key)
            for records in by_parent.values()
        ]
        output[cell] = tuple(sorted(values, key=_candidate_key))
    return output


def _task_quotas(
    by_cell: Mapping[tuple[str, str], Sequence[GateBCandidate]],
) -> tuple[dict[str, int], dict[str, int]]:
    parents_by_task: dict[str, set[str]] = defaultdict(set)
    for (task, _), values in by_cell.items():
        parents_by_task[task].update(value.parent_id for value in values)
    missing = [task for task in GATE_B_TASK_ORDER if not parents_by_task[task]]
    if missing:
        _fail("Gate B selection 缺少 task", details={"missing": missing})
    base, remainder = divmod(GATE_B_SAMPLE_COUNT, len(GATE_B_TASK_ORDER))
    quotas = {
        task: base + int(index < remainder)
        for index, task in enumerate(GATE_B_TASK_ORDER)
    }
    capacities = {
        task: len(parents_by_task[task]) for task in GATE_B_TASK_ORDER
    }
    deficit = 0
    for task in GATE_B_TASK_ORDER:
        if quotas[task] > capacities[task]:
            deficit += quotas[task] - capacities[task]
            quotas[task] = capacities[task]
    while deficit:
        progressed = False
        for task in GATE_B_TASK_ORDER:
            if deficit == 0:
                break
            if quotas[task] < capacities[task]:
                quotas[task] += 1
                deficit -= 1
                progressed = True
        if not progressed:
            _fail("external_val 无法提供 256 个 task-balanced parents")
    return quotas, capacities


def _feasible(
    *,
    by_cell: Mapping[tuple[str, str], Sequence[GateBCandidate]],
    used_parents: set[str],
    remaining_quotas: Mapping[str, int],
    mandatory_cells: set[tuple[str, str]],
) -> bool:
    mandatory_per_task = Counter(task for task, _ in mandatory_cells)
    slots: list[tuple[str, str, str | None, int]] = []
    for task, source in mandatory_cells:
        slots.append(("cell", task, source, 0))
    for task in GATE_B_TASK_ORDER:
        generic = remaining_quotas[task] - mandatory_per_task[task]
        if generic < 0:
            return False
        slots.extend(("task", task, None, index) for index in range(generic))
    if not slots:
        return True

    by_task_parent: dict[str, set[str]] = defaultdict(set)
    for (task, _), values in by_cell.items():
        by_task_parent[task].update(value.parent_id for value in values)

    candidate_parents: list[tuple[str, ...]] = []
    for kind, task, source, _ in slots:
        if kind == "cell":
            assert source is not None
            values = {
                value.parent_id for value in by_cell[(task, source)]
            }
        else:
            values = by_task_parent[task]
        ordered = tuple(
            sorted(
                values - used_parents,
                key=lambda parent: (
                    _stable("feasible", kind, task, source or "", parent),
                    parent,
                ),
            )
        )
        if not ordered:
            return False
        candidate_parents.append(ordered)

    order = sorted(
        range(len(slots)),
        key=lambda index: (
            len(candidate_parents[index]),
            0 if slots[index][0] == "cell" else 1,
            _task_index(slots[index][1]),
            _source_key(slots[index][1], slots[index][2] or ""),
            slots[index][3],
        ),
    )
    owner: dict[str, int] = {}

    def assign(slot_index: int, seen: set[str]) -> bool:
        for parent in candidate_parents[slot_index]:
            if parent in seen:
                continue
            seen.add(parent)
            previous = owner.get(parent)
            if previous is None or assign(previous, seen):
                owner[parent] = slot_index
                return True
        return False

    return all(assign(slot_index, set()) for slot_index in order)


def _select_candidates(
    candidates: Sequence[GateBCandidate],
    *,
    excluded_parents: set[str],
) -> tuple[
    list[GateBCandidate],
    dict[str, int],
    dict[str, int],
    tuple[tuple[str, str], ...],
]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.parent_id not in excluded_parents
    ]
    by_cell = _representatives(eligible)
    cells = tuple(
        sorted(
            by_cell,
            key=lambda cell: (
                _task_index(cell[0]),
                _source_key(cell[0], cell[1]),
            ),
        )
    )
    if not cells:
        _fail("排除 monitoring parents 后没有可用 source-task cell")
    quotas, capacities = _task_quotas(by_cell)
    remaining = dict(quotas)
    mandatory = set(cells)
    selected: list[GateBCandidate] = []
    used: set[str] = set()
    cell_counts: Counter[tuple[str, str]] = Counter()
    if not _feasible(
        by_cell=by_cell,
        used_parents=used,
        remaining_quotas=remaining,
        mandatory_cells=mandatory,
    ):
        _fail("固定 task 配额与全部 source-task cell 覆盖不可同时满足")

    mandatory_order = sorted(
        cells,
        key=lambda cell: (
            len(by_cell[cell]),
            _task_index(cell[0]),
            _source_key(cell[0], cell[1]),
        ),
    )
    for cell in mandatory_order:
        chosen = None
        for candidate in by_cell[cell]:
            if candidate.parent_id in used:
                continue
            next_remaining = dict(remaining)
            next_remaining[candidate.task_family] -= 1
            next_mandatory = mandatory - {cell}
            if _feasible(
                by_cell=by_cell,
                used_parents=used | {candidate.parent_id},
                remaining_quotas=next_remaining,
                mandatory_cells=next_mandatory,
            ):
                chosen = candidate
                remaining = next_remaining
                mandatory = next_mandatory
                break
        if chosen is None:
            _fail("无法在 parent 唯一约束下覆盖 source-task cell", details={"cell": list(cell)})
        selected.append(chosen)
        used.add(chosen.parent_id)
        cell_counts[cell] += 1

    while sum(remaining.values()):
        progressed = False
        for task in GATE_B_TASK_ORDER:
            if remaining[task] == 0:
                continue
            choices = [
                candidate
                for cell, values in by_cell.items()
                if cell[0] == task
                for candidate in values
                if candidate.parent_id not in used
            ]
            choices.sort(
                key=lambda candidate: (
                    cell_counts[candidate.cell],
                    _source_key(task, candidate.source),
                    _candidate_key(candidate),
                )
            )
            chosen = None
            for candidate in choices:
                next_remaining = dict(remaining)
                next_remaining[task] -= 1
                if _feasible(
                    by_cell=by_cell,
                    used_parents=used | {candidate.parent_id},
                    remaining_quotas=next_remaining,
                    mandatory_cells=set(),
                ):
                    chosen = candidate
                    remaining = next_remaining
                    break
            if chosen is None:
                _fail("water-fill 无法保持剩余 task quota 可行", details={"task": task})
            selected.append(chosen)
            used.add(chosen.parent_id)
            cell_counts[chosen.cell] += 1
            progressed = True
        if not progressed:
            _fail("Gate B selection water-fill 未取得进展")
    if len(selected) != GATE_B_SAMPLE_COUNT or len(used) != GATE_B_SAMPLE_COUNT:
        _fail("Gate B selection 未精确得到 256 个唯一 parents")
    selected.sort(
        key=lambda candidate: (
            _task_index(candidate.task_family),
            _source_key(candidate.task_family, candidate.source),
            _candidate_key(candidate),
        )
    )
    return selected, quotas, capacities, cells


def _selection_document(
    *,
    frozen_protocol: Mapping[str, Any],
    access: BenchmarkAccess,
    selected: Sequence[GateBCandidate],
    quotas: Mapping[str, int],
    capacities: Mapping[str, int],
    cells: Sequence[tuple[str, str]],
    probed_shards: Sequence[str],
    monitoring_file_sha256: str,
    monitoring_selection_sha256: str,
    monitoring_parents: Sequence[str],
) -> dict[str, Any]:
    task_counts = Counter(candidate.task_family for candidate in selected)
    source_counts = Counter(candidate.source for candidate in selected)
    cell_counts = Counter(candidate.cell for candidate in selected)
    items = [
        {
            "ordinal": ordinal,
            "record_id": candidate.record_id,
            "parent_id": candidate.parent_id,
            "source": candidate.source,
            "task_family": candidate.task_family,
            "shard_path": candidate.shard_path,
            "line_index": candidate.line_index,
        }
        for ordinal, candidate in enumerate(selected)
    ]
    body = {
        "schema_version": GATE_B_SELECTION_SCHEMA_VERSION,
        "protocol_id": GATE_B_PROTOCOL_ID,
        "protocol_sha256": frozen_protocol["protocol_sha256"],
        "benchmark_identity": access.identity.to_dict(),
        "role": "external_val",
        "algorithm": GATE_B_SELECTION_ALGORITHM,
        "seed": GATE_B_SEED,
        "sample_count": len(selected),
        "parent_count": len({candidate.parent_id for candidate in selected}),
        "task_order": list(GATE_B_TASK_ORDER),
        "task_parent_capacities": {
            task: capacities[task] for task in GATE_B_TASK_ORDER
        },
        "task_counts": {task: task_counts[task] for task in GATE_B_TASK_ORDER},
        "source_counts": dict(sorted(source_counts.items())),
        "source_task_counts": {
            source: {
                task: cell_counts[(task, source)]
                for task in GATE_B_TASK_ORDER
                if (task, source) in cell_counts
            }
            for source in sorted(source_counts)
        },
        "available_source_task_cells": [
            {"task_family": task, "source": source}
            for task, source in cells
        ],
        "probed_shards": list(probed_shards),
        "monitoring_exclusion": {
            "selection_schema": VALIDATION_SELECTION_SCHEMA_VERSION,
            "selection_file_sha256": monitoring_file_sha256,
            "selection_sha256": monitoring_selection_sha256,
            "parent_count": len(monitoring_parents),
            "parent_ids_sha256": sha256_text(
                canonical_json(sorted(monitoring_parents))
            ),
            "intersection_count": len(
                {candidate.parent_id for candidate in selected}
                & set(monitoring_parents)
            ),
        },
        "items": items,
    }
    if body["task_counts"] != {
        task: quotas[task] for task in GATE_B_TASK_ORDER
    }:
        _fail("selected task counts 与固定 quotas 不一致")
    return {
        **body,
        "selection_sha256": sha256_text(canonical_json(body)),
    }


def prepare_gate_b(
    protocol_path: Path | str,
    *,
    training_root: Path,
    output_root: Path,
) -> Path:
    with AtomicArtifactDirectory(Path(output_root)) as writer:
        frozen, source, access = build_frozen_protocol(
            protocol_path,
            training_root=training_root,
        )
        candidates, probed_shards = _scan_candidates(access)
        monitoring_path = Path(training_root) / "validation_selection.json"
        expected_monitoring_file_sha = frozen["training_run"][
            "monitoring_selection"
        ]["file_sha256"]
        if sha256_file(monitoring_path) != expected_monitoring_file_sha:
            _fail("monitoring selection 在 protocol 冻结后发生变化")
        try:
            monitoring = read_json(monitoring_path)
        except RSGeneralDescError as error:
            _fail("无法重新读取 monitoring selection", details={"error": str(error)})
        if not isinstance(monitoring, dict):
            _fail("monitoring selection 必须是对象")
        monitoring_parents = tuple(
            str(item["parent_id"]) for item in monitoring["items"]
        )
        monitoring_identity = frozen["training_run"]["monitoring_selection"]
        if sha256_text(canonical_json(sorted(monitoring_parents))) != (
            monitoring_identity["parent_ids_sha256"]
        ):
            _fail("monitoring parent 排除列表与 frozen protocol 不一致")
        selected, quotas, capacities, cells = _select_candidates(
            candidates,
            excluded_parents=set(monitoring_parents),
        )
        selection = _selection_document(
            frozen_protocol=frozen,
            access=access,
            selected=selected,
            quotas=quotas,
            capacities=capacities,
            cells=cells,
            probed_shards=probed_shards,
            monitoring_file_sha256=expected_monitoring_file_sha,
            monitoring_selection_sha256=monitoring_identity["selection_sha256"],
            monitoring_parents=monitoring_parents,
        )
        if selection["monitoring_exclusion"]["intersection_count"] != 0:
            _fail("Gate B selection 与 monitoring parents 相交")
        if sha256_file(monitoring_path) != expected_monitoring_file_sha:
            _fail("monitoring selection 在 Gate B prepare 期间发生变化")
        writer.write_json("gate_b_protocol.json", frozen)
        writer.write_json("gate_b_selection.json", selection)
        return writer.publish()


def _strict_selection(row: Mapping[str, Any]) -> None:
    if set(row) != _SELECTION_FIELDS:
        _fail(
            "selection 字段不匹配",
            details={
                "unknown": sorted(set(row) - _SELECTION_FIELDS),
                "missing": sorted(_SELECTION_FIELDS - set(row)),
            },
        )
    if (
        row.get("schema_version") != GATE_B_SELECTION_SCHEMA_VERSION
        or row.get("protocol_id") != GATE_B_PROTOCOL_ID
        or row.get("role") != "external_val"
        or row.get("algorithm") != GATE_B_SELECTION_ALGORITHM
        or row.get("seed") != GATE_B_SEED
        or row.get("sample_count") != GATE_B_SAMPLE_COUNT
        or row.get("parent_count") != GATE_B_SAMPLE_COUNT
        or row.get("task_order") != list(GATE_B_TASK_ORDER)
    ):
        _fail("selection 固定身份/算法/count 不匹配")
    expected_sha = row.get("selection_sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha
        )
    ):
        _fail("selection_sha256 非法")
    body = {key: value for key, value in row.items() if key != "selection_sha256"}
    actual_sha = sha256_text(canonical_json(body))
    if actual_sha != expected_sha:
        _fail("selection canonical SHA-256 不匹配")
    items = row.get("items")
    if not isinstance(items, list) or len(items) != GATE_B_SAMPLE_COUNT:
        _fail("selection items 必须恰好为 256")
    records: set[str] = set()
    parents: set[str] = set()
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    previous_key: tuple[Any, ...] | None = None
    for ordinal, value in enumerate(items):
        if not isinstance(value, dict) or set(value) != _ITEM_FIELDS:
            _fail("selection item 字段不匹配", details={"ordinal": ordinal})
        if value.get("ordinal") != ordinal:
            _fail("selection ordinal 不连续", details={"ordinal": ordinal})
        for name in ("record_id", "parent_id", "source", "task_family", "shard_path"):
            if not isinstance(value.get(name), str) or not value[name]:
                _fail("selection item 字符串字段非法", details={"ordinal": ordinal, "field": name})
        line_index = value.get("line_index")
        if isinstance(line_index, bool) or not isinstance(line_index, int) or line_index < 0:
            _fail("selection line_index 非法", details={"ordinal": ordinal})
        if value["task_family"] not in GATE_B_TASK_ORDER:
            _fail("selection item task 非法")
        if value["record_id"] in records or value["parent_id"] in parents:
            _fail("selection record/parent 重复")
        records.add(value["record_id"])
        parents.add(value["parent_id"])
        task_counts[value["task_family"]] += 1
        source_counts[value["source"]] += 1
        cell_counts[(value["task_family"], value["source"])] += 1
        key = (
            _task_index(value["task_family"]),
            _source_key(value["task_family"], value["source"]),
            _stable("parent", value["source"], value["task_family"], value["parent_id"]),
            _stable("record", value["source"], value["task_family"], value["parent_id"], value["record_id"]),
            value["record_id"],
        )
        if previous_key is not None and key < previous_key:
            _fail("selection items 未按 frozen generation order 排序")
        previous_key = key
    if dict(task_counts) != row.get("task_counts"):
        _fail("selection task_counts 与 items 不一致")
    if dict(sorted(source_counts.items())) != row.get("source_counts"):
        _fail("selection source_counts 与 items 不一致")
    expected_source_task_counts = {
        source: {
            task: cell_counts[(task, source)]
            for task in GATE_B_TASK_ORDER
            if cell_counts[(task, source)]
        }
        for source in sorted(source_counts)
    }
    if row.get("source_task_counts") != expected_source_task_counts:
        _fail("selection source_task_counts 与 items 不一致")
    capacities = row.get("task_parent_capacities")
    if (
        not isinstance(capacities, dict)
        or set(capacities) != set(GATE_B_TASK_ORDER)
        or any(
            isinstance(capacities[task], bool)
            or not isinstance(capacities[task], int)
            or capacities[task] < task_counts[task]
            for task in GATE_B_TASK_ORDER
        )
    ):
        _fail("selection task_parent_capacities 非法")
    probed_shards = row.get("probed_shards")
    if (
        not isinstance(probed_shards, list)
        or not probed_shards
        or not all(isinstance(path, str) and path for path in probed_shards)
        or len(probed_shards) != len(set(probed_shards))
        or not {item["shard_path"] for item in items}.issubset(probed_shards)
    ):
        _fail("selection probed_shards 非法")
    cells = row.get("available_source_task_cells")
    if not isinstance(cells, list) or not cells:
        _fail("selection available_source_task_cells 非法")
    available = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {"task_family", "source"}:
            _fail("selection source-task cell 字段非法")
        available.add((cell["task_family"], cell["source"]))
    expected_cells = sorted(
        cell_counts,
        key=lambda cell: (
            _task_index(cell[0]),
            _source_key(cell[0], cell[1]),
        ),
    )
    actual_cells = [
        (cell["task_family"], cell["source"])
        for cell in cells
    ]
    if (
        len(available) != len(cells)
        or actual_cells != expected_cells
        or available != set(cell_counts)
        or any(value <= 0 for value in cell_counts.values())
    ):
        _fail("selection 未覆盖全部发布的 source-task cells")
    exclusion = row.get("monitoring_exclusion")
    if not isinstance(exclusion, dict) or set(exclusion) != _EXCLUSION_FIELDS:
        _fail("selection monitoring_exclusion 字段非法")
    if (
        exclusion.get("selection_schema") != VALIDATION_SELECTION_SCHEMA_VERSION
        or exclusion.get("parent_count") != 128
        or exclusion.get("intersection_count") != 0
        or any(
            not isinstance(exclusion.get(name), str)
            or len(exclusion[name]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in exclusion[name]
            )
            for name in (
                "selection_file_sha256",
                "selection_sha256",
                "parent_ids_sha256",
            )
        )
    ):
        _fail("selection monitoring exclusion 结论非法")


def _selection_paths(selection_path: Path | str) -> tuple[Path, Path]:
    """解析 frozen selection 及其 sibling protocol，并拒绝链接替换。"""

    selection_path = Path(selection_path)
    linked = first_symlink_component(selection_path)
    if linked is not None:
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            "frozen selection 路径含链接组件",
            details={"path": str(linked)},
        )
    if (
        not selection_path.is_file()
        or selection_path.is_symlink()
        or selection_path.stat().st_nlink != 1
    ):
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            "frozen selection 必须是普通单链接文件",
            details={"path": str(selection_path)},
        )
    return selection_path, selection_path.parent / "gate_b_protocol.json"


def _load_selection_context(
    *,
    frozen: Mapping[str, Any],
    source: GateBProtocolSource,
    selection_path: Path,
    historical_implementation_match: bool,
) -> GateBSelectionContext:
    """共享 selection/Benchmark 核验；不在 Stage 5 复制 Gate B 数据合同。"""

    try:
        selection = read_json(selection_path)
    except RSGeneralDescError as error:
        _fail("无法严格读取 frozen selection", details={"error": str(error)})
    if not isinstance(selection, dict):
        _fail("frozen selection 必须是对象")
    _strict_selection(selection)
    if selection["protocol_sha256"] != frozen["protocol_sha256"]:
        _fail("selection protocol SHA 与 sibling frozen protocol 不一致")
    access = open_benchmark_access(source.base_config)
    if selection["benchmark_identity"] != access.identity.to_dict():
        _fail("selection Benchmark identity 与实际 preflight 不一致")
    layout = access.manifest.get("layout")
    role_map = (
        layout.get("role_to_record_shards")
        if isinstance(layout, dict)
        else None
    )
    actual_shards = (
        role_map.get("external_val") if isinstance(role_map, dict) else None
    )
    if selection["probed_shards"] != actual_shards:
        _fail("selection probed_shards 与 manifest external_val layout 不一致")
    frozen_monitoring = frozen["training_run"]["monitoring_selection"]
    exclusion = selection["monitoring_exclusion"]
    if (
        exclusion["selection_file_sha256"] != frozen_monitoring["file_sha256"]
        or exclusion["selection_sha256"] != frozen_monitoring["selection_sha256"]
        or exclusion["parent_ids_sha256"] != frozen_monitoring["parent_ids_sha256"]
    ):
        _fail("selection monitoring exclusion identity 与 protocol 不一致")
    return GateBSelectionContext(
        frozen_protocol=frozen,
        protocol_source=source,
        selection=selection,
        access=access,
        historical_implementation_match=historical_implementation_match,
    )


def load_gate_b_selection(
    protocol_path: Path | str,
    selection_path: Path | str,
) -> GateBSelectionContext:
    """正式 Gate B 读取：要求当前实现与冻结时字节身份完全一致。"""

    selection_path, frozen_path = _selection_paths(selection_path)
    frozen, source = validate_frozen_protocol(protocol_path, frozen_path)
    return _load_selection_context(
        frozen=frozen,
        source=source,
        selection_path=selection_path,
        historical_implementation_match=True,
    )


def load_gate_b_selection_for_stage5_retention(
    protocol_path: Path | str,
    selection_path: Path | str,
) -> GateBSelectionContext:
    """Stage 5 只消费冻结集合，不把新实现伪装成历史 Gate B 重放。"""

    selection_path, frozen_path = _selection_paths(selection_path)
    frozen = read_frozen_protocol(frozen_path)
    source = load_gate_b_protocol(protocol_path)
    current = static_protocol_snapshot(source)
    frozen_static = dict(frozen["static_protocol"])
    historical_implementation = frozen_static.pop("implementation_files", None)
    current_implementation = current.pop("implementation_files", None)
    if current != frozen_static:
        _fail("Stage 5 retention 的 Gate B 数据/模型/评价静态合同发生变化")
    implementation_match = current_implementation == historical_implementation
    return _load_selection_context(
        frozen=frozen,
        source=source,
        selection_path=selection_path,
        historical_implementation_match=implementation_match,
    )


def selection_locations(
    selection: Mapping[str, Any],
) -> tuple[CanonicalRecordLocation, ...]:
    return tuple(
        CanonicalRecordLocation(
            str(item["shard_path"]),
            int(item["line_index"]),
        )
        for item in selection["items"]
    )
