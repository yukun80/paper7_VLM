"""三个外部 adapter 共用的严格解析与确定性选择。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from ..io import normalized_text, stable_hash
from ..config import BuildConfig
from ..contracts import AdapterResult, SourceExample


class SourceAdapter:
    source_name: str

    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def scan(
        self,
        *,
        deep: bool = False,
        for_build: bool = True,
    ) -> AdapterResult:
        raise NotImplementedError


def text_has_pattern(text: str, patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def safe_text(
    text: object,
    config: BuildConfig,
    *,
    reasoning: bool = False,
) -> tuple[str, str | None]:
    normalized = normalized_text(text, location="text")
    patterns = list(config.text_policy.forbidden_patterns)
    if reasoning:
        patterns.extend(config.text_policy.reasoning_forbidden_patterns)
    return normalized, text_has_pattern(normalized, patterns)


def deduplicate_responses(values: Iterable[str]) -> tuple[tuple[str, ...], int]:
    ordered: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for value in values:
        normalized = " ".join(value.split())
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered), duplicates


def select_examples(
    examples: Sequence[SourceExample],
    *,
    config: BuildConfig,
) -> list[SourceExample]:
    """按 task cap 与稳定 hash 选择；full 不设 cap 时保留全部。"""

    grouped: dict[str, list[SourceExample]] = defaultdict(list)
    for example in examples:
        grouped[example.task_family.value].append(example)
    selected: list[SourceExample] = []
    for task in sorted(grouped):
        rows = grouped[task]
        subtask_rows: dict[str, list[SourceExample]] = defaultdict(list)
        for row in rows:
            subtask = ""
            for key in (
                "source_collection",
                "qa_type",
                "source_task",
                "task_dataset",
                "mask_status",
            ):
                value = row.deterministic_facts.get(key)
                if isinstance(value, str) and value:
                    subtask = value
                    break
            subtask_rows[subtask].append(row)
        configured_subtasks = {
            key.rsplit(":", 1)[-1]: value
            for key, value in config.limits.per_task.items()
            if key.startswith(f"{examples[0].source}:{task}:")
        }
        if configured_subtasks:
            task_selected: list[SourceExample] = []
            for subtask, cap in sorted(configured_subtasks.items()):
                candidates = sorted(
                    subtask_rows.get(subtask, []),
                    key=lambda row: (
                        stable_hash(
                            config.seed,
                            row.source,
                            row.task_family.value,
                            subtask,
                            row.source_record_id,
                        ),
                        row.source_record_id,
                    ),
                )
                task_selected.extend(candidates[:cap])
            selected.extend(task_selected)
            continue
        rows = sorted(
            rows,
            key=lambda row: (
                stable_hash(
                    config.seed,
                    row.source,
                    row.task_family.value,
                    row.source_record_id,
                ),
                row.source_record_id,
            ),
        )
        cap = config.limits.per_task.get(f"{examples[0].source}:{task}")
        if cap is None:
            cap = config.limits.per_task.get(task)
        selected.extend(rows if cap is None else rows[:cap])
    selected.sort(
        key=lambda row: (
            row.source,
            row.parent_key,
            row.task_family.value,
            row.source_record_id,
        )
    )
    return selected


def summarize_result(result: AdapterResult) -> None:
    result.audit["candidate_examples"] = len(result.examples)
    result.audit["skip_count"] = len(result.skips)
    result.audit["task_counts"] = dict(
        sorted(Counter(row.task_family.value for row in result.examples).items())
    )
    result.audit["parent_count"] = len({row.parent_key for row in result.examples})
    result.audit["skip_reason_counts"] = dict(
        sorted(Counter(row.reason_code.value for row in result.skips).items())
    )
