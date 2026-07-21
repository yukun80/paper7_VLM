"""Frozen MMRS/RSGPT description-source subset builder for P1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sami_gsd.contracts.config import BenchmarkAuditConfig, LanguageComponentConfig, SourceConfig
from sami_gsd.contracts.language import DescriptionSourceRecord, LanguageAnswer, LanguageImageRef
from sami_gsd.data.adapters.formats import read_image_header
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


LANGUAGE_SUBSET_VERSION = "sami_description_subset_v3_provenance_bound"


class LanguageSubsetError(ValueError):
    """Raised when a selected component violates the frozen subset protocol."""


_MMRS_COMPONENTS = (
    ("rsicd", Path("json/caption/caption_rsicd.json")),
    ("ucm", Path("json/caption/caption_ucm.json")),
    ("sydney", Path("json/caption/caption_syndney.json")),
    ("nwpu", Path("json/caption/caption_nwpu.json")),
    ("rsitmd", Path("json/caption/caption_rsitmd.json")),
)


def _source(config: BenchmarkAuditConfig, key: str) -> SourceConfig:
    """Return one exact configured source row."""

    return next(source for source in config.sources if source.source_key == key)


def _component_policy(source: SourceConfig, component: str) -> LanguageComponentConfig:
    """Return the exact configured scientific component policy."""

    matches = [policy for policy in source.language_components if policy.component == component]
    if len(matches) != 1:
        raise LanguageSubsetError(f"component policy is not unique: {source.source_key}:{component}")
    return matches[0]


def _load_mapping(path: Path) -> Any:
    """Decode strict finite JSON from one selected index."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _image_ref(path: Path, *, logical_path: str) -> LanguageImageRef:
    """Bind a selected image by bytes and a signature-confirmed grid."""

    header = read_image_header(path)
    return LanguageImageRef(
        logical_path=logical_path,
        sha256=sha256_file(path),
        native_hw=(header.height, header.width),
    )


def _mmrs_image_relative(value: str) -> Path:
    """Resolve only the observed portable ``data/...`` source prefix."""

    path = Path(value)
    if len(path.parts) < 3 or path.parts[0] != "data":
        raise LanguageSubsetError("MMRS image path is outside the frozen data/... layout")
    return Path(*path.parts[1:])


def _gpt_answers(row: dict[str, Any]) -> tuple[str, ...]:
    """Extract non-empty assistant answers without accepting other task fields."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise LanguageSubsetError("MMRS row is missing conversations")
    answers = tuple(
        message["value"].strip()
        for message in conversations
        if isinstance(message, dict)
        and message.get("from") == "gpt"
        and isinstance(message.get("value"), str)
        and message["value"].strip()
    )
    if not answers:
        raise LanguageSubsetError("MMRS selected row has no assistant answer")
    return answers


def _make_answers(
    texts: tuple[str, ...],
    *,
    record_id: str,
    origin: str,
    logical_index: str,
    index_sha256: str,
) -> tuple[LanguageAnswer, ...]:
    """Build stable answer IDs with exact index provenance."""

    return tuple(
        LanguageAnswer(
            answer_id=f"answer-{sha256_bytes(canonical_json_bytes({'record': record_id, 'index': index}))[:20]}",
            text=text,
            annotation_origin=origin,
            index_logical_path=logical_index,
            index_sha256=index_sha256,
        )
        for index, text in enumerate(texts)
    )


def _dior_region_pairs(conversations: Any) -> tuple[tuple[int, tuple[float, float, float, float], str], ...]:
    """Extract only forward box-to-short-phrase pairs from alternating DIOR dialogs."""

    if not isinstance(conversations, list) or len(conversations) < 2:
        raise LanguageSubsetError("DIOR-RSVG row lacks box/phrase conversations")
    pairs: list[tuple[int, tuple[float, float, float, float], str]] = []
    for message_index in range(0, len(conversations) - 1, 2):
        prompt_message = conversations[message_index]
        answer_message = conversations[message_index + 1]
        if not isinstance(prompt_message, dict) or not isinstance(answer_message, dict):
            raise LanguageSubsetError("DIOR-RSVG conversation messages must be mappings")
        prompt = prompt_message.get("value")
        phrase = answer_message.get("value")
        if (
            prompt_message.get("from") != "human"
            or answer_message.get("from") != "gpt"
            or not isinstance(prompt, str)
            or not isinstance(phrase, str)
        ):
            raise LanguageSubsetError("DIOR-RSVG conversations must alternate human and gpt text")
        match = re.search(r"(\[[^\[\]]+\])\s*$", prompt)
        if match is None:
            # 本地索引交替包含 phrase->box 的逆向定位任务；P1 只选择 box->phrase，
            # 因此没有终止 box 的 pair 被显式跳过，而不是误当作 caption。
            continue
        try:
            box_payload = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise LanguageSubsetError("DIOR-RSVG terminal box is not valid JSON") from error
        if not isinstance(box_payload, list) or len(box_payload) != 4:
            raise LanguageSubsetError("DIOR-RSVG normalized box must have four values")
        if not phrase.strip():
            raise LanguageSubsetError("DIOR-RSVG short phrase is empty")
        pairs.append((message_index // 2, tuple(float(value) for value in box_payload), phrase.strip()))
    return tuple(pairs)


def _mmrs_records(
    source: SourceConfig,
    *,
    source_root: Path,
    limit_per_component: int,
) -> list[DescriptionSourceRecord]:
    """Build the five caption components plus DIOR short phrases only."""

    records: list[DescriptionSourceRecord] = []
    for component, index_relative in _MMRS_COMPONENTS:
        policy = _component_policy(source, component)
        payload = _load_mapping(source_root / index_relative)
        if not isinstance(payload, list):
            raise LanguageSubsetError(f"MMRS selected index is not an array: {index_relative}")
        index_hash = sha256_file(source_root / index_relative)
        logical_index = f"{source.provenance.source_root}/{index_relative.as_posix()}"
        for row_number, row in enumerate(payload[:limit_per_component]):
            if not isinstance(row, dict) or not isinstance(row.get("image"), str):
                raise LanguageSubsetError(f"MMRS selected row lacks image: {index_relative}:{row_number}")
            image_relative = _mmrs_image_relative(row["image"])
            logical_image = f"{source.provenance.source_root}/{image_relative.as_posix()}"
            record_id = f"mmrs/{component}/{image_relative.as_posix()}"
            texts = _gpt_answers(row)
            records.append(
                DescriptionSourceRecord(
                    schema_version="sami_description_source_v3_provenance_bound",
                    record_id=record_id,
                    source_key="mmrs_1m",
                    component=component,
                    component_key=policy.component_key,
                    source_group_id=f"mmrs/image/{image_relative.as_posix()}",
                    role="global_caption",
                    split_policy="train_candidate",
                    image=_image_ref(source_root / image_relative, logical_path=logical_image),
                    answers=_make_answers(
                        texts,
                        record_id=record_id,
                        origin="source_caption",
                        logical_index=logical_index,
                        index_sha256=index_hash,
                    ),
                    normalized_box_xyxy=None,
                    is_train_candidate=True,
                )
            )

    index_relative = Path("json/RSVG/rsvg_trainval.json")
    policy = _component_policy(source, "dior_rsvg")
    payload = _load_mapping(source_root / index_relative)
    if not isinstance(payload, list):
        raise LanguageSubsetError("DIOR-RSVG selected index is not an array")
    index_hash = sha256_file(source_root / index_relative)
    logical_index = f"{source.provenance.source_root}/{index_relative.as_posix()}"
    dior_count = 0
    for row_number, row in enumerate(payload):
        if dior_count >= limit_per_component:
            break
        if not isinstance(row, dict) or not isinstance(row.get("image"), str):
            raise LanguageSubsetError(f"DIOR-RSVG row lacks image: {row_number}")
        image_relative = _mmrs_image_relative(row["image"])
        for pair_number, box, phrase in _dior_region_pairs(row.get("conversations")):
            if dior_count >= limit_per_component:
                break
            record_id = f"mmrs/dior_rsvg/{image_relative.as_posix()}/{row_number}/{pair_number}"
            records.append(
                DescriptionSourceRecord(
                    schema_version="sami_description_source_v3_provenance_bound",
                    record_id=record_id,
                    source_key="mmrs_1m",
                    component="dior_rsvg",
                    component_key=policy.component_key,
                    source_group_id=f"mmrs/image/{image_relative.as_posix()}",
                    role="region_short_phrase",
                    split_policy="train_candidate",
                    image=_image_ref(
                        source_root / image_relative,
                        logical_path=f"{source.provenance.source_root}/{image_relative.as_posix()}",
                    ),
                    answers=_make_answers(
                        (phrase,),
                        record_id=record_id,
                        origin="source_expression",
                        logical_index=logical_index,
                        index_sha256=index_hash,
                    ),
                    normalized_box_xyxy=box,
                    is_train_candidate=True,
                )
            )
            dior_count += 1
    return records


def _rsgpt_records(
    source: SourceConfig,
    *,
    source_root: Path,
    limit_per_component: int,
) -> list[DescriptionSourceRecord]:
    """Build RSICap train candidates and permanent-test RSIEval records."""

    records: list[DescriptionSourceRecord] = []
    definitions = (
        ("rsicap", "RSICap", Path("dataset/RSICap/captions.json"), "train_candidate"),
        ("rsieval", "RSIEval", Path("dataset/RSIEval/annotations.json"), "permanent_test_only"),
    )
    for component, directory, index_relative, split_policy in definitions:
        policy = _component_policy(source, component)
        if policy.split_policy != split_policy:
            raise LanguageSubsetError(f"component split policy drift: {policy.component_key}")
        payload = _load_mapping(source_root / index_relative)
        rows = payload.get("annotations") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise LanguageSubsetError(f"{directory} annotations must be an array")
        index_hash = sha256_file(source_root / index_relative)
        logical_index = f"{source.provenance.source_root}/{index_relative.as_posix()}"
        for row_number, row in enumerate(rows[:limit_per_component]):
            if not isinstance(row, dict) or not isinstance(row.get("filename"), str):
                raise LanguageSubsetError(f"{directory} row lacks filename: {row_number}")
            text_value = row.get("text_output", row.get("caption"))
            if isinstance(text_value, str):
                texts = (text_value.strip(),)
            elif isinstance(text_value, list):
                texts = tuple(str(value).strip() for value in text_value if str(value).strip())
            else:
                raise LanguageSubsetError(f"{directory} row lacks caption text: {row_number}")
            image_relative = Path("dataset") / directory / "images" / row["filename"]
            record_id = f"rsgpt/{component}/{row['filename']}"
            records.append(
                DescriptionSourceRecord(
                    schema_version="sami_description_source_v3_provenance_bound",
                    record_id=record_id,
                    source_key="rsgpt",
                    component=component,
                    component_key=policy.component_key,
                    source_group_id=f"rsgpt/{directory}/{row['filename']}",
                    role="global_caption",
                    split_policy=split_policy,
                    image=_image_ref(
                        source_root / image_relative,
                        logical_path=f"{source.provenance.source_root}/{image_relative.as_posix()}",
                    ),
                    answers=_make_answers(
                        texts,
                        record_id=record_id,
                        origin="source_caption",
                        logical_index=logical_index,
                        index_sha256=index_hash,
                    ),
                    normalized_box_xyxy=None,
                    is_train_candidate=split_policy == "train_candidate",
                )
            )
    return records


def build_description_subset(
    config: BenchmarkAuditConfig,
    *,
    datasets_root: Path,
    limit_per_component: int,
) -> dict[str, Any]:
    """Build the exact selected scientific subset without reading excluded indexes."""

    if type(limit_per_component) is not int or limit_per_component <= 0:
        raise LanguageSubsetError("limit_per_component must be a positive integer")
    mmrs = _source(config, "mmrs_1m")
    rsgpt = _source(config, "rsgpt")
    records = _mmrs_records(
        mmrs,
        source_root=datasets_root / mmrs.provenance.source_root.removeprefix("datasets/"),
        limit_per_component=limit_per_component,
    ) + _rsgpt_records(
        rsgpt,
        source_root=datasets_root / rsgpt.provenance.source_root.removeprefix("datasets/"),
        limit_per_component=limit_per_component,
    )
    ordered = tuple(sorted(records, key=lambda item: item.record_id))
    if len({record.record_id for record in ordered}) != len(ordered):
        raise LanguageSubsetError("description subset record IDs are not unique")
    payload_records = [record.model_dump(mode="json") for record in ordered]
    report: dict[str, Any] = {
        "schema_version": "sami_description_subset_report_v3_provenance_bound",
        "builder_version": LANGUAGE_SUBSET_VERSION,
        "record_count": len(ordered),
        "train_candidate_count": sum(record.is_train_candidate for record in ordered),
        "permanent_test_only_count": sum(record.split_policy == "permanent_test_only" for record in ordered),
        "components": {
            component: sum(record.component == component for record in ordered)
            for component in ("rsicd", "ucm", "sydney", "nwpu", "rsitmd", "dior_rsvg", "rsicap", "rsieval")
        },
        "component_provenance": {
            policy.component_key: policy.provenance.model_dump(mode="json")
            for source in (mmrs, rsgpt)
            for policy in source.language_components
        },
        "excluded_inputs_read": [],
        "records": payload_records,
        "errors": [],
        "warnings": [],
    }
    report["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


__all__ = ["LANGUAGE_SUBSET_VERSION", "LanguageSubsetError", "build_description_subset"]
