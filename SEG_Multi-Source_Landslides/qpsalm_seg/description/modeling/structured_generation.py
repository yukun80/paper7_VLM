#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema-constrained raw generation for grounded region descriptions.

The decoder owns JSON syntax and schema keys.  Qwen still selects every enum
value and every free-text token from live logits.  The final object is parsed
only for assertion; unconstrained text is never repaired or mapped into fields,
and no value is filled from ground truth after generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Sequence

import torch

from ..protocols.output import (
    canonical_description_json,
    parse_description_output,
)
from ..protocols.versions import STRUCTURED_GENERATION_PROTOCOL


AdvanceTokens = Callable[[int | Sequence[int]], torch.Tensor]


@dataclass(frozen=True)
class DescriptionGenerationResult:
    """One raw decoder result plus an artifact-safe execution audit."""

    text: str
    audit: dict[str, Any]


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    values = tuple(int(value) for value in encoded)
    if not values:
        raise RuntimeError(f"structured decoder 无法编码固定片段: {text!r}")
    return values


def _decode_token(tokenizer: Any, token_id: int) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode([int(token_id)], **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode([int(token_id)], **kwargs))


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(token_ids), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(token_ids), **kwargs))


class _ConstrainedCursor:
    """Advance one cached causal decoder while accounting for every token."""

    def __init__(
        self,
        tokenizer: Any,
        initial_logits: torch.Tensor,
        advance_tokens: AdvanceTokens,
        *,
        max_new_tokens: int,
    ) -> None:
        if initial_logits.ndim != 1:
            raise ValueError("structured decoder initial logits 必须是一维 vocab vector")
        self.tokenizer = tokenizer
        self.logits = initial_logits
        self.advance_tokens = advance_tokens
        self.max_new_tokens = int(max_new_tokens)
        self.token_ids: list[int] = []
        self.forced_tokens = 0
        self.model_selected_tokens = 0
        self.advance_calls = 0

    def feed(self, token_id: int, *, forced: bool) -> None:
        self.feed_many((int(token_id),), forced=forced)

    def feed_many(self, token_ids: Sequence[int], *, forced: bool) -> None:
        values = tuple(int(value) for value in token_ids)
        if not values:
            raise ValueError("structured decoder advance token block 不能为空")
        if len(self.token_ids) + len(values) > self.max_new_tokens:
            raise RuntimeError(
                "schema-constrained JSON 超出 max_new_tokens；"
                "拒绝发布被截断的 raw JSON"
            )
        for token_id in values:
            if token_id < 0 or token_id >= int(self.logits.numel()):
                raise RuntimeError(
                    f"structured decoder token id 越界: {token_id}"
                )
        self.token_ids.extend(values)
        if forced:
            self.forced_tokens += len(values)
        else:
            self.model_selected_tokens += len(values)
        argument: int | Sequence[int] = (
            values[0] if len(values) == 1 else values
        )
        self.logits = self.advance_tokens(argument)
        self.advance_calls += 1
        if self.logits.ndim != 1:
            raise RuntimeError("structured decoder advance logits 必须是一维 vocab vector")

    def force(self, text: str) -> None:
        ids = _token_ids(self.tokenizer, text)
        decoded = "".join(_decode_token(self.tokenizer, value) for value in ids)
        if decoded != text:
            raise RuntimeError(
                "tokenizer 无法无损编码 schema 固定片段: "
                f"expected={text!r} decoded={decoded!r}"
            )
        self.feed_many(ids, forced=True)

    def decoded_stream(self) -> str:
        return _decode_tokens(self.tokenizer, self.token_ids)

    def choose_enum(self, values: Sequence[str]) -> str:
        candidates = {
            str(value): _token_ids(self.tokenizer, str(value))
            for value in values
        }
        if not candidates:
            raise ValueError("structured decoder enum candidates 不能为空")
        prefix: tuple[int, ...] = ()
        active = dict(candidates)
        while active:
            terminals = [
                name for name, ids in active.items() if len(ids) == len(prefix)
            ]
            if terminals:
                if len(terminals) > 1:
                    raise RuntimeError("structured decoder enum tokenization 无法区分候选")
                if len(active) == 1:
                    return terminals[0]
            allowed = sorted({
                ids[len(prefix)] for ids in active.values()
                if len(ids) > len(prefix)
            })
            terminal_token: int | None = None
            if terminals:
                # ``center``/``center_left`` 可能共享 tokenizer 前缀。将下一
                # 个 JSON quote 的 live logit 与子分支比较，而不提前消费它。
                terminal_token = _token_ids(self.tokenizer, '"')[0]
                allowed.append(terminal_token)
            token_id = max(
                sorted(set(allowed)),
                key=lambda value: (float(self.logits[value].float().item()), -value),
            )
            if terminal_token is not None and token_id == terminal_token:
                return terminals[0]
            self.feed(token_id, forced=False)
            prefix = (*prefix, token_id)
            active = {
                name: ids for name, ids in active.items()
                if ids[:len(prefix)] == prefix
            }
        raise RuntimeError("structured decoder enum constrained search 无候选")

    def choose_text(self, *, max_tokens: int) -> tuple[str, str]:
        """Generate a non-empty JSON string payload without consuming its quote."""

        pieces: list[str] = []
        special_ids = {
            int(value) for value in (getattr(self.tokenizer, "all_special_ids", ()) or ())
        }
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            special_ids.add(int(eos))
        for _ in range(int(max_tokens)):
            top_k = min(256, int(self.logits.numel()))
            ranked = torch.topk(self.logits.float(), k=top_k).indices.tolist()
            selected: tuple[int, str] | None = None
            for raw_token_id in ranked:
                token_id = int(raw_token_id)
                if token_id in special_ids:
                    if "".join(pieces).strip():
                        return "".join(pieces), "special_token"
                    continue
                piece = _decode_token(self.tokenizer, token_id)
                if '"' in piece or "\\" in piece or "\ufffd" in piece:
                    if "".join(pieces).strip():
                        return "".join(pieces), "unsafe_string_token"
                    continue
                if not piece or any(ord(character) < 0x20 for character in piece):
                    continue
                selected = token_id, piece
                break
            if selected is None:
                raise RuntimeError("structured decoder 找不到安全的 JSON string token")
            self.feed(selected[0], forced=False)
            pieces.append(selected[1])
        text = "".join(pieces).strip()
        if not text:
            raise RuntimeError("structured decoder 生成了空 JSON string")
        return "".join(pieces), "field_token_limit"


_LOCATION = (
    "upper_left", "upper_center", "upper_right", "center_left", "center",
    "center_right", "lower_left", "lower_center", "lower_right", "distributed",
    "unknown", "unavailable",
)
_SIZE = ("tiny", "small", "medium", "large", "extensive", "unknown", "unavailable")
_SHAPE = (
    "compact", "elongated", "branching", "fragmented", "irregular", "unknown",
    "unavailable",
)
_ELONGATION = ("low", "moderate", "high", "unknown", "unavailable")
_COMPACTNESS = ("compact", "moderate", "dispersed", "unknown", "unavailable")
_FRAGMENTATION = (
    "single", "few_components", "many_components", "highly_fragmented", "unknown",
    "unavailable",
)
_SUPPORT = (
    "supports", "does_not_support", "insufficient_evidence", "unknown", "unavailable",
)
_SUFFICIENCY = ("sufficient", "partial", "insufficient", "unavailable")


def generate_schema_constrained_description(
    tokenizer: Any,
    initial_logits: torch.Tensor,
    advance_tokens: AdvanceTokens,
    *,
    max_new_tokens: int,
) -> DescriptionGenerationResult:
    """Generate the fixed output schema directly, never via post-hoc repair."""

    cursor = _ConstrainedCursor(
        tokenizer,
        initial_logits,
        advance_tokens,
        max_new_tokens=max_new_tokens,
    )
    enum_choices: dict[str, str] = {}
    text_termination: dict[str, str] = {}

    cursor.force('{"schema_version":"qpsalm_description_output_v1","target_status":"')
    target_status = cursor.choose_enum(("present", "absent", "uncertain"))
    enum_choices["target_status"] = target_status

    def enum_field(prefix: str, name: str, values: Sequence[str]) -> str:
        cursor.force(prefix)
        selected = cursor.choose_enum(values)
        enum_choices[name] = selected
        return selected

    absent = target_status == "absent"
    unavailable = ("unavailable",)
    cursor.force('","region":{"location":"')
    region = {
        "location": cursor.choose_enum(unavailable if absent else _LOCATION),
        "size_class": enum_field(
            '","size_class":"',
            "region.size_class",
            unavailable if absent else _SIZE,
        ),
        "shape": enum_field(
            '","shape":"', "region.shape", unavailable if absent else _SHAPE
        ),
        "elongation": enum_field(
            '","elongation":"',
            "region.elongation",
            unavailable if absent else _ELONGATION,
        ),
        "compactness": enum_field(
            '","compactness":"',
            "region.compactness",
            unavailable if absent else _COMPACTNESS,
        ),
        "fragmentation": enum_field(
            '","fragmentation":"',
            "region.fragmentation",
            unavailable if absent else _FRAGMENTATION,
        ),
    }
    enum_choices["region.location"] = region["location"]

    cursor.force('"},"evidence":{"surface_observation":"')
    if absent:
        cursor.force("unavailable")
        surface = "unavailable"
        text_termination["evidence.surface_observation"] = "absent_schema_constraint"
    else:
        surface, reason = cursor.choose_text(max_tokens=12)
        text_termination["evidence.surface_observation"] = reason
    cursor.force('","terrain_support":"')
    support_values = ("insufficient_evidence", "unavailable") if absent else _SUPPORT
    terrain = cursor.choose_enum(support_values)
    enum_choices["evidence.terrain_support"] = terrain
    sar = enum_field('","sar_support":"', "evidence.sar_support", support_values)
    deformation = enum_field(
        '","deformation_support":"', "evidence.deformation_support", support_values
    )
    cursor.force('","surrounding_context":"')
    if absent:
        cursor.force("unavailable")
        context = "unavailable"
        text_termination["evidence.surrounding_context"] = "absent_schema_constraint"
    else:
        context, reason = cursor.choose_text(max_tokens=12)
        text_termination["evidence.surrounding_context"] = reason
    sufficiency_values = ("insufficient", "unavailable") if absent else _SUFFICIENCY
    sufficiency = enum_field(
        '","evidence_sufficiency":"',
        "evidence.evidence_sufficiency",
        sufficiency_values,
    )
    cursor.force('"},"summary":"')
    summary, reason = cursor.choose_text(max_tokens=40)
    text_termination["summary"] = reason
    cursor.force('"}')

    payload = {
        "schema_version": "qpsalm_description_output_v1",
        "target_status": target_status,
        "region": region,
        "evidence": {
            "surface_observation": surface,
            "terrain_support": terrain,
            "sar_support": sar,
            "deformation_support": deformation,
            "surrounding_context": context,
            "evidence_sufficiency": sufficiency,
        },
        "summary": summary,
    }
    raw = canonical_description_json(payload)
    decoded_stream = cursor.decoded_stream()
    if decoded_stream != raw:
        raise RuntimeError(
            "schema-constrained decoder token stream 与发布 raw JSON 不一致: "
            f"stream_sha256={hashlib.sha256(decoded_stream.encode('utf-8')).hexdigest()} "
            f"raw_sha256={hashlib.sha256(raw.encode('utf-8')).hexdigest()}"
        )
    parsed = parse_description_output(raw)
    if not parsed.schema_valid:
        raise RuntimeError(
            "schema-constrained decoder 产生非法 raw JSON: "
            f"{list(parsed.parse_errors)}"
        )
    return DescriptionGenerationResult(
        text=raw,
        audit={
            "protocol": STRUCTURED_GENERATION_PROTOCOL,
            "mode": "schema_constrained_raw_generation",
            "raw_schema_valid": True,
            "repair_used": False,
            "forced_tokens": cursor.forced_tokens,
            "model_selected_tokens": cursor.model_selected_tokens,
            "total_tokens": len(cursor.token_ids),
            "decoder_advance_calls": cursor.advance_calls,
            "max_new_tokens": int(max_new_tokens),
            "enum_choices": enum_choices,
            "text_termination": text_termination,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "token_stream_sha256": hashlib.sha256(
                decoded_stream.encode("utf-8")
            ).hexdigest(),
            "token_stream_matches_raw": True,
        },
    )
