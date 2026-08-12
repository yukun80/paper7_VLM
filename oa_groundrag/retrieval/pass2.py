"""Text-only Pass-2 消息、严格输出与 no-RAG/text-RAG 公平性。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import canonical_json, sha256_text
from oa_groundrag.phase4.errors import ContractError, ReasonCode
from oa_groundrag.phase4.outputs import detect_forbidden_region_claims

from .contracts import (
    PASS2_OUTPUT_SCHEMA,
    RagMode,
    TextRagTask,
    route_text_rag,
    strict_contract_mapping,
)


_OUTPUT_FIELDS = (
    "schema_version",
    "supporting_interpretations",
    "alternative_explanations",
    "limitations",
    "recommended_verification",
    "summary",
)
PASS2_ASSISTANT_PREFILL = (
    '{"schema_version":"oa_groundrag.text_rag.pass2_output.v1",'
    '"supporting_interpretations":'
)
PASS2_CONSTRAINT_SCHEMA = "oa_groundrag.text_rag.json_fsm.v1"
_SAFE_TOKEN_CACHE: dict[int, tuple[int, ...]] = {}
_LIST_FIELDS = (
    "supporting_interpretations",
    "alternative_explanations",
    "limitations",
    "recommended_verification",
)
_EXPECTED_TYPES = {
    "supporting_interpretations": frozenset({"interpretation"}),
    "alternative_explanations": frozenset({"confounder"}),
    "limitations": frozenset({"limitation"}),
    "recommended_verification": frozenset({"limitation"}),
}
_GEOMETRY_PATTERNS = (
    r"\bbbox\b", r"\bbounding box\b", r"\bcentroid\b", r"\barea[_ ]pixels\b",
    r"\barea ratio\b", r"\bperimeter\b", r"\bcompactness\b", r"\belongation\b",
    r"边界框", r"质心", r"像素面积", r"面积比例", r"周长",
)
_CONFIRMED_PATTERNS = (
    r"(?:该|此|这个|目标|候选)?区域?(?:已)?(?:确认|确定|判定)(?:为|是)?(?:一处|典型)?滑坡",
    r"(?:该|此|这个|目标|候选)区域(?:就是|是|为)(?:一处|典型)?滑坡",
    r"\b(?:is|constitutes) (?:a )?(?:confirmed |definite )?landslide\b",
    r"\bconfirmed as (?:a )?landslide\b",
)
_NEGATION_MARKERS = (
    "不能", "无法", "不可", "不足以", "尚不能", "未能", "未明确", "未确认", "未确定",
    "缺乏", "没有", "无证据", "不应", "不要", "并非",
    "cannot", "can not", "insufficient", "not enough", "should not", "must not", "not confirmed",
)
_MISCLASSIFICATION_MARKERS = (
    "误认为", "误判为", "误识别为", "错误识别为",
    "mistaken for", "misclassified as", "incorrectly identified as",
)
_QUALIFIER_BREAKERS = (
    "但", "但是", "然而", "不过", "；", "。", ";", ".",
    " but ", " however ", " yet ",
)
_OBSERVATION_LEAK_PATTERNS = (
    r"(?:根据|依据)(?:知识库|检索资料|文献).{0,30}(?:图像|影像)(?:中)?(?:显示|可见|观察到)",
    r"(?:知识库|检索资料|文献)(?:证明|表明).{0,30}(?:当前|该)(?:图像|影像|区域)(?:存在|出现|具有)",
    r"\b(?:the )?(?:retrieved )?(?:knowledge|literature) (?:shows|proves) that the image\b",
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key: {key}")
        output[key] = value
    return output


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-finite constant: {token}")


@dataclass(frozen=True)
class Pass2Item:
    text: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class Pass2Output:
    supporting_interpretations: tuple[Pass2Item, ...]
    alternative_explanations: tuple[Pass2Item, ...]
    limitations: tuple[Pass2Item, ...]
    recommended_verification: tuple[Pass2Item, ...]
    summary: Pass2Item

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PASS2_OUTPUT_SCHEMA,
            "supporting_interpretations": [item.to_dict() for item in self.supporting_interpretations],
            "alternative_explanations": [item.to_dict() for item in self.alternative_explanations],
            "limitations": [item.to_dict() for item in self.limitations],
            "recommended_verification": [item.to_dict() for item in self.recommended_verification],
            "summary": self.summary.to_dict(),
        }


def _locally_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 56):start].lower()
    return any(marker in prefix for marker in _NEGATION_MARKERS)


def _locally_marked_as_misclassification(text: str, start: int) -> bool:
    """仅豁免紧邻的“误判为滑坡”反例语境，不放宽后续肯定式结论。"""

    prefix = text[max(0, start - 64):start].lower()
    positions = [
        (prefix.rfind(marker), marker)
        for marker in _MISCLASSIFICATION_MARKERS
        if prefix.rfind(marker) >= 0
    ]
    if not positions:
        return False
    position, marker = max(positions, key=lambda item: item[0])
    between = prefix[position + len(marker):]
    return not any(breaker in between for breaker in _QUALIFIER_BREAKERS)


def _asserts_landslide_trigger(text: str) -> bool:
    """区分滑坡诱因断言与“工程活动导致视觉扰动”等混淆因素描述。"""

    patterns = (
        r"(?:降雨|暴雨|强降雨|地震|施工|工程活动)[^，。；,;\n]{0,16}(?:导致|触发|引起)"
        r"[^，。；,;\n]{0,24}(?:滑坡|坡体失稳|边坡失稳|滑动|地质灾害)",
        r"(?:滑坡|坡体失稳|边坡失稳|地质灾害)[^，。；,;\n]{0,24}(?:由|因)"
        r"[^，。；,;\n]{0,16}(?:降雨|暴雨|强降雨|地震|施工|工程活动)",
        r"\b(?:landslide|slope failure|mass movement)[^.;\n]{0,24}(?:triggered|caused) by\b",
        r"\b(?:rainfall|rainstorm|earthquake|construction)[^.;\n]{0,24}(?:triggered|caused)"
        r"[^.;\n]{0,24}(?:landslide|slope failure|mass movement)\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _reject_text(text: str, *, location: str) -> str:
    if not isinstance(text, str) or not text.strip() or len(text.strip()) > 3000:
        raise ContractError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是长度受限的非空字符串")
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _GEOMETRY_PATTERNS):
        raise ContractError(ReasonCode.FORBIDDEN_MODEL_FACT, f"{location}: 不得重写程序几何事实")
    for pattern in _CONFIRMED_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            if not (
                _locally_negated(lowered, match.start())
                or _locally_marked_as_misclassification(lowered, match.start())
            ):
                raise ContractError(ReasonCode.FORBIDDEN_CLAIM, f"{location}: 不得把候选区域升级为确认滑坡")
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _OBSERVATION_LEAK_PATTERNS):
        raise ContractError(ReasonCode.FORBIDDEN_CLAIM, f"{location}: 不得把检索知识写成当前图像观察")
    violations = detect_forbidden_region_claims(normalized)
    if "trigger_cause" in violations and not _asserts_landslide_trigger(lowered):
        violations = tuple(item for item in violations if item != "trigger_cause")
    if violations:
        raise ContractError(ReasonCode.FORBIDDEN_CLAIM, f"{location}: 含禁止式专业结论 {list(violations)}")
    return normalized


def _parse_item(
    value: Any,
    *,
    location: str,
    mode: RagMode,
    packet_types: Mapping[str, str],
    allowed_types: frozenset[str] | None,
) -> Pass2Item:
    row = strict_contract_mapping(value, fields=("text", "evidence_ids"), location=location)
    text = _reject_text(row["text"], location=f"{location}.text")
    ids = row["evidence_ids"]
    if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
        raise ContractError(ReasonCode.TYPE_MISMATCH, f"{location}.evidence_ids 必须是字符串列表")
    evidence_ids = tuple(ids)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{location}.evidence_ids 不得重复")
    unknown = sorted(set(evidence_ids) - set(packet_types))
    if unknown:
        raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{location}: 未知 Evidence ID {unknown}")
    if mode is RagMode.NO_RAG and evidence_ids:
        raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{location}: no_rag 不得引用 evidence")
    if mode is RagMode.TEXT_RAG and not evidence_ids:
        raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{location}: text_rag item 必须引用 evidence")
    if allowed_types is not None:
        mismatched = sorted(evidence_id for evidence_id in evidence_ids if packet_types[evidence_id] not in allowed_types)
        if mismatched:
            raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{location}: Evidence type 不相容 {mismatched}")
    return Pass2Item(text=text, evidence_ids=evidence_ids)


def parse_pass2_output(
    value: str | Mapping[str, Any],
    *,
    mode: RagMode | str,
    packet: Mapping[str, Any] | None,
) -> Pass2Output:
    try:
        rag_mode = mode if isinstance(mode, RagMode) else RagMode(mode)
    except (TypeError, ValueError) as error:
        raise ContractError(ReasonCode.INVALID_ENUM, f"非法 RAG mode：{mode!r}") from error
    if isinstance(value, str):
        try:
            parsed = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ContractError(
                ReasonCode.INVALID_MODEL_OUTPUT,
                "Pass-2 输出不是严格 JSON",
                details={"error": str(error)},
            ) from error
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise ContractError(ReasonCode.TYPE_MISMATCH, "Pass-2 输出必须是 JSON 字符串或对象")
    row = strict_contract_mapping(parsed, fields=_OUTPUT_FIELDS, location="$")
    if row["schema_version"] != PASS2_OUTPUT_SCHEMA:
        raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "Pass-2 schema_version 不匹配")
    items = [] if packet is None else packet.get("items")
    if not isinstance(items, list):
        raise ContractError(ReasonCode.TYPE_MISMATCH, "packet.items 非法")
    packet_types: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("evidence_id"), str) or not isinstance(item.get("knowledge_type"), str):
            raise ContractError(ReasonCode.TYPE_MISMATCH, "packet evidence item 非法")
        evidence_id = str(item["evidence_id"])
        if evidence_id in packet_types:
            raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "packet evidence ID 重复")
        packet_types[evidence_id] = str(item["knowledge_type"])
    if rag_mode is RagMode.NO_RAG and packet_types:
        raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "no_rag parser 必须消费空 packet")
    if rag_mode is RagMode.TEXT_RAG and not packet_types:
        raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "text_rag parser 不接受空 packet")
    parsed_lists: dict[str, tuple[Pass2Item, ...]] = {}
    for field in _LIST_FIELDS:
        values = row[field]
        if not isinstance(values, list) or len(values) > 16:
            raise ContractError(ReasonCode.TYPE_MISMATCH, f"$.{field} 必须是长度受限列表")
        parsed_lists[field] = tuple(
            _parse_item(
                item,
                location=f"$.{field}[{index}]",
                mode=rag_mode,
                packet_types=packet_types,
                allowed_types=_EXPECTED_TYPES[field],
            )
            for index, item in enumerate(values)
        )
    summary = _parse_item(
        row["summary"],
        location="$.summary",
        mode=rag_mode,
        packet_types=packet_types,
        allowed_types=None,
    )
    return Pass2Output(
        supporting_interpretations=parsed_lists["supporting_interpretations"],
        alternative_explanations=parsed_lists["alternative_explanations"],
        limitations=parsed_lists["limitations"],
        recommended_verification=parsed_lists["recommended_verification"],
        summary=summary,
    )


def pass2_output_contract() -> dict[str, Any]:
    item = {
        "type": "object",
        "additional_properties": False,
        "required": ["text", "evidence_ids"],
        "properties": {
            "text": {"type": "string", "non_empty": True},
            "evidence_ids": {
                "type": "array",
                "items": "current_packet_evidence_id",
                "unique": True,
            },
        },
    }
    return {
        "type": "object",
        "additional_properties": False,
        "required": list(_OUTPUT_FIELDS),
        "properties": {
            "schema_version": {"type": "string", "const": PASS2_OUTPUT_SCHEMA},
            **{
                field: {"type": "array", "items": item, "max_items": 16}
                for field in _LIST_FIELDS
            },
            "summary": item,
        },
        "evidence_type_binding": {
            "supporting_interpretations": ["interpretation"],
            "alternative_explanations": ["confounder"],
            "limitations": ["limitation"],
            "recommended_verification": ["limitation"],
            "summary": ["interpretation", "confounder", "limitation"],
        },
    }


def _constraint_evidence_ids(
    *,
    mode: RagMode,
    packet: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = {
        "interpretation": [], "confounder": [], "limitation": [],
    }
    if packet is not None:
        for item in packet.get("items", []):
            knowledge_type = str(item.get("knowledge_type"))
            evidence_id = item.get("evidence_id")
            if knowledge_type in by_type and isinstance(evidence_id, str):
                by_type[knowledge_type].append(evidence_id)
    if mode is RagMode.NO_RAG:
        return {field: [] for field in (*_LIST_FIELDS, "summary")}
    first = {key: values[:1] for key, values in by_type.items()}
    summary = next((values for values in first.values() if values), [])
    return {
        "supporting_interpretations": first["interpretation"],
        "alternative_explanations": first["confounder"],
        "limitations": first["limitation"],
        "recommended_verification": first["limitation"],
        "summary": summary,
    }


def pass2_constraint_identity(
    *,
    mode: RagMode,
    packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """返回可独立重算的生成约束身份，不依赖 tokenizer 或模型运行状态。"""

    return {
        "schema_version": PASS2_CONSTRAINT_SCHEMA,
        "mode": mode.value,
        "citation_ids": _constraint_evidence_ids(mode=mode, packet=packet),
        "field_order": [*_LIST_FIELDS, "summary"],
    }


def _safe_text_token_ids(tokenizer: Any) -> tuple[int, ...]:
    cache_key = id(tokenizer)
    cached = _SAFE_TOKEN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    special = set(int(value) for value in tokenizer.all_special_ids)
    values: list[int] = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            piece
            and '"' not in piece
            and "\\" not in piece
            and not any(ord(character) < 32 for character in piece)
        ):
            values.append(token_id)
    if not values:
        raise ContractError(ReasonCode.MODEL_IDENTITY_MISMATCH, "tokenizer 没有 JSON string 安全 token")
    result = tuple(values)
    _SAFE_TOKEN_CACHE[cache_key] = result
    return result


class Pass2JSONLogitsProcessor:
    """只约束 JSON 结构与 citation ID；自然语言内容仍由冻结 generator 产生。"""

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_length: int,
        mode: RagMode,
        packet: Mapping[str, Any] | None,
    ) -> None:
        if prompt_length < 1:
            raise ContractError(ReasonCode.TYPE_MISMATCH, "constraint prompt_length 非法")
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.safe_ids = _safe_text_token_ids(tokenizer)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.identity = pass2_constraint_identity(mode=mode, packet=packet)
        citations = self.identity["citation_ids"]
        self.segments: list[dict[str, Any]] = []

        def fixed(value: str) -> None:
            ids = tokenizer.encode(value, add_special_tokens=False)
            if not ids:
                raise ContractError(ReasonCode.MODEL_IDENTITY_MISMATCH, "JSON constraint fixed segment 为空")
            self.segments.append({"kind": "fixed", "ids": tuple(int(item) for item in ids), "offset": 0})

        def free_text() -> None:
            self.segments.append({"kind": "free", "count": 0, "minimum": 6, "maximum": 72})

        fields = (
            "supporting_interpretations",
            "alternative_explanations",
            "limitations",
            "recommended_verification",
        )
        for index, field in enumerate(fields):
            if index:
                fixed(f',"{field}":')
            ids = citations[field]
            if mode is RagMode.TEXT_RAG and not ids:
                fixed("[]")
                continue
            fixed('[{"text":"')
            free_text()
            fixed('","evidence_ids":' + canonical_json(ids) + "}]")
        fixed(',"summary":{"text":"')
        free_text()
        fixed('","evidence_ids":' + canonical_json(citations["summary"]) + "}}")
        self.segment_index = 0
        self.consumed = 0

    def _advance(self) -> None:
        self.segment_index += 1

    def _consume(self, token_id: int) -> None:
        if self.segment_index >= len(self.segments):
            if token_id != self.eos_token_id:
                raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "JSON constraint EOS 漂移")
            return
        segment = self.segments[self.segment_index]
        if segment["kind"] == "fixed":
            expected = segment["ids"][segment["offset"]]
            if token_id != expected:
                raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "JSON constraint fixed token 漂移")
            segment["offset"] += 1
            if segment["offset"] == len(segment["ids"]):
                self._advance()
            return
        segment["count"] += 1
        piece = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            segment["count"] >= segment["maximum"]
            or (
                segment["count"] >= segment["minimum"]
                and any(marker in piece for marker in ("。", "！", "？", ".", "!", "?", ";", "；"))
            )
        ):
            self._advance()

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ContractError(ReasonCode.TYPE_MISMATCH, "JSON constraint 只支持 batch size 1")
        generated = input_ids[0, self.prompt_length:].tolist()
        while self.consumed < len(generated):
            self._consume(int(generated[self.consumed]))
            self.consumed += 1
        masked = scores.new_full(scores.shape, float("-inf"))
        if self.segment_index >= len(self.segments):
            masked[:, self.eos_token_id] = 0.0
            return masked
        segment = self.segments[self.segment_index]
        if segment["kind"] == "fixed":
            masked[:, segment["ids"][segment["offset"]]] = 0.0
        else:
            masked[:, self.safe_ids] = scores[:, self.safe_ids]
        return masked


def build_pass2_logits_processor(
    *,
    tokenizer: Any,
    prompt_length: int,
    mode: RagMode,
    packet: Mapping[str, Any] | None,
) -> Pass2JSONLogitsProcessor:
    return Pass2JSONLogitsProcessor(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        mode=mode,
        packet=packet,
    )


def _evidence_context(packet: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if packet is None:
        return []
    output = []
    for item in packet.get("items", []):
        output.append({
            "evidence_id": item["evidence_id"],
            "knowledge_type": item["knowledge_type"],
            "content": item["content"],
            "conditions": item["conditions"],
            "source_id": item["source_id"],
            "pdf_page": item["pdf_page"],
            "section": item["section"],
        })
    return output


def build_pass2_payload(
    *,
    question: str,
    target_status: str,
    program_facts: Mapping[str, Any],
    observation: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "user_question": question,
        "target_status_read_only": target_status,
        "programmatic_facts_read_only": dict(program_facts),
        "pass1_visual_observation_read_only": dict(observation),
        "retrieved_evidence": _evidence_context(packet),
        "required_output_schema": PASS2_OUTPUT_SCHEMA,
        "required_output_fields_in_order": list(_OUTPUT_FIELDS),
        "evidence_type_binding": {
            key: sorted(value) for key, value in _EXPECTED_TYPES.items()
        },
    }


def build_pass2_messages(
    *,
    question: str,
    target_status: str,
    program_facts: Mapping[str, Any],
    observation: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
    task: TextRagTask | str = TextRagTask.CANDIDATE_INTERPRETATION,
) -> list[dict[str, Any]]:
    try:
        task_value = task if isinstance(task, TextRagTask) else TextRagTask(task)
    except (TypeError, ValueError) as error:
        raise ContractError(
            ReasonCode.INVALID_ENUM,
            f"非法 Pass-2 task：{task!r}",
        ) from error
    if not route_text_rag(task_value):
        raise ContractError(
            ReasonCode.RAG_FORBIDDEN,
            f"Pass-2 task 不启用 RAG：{task_value.value}",
        )
    payload = build_pass2_payload(
        question=question,
        target_status=target_status,
        program_facts=program_facts,
        observation=observation,
        packet=packet,
    )
    if task_value is TextRagTask.PROFESSIONAL_QA:
        instruction = (
            "你正在执行文本-only 的遥感与地质灾害专业问答。回答只能依据用户问题和当前 packet；"
            "不得声称存在未提供的当前影像、mask、区域观察或现场事实。区分支持性解释、替代解释、"
            "证据限制和建议核查，不给出当前场景的危险性、稳定性、失效概率或风险等级。"
            "每个 item 必须引用当前 packet 中与字段类型相容的真实 Evidence ID，禁止虚构 ID。"
            "严格保持合同字段与类型。只返回一个严格 JSON 对象，不要复制 Contract，不要输出"
            "Markdown、解释或额外字段。\nContract: "
        )
    else:
        # 默认分支保持既有 Stage 6 candidate prompt 字节不变。
        instruction = (
            "你正在执行文本-only 的候选区域专业解释。视觉事实只能来自只读 Pass-1 observation；"
            "不得重新观察、补造视觉事实、修改 mask/程序事实，或把候选区域确认成滑坡。"
            "区分支持性解释、混淆因素、证据限制和建议核查，不给出危险性、稳定性、失效概率或风险等级。"
            "retrieved_evidence 为空时所有 evidence_ids 必须为空；非空时每个 item 必须引用当前 packet 中"
            "与字段类型相容的真实 Evidence ID，禁止虚构 ID。严格保持合同字段与类型。"
            "只返回一个严格 JSON 对象，不要复制 Contract，不要输出 Markdown、解释或额外字段。"
            "\nContract: "
        )
    return [
        {"role": "user", "content": [{"type": "text", "text": instruction + canonical_json(payload)}]},
        {"role": "assistant", "content": [{"type": "text", "text": PASS2_ASSISTANT_PREFILL}]},
    ]


def prompt_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(list(messages)))


def validate_prompt_fairness(
    no_rag_messages: Sequence[Mapping[str, Any]],
    text_rag_messages: Sequence[Mapping[str, Any]],
) -> str:
    def normalized(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1] != {
                "role": "assistant",
                "content": [{"type": "text", "text": PASS2_ASSISTANT_PREFILL}],
            }
        ):
            raise ContractError(ReasonCode.TYPE_MISMATCH, "Pass-2 必须是 user + 固定 assistant prefill")
        content = messages[0].get("content")
        if not isinstance(content, list) or len(content) != 1 or content[0].get("type") != "text":
            raise ContractError(ReasonCode.TYPE_MISMATCH, "Pass-2 message 必须只有一个文本块")
        text = str(content[0]["text"])
        instruction, marker, contract = text.partition("\nContract: ")
        if not marker or not instruction:
            raise ContractError(ReasonCode.TYPE_MISMATCH, "Pass-2 prompt 缺少自然语言指令或 Contract")
        try:
            row = json.loads(contract, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ContractError(ReasonCode.TYPE_MISMATCH, "Pass-2 prompt payload 非严格 JSON") from error
        if not isinstance(row, dict) or "retrieved_evidence" not in row:
            raise ContractError(ReasonCode.TYPE_MISMATCH, "Pass-2 prompt 缺少 retrieved_evidence")
        row["retrieved_evidence"] = []
        return {"instruction": instruction, "contract": row}
    left, right = normalized(no_rag_messages), normalized(text_rag_messages)
    if left != right:
        raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "no_rag/text_rag prompt 主体不公平")
    return sha256_text(canonical_json(left))
