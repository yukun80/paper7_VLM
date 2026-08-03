"""Stage 4B Silver 候选合同与科学规则过滤。"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import canonical_json, sha256_text

from .contracts import SILVER_PROMPT_VERSION, SILVER_SCHEMA, SilverObservation, fail


CANDIDATE_FIELDS = (
    "schema_version", "candidate_id", "attempt_id", "record_id", "sample_id",
    "candidate_index", "variant", "seed", "evidence_identity", "prompt_identity",
    "raw_output_sha256", "silver_observation",
)


def evidence_identity(record: Mapping[str, Any], hashes: Mapping[str, str]) -> dict[str, Any]:
    assets = record["assets"]
    return {
        "optical_sha256": hashes[assets["optical"]],
        "mask_sha256": hashes[assets["mask"]],
        "overlay_sha256": None if assets["overlay"] is None else hashes[assets["overlay"]],
        "crop_sha256": None if assets["crop"] is None else hashes[assets["crop"]],
        "auxiliaries": {name: hashes[path] for name, path in sorted(assets["auxiliaries"].items())},
    }


def _candidate_id(value: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json({key: value[key] for key in CANDIDATE_FIELDS if key != "candidate_id"}))


def make_silver_candidate(
    record: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    attempt_id: str,
    candidate_index: int,
    variant: str,
    seed: int,
    identity: Mapping[str, Any],
    prompt_identity: Mapping[str, Any],
    raw_output_sha256: str,
) -> dict[str, Any]:
    normalized = SilverObservation.from_mapping(observation).to_dict()
    row = {
        "schema_version": SILVER_SCHEMA, "candidate_id": "", "attempt_id": attempt_id,
        "record_id": record["record_id"], "sample_id": record["sample_id"],
        "candidate_index": candidate_index, "variant": variant, "seed": seed,
        "evidence_identity": dict(identity), "prompt_identity": dict(prompt_identity),
        "raw_output_sha256": raw_output_sha256, "silver_observation": normalized,
    }
    row["candidate_id"] = _candidate_id(row)
    return row


def validate_silver_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(CANDIDATE_FIELDS):
        fail("SILVER_INVALID", "Silver candidate 字段不匹配")
    row = dict(value)
    if row["schema_version"] != SILVER_SCHEMA:
        fail("SILVER_INVALID", "Silver candidate schema 非法")
    for field in ("candidate_id", "attempt_id", "record_id", "sample_id", "raw_output_sha256"):
        if not isinstance(row[field], str) or not row[field]:
            fail("SILVER_INVALID", f"{field} 必须是非空字符串")
    for field in ("candidate_id", "attempt_id", "raw_output_sha256"):
        if len(row[field]) != 64 or any(char not in "0123456789abcdef" for char in row[field]):
            fail("SILVER_INVALID", f"{field} 必须是小写 SHA-256")
    if (
        isinstance(row["candidate_index"], bool) or not isinstance(row["candidate_index"], int)
        or row["candidate_index"] not in {0, 1}
        or isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or row["seed"] < 0
        or row["variant"] not in {"regular", "wrong_mask", "modality_removal"}
        or (row["variant"] != "regular" and row["candidate_index"] != 0)
    ):
        fail("SILVER_INVALID", "candidate index/seed/variant 非法")
    identity = row["evidence_identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "optical_sha256", "mask_sha256", "overlay_sha256", "crop_sha256", "auxiliaries",
    }:
        fail("SILVER_INVALID", "evidence_identity 字段不匹配")
    for field in ("optical_sha256", "mask_sha256"):
        value = identity[field]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            fail("SILVER_INVALID", f"{field} 非法")
    for field in ("overlay_sha256", "crop_sha256"):
        if identity[field] is not None and (
            not isinstance(identity[field], str) or len(identity[field]) != 64
            or any(char not in "0123456789abcdef" for char in identity[field])
        ):
            fail("SILVER_INVALID", f"{field} 非法")
    if not isinstance(identity["auxiliaries"], dict):
        fail("SILVER_INVALID", "auxiliaries identity 非法")
    for name, digest in identity["auxiliaries"].items():
        if (
            not isinstance(name, str) or not name or not isinstance(digest, str)
            or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        ):
            fail("SILVER_INVALID", "auxiliaries identity entry 非法")
    prompt = row["prompt_identity"]
    if (
        not isinstance(prompt, dict) or set(prompt) != {"version", "template_sha256", "request_sha256"}
        or prompt.get("version") != SILVER_PROMPT_VERSION
    ):
        fail("SILVER_INVALID", "prompt_identity 字段不匹配")
    for field in ("template_sha256", "request_sha256"):
        digest = prompt.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            fail("SILVER_INVALID", f"prompt_identity.{field} 非法")
    row["silver_observation"] = SilverObservation.from_mapping(row["silver_observation"]).to_dict()
    expected_id = _candidate_id(row)
    if row["candidate_id"] != expected_id:
        fail("SILVER_INVALID", "candidate_id identity 不一致")
    return row


def _all_text(observation: Mapping[str, Any]) -> str:
    values: list[str] = []
    for value in observation.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return " ".join(values).casefold()


def _mentions(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _landslide_claim_stances(text: str) -> set[str]:
    """将包含 landslide 的短句分为确定、谨慎和否定语境。"""

    stances: set[str] = set()
    clauses = re.split(r"[。！？；，,!?.;\n]+", text.casefold())
    for clause in clauses:
        if not re.search(r"滑坡|\blandslide(?:s)?\b", clause, flags=re.IGNORECASE):
            continue
        negated = _mentions(clause, (
            r"(?:未|没有|无|无法|不能|不可|尚不能|难以|不足以|不存在|不是|并非|不确定)[^。；，,\n]{0,24}(?:滑坡)",
            r"(?:非滑坡)",
            r"(?:滑坡)[^。；，,\n]{0,16}(?:未|没有|无法|不能|不可|尚不能|难以)(?:确认|确定|判定|识别|证明)?",
            r"\b(?:no|not|cannot|can't|unable|insufficient)\b[^.;,\n]{0,32}\blandslide(?:s)?\b",
            r"\blandslide(?:s)?\b[^.;,\n]{0,24}\b(?:not|unconfirmed|undetermined)\b",
        ))
        hedged_cue = _mentions(clause, (
            r"(?:疑似|可能(?:是|为|属于)?|或为|候选(?:的)?|推测(?:为)?)[^。；，,\n]{0,12}滑坡",
            r"滑坡(?:疑似|可能|候选)",
            r"\b(?:possible|possibly|probable|suspected|potential|candidate)\b[^.;,\n]{0,24}\blandslide(?:s)?\b",
            r"\b(?:may|might|could|appears?\s+to)\b[^.;,\n]{0,24}\b(?:be\s+)?(?:a\s+)?landslide\b",
        ))
        hedged_despite_negation = _mentions(clause, (
            r"(?:不能|无法|未能|不)(?:完全)?排除[^。；，,\n]{0,12}滑坡",
            r"(?:但|但是|然而|仍|仍然)[^。；，,\n]{0,12}(?:疑似|可能|候选)[^。；，,\n]{0,12}滑坡",
            r"\b(?:cannot|can't|unable\s+to)\s+rule\s+out\b[^.;,\n]{0,24}\blandslide(?:s)?\b",
            r"\b(?:but|however|still)\b[^.;,\n]{0,24}\b(?:possible|suspected|may|might|could)\b[^.;,\n]{0,24}\blandslide(?:s)?\b",
        ))
        hedged = (hedged_cue and not negated) or hedged_despite_negation
        categorical = _mentions(clause, (
            r"(?:这是|该(?:处|区域|地块|位置)(?:是|为|属于)?|属于|确认|确定|明确|发现|检测到|判定为|识别为|存在)[^。；，,\n]{0,12}滑坡",
            r"滑坡(?:体|区域|边界)",
            r"\b(?:confirmed|detected|identified|is|contains?|present)\b[^.;,\n]{0,24}\blandslide(?:s)?\b",
            r"\blandslide(?:s)?\b[^.;,\n]{0,16}\b(?:exists?|is\s+present|confirmed|detected)\b",
        ))
        if negated:
            stances.add("negated")
        if hedged:
            stances.add("hedged")
        if categorical and not negated and not hedged:
            stances.add("categorical")
        elif not negated and not hedged:
            # 未带限定词的“滑坡”陈述按确定性声明处理，避免静默接受裸断言。
            stances.add("categorical")
    return stances


def rule_violations(candidate: Mapping[str, Any], record: Mapping[str, Any]) -> set[str]:
    observation = candidate["silver_observation"]
    text = _all_text(observation)
    free_text = (
        observation["optical_observation"], observation["short_summary"],
        *observation["possible_confusers"],
    )
    unavailable = set(record["unavailable_modalities"])
    violations: set[str] = set()
    modality_terms = {
        "dem": (r"\bdem\b", r"elevation", r"高程"),
        "slope": (r"\bslope\b", r"坡度"),
        "insar_velocity": (r"\binsar\b", r"形变", r"deformation"),
    }
    for modality, patterns in modality_terms.items():
        if modality in unavailable and _mentions(text, patterns):
            violations.add("UNAVAILABLE_MODALITY_CLAIM")
    insar = record["program_facts"]["modalities"]["insar_velocity"]
    if insar["status"] != "available" and _mentions(text, (r"形变", r"deformation", r"位移", r"velocity")):
        violations.add("INSAR_DEFORMATION_WITHOUT_VALID_EVIDENCE")
    forbidden = (
        r"发生(?:于|时间)", r"\b(?:recent|historical|occurred|date)\b",
        r"触发", r"诱因", r"\b(?:triggered|caused)\b", r"降雨导致", r"地震导致",
        r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m)\s*/\s*(?:day|yr|year|d)\b", r"速度",
        r"稳定性", r"\b(?:stable|unstable|stability)\b",
        r"风险等级", r"高风险", r"低风险", r"\brisk\s+(?:level|grade|high|low)\b",
        r"抬升", r"沉降", r"\b(?:uplift|subsidence)\b",
    )
    landslide_stances: set[str] = set()
    for value in free_text:
        landslide_stances.update(_landslide_claim_stances(value))
    if _mentions(text, forbidden) or (
        record["target_state"] == "target_present" and "categorical" in landslide_stances
    ):
        violations.add("FORBIDDEN_CONCLUSION")
    if record["target_state"] == "no_target" and landslide_stances.intersection({"categorical", "hedged"}):
        violations.add("NO_TARGET_PRESENCE_CLAIM")
    if record["target_state"] == "no_target" and (
        observation["boundary_clarity"] != "not_applicable"
        or observation["surface_or_vegetation_disturbance"] != "not_applicable"
        or observation["terrain_support"] != "unavailable"
        or observation["evidence_sufficiency"] != "insufficient"
    ):
        violations.add("NO_TARGET_FIELD_CONFLICT")
    terrain_available = any(
        record["program_facts"]["modalities"][name]["status"] == "available"
        for name in ("dem", "slope")
    )
    if not terrain_available and observation["terrain_support"] == "supports":
        violations.add("TERRAIN_SUPPORT_WITHOUT_TERRAIN_EVIDENCE")
    if any(value and not re.search(r"[\u3400-\u9fff]", value) for value in free_text):
        violations.add("NON_CHINESE_OBSERVATION")
    mask = record["program_facts"]["mask"]
    number_rules = (
        (r"(?:area_pixels|area pixels?|像素面积)\s*[:=]?\s*(\d+)", mask["area_pixels"]),
        (r"(?:fragment_count|components?|连通(?:分量|区域)(?:数)?)\s*[:=]?\s*(\d+)", mask["fragment_count"]),
        (r"(?:perimeter_pixels|perimeter pixels?|周长像素)\s*[:=]?\s*(\d+)", mask["perimeter_pixels"]),
    )
    for pattern, expected in number_rules:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if int(match.group(1)) != int(expected):
                violations.add("GEOMETRY_CONFLICT")
    ratio = mask["area_ratio"]
    for match in re.finditer(r"(?:area_ratio|foreground ratio|前景比例)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?", text, flags=re.IGNORECASE):
        value = float(match.group(1))
        if "%" in match.group(0):
            value /= 100.0
        if abs(value - float(ratio)) > 1e-6:
            violations.add("GEOMETRY_CONFLICT")
    return violations
