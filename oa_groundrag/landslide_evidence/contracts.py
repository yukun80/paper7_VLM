"""Stage 4A Landslide Evidence Corpus 的严格合同。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import canonical_json, first_symlink_component, sha256_file, sha256_text
from oa_groundrag.phase4.config import _load_yaml

CONFIG_SCHEMA = "oa_groundrag.landslide_evidence.config.v1"
MANIFEST_SCHEMA = "oa_groundrag.landslide_evidence.manifest.v1"
RECORD_SCHEMA = "oa_groundrag.landslide_evidence.record.v1"
SILVER_SCHEMA = "oa_groundrag.landslide_evidence.silver.v1"
SILVER_CONFIG_SCHEMA = "oa_groundrag.landslide_evidence.silver_config.v1"
SILVER_PROMPT_VERSION = "landslide_evidence.silver_zh.v1"
SILVER_COUNTERFACTUAL_ALGORITHM = "landslide_evidence.counterfactual_hash.v1"
SILVER_REVIEW_ALGORITHM = "landslide_evidence.review_priority_hash.v1"
SELECTION_ALGORITHM = "landslide_evidence.stratified_hash.v1"
SUPPORTED_MODALITIES = ("dem", "insar_velocity", "slope")
FALSE_FLAGS = (
    "silver_generated", "expert_review_completed", "oa_grounded_eval",
    "adapter_training_eligible", "case_kb_eligible")
EXPECTED_IDENTITY_FIELDS = (
    "schema_version", "manifest_sha256", "index_sha256", "payload_sha256",
    "build_config_sha256", "ledger_file_sha256", "registry_sha256", "train_split_sha256",
)


class LandslideEvidenceError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.code, self.details = code, dict(details or {})
        super().__init__(f"{code}: {message}")


def fail(code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
    raise LandslideEvidenceError(code, message, details=details)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("TYPE_MISMATCH", f"{location} 必须是字符串键对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: Sequence[str], location: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        fail("SCHEMA_MISMATCH", f"{location} 字段不匹配", details={
            "missing": sorted(expected - set(value)), "unknown": sorted(set(value) - expected),
        })


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("TYPE_MISMATCH", f"{location} 必须是非空字符串")
    return value.strip()


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail("TYPE_MISMATCH", f"{location} 必须是 >= {minimum} 的整数")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("TYPE_MISMATCH", f"{location} 必须是数值")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        fail("TYPE_MISMATCH", f"{location} 必须位于 [0,1]")
    return result


def _sha256(value: Any, location: str) -> str:
    text = _string(value, location)
    if len(text) != 64 or text != text.lower() or any(char not in "0123456789abcdef" for char in text):
        fail("TYPE_MISMATCH", f"{location} 必须是小写 SHA-256")
    return text


@dataclass(frozen=True)
class ForegroundBin:
    name: str
    lower_exclusive: float
    upper_inclusive: float

    def contains(self, value: float) -> bool:
        return self.lower_exclusive < value <= self.upper_inclusive


@dataclass(frozen=True)
class SourceQuota:
    source: str
    total: int
    target: int
    no_target: int


@dataclass(frozen=True)
class PilotConfig:
    config_path: Path
    config_file_sha256: str
    semantic_sha256: str
    benchmark_root: Path
    expected: Mapping[str, str]
    output_root: Path
    sample_count: int
    seed: int
    algorithm_version: str
    source_quotas: tuple[SourceQuota, ...]
    foreground_bins: tuple[ForegroundBin, ...]
    context_margin_ratio: float
    overlay_alpha: float
    min_auxiliary_coverage: float

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "benchmark": {"root": str(self.benchmark_root), **dict(sorted(self.expected.items()))},
            "output_root": str(self.output_root),
            "selection": {
                "sample_count": self.sample_count, "seed": self.seed,
                "algorithm_version": self.algorithm_version,
                "source_quotas": [item.__dict__ for item in self.source_quotas],
                "foreground_bins": [item.__dict__ for item in self.foreground_bins],
            },
            "rendering": {
                "context_margin_ratio": self.context_margin_ratio, "overlay_alpha": self.overlay_alpha,
                "min_auxiliary_coverage": self.min_auxiliary_coverage,
            },
        }


def _absolute(base: Path, value: Any, location: str) -> Path:
    path = Path(_string(value, location))
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def load_config(path: Path | str) -> PilotConfig:
    config_path = Path(os.path.abspath(Path(path)))
    linked = first_symlink_component(config_path)
    if linked is not None:
        fail("OUTPUT_LINK", f"配置路径含 symlink：{linked}")
    row = _load_yaml(config_path)
    _exact(row, ("schema_version", "benchmark", "output_root", "selection", "rendering"), "$")
    if row["schema_version"] != CONFIG_SCHEMA:
        fail("CONFIG_INVALID", f"仅支持 {CONFIG_SCHEMA}")
    benchmark = _mapping(row["benchmark"], "$.benchmark")
    _exact(benchmark, ("root", *EXPECTED_IDENTITY_FIELDS), "$.benchmark")
    benchmark_root = _absolute(config_path.parent, benchmark.pop("root"), "$.benchmark.root")
    expected = {
        field: _string(benchmark[field], f"$.benchmark.{field}") if field == "schema_version"
        else _sha256(benchmark[field], f"$.benchmark.{field}")
        for field in EXPECTED_IDENTITY_FIELDS
    }
    selection = _mapping(row["selection"], "$.selection")
    _exact(selection, ("sample_count", "seed", "algorithm_version", "source_quotas", "foreground_bins"), "$.selection")
    algorithm = _string(selection["algorithm_version"], "$.selection.algorithm_version")
    if algorithm != SELECTION_ALGORITHM:
        fail("CONFIG_INVALID", f"algorithm_version 必须是 {SELECTION_ALGORITHM}")
    quota_rows = selection["source_quotas"]
    if not isinstance(quota_rows, list) or not quota_rows:
        fail("TYPE_MISMATCH", "source_quotas 必须是非空列表")
    quotas = []
    for index, value in enumerate(quota_rows):
        item = _mapping(value, f"source_quotas[{index}]")
        _exact(item, ("source", "total", "target", "no_target"), f"source_quotas[{index}]")
        quota = SourceQuota(
            _string(item["source"], f"quota[{index}].source"), _integer(item["total"], f"quota[{index}].total"),
            _integer(item["target"], f"quota[{index}].target"), _integer(item["no_target"], f"quota[{index}].no_target"),
        )
        if quota.total != quota.target + quota.no_target:
            fail("CONFIG_INVALID", f"{quota.source}: total != target + no_target")
        quotas.append(quota)
    sample_count = _integer(selection["sample_count"], "$.selection.sample_count", 1)
    if len({item.source for item in quotas}) != len(quotas) or sum(item.total for item in quotas) != sample_count:
        fail("CONFIG_INVALID", "source quotas 重复或总数与 sample_count 不一致")
    bin_rows = selection["foreground_bins"]
    if not isinstance(bin_rows, list) or not bin_rows:
        fail("TYPE_MISMATCH", "foreground_bins 必须是非空列表")
    bins = []
    for index, value in enumerate(bin_rows):
        item = _mapping(value, f"foreground_bins[{index}]")
        _exact(item, ("name", "lower_exclusive", "upper_inclusive"), f"foreground_bins[{index}]")
        lower, upper = _number(item["lower_exclusive"], f"bin[{index}].lower"), _number(item["upper_inclusive"], f"bin[{index}].upper")
        if upper <= lower:
            fail("CONFIG_INVALID", f"foreground_bins[{index}] 边界非法")
        bins.append(ForegroundBin(_string(item["name"], f"bin[{index}].name"), lower, upper))
    if bins[0].lower_exclusive != 0 or bins[-1].upper_inclusive != 1 or any(
        left.upper_inclusive != right.lower_exclusive for left, right in zip(bins, bins[1:])
    ):
        fail("CONFIG_INVALID", "foreground_bins 必须连续覆盖 (0,1]")
    rendering = _mapping(row["rendering"], "$.rendering")
    _exact(rendering, ("context_margin_ratio", "overlay_alpha", "min_auxiliary_coverage"), "$.rendering")
    config = PilotConfig(
        config_path, sha256_file(config_path), "", benchmark_root, expected,
        _absolute(config_path.parent, row["output_root"], "$.output_root"), sample_count,
        _integer(selection["seed"], "$.selection.seed"), algorithm, tuple(quotas), tuple(bins),
        _number(rendering["context_margin_ratio"], "context_margin_ratio"),
        _number(rendering["overlay_alpha"], "overlay_alpha"),
        _number(rendering["min_auxiliary_coverage"], "min_auxiliary_coverage"),
    )
    return replace(config, semantic_sha256=sha256_text(canonical_json(config.semantic_dict())))


@dataclass(frozen=True)
class SilverConfig:
    config_path: Path
    config_file_sha256: str
    semantic_sha256: str
    corpus_root: Path
    corpus_expected: Mapping[str, str]
    gate_b_protocol_config: Path
    gate_b_frozen_protocol: Path
    gate_b_selection: Path
    training_root: Path
    base_run: Path
    adapter_run: Path
    evaluation_root: Path
    expected_frozen_protocol_sha256: str
    expected_gate_b_report_sha256: str
    expected_teacher_artifacts: Mapping[str, str]
    output_root: Path
    seed: int
    require_clean_worktree: bool
    candidates_per_record: int
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    prompt_version: str
    prompt_template_sha256: str
    language: str
    counterfactual_sample_count: int
    wrong_mask_count: int
    modality_removal_count: int
    counterfactual_algorithm: str
    review_count: int
    review_seed: int
    review_algorithm: str
    runtime_expected: Mapping[str, str]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SILVER_CONFIG_SCHEMA,
            "corpus": {"root": str(self.corpus_root), **dict(sorted(self.corpus_expected.items()))},
            "teacher": {
                "gate_b_protocol_config": str(self.gate_b_protocol_config),
                "gate_b_frozen_protocol": str(self.gate_b_frozen_protocol),
                "gate_b_selection": str(self.gate_b_selection),
                "training_root": str(self.training_root),
                "base_run": str(self.base_run),
                "adapter_run": str(self.adapter_run),
                "evaluation_root": str(self.evaluation_root),
                "expected_frozen_protocol_sha256": self.expected_frozen_protocol_sha256,
                "expected_gate_b_report_sha256": self.expected_gate_b_report_sha256,
                **dict(sorted(self.expected_teacher_artifacts.items())),
            },
            "run": {"seed": self.seed, "require_clean_worktree": self.require_clean_worktree},
            "generation": {
                "candidates_per_record": self.candidates_per_record,
                "batch_size": self.batch_size,
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "prompt_version": self.prompt_version,
                "prompt_template_sha256": self.prompt_template_sha256,
                "language": self.language,
            },
            "counterfactual": {
                "sample_count": self.counterfactual_sample_count,
                "wrong_mask_count": self.wrong_mask_count,
                "modality_removal_count": self.modality_removal_count,
                "selection_algorithm": self.counterfactual_algorithm,
            },
            "review": {
                "sample_count": self.review_count,
                "seed": self.review_seed,
                "selection_algorithm": self.review_algorithm,
            },
            "runtime": dict(sorted(self.runtime_expected.items())),
        }


def load_silver_config(path: Path | str) -> SilverConfig:
    config_path = Path(os.path.abspath(Path(path)))
    linked = first_symlink_component(config_path)
    if linked is not None:
        fail("OUTPUT_LINK", f"Silver 配置路径含 symlink：{linked}")
    row = _load_yaml(config_path)
    _exact(row, (
        "schema_version", "corpus", "teacher", "run", "generation",
        "counterfactual", "review", "runtime",
    ), "$")
    if row["schema_version"] != SILVER_CONFIG_SCHEMA:
        fail("CONFIG_INVALID", f"仅支持 {SILVER_CONFIG_SCHEMA}")

    corpus = _mapping(row["corpus"], "$.corpus")
    corpus_fields = (
        "root", "manifest_sha256", "corpus_id", "records_sha256",
        "ledger_file_sha256", "ledger_root_sha256",
    )
    _exact(corpus, corpus_fields, "$.corpus")
    corpus_root = _absolute(config_path.parent, corpus.pop("root"), "$.corpus.root")
    corpus_expected = {
        name: _sha256(corpus[name], f"$.corpus.{name}")
        for name in corpus_fields if name != "root"
    }

    teacher = _mapping(row["teacher"], "$.teacher")
    teacher_path_fields = (
        "gate_b_protocol_config", "gate_b_frozen_protocol", "gate_b_selection",
        "training_root", "base_run", "adapter_run", "evaluation_root",
    )
    teacher_hash_fields = (
        "expected_frozen_protocol_sha256", "expected_gate_b_report_sha256",
        "expected_training_report_sha256", "expected_best_pointer_sha256",
        "expected_checkpoint_manifest_sha256", "expected_adapter_sha256",
        "expected_model_identity_sha256", "expected_processor_identity_sha256",
    )
    _exact(teacher, (*teacher_path_fields, *teacher_hash_fields), "$.teacher")
    teacher_paths = {
        name: _absolute(config_path.parent, teacher[name], f"$.teacher.{name}")
        for name in teacher_path_fields
    }

    run = _mapping(row["run"], "$.run")
    _exact(run, ("seed", "output_root", "require_clean_worktree"), "$.run")
    require_clean = run["require_clean_worktree"]
    if require_clean is not True:
        fail("CONFIG_INVALID", "正式 Silver 配置必须 require_clean_worktree=true")

    generation = _mapping(row["generation"], "$.generation")
    _exact(generation, (
        "candidates_per_record", "batch_size", "max_input_tokens", "max_new_tokens", "do_sample",
        "temperature", "top_p", "prompt_version", "prompt_template_sha256", "language",
    ), "$.generation")
    candidates = _integer(generation["candidates_per_record"], "$.generation.candidates_per_record", 1)
    batch_size = _integer(generation["batch_size"], "$.generation.batch_size", 1)
    max_input_tokens = _integer(generation["max_input_tokens"], "$.generation.max_input_tokens", 1)
    max_new_tokens = _integer(generation["max_new_tokens"], "$.generation.max_new_tokens", 1)
    do_sample = generation["do_sample"]
    temperature = _number(generation["temperature"], "$.generation.temperature")
    top_p = _number(generation["top_p"], "$.generation.top_p")
    prompt_version = _string(generation["prompt_version"], "$.generation.prompt_version")
    prompt_template_sha256 = _sha256(
        generation["prompt_template_sha256"], "$.generation.prompt_template_sha256",
    )
    language = _string(generation["language"], "$.generation.language")
    if (
        candidates != 2 or batch_size != 1 or max_input_tokens != 4096 or max_new_tokens != 256
        or do_sample is not True or temperature != 0.2 or top_p != 0.9
        or prompt_version != SILVER_PROMPT_VERSION or language != "zh-CN"
    ):
        fail("CONFIG_INVALID", "Silver v1 生成参数不符合冻结协议")

    counterfactual = _mapping(row["counterfactual"], "$.counterfactual")
    _exact(counterfactual, (
        "sample_count", "wrong_mask_count", "modality_removal_count", "selection_algorithm",
    ), "$.counterfactual")
    counterfactual_count = _integer(counterfactual["sample_count"], "$.counterfactual.sample_count")
    wrong_count = _integer(counterfactual["wrong_mask_count"], "$.counterfactual.wrong_mask_count")
    removal_count = _integer(counterfactual["modality_removal_count"], "$.counterfactual.modality_removal_count")
    counterfactual_algorithm = _string(counterfactual["selection_algorithm"], "$.counterfactual.selection_algorithm")
    if (
        counterfactual_count != 50 or wrong_count != 25 or removal_count != 25
        or wrong_count + removal_count != counterfactual_count
        or counterfactual_algorithm != SILVER_COUNTERFACTUAL_ALGORITHM
    ):
        fail("CONFIG_INVALID", "Silver v1 counterfactual 参数不符合冻结协议")

    review = _mapping(row["review"], "$.review")
    _exact(review, ("sample_count", "seed", "selection_algorithm"), "$.review")
    review_count = _integer(review["sample_count"], "$.review.sample_count", 1)
    review_algorithm = _string(review["selection_algorithm"], "$.review.selection_algorithm")
    if review_count != 150 or review_algorithm != SILVER_REVIEW_ALGORITHM:
        fail("CONFIG_INVALID", "Silver v1 review 参数不符合冻结协议")

    runtime = _mapping(row["runtime"], "$.runtime")
    runtime_fields = ("python", "torch", "transformers", "peft", "qwen-vl-utils")
    _exact(runtime, runtime_fields, "$.runtime")
    runtime_expected = {
        name: _string(runtime[name], f"$.runtime.{name}") for name in runtime_fields
    }

    config = SilverConfig(
        config_path=config_path,
        config_file_sha256=sha256_file(config_path),
        semantic_sha256="",
        corpus_root=corpus_root,
        corpus_expected=corpus_expected,
        gate_b_protocol_config=teacher_paths["gate_b_protocol_config"],
        gate_b_frozen_protocol=teacher_paths["gate_b_frozen_protocol"],
        gate_b_selection=teacher_paths["gate_b_selection"],
        training_root=teacher_paths["training_root"],
        base_run=teacher_paths["base_run"],
        adapter_run=teacher_paths["adapter_run"],
        evaluation_root=teacher_paths["evaluation_root"],
        expected_frozen_protocol_sha256=_sha256(teacher["expected_frozen_protocol_sha256"], "$.teacher.expected_frozen_protocol_sha256"),
        expected_gate_b_report_sha256=_sha256(teacher["expected_gate_b_report_sha256"], "$.teacher.expected_gate_b_report_sha256"),
        expected_teacher_artifacts={
            name: _sha256(teacher[name], f"$.teacher.{name}")
            for name in teacher_hash_fields
            if name not in {"expected_frozen_protocol_sha256", "expected_gate_b_report_sha256"}
        },
        output_root=_absolute(config_path.parent, run["output_root"], "$.run.output_root"),
        seed=_integer(run["seed"], "$.run.seed"),
        require_clean_worktree=require_clean,
        candidates_per_record=candidates,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        prompt_version=prompt_version,
        prompt_template_sha256=prompt_template_sha256,
        language=language,
        counterfactual_sample_count=counterfactual_count,
        wrong_mask_count=wrong_count,
        modality_removal_count=removal_count,
        counterfactual_algorithm=counterfactual_algorithm,
        review_count=review_count,
        review_seed=_integer(review["seed"], "$.review.seed"),
        review_algorithm=review_algorithm,
        runtime_expected=runtime_expected,
    )
    return replace(config, semantic_sha256=sha256_text(canonical_json(config.semantic_dict())))


RECORD_FIELDS = (
    "schema_version", "record_id", "sample_id", "source", "split", "source_identity", "mask_source",
    "target_state", "declared_modality_signature", "available_modalities", "unavailable_modalities", "assets",
    "program_facts", "allowed_claims", "forbidden_claims", "unavailable_claims", "silver_observation",
    "review_status", "case_kb_eligible", "adapter_training_eligible",
)


def unavailable_auxiliary_facts() -> dict[str, Any]:
    return {"status": "unavailable", "unit": None, "sign_convention": None, "full": None,
            "mask": None, "outside": None, "inside_minus_outside_mean": None}


def no_target_mask_facts() -> dict[str, Any]:
    return {
        "status": "no_target", "bbox_xyxy_pixel_half_open": None, "centroid_xy_pixel": None,
        "area_pixels": 0, "area_ratio": 0.0, "location_3x3": None, "fragment_connectivity": 8,
        "fragment_count": 0, "perimeter_connectivity": 4, "perimeter_pixels": 0,
        "compactness_4pi_area_over_perimeter2": None, "covariance_elongation": None,
        "crop_window_xyxy_pixel_half_open": None,
    }


def derive_claims(*, target_state: str, modality_facts: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    allowed = ["optical_visible_observation", "gt_annotation_target_state", "evidence_sufficiency"]
    if target_state == "target_present":
        allowed += ["mask_geometry", "mask_relative_location", "mask_component_count", "possible_confusers"]
    forbidden = [
        "confirmed_landslide_presence", "event_time", "trigger_cause", "precise_motion_speed",
        "uplift_or_subsidence_direction", "stability", "risk_level", "mask_outside_as_target_evidence",
        "unsupported_modality",
    ]
    unavailable = []
    for modality in SUPPORTED_MODALITIES:
        facts = modality_facts[modality]
        if facts["status"] == "unavailable":
            unavailable.append(f"{modality}_observation")
            continue
        if facts["status"] != "available":
            unavailable.append(f"{modality}_encoded_value_distribution")
            continue
        allowed.append(f"{modality}_encoded_value_distribution")
        if target_state == "target_present" and (facts.get("mask") or {}).get("status") != "available":
            unavailable.append(f"{modality}_region_statistics")
        if modality == "dem":
            unavailable.append("physical_elevation")
        elif modality == "slope":
            unavailable.append("physical_slope")
        else:
            unavailable += ["insar_deformation", "insar_velocity", "insar_direction"]
    return sorted(set(allowed)), sorted(set(forbidden)), sorted(set(unavailable))


def validate_record(value: Any, *, location: str = "record") -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, RECORD_FIELDS, location)
    if row["schema_version"] != RECORD_SCHEMA:
        fail("SCHEMA_MISMATCH", f"{location}.schema_version 非法")
    for field in ("record_id", "sample_id", "source", "split", "mask_source", "target_state"):
        _string(row[field], f"{location}.{field}")
    if row["split"] != "train" or row["target_state"] not in {"target_present", "no_target"}:
        fail("SPLIT_LEAKAGE", f"{location} 必须是 train 且 target state 合法")
    if row["silver_observation"] is not None or row["review_status"] != "not_reviewed":
        fail("SCHEMA_MISMATCH", f"{location} Auto 状态非法")
    if row["case_kb_eligible"] is not False or row["adapter_training_eligible"] is not False:
        fail("SCHEMA_MISMATCH", f"{location} eligibility 必须为 false")
    for field in ("available_modalities", "unavailable_modalities", "allowed_claims", "forbidden_claims", "unavailable_claims"):
        values = row[field]
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values) or len(values) != len(set(values)):
            fail("TYPE_MISMATCH", f"{location}.{field} 必须是唯一字符串列表")
    for field in ("source_identity", "assets", "program_facts"):
        _mapping(row[field], f"{location}.{field}")
    return row


SILVER_FIELDS = (
    "optical_observation", "boundary_clarity", "surface_or_vegetation_disturbance", "terrain_support",
    "possible_confusers", "evidence_sufficiency", "short_summary",
)


@dataclass(frozen=True)
class SilverObservation:
    optical_observation: str
    boundary_clarity: str
    surface_or_vegetation_disturbance: str
    terrain_support: str
    possible_confusers: tuple[str, ...]
    evidence_sufficiency: str
    short_summary: str

    @classmethod
    def from_mapping(cls, value: Any) -> "SilverObservation":
        row = _mapping(value, "silver_observation")
        _exact(row, SILVER_FIELDS, "silver_observation")
        confusers = row["possible_confusers"]
        if not isinstance(confusers, list) or not all(isinstance(item, str) and item.strip() for item in confusers):
            fail("SILVER_INVALID", "possible_confusers 必须是字符串列表")
        choices = {
            "boundary_clarity": {"clear", "partially_clear", "unclear", "not_applicable"},
            "surface_or_vegetation_disturbance": {"observed", "not_observed", "uncertain", "not_applicable"},
            "terrain_support": {"supports", "does_not_support", "inconclusive", "unavailable"},
            "evidence_sufficiency": {"sufficient", "limited", "insufficient"},
        }
        normalized = {field: " ".join(_string(row[field], field).split()) for field in ("optical_observation", "short_summary")}
        for field, allowed in choices.items():
            normalized[field] = _string(row[field], field)
            if normalized[field] not in allowed:
                fail("SILVER_INVALID", f"{field} 不在受控枚举中")
        return cls(
            normalized["optical_observation"], normalized["boundary_clarity"],
            normalized["surface_or_vegetation_disturbance"], normalized["terrain_support"],
            tuple(dict.fromkeys(" ".join(item.split()) for item in confusers)),
            normalized["evidence_sufficiency"], normalized["short_summary"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "optical_observation": self.optical_observation, "boundary_clarity": self.boundary_clarity,
            "surface_or_vegetation_disturbance": self.surface_or_vegetation_disturbance,
            "terrain_support": self.terrain_support, "possible_confusers": list(self.possible_confusers),
            "evidence_sufficiency": self.evidence_sufficiency, "short_summary": self.short_summary,
        }
