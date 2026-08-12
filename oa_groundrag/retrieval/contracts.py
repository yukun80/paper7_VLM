"""Stage 6 的严格配置、来源、证据与路由合同。

本模块只定义可重算身份和确定性边界。RAG 不得修改 mask、程序事实或
Pass-1 视觉观察，也不得把候选区域升级成确认滑坡。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    require_bool,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    resolve_config_path,
    safe_join,
)
from oa_groundrag.vlm.config import _load_yaml
from oa_groundrag.vlm.errors import ConfigError, ContractError, ReasonCode


SOURCE_REGISTRY_SCHEMA = "oa_groundrag.text_rag.source_registry.v1"
PAGE_SCHEMA = "oa_groundrag.text_rag.source_page.v1"
EVIDENCE_UNIT_SCHEMA = "oa_groundrag.text_rag.evidence_unit.v1"
BANK_MANIFEST_SCHEMA = "oa_groundrag.text_rag.bank_manifest.v1"
QUERY_SCHEMA = "oa_groundrag.text_rag.query.v1"
PACKET_SCHEMA = "oa_groundrag.text_rag.packet.v1"
SELECTION_SCHEMA = "oa_groundrag.text_rag.dev_selection.v1"
RETRIEVAL_REPORT_SCHEMA = "oa_groundrag.text_rag.retrieval_report.v1"
PASS2_OUTPUT_SCHEMA = "oa_groundrag.text_rag.pass2_output.v1"
PASS2_PREDICTION_SCHEMA = "oa_groundrag.text_rag.pass2_prediction.v1"
PASS2_FAILURE_SCHEMA = "oa_groundrag.text_rag.pass2_failure.v1"
PASS2_RUN_SCHEMA = "oa_groundrag.text_rag.pass2_run.v1"
STAGE6_CONFIG_SCHEMA = "oa_groundrag.text_rag.config.v1"

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_ALLOWED_MODALITIES = frozenset({"general", "optical", "sar", "insar", "dem"})
_ALLOWED_LANGUAGES = frozenset({"zh", "en", "zh_en"})
_ALLOWED_PROFILES = frozenset({"standard", "paper", "handbook", "slides"})
_ALLOWED_STATUSES = frozenset({"current", "published", "historical", "unverified", "excluded"})


class KnowledgeType(StrEnum):
    INTERPRETATION = "interpretation"
    CONFOUNDER = "confounder"
    LIMITATION = "limitation"
    UNCLASSIFIED = "unclassified"


class QueryIntent(StrEnum):
    INTERPRETATION = "interpretation"
    COUNTER_LIMITATION = "counter_limitation"


class TextRagTask(StrEnum):
    SCENE_DESCRIPTION = "scene_description"
    MASK_GROUNDED_REGION_DESCRIPTION = "mask_grounded_region_description"
    CANDIDATE_INTERPRETATION = "candidate_interpretation"
    PROFESSIONAL_QA = "professional_qa"
    EVIDENCE_CONSTRAINED_REPORT = "evidence_constrained_report"


class RagMode(StrEnum):
    NO_RAG = "no_rag"
    TEXT_RAG = "text_rag"


def route_text_rag(task: TextRagTask | str) -> bool:
    """由显式任务枚举决定是否检索；未知任务拒绝，绝不做文本猜测。"""

    try:
        value = task if isinstance(task, TextRagTask) else TextRagTask(task)
    except (TypeError, ValueError) as error:
        raise ContractError(ReasonCode.INVALID_ENUM, f"未注册 Stage 6 task：{task!r}") from error
    return value in {
        TextRagTask.CANDIDATE_INTERPRETATION,
        TextRagTask.PROFESSIONAL_QA,
        TextRagTask.EVIDENCE_CONSTRAINED_REPORT,
    }


def _sha(value: Any, *, location: str) -> str:
    text = require_string(value, location=location)
    if not _SHA_RE.fullmatch(text):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是小写 SHA-256")
    return text


def _float(value: Any, *, location: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是数值")
    result = float(value)
    if result < minimum:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须 >= {minimum}")
    return result


def _string_list(
    value: Any,
    *,
    location: str,
    allowed: frozenset[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是列表")
    result = tuple(require_string(item, location=f"{location}[]") for item in value)
    if not allow_empty and not result:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 不得为空")
    if len(result) != len(set(result)):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 不允许重复")
    if allowed is not None and not set(result) <= allowed:
        raise ConfigError(
            ReasonCode.INVALID_ENUM,
            f"{location}: 非法值 {sorted(set(result) - allowed)}",
        )
    return result


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    title: str
    source_kind: str
    authority_class: str
    source_status: str
    publication_year: int | None
    standard_number: str | None
    language: str
    relative_path: str
    expected_sha256: str
    parser_profile: str
    default_modalities: tuple[str, ...]
    retrieval_enabled: bool
    source_path: Path

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_kind": self.source_kind,
            "authority_class": self.authority_class,
            "source_status": self.source_status,
            "publication_year": self.publication_year,
            "standard_number": self.standard_number,
            "language": self.language,
            "relative_path": self.relative_path,
            "source_sha256": self.expected_sha256,
            "parser_profile": self.parser_profile,
            "default_modalities": list(self.default_modalities),
            "retrieval_enabled": self.retrieval_enabled,
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_text(canonical_json(self.semantic_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "source_identity_sha256": self.identity_sha256}


@dataclass(frozen=True)
class SourceRegistry:
    config_path: Path
    source_root: Path
    sources: tuple[SourceEntry, ...]
    semantic_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_REGISTRY_SCHEMA,
            "source_root": str(self.source_root),
            "source_count": len(self.sources),
            "sources": [source.to_dict() for source in self.sources],
            "registry_semantic_sha256": self.semantic_sha256,
        }


def load_source_registry(path: Path | str, *, verify_files: bool = True) -> SourceRegistry:
    config_path = Path(os.path.abspath(Path(path)))
    if (
        first_symlink_component(config_path) is not None
        or not config_path.is_file()
        or config_path.is_symlink()
        or config_path.stat().st_nlink != 1
    ):
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"source registry 必须是普通单链接文件：{config_path}")
    row = _load_yaml(config_path)
    require_exact_keys(
        row,
        required=("schema_version", "source_root", "sources"),
        location="$",
    )
    if row["schema_version"] != SOURCE_REGISTRY_SCHEMA:
        raise ConfigError(ReasonCode.INVALID_ENUM, f"仅支持 {SOURCE_REGISTRY_SCHEMA}")
    source_root = resolve_config_path(config_path.parent, row["source_root"], location="$.source_root")
    if first_symlink_component(source_root) is not None or not source_root.is_dir() or source_root.is_symlink():
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"source_root 必须是普通目录：{source_root}")
    values = row["sources"]
    if not isinstance(values, list) or not values:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "$.sources 必须是非空列表")
    sources: list[SourceEntry] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    fields = (
        "source_id", "title", "source_kind", "authority_class", "source_status",
        "publication_year", "standard_number", "language", "relative_path",
        "expected_sha256", "parser_profile", "default_modalities", "retrieval_enabled",
    )
    for index, value in enumerate(values):
        location = f"$.sources[{index}]"
        child = require_mapping(value, location=location)
        require_exact_keys(child, required=fields, location=location)
        source_id = require_string(child["source_id"], location=f"{location}.source_id")
        if not _SOURCE_ID_RE.fullmatch(source_id):
            raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}.source_id 格式非法")
        if source_id in seen_ids:
            raise ConfigError(ReasonCode.TYPE_MISMATCH, f"重复 source_id：{source_id}")
        seen_ids.add(source_id)
        relative_path = require_string(child["relative_path"], location=f"{location}.relative_path")
        source_path = safe_join(source_root, relative_path, location=f"{location}.relative_path")
        canonical_relative = source_path.relative_to(source_root).as_posix()
        if canonical_relative in seen_paths:
            raise ConfigError(ReasonCode.TYPE_MISMATCH, f"重复 source path：{relative_path}")
        seen_paths.add(canonical_relative)
        publication_year = child["publication_year"]
        if publication_year is not None:
            publication_year = require_int(
                publication_year,
                location=f"{location}.publication_year",
                minimum=1800,
            )
            if publication_year > 2100:
                raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}.publication_year 非法")
        standard_number = child["standard_number"]
        if standard_number is not None:
            standard_number = require_string(standard_number, location=f"{location}.standard_number")
        language = require_string(child["language"], location=f"{location}.language")
        profile = require_string(child["parser_profile"], location=f"{location}.parser_profile")
        status = require_string(child["source_status"], location=f"{location}.source_status")
        if language not in _ALLOWED_LANGUAGES or profile not in _ALLOWED_PROFILES or status not in _ALLOWED_STATUSES:
            raise ConfigError(ReasonCode.INVALID_ENUM, f"{location}: language/profile/status 非法")
        expected_sha = _sha(child["expected_sha256"], location=f"{location}.expected_sha256")
        if verify_files:
            if (
                first_symlink_component(source_path) is not None
                or not source_path.is_file()
                or source_path.is_symlink()
                or source_path.stat().st_nlink != 1
            ):
                raise ConfigError(ReasonCode.ASSET_MISSING, f"知识源必须是普通单链接文件：{source_path}")
            actual_sha = sha256_file(source_path)
            if actual_sha != expected_sha:
                raise ConfigError(
                    ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                    f"知识源 SHA 漂移：{source_path}",
                    details={"expected": expected_sha, "actual": actual_sha},
                )
        source = SourceEntry(
            source_id=source_id,
            title=require_string(child["title"], location=f"{location}.title"),
            source_kind=require_string(child["source_kind"], location=f"{location}.source_kind"),
            authority_class=require_string(child["authority_class"], location=f"{location}.authority_class"),
            source_status=status,
            publication_year=publication_year,
            standard_number=standard_number,
            language=language,
            relative_path=canonical_relative,
            expected_sha256=expected_sha,
            parser_profile=profile,
            default_modalities=_string_list(
                child["default_modalities"],
                location=f"{location}.default_modalities",
                allowed=_ALLOWED_MODALITIES,
            ),
            retrieval_enabled=require_bool(child["retrieval_enabled"], location=f"{location}.retrieval_enabled"),
            source_path=source_path,
        )
        if source.source_status == "excluded" and source.retrieval_enabled:
            raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: excluded source 不得启用检索")
        sources.append(source)
    semantic_payload = {
        "schema_version": SOURCE_REGISTRY_SCHEMA,
        "sources": [source.semantic_dict() for source in sources],
    }
    return SourceRegistry(
        config_path=config_path,
        source_root=source_root,
        sources=tuple(sources),
        semantic_sha256=sha256_text(canonical_json(semantic_payload)),
    )


@dataclass(frozen=True)
class ExtractionConfig:
    text_min_useful_chars: int
    ocr_min_useful_chars: int
    max_bad_char_ratio: float
    render_dpi: int
    min_unit_chars: int
    max_unit_tokens: int
    ocr_enabled: bool


@dataclass(frozen=True)
class DenseConfig:
    repo_id: str
    revision: str
    model_root: Path
    device: str
    batch_size: int
    max_tokens: int


@dataclass(frozen=True)
class RetrievalConfig:
    lexical_k: int
    dense_k: int
    rrf_k: int
    interpretation_quota: int
    confounder_quota: int
    limitation_quota: int


@dataclass(frozen=True)
class Stage5Binding:
    config_path: Path
    best_pointer_sha256: str
    checkpoint_manifest_sha256: str
    adapter_sha256: str
    workflow_state_sha256: str


@dataclass(frozen=True)
class DevBinding:
    eval_root: Path
    eval_manifest_sha256: str
    prediction_root: Path
    prediction_manifest_sha256: str
    predictions_sha256: str
    question: str
    task: TextRagTask
    expected_records: int
    smoke_limit: int


@dataclass(frozen=True)
class Stage6Config:
    config_path: Path
    source_registry_path: Path
    bank_root: Path
    retrieval_root: Path
    generation_root: Path
    extraction: ExtractionConfig
    dense: DenseConfig
    retrieval: RetrievalConfig
    stage5: Stage5Binding
    dev: DevBinding
    semantic_sha256: str
    source_registry_identity_path: Path | None = None
    stage5_config_identity_path: Path | None = None

    def semantic_dict(self) -> dict[str, Any]:
        """返回 v1 scientific identity；运行路径迁移不改写已发布身份。"""

        source_registry_identity = (
            self.source_registry_identity_path or self.source_registry_path
        )
        stage5_config_identity = (
            self.stage5_config_identity_path or self.stage5.config_path
        )
        return {
            "schema_version": STAGE6_CONFIG_SCHEMA,
            "source_registry": str(source_registry_identity),
            "outputs": {
                "bank_root": str(self.bank_root),
                "retrieval_root": str(self.retrieval_root),
                "generation_root": str(self.generation_root),
            },
            "extraction": self.extraction.__dict__,
            "dense": {**self.dense.__dict__, "model_root": str(self.dense.model_root)},
            "retrieval": self.retrieval.__dict__,
            "stage5": {
                **self.stage5.__dict__,
                "config_path": str(stage5_config_identity),
            },
            "dev": {
                **self.dev.__dict__,
                "eval_root": str(self.dev.eval_root),
                "prediction_root": str(self.dev.prediction_root),
                "task": self.dev.task.value,
            },
        }


def load_stage6_config(path: Path | str) -> Stage6Config:
    config_path = Path(os.path.abspath(Path(path)))
    if first_symlink_component(config_path) is not None or not config_path.is_file() or config_path.is_symlink():
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"Stage 6 配置必须是普通文件：{config_path}")
    row = _load_yaml(config_path)
    require_exact_keys(
        row,
        required=("schema_version", "source_registry", "outputs", "extraction", "dense", "retrieval", "stage5", "dev"),
        location="$",
    )
    if row["schema_version"] != STAGE6_CONFIG_SCHEMA:
        raise ConfigError(ReasonCode.INVALID_ENUM, f"仅支持 {STAGE6_CONFIG_SCHEMA}")
    base = config_path.parent
    outputs = require_mapping(row["outputs"], location="$.outputs")
    require_exact_keys(outputs, required=("bank_root", "retrieval_root", "generation_root"), location="$.outputs")
    extraction_row = require_mapping(row["extraction"], location="$.extraction")
    require_exact_keys(
        extraction_row,
        required=("text_min_useful_chars", "ocr_min_useful_chars", "max_bad_char_ratio", "render_dpi", "min_unit_chars", "max_unit_tokens", "ocr_enabled"),
        location="$.extraction",
    )
    extraction = ExtractionConfig(
        text_min_useful_chars=require_int(extraction_row["text_min_useful_chars"], location="$.extraction.text_min_useful_chars", minimum=1),
        ocr_min_useful_chars=require_int(extraction_row["ocr_min_useful_chars"], location="$.extraction.ocr_min_useful_chars", minimum=1),
        max_bad_char_ratio=_float(extraction_row["max_bad_char_ratio"], location="$.extraction.max_bad_char_ratio"),
        render_dpi=require_int(extraction_row["render_dpi"], location="$.extraction.render_dpi", minimum=72),
        min_unit_chars=require_int(extraction_row["min_unit_chars"], location="$.extraction.min_unit_chars", minimum=1),
        max_unit_tokens=require_int(extraction_row["max_unit_tokens"], location="$.extraction.max_unit_tokens", minimum=64),
        ocr_enabled=require_bool(extraction_row["ocr_enabled"], location="$.extraction.ocr_enabled"),
    )
    if extraction.max_bad_char_ratio > 0.2:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "$.extraction.max_bad_char_ratio 过大")
    dense_row = require_mapping(row["dense"], location="$.dense")
    require_exact_keys(dense_row, required=("repo_id", "revision", "model_root", "device", "batch_size", "max_tokens"), location="$.dense")
    revision = require_string(dense_row["revision"], location="$.dense.revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "$.dense.revision 必须是完整 Hub commit")
    device = require_string(dense_row["device"], location="$.dense.device")
    if device not in {"cpu", "cuda"}:
        raise ConfigError(ReasonCode.INVALID_ENUM, "$.dense.device 只允许 cpu/cuda")
    dense = DenseConfig(
        repo_id=require_string(dense_row["repo_id"], location="$.dense.repo_id"),
        revision=revision,
        model_root=resolve_config_path(base, dense_row["model_root"], location="$.dense.model_root"),
        device=device,
        batch_size=require_int(dense_row["batch_size"], location="$.dense.batch_size", minimum=1),
        max_tokens=require_int(dense_row["max_tokens"], location="$.dense.max_tokens", minimum=64),
    )
    if dense.max_tokens != extraction.max_unit_tokens:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "dense/extraction token 上限必须一致")
    retrieval_row = require_mapping(row["retrieval"], location="$.retrieval")
    require_exact_keys(
        retrieval_row,
        required=("lexical_k", "dense_k", "rrf_k", "interpretation_quota", "confounder_quota", "limitation_quota"),
        location="$.retrieval",
    )
    retrieval = RetrievalConfig(**{
        key: require_int(retrieval_row[key], location=f"$.retrieval.{key}", minimum=1)
        for key in retrieval_row
    })
    stage5_row = require_mapping(row["stage5"], location="$.stage5")
    require_exact_keys(
        stage5_row,
        required=("config", "best_pointer_sha256", "checkpoint_manifest_sha256", "adapter_sha256", "workflow_state_sha256"),
        location="$.stage5",
    )
    stage5 = Stage5Binding(
        config_path=resolve_config_path(base, stage5_row["config"], location="$.stage5.config"),
        best_pointer_sha256=_sha(stage5_row["best_pointer_sha256"], location="$.stage5.best_pointer_sha256"),
        checkpoint_manifest_sha256=_sha(stage5_row["checkpoint_manifest_sha256"], location="$.stage5.checkpoint_manifest_sha256"),
        adapter_sha256=_sha(stage5_row["adapter_sha256"], location="$.stage5.adapter_sha256"),
        workflow_state_sha256=_sha(stage5_row["workflow_state_sha256"], location="$.stage5.workflow_state_sha256"),
    )
    dev_row = require_mapping(row["dev"], location="$.dev")
    require_exact_keys(
        dev_row,
        required=("eval_root", "eval_manifest_sha256", "prediction_root", "prediction_manifest_sha256", "predictions_sha256", "question", "task", "expected_records", "smoke_limit"),
        location="$.dev",
    )
    try:
        task = TextRagTask(require_string(dev_row["task"], location="$.dev.task"))
    except ValueError as error:
        raise ConfigError(ReasonCode.INVALID_ENUM, "$.dev.task 非法") from error
    if not route_text_rag(task):
        raise ConfigError(ReasonCode.RAG_FORBIDDEN, "Stage 6 dev task 必须启用 RAG")
    dev = DevBinding(
        eval_root=resolve_config_path(base, dev_row["eval_root"], location="$.dev.eval_root"),
        eval_manifest_sha256=_sha(dev_row["eval_manifest_sha256"], location="$.dev.eval_manifest_sha256"),
        prediction_root=resolve_config_path(base, dev_row["prediction_root"], location="$.dev.prediction_root"),
        prediction_manifest_sha256=_sha(dev_row["prediction_manifest_sha256"], location="$.dev.prediction_manifest_sha256"),
        predictions_sha256=_sha(dev_row["predictions_sha256"], location="$.dev.predictions_sha256"),
        question=require_string(dev_row["question"], location="$.dev.question"),
        task=task,
        expected_records=require_int(dev_row["expected_records"], location="$.dev.expected_records", minimum=1),
        smoke_limit=require_int(dev_row["smoke_limit"], location="$.dev.smoke_limit", minimum=1),
    )
    if dev.smoke_limit > dev.expected_records:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "smoke_limit 不得超过 expected_records")
    source_registry_path = resolve_config_path(
        base,
        row["source_registry"],
        location="$.source_registry",
    )
    source_registry_identity_path: Path | None = None
    stage5_config_identity_path: Path | None = None
    if base.name == "retrieval" and base.parent.name == "configs":
        repository_root = base.parent.parent
        source_registry_identity_path = (
            repository_root
            / "configs/stage6_text_rag"
            / source_registry_path.name
        )
        stage5_config_identity_path = (
            repository_root
            / "configs/phase4_rs_vlm"
            / stage5.config_path.name
        )
    provisional = Stage6Config(
        config_path=config_path,
        source_registry_path=source_registry_path,
        bank_root=resolve_config_path(base, outputs["bank_root"], location="$.outputs.bank_root"),
        retrieval_root=resolve_config_path(base, outputs["retrieval_root"], location="$.outputs.retrieval_root"),
        generation_root=resolve_config_path(base, outputs["generation_root"], location="$.outputs.generation_root"),
        extraction=extraction,
        dense=dense,
        retrieval=retrieval,
        stage5=stage5,
        dev=dev,
        semantic_sha256="",
        source_registry_identity_path=source_registry_identity_path,
        stage5_config_identity_path=stage5_config_identity_path,
    )
    semantic = sha256_text(canonical_json(provisional.semantic_dict()))
    return Stage6Config(**{**provisional.__dict__, "semantic_sha256": semantic})


def evidence_unit_id(row: Mapping[str, Any]) -> str:
    payload = {
        "source_id": row["source_id"],
        "source_sha256": row["source_sha256"],
        "pdf_page": row["pdf_page"],
        "section": row["section"],
        "clause": row["clause"],
        "knowledge_type": row["knowledge_type"],
        "content_sha256": row["content_sha256"],
    }
    return f"ev_{sha256_text(canonical_json(payload))}"


def query_identity(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key not in {"schema_version", "query_id"}}
    return f"qry_{sha256_text(canonical_json(payload))}"


def packet_identity(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key not in {"schema_version", "packet_id"}}
    return f"pkt_{sha256_text(canonical_json(payload))}"


def strict_contract_mapping(
    value: Any,
    *,
    fields: Sequence[str],
    location: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是对象")
    row = dict(value)
    expected = set(fields)
    if set(row) != expected:
        raise ContractError(
            ReasonCode.UNKNOWN_FIELD,
            f"{location}: 字段不匹配",
            details={"missing": sorted(expected - set(row)), "unknown": sorted(set(row) - expected)},
        )
    return row
