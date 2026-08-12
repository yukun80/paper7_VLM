"""Text Evidence Bank 的 PDF/OCR 构建、分类、索引与严格验证。"""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.errors import ContractError, ModelError, ReasonCode

from .contracts import (
    BANK_MANIFEST_SCHEMA,
    EVIDENCE_UNIT_SCHEMA,
    PAGE_SCHEMA,
    KnowledgeType,
    SourceEntry,
    Stage6Config,
    evidence_unit_id,
    load_source_registry,
)
from .retrieval import BGEM3DenseEmbedder, build_lexical_index


_BANK_PAYLOAD_FILES = frozenset({
    "normalized_registry.json",
    "source_pages.jsonl",
    "evidence_units.jsonl",
    "lexical.sqlite3",
    "dense_embeddings.npy",
    "dense_index.json",
    "environment.json",
    "source_report.json",
})
_BANK_ALL_FILES = _BANK_PAYLOAD_FILES | {"SHA256SUMS.jsonl", "manifest.json"}
_HEADING_RE = re.compile(
    r"^(?P<clause>(?:\d+(?:\.\d+){0,5}|[一二三四五六七八九十百]+)[、.．]?)[ \t]+(?P<title>\S.{0,180})$"
)
_SENTENCE_RE = re.compile(r"(?<=[。！？!?;；.])\s+")

_LIMITATION_KEYWORDS = (
    "限制", "局限", "不能", "无法", "不足", "误差", "精度", "分辨率", "单时相",
    "低相干", "失相干", "相干性", "解缠", "大气延迟", "大气误差", "叠掩", "阴影",
    "视线向", "los", "limitation", "uncertainty", "cannot", "insufficient", "resolution",
    "decorrelation", "coherence", "unwrapping", "atmospheric", "layover", "shadow",
    "line-of-sight", "ambiguity", "error source",
)
_CONFOUNDER_KEYWORDS = (
    "道路切坡", "道路边坡", "采石场", "矿山", "裸岩", "河滩", "冲沟", "施工扰动",
    "人工开挖", "梯田", "露天矿", "弃渣", "quarry", "mine", "road cut", "bare rock",
    "river terrace", "riverbed", "gully", "construction", "terrace", "excavation",
)
_INTERPRETATION_KEYWORDS = (
    "滑坡", "崩塌", "地表形态", "遥感特征", "识别标志", "后缘", "前缘", "滑坡壁",
    "陡坎", "拉张裂缝", "滑坡舌", "鼓丘", "色调", "纹理", "植被异常", "边界",
    "landslide", "slope movement", "morphology", "scarp", "crown", "toe", "hummocky",
    "debris slide", "earth flow", "rock fall", "remote sensing", "texture", "tone", "vegetation",
)
_CONDITION_MARKERS = (
    "当", "如果", "若", "在", "条件", "适用于", "前提", "when ", "if ", "under ",
    "provided that", "applicable",
)
_TAG_KEYWORDS = tuple(dict.fromkeys(
    _LIMITATION_KEYWORDS + _CONFOUNDER_KEYWORDS + _INTERPRETATION_KEYWORDS
))


def normalize_extracted_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\x00", "")
    lines = [" ".join(line.split()) for line in value.replace("\r", "\n").split("\n")]
    output: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if output and not blank:
                output.append("")
            blank = True
        else:
            output.append(line)
            blank = False
    return "\n".join(output).strip()


def page_quality(text: str) -> dict[str, Any]:
    normalized = normalize_extracted_text(text)
    useful = sum(
        char.isalnum() or "\u3400" <= char <= "\u9fff"
        for char in normalized
    )
    bad = sum(
        char == "\ufffd"
        or (unicodedata.category(char) in {"Cc", "Co"} and char not in "\n\t")
        for char in normalized
    )
    denominator = max(1, len(normalized))
    return {
        "useful_char_count": int(useful),
        "bad_char_count": int(bad),
        "bad_char_ratio": round(bad / denominator, 8),
        "text_length": len(normalized),
    }


def _quality_usable(metrics: Mapping[str, Any], *, minimum: int, max_bad_ratio: float) -> bool:
    return (
        int(metrics["useful_char_count"]) >= minimum
        and float(metrics["bad_char_ratio"]) <= max_bad_ratio
    )


def _rapidocr_identity() -> dict[str, Any]:
    spec = importlib.util.find_spec("rapidocr")
    if spec is None or spec.origin is None:
        raise ModelError(ReasonCode.ASSET_MISSING, "未安装 rapidocr")
    root = Path(spec.origin).resolve().parent
    models: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.onnx")):
        if path.is_symlink() or not path.is_file():
            raise ModelError(ReasonCode.OUTPUT_LINK, f"RapidOCR model 非普通文件：{path}")
        models[path.relative_to(root).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "backend": "rapidocr",
        "rapidocr_version": importlib.metadata.version("rapidocr"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "package_root": str(root),
        "models": models,
        "models_identity_sha256": sha256_text(canonical_json(models)),
    }


class RapidOCRAdapter:
    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as error:
            raise ModelError(ReasonCode.ASSET_MISSING, "OCR 需要 rapidocr") from error
        self.engine = RapidOCR()
        self.identity = _rapidocr_identity()

    def extract(self, image: np.ndarray) -> str:
        result = self.engine(image)
        texts: Sequence[Any] | None = getattr(result, "txts", None)
        if texts is None and isinstance(result, tuple) and result:
            rows = result[0]
            if rows:
                texts = [row[1] for row in rows if isinstance(row, (list, tuple)) and len(row) >= 2]
        if texts is None:
            return ""
        return normalize_extracted_text("\n".join(str(value) for value in texts if str(value).strip()))


def _pdf_text_blocks(page: Any) -> str:
    blocks = page.get_text("blocks", sort=True)
    values = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = normalize_extracted_text(str(block[4]))
        if text:
            values.append(text)
    return "\n\n".join(values)


def audit_sources(
    config: Stage6Config,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pymupdf as fitz
    except ImportError as error:
        raise ModelError(ReasonCode.ASSET_MISSING, "PDF 审计需要 PyMuPDF") from error
    registry = load_source_registry(config.source_registry_path)
    ocr: RapidOCRAdapter | None = None
    pages: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    method_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for source in registry.sources:
        try:
            document = fitz.open(source.source_path)
        except Exception as error:
            raise ContractError(
                ReasonCode.ASSET_MISSING,
                f"无法打开 PDF：{source.source_path}",
                details={"error": str(error)},
            ) from error
        try:
            page_counts[source.source_id] = document.page_count
            if progress is not None:
                progress({"event": "source_started", "source_id": source.source_id, "page_count": document.page_count})
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text_layer = _pdf_text_blocks(page)
                text_metrics = page_quality(text_layer)
                attempts = [{
                    "method": "text_layer",
                    "status": (
                        "usable"
                        if _quality_usable(
                            text_metrics,
                            minimum=config.extraction.text_min_useful_chars,
                            max_bad_ratio=config.extraction.max_bad_char_ratio,
                        )
                        else "low_quality"
                    ),
                    "metrics": text_metrics,
                    "text_sha256": sha256_text(text_layer),
                }]
                accepted = ""
                method = "none"
                status = "ocr_required"
                if attempts[0]["status"] == "usable":
                    accepted = text_layer
                    method = "text_layer"
                    status = "usable"
                elif config.extraction.ocr_enabled:
                    if ocr is None:
                        ocr = RapidOCRAdapter()
                    pixmap = page.get_pixmap(dpi=config.extraction.render_dpi, alpha=False)
                    channels = pixmap.n
                    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, channels)
                    try:
                        ocr_text = ocr.extract(image)
                        ocr_metrics = page_quality(ocr_text)
                        ocr_usable = _quality_usable(
                            ocr_metrics,
                            minimum=config.extraction.ocr_min_useful_chars,
                            max_bad_ratio=config.extraction.max_bad_char_ratio,
                        )
                        attempts.append({
                            "method": "rapidocr",
                            "status": "usable" if ocr_usable else "low_quality",
                            "metrics": ocr_metrics,
                            "text_sha256": sha256_text(ocr_text),
                        })
                        if ocr_usable:
                            accepted = ocr_text
                            method = "rapidocr"
                            status = "usable"
                        else:
                            status = "ocr_failed"
                    except Exception as error:
                        attempts.append({
                            "method": "rapidocr",
                            "status": "failed",
                            "metrics": None,
                            "text_sha256": None,
                            "error": f"{type(error).__name__}: {error}",
                        })
                        status = "ocr_failed"
                metrics = page_quality(accepted)
                row = {
                    "schema_version": PAGE_SCHEMA,
                    "source_id": source.source_id,
                    "source_sha256": source.expected_sha256,
                    "pdf_page": page_index + 1,
                    "page_width": round(float(page.rect.width), 4),
                    "page_height": round(float(page.rect.height), 4),
                    "extraction_attempts": attempts,
                    "extraction_method": method,
                    "quality_status": status,
                    "useful_char_count": metrics["useful_char_count"],
                    "bad_char_ratio": metrics["bad_char_ratio"],
                    "text": accepted,
                    "text_sha256": sha256_text(accepted),
                }
                pages.append(row)
                method_counts[method] += 1
                status_counts[status] += 1
            if progress is not None:
                progress({"event": "source_complete", "source_id": source.source_id, "page_count": document.page_count})
        finally:
            document.close()
    environment = {
        "pymupdf_version": importlib.metadata.version("PyMuPDF"),
        "ocr": None if ocr is None else ocr.identity,
        "page_counts": page_counts,
        "method_counts": dict(sorted(method_counts.items())),
        "quality_status_counts": dict(sorted(status_counts.items())),
    }
    return pages, environment


def _paragraph_blocks(text: str, profile: str) -> list[tuple[str | None, str | None, str]]:
    lines = text.splitlines()
    output: list[tuple[str | None, str | None, str]] = []
    section: str | None = None
    clause: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        content = " ".join(buffer).strip()
        if content:
            output.append((section, clause, content))
        buffer = []

    for line in lines:
        value = " ".join(line.split())
        if not value:
            flush()
            continue
        heading = _HEADING_RE.match(value)
        # OCR 中文行通常没有句末标点，不能据此把整页误判成标题。
        # 数字条款由 _HEADING_RE 处理；这里只保留明确全大写的英文短标题。
        short_heading = len(value) <= 120 and value.isupper()
        if heading:
            flush()
            clause = heading.group("clause")
            section = value
            continue
        if short_heading and len(value.split()) <= 14:
            flush()
            section = value
            clause = None
            continue
        buffer.append(value)
        if profile == "slides" and len(buffer) >= 8:
            flush()
    flush()
    return output


def _split_long_content(
    content: str,
    *,
    token_count: Callable[[str], int],
    max_tokens: int,
) -> tuple[str, ...]:
    if token_count(content) <= max_tokens:
        return (content,)
    sentences = [value.strip() for value in _SENTENCE_RE.split(content) if value.strip()]
    if len(sentences) <= 1:
        words = content.split()
        if len(words) <= 1:
            midpoint = max(1, len(content) // 2)
            return (
                *_split_long_content(content[:midpoint], token_count=token_count, max_tokens=max_tokens),
                *_split_long_content(content[midpoint:], token_count=token_count, max_tokens=max_tokens),
            )
        sentences = words
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join((*current, sentence)).strip()
        if current and token_count(candidate) > max_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    output: list[str] = []
    for chunk in chunks:
        if token_count(chunk) <= max_tokens:
            output.append(chunk)
        elif len(chunk) < len(content):
            output.extend(_split_long_content(chunk, token_count=token_count, max_tokens=max_tokens))
        else:
            raise ContractError(ReasonCode.TOKEN_LIMIT_EXCEEDED, "无法在自然边界内拆分 Evidence Unit")
    return tuple(output)


def split_page_units(
    text: str,
    *,
    parser_profile: str,
    token_count: Callable[[str], int],
    min_chars: int,
    max_tokens: int,
) -> tuple[dict[str, Any], ...]:
    units: list[dict[str, Any]] = []
    for section, clause, content in _paragraph_blocks(normalize_extracted_text(text), parser_profile):
        for chunk in _split_long_content(content, token_count=token_count, max_tokens=max_tokens):
            useful = page_quality(chunk)["useful_char_count"]
            if useful >= min_chars:
                units.append({"section": section, "clause": clause, "content": chunk})
    return tuple(units)


def classify_unit(content: str) -> tuple[KnowledgeType, dict[str, Any]]:
    lowered = content.lower()
    limitation = tuple(keyword for keyword in _LIMITATION_KEYWORDS if keyword in lowered)
    confounder = tuple(keyword for keyword in _CONFOUNDER_KEYWORDS if keyword in lowered)
    interpretation = tuple(keyword for keyword in _INTERPRETATION_KEYWORDS if keyword in lowered)
    if limitation:
        value = KnowledgeType.LIMITATION
        matched = limitation
    elif confounder:
        value = KnowledgeType.CONFOUNDER
        matched = confounder
    elif interpretation:
        value = KnowledgeType.INTERPRETATION
        matched = interpretation
    else:
        value = KnowledgeType.UNCLASSIFIED
        matched = ()
    return value, {
        "method": "deterministic_keyword_priority.v1",
        "priority": ["limitation", "confounder", "interpretation"],
        "matched_rule_ids": [f"kw:{keyword}" for keyword in matched],
    }


def _modalities(content: str, source: SourceEntry) -> tuple[str, ...]:
    lowered = content.lower()
    detected: set[str] = set()
    if any(value in lowered for value in ("insar", "干涉", "相干", "解缠", "los", "视线向")):
        detected.add("insar")
    if any(value in lowered for value in ("sar", "雷达", "microwave", "微波")):
        detected.add("sar")
    if any(value in lowered for value in ("dem", "高程", "坡度", "terrain model", "elevation")):
        detected.add("dem")
    if any(value in lowered for value in ("光学", "可见光", "影像色调", "影像纹理", "optical", "visible image")):
        detected.add("optical")
    if not detected:
        detected.update(source.default_modalities)
    return tuple(sorted(detected))


def _conditions(content: str) -> tuple[str, ...]:
    sentences = [value.strip() for value in re.split(r"(?<=[。！？!?;；.])", content) if value.strip()]
    selected = [sentence for sentence in sentences if any(marker in sentence.lower() for marker in _CONDITION_MARKERS)]
    return tuple(dict.fromkeys(selected[:8]))


def _tags(content: str) -> tuple[str, ...]:
    lowered = content.lower()
    return tuple(keyword for keyword in _TAG_KEYWORDS if keyword in lowered)[:32]


def build_evidence_units(
    *,
    pages: Sequence[Mapping[str, Any]],
    sources: Sequence[SourceEntry],
    token_count: Callable[[str], int],
    min_chars: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    source_by_id = {source.source_id: source for source in sources}
    units: list[dict[str, Any]] = []
    for page in pages:
        if page["quality_status"] != "usable":
            continue
        source = source_by_id[str(page["source_id"])]
        for value in split_page_units(
            str(page["text"]),
            parser_profile=source.parser_profile,
            token_count=token_count,
            min_chars=min_chars,
            max_tokens=max_tokens,
        ):
            content = value["content"]
            knowledge_type, provenance = classify_unit(content)
            content_sha = sha256_text(content)
            row = {
                "schema_version": EVIDENCE_UNIT_SCHEMA,
                "unit_id": "",
                "source_id": source.source_id,
                "source_title": source.title,
                "source_sha256": source.expected_sha256,
                "pdf_page": int(page["pdf_page"]),
                "section": value["section"],
                "clause": value["clause"],
                "knowledge_type": knowledge_type.value,
                "content": content,
                "applicable_modalities": list(_modalities(content, source)),
                "conditions": list(_conditions(content)),
                "tags": list(_tags(content)),
                "authority_class": source.authority_class,
                "source_status": source.source_status,
                "extraction_method": page["extraction_method"],
                "classification_provenance": provenance,
                "content_sha256": content_sha,
                "duplicate_of": None,
                "indexed": source.retrieval_enabled and knowledge_type is not KnowledgeType.UNCLASSIFIED,
            }
            row["unit_id"] = evidence_unit_id(row)
            units.append(row)
    canonical_by_key: dict[tuple[str, str], str] = {}
    for unit in sorted(units, key=lambda row: row["unit_id"]):
        key = (str(unit["knowledge_type"]), str(unit["content_sha256"]))
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = str(unit["unit_id"])
        else:
            unit["duplicate_of"] = canonical
            unit["indexed"] = False
    return sorted(units, key=lambda row: row["unit_id"])


def _ledger_rows(root: Path, files: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(files):
        path = root / relative
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _write_environment(
    *,
    extraction_environment: Mapping[str, Any],
    dense_identity: Mapping[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        fts5_available = bool(connection.execute(
            "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
        ).fetchone()[0])
    finally:
        connection.close()
    return {
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "sqlite": sqlite3.sqlite_version,
        "fts5_available": fts5_available,
        "extraction": dict(extraction_environment),
        "dense": dict(dense_identity),
    }


def build_bank(
    config: Stage6Config,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    registry = load_source_registry(config.source_registry_path)
    if config.bank_root.exists() or config.bank_root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"Bank 已存在：{config.bank_root}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ModelError(ReasonCode.ASSET_MISSING, "Evidence Unit 切分需要本地 BGE tokenizer") from error
    tokenizer = AutoTokenizer.from_pretrained(
        config.dense.model_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_count = lambda text: len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    pages, extraction_environment = audit_sources(config, progress=progress)
    units = build_evidence_units(
        pages=pages,
        sources=registry.sources,
        token_count=token_count,
        min_chars=config.extraction.min_unit_chars,
        max_tokens=config.extraction.max_unit_tokens,
    )
    per_source_units = Counter(str(unit["source_id"]) for unit in units)
    empty_sources = sorted(source.source_id for source in registry.sources if per_source_units[source.source_id] == 0)
    if empty_sources:
        raise ContractError(ReasonCode.ASSET_MISSING, f"知识源没有可用 Evidence Unit：{empty_sources}")
    indexed = [unit for unit in units if unit["indexed"]]
    type_counts = Counter(str(unit["knowledge_type"]) for unit in indexed)
    missing_types = [value.value for value in KnowledgeType if value is not KnowledgeType.UNCLASSIFIED and not type_counts[value.value]]
    if missing_types:
        raise ContractError(ReasonCode.ASSET_MISSING, f"Bank 缺少核心 knowledge type：{missing_types}")
    embedder = BGEM3DenseEmbedder(
        config.dense.model_root,
        repo_id=config.dense.repo_id,
        revision=config.dense.revision,
        device=config.dense.device,
        batch_size=config.dense.batch_size,
        max_tokens=config.dense.max_tokens,
    )
    dense_order = [str(unit["unit_id"]) for unit in indexed]
    embeddings = embedder.encode([str(unit["content"]) for unit in indexed])
    environment = _write_environment(
        extraction_environment=extraction_environment,
        dense_identity=embedder.identity,
    )
    source_report = {
        "source_count": len(registry.sources),
        "page_count": len(pages),
        "usable_page_count": sum(row["quality_status"] == "usable" for row in pages),
        "ocr_page_count": sum(row["extraction_method"] == "rapidocr" for row in pages),
        "failed_page_count": sum(row["quality_status"] != "usable" for row in pages),
        "unit_count": len(units),
        "indexed_unit_count": len(indexed),
        "knowledge_type_counts": dict(sorted(type_counts.items())),
        "unclassified_count": sum(unit["knowledge_type"] == KnowledgeType.UNCLASSIFIED.value for unit in units),
        "exact_duplicate_count": sum(unit["duplicate_of"] is not None for unit in units),
        "per_source": {
            source.source_id: {
                "pages": sum(page["source_id"] == source.source_id for page in pages),
                "usable_pages": sum(page["source_id"] == source.source_id and page["quality_status"] == "usable" for page in pages),
                "units": per_source_units[source.source_id],
                "indexed_units": sum(unit["source_id"] == source.source_id and unit["indexed"] for unit in units),
            }
            for source in registry.sources
        },
    }
    with AtomicArtifactDirectory(config.bank_root) as writer:
        writer.write_json("normalized_registry.json", registry.to_dict())
        writer.write_jsonl("source_pages.jsonl", pages)
        writer.write_jsonl("evidence_units.jsonl", units)
        build_lexical_index(writer.path("lexical.sqlite3"), units)
        with writer.path("dense_embeddings.npy").open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        dense_index = {
            "schema_version": "oa_groundrag.text_rag.dense_index.v1",
            "model_identity": dict(embedder.identity),
            "embedding_dimension": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "unit_ids": dense_order,
            "unit_order_sha256": sha256_text(canonical_json(dense_order)),
        }
        writer.write_json("dense_index.json", dense_index)
        writer.write_json("environment.json", environment)
        writer.write_json("source_report.json", source_report)
        assert writer.staging is not None
        payload_hashes = {relative: sha256_file(writer.path(relative)) for relative in sorted(_BANK_PAYLOAD_FILES)}
        bank_id = sha256_text(canonical_json({
            "config_semantic_sha256": config.semantic_sha256,
            "registry_semantic_sha256": registry.semantic_sha256,
            "payload_hashes": payload_hashes,
        }))
        writer.write_jsonl("SHA256SUMS.jsonl", _ledger_rows(writer.staging, _BANK_PAYLOAD_FILES))
        writer.write_json("manifest.json", {
            "schema_version": BANK_MANIFEST_SCHEMA,
            "bank_id": bank_id,
            "config_semantic_sha256": config.semantic_sha256,
            "registry_semantic_sha256": registry.semantic_sha256,
            "source_count": len(registry.sources),
            "page_count": len(pages),
            "unit_count": len(units),
            "indexed_unit_count": len(indexed),
            "knowledge_type_counts": dict(sorted(type_counts.items())),
            "dense_model_identity_sha256": sha256_text(canonical_json(embedder.identity)),
            "payload_hashes": payload_hashes,
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "development_only": True,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_accessed": False,
        })
        root = writer.publish()
    return validate_bank(root, config=config, verify_sources=True)


_UNIT_FIELDS = {
    "schema_version", "unit_id", "source_id", "source_title", "source_sha256", "pdf_page",
    "section", "clause", "knowledge_type", "content", "applicable_modalities", "conditions", "tags",
    "authority_class", "source_status", "extraction_method", "classification_provenance",
    "content_sha256", "duplicate_of", "indexed",
}


def _validate_unit(unit: Mapping[str, Any], sources: Mapping[str, SourceEntry]) -> None:
    if set(unit) != _UNIT_FIELDS or unit.get("schema_version") != EVIDENCE_UNIT_SCHEMA:
        raise ContractError(ReasonCode.UNKNOWN_FIELD, "Evidence Unit 字段/schema 不匹配")
    source = sources.get(str(unit["source_id"]))
    if source is None or unit["source_sha256"] != source.expected_sha256 or unit["source_title"] != source.title:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Evidence Unit source identity 不一致")
    if isinstance(unit["pdf_page"], bool) or not isinstance(unit["pdf_page"], int) or unit["pdf_page"] < 1:
        raise ContractError(ReasonCode.TYPE_MISMATCH, "Evidence Unit pdf_page 非法")
    if not isinstance(unit["content"], str) or not unit["content"].strip():
        raise ContractError(ReasonCode.TYPE_MISMATCH, "Evidence Unit content 非法")
    if sha256_text(unit["content"]) != unit["content_sha256"] or evidence_unit_id(unit) != unit["unit_id"]:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Evidence Unit content/id 无法重算")
    try:
        knowledge_type = KnowledgeType(unit["knowledge_type"])
    except ValueError as error:
        raise ContractError(ReasonCode.INVALID_ENUM, "Evidence Unit knowledge_type 非法") from error
    if not isinstance(unit["indexed"], bool) or (unit["indexed"] and knowledge_type is KnowledgeType.UNCLASSIFIED):
        raise ContractError(ReasonCode.TYPE_MISMATCH, "Evidence Unit indexed 状态非法")
    for field in ("applicable_modalities", "conditions", "tags"):
        value = unit[field]
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value) or len(value) != len(set(value)):
            raise ContractError(ReasonCode.TYPE_MISMATCH, f"Evidence Unit {field} 非法")


def validate_bank(
    root: Path | str,
    *,
    config: Stage6Config,
    verify_sources: bool = True,
) -> dict[str, Any]:
    root = Path(os.path.abspath(Path(root)))
    linked = first_symlink_component(root)
    if linked is not None or not root.is_dir() or root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_LINK, f"Bank root 非普通目录：{root}")
    linked_payload = next((path for path in root.rglob("*") if path.is_symlink()), None)
    if linked_payload is not None:
        raise ContractError(ReasonCode.OUTPUT_LINK, f"Bank 含链接：{linked_payload}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != _BANK_ALL_FILES:
        raise ContractError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "Bank 文件集不匹配",
            details={"missing": sorted(_BANK_ALL_FILES - actual), "unexpected": sorted(actual - _BANK_ALL_FILES)},
        )
    ledger = read_jsonl(root / "SHA256SUMS.jsonl")
    ledger_map = {row.get("path"): row for row in ledger}
    if set(ledger_map) != _BANK_PAYLOAD_FILES or len(ledger_map) != len(ledger):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Bank ledger 路径集非法")
    for relative, row in ledger_map.items():
        path = root / str(relative)
        if set(row) != {"path", "size_bytes", "sha256"} or path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, f"Bank ledger 漂移：{relative}")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != BANK_MANIFEST_SCHEMA or manifest.get("sealed_test_accessed") is not False:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Bank manifest schema/boundary 非法")
    if manifest.get("ledger_sha256") != sha256_file(root / "SHA256SUMS.jsonl"):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Bank ledger identity 漂移")
    registry = load_source_registry(config.source_registry_path, verify_files=verify_sources)
    normalized_registry = read_json(root / "normalized_registry.json")
    if normalized_registry != registry.to_dict():
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Bank normalized registry 漂移")
    sources = {source.source_id: source for source in registry.sources}
    pages = read_jsonl(root / "source_pages.jsonl")
    page_keys: set[tuple[str, int]] = set()
    for page in pages:
        required = {
            "schema_version", "source_id", "source_sha256", "pdf_page", "page_width", "page_height",
            "extraction_attempts", "extraction_method", "quality_status", "useful_char_count",
            "bad_char_ratio", "text", "text_sha256",
        }
        if set(page) != required or page.get("schema_version") != PAGE_SCHEMA:
            raise ContractError(ReasonCode.UNKNOWN_FIELD, "source page 字段/schema 非法")
        source = sources.get(str(page["source_id"]))
        key = (str(page["source_id"]), int(page["pdf_page"]))
        if source is None or key in page_keys or page["source_sha256"] != source.expected_sha256:
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "source page identity/duplicate 非法")
        page_keys.add(key)
        if sha256_text(str(page["text"])) != page["text_sha256"]:
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "source page text SHA 漂移")
    units = read_jsonl(root / "evidence_units.jsonl")
    unit_by_id: dict[str, Mapping[str, Any]] = {}
    canonical_by_key: dict[tuple[str, str], str] = {}
    for unit in units:
        _validate_unit(unit, sources)
        unit_id = str(unit["unit_id"])
        if unit_id in unit_by_id or (str(unit["source_id"]), int(unit["pdf_page"])) not in page_keys:
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Evidence Unit duplicate/page traceability 非法")
        unit_by_id[unit_id] = unit
        key = (str(unit["knowledge_type"]), str(unit["content_sha256"]))
        duplicate_of = unit["duplicate_of"]
        if key not in canonical_by_key:
            canonical_by_key[key] = unit_id
            if duplicate_of is not None:
                raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "首个 exact content 不得是 duplicate")
        elif duplicate_of != canonical_by_key[key] or unit["indexed"]:
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "exact duplicate 关系非法")
    dense_index = read_json(root / "dense_index.json")
    dense_ids = dense_index.get("unit_ids")
    expected_ids = [str(unit["unit_id"]) for unit in units if unit["indexed"]]
    if dense_ids != expected_ids or dense_index.get("unit_order_sha256") != sha256_text(canonical_json(expected_ids)):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "dense unit order 漂移")
    with (root / "dense_embeddings.npy").open("rb") as handle:
        embeddings = np.load(handle, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape != (len(expected_ids), dense_index.get("embedding_dimension")) or embeddings.dtype != np.float32:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "dense matrix shape/dtype 非法")
    if not np.isfinite(embeddings).all() or (len(embeddings) and not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "dense matrix 非有限或未归一化")
    connection = sqlite3.connect(f"file:{root / 'lexical.sqlite3'}?mode=ro", uri=True)
    try:
        indexed_rows = connection.execute("SELECT unit_id FROM evidence ORDER BY unit_id").fetchall()
        fts_count = int(connection.execute("SELECT count(*) FROM evidence_fts").fetchone()[0])
    finally:
        connection.close()
    if [row[0] for row in indexed_rows] != sorted(expected_ids) or fts_count != len(expected_ids):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "FTS/unit set 不一致")
    payload_hashes = {relative: sha256_file(root / relative) for relative in sorted(_BANK_PAYLOAD_FILES)}
    bank_id = sha256_text(canonical_json({
        "config_semantic_sha256": config.semantic_sha256,
        "registry_semantic_sha256": registry.semantic_sha256,
        "payload_hashes": payload_hashes,
    }))
    if manifest.get("bank_id") != bank_id or manifest.get("payload_hashes") != payload_hashes:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Bank identity 无法重算")
    report = read_json(root / "source_report.json")
    if report.get("source_count") != len(registry.sources) or report.get("page_count") != len(pages) or report.get("unit_count") != len(units):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Bank source report count 漂移")
    return {
        "ok": True,
        "root": str(root),
        "bank_id": bank_id,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "ledger_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
        "source_count": len(registry.sources),
        "page_count": len(pages),
        "unit_count": len(units),
        "indexed_unit_count": len(expected_ids),
        "knowledge_type_counts": manifest["knowledge_type_counts"],
        "sealed_test_accessed": False,
    }
