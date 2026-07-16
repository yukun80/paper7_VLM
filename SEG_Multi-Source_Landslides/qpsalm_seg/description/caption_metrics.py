"""Official post-hoc caption metrics for a frozen RSIEval generation set."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable

from qpsalm_seg.paths import resolve_project_path

from .json_protocol import strict_json_loads
from .metrics import bootstrap_mean_ci


CAPTION_METRICS_PROTOCOL = "qpsalm_rsieval_caption_metrics_v1_official_backends"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        strict_json_loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def caption_metric_population(
    eval_dir: str | Path,
    *,
    source_dataset: str = "RSIEval",
    expected_samples: int | None = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind metrics to a complete independent global-caption evaluation."""
    # 延迟导入避免 package export 与 evaluator 形成初始化环。
    from .evaluator import (
        DESCRIPTION_EVALUATION_PROTOCOL,
        EVALUATION_POPULATION_FIELDS,
        evaluation_population_sha256,
        revalidate_evaluation_publication,
    )

    root = resolve_project_path(eval_dir) or Path(eval_dir)
    report_path = root / "eval_report.json"
    generation_path = root / "raw_generations.jsonl"
    if not report_path.is_file() or not generation_path.is_file():
        raise FileNotFoundError(
            f"caption metrics 缺少 eval_report/raw_generations: {root}"
        )
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL
        or report.get("stage") != "rsicap_caption"
        or report.get("split") != "test"
    ):
        raise ValueError(
            "正式 RSIEval caption metrics 要求当前 evaluation protocol、"
            "stage=rsicap_caption、split=test"
        )
    limit = report.get("evaluation_limit_audit") or {}
    if (
        limit.get("protocol") != "qpsalm_description_evaluation_limit_v1"
        or int(limit.get("requested_max_samples", -1)) != 0
        or limit.get("full_population_requested") is not True
        or int(limit.get("dataset_rows_evaluated", -1))
        != int(report.get("num_samples", -1))
    ):
        raise ValueError(
            "正式 RSIEval caption metrics 要求 --max-val-samples 0"
        )
    coverage = report.get("generation_coverage") or {}
    if coverage.get("complete") is not True:
        raise ValueError(
            "正式 caption metrics 要求完整 frozen generation；"
            "请使用 --max-val-samples 0 --max-generate-samples 0"
        )
    if coverage.get("population_identity_fields") != list(EVALUATION_POPULATION_FIELDS):
        raise ValueError(
            "caption eval population identity fields 与当前协议不一致；"
            "必须重新生成包含 reference_texts 的正式报告"
        )
    publication_audit = revalidate_evaluation_publication(root, report)
    all_rows = _read_jsonl(generation_path)
    observed_population_sha256 = evaluation_population_sha256(all_rows)
    if coverage.get("population_sha256") != observed_population_sha256:
        raise ValueError("caption eval report 与 raw_generations population hash 不一致")
    expected_total = int(coverage.get("eligible_samples") or 0)
    generated_total = int(coverage.get("generated_samples") or 0)
    if (
        expected_total != len(all_rows)
        or generated_total != len(all_rows)
        or int(report.get("num_samples") or 0) != len(all_rows)
        or int(report.get("num_generated") or 0) != len(all_rows)
    ):
        raise ValueError(
            "caption eval complete 标志与 raw generation 行数不一致: "
            f"eligible={expected_total} generated={generated_total} "
            f"report={report.get('num_generated')} rows={len(all_rows)}"
        )
    source_filter = report.get("source_filter_audit") or {}
    if (
        source_filter.get("protocol")
        != "qpsalm_description_evaluation_source_filter_v1"
        or source_filter.get("stage") != "rsicap_caption"
        or source_filter.get("split") != "test"
        or source_filter.get("source_dataset") != str(source_dataset)
        or int(source_filter.get("rows_after_filter") or 0) != len(all_rows)
    ):
        raise ValueError(
            "正式 RSIEval caption metrics 要求从 DataLoader 起冻结 RSIEval-only source filter"
        )
    if any(
        str(row.get("source_dataset") or "") != str(source_dataset)
        or str(row.get("task_family") or "") != "global_caption"
        for row in all_rows
    ):
        raise ValueError("RSIEval-only eval 中混入其他 source/task rows")
    rows = list(all_rows)
    if not rows:
        raise ValueError(f"eval 中没有 source_dataset={source_dataset!r} global captions")
    if any(str(row.get("split") or "") != "test" for row in rows):
        raise ValueError("正式 RSIEval caption metric row 必须全部属于 test split")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("caption metric population 要求非空且唯一 sample_id")
    parent_ids = [str(row.get("parent_sample_id") or "") for row in rows]
    if any(not value for value in parent_ids) or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("正式 RSIEval caption population 要求每个 parent 恰好一条 generation")
    if expected_samples is not None and len(rows) != int(expected_samples):
        raise ValueError(
            f"正式 {source_dataset} caption population 应为 {int(expected_samples)} 条，"
            f"实际 {len(rows)} 条"
        )
    for row in rows:
        references = list(dict.fromkeys(
            str(value).strip() for value in row.get("reference_texts", [])
            if str(value).strip()
        ))
        if not str(row.get("raw_generation") or "").strip() or not any(references):
            raise ValueError(
                f"caption metric row 缺少 prediction/reference: {row.get('sample_id')}"
            )
        row["reference_texts"] = references
    rows.sort(key=lambda value: str(value["sample_id"]))
    identities = [
        {
            "sample_id": str(row["sample_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "prediction_sha256": hashlib.sha256(
                str(row["raw_generation"]).encode("utf-8")
            ).hexdigest(),
            "reference_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in row["reference_texts"]
            ],
        }
        for row in rows
    ]
    encoded = json.dumps(
        identities,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return rows, {
        "eval_dir": str(root.resolve(strict=False)),
        "eval_report": str(report_path.resolve(strict=False)),
        "eval_report_sha256": _sha256(report_path),
        "raw_generations": str(generation_path.resolve(strict=False)),
        "raw_generations_sha256": _sha256(generation_path),
        "evaluation_publication_audit": publication_audit,
        "evaluation_population_sha256": coverage.get("population_sha256"),
        "metric_population_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_dataset": str(source_dataset),
        "expected_samples": expected_samples,
        "num_samples": len(rows),
        "num_parents": len({str(row["parent_sample_id"]) for row in rows}),
    }


def _metric_summary(
    corpus_score: float,
    sample_scores: Iterable[float],
    *,
    seed: int,
) -> dict[str, Any]:
    values = [float(value) for value in sample_scores]
    if not values:
        raise ValueError("caption metric backend 返回空 per-sample scores")
    return {
        "corpus_score": float(corpus_score),
        "num_samples": len(values),
        "parent_macro": sum(values) / len(values),
        "parent_macro_bootstrap_95ci": bootstrap_mean_ci(
            values, seed=int(seed), samples=10000
        ),
    }


def _pycoco_metrics(
    rows: list[dict[str, Any]], *, seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("正式 METEOR/SPICE 需要可执行 Java runtime")
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
    except ImportError as exc:
        raise RuntimeError(
            "正式 BLEU/METEOR/ROUGE/CIDEr/SPICE 需要 pycocoevalcap；"
            "请安装项目的 caption-eval optional dependencies"
        ) from exc
    ground_truth = {
        str(index): list(row["reference_texts"])
        for index, row in enumerate(rows)
    }
    results = {
        str(index): [str(row["raw_generation"])]
        for index, row in enumerate(rows)
    }
    output: dict[str, Any] = {}
    backend = {
        "bleu": f"{Bleu.__module__}.{Bleu.__name__}",
        "meteor": f"{Meteor.__module__}.{Meteor.__name__}",
        "rouge": f"{Rouge.__module__}.{Rouge.__name__}",
        "cider": f"{Cider.__module__}.{Cider.__name__}",
        "spice": f"{Spice.__module__}.{Spice.__name__}",
    }
    source_hashes = {}
    for scorer in (Bleu, Meteor, Rouge, Cider, Spice):
        source_path = Path(inspect.getfile(scorer))
        source_hashes[f"{scorer.__module__}.{scorer.__name__}"] = {
            "path": str(source_path.resolve(strict=False)),
            "sha256": _sha256(source_path),
        }
    package_root = Path(inspect.getfile(Spice)).resolve(strict=False).parents[1]
    jar_hashes = {
        candidate.relative_to(package_root).as_posix(): _sha256(candidate)
        for candidate in sorted(package_root.rglob("*.jar"))
        if candidate.is_file()
    }
    if not jar_hashes:
        raise RuntimeError(
            "pycocoevalcap 未发现 METEOR/SPICE JAR resources；"
            "请按该包说明完成 Stanford CoreNLP 资源准备"
        )
    java_result = subprocess.run(
        [java, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if java_result.returncode != 0:
        raise RuntimeError(f"Java runtime 检查失败: {java_result.stderr.strip()}")
    java_path = Path(java).resolve(strict=False)
    backend["source_file_audit"] = source_hashes
    backend["resource_jar_sha256"] = jar_hashes
    backend["java"] = {
        "executable": str(java_path),
        "executable_sha256": _sha256(java_path),
        "version": (java_result.stderr or java_result.stdout).strip(),
    }
    bleu_score, bleu_samples = Bleu(4).compute_score(ground_truth, results)
    for order in range(4):
        output[f"BLEU-{order + 1}"] = _metric_summary(
            float(bleu_score[order]),
            bleu_samples[order],
            seed=seed + 1009 * (order + 1),
        )
    for name, scorer, offset in (
        ("METEOR", Meteor(), 5003),
        ("ROUGE-L", Rouge(), 7001),
        ("CIDEr", Cider(), 9001),
    ):
        score, samples = scorer.compute_score(ground_truth, results)
        output[name] = _metric_summary(score, samples, seed=seed + offset)
    spice_score, spice_samples = Spice().compute_score(ground_truth, results)
    spice_values = [
        float((value.get("All") or {}).get("f") or 0.0)
        for value in spice_samples
    ]
    output["SPICE"] = _metric_summary(
        spice_score, spice_values, seed=seed + 11003
    )
    return output, backend


def _bertscore_model_audit(path_ref: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_project_path(path_ref) or Path(path_ref)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            "BERTScore 必须使用显式本地 Hugging Face 模型目录（含 config.json）: "
            f"{path}"
        )
    binding_files: dict[str, dict[str, Any]] = {}
    weight_count = 0
    for candidate in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = candidate.relative_to(path).as_posix()
        if candidate.suffix in {".safetensors", ".bin", ".pt"}:
            weight_count += 1
        # 绑定 encoder 权重、配置和 tokenizer；忽略缓存锁与说明文件。
        if (
            candidate.suffix in {
                ".json", ".safetensors", ".bin", ".pt", ".txt", ".model"
            }
            or candidate.name in {"merges.txt", "sentencepiece.bpe.model", "spiece.model"}
        ):
            stat = candidate.stat()
            binding_files[relative] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256(candidate),
            }
    if weight_count == 0:
        raise FileNotFoundError(f"BERTScore 本地模型目录没有权重文件: {path}")
    encoded = json.dumps(
        binding_files,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return path, {
        "model_dir": str(path.resolve(strict=False)),
        "binding_files": binding_files,
        "model_snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _assert_model_snapshot_unchanged(
    path: Path, audit: dict[str, Any]
) -> None:
    for relative, expected in (audit.get("binding_files") or {}).items():
        candidate = path / relative
        if not candidate.is_file():
            raise RuntimeError(f"BERTScore 模型在 scoring 期间删除文件: {relative}")
        stat = candidate.stat()
        if (
            int(stat.st_size) != int(expected["size"])
            or int(stat.st_mtime_ns) != int(expected["mtime_ns"])
        ):
            raise RuntimeError(f"BERTScore 模型在 scoring 期间发生变化: {relative}")


def _bertscore(
    rows: list[dict[str, Any]],
    *,
    model_path: str | Path,
    num_layers: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "正式 BERTScore 需要 bert-score；"
            "请安装项目的 caption-eval optional dependencies"
        ) from exc
    model_dir, audit = _bertscore_model_audit(model_path)
    if int(num_layers) <= 0 or int(batch_size) <= 0:
        raise ValueError("BERTScore num_layers/batch_size 必须为正整数")
    candidates = []
    references = []
    owners = []
    for index, row in enumerate(rows):
        for reference in row["reference_texts"]:
            candidates.append(str(row["raw_generation"]))
            references.append(str(reference))
            owners.append(index)
    _precision, _recall, f1 = bert_score(
        candidates,
        references,
        model_type=str(model_dir),
        num_layers=int(num_layers),
        batch_size=int(batch_size),
        device=str(device),
        verbose=False,
        rescale_with_baseline=False,
    )
    best = [float("-inf")] * len(rows)
    for owner, value in zip(owners, f1.detach().cpu().tolist()):
        best[owner] = max(best[owner], float(value))
    if any(value == float("-inf") for value in best):
        raise RuntimeError("BERTScore 未返回完整 sample/reference 对")
    _assert_model_snapshot_unchanged(model_dir, audit)
    audit["num_layers"] = int(num_layers)
    audit["batch_size"] = int(batch_size)
    return _metric_summary(sum(best) / len(best), best, seed=seed + 13001), audit


def score_caption_metrics(
    eval_dir: str | Path,
    *,
    bertscore_model: str | Path,
    bertscore_num_layers: int,
    bertscore_batch_size: int,
    device: str,
    seed: int,
    source_dataset: str = "RSIEval",
) -> dict[str, Any]:
    if str(source_dataset) != "RSIEval":
        raise ValueError(
            "qpsalm_rsieval_caption_metrics 只接受 source_dataset=RSIEval"
        )
    rows, input_audit = caption_metric_population(
        eval_dir, source_dataset=source_dataset, expected_samples=100
    )
    metrics, backends = _pycoco_metrics(rows, seed=seed)
    bertscore, bertscore_audit = _bertscore(
        rows,
        model_path=bertscore_model,
        num_layers=bertscore_num_layers,
        batch_size=bertscore_batch_size,
        device=device,
        seed=seed,
    )
    metrics["BERTScore-F1"] = bertscore
    return {
        "protocol": CAPTION_METRICS_PROTOCOL,
        "status": "engineering-valid",
        "role": "secondary_language_metrics",
        "region_grounding_claimed": False,
        "input_audit": input_audit,
        "backend_audit": {
            "pycocoevalcap": backends,
            "package_versions": {
                "pycocoevalcap": importlib_metadata.version("pycocoevalcap"),
                "bert-score": importlib_metadata.version("bert-score"),
            },
            "bertscore_model": bertscore_audit,
        },
        "statistics": {
            "unit": "image_parent",
            "bootstrap_samples": 10000,
            "confidence": 0.95,
            "seed": int(seed),
        },
        "metrics": metrics,
        "errors": [],
    }


def write_caption_metric_report(path: str | Path, report: dict[str, Any]) -> None:
    target = resolve_project_path(path) or Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
