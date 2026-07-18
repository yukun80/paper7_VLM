#!/usr/bin/env bash
# 用途：一次性重建 Bridge v7、Unified v3，并严格迁移和只读验证 M3 v3 cache。
# 推荐命令：BRIDGE_REBUILD_CONFIRM=overwrite_auto_only_v7 PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python bash scripts/run_segdesc_artifact_acceptance.sh small
# 完整重建回退：cache build 完成后增加 CACHE_ACTION=verify-existing，只读验收当前原生 M3 v3 cache。
# 输入：Landslide V2、Description v4、旧 M3 v2 cache、只读 segmentation Vision Cache v3；
#       ARTIFACT_SEED 可显式覆盖统一构建 seed（默认 42）。
# 输出：Bridge v7、Unified v3、side-by-side M3 v3 cache 及其 validation/migration reports。
# 写入行为：覆盖当前 auto-only Bridge/Unified；只新建 M3 v3，不修改旧 cache、datasets 或 segmentation cache。
# 所属流程：首次 D-1 前的 M2/M3 artifact 工程验收；不会运行测试、训练或专家 merge。

set -euo pipefail

MODE="${1:-small}"
if [[ "${MODE}" != "small" && "${MODE}" != "full" ]]; then
  echo "用法：bash scripts/run_segdesc_artifact_acceptance.sh [small|full]" >&2
  exit 2
fi
if [[ "${BRIDGE_REBUILD_CONFIRM:-}" != "overwrite_auto_only_v7" ]]; then
  echo "拒绝覆盖 Bridge：请显式设置 BRIDGE_REBUILD_CONFIRM=overwrite_auto_only_v7。" >&2
  echo "该确认只适用于尚未冻结专家数据的 prepare/auto-only artifact。" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_SEED="${ARTIFACT_SEED:-42}"
LANDSLIDE_BENCHMARK="benchmark/multisource_landslide_v2_${MODE}"
DESCRIPTION_BENCHMARK="benchmark/qpsalm_description_v2_${MODE}"
BRIDGE_BENCHMARK="benchmark/landslide_region_description_v1_${MODE}"
UNIFIED_BENCHMARK="benchmark/multisource_landslide_segdesc_v1_${MODE}"
LEGACY_DESCRIPTION_CACHE="${LEGACY_DESCRIPTION_CACHE:-outputs/qpsalm_description/cache/${MODE}_vision_v1}"
DESCRIPTION_CACHE_OUTPUT="${DESCRIPTION_CACHE_OUTPUT:-outputs/qpsalm_description/cache/${MODE}_vision_v1_m3v3}"
READINESS_REPORT="${READINESS_REPORT:-outputs/qpsalm_description/readiness/${MODE}_artifact_readiness.json}"
SEGMENTATION_VISION_CACHE="${SEGMENTATION_VISION_CACHE:-outputs/qpsalm_v2/cache/${MODE}_qwen_psalm_full_qwen_vision_v3}"
SEGMENTATION_CONFIG="${SEGMENTATION_CONFIG:-SEG_Multi-Source_Landslides/configs/qpsalm_v2_${MODE}.yaml}"
CACHE_ACTION="${CACHE_ACTION:-migrate}"
if [[ "${CACHE_ACTION}" != "migrate" && "${CACHE_ACTION}" != "verify-existing" ]]; then
  echo "CACHE_ACTION 只允许 migrate|verify-existing: ${CACHE_ACTION}" >&2
  exit 2
fi

PYTHONPATH=SEG_Multi-Source_Landslides \
"${PYTHON_BIN}" -B -c '
import json
import csv
import sys
from pathlib import Path
from qpsalm_seg.paths import resolve_project_path

root = resolve_project_path(sys.argv[1]) or Path(sys.argv[1])
report_path = root / "reports/validation_report.json"
if report_path.is_file():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expert_records = int((report.get("expert") or {}).get("expert_records", 0))
    if report.get("status") == "expert_pilot_frozen" or expert_records > 0:
        raise SystemExit(
            "拒绝覆盖含专家数据或已冻结的 Bridge；请改用独立输出目录。"
        )

# awaiting report 之外还要保护尚未 merge 的人工文件和被原地填写的模板。
protected_outputs = (
    root / "indexes/expert_all.jsonl",
    root / "indexes/pending_arbitration.jsonl",
    root / "manifests/evaluation_gate_manifest.json",
    root / "reports/expert_review_report.json",
)
if any(path.exists() for path in protected_outputs):
    raise SystemExit(
        "拒绝覆盖含 expert/pending/gate merge 产物的 Bridge；请先人工审计。"
    )
manifests = root / "manifests"
expected_manifests = {
    "evaluation_gate_manifest.template.json",
    "pilot_parent_manifest.jsonl",
    "review_package_manifest.jsonl",
    "review_selection.jsonl",
}
if manifests.is_dir():
    unexpected = sorted(
        path.name for path in manifests.iterdir()
        if path.is_file() and path.name not in expected_manifests
    )
    if unexpected:
        raise SystemExit(
            "拒绝覆盖 manifests 中可能的人工文件: "
            + ", ".join(unexpected[:8])
        )
gate_template = manifests / "evaluation_gate_manifest.template.json"
if gate_template.is_file():
    gate = json.loads(gate_template.read_text(encoding="utf-8"))
    thresholds = gate.get("thresholds")
    if (
        gate.get("frozen") is not False
        or gate.get("status") != "pending_pilot_review"
        or not isinstance(thresholds, dict)
        or any(value is not None for value in thresholds.values())
        or gate.get("note")
        != "Thresholds must be filled and frozen only after completed expert review and Pilot analysis."
    ):
        raise SystemExit(
            "拒绝覆盖已被人工填写或修改的 evaluation gate template。"
        )
package = root / "review_package"
expected_templates = {
    "reviewer_1_template.jsonl",
    "reviewer_2_template.jsonl",
    "reviewer_1_template.csv",
    "reviewer_2_template.csv",
}
if package.is_dir():
    unexpected = sorted(
        path.name for path in package.iterdir()
        if path.is_file() and path.name not in expected_templates
    )
    if unexpected:
        raise SystemExit(
            "拒绝覆盖 review_package 中可能的人工文件: "
            + ", ".join(unexpected[:8])
        )
    review_fields = (
        "decision", "corrected_structured_targets", "revised_summary", "notes",
    )
    for path in sorted(package.glob("reviewer_*_template.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if any(row.get(field) not in (None, "") for field in review_fields):
                    raise SystemExit(
                        f"拒绝覆盖已填写的 reviewer template: {path}:{line_number}"
                    )
    for path in sorted(package.glob("reviewer_*_template.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), 2):
                if any(str(row.get(field) or "").strip() for field in review_fields):
                    raise SystemExit(
                        f"拒绝覆盖已填写的 reviewer template: {path}:{line_number}"
                    )
' "${BRIDGE_BENCHMARK}"

if [[ "${CACHE_ACTION}" == "migrate" ]]; then
  if [[ ! -d "${LEGACY_DESCRIPTION_CACHE}" ]]; then
    echo "旧 M3 v2 cache 不存在：${LEGACY_DESCRIPTION_CACHE}" >&2
    exit 2
  fi
  if [[ -e "${DESCRIPTION_CACHE_OUTPUT}" ]]; then
    echo "M3 v3 输出已存在，严格迁移拒绝覆盖：${DESCRIPTION_CACHE_OUTPUT}" >&2
    echo "请先审计现有目录，或为 DESCRIPTION_CACHE_OUTPUT 指定新的 side-by-side 路径。" >&2
    exit 2
  fi
elif [[ ! -d "${DESCRIPTION_CACHE_OUTPUT}" ]]; then
  echo "待验证的原生 M3 v3 cache 不存在：${DESCRIPTION_CACHE_OUTPUT}" >&2
  exit 2
fi

echo "[SEGDESC-ARTIFACT] mode=${MODE} seed=${ARTIFACT_SEED} full_population=true"
echo "[SEGDESC-ARTIFACT] 1/4 rebuild Bridge v7 prepare (auto-only)"
BRIDGE_STAGE=prepare \
BRIDGE_PILOT_PARENTS=300 \
MAX_SAMPLES=0 \
SEED="${ARTIFACT_SEED}" \
RUN_CONTROL=--overwrite \
PYTHON_BIN="${PYTHON_BIN}" \
SOURCE_BENCHMARK="${LANDSLIDE_BENCHMARK}" \
BRIDGE_OUTPUT_DIR="${BRIDGE_BENCHMARK}" \
bash scripts/run_4_build_landslide_bridge.sh "${MODE}"

echo "[SEGDESC-ARTIFACT] 2/4 rebuild Unified v3 (expert publication remains gated)"
DRY_RUN= \
MAX_SAMPLES=0 \
SEED="${ARTIFACT_SEED}" \
RUN_CONTROL=--overwrite \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_DIR="${UNIFIED_BENCHMARK}" \
bash scripts/run_5_build_segdesc_dataset.sh "${MODE}"

if [[ "${CACHE_ACTION}" == "migrate" ]]; then
  echo "[SEGDESC-ARTIFACT] 3/4 migrate M3 v2 -> v3 with strict content-bound hardlinks"
  PYTHONPATH=SEG_Multi-Source_Landslides \
  "${PYTHON_BIN}" -B -m qpsalm_seg.cli.segdesc cache migrate \
    --config "${SEGMENTATION_CONFIG}" \
    --legacy-cache "${LEGACY_DESCRIPTION_CACHE}" \
    --description-benchmark "${DESCRIPTION_BENCHMARK}" \
    --bridge-benchmark "${BRIDGE_BENCHMARK}" \
    --segmentation-vision-cache "${SEGMENTATION_VISION_CACHE}" \
    --output-dir "${DESCRIPTION_CACHE_OUTPUT}"
else
  echo "[SEGDESC-ARTIFACT] 3/4 use existing native M3 v3 cache (read-only)"
fi

echo "[SEGDESC-ARTIFACT] 4/4 aggregate live artifact readiness and replay all M3 shards"
PYTHONPATH=SEG_Multi-Source_Landslides \
"${PYTHON_BIN}" -B -m qpsalm_seg.cli.segdesc validate artifacts \
  --mode "${MODE}" \
  --description-benchmark "${DESCRIPTION_BENCHMARK}" \
  --bridge-benchmark "${BRIDGE_BENCHMARK}" \
  --unified-benchmark "${UNIFIED_BENCHMARK}" \
  --description-cache "${DESCRIPTION_CACHE_OUTPUT}" \
  --output "${READINESS_REPORT}"

echo "[SEGDESC-ARTIFACT] complete mode=${MODE}"
echo "[SEGDESC-ARTIFACT] Bridge=${BRIDGE_BENCHMARK}/reports/validation_report.json"
echo "[SEGDESC-ARTIFACT] Unified=${UNIFIED_BENCHMARK}/reports/validation_report.json"
echo "[SEGDESC-ARTIFACT] Cache=${DESCRIPTION_CACHE_OUTPUT}/validation_report.json"
echo "[SEGDESC-ARTIFACT] Readiness=${READINESS_REPORT}"
echo "[SEGDESC-ARTIFACT] M2 仍须以实际 report 为准；prepare 不代表专家审核完成。"
