"""Unified Demo 的严格双语表示层；不参与模型 prompt 或科学数据合同。"""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence

from oa_groundrag.runtime.contracts import RegionSource, UnifiedTask


DEFAULT_LOCALE = "zh"
SUPPORTED_LOCALES = ("zh", "en")

RUN_MODE_SINGLE = "SINGLE_TASK"
RUN_MODE_SUITE = "TASK_SUITE"
SOURCE_FILTER_ALL = "__ALL_SOURCES__"


class DemoI18nError(RuntimeError):
    """翻译 key、locale 或格式化参数不满足严格合同。"""


TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "app.document_title": "OA-GroundRAG 统一演示工作台",
        "app.header": (
            "# OA-GroundRAG 统一演示工作台\n"
            "**只读推理工作台。** 基准数据、检查点、适配器、文本证据库和正式输出"
            "均不修改；唯一持久写入为独立定性样本库、测试访问凭据与演示推理产物。"
        ),
        "app.language.label": "界面语言 / Interface Language",
        "tab.browser": "A. 基准数据浏览",
        "tab.gallery": "B. Demo Gallery",
        "tab.runner": "C. 任务运行 / 结果查看 / 执行轨迹",
        "browser.intro": (
            "### Demo / 定性探索\n"
            "筛选与随机选择只读取规范索引和人工条件，不读取模型得分、不做 Top-K。"
        ),
        "browser.split.label": "数据划分",
        "browser.source.label": "数据源",
        "browser.target.label": "目标状态",
        "browser.size.label": "目标尺寸",
        "browser.modalities.label": "可用模态（所选模态必须全部存在）",
        "browser.sample_query.label": "sample_id 筛选 / 精确定位",
        "browser.current_sample.label": "当前 sample_id",
        "browser.apply": "应用筛选",
        "browser.previous": "上一条",
        "browser.next": "下一条",
        "browser.random": "随机样本",
        "browser.locate": "指定 sample_id",
        "browser.metadata.label": "样本元数据",
        "browser.optical.label": "完整光学影像 / RGB 预览",
        "browser.reference.accordion": "Reference / Audit Only（绝不作为 USER_MASK）",
        "browser.reference.label": "Reference / Audit Only",
        "browser.optical_channels.label": "光学 / 多光谱逐通道预览（确定性渲染）",
        "browser.auxiliary.label": (
            "空间专家输入预览——辅助模态 | 当前 P0 中不是 MLLM 正式 Grounded 输入"
        ),
        "browser.channel_metadata.label": "通道数值 / 有效性 / 显示变换",
        "browser.spatial_inputs.label": "空间专家输入",
        "browser.formal_inputs.label": "MLLM 正式 Grounded 输入",
        "gallery.intro": (
            "### 定性 Demo Gallery\n"
            "仅用于人工定性选择并保留独立修订记录；不是金标准、评价样本选择、"
            "基准分数或科学验收。"
        ),
        "gallery.tags.label": "展示标签（逗号或换行分隔）",
        "gallery.tasks.label": "所选任务",
        "gallery.note.label": "备注",
        "gallery.add": "加入 / 更新定性样本库",
        "gallery.remove": "从当前视图移除（保留逻辑删除记录）",
        "gallery.table.label": "当前定性选择",
        "runner.intro": (
            "### 任务路由\n"
            "仅支持六个显式 UnifiedTask 枚举；任务套件会先完整预检，再按固定能力顺序执行。"
        ),
        "runner.mode.label": "运行模式",
        "runner.single_task.label": "单任务",
        "runner.suite_tasks.label": "任务套件（强制 canonical 顺序）",
        "runner.instruction.label": "视觉 / 区域指令（留空使用对应任务的配置提示词）",
        "runner.question.label": "KNOWLEDGE_QA 问题（留空使用配置提示词）",
        "runner.prompts.label": "Demo 编排实际发送的提示词（只读）",
        "runner.user_mask.label": "独立用户 / 演示 Mask——严格 PNG-L、仅 0/255",
        "runner.region_source.label": "候选区域专业解释的区域来源",
        "runner.candidate_selector.label": "候选区域选择（仅限同一样本、同一空间快照）",
        "runner.run": "运行只读演示推理",
        "runner.summary.label": "任务套件摘要",
        "candidate.intro": (
            "### 候选区域预览\n"
            "这里只展示真实 OA-AuxSeg 候选区域；不使用 GT/reference mask，也不自动选择 Top-1。"
        ),
        "candidate.gallery.label": "OA-AuxSeg 候选区域叠加图",
        "candidate.mask.label": "所选候选区域 / 显式全局 Mask",
        "candidate.overlay.label": "所选候选区域叠加图",
        "candidate.metadata.label": "候选区域 ID / 边界框 / 像素面积 / 置信度 / 绑定信息",
        "candidate.run": "运行所选候选区域专业解释",
        "result.task.label": "结果任务",
        "result.original.label": "原始影像",
        "result.mask.label": "预测 / 用户 Mask",
        "result.probability.label": "Mask 概率图",
        "result.overlay.label": "预测叠加图",
        "result.spatial.label": "空间结果 / 候选区域",
        "result.raw_output.label": "模型原始输出",
        "grounded.heading": "### 基于区域的理解",
        "grounded.full.label": "完整 RGB",
        "grounded.mask.label": "二值 Mask",
        "grounded.crop.label": "上下文裁剪",
        "grounded.facts.label": "程序化事实（Programmatic Facts）",
        "grounded.pass1.label": "Pass-1 结构化观察",
        "grounded.limitations.label": "局限性",
        "knowledge.heading": "### 知识增强",
        "knowledge.evidence.label": "检索证据",
        "knowledge.pass2.label": "Pass-2 专业解释",
        "knowledge.citations.label": "引用",
        "trace.accordion": "执行轨迹",
        "trace.label": "执行计划 / 能力调用顺序 / 回退 / CUDA 峰值",
        "choice.run.single": "单任务",
        "choice.run.suite": "任务套件",
        "choice.source.all": "全部数据源",
        "choice.split.train": "训练集（train）",
        "choice.split.val": "验证集（val）",
        "choice.split.test": "测试集（test）",
        "choice.target.all": "全部目标状态",
        "choice.target.target": "有目标",
        "choice.target.no_target": "无目标",
        "choice.size.all": "全部尺寸",
        "choice.size.empty": "无目标（empty）",
        "choice.size.small": "小目标（small）",
        "choice.size.medium": "中目标（medium）",
        "choice.size.large": "大目标（large）",
        "choice.task.VLM_ONLY": "场景理解（VLM_ONLY）",
        "choice.task.SEGMENT_ONLY": "空间分割（SEGMENT_ONLY）",
        "choice.task.REGION_UNDERSTANDING": "区域理解（REGION_UNDERSTANDING）",
        "choice.task.SEGMENT_AND_UNDERSTAND": "分割并理解（SEGMENT_AND_UNDERSTAND）",
        "choice.task.KNOWLEDGE_QA": "知识问答（KNOWLEDGE_QA）",
        "choice.task.REGION_INTERPRETATION": "候选区域专业解释（REGION_INTERPRETATION）",
        "choice.region.OA_AUXSEG_CANDIDATE": "OA-AuxSeg 候选区域",
        "choice.region.USER_MASK": "用户提供 Mask",
        "choice.mode.demo": "Demo / 定性探索",
        "choice.modality.optical": "光学（optical）",
        "choice.modality.dem": "DEM（dem）",
        "choice.modality.insar_velocity": "InSAR 速度（insar_velocity）",
        "choice.modality.slope": "坡度（slope）",
        "candidate.choice": "候选区域 {candidate_id} | 像素面积={area_pixels} | 置信度={confidence}",
        "candidate.global_choice": "显式使用 OA-AuxSeg 全局 Mask（回退）",
        "preview.full_optical": "完整光学影像 / 确定性 RGB 预览",
        "preview.channel": (
            "空间专家输入预览 | {modality} / {channel_name} | valid={valid_fraction}\n"
            "原始光学通道预览；完整光学 RGB 才是当前 P0 的视觉输入"
        ),
        "preview.auxiliary": (
            "空间专家输入预览 | {modality} / {channel_name} | valid={valid_fraction}\n"
            "当前 P0 中不是 MLLM 正式 Grounded 输入"
        ),
        "status.browser.loaded": "✅ Benchmark payload 已只读加载。",
        "status.browser.test_locked": (
            "🔒 **test 已锁定**——仅显示 metadata；未访问 HDF5 payload、未执行推理、"
            "未写入 Gallery。只有负责人授权并设置 `allow_test_demo: true` 后才能访问。"
        ),
        "status.browser.test_access": (
            "⚠️ **test Demo 访问已记录**——该样本已永久失去 blind/sealed evaluation 属性；"
            "本次访问不是正式 test evaluation。"
        ),
        "status.browser.filtered": "✅ 筛选结果 {total} 条；显示第 {position} 条。",
        "status.browser.random": "✅ 已按当前人工筛选条件随机选择；未读取模型分数。",
        "status.browser.located": "✅ 已按 exact sample_id 定位并同步筛选状态。",
        "status.candidate.cleared": "候选区域选择已清空：当前样本发生变化。",
        "status.candidate.none": "当前样本没有有效空间结果。",
        "status.candidate.snapshot": (
            "Spatial snapshot `{snapshot_id}` 来自 `{source_task}`；请选择候选区域，"
            "或显式确认全局 mask。"
        ),
        "status.candidate.unselected": "尚未选择候选区域。",
        "status.candidate.global": "已显式确认全局 fallback。",
        "status.candidate.selected": "已选择候选区域 `{candidate_id}` 用于解释。",
        "status.gallery.published": (
            "✅ Gallery revision `{revision}` 已发布。仅用于定性演示；"
            "正式验收和科学验收均为 false。"
        ),
        "status.banner.invalid_tasks": "⚠️ 请至少选择一个有效 UnifiedTask。",
        "status.banner.knowledge": (
            "### 当前执行输入\n"
            "**KNOWLEDGE_ONLY——未消费 Benchmark payload。** 当前 Browser selection "
            "仅作为 UI metadata，不会打开 HDF5、构造 spatial input 或创建 test receipt。"
        ),
        "status.banner.visual": (
            "### 当前执行输入\nmode=`{mode}` · split=`{split}` · "
            "sample=`{sample_id}` · source=`{source}`"
        ),
        "status.run.published": (
            "✅ Demo run `{run_id}` 已发布：成功={success}，失败={failed}，等待={waiting}。"
            "仅为工程证据，不构成科学验收。"
        ),
        "status.run.knowledge": "KNOWLEDGE_ONLY：未消费 Benchmark payload。",
        "status.run.pending": "REGION_INTERPRETATION 正在等待显式候选区域或全局 mask 选择。",
        "status.run.test": "已记录 test receipt；盲测/封存属性为 false。",
        "error.no_samples": "当前筛选条件下没有可用样本。",
        "error.gallery_mode": "Demo Gallery 只接受当前 Benchmark Browser selection。",
        "error.candidate_required": "请先从 Candidate 预览选择具体候选区域，或显式选择全局 mask。",
        "error.candidate_index": "候选区域 Gallery 的选择索引无效。",
        "error.browser_operation": "Benchmark Browser 操作失败；详情已写入服务端日志。",
        "error.gallery_operation": "Demo Gallery 操作失败；详情已写入服务端日志。",
        "error.runner_operation": "Demo 推理请求失败；详情已写入服务端日志。",
        "error.candidate_operation": "候选区域操作失败；详情已写入服务端日志。",
        "error.i18n_state": "语言切换检测到不一致的界面状态；未修改当前选择。",
    },
    "en": {
        "app.document_title": "OA-GroundRAG Unified Demo Workbench",
        "app.header": (
            "# OA-GroundRAG Unified Demo Workbench\n"
            "**Read-only inference workbench.** Benchmark data, checkpoints, Adapters, the Text Bank, "
            "and formal outputs are never modified. Persistent writes are limited to the independent "
            "Demo Gallery, test receipts, and Demo runs."
        ),
        "app.language.label": "Interface Language / 界面语言",
        "tab.browser": "A. Benchmark Browser",
        "tab.gallery": "B. Demo Gallery",
        "tab.runner": "C. Task Runner / Result Viewer / Trace",
        "browser.intro": (
            "### Demo / Qualitative Exploration\n"
            "Filtering and random selection use only the canonical index and manual criteria. "
            "Model scores are never read and no Top-K selection is performed."
        ),
        "browser.split.label": "Split",
        "browser.source.label": "Source",
        "browser.target.label": "Target Status",
        "browser.size.label": "Target Size",
        "browser.modalities.label": "Available Modalities (all selected modalities must exist)",
        "browser.sample_query.label": "sample_id Filter / Exact Lookup",
        "browser.current_sample.label": "Current sample_id",
        "browser.apply": "Apply Filters",
        "browser.previous": "Previous",
        "browser.next": "Next",
        "browser.random": "Random Sample",
        "browser.locate": "Locate sample_id",
        "browser.metadata.label": "Sample Metadata",
        "browser.optical.label": "Full Optical / RGB Preview",
        "browser.reference.accordion": "Reference / Audit Only (never USER_MASK)",
        "browser.reference.label": "Reference / Audit Only",
        "browser.optical_channels.label": "Optical / Multispectral Channel Previews (deterministic rendering)",
        "browser.auxiliary.label": (
            "Spatial Expert Input Preview — Auxiliary Modalities | "
            "Not a Formal MLLM Grounded Input in Current P0"
        ),
        "browser.channel_metadata.label": "Channel Values / Validity / Display Transform",
        "browser.spatial_inputs.label": "Spatial Expert Inputs",
        "browser.formal_inputs.label": "MLLM Formal Grounded Inputs",
        "gallery.intro": (
            "### Qualitative Demo Gallery\n"
            "Manual qualitative selections use independent revisions. This is not Gold, an evaluation "
            "selection, a benchmark score, or scientific acceptance."
        ),
        "gallery.tags.label": "Demo Tags (comma or newline separated)",
        "gallery.tasks.label": "Selected Tasks",
        "gallery.note.label": "Note",
        "gallery.add": "Add / Update Gallery",
        "gallery.remove": "Remove from Current View (retain tombstone)",
        "gallery.table.label": "Current Qualitative Selection",
        "runner.intro": (
            "### Task Routing\n"
            "Only the six explicit UnifiedTask values are supported. A Task Suite is fully preflighted "
            "and then executed in canonical capability order."
        ),
        "runner.mode.label": "Run Mode",
        "runner.single_task.label": "Single Task",
        "runner.suite_tasks.label": "Task Suite (canonical order enforced)",
        "runner.instruction.label": "Visual / Region Instruction (blank uses the task-specific config prompt)",
        "runner.question.label": "KNOWLEDGE_QA Question (blank uses the config prompt)",
        "runner.prompts.label": "Effective Prompts Sent by Demo Orchestration (read only)",
        "runner.user_mask.label": "Independent User/Demo Mask — strict PNG-L with values 0/255",
        "runner.region_source.label": "REGION_INTERPRETATION Region Source",
        "runner.candidate_selector.label": "Candidate Selection (same sample and spatial snapshot only)",
        "runner.run": "Run Read-only Demo Inference",
        "runner.summary.label": "Task Suite Summary",
        "candidate.intro": (
            "### Candidate Preview\n"
            "Only real OA-AuxSeg candidates are shown. No GT/reference mask is used and Top-1 is never "
            "selected automatically."
        ),
        "candidate.gallery.label": "OA-AuxSeg Candidate Overlays",
        "candidate.mask.label": "Selected Candidate / Explicit Global Mask",
        "candidate.overlay.label": "Selected Candidate Overlay",
        "candidate.metadata.label": "Candidate ID / bbox / area / confidence / binding",
        "candidate.run": "Run Selected REGION_INTERPRETATION",
        "result.task.label": "Result Task",
        "result.original.label": "Original Image",
        "result.mask.label": "Predicted / User Mask",
        "result.probability.label": "Mask Probability",
        "result.overlay.label": "Predicted Overlay",
        "result.spatial.label": "Spatial Result / Candidates",
        "result.raw_output.label": "Raw Model Output",
        "grounded.heading": "### Grounded Understanding",
        "grounded.full.label": "Full RGB",
        "grounded.mask.label": "Binary Mask",
        "grounded.crop.label": "Context Crop",
        "grounded.facts.label": "Programmatic Facts",
        "grounded.pass1.label": "Pass-1 Structured Observation",
        "grounded.limitations.label": "Limitations",
        "knowledge.heading": "### Knowledge Augmentation",
        "knowledge.evidence.label": "Retrieved Evidence",
        "knowledge.pass2.label": "Pass-2 Interpretation",
        "knowledge.citations.label": "Citations",
        "trace.accordion": "Execution Trace",
        "trace.label": "ExecutionPlan / Provider Order / Fallback / CUDA Peak",
        "choice.run.single": "Single Task",
        "choice.run.suite": "Task Suite",
        "choice.source.all": "All Sources",
        "choice.split.train": "Train (train)",
        "choice.split.val": "Validation (val)",
        "choice.split.test": "Test (test)",
        "choice.target.all": "All Target States",
        "choice.target.target": "Target Present",
        "choice.target.no_target": "No Target",
        "choice.size.all": "All Sizes",
        "choice.size.empty": "No Target (empty)",
        "choice.size.small": "Small",
        "choice.size.medium": "Medium",
        "choice.size.large": "Large",
        "choice.task.VLM_ONLY": "Scene Understanding (VLM_ONLY)",
        "choice.task.SEGMENT_ONLY": "Spatial Segmentation (SEGMENT_ONLY)",
        "choice.task.REGION_UNDERSTANDING": "Region Understanding (REGION_UNDERSTANDING)",
        "choice.task.SEGMENT_AND_UNDERSTAND": "Segment and Understand (SEGMENT_AND_UNDERSTAND)",
        "choice.task.KNOWLEDGE_QA": "Knowledge QA (KNOWLEDGE_QA)",
        "choice.task.REGION_INTERPRETATION": "Candidate Region Interpretation (REGION_INTERPRETATION)",
        "choice.region.OA_AUXSEG_CANDIDATE": "OA-AuxSeg Candidate Region",
        "choice.region.USER_MASK": "User-provided Mask",
        "choice.mode.demo": "Demo / Qualitative Exploration",
        "choice.modality.optical": "Optical",
        "choice.modality.dem": "DEM",
        "choice.modality.insar_velocity": "InSAR Velocity",
        "choice.modality.slope": "Slope",
        "candidate.choice": "Candidate {candidate_id} | area={area_pixels} | confidence={confidence}",
        "candidate.global_choice": "Use OA-AuxSeg Global Mask (explicit fallback)",
        "preview.full_optical": "Full Optical / Deterministic RGB Preview",
        "preview.channel": (
            "Spatial Expert Input Preview | {modality} / {channel_name} | valid={valid_fraction}\n"
            "Raw optical channel preview; Full Optical RGB is the current P0 visual input"
        ),
        "preview.auxiliary": (
            "Spatial Expert Input Preview | {modality} / {channel_name} | valid={valid_fraction}\n"
            "Not a formal MLLM grounded input in current P0"
        ),
        "status.browser.loaded": "✅ Benchmark payload loaded read-only.",
        "status.browser.test_locked": (
            "🔒 **test locked** — metadata only. No HDF5 payload, inference, or Gallery access occurred. "
            "Access requires owner authorization and `allow_test_demo: true`."
        ),
        "status.browser.test_access": (
            "⚠️ **test Demo access recorded** — this sample has permanently lost its blind/sealed "
            "evaluation property. This is not formal test evaluation."
        ),
        "status.browser.filtered": "✅ {total} samples match; showing item {position}.",
        "status.browser.random": "✅ Randomly selected under the current manual filters; model scores were not read.",
        "status.browser.located": "✅ Located the exact sample_id and synchronized the filter state.",
        "status.candidate.cleared": "Candidate selection cleared because the current sample changed.",
        "status.candidate.none": "No valid spatial result exists for the current sample.",
        "status.candidate.snapshot": (
            "Spatial snapshot `{snapshot_id}` is from `{source_task}`. Select a candidate or explicitly "
            "confirm the global mask."
        ),
        "status.candidate.unselected": "No candidate selected.",
        "status.candidate.global": "Explicit global fallback confirmed.",
        "status.candidate.selected": "Candidate `{candidate_id}` selected for interpretation.",
        "status.gallery.published": (
            "✅ Gallery revision `{revision}` published. Qualitative demo only; formal and scientific "
            "acceptance remain false."
        ),
        "status.banner.invalid_tasks": "⚠️ Select at least one valid UnifiedTask.",
        "status.banner.knowledge": (
            "### Current Execution Input\n"
            "**KNOWLEDGE_ONLY — No Benchmark payload consumed.** The current Browser selection is UI "
            "metadata only; no HDF5 file, spatial input, or test receipt is created."
        ),
        "status.banner.visual": (
            "### Current Execution Input\nmode=`{mode}` · split=`{split}` · "
            "sample=`{sample_id}` · source=`{source}`"
        ),
        "status.run.published": (
            "✅ Demo run `{run_id}` published: success={success}, failed={failed}, waiting={waiting}. "
            "Engineering evidence only; this is not scientific acceptance."
        ),
        "status.run.knowledge": "KNOWLEDGE_ONLY: No Benchmark payload consumed.",
        "status.run.pending": "REGION_INTERPRETATION is waiting for explicit candidate/global selection.",
        "status.run.test": "A test receipt was recorded; the blind/sealed property is false.",
        "error.no_samples": "No samples match the current filters.",
        "error.gallery_mode": "The Demo Gallery accepts only the current Benchmark Browser selection.",
        "error.candidate_required": "Select a candidate in Candidate Preview or explicitly select the global mask first.",
        "error.candidate_index": "The Candidate Gallery selection index is invalid.",
        "error.browser_operation": "The Benchmark Browser operation failed. Details are available in the server log.",
        "error.gallery_operation": "The Demo Gallery operation failed. Details are available in the server log.",
        "error.runner_operation": "The Demo inference request failed. Details are available in the server log.",
        "error.candidate_operation": "The candidate operation failed. Details are available in the server log.",
        "error.i18n_state": "The language switch detected inconsistent UI state; the current selection was not changed.",
    },
}


def _placeholders(template: str) -> frozenset[str]:
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name or any(token in field_name for token in (".", "[", "]")):
            raise DemoI18nError(f"translation placeholder 非法：{field_name!r}")
        names.add(field_name)
    return frozenset(names)


def validate_catalogs() -> None:
    """要求两种语言 key 与 placeholder 集合完全一致。"""

    if set(TEXT) != set(SUPPORTED_LOCALES):
        raise DemoI18nError("translation locale 集合不匹配")
    reference = set(TEXT[DEFAULT_LOCALE])
    for locale in SUPPORTED_LOCALES:
        if set(TEXT[locale]) != reference:
            missing = sorted(reference - set(TEXT[locale]))
            extra = sorted(set(TEXT[locale]) - reference)
            raise DemoI18nError(
                f"translation key 不一致：locale={locale}, missing={missing}, extra={extra}"
            )
    for key in sorted(reference):
        expected = _placeholders(TEXT[DEFAULT_LOCALE][key])
        for locale in SUPPORTED_LOCALES[1:]:
            actual = _placeholders(TEXT[locale][key])
            if actual != expected:
                raise DemoI18nError(
                    f"translation placeholder 不一致：key={key}, locale={locale}"
                )
    for name, headers in (
        ("gallery", GALLERY_HEADERS),
        ("evidence", EVIDENCE_HEADERS),
    ):
        if set(headers) != set(SUPPORTED_LOCALES):
            raise DemoI18nError(f"{name} Dataframe header locale 集合不匹配")
        widths = {len(headers[locale]) for locale in SUPPORTED_LOCALES}
        if len(widths) != 1:
            raise DemoI18nError(f"{name} Dataframe header 列数不匹配")


def tr(locale: str, key: str, **params: Any) -> str:
    """严格翻译；不允许 locale/key/placeholder 静默回退。"""

    if locale not in SUPPORTED_LOCALES:
        raise DemoI18nError(f"不支持的 UI locale：{locale!r}")
    if key not in TEXT[locale]:
        raise DemoI18nError(f"未知 translation key：{key}")
    template = TEXT[locale][key]
    expected = _placeholders(template)
    actual = frozenset(params)
    if actual != expected:
        raise DemoI18nError(
            f"translation 参数不匹配：key={key}, expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return template.format(**params)


@dataclass(frozen=True)
class MessageSpec:
    """可在不重做业务操作的情况下按当前 locale 重绘的 UI 消息。"""

    key: str
    params: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    @classmethod
    def create(cls, key: str, **params: str | int | float | bool | None) -> "MessageSpec":
        return cls(key=key, params=tuple(sorted(params.items())))

    def render(self, locale: str) -> str:
        return tr(locale, self.key, **dict(self.params))


def render_messages(locale: str, specs: Sequence[MessageSpec] | None) -> str:
    return " ".join(spec.render(locale) for spec in (specs or ()))


def labeled_choices(
    locale: str,
    values: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    return [(tr(locale, key), value) for key, value in values]


def run_mode_choices(locale: str) -> list[tuple[str, str]]:
    return labeled_choices(locale, (
        ("choice.run.single", RUN_MODE_SINGLE),
        ("choice.run.suite", RUN_MODE_SUITE),
    ))


def task_label(locale: str, task: UnifiedTask | str) -> str:
    value = task.value if isinstance(task, UnifiedTask) else UnifiedTask(str(task)).value
    return tr(locale, f"choice.task.{value}")


def task_choices(
    locale: str,
    tasks: Iterable[UnifiedTask | str] = tuple(UnifiedTask),
) -> list[tuple[str, str]]:
    values = [task.value if isinstance(task, UnifiedTask) else UnifiedTask(str(task)).value for task in tasks]
    return [(task_label(locale, value), value) for value in values]


def region_source_choices(locale: str) -> list[tuple[str, str]]:
    return [
        (tr(locale, f"choice.region.{source.value}"), source.value)
        for source in (RegionSource.OA_AUXSEG_CANDIDATE, RegionSource.USER_MASK)
    ]


def split_choices(locale: str) -> list[tuple[str, str]]:
    return [(tr(locale, f"choice.split.{value}"), value) for value in ("train", "val", "test")]


def source_choices(locale: str, sources: Iterable[str]) -> list[tuple[str, str]]:
    values = tuple(str(value) for value in sources)
    if SOURCE_FILTER_ALL in values:
        raise DemoI18nError("Benchmark source 与 UI reserved value 冲突")
    return [
        (tr(locale, "choice.source.all"), SOURCE_FILTER_ALL),
        *((value, value) for value in values),
    ]


def target_choices(locale: str) -> list[tuple[str, str]]:
    return [(tr(locale, f"choice.target.{value}"), value) for value in ("all", "target", "no_target")]


def size_choices(locale: str) -> list[tuple[str, str]]:
    return [
        (tr(locale, f"choice.size.{value}"), value)
        for value in ("all", "empty", "small", "medium", "large")
    ]


def modality_choices(locale: str, modalities: Iterable[str]) -> list[tuple[str, str]]:
    known = {"optical", "dem", "insar_velocity", "slope"}
    return [
        (
            tr(locale, f"choice.modality.{value}")
            if value in known
            else value,
            value,
        )
        for value in modalities
    ]


def preview_caption(locale: str, preview: Any) -> str:
    return tr(
        locale,
        "preview.auxiliary" if bool(preview.is_auxiliary) else "preview.channel",
        modality=str(preview.modality),
        channel_name=str(preview.channel_name),
        valid_fraction=f"{float(preview.valid_fraction):.3f}",
    )


def preview_gallery(
    locale: str,
    full_optical: Any,
    optical_channels: Sequence[Any],
    auxiliary_channels: Sequence[Any],
) -> tuple[list[tuple[Any, str]], list[tuple[Any, str]]]:
    optical: list[tuple[Any, str]] = []
    if full_optical is not None:
        optical.append((full_optical, tr(locale, "preview.full_optical")))
    optical.extend((item.image, preview_caption(locale, item)) for item in optical_channels)
    auxiliary = [(item.image, preview_caption(locale, item)) for item in auxiliary_channels]
    return optical, auxiliary


def candidate_label(locale: str, option: Mapping[str, Any]) -> str:
    kind = option.get("kind")
    if kind == "EXPLICIT_GLOBAL":
        return tr(locale, "candidate.global_choice")
    if kind != "CANDIDATE":
        raise DemoI18nError(f"未知 Demo candidate kind：{kind!r}")
    return tr(
        locale,
        "candidate.choice",
        candidate_id=int(option["candidate_id"]),
        area_pixels=int(option["area_pixels"]),
        confidence=f"{float(option['confidence']):.4f}",
    )


def candidate_choices(
    locale: str,
    payload: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    return [
        (candidate_label(locale, option), str(option["token"]))
        for option in (payload or {}).get("options", ())
    ]


def candidate_gallery(
    locale: str,
    payload: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    return [
        (str(option["overlay_path"]), candidate_label(locale, option))
        for option in (payload or {}).get("options", ())
        if option.get("kind") == "CANDIDATE"
    ]


GALLERY_HEADERS = {
    "zh": ["样本 ID", "数据划分", "数据源", "展示标签", "备注", "任务", "更新时间"],
    "en": ["Sample ID", "Split", "Source", "Demo Tags", "Note", "Tasks", "Updated At"],
}

EVIDENCE_HEADERS = {
    "zh": ["知识类型", "来源标题", "页码", "章节", "Evidence ID", "内容"],
    "en": ["Knowledge Type", "Source Title", "Page", "Section", "Evidence ID", "Content"],
}


validate_catalogs()
