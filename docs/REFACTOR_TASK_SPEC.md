# 文档 A：多源遥感滑坡 Grounded Segmentation–Description 项目彻底重构任务书

> 文档状态：重构执行基线 v1  
> 适用项目：`yukun80/paper7_VLM`  
> 编制日期：2026-07-20  
> 执行方式：先冻结或删除旧框架，再按 P0–P8 逐阶段实施 greenfield rewrite  
> 本文不授权自动启动正式长时间训练、付费 API 批处理或专家标签冻结

## 1. 执行摘要

本项目的新目标不是继续扩展现有 SANE—QMEF—PMRD—MGRR 链路，而是将其冻结或删除为旧研究基线，以一个不兼容旧 Benchmark、旧配置、旧 cache、旧 checkpoint 格式和旧类名的 greenfield 主线重新实现多源遥感滑坡 Grounded Segmentation–Description 系统。

新主线只保留三个一级模块：

1. **Sensor-Aware Multi-Image Adapter**：把任意数量、任意组合的光学、多光谱、SAR、DEM、slope、InSAR 等观测组织为官方 Qwen3-VL 原生多图输入，并提供最少量、可审计的空间支持特征。
2. **Unified Grounded Segmentation**：优先采用 Qwen3-VL-Seg 风格的轻量 box-guided mask decoder；若 G0 证明 box bottleneck 不可接受，则只替换本模块为 PSALM-Lite。
3. **Mask-Grounded Multi-Source Region Reader**：采用 GAR 的全图上下文和 RoI-aligned feature replay 思想，增加 exact-mask 多传感器证据读取、coverage 与 null 状态，不再恢复旧 MGRR 的 context ring、CPU 连通域、residual component 和第二套 reliability。

论文定位冻结为：

> **以多源滑坡 Benchmark 和统一 grounded task 为核心，以轻量多源 VLM 适配及分割—区域描述接口为方法支撑的数据—方法论文。**

论文主张最多三项：

1. 面向任意单时相或同期多源组合的滑坡分割—区域描述 Benchmark 与任务协议；
2. 轻量、支持缺失模态、显式传感器身份且可审计的多图输入适配；
3. 将预测 mask 与逐模态区域证据绑定，并通过 mask/region swap、modality removal 和 unsupported-claim 测试验证的 grounded region understanding 接口。

PSALM 已提出 LMM-updated mask tokens、proposal 与 condition classification 解耦；Qwen3-VL-Seg 已提出 box-guided 多尺度空间注入、高分辨率融合和 mask-aware refinement；GAR 已提出 mask prompt、全图上下文和 RoI-aligned feature replay。以上均不得写成本项目首创。fileciteturn56file1 fileciteturn56file2 fileciteturn56file3

---

## 2. 实时仓库核查基线

截至本文编制时，公开仓库默认分支仍为 `master`，最近检索到的 HEAD 为 `834b5ad7233e1f288dc074078831d09838e8cfb4`，提交内容是 2026-07 的 SANE—QMEF—PMRD 重构评审归档，不是新的训练或科学验收。fileciteturn58file0L1-L3

当前公开文件仍将活动主线定义为：

```text
SANE -> QMEF -> PMRD -> semantic-evidence verifier
encode_multisource -> segment -> build region evidence -> describe
```

同时明确仓库只是 “M0–M7 engineering-complete candidate”，Bridge 仍为 `awaiting_expert_review`，D0 已完成而 D1/D2/D3a 尚未正式运行，代码存在不能代替科学验收。fileciteturn88file0L16-L43 fileciteturn88file0L45-L69

根 README 的公开快照仍记录：

- Landslide V2 Small、Description M1.1、auto-only Unified、M3 cache、D-1 和 D0 为工程有效；
- D1/D2/D3a 未正式运行；
- D3b/D4/M6 expert/M7 受真实专家 Bridge 阻塞；
- D0 monitor 只证明训练闭环可运行，不构成 grounded-region 科学结论。fileciteturn89file0L36-L58

公开默认分支中，对下列路径的直接读取返回不存在：

```text
external/PSALM/
external/Grasp-Any-Region-main/
external/RSGPT/
.gitmodules
```

因此不得假设这些目录仍存在于公开主分支。P0 必须在用户本地 worktree 中再次检查它们是否为未跟踪目录、被忽略目录或历史残留；本文将其标记为“本地待实时核查”。

### 2.1 当前仓库结构事实

当前 `AGENTS.md` 给出的活动布局为：

```text
scripts/1-benchmark/
scripts/2-instruction/
scripts/3-description/
scripts/4-landslide-bridge/
scripts/5-segdesc/

SEG_Multi-Source_Landslides/qpsalm_seg/
├── data/
├── models/
├── description/
├── engine/
└── cli/
```

旧 Benchmark、Bridge、Unified、cache、训练和评价协议彼此绑定，已经形成多套派生产物和严格 checkpoint lineage。fileciteturn88file0L143-L177

### 2.2 当前最近完成的训练阶段

公开证据只支持以下结论：

```text
最近完成：D0 MMRS Caption，1000 steps，engineering-valid
尚未完成：D1、D2、D3a 当前 Small 正式运行
未科学验收：M4、D3b、D4、M6 expert、M7
```

P0 若在本地发现比公开 README 更新的 report，只能在核验文件内容、hash、checkpoint 和 git 状态后更新这一结论。

---

## 3. 当前系统为什么必须重构

### 3.1 科学叙事已经过载

当前系统同时承担：

- 多波段原始物理编码；
- Qwen-ViT 多层缓存；
- 模态 reliability；
- 多尺度 deformable sampling；
- Qwen mask query；
- proposal set；
- relevance verifier；
- noisy union；
- detail refinement；
- exact mask pooling；
- RoI replay；
- context ring；
- component replay；
- 第二套区域 reliability；
- 双 Adapter；
- 多阶段数据与 checkpoint gate。

这种结构不是简单的“四个模块”，而是多套视觉编码、空间采样、可靠性调节和区域读取机制叠加。每个结构都需要独立三随机种子和反事实验证，组合消融成本超过单卡项目能够可靠承担的范围。

### 3.2 多处重复成熟模型已有能力

旧 `SensorAwareNativeScaleEncoder` 同时维护 raw physical branch 和缓存的 Qwen-ViT 多层特征；旧 `QwenGuidedEvidenceFusion` 又执行 reliability、deformable alignment 和 query-conditioned sampling；旧 `ProposalSetMaskRefinementDecoder` 再执行 proposal、detail fusion、refinement 和 noisy union。旧算法文档也承认 reliability 可能多次衰减、坐标约定尚未统一、缓存视觉证据不等价于原生在线 Qwen3-VL 前向。fileciteturn90file0L24-L44 fileciteturn90file0L48-L72 fileciteturn90file0L104-L130

旧 schema 中 `EvidenceFeatures`、`ProposalSet`、`SegmentationOutput` 已把 reliability、proposal relevance、query modality/scale attention 等诊断固定为主模型合同，使后续简化必须同时修改数据、模型、损失、评价和 checkpoint。fileciteturn92file0L280-L344

### 3.3 数据与协议叠加已经影响可维护性

旧配置同时引用：

- Landslide V2；
- Description V2；
- Bridge；
- Unified SegDesc；
- segmentation vision cache；
- description vision cache；
- segmentation checkpoint；
- D-1、D0–D4、M6、M7 gate。

`qpsalm_segdesc_small.yaml` 本身已经同时绑定三套 Benchmark、两套 cache、两个 Adapter、多个阶段和 counterfactual modes。fileciteturn94file0L6-L28 fileciteturn94file0L30-L77

重构必须从唯一 canonical parent record 开始，不能再在旧 Landslide V2、Description M1.1、Bridge 和 Unified 外面增加第五层索引。

### 3.4 自动文本不能承担专业真值

旧 Bridge 自动文本具有良好数据血缘，但其主要内容来自确定性几何、规则和教师候选。它可用于辅助训练和审核预填充，不能作为 expert val/test，也不能证明模型具备地学视觉理解。新流水线必须从 Evidence Packet、字段级双模型独立观察和专家冻结 canonical fact record 出发。

---

## 4. 科学问题和论文定位

### 4.1 固定科学问题

> 给定同一地理区域、单时相或近同期的任意可用遥感传感器集合，能否只增加轻量的传感器输入适配和 mask-grounded 区域证据读取机制，复用成熟 VLM 的多图理解、语言 grounding、像素分割和区域理解能力，在统一参考画布上输出滑坡 mask，并生成只陈述当前模态所支持事实的区域描述？

### 4.2 论文定位

优先定位：

```text
多源滑坡 Benchmark 与统一 grounded task
+ 轻量多源 VLM 适配
+ 分割—区域描述统一接口
```

不优先定位为：

- 完整新视觉骨干论文；
- 通用遥感多任务 assistant；
- 变化检测或多时相推理论文；
- 仅靠 learned reliability 的鲁棒分割论文；
- 只展示工程代码量的系统论文。

MIGRANT 的启发是：任务 taxonomy、数据构建、指令格式和阶段训练本身可以形成主要贡献，不必依赖大量结构创新。MIGRANT 采用单图遥感适配后再进行多图 grounding advancement，并以 MIG-RS-Instruct/MIG-RS-Bench 组织贡献；其公开仓库主要围绕数据预处理、QA 模板、ms-swift 微调和评价展开。fileciteturn56file6 fileciteturn77file0L34-L50

---

## 5. 研究范围与明确排除项

### 5.1 输入范围

允许的输入是同一地理区域、单时相、同期或近同期的任意组合：

- 高分辨率光学 RGB；
- 多波段光学；
- Sentinel-2 多光谱；
- Sentinel-1 SAR；
- DEM；
- slope；
- InSAR LOS 形变速率；
- 其他具有明确 product type、单位、sign convention 和 valid coverage 的形变产品。

### 5.2 正式任务

| 任务 | 输入 | 输出 |
|---|---|---|
| T1 global/no-target segmentation | 多源 parent + 指令 | 全部滑坡 semantic mask 或空 mask |
| T2 referring segmentation | 多源 parent + referring expression | 对应滑坡区域 mask 或空 mask |
| T3 GT-mask region understanding | 多源 parent + GT mask | 字段级视觉事实、证据状态、摘要 |
| T4 fixed/end-to-end region understanding | 多源 parent + OOF fixed mask 或在线预测 mask | 与实际预测区域对应的描述 |

模态组合、缺失模式、coverage、quality 和 GSD 是输入条件轴，不另行扩张为任务族。

### 5.3 明确排除

不得加入：

- 灾前—灾后变化检测；
- 时间差分或 change branch；
- 目标跟踪、视频、跨时相关系推理；
- 跨视图地理定位；
- 多 region 关系推理；
- MMRS 分类、普通检测、普通 VQA、红外全任务；
- 未经证据支持的触发因素、发生时间、运动速度、未来失稳和风险等级；
- 缺失单位或 sign convention 时的定量物理结论；
- oracle matched proposal 作为部署结果。

DisasterM3 只借鉴“任务 taxonomy—数据清洗—人工验证—多维评价”的组织方式。其核心数据和任务是双时相灾害评估，因此不得迁移其 pre/post、变化、恢复建议等任务进入本项目。fileciteturn56file0

---

## 6. 旧系统审计与新旧映射

### 6.1 旧核心类和文件

| 旧资产 | 当前职责 | 新主线处置 |
|---|---|---|
| `qpsalm_seg/schema.py::ModalityInstance` | 多源物理和空间语义 | 设计思想保留，定义新 `ModalityRecord`；不保留旧类名 alias |
| `schema.py::MultisourceBackboneState` | task-neutral state | 设计思想保留，定义新 `QwenBackboneState` |
| `schema.py::SegmentationState` | QMEF/semantic state | 替换为轻量 `GroundingState` |
| `schema.py::RegionEvidenceState` | MGRR state | 替换为 `RegionReadout` |
| `models/sane.py::SensorAwareNativeScaleEncoder` | raw + cached pyramid | 删除；由多图 Adapter 和可选 support residual 取代 |
| `models/qmef.py::QwenGuidedEvidenceFusion` | reliability + deformable fusion | 删除 |
| `models/qmef.py::ScaleAwareDeformableAggregator` | 多尺度坐标采样 | 删除 |
| `controllers.py::QwenMaskQueryController` | 缓存 token 条件化 language decoder | 删除；改用官方在线 Qwen3-VL forward wrapper |
| `models/pmrd.py::ProposalSetMaskRefinementDecoder` | proposal/refinement/union | 删除；由 G0 选定的成熟分割内核取代 |
| `models/qpsalm.py::MultiSourceQwenPSALMSeg` | 旧总装 | 删除 |
| `description/modeling/mgrr.py::MultiGranularityRegionReplay` | exact/RoI/context/component/reliability | 删除；以 GAR-lite 重写 |
| `description/modeling/region_baselines.py` | MGRR 消融 | P6 后删除 |
| `qpsalm_v2_*.yaml` | 旧分割配置 | P8 删除 |
| `qpsalm_segdesc_small.yaml` | 旧 SegDesc 配置 | P8 删除 |
| `scripts/1-benchmark` 至 `scripts/5-segdesc` | 多层派生 Benchmark | P1 验收后进入删除清单，P8 清理 |
| `ALGORITHM.md`、`benchmark_GAR.md` | 旧算法正式协议 | P8 移入 Git 历史，不在活动 docs 保留 |

旧 `ModalityInstance` 已明确包含 sensor、product、bands、orbit、units、signed、GSD、valid mask 和 quality；这些信息是多源遥感任务的必要复杂性，必须进入新 canonical schema。fileciteturn92file0L21-L39

### 6.2 必须保留的资产

以下资产不因 greenfield rewrite 删除：

1. 原始 `datasets/`；
2. 已接受的旧 benchmark、cache、checkpoint、报告和可视化结果，只读保存；
3. 旧系统 Git tag 和 baseline branch；
4. 原始 source registry、最小 scientific provenance、来源 hash；
5. 现有专家审核材料、reviewer 文件和仲裁文件；
6. 当前 ontology 中已明确区分的 deterministic geometry 与 modality evidence 语义；
7. 原始实验日志，用于论文公平对照；
8. 可复用的原子写、hash、split 隔离和 checkpoint lineage 设计思想。

### 6.3 最终删除资产

在替代阶段验收后，活动主分支必须删除：

- 旧 SANE/QMEF/PMRD/MGRR 代码；
- 旧 Benchmark reader 和 builder；
- 旧 cache protocol；
- 旧 checkpoint compatibility loader；
- 旧 CLI；
- 旧配置；
- 旧测试；
- 失效文档；
- 本地整体嵌套的第三方仓库；
- 旧类名 alias、兼容 shim、迁移 fallback。

删除不等于丢失：Git tag、baseline branch 和旧 accepted artifacts 是唯一历史存档。

---

## 7. 新项目目录和唯一运行入口

新主线采用标准 `src/` 布局：

```text
paper7_VLM/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── REFACTOR_PROGRESS.md
├── configs/
│   ├── benchmark_v3_small.yaml
│   ├── benchmark_v3_full.yaml
│   ├── model_sami.yaml
│   ├── train_s1.yaml
│   ├── train_s2.yaml
│   ├── train_s3.yaml
│   ├── eval_segmentation.yaml
│   └── eval_description.yaml
├── schemas/
│   ├── canonical_parent_v3.schema.json
│   ├── task_view_v3.schema.json
│   ├── evidence_packet_v1.schema.json
│   ├── canonical_fact_v1.schema.json
│   └── model_output_v1.schema.json
├── src/sami_gsd/
│   ├── cli.py
│   ├── contracts/
│   ├── data/
│   ├── annotation/
│   ├── model/
│   │   ├── qwen_backbone.py
│   │   ├── sensor_adapter.py
│   │   ├── segmentation/
│   │   ├── region_reader.py
│   │   └── model.py
│   ├── training/
│   ├── evaluation/
│   └── utilities/
├── tests/
├── tools/
└── docs/
    ├── adr/
    ├── audits/
    ├── handoffs/
    └── reports/
```

唯一 CLI：

```text
sami-gsd
```

允许的子命令：

```text
sami-gsd data audit
sami-gsd benchmark build
sami-gsd benchmark validate
sami-gsd benchmark summarize
sami-gsd model smoke
sami-gsd g0 evaluate
sami-gsd train segmentation
sami-gsd train description
sami-gsd train joint
sami-gsd evaluate segmentation
sami-gsd evaluate description
sami-gsd evaluate counterfactual
sami-gsd export predictions
```

不得为同一功能保留第二个 Python module CLI、旧 shell wrapper 或兼容命令。README 是最终命令的唯一来源。

---

## 8. 新 Benchmark 任务体系

新 Benchmark 名称冻结为：

```text
SAMI Landslide Grounded Benchmark v3
benchmark/sami_landslide_v3/<small|full>/
```

目录：

```text
manifests/
  source_registry.yaml
  benchmark_manifest.json
  split_manifest.json
  duplicate_clusters.jsonl
  evaluation_conditions.json
parents/
  all.jsonl
  train.jsonl
  val.jsonl
  test.jsonl
tasks/
  t1_global/
  t2_referring/
  t3_gt_region/
  t4_predicted_region/
assets/
  <parent_id>/
reports/
  validation_report.json
  summary_report.json
  provenance_report.json
  duplicate_report.json
```

新 Benchmark 不读取旧 Landslide V2、Description M1.1、Bridge、Unified 或旧 cache。旧派生产物只允许用于 P0 审计、source crosswalk 和公平对照。

---

## 9. Canonical data schema

### 9.1 Canonical parent record

每行必须满足 `canonical_parent_v3.schema.json`，顶层字段固定为：

```json
{
  "schema_version": "sami_canonical_parent_v3",
  "parent_id": "...",
  "source": {
    "dataset": "...",
    "record_id": "...",
    "scene_id": "...",
    "event_id": null,
    "region_id": "...",
    "source_group_id": "..."
  },
  "split": "train",
  "reference_canvas": {},
  "modalities": [],
  "annotations": {},
  "provenance": {},
  "hashes": {},
  "annotation_status": "..."
}
```

### 9.2 Reference canvas

`reference_canvas` 必须包含：

```text
reference_modality_id
coordinate_space = reference_pixel_half_open
original_hw
canvas_hw
valid_mask_path
transform_chain
inverse_transform_available
crs / geotransform（若源数据提供）
```

内部 bbox 一律使用像素半开区间：

```text
[x0, y0, x1, y1)
```

只有在 Qwen grounding 序列化边界，才转换为 `[0,1000]` 坐标。转换前后必须有 round-trip test。禁止在模型或 evaluator 中猜测坐标制。

### 9.3 Reference canvas 选择规则

按以下顺序确定，不允许训练时随机选择：

1. 若人工/官方 mask 原生定义在某个栅格上，该栅格为 reference canvas；
2. 若多个已配准栅格均有 mask，选有效覆盖完整且 GSD 最细者；
3. 若为单图语言数据，原图即 reference canvas；
4. 若 support modality 无可靠 source-to-reference transform，它可以作为全局 Qwen view，但不得进入 aligned support residual、exact-mask pooling 或像素级证据评价；
5. 无可逆映射的 parent 不得进入 T1–T4 空间任务，只能进入独立的全图语言适配流。

### 9.4 Modality record

每个 modality 必须包含：

```text
modality_id
family
sensor
product_type
band_names
band_metadata
orbit
acquisition_time / time_range
native_gsd_m
aligned_gsd_m
units
signed
sign_convention
normalization
quality
availability_status
valid_coverage
native_asset_path
aligned_asset_path
valid_mask_path
source_to_reference_transform
reference_to_source_transform
alignment_status
render_policy
hashes
```

状态必须区分：

```text
missing
present_zero_valid
present_partial_valid
present_valid
```

不得用全零图代替 missing，也不得把 zero-valid 当成正常输入。

### 9.5 Annotation record

`annotations` 包含：

```text
global_landslide_mask
global_target_status
referring_regions[]
no_target_eligibility
region_fact_refs[]
```

多连通滑坡是一个 semantic mask。为 box-guided decoder 派生的连通组件只能称为 `component_region`，不得称为人工滑坡实例。

### 9.6 Task view

任务视图只能在 parent split 冻结后派生。每行包含：

```text
task_id
parent_id
task_type
instruction
target_status
region_geometry
target_mask_ref
target_box_ref
answer_ref
annotation_origin
weight
```

模态 dropout 不通过复制任务行实现；训练由 sampler 根据 parent 的 modality list 采样，评价由冻结的 `evaluation_conditions.json` 指定 active modality subset。

---

## 10. 数据来源选择和排除

### 10.1 核心滑坡数据

P0 必须从原始数据根目录生成 source inventory。允许进入 spatial task 的源必须满足：

- 有可追溯 parent/group；
- 有 global 或 referring mask；
- split 可按原始大图、事件、区域或 source group 冻结；
- 本地路径存在且可读，数据格式和监督可解析；
- valid mask 和坐标变换可以重建。

旧 Benchmark v2 的 materialized arrays 不能作为新 Benchmark 唯一来源；若原始数据确实丢失，必须由人工 ADR 决定是否把旧 materialized asset 作为受限 source，Codex 不得自行降级。

### 10.2 遥感语言辅助数据

固定选择：

- MMRS 中的五个光学 caption source：
  - RSICD；
  - UCM-Captions；
  - Sydney-Captions；
  - NWPU-Captions；
  - RSITMD。
- DIOR-RSVG：仅 box/region—短语对应；
- RSICap：详细遥感描述风格；
- RSIEval：永久 test-only。

固定排除：

- `total.json`；
- MMRS classification；
- MMRS ordinary detection；
- MMRS ordinary VQA；
- infrared；
- 无关 SAR ship/infrared task；
- DIOR-RSVG 作为详细区域 caption 真值；
- RSIEval 参与训练、早停或 prompt 调参。

EarthGPT 的 MMRS-1M 涵盖大量分类、检测、VQA、grounding 和多传感器任务，本项目只使用任务相关子集，不继承其“通用遥感 assistant”目标。fileciteturn56file5

RSGPT 报告 RSICap 为 2,585 条人工详细 caption，RSIEval 为 100 张人工评测图；其图像来自 DOTA，官方仓库声明仅限学术用途、禁止商业使用。fileciteturn56file4 fileciteturn79file0L67-L73 fileciteturn79file0L104-L110

### 10.3 Scene–region ontology v2

新文件：

```text
configs/scene_region_ontology_v2.yaml
```

至少包括：

```text
vegetation
water_system
farmland
bare_soil
exposed_rock
road
railway
bridge
building
settlement
valley
channel
ridge
slope_position
target_location
target_shape
boundary_clarity
surface_disturbance
vegetation_disturbance
internal_texture
relation_to_river
relation_to_road
relation_to_settlement
alternative_explanation
evidence_limitation
```

每个字段必须声明：

```text
kind
allowed_values
synonyms
direct_observation_or_inference
permitted_source_views
forbidden_without_metadata
evaluation_metric
```

### 10.4 最小 provenance registry

`source_registry.yaml` 每个 source 必须包含：

```text
source_key
source_name
source_root
source_document
citation_key
upstream_url
provenance_notes
```

这些字段仅用于科学可追溯，不表达许可、审批或再分发结论，也不参与 runtime gate。Builder
只按本地可用性、格式/监督/研究边界、parent/group、split、坐标、valid 和 duplicate 等技术条件
选择数据。P0-P7 不查询或推断 raw data license，不生成 source 授权请求。未来公开分发原始图像、
物化 Benchmark 或派生数据包时，在独立 publication/release 阶段人工审查，不反向阻塞本阶段构建。

---

## 11. 数据预处理、坐标与 split

### 11.1 预处理原则

- 原始数据只读；
- 所有 derived asset 写入 Benchmark；
- 图像 resize 使用固定策略；
- mask 和 valid mask 只用 nearest；
- 每次 crop/resize/pad 记录在 `TransformChain`；
- 所有 transform 必须能 round-trip 或显式标记不可逆；
- padding 和 nodata 从 loss、metric、pooling 中排除；
- normalization 仅记录为机器字段，不写进自然语言 prompt；
- dataset name 不作为模型输入。

### 11.2 物理字段

SAR、DEM、slope、InSAR renderer 只能按其实际数据级别描述：

- 只有原始物理量、units、sign、GSD 均可信时，允许定量；
- 只有归一化数组时，只能做 relative/qualitative observation；
- 缺失 sign convention 时，不得解释 InSAR 正负方向；
- 无 colorbar/time range/units 时，不得生成形变速率；
- `normalization` 不得被模型当成地学证据。

### 11.3 Split

固定顺序：

```text
source inventory
→ source groups
→ exact duplicate clusters
→ perceptual candidates
→ verified duplicate clusters
→ group-level split
→ task expansion
```

分组约束的并集包括：

- original large image；
- scene；
- event；
- region；
- source group；
- exact/perceptual duplicate cluster。

同一 connected group 只能进入一个 split。任务视图不得再次随机切分。

### 11.4 Duplicate protocol

采用：

1. SHA-256 exact；
2. dHash 只负责 candidate recall；
3. 统一 RGB64；
4. MAE `<= 3.0` 验证近重复；
5. verified edges 形成 connected components；
6. split 前合并或绑定。

### 11.5 稳定构建

所有 builder：

- 输入按稳定 key 排序；
- 随机过程显式 seed；
- 输出 `allow_nan=false`；
- atomic write；
- manifest 保存 builder version、config hash、source hash、record hash；
- 同一输入重复运行必须得到相同 aggregate hash。

---

## 12. 滑坡语义标注流水线

### 12.1 Evidence Packet

每个区域生成 `evidence_packet_v1`：

1. 原始 reference image；
2. mask overlay；
3. 保留上下文的局部 crop；
4. 各可用模态 view；
5. deterministic geometry；
6. availability 和 valid coverage；
7. units、sign convention、normalization；
8. source hashes；
9. 禁止推断列表。

### 12.2 三层事实

**第一层：确定性事实**

```text
target_status
location
area_ratio
bbox
centroid
elongation
compactness
fragmentation
coverage
```

程序计算，不计作模型视觉推理。

**第二层：视觉观察**

```text
slope_position
boundary_clarity
surface_condition
vegetation_disturbance
internal_texture
channel_relation
road_relation
settlement_relation
alternative_explanation
```

由模型提出、专家审核。

**第三层：受约束解释**

```text
optical_support
multispectral_support
sar_support
terrain_support
deformation_support
evidence_sufficiency
```

只允许 `supports / does_not_support / insufficient / unavailable` 等受控值。

### 12.3 双模型独立观察

流水线固定为：

```text
deterministic facts
→ Evidence Packet
→ Model A JSON
→ Model B JSON（不看 A）
→ field verifier
→ expert review/arbitration
→ canonical fact record
→ derived text/tasks
```

每个模型调用必须保存：

```text
provider
model
model_version
prompt_hash
packet_hash
image_hashes
raw_response
parsed_response
timestamp
cost
status
```

API idempotency key：

```text
sha256(provider + model_version + prompt_hash + packet_hash)
```

相同 key 已成功时禁止重复付费调用。

### 12.4 字段级规则审核

至少检查：

- target status 与输入一致；
- deterministic fields 与程序结果一致；
- 声称某模态支持时该模态存在且 coverage 合格；
- 数字是否有单位和 sign；
- 禁止项是否出现；
- 描述是否对应当前 mask；
- 两模型字段是否一致；
- referring expression 能否反向 grounding 到当前区域。

### 12.5 Gold / Silver / Auto

| 等级 | 审核 | 用途 |
|---|---|---|
| Gold | 两名专家独立审核 + 仲裁 | expert val/test、高质量 train |
| Silver | 双模型 + 单专家修订 | 主体训练 |
| Auto | 双模型一致 + 规则 + cycle grounding | 低权重辅助训练，不作 test truth |

现有 300-parent Pilot 和 485 review items 可作为 Gold Pilot 候选，但必须重新映射到 v3 canonical parent，旧 reviewer 模板不能被直接重命名为新协议。

### 12.6 派生输出

同一 canonical fact record 程序化派生：

- detailed region description；
- short report；
- referring expression；
- targeted QA；
- unsupported-evidence negative QA。

不得分别重复标注多套文本。

---

## 13. 新模型总架构

模型暂定名称：

```text
Sensor-Aware Multi-Image Grounded SegDesc
SAMI-GroundSegDesc
```

数据流：

```text
Canonical parent + active modality subset
→ Sensor-Aware Multi-Image Adapter
→ Qwen3-VL-2B native multi-image forward
→ task-neutral QwenBackboneState
   ├─ Grounded box(es) + <seg> hidden state
   │  → Grounded Mask Decoder
   │  → semantic mask
   └─ mask + per-view spatial maps
      → GAR-lite Region Reader
      → visual fact JSON
→ deterministic fact compiler
→ constrained natural-language summary
```

官方在线 Qwen3-VL forward 是模型定义。cache 只是可关闭的透明 memoization。

Qwen3-VL 官方代码采用 Apache-2.0，支持更强 2D grounding，并通过 DeepStack 融合多层 ViT 特征；本项目应使用官方 processor、chat template 和模型类，不复制整个仓库。fileciteturn62file0L18-L55 fileciteturn61file0L3-L7

---

## 14. 模块一：Sensor-Aware Multi-Image Adapter

### 14.1 唯一职责

只负责：

- 将 variable-cardinality modalities 变为 Qwen 多图序列；
- 维护 sensor identity；
- 维护 reference/support 关系；
- 区分 missing 与 zero-valid；
- 提供可选的 valid-gated support residual。

不负责：

- mask query；
- proposal；
- segmentation loss；
- region description；
- learned reliability；
- deformable attention。

### 14.2 输入/输出接口

```python
class SensorAwareMultiImageAdapter(nn.Module):
    def prepare(
        self,
        parents: list[CanonicalParent],
        active_modalities: list[tuple[str, ...]],
    ) -> MultiImageBatch: ...

    def encode(
        self,
        batch: MultiImageBatch,
        *,
        return_spatial_features: bool,
    ) -> QwenBackboneState: ...
```

`QwenBackboneState` 至少包含：

```text
language_aligned_visual_tokens
per_view_spatial_features
grid_thw
view_order
view_ids
reference_view_id
sensor_cards
valid_masks
view_transforms
processor_fingerprint
model_fingerprint
```

### 14.3 View 顺序

固定：

1. reference view；
2. optical；
3. multispectral；
4. SAR；
5. terrain；
6. deformation；
7. 同 family 内按 `modality_id` 排序。

输入排列变化测试必须证明 identity mapping 不依赖调用方原始 list 顺序。

### 14.4 Sensor card

每幅有效 view 前加入 compact card：

```text
view_id
family
sensor
product_type
bands/polarizations
orbit
GSD
units
sign convention
valid coverage
quality
```

禁止加入：

- dataset name；
- normalization 方法；
- split；
- GT label；
- target mask geometry；
- annotation origin。

### 14.5 Pixel budget

G0 必须 profile 两个固定候选：

```text
Profile S: reference <= 512², support <= 384², max 4 views
Profile M: reference <= 768², support <= 448², max 6 views
```

选择峰值显存不超过 22 GiB、且科学指标更高的 profile；保留约 2 GiB 安全余量。

### 14.6 Optional aligned support residual

只允许一个简化路径：

```text
support feature
→ 1×1 projection
→ stored transform warp to reference
→ binary valid masking
→ valid-count average
→ near-zero residual scale
→ mask decoder memory
```

没有 learned reliability、query-conditioned sampling 或 family-specific pyramid。该路径只有在相对 native multi-image baseline 有稳定增益时才启用。

### 14.7 Cache 等价性

cache key 必须包含：

```text
model weight hash
processor hash
Qwen code revision
input image hashes
sensor card hash
pixel budget
view order
dtype
```

同设备同 dtype 等价性要求：

```text
shape exact
metadata exact
cosine_similarity >= 0.9999
FP32 max_abs <= 1e-4
BF16 max_abs <= 5e-3
```

不满足则禁用 cache，不允许 fallback 到旧 cache。

---

## 15. 模块二：Unified Grounded Segmentation

### 15.1 首选内核

首选实现 Qwen3-VL-Seg-style decoder：

```text
Qwen grounded boxes
+ per-object <seg> hidden states
+ Qwen multi-scale visual features
+ reference shallow details
+ optional support residual
→ masks + IoU confidence
```

Qwen3-VL-Seg 把 grounded box 作为结构先验，并使用 multi-scale spatial feature injection、spatial-semantic query、box-guided high-resolution fusion 和 iterative mask-aware refinement，新增约 17M 参数。该设计是借鉴对象，不是本项目创新。fileciteturn56file3

### 15.2 训练序列

每个目标输出严格 JSON：

```json
{
  "target_status": "present",
  "objects": [
    {
      "label": "landslide",
      "bbox_2d": [x0, y0, x1, y1],
      "mask_token": "<seg>"
    }
  ]
}
```

no-target：

```json
{"target_status": "absent", "objects": []}
```

一个 box 对应一个 `<seg>` hidden state和一个 mask。global semantic mask 的多个 component 按面积降序、centroid tie-break 排序，称为 component regions，不称实例。

### 15.3 Decoder 最小结构

固定包括：

1. 4 层 Qwen vision feature 的轻量 1×1 projection；
2. near-zero depthwise spatial adapter；
3. multimodal visual embedding 与 spatial feature 的 memory；
4. box Fourier encoding + `<seg>` hidden state构成 query；
5. reference RGB/显示 view 的浅层 stem；
6. 15% 扩张 soft box gate；
7. 两阶段 upsample/fusion；
8. 一次 mask-aware query refinement；
9. IoU head。

首版不增加更多 refinement rounds。

### 15.4 Multi-box 和 semantic union

- 每个 box 独立预测 mask；
- 保留 box、mask、IoU score；
- final semantic probability 使用 pixelwise `max`；
- 不使用 PMRD verifier；
- 不使用 calibrated noisy union；
- 不使用 oracle proposal；
- no-target 直接输出全零 mask。

### 15.5 PSALM-Lite 回退

若 G0 失败，只替换本模块为：

```text
Qwen-updated mask tokens
→ standard Mask2Former-style proposal decoder
→ condition classification
→ semantic mask
```

保留：

- LMM-updated mask tokens；
- proposal/classification 解耦；
- standard bipartite matching。

删除：

- QMEF；
- PMRD verifier；
- noisy union；
- query-specific 1/2 detail path；
- 多层 reliability。

PSALM 官方代码和模型使用 Apache-2.0；其主要设计可作为 fallback 参考，但不得整体复制或嵌套。fileciteturn43file0L10-L47 fileciteturn73file0L3-L7

---

## 16. G0 架构选择门

G0 不是正式长训练，而是冻结分割内核的实验门。

### 16.1 必须比较

```text
G0-A Qwen native multi-image box grounding
G0-B GT-box → Qwen3-VL-Seg-style mask decoder
G0-C predicted-box → decoder
G0-D multi-box semantic union
G0-E no-target
G0-F PSALM-Lite pilot
G0-G optional SAM2 box-prompt engineering baseline
```

SAM2 只作为外部分割 baseline，不进入主方法。其图像 predictor 提供清晰的 model builder / predictor API，但主仓库不得引入视频或 tracking 路径。fileciteturn86file0L77-L99

### 16.2 固定人口

- 200–300 个 parent；
- 覆盖 positive、no-target；
- 覆盖单组件、多组件；
- 覆盖主要模态组合；
- parent 固定；
- seed 固定；
- 不用 expert test；
- 不用旧 oracle proposal。

### 16.3 决策规则

选择 Qwen3-VL-Seg-style 仅当同时满足：

1. GT-box mask Dice 与预算匹配 legacy route A 差距不超过 2 个绝对百分点；
2. predicted-box end-to-end Dice 至少达到 GT-box Dice 的 90%；
3. multi-component target area coverage 不低于 90%；
4. no-target false-positive rate 不比 PSALM-Lite 高超过 1 个百分点；
5. 4–6 views 峰值显存不超过 22 GiB；
6. 无坐标 round-trip 错误；
7. 64 样本 overfit 可通过。

任一核心条件失败，选择 PSALM-Lite。决策写入：

```text
docs/adr/ADR-0002-segmentation-kernel.md
reports/g0/g0_decision.json
```

决策后，未选路线不得进入正式模型；其试验代码在 P8 删除，结果留在 report 和 Git 历史。

---

## 17. 模块三：Mask-Grounded Multi-Source Region Reader

### 17.1 唯一职责

给定 task-neutral Qwen visual state 和一个 exact region mask，构造少量可供 description adapter 使用的区域证据 token。

### 17.2 输入/输出

```python
class MultiSourceRegionReader(nn.Module):
    def forward(
        self,
        state: QwenBackboneState,
        region_masks: Tensor,
        region_sources: list[str],
    ) -> RegionReadout: ...
```

`RegionReadout` 包含：

```text
global_context_tokens
exact_mask_tokens
roi_replay_tokens
sensor_evidence_tokens
coverage
null_states
region_token_mask
diagnostics
```

### 17.3 Token 组成

每 region 首版固定：

```text
1 global context token
1 reference exact-mask token
1 reference RoI token
0–1 exact-mask token per valid support modality
0–1 RoI token per valid support modality
1 availability/coverage summary token
```

由实际模态数形成变长序列，只在 token 维 padding。

### 17.4 Exact-mask pooling

- reference mask 通过记录的 transform 映射到每个 view；
- 与 view valid mask 相交；
- coverage 为 0 返回 explicit null；
- 不得用 projection bias制造伪证据；
- alignment unknown 的 view 只贡献 global token，不贡献 exact/RoI token。

### 17.5 RoI replay

使用标准 `torchvision.ops.roi_align` 或经测试等价的纯 PyTorch 实现，从完整图像 feature map 读取 bbox 内网格。GAR 的关键价值是：RoI 特征来自完整图像 feature map，从而同时保留 global context 和 local details。fileciteturn56file2

### 17.6 明确删除

首版不实现：

- context ring；
- SciPy/NumPy CPU connected components；
- residual components；
- component slot；
- 第二套 learned reliability；
- 12–20 个手工 token；
- multi-region relation；
- GAR AnyRes/PerceptionLM/XTuner 全栈。

### 17.7 输出分工

程序计算：

```text
location
size
area ratio
bbox
elongation
compactness
fragmentation
coverage
```

VLM 生成：

```text
surface observation
boundary clarity
vegetation disturbance
slope/channel/road/settlement context
alternative explanation
per-modality support
evidence limitation
```

最终发布保存两层：

1. `visual_fact_record`：vision-only，承担科学评价；
2. `compiled_report`：程序合并 deterministic facts 后生成，承担应用展示。

JSON 格式正确率仅为工程指标。

---

## 18. 训练体系

训练只保留 G0、S1、S2、可选 S3。

### 18.1 Trainer 选择

**冻结选择 B：自研最小 Trainer。**

实现基础：

```text
PyTorch
Transformers
Accelerate
PEFT
bitsandbytes
Pydantic
```

不把 ms-swift 作为运行时依赖，不维护第二套 Trainer。原因是本项目需要自定义 dense mask loss、Qwen hidden/spatial state、region token 注入和两个任务 Adapter；直接嵌入持续快速变化的 ms-swift 会把 greenfield 主线再次绑定到大型外部框架。ms-swift 只借鉴 LoRA/QLoRA、resume、initialize-from、CLI lineage 和日志规范。其当前项目覆盖 Qwen3-VL、多模态训练、LoRA/QLoRA 和量化训练，但不是本项目运行依赖。fileciteturn87file0L55-L78

### 18.2 Trainer 分层

不得建立 god class。至少拆分为：

```text
TrainingLoop
SegmentationStep
DescriptionStep
JointSchedule
CheckpointManager
RunManifest
TaskSampler
MetricWriter
```

### 18.3 S1 分割适配

训练：

- Qwen3-VL-2B NF4；
- `seg_adapter`；
- sensor adapter trainable projections；
- grounded mask decoder；
- optional support residual。

冻结：

- Qwen vision tower；
- Qwen base parameters；
- description adapter。

数据：

```text
T1 global/no-target
T2 referring
modality dropout
```

推荐初始 task sampling：

```text
global 0.4
referring 0.4
no-target 0.2
```

### 18.4 S2 区域描述适配

课程：

```text
S2a MMRS/RSICap scene-language adaptation
S2b DIOR region alignment
S2c landslide GT-mask facts: Auto/Silver/Gold
S2d OOF fixed predicted-mask facts
```

训练：

- `desc_adapter`；
- region reader；
- visual-to-language projector；
-必要 special embeddings。

不更新 segmentation adapter 和 mask decoder。

长 caption、短 QA 和字段 JSON 均按 sample mean loss；不得因 token 数更多而支配总 loss。

### 18.5 S3 可选联合训练

只有 S1、S2 分别验收后允许。

约束：

- 独立 DataLoader；
- 一个 accumulation window 只包含同一 task；
- 只激活对应 Adapter；
- segmentation retention 最大允许 positive Dice 下降 1 个绝对百分点；
- 若端到端 factuality 无显著提升，最终模型保持顺序推理，不运行联合优化。

### 18.6 Resume 和 initialize

```text
--resume：同一 stage、同一 run，恢复 optimizer/scheduler/RNG/cursor
--initialize-from：进入新 stage，只加载允许的模型权重，重建优化器
```

不得混用，不得以复制 checkpoint 改变角色。

---

## 19. 推理阶段

固定流程：

```text
1. 解析 canonical parent 或用户输入
2. 验证 modality metadata、coverage、units、sign
3. 构建 reference/support views 和 sensor cards
4. 官方 Qwen3-VL native multi-image forward
5. Qwen 输出 target status、boxes 和 <seg>
6. mask decoder 输出 individual masks
7. pixelwise max 得到 semantic mask
8. mask 映射到各 view
9. exact-mask pooling + RoI replay
10. desc_adapter 输出 visual fact JSON
11. deterministic fact compiler 合并几何
12. 生成 compiled report
13. 保存所有 diagnostics 和 provenance
```

正式输出：

```text
semantic_mask
individual_masks
grounded_boxes
mask_iou_scores
target_status
active_modalities
visual_fact_record
deterministic_facts
compiled_report
provenance
```

不得输出：

- oracle matched proposal；
- attention weight 作为证据；
- 未校准的 physical quantity；
- 不可追溯的自由报告。

---

## 20. Loss 和指标

### 20.1 分割 loss

初始配置冻结为：

```text
L = 1.0 * L_language
  + 2.0 * L_mask_bce
  + 2.0 * L_mask_dice
  + 0.5 * L_iou
```

规则：

- BCE/Dice 只在 valid mask 内计算；
- no-target 依赖严格空 object list 和空 semantic mask；
- 首版无 boundary loss；
- 无 PMRD relevance loss；
- 无 learned reliability loss；
- 无 oracle loss。

### 20.2 描述 loss

```text
L_desc = mean_per_sample_causal_loss
```

不同 task 先各自归一化，再按配置权重相加。自动文本权重不得高于 Gold/Silver。

### 20.3 分割指标

至少：

```text
overall IoU/Dice
positive-only IoU/Dice
no-target specificity
empty-mask false-positive rate
component-region recall
target-area coverage
boundary F-score
bbox recall/IoU
GT-box / predicted-box / end-to-end
modality-subset robustness
peak VRAM
latency
```

### 20.4 区域理解指标

至少：

```text
target-status macro-F1
field-value F1 by provenance
same-image region retrieval
unsupported claim rate
UFCR
mask swap sensitivity
region swap sensitivity
modality removal sensitivity
cross-parent modality swap
expert factuality
calibration
raw JSON validity（工程指标）
```

Caption overlap 和 LLM judge 只能作为补充，不能单独证明 grounded understanding。

---

## 21. Baseline 与公平对照

### 21.1 旧系统

旧 tag/branch 运行：

```text
SANE-QMEF-PMRD-MGRR historical accepted
```

不修改旧 checkpoint，不迁移成新格式。

### 21.2 新系统 baseline

```text
B0 reference view only
B1 native multi-image
B2 B1 + sensor card
B3 B2 + valid/null
B4 B3 + aligned support residual

R0 crop/single-vector
R1 global + GAR RoI replay
R2 R1 + exact-mask per-sensor tokens
```

### 21.3 外部分割 baseline

- Mask2Former/Detectron2 specialist；
- SAM2 box-prompt；
- PSALM-Lite；
- Qwen3-VL-Seg-style。

Mask2Former 官方仓库已归档，主体为 MIT，部分依赖为 Apache-2.0；不应成为新主线的整库依赖。fileciteturn85file0L13-L16 fileciteturn85file0L45-L52

### 21.4 公平条件

固定：

```text
same raw-source parent population
same split
same Qwen3-VL-2B base
same active modality conditions
same pixel/token budget
same target canvas
same LoRA rank and layers
same optimizer steps
same seeds
same checkpoint selection metric
same GT/fixed/end-to-end evaluation
```

旧 v2 与新 v3 通过 source fingerprint 生成 `legacy_overlap_manifest.json`。旧模型在旧 branch 运行，结果按 source ID 对齐；新主线不得实现旧 reader。

正式科学结论使用三随机种子。G0 和开发消融可先单 seed，但不得写成最终结论。

---

## 22. 代码工程规范

### 22.1 配置

- YAML + Pydantic 是唯一配置真值；
- CLI 只允许白名单 override；
- 每次运行保存 resolved config；
- 禁止 YAML、Python 常量、CLI 三份维护；
- protocol version 只在 schema/manifest 定义，不在多处复制。

### 22.2 公共 API

所有 public API：

- 类型标注；
- docstring；
- shape contract；
- device/dtype contract；
- 明确异常；
- 不读文件。

### 22.3 文件与错误

禁止：

- 机器绝对路径；
- silent fallback；
- `except Exception: continue`；
- forward 中读文件；
- forward 中 SciPy/NumPy；
- tensor layout 猜测；
- 未记录 resize；
- 未记录 coordinate normalization；
- 自动覆盖 accepted artifact；
- old class alias；
- old benchmark compatibility reader；
- 同一功能两个 CLI。

所有 JSON/JSONL：

```text
UTF-8
allow_nan=false
atomic write
stable ordering
schema validation
```

### 22.4 测试

- unit tests：schema、transform、mask、valid、parser、loss；
- integration tests：single/multi-image forward、segmentation、description；
- regression tests：hash、split、duplicate、checkpoint reload；
- smoke tests：CPU synthetic + GPU one-batch；
- micro overfit：32–64 samples；
- no long training in Codex session。

---

## 23. 外部项目复用和许可证

| 项目 | 借鉴算法 | 借鉴工程 | 可直接复用 | 只能参考/限制 | 许可证与冲突 |
|---|---|---|---|---|---|
| Qwen3-VL | 原生多图、2D grounding、DeepStack | processor/chat template/model loading | 官方 Transformers model/processor 依赖 | 不复制整库；中间特征只经 wrapper/hook | 代码 Apache-2.0；权重许可证 P0 单独核验 |
| Detectron2 | evaluator/model input-output contract | dataset/model/loss/evaluator 分离、config baseline | evaluator 思路、少量通用 utility 经许可 | 不整库嵌套 | Apache-2.0；与现代 Qwen 栈版本需隔离 |
| Mask2Former | masked attention、matching、proposal decoder | baseline config/model zoo | 仅 G0/PSALM fallback 的最小实现 | 仓库 archived，不作为主依赖 | 主体 MIT，部分 Apache-2.0 |
| SAM2 | box/point prompt segmentation baseline | builder/config/checkpoint/predictor 分离 | 可作为独立 optional baseline | 不加入视频、tracking，不进主模型 | Apache-2.0；torch/CUDA 扩展可能冲突 |
| ms-swift | PEFT/QLoRA、stage init、resume | CLI lineage、日志、dry-run | 无运行时代码复用 | 不嵌套、不维护第二 Trainer | Apache-2.0；更新快，版本漂移风险高 |
| PSALM | LMM-updated mask tokens、proposal/classification decoupling | multi-task input schema | G0 fallback 可移植最小 decoder | 不复制 LLaVA/Swin 全栈 | Apache-2.0 |
| GAR | mask prompt、global context、RoI replay | region evaluation思路 | 可移植 prompt/RoI 小段并保留 attribution，优先自行实现 | 不复制 AnyRes、XTuner、完整数据管线 | Apache-2.0；原依赖与 Qwen 栈冲突 |
| MIGRANT | task taxonomy、data-centric two-stage curriculum | data/QA/eval 目录组织 | 不需要代码复用 | 不复制 ms-swift/PowerPaint vendor | MIT |
| RSGPT | RSICap/RSIEval 数据职责 | 高质量 caption 与独立 test | 解析项目负责人提供的本地科研副本 | 代码根 LICENSE 未发现，禁止复制；数据公开发布不在 P1 范围 | 仅保留 citation/provenance，不作 runtime 许可判断 |
| EarthGPT | MMRS 子集和任务格式 | source-specific technical audit | 解析冻结的本地科研子集 | 根 LICENSE 未发现，不得复制代码；数据公开发布不在 P1 范围 | 仅保留 citation/provenance，不作 component 授权表 |
| Qwen3-VL-Seg | box-guided 17M decoder | GT-box/predicted-box 分层评价 | 独立按论文实现 | 未确认官方代码；第三方复现不视为官方 | P0 记录论文和独立实现来源 |

Apache-2.0 与 MIT 代码通常可在保留通知条件下组合；当前 greenfield 根代码许可证已由项目负责人
在 P0 接受为 Apache-2.0。该代码许可证不对 raw data 或未来数据包公开发布作结论。

---

## 24. P0–P8 实施路线总表

| Phase | 目标 | 关键输出 | 阶段验收 |
|---|---|---|---|
| P0 | 实时审计与设计冻结 | inventory、provenance、ADR、backup plan | 人工批准 ADR 和备份 |
| P1 | Canonical Benchmark v3 | parent schema、builder、split、validator | small errors=[]、hash stable |
| P2 | 最小模型骨架 | native Qwen multi-image、Adapter、state | single/multi smoke、cache equivalence |
| P3 | G0 分割内核选择 | box/GT-box/pred-box/PSALM pilot | ADR-0002 明确只选一条 |
| P4 | 分割完整实现 | S1 train/eval/export | overfit、reload、no-target、24 GiB |
| P5 | Description Benchmark | Evidence Packet、facts、review package | auto/expert 分离、API resumable |
| P6 | GAR-lite | exact mask + RoI + desc adapter | swap/removal/unsupported tests |
| P7 | 训练与评价入口 | S1/S2/S3、lineage、3-seed aggregator | 人工可启动正式训练 |
| P8 | 彻底清理 | deletion manifest、README、AGENTS | 全仓库无旧引用 |

---

## 25. 每阶段输入、输出、验收和删除项

### P0：实时审计和设计冻结

**输入**

- 当前 worktree；
- 公开 HEAD；
- 本任务书；
- 旧 reports/checkpoints；
- 外部代码/依赖项目及其许可证（不含 raw data license 判断）。

**实现**

1. `git status --short`、HEAD、branch、untracked、ignored audit；
2. 生成完整文件树；
3. 类、函数、CLI、配置、测试和 artifact inventory；
4. 检查本地 `external/`；
5. 生成 reuse/deletion matrix，并记录最小 raw-source provenance；
6. 写 ADR-0001；
7. 提出 backup 命令。

**输出**

```text
docs/audits/repo_inventory.json
docs/audits/reuse_matrix.md
docs/audits/deletion_plan.yaml
docs/adr/ADR-0001-greenfield-rewrite.md
REFACTOR_PROGRESS.md
docs/handoffs/P0.md
```

**备份方案**

```bash
git tag -a pre-sami-rewrite-2026-07-20 <approved_sha>
git branch baseline/sane-qmef-pmrd-mgrr <approved_sha>
git switch -c refactor/sami-groundsegdesc
```

Codex 不自动 push。若 worktree dirty，停止并要求人工决定是否先 commit；禁止自动 stash/reset。

**验收**

- inventory 覆盖全部 tracked/untracked/ignored relevant files；
- 外部代码/依赖许可证无“猜测”；raw data 不在 P0 作许可判断；
- current accepted artifacts 有路径和 hash；
- 人工书面批准 ADR 和 tag/branch。

**删除项**

P0 不删模型代码。

### P1：Canonical Benchmark v3

**输入**

- raw source registry；
- minimal provenance registry；
- v3 schemas；
- old benchmark 仅用于 cross-check。

**实现**

- scanner；
- materializer；
- transform engine；
- duplicate grouping；
- group split；
- task expansion；
- language subset；
- validator；
- summary。

**输出**

```text
benchmark/sami_landslide_v3/small/
reports/p1/
```

**验收**

- small 全流程完成；
- `validation_report.errors == []`；
- 两次构建 aggregate hash 一致；
- verified cross-split duplicates = 0；
- mask/box transform round-trip 通过；
- padding/nodata exclusion 通过；
- 不包含 pre/post/change 字段；
- 不读取旧派生 Benchmark；
- provenance report 对全部物化 source/component 绑定可重放。

**删除项**

把 `scripts/1-benchmark` 至 `scripts/5-segdesc` 和旧 data readers 加入 deletion manifest；P1 验收后新代码不再调用它们。

### P2：模型最小骨架

**输入**

- v3 small；
- Qwen3-VL-2B；
- official processor；
- model config。

**实现**

- `SensorAwareMultiImageAdapter`；
- `QwenBackboneWrapper`；
- typed states；
- sensor cards；
- missing/zero-valid；
- view ordering；
- optional cache writer/reader；
- one-forward CLI。

**验收**

- optical-only、SAR-only、terrain-only、multi-view 均可 forward；
- list 顺序变化不改变 view identity；
- modality dropout 不泄漏；
- zero-valid 不制造视觉 token；
- spatial grid 可重建；
- cache 等价；
- Profile S 峰值记录。

**删除项**

旧 Qwen vision cache/controller 加入 deletion manifest；新模型不得 import。

### P3：G0 分割内核选择

**输入**

- P2 state；
- 固定 200–300 parent pilot；
- legacy report；
- G0 configs。

**实现**

- strict box parser；
- GT-box decoder；
- predicted-box；
- multi-box；
- no-target；
- PSALM-Lite pilot；
- memory profiler；
- decision report。

**验收**

- 人工执行 formal G0 命令；
- report 完整；
- ADR-0002 明确唯一内核；
- 非选内核从 formal config/CLI 移除。

**删除项**

非选内核列入 P8 deletion；不得保留 hidden auto-switch。

### P4：分割完整实现

**输入**

- G0 chosen kernel；
- T1/T2；
- S1 config。

**实现**

- seg LoRA；
- decoder/loss；
- valid mask；
- sampler；
- checkpoint；
- resume/init；
- evaluator；
- visualization；
- prediction export。

**验收**

- 32–64 sample overfit；
- checkpoint reload logits/masks 一致；
- resume cursor/RNG 一致；
- no-target 输出空 mask；
- invalid pixels不参与 loss/metric；
- 无 oracle output；
- GPU smoke；
- peak <=22 GiB；
- README 有人工正式命令。

**删除项**

旧 SANE/QMEF/PMRD/model assembly 和旧 segmentation CLI 在 replacement commit 后删除；baseline branch 保留。

### P5：滑坡语义标注和 Description Benchmark

**输入**

- v3 parents/masks；
- ontology v2；
- API provider configs；
- reviewer registry。

**实现**

- deterministic fact extractor；
- Evidence Packet；
- dual-provider adapters；
- idempotent API job store；
- raw response store；
- field verifier；
- cycle grounding；
- review package；
- canonical fact merge；
- derived task generator。

**验收**

- auto/Silver/Gold 严格分离；
- API dry-run；
- resume 不重复成功调用；
- 无单位时定量声明被拒；
- source view/evidence region 可追溯；
- review package 可人工填写；
- expert val/test 只来自 Gold。

**删除项**

旧 Bridge/Unified/description benchmark builder 列入删除并停止调用。

### P6：GAR-lite 区域描述模型

**输入**

- P2 backbone；
- P5 facts；
- GT/fixed masks。

**实现**

- global token；
- exact-mask token；
- RoIAlign；
- per-view tokens；
- null/coverage；
- desc adapter；
- raw structured generation；
- fact compiler；
- summary。

**验收**

- GT mask forward/train/generate；
- fixed predicted mask；
- mask swap 敏感；
- region swap 检出；
- modality removal 后对应字段为 unavailable/insufficient；
- unsupported claim 可统计；
- batch size 1 smoke；
- 不 import MGRR；
- 不调用 SciPy/NumPy connected components。

**删除项**

旧 MGRR、region baselines、旧 desc controller/CLI 删除。

### P7：训练和评价入口

**输入**

- S1/S2/S3 config；
- P4/P6 models；
- v3 task manifests。

**实现**

- minimal Trainer；
- checkpoint lineage；
- resume/init；
- S1/S2/S3 commands；
- GT/fixed/end-to-end eval；
- counterfactual suite；
- baseline comparison；
- 3-seed aggregator；
- memory/time report。

**验收**

- dry-run；
- one-batch smoke；
- terminal checkpoint；
- strict reload；
- config/data/model fingerprint；
- formal commands由 README 发布；
- Codex 不运行长训练。

**删除项**

旧 engine、training/evaluation protocols、旧 run scripts、旧 config 删除。

### P8：彻底清理

**输入**

- P1–P7 acceptance reports；
- deletion manifest；
- baseline tag/branch。

**实现**

- 执行全部 approved deletion；
- 清理 imports/requirements/docs/tests；
- 更新 README、AGENTS；
- 扫描旧类名、旧命令、旧 protocol；
- license/NOTICE；
- final tree audit。

**验收**

以下搜索必须为空，或只出现在 deletion/history report 中：

```text
SensorAwareNativeScaleEncoder
QwenGuidedEvidenceFusion
ScaleAwareDeformableAggregator
ProposalSetMaskRefinementDecoder
MultiGranularityRegionReplay
qwen_psalm_full
qpsalm_segdesc
multisource_landslide_v2
qpsalm_description_v2
landslide_region_description_v1
```

活动主分支不存在 old CLI、old reader、compat shim、legacy package。README 只含新命令。

---

## 26. 风险与回退策略

| 风险 | 触发条件 | 回退 |
|---|---|---|
| box bottleneck | predicted-box 保留率不足 | G0 选择 PSALM-Lite，只换模块二 |
| native multi-image 无增益 | B1≈B0 | 开启单次 aligned support residual |
| renderer 丢失物理信息 | render-only 在关键模态显著失败 | 增加共享浅层 BandSetResidual，必须单独 ADR/消融 |
| 多图显存超限 | Profile S >22 GiB | 减 view/pixel budget、冻结 vision、cache；不加新 backbone |
| GAR-lite region 对应弱 | swap/retrieval 不通过 | 增加一个 inside-minus-local-context token；不恢复完整 MGRR |
| 多组件描述不足 | 全局 mask 描述混乱 | 按预测 boxes 分区域描述，再程序汇总 |
| missing modality 过度声称 | removal 后仍声称 | 增加 unavailable negatives 和 field rules，不先加 reliability |
| joint 损伤分割 | retention drop >1% | 取消 S3，保持顺序推理 |
| expert Bridge 延迟 | Gold 未冻结 | 只发布工程结果，不宣称 expert factuality |
| 新模型不及旧模型 | 三 seed 显著退化 | 从 tag 恢复一个经证实必要的 SANE-lite 或 QMEF-lite；一次只恢复一个 |

---

## 27. 单卡算力计划

硬约束：

```text
GPU peak <= 22 GiB
Qwen3-VL-2B
NF4
vision tower frozen
BF16 trainable heads
batch size 1
gradient accumulation
no full-parameter pretraining
```

预估 trainable components：

```text
sensor adapter: 0.5–2M
mask decoder: 12–17M
region reader/projector: 0.5–2M
two LoRA adapters: several million
```

正式数字由 P2/P4 parameter report 计算，不在文档中虚构。

Codex 可运行：

- CPU unit tests；
- synthetic integration tests；
- 1–10 step GPU smoke；
- 32–64 sample micro overfit；
- memory profile。

Codex 不运行：

- S1/S2/S3 formal；
- three-seed full；
- paid API batch；
- expert merge。

---

## 28. 最终完成定义

项目只有同时满足以下条件才称为“重构完成”：

1. v3 small canonical Benchmark 从 raw source 独立构建、验证零错误、重复运行 hash 一致；
2. G0 已人工冻结唯一 segmentation kernel；
3. S1 程序完成 micro overfit、reload、no-target 和 24 GiB gate；
4. Evidence Packet、field annotation、review 和 canonical facts 可闭环；
5. GAR-lite 通过 mask/region/modality counterfactual smoke；
6. S1/S2 正式训练命令可由人工直接运行；
7. GT/fixed/end-to-end evaluation 均存在；
8. baseline 和三 seed aggregator 已实现；
9. 旧主线代码、配置、CLI、reader、cache protocol 已从活动分支删除；
10. README/AGENTS 只描述当前系统；
11. greenfield 代码的 LICENSE/NOTICE 与数据 provenance/data usage 记录完整；未来数据包公开发布审查保持独立；
12. 未把工程门禁写成科学成功。

“正式论文模型完成”还需人工运行三 seed S1/S2、冻结 Gold expert val/test，并通过最终科学门。

---

## 29. `REFACTOR_PROGRESS.md` 模板

```markdown
# REFACTOR_PROGRESS

## Current status
- phase:
- phase_status: not_started | in_progress | blocked | engineering_passed | human_accepted
- current_branch:
- current_commit:
- baseline_tag:
- baseline_branch:
- dirty_worktree:
- task_spec_version:
- active_adr:

## Objective
...

## Scope for this session
### Allowed
- ...
### Explicitly excluded
- ...

## Changes
### Files added
- ...
### Files modified
- ...
### Files deleted
- ...

## Commands executed
| command | start/end | exit code | result artifact |
|---|---|---:|---|

## Tests
| test | status | evidence |
|---|---|---|

## Smoke / micro-overfit
- config:
- device:
- steps:
- peak_vram:
- result:

## Data and artifact bindings
- benchmark:
- manifest_sha256:
- config_sha256:
- checkpoint:
- checkpoint_sha256:

## Blockers
- ...

## Human action required
- ...

## Next exact command
```bash
...
```

## Next phase scope
- ...

## Known technical debt
- ...
```

---

## 30. 阶段 handoff 模板

```markdown
# Handoff: P<phase>

## What was completed
...

## What was not completed
...

## Acceptance evidence
- report:
- status:
- errors:
- warnings:
- commit:

## Public contracts introduced or changed
- schema:
- API:
- CLI:
- config:

## Deleted paths
- ...

## Preserved read-only artifacts
- ...

## Known risks
- ...

## Human decisions
- decided:
- pending:

## Start point for next session
1. read:
2. inspect:
3. run:
4. do not:
```

---

## 31. ADR 模板

```markdown
# ADR-XXXX: <decision>

- Status: proposed | accepted | superseded | rejected
- Date:
- Owners:
- Phase:
- Commit:

## Context
...

## Decision
...

## Alternatives considered
1. ...
2. ...

## Evidence
- experiment/report:
- license/dependency:
- hardware:

## Consequences
### Positive
- ...
### Negative
- ...

## Implementation constraints
- ...

## Rollback
- tag/branch:
- trigger:
- procedure:

## Human approval
- approver:
- date:
```

---

## 32. Deletion manifest 模板

```yaml
version: sami_deletion_manifest_v1
baseline_tag: null
baseline_branch: null
entries:
  - path: SEG_Multi-Source_Landslides/qpsalm_seg/models/sane.py
    kind: file
    owner_phase: P4
    replacement: src/sami_gsd/model/sensor_adapter.py
    delete_after:
      - p2_adapter_accepted
      - p4_segmentation_accepted
      - baseline_tag_verified
    references_checked: false
    tests_removed_or_replaced: []
    docs_removed_or_replaced: []
    license_notes: null
    approved_by: null
    deleted_commit: null
```

---

## 33. 最终决策表

| 类别 | 决策 |
|---|---|
| 已冻结决策 | greenfield rewrite；三个一级模块；Qwen3-VL-2B；原生在线多图 forward；自研最小 Trainer；canonical Benchmark v3；旧系统只存 tag/branch；不兼容旧协议 |
| 待 G0 决定 | Qwen3-VL-Seg-style 或 PSALM-Lite；Profile S/M；是否启用 aligned support residual |
| 需要人工决定 | baseline tag/branch；项目根代码许可证；G0 formal 运行；付费 API；专家审核/仲裁；正式训练；最终阈值；未来 publication/release |
| Codex 可自主决定 | 内部函数拆分、类型名的非公共细节、测试 fixture、纯重构、日志字段的附加非语义信息；不得改变冻结接口 |
| 明确禁止 | pre/post/change；tracking/video；完整第三方仓库嵌套；双 Trainer；旧 reader/shim；silent fallback；oracle output；自动正式训练；自动付费 API；自动专家结论；无单位定量物理声称 |
