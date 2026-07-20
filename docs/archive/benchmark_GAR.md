# 多源滑坡分割—区域描述统一模型：Benchmark 与算法实施方案

> 文档状态：实施基线 v2；工程状态复核于 2026-07-18
>
> 当前前置能力：Landslide Benchmark V2、SANE、QMEF、PMRD、Qwen mask-query controller
>
> 本阶段目标：在不破坏现有分割路径的前提下，建立可审计的整图描述、区域对齐和滑坡区域描述能力。

当前实现状态（代码状态不等于科学验收）：

| 阶段 | 工程状态 | 仍需人工完成的门槛 |
|---|---|---|
| M0/M1/M1.1 | Description Small v4 已重建并通过工程验证，verified cluster 不跨 split | 保留 RSIEval 数量 warning；未授权 Full |
| M2 | Bridge v7 prepare 与 300-parent Pilot review package 已通过工程验证 | 当前为 `awaiting_expert_review`；完成两名专家审核、必要仲裁并冻结 Pilot gate |
| M3 | task-neutral state 与 Description Vision Cache v1 M3 v3 migration/deep validation 已通过 | 保留旧 cache 与 segmentation Vision Cache v3 只读证据，不得覆盖 |
| M4 | 六种 region encoder 消融、MGRR 多粒度 token sequence 与反事实接口已实现 | Small 三 seed 消融、ERFS、retrieval 和 UFCR 门槛 |
| M5 | `desc_adapter`、causal generation、raw parser/repair 与 D-1 工程门禁已通过 | D-1 不替代 M4 专家或科学门禁 |
| M6 | D0 Small 已完成；D0-D4、GT/fixed/end-to-end、OOF replay、事实性和 paired CI 已实现 | 下一阶段为 D1；随后运行 D2、D3a，专家阶段等待 M2 冻结 |
| M7 | 同任务梯度累积的独立 DataLoader 交替训练、严格 full-val retention 与三种子聚合 gate 已实现 | 从通过 M6 的 checkpoint 初始化并让三条独立 seed 链全部通过 full-val retention |

因此，当前仓库仍是 **M0-M7 engineering-complete candidate**：D0 的完成证明训练闭环可运行，
但不能替代 M2 人工审核、Small 科学实验和固定统计门槛，也不能据此进入 Full。

---

## 一、研究目标与范围

下一阶段不是将 MMRS-1M 的全部任务加入现有模型，而是完成以下闭环：

```text
多源遥感输入
    -> 滑坡分割或用户指定区域
    -> 区域级多源证据提取
    -> 可审计结构化事实
    -> 自然语言摘要
```

第一版研究：

1. 遥感整图描述；
2. box、mask、full-image 指定区域理解；
3. 滑坡 GT mask、referring mask 和预测 mask 描述；
4. optical、multispectral、SAR、terrain、deformation 证据约束；
5. 无目标、缺失模态和低可信证据下的拒绝或保守描述；
6. 分割与描述共享 Qwen 基础模型、顺序执行的统一推理。

首版明确不做：

- MMRS 分类、通用检测、普通 VQA 和红外任务；
- 灾前灾后变化描述和多区域关系推理；
- 未经证据支持的成因、触发因素、活动加速或未来失稳预测；
- 将 DIOR-RSVG 短 referring phrase 当作详细区域 caption；
- 将归一化特征值伪装成有物理单位的测量值。

---

## 二、依据、审计事实与待验证假设

文档结论必须使用以下标签之一：

- **论文报告**：论文或官方项目明确报告；
- **本地审计**：从当前本地文件直接统计；
- **本项目设计**：为本研究提出的新模块或协议；
- **待验证**：仍需实验或人工审核确认。

### 2.1 论文依据

#### EarthGPT / MMRS-1M

**论文报告**：EarthGPT 使用 MMRS-1M 统一多传感器、多任务遥感理解。MMRS-1M 覆盖 caption、VQA、分类、检测、visual grounding 和 region-level caption 等任务。

本项目只采用五个光学 caption 数据源，以及 DIOR-RSVG 的 box-to-text 和 text-to-box 任务视图。

论文：<https://arxiv.org/abs/2401.16822>

#### RSGPT / RSICap / RSIEval

**论文报告**：RSICap 包含 2,585 条人工详细遥感图像描述；RSIEval 用于整图 caption 和 VQA 评价。

论文：<https://arxiv.org/abs/2307.15266>

#### Grasp Any Region

**论文报告与代码核对**：GAR 保留完整图像上下文，并通过 RoI-aligned feature replay 将指定 bbox 的视觉特征插入语言模型序列。

论文：<https://arxiv.org/abs/2510.18876>

以下能力不是 GAR 原实现的直接复现，而是**本项目设计**：

- exact-mask 多尺度池化；
- context-ring token；
- 多源模态证据 token；
- valid-mask 与 GSD 感知；
- batch-safe 多源 region replay；
- 与 SANE/QMEF/PMRD 的顺序式分割—描述衔接。

### 2.2 本地数据审计基线

以下数字来自当前本地数据。构建程序仍必须重新统计并写入报告，不得硬编码为验证条件。

| 数据源 | 本地审计结果 | 解释 |
| --- | ---: | --- |
| RSICap annotations | 2,585 | 2,585 个唯一 image_id 和 filename |
| RSICap instruction | 4 种 | 官方 `text_input` 的四种表达 |
| RSICap image files | 3,000 | 415 张未被 caption 标注引用 |
| RSIEval images/captions | 100 / 100 | 仅作为 test |
| RSIEval local QA | 943 | 与 RSGPT README 所述 936 存在差异 |
| MMRS caption parents | 46,275 | 五个 caption JSON 的 parent 总数 |
| DIOR-RSVG parents | 15,709 | 每个 parent 有两个相反方向 source record |
| DIOR-RSVG source JSON records | 31,418 | 15,709 box-to-text record + 15,709 text-to-box record |
| DIOR-RSVG expanded task turns | 61,640 | 30,820 box-to-text + 30,820 text-to-box；单个 source record 可含多轮 |
| DIOR-RSVG valid region pairs | 30,809 | 排除 11 个因两位小数坐标退化为零宽或零高的非法源 bbox |
| Landslide V2 Small parents | 5,561 | train/val/test parent 隔离 |
| Landslide V2 referring targets | 31,998 | 位置、尺度、形态和 no-target 目标 |
| Landslide V2 instructions | 39,136 | instruction 数，不等于 parent 数 |

审计注意事项：

1. RSIEval 相同 question/answer 出现在不同图像上时，不属于重复样本；只允许在同图或相同 image hash 内去重。
2. DIOR-RSVG 单条 JSON record 可能包含多轮 region pair，必须展开全部轮次。
3. DIOR 的 box-to-text 答案通常是 `The tiny vehicle` 一类短 referring expression，只提供区域语义对齐和指代词汇监督。
4. Landslide V2 的 materialized `.npy` 可能已经归一化，不能默认恢复原始物理单位。

---

## 三、数据职责与最终 Benchmark

### 3.1 数据职责

| 数据 | 核心训练 | 主要职责 | 不承担的职责 |
| --- | --- | --- | --- |
| RSICap | 是 | 人工详细整图描述和风格校准 | 滑坡专业事实、区域定位 |
| RSIEval Caption | 否，test-only | 独立整图描述测试 | 早停、阈值选择、prompt 调参 |
| RSIEval VQA | 否，test-only | 能力保持检查 | 描述模型选择 |
| MMRS Caption | 是，选择性 | 场景、目标和词汇广度 | 最终详细描述风格 |
| DIOR-RSVG box-to-text | 是，辅助 | box 到短 referring expression 对齐 | 详细 region caption |
| DIOR-RSVG text-to-region | 是，辅助 | 短语到同图候选区域的反向检索 | 自由坐标 bbox 回归、PMRD 通用目标分割 |
| Landslide V2 | 是 | 精细 mask、多源语义、referring/no-target | 时序因果和触发因素 |
| 专家 Bridge | 是，核心 | 专业滑坡区域描述和事实性评价 | 未审核伪标签 test 真值 |

### 3.2 `rs_global_caption_v1`

逻辑组成：

```text
rsicap_train
rsicap_dev
mmrs_caption_train
mmrs_caption_dev
rsieval_caption_test
rsieval_vqa_test
```

规则：

1. RSICap 优先按经审计确认的 source-scene ID 分组，再固定划分 90% train、10% dev；同一源场景的 patch 不跨 split。
2. 文件名前缀（如 `P1384`）只能作为候选 scene group。只有在 M0 中验证其稳定对应原始 DOTA 场景后才能作为 fallback，并将依据写入审计报告。
3. 当前本地 RSICap 与 RSIEval 候选场景前缀无重叠，构建程序仍必须使用最终 scene group 再验证。
4. RSIEval 永久 test-only，优先级高于任何训练来源。
5. MMRS 同一 parent 的去重参考答案全部保留；训练期可复现地采样一条，评价使用全部参考。
6. `rsicap_train_all` 只能作为模型和超参数冻结后的 final-fit manifest，不进入常规开发流程。

### 3.3 `rs_region_alignment_v1`

DIOR-RSVG 统一解析为 `region_pair`，再建立两个 task view：

```text
region_referring_expression: image + box -> short phrase
region_grounding: image + short phrase + same-image candidate regions -> matched region
```

规则：

1. 保存原始归一化 bbox 和像素半开区间 `[x0, y0, x1, y1)`；
2. 同一 parent 的全部 region pair 和双向任务必须同 split；
3. 正反方向是同一监督事实的两个视图，不能统计为两个独立 parent；
4. 首版反向任务只做同图候选区域检索或对比对齐，不建立自由坐标 bbox 回归 head，不混入滑坡 PMRD 主损失；
5. 同图重复短语、非法 bbox、零面积和越界 bbox 单独标记。

### 3.4 `landslide_region_description_v1`

| region source | 定义 | 监督可信度 |
| --- | --- | --- |
| `gt_global_mask` | parent 完整滑坡语义 mask | 高 |
| `pseudo_instance_component` | 二值 mask 连通域 | 中，不等同真实实例 |
| `gt_referring_mask` | Landslide V2 referring target | 高，但文本属性多为程序派生 |
| `no_target` | 无目标或 referent 不存在 | 高 |
| `predicted_proposal` | 固定 checkpoint 的 PMRD proposal | 仅课程学习和端到端评价 |
| `user_region` | 交互推理输入 | 无训练真值 |

Bridge 输出：

```text
auto_train
expert_train
expert_val
expert_test
```

专家 val/test 必须 parent 级独立，且不能使用未经审核的教师文本作为真值。

### 3.5 `multisource_landslide_segdesc_v1`

统一索引引用现有 segmentation、Description V2 和 Landslide Bridge，不重复复制图像或 mask。
统一训练调度使用以下顶层 `task_group`：

```text
segmentation
global_caption
region_alignment
region_description_auto
region_description_expert
```

component 内部仍保留更细的 `task_family`，例如 `region_referring_expression`、
`region_grounding`、`landslide_region_structured_description`、`landslide_region_caption` 和
`no_target_response`。首版不包含 `landslide_region_vqa`；只有获得独立、可验证的 VQA 标注后
才新增。

### 3.6 存储策略

正式 benchmark 使用**入选 parent 图片物化模式**：

- source JSONL 保存原始 datasets 逻辑路径、图像 hash、尺寸和 provenance；
- split 冻结后只复制入选 parent 图片，不复制 MMRS 的其他任务或未入选图像；
- 最终模型索引只引用 `benchmark/qpsalm_description_v2_<mode>/data/...`；
- 图片保持原始字节、编码和尺寸，不重采样、不转码；
- `external/RSGPT`、`external/Grasp-Any-Region-main` 和 `../datasets/MMRS-1M` 保持只读。

目录协议固定为：

```text
data/<split>/<source_slug>/<parent_sample_id>.<source_suffix>
```

同一 parent 的多个 task view 共用一份物理图片。source indexes 与 provenance 保留
`datasets/...` 作为数据血缘，但训练、验证、推理和描述视觉缓存不得运行时回退到源图片。

---

## 四、统一描述 Schema

描述样本使用独立 schema：

```text
qpsalm_description_v2
```

### 4.1 示例

```json
{
  "schema_version": "qpsalm_description_v2",
  "sample_id": "rsicap_p0378_0001__global_caption",
  "parent_sample_id": "rsicap_p0378_0001",
  "source_dataset": "RSICap",
  "split": "train",
  "task_family": "global_caption",
  "visual_ref": {
    "type": "single_image",
    "path": "benchmark/qpsalm_description_v2_small/data/train/rsicap/rsicap_p0378_0001.png",
    "storage_mode": "materialized_copy",
    "width": 512,
    "height": 512,
    "sha256": "..."
  },
  "region_geometry": {
    "type": "full_image",
    "mask_path": null,
    "bbox_xyxy_normalized": null,
    "bbox_xyxy_pixel_half_open": null,
    "coordinate_space": "original_image"
  },
  "target_status": "present",
  "region_source": "full_image",
  "instruction": "Describe this remote sensing image in detail.",
  "answer_type": "natural_caption",
  "answers": [{
    "text": "...",
    "language": "en",
    "annotation_origin": "human",
    "quality": 1.0,
    "caption_quality_weight": 1.0
  }],
  "structured_targets": {},
  "provenance": {
    "annotation_path": "datasets/RSGPT/dataset/RSICap/captions.json",
    "source_image_path": "datasets/RSGPT/dataset/RSICap/images/P0378_0001.png",
    "original_record_id": "0",
    "license_status": "academic_only"
  },
  "quality_flags": []
}
```

### 4.2 `visual_ref`

允许：

```text
single_image
multisource_parent
```

`single_image` 保存图像路径和尺寸；运行时必须通过统一适配器转换为一个光学 `ModalityInstance`：

```text
family = optical
sensor = source_dataset_specific 或 generic_aerial_rgb
product_type = rgb
band_names = [R, G, B]
units = display_rgb
native_gsd_m = unknown
aligned_gsd_m = unknown
valid_mask = all ones over decoded pixels
quality = caption/image audit quality
```

适配器不得伪造 GSD、轨道、物理单位或传感器型号。解码失败、alpha-only、灰度伪 RGB 和无有效像素必须在索引验证阶段报错。

`multisource_parent` 保存 Landslide V2 benchmark 与 parent sample ID，由现有 resolver 读取模态和 valid mask。

不使用单个 `sensor_family` 表示多源样本，模态语义以 Landslide V2 parent 的结构化 metadata 为准。

### 4.3 `region_geometry` 与 `target_status`

`region_geometry.type`：

```text
full_image
box
mask
null
```

`target_status`：

```text
present
absent
uncertain
```

二者必须分离：非空预测 mask 可能是 false positive；no-target referent 没有区域；full-image caption 也不需要落盘全 1 mask。

运行时规则：

- `full_image`：在有效区域内懒生成全 1 mask；
- `box`：懒生成矩形 mask，同时保留 bbox 类型；
- `mask`：读取精细 mask 并与 reference canvas 对齐；
- `null`：生成全 0 region mask 和 null-region token。

### 4.4 `structured_targets`

每个字段保存：

```json
{
  "value": "moderately_elongated",
  "value_space": "categorical",
  "source": "deterministic_mask_geometry",
  "confidence": 1.0,
  "units": null,
  "evidence_modalities": []
}
```

允许的 `value_space`：

```text
physical
normalized_relative
categorical
unavailable
```

模型输入不得包含作为监督答案的 `structured_targets` 文本。输入 mask 可派生 geometry token，但报告时应说明这些字段不是模型从影像中估计得到。

### 4.5 provenance 与 license

每条记录必须包含原始 annotation/image 路径、record ID、构建器版本、hash、annotation origin、license source/status 和 quality flags。

RSGPT 图像来自 DOTA，按官方说明仅限学术用途。MMRS 各组成数据集许可需逐源审计；缺失时标记 `license_unknown`，禁止假定可以再分发。

### 4.6 描述本体、语言与输出约束

正式实现前冻结 `description_ontology_v1`，计划保存为：

```text
configs/description_ontology_v1.yaml
configs/qpsalm_description_output_v1.schema.json
```

ontology 至少定义：

```text
target_status
location
size_class
shape
elongation
compactness
fragmentation
surface_observation
terrain_support
sar_support
deformation_support
surrounding_context
evidence_sufficiency
confidence
```

每个字段必须声明允许值、连续值到类别的阈值、`unknown/unavailable`、允许的文本同义词、监督来源、是否允许模型预测以及对应评价方式。确定性 geometry 与模型 observations 使用不同 provenance，不能混成一个 F1。

首版训练、生成和正式评价语言固定为英文。中文仅作为后续翻译或应用展示，不与英文 benchmark 指标混算。

结构化生成必须保存三层结果：

```text
raw_generation
parse_status + schema_errors
deterministically_repaired_generation
```

允许的确定性修复仅包括去除代码围栏、截取唯一 JSON object、补齐 schema 中有默认 `unavailable` 的缺失字段和规范化已登记同义词。禁止根据 GT 或图像补写内容。主结构指标使用原始生成：非法 JSON 计为对应字段错误；修复后结果只作为二级工程可用性指标，同时报告 raw/repair invalid rate。

Caption 原文保持不变，训练权重由审计结果单独给出：

```text
1.0  human_and_verifiable
0.5  partially_verifiable_or_weak_inference
0.0  clearly_unsupported_or_audit_only
```

`caption_quality_weight=0.0` 的文本保留在审计索引中，但不进入主训练采样。

---

## 五、去重与 Split 协议

### 5.1 固定顺序

必须先去重聚类，再生成 split：

```text
source records
    -> source visual parents
    -> SHA-256 exact groups
    -> perceptual near-duplicate candidates
    -> RGB64 MAE verification
    -> verified/exact connected components
    -> canonical visual parent selection and caption merge
    -> source scene/group constraints
    -> split assignment
    -> task view expansion
```

不得先展开多条 instruction/caption，再按 instruction 随机划分。

canonical caption 的每个去重答案必须保存 `source_answer_index`、原文 SHA-256 和
source record provenance。verified cluster manifest 必须逐成员记录 canonical 选择、source
declared split、最终 split 和 split action；含官方 test 成员的整个簇只能进入 test。

### 5.2 冲突优先级

同一或近重复图像出现在多个 split 时，保留优先级为：

```text
RSIEval test
> expert Bridge test
> expert Bridge val
> source official test
> dev
> train
```

发生冲突时删除或迁移低优先级训练引用，不修改 test 真值。

### 5.3 感知 hash 原则

感知 hash 只生成候选簇，不能单独触发合并。大面积农田、机场和住宅区可能视觉相似，但并非同一图像。

M1.1 使用确定性的二阶段验证：dHash 完全相同负责召回，候选图统一转换为 RGB 64x64，
逐通道 MAE 不超过 3.0 才标记为 `verified_near_duplicate`。verified 连通簇在 split
之前合并为一个 canonical parent，caption 合并为多参考答案并保留 answer-level provenance。
scene group 只约束 split，禁止把同一场景的不同裁剪合并为同一图像。

报告区分：

```text
exact_duplicate
verified_near_duplicate
possible_near_duplicate
same_source_scene
```

### 5.4 Small 与 Full

Small 不只取索引前 N 条：

- RSICap 保留全部 2,585 parent；
- RSIEval 保留全部 test；
- MMRS caption 按 source、场景、caption 长度和去重簇分层抽取 10,000–15,000 parent；
- DIOR-RSVG 按 parent 和 region 数量分层抽样；
- Landslide Bridge pilot 默认 300 parent，覆盖数据源、模态组合、region source 和 target status。

Full 使用清洗后的全部可用 parent，但仍遵守 split 和 license 约束。

---

## 六、数据构建程序

沿用现有编号式 pipeline，但合并重复职责。

### 6.1 `scripts/3-description/`

```text
3-1_scan_description_sources.py
3-2_build_global_caption_index.py
3-3_build_region_alignment_index.py
3-4_deduplicate_and_split.py
3-5_materialize_description_images.py
3-6_validate_description_benchmark.py
3-7_summarize_description_benchmark.py
description_common.py
```

总控入口：

```text
scripts/run_3_build_description_benchmark.sh
```

#### `3-1_scan_description_sources.py`

只读扫描：

- `../datasets/RSGPT/dataset/RSICap`；
- `../datasets/RSGPT/dataset/RSIEval`；
- `../datasets/MMRS-1M/json/caption`；
- `../datasets/MMRS-1M/json/RSVG/rsvg_trainval.json`；
- 对应图像目录。

同时兼容旧的 `external/RSGPT/dataset` 布局，并允许用
`PAPER7_RSGPT_DATA_ROOT` 显式覆盖。索引始终写可移植逻辑路径，不写机器绝对路径。

明确禁止读取：

```text
../datasets/MMRS-1M/json/total.json
classification
detection
VQA
infrared
```

扫描输出覆盖路径、解码、尺寸、caption 长度、空文本、多轮结构、bbox、source split、hash 和 license 状态。

#### `3-2_build_global_caption_index.py`

- RSICap 保留原始四种 instruction 和 canonical instruction；
- MMRS 每图一个 parent，多参考 caption 存入 answers；
- 只删除同 parent 内完全重复答案，不跨图删除相同短句；
- 原文不静默改写，语法异常和低信息 caption 使用 quality flag；
- RSICap 中天气、季节等难验证陈述标记 `low_verifiability`；
- 根据第四节协议写入 `caption_quality_weight`，权重为 0 的原文保留在审计索引但不进入主训练。

#### `3-3_build_region_alignment_index.py`

- 展开 DIOR-RSVG 每条 record 的全部 conversation pair；
- 将正反向记录归并到同一 region pair ID；
- 保存 normalized 和 pixel-half-open bbox；
- 区分短语、类别、尺寸/位置修饰词；
- 标记歧义短语和重复 box。

#### `3-4_deduplicate_and_split.py`

执行第五节协议，验证并合并同图重编码，输出 immutable split manifest、verified duplicate
manifest、canonical selected source records 和 selected parent source manifest。它不发布模型侧最终索引。

#### `3-5_materialize_description_images.py`

- 每个 selected parent 复制一份图片到固定的 `data/<split>/<source>/<parent>` 路径；
- 使用 `.part` 临时文件、复制时 SHA-256 和原子替换；
- 已存在且 hash 正确的图片可复用，错误文件必须重新复制；
- 所有图片成功后才发布 `all/train/dev/test`、component indexes 和最终 parent manifest；
- 写入 materialization manifest/report，并清理仅位于当前 benchmark `data/` 内的未登记文件。

#### 验证和汇总

至少检查：

- 图像路径存在且可解码；
- 最终模型索引不得引用 `datasets/...`，且图片必须位于当前 benchmark 的 `data/`；
- materialization、parent、task records 和实际文件一一对应；
- 不存在未登记文件、符号链接或残留 `.part` 文件；
- caption/phrase 非空；
- bbox 有效且坐标协议一致；
- parent、scene、duplicate cluster 不跨 split；
- verified perceptual cluster 只发布一个 canonical parent；
- `train_eligible.jsonl` 不包含零权重答案，完整审计索引仍保留原始答案；
- RSIEval 只在 test；
- DIOR 双向 view 同 split；
- 不读取 `total.json`；
- provenance 和 license 字段存在；
- 本地 943 QA 与官方统计差异写入 warning，而不是自动删除。

### 6.2 `scripts/4-landslide-bridge/`

```text
4-1_inventory_regions.py
4-2_extract_region_facts.py
4-3_build_candidate_descriptions.py
4-4_build_review_package.py
4-5_merge_expert_reviews.py
4-6_validate_landslide_bridge.py
landslide_bridge_common.py
```

总控入口：

```text
scripts/run_4_build_landslide_bridge.sh
```

职责：

1. 统计 global/referring/component/no-target region；predicted region 延后到 M6；
2. 生成确定性几何和条件允许的多源证据；
3. 生成规则化候选文本，提供可插拔离线教师接口；
4. 导出多源面板、mask overlay、可编辑 JSON/CSV；
5. 合并 accept/revise/reject 和双人仲裁结果；
6. 验证文本、结构字段、模态和 split 一致性。

### 6.3 `scripts/5-segdesc/`

```text
5-1_build_unified_index.py
5-2_validate_unified_index.py
5-3_summarize_unified_index.py
```

总控入口：

```text
scripts/run_5_build_segdesc_dataset.sh
```

统一索引只保存 component benchmark 引用和任务采样元数据，不再次复制源样本。每条引用同时
绑定 component index 的 SHA-256、精确 JSONL 行号和 record ID；component manifest 还绑定三个
输入 benchmark 的 validation report hash。`qpsalm_segdesc_index_builder_v3_component_contract_bound`
进一步把 Landslide V2 final stage/root 及 instruction validation、Description M1.1 v4 以及
当前 Bridge M2 v7 合同写入 manifest；validator 会重新读取实际 report 独立核对，而不是只信
manifest 自报字段。旧 Bridge
v4-v6 和旧 unified v2 即使自身 `errors=[]` 也不得作为当前输入。Bridge 处于
`awaiting_expert_review` 时只允许发布
`region_description_auto`，即使目录中残留旧 `expert_all.jsonl` 或 evaluation gate，也必须忽略
并在报告中明确记录。只有 Bridge validation 为 `expert_pilot_frozen`，且当前 Bridge 的
`evaluation_gate_manifest.json` 通过路径、hash 和协议校验后，才允许发布
`region_description_expert`。dry-run 只执行 5-1 统计，不调用依赖已发布索引的验证和汇总。

### 6.4 统一脚本约束

每个入口必须支持：

```text
--dry-run
--max-samples
--overwrite
--seed
--output-dir
```

并满足：

- 中文关键注释；
- 文件头包含用途、运行命令、输入、输出和写入行为；
- 非零错误退出；
- 不修改原始数据；
- 临时文件加原子替换；
- validation `errors == []` 才允许进入下一阶段。

### 6.5 单图适配与描述视觉缓存

描述数据加载层提供唯一入口：

```python
build_single_image_modality_instance(visual_ref) -> ModalityInstance
```

RSICap、RSIEval、MMRS Caption 和 DIOR-RSVG 必须经过该入口，不能在各 Dataset 内分别实现 resize、归一化或伪造 metadata。单图和多源 parent 最终都进入 `encode_multisource`，但保留各自 source type 和恢复变换。

描述任务建立独立缓存协议：

```text
qpsalm_description_vision_cache_v1
```

缓存规则：

1. 支持 `source_type=single_image|multisource_parent`；
2. 以 parent、源图/渲染内容 hash、Qwen model/processor revision、renderer version 和空间变换作为 key；
3. 只缓存任务无关的视觉空间特征与 view token，不缓存 instruction、region mask、region token、segmentation query 或答案；
4. 同一 parent 的多个 caption、box、mask 和 no-target 任务共享视觉缓存；
5. 缓存 manifest 保存 source type、原图尺寸、grid/padding transform、特征层和完整 revision；
6. 保留现有 `qpsalm_qwen_vision_cache_v3` 供分割训练使用，不覆盖、不就地升级，也不要求重新生成；
7. 只有 manifest 和 view hash 完全兼容时，才允许显式转换 v3 中的多源视觉 payload；转换结果写入新目录并记录来源，禁止将两个协议当作同一 cache 静默复用。

描述缓存验证必须覆盖单图、多源 parent、同 parent 多任务复用、内容 hash 变化、错误 revision 和丢失 source view。

当前 builder 为 `description_vision_cache_m3_v3_shard_content_bound`，validation 为
`qpsalm_description_vision_cache_validation_v2_shard_content_bound`。manifest 必须为每个 shard
保存路径、字节数、record 数和 SHA-256；深度验证和训练期首次加载都核验 shard 内容，因此即使
tensor shape、lookup 和来源 metadata 均未改变，任一 tensor 数值位翻转也会失败。旧 v2 cache
不能被运行时直接读取；只允许一次 side-by-side 严格迁移：逐 shard 补齐 SHA/size/record count，
逐 record 校验 shape/finite/fingerprint/lookup/task-neutral 字段，再按当前 Description v4、Bridge
v7 和 segmentation cache v3 重放每个 parent 的 `source_content_hash`。全部一致且同文件系统时
才允许 hardlink 复用并记录 inode audit；否则必须完整重建。迁移不修改旧 cache，也不引入宽松
runtime fallback。staging 验证不能直接代表成功：发布到正式目录后必须再次遍历全部 shards，
重放 source/target inode、旧 cache report hash 和 segmentation cache v3 的 build-time snapshot，
只有终态报告仍有效时 migration 才返回成功。
当前 migration report 协议为
`qpsalm_description_vision_cache_migration_v2_published_replay_bound`；旧的无发布终态 replay 报告
不得进入 readiness。
公开 `qpsalm_description_vision_cache_v1` format 与
`task_neutral_parent_visual_features_v1` 协议不变。
首次 D-1 前的手工 artifact 批次还必须发布
`qpsalm_segdesc_artifact_readiness_v2_training_consumable`：它重新打开 Description v4、
Bridge v7、Unified v3、M3 artifact origin 和全部 cache shards，核对当前 component validation、
Unified all/split population、expert publication 状态与嵌套文件 SHA。只有 readiness 报告为
`status=engineering-valid`、`ready=true`、`errors=[]` 才可进入 D-1；Bridge 为
`awaiting_expert_review` 时必须仍为 0 expert、gate 未冻结，readiness 不提升其科学状态。
当前 D-1 overfit 必须显式传入该报告；训练入口先完整重算报告，再把
`qpsalm_segdesc_artifact_readiness_acceptance_v1_live_replayed` 写入 dataset summary、checkpoint
data audit 和 overfit report。后续 D-1/D0 门禁绑定原报告字节及其嵌套文件 SHA，不能只依赖脚本
执行顺序或手工声称 Unified v3 已发布。
严格迁移 origin 必须重放旧 cache manifest/report hash 和每个 hardlink 的 source/target inode；
若内容漂移后改为完整重建，则只接受无 migration metadata 的当前 M3 v3 builder artifact。
运行时 reader 必须要求已发布且成功的 `validation_report.json`，并逐项核对当前 manifest SHA、
输入指纹、component/record/shard 数、全部 shard bytes/hash 与 source cache 隔离统计。builder
深度扫描阶段只能通过显式内部参数在报告发布前打开 cache；训练、评价和 demo 不允许绕过。
`qpsalm_description_vision_cache_artifact_binding_v1_validation_bound` 将 cache 目录、manifest、
validation report、输入/source provenance 和 shard inventory 写入每个 SegDesc checkpoint。
正式 M4/M6 使用
`qpsalm_description_vision_cache_artifact_revalidation_v1_checkpoint_bound` 重开该目录，并以
`qpsalm_description_vision_cache_shard_replay_v1_sha256_complete` 重放全部 shard。仅在同一验收
进程内目录文件 size/mtime/ctime/inode 快照完全未变时允许复用一次全量 hash 结果。

---

## 七、滑坡区域事实提取

### 7.1 确定性几何

从 mask 和 valid canvas 计算：

```text
centroid
bbox
valid_area_ratio
perimeter
aspect_ratio
elongation
compactness
orientation
fragmentation
component_count
distance_to_valid_boundary
absolute_position
```

这些字段是程序计算结果，不作为“模型视觉推理能力”计分。模型负责将其组织为可读文本，并与视觉/物理证据结合。

### 7.2 证据可信度三级协议

#### Level A：physical

仅在以下条件满足时允许定量描述：

- 能读取原始物理数据；
- units 已知；
- band/polarization/product_type 已知；
- sign convention 已知或不涉及符号；
- GSD/空间对齐支持区域统计；
- valid coverage 达到配置阈值。

#### Level B：normalized relative

只有归一化特征时，只允许描述区域内外相对高低、异常强弱、纹理或反射差异和有效覆盖率。不得附加米、度、毫米/年、dB 等单位。

#### Level C：unavailable

单位、符号、波段或覆盖不足时，结构字段为 unavailable；文本输出 `insufficient evidence` 或省略该模态句子。

`target_status=absent` 是更强的 no-target 约束：六个 region 字段必须全部为
`unavailable`，三类模态 support 只能为 `insufficient_evidence/unavailable`，总体证据充分性只能
为 `insufficient/unavailable`。自动候选、双人修订、仲裁结果和最终 Bridge validator 使用同一
约束，禁止把整幅场景先验或邻域观测写成不存在目标的区域事实。

### 7.3 模态约束

#### Optical / Multispectral

- RGB 允许颜色、亮度、纹理和区域内外对比；
- NDVI 仅在明确存在校准 red/NIR 时计算；
- NDBI 仅在明确存在 SWIR/NIR 时计算；
- 未知 band order 禁止推断指数。

#### SAR

- VV/VH 必须区分线性幅度、功率和 dB；
- ratio/difference 必须符合 value encoding；
- 升降轨分别记录；
- 归一化后无法逆变换时仅描述相对异常。

#### Terrain

- elevation meter 仅来自原始 DEM；
- slope 等级仅来自有单位 slope，或由已知 GSD/单位的 DEM 合法推导；
- sample-level robust normalized DEM 不输出绝对高程。

#### InSAR

- 保存 LOS、units、sign convention 和 coverage；
- `source_defined` sign 不解释为抬升或沉降；
- 单时相或单速率图不描述加速趋势；
- 形变异常只是观测证据，不能直接等价为活动滑坡。

### 7.4 候选文本与专家审核

规则化文本和教师文本都只是 candidate：

```text
structured facts
    -> deterministic text candidate
    -> optional teacher candidate
    -> expert review
    -> accepted/revised/rejected annotation
```

Bridge pilot 默认 300 个 parent，按数据源、模态组合、region source、面积和 target status
分层，并精确满足 train/val/test = 180/60/60。任一 split 候选不足均为正式验证错误；
`--max-samples` 生成的缩小版本只属于 smoke，不允许冻结 evaluation gate。

专家 val/test：

- 至少两名审核者；
- 不一致样本仲裁；
- 保存原始 candidate、修改后文本和状态；
- 报告字段一致率、文本接受率、修改率和争议字段分布；
- 两名审核者的分类字段报告 Cohen's kappa；超过两名审核者或存在缺失标注时报告 Krippendorff's alpha；
- parent 不得进入 auto/expert train。

### 7.5 自动 Bridge 与专家 Bridge 的职责

Bridge 监督分为两层，不能把自动事实和专家自然语言混成同一种真值：

```text
auto_train
    = deterministic geometry
    + protocol-valid physical/relative evidence
    + target/modality availability

expert_train/val/test
    = accepted or revised structured fields
    + geoscience wording
    + evidence sufficiency
    + conservative summary
```

`auto_train` 可以覆盖全部合法 Landslide V2 mask，但只包含可追溯字段和规则化文本；教师自由文本仍是 review candidate。`expert_val/test` 只使用双人审核并完成仲裁的记录。模型训练时不得将 `structured_targets` 的答案文本放入 prompt。

---

## 八、统一模型接口

### 8.1 现有分割路径保持不变

现有模型：

```text
SANE -> QMEF -> PMRD
```

必须继续支持：

```python
output = model(modality_batch)
```

现有分割 checkpoint 必须可加载。新增描述模块不能改变默认 segmentation forward 的输出语义。

### 8.2 状态拆分

#### `MultisourceBackboneState`

任务无关，保存：

- SANE 逐模态 detail/high/mid/low 金字塔；
- 各尺度 valid mask；
- active modality subset；
- reference canvas 和恢复 transform；
- Qwen vision-cache token 与 family metadata。

不得把 segmentation instruction-conditioned QMEF fused feature 作为其唯一视觉状态。

#### `SegmentationState`

任务相关，保存：

- segmentation `SemanticEvidence`；
- QMEF evidence；
- PMRD proposal/query/relevance；
- proposal 到 final union 的映射。

#### `RegionEvidenceState`

由 `MultisourceBackboneState + RegionPrompt` 构建，保存：

- global context token；
- exact-mask token；
- RoI replay token；
- context-ring token；
- geometry token；
- modality evidence/null token；
- coverage、reliability 和 source diagnostics。

### 8.3 新增公开方法

```python
state = model.encode_multisource(batch)
segmentation = model.segment_from_state(state, segmentation_prompt)
description = model.describe_from_state(state, region_prompts)
```

默认 `forward(ModalityBatch)` 内部可以调用这些方法，但不返回大体积 state。统一推理显式复用 state，避免 SANE 重复编码。

### 8.4 `RegionPrompt`

至少包含：

```text
region_id
geometry_type
mask_or_box
target_status
region_source
instruction
active_modalities
```

首版每个 decoder sample 一次只描述一个 region；batch 内不同样本可以使用不同 region。`gt_global_mask` 由数据层展开为一个 global summary prompt 和若干主要 component prompt，按独立 region 顺序生成，不在首版做多 region 关系推理。

---

## 九、MGRR：Multi-Source Grounded Region Replay

MGRR 是**本项目设计**，借鉴 GAR 的全局上下文与 RoI feature replay 思想，但不复制其 PerceptionLM、AnyRes 或 XTuner 框架。

### 9.1 全局上下文

来源：

- Qwen global view token；
- SANE low/mid 的 valid-aware pooling；
- 可用模态和场景尺度 embedding。

全局 token 描述道路、河谷、植被和整体地形背景，但不能覆盖或替代 region token。

### 9.2 Exact-mask token

对每个模态和尺度执行：

```text
inside = sum(mask * valid * feature) / sum(mask * valid)
outside/local contrast = inside - context_ring_pool
```

空 mask 不执行无效除法，直接返回显式 null-region token。

### 9.3 RoI replay token

根据 mask bbox 或输入 box，在原始 per-modality feature 上执行 RoIAlign。初始网格：

```text
detail: 7 x 7
high:   7 x 7
mid:    4 x 4
low:    2 x 2
```

网格加入尺度和模态 embedding，再由固定数量 learnable region queries 压缩；不能直接将全部格点送入 Qwen。

坐标转换必须使用 dataset resize/pad transform，而不是只按特征宽度缩放。

### 9.4 Context ring

```text
ring = dilate(mask, adaptive_radius) - mask
```

ring radius 根据 region 面积和 canvas 大小自适应，并设上下限。ring 与 valid mask 相交，只用于区域内外对照。

### 9.5 多源融合

MGRR 不直接复用 segmentation query 的融合结果，而是用 region query 对逐模态、逐尺度 token 做可靠性融合：

- coverage 为 0 的模态不参与；
- 低质量模态保留但降低 prior；
- 缺失模态使用 family-specific null token；
- reliability 和最终 token attention 写入 diagnostics。

### 9.6 Token 预算与实际序列

首版每个 region 使用 12–20 个 token：

```text
2 global
4-8 local replay
2 inside/contrast
2 context ring
1 geometry
1-5 modality evidence/null
```

具体数量通过 Small 消融确定，不能只依据显存选择。

工程实现使用变长 `region_sequence_tokens`，而不是先把所有证据压成单一 region vector。present
区域依次包含 region summary、global context、exact mask、replay、context、geometry、逐模态 token
和主要 component replay slots；absent 区域使用 global/null-region/geometry/null-evidence。batch
内仅在 token 维 padding，并保存 `region_sequence_mask`，不得用预分配切片写入破坏梯度链。
当前 MGRR replay 协议为 `qpsalm_mgrr_v2_multiscale_grid_replay`：在已应用 renderer
resize/pad transform 的 reference canvas 上确定 bbox，再对 detail/high/mid/low 原生特征执行
`7×7/7×7/4×4/2×2` 的 `grid_sample`，加入尺度 embedding 后由两个可学习 region query
压缩。residual components 仅做 exact-mask 聚合，禁止使用跨多个残余斑块的 union bbox。
每个主要 component token 显式组合 inside、multiscale RoI replay 与
`inside - component_context_ring`；`mgrr_no_context` 同时关闭整区和组件 contrast，
`roi_replay_only` 只保留空间 replay，确保消融名称对应真实信息路径。
8 邻域组件只在每个 region 解析一次，所有模态共享同一组 component slots；禁止在逐模态循环
中重复连通域分析，以免产生重复 CPU 同步或模态间 slot 漂移。

### 9.7 必须比较的 baseline

```text
crop-only
full-image + box coordinates
single-vector masked pooling
RoI replay only
MGRR without context ring
full MGRR
```

工程参数依次为 `crop_only`、`full_image_box`、`masked_pooling`、`roi_replay_only`、
`mgrr_no_context` 和 `mgrr`。各组必须从同一上游 checkpoint 初始化并使用相同 split、seed、
训练预算与输出协议。

六路 Assisted 对照必须接收相同的确定性 geometry 输入，不能只给 full MGRR 注入 geometry；
Vision-only 的 `crop_only`/`masked_pooling` 则保持无答案字段，`full_image_box` 仅保留其 baseline
定义所需的 box 坐标。`full_image_box` 在 no-target/null box 上仍必须读取完整有效视野并编码
显式 no-box 状态，不能因 region mask 为空而退化成 learned null-only baseline。

工程门禁将同一 seed 的六条 encoder 链从共同 D1 checkpoint 分叉，并分别运行 D2-D3a-D3b；
checkpoint 保存 `qpsalm_description_stage_lineage_v3_run_completion_bound`。每个 lineage entry 除
选择角色外还绑定源 run 的成功 completion report、所选 best/last 与 validation selection；中断
run 的孤立 best 不得进入下一阶段。正式 full-MGRR 配对会核验共同 D1 SHA、
每一上游 stage 的受控 config/data audit、当前训练 population 与只允许变化的 region encoder。
三 seed 聚合协议为 `qpsalm_description_seed_gate_v12_strict_json_finite`，还要求跨
seed 的 expert/retrieval population、scientific config 和训练 population 一致。单 seed 内的完整
loader audit 必须逐字一致；跨 seed 则先验证 bridge/DIOR/global-caption loader 分别使用
`seed+11003/21013/31019`，随后仅规范化 loader/sampler 的运行局部 seed，样本总体、task pattern、
batch/sampler 合同、验证总体和 frozen Bridge 绑定不得变化。旧 v4 artifact-only gate 与 v10
直接比较 run-local loader seed 的 gate 都不能作为 M4 科学准入。

五种 baseline 的当前 seed gate 必须再由
`qpsalm_m4_region_encoder_suite_v8_strict_json_finite` 聚合。每个 formal artifact 还必须绑定
当前 ontology、record schema 与 output schema 的字节级规格。聚合器从每份 gate 的 input bindings 重算
原始 eval/retrieval/ERFS，要求五份 2/3 seed gate 全部通过、共享同一 frozen Bridge、同一组三个
full-MGRR candidate checkpoints 和同一份规范化 training population。只比较 crop-only 或任意单一
baseline 不构成 M4 完成。

### 9.8 多连通区域 replay

`component_mask`、`gt_referring_mask` 和单个 `predicted_proposal` 可以使用单一 bbox 的 RoI replay。`gt_global_mask` 可能包含相距很远的滑坡斑块，禁止对其 union bbox 只执行一次 RoIAlign。

`gt_global_mask` 使用固定策略：

1. 对完整 mask 生成一个 exact-mask global token；
2. 对 8 邻域连通域按有效面积降序排列；
3. 依次保留主要 component，直到累计覆盖至少 90% 的有效目标面积，最多保留 8 个；
4. 每个主要 component 生成独立 RoI、inside/contrast 和 context-ring token；
5. 未进入 top components 的小区域汇总为 `residual_components` token，并记录数量和面积覆盖；
6. 输出由一条全局滑坡摘要和主要 component 的独立描述组成，不把 component 当作经人工确认的真实滑坡实例。

报告必须包含 component coverage、截断数量和 residual area ratio。若只有一个有效连通域，该路径退化为普通单区域 replay。

端到端 `segment -> describe` 评价不得仅按 parent 选择一个全图 segmentation instruction。
`gt_referring_mask`、no-target 和与 referring mask 完全同掩膜的伪 component 必须通过
`source_region_aliases.sample_id -> parent_referring_target_sample_id` 精确映射到对应指令。
没有 referring alias 的 `pseudo_instance_component` 没有可复现的语言定位目标，只参加
GT-mask oracle 与固定 mask 描述评价，不参加正式端到端评价。报告必须给出源 Bridge 行数、
可映射行数、排除原因、实际映射类型和唯一分割推理次数，禁止静默回退到全图 mask。

---

## 十、描述控制器与输出协议

### 10.1 共享 Qwen、双 Adapter

同一个 Qwen3-VL-2B 量化基础模型承载：

- 现有 PEFT `default` adapter：逻辑上的 segmentation adapter；
- 新增 `desc_adapter`：自回归描述。

保留 `default` 名称以兼容当前分割 checkpoint，不执行 adapter 重命名迁移。

每个 batch 只激活一个 adapter：

- segmentation batch 激活 `default`；
- description batch 激活 `desc_adapter`；
- 首版不做 adapter fusion。

统一模型表示共享基础模型和视觉状态，不表示一次 Qwen forward 同时输出 mask 与文本。

### 10.2 描述输入序列

```text
<SYSTEM>
<TASK>
<GLOBAL_CONTEXT>
<REGION_START>
<REGION_GEOMETRY_ASSISTED or REGION_GEOMETRY_TYPE_ONLY>
<LOCAL_DETAIL>
<NEIGHBOR_CONTEXT>
<OPTICAL_EVIDENCE or NULL>
<MULTISPECTRAL_EVIDENCE or NULL>
<SAR_EVIDENCE or NULL>
<TERRAIN_EVIDENCE or NULL>
<DEFORMATION_EVIDENCE or NULL>
<REGION_END>
<ANSWER>
```

训练时 causal LM labels 只覆盖答案 token，system/task/region token 使用 `-100`。

### 10.3 混合输出

滑坡区域描述首版输出与 `configs/qpsalm_description_output_v1.schema.json` 保持一致：

```json
{
  "schema_version": "qpsalm_description_output_v1",
  "target_status": "present",
  "region": {
    "location": "upper_left",
    "size_class": "small",
    "shape": "elongated",
    "elongation": "high",
    "compactness": "moderate",
    "fragmentation": "few_components"
  },
  "evidence": {
    "surface_observation": "A disturbed surface is visible inside the region.",
    "terrain_support": "supports",
    "sar_support": "unavailable",
    "deformation_support": "insufficient_evidence",
    "surrounding_context": "The region lies on a vegetated slope.",
    "evidence_sufficiency": "partial"
  },
  "confidence": 0.72,
  "summary": "..."
}
```

规则：

1. Assisted 中 `region` 来自确定性 region 计算，单独报告，不宣称是视觉模型预测；
2. Vision-only 中 `region` 由视觉与区域 token 生成，必须与 Assisted 分开评价；
3. `evidence` 由模型生成并接受字段级监督；
4. 缺失模态字段必须是 unavailable/null；
5. `target_status=absent` 时明确拒绝，不继续生成滑坡属性；output schema 以条件约束要求
   六个 `region` 字段全部为 `unavailable`，三类模态 support 只能为
   `insufficient_evidence/unavailable`，evidence sufficiency 只能为 `insufficient/unavailable`；
   外部 `jsonschema` 与内置固定子集校验器执行相同
   语义；违反该条件的 raw output 整体作为 schema-invalid 进入结构字段和 status 的
   `invalid` 桶，不能仅凭写出 `target_status=absent` 获得 absent recall；即使结构合法，absent
   样本产生 unsupported evidence claim 仍计入 false-description rate；
6. global caption 使用自由文本，不强制套用滑坡 JSON。

### 10.4 Assisted 与 Vision-only 协议

区域描述必须分别实现并报告两种输入模式：

#### Assisted

- 输入由程序计算的 location、size、shape 和 fragmentation geometry token；
- 目标是可靠地组织确定性几何、视觉观察和多源证据；
- geometry 指标只评价 verbalization fidelity，不作为视觉理解增益。

#### Vision-only

- 只输入 region mask/box、geometry type、坐标变换和视觉特征；
- 不输入 location、size class、elongation、compactness、fragmentation 等离散答案；
- 目标是评价模型能否从空间特征和 region 约束自主恢复形态语义。

MGRR 的主要视觉理解结论以 Vision-only 为准；Assisted 用于工程报告能力。两种模式使用同一数据 split 和输出 schema，禁止在同一个指标表中混合样本。

### 10.5 Checkpoint 与 Adapter 迁移

现有分割 checkpoint 格式 `qpsalm_sane_qmef_pmrd_v5` 通过显式导入入口加载：

```python
load_segmentation_backbone_checkpoint(...)
```

迁移规则：

1. 只加载白名单中的现有 SANE、QMEF、PMRD、controller projection 和 `default` adapter 参数；
2. MGRR、description projection、special embeddings 和 `desc_adapter` 随机初始化并写入迁移报告；
3. 参数 shape、architecture spec 或 evidence protocol 不匹配时立即失败；
4. 禁止使用无白名单的 `strict=False` 静默跳过参数；
5. 新统一 checkpoint 使用 `qpsalm_segdesc_v1`，记录来源分割 checkpoint hash、两个 adapter 状态、ontology/schema 版本和描述 cache protocol；当前实现逐文件绑定 `description_ontology_v1.yaml`、record v2 schema 与 output v1 schema 的 SHA/bytes，resume/initialize 时重新计算，漂移后拒绝旧描述 checkpoint；
6. Adapter 名称固定为 `default` 和 `desc_adapter`，不执行隐式重命名。
7. 同一 stage 的 `resume` 必须要求 region encoder 和完整 state 完全一致；跨 stage 的
   `initialize-from` 只允许显式替换 region encoder，其他共享参数必须逐 key/shape 严格迁移，
   被重新初始化的 region keys 必须写入迁移报告。
   当前 checkpoint 还必须保存并恢复 Python/NumPy/Torch CPU/CUDA RNG。D0-D4 的每条 stream
   使用 `qpsalm_description_training_progress_v1_loader_cursor_bound`，联合阶段使用
   `qpsalm_segdesc_joint_progress_v3_parent_population_list_bound`；恢复时绑定有序 population、sampler、
   worker 配置、epoch 和 batch cursor，禁止从 epoch 0 重放已训练数据。
   D0–D4 与 M7 还必须用
   `qpsalm_segmentation_migration_lineage_v1_source_bytes_bound` 比较当前配置和源 checkpoint 的
   原始 segmentation SHA/format/step/白名单，并重新读取 migration source bytes；后续 stage
   不得悄悄换用另一分割基线。
8. 正式 M4/D4/M6/M7 验收不能只信任 evaluation JSON 中复制的 checkpoint metadata。
   `qpsalm_segdesc_checkpoint_provenance_v3_segmentation_lineage_bound` 必须以内存映射重放 checkpoint 本体，并逐字比较
   step、stage/lineage/config、segmentation migration、Adapter 清单、state-key inventory 和当前
   ontology/schema assets；还必须按 checkpoint 内的 artifact binding 重开 Description Vision
   Cache，核对 manifest/validation report 并重放全部 shard SHA。payload、cache 与报告任一不一致
   时门禁失败。

---

## 十一、训练协议

### D-1：最小正确性与过拟合

正式预适配前完成：

- Qwen zero-shot baseline；
- 固定 64 条 global/box/mask/null 混合样本、100 optimizer steps 过拟合；其
  `StageSpec.region_token_policy=mixed_explicit`，global 禁止区域路径，
  box/mask/null 必须消费区域证据；
- causal label mask 检查；
- region token 梯度检查；
- adapter 切换和 checkpoint reload；
- batch size > 1、不同图像尺寸和空 region。

D-1 的 64 条样本不是从单一 Bridge index 直接截断。当前协议按 seed 做确定性
分层并以 round-robin 顺序组成四路等额混合：M1.1 `global_caption/full_image`、M1.1
`region_referring_expression/box`、M2 有掩膜 candidate、M2 `no_target`。Bridge
`region_geometry` 中由 mask 推导的 bbox 不能冒充 box-conditioned 样本。M2 candidate
只用于工程过拟合，报告必须写明 `expert_truth_used=false`；它不构成专家评价。
输出格式与区域视觉路由必须解耦：M1.1 DIOR box 的训练目标可以是自由文本，但 batch 中必须设置
`use_region_tokens=true` 并实际消费 box/MGRR token；`structured_outputs` 只控制 JSON/free-text
response contract，不能再兼任视觉路由开关。
统一 `train d-minus-one` 入口将工程 overfit 固定为 64 条、100 optimizer steps，batch size
默认 2 且不得小于 2；`evaluate zero-shot` 固定选择 64 条。冲突的 CLI 覆盖在加载模型前拒绝，
避免以非预注册预算生成看似同名的 D-1 报告。

过拟合 run 结束后必须生成当前
`qpsalm_d_minus_one_overfit_validation_v10_structured_decoder_bound` 的
`d_minus_one_overfit_validation.json`，绑定四路 population、Description/Bridge validation
与 index 哈希、causal v5、task-path-aware gradient gate、`desc_adapter` 隔离、loss 下降、raw
JSON/schema/nonempty-summary smoke、严格 checkpoint reload、batch size > 1、多种原生尺寸、
null region 和不超过 24 GiB 的峰值显存。严格 reload 必须先改变一个 checkpoint 内的
`desc_adapter` LoRA 哨兵，并分别扰动 optimizer、scheduler、训练 RNG 及启用时的 GradScaler，
再通过正常 loader 恢复并核对各自状态指纹；
structured route 不再让 decoder 自由生成整段 JSON 语法，而采用
`qpsalm_description_structured_generation_v2_token_stream_bound`：固定 schema key/标点，
由 Qwen live logits 选择 enum 与自由文本 token，并在 decoding 期间执行 absent 条件约束。
最终 JSON 是模型 decoder 的 raw 输出，不允许从已生成文本做 repair，也不读取 GT；逐行 audit
必须绑定 raw SHA、实际 causal token-stream SHA、forced/model-selected token 数和字段终止原因；
token stream 与发布 raw JSON 必须逐字节相同。
该 protocol 进入 checkpoint architecture spec；旧 unconstrained D-1 checkpoint/report 不允许
原位补字段或 resume，必须从 segmentation checkpoint 新建 run。
当前梯度协议为 `qpsalm_description_gradient_gate_v4_window_homogeneous`。当前
`qpsalm_d_minus_one_task_path_batch_sampler_v1_window_homogeneous` 将连续
`grad_accum_steps` 个 microbatch 组织为单一 global 或 region 路径；纯 global window 必须证明
MGRR、spatial backbone 和 region projector 为零梯度，region window 必须证明这些模块获得非零
有限梯度。混合路径的梯度不得同时充当两条 path report。只有两条任务路径都被真实观察后，
D-1 梯度子门禁才完成，不能让随机首批样本的类别顺序决定 run 成败。
最终只读门禁必须逐 path 重放 required nonzero/zero module inventory、梯度计数、有限 norm 和
checks 全集；只有顶层 `passed=true`/`all_required_streams_checked=true` 的旧文件无效。
第 100 步 terminal checkpoint 只有在两条路径门禁完成后才能保存，并在 checkpoint metadata 中
嵌入同一完整 gradient proof。若进程在 checkpoint 已落盘、strict reload/报告发布前中断，
同 run `--resume` 从 checkpoint 重放该 proof 与 final validation artifact 后只完成终态发布，
不得伪造新的 optimizer step，也不得只信任目录中的独立 gradient JSON。
显存门禁必须观察到真实 CUDA 正峰值，CPU/缺失记录的 `0 GiB` 不得通过。history、resolved
config、dataset summary、gradient/trainable manifest、validation/raw generation、checkpoint
和显式 segmentation migration 均写入不可变哈希绑定；ontology、record schema 与 output
schema 也作为独立源文件绑定，并在后续门禁中与当前仓库字节级重验。checkpoint 必须以内存映射
重放当前 format/step/metadata 和实际 migration，再按其 architecture binding 核验 M3 manifest、
validation report 与全部 shard SHA；只伪造 reload 汇总不能通过。旧 v1-v7 报告不兼容。该报告
还必须逐 batch 检查 prefix/padding label mask、target 连续性与监督 EOS，将结果写入每条 history，
并由只读验证器从绑定的 history 重放；仅在汇总报告中声明 causal v5 不得通过。该报告只表示
overfit subgate，不能单独把 D-1 标为完成。原生 Qwen zero-shot report 还必须逐文件绑定
本地模型权重、tokenizer、配置字节，以及 M1.1 输入 population 与 raw generation；最后由
`qpsalm_d_minus_one_engineering_gate_v13_structured_decoder_bound` 从当前 M1.1 index 重建 zero-shot
population，逐张重开所选 benchmark `data/` 图像并核对 `materialized_copy`、登记 SHA 与 live SHA，
重验 Qwen 模型文件和 overfit 运行源，再核验两个 run 使用同一 M1.1 validation
report 和 seed，才允许 `d_minus_one_complete=true`。zero-shot 不声称 region capability，也不设置
人为性能阈值，只作为明确记录的基线。当前 gate 还把 CLI 成功发布的 `training_report.json` 纳入必需
证据：逐项重放其 artifact binding，要求终态 checkpoint 声明 `terminal_last`，checkpoint/
progress/history 共享同一最终 step，history 严格递增结束于该 step，并要求当前 overfit validation
正是完成报告所绑定的文件。训练在完成报告发布前中断、完成报告漂移或目录仍含
`failure_report.json` 时，即使 overfit report 表面通过也必须拒绝。

D-1 取样前还必须重放两条 live 数据链：
`qpsalm_description_engineering_audit_v1_cache_partition_bound` 验证 `train/dev/test` 是 cache
所绑定 `all.jsonl` 的精确分区，并从 train 重建正权重 `train_eligible`；
`landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound` 验证 live
`candidate_all` 与 Description Cache 的 `multisource_parent` 输入完全一致，再验证
`auto_train` 是 candidate train 的逐行投影且不含 expert truth。两项 audit 均进入 dataset
summary/checkpoint data binding，任何 cache 构建后的索引漂移都必须拒绝。

D0 是第一个正式预适配 stage，必须显式传入上述当前 D-1 gate。训练器深度重算 gate 后写入
`qpsalm_d_minus_one_acceptance_v11_structured_decoder_bound`，将 D-1 使用的 M1.1 benchmark root、
builder、validation report SHA、zero-shot materialized-image population SHA 和 overfit training
completion report SHA 固化，并将其逐
stage 保存到 checkpoint metadata 与 lineage；
D1–D4、M7 初始化/续训和最终 retention 都必须从原 zero-shot/overfit 源重新验证该 acceptance。
仅有 overfit subgate、旧 v1-v9 gate、编辑过的 gate 或已漂移输入都不能启动/延续正式课程。

正式 D0 前必须先发布 `qpsalm_d0_preflight_v6_region_route_bound`：重验 D-1、Description/Bridge、
当前 M3 cache 全 shard 和 segmentation migration，构建 model、dataset、collator、optimizer 与
trainable parameter manifest，但不得执行 backward 或 `optimizer.step`。只有报告满足
`status=engineering-valid`、`ready=true`、`optimizer_steps=0`、`errors=[]`，并且
`formal_training_launch.unique=true`，才允许执行报告发布的唯一正式 D0 训练命令。preflight
与正式入口必须用集中协议中的 `seed + 11003` 构造相同 D0 sampler，逐项重放 dataset
population、首批 collator tensor 和完整 stream binding；首批审计使用
`qpsalm_description_collator_audit_v3_output_format_region_route_separated`，必须直接消费训练 collator 的
`requests/instructions/target_texts/reference_texts/structured_outputs/use_region_tokens/metadata/region_masks/weights`
契约，不允许用 synthetic-only 的影子字段代替。该命令必须绑定
`qpsalm_d0_training_launch_v2_exact_command_bound` 的完整 argv 与 shell rendering；正式入口必须
从当前 config、Python executable、device、gate、output 和 report path 重建后逐字段相等，
只比较部分参数或信任可编辑的 command 字符串不得通过。该命令还必须绑定
preflight 原子写出的 resolved config SHA 和一个与 preflight 目录完全分离的正式 output-dir；
正式目录必须不存在或为空，发布命令不得携带 `--overwrite-output`。报告之外手工重构的近似命令
不属于验收协议；若目录在预检后被占用，应安全失败并重新预检，而不是覆盖已有 run。
正式 D0 还必须通过 `--d0-preflight-report` 绑定该报告；启动前重验 resolved config SHA、device、D-1
acceptance、Description/Bridge 工程 binding、cache 文件元数据快照及其 segmentation cache v3
source provenance，并把当前 acceptance 原子
写入训练目录。该 acceptance 使用
`qpsalm_d0_preflight_acceptance_v6_region_route_consumed`，并携带
`qpsalm_d0_construction_contract_v2_region_route_replayed`；正式 trainer 在首个 optimizer step 前必须
用实际 migration、dataset/collator/loader、trainable manifest 和 optimizer spec 重建后完全相等。
没有 ready report、报告或输入漂移、以及绕开统一入口的 D0 均不得训练。
preflight 不是训练结果，也不能代替 D0 checkpoint。

D-1/D0-D4 的 checkpoint、run completion report 与训练/验证 JSONL history 均采用同目录临时
文件替换发布；history 只允许单训练进程写入，若既有文件以不完整 JSON 行结尾则拒绝继续追加，
由显式 resume reconciliation 处理可恢复时间线。

### D0：MMRS Caption 场景预适配

目的：学习遥感场景和目标词汇。

只训练：

```text
desc_adapter
task-neutral visual projection
instruction/visual special embeddings
```

冻结 SANE、QMEF、PMRD、segmentation adapter、MGRR、region projector 和 region special
embedding。D0/D1 的 causal sequence 只使用 instruction token 与 parent-level task-neutral visual
tokens，不注入随机初始化的 exact-mask、component RoI 或 context-ring token。区域空间能力从 D2
开始训练，避免全图 caption 预适配被尚未校准的 region replay 污染。缓存读取也采用
`include_spatial=false` 快速路径，只加载 view tokens 和 valid mask，不搬运或投影四尺度 spatial
features。每次运行必须输出 `trainable_parameter_manifest.json`，逐组记录参数名、数量、学习率和
weight decay。对应 causal sequence 协议为
`qpsalm_description_causal_v5_stage_separated_schema_ordered`；训练 target 和 schema-constrained
raw generation 使用相同字段顺序，旧的 v4 及更早描述 checkpoint 不允许继续初始化 D0-D4，
但分割 checkpoint、segmentation Vision
Cache v3 和 Description Vision Cache v1 无需重建。

每个 parent 每个 epoch 采样一条参考 caption，受 source 和 caption quality 权重控制，避免 NWPU 数量支配训练。

### D1：RSICap 详细描述校准

起始比例：

```text
RSICap 70%
MMRS Caption 30%
```

RSICap 校准详细描述风格，MMRS 防止场景覆盖快速收缩。该比例是 preset 起点，需由 dev 结果验证。

### D2：DIOR Region Alignment

目标：

```text
box -> referring expression
image/box/text contrastive alignment
text -> same-image candidate-region retrieval
```

不将此阶段称为详细 region caption，不建立自由坐标 bbox 回归 head，不更新滑坡 PMRD 主损失。反向任务只在同一图像的标注 region candidates 中检索正确区域；自由坐标 grounding 延后到存在独立研究必要性时再评估。

D2 首次启用 description spatial backbone、MGRR、alignment text projection、phrase 使用的
instruction embedding 和 alignment temperature；不训练此路径未使用的 causal region projector、
region special embedding 或 visual special embedding。D3/D4 再联合启用完整描述模块。各 stage
的 optimizer 参数集合必须写入运行产物，跨 stage 使用 `initialize-from` 重建 optimizer，禁止沿用
上一阶段 optimizer state。

D2 训练 DataLoader 必须使用 parent-grouped batch sampler：先按 `parent_sample_id` 聚合并在组内
打乱区域，再将连续候选装入 batch，使同图不同 phrase/region 成为真实 hard negatives。相同 parent
且规范化 phrase 相同的重复标注使用 multi-positive target，不得互相作为假负样本。普通全局随机
shuffle 不能用于 D2 主实验。没有任何同图候选对的 batch 不进入 D2 optimizer step；单区域 parent
仍保留在完整 dev/test 对齐评价中，并单独统计其覆盖，不能将跨图 negatives 伪称为同图检索监督。

### D3a：自动结构化 GT-mask 预训练

使用全部满足 schema、valid mask 和证据协议的 Landslide V2 区域训练：

```text
target_status
deterministic geometry verbalization
available/unavailable modality state
protocol-valid physical or relative evidence
structured JSON fields
```

此阶段只使用 `auto_train` 中可追溯的结构字段和规则化文本，训练 MGRR、region projector 和
`desc_adapter`，冻结 segmentation adapter 和 PMRD。加载前必须以
`landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound` 重读当前 M2 validation、完整
`candidate_all` 与 `auto_train`，证明当前 builder、完整 Pilot、零错误、精确 train 投影以及所有
candidate 的 `is_expert_truth=false`，并证明 candidate 路径、字节数和 SHA 与当前 Description
Cache 的 `multisource_parent` 输入完全相同；live index/report/cache SHA 和 population hash 必须
进入训练 data binding。D3b/D4 打开 Bridge region stream 时同样执行该 cache/candidate 审计，
但仍额外要求冻结 expert gate。对于 Level C 证据，模型必须学习输出 unavailable/insufficient，而不是补全看似合理的
地学结论。

### D3b：专家 Bridge 文本校准

D3a 不使用人工 validation 选择 scientific best。它保存的完整训练终点必须在 checkpoint 内声明
`checkpoint_role=terminal_last`；D3b 初始化按该内部角色而非文件名核验，拒绝
`validation_best`、未声明角色或仅重命名得到的迁移源。
其他带 validation 的 D0/D1/D2/D3b/D4 阶段在跨阶段迁移与独立评价时必须声明并使用
`checkpoint_role=validation_best`；正式报告通过
`qpsalm_description_evaluation_checkpoint_binding_v5_run_completion_bound` 把期望/实际角色和
成功 run completion 一并写入 checkpoint binding。

起始采样：

```text
expert/accepted Bridge 60%
DIOR region alignment 20%
RSICap/MMRS global caption 20%
```

使用 `expert_train` 校准自然语言流畅性、地学措辞、证据充分性、保守表达和 summary。训练 MGRR、region projector 和 `desc_adapter`，冻结分割模块和 segmentation adapter。`expert_val/test` 只用于模型选择和评价，任何未经审核的教师文本不得进入这两个 split。

### D4：预测 mask 课程学习

预测 mask 必须离线生成并固定版本：

- train 使用 out-of-fold 预测或受控 mask 扰动，避免同一训练 checkpoint 的过度乐观结果；
- val/test 使用固定 checkpoint 的固定预测；
- 保存 checkpoint hash、threshold、proposal ID 和 GT IoU；
- 包含腐蚀、膨胀、遗漏、合并和 false positive；
- 不把低 IoU 预测错误地标成 GT region caption。

课程起点：

```text
100% GT
75% GT + 25% predicted/perturbed
50% GT + 50% predicted/perturbed
25% GT + 75% predicted/perturbed
```

是否进入下一档由固定 val 的 region factuality 决定，不只看训练 loss。

正式 OOF 协议只从冻结的 `expert_train` 中选择带专家 target 的 `gt_global_mask` parent，
由 `qpsalm-build-oof-folds` 生成 parent-level 分层 fold、每 fold 排除索引和
holdout 索引。每个 fold 必须使用对应 train index 与独立 Vision Cache v3 从头训练 segmentation
checkpoint；导出器校验 checkpoint 内的 `config.train_index`、fold train hash 和本次 prediction
holdout hash。当前 `qpsalm_segmentation_oof_folds_v3_source_partition_replayed` 会从冻结 Bridge
重新计算 parent assignment，并逐行证明 train/holdout 是源 segmentation index 的精确排除/保留
分区。`qpsalm_predicted_region_oof_merge_v4_exact_fold_publications_replayed` 只有在每个 parent 恰好出现
一次、fold 归属正确、checkpoint config 与 Vision Cache v3 的 train/val index fingerprint 都匹配、
expert source row 和 mask 内容均可重放时才发布 D4 train index。每个 fold mask 必须精确位于
`masks/train/<parent>.npy`，fold 目录不得包含未绑定或 `.part` 文件。仅传入一个 fold 名称或复制
`out_of_fold_verified=true` 不算 out-of-fold。

D4 训练必须同时传入两个不同发布物：`predicted_index` 是仅含 train 的 OOF merge index，
`predicted_val_index` 是固定原分割 checkpoint 对 expert val 独立导出的 prediction index。周期
validation 只读取后者并显式报告 `fixed_prediction`；缺失、路径相同、split/report/hash 不匹配或
val 集为空都必须在训练前失败。M7 选择 `joint_region_stage=predicted_mask` 时沿用同一分离协议，
不能用 train-only OOF index 产生空 region val，也不能把预测 mask validation 标为 GT-mask。
fixed val/test 发布物使用
`qpsalm_fixed_predicted_region_artifact_v3_exact_mask_directory_bound`；它必须完整覆盖当前 frozen
expert split 的 `gt_global_mask` parents，并在 D4/M6/M7 消费时重新读取 segmentation checkpoint、
专家 target 与每个 mask。mask 必须位于确定性的 `masks/<split>/<parent>.npy`，目录不得包含未绑定、
临时或缺失文件。保存 audit 后再替换源 index、checkpoint 或 mask 必须失败。

课程升档使用 `qpsalm_d4_curriculum_gate_v6_strict_json_finite`，并严格执行
`D3b(0%) -> 25% -> 50% -> 75%`。每一档先在完整 expert val 上运行 Vision-only GT-mask（仅
D3b）或 fixed-prediction（D4）generation，再由两名 reviewer 汇总 ERFS；只有当前 frozen Pilot
绝对事实性阈值全部通过，才能发布到下一相邻档的 gate。发布 CLI 必须先写候选文件，从候选绑定的
eval report、未修复 raw generation、ERFS、checkpoint、M4 suite 和当前 Bridge gate 逐字段重建，
完全一致后才原子发布；训练器消费时再次执行同一重放，不能只信任可编辑的 `passed` 字段。
`passed=false` 可作为阈值未通过的审计结果保留，但不能授权升档。相邻 D4 档必须复用同一 OOF train 与 fixed val prediction 发布物，但各自保存
实际 curriculum fraction 和精确 train population；同 stage checkpoint 初始化仅允许通过该相邻
gate，不能从 25% 直接跳到 75%。

首个 `D3b -> 25%` gate 还必须绑定通过的五-baseline M4 suite，并证明当前 seed 的 D3b source
checkpoint 正是 suite 验收的 full-MGRR candidate。该 acceptance 写入后续 D4 checkpoint 并沿
25%->50%->75% 继承；suite 文件 hash 或 frozen Bridge 漂移时后续升档直接失败。

D4 predicted-row 选择使用独立的 `d4_curriculum_sampling_seed=42`，不复用模型训练 seed。三条
模型 seed 链必须得到相同的 OOF/GT 混合 population 指纹；模型 seed 只改变优化随机性。否则三
seed 结果同时混入训练数据差异，不能作为受控重复实验。

OOF fold 和 predicted-region 目录属于单次运行输出。执行覆盖前必须确认 segmentation
checkpoint、Vision Cache、Bridge expert index、fold manifest 及 train/val/prediction index 均不在
待删除目录内；merge 输出也不得覆盖任一输入 index 或 manifest。该约束保护后续 gate 所需的
live artifact replay 证据。

进入 M7 前，75% checkpoint 自身还必须在完整 fixed expert-val 上再次通过，并用
`purpose=m7_acceptance` 发布 final gate。M7 predicted-mask 主路线显式固定 75%，其 region loader
的 OOF/fixed index、curriculum audit、frozen Bridge 和 train population 必须与 final gate 中的
75% checkpoint 完全一致；YAML 的 D4 25% 起始默认值不得隐式泄漏到联合训练。

### D5：分割—描述交替训练

起始 task sampling：

```text
segmentation                    50%
landslide region description   25%
global caption                  25%
```

DIOR alignment 已在 D2 和 D3b replay 中训练；M7 主路线不再加入第四个 DIOR loader，避免
contrastive objective 与 retention 判断同时改变。若研究 DIOR continual replay，必须作为单独
消融增加第四个 loader，不能修改主协议的 50/25/25 比例。

联合训练使用三个独立 DataLoader：

```text
segmentation_loader
global_caption_loader
region_description_loader
```

任务调度器按上述比例选择 loader。每个 optimizer step 只选择一种任务并激活对应 adapter；
若启用梯度累积，则该 step 的全部 microbatch 均来自同一任务，禁止在一个梯度累积窗口中
混合 adapter。不使用一个容纳所有 schema 的巨大混合 collate。DIOR alignment 已在 D2/D3b
完成，M7 主路线不再隐式混入 DIOR batch。

联合阶段使用一个 optimizer，并用命名参数组分别控制：

```text
default segmentation adapter
desc_adapter
MGRR + region projector
shared description/vision projection
optional segmentation dense heads (ablation only)
```

未激活 adapter 在当前 step 不产生梯度。主协议设置
`joint_train_shared_segmentation_dense=false`，冻结 SANE/QMEF/PMRD 和 segmentation controller
dense projection，只交替更新 `default` adapter、`desc_adapter`、MGRR 与描述投影。只有该主路线
通过 segmentation retention 后，才允许在独立消融中显式开启共享 segmentation dense 参数，
且不得与默认结果混报。每个 loader 的步数、parent 覆盖、采样比例和 optimizer group LR
必须写入 run manifest。

当前联合运行协议为 `qpsalm_segdesc_joint_v7_strict_json_finite`，梯度门禁必须分别验证
segmentation、global-caption 和 region-description 三条路径。global-caption 要求
`desc_adapter + description projection` 有效且 MGRR 为零梯度；region-description 额外要求
MGRR 有效；segmentation 只允许 default adapter（以及显式消融启用的 dense heads）更新。任一
inactive adapter 或非目标模块出现非零梯度都必须中止训练。`joint_manifest.json` 记录 optimizer
逐参数清单、各 loader 的有序 rows/sampler/worker binding、每 epoch batch 数和 parent
population hash；`joint_coverage_latest.json` 持续记录每个任务的 optimizer steps、样本数、
parent 覆盖及下一 microbatch cursor。resume 必须由 task pattern 与梯度累积重算每条 loader
已消费 microbatch 数，按 epoch 确定性重建并跳过 cursor；重放期间必须保护 checkpoint 恢复的
模型 RNG，禁止重复训练或改变后续 dropout/augmentation 随机流。
首次进入 M7 还必须生成
`qpsalm_segdesc_joint_initialization_v4_run_completion_bound`，把实际加载的 D4 checkpoint 路径、SHA、
step、metadata/state inventory、seed、region-data audit 以及 D4/M6 gate 路径与 SHA 固化进
best/last checkpoint。它还重放 D4 成功 training completion，证明所选 best/last 来自已完成 run；
resume 与 retention 都重新打开该 D4 payload 复算；复制 M6/D4 audit 字段但由另一组权重初始化，
或使用中断 run 遗留 best 的 joint checkpoint 必须失败。
M7 新 run 的 D4 source 必须是 `validation_best`，M7 resume source 必须是同 run 的
`terminal_last`；两种角色都从 checkpoint metadata 重放，不按文件名推断。
同一初始化审计还必须证明 joint runtime、D4 source 与 full-val retention baseline 继承同一份
原始 segmentation checkpoint bytes。

新 M7 run 只在联合优化开始前建立一次固定 monitor baseline，并同时冻结精确
sample-population identity、阈值和 positive Dice。resume 必须从同一 run 目录读取该 baseline，
并校验 checkpoint 中的 baseline identity、progress step、三类 parent population hash 和已覆盖
parent 子集；禁止在加载已联合训练的 checkpoint 后重新计算 baseline。周期 monitor 只有在样本
身份和阈值完全一致时才可参与 best-checkpoint 选择，并始终标记为 `monitor_only`。CLI 禁止
`--resume` 与 `--overwrite-output` 同时使用。

正式 full-val retention 的 baseline eval manifest 所绑定的 segmentation checkpoint SHA 必须与
joint checkpoint 的 `segmentation_migration.source_sha256` 完全一致。样本 population 相同但基线
模型不同仍不得参与门禁，以防通过替换较弱 baseline 人为缩小 Dice drop。
被评价的 joint checkpoint 必须声明 `checkpoint_role=validation_best`；terminal last 只用于恢复
同一 M7 run，不参与正式保持门禁。

正式单 seed 协议为 `qpsalm_segdesc_retention_v22_run_completion_bound`：除了 gate 中的汇总字段，还
必须重新验证 joint checkpoint 保存的 D-1 acceptance、D4 75% final acceptance 和完整 M6
GT/fixed/end-to-end acceptance，并绑定原始 `joint_segmentation_eval.json` 和 joint checkpoint 的
文件 SHA，重验 joint checkpoint 的 ontology、record/output schema 与当前仓库字节级一致，
并从 checkpoint payload 重算 joint run protocol、task schedule、loader binding、cursor 与实际
M6/D4 初始化源。gate 还必须重放 joint `training_report.json`，核对成功终态、terminal last、
coverage/history 以及被评价的 `validation_best` selection binding。
baseline `eval_manifest.json` 还必须使用
`qpsalm_segmentation_eval_manifest_v3_replay_config_bound` 绑定实际 `eval_report.json` 的路径、
SHA-256 和字节数、冻结的 eval threshold/threshold sweep，以及逐样本
prediction/target/valid-mask SHA 人口。正式 retention 必须按这些冻结评价参数现场重放声明的原
segmentation checkpoint，并让 joint checkpoint 使用同一 threshold/sweep；冻结报告与 replay 的
sample population、阈值、二值预测人口和指标必须完全一致。只改低 baseline Dice，或同时重写
report 与 manifest 但没有匹配现场 replay，都必须失败。baseline 与 joint 还必须逐样本比较
shape、target SHA 和 valid-mask SHA；相同 sample identity 但实际监督字节已漂移时不得计算
retention drop。
M7 Small 最终由 `qpsalm_segdesc_retention_seed_gate_v18_run_completion_bound` 聚合恰好
三条 seed 链；三个保存 seed、CLI seed 和 checkpoint seed 必须一致，checkpoint 必须互不相同，
scientific joint config 必须一致，并共享完全相同的 D-1 gate、frozen Bridge、D4 OOF/fixed
train-val population、M6 三模式 expert parent population、baseline checkpoint/report、full-val
population、阈值、最大允许 drop，以及相同内容的 Description Vision Cache manifest、validation
report 和 shard inventory。三条 M7 loader 的 dataset contract 与完整 parent population list/SHA
也必须相同；仅 loader/sampler seed 可按各 run seed 的固定 offset 改变。cache 路径可以是不同的
只读副本，但内容指纹必须一致。`output_dir`、
各 seed 的 final-gate 与 M6 gate 路径是 run-local
字段，不参与 config 相等比较，但 gate 内容仍逐条深度验证。Retention 属于安全不回退约束，要求
3/3 seed 全部通过，不采用 M4 比较的 2/3 增益规则。聚合产物本身必须从内嵌的三个单 seed
gate 路径重新执行完整 validator 并逐字段重建，候选 JSON 与重建结果完全一致后才原子发布；
不能把已经漂移的单 seed gate 留在一个表面通过的汇总里。旧 v7 只绑定 D-1/D4，未绑定完整三模式 M6
acceptance；旧 v9 retention 还不能证明确定性 joint execution，v10-v12 又未绑定 M6 的逐行
counterfactual input/delta 与 Description cache 全 shard 重放，v13 仍未重放实际评价 mask，
v14-v16 又未绑定 baseline report 字节，v17 仍未现场重放 baseline checkpoint，v18 未绑定并
重放冻结的 threshold sweep；这些旧协议均不能进入正式三种子聚合，必须重跑。

正式 CLI 在 baseline replay 前先以内存映射重放 joint checkpoint payload、D-1/D4/M6、joint
initialization、Description cache 和 segmentation lineage；preflight 与实际模型加载之间的文件
SHA、step 或 metadata 漂移必须失败，避免无效输入先消耗一遍完整 full-val。
joint execution replay 必须从 task pattern、checkpoint step 与 grad accumulation 重算每条
DataLoader cursor，并逐 task 验证完整 population parent list、coverage 唯一 ID、
covered/population/fraction、covered/population SHA 与 samples seen；coverage 不得包含 loader
population 外 parent。
正式流程先原子写 `retention_gate.candidate.json`，从磁盘重开并运行完整单 seed validator，
验证通过后才原子改名为 `retention_gate.json`，封闭 preflight 之后的 artifact 漂移窗口；有限
样本 smoke 不得进入这条正式通过路径。

正式 retention 使用独占输出目录，覆盖前必须证明 baseline report 与 joint checkpoint 都不位于
待删除目录内；协议重放异常原子写入 `failure_report.json` 且不发布正式 gate，证据自洽但科学
阈值未通过则保留 `passed=false` gate 报告并返回非零状态。

独立 D3/D4 checkpoint 必须显式保存
`qpsalm_region_training_data_binding_v2_cache_candidate_bound`，包括 stage、冻结专家 gate、
Bridge cache-candidate live audit、OOF predicted-index、curriculum audit 和精确 train population
指纹。M7 初始化和续训只接受这个明确字段，不从完整
训练 data audit 的其他嵌套结构推断；旧的缺失该字段或只保存 expert/index 而未保存
cache binding/fraction/population 的中间 checkpoint 必须重建。
正式 D4/M6 重放必须同时比较 frozen Pilot gate 当前绑定的 candidate SHA、region data audit
保存的 candidate SHA 与 Description Cache `multisource_parent` fingerprint；三者任一不一致都
表示混用了不同代 Bridge/cache，不得进入下一阶段。

D-1/D0-D4 与 M7 的输出目录都属于单一 run。新 run 不得向非空目录追加，resume 必须使用同一
目录内的 checkpoint，从而保留完整 history、baseline、coverage 和 data audit。输出还必须与
config、benchmark、cache、checkpoint、prediction index 和 gate 输入路径完全隔离；禁止把新 run
建在任何重放证据目录内部，或让覆盖目录包含这些输入。进程成功或失败分别写入原子的
`training_report.json` 或 `failure_report.json`；成功终态升级为
checkpoint-replayed v3 协议，必须绑定 last checkpoint、active history、progress/coverage
和核心 manifest 的当前 SHA，要求 `checkpoint_role=terminal_last`，并从 checkpoint 本体重放最终
step、stage 与 progress protocol/cursor。
磁盘 progress/coverage 必须与 checkpoint metadata 逐字段相等，active history 必须严格递增并
恰好结束于最终 checkpoint step；terminal audit 与 completion 的二次 artifact binding 必须一致。
独立训练与 M7 分别使用 `qpsalm_description_training_completion_v3_checkpoint_replayed` 和
`qpsalm_segdesc_joint_training_completion_v3_checkpoint_replayed`。
所有跨阶段或正式 checkpoint 消费再通过
`qpsalm_segdesc_checkpoint_run_completion_v1_selection_role_bound` 选择当前 best/last：重放同目录
completion 的全部 artifact，核对 selected checkpoint payload 的 stage/role/step，并在 best 路线
重放 `validation_best.json` 的 selection step。只有 checkpoint 文件而没有成功 run 终态不构成
可迁移或可正式评价的阶段产物。
调用方不得覆盖 completion 的 `protocol`、`terminal_status` 或 `artifacts` 保留字段。
既有失败报告在下一次合法 resume 前归档到 `failure_history.json`，完成 run 若已有
`training_report.json` 则禁止再次 resume；只有启动 manifest 而没有终态报告的目录属于未完成产物。

resume 必须逐字段匹配 checkpoint 保存的完整 run config，包括 seed、scheduler 总步数、学习率、梯度
累积、task pattern、课程比例、数据路径和输出目录；禁止借 resume 改变实验。跨 stage 或改变任一
运行协议时必须新建输出目录并显式使用 `--initialize-from`。跨 D0-D4 与进入 M7 时，初始化
checkpoint 的保存 seed 必须与目标 run seed 相同，禁止拼接不同 seed 的阶段后把最终目录标成
新的独立 seed。resume 还必须恢复进程 RNG 和每条独立 DataLoader 的下一 batch cursor；缺少当前
progress/RNG 协议的旧实验 checkpoint 不兼容，不得悄悄从 epoch 0 继续。
恢复点还必须不早于同目录现存的 best/last checkpoint。若进程在最近一次 checkpoint 后已经写入
history，`qpsalm_segdesc_resume_reconciliation_v1_checkpoint_cursor_bound` 会先逐字节归档该 history，
再把 active JSONL 原子回退到 checkpoint step；重复/逆序 step 或更旧恢复点必须直接失败，不能在
同一 run 中形成分叉时间线。

---

## 十二、评价协议

### 12.0 主要终点与统计协议

区域描述预注册三个共同主要终点：

1. **Expert Region Factuality Score（ERFS）**：审核者按固定八个 ontology family 标记
   `supported=1`、`partially_supported=0.5`、`unsupported=0`；每个样本必须填写全部 family，
   再先对 family 求均值、后对 parent 求宏平均，避免长文本和多字段样本占更大权重。
2. **Same-image Region Retrieval R@1**：使用生成描述或结构字段在同一 parent 的候选 regions 中检索目标区域，按 parent 聚合；候选集合和打分器在 Pilot 后冻结。
3. **Unsupported Factual Claim Rate（UFCR）**：

   ```text
   unsupported generated factual claims / all generated factual claims
   ```

   分母按 claim 计数，不按句子或样本计数；没有 factual claim 的样本不进入该比率分母，但必须单独报告数量和 empty-description rate。

MGRR 进入主模型要求：至少 2/3 seeds 同时提高 ERFS 和 R@1，且 UFCR 相对 baseline 满足 Pilot 冻结的非劣界限。CIDEr、BLEU、METEOR、ROUGE、SPICE 和 BERTScore 均为次要语言指标，不能替代区域事实性结论。

统计单位固定为 parent：

- 三个随机种子分别报告，并汇总 mean ± std；
- paired bootstrap 在同一固定 test parent 集合的逐 parent 差值上重采样，默认 10,000 次；
- 同一 parent 的多个 task view、component 和正反向 region pair 先在 parent 内聚合；
- 不把三个 seed 当作大量独立样本混入 bootstrap；
- Assisted、Vision-only、GT-mask、fixed predicted-mask 和 end-to-end 使用独立结果表。

在 M2 Pilot 完成后、正式比较模型前，冻结 `evaluation_gate_manifest.json`，记录 ERFS rubric、claim 切分规则、retrieval scorer、UFCR 非劣界限、target-status 阈值和 bootstrap seed。不得根据 Full 结果回改门槛。
冻结文件必须绑定本轮 Pilot parent manifest、review selection 与 candidate index 的 SHA-256；
其他 run 的 gate 即使阈值字段相同也不得复用。

### 12.1 整图描述

在 RSIEval test 报告：

- BLEU-1/4；
- METEOR；
- ROUGE-L；
- CIDEr；
- SPICE；
- BERTScore；
- 人工事实性、详细度和可读性。

RSIEval 只有 100 张图，caption 指标必须报告 bootstrap 95% 置信区间。首版不消费本地
RSIEval VQA；943/936 数量差异只保留为数据审计 warning，不进入训练或 checkpoint 选择。

自动 scorer 采用 `qpsalm_rsieval_caption_metrics_v1_official_backends`。它只读取冻结 D1
checkpoint 通过 `qpsalm_description_evaluation_source_filter_v1` 从 DataLoader 起隔离的完整
`rsicap_caption/test/RSIEval` raw generations，要求恰好 100 个唯一 RSIEval parent，
并同时核验当前 population identity fields、eval-report population hash、raw-generation 文件哈希
和逐样本 prediction/reference hash。BLEU-1..4、METEOR、ROUGE-L、CIDEr、SPICE 必须来自
`pycocoevalcap`，BERTScore 必须使用显式本地 encoder、显式输出层和全部合法 references；不允许
缺包时回退到自制近似。报告保存 corpus score，并以图像 parent 为统计单位计算 macro score 和
10,000 次 bootstrap 95% CI。BERTScore 模型权重、config 和 tokenizer 全部进入输入绑定；Java、
权重或依赖缺失时非零退出。自动指标仍不包含本节要求的人工事实性、详细度和可读性。
正式报告还必须保存 Java executable、pycocoevalcap scorer 源文件和 METEOR/SPICE JAR
资源哈希；SPICE 首次准备 Stanford CoreNLP 后端属于显式环境准备，不得在资源不同的 run 间
直接比较。

人工指标采用 `qpsalm_rsieval_caption_human_review_v1_blind_two_rater`。模板冻结同一 100-parent
population、materialized 图像 SHA 和未修复模型文本，但隐藏 reference captions；至少两名不同
reviewer 独立给出事实性、详细度和可读性 1–5 整数分。聚合器拒绝漏评、重复 reviewer、图像或
generation 改写，按 parent 报告三维均值、10,000 次 bootstrap 95% CI、exact/within-one
agreement 和 quadratic weighted kappa。该 test-only 人工报告不得用于 prompt 调参、早停或
checkpoint 选择，也不自动声明模型达到科学阈值。

### 12.2 DIOR 区域对齐

box-to-text：

- exact/normalized phrase match；
- token F1；
- 同图 region-text retrieval R@1/R@5；
- 修饰词准确率。

text-to-region（由原始 text-to-box view 派生）：

- 同图 candidate-region retrieval R@1/R@5；
- contrastive ranking margin；
- 歧义短语分组准确率。

首版不报告自由坐标 IoU 或 Acc@0.5，因为模型不包含 bbox 回归 head。标注 bbox 只定义候选 region 和 box-to-text 输入。

双向 view 来自同一 region pair，评价按 parent 聚合，不能把两个方向当作独立样本计算置信区间。

当前 scorer 为 `qpsalm_same_image_region_retrieval_v2_parent_ranked`：在每个 parent 的候选
region 集合内分别计算 region-to-text 与 text-to-region R@1/R@5、最佳正例相对最佳负例的
ranking margin、normalized phrase match 和 modifier accuracy，再先在 parent 内平均、最后做
parent macro。相同规范化 phrase 是 multi-positive，不得互相作为负例；没有不同 phrase 负例的
query 不进入 margin 分母。该 scorer 名称随 Pilot gate 冻结，旧 v1 gate 不可用于正式比较。

### 12.3 滑坡结构字段

- target-status macro-F1 与 balanced accuracy；
- present recall、absent recall；
- no-target false description rate；
- positive target false rejection rate；
- location、size class accuracy；
- elongation/compactness/fragmentation macro-F1；
- surface、terrain、SAR、deformation 字段 F1；
- evidence sufficiency accuracy；
- ERFS 与 field-level UFCR；
- raw JSON invalid rate、repair success rate 和 repaired-only field score。
- raw summary token-F1、exact-match 与 non-empty rate；summary 人工事实性仍进入 ERFS。

Raw structured 主指标只接受输出全文恰好为一个 schema-valid JSON object。Markdown fence、前后
解释文字或从混合文本中截取出的 JSON 只能进入 deterministic repair 分析，不能计为 raw parse
成功。Python JSON 扩展中的 `NaN`、`Infinity` 与 `-Infinity` 不是标准 JSON，raw 与 repair parser
必须拒绝，非有限 `confidence` 不能绕过数值范围验证。`summary` 是必需的非空字段；训练数据在
collate 前必须拒绝缺失或无效 summary。训练历史、评估报告、raw generations、OOF/fixed-mask
索引及审计 JSON/JSONL 统一以 `allow_nan=False` 发布；编码失败必须发生在原子替换旧产物之前，
不能把 Python 扩展 JSON 写入正式 artifact。M3–M7 loader 和正式 gate 也必须使用同一严格
decoder 重读 benchmark、cache manifest、checkpoint 旁路报告和人工审核 JSONL；不能接受外部
篡改或旧程序写出的非有限数。SegDesc checkpoint 保存、resume、跨 stage initialize 和 formal
provenance 还必须递归校验其可发布 non-tensor metadata；尚无有效 best score 时保存 `null`，不能
把内部的 `-inf` 哨兵写进 checkpoint。SegDesc config 在模型/optimizer 构建前必须拒绝非有限
学习率、loss scale、weight decay、gradient norm、curriculum fraction、mask threshold 与 retention
阈值，并验证计数/seed 的正负范围。

Assisted 模式的几何字段只评价 verbalization fidelity，不作为模型视觉感知增益；Vision-only 模式单独评价 geometry 字段预测能力。

Repair 分析必须显式报告 raw invalid 分母、实际 repair attempts、repair success rate，以及仅在
raw schema-invalid 且 repair schema-valid 样本上计算的 repaired-only field score。不得用全体
样本的 repair-schema-valid 比例或 repair 后字段分数替换 raw 主指标；同时报告空描述率和没有
factual claim 的样本数，防止通过不生成可核验主张降低 UFCR。

### 12.4 区域真实性与敏感性

必须执行：

```text
normal mask
full mask
zero mask
shuffled mask
same-image region swap
cross-parent region swap
modality removal
cross-parent modality swap
```

`same-image region swap` 必须从同一 `parent_sample_id` 的 Bridge/DIOR 区域目录加载另一个
真实 mask 或 box，并复用相同 cache transform。禁止使用 batch 内跨 parent 的 mask、水平翻转
或其他几何变换伪造同图区域。若同一 parent 没有第二个不同的有效区域，该样本记为不可用，
不能计入反事实覆盖率；正式 gate 因覆盖不足失败，而不是用人工扰动补足样本数。
`cross-parent region swap` 是另一独立模式：donor 必须来自不同 parent 的真实 mask/box，报告
必须记录 target/donor parent、sample、region 与 mask 路径。它不能借用 same-image 模式的覆盖，
且只读取 donor geometry，不得读取未审核 candidate 文本。
当前 Bridge builder 为 `landslide_bridge_m2_v7_expert_review_replay_bound`。旧 v4 prepare 模板缺少
冻结覆盖字段，v5 没有绑定 reviewer/arbitration 原文件和 expert split 派生产物，v6 又没有由
validator 独立重放双审/仲裁归并语义；三者都必须在人工审核开始前重建，禁止手工把旧 gate 或
validation report 改名。
`cross-parent modality swap` 同样必须显式核验 donor 与 target 的 `parent_sample_id` 不同，
并在逐样本报告中记录 donor parent 和被替换的模态族；同一 parent 的不同 instruction/view
不能被误当作 cross-parent 对照。donor 由数据集 parent 目录显式解析并单独编码，不依赖当前
DataLoader batch 中恰好出现另一个 parent，因此 `batch_size=1` 也必须能够执行该对照。
正式 paired gate 只接受
`qpsalm_description_evaluation_v17_structured_decoder_bound` 报告；该协议除上述 donor/region
身份外，还冻结 `max_val_samples=0` 的完整 population 请求、generation population SHA-256、
逐样本全部 references、materialized visual
identity、DIOR retrieval population SHA-256、description checkpoint SHA、训练/评价 stage、
segmentation lineage、训练/runtime seed 和 parent-level 反事实统计，还逐样本保存 region-mask 与
task-neutral backbone state 的前后指纹。正式 gate 必须从 raw baseline/counterfactual/target 重算
sensitivity、score delta 与 claim delta，并拒绝实际输入未变化或 donor 身份不合法的行；还要求
checkpoint 内精确绑定 M3 manifest/validation report/shard inventory；D4 fixed 模式还携带可重放的
OOF/fixed prediction artifact audit。每条实际送入 descriptor 的二值 region mask，以及 cycle
localization 在 valid-mask 后的 prediction/target，都必须按 sample/role 原子物化为 NPY；正式 gate
重新打开这些文件并重算 area 与 pixel IoU。GT/fixed 输入还必须从绑定的 Bridge/predicted source
NPY 逐像素重放，并从 checkpoint 绑定的 M3 shard record 重开 lookup key、cache fingerprint 与
reference-view render transform，禁止只信任评价行复制的 transform。若 M3 复用了经过
segmentation size bucket 的 Vision Cache v3，Bridge/native mask 必须先按完整场景范围以 nearest
映射到该 transform 的 `source_h/source_w`，再执行 resize/pad；两段映射及其源/目标尺寸必须写入
`qpsalm_description_region_input_source_v2_native_cache_projection_bound` 并可逐像素重放，不能把
尺寸不一致视为任意 resize。cycle 还必须保存 source-space prediction 和
descriptor valid mask，后者必须等于该 record 全部 view-valid masks 的 union，并据此重新生成
effective prediction/target；mask artifact 目录不得残留未绑定文件或 `.part`；end-to-end 必须另存
source-space 在线预测并重放其 cache 投影。不能只比较 audit 字典或接受彼此自洽的伪造计数。
独立评价必须先写完这些原子 JSONL，再由
`qpsalm_description_evaluation_publication_v1_artifact_bound` 从磁盘重开 raw、counterfactual、
可选 end-to-end/cycle 记录，核对完整 population、行数、checkpoint、字节数与 SHA-256，最后才
原子发布 `eval_report.json`；异常路径不得留下未绑定的同名报告。正式 gate 会重建该 audit，
因此在报告发布后替换 generation 或 checkpoint 会直接失效。旧 v4-v15 报告不能用于 MGRR
科学准入。三 seed 比较器必须核验 CLI seed、checkpoint 保存 config seed、主评价 seed 和
DIOR retrieval seed 完全一致，并拒绝同一 description checkpoint 在多个 seed 槽位重复使用；
只把同一 run 复制或重命名不能构成独立 seed。
每个 seed 的 baseline/candidate 主评价与 DIOR retrieval 都必须共享同一个原始 segmentation
checkpoint SHA；跨 seed 可以使用各自预注册的 segmentation seed，但同一配对内不得改变。
评估报告必须分别记录 `skipped_unavailable` 与 `skipped_no_effect`：前者表示数据中没有合法
配对，后者表示已构造对照但输入未发生变化；两者都不能计入正式覆盖率。
正式比较不得由命令行重新指定 UFCR 非劣界限或 bootstrap seed；这些值、四类反事实的最小
有效 parent 数、ERFS/target-status/UFCR 绝对阈值和 retrieval scorer 必须从当前绑定的
`expert_pilot_frozen` Bridge gate 读取。Assisted、fixed-prediction 或 end-to-end 报告可单独
分析，但不能替代 Vision-only + GT-mask 的 MGRR 主模型准入。

核心指标：

- 同图不同 region 描述差异；
- 正确 region retrieval；
- mask 外属性泄漏率；
- target-status macro-F1、present/absent recall；
- unavailable-modality hallucination；
- region-swap degradation；
- shuffled-mask degradation。

### 12.5 GT-mask、预测 mask 和端到端拆分

分别报告：

```text
GT-mask oracle description
fixed predicted-mask description
end-to-end segmentation -> description
```

避免将 segmentation 错误误判为 descriptor 错误。

三类结果必须保存 description checkpoint SHA、训练 stage、评价数据 stage 和 segmentation
migration lineage。GT-mask 与 fixed-prediction 要求 checkpoint stage 分别匹配 expert Bridge 与
D4 predicted-mask 数据；end-to-end 使用 D4 `predicted_mask` checkpoint，但以 `bridge_expert`
作为评价数据 stage，以便在线分割严格解析原 Bridge region identity。fixed-prediction 发布报告中
的 segmentation checkpoint SHA 必须与 description checkpoint 的原始 segmentation migration
一致，禁止混用另一分割模型的离线 mask。

正式 M6 三模式共同总体固定为 frozen expert val 中的 `gt_global_mask` parent，每个 parent 恰好一
行。GT/end-to-end 由 `qpsalm_description_region_source_filter_v1` 在 sample limiting 前过滤并
保存精确 population SHA；fixed prediction 只能来自同一批 parent 的独立冻结分割导出。统一
`qpsalm_m6_acceptance_v10_strict_json_finite` gate 深度重算三份 ERFS 绝对门槛、五种反事实
的 parent-level CI 与最小有效 parent、D-1/stage lineage、M4/D4 acceptance、GT cycle localization
以及 end-to-end target mapping。cycle 与在线 target audit 都必须绑定当前 segmentation instruction
index 的路径、SHA-256、字节数、task-family filter 和过滤后 population SHA，并从该源索引逐条重放
Bridge-to-segmentation 映射；只绑定可编辑 audit 文件哈希不构成验收。在线映射协议为
`qpsalm_end_to_end_region_target_v3_source_bound`。fixed/end-to-end 必须绑定同一个 D4 75%
checkpoint；fixed 发布物的 segmentation checkpoint 还必须等于 description checkpoint 保存的
`segmentation_migration.source_sha256`。M7 只能从该 gate 返回的
`qpsalm_m6_acceptance_audit_v10_strict_json_finite` 初始化。三份评价报告的内嵌 checkpoint
metadata 必须与 checkpoint 本体重放的 step、stage lineage 和 ontology/schema assets 完全一致；
只保持文件 SHA 不变而改写报告 metadata 不能通过。M4/D3b 的全 region GT-mask population 仍单独
保留，不能用 M6 的 global-only 目录替代。
M6 发布必须先写候选 gate，再从其绑定的三模式 evaluation、ERFS、D4、Bridge 和 checkpoint
完整重建，逐字段一致后才原子发布。证据自洽但科学阈值失败时应保留可审计的 `passed=false`
产物并返回非零；artifact replay validator 可接受该结果，但 M7 authorization validator 必须另外
要求 `passed=true` 且 `errors=[]`，二者不能混为一个判断。

### 12.6 Cycle localization

```text
generated text -> segment/ground -> region IoU
```

该指标只作为同模型自一致性辅助指标。主要真实性证据来自独立结构字段、同图 retrieval、人工审核和反事实测试，避免自循环虚高。

当前实现协议为 `qpsalm_cycle_localization_v1_raw_text_grounding`。它只在 frozen expert
Bridge 的 Vision-only + GT-mask 评价中启用，直接使用未修复 raw generation 替换
segmentation semantic prompt，并强制激活 `default` segmentation adapter。物理模态、active
subset 和 segmentation Vision Cache v3 保持不变；预测 mask 先按原 segmentation resize/pad
恢复到原图；若 cache view 来自 size-bucketed modality，先以 nearest/full-extent 映射到其
render-source canvas，再使用 Description cache 的 render transform 投影到 reference canvas，与真实
region mask 计算 IoU。正式评价同时物化恢复后的 source mask、descriptor valid mask 及应用
valid mask 后的 prediction/target；gate 必须从前两者重放后两者。no-target 的
target/prediction 均为空时 IoU 定义为 1，并单独报告
empty-target accuracy。报告按 parent 宏平均和 bootstrap，记录 eligible/excluded/empty-generation
覆盖。它不得用于 checkpoint 选择、MGRR 主准入或替代 ERFS、R@1、UFCR 和反事实门禁。

### 12.7 分割保持

使用当前固定 segmentation checkpoint 和完整 val 协议比较：

```text
delta_positive_dice = joint_positive_dice - seg_only_positive_dice
```

同时报告 overall、positive-only、no-target false positive 和 grouped modality metrics。

### 12.8 专家标注一致性

专家 Bridge 报告：

- 两名审核者的字段级 Cohen's kappa；
- 多于两名审核者或存在缺失标签时的 Krippendorff's alpha；
- candidate 接受率、修改率、拒绝率；
- summary 平均编辑距离和 factual claim 修改率；
- 争议字段、证据等级和模态类型分布。

区域事实性汇总使用 `qpsalm_expert_region_factuality_v2_source_revalidated`。正式 M4/D4/M6 gate
不得只信任汇总 JSON：必须核对至少两份 reviewer JSONL 的路径与 SHA，并按报告冻结的 reviewer
门槛和聚合 seed 重新运行完整 ERFS/UFCR/一致性计算；旧 v1 汇总不能继续发布科学结果。

M2 merge 的 `expert_review_report.json` 必须同时保存 final accept/revise/reject rate、候选到
最终 expert summary 的字符与归一化 edit distance、结构化 claim field 修改率，以及分歧样本的
region source、modality-family combo 和 evidence-level 分布。文本 evidence 字段
`surface_observation`、`surrounding_context` 也进入字段一致性/修改统计，不能只统计离散支持标签。
这些统计来自冻结 candidate 与最终双审/仲裁结果；未完成仲裁时不得以 pending 项推算。

M2 v7 的 `expert_review_report.json` 还必须按
`landslide_bridge_expert_artifact_binding_v1_review_sources_and_outputs` 冻结两份 reviewer 原文件、
可选 arbitration、人工 frozen-gate 源文件，以及 `expert_all/train/val/test`、pending 和发布 gate
的路径、SHA-256、字节数与适用的 JSONL/CSV 记录数。reviewer ID 必须非空，仲裁者不得与两名
reviewer 重合，且 arbitration 只能覆盖真实分歧项。Bridge validator 要求三个 split 均非空并
逐行等于 `expert_all` 的对应投影，再由 validation report 绑定 merge report 本身。validator
必须按 `landslide_bridge_expert_review_replay_v1_exact_semantic_projection` 从 current candidate、
review selection、双审与仲裁源独立重建 `expert_all` 和 pending；重放结果必须逐行相同，且决策计数
必须与 merge report 一致。accept/revise/reject rate、双审一致性、字段分歧、证据/模态分布与
expert summary/claim 修改统计也必须从 raw sources 和重放产物逐字段重算，不能只检查取值范围。
D3b/D4/M6/M7 和 unified expert publication 必须从 live bytes 重放
两层绑定并要求该语义审计；不能仅凭旧的 `status=expert_pilot_frozen` 或一次成功的
merge/validate 日志接受专家真值。

一致性结果不足时必须先修订 ontology、审核手册或样本展示，不得直接将仲裁结果当作可靠 test benchmark。

---

## 十三、实施里程碑

### M0：数据审计

- 验证本地统计；
- 记录 license 与 QA 数量差异；
- 验证 RSICap source-scene 映射及文件名前缀 fallback；
- 统计 caption verifiability 和建议训练权重；
- 不生成训练数据。

### M1：Description Benchmark Small

- 完成 source audit index 与图片物化后的模型索引；
- 冻结 `description_ontology_v1` 和输出 JSON Schema；
- 完成 single-image `ModalityInstance` 适配；
- 完成 exact/verified perceptual canonical merge 与 split freeze；
- 构建 global caption 和 region alignment Small；
- validation `errors == []`。

### M2：Landslide Bridge Pilot

- 300 parent 分层抽样并硬校验 180/60/60 split 配额；
- 三级证据协议；
- `BRIDGE_STAGE=prepare` 生成规则候选与双人 review package，但不生成专家真值；
- `BRIDGE_STAGE=merge` 只接收显式完成的双人审核、必要仲裁和人工冻结 gate；
- expert val/test 与标注一致性报告；
- 冻结 `evaluation_gate_manifest.json` 中的 Pilot 阈值和统计协议。

人工入口和环境变量只在根 `README.md` 维护。prepare 验证通过只表示自动构建有效，状态必须是
`awaiting_expert_review`。审核完成后的 merge 必须显式提供两份 reviewer 结果、必要仲裁和人工
冻结的 evaluation gate；程序不得从空模板或规则候选推断专家标签。

M3-M7 工程包固定使用 `qpsalm_seg.description` 下的一层子包：`modeling/`、`data/`、
`training/`、`evaluation/`、`protocols/`、`workflows/`。依赖按
`CLI -> workflows -> training/evaluation -> data/modeling -> protocols` 单向流动；同层可调用
明确的公共契约，但禁止跨模块导入下划线私有符号。`description/__init__.py` 只惰性导出三类
state、region encoding/geometry 等少量稳定契约，不 eager-import trainer/evaluator。配置协议为
`qpsalm_segdesc_config_v2`，model/data/training/evaluation/joint 使用独立 dataclass；
D-1 与 D0-D4 的 stage 条件统一由不可变 `StageSpec` 注册表给出。统一薄入口为
`qpsalm-segdesc cache|train|evaluate|validate`，其中 M7 由 `train joint` 进入；算法不得写入 CLI。
运行时不得提供 flat attribute/config 兼容视图；run artifact 与 checkpoint 必须保存字段完整的
嵌套 v2 配置，resume、stage lineage、M4/M7 paired audit 均以该 canonical object 为准。

### M3：Task-neutral Backbone State

- 暴露 `encode_multisource`、`build_segmentation_state` 和 `segment_from_state`；
- 构建并验证 `qpsalm_description_vision_cache_v1`；
- 保持 segmentation forward 和 checkpoint；
- 验证 state 复用不改变分割输出；
- 验证现有 cache v3 不被修改或覆盖。

Description cache 使用独立 key `qdcv1:<component>:<parent>`，支持 `single_image` 与
`multisource_parent`。缓存 record 禁止包含 instruction、condition、region geometry 或
segmentation state；若复用 segmentation cache v3，其 backend、模型/processor revision、
层号、空间尺寸和 view token 数必须严格一致。

M3 builder 在启动视觉编码前必须读取 benchmark validation：`single_image` 只接受当前
Description M1.1 v4、`errors=[]` 且 verified cross-split cluster 为零；`multisource_parent`
只接受当前 Bridge M2 v7、完整 Pilot、`errors=[]`，状态可为 `awaiting_expert_review` 或
`expert_pilot_frozen`。manifest 的 component input fingerprint 除索引路径/size/SHA 外，还必须
保存 validation report 相对路径/size/SHA、builder 和状态；verify-only、runtime dataset audit
与正式 checkpoint artifact replay 都必须比较这些字段。该门禁不把 awaiting candidate 提升为
expert truth，只防止昂贵 cache 基于旧代或无效 benchmark 构建。

cache 自身的深度报告通过后，消费者还必须在打开数据集时重算输入索引投影：Description
stream 以 `qpsalm_description_engineering_audit_v1_cache_partition_bound` 逐行重放
`all/train/dev/test/train_eligible`；Bridge region stream 以
`landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound` 重放
`candidate_all/auto_train`，并把 live candidate 与 cache manifest 的 `multisource_parent`
fingerprint 绑定。该审计不能替代专家冻结，只防止训练期间使用与 cache 不同代的索引。

### M4：MGRR

- full/box/mask/null；
- exact-mask、RoI replay、context ring；
- global mask top-component replay 与 residual aggregation；
- 多模态、valid mask、多尺寸、batch > 1；
- Assisted/Vision-only 双协议；
- baseline 消融和梯度测试。

### M5：Description Controller

- `desc_adapter`；
- teacher forcing 与 causal labels；
- JSON + summary；
- raw parse、schema validation 和 deterministic repair；
- autoregressive generation；
- adapter 切换、显式分割 checkpoint 迁移、resume 和 24GB 单卡 smoke。

### M6：训练与评价闭环

- D0–D2、D3a、D3b、D4 独立描述训练；
- GT/predicted/end-to-end 三套评价；
- counterfactual suite；
- raw generated text -> frozen segmentation 的 cycle localization 辅助评价；
- Gradio 中按 proposal/region 展示描述；overlay 必须选择 cache reference view 对应的物理模态，
  先逆 renderer padding/resize 恢复源尺寸，并显示 checkpoint、mask source、region identity 与
  end-to-end mapping audit，禁止直接拉伸 cache canvas 冒充原图坐标。
- `qpsalm_m6_acceptance_v10_strict_json_finite` 将三套正式 expert-val 结果与 D4 final gate
  合并为 M7 唯一入口；M2 未冻结或任一人工 ERFS/反事实/人口绑定缺失时必须失败。
  三套 checkpoint metadata 中的 ontology、record schema 与 output schema 不仅要彼此一致，还要
  与当前仓库的字节级资产规格完全一致；任一协议资产漂移都必须重跑评价，不能沿用旧 M6 gate。

独立正式评价必须设置 `max_val_samples=0` 与 `max_generate_samples=0`（CLI 默认如此）；训练
周期 monitor 才允许有限样本。反事实只统计输入确实发生变化的样本，并报告同一样本目标得分
差与事实 claim 数差的 paired bootstrap CI。专家事实性模板必须包含冻结 generation、盲审
可视面板和不可改写的 claim inventory。正式反事实门槛还要求每种模式达到预设有效样本数，
不能由少量未发生 no-op 的样本替代完整覆盖。

### M7：联合训练

- segmentation/description 交替 batch；
- 双 adapter；
- segmentation retention；
- Small 三种子门槛。

一个 optimizer step 内的 gradient accumulation 只能来自同一任务；默认任务序列为
`segmentation, global_caption, segmentation, region_description`。最终 retention 只有在完整
val 样本数、样本身份 SHA-256 与原分割 baseline 一致，身份记录完整且无重复、阈值一致且
checkpoint 明确来自 joint stage 时才可通过；有限样本只产生 preliminary 结果。样本身份由
sample/parent ID、任务模板、instruction、target mask 引用和 active subset 的规范化记录计算，
并包含 target size、resize/pad transform、prompt version 与 instruction ablation，避免仅凭
相同样本数或同名 sample 误判为同一评估总体。
有限样本 smoke 必须在同一受限总体上分别现场运行 baseline 与 joint 后计算临时 drop；不得将
冻结 full-val baseline Dice 与部分 joint population 直接相减。其 comparison mode 固定为
`live_limited_replay`，正式门禁只接受 `frozen_full_report`。

联合训练的同 run 续训还必须由 checkpoint 保存的 task pattern、每任务 optimizer step 和
`grad_accum_steps` 重算总 microbatch 数，并与三条 loader 的 epoch/cursor、完整有序 rows hash、
sampler 和 worker seed contract 一致。正式 retention 会从 checkpoint 本体重算该 execution
contract；仅在外部 JSON 中声明 resume 成功不能通过。

三种子门槛不是三个 JSON 中 `passed` 字段的简单计数。聚合时必须重新读取并校验每份 gate 所
绑定的原始 joint eval report、joint checkpoint、共同 baseline manifest/report/checkpoint 和
population 指纹；还必须比较 execution audit 导出的三任务训练 loader/parent population binding。
三份 joint checkpoint SHA 必须唯一，且三条链都通过，才可发布 M7 Small gate。

训练中使用的固定 monitor subset 只负责选择 checkpoint。其 baseline 在新 run 开始时冻结，
续训不得重建；正式 retention 仍必须通过独立 full-val CLI，并与原分割 checkpoint 的 full-val
报告比较。

只有 M0–M7 的 Small 验收全部通过，才构建 Full 并运行正式训练。

---

## 十四、进入 Full 的硬门槛

1. 所有 schema、path、bbox、split、provenance 和 license 验证 `errors == []`；
2. RSIEval 与 train/dev 的 exact、near-duplicate、source-scene 检查通过；
3. DIOR 多轮 pair 展开和 bbox 转换人工抽检无系统性错误；
4. single-image 适配、`qpsalm_description_vision_cache_v1` 和 cache v3 隔离验证通过；
5. 固定 64 样本、100 optimizer steps 过拟合，显式 checkpoint 迁移/reload、raw JSON generation 和 parser smoke 通过；
6. target-status macro-F1、present recall、absent recall、false description 和 false rejection 均达到 Pilot 冻结门槛，不允许只靠预测 absent 通过；
7. 总体及 unavailable-modality 子集的 UFCR 达到 Pilot 冻结门槛，claim 级分母和 empty-description 数量完整报告；
8. full MGRR 相比 crop-only 和 single-vector pooling 在至少 2/3 seeds 同时改善 ERFS 与 same-image retrieval R@1，且 UFCR 满足冻结的非劣界限；ERFS 和 R@1 的 paired bootstrap 95% CI 均不跨 0；
9. shuffled-mask 与 region-swap 使区域一致性显著下降，配对 bootstrap 95% CI 不跨 0；
10. modality removal 后对应证据陈述显著减少，cross-parent modality swap 不能维持原区域事实得分；
11. Assisted 与 Vision-only 分开报告，MGRR 的视觉理解结论必须在 Vision-only 中成立；
12. GT-mask oracle 明显优于 full-image-only 区域描述；
13. fixed predicted-mask 和端到端描述稳定运行并单独报告；
14. 联合训练后 full-val positive Dice 下降不超过 1 个绝对百分点；
15. 专家 val/test parent 独立，未审核 teacher caption 不进入最终真值，标注一致性达到 Pilot 冻结门槛。

---

## 十五、剩余人工实施顺序（2026-07-18 游标）

已接受的 M1.1、M3、D-1 和 D0 产物不得覆盖。剩余流程为：

1. 从 D0 `validation_best` 初始化并完成 D1 RSICap 校准，冻结独立 RSIEval test generation；
2. 从 D1 `validation_best` 初始化并完成 D2 DIOR region alignment；
3. 从 D2 `validation_best` 初始化并完成 D3a 自动 Bridge 工程训练；
4. 与 D1–D3a 并行完成两份真实 M2 reviewer 结果、必要仲裁和人工 Pilot gate；
5. 仅在 Bridge 达到 `expert_pilot_frozen` 后发布 expert Unified，并运行 D3b；
6. 完成三 seed M4、专家事实性、retrieval、ERFS、UFCR 和反事实门槛；
7. 生成严格 OOF/fixed predictions，训练并验收 D4；
8. 完成 GT-mask、fixed-prediction、end-to-end M6 接受门禁；
9. 从接受的 D4/M6 权重运行三 seed M7，并通过 exact-population full-val retention；
10. 只有全部 Small 门槛通过后才进入 Full。

---

## 十六、Codex 实施约束

1. `external/RSGPT`、`external/Grasp-Any-Region-main` 和原始 MMRS 数据只读；
2. 只复制正式 benchmark 入选 parent 的图片，不复制完整原始数据集，也不新增下载要求；
3. 不读取 MMRS `total.json`；
4. 不处理 classification、detection、普通 VQA 和 infrared；
5. 不破坏现有 segmentation forward、eval 和 checkpoint 加载；
6. description 能力由配置显式开启；
7. 新增脚本使用中文关键注释和标准文件头；
8. Codex 不自动运行 benchmark 构建、测试、smoke 或 GPU 训练；
9. Codex 负责实现代码、提供手动命令并依据用户返回的日志继续诊断；所有命令由用户执行；
10. 长训练、专家审核和最终科学解释由人工完成；
11. 所有生成数据保存 provenance、构建版本和 source hash；
12. teacher/pseudo caption 不进入最终 test 真值；
13. 物理量不满足单位和符号协议时，降级为定性描述或 unavailable；
14. 每个阶段先通过最小测试，再进入下一阶段。

---

## 十七、最终研究主线

```text
MMRS Caption
    -> 遥感场景和词汇广度

RSICap
    -> 人工详细描述风格

RSIEval
    -> 独立整图 caption 保持测试（首版不使用本地 VQA）

DIOR-RSVG
    -> box、短语和区域 token 对齐

Landslide V2 + Expert Bridge
    -> 精细 mask、多源证据和专业区域描述

SANE task-neutral features
    -> 多源、原生尺度空间表示

MGRR
    -> 全局、区域内部、局部细节、邻域和模态证据 token

Qwen shared base + default/desc adapters
    -> 顺序式 segmentation -> grounded region description
```

最终研究价值不只是“为 mask 生成一句 caption”，而是验证：

1. 描述是否真正随 region mask 改变；
2. 多源证据是否被正确使用；
3. 缺失证据时是否停止声称对应事实；
4. 分割误差和描述误差能否独立测量；
5. 共享 Qwen 基础模型后，分割能力是否保持；
6. MGRR 是否比 crop-only 和简单 masked pooling 生成更准确、上下文充分且可审计的区域描述。
