# 多源滑坡分割—区域描述统一模型：Benchmark 与算法实施方案

> 文档状态：实施基线 v2
>
> 当前前置能力：Landslide Benchmark V2、SANE、QMEF、PMRD、Qwen mask-query controller
>
> 本阶段目标：在不破坏现有分割路径的前提下，建立可审计的整图描述、区域对齐和滑坡区域描述能力。

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
| DIOR-RSVG parents | 15,709 | 每个 parent 有两个相反方向任务视图 |
| DIOR-RSVG task records | 31,418 | 15,709 box-to-text + 15,709 text-to-box |
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

统一索引引用现有 segmentation 和三个描述 benchmark，不重复复制图像。任务族固定为：

```text
segmentation
global_caption
region_referring_expression
region_grounding
landslide_region_structured_description
landslide_region_caption
no_target_response
```

首版不包含 `landslide_region_vqa`；只有获得独立、可验证的 VQA 标注后才新增。

### 3.6 存储策略

默认使用**薄索引模式**：

- JSONL 保存可移植逻辑路径、图像 hash、尺寸和 provenance；
- 不将 66GB MMRS 原图复制到 benchmark；
- `external/RSGPT`、`external/Grasp-Any-Region-main` 和 `../datasets/MMRS-1M` 保持只读。

发布或迁移时可显式选择：

```text
--materialize-mode copy
--materialize-mode hardlink
```

默认值为 `none`。验证程序应区分“逻辑路径可解析”和“benchmark 已自包含”。

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
    "path": "external/RSGPT/dataset/RSICap/images/P0378_0001.png",
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
    "annotation_path": "external/RSGPT/dataset/RSICap/captions.json",
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
    -> canonical parent
    -> SHA-256 exact groups
    -> perceptual near-duplicate candidates
    -> source scene/group constraints
    -> split assignment
    -> task view expansion
```

不得先展开多条 instruction/caption，再按 instruction 随机划分。

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

感知 hash 只生成候选簇，不能自动删除所有相似遥感图像。大面积农田、机场和住宅区可能视觉相似，但并非同一图像。

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
3-5_validate_description_benchmark.py
3-6_summarize_description_benchmark.py
description_common.py
```

总控入口：

```text
scripts/run_3_build_description_benchmark.sh
```

#### `3-1_scan_description_sources.py`

只读扫描：

- `external/RSGPT/dataset/RSICap`；
- `external/RSGPT/dataset/RSIEval`；
- `../datasets/MMRS-1M/json/caption`；
- `../datasets/MMRS-1M/json/RSVG/rsvg_trainval.json`；
- 对应图像目录。

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

执行第五节协议并输出 immutable split manifest。后续程序只能读取该 manifest，不能重新随机划分。

#### 验证和汇总

至少检查：

- 图像路径存在且可解码；
- caption/phrase 非空；
- bbox 有效且坐标协议一致；
- parent、scene、duplicate cluster 不跨 split；
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

1. 统计 global/referring/component/no-target/predicted region；
2. 生成确定性几何和条件允许的多源证据；
3. 生成规则化候选文本，提供可插拔离线教师接口；
4. 导出多源面板、mask overlay、可编辑 JSON/CSV；
5. 合并 accept/modify/reject 和双人仲裁结果；
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

统一索引只保存 component benchmark 引用和任务采样元数据，不再次复制源样本。

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

Bridge pilot 默认 300 个 parent，按数据源、模态组合、region source、面积和 target status 分层。

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

### 9.6 Token 预算

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

### 9.7 必须比较的 baseline

```text
crop-only
full-image + box coordinates
single-vector masked pooling
RoI replay only
MGRR without context ring
full MGRR
```

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

滑坡区域描述首版输出：

```json
{
  "region_id": "Prompt0",
  "target_status": "present",
  "geometry": {
    "location": "upper_left",
    "size_class": "small",
    "shape": "elongated",
    "fragmentation": "moderate"
  },
  "observations": {
    "surface": {"value": "...", "confidence": "..."},
    "terrain": {"value": "...", "confidence": "..."},
    "sar": {"value": "...", "confidence": "..."},
    "deformation": {"value": "...", "confidence": "..."},
    "surrounding_context": {"value": "...", "confidence": "..."}
  },
  "evidence_sufficiency": "sufficient|partial|insufficient",
  "summary": "..."
}
```

规则：

1. geometry 来自确定性 region 计算，单独报告，不宣称是视觉模型预测；
2. observations 由模型生成并接受字段级监督；
3. 缺失模态字段必须是 unavailable/null；
4. `target_status=absent` 时明确拒绝，不继续生成滑坡属性；
5. global caption 使用自由文本，不强制套用滑坡 JSON。

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
5. 新统一 checkpoint 使用 `qpsalm_segdesc_v1`，记录来源分割 checkpoint hash、两个 adapter 状态、ontology/schema 版本和描述 cache protocol；
6. Adapter 名称固定为 `default` 和 `desc_adapter`，不执行隐式重命名。

---

## 十一、训练协议

### D-1：最小正确性与过拟合

正式预适配前完成：

- Qwen zero-shot baseline；
- 32–64 条 global/box/mask/null 混合样本过拟合；
- causal label mask 检查；
- region token 梯度检查；
- adapter 切换和 checkpoint reload；
- batch size > 1、不同图像尺寸和空 region。

### D0：MMRS Caption 场景预适配

目的：学习遥感场景和目标词汇。

只训练：

```text
desc_adapter
description projection
region/global special embeddings
```

冻结 SANE、QMEF、PMRD 和 segmentation adapter。

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

### D3a：自动结构化 GT-mask 预训练

使用全部满足 schema、valid mask 和证据协议的 Landslide V2 区域训练：

```text
target_status
deterministic geometry verbalization
available/unavailable modality state
protocol-valid physical or relative evidence
structured JSON fields
```

此阶段只使用 `auto_train` 中可追溯的结构字段和规则化文本，训练 MGRR、region projector 和 `desc_adapter`，冻结 segmentation adapter 和 PMRD。对于 Level C 证据，模型必须学习输出 unavailable/insufficient，而不是补全看似合理的地学结论。

### D3b：专家 Bridge 文本校准

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

### D5：分割—描述交替训练

起始 task sampling：

```text
segmentation                    50%
landslide region description   25%
DIOR region alignment           15%
global caption                  10%
```

联合训练使用三个独立 DataLoader：

```text
segmentation_loader
global_caption_loader
region_description_loader
```

任务调度器按上述比例选择 loader，每个 optimizer step 只消费一种 batch 并激活对应 adapter；不使用一个容纳所有 schema 的巨大混合 collate。DIOR region alignment 由 region loader 内的独立 task sampler 提供。

联合阶段使用一个 optimizer，并用命名参数组分别控制：

```text
default adapter + segmentation heads
desc_adapter
MGRR + region projector
shared description/vision projection
```

未激活 adapter 在当前 step 不产生梯度。先冻结 SANE/PMRD 做双 adapter 交替；只有 segmentation retention 通过后，才允许小学习率更新共享 projection。每个 loader 的步数、parent 覆盖、采样比例和 optimizer group LR 必须写入 run manifest。

---

## 十二、评价协议

### 12.0 主要终点与统计协议

区域描述预注册三个共同主要终点：

1. **Expert Region Factuality Score（ERFS）**：审核者按 ontology 字段和 summary factual claim 标记 `supported=1`、`partially_supported=0.5`、`unsupported_or_contradictory=0`；先对每个 ontology family 求均值，再对 parent 求宏平均，避免长文本和多字段样本占更大权重。
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

### 12.1 整图描述

在 RSIEval test 报告：

- BLEU-1/4；
- METEOR；
- ROUGE-L；
- CIDEr；
- SPICE；
- BERTScore；
- 人工事实性、详细度和可读性。

RSIEval 只有 100 张图，指标必须报告 bootstrap 95% 置信区间。RSIEval VQA 仅评价能力保持，不选择描述 checkpoint。

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

Assisted 模式的几何字段只评价 verbalization fidelity，不作为模型视觉感知增益；Vision-only 模式单独评价 geometry 字段预测能力。

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

### 12.6 Cycle localization

```text
generated text -> segment/ground -> region IoU
```

该指标只作为同模型自一致性辅助指标。主要真实性证据来自独立结构字段、同图 retrieval、人工审核和反事实测试，避免自循环虚高。

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

- 完成 thin-index schema；
- 冻结 `description_ontology_v1` 和输出 JSON Schema；
- 完成 single-image `ModalityInstance` 适配；
- 完成 split/dedup；
- 构建 global caption 和 region alignment Small；
- validation `errors == []`。

### M2：Landslide Bridge Pilot

- 300 parent 分层抽样；
- 三级证据协议；
- review package；
- expert val/test 与标注一致性报告；
- 冻结 `evaluation_gate_manifest.json` 中的 Pilot 阈值和统计协议。

### M3：Task-neutral Backbone State

- 暴露 `encode_multisource`；
- 构建并验证 `qpsalm_description_vision_cache_v1`；
- 保持 segmentation forward 和 checkpoint；
- 验证 state 复用不改变分割输出；
- 验证现有 cache v3 不被修改或覆盖。

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
- Gradio 中按 proposal/region 展示描述。

### M7：联合训练

- segmentation/description 交替 batch；
- 双 adapter；
- segmentation retention；
- Small 三种子门槛。

只有 M0–M7 的 Small 验收全部通过，才构建 Full 并运行正式训练。

---

## 十四、进入 Full 的硬门槛

1. 所有 schema、path、bbox、split、provenance 和 license 验证 `errors == []`；
2. RSIEval 与 train/dev 的 exact、near-duplicate、source-scene 检查通过；
3. DIOR 多轮 pair 展开和 bbox 转换人工抽检无系统性错误；
4. single-image 适配、`qpsalm_description_vision_cache_v1` 和 cache v3 隔离验证通过；
5. 32–64 样本过拟合、显式 checkpoint 迁移/reload、raw JSON generation 和 parser smoke 通过；
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

## 十五、人工实施顺序

1. 运行 RSGPT/MMRS/DIOR 数据审计；
2. 人工确认 license、RSIEval QA 差异、RSICap scene mapping 和 DIOR 多轮解析；
3. 冻结 `description_ontology_v1`、输出 JSON Schema 和英语协议；
4. 构建 `rs_global_caption_v1_small`；
5. 构建 `rs_region_alignment_v1_small`；
6. 抽检 100 条 global caption 和 100 个 region pair；
7. 完成跨数据集去重和 split freeze；
8. 实现单图适配并建立 `qpsalm_description_vision_cache_v1`；
9. 暴露 task-neutral backbone state 并验证 cache v3 隔离；
10. 进行 32–64 条样本过拟合；
11. 训练 D0 MMRS Caption Small；
12. 训练 D1 RSICap 校准；
13. 训练 D2 DIOR region alignment；
14. 构建 300-parent Landslide Bridge pilot；
15. 完成专家审核、标注一致性分析并冻结 expert val/test；
16. 冻结 `evaluation_gate_manifest.json`；
17. 实现和消融 MGRR、global component replay 及 Assisted/Vision-only；
18. 训练 D3a 自动结构化 GT-mask 描述；
19. 训练 D3b 专家 Bridge 文本校准；
20. 执行 mask、region 和 modality 反事实测试；
21. 生成 out-of-fold/fixed predicted masks；
22. 训练 D4 predicted-mask curriculum；
23. 训练 D5 双 adapter 交替任务并检查 segmentation retention；
24. Small 门槛全部通过后进入 Full。

---

## 十六、Codex 实施约束

1. `external/RSGPT`、`external/Grasp-Any-Region-main` 和原始 MMRS 数据只读；
2. 默认不复制原图，不新增外部数据下载要求；
3. 不读取 MMRS `total.json`；
4. 不处理 classification、detection、普通 VQA 和 infrared；
5. 不破坏现有 segmentation forward、eval 和 checkpoint 加载；
6. description 能力由配置显式开启；
7. 新增脚本使用中文关键注释和标准文件头；
8. Codex 不自动运行长时间 GPU 训练；
9. Codex 负责代码、测试、smoke、命令和失败诊断；
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
    -> 独立整图 caption / VQA 保持测试

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
