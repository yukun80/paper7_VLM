# Multi-Source Qwen-PSALM-Seg 二次重构问题与修改方案

## 一、重构定位

后续重构继续保留 **SANE—QMEF—PMRD** 三模块主框架，不再增加新的一级模块。17 个问题应分别收敛到三条主线：SANE 负责预训练密集视觉特征与传感器结构表达，QMEF 负责无泄漏的多源语义证据融合，PMRD 负责经 Qwen 更新的 mask queries、query-specific 像素特征与 proposal-set 分割。

PSALM 的关键依据是：task instruction、condition prompt 和 mask tokens 均进入 LMM，经过 LMM 更新后的 mask-token hidden states 再用于 mask proposal 生成；其消融结果表明，mask tokens 不经过 LMM 会降低 referring 和开放词汇分割表现。 Qwen3-VL-Seg 则说明，预训练视觉编码器的中间层特征、高分辨率浅层特征和 mask-aware query refinement 对精细边界恢复具有实际作用。

---

## 1. Mask tokens 尚未进入 Qwen

### 当前问题

当前 mask tokens 只存在于 PMRD 内部。Qwen 输出 task、condition、reasoning 和 visual evidence，随后 PMRD 再独立初始化 queries。这使 mask queries 虽然受到 Qwen semantic token 的加法或 attention 条件控制，但没有真正经过 Qwen 的多模态语义更新，与 PSALM 的核心机制仍有明显差异。

### 修改方案

在 Qwen 输入序列末尾加入固定数量的专用 `<MASK_i>` tokens，使任务指令、条件提示、多源视觉 token 和 mask tokens 在同一序列内交互。Qwen 输出位置对应的 hidden states 应直接作为 PMRD 的初始 coarse queries，替代当前独立的 `mask_tokens + task_to_query` 初始化。

考虑单卡算力，不重新在线运行完整 Qwen 视觉编码。多源视觉 token 可继续离线缓存，训练阶段只加载缓存视觉 token，与在线文本 token 和 learnable mask tokens 拼接后进入 Qwen language decoder。Qwen3-VL-2B 主体冻结，仅对最后若干 LM block 施加 QLoRA/LoRA，并训练新增 mask-token embedding。这样既能让 mask tokens 真正进入 Qwen，又避免重复计算视觉塔。

PMRD 仍负责 dense mask decoding，Qwen 只负责将 mask query 更新成与指令和多源证据一致的语义 query。主模型中不应同时保留“Qwen 更新 query”和“独立 PMRD mask token 初始化”两套平行机制。

---

## 2. VLM 指令作用不可辨识

### 当前问题

普通滑坡、多源证据、DEM 证据、SAR 证据和 InSAR 证据等指令，大多对应同一个 parent semantic mask。即使模型完全忽略文本指令，也可能得到同样的监督结果。当前训练协议因此无法证明 Qwen、condition prompt 和 evidence reasoning 确实影响了分割目标。

### 修改方案

训练集中必须加入“同一图像、不同指令、不同目标 mask”的样本。优先使用现有 mask 自动派生以下监督：指定位置的滑坡、最大滑坡、小型滑坡斑块、狭长滑坡、碎片化滑坡、指定连通域、多目标滑坡和 no-target 指令。

还应构造反事实样本，例如要求分割不存在的位置或不存在的特征，目标为空 mask；或交换同一图像不同区域的 referring expression，要求模型拒绝错误指令。这样才能区分“理解指令”与“固定语义分割”。

训练任务至少应包含三类：

1. `global semantic segmentation`：分割全部滑坡；
2. `referring/conditional segmentation`：按位置、尺度、形态或指定区域分割；
3. `no-target segmentation`：指令描述目标不存在时输出空 mask。

正式消融必须比较正常指令、随机打乱指令、固定通用指令和删除 Qwen 语义输入。若打乱指令后结果不下降，说明 VLM 部分尚未发挥作用。

---

## 3. Modality dropout 与 Qwen 视觉证据存在信息泄漏

### 当前问题

当前 semantic evidence 和 Qwen 多视图缓存可能基于完整模态集合生成，而 SANE 之后才随机丢弃模态。此时 student dense branch 虽然缺少某个模态，Qwen visual token 和 prompt 仍可能包含该模态的信息，缺失模态实验不再真实。

### 修改方案

模态子集必须在所有语义和视觉证据构造之前确定。每次训练首先生成 `active_modality_subset`，然后统一作用于：

* SANE 输入实例；
* availability metadata；
* Qwen prompt；
* evidence reasoning text；
* Qwen visual view mask；
* QMEF reliability prior。

Qwen multi-view cache 应增加 `subset_signature`。不必为每个样本缓存所有可能组合，可以为每个样本缓存完整组合、各单模态组合以及训练计划中实际使用的若干随机子集。训练时 modality dropout 只能选择已经具有匹配 Qwen evidence cache 的 subset。

missing-modality consistency 的 teacher 使用完整模态及完整 evidence，student 使用明确记录的子模态及相应 Qwen evidence。任何 student 不可见的模态，其文本描述和视觉 token 也必须被移除。

---

## 4. 不引入地理坐标识别模块

### 当前处理原则

本研究面向已经完成 patch 级配准和数据预处理的遥感图像分割，不要求模型理解 CRS、经纬度、仿射变换或地理坐标。因此不增加地理坐标编码、空间参考系转换或地图坐标 attention。

### 保留要求

“原生尺度”在本文中应严格定义为：

> 保留不同模态的原始 H/W、通道结构和 GSD 信息，并在 feature level 完成尺度对齐。

数据构建阶段必须保证同一样本各模态覆盖相同或基本一致的地表范围。模型只处理剩余的分辨率差异和轻微空间误差，不承担完整遥感配准任务。论文和文档中避免使用“地理坐标对齐”，改用“GSD-aware native-resolution feature alignment”。

---

## 5. Qwen multi-view token 尚不是真正的多源推理

### 当前问题

当前方案主要从各图像 token 内部池化得到 per-view embedding。由于 Qwen 是因果语言模型，图像 token 本身不一定能访问其后出现的总结指令，也不代表模型已经完成所有传感器之间的联合推理。

### 修改方案

在所有传感器 view 和文字说明之后，追加明确的 evidence query anchors，例如：

* `<GLOBAL_EVIDENCE>`；
* `<OPTICAL_EVIDENCE>`；
* `<SAR_EVIDENCE>`；
* `<TERRAIN_EVIDENCE>`；
* `<DEFORMATION_EVIDENCE>`。

缓存的最终证据不应以图像 token mean pooling 为主，而应提取这些位于完整多视图上下文之后的 anchor hidden states。它们能够访问前面全部图像和文本，更适合作为 Qwen 的多源推理输出。

per-view visual tokens仍可保留，用于局部传感器证据；post-context global token用于跨视图综合。SemanticEvidenceController 最终接收：

[
E={E_{\text{task}},E_{\text{condition}},
E_{\text{global}},E_{\text{view}*1},...,E*{\text{view}_M}}
]

需要继续保留 image shuffle、view removal、text-only 和 image-text-delta 消融。若图像被打乱后 global evidence 和最终性能基本不变，则不能将其表述为真正的视觉推理。

---

## 6. SANE 不再从零训练，改用预训练模型中间层特征

### 当前问题

当前 SharedBandPyramid 和 family-specific blocks 基本随机初始化。对异构且规模有限的滑坡数据，从零学习光学纹理、小目标边界、SAR 结构和多尺度空间模式风险较高。

### 修改方案

SANE 改造成**预训练特征适配器**，而不是独立从零训练的视觉主干。

主推荐方案是复用 Qwen3-VL 视觉编码器的多层中间特征。对每个模态实例先生成物理意义明确的 sensor-aware 视图，再由冻结的 Qwen3-VL vision tower提取若干层视觉特征，例如浅层、中层和高层特征。Qwen3-VL-Seg 已证明中间 ViT 特征经过轻量 spatial injection 后，能够为 dense prediction 提供比单一顶层表示更充分的空间信息。

原始多波段、SAR、DEM 和 InSAR 数值不能完全依赖三通道渲染，因此保留轻量 raw-physical adapter。其作用从“主视觉编码器”降级为“物理残差分支”：

[
F_m^l =
A_l(F_{\text{Qwen-ViT},m}^l)
+\alpha_l P_l(F_{\text{raw},m}^l)
]

其中 (A_l) 是中间层适配器，(P_l) 是原始波段投影，(\alpha_l) 采用接近零的初始化，保证训练初期主要继承预训练视觉表示。

不建议同时引入多个大型预训练 backbone。首先使用 Qwen3-VL vision tower保持模型统一；若实验显示 S1/S2 表征明显不足，再单独将 CROMA 或 AnySat 作为遥感预训练 backbone 替代方案进行对比，而不是与 Qwen、CROMA、AnySat 同时堆叠。

Qwen vision tower 默认冻结，允许对最后一到两层或 spatial adapters 使用较小学习率微调。SANE 的创新重点应从“重新学习视觉特征”转为“如何适配和融合预训练中间层特征与遥感原始物理通道”。

---

## 7. QMEF reliability pooling 未排除无效区域

### 当前问题

当前 reliability 使用整幅特征均值进行 pooling，padding、nodata 和无效覆盖区域的零值会进入统计，使不同有效覆盖比例的模态产生偏差。

### 修改方案

所有 reliability pooling 改成 valid-mask weighted pooling：

[
\bar F_m=
\frac{\sum_{x,y}V_m(x,y)F_m(x,y)}
{\sum_{x,y}V_m(x,y)+\epsilon}
]

同时增加有效覆盖比例：

[
c_m=\frac{\sum V_m}{HW}
]

将 (c_m) 作为 reliability head 的显式输入。对有效面积过小的模态设置最小可靠度上限，避免极少有效像素被错误赋予高权重。

high、mid、low 特征均需要对应尺度的 valid mask。QMEF 中任何 global pooling、attention key 和 reliability 计算都必须使用有效区域。

---

## 8. Reliability softmax 强制所有模态权重和为 1

### 当前问题

softmax 必须在可用模态中选择至少一个，即使所有模态质量都很差，仍会人为提高某个模态的权重，模型无法表达“当前所有证据均不可靠”。

### 修改方案

在 reliability 分布中加入一个 learnable `null evidence` 槽位：

[
[r_1,\ldots,r_M,r_{\varnothing}]
=\operatorname{softmax}(z_1,\ldots,z_M,z_{\varnothing})
]

真实模态融合只使用 (r_1,\ldots,r_M)，不再重新归一化到 1。若 null evidence 权重较高，融合特征整体幅度相应降低，并向 PMRD 输出较高不确定性。

另一种实现是独立 sigmoid gate，但 null-slot softmax更容易保持训练稳定，也能直接解释“没有可信辅助证据”的状态。

QMEF 日志中应增加 null reliability、有效模态总质量和 reliability calibration，而不仅记录最大模态权重。

---

## 9. Query attention 只使用 mid-level 特征

### 当前问题

当前 query-spatial attention 只读取 1/8 特征，虽然能够选择不同模态，但无法同时利用高分辨率边界和低分辨率大范围地貌上下文。

### 修改方案

将 high、mid、low 三层特征组织为统一的 multi-scale evidence memory，并为每层加入 scale embedding。每个 query 在三个尺度上进行少量 deformable sampling：

[
Z_q=\sum_{l\in{h,m,l}}
\sum_{m,p}A_{q,l,m,p}V_{l,m,p}
]

不再恢复独立的 high/mid/low gate。尺度选择、模态选择和空间位置选择由同一个 query-conditioned deformable attention 完成。

为了控制显存，每个 query 每个尺度只采样固定数量位置，例如 4 个点，不对所有 high-resolution pixels 展开全局 attention。

---

## 10. Query 选择的模态没有形成 query-specific pixel feature

### 当前问题

query attention 目前只更新 query embedding，最终动态 mask kernel仍作用于所有 query共享的 fused high feature。不同 query 即使选择了不同模态，实际像素特征仍相同，query-level 模态选择对 mask边界的影响较弱。

### 修改方案

PMRD refinement阶段根据 query-modality-spatial attention生成 query-specific pixel feature：

[
F_q(x,y)=
\sum_m A_{q,m}(x,y)F_m^{detail}(x,y)
]

随后第 (q) 个 mask embedding只作用于 (F_q)，而不是共享的 (F_{\text{fused}})。

为避免显存过高，不必长期保存完整的 `[B,Q,D,H,W]`。可按 query 分块计算，或将 attention分解为：

[
A_{q,m}(x,y)\approx a_{q,m}\cdot s_q(x,y)
]

先按 query 模态权重融合，再用 coarse mask或空间 attention做局部调制。

这一修改应与第 9 项合并实现，使“query选择哪些证据”和“query用哪些像素生成 mask”成为同一逻辑。

---

## 11. 高分辨率细节只有 1/4 分辨率

### 当前问题

当前 PMRD detail branch输入仍是 1/4 feature，最终 mask主要通过插值恢复至原分辨率。对高分辨率光学影像中的小滑坡、细长边界和碎片化目标，细节恢复能力可能不足。

### 修改方案

SANE 保留一条 1/2 分辨率的 shallow detail feature，来源优先采用预训练 vision tower浅层中间特征，并融合轻量原始图像卷积分支。该 feature只提供给 PMRD refinement，不进入复杂全局语义融合。

由于没有 box prior，使用第一轮 coarse mask形成 soft spatial gate：

[
F^{detail}*q=
\sigma(M_q^{coarse})\odot F^{1/2}*{detail}
]

再与上采样后的语义特征融合。这样可借鉴 Qwen3-VL-Seg 的高分辨率像素融合思想，同时用 coarse mask替代 box，降低无关浅层纹理干扰。

最终 mask在 1/2 特征上预测，再插值一次恢复原分辨率。只有边界指标仍明显不足时，才考虑 full-resolution stem，不应一开始使用全分辨率重型分支。

---

## 12. Verifier 诊断指标与 component-set 监督不一致

### 当前问题

proposal 已经按连通域进行 Hungarian matching，但当前部分诊断仍使用“proposal 与完整 semantic union mask 的 Dice”定义 best query。正确分割单个连通域的 query未必对完整 union mask具有最高 Dice，因此 verifier accuracy可能产生误导。

### 修改方案

诊断指标改为与 component-set matching一致：

* matched proposal mean Dice；
* component recall；
* component precision；
* unmatched proposal rejection rate；
* relevance AP/AUC；
* matched proposal relevance rank；
* proposal union Dice；
* missed-component rate；
* duplicate-component rate。

`best_query_dice against full semantic mask` 仅在 `num_mask_tokens=1` 的 semantic baseline中使用。多 query模型不再以它作为 verifier主指标或 checkpoint选择依据。

Verifier训练目标也应明确：matched proposal为正，unmatched proposal为负；当一个 proposal覆盖多个连通域时，另行记录 merge error；多个 proposal匹配同一连通域时记录 duplicate error。

---

## 13. Prompt 中仍包含 dataset name 和 normalization 信息

### 当前问题

dataset name、normalization combo 和内部数据处理信息进入 Qwen prompt，可能让模型利用数据集来源和预处理方式作为捷径。这些内容也不是自然语言分割任务所需的语义。

### 修改方案

从 Qwen prompt 中彻底删除：

* dataset name；
* normalization method；
* 文件来源；
* 内部 value encoding 名称；
* 采样和清洗标记。

Qwen prompt只保留：

* 分割任务指令；
* 目标条件；
* 当前可用的证据类型；
* 简短的遥感证据角色说明；
* 必要的尺度描述。

sensor、band、orbit、GSD、quality和availability全部通过 SANE/QMEF结构化 embedding输入，不重复写入自然语言 prompt。

正式实验应加入“完整语义 prompt”和“仅任务指令 prompt”对比，判断长 evidence reasoning是否真正有益。

---

## 14. 模态 family 映射仍依赖旧 canonical 名称

### 当前问题

虽然模型内部已经使用 `ModalityInstance`，但 family 仍通过旧 raw modality名称映射得到。新增传感器、波段产品或地形变量仍需要修改代码中的映射表。

### 修改方案

benchmark schema直接保存标准字段：

```text
family
sensor
product_type
band_names
orbit
native_gsd_m
units
quality
```

Dataset优先读取 `family` 和 `product_type`，旧 canonical映射只作为历史数据 fallback，并在读取时输出迁移警告。

`canonical_combo` 只用于数据统计和结果分组，不再参与模型输入构建和训练逻辑。完成数据迁移后逐步删除旧 canonical依赖。

---

## 15. Hash bucket embedding 存在碰撞和弱语义

### 当前问题

sensor和band name通过字符串 hash映射到固定 embedding bucket，可能产生碰撞；同时 `R`、`red`、`B04` 等同义波段无法共享语义，模型也无法知道不同光谱波段的物理关系。

### 修改方案

建立标准传感器与波段注册表。已知波段使用标准 ID，并增加连续物理描述：

* 中心波长；
* 波段宽度；
* 极化类型；
* terrain variable type；
* deformation type；
* 单位；
* signed/unsigned 标志。

band token由离散 ID embedding和连续物理属性 projection共同构成。未知传感器或未知波段才使用 hash fallback。

对于 Sentinel-2 混合分辨率波段，增加 per-band GSD；也可以按 10 m、20 m band group拆成多个 ModalityInstance，避免一个实例只保存单一 GSD。

---

## 16. Qwen 多视图 renderer 的物理问题

### 当前问题

当前 renderer已经支持光学真彩色、S2假彩色、SAR组合、terrain view和signed InSAR，但仍存在物理表达不严格的问题。

### 修改方案

SAR 第三通道当前若计算的是 `VV - VH`，描述必须写成 difference，而不能称为 ratio。若输入为 dB，`VV_dB - VH_dB` 可解释为线性域比值的对数形式；若输入为线性值，则应显式计算安全 ratio。

Terrain renderer必须根据 `product_type` 区分 DEM、slope、aspect和curvature。只有 DEM可以派生 hillshade和坡度；输入本身为 slope时不能再次将其当高程求坡度。

InSAR使用以零为中心的固定发散色图，并在描述中写明单位、LOS方向和正负号约定。为保留跨样本形变幅度差异，优先使用数据集级或区域级固定裁剪范围，而不是完全使用每张图自己的 98% 分位缩放。可以同时提供“固定尺度 view”和“局部增强 view”。

所有 renderer必须应用 valid mask。nodata区域统一显示为中性灰色，并在 view description中说明灰色区域无有效数据。

S2 true/false-color必须根据标准 band name索引，不允许在未知 band顺序下静默取前三通道。fallback view必须明确标记为 `uncertain_band_order`，并可在正式 Qwen evidence中选择禁用。

---

## 17. 代码工程仍需整理

### 当前问题

重构后算法主模块已经清晰，但 `data.py` 和 `train_eval.py` 仍承担过多职责。部分历史字段如 `visual_preview` 已不参与主模型，却仍保留在核心 batch接口中。缓存、数据、prompt、评估和诊断逻辑尚未完全解耦。

### 修改方案

`data.py` 应拆分为数据索引读取、模态归一化、空间变换、prompt生成、Dataset和Sampler。`train_eval.py` 应拆分为 trainer、evaluator、checkpoint manager、threshold evaluator和diagnostics exporter。

核心 `ModalityBatch` 只保留模型训练必需字段。`visual_preview`、可视化路径和调试信息移入可选 diagnostics/meta对象，不再作为主模型必需输入。

Qwen text cache和multi-view cache必须采用严格版本校验，至少检查：

* 模型 revision；
* processor revision；
* renderer version；
* pooling method；
* subset signature；
* 内容 hash；
* prompt version。

配置继续以 Python dataclass和preset为唯一事实来源。shell脚本只负责编排，不能重新维护算法参数。

测试体系需要补充四类真实集成测试：真实 benchmark sample完成 forward/backward；Qwen text cache与mask-token QLoRA链路；subset-aware multi-view cache；CUDA mixed-precision单步训练。合成单元测试继续保留，但不能作为完成训练闭环的唯一依据。

---

# 实施优先级

第一优先级是解决科学有效性问题，包括第 2、3、7、12、13 项。它们分别决定指令是否有效、缺失模态是否真实、指标是否可信以及模型是否利用数据集捷径。

第二优先级是增强模型核心能力，包括第 1、5、6、9、10、11 项。完成后，mask tokens、Qwen多源语义、预训练视觉中间层、query-level多尺度融合和高分辨率细化才能真正形成闭环。

第三优先级是完善结构表达和工程质量，包括第 8、14、15、16、17 项。它们影响模型扩展性、物理合理性和代码长期维护。

第 4 项不增加模型模块，只需在论文和数据规范中明确：本研究假定输入 patch已经配准，不研究地理坐标理解。

# 重构后的目标链路

最终模型应形成如下唯一主链：

[
\text{预训练多源中间层特征}
\rightarrow
\text{Qwen更新的mask queries}
\rightarrow
\text{subset-aware多源证据融合}
\rightarrow
\text{query-specific多尺度像素特征}
\rightarrow
\text{proposal set与mask-aware refinement}
]

该链路分别对应三个一级模块：

**SANE**：预训练视觉中间层与遥感物理通道适配；
**QMEF**：无泄漏、多尺度、可拒绝的多源证据融合；
**PMRD**：经 Qwen 更新的 mask queries、component-set proposals和高分辨率 mask细化。

后续新增设计必须归入这三者之一，不能再建立作用相似的平行 scorer、gate或视觉分支。
