# Multi-Source Qwen-PSALM-Seg 算法改进重构指导方针

## 一、重构目标与基本原则

当前模型已经具备多源模态适配、空间门控、query 级模态注意力、Qwen 文本与视觉证据、PSALM-style mask proposals、多尺度融合和多种 verifier，但这些能力分别以独立小模块叠加，导致同一种功能被多次实现。例如，全局模态 gate、尺度 gate、空间 gate 和 query modality attention 都在解决“选择什么模态”；condition scorer、evidence scorer 和 visual-evidence scorer 都在解决“选择哪个 proposal”；本地 visual evidence adapter 和 Qwen visual evidence cache 又都在承担视觉语义增强。当前代码因此面临模块职责不清、损失目标重复、消融组合过多和论文逻辑难以概括的问题。

重构的总体目标不是删除所有现有能力，而是将功能相近的小模块整合为三个有明确边界的核心模块，使整个算法形成一条清晰逻辑链：

**多源遥感数据如何保留原生尺度和传感器差异 → Qwen 如何提供任务语义并调度多源证据 → PSALM mask queries 如何生成和细化滑坡 mask。**

PSALM 最值得保留的是“任务指令、条件提示和 mask tokens 的统一输入范式”，以及“mask proposal 生成与条件分类解耦”的设计，而不是完整照搬其旧视觉主干。PSALM 的 mask generator 本身接收多层视觉特征、mask tokens 和 condition embeddings，并通过匹配机制训练 proposals。 Qwen3-VL-Seg 最值得借鉴的是多尺度空间特征注入、高分辨率细节融合和 mask-aware iterative refinement，而不是必须复现 box grounding。

重构后的主模型必须遵循三个原则。第一，每个核心模块只能回答一个主要问题，不能同时承担模态编码、语义推理、proposal 选择和像素恢复。第二，任何额外分支都必须能通过独立消融说明其作用，否则不进入主模型。第三，正式方法应避免依赖人工设置的弱模态权重、GT bbox 或大量辅助正则来维持性能。

---

## 二、收缩研究范围，彻底删除灾前灾后相关设计

当前研究范围已经调整为**单时相或同期多源遥感条件下的滑坡分割**，不再包含灾前灾后变化检测。因此，项目中的数据定义、任务模板、模型接口、配置、注释、实验脚本和文档必须同步收缩。

需要删除所有关于以下内容的表述与预留：

* `pre_optical`、`post_optical`、`pre_image`、`post_image`；
* change detection、new landslide、post-disaster change；
* temporal token、change-aware branch、difference feature；
* 灾前灾后配准、时间顺序和变化区域分割；
* 与变化检测有关的 prompt、task type、损失和评价指标。

不能仅在训练时不使用这些字段而继续在框架中保留。保留未使用接口会增加数据 schema 和模型逻辑复杂度，也容易让后续学生误以为需要维护一个并不存在的变化检测任务。

重构后的任务范围应只包括：

1. 普通滑坡语义分割；
2. 多源证据约束滑坡分割；
3. 困难负样本感知滑坡分割；
4. 后续可扩展的语言指令或 referring segmentation。

其中多源证据只来自当前有效的高分辨率光学、Sentinel-2、Sentinel-1、DEM/地形变量和 InSAR 形变速率。

---

# 三、重构后的三个核心模块

## 模块一：传感器感知的原生尺度多源编码器

### Sensor-Aware Native-Scale Encoder，简称 SANE

该模块只负责解决两个问题：

**不同传感器和波段如何被正确编码，以及不同尺寸和空间分辨率如何在不丢失原生尺度信息的情况下形成统一的多尺度特征。**

当前实现将原始模态压入 `hr_optical/s2/s1/dem/insar` 五个固定槽位，并规定固定通道数，例如高分光学 5 通道、S1 4 通道、DEM 2 通道。超过容量的通道会被截断。  这个设计适合快速原型，但不适合作为最终方法，因为它掩盖了波段、极化、升降轨和地形变量的真实语义。

重构后，输入单位不再是固定通道槽位，而是“模态实例”。每个输入实例至少应携带：

* 模态族：optical、multispectral、SAR、terrain、deformation；
* 传感器类型；
* 波段或极化标识；
* 原生 GSD；
* 当前对齐后 GSD；
* 有效像素掩码；
* 可选质量信息；
* 实际图像张量。

同一模态族可以共享主 adapter，但必须通过 sensor embedding、band embedding、orbit embedding、GSD embedding 和 quality embedding 保留差异。这样既不会为每种传感器单独建立完整网络，也不会把不同物理含义的通道盲目拼接。

当前 `RemoteSensingModalityAdapter` 中的通道注意力、梯度特征和空洞上下文可以保留，但应作为 SANE 的内部实现，而不是独立方法贡献。SAR、DEM 和 InSAR 使用梯度信息的思路是合理的，能够保留散射变化、地形坡度和形变梯度。

SANE 的最终输出应是每个模态的原生尺度层级特征，而不是已经融合的单一 feature map。输出形式应类似：

[
{F_m^{1/4},F_m^{1/8},F_m^{1/16},V_m,Q_m,S_m}_{m=1}^{M}
]

其中 (V_m) 是有效区域，(Q_m) 是质量或可靠性，(S_m) 是尺度信息。

该模块不读取自然语言指令，也不进行 proposal 生成。它只负责形成可信、可比较的多源密集空间特征。

---

## 模块二：Qwen 引导的多源语义证据融合器

### Qwen-Guided Multi-Source Evidence Fusion，简称 QMEF

该模块只负责解决一个问题：

**当前任务条件下，哪些模态、哪些位置和哪些尺度上的证据应该被使用。**

当前实现同时使用 sample-level gate、scale gate、spatial gate 和 query-level modality attention。   这些设计逐步解决了不同层级的证据选择问题，但功能严重重叠，且每个 gate 都需要独立参数、正则和诊断。

重构后只保留两级机制。

第一级是**模态可靠性先验**。它根据 availability、传感器、GSD、质量信息和全局统计，输出每个模态的基础可靠度。它回答的是：“这个样本中哪些模态总体可信？”

第二级是**query-spatial cross-modal attention**。它根据任务语义，在每个空间位置或 mask query 上动态读取不同模态特征。它回答的是：“当前候选滑坡区域在这个位置应使用哪种证据？”

不再单独维护 scale gate。尺度信息通过多尺度 token 和 scale positional embedding 进入统一 cross-modal attention。这样可以将当前的四层 gate 压缩为“可靠度 prior + query-spatial attention”两层逻辑。

Qwen3-VL 在该模块中只承担语义控制和多源证据理解。Qwen 输入应包括自然语言任务和多源视觉概览，输出一个统一的 semantic-evidence embedding，而不再分别构造 condition、evidence-text 和 visual-evidence 三个相互竞争的 embedding。

当前代码中三个 scorer 本质上都使用同一种 `ConditionAwareProposalScorer`，且最终通过固定权重相加。  这一部分需要合并为一个统一的 semantic-evidence verifier。它的输入是：

* 指令语义；
* Qwen 多源视觉证据；
* 当前 mask query；
* 模态可靠性信息。

它的输出是一个 proposal relevance score。

重构后不再分别设置：

* `condition_scorer`；
* `evidence_scorer`；
* `visual_evidence_scorer`；
* 三组分类损失；
* 三组 ranking loss；
* 三组手工 selection weight。

如果确实需要区分文本证据和视觉证据，应在统一 verifier 内使用两个子 token，通过 attention 自动融合，而不是建立三个独立排序器。

---

## 模块三：PSALM-style 滑坡 proposal 与 mask 细化解码器

### Proposal-Set Mask Refinement Decoder，简称 PMRD

该模块负责解决最后一个问题：

**如何从融合后的多源证据中生成一个或多个滑坡候选区域，并恢复精确边界。**

该模块保留 PSALM 的 mask token 和 proposal—classification 解耦思想。PSALM 的关键价值在于 mask tokens 先经多模态模型更新，再由 mask generator 生成 proposals，condition embedding 用于分类，而不是用一个 segmentation token 直接输出唯一 mask。

当前 Transformer mask decoder、mask embeddings 和多 proposal 输出可以保留。但需要重新定义多个 query 的语义，避免目前“多个 proposal 都学习整幅 GT，同时又被 diversity loss 要求不同”的矛盾。

如果一幅 patch 中的滑坡 mask 可以可靠分解为多个连通域，应将每个连通域视为一个滑坡实例，使用 Hungarian matching 将 queries 与连通域匹配。PSALM 本身也采用 bipartite matching 来分配 proposals 和 GT。

如果连通域不能可靠表示单独滑坡，则应将多 query 定义为“覆盖集合”，训练目标改为：

* proposal union 覆盖完整滑坡 mask；
* proposal 间减少冗余重叠；
* 每个滑坡区域至少被一个 proposal 覆盖。

不能继续让多个 top-k proposal 都独立逼近整幅语义 GT，再通过 noisy-or 合并。

PMRD 中应引入 Qwen3-VL-Seg 的两个核心思想：

第一，**高分辨率细节注入**。浅层高分辨率特征只用于边界恢复，不再重复承担语义编码。

第二，**mask-aware iterative refinement**。第一轮 mask 产生后，用 soft mask 从多源高分辨率特征中聚合区域证据，将其反馈给 query，再进行第二轮 mask 预测。Qwen3-VL-Seg 通过第一轮 mask 对像素特征进行加权池化，再更新 query，证明这种 coarse-to-fine 路线适合轻量 mask decoder。

这个 refinement 直接替代 box prior。正式模型中应删除 `box_prior_adapter` 和 GT bbox 主路径。若需要与 Qwen3-VL-Seg 做对照，box prior 只能放在独立的 legacy ablation 中，不能进入主模型。

---

# 四、原生多尺度建模模块的重构方针

## 1. 当前问题

当前实现将所有模态 resize 和 padding 到同一个 target size，然后在统一尺寸的 feature 上用卷积下采样形成 high/mid/low 金字塔。

因此，目前的“多尺度”只表示网络内部的特征金字塔，不表示不同传感器原生空间尺度。高分辨率光学和 10 m Sentinel-2 最终都在同一个像素网格中编码，它们的真实尺度差异主要依靠 GSD embedding 补偿。

此外，当前 `MultiScaleFeatureFusion` 使用卷积下采样、双线性上采样和特征相加形成 mask/memory features。 FPN 本身并非无效，但这种简单加和式 FPN 已不适合承担本文的主要创新，也无法充分解决多传感器原生尺度差异。

## 2. 重构目标

多尺度建模应单独成为 SANE 中的核心子系统，命名为：

**Native-Scale Spatial Aggregator，原生尺度空间聚合器。**

该模块应采用以下逻辑：

不同模态先在各自合理的输入尺度上编码，形成各自的多层特征；随后不是将所有特征直接 resize 后相加，而是通过 scale-aware deformable attention 或可变采样位置的 cross-attention，将不同模态的信息映射到一组统一的 decoder reference grids。

PSALM 的 mask generator采用 Mask2Former-style multi-scale deformable attention，而 Qwen3-VL-Seg采用轻量 spatial injection 加高分辨率细节恢复。  重构后的模块可以综合两者：

* 中低分辨率语义特征通过轻量多尺度 deformable attention 聚合；
* 高分辨率边界特征通过 shallow detail branch 保留；
* GSD 和 scale embedding用于控制采样范围；
* decoder query 根据目标区域动态读取不同尺度，而不是固定上采样相加。

为了满足单卡条件，可限制为三层特征、少量 deformable attention 层和较低 decoder dimension。目标不是复制完整 Mask2Former，而是用更现代的可变位置特征聚合替代当前固定 FPN 加和。

## 3. 数据与 batch 策略

不能再将所有样本无条件压缩到唯一 target size。建议采用尺寸分桶，例如 128、192、256、384，同一个 batch 内使用相近尺寸。

样本内部如果不同模态必须完成地理配准，可以在数据层统一到共同地理范围，但应保留每个模态的原生 GSD 和缩放比例。模型对齐应主要发生在 feature level，而不是依赖输入层的反复重采样。

正式论文中应将当前表述从“支持任意大小图像”改为更准确的：

**支持不同原始尺寸、不同 GSD 和不同传感器尺度的动态编码与 feature-level 对齐。**

---

# 五、padding 无效区域的统一处理方案

当前等比例 resize 后使用零值 padding，但 loss、proposal matching 和 metrics 仍对完整 target canvas 计算。  这会把 padding 当作真实背景，影响小目标、非方形样本和不同尺寸样本之间的公平比较。

数据层必须在 resize/pad 时同步生成：

[
V\in{0,1}^{H\times W}
]

其中有效影像区域为 1，padding 区域为 0。

该 valid mask 必须贯穿整个训练与评价流程。

在 BCE/Focal 中，只累计 (V=1) 的像素，并用有效像素数量归一化。Dice、IoU 和 Tversky 中，交集、预测面积和 GT 面积都必须乘以 (V)。proposal 与 GT 的 Dice matching 也必须使用 valid mask，否则 proposal ranking 仍会受 padding 影响。

Boundary loss 需要进一步收缩 valid mask 边缘，避免真实影像与 padding 的接缝被当作目标边界。注意力模块应将无效 token 作为 key padding mask，禁止 decoder 从 padding 区域读取特征。

验证时建议同时保留两套方式：

1. 在 target canvas 上使用 valid mask 计算；
2. 去除 padding 并恢复到原始 H/W 后计算。

两者结果应基本一致。若差异明显，说明 resize 或恢复逻辑存在问题。

此外，空 mask 样本不能继续只用“预测和 GT 均为空则 IoU=1”混入总体均值。应分别报告：

* positive-only IoU/Dice；
* negative sample accuracy；
* empty false-positive rate；
* overall IoU/Dice。

---

# 六、真正的 Qwen 多源视觉证据方案

## 1. 当前问题

当前 `build_visual_preview` 只按优先顺序选择第一个可用模态，通常是高分光学或 S2。

这意味着即使一个样本同时包含 S2、S1、DEM 和 InSAR，Qwen3-VL 也可能只看到一张 S2 三通道图。当前 Qwen visual evidence 因此不能称为真正的多源视觉证据。

此外，直接选前三个通道也存在物理含义错误：

* S2 前三通道不一定构成正确 RGB；
* SAR 需要 VV/VH/ratio 等明确视图；
* DEM 应使用 elevation、slope 或 hillshade；
* InSAR 应保留正负形变方向，不能简单裁剪到 `[0,1]`；
* 不同 sensor view 应向 Qwen 显式说明其类型。

## 2. 重构目标

建立独立的：

**Sensor-Aware Multi-View Renderer，多源传感器视觉渲染器。**

每个可用模态生成一张或少量具备明确物理含义的可视图：

* 高分辨率光学：自然 RGB；
* Sentinel-2：真彩色和 NIR/SWIR 假彩色；
* Sentinel-1：VV、VH 和 ratio 合成；
* DEM：高程、坡度或 hillshade；
* InSAR：以零为中心的发散色图，并标注单位与方向。

Qwen3-VL 使用多图输入，而不是将这些 view 拼成一个大图或只选择优先模态。每个 view 前应附加简短类型说明，例如该图是 S1 SAR、DEM slope 或 InSAR velocity。

Qwen 输出只作为**全局语义证据 embedding**进入 QMEF 和 proposal verifier，不应直接承担密集像素编码。密集边界仍由 SANE 的原始数值特征提供。

## 3. 删除重复的本地 visual evidence 分支

当前 `VisualEvidenceAdapter` 又用 CNN 对三通道 preview 生成 dense features，并直接加到 mask features 和 memory features。

这与原始模态 adapter 重复编码同一视觉信息。重构后建议删除该 dense preview branch。高分辨率细节统一由 SANE 和 PMRD 的 detail branch 提供，Qwen multi-view 只提供语义证据。

若需要保留一个快速非 Qwen 对照，应将其明确命名为 `local_preview_baseline`，只用于消融，不进入主模型。

## 4. Qwen pooling 与真实性验证

当前 Qwen 图文 cache 对全部有效 token 做 mean pooling。 长文本可能使 embedding 主要反映 prompt，而不是图像。

需要比较以下 pooling 方案：

* vision-token pooling；
* image-end token；
* learnable attention pooling；
* image-text embedding 与 text-only embedding 的差值。

必须增加两个真实性测试：

第一，将图像随机打乱而保留文本不变，观察 verifier 性能是否明显下降。
第二，将文本保持不变，只删除某一模态 view，观察对应 evidence score 是否变化。

若图像打乱后性能基本不变，说明所谓 visual evidence 实际仍是文本条件，不能作为方法贡献。

Qwen visual cache key 还必须包含 preview 内容 hash、renderer version、模型 revision 和 processor revision，不能只依赖 sample ID 和 prompt 文本。

---

# 七、proposal 与损失函数的简化方针

当前完整配置同时启用了大量损失：最终 mask、proposal mask、proposal 分类、condition 分类、evidence 分类、visual evidence 分类、多个 ranking、空 mask 抑制、query diversity、proposal diversity、gate entropy 和 query usage balance。

这种设计适合调试，但不适合作为最终算法。它会导致性能提升无法归因，且不同正则可能互相冲突。

重构后的主损失建议只保留四类。

第一类是最终 mask loss，由 BCE/Focal、Dice/Tversky 和可选 Boundary loss 组成。

第二类是 proposal set matching loss。根据连通域或 coverage-set 定义，对 proposals 进行 Hungarian matching 或 set coverage 监督。

第三类是统一 semantic-evidence verifier loss。只保留一个 proposal relevance classification/ranking 目标。

第四类是 missing-modality consistency loss。对同一样本的完整模态和随机子模态预测进行一致性约束。

query diversity、gate entropy、query usage balance 和人工 hard-combo 权重不应默认进入主模型。只有在消融证明 query 塌缩或模态塌缩无法由结构解决时，才作为训练技巧启用。

当前 `canonical_combo_loss_weights` 对弱组合进行手动加权。 主模型应改为均衡采样、modality dropout 和 reliability-aware fusion，不应把人工组合权重作为核心方法的一部分。

---

# 八、代码“屎山”问题的重构方针

当前代码的问题不只是文件数量，而是算法逻辑、实验逻辑和历史兼容逻辑混在一起。模型 forward 中同时处理 Qwen 文本、evidence 文本、visual cache、本地 preview、GSD FiLM、多源 adapter、box prior、fusion、decoder 和十余种 loss；训练脚本和 shell 又分别维护一套 loss stage 和 verifier stage。

代码整理应遵循“算法边界即代码边界”。

## 1. 模型代码只保留三大模块

模型主干只依赖：

* SANE；
* QMEF；
* PMRD。

总装模型只负责调用三者，不再在 forward 中直接实现 evidence 权重、GSD 调制、visual feature 加法或多个 scorer 的组合逻辑。

## 2. 建立统一输入和输出数据结构

所有模块通过统一的数据对象传递：

* `ModalityBatch`；
* `MultiScaleFeatures`；
* `SemanticEvidence`；
* `ProposalSet`；
* `SegmentationOutput`。

避免当前通过大型字典和字符串 key 隐式约定几十个字段。统一结构必须包含 shape、valid mask、GSD、availability 和 modality metadata。

## 3. 配置只能有一个事实来源

当前 Python 配置和 shell 脚本都在维护 loss stage 和 verifier stage。重构后所有 preset 只能定义在 Python 配置层，shell 只负责选择实验名称和传入少量参数。

不允许 shell 脚本逐项改写几十个 loss 权重。否则配置文件、命令行和 shell 默认值会不断漂移。

## 4. 主算法与实验消融分离

以下能力不应继续存在于主模型内部：

* legacy box prior；
* local preview baseline；
* text probe；
* hash-smoke；
* hard-combo weighting；
* verifier 独立实验；
  -旧 FPN 兼容路径。

这些可以作为插件、实验 wrapper 或 development utility 存在，但不能污染主模型 forward。

## 5. 删除所有历史兼容 shim 和无效接口

当新版训练和 checkpoint迁移完成后，应删除：

* 旧 `model.py` 兼容导入；
* 废弃参数；
* 灾前灾后字段；
* box-prior 主线参数；
* 未使用的 prompt builder；
* 重复的 modality mapping；
* 同时存在于 `data.py` 和 `indexing.py` 的重复逻辑。

必要的旧 checkpoint 兼容应通过单独的 conversion script 完成，而不是长期在主模型中保留条件分支。

## 6. 每个核心模块必须有独立测试

SANE 测试任意模态组合、不同通道数、不同 GSD、不同 H/W 和 valid mask。

QMEF 测试缺失模态、模态顺序交换、错误 view、图像 shuffle 和 evidence attention。

PMRD 测试空 mask、单连通域、多连通域、不同 proposal 数量和 valid-region matching。

训练系统测试 checkpoint reload、cache version、配置一致性和单卡 smoke run。

---

# 九、建议的重构实施顺序

第一阶段先做**范围和正确性清理**。删除灾前灾后相关内容；加入 valid pixel mask；区分 positive 与 empty 指标；将 box prior 移出主线；统一 Python 配置；更新 README 和算法说明。该阶段不改变主体网络，优先保证现有实验结果可信。

第二阶段完成**三模块结构重组**。将现有 adapter、GSD 和多尺度编码重组为 SANE；将多个 gate 和多个 verifier 合并为 QMEF；将 mask token、proposal 和 refinement 重组为 PMRD。主模型图中只允许出现这三个模块。

第三阶段完成**原生多尺度建模**。引入尺寸 bucket、per-modality GSD、多尺度 deformable aggregator 和有效区域 attention mask，替换当前简单 FPN 加和。

第四阶段完成**Qwen 多源视觉证据**。建立多传感器 renderer、多图 Qwen cache、vision-token pooling、image-shuffle 评价，并删除主线中的本地 preview dense branch。

第五阶段完成**proposal 监督重定义**。选择 connected-component matching 或 coverage-set supervision，加入 mask-aware iterative refinement，简化 loss。

最后阶段再进行代码清理、模块单测、配置冻结和正式消融实验。

---

# 十、重构完成后的算法逻辑与创新点

重构后的论文方法应只强调三个实质性模块。

第一，**传感器感知的原生尺度多源编码器**，解决不同传感器、不同波段、不同 GSD 和任意模态组合的统一密集表示问题。

第二，**Qwen 引导的多源语义证据融合器**，利用多图 Qwen 语义证据和 query-spatial cross-modal attention，在不同任务条件下动态选择光学、SAR、DEM 和 InSAR 证据。

第三，**PSALM-style proposal set 与 mask-aware refinement decoder**，通过多个 mask tokens、proposal—classification 解耦和迭代区域证据回放，生成不规则、多斑块滑坡 mask，而不依赖 box grounding。

这三个模块分别对应：

**看懂不同传感器 → 拼好不同证据 → 分出精确滑坡区域。**

最终模型图、代码主干、实验消融和论文贡献都应围绕这三步展开。任何不能明确归入这三步、且无法通过独立实验说明必要性的模块，都不应进入最终主模型。
