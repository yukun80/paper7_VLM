# SANE–QMEF–PMRD 重构评审快照（历史归档，2026-07）

> 归档状态：本文记录一次阶段性重构评审，其中“当前”“仍需处理”等措辞只代表归档时点。
> 当前分割算法契约与仍有效的限制统一维护在
> `SEG_Multi-Source_Landslides/ALGORITHM.md`；SegDesc 设计见 `docs/benchmark_GAR.md`。

# 一、归档时模型的实际设计

## 1. SANE：预训练特征与遥感物理信息共同编码

当前 SANE 已经不再依赖固定的五组通道槽位，而是将每个输入表示为 `ModalityInstance`。每个实例显式保存 family、sensor、product type、band names、band metadata、orbit、units、signed、原生与对齐 GSD、valid mask 和 quality。一个样本可以包含任意数量、任意通道数的模态实例。

SANE 同时保留了两条互补路径。

一条是原始物理数据路径。每个物理波段通过共享单波段 stem 编码，波段权重由有效区域池化后的视觉特征和物理 band token 联合决定。输入在第一次卷积前已经乘以 valid mask，避免 nodata 数值进入卷积。

另一条是预训练 Qwen-ViT 路径。视觉缓存保存 Qwen-ViT 第 5、11、17、23 层的中间特征，再分别映射到 detail、high、mid、low 四个尺度；原始遥感特征以接近零初始化的残差形式加入预训练特征。

这已经基本解决上一版“SANE 从零训练密集特征”的主要风险，也符合 Qwen3-VL-Seg 使用预训练 ViT 中间层特征、再通过轻量空间适配进行 dense prediction 的思路。

## 2. Qwen controller：mask queries 真正进入语言模型

当前正式路径使用 NF4 4-bit Qwen3-VL-2B language decoder。视觉塔离线运行并缓存特征，在线训练时释放视觉塔，仅保留 language decoder。Qwen 基础权重冻结，最后四个语言层的 q/k/v/o projection 通过 QLoRA 训练。

每个样本的序列按以下顺序构造：

[
\text{task}
\rightarrow
\text{condition}
\rightarrow
\text{reasoning}
\rightarrow
\text{view descriptions + visual tokens}
\rightarrow
\text{evidence anchors}
\rightarrow
\text{mask embeddings}
]

每个视觉 view 使用 Qwen 原生 vision-start 和 vision-end embedding 包围，mask embeddings 位于所有多源视觉证据和 evidence anchors 之后，因此能够访问完整的前置语义信息。

Qwen 输出中，mask 位置的 hidden states被投影到 decoder dimension，作为 PMRD 唯一的 query 初始化。

这与 PSALM 的核心发现保持一致：mask tokens 经过 LMM 更新后再送入 mask generator，比直接使用独立 mask queries 更有效，尤其有利于 referring 和开放词汇任务。

## 3. QMEF：可拒绝的多源、多尺度、query-conditioned 融合

QMEF 已经将此前多个重叠 gate 收敛为两层逻辑。

第一层是模态 reliability prior。它使用有效区域池化后的视觉特征、模态 metadata、对应 family evidence anchor、质量和覆盖比例估计每个模态的可靠度，并加入一个 learnable null-evidence slot。真实模态权重不再被强制重新归一化为 1，因此模型可以表达“所有辅助模态均不可靠”。

第二层是 query-conditioned deformable evidence attention。每个 coarse mask query 根据自身 mask 质心和学习偏移，在 high、mid、low 三个尺度、所有模态和多个采样点上联合归一化注意力。注意力分数同时包含视觉 key、family semantic anchor 和 reliability prior。

因此当前模型不再是“先融合一次、所有滑坡共用一套特征”，而是允许不同滑坡 proposal分别选择光学、SAR、地形或形变证据。

## 4. PMRD：无 box 的 proposal-set 滑坡分割

PMRD 先用 Qwen mask-position states读取低分辨率融合 memory，生成多个 coarse masks。随后，QMEF 提供每个 query 的模态与尺度证据权重，PMRD据此构建 query-specific 1/2 尺度 detail features。coarse mask作为空间 gate，区域池化后的多源证据被反馈到 query，并进行第二轮 refinement。

最终不是只选择一个 proposal，而是由统一 verifier输出 proposal relevance，再以 query-count-calibrated noisy union 合并相关 proposals。

该路线保留了 Qwen3-VL-Seg 的“预训练语义先验—多尺度空间注入—高分辨率细节—mask-aware refinement”思想，但用 coarse mask和多源 evidence替代了 bbox prior，更符合不规则、多斑块滑坡对象。

## 5. Proposal-set 监督和评价

训练 mask会被拆分为 8 邻域连通域。连通域数量不超过 query数量时，使用 BCE+Dice cost进行 Hungarian matching；超过 query数量时切换到 coverage-set监督，允许一个 query覆盖多个组件。

主损失已经收缩为：

[
L=L_{\text{final}}
+\lambda_{\text{set}}L_{\text{proposal set}}
+\lambda_{\text{verifier}}L_{\text{relevance}}
+\lambda_{\text{consistency}}L_{\text{missing modality}}
]

旧版本中的 query diversity、gate entropy、hard-combo weighting和多套 verifier ranking loss已经退出主目标。

评价同时包含 positive-only Dice/IoU、negative accuracy、empty false-positive rate、component recall/precision、relevance AP/AUC、duplicate/merge/missed-component rate 和 proposal-union Dice。

---

# 二、这次重构中设计合理的地方

## 1. 此前列出的多数核心问题已经真正解决

“mask tokens没有进入Qwen”已经解决；当前 mask embeddings在 Qwen序列中经过语言 decoder更新，并且是 PMRD唯一 query来源。

“VLM指令不可辨识”在数据结构上已经基本解决。benchmark-v2同时包含 global、referring和 no-target任务，同一 parent可以对应全滑坡 mask、不同方位目标 mask和空 mask。集成测试也显式构造了同一父图的全局、左上、右下和无目标四类指令。

“modality dropout与Qwen视觉信息泄漏”也已经从流程上解决。active subset在 prompt、Qwen view选择和 SANE之前确定，full和active prompts分别生成；视觉缓存按 `ActiveModalitySubset` 动态筛选 view。

“Qwen multi-view token不等于多源推理”得到较大改善。所有多源 view后面加入 global、optical、multispectral、SAR、terrain、deformation anchors，再在其后放置 mask embeddings。anchors和mask queries能够读取前面的完整多图上下文。

“SANE从零训练”已经改为预训练 Qwen-ViT中间层特征加遥感物理残差路径。

“reliability pooling未使用有效区域”“softmax不能拒绝所有模态”“query只用mid-level”“query选择的模态没有直接形成像素特征”“细节只有1/4”等问题，也分别通过 valid-weighted pooling、null slot、多尺度联合 attention、query-specific detail feature和1/2 detail branch进行了处理。

## 2. 数据、模型和评价形成了较清楚的因果链

当前数据层先决定 active subset，再生成 prompt和视觉证据；SANE只看 active dense features，Qwen只看 active views，QMEF只对 active modalities计算 reliability和 attention。这个因果链比上一版清楚，适合做缺失模态实验。

视觉真实性消融只替换或删除 Qwen evidence tokens，不改变 SANE dense features，因此可以单独判断“Qwen视觉证据”是否有效，而不会同时改变视觉 backbone。

## 3. 工程实现已经接近可长期维护状态

代码已经拆分为 `data/`、`engine/`、`models/`、schema、matching、losses和 inference服务。训练优化器具有明确的参数角色和分阶段 QLoRA 开启逻辑，LoRA梯度和实际参数更新都会在阶段切换时检查。

仓库还新增了严格 integration gate、Gradio demo、gallery筛选和共享 inference session。最新提交已经把模型训练之外的推理和展示链路也补齐。

---

# 三、当前算法的创新性判断

当前创新属于**中上水平的任务与框架创新**，而不是底层 Transformer 原理创新。

最有价值的潜在创新点有三个。

第一，**任意数量的传感器—产品—波段实例编码**。模型不要求固定 S2+S1+DEM通道组合，而是将不同传感器产品表示为变长集合，并同时利用预训练 Qwen-ViT特征和原始物理数值残差。

第二，**Qwen更新的 mask queries与多源 evidence anchors**。mask queries真正经过 Qwen，并在所有多源 view和 evidence anchors之后生成，比“Qwen只做文本编码器”更接近真正的 VLM分割。

第三，**null-aware query-spatial multi-source fusion + no-box proposal-set decoding**。不同滑坡候选可在不同尺度、位置和模态上选择证据，并通过多 proposal union处理多斑块滑坡，而不是用一个 bbox约束整个滑坡。

需要准确区分已有思想与本文创新。mask tokens经过 LMM、proposal生成与分类解耦来自 PSALM；预训练 ViT中间层、浅层细节和 mask-aware refinement受 Qwen3-VL-Seg启发。真正需要通过实验建立的新贡献是：

> 这些机制能否在异构传感器、任意模态子集、不同 GSD和多斑块滑坡条件下形成稳定收益。

DisasterM3的结果也表明，PSALM在灾害 referring segmentation中经领域微调可获得显著提升，并且多 mask tokens优于单 token路线；但其跨光学—SAR场景仍存在明显差距。 这说明你的方向有充分动机，但最终价值取决于是否真正缩小跨传感器差距。

---

# 四、当前仍需处理的问题

## 1. 当前 Qwen视觉输入仍是“缓存 token注入”，不是原生在线多模态前向

正式训练会移除 Qwen视觉塔，将缓存的 visual tokens经过线性映射后，以 `inputs_embeds`形式注入 language decoder。

这对单卡训练很合理，但它不完全等价于 Qwen3-VL原生视觉前向：缓存 token没有显式复用原生图像 grid对应的完整 mRoPE或多模态 position protocol。因此论文中应表述为：

**Qwen language decoder conditioned on cached multi-view visual tokens**

而不宜直接声称“完整端到端 Qwen3-VL视觉 grounding”。

这不是阻断问题，因为密集空间定位由 SANE和PMRD完成。但正式实验需要加入：

* frozen Qwen mask-query；
* QLoRA mask-query；
* text-only；
* shuffled-view；
* native或近似在线 Qwen小样本对照。

只有 QLoRA 和正常多视图证据稳定优于这些对照，才能证明 Qwen语义链路的必要性。

## 2. “预训练1/2细节特征”仍应谨慎表述

缓存的四层空间特征实际大小约为 16、8、6、4，再插值到 detail/high/mid/low目标尺度。

因此真正的1/2高频细节主要来自 raw physical branch，而不是从 Qwen-ViT直接获得原生1/2特征。当前设计是合理的“预训练语义 + 原始浅层细节”，但不能表述为 Qwen-ViT本身提供了原生半分辨率边界。

此外，同一模态可能生成多个 view，例如 Sentinel-2真彩与假彩。`features_for`目前对这些 view的同层空间特征做加权平均。 这可能把真彩和假彩的差异过早混合。若后续消融发现 S2提升有限，应将简单平均改为小型 view-attention，而不是增加新的大模块。

## 3. QMEF 的 evidence强度可能被重复衰减

当前 reliability已经用于形成 fused features；query attention中又加入 `log reliability`；得到 query context后再乘 `real_reliability_mass`；PMRD构建 query-specific detail时又乘一次 real evidence mass。

当 null evidence较高时，这可能造成多次衰减，导致视觉证据过弱。建议将 null/reliability的幅度作用限定在一处：

* reliability作为 attention prior；
* 或 reliability作为特征幅度；
* 最终再统一乘一次 real evidence mass。

不建议三处同时起作用。正式训练前应记录 null mass分布和 query context norm，检查是否存在大量接近零的 evidence。

## 4. 尺度对齐中仍有一个数值坐标约定不一致

QMEF先用 `F.interpolate(..., align_corners=False)`，随后 `grid_sample(..., align_corners=True)`。

这可能产生半像素偏移，尤其影响小滑坡和细边界。应统一为同一种坐标约定，并增加一个已知平移图案的对齐单元测试。这是一个应在正式大规模训练前修复的小问题。

## 5. 缺失模态实验的有效区域需要双重报告

当前 `valid_mask`等于目标 canvas有效区与 active modalities有效覆盖区的交集。

这在部署意义上合理：模型只对当前确实有观测的区域负责。但在比较“完整模态”和“子模态”性能时，两者可能实际在不同像素集合上评价，导致结果不完全公平。

建议同时保存两种 mask：

* `active_support_valid_mask`：当前可用模态覆盖区；
* `reference_valid_mask`：完整样本共同或标准参考覆盖区。

正式报告应同时给出：

* deployable active-support性能；
* common/reference-support性能。

这样才能判断性能变化是因为模态信息缺失，还是因为评价范围被改变。

## 6. 16个 mask queries是否足够尚未由数据证明

当前正式 preset使用16个 mask tokens。 coverage mode可以在组件数量超过16时训练 union，但一个 query可能需要同时覆盖多个不连续滑坡。

需要先统计每个 patch的连通域数量分布，再做8、16、32 queries消融。若超过16个有效组件的样本很少，当前设计足够；若群发滑坡样本占比较高，应增加 query数或按更大 patch分块。

同时要注意，连通域不一定等价于独立滑坡实例。应比较：

* component matching；
* coverage-set only；
* 单 query semantic baseline。

## 7. 指令数据已经可辨识，但自动派生质量仍需审核

benchmark-v2包含位置、尺度、形态、数量和 no-target指令。

但这些 target大多由原 mask自动派生。需要人工抽查以下问题：

* “最大滑坡”是否确实只有一个明确最大目标；
* “狭长”“碎片化”“紧凑”等形态规则是否稳定；
* 位置网格边界上的滑坡是否被错误分割；
* no-target是否真的为空；
* 翻转增强后指令和 target是否同步。

此外，`1-6_build_referring_targets.py` 的文件说明仍写着“当前主模型不直接使用 referring 数据”，但当前模型已经正式训练 referring任务。 这是需要清理的过时文档。

## 8. Renderer仍有两个物理表达问题

SAR第三通道当前对原始 `VV-VH`直接裁剪到 ([-1,1])。 如果输入为 dB，实际差值常超过该范围，会造成大面积饱和。应使用数据集级固定 dB差值范围或稳健分位范围。

DEM的 hillshade和 slope目前基于归一化高程计算，而不是真实高程差、像元 GSD和坡度角。因此它更接近“归一化地形梯度可视图”，不应被描述为严格物理坡度。

这些问题主要影响 Qwen视觉语义解释，不影响 raw physical branch，但正式实验前仍应修正描述或计算方法。

## 9. 最佳 checkpoint只看 positive-only Dice可能忽视空目标误报

当前默认按 positive-only Dice选择最佳 checkpoint。

这能避免空样本抬高总体指标，但也可能选中一个正样本分割很好、no-target误报严重的模型。建议同时保存：

* best-positive checkpoint；
* best-composite checkpoint。

综合分数可采用：

[
S=D_{\text{positive}}
-\lambda,FPR_{\text{empty}}
+\mu,S_{\text{instruction contrast}}
]

论文主结果应在验证集预先确定选择规则，不应根据测试集临时选择。

## 10. 科学结果尚未闭环

仓库已经提供完整的 integration gate、真实 batch反向检查、QLoRA梯度和参数更新检查，以及消融 suite。

但从当前可见代码和文档中，还不能确认以下正式结果已经完成：

* small-v2三随机种子；
* 六个 preset完整对照；
* normal与指令/视觉消融的成对显著退化；
* 跨数据集或区域留出；
* 各模态组合的完整结果；
* null evidence校准；
* 16 queries充分性。

因此当前第一阶段应定义为**实现完成，实证待闭环**，而不是已经完成论文结论。

---
