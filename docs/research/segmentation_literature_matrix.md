# 分割模型文献机制矩阵

> 状态：P1 prerequisite 阶段的文献审计，不是实验结果
> 更新时间：2026-07-24
> 证据范围：原论文、正式会议页面、作者仓库或官方文档
> 项目范围：HDF5-first、单时相、二值滑坡语义分割

## 1. 读取规则

本矩阵把论文当作“可检验机制来源”，而不是可直接采用的完整架构。支持性使用以下术语：

- **直接支持**：论文的输入和训练过程原生允许该变化；
- **条件支持**：需要固定元数据、专用输入适配器或下游头；
- **不支持**：论文没有给出该能力，不能从模型规模或相邻任务推断；
- **未证明**：论文可能能运行，但没有与本项目问题等价的实验证据。

复杂度为论文机制相对本项目 K1 的工程判断。跨论文的参数量、显存和预训练预算口径不一致，
因此没有可靠原文数字时不补造数值。

五个可训练 source 的缩写：

- **GDCLD**：以显式注册 RGB 为主；
- **LMHLD**：B/G/R/NIR；
- **LBA**：LandslideBench_agent，以显式注册光学通道为主；
- **L4S**：Landslide4Sense 多光谱/地形通道；
- **MM**：Multimodal 的 RGB/DEM/InSAR-like 通道。

## 2. 可变光谱与通道

### 2.1 DOFA

- **论文、年份、正式来源**：[Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation](https://arxiv.org/abs/2403.15356)，2024 年首发；[作者仓库](https://github.com/zhu-xlab/DOFA)。
- **解决的问题**：用一个 EO foundation encoder 处理来自不同传感器、不同波段数量和不同光谱范围的输入，而不是每个传感器训练一个 backbone。
- **核心机制**：用波长条件的动态 hypernetwork 生成与输入波段集合匹配的权重，并在五类 EO 模态上进行连续/混合预训练。
- **必需输入元数据**：每个光谱通道的可靠中心波长；若使用论文的跨传感器尺度设定，还需要与预训练协议一致的传感器和空间分辨率信息。
- **不同通道数**：直接支持光谱通道数变化，但前提是所有参与动态权重的通道拥有可靠波长。
- **缺失模态**：能接受不同波段集合不等于已证明对随机缺失、错误 validity 或整模态缺失鲁棒；本项目仍需独立 modality-dropout 实验。
- **不同 GSD/尺度**：论文面向异构 GSD，但动态光谱权重本身不解决像素到物理尺度的绑定。
- **dense segmentation**：可作为下游 dense task encoder；它不是完整的二值分割 decoder、loss 或 no-target 协议。
- **参数/显存/训练复杂度**：ViT foundation-model 级 encoder、动态权重生成和跨数据集预训练，显著高于 K1 的共享卷积 stem；预训练权重还引入外部 lineage。
- **五源适用性**：LMHLD 与 L4S 的光谱通道只有在 HDF5 明确给出可靠波长后才可能适用；GDCLD/LBA 的 RGB 价值有限；MM 的 DEM/InSAR 不得接受波长条件。
- **最小可采用机制**：只保留“物理元数据必须显式、known/unknown 必须区分”这一原则；若未来出现可靠波长 cohort，再对纯光谱通道预注册一个动态 stem 消融。
- **拒绝照搬**：当前不采用动态 hypernetwork、foundation 预训练和完整 ViT。现有 source contract 没有可作为模型条件的可靠逐通道波长，强行补值会违反 authority。

### 2.2 OFA-Net

- **论文、年份、正式来源**：[One for All: Toward Unified Foundation Models for Earth Vision](https://arxiv.org/abs/2401.07527)，2024。
- **解决的问题**：避免因 EO 模态或空间分辨率改变而切换整套 backbone。
- **核心机制**：多个输入域共享一个 Transformer backbone，通过 masked image modeling 在整理后的多模态集合上联合预训练。
- **必需输入元数据**：至少需要知道输入属于哪个适配/patch-embedding 协议及其空间分辨率；论文不能替代本项目的 channel identity 和 validity。
- **不同通道数**：条件支持；共享的是 backbone，输入适配仍需与各模态张量结构匹配，并非任意标量 channel set。
- **缺失模态**：没有直接证明任意 subset、随机 modality dropout 或零填充缺失的等价性。
- **不同 GSD/尺度**：面向不同空间分辨率，但依赖预训练数据和适配方式；未知 GSD 不能被模型名称自动处理。
- **dense segmentation**：可迁移到下游任务；共享 encoder 本身不是 dense segmentation 完整实现。
- **参数/显存/训练复杂度**：共享 Transformer 降低“多 backbone”总量，但单模型仍是 foundation 预训练成本；相对 K1 需要更多 token 内存。
- **五源适用性**：共享特征抽取思想对五源都相关；固定 modality adapter 对 LMHLD/L4S/MM 的可变 channel subset 不充分。
- **最小可采用机制**：共享主干，输入差异在最窄的 typed channel stem 中吸收。
- **拒绝照搬**：不采用大规模 masked pretraining、固定模态适配器集合或 Transformer backbone；它们尚未证明比共享 CNN 的 masked set fusion 更必要。

### 2.3 S2MAE

- **论文、年份、正式来源**：[S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data](https://openaccess.thecvf.com/content/CVPR2024/html/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.html)，CVPR 2024。
- **解决的问题**：让预训练模型同时学习高/多光谱数据的空间结构和连续光谱结构。
- **核心机制**：3D Transformer、空间-光谱 embedding、紧凑 cube token 和 90% masked autoencoding；在超过百万幅光谱图像上渐进预训练。
- **必需输入元数据**：稳定的通道顺序、空间配准，以及通道轴具有局部光谱连续性的假设。
- **不同通道数**：对论文覆盖的光谱输入有一定适配性，但不等价于无序、异构的任意 channel set。
- **缺失模态**：3D 随机遮挡是预训练策略，不是 channel validity 或整模态缺失合同。
- **不同 GSD/尺度**：没有把可靠物理 GSD 作为核心条件；像素尺度增强不能替代 GSD。
- **dense segmentation**：预训练表示可下游微调；MAE 本身不是 landslide dense decoder。
- **参数/显存/训练复杂度**：3D tokenization、Transformer 和百万级预训练明显高于 K1，且 token 成本随空间/光谱体积增长。
- **五源适用性**：LMHLD 和 L4S 的连续光谱子集可能受益；GDCLD/LBA 的 RGB 信息有限；MM 的 RGB/DEM/InSAR 不满足连续光谱轴假设。
- **最小可采用机制**：仅把“光谱身份不能退化为匿名通道序号”写入 metadata 编码与消融。
- **拒绝照搬**：不采用 3D cube token、90% MAE 或连续光谱假设，因为它会把 DEM、slope、InSAR 和光学通道错误地视作一条光谱序列。

## 3. 多模态与缺失模态

### 3.1 MultiMAE

- **论文、年份、正式来源**：[MultiMAE: Multi-modal Multi-task Masked Autoencoders](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7102_ECCV_2022_paper.php)，ECCV 2022；[作者项目页](https://multimae.epfl.ch/)。
- **解决的问题**：让同一网络在 RGB 单输入或 RGB 加额外模态时都能迁移，并通过跨模态预测学习互补信息。
- **核心机制**：跨空间 patch 和输入模态分配有限可见 token，联合重建多个输出；训练中的 modality/patch masking 强迫跨模态预测。
- **必需输入元数据**：预先定义的模态类型、对齐关系、各模态输入/输出 adapter 和有效值预处理。
- **不同通道数**：仅条件支持已注册模态 adapter；不是任意 channel identity 的开放集合。
- **缺失模态**：直接支持“部分已知模态可用”的训练/推理思想，是本项目 dropout 假设的主要机制证据。
- **不同 GSD/尺度**：原设定依赖空间对齐，未直接解决不同物理 GSD。
- **dense segmentation**：论文在 semantic segmentation 等 dense transfer 任务上评估。
- **参数/显存/训练复杂度**：多 adapter、Transformer、重建 heads 和预训练损失高于监督 K1；有限可见 token 缓解但不消除成本。
- **五源适用性**：对 MM 的 RGB/DEM/InSAR 和 L4S 的异构通道最相关；对五源统一训练提示应显式随机丢弃可用证据。
- **最小可采用机制**：训练期 channel/modality dropout，缺失标记与真实零严格分离，并报告单模态推理退化。
- **拒绝照搬**：不采用伪标签多任务输出、MAE 重建 heads 或固定 modality adapter 清单；当前监督目标只有可靠 binary mask。

### 3.2 OmniSat

- **论文、年份、正式来源**：[OmniSat: Self-Supervised Modality Fusion for Earth Observation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4127_ECCV_2024_paper.php)，ECCV 2024；[作者仓库](https://github.com/gastruc/OmniSat)。
- **解决的问题**：利用对齐的单日期、高分辨率、光学时间序列和雷达等 EO 模态进行无监督融合。
- **核心机制**：模态专用编码后进行 token 对齐/跨模态融合，利用共同覆盖区域的 alignment 构造自监督目标。
- **必需输入元数据**：可靠空间共注册、模态身份；时间序列分支还需时间结构。没有 alignment 时论文的正对齐假设不成立。
- **不同通道数**：条件支持论文注册的传感器/模态，不直接支持任意 channel enumeration。
- **缺失模态**：论文报告多模态预训练在推理只剩一个模态时仍可迁移，是缺失模态训练的正面证据；本项目仍需针对 validity 重放。
- **不同 GSD/尺度**：能融合不同 EO 数据类型，但需要明确对齐和各模态 encoder，不能把 unknown GSD 当 known。
- **dense segmentation**：包含 land-cover 等空间预测下游，但 foundation fusion 不是本项目二值 decoder。
- **参数/显存/训练复杂度**：多个模态 encoder、attention fusion、自监督预训练和对齐数据构建显著高于 K1。
- **五源适用性**：MM 最接近论文的互补模态场景；L4S 次之；GDCLD/LMHLD/LBA 的收益取决于实际存在的多模态而非 source 名称。
- **最小可采用机制**：只采用“模态对齐必须显式”和“训练时模拟可用模态子集”的原则。
- **拒绝照搬**：不采用多 encoder 自监督堆栈、时间序列模块或 cross-modal token alignment；当前 HDF5 是单时相监督任务，先验证 masked mean。

### 3.3 AnySat

- **论文、年份、正式来源**：[AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)，CVPR 2025；[作者仓库](https://github.com/gastruc/AnySat)。
- **解决的问题**：用一个自监督 EO 模型处理不同分辨率、物理覆盖尺度、模态、时间长度和通道数。
- **核心机制**：JEPA 训练；模态专用 projector 把不同形状的子 patch 映射到共享空间；基于 GSD 的位置编码和 scale-adaptive spatial encoder；attention combiner 融合可用模态。
- **必需输入元数据**：模态身份、空间共注册、每模态可靠 meters-per-pixel、patch 的物理尺寸；时间模态还需要时间轴。
- **不同通道数**：条件支持已注册 modality projector；不是无需 schema 的任意 channel set。
- **缺失模态**：combiner 可消费 available modalities，但对本项目 channel-level missing/validity 仍需单独验证。
- **不同 GSD/尺度**：直接支持，且明确依赖可靠 GSD 和物理 patch size。
- **dense segmentation**：论文下游包括 flood、burn scar、deforestation segmentation；预训练 encoder 仍需任务头。
- **参数/显存/训练复杂度**：多 projector、shared Transformer、attention combiner 和 JEPA 多数据集预训练高于 K1；物理 patch token 数随尺度变化。
- **五源适用性**：若未来五源存在可信 GSD，scale-adaptive 思想具有价值；当前逐样本 GSD 未获可靠绑定，五源都不能据名称启用该条件。
- **最小可采用机制**：`gsd_known` 与 unknown embedding；只在 verified cohort 上比较显式 GSD 条件。
- **拒绝照搬**：当前不采用 JEPA、模态 projector 集合、时间 encoder 或 attention combiner；先用共享 scalar stem 与 masked mean 证伪 H1。

### 3.4 Scale-MAE

- **论文、年份、正式来源**：[Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.html)，ICCV 2023。
- **解决的问题**：避免把不同地面覆盖尺度的 EO 图像仅当作普通 resize augmentation。
- **核心机制**：用已知输入地面覆盖尺度调整 ViT 位置编码；decoder 分别重建低/高频、低/高尺度目标。
- **必需输入元数据**：可靠的输入 GSD 或图像地面覆盖范围；仅有像素宽高不够。
- **不同通道数**：不是核心能力；主要面向影像 scale。
- **缺失模态**：不支持。
- **不同 GSD/尺度**：直接支持已知尺度，并明确区分物理尺度和数组分辨率。
- **dense segmentation**：论文报告 SpaceNet building segmentation transfer 增益；预训练目标仍不是任务 mask。
- **参数/显存/训练复杂度**：ViT MAE、双频重建 decoder 和预训练阶段高于 K1；推理 encoder 成本仍为 Transformer 级。
- **五源适用性**：只有出现可靠 GSD/footprint 的 source cohort 才能检验；当前五源均不能用近似说明补数值。
- **最小可采用机制**：在 transform record 中分离 `array_resize` 与 `physical_gsd`，为 unknown GSD 保留显式状态。
- **拒绝照搬**：不采用 MAE 预训练、频率重建 decoder 或用未知 GSD驱动位置编码；这不能解决当前首要的 channel validity。

## 4. 轻量语言条件分割

### 4.1 CLIPSeg

- **论文、年份、正式来源**：[Image Segmentation Using Text and Image Prompts](https://openaccess.thecvf.com/content/CVPR2022/html/Luddecke_Image_Segmentation_Using_Text_and_Image_Prompts_CVPR_2022_paper.html)，CVPR 2022；[作者代码页](https://eckerlab.org/code/clipseg)。
- **解决的问题**：用测试时文本或示例图像指定开放查询，而不是把类别集合固定在训练期。
- **核心机制**：以 CLIP 为 backbone，加入 Transformer dense decoder，把 prompt embedding 与视觉特征结合并输出 binary mask。
- **必需输入元数据**：RGB 图像和确实改变分割目标的文本/图像 prompt；不消费波长、GSD 或 channel validity。
- **不同通道数**：不支持；CLIP visual encoder 是固定 RGB 接口。
- **缺失模态**：不支持本项目的光谱/地形/radar 缺失；prompt 类型变化不是传感器缺失。
- **不同 GSD/尺度**：没有物理 GSD 条件。
- **dense segmentation**：直接支持文本/图像条件的 binary dense segmentation。
- **参数/显存/训练复杂度**：冻结或微调 CLIP 再加 Transformer decoder，明显高于单层 FiLM；RGB visual tower 还会排除原始非 RGB 通道。
- **五源适用性**：仅能消费五源注册 RGB view，不能成为 LMHLD/L4S/MM 全证据的主 spatial encoder。
- **最小可采用机制**：复用“prompt embedding 只需在窄接口调制 dense decoder”的思想，并运行 null/random/equivalent prompt 对照。
- **拒绝照搬**：不采用 CLIP visual tower、PhraseCut 预训练或开放类假设；本项目 binary target 若始终相同，文本可能没有新增信息。

### 4.2 CRIS

- **论文、年份、正式来源**：[CRIS: CLIP-Driven Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_CRIS_CLIP-Driven_Referring_Image_Segmentation_CVPR_2022_paper.html)，CVPR 2022。
- **解决的问题**：把 referring expression 与像素级视觉特征精细对齐。
- **核心机制**：CLIP 视觉/文本表示、vision-language decoder，以及 text-to-pixel contrastive learning。
- **必需输入元数据**：RGB 图像与有区分力的 referring expression；像素-文本正负对应关系必须由标注支持。
- **不同通道数**：不支持固定 RGB 以外的传感器 channel set。
- **缺失模态**：不支持 EO missing modality。
- **不同 GSD/尺度**：没有物理尺度建模。
- **dense segmentation**：直接用于 referring image segmentation。
- **参数/显存/训练复杂度**：CLIP 双 encoder、跨模态 decoder 和额外 contrastive objective 高于 K2 单 FiLM。
- **五源适用性**：五源只有统一 landslide binary mask，当前没有对象级多目标 referring supervision；只能以注册 RGB 做不完整输入。
- **最小可采用机制**：把 prompt sensitivity 定义成可测的 logit/mask 差异，而不是假定语言一定有益。
- **拒绝照搬**：不采用 pixel-text contrastive loss或多层 vision-language decoder；没有合法的 pixel-text 正负标注时会制造监督。

### 4.3 DenseCLIP

- **论文、年份、正式来源**：[DenseCLIP: Language-Guided Dense Prediction With Context-Aware Prompting](https://openaccess.thecvf.com/content/CVPR2022/html/Rao_DenseCLIP_Language-Guided_Dense_Prediction_With_Context-Aware_Prompting_CVPR_2022_paper.html)，CVPR 2022。
- **解决的问题**：把 CLIP 的全图-文本匹配知识迁移到像素级语义分割、检测和实例分割。
- **核心机制**：将 image-text matching 改写为 pixel-text matching，以 score map 指导 dense model，并用图像上下文调整文本 prompt。
- **必需输入元数据**：RGB 图像、固定或开放的类别文本集合，以及像素类别监督。
- **不同通道数**：不支持原生可变 EO channel。
- **缺失模态**：不支持。
- **不同 GSD/尺度**：不支持物理 GSD；普通多尺度特征不能视作物理 scale conditioning。
- **dense segmentation**：直接支持多类 semantic segmentation 等 dense task。
- **参数/显存/训练复杂度**：CLIP 特征、像素-文本 score maps 与 context prompt 模块增加明显计算；多类别文本矩阵对二值固定目标没有同等收益来源。
- **五源适用性**：只能以注册 RGB 进入 CLIP；对 LMHLD NIR、L4S 地形/多光谱、MM DEM/InSAR 丢失证据。
- **最小可采用机制**：将语言注入限制在 decoder/bottleneck，而不让语言承担空间编码。
- **拒绝照搬**：不采用 CLIP dense visual path、类别 score-map head 或 context prompt learning；本项目不是开放类别 semantic segmentation。

## 5. LMM/VLM 像素分割复杂度上界

### 5.1 LISA

- **论文、年份、正式来源**：[LISA: Reasoning Segmentation via Large Language Model](https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html)，CVPR 2024；[作者仓库](https://github.com/dvlab-research/LISA)。
- **解决的问题**：根据需要常识或隐式推理的文本查询输出目标 mask。
- **核心机制**：扩展 LMM 词表加入 `<SEG>` token，把其 hidden embedding 作为 mask 条件并交给分割模型解码。
- **必需输入元数据**：RGB、复杂且确实需要推理的 query、文本生成监督与 mask；还依赖 LMM 和分割模型的预训练权重。
- **不同通道数**：不支持原始 EO channel set。
- **缺失模态**：不支持传感器模态缺失。
- **不同 GSD/尺度**：无可靠物理 GSD 条件。
- **dense segmentation**：直接生成 mask，但空间信息仍来自专用视觉/分割路径。
- **参数/显存/训练复杂度**：LMM、视觉编码器、分割 decoder 和自回归训练构成数量级更高的上界，远超 K1/K2。
- **五源适用性**：五源目标都是显式 landslide binary mask，当前无复杂隐式推理监督；非 RGB 通道也不能直接进入其视觉塔。
- **最小可采用机制**：仅确认“语言表示可以作为窄条件向量”，这已由 K2 单 FiLM 覆盖。
- **拒绝照搬**：拒绝 `<SEG>` 自回归 token、LLM mask 控制、SAM 类 decoder 和 reasoning 数据混训；没有与当前任务匹配的信息增量。

### 5.2 PixelLM

- **论文、年份、正式来源**：[PixelLM: Pixel Reasoning with Large Multimodal Model](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_CVPR_2024_paper.html)，CVPR 2024。
- **解决的问题**：让 LMM 对开放世界、多目标推理请求生成多个像素级 mask，同时避免额外的大型 segmentation foundation model。
- **核心机制**：segmentation codebook tokens、轻量 pixel decoder 和 token fusion，把多个语言 hidden embeddings 解码为 masks。
- **必需输入元数据**：RGB、目标有区分力的语言、多目标/推理监督和 codebook-token 对齐。
- **不同通道数**：不支持 EO 可变通道。
- **缺失模态**：不支持传感器缺失。
- **不同 GSD/尺度**：无物理尺度条件。
- **dense segmentation**：直接支持多目标 reasoning segmentation。
- **参数/显存/训练复杂度**：pixel decoder 虽相对 SAM 轻，但完整 LMM、视觉编码和自回归 token 学习仍远高于 K2。
- **五源适用性**：本项目是单目标类别的 binary dense mask，五源没有多目标语言 codebook 真值；只使用 RGB 还会损失关键模态。
- **最小可采用机制**：如果未来 prompt 有真实信息，只保留低秩/FiLM 的 hidden-vector conditioning，不引入 mask tokens。
- **拒绝照搬**：拒绝 codebook、token fusion、自回归语言训练和 open-world reasoning 数据。

### 5.3 GSVA

- **论文、年份、正式来源**：[GSVA: Generalized Segmentation via Multimodal Large Language Models](https://openaccess.thecvf.com/content/CVPR2024/html/Xia_GSVA_Generalized_Segmentation_via_Multimodal_Large_Language_Models_CVPR_2024_paper.html)，CVPR 2024。
- **解决的问题**：对一个表达中的多个对象生成多个 mask，并在查询对象不存在时显式拒绝。
- **核心机制**：多个 `[SEG]` token 驱动分割模型，新增 `[REJ]` token 表示 absent referent。
- **必需输入元数据**：RGB、referring expressions、多对象 mask、empty-referent 标注和 LMM/segmentation-model 权重。
- **不同通道数**：不支持。
- **缺失模态**：不支持 EO 模态缺失。
- **不同 GSD/尺度**：不支持物理尺度。
- **dense segmentation**：直接支持 generalized referring segmentation。
- **参数/显存/训练复杂度**：论文使用 7B 级 MLLM 路线并连接分割模型，远超 24 GiB 内简洁 K1 的科学需求。
- **五源适用性**：no-target 标签与“文本所指对象不存在”在统计上相似，但语义不同；五源 no-target 是 binary landslide mask 为空，不需要 LLM 拒绝 token。
- **最小可采用机制**：只采用显式 no-target 指标和 false-positive gate，不采用 token 架构。
- **拒绝照搬**：拒绝 `[SEG]/[REJ]`、多对象语言生成和外部分割模型；直接的空 mask supervision 更简单且更强。

### 5.4 GLaMM

- **论文、年份、正式来源**：[GLaMM: Pixel Grounding Large Multimodal Model](https://openaccess.thecvf.com/content/CVPR2024/html/Rasheed_GLaMM_Pixel_Grounding_Large_Multimodal_Model_CVPR_2024_paper.html)，CVPR 2024。
- **解决的问题**：生成自然语言回复并将回复中的对象概念与像素 mask 交织绑定，同时接受文本和可选视觉区域 prompt。
- **核心机制**：全局图像 encoder、grounding image encoder、region encoder、LLM、语言到像素投影和 pixel decoder 的组合；在大规模自动 grounded-conversation 数据上训练。
- **必需输入元数据**：RGB、区域/对话 grounding、对象 mask、语言生成监督和多套预训练模型。
- **不同通道数**：不支持原始 EO channel set。
- **缺失模态**：不支持 EO missing modality。
- **不同 GSD/尺度**：无物理 GSD 条件。
- **dense segmentation**：能输出对象 masks，但主要任务是 grounded conversation，不是单一 binary semantic mask。
- **参数/显存/训练复杂度**：论文补充材料报告 7B Vicuna、双图像 encoder、LoRA 和 pixel decoder，并使用 8 张 A100 40 GB 训练；明显超过本项目单卡上限。
- **五源适用性**：对五源没有必要的 grounded-conversation supervision，且 RGB 双塔不能消费 LMHLD/L4S/MM 的完整证据。
- **最小可采用机制**：没有超出 K2 pooled query + FiLM 所需的最小机制。
- **拒绝照搬**：拒绝双视觉 encoder、region tokenizer、LLM 自回归、自动标注流水线和 grounded conversation 数据；其任务和预算都不匹配。

## 6. 跨论文机制结论

| 项目问题 | 最小被支持机制 | 当前结论 |
| --- | --- | --- |
| 可变通道数 | typed scalar-channel stem + 显式 identity | 先检验 K1；DOFA 动态权重因波长 unknown 暂不具备资格 |
| 缺失通道/模态 | validity-aware aggregation + training-time dropout | MultiMAE/OmniSat 支持这一训练原则，但不支持照搬完整预训练架构 |
| 通道顺序 | set aggregation | 论文没有替代本项目 permutation gate 的证据，必须自行验证 |
| 多尺度 | CNN pyramid | 先处理图像尺度；只有 `gsd_known=true` cohort 才检验 AnySat/Scale-MAE 式物理条件 |
| 语言条件 | pooled text embedding + 单层 FiLM | CLIPSeg/CRIS/DenseCLIP 证明语言可调制 dense prediction，不证明固定 landslide prompt 有信息 |
| no-target | 直接空 mask 监督与 FPR | GSVA 的拒绝思想可作为评测提醒，不需要 LMM token |
| 复杂 LMM | 无 | LISA/PixelLM/GSVA/GLaMM 是复杂度上界，不是默认候选 |

文献审计只支持从 K1 开始并对 K2 设置信息量门。它没有产生任何本项目 Dice、IoU、显存、
吞吐或优胜候选结果。
