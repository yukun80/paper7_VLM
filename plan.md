## 一、研究背景说明

本研究要解决的是地质灾害遥感智能判读中的可靠性问题。我们不是要训练一个覆盖所有遥感场景的通用大模型，而是要在单卡 RTX 4090 低算力条件下，基于 Qwen3-VL-2B/4B 这类小参数视觉语言模型，构建一个面向崩塌、滑坡、泥石流、岩屑坡识别的多传感器 benchmark，并进一步验证质量感知训练、证据定位约束和 GeoSODA-lite 半 on-policy 蒸馏是否能提升小模型在地灾识别中的准确率、拒答能力和定位可靠性。当前研究设想已经明确：数据对象包括星载光学、SAR、InSAR 形变速率/累计形变图、无人机巡查图像、无人机地灾现场图像、非地灾困难负样本和地灾行业规程文本；模型对象以 Qwen3-VL-2B 和 Qwen3-VL-4B 为主；任务只保留 image-level 和 grounding-level，不设置 region-level。

这个问题在实际应用中很重要。地质灾害隐患识别通常服务于山区巡查、灾后应急、工程边坡排查、InSAR 隐患筛查和无人机复核。如果模型把裸岩、采石场、河滩、道路边坡、施工扰动、阴影、云雪、SAR 斑点噪声或 InSAR 低相干区误判为滑坡，会造成大量误报；如果模型面对模糊、低分辨率、缺少色标的 InSAR 图像时仍然高置信度输出“活动滑坡”，则会误导后续人工核查和风险排序。RSHallu 已经指出，遥感多模态大模型的幻觉会影响应急管理等高风险场景，且遥感幻觉不仅是对象不存在的问题，还包括模态、分辨率和场景级语义错误。

现有方法大致有三类。第一类是传统遥感分类、检测、分割模型，例如 CNN、ViT、YOLO、Mask R-CNN、U-Net、SegFormer、SAM/SAM2 或滑坡专用分割模型。这类方法能完成特定任务，但通常只能输出类别、框或掩膜，不能解释“为什么判断为地灾”“证据是否充分”“是否需要补充 InSAR 或无人机核查”。第二类是通用或遥感视觉语言模型，例如 Qwen-VL、LLaVA、GeoChat、EarthDial、VHM、Earth-OneVision 等。这类模型可以做图像描述、问答和定位，但多数工作面向通用遥感场景或广义灾害场景，缺少针对崩滑流、岩屑坡、InSAR 形变证据和地灾规程约束的专业设计。第三类是遥感幻觉评估和诚实问答方法，例如 RSHallu、VHM/H2RSVLM 等，它们关注模型不要对不存在对象乱答，但还没有充分覆盖地灾低质影像、InSAR 证据误读和 grounding-level 证据失配。

现有方法的不足主要体现在四个方面。第一，很多遥感 VLM 追求大数据、多任务和通用模型，训练条件远超我们当前的一块 4090。第二，已有遥感幻觉研究多针对通用遥感对象，还没有系统讨论地灾场景中的“低质影像诱导幻觉”，例如云影、模糊、SAR 噪声、InSAR 低相干和无人机倾斜视角导致的过度判断。第三，很多 VLM 即使能回答“图中有滑坡”，也不能保证它定位到了正确证据区域。第四，InSAR 形变图不是普通 RGB 图像，必须结合时间跨度、单位、LOS 方向、升降轨、色标和相干性说明，否则模型容易编造形变证据。

因此，本研究需要尝试“小模型、强约束、可解释、可拒答、可定位”的路线。深度研究报告建议将主线收缩为 GeoHazard-HalluGround benchmark、质量感知诚实 SFT、grounding 证据约束和 GeoSODA-lite 学生错误驱动蒸馏。该路线的目标不是让模型替代专家做最终地质判定，而是让模型成为地灾遥感初筛和辅助判读工具：能识别、能说明依据、能定位证据、能判断证据是否充分，并且在看不清或证据不足时不乱答。

本研究希望解决的关键问题是：在低算力、小数据、强专业约束条件下，能否通过 benchmark 设计和轻量后训练，使 Qwen3-VL-2B/4B 在崩塌、滑坡、泥石流、岩屑坡识别中实现更高准确率、更低幻觉率、更强拒答能力和更可靠的视觉定位能力。

---

## 二、核心研究目标

目标 1：构建一个面向崩滑流与岩屑坡识别的多传感器 GeoHazard-HalluGround benchmark。

该 benchmark 需要同时支持 image-level 和 grounding-level 两类任务。image-level 包括图像描述、多标签分类、地灾 VQA、图像质量判断、证据充分性判断、可回答性判断和拒答。grounding-level 包括疑似地灾体、滑坡体、崩塌源区、泥石流沟道/堆积扇、岩屑坡范围的 bbox 定位和 mask 分割。验证标准是完成数据清单、标签体系、instruction 模板、训练/验证/测试划分、评估脚本和一批可视化标注案例。

目标 2：验证 Qwen3-VL-2B/4B 在多传感器地灾任务中的零样本能力和微调提升空间。

需要比较 Qwen3-VL-2B、Qwen3-VL-4B，以及可选的 Qwen2.5-VL-3B、TinyRS-R1 等模型在 image-level 和 grounding-level 上的表现。验证标准是得到 zero-shot、prompt-only、普通 SFT、quality-aware SFT、grounding SFT 后的定量结果，明确 2B 与 4B 的差异，分析小模型在哪些传感器、哪些地灾类型和哪些质量条件下最容易失败。研究计划中已将 Qwen3-VL-2B/4B 作为主模型，并把 Qwen2.5-VL-3B、TinyRS-R1 等作为可选对照。

目标 3：验证质量感知诚实 SFT 是否能降低低质影像和无灾害困难负样本中的幻觉与过度自信。

重点考察云遮挡、阴影、模糊、低分辨率、SAR 噪声、InSAR 低相干、缺失色标、无人机遮挡等情况下，模型是否仍然高置信度输出错误地灾判断。验证标准是：低质图像分层准确率提高，false alarm rate、overclaim rate、unanswerable QA error 下降，模型能够更合理地输出“证据不足”“需补充 InSAR/UAV/现场核查”。

目标 4：验证 GeoSODA-lite 学生错误驱动蒸馏是否能进一步提升小模型准确率和可靠性。

先让 SFT 后的小模型在困难负样本、低质图像和 grounding 错误样本上生成回答，建立 student-error bank；再用教师模型、规程知识、规则检查器和少量专家抽检生成纠错答案；最后进行 correction SFT 或 DPO/ORPO。验证标准是：GeoSODA-lite 相比 quality-aware SFT，在地灾误报、证据幻觉、定位错误、规程违规和过度自信指标上有稳定改进。深度研究报告建议主实验至少包含 zero-shot、普通 SFT、质量感知 SFT、困难负样本扩充、grounding 约束和 GeoSODA-lite 六组。

---

## 三、研究假设或预期结论

假设 1：Qwen3-VL-2B/4B 经过地灾 instruction 微调后，可以在 image-level 地灾描述、分类和 VQA 上明显优于 zero-shot。

为什么可能成立：Qwen3-VL 已具备通用视觉语言能力和 grounding 基础，地灾任务虽然专业，但 image-level 描述、分类和 VQA 可通过高质量 instruction 数据快速适配。Earth-OneVision 的 2B 遥感 MLLM 已证明，小参数模型在多传感器、多任务遥感场景中具有可竞争表现；其设计覆盖六类传感器、九类任务，并通过 FGVLA、SLIS、PCMA 处理视觉语言对齐、空间输出统一和跨模态适配问题。

需要用什么结果证明：比较 zero-shot 与 SFT 后的 Accuracy、F1、VQA accuracy、caption factuality score、结构化输出正确率。如果 SFT 后多个 image-level 指标稳定提升，说明假设成立。

如果不成立，可能解释为：训练数据质量不足；instruction 模板过于机械；类别定义不清；Qwen3-VL 对遥感俯视视角适配不足；输入分辨率或裁剪方式损失了地灾关键证据；或训练集与测试集传感器分布差异过大。

假设 2：加入质量标签、证据充分性标签和拒答样本后，模型在低质图像和无灾害困难负样本上的幻觉会减少。

为什么可能成立：遥感幻觉与模态、分辨率、场景语义等图像级因素相关。RSHallu 将 image-level hallucination 纳入遥感幻觉 taxonomy，并提出 RSHalluEval、RSHalluCheck 和 RSHalluShield，支持遥感幻觉评估和缓解。 本课题将这一思想地灾化，把“质量不足时不可判断”“无 InSAR 输入不能编造形变”“低相干区域不能直接等同活动滑坡”显式加入训练目标。

需要用什么结果证明：低质图像中的 overconfident hallucination rate 降低；无灾害困难负样本中的 false alarm rate 降低；unanswerable QA accuracy 提高；模型输出“证据不足”“建议补充数据”的比例更合理。

如果不成立，可能解释为：低质标签不一致；拒答样本数量过少；模型学成了“低质图像一律拒答”；正负样本比例不平衡；评价指标没有区分“合理不确定”和“错误拒答”。

假设 3：grounding-level 证据定位可以减少证据幻觉，使模型回答更可信。

为什么可能成立：仅有 image-level 回答时，模型可能说对类别但找错证据区域；加入 bbox/mask 后，模型必须把“判断为地灾”与具体空间区域对应起来。Earth-OneVision 使用 SLIS 将文本、坐标和分割 token 统一到自回归输出中，并在消融中显示 SLIS 对 grounding 有帮助。

需要用什么结果证明：grounding P@0.25、P@0.5、mIoU、mask IoU、Dice 提升；错误案例中“判断为滑坡但框在阴影、采石场、道路或河滩上”的比例下降；evidence consistency score 提高。

如果不成立，可能解释为：bbox/mask 标注质量不足；地灾边界本身模糊；模型坐标输出格式不稳定；高分辨率图像被压缩后关键证据丢失；或者 grounding 数据量不足以改变模型的视觉注意模式。

假设 4：GeoSODA-lite 能进一步纠正小模型在困难负样本和低质影像上的典型错误。

为什么可能成立：半 on-policy 的核心思想是让学生模型在自己生成的错误分布上学习，而不是只模仿理想答案。地灾任务中小模型的错误模式高度集中，例如把采石场当滑坡、把云影当裂缝、把 InSAR 低相干当形变异常、没有 InSAR 图却编造形变证据，因此 student-error bank 有明确价值。

需要用什么结果证明：GeoSODA-lite 后，false alarm rate、overclaim rate、evidence hallucination rate、rule violation rate 下降；student-error bank 中高频错误类别被明显纠正；普通分类、VQA 和 grounding 性能不明显下降。

如果不成立，可能解释为：教师纠错答案不可靠；偏好对质量不足；DPO/ORPO 参数不合适；错误库覆盖不够；学生模型 cold-start 不充分；或规则检查器过强导致模型过度保守。

---

## 四、实验或分析对象

1. 数据类型。

需要准备星载光学图像、SAR 图像、InSAR 形变速率/累计形变图、无人机巡查图像、无人机地灾现场记录图像、非地灾困难负样本和行业规程文本。正样本包括滑坡、崩塌、泥石流、岩屑坡、复合型崩滑流灾害。困难负样本包括裸岩、采石场、道路边坡、工程开挖面、河滩、冲沟、弃渣场、农田裸土、阴影、云雪覆盖、SAR 斑点噪声、InSAR 低相干区。

2. 时间长度。

静态 image-level 和 grounding-level 图像没有固定时间长度要求。若使用双时相或多时相样本，需要记录 T1、T2 的日期。若使用 InSAR 形变速率图，需要记录时间窗口，例如“2021-01 至 2022-12 平均 LOS 形变速率图”。无人机巡查图像若来自周期性巡查，应记录拍摄日期和巡查批次。

3. 采样频率。

普通单幅图像不涉及采样频率。InSAR 数据需要记录产品时间跨度、影像数量或时间间隔。无人机巡查数据如为周期巡查，应记录巡查周期，例如汛前、汛后、月度、季度或灾后应急。

4. 输入变量。

输入包括图像、传感器类型、任务指令和可选元数据。元数据包括传感器类型、空间分辨率、日期、地点、SAR 极化方式、升降轨、InSAR 单位、色标、时间跨度、LOS 方向、相干性说明、无人机视角或飞行高度。

5. 输出变量。

image-level 输出包括：图像描述、是否存在疑似地灾、地灾类型、多标签类别、图像质量、证据充分性、可回答性、拒答原因、主要证据、缺失证据、核查建议。

grounding-level 输出包括：地灾目标 bbox 坐标、可选 mask、定位对象类型、定位置信度、定位区域证据描述。

6. 是否需要真实值或基准值。

需要。image-level 需要专家标签、清单标签或人工审核标签，包括灾害类型、负类类型、质量等级、证据充分性、可回答性和 VQA 标准答案。grounding-level 需要 bbox 或 mask 标注。InSAR 证据需要人工判断“形变异常是否支持活动性判断”，不能让模型自动生成真值。

7. 如果没有真实值，替代评价方式。

缺少严格真值时，采用三类替代评价。第一，专家盲评，按 0、0.5、1 给模型回答打分。第二，多源交叉验证，用光学、InSAR、UAV、历史清单、现场记录互相验证。第三，弱标注或 pseudo-label，用已有滑坡分割模型、SAM/SAM2 或人工快速审核生成候选 mask，再由学生完成复核。

8. 如果是仿真数据，应该如何构造。

本研究不建议以仿真数据作为主数据来源。但可以构造质量退化数据用于鲁棒性测试，例如模糊、降分辨率、遮挡、亮度变化、压缩噪声、SAR speckle 模拟、InSAR 色标遮挡、相干性遮罩缺失。仿真退化样本只能用于鲁棒性和幻觉测试，不能替代真实低质样本。

9. 如果是真实数据，应该如何清洗和整理。

每张图像必须统一命名，保存原图、裁剪图、传感器元数据、标签文件、标注来源和数据版本。相邻切片、同一滑坡体、同一无人机航线、同一 InSAR 图的相邻 patch 不能随机同时进入训练集和测试集，避免空间泄漏。数据划分应优先按区域、事件、传感器划分，保留 region-holdout、event-holdout、sensor-holdout 三类测试。

---

## 五、方法路线

步骤 1：建立项目目录和实验环境。

这一步做什么：搭建 Qwen3-VL-2B/4B 推理与微调环境，整理代码仓库结构，包括 data、annotations、scripts、configs、outputs、logs、figures、reports、models。

为什么要做：后续数据构建、训练、评估和复现实验必须统一管理。

输入是什么：RTX 4090 工作站、Python/PyTorch 环境、Qwen3-VL 权重、LoRA/QLoRA 工具链、数据处理脚本。

输出是什么：可运行的 zero-shot 推理脚本、训练脚本、评估脚本和配置模板。

需要注意什么：所有实验必须固定随机种子，保存 config、模型版本、数据版本、训练日志和显存记录。

这一步如何进入下一步：环境稳定后，开始数据清洗和 benchmark 构建。

步骤 2：建立数据清单和标签体系。

这一步做什么：汇总星载光学、SAR、InSAR、UAV 和非地灾负样本，建立统一 metadata 表。

为什么要做：benchmark 的价值首先取决于数据组织、标签一致性和可追溯性。

输入是什么：原始图像、地灾清单、无人机记录、InSAR 图、行业规程文本、公开遥感负样本。

输出是什么：metadata.jsonl、标签定义文档、数据划分方案、标注规范。

需要注意什么：每张图像必须记录 source、sensor_type、region_id、event_id、date、resolution、hazard_label、negative_label、quality_label、evidence_labels、bbox、mask_path、split。

这一步如何进入下一步：根据 metadata 生成 image-level instruction 和 grounding 标注任务。

步骤 3：构建 pilot benchmark。

这一步做什么：先选 1000–2000 张图像构建小规模 pilot benchmark。

为什么要做：先验证任务定义、标签体系和评估脚本是否可行，避免一开始标注几万张后发现任务不可评估。

输入是什么：清洗后的图像、初步标签、少量专家审核样本。

输出是什么：pilot 版 image-level QA、分类标签、描述文本、质量标签、bbox/mask 标注。

需要注意什么：正样本、困难负样本、低质图像、InSAR 样本和 UAV 样本都要包含，不要只选清晰滑坡图。

这一步如何进入下一步：pilot benchmark 用于 zero-shot 测试、错误分析和首轮 SFT。

步骤 4：设计 instruction 模板。

这一步做什么：为每类任务设计标准 prompt 和答案格式。

为什么要做：小模型训练稳定性高度依赖 instruction 格式，必须避免同一任务有多种混乱输出格式。

输入是什么：标签体系、任务定义、专家判读规则、InSAR 元数据规范。

输出是什么：instruction 模板库，包括描述、分类、VQA、拒答、grounding、分割、InSAR 证据问答模板。

需要注意什么：答案尽量采用半结构化格式，例如“判断—类别—证据—缺失证据—质量—位置—建议”。不要鼓励自由长篇报告。

这一步如何进入下一步：模板用于生成训练集、验证集和测试集。

步骤 5：生成 VLM 指令样本。

这一步做什么：根据 metadata 自动生成 qwen_vl_sft.jsonl。每张正样本至少生成“是否存在地灾”“属于哪一类”“证据在哪里”三类问题；每张困难负样本至少生成“是否存在地灾”“为什么不应判为地灾”两类问题；每张低质样本至少生成“图像质量是否足以支持判断”一类问题。InSAR 样本必须在 prompt 中附带单位、时间段、LOS 方向和色标说明。该要求已在工程方案中明确列为指令样本生成原则。

为什么要做：将传统视觉标注转化为 VLM 可训练数据。

输入是什么：metadata、bbox/mask、质量标签、证据标签、模板库。

输出是什么：image-level instruction、grounding instruction、GeoSODA 预留字段。

需要注意什么：不得让自动生成答案编造不存在的证据。所有 InSAR 相关答案必须受元数据约束。

这一步如何进入下一步：生成的数据用于 zero-shot 测试和 SFT。

步骤 6：数据划分。

这一步做什么：按区域、事件和数据源划分训练集、验证集和测试集。

为什么要做：防止空间泄漏和事件泄漏，高估模型泛化能力。

输入是什么：metadata、region_id、event_id、sensor_type。

输出是什么：train/val/test split 文件，建议基础比例为 70%/10%/20%。

需要注意什么：禁止同一滑坡体、同一无人机航线、同一 InSAR 图的相邻 patch 同时进入训练集和测试集。测试集至少包括公开数据测试集、自采区域测试集、困难负样本测试集；尽量设置 region-holdout 和 sensor-holdout。

这一步如何进入下一步：划分完成后开始 zero-shot 基线实验。

步骤 7：Qwen3-VL 零样本基线测试。

这一步做什么：使用 Qwen3-VL-2B-Instruct 和 Qwen3-VL-4B-Instruct 在测试集上直接推理，不进行微调。

为什么要做：确定基础模型短板，例如是否把裸岩误判为滑坡、是否在没有 InSAR 输入时编造形变、是否无法输出稳定坐标格式。工程方案已明确将 zero-shot 作为第一个模型阶段。

输入是什么：测试集图像、统一 prompt、Qwen3-VL-2B/4B。

输出是什么：原始模型输出、解析后的类别/拒答/bbox 字段、zero-shot 指标表、错误样本库。

需要注意什么：必须保存每个样本的原始回答，不只保存最终指标。

这一步如何进入下一步：错误样本用于困难负样本增强和 student-error bank 初始化。

步骤 8：第一阶段普通地灾 SFT。

这一步做什么：用 image-level 描述、地灾/非地灾分类、地灾类型问答、典型证据说明对 Qwen3-VL-2B/4B 进行 LoRA/QLoRA 微调。

为什么要做：先让模型具备基本地灾术语、类别识别和输出格式能力。

输入是什么：训练集 image-level instruction、Qwen3-VL-2B/4B、LoRA 配置。

输出是什么：GeoHazard-Qwen3VL-2B-SFT-v1 和 GeoHazard-Qwen3VL-4B-SFT-v1。

需要注意什么：一块 4090 下优先冻结视觉编码器和大部分主干参数，只训练语言侧 LoRA 和必要的多模态投影层适配参数。

这一步如何进入下一步：普通 SFT 模型作为 quality-aware SFT 和 GeoSODA-lite 的 cold-start 模型。

步骤 9：第二阶段质量与困难负样本增强微调。

这一步做什么：加入困难负样本、图像质量判断、证据充分性判断、可回答性判断和保守回答样本。

为什么要做：降低无灾害误报、低质影像过度自信和证据编造。

输入是什么：普通 SFT 模型、困难负样本、低质样本、拒答样本、InSAR 元数据样本。

输出是什么：GeoHazard-Qwen3VL-2B-QA-v1 和 GeoHazard-Qwen3VL-4B-QA-v1。

需要注意什么：训练目标不是让模型一律拒答，而是让模型区分“可判断为无明显地灾”“存在疑似地灾但证据不足”“图像质量不足需补充数据”。工程方案已强调这一点。

这一步如何进入下一步：quality-aware 模型用于 grounding 微调和 student-error bank 生成。

步骤 10：第三阶段 grounding-level 定位能力微调。

这一步做什么：加入 bbox 定位任务。对有 mask 的样本先自动生成 bbox，对关键样本人工校正 bbox。训练模型按统一格式输出“类别：[x1,y1,x2,y2]”。

为什么要做：让模型判断“有地灾”时必须指出证据在哪里。

输入是什么：带 bbox/mask 的训练样本、Qwen3-VL grounding 格式、quality-aware 模型。

输出是什么：GeoHazard-Qwen3VL-2B-Grounding-v1、GeoHazard-Qwen3VL-4B-Grounding-v1、bbox 可视化结果。

需要注意什么：如果 mask 训练成本较高，工程初版只做 bbox，不强制端到端 mask 输出；需要 mask 时，可在 bbox 后调用 SAM/SAM2 或传统分割模型做框内掩膜生成。该折中路线已在工程方案中提出。

这一步如何进入下一步：grounding 错误样本进入 student-error bank。

步骤 11：构建 GeoSODA-lite 学生错误库。

这一步做什么：让 quality-aware + grounding 模型在困难负样本、低质图像和跨区域测试样本上生成回答，筛选错误样本。

为什么要做：半 on-policy 的价值在于纠正学生自己的真实错误。

输入是什么：训练外困难样本、低质样本、InSAR 样本、grounding 错误样本、当前学生模型。

输出是什么：student-error bank，包含原图、prompt、学生答案、错误类型、参考纠正答案、是否需要专家复核。

需要注意什么：错误类型至少包含无灾害误报、低质误答、证据编造、定位错位、规程误读、不可答却答六类。

这一步如何进入下一步：错误库用于构造 correction SFT 样本或 chosen/rejected pair。

步骤 12：GeoSODA-lite 蒸馏训练。

这一步做什么：用教师模型、规程 RAG、规则检查器和少量专家抽检生成纠正答案，再对小模型进行 correction SFT 或 DPO/ORPO。

为什么要做：进一步降低小模型在困难样本上的稳定错误。

输入是什么：student-error bank、教师纠错答案、偏好对、quality-aware + grounding 模型。

输出是什么：GeoHazard-Qwen3VL-2B-GeoSODA-lite 和 GeoHazard-Qwen3VL-4B-GeoSODA-lite。

需要注意什么：教师答案必须经过规则过滤，不能让教师也编造 InSAR 证据；如果 DPO/ORPO 不稳定，优先回退到 correction SFT。深度研究报告也建议 GeoSODA-lite 作为轻量增强模块，不宜一开始做完整重型 OPD/RL。

这一步如何进入下一步：最终模型进入完整评估。

步骤 13：完整评估和误差分析。

这一步做什么：在 image-level、grounding-level、低质图像、困难负样本、跨区域、跨传感器测试集上评估所有模型版本。

为什么要做：证明每个模块是否真正有贡献。

输入是什么：zero-shot、SFT、quality-aware SFT、grounding、GeoSODA-lite 各阶段模型。

输出是什么：结果汇总表、消融表、误差分析图、典型案例图、失败案例库。

需要注意什么：不要只报总准确率，必须按质量等级、传感器、灾害类型、负样本类型和错误类型分层。

这一步如何进入下一步：结果用于论文实验部分、benchmark 完善和后续扩展。

步骤 14：模型封装与最小 Demo。

这一步做什么：封装 LoRA 权重、推理脚本、批量预测脚本、bbox 可视化脚本和简单 Web Demo。

为什么要做：保证研究结果可复现、可展示、可交付。

输入是什么：最终模型、测试图像、推理配置。

输出是什么：可上传遥感图像并输出地灾类型、证据说明、bbox、质量提示和核查建议的 demo。

需要注意什么：Demo 不要求达到专家系统水平，但必须保证输入输出格式稳定、可解释案例清楚、失败案例可追踪。

这一步如何进入下一步：形成论文/项目演示材料和后续工程化基础。

---

## 六、参数设置与对比方案

关键参数如下。

| 参数                | 含义                           | 推荐测试范围                                             | 为什么测试                         | 可能影响                            |
| ----------------- | ---------------------------- | -------------------------------------------------- | ----------------------------- | ------------------------------- |
| 模型基座              | 选择主模型与对照模型                   | Qwen3-VL-2B、Qwen3-VL-4B；可选 Qwen2.5-VL-3B、TinyRS-R1 | 2B 适合单卡训练，4B 检验参数量提升是否带来可靠性提升 | 4B 可能更稳，但显存压力和训练时间更高            |
| 图像分辨率/图像 token 上限 | 控制视觉信息量和显存                   | 448、672、896；或图像 token 上限 512、1024                  | 地灾证据常是局部细节，分辨率过低会损失裂缝和边界      | 分辨率提高可能提升 grounding，但可能 OOM     |
| LoRA rank         | 低秩适配器容量                      | 8、16、32                                            | r=8 低成本，r=16 稳妥，r=32 能力更强     | rank 太低欠拟合，太高可能过拟合              |
| 学习率               | LoRA/QLoRA 微调学习率             | 1e-5、2e-5、5e-5，必要时 1e-4                            | VLM 微调对学习率敏感                  | 过高可能输出格式崩坏或幻觉增加                 |
| 训练轮数              | SFT 或 correction SFT 的 epoch | 1、2、3                                              | 数据规模有限，训练过多可能记模板              | epoch 增加可能训练集提升、跨区泛化下降          |
| 正负样本比例            | 地灾正样本、困难负样本、普通负样本比例          | 1:1:0.5、1:1:1、1:2:1                                | 幻觉抑制依赖高质量负样本                  | 负样本少误报多，负样本多可能漏检                |
| 低质样本比例            | 低质图像和退化样本比例                  | 10%、20%、30%                                        | 训练模型质量不足时降级判断                 | 比例高可降低过度自信，但可能过度拒答              |
| InSAR 元数据         | 是否输入时间、单位、色标、LOS 方向等         | 无元数据、基础元数据、完整元数据                                   | 检验模型是否依赖物理信息而非颜色模板            | 完整元数据应降低形变证据幻觉                  |
| grounding 任务比例    | 定位样本在训练集中占比                  | 10%、20%、30%                                        | 检验定位训练是否提升证据一致性               | 比例过低定位弱，过高可能影响问答                |
| mask 表示方式         | 分割输出路线                       | bbox only、bbox+SAM、bbox+mask token                 | 单卡条件下端到端 mask 风险较高            | bbox+SAM 更稳，mask token 创新更强但难度高 |
| GeoSODA 错误库规模     | 学生错误驱动蒸馏样本数量                 | 500、1000、3000、5000                                 | 小规模先验证，大规模提升稳定性               | 规模小信号不足，规模大纠错成本高                |
| DPO/ORPO beta     | 偏好优化强度                       | 0.01、0.05、0.1、0.2                                  | 检验偏好学习稳定性                     | beta 过大可能损伤原能力                  |
| 专家审核比例            | 教师纠错答案人工复核比例                 | 5%、10%、20%                                         | 控制纠错质量和人工成本                   | 审核比例高质量好但成本高                    |

基准方法包括：zero-shot Qwen3-VL-2B/4B、prompt-only Qwen3-VL、普通 SFT、传统分类模型 ResNet/ConvNeXt、传统检测模型 YOLO、传统分割模型 U-Net/SegFormer/SAM 后处理、可选 Qwen2.5-VL-3B、TinyRS-R1。

改进方法包括：quality-aware SFT、hard-negative SFT、grounding SFT、GeoSODA-lite correction SFT、GeoSODA-lite DPO/ORPO、规程 RAG/规则检查器增强。

消融实验至少包括：去掉困难负样本、去掉质量标签、去掉拒答样本、去掉 InSAR 元数据、去掉 grounding、去掉 GeoSODA-lite、GeoSODA-lite 中只用教师 SFT 不用学生错误库、只用全图教师不用 crop/mask 证据教师。

敏感性分析包括：模型规模、图像分辨率、LoRA rank、正负样本比例、低质样本比例、GeoSODA 错误库规模、DPO/ORPO beta、专家审核比例。

不同工况分析包括：光学、SAR、InSAR、UAV；清晰图像、低质图像；公开数据、自采数据；region-holdout、sensor-holdout、event-holdout；清晰正样本、困难负样本、证据不足样本。

---

## 七、评价指标

有真实值时，使用以下定量指标。

image-level 识别指标：Accuracy、Precision、Recall、F1、Balanced Accuracy、per-class F1、多标签 mAP。

VQA 指标：VQA accuracy、exact match、yes/no accuracy、unanswerable QA accuracy、answer format accuracy。

幻觉指标：false alarm rate、hard-negative false alarm rate、overclaim rate、evidence hallucination rate、modality hallucination rate、InSAR evidence fabrication rate、unanswerable error rate。

低质影像可靠性指标：quality-stratified accuracy、overconfidence under degradation、low-quality refusal precision、low-quality refusal recall、uncertain recall。

校准指标：ECE、Brier score、confidence-error correlation。如果模型没有显式概率，可通过输出置信度等级 high/medium/low 转换为离散校准指标。

grounding 指标：bbox P@0.25、bbox P@0.5、mAP、pointing accuracy、mean IoU、mask IoU、Dice。

证据一致性指标：evidence consistency score、modality availability consistency、rule violation rate、structure output validity rate。

效率指标：单图推理时间、batch 推理时间、显存峰值、LoRA 权重大小、训练时长。

没有严格真实值时，使用工程判断指标。

第一，曲线或空间分布是否合理。例如 InSAR 形变异常是否与坡体、沟道或边坡单元空间一致。

第二，是否符合地灾判读逻辑。例如仅凭单幅光学图不能直接判断活动性；无 InSAR 图时不能输出具体形变速率。

第三，是否具有空间一致性。例如 grounding 框是否落在裸露坡体、堆积扇、沟道或崩塌源区，而不是落在道路、阴影或河面上。

第四，是否能与工程记录、专家经验、现场照片、无人机图像或历史清单互相验证。

第五，低质图像下是否能合理保守表达，而不是全部拒答或全部高置信判断。

---

## 八、建议绘制的图表

图 1：数据组成统计图。横轴为数据来源或传感器类型，纵轴为样本数量。需要画光学、SAR、InSAR、UAV、困难负样本、规程文本转 QA 数量。说明 benchmark 的多源组成是否均衡。预期看到光学样本最多，InSAR 和 UAV 样本较少但具有专业价值。

图 2：地灾类别与困难负类分布图。横轴为类别，纵轴为样本数量。需要包括滑坡、崩塌、泥石流、岩屑坡、复合灾害、裸岩、采石场、道路边坡、河滩、阴影、低相干区等。说明负样本是否足够覆盖高混淆对象。预期困难负样本占比较高。

图 3：典型原始图像与标注示例图。横轴不需要；每个子图展示原图、bbox、mask、质量标签、证据标签。需要覆盖光学、SAR、InSAR、UAV。说明 benchmark 的标注质量和任务形式。

图 4：研究框架图。横向流程为数据构建、zero-shot、SFT、quality-aware SFT、grounding、GeoSODA-lite、评估。说明整体技术路线。预期框架清楚体现 benchmark + method 两条贡献。

图 5：方法对比柱状图。横轴为模型阶段，纵轴为 F1、false alarm rate、unanswerable QA accuracy、bbox P@0.5 等。需要比较 zero-shot、SFT、quality-aware SFT、grounding、GeoSODA-lite。说明每个模块的增益。

图 6：质量退化—过度自信曲线。横轴为退化强度或质量等级，纵轴为 overconfident hallucination rate 或 false alarm rate。需要画不同模型阶段曲线。说明质量感知训练是否能降低低质图像过度自信。预期 GeoSODA-lite 或 quality-aware SFT 曲线更低、更平滑。

图 7：混淆矩阵。横轴为预测类别，纵轴为真实类别。需要包括地灾类和困难负类。说明模型最容易混淆哪些对象。预期初始模型会把裸岩、采石场、阴影误判为滑坡，增强后误判减少。

图 8：grounding/mask 可视化图。展示原图、真实 bbox/mask、模型 bbox/mask、模型解释文本。说明模型是否把证据定位到正确区域。预期 grounding SFT 后框更靠近真实滑坡体、沟道或岩屑坡范围。

图 9：参数敏感性热力图。横轴为 LoRA rank 或学习率，纵轴为图像分辨率或正负样本比例，颜色为 F1 或 hallucination-free rate。说明哪些参数最敏感。预期学习率和负样本比例对幻觉指标影响较大。

图 10：GeoSODA 错误迁移图。横轴为错误类型，纵轴为错误数量或错误率，比较蒸馏前后。错误类型包括无灾害误报、低质误答、证据幻觉、定位幻觉、规程幻觉、不可答却答。说明学生错误驱动蒸馏是否纠正高频错误。

图 11：典型局部放大图。展示模型容易误判的局部区域，例如采石场台阶、云影、低相干 InSAR 斑块、岩屑坡边界。说明错误产生原因和模型改进效果。

图 12：结果汇总表。行是模型和训练阶段，列是 Accuracy、F1、Recall、false alarm rate、hard-negative FAR、VQA accuracy、bbox P@0.5、mask IoU、推理时间、显存。说明最终模型是否达到论文或项目要求。

---

## 九、结果分析要求

结果出来后，学生必须从以下角度分析。

第一，结果是否支持原假设。分别回答 SFT 是否提升 image-level 能力，质量感知训练是否降低幻觉，grounding 是否提升证据一致性，GeoSODA-lite 是否进一步纠错。

第二，哪个方法最好，为什么。不能只看总准确率，要同时看召回率、误报率、困难负样本误报率、低质图像过度自信率和 grounding 指标。如果某方法 F1 高但误报率也高，不应简单认为最好。

第三，哪个参数最敏感，为什么。重点分析图像分辨率、LoRA rank、学习率、正负样本比例、低质样本比例和 GeoSODA 错误库规模。需要结合训练曲线、验证集指标和错误类型解释。

第四，是否存在反常结果。例如加入低质样本后总体 F1 降低；加入困难负样本后漏检率升高；4B 不如 2B；GeoSODA-lite 后回答过于保守；grounding 训练后 VQA 下降。

第五，反常结果可能由什么原因导致。可能原因包括数据分布不均、低质标签错误、instruction 模板过拟合、坐标格式不稳定、教师纠错质量差、DPO 参数过强、训练轮数过多、测试集空间泄漏或跨区域差异过大。

第六，结果是否有工程解释。例如模型降低了采石场误报，说明困难负样本有效；模型在缺少 InSAR 元数据时仍编造形变，说明规则约束不足；模型在 UAV 近景图像上表现好但在卫星图上差，说明尺度适配不足。

第七，结果是否足以支撑论文结论。论文结论不能只写“模型提升了准确率”，必须支撑“质量诱导地灾幻觉是可量化问题”“质量感知训练降低过度自信”“grounding 改善证据一致性”“学生错误驱动蒸馏能纠正小模型典型错误”。

第八，下一步还需要补什么实验。如果 GeoSODA-lite 效果不明显，需要补充教师纠错质量评估、错误库规模敏感性、普通 teacher SFT 对照。如果 grounding 不稳定，需要补充坐标格式检查、bbox 质量审核和 SAM/SAM2 后处理实验。如果低质拒答过多，需要调整低质样本比例和拒答模板。

---

## 十、最终需要提交的材料

学生最终需要提交以下材料。

1. 完整代码仓库，包括数据处理、instruction 生成、训练、推理、评估、可视化和 demo 脚本。

2. 原始数据索引，不要求提交所有原始大图，但必须提交可追溯的 metadata、数据来源说明和下载/裁剪脚本。

3. 处理后数据，包括裁剪图像、mask、bbox、metadata.jsonl、classification.csv、detection_coco.json、qwen_vl_sft.jsonl。

4. 参数设置表，包括模型名称、图像分辨率、图像 token 上限、LoRA rank、学习率、epoch、batch size、gradient accumulation、训练数据比例、随机种子。

5. 结果汇总表，包括所有模型阶段的 image-level、hallucination、grounding、效率指标。

6. 所有图表，包括数据统计图、类别分布图、框架图、方法对比图、质量退化曲线、混淆矩阵、grounding 可视化图、参数敏感性图、GeoSODA 错误迁移图、典型失败案例图。

7. student-error bank，包括原图路径、prompt、学生答案、错误类型、纠正答案、教师来源、是否专家复核。

8. 模型权重，包括 Qwen3-VL-2B LoRA 权重、Qwen3-VL-4B LoRA 权重，以及 tokenizer/processor/config 说明。

9. 推理脚本，包括单图推理、多图推理、批量推理、bbox 可视化、结果导出。

10. 评估脚本，包括分类指标、VQA 指标、幻觉指标、bbox/mask 指标、误报率、低质分层指标、推理效率统计。

11. 一页文字总结，说明做了什么、得到什么结果、哪个模块最有效、存在什么问题、下一步怎么做。

12. 主要结论列表，至少包括 3–5 条可直接用于论文或项目报告的结论。

13. 存在问题列表，明确当前数据、模型、训练和评估的不足。

14. 下一步计划，包括扩展数据、优化 GeoSODA-lite、加入更多传感器、改进 mask 输出或增加专家审核。

---

## 十一、最终总结模板

本实验针对多传感器遥感图像中崩塌、滑坡、泥石流、岩屑坡识别的可靠性问题，采用 Qwen3-VL-2B/4B 小参数视觉语言模型，构建了包含 image-level 地灾理解和 grounding-level 证据定位的 GeoHazard-HalluGround benchmark。实验首先评估了 Qwen3-VL 的 zero-shot 能力，然后依次开展普通 SFT、质量感知诚实 SFT、grounding 定位增强和 GeoSODA-lite 学生错误驱动蒸馏。结果表明，……。与 zero-shot 和普通 SFT 相比，……方法在……指标上表现更好，尤其在困难负样本误报率、低质图像过度自信率和 bbox 定位精度方面取得……提升，说明……。参数敏感性分析表明，……对结果影响最大，其中……设置较为稳妥。总体来看，该方法能够在单卡低算力条件下提升小参数 VLM 的地灾识别可靠性，使模型具备更好的图像级判断、证据说明、拒答和定位能力，但仍存在……局限，后续需要进一步通过更大规模专家标注、跨区域测试、InSAR 元数据增强和更稳定的 GeoSODA-lite 蒸馏验证。

---

## 十二、我的研究想法如下

研究主题：

基于 Qwen3-VL-2B/4B 的小参数质量感知地灾视觉语言模型研究：面向崩滑流与岩屑坡识别的多传感器 benchmark 与半 on-policy 方法验证。

我希望验证的核心结论：

在一块 RTX 4090 的低算力条件下，不训练通用遥感大模型，也可以通过高质量地灾 benchmark、质量感知诚实 SFT、困难负样本、grounding 证据约束和 GeoSODA-lite 学生错误驱动蒸馏，使 Qwen3-VL-2B/4B 在崩塌、滑坡、泥石流和岩屑坡识别中实现更高准确率、更低幻觉率、更强拒答能力和更可靠的视觉定位能力。

研究对象：

数据对象包括星载光学图像、SAR 图像、InSAR 形变速率/累计形变图、无人机巡查图像、无人机地灾现场记录图像、非地灾困难负样本和地灾行业规程文本。模型对象包括 Qwen3-VL-2B 和 Qwen3-VL-4B，必要时加入 Qwen2.5-VL-3B、TinyRS-R1 或其他小模型作对照。任务对象只包括 image-level 和 grounding-level，不设置 region-level。

已有方法或基准方法：

基准方法包括 zero-shot Qwen3-VL-2B/4B、prompt-only Qwen3-VL、普通 SFT、小型传统分类模型、传统检测/分割模型、可选 Qwen2.5-VL-3B 和 TinyRS-R1。传统视觉基线包括 ResNet/ConvNeXt 分类、YOLO 检测、U-Net/SegFormer 分割、SAM/SAM2 后处理。文献参考包括 RSHallu 的遥感幻觉 taxonomy、dual-mode checking 和 RSHalluShield 缓解思路，以及 Earth-OneVision 的 2B 多传感器多任务遥感 VLM、FGVLA、SLIS、PCMA 和统一空间 token 输出思路。 

我提出的方法：

提出 GeoHazard-HalluGround benchmark，包含 image-level 地灾理解和 grounding-level 证据定位/分割。提出质量诱导地灾幻觉评估，覆盖无灾害误报、低质影像误答、证据幻觉、定位幻觉、规程幻觉和过度自信。提出质量感知诚实 SFT，使模型学会判断图像质量、证据充分性和可回答性。提出 GeoSODA-lite，基于 student-error bank、教师纠错、规程规则和专家抽检进行轻量半 on-policy 蒸馏。

关键变量或参数：

模型基座、图像分辨率、图像 token 上限、LoRA rank、学习率、训练轮数、正负样本比例、低质样本比例、InSAR 元数据是否输入、grounding 是否加入、mask 表示方式、GeoSODA 错误库规模、DPO/ORPO beta、教师纠错策略、专家审核比例。

希望得到的图表：

数据组成统计图、地灾类别与困难负类分布图、典型原始图像与标注示例图、研究框架图、方法对比柱状图、质量退化—过度自信曲线、混淆矩阵、grounding/mask 可视化图、参数敏感性热力图、GeoSODA 错误迁移图、误差统计图、典型局部放大图、最终结果汇总表。

希望学生最终完成的成果：

完整代码、数据索引与标注文件、instruction 模板、训练配置、zero-shot 与各阶段微调结果、student-error bank、定量结果表、所有图表、LoRA 权重、推理脚本、评估脚本、demo、一页总结、主要结论、存在问题和下一步计划。

特别注意事项：

第一，不要做通用遥感大模型，不要扩大到全场景遥感任务。第二，不设置 region-level，只做 image-level 和 grounding-level。第三，InSAR 图像必须携带单位、时间段、LOS 方向、升降轨、色标和相干性说明，不能只当 RGB 图输入。第四，低质图像不能简单删除，要作为质量诱导幻觉评估的重要样本。第五，困难负样本必须主动构建，尤其是裸岩、采石场、道路边坡、河滩、阴影、低相干区。第六，GeoSODA-lite 必须先有 SFT cold-start，再做学生错误库，不要一开始直接做 DPO/ORPO。第七，所有测试必须防止空间泄漏，同一滑坡体、同一航线、同一 InSAR 图相邻 patch 不能同时进入训练和测试。第八，最终结论不能只报告准确率，必须同时报告误报率、过度自信率、拒答正确率、grounding 精度和典型失败案例。
