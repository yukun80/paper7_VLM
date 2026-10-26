# 6. 五个核心模块：为什么这样设计，实际上怎样工作

[返回本节目录](README.md) · [文档总目录](../../README.md)

<a id="guide-modules"></a>

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

### 6.1 OA-AuxSeg：专业“定位员”

**解决的问题。** 普通 VLM 擅长用语言概括图像，但像素级边界是另一种任务。OA-AuxSeg
专门回答候选区域在哪里，保留独立的分割模型比让语言模型顺便输出 mask 更容易冻结数学、
验证 checkpoint 和单独评价空间能力。

**实际结构。** 当前模型使用 ConvNeXt-Small 光学主干、多阶段辅助模态融合和
SegFormer-style 四尺度 decoder。它输出 mask logits、mask probability、no-target score、
modality weights、候选连通区域和区域特征。详见
[model.py](<../../../oa_groundrag/segmentation/model.py>)、
[fusion.py](<../../../oa_groundrag/segmentation/fusion.py>) 和
[regions.py](<../../../oa_groundrag/segmentation/regions.py>)。

**为什么光学锚定。** 光学影像定义空间边界基准；DEM、InSAR velocity、slope 等是可缺失
的辅助证据。registry 明确记录某个样本有哪些模态，模型通过 availability 与 null auxiliary
处理缺失，而不是从文件名猜通道含义。当前受支持辅助 registry 的实际边界应以
[contracts.py](<../../../oa_groundrag/segmentation/contracts.py>) 和实时状态为准，不能把未来计划
中的 SAR 说成已经接入。

**no-target 为什么重要。** 空 mask 可能意味着当前没有候选，而不是系统坏了。后续合同要求
no-target 时不得虚构 bbox、crop、区域数组或目标存在性。严格失败比“为了输出好看而自动补一个
区域”更适合科研审计。

### 6.2 Shared MLLM：共享“观察员”

项目使用本地 Qwen3-VL-2B-Instruct 作为共享语义主体。视觉 encoder/merger 保持冻结，
只在 Attention 的 <code>q_proj/k_proj/v_proj/o_proj</code> 上训练 rank-8 LoRA；配置见
[RS-General LoRA](<../../../configs/vlm/rs_general/rs_generaldesc_lora_qwen3vl_2b.yaml>)。

这里的“Shared”有两个含义：

1. 普通遥感描述和 mask-grounded 区域观察使用同一个 MLLM 主体。
2. 不为每个 runtime task 串接不同 Adapter；最终 Grounded Adapter 已从 RS-General
   warm start，并通过 replay 维持通用能力。

Grounded curriculum 的 region/replay 比例、warm start 和生成限制见
[Grounded train-only 配置](<../../../configs/vlm/grounded/train_v2.yaml>)。
运行 loader 只读取最终 trainable LoRA state，见
[grounded_runtime.py](<../../../oa_groundrag/vlm/grounded_runtime.py>)。

### 6.3 Grounding：把“哪里”变成可审计输入

Grounding 不是再训练一个分割器，而是一份明确的证据接口：

- <code>optical_full</code>：完整、无标记的光学图，保留全局场景。
- <code>binary_mask</code>：严格二值位置提示；白色不是地物颜色。
- <code>context_crop</code>：直接从原始 RGB 裁剪的无标记局部图；crop 边缘不是目标边界。
- Programmatic Facts：bbox、质心、面积比例、组件数、紧致度、伸长率等确定性事实。
- Limitations：对齐、覆盖率、单位、符号约定和当前证据缺口。

为什么不是只给 crop？因为 crop 会丢失全局环境，还容易让模型把裁剪边缘误认为目标边界。
为什么不是只给 mask overlay？因为彩色覆盖会污染真实颜色。正式输入使用原图、二值 mask 和
clean crop；overlay 只允许作为明确标记的审计消融。

程序事实由 [evidence.py](<../../../oa_groundrag/grounding/evidence.py>) 计算，消息合同由
[messages.py](<../../../oa_groundrag/grounding/messages.py>) 构造。模型只能观察视觉内容，不能
重新生成或修改几何。

还要区分空间模型的多源输入与 Shared MLLM 的多源证据：当前 Unified P0 合同中的
<code>auxiliary_views</code> 只允许 region task 携带对齐、覆盖率、单位和符号约定等元数据，
不能据此宣称 Shared MLLM 已实际消费完整 SAR/InSAR/DEM grounded 图像证据。多源 Grounded
Evidence 的实时实施状态与下一任务只看 [当前进度](<../../status/README.md>)。

### 6.4 Evidence-Constrained Text RAG：知识“查证员”

Text Evidence Bank 保存的不是任意网页，而是登记来源的专业 PDF 证据单元。每个单元带有
来源、页码、章节、知识类型、适用模态和可重算身份。构建逻辑见
[bank.py](<../../../oa_groundrag/retrieval/bank.py>)，来源登记见
[sources_v1.yaml](<../../../configs/retrieval/sources_v1.yaml>)。

在线检索的真实组合是：

~~~text
解释查询 + 混淆/限制查询
        ↓
每条查询：FTS5 BM25 + BGE-M3 dense
        ↓
RRF 融合
        ↓
interpretation / confounder / limitation 平衡 packet
        ↓
text-only Pass-2
~~~

当前规模下使用 SQLite FTS5 和本地 NumPy dense matrix 足够透明、可复现，也没有证据表明
必须引入独立向量数据库或 learned reranker。配置中的召回深度、RRF 常数和配额见
[runtime_v1.yaml](<../../../configs/retrieval/runtime_v1.yaml>)，实现见
[search.py](<../../../oa_groundrag/retrieval/search.py>)。

Pass-2 不是“搜到什么就自由发挥”。解析器检查严格 JSON、Evidence ID、证据类型、禁止重写
几何、禁止把候选确认成滑坡、禁止把文献知识说成当前图像观察。见
[pass2.py](<../../../oa_groundrag/retrieval/pass2.py>)。

### 6.5 Unified Runtime：确定性“调度员”

Router 只把显式 <code>UnifiedTask</code> 映射为 <code>ExecutionPlan</code>。它不调用 LLM
猜意图，也不做 ReAct、反思或自主规划。Provider 失败会返回稳定 reason code，不会偷偷换用
另一个能力。

单卡约 24 GB 的关键不是把所有东西永久放进显存，而是：

1. 按任务惰性加载所需 provider。
2. 空间推理完成后按配置释放空间模型。
3. 第一遍视觉观察完成后释放 Shared MLLM，再运行检索。
4. 第二遍需要生成时重新获得 Shared MLLM，并在结束后释放。
5. 记录 provider 调用、释放、身份和显存 trace。

配置见 [inference_v2.yaml](<../../../configs/runtime/inference_v2.yaml>)，编排见
[inference.py](<../../../oa_groundrag/runtime/inference.py>)。

候选 ID 缺失或不存在时，runtime 可以按明确合同退到 OA-AuxSeg global mask，并在
<code>RegionSelection</code> 和 limitation 中记录原因。这是**特定且可见的候选选择规则**，
不是“所有错误都自动降级”。模型输出非法、身份不匹配、test/sealed 路径或 provider 失败仍然
fail closed。
