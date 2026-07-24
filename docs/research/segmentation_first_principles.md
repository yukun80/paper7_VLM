# 滑坡分割第一性原理与候选边界

> 状态：设计冻结草案，等待 P1/P2 owner-run prerequisite
> 证据性质：由任务合同、HDF5 数据语义和原始文献推导；没有模型实验结果
> 详细接口权威：[`../REFACTOR_TASK_SPEC.md`](../REFACTOR_TASK_SPEC.md)

## 1. 预测目标需要哪些信息？

预测目标是每个 `target_valid` 像素是否属于滑坡，而不是场景描述、对象检测或灾害前后变化。
模型所需信息按必要性分三层：

1. **监督与几何必要信息**
   - 二值 target 和 target validity；
   - 输入 channel 与 target 的空间配准；
   - resize、crop、pad 对 values、pixel validity 和 target validity 的同构变换。
2. **观测解释信息**
   - channel identity 与 modality family；
   - channel 是否存在；
   - 每个像素是否可观测；
   - 只从 train valid pixels 得到的 source/channel normalization。
3. **条件性物理信息**
   - 仅在来源明确绑定时使用 wavelength、GSD、unit、sign；
   - unknown 本身是合法状态，不能用估计值伪装 known。

地形形态、光谱响应、雷达/干涉响应可能互补，但“可能互补”不等于每个样本必须包含所有模态。
模型必须在任意合法 subset 上定义良好。

## 2. 哪些差异是物理语义，哪些只是存储格式？

### 2.1 物理或观测语义

- red、NIR、DEM、slope、SAR、InSAR 等 channel identity；
- 光学、地形、雷达等 modality family；
- 已验证 wavelength、GSD、unit 和 sign；
- channel 缺失、pixel 无效、饱和或 source 明确提供的质量状态；
- 不同 source 的传感器与预处理 provenance。

### 2.2 存储与表示

- HDF5 dataset key、CHW/HWC、dtype、byte order；
- channel 在数组中的枚举位置；
- 文件名、机器绝对路径和当前挂载点；
- batch 内为对齐 C/H/W 添加的 padding；
- 在不改变物理覆盖定义时的内存布局和无损序列化。

存储差异必须由 data contract 消除或显式记录，不能让模型学习。例如 channel index 只用于读取，
模型身份来自 `channel_key`，绝对路径不能进入 canonical identity。

## 3. 哪些元数据可信，哪些必须是 unknown？

### 3.1 可直接信任

- 从 HDF5 key、shape、dtype 和显式 channel schema 重放出的结构；
- source 明确声明且 P1 可重放的 native split；
- 文件字节 hash；
- source 提供并能独立重开的 valid/channel-valid/pixel-valid；
- 明确的 channel identity，例如实际 schema 中注册的 red、NIR、DEM。

### 3.2 只能降低可信度、不能排除训练

- 缺失或不完整的 location、group、canonical index、duplicate component；
- source-declared split 尚无 verified group isolation；
- LandslideBench_agent 的对话文字；
- 空间邻接风险但没有可靠 group graph。

### 3.3 必须标为 unknown

- 未由 source 文件或正式 source schema 数值绑定的中心波长和 GSD；
- 仅凭传感器名、波段昵称或“约 10 m”推断的数值；
- 不能重放的物理 unit、sign、offset；
- 用 zero-fill 推断的 missing 状态；
- 没有可靠 location/group 证据时的地理独立性。

当前 P1 prerequisite 之前，没有任何 source cohort 获准启用 wavelength/GSD 数值条件。

## 4. 哪些运算必须对 channel/modality 顺序不敏感？

输入枚举顺序不是物理语义，因此以下操作必须是 permutation-equivariant 或
permutation-invariant：

- 每个标量 channel 的共享 stem；
- channel metadata 编码；
- `channel_valid × pixel_valid` 掩码；
- 每级金字塔的 sum-normalized masked mean；
- modality dropout 对 channel identity 的选择；
- batch collator 的 padding 与 unpadding；
- 最终 fused pyramid 和 logits。

排列必须连同 values、descriptor 和 validity 一起变换。冻结的 eval-mode permutation gate 要求
重排前后 logits 在规定数值容差内等价。任何依赖数组位置的 learned source slot、五槽拼接或
位置特定 stem 都违反此原则。

## 5. missing channel 与真实零观测如何区分？

数值张量不能单独表达缺失：

- **真实零**：`channel_valid=true` 且对应 `pixel_valid=true`，零值进入 stem 和聚合；
- **整 channel 缺失**：`channel_valid=false`，无论存储占位值是什么都不进入聚合；
- **局部无效**：`channel_valid=true`，但局部 `pixel_valid=false`，这些位置权重为零；
- **batch padding**：同时令 channel-valid 和 pixel-valid 为 false；
- **target invalid**：只由 `target_valid=false` 表达，loss/metric 精确为零贡献。

聚合分母是当前位置有效 channel 权重之和，并显式处理分母为零。训练 dropout 只能改变输入
validity mask，不能改 target 或把缺失写成一个“特殊零值”。

## 6. 多尺度究竟是什么？

必须区分三个量：

1. **数组/网络尺度**：同一注册图像在 encoder pyramid 中的下采样层级；
2. **增强尺度**：resize/crop 后目标在像素坐标中的大小变化；
3. **物理尺度**：每像素覆盖多少米以及 patch 覆盖多少平方米。

K0/K1 的 CNN pyramid 只解决前两项。只有 `gsd_known=true` 且 transform record 能重放物理
覆盖时，才允许检验第三项。把不同尺寸数组统一 resize 后，不能声称模型已经获得物理
scale invariance；把 source 报告的近似分辨率补成精确 GSD 同样不允许。

## 7. language prompt 是否真正增加信息？

若每个样本的目标都是“分割滑坡”，措辞变化不改变条件分布：

```text
P(mask | image, equivalent landslide prompt) = P(mask | image)
```

此时语言最多是常量偏置，不能凭参数量成为科学贡献。K2 的资格来自 prompt 是否改变合法目标，
而非来自 language model 是否强大。必须比较：

- correct prompt；
- null prompt；
- random prompt；
- semantic-equivalent prompt；
- wrong-object prompt。

wrong-object 若没有对应标注，只能测 sensitivity，不能把空 mask 当真值。若 P1 的 prompt audit
确认所有文本语义等价，记录 `prompt_information_status: redundant`；K2 只做有界工程覆盖，不进入
正式三 seed 选择。

## 8. 每增加一个模块，哪个可证伪假设支持它？

| 模块 | 允许它存在的假设 | 最小证伪方式 | 失败动作 |
| --- | --- | --- | --- |
| channel metadata embedding | H1 | 去掉 identity 后跨 channel subset 显著退化 | 若无收益，缩减 metadata 参数 |
| masked set aggregation | H1 | permutation、missing、subset gate | 若失败，先审计 validity，再考虑一层 attention |
| source-balanced sampler | H2 | 与样本比例采样在同预算下比较 source collapse | 若无跨 source 收益，采用更简单采样 |
| channel/modality dropout | H2 | 完整输入与缺失输入分层比较 | 若只损害完整输入且不改善缺失鲁棒性，降低或删除 |
| GSD conditioning | H3 | 仅在 verified-GSD cohort 比较 known/null/randomized GSD | 无 eligible cohort 时不实现 |
| FiLM prompt conditioning | H4 | correct/null/random/equivalent/wrong prompt | prompt 冗余或无稳定增益则拒绝 K2 |
| 单层 set/cross-modal attention | H5 | masked mean 的冻结失败先复现，再同预算比较 | 未达到替代门限则删除 |
| non-RGB channel-set path | H6 | 与注册 RGB K0 在异构模态 cohort 比较 | 若无益，回查 normalization/validity，不转向 RGB VLM |

## 9. 冻结假设

### H1：channel set 足够性

显式 channel identity、modality metadata 与 validity-aware masked set aggregation 足以统一合法的
可变 channel 组合。它被公平的 channel subset、missing channel 和 permutation 实验证伪。

### H2：采样与 dropout

source-balanced sampling 和 channel/modality dropout 能降低 dominant-source collapse，同时不越过
冻结的完整输入退化上限。只改善 unified 而牺牲少数 source 不算支持。

### H3：物理尺度条件

可信 GSD conditioning 只在真实、可重放的物理 scale 存在时有益。unknown cohort 不能用于支持
或反驳该机制；没有 eligible cohort 时结论是 unavailable，不是 negative。

### H4：prompt 信息量

轻量 prompt conditioning 只有在 prompt 改变目标信息时才产生跨 seed、跨 source 的增益。
语义等价文本之间的大幅输出差异反而是失败。

### H5：融合深度

一层或零层 learned fusion 已足够；更深融合的增益不足以抵消参数、FLOPs、显存和复现成本。
增加 attention 前必须先有 masked mean 的已冻结失败证据。

### H6：非 RGB 证据路径

非 RGB 模态通过 typed channel-set encoder 比渲染/塞入 RGB VLM visual tower 更可靠。比较必须使用
同一 population、validity 和预算，并单列 L4S/MM 等真正含非 RGB 证据的 cohort。

正式 seeds 冻结为 `3407`、`3408`、`3409`。在 P1/P2 live acceptance 之前，H1-H6 都只是 proposed，
不能产生 accepted/rejected 结论。

## 10. 候选边界

### K0：P2 registered-view direct-dense sanity

- 只用显式注册 RGB view 和 validity；
- 小于等于 5M trainable parameters；
- 最小 convolutional encoder-decoder；
- 用于证明 image/mask/transform/validity 可学习；
- 永不具备 P3/P4 内核选择资格。

### K1：Channel-Set Dense Kernel

- 共享 scalar-channel stem；
- identity/modality/known-unknown metadata embedding；
- 每级 pyramid 使用 masked mean；
- shared hierarchical CNN + simple FPN/U-Net decoder；
- 不含 source embedding、attention、dynamic spectral weights、语言或固定 slots；
- 小于等于 20M trainable parameters，是 P3 默认 Pareto 基线。

### K2：K1 + lightweight prompt conditioning

- 只能在 K1 工程 prerequisite 通过后实现；
- 冻结本地 language interface，visual tower 不调用；
- pooled query 只经一个 FiLM block 调制 bottleneck/decoder；
- 新增 trainable parameters 小于等于 2M；
- prompt audit 冗余时不能进入正式选择。

### K3：复杂 VLM pixel decoder

- 当前明确不实现；
- LMM segmentation token、SAM/proposal、bbox、autoregressive mask、第二视觉 encoder 均在边界外；
- 只有 K2 有 eligible positive evidence 且 owner 接受新 ADR 后才能重新讨论；
- 本 Goal 不授权该 ADR。

## 11. 决策顺序与失败归因

```text
P1 source/validity/split replay
  -> P2 K0 learnability
  -> K1 H1/H6
  -> H2 sampler/dropout
  -> H3 only if verified GSD cohort exists
  -> K2 H4 only after K1 and prompt eligibility
  -> H5 attention only after reproducible masked-mean failure
  -> Pareto-simplest unique kernel
```

失败优先归因顺序是 source bytes与 dataset key、label、registered transform、normalization、
channel/pixel/target validity、sampler/population、模型。不得用更大的 VLM 掩盖前六类问题。
