# 分割研究 Findings

> 当前状态：`P1_owner_gate`
> 结果声明：本文没有记录任何模型实验、builder、validator、overfit、GPU、显存或三 seed 结果。
> 允许证据：当前 authority 中的 HDF5 现场事实、原论文机制和由两者得到的可证伪推论。

## 1. 当前可以成立的结论

1. 五个非空 HDF5 source 都可进入训练；location、group、canonical index 或 duplicate component
   不完整影响的是 evaluation assurance，而不是 image/mask 的可学习性。
2. 当前 strict cohort 为零。因此后续任何 exploratory 指标都不能改写成严格泛化结论。
3. source tensor 的 channel 数量和物理意义不同；固定 C 拼接、五个 source slots 或数组位置身份
   都会把存储偶然性写进模型。
4. `channel_valid`、`pixel_valid` 和 `target_valid` 是三种不同语义。真实零值不能作为 missing
   sentinel，invalid target 不能进入 loss/metric。
5. 现阶段没有可用于模型条件的可靠逐样本 wavelength/GSD cohort。DOFA、AnySat 和 Scale-MAE
   提供了机制参考，但不能授权补造物理元数据。
6. 五源任务的主要监督是同一个 binary landslide mask。语言是否提供额外目标信息尚未由 P1
   prompt audit 证明，因此 K2 暂无正式选择资格。

以上是合同与文献推论，不是模型效果。

## 2. 文献支持的最小机制

### 2.1 保留

- **共享 backbone**：OFA-Net 表明跨 EO 输入共享主干是合理方向；本项目先用更小的 shared CNN。
- **typed channel metadata**：DOFA/S2MAE 说明光谱身份不能退化为匿名数组位置；本项目用显式
  identity/modality/known-unknown embedding。
- **modality masking**：MultiMAE 与 OmniSat 支持在训练时暴露不同可用模态组合；本项目采用
  channel/modality dropout，但必须由 validity 合同约束。
- **物理 scale 与 resize 分离**：AnySat 与 Scale-MAE 都依赖真实尺度信息；本项目先记录
  `gsd_known`，没有可靠数值时保持 unknown。
- **窄语言条件**：CLIPSeg、CRIS、DenseCLIP 证明文本 embedding 可以调制 dense prediction；
  本项目将上限缩到 pooled query + 单 FiLM，并先检查 prompt 信息量。
- **显式 no-target gate**：GSVA 显示 absent target 需要专门处理；本项目已有更直接的空 mask
  supervision 和 false-positive metric，不需要拒绝 token。

### 2.2 暂时拒绝的复杂度

- DOFA wavelength-conditioned dynamic hypernetwork：缺少可靠波长 cohort。
- S2MAE 3D spectral cube：异构 DEM/slope/InSAR 不构成连续光谱轴。
- MultiMAE/OFA/AnySat/OmniSat 的完整 foundation pretraining：当前没有证据支持其相对 K1 的
  额外预算。
- set/cross-modal attention：masked mean 尚未有冻结失败。
- CLIP visual tower：只能接收 RGB，不能作为五源完整 spatial encoder。
- pixel-text contrastive objective：当前没有合法像素-文本正负监督。
- LISA、PixelLM、GSVA、GLaMM 完整路线：其 reasoning/referring/grounded-conversation 任务、
  参数和训练依赖均不匹配当前 binary segmentation。
- bbox-first、SAM/proposal cascade、oracle box、自回归 mask 和运行时候选切换：authority 明确排除。

“拒绝”在此表示当前路线无资格实现，不表示论文机制在其原任务无效。

## 3. 候选的当前证据状态

| 候选 | 当前状态 | 已有证据 | 缺少的真实证据 |
| --- | --- | --- | --- |
| K0 | blocked by P1 | authority 定义与门限 | Benchmark v4、owner overfit/reload |
| K1 | proposed | 文献和第一性原理支持最简 channel-set 路线 | P1/P2 acceptance、H1/H6 三 seed 结果 |
| K2 | proposed, ineligible now | 文献支持窄语言调制可行 | K1 prerequisite、prompt 信息量、H4 结果 |
| K3 | excluded | 仅作为复杂度上界完成文献审计 | 当前 Goal 不授权 |

没有候选被接受、拒绝或冻结；也没有 class/config/report/checkpoint hash 可以登记。

## 4. H1-H6 当前状态

- **H1**：proposed；需要 Benchmark v4 的可变 channel batch、permutation 和 subset protocol。
- **H2**：proposed；依赖 K1 和 source-balanced sampler，不能用 source 数量差异直接声称 collapse。
- **H3**：proposed but no eligible cohort；可靠 GSD 出现前不实现 conditioner。
- **H4**：proposed but blocked by K1/prompt audit；语义等价 prompt 不能当多任务指令。
- **H5**：proposed；masked mean 没有失败证据，因此 attention 目前被拒绝。
- **H6**：proposed；需要 K0 RGB 与 K1 full-channel 的同 population 比较，尤其是 L4S/MM。

## 5. 需要由真实 artifact 回答的问题

### P1

- 每个 source 的实际 channel descriptor、validity key 和 registered RGB eligibility 是否能完整重放？
- normalization 是否只使用 canonical train valid pixels？
- native split、train-only 和 evaluation eligibility 是否无漂移？
- prompt 字段是否存在，若存在是否真的改变目标语义？

### P2

- 五源及 unified 的 1/4/8/32-parent 能否达到冻结门限？
- no-target false-positive、invalid-target 零贡献和 FP32 reload 是否通过？
- 若 K0 失败，问题位于 source、label、transform、normalization 还是 validity？

### P3

- masked mean 是否在 channel permutation 下数值等价？
- 非 RGB 通道是否在跨 source、跨 seed 结果中带来稳定增益？
- source-balanced sampling/dropout 是否降低 dominant-source collapse，而非只改变总体均值？
- 是否存在可合法检验 GSD 或 prompt conditioning 的 cohort？

## 6. 下一动作

当前最高优先级不是实现 K1。P1 Benchmark v4 的实现与 focused checks 已完成，但 builder
和 independent validator 尚未由项目负责人运行。下一动作是执行 README 中的 P1 owner
commands，并重开真实 artifact 核对 `errors == []`、source/manifest/normalization hash、native
split、311 个 location conflicts 与 strict population 0。P1 live acceptance 后再实现 P2；
P2 live acceptance 前，H1-H6 都保持 proposed，不得创建形式上像结果的空 report、hash 或
checkpoint。
