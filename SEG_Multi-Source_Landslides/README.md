# Multi-Source Qwen-PSALM-Seg

当前实现采用 **SANE -> QMEF -> PMRD**：

- **SANE**：按 benchmark-v2 的 family/sensor/product/band/GSD/units 编码原始物理模态，并可接入 Qwen-ViT cache v3 的 layers 5/11/17/23 空间特征。
- **QMEF**：使用 valid-weighted reliability、coverage ratio、null evidence，并在 scale × modality × sampling point 上执行一次联合 query-conditioned attention。
- **PMRD**：由 Qwen mask-position hidden states 初始化 proposal queries；组件数不超过 queries 时使用 Hungarian，超过时切换为覆盖全部组件的 coarse/refined coverage-set 监督，再进行 query-specific detail refinement 和 relevance-gated union。

`qwen_psalm_full` 在线加载 NF4 4-bit Qwen language decoder，仅在最后 4 个语言 block 的
q/k/v/o projection 上训练 LoRA；新增 mask/evidence 参数保留 FP32 master weights。视觉塔只用于离线 cache，
训练时将每个物理 view description 与对应视觉 token 交错送入 language decoder，并使用 Qwen 原生
vision-start/vision-end embedding 包围每个压缩 view。模型不生成 bbox，
也不做灾前灾后变化检测。
24GB 单卡性能路线默认关闭 Qwen activation checkpoint，用激活显存换取吞吐；显存不足时才
显式选择 `reentrant`，运行时不会自动回退。训练 batch 按空间尺寸与 Qwen 序列负载分桶，
consistency teacher 只计算实际 dropped-modality 样本。
Qwen forward 始终经过 PEFT wrapper；默认先用 450 steps 训练 controller prompts、SANE、QMEF
和 PMRD，再以 dense learning rate 的 0.2 倍启用 QLoRA。`qwen_mask_query_frozen` 提供不注入
LoRA、只训练软提示与分割模块的科学对照。NF4 Qwen 与 FP32 LoRA/controller projection
显式隔离于 dense segmentation autocast，避免外层混合精度改变 adapter 计算图。
视觉 evidence 消融支持 `shuffled`、`text-only`、`image-text-delta` 和
`remove:<family>`；它们不改变 SANE 的预训练空间特征，避免把 Qwen evidence 效果与
dense backbone 变化混在一起。

checkpoint 协议为 `qpsalm_sane_qmef_pmrd_v5`，绑定在线 Qwen mask-query 序列结构和 QLoRA 阶段配置；`resume_training_stage` 明确记录恢复后下一步所处阶段。
vision cache manifest 绑定 train/val/test instruction index 指纹，重建 benchmark 后不会静默复用旧 cache。
`qpsalm-integration-check` 是正式实验前的硬门槛：raw 三任务检查保持不变；Qwen 侧只用一个
同空间/负载/任务组的代表性 batch 验证 LoRA projection 确实执行、A/B 梯度、参数更新、
teacher consistency 和显存。深度诊断由 `--qwen-check diagnostic` 显式启用，依次检查
controller-only、student-only segmentation 和 full/dropped consistency，不增加普通启动开销。

评估严格区分 verifier 可部署选择与 GT-only 诊断：`selected_proposal` 来自 relevance
argmax，`oracle_matched_proposal` 来自统一 component assignment，仅用于分析 proposal
生成上限。原尺寸指标遇到损坏的 resize transform 会直接失败，不会静默退回 canvas 指标。

raw 与 pretrained-SANE preset 使用 `64/128/256/384` 尺寸桶；24GB 单卡主路线
`qwen_psalm_full` 使用 `64/128/256` 尺寸桶。算法 preset 不绑定硬件参数；正式
运行参数直接由 small/full YAML 定义，当前24GB配置为BF16、`batch_size=4`、
`grad_accum_steps=1`、`query_chunk_size=16`和disabled Qwen checkpoint。

主 preset：

```text
raw_sane_baseline
raw_sane_qmef
raw_sane_qmef_pmrd
pretrained_sane_qmef_pmrd
qwen_psalm_full
qwen_mask_query_frozen
```

配置：

```text
configs/qpsalm_v2_small.yaml
configs/qpsalm_v2_full.yaml
configs/qpsalm_v2_smoke.yaml
```

完整构建、训练、cache、val/test、消融和分析命令统一维护在仓库根目录
[README.md](../README.md)。

交互推理使用 `qpsalm-demo`，PPT 分层精选使用 `qpsalm-curate-gallery`。两者共享
`InferenceSession`，只支持 benchmark val/test 样本，并复用 parent-level Qwen vision cache v3。

## Segmentation-Grounded Description 扩展

描述主线在保留原分割 `forward(ModalityBatch)` 的同时增加：

```text
MultisourceBackboneState
    -> RegionPrompt
    -> MGRR multi-granularity token sequence
    -> Qwen desc_adapter
    -> raw JSON + summary
```

- `MultisourceBackboneState` 是任务无关的 SANE/视觉 cache 状态；分割和描述可复用一次编码。
- `SegmentationState` 保存 QMEF、PMRD proposal 和 relevance，只服务分割任务。
- `RegionEvidenceState` 保存 exact-mask、context ring、component replay、geometry、逐模态证据及
  变长 `region_sequence_tokens`，不把 segmentation-conditioned fused feature 当作通用 caption 表征。
- `qpsalm_description_vision_cache_v1` 独立于 segmentation Vision Cache v3，按 parent 缓存，
  禁止包含 instruction、region mask、答案或 segmentation state。
- Qwen 基座共享，`default` adapter 用于分割，`desc_adapter` 用于描述；每个 batch 只激活一个。
- description causal sequence 使用
  `qpsalm_description_causal_v5_stage_separated_schema_ordered`，checkpoint 使用
  `qpsalm_segdesc_v1`；跨 D-stage 使用 `--initialize-from`，同 stage 中断续训才使用 `--resume`。
- D0/D1 是全图 caption，D2 首次启用区域 replay，D3a 是 auto Bridge；D3b 及其后的专家路径
  必须等待冻结的真实 Bridge。
- M6 分开报告 GT-mask、fixed-prediction 和 end-to-end；M7 使用独立 task DataLoader 和
  exact-population segmentation retention，不允许用 monitor subset 代替正式门禁。

当前 M1.1、M3、D-1 和 D0 已通过 Small 工程门禁；M2 仍是
`awaiting_expert_review`，下一训练阶段是 D1。工程通过不等于科学验收。完整命令与当前状态只在
仓库根目录 [README.md](../README.md) 维护；分割算法见 [ALGORITHM.md](ALGORITHM.md)，SegDesc
研究协议见 [docs/benchmark_GAR.md](../docs/benchmark_GAR.md)。
