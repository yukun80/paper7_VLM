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
视觉 evidence 消融支持 `shuffled`、`text-only`、`image-text-delta` 和
`remove:<family>`；它们不改变 SANE 的预训练空间特征，避免把 Qwen evidence 效果与
dense backbone 变化混在一起。

checkpoint 协议为 `qpsalm_sane_qmef_pmrd_v3`，绑定在线 Qwen mask-query 序列结构。
vision cache manifest 绑定 train/val/test instruction index 指纹，重建 benchmark 后不会静默复用旧 cache。
`qpsalm-integration-check` 是正式实验前的硬门槛：真实三任务 raw optimizer step 与真实
多模态 Qwen QLoRA BF16 step 均通过后，才进入 small-v2 三 seed 对比。

评估严格区分 verifier 可部署选择与 GT-only 诊断：`selected_proposal` 来自 relevance
argmax，`oracle_matched_proposal` 来自统一 component assignment，仅用于分析 proposal
生成上限。原尺寸指标遇到损坏的 resize transform 会直接失败，不会静默退回 canvas 指标。

raw 与 pretrained-SANE preset 使用 `64/128/256/384` 尺寸桶；24GB 单卡主路线
`qwen_psalm_full` 使用 `64/128/256` 尺寸桶。算法 preset 不绑定硬件参数；正式
运行参数直接由 small/full YAML 定义，当前24GB配置为BF16、`batch_size=6`、
`grad_accum_steps=1`、`query_chunk_size=16`和disabled Qwen checkpoint。

主 preset：

```text
raw_sane_baseline
raw_sane_qmef
raw_sane_qmef_pmrd
pretrained_sane_qmef_pmrd
qwen_psalm_full
```

配置：

```text
configs/qpsalm_v2_small.yaml
configs/qpsalm_v2_full.yaml
configs/qpsalm_v2_smoke.yaml
```

完整构建、训练、cache、val/test、消融和分析命令统一维护在仓库根目录
[README.md](../README.md)。
