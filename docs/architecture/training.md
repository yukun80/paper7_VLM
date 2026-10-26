# 6. 两段训练 curriculum

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

**设计与现状差异：** 原设计中的 resume 步骤属于冻结记录；当前训练入口只允许从 step 0 写入全新输出目录，不支持恢复训练。4B 后续资源检查仍未启动，不能直接执行旧步骤；继续前须重新确认方案。见[使用说明](../usage/commands.md)与[下一步](../status/next.md)。

### 6.1 RS-General Adaptation

- 数据：同一冻结 RS-GeneralDesc Benchmark，仅 train/val；
- 初始化：Qwen3.5-4B 固定基模 revision；
- template：non-thinking；
- batch：micro batch 1，gradient accumulation 16；
- optimizer/scheduler/LR：沿用冻结 2B 配方，LR `2e-4`；
- 预算：`max_steps=1000`；
- 发布：同模型 val monitor 选择 best，输出 checkpoint manifest、Adapter、best pointer、
  配置/数据/模型/processor/ledger/实现身份。

先通过 bounded smoke 才能进入正式训练。输出根必须全新且拒绝覆盖。

### 6.2 Mask-Grounded Region Adaptation

- 初始化：只允许同一 Qwen3.5 家族的 RS-General best Adapter；
- 输入：原始 full optical、严格 binary mask、clean context crop；
- 数据：冻结 6,974 条 compact supervision，90% Region / 10% RS-General replay；
- optimizer/scheduler/RNG/sampler：fresh state；
- template：non-thinking；
- LR：`5e-5`；
- 预算：`max_steps=1000`；
- 输出：独立 checkpoint、Adapter、best pointer 与完整身份 manifest。

配置使用严格 `rs_vlm.mask_grounded_train_config.v4`，显式绑定监督位置 loss、FP32 CE、
allocator、microbatch cache、2 GiB 余量门和有界资源遥测。旧 M7 v3 配置与解析分支在新
资源门通过后删除，不提供 alias。通过 1→20→100-step 资源门后，正式训练必须从冻结
M5 step-1000 best fresh warm-start 到独立新根，不得从 smoke-100 恢复；任何跨家族
warm-start 必须拒绝。正式 M7 完成与 retention 只构成工程完成，`formal_acceptance` 与
`scientific_acceptance` 均保持 false，不新增 Gate C 或 retention 科学阈值。
