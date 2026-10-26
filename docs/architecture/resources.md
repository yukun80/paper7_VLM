# 5. Qwen3.5 LoRA 与单卡资源合同

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

**设计与现状差异：** 原设计中的 resume 步骤属于冻结记录；当前训练入口只允许从 step 0 写入全新输出目录，不支持恢复训练。4B 后续资源检查仍未启动，不能直接执行旧步骤；继续前须重新确认方案。见[使用说明](../usage/commands.md)与[下一步](../status/next.md)。

硬件目标是单张约 24 GB GPU，基础模型 BF16、SDPA、gradient checkpointing，视觉塔与
merger 严格冻结。首版 LoRA 只覆盖 Qwen3.5 8 个 full-attention 层中的：

```text
q_proj, k_proj, v_proj, o_proj
r=8, alpha=16, dropout=0.05, bias=none
```

DeltaNet 的 `in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj` 不进入首版。
实现必须按真实模块拓扑选择目标，不能仅凭叶子名误命中非 full-attention 层；只能有
上述 LoRA 参数和允许的 adapter wrapper state 可训练。

M4 资源门仍按顺序执行：tiny forward/backward → state round-trip → 跨家族拒绝 → 真实
CUDA 推理 → 1-step → 20-step。M7 因完整 `[B,S,V]` logits 与 FP32 CE 临时张量触发
24 GB 单卡 OOM 后，负责人只授权数学目标等价的 Qwen3.5 训练期优化：对 causal shift 后
实际受监督的位置使用原生一维 `logits_to_keep`，再以 FP32 cross entropy 计算原有
assistant-token mean loss。该包装器位于 `oa_groundrag/training/vlm/`，不得修改或进入
Qwen3.5 backend、Gate B 或 runtime/generation 路径。

M7 新资源门使用完整两 epoch/16,000 条确定性序列画像，并冻结 input tokens、supervised
tokens、pixel tensor、vision grid 与综合 footprint 的最坏样本。随后对每个 allocator
profile 在独立新根依次执行最坏样本 forward/backward、fresh 1-step、resume 20-step、
resume 100-step。profile 顺序固定为 native、`expandable_segments=True`、以及相同
expandable allocator 加 post-backward `empty_cache`。仅选择首个完整通过者，不跨 profile
恢复。

资源门要求无 OOM、非有限 loss/gradient 或 allocator retry；所有同步采样点设备可用显存
不少于 2 GiB；step 20/100 post-zero-grad allocated 不超过 step 1 稳态基线 256 MiB；
sampler/cursor/resume 身份严格一致。监督位置实现相对 full-logits 参考的 loss 容差固定为
`atol<=1e-4, rtol<=1e-3`，64 个 LoRA 梯度 aggregate relative-L2 `<=1e-3` 且 cosine
`>=0.9999`；新 step-1 loss 相对旧证据差异 `<=1e-3`。三个 profile 均失败则停止，不得
自动安装 kernel、量化、缩减输入、改变 accumulation/LoRA 或扩展成 hybrid LoRA。
