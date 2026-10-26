# 1. 模型家族与独立产物链

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

**后来实施的变化：** 下文“字节身份保持不变”是该版设计冻结时的保护要求；后续经授权的 2B 便携改造调整了路径元数据，原始模型与 Adapter 权重未随之改写。当前发布身份以[发布基线](../status/release.md)为准，不用原方案覆盖它。

### 1.1 保留链：Qwen3-VL-2B

```text
Qwen3-VL-2B-Instruct
→ RS-General Adapter 2B
→ Gate B 2B（冻结）
→ Mask-Grounded Adapter 2B
→ Runtime Profile 2B
```

现有 v2 配置、模型目录、Adapter、checkpoint、Gate B、Benchmark、outputs 和运行时
YAML 的字节身份保持不变。新实现可以改变活动源码的组织，但不得伪造旧 verifier、
重写旧 manifest 或通过兼容 alias 延续旧具体类接口。

### 1.2 新增链：Qwen3.5-4B

```text
Qwen/Qwen3.5-4B（官方后训练多模态模型，固定 revision）
→ RS-General Adapter 4B
→ Gate B 4B（独立协议）
→ Mask-Grounded Adapter 4B
→ Runtime Profile 4B
```

模型必须是官方 `Qwen/Qwen3.5-4B`，不得替换为 `Qwen/Qwen3.5-4B-Base`。
Qwen3.5 使用 `qwen3_5` 混合架构；其默认 thinking 行为必须由任务策略显式控制。

两个家族必须绑定各自的：

- backend name 与 architecture/model type；
- 固定本地模型根、Hub repository 与不可变 revision；
- processor/tokenizer/template 身份；
- 完整模型文件与权重分片 SHA-256 ledger；
- LoRA topology、Adapter、checkpoint 和 best pointer；
- 配置 schema、训练输出、Gate 协议与运行时 profile。

跨家族 Adapter、checkpoint 或 warm-start 必须 fail closed。
