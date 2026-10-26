# 2. 后端架构与公共合同

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

生产代码按能力组织在：

```text
oa_groundrag/vlm/
├── backends/
│   ├── contracts.py
│   ├── registry.py
│   ├── qwen3_vl/
│   └── qwen3_5/
├── checkpoint.py
├── inference.py
└── grounded_runtime.py
```

`VLMBackendSpec` 是不可变规格，至少绑定 backend、model type、architecture、
processor type、模型/processor 构造器、身份策略、thinking 能力和 LoRA target 解析。
registry 必须拒绝未知 backend 和重复注册。对外稳定构造入口为：

```python
resolve_vlm_backend()
build_processor_adapter()
build_model_adapter()
```

训练器、checkpoint loader、Grounded provider 与 Unified Runtime 只依赖模型无关协议，
不得导入具体家族实现。`qwen3_vl` 与 `qwen3_5` 可以共享无状态公共 utility，但不得
共享家族常量、topology 假设、template 身份或模型参数计数。

不提供旧具体类的 legacy alias、旧目录包装或 phase/stage 一级源码入口。
