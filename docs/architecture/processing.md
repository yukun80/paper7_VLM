# 4. Qwen3.5 processor 与 reasoning 合同

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

Qwen3.5 processor 后端必须验证：文本、单图、三图、assistant-only label mask、tensor keys、
离线加载以及 template 参数透传。训练和所有结构化任务强制：

```python
chat_template_kwargs={"enable_thinking": False}
```

运行时策略固定为：

| 任务/路径 | thinking | decoding | max_new_tokens | wire response |
| --- | --- | --- | ---: | --- |
| `VLM_ONLY` | enabled | greedy | 2048 | 仅最终答案 |
| Grounded Pass-1 | disabled | greedy | 768 | 严格 structured output |
| Text-only Pass-2 | disabled | greedy | 768 | 证据受限最终答案 |
| 训练与评价 | disabled | deterministic | 配置固定 | 不消费 reasoning |

`<think>` 或等价 reasoning 内容不得进入 `UnifiedResponse` 公共正文。provider metadata
只记录 `thinking_enabled` 与 reasoning token 数，不记录 reasoning 文本。
