# 14. 相关但不能包装成项目经验的题

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

下面这些是大模型面试常见基础。它们与底层模型或未来部署相关，但当前仓库没有把它们作为
自研模块或已完成实验。面试时可以解释概念和选型边界，不能说“项目已经做过”。

| 题目 | 应理解到什么程度 | 与当前项目的准确关系 |
| --- | --- | --- |
| RoPE / 位置编码 | 为什么 token 顺序需要位置信息，RoPE 如何编码相对位置 | Qwen3-VL 基座内部能力；本项目没有修改或消融 |
| MHA、GQA、FlashAttention | 计算/显存瓶颈与共享 KV、IO-aware kernel 的基本思想 | 配置使用 PyTorch SDPA；没有自研 Attention kernel 或性能横评 |
| KV Cache / Prompt Caching | 自回归时复用历史 K/V 与跨请求前缀缓存的区别 | 模型库可能在单次 generate 内使用 cache；项目没有发布跨请求缓存系统 |
| INT8/INT4、AWQ/GPTQ | 权重量化的精度、校准、显存和吞吐取舍 | 当前 bfloat16 路线未量化；不能声称做过量化无损实验 |
| DPO / PPO / GRPO / RLHF | SFT 后的偏好或奖励优化目标与数据要求 | 当前只做 SFT LoRA，没有偏好优化或强化学习 |
| MCP / Function Calling / Skills | 模型决定调用、工具协议与操作知识封装的层次 | 当前 provider 是项目内部 Python 协议，不是 MCP server，也不是模型 Function Call |
| LangChain / LangGraph | 通用链式组件和状态图编排解决的问题 | 当前固定 Workflow 直接实现，没有使用这些框架 |
| ReAct / Reflection / Multi-Agent | 自主规划、工具循环、反思与多角色协作 | 明确不在当前主线；不能把确定性 Router 称作 Agent |
| Knowledge Graph | 实体、关系、多跳查询的适用问题 | 当前 Text Bank 是可追溯文本检索，没有知识图谱 |

这些专题可从小林的
[大模型工程目录](https://xiaolinnote.com/ai/llm/llm_info.html)、
[Agent 目录](https://xiaolinnote.com/ai/agent/agent_info.html)和
[工具调用目录](https://xiaolinnote.com/ai/)继续学习。回答时建议采用：

> 我理解该技术解决的问题是……；当前项目由于……没有采用；如果出现……瓶颈，我会先用……
> 指标验证，再决定是否引入。

这比为了显得技术栈多而虚构实践更可靠。
