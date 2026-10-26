# 15. 面试前的十句复习卡

[返回本节目录](README.md) · [文档总目录](../../README.md)

<a id="review-cards"></a>

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

1. OA-GroundRAG 的主问题是 Where、What、How，不是让一个模型包办全部任务。
2. OA-AuxSeg 是独立空间专家；mask 是关注区域，不是确认滑坡标签。
3. Grounding 用原图、二值 mask、clean crop 和只读程序事实连接像素与语言。
4. Shared MLLM 是 Qwen3-VL-2B 加最终 Grounded LoRA，不是 runtime 串联两套 Adapter。
5. 项目使用 SFT LoRA：SFT 是训练目标，LoRA 是参数更新方式。
6. Pass-1 只说图像观察，Pass-2 才结合文本知识谨慎解释。
7. Retrieval 是双查询、FTS5 + BGE-M3、RRF 和三类平衡 Evidence Packet。
8. Router 是显式 task 驱动的 Workflow，不是 Agent，也不让 LLM 猜任务。
9. 严格失败、证据 ID 和 artifact identity 是科研复现的一部分，不是多余的工程负担。
10. 工程完成、自动开发评价和科学 Gate 是三个层次；任何时候都不要越级表述。
