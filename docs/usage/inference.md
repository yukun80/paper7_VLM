# Unified Inference

[返回本节目录](README.md) · [文档总目录](../README.md)


调用方必须显式提供 `UnifiedTask`，不会让 LLM 猜任务。稳定 Python 流程是：

```text
UnifiedRequest
→ CapabilityRouter
→ ExecutionPlan
→ UnifiedInferenceRuntime
→ UnifiedResponse
```

六个任务保持为 `VLM_ONLY`、`SEGMENT_ONLY`、`REGION_UNDERSTANDING`、
`SEGMENT_AND_UNDERSTAND`、`KNOWLEDGE_QA` 和 `REGION_INTERPRETATION`。Router 只产生
确定性 capability 计划，不导入 torch，也不加载模型或 Bank。

只校验请求和执行计划：

```bash
python scripts/infer/oa_groundrag.py \
  --config configs/runtime/inference_v2.yaml \
  --request request.json \
  --dry-run
```

执行只读推理并写入调用方指定的全新根：

```bash
python scripts/infer/oa_groundrag.py \
  --config configs/runtime/inference_v2.yaml \
  --request request.json \
  --output-root /tmp/oa_groundrag_request_001
```

`UnifiedRequest`、`UnifiedResponse` 和 `UnifiedTask` 的 wire schema 保持 v1。普通 runtime
拒绝 `GT_MASK`、test/sealed 路径和已有输出根；candidate ID 缺失或无匹配时按现有合同回退
到 OA-AuxSeg global mask，不自动选择 Top-1。
