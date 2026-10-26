# 稳定 Python 接口

[返回本节目录](README.md) · [文档总目录](../README.md)

大型根级 facade 不导出训练和评价实现。按职责导入：

```python
from oa_groundrag.segmentation import SpatialInferenceSession
from oa_groundrag.grounding import EvidenceBuilder, RegionSelector
from oa_groundrag.runtime import UnifiedInferenceRuntime, UnifiedRequest, UnifiedTask
```

- `oa_groundrag.segmentation`：OA-AuxSeg model/contracts/inference session。
- `oa_groundrag.vlm`：Qwen model、processor、checkpoint 和 grounded adapter loader。
- `oa_groundrag.grounding`：region、evidence、messages 和 output contracts。
- `oa_groundrag.retrieval`：Text RAG contracts、retriever 与 Pass-2。
- `oa_groundrag.runtime`：统一请求、路由、provider 和 runtime。
- 训练和评价通过 `oa_groundrag.training.*`、`oa_groundrag.evaluation.*` 显式导入。
