# 科学合同边界

[返回本节目录](README.md) · [文档总目录](../README.md)

- OA-AuxSeg 只输出 probability、mask、no-target、候选区域和 modality evidence；模型
  `forward` 不读取 HDF5。
- RS-GeneralDesc 只接受 canonical manifest/payload/ledger/asset identity，不接入 mask 或
  grounded supervision。
- Grounded Evidence 使用原始光学图、严格 binary mask、clean crop、Programmatic Facts 和
  Pass-1 structured observation；empty/no-target 合同不得静默修复。
- Text RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1 观察，不把检索知识说成
  当前图像事实，也不把候选区域升级为确认滑坡。
- Gate B、Gate D、Benchmark、checkpoint、Text Evidence Bank 和正式 outputs 的身份与历史
  实现提交绑定；路径迁移不伪造旧 verifier 继续通过。
