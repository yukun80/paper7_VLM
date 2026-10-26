# 科学验收与不可扩张边界

[返回本节目录](README.md) · [文档总目录](../README.md)

| Gate / 边界 | 当前状态 | 允许的结论 |
| --- | --- | --- |
| Gate A | 未执行 | OA-AuxSeg 只有工程定版 checkpoint；不得导出或宣称正式 fixed predicted masks |
| Gate B | 2B 与 4B 各自冻结作用域正式通过 | 分别只证明同一模型家族中 RS-General Adapter 相对自身 base 的提升；跨模型报告不作升级判据或优越性结论 |
| Gate C | 未执行 | Region Adapter checkpoint 可用，但已无当前 Eval-dev artifact；不能证明模型正确依赖 mask |
| Gate D | 未执行；旧 development artifacts 已退役 | 只能报告在线 RAG 工程合同；不能声称 RAG 有科学增益 |
| Gate E / F | 条件未触发 | 不启动 Landslide-Evidence Adapter；不宣称 retention 通过 |
| sealed test | 未访问 | 任何阈值、协议或实现均不得从 test 反推 |

持续有效的数据与事实边界：

- Corpus 和 supervision 只使用 train；如按保留配方重建 OA-GroundedEval-dev，只能使用 val；
  test/sealed test 不得进入 Corpus、开发阈值或当前报告。
- 6,974 条 compact supervision 是 `mixed_model_and_single_expert`，不是 Gold 或专家共识。
- OA-AuxSeg 当前辅助 registry 只接受 `dem / insar_velocity / slope`；SAR 尚未接入。
- encoded InSAR 的物理单位与 sign convention 未确认时，只能作为 encoded evidence；不得
  生成定量位移或物理方向结论。
- LMHLD 和 Landslide4Sense 缺少可靠地理 group；不得从文件名或 sample ID 伪造空间身份。
- Text RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1 observation，不把
  候选区域升级为确认滑坡。
- Stage 7 Case RAG 只在正式 Gate D 证明稳定增益后考虑；Stage 8
  Landslide-Evidence Adapter 只在 Gate E 失败时考虑。

Geo-GroundRAG v4 当前仅完成工程与有限资源检查，未新增任何科学 Gate 结论，见[本次状态](shared-segmentation.md)。
