# 数据与训练资料的职责

[返回本节目录](README.md) · [文档总目录](../README.md)

三类数据各有用途，不能因为文件格式相同就合并其含义。

| 数据 | 用途 | 实现入口 |
| --- | --- | --- |
| OA-AuxSeg Benchmark | 光学主模态与可选辅助证据的空间分割 | [分割数据](../../oa_groundrag/data/oa_auxseg/) |
| RS-GeneralDesc Benchmark | 通用遥感图文理解，不接受区域监督资料混入 | [通用遥感数据](../../oa_groundrag/data/rs_general/) |
| Grounded Corpus 与 supervision | 由区域、观察和程序事实构造区域理解监督 | [区域数据](../../oa_groundrag/data/grounded/) |
| Text Evidence Bank | 提供解释、混淆因素和限制三类专业文本知识 | [检索](../../oa_groundrag/retrieval/) |

## 来源与隔离

原始 HDF5 在仓库同级 datasets 中，只读。衍生 Benchmark 的生成必须在明确授权下写入全新目录，不覆盖现有资产。具体字段、模态、有效值与空间配准以数据读取实现和现场检查为准，不能从文件名推断。

通用遥感训练、区域训练、评价资料与运行输入分别管理。区域监督不因单人审核或模型辅助生成就变成专家共识。训练材料不能混入 test/sealed；正式评价的协议和阈值必须事先冻结。

## 查找具体资料

数据总量、实际目录及身份只维护在[成果清单](../status/assets.md)。已经退役的开发评价资料见[退役记录](../status/retired.md)，不得根据旧命令自动重建。

训练关系见[训练设计](training.md)，使用入口见[命令说明](../usage/commands.md)。
