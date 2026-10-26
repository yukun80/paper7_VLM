# 测试

[返回本节目录](README.md) · [文档总目录](../README.md)

测试按能力镜像源码职责，并全部带 package marker：

```text
tests/
├── test_architecture.py
├── data/{oa_auxseg,rs_general,grounded}/
├── segmentation/
├── vlm/
├── grounding/
├── retrieval/
├── runtime/
├── training/
├── evaluation/{rs_general,grounding,retrieval}/
└── integration/
```

以下是完整测试的发现命令，可能包含依赖本地资产的检查；不属于每次文档整理必须运行的命令。执行前先核对当前授权与相关测试的读取范围：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

需要本地 Benchmark、checkpoint 或 CUDA 的测试必须显式 skip 或使用独立 bounded smoke，
不得为追求绿色结果重建正式资产或访问 sealed test。实时测试结果只记录在
[当前进度](../status/README.md)。
