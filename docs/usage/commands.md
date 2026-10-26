# CLI

[返回本节目录](README.md) · [文档总目录](../README.md)

安装后的稳定统一入口：

```bash
oa-groundrag --help
```

从源码 checkout 运行时，用户按动作查找脚本：

```text
scripts/
├── data/       Benchmark、Corpus、annotation、supervision、Text Bank
├── train/      OA-AuxSeg、RS-VLM、Mask-Grounded Adapter
├── infer/      OA-AuxSeg、RS-VLM、Text RAG、OA-GroundRAG
└── evaluate/   OA-AuxSeg、RS-VLM、Gate B、Grounded、Gate D
```

所有脚本都是薄入口；算法位于 `oa_groundrag/`。常用帮助命令：

```bash
python scripts/data/oa_auxseg_benchmark.py --help
python scripts/data/rs_general_benchmark.py --help
python scripts/data/grounded_corpus.py --help
python scripts/data/grounded_annotation.py --help
python scripts/data/grounded_supervision.py --help
python scripts/data/text_evidence_bank.py --help

python scripts/train/oa_auxseg.py --help
python scripts/train/rs_vlm.py --help
python scripts/train/grounded_adapter.py --help

python scripts/infer/oa_auxseg.py --help
python scripts/infer/rs_vlm.py --help
python scripts/infer/text_rag.py --help
python scripts/infer/oa_groundrag.py --help
python scripts/infer/demo.py --help

python scripts/evaluate/oa_auxseg.py --help
python scripts/evaluate/rs_vlm.py --help
python scripts/evaluate/gate_b.py --help
python scripts/evaluate/grounded.py --help
python scripts/evaluate/gate_d.py --help
```

OA-AuxSeg、RS-VLM 和 Mask-Grounded 的训练入口只允许从 step 0 写入全新输出目录，不提供
续跑参数。配置中的 `resume_checkpoint` 只为兼容既有配置 schema 而保留，值必须为 `null`。
旧 checkpoint 继续支持只读推理和身份核验；新 checkpoint 只保存模型权重、step、模型选择
指标及资产身份，不保存 optimizer、scheduler、RNG 或 sampler 状态。
