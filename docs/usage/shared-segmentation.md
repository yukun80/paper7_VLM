# Geo-GroundRAG 实验入口

[返回使用目录](README.md) · [设计](../architecture/shared-segmentation.md) · [实际验证](../status/shared-segmentation.md)

当前代码位于独立 v4 工作区。所有输出都必须使用新目录，原模型/Benchmark 保持只读。
已有 Lens 训练权重未在本机发现，因此下述正式模型命令已具备接口，但没有完成真实 Lens 权重验证。
不要把本地 Qwen3.5 基模填入 Lens 模型入口；它缺少训练好的分割部分。

## 环境

沿用已有 `qwen3vl` 环境。需要安装固定的 `shared-segmentation` 可选依赖以及 `vlm` 依赖。
项目外部资产不作为包依赖；官方源码必须保持固定版本。

本次工作区已经准备了固定源码，可直接复现已运行的检查：

```bash
conda activate qwen3vl
export PYTHONPATH="$PWD/.research-deps/xiaomi-lens/src${PYTHONPATH:+:$PYTHONPATH}"
export XIAOMI_LENS_DECODER_ATTN=sdpa
python -m oa_groundrag.runtime.shared_cli --help
```

拿到纯代码包后，可以在独立环境中安装 `.[vlm,shared-segmentation]`。
不要在原共享环境中自动升级依赖；安装与大模型下载不是本次交付的后续自动动作。

## 小样本转换

```bash
python -m oa_groundrag.runtime.shared_cli prepare \
  --benchmark ../benchmark/oa_auxseg_hdf5_v1/full \
  --split train --indices 0 1 --modalities dem \
  --output research_runs/my_joint_samples
```

输出包含新图像、数组、逐条身份、监督来源及文件校验清单，不改写 Benchmark。
索引是 v4 父样本隔离后的视图索引。最多一次选择 8 条，禁止 test/sealed。
加 `--compact <compact目录>` 可转换已有区域文字，仍需 `--benchmark` 来确定共同父样本划分。
选择不属于请求 split 的记录会报错。不要将同一父样本分开导出后混入不同划分。

Python 接口 `RSGeneralReplay` 复用原有通用遥感模板和资产校验，只产生文字监督；
其派生裁剪目录也必须全新。多图顺序由原模板保留。

## 已训练模型的身份要求

`--model` 指向完整本地 Lens/Geo 模型目录，`--identity` 指向目录之外的 JSON 清单：

```json
{
  "kind": "xiaomi_lens",
  "lens_revision": "05106853ed6222763233b4cc31de117b30adef49",
  "files": {"config.json": "完整SHA256", "tokenizer_config.json": "完整SHA256"}
}
```

示例只展示结构；实际 `files` 必须列出目录中每一个文件，包括全部权重分片和 processor 文件。
清单应来自核实过来源的资产接收过程，不能用自行计算哈希替代权重来源核验。
`geo_groundrag` 用于完整的新模型；额外文件、缺少分片、token 不匹配或缺少 decoder 都会拒绝。
模型加载已用实际小型 Qwen3.5 + 官方 decoder 的保存/加载测试验证；真实 Lens 权重尚待验证。

## 推理与知识解释

```bash
python -m oa_groundrag.runtime.shared_cli infer \
  --model /path/to/verified-lens --identity /path/to/identity.json \
  --data research_runs/my_joint_samples --index 0 --mode feedback \
  --device cuda --max-pixels 65536 \
  --rag-config /home/yukun80/codes/paper7_VLM/configs/retrieval/runtime_v1.yaml \
  --output research_runs/my_prediction
```

`--mode joint` 比较单次回答；`feedback` 重新观察实际预测区域。`--image` 支持独立 RGB 图片，
不能与 `--data` 同时指定。辅助观测由数据包提供，缺少时自动走 RGB。
省略 `--rag-config` 时解释标为关闭；知识问答使用 `--knowledge-only --question ...`。
知识检索默认在 CPU，避免与共享模型争抢 GPU。
非方形区域裁剪可能无法放入极小像素预算（例如 1024）；实际处理结果超预算时会拒绝，
此时应在资源允许范围内增大预算，不能把拒绝当作无目标。

输出包含每个区域的 PNG、程序事实、描述、来源、限制及校验清单。
生成不完整或无合法区域时报告失败状态，不伪装成有效无目标结果。

## 有界训练与开发评分

```bash
python -m oa_groundrag.runtime.shared_cli train \
  --model /path/to/verified-lens --identity /path/to/identity.json \
  --data research_runs/my_joint_samples --steps 2 --device cuda \
  --output research_runs/my_delta
```

默认训练 decoder、辅助融合和语言 LoRA；`--visual-last-block` 显式开放最后一个视觉块。
`--description-supervision feedback` 在有文字标注的非空 GT 区域上进行反馈描述训练；
其他分割记录仍进行分割训练。推理用 `--delta` 加载保存的增量，基模身份必须一致。
默认入口不允许超过 2 次更新；`--formal-run --steps N --time-limit S --learning-rate LR`
可在另获授权后运行更长训练。该入口本次未运行正式训练，也不支持优化器断点恢复。
完整数据导出可使用 `export_samples` Python 接口；命令行 `prepare` 默认仅用于小样本检查。

```bash
python -m oa_groundrag.runtime.shared_cli score \
  --data /path/to/val_samples --index 0 \
  --prediction research_runs/my_prediction --output research_runs/my_metrics
```

开发评分拒绝 train、错误影像身份、截断输出和未登记 mask。
对照设计见 `configs/vlm/shared_segmentation/comparisons.json`，其中各实验尚未正式执行。

## 复现检查

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/vlm/shared_segmentation -t . -v
python scripts/train/shared_resource_probe.py --help
```

资源脚本会以原始 Qwen3.5 加随机 decoder 运行最多 2 次更新；它不是正式 Lens 推理。
本次 GPU 更新预算已用完，不应在未取得后续授权时重复运行。报告中的受提示解码 mask
不能冒充自由预测。既有资源测量见[当前状态](../status/shared-segmentation.md)。
