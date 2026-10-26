# 2B 可迁移部署

[返回本节目录](README.md) · [文档总目录](../README.md)

2B Unified Runtime 可以放到任意服务器目录，但解压后必须保留 `paper7_VLM` 与
`benchmark` 两个同级目录。配置文件中的普通引用按配置文件所在目录解析；模型、Adapter、
Benchmark 和 Text Bank 的发布身份使用相对 `<bundle-root>` 的 POSIX 路径。运行时会拒绝
绝对发布路径、`..`、目录越界和符号链接逃逸。

此前交付验证所用环境版本为 Python 3.11.15、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、
Transformers 5.3.0、Accelerate 1.10.1、PEFT 0.15.2、safetensors 0.6.2、
qwen-vl-utils 0.0.14、NumPy 2.1.2、h5py 3.16.0、Pillow 11.3.0、PyYAML 6.0.2、
SciPy 1.16.2、Gradio 6.20.0、PyMuPDF 1.28.2、RapidOCR 3.9.2 和 ONNX Runtime
1.28.0。交付包不包含 Python、CUDA 或 NVIDIA 驱动，新服务器应先安装与本机驱动兼容的
PyTorch，再从仓库目录安装项目依赖：

```bash
export OA_BUNDLE_ROOT=/path/to/bundle-root
cd "$OA_BUNDLE_ROOT/paper7_VLM"
python -m pip install -e '.[vlm,retrieval,demo]'
```

运行前可从任意工作目录执行 CPU 静态回归；它会在临时目录复制最小运行元数据、真实
step-900 Adapter 和 Text Bank，不加载完整 Qwen 模型或调用 GPU：

```bash
cd "$OA_BUNDLE_ROOT/paper7_VLM"
python -m unittest tests.runtime.test_portable_bundle -v
python scripts/infer/oa_groundrag.py \
  --config configs/runtime/inference_v2.yaml \
  --request request.json \
  --dry-run
```

交付包应只包含运行代码、运行配置、必要测试、2B 模型、OA-AuxSeg 权重、step-900 Region
Adapter 的最小运行元数据、BGE-M3、完整 Text Bank、知识 PDF 和完整 OA-AuxSeg full
Benchmark，以及本项目文档。以下命令必须从 `<bundle-root>` 执行，本仓库不会自动创建归档：

```bash
cd "$OA_BUNDLE_ROOT"
tar -czf oa-groundrag-2b-runtime.tar.gz \
  paper7_VLM/oa_groundrag \
  paper7_VLM/scripts/infer \
  paper7_VLM/configs/runtime/inference_v2.yaml \
  paper7_VLM/configs/runtime/demo_v1.yaml \
  paper7_VLM/configs/segmentation/full_proposed_dropout_b16_nockpt_e100.json \
  paper7_VLM/configs/vlm/grounded/runtime_v1.yaml \
  paper7_VLM/configs/retrieval/runtime_v1.yaml \
  paper7_VLM/configs/retrieval/sources_v1.yaml \
  paper7_VLM/pyproject.toml \
  paper7_VLM/README.md \
  paper7_VLM/AGENTS.md \
  paper7_VLM/docs \
  paper7_VLM/tests/__init__.py \
  paper7_VLM/tests/runtime/__init__.py \
  paper7_VLM/tests/runtime/test_portable_bundle.py \
  paper7_VLM/models_zoo/Qwen3-VL-2B-Instruct \
  paper7_VLM/models_zoo/ConvNeXt/convnext_small-0c510722.pth \
  paper7_VLM/models_zoo/text_embeddings/bge-m3-5617a9f61b028005a4858fdac845db406aefb181 \
  paper7_VLM/outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt \
  paper7_VLM/outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1/workflow_state.json \
  paper7_VLM/outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1/training/best_checkpoint.json \
  paper7_VLM/outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1/training/checkpoints/step-00000900/manifest.json \
  paper7_VLM/outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1/training/checkpoints/step-00000900/adapter/adapter_model.safetensors \
  paper7_VLM/outputs/stage6_text_rag/text_evidence_bank_v1 \
  benchmark/oa_auxseg_hdf5_v1/full
sha256sum oa-groundrag-2b-runtime.tar.gz \
  > oa-groundrag-2b-runtime.tar.gz.sha256
sha256sum -c oa-groundrag-2b-runtime.tar.gz.sha256
```

归档不得包含 Qwen3.5-4B、8B、PSALM、训练用 RS-GeneralDesc 或 Grounded Benchmark、旧
checkpoint、optimizer、scheduler、RNG、Gate B 目录、Demo 历史输出、缓存、`.git`、
`datasets` 或 `external`。完整 OA-AuxSeg full Benchmark 不能拆分，因为分割 checkpoint
绑定的是该数据集的完整 manifest 与 index 身份。

文档目录整体随包交付，包含归档与讲解附件，保证文档之间的跳转完整。指向未随运行包交付的训练源码或实验产物的链接只用于开发仓库查阅。
