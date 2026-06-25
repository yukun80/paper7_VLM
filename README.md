# GeoHazard-HalluGround V2

本仓库用于把 Sen12Landslides 和 GDCLD 整理成 Qwen3-VL 可用的滑坡 VLM benchmark，并准备 Qwen3-VL-2B LoRA 微调数据。

## 环境安装

默认训练环境为 `qwen3vl`：

```bash
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl

cd /home/yukun/codes/paper7_VLM
git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git external/Qwen-VL-Series-Finetune
cd external/Qwen-VL-Series-Finetune
```

安装 PyTorch CUDA 12.8 版本：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装训练基础依赖：

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
pip install qwen-vl-utils
```

如果 `av` 或 `decord` 因 FFmpeg 报错，且当前只训练图像样本，可先跳过视频相关依赖。

WSL2 + Ubuntu 20.04 下建议使用 CUDA Toolkit 12.8：

```bash
export CUDA_HOME=/usr/local/cuda-12.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}

which nvcc
nvcc --version
```

## flash-attn

Ubuntu 20.04 的 glibc 通常是 2.31，预编译 `flash-attn` wheel 可能报 `GLIBC_2.32 not found`。需要强制源码编译，并降低 WSL2 编译资源占用：

```bash
cd /home/yukun/codes/paper7_VLM
conda activate qwen3vl

pip uninstall -y flash-attn flash_attn
pip cache remove flash-attn || true
rm -rf /tmp/pip-* ~/.cache/pip/wheels/*flash* 2>/dev/null || true

export CUDA_HOME=/usr/local/cuda-12.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}

mkdir -p /home/yukun/codes/paper7_VLM/tmp_build
export TMPDIR=/home/yukun/codes/paper7_VLM/tmp_build
export TORCH_EXTENSIONS_DIR=/home/yukun/codes/paper7_VLM/tmp_build/torch_extensions

export TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=1
export NVCC_THREADS=1

pip install flash-attn==2.8.3.post1 \
  --no-build-isolation \
  --no-cache-dir \
  --no-binary=flash-attn \
  -v 2>&1 | tee build_flash_attn.log
```

如果安装日志出现 `Guessing wheel URL`，说明仍可能在尝试预编译 wheel；正确源码编译应看到大量 `nvcc` 和 `.cu` 编译输出。

不使用 `flash-attn` 也可以训练，保持：

```bash
DISABLE_FLASH_ATTN2=True
```

该配置已完成 100 step smoke。

## 环境检查

基础环境检查不强制导入 `flash_attn`：

```bash
conda activate qwen3vl

python - <<'PY'
import torch, transformers, deepspeed, peft, qwen_vl_utils
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

nvidia-smi
```

`flash-attn` 安装完成后单独检查：

```bash
python - <<'PY'
import torch, flash_attn
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("flash_attn", flash_attn.__version__)
print("cuda", torch.cuda.is_available())
PY
```

## 模型权重

下载 Qwen3-VL-2B 权重到本地：

```bash
cd /home/yukun/codes/paper7_VLM
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir models/Qwen3-VL-2B-Instruct
```

训练时使用绝对路径，避免 wrapper 进入外部训练仓库后相对路径失效：

```bash
MODEL_ID=/home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct
```

## 目录

```text
datasets/
  Sen12Landslides/
  GDCLD/extracted/
benchmark/
scripts/
docs/geohazard_benchmark.md
```

GDCLD 全量数据需解压到：

```text
datasets/GDCLD/extracted/
  train_data/
  train_label/
  val_data/
  val_label/
  test_data/
  test_label/
```

## 数据构建

一键构建 V2 benchmark：

```bash
cd /home/yukun/codes/paper7_VLM
bash scripts/run_geohazard_v2_pipeline.sh
```

默认输出：

```text
benchmark/geohazard_halluground_v2_full/
```

可用环境变量覆盖参数：

```bash
OUT_DIR=benchmark/geohazard_halluground_v2_full \
GDCLD_TILE_SIZE=512 \
GDCLD_STRIDE=512 \
GDCLD_NEGATIVE_RATIO=1.0 \
AUDIT_SAMPLES=100 \
bash scripts/run_geohazard_v2_pipeline.sh
```

如需加入 GDCLD `Future work` 候选测试集：

```bash
INCLUDE_FUTURE_WORK=1 bash scripts/run_geohazard_v2_pipeline.sh
```

构建成功后检查：

```bash
cat benchmark/geohazard_halluground_v2_full/validation_report.json
cat benchmark/geohazard_halluground_v2_full/summary.json
```

`validation_report.json` 中 `errors` 应为空。

分阶段脚本：

```text
scripts/1-1_scan_sources.py           扫描 Sen12 和 GDCLD 源数据
scripts/1-2_prepare_sen12_views.py    生成 Sen12 VLM RGB 视图和 mask
scripts/1-3_prepare_gdcld_tiles.py    生成 GDCLD 512x512 tile、mask、bbox
scripts/1-4_merge_annotations.py      合并 metadata 并生成 split
scripts/1-5_export_training_files.py  导出 qwen_vl_sft.jsonl 和 COCO bbox
scripts/1-6_validate_and_summarize.py 校验输出并生成统计和抽查图
```

共享逻辑在 `scripts/geohazard_common.py`。

## 微调数据导出

不要直接用总文件 `qwen_vl_sft.jsonl` 训练，因为它包含 train/val/test。

512 条 smoke 数据：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --max-train-samples 512

python scripts/2-2_convert_qwen_to_llava.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --check-images
```

完整 train 数据：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full

python scripts/2-2_convert_qwen_to_llava.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --check-images
```

默认只导出 `classification,grounding`。如需加入质量判断任务：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --tasks classification,grounding,quality
```

输出文件：

```text
llava_train.json
llava_val.json
llava_test.json
```

## Smoke 训练

脚本直接调用 `external/Qwen-VL-Series-Finetune/src/train/train_sft.py`。默认适配 RTX 4070 12GB：

```text
BATCH_SIZE=1
GRAD_ACCUM=8
LORA_RANK=8
IMAGE_MIN_PIXELS=100352
IMAGE_MAX_PIXELS=200704
```

不使用 `flash-attn` 的稳定 smoke：

```bash
cd /home/yukun/codes/paper7_VLM
conda activate qwen3vl

DATA_PATH=benchmark/geohazard_halluground_v2_full/llava_train.json \
MODEL_ID=/home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/qwen3vl2b_smoke_lora \
MAX_STEPS=100 \
DISABLE_FLASH_ATTN2=True \
bash scripts/train_qwen3vl2b_lora.sh
```

`flash-attn` 验证通过后可测试加速：

```bash
DATA_PATH=benchmark/geohazard_halluground_v2_full/llava_train.json \
MODEL_ID=/home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/qwen3vl2b_smoke_lora_flash \
MAX_STEPS=100 \
DISABLE_FLASH_ATTN2=False \
bash scripts/train_qwen3vl2b_lora.sh
```

脚本默认只做 LoRA，并冻结 vision tower、LLM 和 merger。

## 全量训练

smoke 成功后先重新导出完整 train，再开始第一阶段训练：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full

python scripts/2-2_convert_qwen_to_llava.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --check-images
```

正式第一阶段 LoRA：

```bash
SAVE_STEPS=1000 \
DATA_PATH=benchmark/geohazard_halluground_v2_full/llava_train.json \
MODEL_ID=/home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/qwen3vl2b_stage1_lora \
MAX_STEPS=-1 \
EPOCHS=1 \
DISABLE_FLASH_ATTN2=True \
bash scripts/train_qwen3vl2b_lora.sh 2>&1 | tee outputs/qwen3vl2b_stage1_lora.log
```

如果 `flash-attn` 已验证通过，将最后一段改为：

```bash
DISABLE_FLASH_ATTN2=False
```

## Zero-Shot 推理

Dry run 检查输入和图像路径：

```bash
python scripts/eval_qwen3vl2b_zero_shot.py \
  --input benchmark/geohazard_halluground_v2_full/llava_test.json \
  --output outputs/qwen3vl2b_zeroshot/raw_predictions.jsonl \
  --max-samples 10 \
  --dry-run
```

实际推理：

```bash
python scripts/eval_qwen3vl2b_zero_shot.py \
  --input benchmark/geohazard_halluground_v2_full/llava_test.json \
  --output outputs/qwen3vl2b_zeroshot/raw_predictions.jsonl \
  --max-samples 100
```

该脚本只保存 raw response，指标解析后续单独实现。

## 主要输出

```text
benchmark/<run>/metadata.jsonl
benchmark/<run>/qwen_vl_sft.jsonl
benchmark/<run>/qwen_sft_train.jsonl
benchmark/<run>/qwen_sft_val.jsonl
benchmark/<run>/qwen_sft_test.jsonl
benchmark/<run>/llava_train.json
benchmark/<run>/llava_val.json
benchmark/<run>/llava_test.json
benchmark/<run>/detection_coco.json
benchmark/<run>/validation_report.json
benchmark/<run>/summary.json
outputs/qwen3vl2b_smoke_lora/
outputs/qwen3vl2b_stage1_lora/
```

更多分阶段细节见 `docs/geohazard_benchmark.md`。
