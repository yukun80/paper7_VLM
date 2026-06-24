# GeoHazard-HalluGround V2

本仓库用于把 Sen12Landslides 和 GDCLD 整理成 Qwen3-VL 可用的滑坡 VLM benchmark，并准备 Qwen3-VL-2B LoRA 微调数据。

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

## 一键构建 V2 Benchmark

```bash
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

## 分阶段脚本

```text
scripts/1-1_scan_sources.py          扫描 Sen12 和 GDCLD 源数据
scripts/1-2_prepare_sen12_views.py   生成 Sen12 VLM RGB 视图和 mask
scripts/1-3_prepare_gdcld_tiles.py   生成 GDCLD 512x512 tile、mask、bbox
scripts/1-4_merge_annotations.py     合并 metadata 并生成 split
scripts/1-5_export_training_files.py 导出 qwen_vl_sft.jsonl 和 COCO bbox
scripts/1-6_validate_and_summarize.py 校验输出并生成统计和抽查图
```

共享逻辑在：

```text
scripts/geohazard_common.py
```

## 导出 Qwen3-VL 微调数据

不要直接用总文件 `qwen_vl_sft.jsonl` 训练，因为它包含 train/val/test。

先按 split 导出：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full
```

默认只导出 `classification,grounding`。如需加入质量判断任务：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --tasks classification,grounding,quality
```

12GB 显存 smoke 数据：

```bash
python scripts/2-1_export_qwen_splits.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --max-train-samples 512
```

转换为 Qwen-VL-Series-Finetune 使用的 LLaVA JSON：

```bash
python scripts/2-2_convert_qwen_to_llava.py \
  --out-dir benchmark/geohazard_halluground_v2_full \
  --check-images
```

输出：

```text
llava_train.json
llava_val.json
llava_test.json
```

## Qwen3-VL-2B LoRA 微调

准备训练代码：

```bash
git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git external/Qwen-VL-Series-Finetune
```

安装基础依赖：

```bash
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils peft accelerate bitsandbytes deepspeed
```

启动默认 LoRA 训练：

```bash
bash scripts/train_qwen3vl2b_lora.sh
```

默认配置：

```text
MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
DATA_PATH=benchmark/geohazard_halluground_v2_full/llava_train.json
OUTPUT_DIR=outputs/qwen3vl2b_geohazard_lora
BATCH_SIZE=1
GRAD_ACCUM=8
LORA_RANK=8
IMAGE_SIZE=448
```

可覆盖示例：

```bash
DATA_PATH=benchmark/geohazard_halluground_v2_full/llava_train.json \
OUTPUT_DIR=outputs/qwen3vl2b_smoke_lora \
EPOCHS=1 \
bash scripts/train_qwen3vl2b_lora.sh
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
```

更多分阶段细节见 `docs/geohazard_benchmark.md`。
