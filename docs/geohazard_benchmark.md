# GeoHazard-HalluGround Benchmark 第一阶段构建流水线

本目录说明如何把 Sen12Landslides 和 GDCLD 整理成可用于 Qwen3-VL 等视觉语言模型微调的滑坡 benchmark。脚本不会改动 `datasets/` 下的原始数据，只会在 `benchmark/` 下写入派生图像、掩膜、标注和训练样本。

旧单体脚本 `scripts/build_geohazard_benchmark.py` 已删除。后续扩展新数据集、新视图或新标注格式，应优先修改 `scripts/geohazard_common.py` 和 `scripts/1-1_*` 到 `scripts/1-6_*` 阶段脚本。

## 输出目录

标准输出结构如下：

```text
benchmark/<run>/
  intermediate/
    source_manifest.jsonl
    sen12_samples.jsonl
    gdcld_samples.jsonl
  vlm_views/
  segmentation_masks/
  segmentation_masks_redblack/
  metadata.jsonl
  qwen_vl_sft.jsonl
  detection_coco.json
  splits/
  figures/
  audit/
  summary.json
  validation_report.json
```

主要文件含义：

- `intermediate/source_manifest.jsonl`：原始数据源清单，记录 Sen12 patch 和 GDCLD image-label pair。
- `intermediate/sen12_samples.jsonl`：Sen12 渲染视图后的样本 metadata。
- `intermediate/gdcld_samples.jsonl`：GDCLD 512×512 tile 样本 metadata。
- `metadata.jsonl`：合并后的统一样本 metadata。
- `qwen_vl_sft.jsonl`：Qwen 风格图像指令微调样本，prompt 和回答说明为中文。
- `detection_coco.json`：由语义 mask 自动派生的 COCO bbox 标注。
- `segmentation_masks/`：与 VLM 输入图对齐的二值 `0/1` PNG 掩膜。
- `segmentation_masks_redblack/`：与二值 mask 平行的红黑 RGB 可视化标签，黑色为背景，红色为滑坡，仅用于人工查看。
- `vlm_views/`：供 VLM 输入的 RGB PNG 视图。
- `splits/`：按 `train`、`val`、`test`、`test_candidate` 拆分后的 metadata。
- `figures/`：数据来源、传感器、split 和质量标签统计图。
- `audit/`：随机抽查图，叠加 mask 和 bbox。
- `summary.json`：样本数量和分布统计。
- `validation_report.json`：一致性校验结果。

## Benchmark 版本

- `v0`：通用 RGB 滑坡 benchmark。使用 Sen12 S2 事件后真彩色视图和 GDCLD RGB tile，适合先跑通训练链路。
- `v1`：多视图地灾 benchmark。在 `v0` 基础上加入 Sen12 S2 假彩色、事前/事后多图样本，以及 Sentinel-1 升轨/降轨 SAR 视图。灾前和灾后图像分别保存为独立 `128×128` RGB 图，并在 SFT 中作为两个 image object 输入；`pair_preview` 只用于人工查看，不作为训练主输入。
- `v2`：HalluGround 试验版。在 `v1` 基础上加入 DEM hillshade 辅助视图，并在 SFT 中增加质量判断和证据充分性任务。

## 分阶段脚本

### 1-1 扫描原始数据源

扫描 Sen12 和 GDCLD 原始数据，生成 `source_manifest.jsonl`。当前 `datasets/GDCLD/extracted` 的小样本结构已支持，包括 `test_data` 和 `Future work` 子目录。

```bash
python scripts/1-1_scan_sources.py \
  --sen12-root datasets/Sen12Landslides \
  --gdcld-root datasets/GDCLD/extracted \
  --out-dir benchmark/geohazard_halluground_v0 \
  --clean
```

smoke test 可限制 Sen12 数量：

```bash
python scripts/1-1_scan_sources.py \
  --sen12-root datasets/Sen12Landslides \
  --gdcld-root datasets/GDCLD/extracted \
  --out-dir benchmark/smoke_pipeline \
  --max-sen12-s2 5 \
  --max-sen12-s1asc 0 \
  --max-sen12-s1dsc 0 \
  --clean
```

### 1-2 生成 Sen12 VLM 视图

Sen12 原始数据是固定大小 patch，因此按 patch-level 生成 RGB 视图，不再切片。

```bash
python scripts/1-2_prepare_sen12_views.py \
  --out-dir benchmark/geohazard_halluground_v0 \
  --version v0
```

构建 `v1` 或 `v2` 时替换 `--version` 即可：

```bash
python scripts/1-2_prepare_sen12_views.py \
  --out-dir benchmark/geohazard_halluground_v1 \
  --version v1
```

### 1-3 生成 GDCLD tile

GDCLD 是高分辨率 RGB 整景图，不再整图缩放。默认使用 `512×512` tile，步长也是 `512`。脚本使用 `rasterio` windowed read，只读取当前窗口，避免超大 PNG/TIF 触发整图解码和内存压力。

```bash
python scripts/1-3_prepare_gdcld_tiles.py \
  --out-dir benchmark/geohazard_halluground_v0 \
  --gdcld-tile-size 512 \
  --gdcld-stride 512
```

smoke test：

```bash
python scripts/1-3_prepare_gdcld_tiles.py \
  --out-dir benchmark/smoke_pipeline \
  --gdcld-tile-size 512 \
  --gdcld-stride 512 \
  --max-gdcld-tiles 20
```

GDCLD 标签中可能出现 `1`、`85`、`255` 等不同正类编码。脚本统一使用 `>0` 二值化，输出 mask 只有 `0/1`。部分 GDCLD 标签存在边缘整行或整列正类伪影，脚本会先清理这类边框，再过滤 bbox 过小、面积比例过低的细线伪标签，避免把 `bbox=[0,0,512,1]` 这类 tile 作为滑坡正样本。

`Future work` 默认不导出到正式训练集；如需作为候选测试数据，可显式加入：

```bash
python scripts/1-3_prepare_gdcld_tiles.py \
  --out-dir benchmark/geohazard_halluground_v0 \
  --include-gdcld-future-work
```

显式加入后，Future work 样本的 split 为 `test_candidate`，不会混入 `train`、`val` 或正式 `test`。

### 1-4 合并 annotation

合并 Sen12 和 GDCLD 样本 metadata，并生成 split 文件。

```bash
python scripts/1-4_merge_annotations.py \
  --out-dir benchmark/geohazard_halluground_v0
```

划分规则：

- Sen12 按 `region_id + event_date` 生成 deterministic split，避免同一事件随机泄漏。
- GDCLD 官方 `test_data` 保持为 `test`。
- GDCLD `Future work` 只有显式导出时进入 `test_candidate`。

### 1-5 导出训练和评估文件

从 `metadata.jsonl` 导出 Qwen-VL SFT 样本和 COCO bbox。

```bash
python scripts/1-5_export_training_files.py \
  --out-dir benchmark/geohazard_halluground_v0 \
  --version v0
```

初版训练建议让 VLM 输出分类结果和 bbox。segmentation mask 保留用于 IoU、Dice 或后续专门分割模型评估，不建议在第一版强制 Qwen 直接生成 mask token。

### 1-6 校验和统计汇总

校验图像、二值 mask、红黑可视化 mask、bbox、split 泄漏，并生成统计图和随机抽查图。每次生成 audit 前会清空旧 audit 目录，避免旧抽查图残留造成误判。

```bash
python scripts/1-6_validate_and_summarize.py \
  --out-dir benchmark/geohazard_halluground_v0 \
  --audit-samples 100
```

若 `validation_report.json` 中 `errors` 为空，则第一阶段数据构建通过基础验收。

## 一键顺序示例

V0 smoke test：

```bash
python scripts/1-1_scan_sources.py \
  --sen12-root datasets/Sen12Landslides \
  --gdcld-root datasets/GDCLD/extracted \
  --out-dir benchmark/smoke_pipeline \
  --max-sen12-s2 5 \
  --max-sen12-s1asc 0 \
  --max-sen12-s1dsc 0 \
  --clean

python scripts/1-2_prepare_sen12_views.py \
  --out-dir benchmark/smoke_pipeline \
  --version v0

python scripts/1-3_prepare_gdcld_tiles.py \
  --out-dir benchmark/smoke_pipeline \
  --gdcld-tile-size 512 \
  --gdcld-stride 512 \
  --max-gdcld-tiles 20

python scripts/1-4_merge_annotations.py \
  --out-dir benchmark/smoke_pipeline

python scripts/1-5_export_training_files.py \
  --out-dir benchmark/smoke_pipeline \
  --version v0

python scripts/1-6_validate_and_summarize.py \
  --out-dir benchmark/smoke_pipeline \
  --audit-samples 5
```

V1 小样本：

```bash
python scripts/1-1_scan_sources.py \
  --sen12-root datasets/Sen12Landslides \
  --gdcld-root datasets/GDCLD/extracted \
  --out-dir benchmark/smoke_pipeline_v1 \
  --max-sen12-s2 2 \
  --max-sen12-s1asc 2 \
  --max-sen12-s1dsc 2 \
  --clean

python scripts/1-2_prepare_sen12_views.py --out-dir benchmark/smoke_pipeline_v1 --version v1
python scripts/1-3_prepare_gdcld_tiles.py --out-dir benchmark/smoke_pipeline_v1 --max-gdcld-tiles 20
python scripts/1-4_merge_annotations.py --out-dir benchmark/smoke_pipeline_v1
python scripts/1-5_export_training_files.py --out-dir benchmark/smoke_pipeline_v1 --version v1
python scripts/1-6_validate_and_summarize.py --out-dir benchmark/smoke_pipeline_v1 --audit-samples 5
```

## 必须保留英文的内容

为了避免破坏后续训练和评估脚本，以下内容保留英文或原始约定：

- 文件名和目录名，例如 `metadata.jsonl`、`qwen_vl_sft.jsonl`、`vlm_views/`。
- CLI 参数名，例如 `--sen12-root`、`--gdcld-tile-size`、`--version`。
- JSON 字段名，例如 `sample_id`、`hazard_present`、`bbox_norm_1000`。
- split 名称 `train`、`val`、`test`、`test_candidate`。
- 数据集名、模型名和格式名，例如 Sen12Landslides、GDCLD、Qwen、COCO、SFT。
- 类别规范值 `landslide` 和 `none`。

## SFT 样本示例

`qwen_vl_sft.jsonl` 中每行是一个训练样本，结构类似：

```json
{
  "id": "gdcld_gdcld_pair_00000_Lushan_tile_000001::grounding",
  "image": "benchmark/smoke_pipeline/vlm_views/gdcld/gdcld_gdcld_pair_00000_Lushan_tile_000001.png",
  "task": "grounding",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "benchmark/smoke_pipeline/vlm_views/gdcld/gdcld_gdcld_pair_00000_Lushan_tile_000001.png"
        },
        {
          "type": "text",
          "text": "传感器：RGB 高分辨率遥感影像；模态：rgb_high_resolution_tile；空间分辨率：未知 m；事件或观测日期：未知。如果图像中存在可见滑坡，请返回滑坡证据区域 bbox。bbox 使用 0-1000 归一化图像坐标，格式为 [x1,y1,x2,y2]；如果没有可见滑坡，则返回空 bbox。"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"hazard_present\": true, \"hazard_type\": \"landslide\", \"bbox_0_1000\": [120, 240, 640, 810], \"evidence_sufficiency\": \"足以支持可见滑坡存在判断\"}"
    }
  ]
}
```

## 注意事项

- 当前两个公开数据集只支撑 `landslide` 与 `none`，不能直接扩展为崩塌、泥石流、岩屑坡或 InSAR 形变证据。
- Sen12 灾前/灾后样本的 `rendered_image` 指向灾后图像；`pre_image`、`post_image` 和 `image_sequence` 用于多图 VLM 输入；`pair_preview_image` 仅用于 audit 和人工查看。
- GDCLD bbox 是从语义 mask 自动派生的，不代表实例级滑坡标注。
- GDCLD tile 的新增追溯字段包括 `source_scene_file`、`source_label_file`、`source_window_xywh`、`source_scene_width`、`source_scene_height`、`tile_index`、`is_future_work`、`mask_positive_pixels`。
- `mask_path` 指向训练和评估使用的二值标签；`mask_visual_path` 指向人工查看用的红黑标签，不参与 SFT 或 COCO 导出。
- 对 Sen12 灾前/灾后样本，`mask_path`、bbox 和 COCO 标注始终对齐灾后 `post_image` 的 `128×128` 坐标，不以 pair preview 坐标为准。
- 全量构建会生成大量 PNG 和 JSONL，`benchmark/` 已被 `.gitignore` 忽略，避免派生数据误入版本管理。
