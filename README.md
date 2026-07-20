# SAMI-GroundSegDesc

当前主线是经 ADR-0001 接受的 **SAMI-GroundSegDesc greenfield rewrite**。科学问题固定为：使用
同一区域、单时相或同期的多源遥感观测，在一个 reference canvas 上分割滑坡，并生成只受当前有效
模态支持的区域描述。P1 正在实施 Canonical Benchmark v3；P1.1 的严格合同/只读审计和 P1.2 的
reference-canvas/空间原语均已工程通过，但尚未构建真实 Small benchmark。

本文件后部的 Multi-Source Qwen-PSALM-Seg/Benchmark v2 命令暂时保留为只读 legacy baseline
运行记录，不属于 greenfield runtime。新代码不读取旧 benchmark、cache、config 或 checkpoint；
旧资产仅通过已验证的 baseline tag/branch 和 deletion manifest 分阶段保留。

## Greenfield P1.1–P1.2：合同、只读审计与空间原语

从仓库根目录安装当前包与测试依赖：

```bash
conda activate qwen3vl
python -m pip install -e '.[test]'
```

当前唯一 CLI 是 `sami-gsd`。P1.1 开放的命令仍只有 `data audit`；P1.2 不增加第二个 CLI：

```bash
sami-gsd data audit --help
```

运行不依赖 `pytest` 的 CPU/synthetic 验收：

```bash
PYTHONPATH=src python -m unittest discover -s tests/p1 -v
```

安装 test extra 后，也可通过同一测试目录运行：

```bash
PYTHONPATH=src python -m pytest -q tests/p1
```

手动执行 Small raw-source 审计（会递归读取并计算所有可见文件的 SHA-256，可能耗时）：

```bash
sami-gsd data audit \
  --config configs/benchmark_v3_small.yaml \
  --output-dir ../benchmark/sami_landslide_v3/p1_1_audit_small
```

该命令只读 raw data，输出 `inventory.json`、`source_registry.yaml`、`license_report.json` 和
`audit_manifest.json`，拒绝覆盖已有输出目录。配置中的九个 source 当前全部
`allowed_for_training=false`；未知或未审核许可证不能进入 training-eligible index。

P1.2 冻结以下空间边界：reference 依次选 official/human 原生 mask 栅格、完整覆盖且 GSD 最细的
registered mask 栅格、或唯一语言图像；候选顺序不影响结果。内部 box 永远使用 reference pixel
半开区间，仅在 Qwen grounding 序列化边界转换为 `[0,1000]` 整数。`TransformStep` 记录
pixel-edge 坐标、half-pixel-center sampling 和 clamp border；image 固定 bilinear，mask/valid
固定 nearest。crop/resize/pad 的 coordinate inverse 只对 retained valid-content footprint 承诺，
padding 与 nodata 始终从有效 target 中排除。无可靠双向变换的 support 只能是 `global_only`，不得
暴露 pixel-level transform。

P1.2 仍不执行 source-specific materialization、真实 Small/Full build、split、duplicate grouping、
task expansion 或语言子集构建，也不物理删除任何旧文件。工程证据见
`docs/reports/p1/p1_2_spatial_report.json`；后续工作仍需新的明确 P1 子任务授权。

## Legacy baseline 目录约定

默认从 `paper7_VLM` 根目录运行命令，并使用同级大数据目录：

```text
/home/yukun80/codes/
├── datasets/
├── benchmark/
└── paper7_VLM/
```

可用 `PAPER7_DATASETS_ROOT` 和 `PAPER7_BENCHMARK_ROOT` 覆盖物理位置。JSONL 始终保存
`datasets/...`、`benchmark/...` 逻辑路径，不写机器绑定的绝对路径。

推荐环境：

```bash
conda activate qwen3vl
export PYTHONPATH=SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}
```

也可安装命令别名：

```bash
python -m pip install -e SEG_Multi-Source_Landslides
```

## 当前 Small 工程进展（2026-07-18）

以下状态来自当前落盘 validation、gate 和 training completion report，不以“代码存在”或
“命令退出成功”替代验收：

| 对象 | 当前证据 | 状态 |
| --- | --- | --- |
| Landslide V2 Small | final 5,561 parents，`errors=[]`；保留 11 条 GDCLD preview 低对比 warning | engineering-valid |
| Description M1.1 Small | `description_benchmark_m1_v4_answer_trace`，40,963 records / 19,685 parents，verified cross-split clusters = 0 | engineering-valid |
| M2 Bridge Small | v7 prepare，33,114 regions；300 Pilot parents = 180/60/60；485 review items；0 expert labels | `awaiting_expert_review` |
| Unified SegDesc Small | v3 component contract，96,069 references / 27,982 parents；0 expert records | engineering-valid、auto-only |
| Description Vision Cache | M3 v3 strict migration，25,239 records / 99 shards；24,212,256,037 bytes 严格复用并重放 | engineering-valid |
| D-1 | v13 gate，zero-shot + 64-sample/100-step overfit，`d_minus_one_complete=true`、`errors=[]` | engineering-valid |
| D0 | preflight v6 ready；MMRS caption 1,000 steps；completion v3 `terminal_status=completed` | engineering-valid |
| D1/D2/D3a | 实现与门禁已存在，尚未运行当前 Small 正式课程 | pending engineering runs |
| D3b/D4/M6 expert/M7 final | 需要冻结的真实专家 Bridge 及后续科学门禁 | blocked by human review |

D0 当前接受的 run 是 `outputs/qpsalm_description/d0_mmrs_seed42`：best/last 均到 step 1000，
best score 为 0.5602917596，256 条 dev 样本的 teacher-forced loss 为 1.6976218831；monitor
generation 只覆盖其中 32 条，caption token F1 为 0.5602917596（95% parent bootstrap CI
0.5039–0.6137）。这些数值证明训练、验证、checkpoint 和 completion 链路有效，不是完整
RSIEval、专家事实性或 grounded-region 科学结论。不要覆盖该 D0 目录；下一阶段从其
`checkpoint_best.pt` 显式初始化 D1。

## 构建 Benchmark V2

构建 small：

```bash
SMALL_LIMIT=500 \
bash scripts/run_1_build_benchmark.sh small
bash scripts/run_2_build_instruction_dataset.sh small
```

`SMALL_LIMIT` 表示每个 `dataset_name + split` 的父样本上限，不是整个 split 的
总上限。instruction 构建还会从父样本派生 global、referring 和 no-target 任务，
因此 instruction 行数通常明显大于父样本数。

构建 full：

```bash
bash scripts/run_1_build_benchmark.sh full
bash scripts/run_2_build_instruction_dataset.sh full
```

输出分别位于同级：

```text
../benchmark/multisource_landslide_v2_small
../benchmark/multisource_landslide_v2_full
```

质量门要求 source、final、referring-target 和 instruction validation 的 `errors == []`。
v2 每个模态必须显式包含 `family`、`sensor`、`product_type`、band metadata、GSD、units、
signed、quality、结构化 normalization 和归一化前物化的 valid mask。

独立阶段入口：

```bash
python scripts/1-benchmark/1-1_scan_sources.py --datasets-root datasets --out-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-2_build_index.py --mode small --datasets-root datasets --out-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-3_validate_index.py --benchmark-dir benchmark/multisource_landslide_v2_small --stage source
python scripts/1-benchmark/1-4_preprocess_samples.py --benchmark-dir benchmark/multisource_landslide_v2_small --strategy materialize
python scripts/1-benchmark/1-3_validate_index.py --benchmark-dir benchmark/multisource_landslide_v2_small --stage final
python scripts/1-benchmark/1-5_build_splits.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-6_build_referring_targets.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-3_validate_index.py --benchmark-dir benchmark/multisource_landslide_v2_small --stage referring_target
python scripts/1-benchmark/1-7_summarize_benchmark.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/2-instruction/2-1_build_instruction_templates.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/2-instruction/2-2_apply_instruction_templates.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/2-instruction/2-3_validate_instruction_index.py --benchmark-dir benchmark/multisource_landslide_v2_small
```

## 构建 Description Benchmark M0/M1

`docs/benchmark_GAR.md` 的下一阶段先构建遥感全图描述与区域对齐 benchmark。
脚本只从原始数据中选择所需 parent，并将对应图片原样复制到 benchmark；不会修改
现有 Landslide Benchmark V2。默认读取：

```text
../datasets/MMRS-1M
../datasets/RSGPT/dataset/RSICap
../datasets/RSGPT/dataset/RSIEval
```

同时兼容旧的 `external/RSGPT/dataset` 布局；可用 `PAPER7_RSGPT_DATA_ROOT` 覆盖。
构建并验证 small：

```bash
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

如需重建已有派生产物：

```bash
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

默认 small 保留全部 RSICap/RSIEval，分层选择 12,000 个 MMRS Caption canonical parent 和
5,000 个 DIOR-RSVG parent。dHash 只用于召回候选；统一 RGB 64x64 后 MAE 不超过 3.0 的
同图重编码会在 split 前合并为一个 canonical parent，并合并多来源 caption。输出位于：

```text
../benchmark/qpsalm_description_v2_small/
├── data/
├── indexes/
├── manifests/
└── reports/
```

其中 `3-4_deduplicate_and_split.py` 只冻结 selected source records，
`3-5_materialize_description_images.py` 才复制图片并发布最终索引；验证与汇总依次为
`3-6_validate_description_benchmark.py` 和 `3-7_summarize_description_benchmark.py`。
可用 `DESCRIPTION_COPY_WORKERS=8` 调整本地复制并行度；研究复现时应保持默认
`DESCRIPTION_PERCEPTUAL_MAE_THRESHOLD=3.0`，修改门槛必须形成新的 split protocol。
canonical 合并后的每条 caption answer 都保留 source answer index、原文 hash 和来源记录；
`verified_perceptual_duplicates.jsonl` 同时记录 canonical 选择与每个成员的 split action。

在 `/tmp` 手动执行小规模闭环而不覆盖正式 benchmark：

```bash
PAPER7_BENCHMARK_ROOT=/tmp \
MAX_SAMPLES=8 \
RUN_CONTROL=--overwrite \
DESCRIPTION_COPY_WORKERS=4 \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

最终 `all/train/dev/test` 索引只引用 `data/` 中的图片；`*_source.jsonl` 和
provenance 保留 `datasets/...` 路径用于审计。质量门为
`reports/validation_report.json` 中 `errors == []`，并要求
`verified_perceptual_duplicate_cross_split_groups == 0`。训练应读取
`indexes/train_eligible.jsonl`，完整 `train.jsonl` 仍保留零权重审计答案。DIOR 只提供
box-to-phrase 和 phrase-to-candidate-region 对齐监督，不作为详细区域 caption 真值。
物化图片仅用于本地研究；未经各源数据许可审核不得公开重新分发。
本阶段通过后再进入 Landslide Bridge、MGRR 和描述 Adapter 训练。

## Landslide Bridge Pilot

M2 从 `multisource_landslide_v2_<mode>` 构建区域清单、三级多源证据、规则候选文本和
双人专家审核包。`pseudo_instance_component` 仅表示 8 邻域伪实例组件，不等同人工实例。
正式 Pilot 必须精确包含 300 个不同 parent，并满足 train/val/test = 180/60/60；候选不足会
作为验证错误报告。使用 `MAX_SAMPLES` 的缩小构建只用于 smoke，不能进入专家 gate。
准备阶段不会生成专家标签：

```bash
BRIDGE_STAGE=prepare \
BRIDGE_PILOT_PARENTS=300 \
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_4_build_landslide_bridge.sh small
```

审核者分别填写 `review_package/reviewer_1_template.*` 和
`review_package/reviewer_2_template.*` 的副本。`decision` 只允许 `accept/revise/reject`；
双人分歧必须提供仲裁文件。Pilot 分析后，还需人工复制并填写
`manifests/evaluation_gate_manifest.template.json`，将状态显式冻结，再执行合并：
模板已绑定本轮 Pilot parent、review selection 和 candidate index 的 SHA-256；不得从其他
run 复制 gate，也不得修改 `bindings`。当前模板还要求人工冻结 target-status、总体及
unavailable-modality UFCR、ERFS、UFCR 非劣界限、parent-level bootstrap seed，以及五类
正式反事实的最小有效 parent 数；任一项仍为 `null` 时 merge 会拒绝发布专家 gate。若当前
prepare 产物来自缺少这些字段的旧模板，应在开始人工审核前重跑 prepare；candidate/index
协议不变，原始 datasets 不受影响。

```bash
BRIDGE_STAGE=merge \
REVIEWER_1=/path/to/reviewer_1_completed.jsonl \
REVIEWER_2=/path/to/reviewer_2_completed.jsonl \
ARBITRATION_FILE=/path/to/arbitration_completed.jsonl \
EVALUATION_GATE=/path/to/evaluation_gate_frozen.json \
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_4_build_landslide_bridge.sh small
```

正式 prepare 验收状态为 `awaiting_expert_review`、`pilot_protocol_complete == true` 且
`errors == []`；带 `MAX_SAMPLES` 的构建状态为 `smoke_only`。只有仲裁清零、三个 split
均有审核通过记录且 gate 已冻结时，状态才会变为 `expert_pilot_frozen`。Bridge 只引用
Landslide V2 已物化模态，不读取原始 `datasets/`，也不重复复制多源数组。

M2 prepare 有效后，可构建独立的 task-neutral Description Vision Cache v1。它按 parent
缓存，不包含 instruction、region geometry 或 segmentation state；现有 segmentation
Vision Cache v3 只读复用，不会被覆盖：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc cache build \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --description-benchmark benchmark/qpsalm_description_v2_small \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --segmentation-vision-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_description/cache/small_vision_v1_m3v3 \
  --device cuda --backend qwen --overwrite
```

当前 builder 为 `description_vision_cache_m3_v3_shard_content_bound`。构建器为每个 shard
保存路径、字节数、record 数和 SHA-256，深度遍历全部 shard，重新绑定当前
Description/Bridge 索引，并逐条核对复用的 segmentation cache v3 record；正式编码开始前还会
要求 Description M1.1 v4 零 verified cross-split cluster，以及 Bridge M2 v7 Pilot 完整且状态为
`awaiting_expert_review`/`expert_pilot_frozen`。两份 benchmark validation report 的相对路径、
builder、状态、字节数和 SHA 与输入索引一起写入 manifest，避免先用旧 Bridge 昂贵重建 cache。
结果原子写入
`outputs/qpsalm_description/cache/small_vision_v1_m3v3/validation_report.json`。正式 M3 验收要求
`errors == []`、`source_cache.isolation_unchanged == true`，且复用 record 的
`validated_records == reused_records`、`shard_integrity.all_verified == true`。训练期首次读取
cache 时必须存在与当前 manifest 精确绑定的成功深度报告；缺失、失败或报告/manifest 漂移会在
模型构建前终止。每个 shard 首次读取也会核验 SHA，避免同形状 tensor 数值损坏绕过
shape/fingerprint 检查。checkpoint 进一步保存 manifest、validation report、输入索引、源 cache
provenance 与 shard inventory 的 artifact binding；正式 M4/M6 会从该 binding 重开 cache 并重放
全部 shard SHA 和 benchmark validation 指纹。训练/评价 DataLoader 打开时还会执行 live benchmark 审计：Description stream
以 `qpsalm_description_engineering_audit_v1_cache_partition_bound` 重读 `all/train/dev/test`，证明
三 split 精确分区 cache 绑定的 `all.jsonl`，并重建正权重 `train_eligible`；Bridge region stream
以 `landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound` 证明当前
`candidate_all.jsonl` 同时绑定 cache 的 `multisource_parent` 输入，且 `auto_train` 是未冒充
expert truth 的精确 train 投影。两项 audit 都进入 checkpoint data binding，因此构建 cache 后再
改索引不能被后续课程静默接受。旧的 M3 v2 cache 不能被运行时直接读取，但允许执行一次严格、
side-by-side 的迁移。迁移逐 shard 计算 SHA/字节数/record 数，逐条校验 tensor shape/finite、
lookup、task-neutral 字段和 `source_content_hash`，并按当前 Description v4、Bridge v7、
segmentation cache v3 重放每个 parent。只有所有绑定完全一致时才在同一文件系统创建 hardlink；
任何漂移、损坏或跨文件系统都会明确要求完整重建。迁移不修改旧的 23 GiB cache，也不增加运行时
fallback。staging 深验后还会从正式发布目录重新遍历全部 shards，并重放旧/新 inode 与 source
hash；正式路径终态未通过时会删除新建的 hardlink 目录并返回失败：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc cache migrate \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --legacy-cache outputs/qpsalm_description/cache/small_vision_v1 \
  --description-benchmark benchmark/qpsalm_description_v2_small \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --segmentation-vision-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_description/cache/small_vision_v1_m3v3
```

Description cache 的公开 v1 format/protocol 和 segmentation cache v3 均未改变。迁移或新建后可
只读复核 cache，不重新编码视觉特征：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc cache verify \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --description-benchmark benchmark/qpsalm_description_v2_small \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --segmentation-vision-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_description/cache/small_vision_v1_m3v3
```

## Segmentation-Grounded Description M3-M7

统一入口为 `python -m qpsalm_seg.cli.segdesc`（editable 安装后也可使用
`qpsalm-segdesc`）；旧 `python -m qpsalm_seg.cli.*` 命令仅作为薄转发保留。配置使用
`qpsalm_segdesc_config_v2`，D-1、D0-D4 的数据源、region-token 路由、初始化角色和人工门禁由
`StageSpec` 固定。公开状态、MGRR、双 Adapter、输出 schema 和 artifact replay 的设计依据统一见
[docs/benchmark_GAR.md](docs/benchmark_GAR.md)；本 README 只维护可运行命令及其准入条件。

以下命令均从仓库根目录手动运行。`--resume` 只用于同一 stage、同一 run，会恢复优化器、
scheduler、RNG、数据游标和历史；`--initialize-from` 只用于进入下一 stage，并重置优化状态。
训练顺序固定为 `D0 -> D1 -> D2 -> D3a -> D3b -> D4`，不得跳级或用 resume 跨 stage。
D3a 工程训练消费 `terminal_last`，具有正式 validation 的迁移与评价消费
`validation_best`；复制或重命名 checkpoint 不会改变角色。

每个输出目录只属于一个 run，且不得位于 config、benchmark、cache、checkpoint、prediction
index 或 gate 输入内部。所有训练和正式评价显式传入 `--seed`；更换 seed 必须从对应 D0
重建完整链。成功运行以原子发布的 `training_report.json` 和绑定 checkpoint 为准，失败运行以
`failure_report.json` 为准；只有启动 manifest 的目录不能视为完成。完整的 resume reconciliation、
selection-role 和 completion replay 字段由 GAR 定义，运行时会严格重验。

M2 未冻结前，只允许继续 M3–M5、D-1、D0–D2 和 D3a 等不依赖专家真值的工程路径。
D3b、正式 M4、D4、M6 专家验收和 M7 最终验收必须等待真实 Bridge 冻结；candidate 或
auto-only 文本不得作为 expert val/test。

M4 使用 `qpsalm_mgrr_v2_multiscale_grid_replay`。其多尺度 replay、component/residual、跨 view
坐标映射和六种 region encoder 的精确定义见 GAR；旧实验性 SegDesc checkpoint 不兼容，
但现有 segmentation checkpoint 和两类 vision cache 不因此重建。

先发布只含 component 引用、hash 和精确 JSONL 行号的统一索引。它不复制 M1/M2 图片或 mask，
并绑定三个 component validation report。v3 publication contract 还会独立核对 Landslide V2
final report 及其 instruction validation、Description M1.1 v4 和当前 Bridge M2 v7；旧 Bridge
v4-v6 即使 `errors=[]` 也会被拒绝，
因此旧 unified v2 产物必须重建。Bridge 尚在专家审核时只发布自动描述；目录中残留的
expert index 或旧 gate 会被明确忽略。只有冻结后的 v2 evaluation gate 才能启用专家监督：

```bash
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_5_build_segdesc_dataset.sh small
```

### D-1 基线与过拟合

首次运行 D-1 前，推荐用一个显式确认的手工批次重建 Bridge v7、发布 auto-only Unified v3，
并把旧 M3 v2 cache 严格 side-by-side 迁移为 M3 v3 后做只读深验证：

```bash
BRIDGE_REBUILD_CONFIRM=overwrite_auto_only_v7 \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_segdesc_artifact_acceptance.sh small
```

该批次不会运行训练或专家 merge，也不会修改旧 M3 cache、原始 datasets 或 segmentation cache。
批次显式使用完整 population（不会继承外层 shell 的 `MAX_SAMPLES`/`DRY_RUN`）；确定性种子只由
`ARTIFACT_SEED` 控制，默认 42。
它只允许覆盖尚未冻结专家数据的 auto-only Bridge/Unified；若发现 completed review、pending/
expert/gate merge 产物、额外人工文件或被原地填写的 reviewer template，会在覆盖前拒绝执行。
若严格 migration 检测到 parent、视觉内容、
source cache 或 shard 漂移会立即停止，并明确要求改用上文 `cache build` 完整重建，而不会偷偷复制或
接受旧格式。最后的
`outputs/qpsalm_description/readiness/small_artifact_readiness.json` 会重新打开当前 Description v4、
Bridge v7、Unified v3、M3 origin 和全部 shards，重放 component/report/index binding。严格迁移
成功时还会重放旧/新 cache 的来源哈希、inode 与 segmentation cache v3 snapshot；若按提示完整重建，则只接受无 migration metadata
的当前原生 M3 v3 builder artifact。只有
该报告同时满足 `status=engineering-valid`、`ready=true`、`errors=[]` 才继续 D-1。Bridge 的状态仍以
报告中的实际值为准：`awaiting_expert_review` 只允许 auto-only，不能解释为专家审核完成。
如果默认 migration 明确要求完整重建，先执行上文 `cache build`；构建成功后用同一批次只读接管该
原生 v3 cache（不会再次迁移或覆盖它）：

```bash
BRIDGE_REBUILD_CONFIRM=overwrite_auto_only_v7 \
CACHE_ACTION=verify-existing \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_segdesc_artifact_acceptance.sh small
```

原生 Qwen3-VL 全图描述 zero-shot 基线：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc evaluate zero-shot \
  --model models_zoo/Qwen3-VL-2B-Instruct \
  --benchmark benchmark/qpsalm_description_v2_small \
  --split dev --seed 42 --device cuda \
  --output-dir outputs/qpsalm_description/d_minus_1_zero_shot_dev \
  --overwrite-output
```

固定 64 条混合样本过拟合，用于检查 `desc_adapter`、MGRR、causal labels 和 checkpoint：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc train d-minus-one \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --seed 42 --region-encoder mgrr --device cuda --batch-size 2 \
  --artifact-readiness-report \
    outputs/qpsalm_description/readiness/small_artifact_readiness.json \
  --max-val-samples 64 \
  --val-interval 25 --save-interval 50 \
  --output-dir outputs/qpsalm_description/d_minus_1_overfit_structured_v2_seed42 \
  --overwrite-output
```

该 run 使用确定性的四路等额混合：M1.1 full-image global、M1.1 DIOR box、Bridge
mask candidate、Bridge null candidate；Bridge candidate 明确不是 expert truth。当前 D-1
要求 M2 已用当前 `landslide_bridge_m2_v7_expert_review_replay_bound` 至少重建到
`awaiting_expert_review`，不要求伪造或提前完成专家审核。训练结束检查
`d_minus_one_overfit_validation.json`；当前验收会哈希绑定 history、resolved config、dataset
summary、gradient gate、trainable manifest、validation/raw generation、checkpoint、artifact
readiness report 和显式
segmentation migration，并独立绑定 ontology、record schema 与 output schema；后续 gate 会按
当前仓库字节级重验三份协议资产。当前 overfit report 为
`qpsalm_d_minus_one_overfit_validation_v10_structured_decoder_bound`，还会从 checkpoint 本体重放
format/step/metadata、segmentation migration 和精确 M3 cache 的全部 shard。reload 会先破坏一个
`desc_adapter` LoRA 哨兵，并分别扰动 optimizer、scheduler、训练 RNG（以及启用时的
GradScaler）状态，再通过严格 loader 恢复并核对各自状态指纹；24 GiB 门禁要求真实 CUDA 记录和正的峰值，CPU 产生的
`0 GiB` 不再算通过。每个 causal batch 还会实测 prefix/padding mask、target 连续性与监督 EOS，
把五项证据写入 history；`qpsalm_description_gradient_gate_v4_window_homogeneous` 使用
`qpsalm_d_minus_one_task_path_batch_sampler_v1_window_homogeneous` 保证每个 accumulation window
只含 global 或 region 一条视觉路由。纯 global 窗口要求区域模块为零，region 窗口要求
MGRR/spatial backbone/region projector 非零有限，并在两条路径都被真实观察后才完成子门禁。
Structured route 使用 `qpsalm_description_structured_generation_v2_token_stream_bound`：
decoder 固定 JSON 语法和 schema key，Qwen 从 live logits 选择 enum 与自由文本 token；最终发布值
是 decoder 的原始输出，不读取 GT、不调用 deterministic repair。每条 structured generation 都
记录 forced/model-selected token 数、字段终止原因、raw SHA 与实际 causal token-stream SHA；
只有逐字节相等才允许发布，门禁会逐行重放。
该 decoder protocol 已进入 checkpoint architecture binding；缺少此字段的旧 D-1 checkpoint/report
不能升级或 resume，应保留原目录作为失败证据，并从 segmentation checkpoint 在新目录重跑 100-step。
统一门禁从绑定 history 和 gradient artifact 重新计算，并逐 path 核对 module inventory、梯度计数、
有限 norm 与 checks 全集；只有顶层 `passed=true` 的旧/手写文件不能替代该证据。
第 100 步 `checkpoint_last.pt` 还嵌入相同的完整 gradient proof，并拒绝在路径证据未齐时保存。
如果进程恰好在 terminal checkpoint 成功写入、strict reload 或 `training_report.json` 发布前中断，
可对同目录 terminal checkpoint 使用原参数 `--resume`；入口会恢复 checkpoint 内 proof 和已落盘的
final validation，只补做严格 reload 与终态报告，不额外执行 optimizer step。
旧 overfit report 不兼容，需重跑该短 run。该报告不会单独声称
D-1 完成。

zero-shot 与过拟合都完成后运行只读统一门禁：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc validate d-minus-one \
  --zero-shot-dir outputs/qpsalm_description/d_minus_1_zero_shot_dev \
  --overfit-dir outputs/qpsalm_description/d_minus_1_overfit_structured_v2_seed42 \
  --output outputs/qpsalm_description/d_minus_1_gate_structured_v2_seed42.json
```

统一门禁使用 `qpsalm_d_minus_one_engineering_gate_v13_structured_decoder_bound`，会从当前 M1.1
索引重新选择 zero-shot population，并逐张重开所选 M1.1 `data/` 图像，要求路径属于当前
benchmark、`storage_mode=materialized_copy` 且 live SHA 与 `visual_ref.sha256` 一致；同时逐文件重验
Qwen 权重、tokenizer、配置字节，及 checkpoint payload、Description cache 和上述 overfit 源文件。它还要求 overfit run 已原子
发布当前 `training_report.json`，深度重放其中每个 artifact binding，并证明 `checkpoint_last.pt`
为 `terminal_last`、checkpoint step 与 progress/history 终点一致、history 严格递增且
`d_minus_one_overfit_validation.json` 正是该完成报告绑定的文件；缺少完成报告、残留
`failure_report.json` 或只有中途生成的 overfit report 均不能通过。只有输出同时为
`status=engineering-valid`、`d_minus_one_complete=true` 和
`errors=[]`，D-1 工程门禁才通过；这仍不替代 M2 专家冻结、D3b/D4 或正式科学评价。

### D0-D3 课程训练

D0 使用 MMRS Caption 做遥感场景预适配。先运行 construction-only preflight；它会重验 D-1、
benchmark、M3 全部 shard、segmentation migration，并构建 model/dataset/collator/optimizer 与
trainable manifest，但保持 `optimizer_steps=0`：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage mmrs_caption --seed 42 --device cuda \
  --d-minus-one-gate outputs/qpsalm_description/d_minus_1_gate_structured_v2_seed42.json \
  --output-dir outputs/qpsalm_description/d0_preflight_seed42 \
  --formal-output-dir outputs/qpsalm_description/d0_mmrs_seed42 \
  --preflight-only --overwrite-output
```

只有当前 `qpsalm_d0_preflight_v6_region_route_bound` 的 `preflight_report.json` 同时满足
`status=engineering-valid`、`ready=true`、`errors=[]`，并且
`formal_training_launch.unique=true`、其 resolved config SHA 仍匹配时，才运行报告字段
`formal_training_launch.command` 中发布的唯一正式 D0 命令；不要从上面的 preflight 参数手工重构
另一条命令，也不要手工追加 `--overwrite-output` 或 `--initialize-from`；D0 只能从已绑定的
segmentation checkpoint 新建，正式入口会在读取 artifact 或创建 run 目录前拒绝这些旁路。
预检拒绝非空的正式 output-dir，发布的正式命令也不含 `--overwrite-output`，因此不会
删除既有 D0 run；若预检后目录被其他进程占用，正式启动会安全失败并要求重新预检。
preflight 和正式入口都按 `seed + 11003` 重建同一个 D0 sampler，并以
`qpsalm_description_collator_audit_v3_output_format_region_route_separated` 核对首批真实 training batch 的
request、instruction/target/reference、structured flag、region-token route、metadata、mask、weight 及完整 stream
binding；审计器不使用测试专属的影子字段。正式命令必须携带 `--d0-preflight-report`；训练入口会按
`qpsalm_d0_training_launch_v2_exact_command_bound` 重建并逐字段比较 argv 与 shell command，再重验报告状态、resolved config SHA、D-1
acceptance、construction device、Description/Bridge binding、cache 元数据快照及其 segmentation
cache v3 source provenance，并在正式目录原子写入
当前 `qpsalm_d0_preflight_acceptance_v6_region_route_consumed` 的
`d0_preflight_acceptance.json`。正式 trainer 还会在首个 optimizer step 前按
`qpsalm_d0_construction_contract_v2_region_route_replayed` 比较实际 segmentation migration、dataset、
collator、stream binding、trainable manifest 和 optimizer spec。省略该参数或在预检后修改输入均不能启动 D0。

D0 不接受只显示 overfit success 的单个报告；它会深度重算上述统一 gate。D0 保存
`qpsalm_d_minus_one_acceptance_v11_structured_decoder_bound`，把 M1.1 benchmark root、builder、
validation SHA、zero-shot materialized-image population SHA 与 overfit training completion
report SHA 一并固化。D1–D4 初始化、M7
初始化/续训和正式 retention 都继续按原 gate 路径、
SHA 与当前 `description_benchmark` 重验；任一 zero-shot、overfit、checkpoint、输入 benchmark
漂移或中途切换另一份 M1.1 root 都会使后续链失效。

当前 Small 已完成上述 D0 闭环：

```text
preflight: outputs/qpsalm_description/d0_preflight_seed42/preflight_report.json
run:       outputs/qpsalm_description/d0_mmrs_seed42
status:    terminal_status=completed, steps=1000, best checkpoint step=1000
```

`training_report.json` 使用
`qpsalm_description_training_completion_v3_checkpoint_replayed`，绑定 best/last checkpoint、
D0 preflight acceptance、dataset/config/history/progress/trainable manifest 和 validation-best；
目录中没有 `failure_report.json`。因此后续不得用同名新 run 覆盖它，D1 必须通过下文
`--initialize-from .../checkpoint_best.pt` 建立新的 optimizer 与 stage lineage。

每次 D0–D4 initialize/resume 还会重算
`qpsalm_segmentation_migration_lineage_v1_source_bytes_bound`，要求当前配置与源 SegDesc
checkpoint 都指向同一份原始 segmentation checkpoint SHA/format/step/白名单，并重新读取可访问的
source bytes。不能在后续 stage 更换分割基线后继续沿用旧 lineage。
统一 `qpsalm_segdesc_v1` checkpoint 还逐文件绑定 ontology、训练 record schema 和输出 schema；
resume/initialize 时会对当前文件重新计算 SHA。输出 schema 或 ontology 改动后，旧实验性描述
checkpoint 必须重训，不能用宽松加载掩盖协议变化。
正式 M4/D4/M6/M7 门禁还通过
`qpsalm_segdesc_checkpoint_provenance_v3_segmentation_lineage_bound` 以内存映射方式直接重放 checkpoint 本体，核对
`checkpoint_step`、完整非 tensor metadata、Adapter 清单、state-key inventory 和当前协议资产；
并从 metadata 中的 artifact binding 重放精确 Description cache。只改评估 JSON 中复制出的
stage/lineage、替换 cache shard，或把报告指向另一个同名 checkpoint，均不能通过。

D0/D1 只训练 `desc_adapter`、task-neutral visual projection 和 instruction/visual special
embeddings；全图 caption 序列不注入 MGRR region tokens。MGRR、description spatial backbone、
region projector 和 region special embedding 从 D2 才进入训练，跨 stage 必须使用
`--initialize-from` 重建 optimizer。D0/D1 还会跳过四尺度 spatial cache 投影；实际参数集合写入
运行目录的 `trainable_parameter_manifest.json`。该路径使用
`qpsalm_description_causal_v5_stage_separated_schema_ordered`；训练 target 与受约束生成均按
固定 schema 顺序序列化，旧的 v4 及更早实验描述 checkpoint 需要舍弃并从 D0
重训，现有分割 checkpoint 和 vision cache 不需要重建。
每个 caption parent 每个 epoch 只选择一条 reference；选择由
`caption_quality_weight` 确定性加权。D0 在 MMRS source 间平衡总 loss mass，D1 固定保持
RSICap/MMRS=70/30 后再在各组内部平衡 source。精确 source counts、总权重和均值写入
`dataset_summary.json`；不会通过丢弃大 source 的 parent 来制造平衡。

D1 使用 RSICap 校准详细描述，并按配置回放 30% MMRS：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage rsicap_caption --seed 42 --device cuda \
  --initialize-from outputs/qpsalm_description/d0_mmrs_seed42/checkpoint_best.pt \
  --output-dir outputs/qpsalm_description/d1_rsicap_seed42 --overwrite-output
```

D1 checkpoint 的 RSIEval test 先冻结完整 generation；该 test 只作独立评价，不能参与早停、
prompt 调参或 checkpoint 选择：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage rsicap_caption --seed 42 \
  --checkpoint outputs/qpsalm_description/d1_rsicap_seed42/checkpoint_best.pt \
  --split test --source-dataset RSIEval \
  --evaluation-mode gt_mask --no-counterfactuals \
  --max-val-samples 0 --max-generate-samples 0 --device cuda \
  --output-dir outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test \
  --overwrite-output
```

自动 caption 指标是可选的正式后处理依赖。安装后，使用一个已经下载到本机的显式 encoder
目录，并根据其架构填写输出层号；BERTScore 路径不会联网下载或按模型名称猜测层号：

```bash
pip install -e 'SEG_Multi-Source_Landslides[caption-eval]'

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_caption_metrics \
  --eval-dir outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test \
  --bertscore-model models_zoo/bert_score_encoder \
  --bertscore-num-layers 12 --bertscore-batch-size 16 --device cuda \
  --output outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test/caption_metrics.json \
  --overwrite-output
```

`qpsalm_rsieval_caption_metrics_v1_official_backends` 只接受从 DataLoader 起通过
`qpsalm_description_evaluation_source_filter_v1` 冻结的 100 个唯一 RSIEval test parent、
完整 generation 以及与当前 population identity fields 一致的 eval report。它绑定 report、raw
generation、参考文本和本地 BERTScore 权重/配置/tokenizer 哈希，使用 pycocoevalcap 计算
BLEU-1..4、METEOR、ROUGE-L、CIDEr、SPICE，并对每个 parent 取最佳 reference BERTScore-F1；
每项同时报告 corpus score、parent macro 和 10,000 次 bootstrap 95% CI。缺少 Python 依赖、
Java、模型权重或输入绑定不一致时直接失败，不使用近似实现。所有这些仍是次要语言指标，不能
替代人工 caption 事实性/详细度/可读性或区域 grounding 指标。
`pycocoevalcap` 的 SPICE 后端首次使用时可能按上游行为准备 Stanford CoreNLP 资源；请在正式
离线运行前显式完成该准备。最终报告会绑定 Java executable、五个 scorer 源文件和全部 JAR
资源哈希，不能把资源不同的 run 当作同协议比较。

同一冻结 generation 的人工三维评价先生成一份 blind 模板，再分别复制给两名 reviewer 独立
填写 `reviewer_id` 和三个 `scores`（整数 1–5）；模板不包含 reference caption：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_caption_human_review \
  --eval-dir outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test \
  --write-template \
  --output outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test/caption_human_review_template.jsonl
```

两人审核完成后再汇总，不能由同一 reviewer ID 提交两份文件：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_caption_human_review \
  --eval-dir outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test \
  --review /path/to/reviewer_1.jsonl \
  --review /path/to/reviewer_2.jsonl \
  --seed 42 \
  --output outputs/qpsalm_description/d1_rsicap_seed42/eval_rsieval_test/caption_human_review_report.json
```

审核文件绑定 100 张 materialized RSIEval 图像及其 SHA、raw generation、eval report 和完整
population；改写模型文本、替换图像、漏评、重复 reviewer 或越界分数都会失败。报告对事实性、
详细度、可读性分别给出 parent macro、10,000 次 bootstrap 95% CI、exact/within-one agreement
和 quadratic weighted kappa。完成审核只表示得到了人工测量，不构成 test-set checkpoint 选择。

D2 只做 DIOR 同图候选区域对齐；batch 必须至少为 2：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage dior_alignment --seed 42 --batch-size 4 --device cuda \
  --initialize-from outputs/qpsalm_description/d1_rsicap_seed42/checkpoint_best.pt \
  --output-dir outputs/qpsalm_description/d2_dior_seed42 --overwrite-output
```

同一 parent 内文本完全相同的区域使用 multi-positive contrastive target，不会互相充当
假负样本；训练 loss 与 same-image R@1 采用同一正样本定义。D2 训练使用 parent-grouped batch
sampler，使同图不同区域稳定成为 hard negatives，而不是依赖普通随机 batch 偶然相遇。

D3a 使用全部合法 train mask 和规则化结构事实。该 stage 没有人工 val，因此后续初始化使用
`checkpoint_last.pt`，不能按自动 candidate 指标选择“科学 best”。训练器把该终点角色写为
`checkpoint_role=terminal_last`，D3b 按内部元数据强制核验，因此重命名或复制一个 best/未声明
角色的 checkpoint 也不能绕过。启动时会以
`landslide_bridge_engineering_audit_v2_cache_candidate_projection_bound` 重读当前 M2 validation、
`candidate_all.jsonl` 与 `auto_train.jsonl`，要求当前 builder、完整 Pilot、零错误、精确 train
投影以及 `is_expert_truth=false`，并要求 live candidate 的路径、字节数和 SHA 与当前 Description
Vision Cache 的 `multisource_parent` 输入一致；audit 会进入 checkpoint data binding：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_auto --seed 42 --region-protocol vision_only --region-encoder mgrr --device cuda \
  --initialize-from outputs/qpsalm_description/d2_dior_seed42/checkpoint_best.pt \
  --output-dir outputs/qpsalm_description/d3a_bridge_auto_seed42 --overwrite-output
```

只有 M2 双人审核、仲裁和 gate 冻结后才能运行 D3b。D3b 使用独立 Bridge、DIOR 和 global-caption
DataLoader，默认按 3:1:1 交替：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert --seed 42 --region-protocol vision_only --region-encoder mgrr --device cuda \
  --initialize-from outputs/qpsalm_description/d3a_bridge_auto_seed42/checkpoint_last.pt \
  --output-dir outputs/qpsalm_description/d3b_bridge_expert_seed42 --overwrite-output
```

将 `--region-encoder` 分别设为 `crop_only`、`full_image_box`、`masked_pooling`、
`roi_replay_only`、`mgrr_no_context`、`mgrr`。六条链必须从同一个 D1 checkpoint 分叉，并各自
按 D2 -> D3a -> D3b 顺序训练；D2 已首次训练 region encoder，因此不能先只训练 MGRR D2、再在
D3a 临时替换 baseline encoder。正式 M4 gate 会核验 D0-D3 stage lineage、共同 D1 checkpoint、
相同数据 population/训练预算，以及 baseline 对 full MGRR 的配对身份。Assisted 和 Vision-only
也必须分开训练、分开报告。

Vision-only 的 MGRR 不接收确定性面积、中心或形状答案；`full_image_box` 对照只保留协议允许的
归一化 box 坐标与存在标记。Assisted 给六种 encoder 注入同一份完整连续几何，避免把输入协议
差异误报为 MGRR 增益。`full_image_box` 即使面对 null box 仍读取完整有效视野，并用零坐标编码
no-box；它不会退化成 learned null-only。六种 encoder 均保持从 region sequence 到 SANE
feature 的梯度链，零有效覆盖模态由显式 null evidence 替换。

`bridge_expert`、`predicted_mask` 和 M7 region loader 会重新核验 Bridge validation status、
人工冻结 v2 gate 及其 Pilot/selection/candidate hashes。目录中残留的 `expert_*.jsonl` 不足以
解锁这些 stage；expert row 缺少人工 target 时也不会回退到 candidate。

### GT、固定预测与端到端评价

GT-mask oracle：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert --seed 42 \
  --evaluation-mode gt_mask --region-protocol vision_only --region-encoder mgrr \
  --checkpoint outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --split val --device cuda \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --cycle-localization-samples 128 \
  --output-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --overwrite-output
```

这条未过滤的 GT-mask 命令用于 M4/MGRR 与 D3b→D4 gate，保留全部 expert region 类型。M6
三模式可比性另用下文的 `gt_global_mask` population，不能用其中一个目录冒充另一个协议。

该 GT-mask Vision-only 命令同时产生 `cycle_localization.jsonl` 和报告中的
`cycle_localization`。它把未修复 raw generation 作为 frozen segmentation `default` adapter
的 grounding prompt，恢复并重投影预测 mask 后计算 parent-macro region IoU。该结果仅是辅助
自一致性指标；Assisted、fixed-prediction、end-to-end、未冻结 Bridge 或训练期 monitor 均禁止
启用。`-1` 关闭，`0` 表示覆盖全部可定位 expert rows，正整数表示显式上限。

界面按 `--evaluation-mode` 使用真实 GT、固定预测或在线 segmentation mask；`end_to_end`
会复用正式 evaluator 的 Bridge-to-segmentation target resolver，并在 overlay/diagnostics 中显示
segmentation sample、mapping kind、threshold 与 resize transform。`full/zero` 仅作为显式
反事实覆盖，不能再把 GT overlay 冒充端到端输入。

固定 val prediction 先由同一冻结分割 checkpoint 离线导出：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.export_predicted_regions \
  --segmentation-config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --source-index benchmark/landslide_region_description_v1_small/indexes/expert_val.jsonl \
  --split val \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_description/predicted_val --device cuda --overwrite-output
```

固定预测评价必须同时使用 `stage=predicted_mask` 和 `evaluation_mode=fixed_prediction`：
评估器会核对 predicted-index 发布报告中的 segmentation checkpoint SHA 与 D4 description
checkpoint 的 `segmentation_migration.source_sha256`，因此不能用另一分割模型生成的 mask 混报。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage predicted_mask --seed 42 \
  --checkpoint outputs/qpsalm_description/d4_predicted_75_seed42/checkpoint_best.pt \
  --split val --evaluation-mode fixed_prediction \
  --predicted-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --device cuda --output-dir outputs/qpsalm_description/d4_predicted_75_seed42/eval_fixed_val \
  --overwrite-output
```

端到端使用 `--stage bridge_expert --evaluation-mode end_to_end`。评估器按 Bridge region 身份
严格选择 segmentation instruction：`gt_global_mask` 使用全图指令，`gt_referring_mask`、
no-target 以及带 referring alias 的伪 component 使用对应的 referring/no-target 指令。
没有语言可识别 alias 的纯 `pseudo_instance_component` 不会回退为全图 mask，而是从端到端集合
排除并记录在 `end_to_end_coverage.excluded_by_reason`。逐条映射写入
`end_to_end_target_audit.jsonl`；默认以 `segmentation_mask_threshold=0.5` 二值化分割输出。
三种评价必须生成独立目录，不能混合统计。原始生成保存在
`raw_generations.jsonl`，主指标只读取未修复 JSON；deterministic repair 仅作错误分析。
Raw structured output 必须是文件内容中的唯一 JSON object；Markdown fence、前后说明文字或
截取出的花括号只允许进入 repair diagnostics，不能提高 raw parse/schema 指标。`summary` 必须
为非空字符串，并独立报告 raw summary token-F1、exact match 和 non-empty rate；正式事实性审核
仍以人工 summary rubric 为准。Python 解码器扩展接受的 `NaN`、`Infinity` 和 `-Infinity` 不属于
JSON 标准，raw parser 与 repair 提取都必须拒绝；非有限 `confidence` 不能绕过 `[0,1]` schema
约束。description 训练/评估侧 JSON 与 JSONL 发布同样使用 `allow_nan=False`，编码失败时保留原有
原子产物不变；M3–M7 loader 和正式 gate 重读外部 artifact 时也拒绝这些非标准数值。
SegDesc checkpoint 的可发布 non-tensor metadata 使用同一有限性边界；首次有效 validation 之前
的内部 best-score 哨兵保存为 `null`，resume 时再从已绑定的 best report 恢复。
配置解析也会在模型或 optimizer 构建前拒绝非有限超参数及非法负计数。
`target_status=absent` 时 output schema 还强制六个 `region`
字段全为 `unavailable`，并禁止模态 support/sufficiency 给出肯定结论；继续输出
location/shape 或 `supports/sufficient` 等滑坡属性会使整条 raw output 落入
schema-invalid/status-invalid，不能获得 absent recall。deterministic repair 只按该 schema 条件
清空区域/肯定证据属性并单独记录 action，不改变 raw 主指标。结构合法但仍产生 unsupported
evidence claim 的 absent 样本也计入 false-description rate。
独立 `eval_description` 默认完整评估并完整生成；只有 smoke 才显式传正整数上限。
端到端正式命令为：

这里 `stage=bridge_expert` 表示评价数据来自冻结 expert Bridge，而 checkpoint 必须明确为
`stage=predicted_mask` 的 D4 权重；报告分别记录 `evaluation_data_stage` 与 `checkpoint_stage`。
其他评价模式要求两者 stage 相同。当前运行所加载的 segmentation source SHA 还必须与
checkpoint 保存的 migration lineage 一致。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert --seed 42 \
  --checkpoint outputs/qpsalm_description/d4_predicted_75_seed42/checkpoint_best.pt \
  --split val --evaluation-mode end_to_end --region-source gt_global_mask \
  --region-encoder mgrr --device cuda \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --output-dir outputs/qpsalm_description/d4_predicted_75_seed42/eval_end_to_end_val \
  --overwrite-output
```

端到端 mask 会先按 segmentation resize transform 恢复到原图；若 M3 复用了经过 size bucket
的 segmentation cache，则按 nearest/full-extent 显式映射到 reference view 的 render-source
canvas，最后再执行 Description Cache render transform。禁止直接在两个 padded canvas 之间插值，
源/目标尺寸与映射协议会写入 region-input source binding 供正式 gate 重放。

M6 的 GT oracle 必须另建与 fixed/end-to-end 完全一致的一-parent-one-row population；它只选择
人工冻结 Bridge 中的 `gt_global_mask`，并对全部选中 parent 完成 cycle localization：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert --seed 42 \
  --evaluation-mode gt_mask --region-source gt_global_mask \
  --region-protocol vision_only --region-encoder mgrr \
  --checkpoint outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --split val --device cuda \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --cycle-localization-samples 0 \
  --output-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_global_val \
  --overwrite-output
```

`region_source_filter_audit` 会保存过滤前后数量和精确 population SHA。正式 M6 gate 还会逐行
确认 raw generation 的 `region_source=gt_global_mask`；仅手工补一个 audit 字段不能通过。

### D4 Out-of-Fold predicted-mask curriculum

先建立 parent-level 三折索引。OOF v3 只接受内容全部为 `split=train` 的 segmentation index：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.build_oof_folds \
  --segmentation-index benchmark/multisource_landslide_v2_small/indexes/instruction_train.jsonl \
  --bridge-index benchmark/landslide_region_description_v1_small/indexes/expert_train.jsonl \
  --num-folds 3 --seed 42 \
  --output-dir outputs/qpsalm_description/oof_folds_small --overwrite-output
```

对 fold 0、1、2 分别构建与其 train/holdout 指纹绑定的 segmentation Vision Cache v3，随后从头
训练一个排除该 fold 的 segmentation checkpoint。以 fold 0 为例：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --train-index outputs/qpsalm_description/oof_folds_small/fold_0_train.jsonl \
  --val-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --output-dir outputs/qpsalm_description/oof_folds_small/cache_fold_0 \
  --device cuda --backend qwen --overwrite

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda \
  --train-index outputs/qpsalm_description/oof_folds_small/fold_0_train.jsonl \
  --val-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --vision-feature-cache outputs/qpsalm_description/oof_folds_small/cache_fold_0 \
  --output-dir outputs/qpsalm_description/oof_folds_small/seg_fold_0 \
  --overwrite-output --skip-torch-preflight
```

用 fold-specific checkpoint 只预测对应 holdout。导出器会同时核验 checkpoint 中的
`config.train_index`、fold train hash 和 prediction holdout hash：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.export_predicted_regions \
  --segmentation-config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_description/oof_folds_small/seg_fold_0/checkpoint_best.pt \
  --source-index benchmark/landslide_region_description_v1_small/indexes/expert_train.jsonl \
  --split train --checkpoint-fold 0 \
  --fold-manifest outputs/qpsalm_description/oof_folds_small/fold_manifest.json \
  --train-index outputs/qpsalm_description/oof_folds_small/fold_0_train.jsonl \
  --val-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --prediction-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --vision-feature-cache outputs/qpsalm_description/oof_folds_small/cache_fold_0 \
  --output-dir outputs/qpsalm_description/predicted_fold_0 --device cuda \
  --overwrite-output
```

三个 fold 全部导出后合并；缺失、重复、in-fold、错误 checkpoint、mask hash/shape 错误或
fold 覆盖不完整都会直接失败：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.merge_oof_predictions \
  --fold-manifest outputs/qpsalm_description/oof_folds_small/fold_manifest.json \
  --input outputs/qpsalm_description/predicted_fold_0/predicted_train_0.jsonl \
  --input outputs/qpsalm_description/predicted_fold_1/predicted_train_1.jsonl \
  --input outputs/qpsalm_description/predicted_fold_2/predicted_train_2.jsonl \
  --output outputs/qpsalm_description/predicted_train_oof.jsonl
```

当前 fold 协议为 `qpsalm_segmentation_oof_folds_v3_source_partition_replayed`。加载器会从当前
frozen `expert_train` 重算分层 parent assignment，并把每个 train/holdout JSONL 与源 segmentation
rows 做有序精确分区比较；同步篡改文件和 manifest hash 也不能隐藏 held-out parent 泄漏。
`qpsalm_predicted_region_oof_merge_v4_exact_fold_publications_replayed` 还会重新打开每个 segmentation
checkpoint，核验其 `config.train_index`/`config.val_index`、Vision Cache v3 train/val index
fingerprint、expert source row、fold manifest 以及每个 mask 的 hash/shape/binary 内容。D4/M6/M7
消费保存的 audit 时会再次执行同一重放；每个 fold 的 mask 必须精确位于
`masks/train/<parent>.npy`，且 fold 目录不得残留 stale/`.part` 文件。fixed val/test 使用
`qpsalm_fixed_predicted_region_artifact_v3_exact_mask_directory_bound`，要求完整覆盖当前 expert
split 的 `gt_global_mask` parents，并逐行重验 segmentation checkpoint、专家 target 和 mask；
每个 mask 必须位于 `masks/<split>/<parent>.npy`，目录不得有 stale 或 `.part` 文件。
旧的 D4 fold/prediction 中间产物不兼容，必须从 fold 构建或 predicted export 阶段重建，但不需要
修改原始 dataset。
OOF fold 与 prediction CLI 采用单次运行目录；覆盖前会解析 checkpoint、cache、Bridge/fold/index
输入的真实路径。任一输入位于待删除目录内时拒绝覆盖，OOF merge 也拒绝把输出文件写回任一
fold input 或 manifest，避免在生成审计产物时破坏后续需要重放的源证据。

D4 不能仅凭训练结束自动升档。先对 D3b checkpoint 完成上文的全量 Vision-only GT-mask val
评价和双人 ERFS 汇总，再发布 `0 -> 0.25` gate：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.validate_d4_curriculum \
  --eval-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --expert-report outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val/expert_factuality_report.json \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --m4-suite-gate outputs/qpsalm_description/m4_region_encoder_suite_gate.json \
  --current-fraction 0 --next-fraction 0.25 --seed 42 \
  --output outputs/qpsalm_description/d3b_bridge_expert_seed42/d4_to_25_gate.json
```

只有 gate 返回 `passed=true` 才能从 D3b 权重开始 25% 档；其余仍为 expert GT regions：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage predicted_mask --seed 42 \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --predicted-val-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --d4-curriculum-gate outputs/qpsalm_description/d3b_bridge_expert_seed42/d4_to_25_gate.json \
  --predicted-mask-fraction 0.25 \
  --d4-curriculum-sampling-seed 42 \
  --initialize-from outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --device cuda --output-dir outputs/qpsalm_description/d4_predicted_25_seed42 \
  --overwrite-output
```

`--predicted-index` 必须是 OOF train 发布物，`--predicted-val-index` 必须是独立冻结分割模型生成的
val 发布物；训练器分别校验两份 report/path/hash，禁止用 train-only OOF index 构造空 val，或把
同一 index 同时冒充 train/val。`--predicted-mask-fraction` 只接受预注册的
`0.25/0.50/0.75` 三档；训练数据审计记录请求值、实际 GT/predicted 数量和 population hash，预测
index 数量不足时直接失败。D4 周期 val 始终标记为 `fixed_prediction`，不会把固定预测 mask
误报为 `gt_mask`。

25% checkpoint 必须按上文 fixed-prediction 命令完整生成 val，并用两名 reviewer 汇总 ERFS；
通过后才能发布相邻的 `0.25 -> 0.50` gate：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.validate_d4_curriculum \
  --eval-dir outputs/qpsalm_description/d4_predicted_25_seed42/eval_fixed_val \
  --expert-report outputs/qpsalm_description/d4_predicted_25_seed42/eval_fixed_val/expert_factuality_report.json \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --current-fraction 0.25 --next-fraction 0.50 --seed 42 \
  --output outputs/qpsalm_description/d4_predicted_25_seed42/d4_to_50_gate.json

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage predicted_mask --seed 42 \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --predicted-val-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --d4-curriculum-gate outputs/qpsalm_description/d4_predicted_25_seed42/d4_to_50_gate.json \
  --predicted-mask-fraction 0.50 \
  --d4-curriculum-sampling-seed 42 \
  --initialize-from outputs/qpsalm_description/d4_predicted_25_seed42/checkpoint_best.pt \
  --device cuda --output-dir outputs/qpsalm_description/d4_predicted_50_seed42 \
  --overwrite-output
```

对 50% checkpoint 重复同一套完整 fixed-val generation 和双人 ERFS 后，发布并训练 75% 档：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.validate_d4_curriculum \
  --eval-dir outputs/qpsalm_description/d4_predicted_50_seed42/eval_fixed_val \
  --expert-report outputs/qpsalm_description/d4_predicted_50_seed42/eval_fixed_val/expert_factuality_report.json \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --current-fraction 0.50 --next-fraction 0.75 --seed 42 \
  --output outputs/qpsalm_description/d4_predicted_50_seed42/d4_to_75_gate.json

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage predicted_mask --seed 42 \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --predicted-val-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --d4-curriculum-gate outputs/qpsalm_description/d4_predicted_50_seed42/d4_to_75_gate.json \
  --predicted-mask-fraction 0.75 \
  --d4-curriculum-sampling-seed 42 \
  --initialize-from outputs/qpsalm_description/d4_predicted_50_seed42/checkpoint_best.pt \
  --device cuda --output-dir outputs/qpsalm_description/d4_predicted_75_seed42 \
  --overwrite-output
```

最后对 75% checkpoint 完成全量 fixed-val 和双人 ERFS，并为 M7 发布该 checkpoint 自身的
acceptance gate；`--final-m7` 不能替代前三次相邻升档：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.validate_d4_curriculum \
  --eval-dir outputs/qpsalm_description/d4_predicted_75_seed42/eval_fixed_val \
  --expert-report outputs/qpsalm_description/d4_predicted_75_seed42/eval_fixed_val/expert_factuality_report.json \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --current-fraction 0.75 --final-m7 --seed 42 \
  --output outputs/qpsalm_description/d4_predicted_75_seed42/d4_final_m7_gate.json
```

每个 gate 都先写隐藏候选，再从候选声明的路径重新读取并核验完整 generation population、
checkpoint/seed、未修复 raw output、ERFS、当前 frozen Bridge 阈值、M4 suite 和源 checkpoint 的
region training population；逐字段重建完全一致后才原子发布。`passed=false` 是可保留的科学结果，
但仅手工把 JSON 中的 `passed` 改为 `true` 无法解锁训练。三个模型 seed 都固定
`--d4-curriculum-sampling-seed 42`，使 predicted-row 子集保持一致；模型 seed 只改变优化顺序和
随机性，不能同时改变训练 parent population。

完成 GT-global、fixed-prediction、end-to-end 三个独立目录的双人盲审 ERFS 后，发布统一 M6
acceptance。三者必须是同一个 expert parent population；GT/end-to-end 必须绑定
`gt_global_mask` filter，fixed/end-to-end 必须使用同一个 D4 75% checkpoint。Gate 会重算三份
绝对事实性阈值、五种反事实的 parent-level CI/最低有效 parent、D-1 与完整 stage lineage、M4/D4
acceptance、cycle localization 和在线 target mapping：

当前 M6 gate 为 `qpsalm_m6_acceptance_v10_strict_json_finite`。它不只绑定
`cycle_localization.jsonl` 与 `end_to_end_target_audit.jsonl`：还冻结实际使用的 segmentation
instruction index、task-family filter 和过滤后 population，并从该源逐条重放 target mapping、
raw-text SHA。当前 evaluator 还会把每条实际送入 descriptor 的 region mask，以及 cycle 的
source-space prediction、descriptor valid mask 和应用 valid mask 后的 prediction/target，原子保存为
role/sample-bound 二值 NPY；正式 gate 逐文件重开，重放 source→cache 投影与 valid-mask 应用，再从
像素重算 area、intersection、union 与 IoU，不再信任 JSON 中彼此自洽的计数。GT/fixed 还从当前
Bridge/predicted source NPY 重新投影，并逐条打开 checkpoint 绑定的 M3 shard record，核对 lookup
key、cache fingerprint 与 reference-view render transform 后再逐像素核对 descriptor
输入；cycle valid 还必须等于该 record 全部 view-valid masks 的 union。end-to-end 另存恢复到
source space 的在线预测，再重放到 cache canvas。三种评价还会从实际 checkpoint 本体重放 step、metadata、stage
lineage、ontology/schema assets，以及精确 M3 manifest/validation/shard 内容；fixed 模式还会从
保存的 audit 重放完整 fixed prediction 发布物，并核对其 segmentation checkpoint 正是 description
checkpoint 的 migration source。只改写 eval report 内嵌 metadata 或替换 cache/prediction 的伪证据
会失败。每条反事实
还必须绑定真实输入前后指纹并从 raw generation 重算 delta。协议资产变化后必须重新评价和审核。
旧 v1-v7 gate/audit 不兼容，不能通过改写汇总 JSON 升级。
发布 CLI 先写隐藏候选，并从候选声明的三模式路径重建完整 gate 后才原子替换最终文件。
若证据链有效但任一科学阈值未通过，仍发布可重放的 `passed=false` 报告并返回非零；M7 授权
validator 会单独要求 `passed=true` 和空 `errors`，失败报告不能被当作初始化许可。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.validate_m6_acceptance \
  --gt-eval-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_global_val \
  --gt-expert-report outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_global_val/expert_factuality_report.json \
  --fixed-eval-dir outputs/qpsalm_description/d4_predicted_75_seed42/eval_fixed_val \
  --fixed-expert-report outputs/qpsalm_description/d4_predicted_75_seed42/eval_fixed_val/expert_factuality_report.json \
  --end-to-end-eval-dir outputs/qpsalm_description/d4_predicted_75_seed42/eval_end_to_end_val \
  --end-to-end-expert-report outputs/qpsalm_description/d4_predicted_75_seed42/eval_end_to_end_val/expert_factuality_report.json \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --d4-final-gate outputs/qpsalm_description/d4_predicted_75_seed42/d4_final_m7_gate.json \
  --seed 42 \
  --output outputs/qpsalm_description/d4_predicted_75_seed42/m6_acceptance_gate.json
```

当前 M2 仍为 `awaiting_expert_review` 时，本命令按设计失败；candidate/auto_train 不能替代三份
expert report，也不能人工伪造 `passed=true`。

### 专家事实性与三 seed MGRR 门槛

先冻结模型生成，再为同一批 parent 生成两份独立审核文件：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_expert_factuality \
  --eval-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --write-template \
  --output outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val/expert_review_template.jsonl
```

模板包含冻结 instruction、model generation、区域/多模态面板路径和自动枚举的 factual
claims。审核者只能填写 `reviewer_id`、family score、每条 claim 的 support 和 notes；改写
generation、claim ID、来源字段或文本会被汇总器拒绝。模板不显示 reference target，避免
用标签措辞代替视觉事实判断。

当前汇总协议为 `qpsalm_expert_region_factuality_v2_source_revalidated`：报告保存两份 reviewer
JSONL 的路径/SHA、最小 reviewer 数和聚合 seed；M4、D4、M6 每次验收都会重新读取审核源并完整
重算，旧 v1 汇总或仅手工编辑过的报告不再进入正式 gate。

两名审核者填写后汇总 parent-level ERFS：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_expert_factuality \
  --eval-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --review /path/to/reviewer_1.jsonl --review /path/to/reviewer_2.jsonl \
  --output outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val/expert_factuality_report.json
```

`qpsalm-compare-description-runs` 需要三个 seed 各自成对的 generation、DIOR retrieval 和
expert factuality 报告；正式准入同时要求 ERFS 的 paired bootstrap CI 下界大于 0、R@1 提升，
专家 unsupported claim rate 不越过预注册非劣界限，并要求 shuffled-mask、region-swap、
cross-parent region swap、cross-parent modality swap 的 paired target-score CI 上界小于 0，
以及 modality removal 后 factual-claim count 的 paired CI 上界小于 0。五种正式反事实都必须
达到冻结 Pilot 配置的最小有效 parent 数；覆盖不足时即使 CI 数值看似有利也不能通过门槛。
其中 region-swap 仅使用同一 parent 中另一个真实区域；跨 parent mask 和几何翻转不属于
same-image region-swap，不能用于补齐其覆盖率。`cross_parent_region_swap` 是独立模式，必须
从不同 parent 的真实 mask/box 加载 donor，并记录 target/donor parent 和 region identity。
两种 region swap 都不消费 donor 的 candidate 文本，因此不会把未审核文本当作专家真值。
该门禁集合由 `landslide_bridge_m2_v7_expert_review_replay_bound` 发布。v7 把两份原始 reviewer
文件、可选 arbitration、人工 frozen-gate 源文件、`expert_all/train/val/test`、pending 索引和
发布 gate 的路径、SHA-256、字节数及 JSONL 记录数写入 `expert_review_report.json`，随后由
`validation_report.json` 再绑定该 merge report。validator 还按
`landslide_bridge_expert_review_replay_v1_exact_semantic_projection` 从 candidate、selection、双审和
仲裁源独立重建 expert/pending，并要求它们逐行等于发布产物；仅重新计算一组自洽哈希不能掩盖
错误归并。accept/revise/reject rate、双审一致性、字段分歧、证据/模态分布以及 expert 修改率
也必须从这些 raw sources 和重放产物逐字段重算。D3b/D4/M6/M7 和 unified expert publication
都要求当前 v7 重放审计。现有
v4/v5/v6 prepare/merge 产物必须在人工填写 review/gate 前重新执行 M2 prepare，不能手工补字段
或改名冒充新协议。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.compare_description_runs \
  --baseline outputs/crop_s42/eval --candidate outputs/mgrr_s42/eval --seed 42 \
  --baseline outputs/crop_s123/eval --candidate outputs/mgrr_s123/eval --seed 123 \
  --baseline outputs/crop_s3407/eval --candidate outputs/mgrr_s3407/eval --seed 3407 \
  --baseline-retrieval outputs/crop_s42/dior_eval \
  --candidate-retrieval outputs/mgrr_s42/dior_eval \
  --baseline-retrieval outputs/crop_s123/dior_eval \
  --candidate-retrieval outputs/mgrr_s123/dior_eval \
  --baseline-retrieval outputs/crop_s3407/dior_eval \
  --candidate-retrieval outputs/mgrr_s3407/dior_eval \
  --baseline-expert outputs/crop_s42/expert_factuality_report.json \
  --candidate-expert outputs/mgrr_s42/expert_factuality_report.json \
  --baseline-expert outputs/crop_s123/expert_factuality_report.json \
  --candidate-expert outputs/mgrr_s123/expert_factuality_report.json \
  --baseline-expert outputs/crop_s3407/expert_factuality_report.json \
  --candidate-expert outputs/mgrr_s3407/expert_factuality_report.json \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --output outputs/qpsalm_description/mgrr_seed_gate.json
```

上面命令只生成一个 baseline 的三 seed gate。必须分别以 `crop_only`、`full_image_box`、
`masked_pooling`、`roi_replay_only`、`mgrr_no_context` 作为 baseline 重复运行，并保持 candidate
始终是同一组三个 full-MGRR artifacts；然后发布完整 suite gate：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.validate_m4_region_encoder_suite \
  --gate crop_only=outputs/qpsalm_description/m4_crop_only_vs_mgrr_gate.json \
  --gate full_image_box=outputs/qpsalm_description/m4_full_image_box_vs_mgrr_gate.json \
  --gate masked_pooling=outputs/qpsalm_description/m4_masked_pooling_vs_mgrr_gate.json \
  --gate roi_replay_only=outputs/qpsalm_description/m4_roi_replay_only_vs_mgrr_gate.json \
  --gate mgrr_no_context=outputs/qpsalm_description/m4_mgrr_no_context_vs_mgrr_gate.json \
  --output outputs/qpsalm_description/m4_region_encoder_suite_gate.json
```

Suite CLI 会从五份 gate 的 input bindings 重新读取原始 eval/retrieval/ERFS 并重算，要求五种
baseline 全部覆盖、五份 2/3 seed gate 均通过、共享同一 frozen Bridge 和同一组三个 MGRR
candidate，并共享同一份规范化 training population。D3b→D4 25% gate 还会核对当前 seed 的 source checkpoint 正是该 suite 中的 MGRR
candidate；单独一份 crop-only comparison 不能解锁 D4。

正式比较只从上述冻结 Bridge gate 读取阈值和统计 seed，并要求当前
`qpsalm_description_evaluation_v17_structured_decoder_bound` 的完整 GT-mask、Vision-only
population
指纹及 checkpoint lineage。每个 seed 内的 baseline/candidate（包括 DIOR retrieval）还必须共享
完全相同的原始 segmentation checkpoint SHA，避免把分割模型差异归因于 region encoder。旧 v4
到 v10 报告不再参与正式 gate；v11 除证明 `--max-val-samples 0` 外，还绑定每条反事实的 mask/
backbone 输入前后指纹、donor identity 和原始 generation，并在 gate 端重算 score/claim delta，防止对截断 DataLoader 的
“全量生成”冒充完整 expert population；其 checkpoint 还必须绑定并重放精确 Description cache
artifact。v12 的 D4 fixed evaluation 另外携带可重放的 OOF/fixed artifact audit；v13 再物化并逐条
重开评价实际使用的二值 region/cycle mask，并由 source mask + M3 transform 重放 GT/fixed 输入；
v16 的独立评价不再提前写入半成品 `eval_report.json`：CLI 先原子写完 generation、反事实、
end-to-end/cycle JSONL，再通过
`qpsalm_description_evaluation_publication_v1_artifact_bound` 重开全部记录，核对行数、population、
checkpoint 与文件 SHA-256，最后一次性原子发布唯一的终态报告。失败时只写
`failure_report.json`，不会遗留可被正式 gate 误读的报告。正式 curriculum、M6 和 M7 gate
会重放该 publication audit，不再只比较复制字段。
每个主评价与 DIOR retrieval 还必须证明 runtime seed、checkpoint
保存的训练 seed 和 CLI seed 一致；四类 checkpoint 在三个 seed 槽位中必须各自唯一，禁止将
同一 run 重复传入并改写 seed 标签。
当前三 seed gate 协议为
`qpsalm_description_seed_gate_v12_strict_json_finite`；除了 artifact 唯一性，还要求
当前 ontology、record/output schema 未漂移，且三条链共享相同 expert/retrieval population、
scientific config 和训练 population。每个 seed 内仍逐字绑定完整 loader audit；跨 seed 比较会先验证
`seed+11003/21013/31019` 三个预注册 loader seed，再只剥离 loader/sampler 的运行局部 seed，保留样本
population、三流 task pattern、batch/sampler 合同、验证集和 frozen Bridge 绑定。这样不同 seed 的合法
shuffle 不会被误判为数据漂移，而改变样本或预算仍会失败。
旧 v4 只绑定 artifact seed，不能证明共同 D1 upstream 或跨 seed 可比性，不再用于正式准入。
反事实 CI 在 gate 时按 parent 重新聚合，不能用命令行阈值或运行时样本级 CI 替代。

### M7 联合训练与分割保持

M7 新运行必须用 `--initialize-from` 加载通过 M6 的 checkpoint，不能随机初始化描述头：
该来源必须是 D4 `validation_best`；M7 `--resume` 则只接受同 run 的 `terminal_last`。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc train joint \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --seed 42 \
  --initialize-from outputs/qpsalm_description/d4_predicted_75_seed42/checkpoint_best.pt \
  --region-stage predicted_mask \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --predicted-val-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --predicted-mask-fraction 0.75 \
  --d4-curriculum-sampling-seed 42 \
  --d4-final-acceptance-gate outputs/qpsalm_description/d4_predicted_75_seed42/d4_final_m7_gate.json \
  --m6-acceptance-gate outputs/qpsalm_description/d4_predicted_75_seed42/m6_acceptance_gate.json \
  --device cuda --output-dir outputs/qpsalm_description/m7_joint_seed42 \
  --overwrite-output
```

主路线默认令 `joint_train_shared_segmentation_dense=false`：SANE/QMEF/PMRD 与 controller
dense projection 保持冻结，segmentation batch 只更新 `default` adapter，description batch
只更新 `desc_adapter`、MGRR 和描述投影。`--train-shared-segmentation-dense` 仅用于已经通过
主路线 retention 后的独立消融，不能与默认结果混报。

`grad_accum_steps` 表示一个 optimizer step 内、对当前选中任务连续累积的 microbatch 数；
任务不会在同一次梯度累积中切换。默认 task pattern 为
`segmentation, global_caption, segmentation, region_description`，即 50/25/25。

联合运行使用 `qpsalm_segdesc_joint_v7_strict_json_finite`：首次遇到三类任务时分别检查目标参数有
非零有限梯度，并检查 inactive Adapter/模块保持零梯度。`joint_manifest.json` 保存 optimizer
逐参数清单、完整 loader binding 和 parent population，`joint_coverage_latest.json` 保存各任务
真实步数、样本数、parent 覆盖及 epoch/batch cursor。resume 从 task pattern 与
`grad_accum_steps` 重算每条流应消费的 microbatch 总数，再重建当前 epoch 并只跳过已消费 batch；
跳过过程不得推进已恢复的模型 RNG。旧的二分类 description 梯度门禁或 joint v3 progress
不能作为 M7 验收依据。checkpoint/retention execution replay 还逐 task 核对完整 population
parent list、coverage 唯一 ID、covered/population/fraction、covered/population SHA 与
`samples_seen`；伪造覆盖率
或写入 loader population 外 parent 会使门禁失败。

新 M7 predicted-mask run 只接受显式 `--predicted-mask-fraction 0.75`，且 final gate 必须由同一个
75% checkpoint 的完整 fixed expert-val 评价发布。它还要求 `--initialize-from` checkpoint 的
stage 与 `--region-stage` 一致，并核对该
M6 checkpoint 显式保存的
`qpsalm_region_training_data_binding_v2_cache_candidate_bound` frozen expert/predicted-index、
cache-candidate、curriculum fraction 和精确 train population audit；resume 则核对 joint
checkpoint 中同一份 region data audit。D4/M6 重放还要求 frozen Pilot gate 当前绑定的
candidate SHA、region audit 的 candidate SHA 和 Description Cache 输入 SHA 三者完全一致；
仅参数 shape 匹配不能绕过数据身份门禁。
首次加载还会发布
`qpsalm_segdesc_joint_initialization_v4_run_completion_bound`：它记录并重放实际加载的 D4 checkpoint
路径、文件 SHA、step、metadata/state inventory、seed、region-data audit，以及 D4/M6 gate
路径与 SHA。best/last checkpoint 都保存该 audit；resume 和正式 retention 必须重新打开原 D4
checkpoint 和其成功 `training_report.json` 复算，不能只复制一份看似正确的 M6 metadata，
也不能使用中断 run 留下的孤立 best。
该 audit 同时比较 joint 配置、D4 source 和 retention baseline 的原始 segmentation migration；
三者不是同一 checkpoint bytes 时，在联合优化或正式评价前失败。

新 run 会在联合优化前一次性冻结 `segmentation_monitor_baseline.json`。续训必须保留原输出
目录，并校验 checkpoint 中的 baseline identity、progress step 和三类 parent population；不能
把已联合训练的模型重新当作 baseline。因此 `--resume` 禁止与 `--overwrite-output` 同时使用：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.segdesc train joint \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --seed 42 \
  --resume outputs/qpsalm_description/m7_joint_seed42/checkpoint_last.pt \
  --region-stage predicted_mask \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --predicted-val-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --predicted-mask-fraction 0.75 \
  --d4-curriculum-sampling-seed 42 \
  --d4-final-acceptance-gate outputs/qpsalm_description/d4_predicted_75_seed42/d4_final_m7_gate.json \
  --m6-acceptance-gate outputs/qpsalm_description/d4_predicted_75_seed42/m6_acceptance_gate.json \
  --device cuda --output-dir outputs/qpsalm_description/m7_joint_seed42
```

若新 run 还覆盖了 max steps、batch size、验证/保存间隔或梯度累积，resume 必须原样重复这些参数；
checkpoint 会逐字段拒绝改变训练协议的续跑。

训练结束必须在与分割基线完全相同的完整 val 上执行 retention，而不是只看 monitor subset。
正式 retention 只接受 M7 `checkpoint_role=validation_best`；`checkpoint_last.pt` 仅用于同 run
resume，不能替代保留约束下选择出的联合最优点。
当前评估器会在 `coverage.sample_population` 中写入样本 ID、任务、目标引用、空间变换和
prompt 协议的确定性 SHA-256。retention 同时要求两份报告样本数相同、身份指纹相同、阈值相同；
基线 `eval_manifest.json` 还必须绑定生成报告的 segmentation checkpoint SHA、step、val split、
`eval_report.json` 的路径/SHA-256/字节数，以及 normal instruction/visual 协议。旧版不含
population、checkpoint SHA 或 report 字节绑定的 baseline report
不能通过正式门禁，需要先用当前 `qpsalm-eval` 对原分割 checkpoint 重新执行一次 full-val。
正式 retention 还会将 baseline manifest 的 checkpoint SHA 与 joint checkpoint 内保存的
`segmentation_migration.source_sha256` 精确比较；不能换用另一个较弱分割 checkpoint 的报告作为
基线。当前 `qpsalm_segdesc_retention_v22_run_completion_bound` 还会重新验证 joint
checkpoint 保存的 D-1 acceptance、D4 75% final acceptance 与完整 M6 三模式 acceptance，重放
联合训练实际初始化的 D4 source checkpoint，并绑定原始 `joint_segmentation_eval.json` 的路径与
SHA；同时重放 joint 成功训练终态，证明被评价的 `validation_best`、terminal last、coverage 和
history 属于同一完成 run。旧 v7 只绑定 D-1/D4、没有绑定完整
GT/fixed/end-to-end M6 acceptance；当前 v18 还从 checkpoint payload 重算 M7 task schedule、loader
binding 和 cursor，并要求当前 M6 v8 的反事实输入/原始输出、Description cache artifact、D4
OOF/fixed prediction artifacts 与正式评价 mask artifacts 逐行、逐 shard、逐文件重放，并从
`qpsalm_segmentation_eval_manifest_v3_replay_config_bound` 重验 baseline report 字节、冻结的
eval threshold/threshold sweep 及逐样本 prediction/target/valid-mask SHA。正式命令还会在加载
joint checkpoint 前，用声明的原始 segmentation checkpoint 和冻结评价参数重新执行一次同一
full-val，写入 `baseline_segmentation_replay.json`；joint 评价也使用相同 threshold/sweep，并
逐样本比较 shape、target SHA 与 valid-mask SHA，证明两次指标使用相同监督字节。冻结报告与现场
replay 的 population、阈值、threshold sweep、逐样本二值 mask 指纹和指标必须完全一致。旧 v18
及更早协议均不兼容，
必须使用当前 CLI 重跑。完整门禁失败
时 CLI 返回非零；`--max-samples` smoke 只按
`preliminary_passed` 返回状态：它会在同一有限总体上分别现场运行原 baseline 与 joint，再计算
临时 drop，永远不会把 full-val baseline 指标和有限 joint 总体混用，也不能发布正式
`passed=true`：

CLI 会在第一遍 full-val 前先以内存映射检查 joint checkpoint 本体，并重放 D-1、D4、M6、
initialization、cache 和 segmentation lineage；明显无效的 checkpoint/gate 会在占用完整 GPU
评价前失败。preflight 与之后实际加载的 checkpoint 字节、step 或 metadata 不一致同样失败。
正式运行先写 `retention_gate.candidate.json`，从磁盘完整重开候选 gate 及其全部 bound
artifacts；只有单 seed 深度验证通过才原子改名为 `retention_gate.json`。

retention 输出目录同样采用单次运行所有权；非空目录必须显式覆盖，且 baseline report 或 joint
checkpoint 位于待覆盖目录内时会拒绝删除。协议重放异常只写 `failure_report.json`，不会遗留
半成品正式 gate；证据自洽但 Dice retention 未达阈值时仍发布 `passed=false` 的
`retention_gate.json` 并返回非零，保留科学门禁失败原因。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --split val --device cuda --max-val-samples 0 \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val_population_v1 \
  --overwrite-output --skip-torch-preflight
```

然后使用新报告执行 retention：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_segdesc_retention \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --seed 42 \
  --checkpoint outputs/qpsalm_description/m7_joint_seed42/checkpoint_best.pt \
  --baseline-eval-report outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val_population_v1/eval_report.json \
  --device cuda --output-dir outputs/qpsalm_description/m7_joint_seed42/retention_full_val \
  --overwrite-output
```

三个 seed 必须分别完成同一课程链和上述 full-val retention，然后执行统一聚合。聚合器要求三个
训练 seed 与 checkpoint metadata 一致、joint checkpoint SHA 互不相同、scientific config
完全相同，并共享同一 D4/Bridge train-val population、baseline report/checkpoint、full-val
population 和阈值。`output_dir`、每个 seed 的 final-gate 路径属于 run-local 字段，不参与 config
相等比较；`m6_acceptance_gate` 同样是 run-local 路径，但其三模式 population 与绑定证据会逐条
深度验证，并要求 joint checkpoint 的 ontology、record/output schema 与当前仓库字节级一致。
当前聚合协议为 `qpsalm_segdesc_retention_seed_gate_v18_run_completion_bound`；除共同
baseline、scientific config 与相同内容的 Description Vision Cache 外，还要求三条 seed 链的
segmentation/global-caption/region-description loader contract、完整 parent population list 和
population SHA 完全相同；loader/sampler seed 可随 run seed 按预注册 offset 改变。cache 允许
路径不同，但 manifest、validation report 与 shard inventory 必须一致。聚合 CLI 先写候选文件，
再从其中列出的三个单 seed gate 路径重跑完整 validator 并逐字段重建聚合结果，完全一致后才
原子发布最终 JSON；任一源 gate/checkpoint 后续漂移都会让聚合 gate 失效。
Retention 是安全约束，
因此这里要求 **3/3 全部通过**，不能套用 M4 增益比较的 2/3 规则：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.compare_segdesc_retention \
  --retention-gate outputs/qpsalm_description/m7_joint_seed42/retention_full_val/retention_gate.json \
  --seed 42 \
  --retention-gate outputs/qpsalm_description/m7_joint_seed123/retention_full_val/retention_gate.json \
  --seed 123 \
  --retention-gate outputs/qpsalm_description/m7_joint_seed3407/retention_full_val/retention_gate.json \
  --seed 3407 \
  --output outputs/qpsalm_description/m7_retention_seed_gate.json
```

交互质检使用 `qpsalm-demo-description`，默认监听 `127.0.0.1:7861`。所有 M3-M7
工程入口完成并不等于 Full 准入；只有专家 Pilot、三 seed 和 retention 门槛均通过后才能构建 Full。
下拉项显式显示 `parent | proposal/region | sample`。overlay 使用 cache reference view 对应的
真实模态，并先裁掉 renderer padding、再按保存的 resize transform 恢复到原始影像尺寸；禁止把
正方形 cache canvas 直接拉伸为原图。Demo 也执行与 evaluator 相同的 checkpoint stage、seed、
segmentation migration 和 fixed-prediction source binding，诊断面板保存这些审计字段。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.demo_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert --seed 42 \
  --evaluation-mode gt_mask --region-source gt_global_mask \
  --checkpoint outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --split val --max-val-samples 0 --device cuda
```

## 模型 Preset

主 preset 由 `qpsalm_seg/presets.py` 定义：

| Preset | 作用 |
|---|---|
| `raw_sane_baseline` | 单 query、无语义 reliability/null gate 的均匀多源 SANE 基线 |
| `raw_sane_qmef` | 增加 null-aware、query-conditioned QMEF |
| `raw_sane_qmef_pmrd` | 增加 proposal set 与两轮 PMRD |
| `pretrained_sane_qmef_pmrd` | 使用 Qwen-ViT cache v3 的中间空间特征 |
| `qwen_psalm_full` | 在线 4-bit Qwen language decoder + QLoRA mask-query states |
| `qwen_mask_query_frozen` | 冻结 Qwen language decoder，仅训练软提示、SANE/QMEF/PMRD 的消融基线 |

正式 Qwen 路线固定使用离线视觉塔和在线语言 decoder。Qwen 不生成 bbox；它负责语义条件、
多视图证据 token、evidence anchors 和 mask-query hidden states。

## Smoke 回归

先重建 small v2，再运行：

```bash
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_smoke.sh
```

该入口使用 development-only `text_probe`，执行 5-step forward/backward、validation、
checkpoint reload 和可视化，不加载 Qwen 权重。

## 正式训练

small：

```bash
BENCHMARK_SIZE=small \
PRESET=qwen_psalm_full \
SEED=42 \
RUN_NAME=small_qwen_b4_bf16_nockpt \
RUN_CONTROL=--overwrite \
CACHE_CONTROL=reuse \
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
```

full：

```bash
BENCHMARK_SIZE=full \
PRESET=qwen_psalm_full \
SEED=42 \
RUN_NAME=full_qwen_b4_bf16_nockpt \
RUN_CONTROL=--overwrite \
CACHE_CONTROL=reuse \
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
```

24GB 单卡参数直接定义在 small/full YAML：BF16、`batch_size=4`、
`grad_accum_steps=1`、`query_chunk_size=16`，并关闭 Qwen gradient checkpoint。
脚本不再接受隐藏的精度、batch或checkpoint覆盖。正式训练先进行 450-step decoder warmup，
随后以 `0.2 × lr` 启用最后四层 QLoRA。首次运行使用 YAML 中的 batch 规模执行代表性反向门禁，
峰值上限为22.5 GiB；可用
`MEMORY_GATE=0` 显式跳过，但正式实验不建议关闭。

周期验证使用固定的 parent-aware monitor subset：small 为 512 条，full 为 1024 条；
训练结束后脚本再用 best checkpoint 完整评估 val。

## Vision Cache V3

单独准备 small cache：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --output-dir outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --backend qwen --device cuda --overwrite
```

cache 按 parent sample 和 view 分片，默认将物理视图渲染为 256，并以 `16/8/6/4`
保存 ViT layers 5/11/17/23
空间特征，使浅层保留更多边界、深层压缩语义上下文；同时保存原生 view tokens、
grid/padding transform、content hash、renderer/model/processor/prompt revision、pooling method、
full-subset signature 和 preset/尺寸 input protocol。训练时按
`ActiveModalitySubset` 动态选择，多个 instruction 不重复编码同一父图像。
cache 构建采用流式 parent 编码：Qwen 视觉塔只加载一次，内存中最多保留
`--shard-size` 个已编码父样本，写出 shard 后立即释放。`manifest.json` 中的
`peak_buffer_records` 可用于核对实际缓存上界；full 数据不再先把所有渲染视图驻留内存。
manifest 同时绑定 train/val/test instruction index 的 SHA-256；benchmark 或 instruction
索引重建后，`--verify-only` 会拒绝旧 cache。`RUN_CONTROL` 只控制训练目录；视觉 cache
由 `CACHE_CONTROL=reuse|verify|overwrite` 独立控制，默认校验并复用。
本地 Qwen revision 对配置和全部权重文件计算完整 SHA-256；每个进程只计算一次，因此首次
启动会多一次顺序读盘，但不会用仅哈希 `config.json` 的弱 revision 误判权重一致。

开发结构测试可使用：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_smoke.yaml \
  --output-dir /tmp/qpsalm_vision_v3_smoke --backend hash-smoke --max-samples 4 --overwrite
```

校验已有 cache 的 renderer/prompt/pooling/revision/subset 协议：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --output-dir outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 --verify-only
```

## 独立训练与评估

```bash
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_v2/manual_run --skip-torch-preflight
```

验证集：

```bash
python -m qpsalm_seg.cli.eval \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --checkpoint outputs/qpsalm_v2/manual_run/checkpoint_best.pt \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda --output-dir outputs/qpsalm_v2/manual_run/eval_val \
  --export-multimodal-overview --skip-torch-preflight
```

测试集需先生成 test 专用 cache：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_full.yaml \
  --preset qwen_psalm_full \
  --eval-index indexes/instruction_test.jsonl --eval-split test \
  --output-dir outputs/qpsalm_v2/cache/full_qwen_psalm_full_test_vision_v3 \
  --backend qwen --device cuda --overwrite
```

然后将 eval 命令改为 `--split test --vision-feature-cache outputs/qpsalm_v2/cache/full_qwen_psalm_full_test_vision_v3`。

## 交互推理与 PPT 图库

在浏览器中选择 benchmark 样本、活动模态并输入指令。模型和 Qwen cache 在进程内只加载一次：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.demo \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda --inbrowser
```

默认地址为 `http://127.0.0.1:7860`。原始 benchmark 指令对应正式 GT 指标；修改 instruction
或 condition 后，页面会把 Dice/IoU 标记为参考指标。关闭模态会同时更新像素输入、Qwen view、
prompt 和 QMEF availability，不会只改变界面文字。

从已有完整 val 报告生成强、典型、失败、指令对、小目标和 Sen12 专题图库：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.curate_gallery \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --eval-report outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda \
  --output-dir outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/ppt_gallery \
  --overwrite-output
```

入口只重新推理入选样本，输出 `gallery_index.html`、presentation overview、独立 mask、
`gallery_manifest.jsonl` 和 `gallery_summary.json`。PPT 图不导出 GT-only oracle proposal。

只生成局部/指代分割的 PPT 图库：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.curate_gallery \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --eval-report outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --task-family referring_landslide_segmentation --device cuda \
  --max-items 120 \
  --output-dir outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/ppt_gallery_referring \
  --overwrite-output
```

`--task-family` 在选样前过滤评估记录；可重复传入以组合多个任务族。该命令会从同一
parent 的不同位置、尺度、形态和数量指令中优先保留对照 pair，并同时抽取强、典型和失败案例。

## 消融与真实性测试

instruction 消融：

```bash
python -m qpsalm_seg.cli.eval ... --instruction-ablation shuffled
python -m qpsalm_seg.cli.eval ... --instruction-ablation fixed-generic
python -m qpsalm_seg.cli.eval ... --instruction-ablation no-semantic
```

`shuffled` instruction 同样要求至少两个不同 parent 和不同文本；不能构造有效反事实时会
直接报错。

视觉真实性消融：

```bash
python -m qpsalm_seg.cli.eval ... --visual-ablation shuffled
python -m qpsalm_seg.cli.eval ... --visual-ablation text-only
python -m qpsalm_seg.cli.eval ... --visual-ablation image-text-delta
python -m qpsalm_seg.cli.eval ... --visual-ablation remove:deformation
python -m qpsalm_seg.cli.eval ... --visual-ablation remove:sar
```

`visual_ablation` 只改变送入 Qwen language decoder 的语义 view tokens，不改变 SANE
读取的当前样本 Qwen-ViT 空间特征。因此这些实验衡量的是 Qwen 多视图 evidence 的作用，
不会同时替换 dense visual backbone。`image-text-delta` 使用完整图文上下文与 text-only
上下文的 post-context evidence anchor 差值进行 QMEF/verifier 消融，PMRD mask query 仍取
完整图文序列的 Qwen hidden states。
`shuffled` 要求 cache 中每种 raw modality 或 family 组合至少有两个不同 parent；否则评估
会明确报错，不会静默使用原图冒充 shuffle。

推荐使用单进程 suite，只加载一次 Qwen/checkpoint 并自动生成全部评估和证据报告：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_ablation_suite \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --checkpoint outputs/RUN/checkpoint_best.pt \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda \
  --visual-remove terrain --visual-remove sar --visual-remove deformation \
  --include-image-text-delta --min-delta 0 \
  --output-dir outputs/RUN/ablation_suite --overwrite-output --skip-torch-preflight
```

suite 依次切换 Dataset instruction 和 Qwen token-only visual evidence，SANE dense features
始终不变。每个条件仍生成标准 `eval_report.json/eval_manifest.json`，最后自动写
`ablation_evidence.json`；任一必需消融未出现性能退化时非零退出。

已有独立 eval 目录时，也可以单独生成严格成对证据报告：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.ablation_report \
  --normal outputs/ablations/normal \
  --instruction-shuffled outputs/ablations/instruction_shuffled \
  --instruction-fixed-generic outputs/ablations/instruction_fixed \
  --instruction-no-semantic outputs/ablations/instruction_no_semantic \
  --visual-shuffled outputs/ablations/visual_shuffled \
  --visual-text-only outputs/ablations/visual_text_only \
  --visual-remove terrain=outputs/ablations/remove_terrain \
  --visual-remove sar=outputs/ablations/remove_sar \
  --image-text-delta outputs/ablations/image_text_delta \
  --min-delta 0 --output outputs/ablations/ablation_evidence.json
```

汇总器要求所有目录包含 `eval_report.json` 和 `eval_manifest.json`，并严格检查 checkpoint、
step、split、preset 与 sample IDs 完全相同。Instruction 比较联合逐样本 proposal/final-mask
退化与 paired/no-target sensitivity；`remove:<family>` 只在确实包含该 family 的样本上比较。
normal 未优于任意必需消融时命令非零退出，不能据此声称模型使用了对应语义或视觉证据。

Qwen view token pooling 需要分别训练，可通过
`--qwen-view-pooling tokens|image-end|attention` 选择。`tokens` 是主路线；`image-end`
仅保留每个 view 的最后一个视觉 token；`attention` 使用可学习查询池化。该选项属于
checkpoint architecture protocol，不能在加载同一权重时临时切换。

报告包含 positive-only IoU/Dice、negative accuracy、empty false-positive rate、component
recall/precision、relevance AP/AUC、unmatched rejection、merge/duplicate/missed-component rate、
proposal-union Dice、同 parent paired target/prediction IoU、instruction contrast ratio 和
no-target rejection。proposal CSV、mask export 和可视化同时保留 verifier 实际选择的
`selected_proposal` 与由 GT assignment 得到的 `oracle_matched_proposal`；后者只用于测量
proposal capacity 与 selection gap，不是推理时可用的模型输出。

## 真实集成门槛

重建 small-v2 并生成正式 vision cache 后，在三 seed 实验前运行一次严格单卡检查：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.integration_check \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --mode all --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --max-memory-gib 22.5 \
  --output outputs/qpsalm_v2/real_integration_report.json
```

`raw` 检查会从真实 train split 各选择 global、referring 和 no-target 样本并完成一次
optimizer step。`qwen` 检查只运行一个 batch：从同一空间桶、Qwen sequence-load 桶和 task
group 中选择六个不同 parent 的多源正样本，并交替使用 full/dropped evidence。该 batch 完成
一次 forward、backward 和 optimizer step；只有聚合 LoRA 梯度有限非零、LoRA 参数实际更新、
teacher consistency 生效且峰值 reserved memory 不超过 22.5 GiB 时才通过。单个样本 LoRA
梯度为零不作为失败条件。结果写入
`qpsalm_real_integration_v2` JSON，任一检查失败时命令非零退出。

需要定位 Qwen/PEFT 梯度链路时运行深度诊断；它额外执行 controller-only 两个优化步骤，
第一步检查 `lora_B`，第二步检查 `lora_A`；随后分别检查 student-only segmentation 和
full/dropped consistency 的 Qwen hidden、mask/coarse/refined query 梯度：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.integration_check \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --mode qwen --qwen-check diagnostic --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --max-memory-gib 22.5 \
  --output outputs/qpsalm_v2/qwen_trainability_diagnostic.json
```

门禁通过后，先运行 5-step 阶段切换 smoke；它在 step 2 启用 QLoRA，并要求 trainer 同时
观测到非零 LoRA 梯度和真实参数更新。正式 YAML 仍保持 step 450：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --qwen-lora-start-step 2 --max-steps 5 \
  --max-train-samples 24 --max-val-samples 12 --monitor-val-samples 12 \
  --num-workers 0 --val-interval 5 --save-interval 5 --num-visualizations 4 \
  --output-dir outputs/qpsalm_v2/qwen_stage_smoke \
  --overwrite-output --skip-torch-preflight
```

成功时终端会出现一次 `[QLORA]`，详细阶段证据写入 `stage_events.jsonl`，并生成
`checkpoint_best.pt`、`checkpoint_last.pt`、validation report 和可视化。

Qwen 主训练默认关闭 activation checkpoint，以增加激活显存换取更高吞吐；显存不足时可显式
传入 `--qwen-gradient-checkpointing reentrant`。运行时不会自动回退，实际模式会写入 resolved
config、checkpoint protocol、训练启动日志和 integration report。当前配置还会使用 SDPA、
序列负载分桶和 dropped-only teacher batch，减少 padding 与重复 teacher forward。

完成门禁后可用独立 20-step batch 4 运行检查稳定吞吐，并关闭周期验证与可视化：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda --batch-size 4 --max-steps 20 \
  --qwen-lora-start-step 0 \
  --val-interval 20 --max-val-batches 1 --num-visualizations 0 \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_v2/throughput_b4_nf4 --overwrite-output --skip-torch-preflight
```

`train_history.jsonl` 会记录 `samples_per_sec`、`qwen_tokens_per_sec`、峰值显存、Qwen padding
比例和 teacher 样本比例。冻结 BF16 Qwen 对照可在相同命令中增加 `--no-qwen-4bit`；只有吞吐
更高且峰值不超过 22.5 GiB 时才应修改正式 YAML。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.summarize_run \
  --run-dir outputs/qpsalm_v2/throughput_b4_nf4 --no-export-tables
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.summarize_run \
  --run-dir outputs/qpsalm_v2/throughput_b4_frozen_bf16 --no-export-tables
```

终端 `train_performance.steady_state_last_window` 用于比较稳定吞吐，
`weighted_mean` 用于查看包含冷启动在内的整体效率。

## 数据与结果工具

```bash
python -m qpsalm_seg.cli.inspect_data --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --split train
python -m qpsalm_seg.cli.cache_index --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --output-dir outputs/qpsalm_v2/index_cache --split both --strategy round-robin-family
python -m qpsalm_seg.cli.summarize_run --run-dir outputs/qpsalm_v2/RUN --eval-dir outputs/qpsalm_v2/RUN/eval_val
python -m qpsalm_seg.cli.compare_runs --help
python -m qpsalm_seg.cli.diagnose_run --help
python -m qpsalm_seg.cli.recommend_threshold --help
python -m qpsalm_seg.cli.export_tables --help
```

三组固定 seed 的模块准入检查可重复传入成对 summary；只有至少 2/3 seed 的 candidate
pipeline ready，且 positive-only、instruction sensitivity 或 component-set 主指标超过
`--min-delta`，报告中的 `passed_2_of_3_gate` 才为 true：

```bash
python -m qpsalm_seg.cli.compare_runs \
  --baseline-summary outputs/base_s42/run_summary.json \
  --candidate-summary outputs/candidate_s42/run_summary.json \
  --baseline-summary outputs/base_s123/run_summary.json \
  --candidate-summary outputs/candidate_s123/run_summary.json \
  --baseline-summary outputs/base_s3407/run_summary.json \
  --candidate-summary outputs/candidate_s3407/run_summary.json \
  --min-delta 0 --output outputs/seed_gate.json
```

可编辑安装后的命令别名：

| 命令 | 模块 |
|---|---|
| `qpsalm-inspect-data` | `qpsalm_seg.cli.inspect_data` |
| `qpsalm-cache-index` | `qpsalm_seg.cli.cache_index` |
| `qpsalm-check-env` | `qpsalm_seg.cli.check_env` |
| `qpsalm-cache-qwen-vision-features` | `qpsalm_seg.cli.cache_qwen_vision_features` |
| `qpsalm-cache-description-vision-features` | `qpsalm_seg.cli.cache_description_vision_features` |
| `qpsalm-integration-check` | `qpsalm_seg.cli.integration_check` |
| `qpsalm-ablation-report` | `qpsalm_seg.cli.ablation_report` |
| `qpsalm-eval-ablation-suite` | `qpsalm_seg.cli.eval_ablation_suite` |
| `qpsalm-train` | `qpsalm_seg.cli.train` |
| `qpsalm-eval` | `qpsalm_seg.cli.eval` |
| `qpsalm-summarize-run` | `qpsalm_seg.cli.summarize_run` |
| `qpsalm-compare-runs` | `qpsalm_seg.cli.compare_runs` |
| `qpsalm-diagnose-run` | `qpsalm_seg.cli.diagnose_run` |
| `qpsalm-recommend-threshold` | `qpsalm_seg.cli.recommend_threshold` |
| `qpsalm-export-tables` | `qpsalm_seg.cli.export_tables` |
| `qpsalm-curate-gallery` | `qpsalm_seg.cli.curate_gallery` |
| `qpsalm-demo` | `qpsalm_seg.cli.demo` |
| `qpsalm-train-description` | `qpsalm_seg.cli.train_description` |
| `qpsalm-eval-description` | `qpsalm_seg.cli.eval_description` |
| `qpsalm-build-oof-folds` | `qpsalm_seg.cli.build_oof_folds` |
| `qpsalm-export-predicted-regions` | `qpsalm_seg.cli.export_predicted_regions` |
| `qpsalm-merge-oof-predictions` | `qpsalm_seg.cli.merge_oof_predictions` |
| `qpsalm-train-segdesc-joint` | `qpsalm_seg.cli.train_segdesc_joint` |
| `qpsalm-eval-segdesc-retention` | `qpsalm_seg.cli.eval_segdesc_retention` |
| `qpsalm-compare-segdesc-retention` | `qpsalm_seg.cli.compare_segdesc_retention` |
| `qpsalm-compare-description-runs` | `qpsalm_seg.cli.compare_description_runs` |
| `qpsalm-eval-description-zero-shot` | `qpsalm_seg.cli.eval_description_zero_shot` |
| `qpsalm-validate-d-minus-one` | `qpsalm_seg.cli.validate_d_minus_one` |
| `qpsalm-validate-d4-curriculum` | `qpsalm_seg.cli.validate_d4_curriculum` |
| `qpsalm-validate-m6-acceptance` | `qpsalm_seg.cli.validate_m6_acceptance` |
| `qpsalm-validate-m4-region-encoder-suite` | `qpsalm_seg.cli.validate_m4_region_encoder_suite` |
| `qpsalm-demo-description` | `qpsalm_seg.cli.demo_description` |
| `qpsalm-score-expert-factuality` | `qpsalm_seg.cli.score_expert_factuality` |
| `qpsalm-score-caption-metrics` | `qpsalm_seg.cli.score_caption_metrics` |
| `qpsalm-score-caption-human-review` | `qpsalm_seg.cli.score_caption_human_review` |

## 静态检查与单元测试

```bash
bash -n scripts/run_1_build_benchmark.sh scripts/run_2_build_instruction_dataset.sh \
  scripts/run_3_build_description_benchmark.sh \
  scripts/run_4_build_landslide_bridge.sh \
  scripts/run_5_build_segdesc_dataset.sh \
  SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh \
  SEG_Multi-Source_Landslides/scripts/run_qpsalm_smoke.sh

python -B -m py_compile $(find scripts/1-benchmark scripts/2-instruction \
  scripts/3-description scripts/4-landslide-bridge scripts/5-segdesc \
  SEG_Multi-Source_Landslides/qpsalm_seg -name '*.py' -type f)

PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest \
  SEG_Multi-Source_Landslides/tests/test_benchmark_v2.py \
  SEG_Multi-Source_Landslides/tests/test_refactor_core.py \
  SEG_Multi-Source_Landslides/tests/test_inference_gallery.py \
  SEG_Multi-Source_Landslides/tests/test_renderer.py \
  SEG_Multi-Source_Landslides/tests/test_v2_integration.py \
  SEG_Multi-Source_Landslides/tests/test_description_benchmark.py \
  SEG_Multi-Source_Landslides/tests/test_landslide_bridge.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_unified_index.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_protocol.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_architecture.py -v
```

当前分割算法与已知限制见
[SEG_Multi-Source_Landslides/ALGORITHM.md](SEG_Multi-Source_Landslides/ALGORITHM.md)，描述
benchmark、MGRR、双 Adapter、训练课程和科学门槛见
[docs/benchmark_GAR.md](docs/benchmark_GAR.md)。早期研究任务与重构评审只保留在
`docs/archive/`，不作为当前运行或协议依据。
