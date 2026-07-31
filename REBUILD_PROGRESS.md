# REBUILD_PROGRESS

## 当前状态

- program: `OA_GROUNDRAG_V2`
- authority: `docs/OA-GroundRAG_算法构建方案.md`
- stage: `0`
- stage_name: `ASSET_FREEZE_AND_ROUTE_MIGRATION`
- stage_status: `complete`
- next_stage: `1`
- next_stage_name: `OA_AUXSEG_FORMAL_ACCEPTANCE`
- next_gate: `A`
- scientific_status: `Stage 0 complete / Stage 1 OA-AuxSeg formal acceptance pending`
- execution_date: `2026-07-31`
- branch: `main`
- stage0_baseline_head: `1436c9dab5121f8d766bb939d6812334d2ca6409`
- active_training_process_found: `false`
- gpu_run_performed: `false`
- formal_evaluation_performed: `false`
- commit_performed: `false`
- push_performed: `false`

当前状态只表示 Stage 0 文档迁移和只读资产冻结完成，不表示 Stage 1–9 已获授权或通过
验收。代码存在、checkpoint 存在、训练结束或中途指标都不能单独作为科学验收。

## Stage 0 已完成

- 重构版已接管 canonical 路径 `docs/OA-GroundRAG_算法构建方案.md`，成为唯一活动
  算法方案。
- 旧版已移至 `docs/archive/OA-GroundRAG_算法构建方案_v1.md`，并明确标记为历史资料。
- `AGENTS.md`、`README.md` 和本文件已统一为新方案 Stage 0–9。
- 历史工程路径 `oa_groundrag/phase2/phase3/phase4` 保持不变，只更新语义映射。
- 未修改模型、Trainer、Evaluator、Benchmark builder、RAG、测试或配置代码。
- 未修改 `../datasets`、`../benchmark`、`../external`、`models_zoo/`、`outputs/`、
  `docs/RAG_knowledge/` 或任何既有 archive 文件。

## 冻结资产

### OA-AuxSeg Benchmark

- root:
  `/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/full`
- schema: `oa_auxseg_hdf5_v1`
- sample_count: `53645`
- split_counts: `train=36761 / val=12375 / test=4509`
- source_counts:
  `gdcld=13447 / lmhld=28185 / landslidebench_agent=2130 / landslide4sense=3799 / multimodal_landslide=6084`
- included_sources:
  `gdcld / lmhld / landslidebench_agent / landslide4sense / multimodal_landslide`
- excluded_sources: `sen12landslides`
- multimodal_landslide test count: `0`

这些值来自已发布 manifest 的轻量读取。本次未重跑 full validator、样本遍历、逐文件
hash 或数据源扫描。

### OA-AuxSeg 实现与训练

- repository path: `oa_groundrag/phase2`
- model schema: `oa_auxseg_model_v6`
- checkpoint schema: `oa_auxseg_checkpoint_v6`
- runtime config:
  `configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json`
- output root:
  `outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100`
- configured max steps: `229757`
- last logged training step: `213200`
- `checkpoint_best.pt`: present
- `checkpoint_last.pt`: present
- final `training_report.json`: absent
- visible training process on 2026-07-31: absent

最近一次已记录 validation 位于 step `211416`：

- Dice: `0.8249502019193231`
- positive-only Dice: `0.8334839149584277`
- no-target FPR: `0.41858367162832655`
- sample_count: `12375`

它是未完成训练中的 validation 快照，不是 Gate A 结果。当前没有已验收 OA-AuxSeg
checkpoint，也不能从该输出导出正式 fixed predicted masks。

### RS-GeneralDesc External Benchmark

- root:
  `/home/yukun80/codes/benchmark/oa_landslidedesc_external_v1`
- manifest schema: `oa_landslidedesc.manifest.v3`
- canonical schema: `oa_landslidedesc.canonical.v3`
- benchmark scope: `external_train_val`
- build_id:
  `build_8adb325c14ed7a8419b7d0e95ab2871ee277c3eac7d0409b3dbf64a9f831f96e`
- payload SHA-256:
  `f43ab63d2bb452e72648b108c43072d60b682ce936c3fbc196cde3c04fa623ec`
- records / parents: `274693 / 104954`
- external_train / external_val: `261646 / 13047`
- saved deep validation: `errors=0 / warnings=0`
- source roots embedded: `false`
- historical manifest formal flag: `false`
- historical blocker: `oa_component_disabled`

该资产已完成数据构建和 deep validation。旧 formal flag 描述的是“必须同时包含 OA
组件”的 v1 范围，不再作为 RS-GeneralDesc 数据产品失败的结论。Stage 2 将迁移
作用域验收合同，但必须保持上述 v3 manifest、build ID、payload 和资产字节只读；
不为改名重建约 40 GB 数据。

### 候选 RS-General Adapter

- output root:
  `outputs/phase4_mask_grounded_description/external_lora_qwen3vl_2b_workers2_20260730_192651`
- training report schema:
  `oa_mask_grounded_description.training_report.v1`
- report status: `completed`
- formal_acceptance: `false`
- config semantic SHA-256:
  `830ed95a289be7804dd51838583a3a78b4d69e524cbd76cdaafe4b846169fda9`
- benchmark build/payload: 与上方 RS-GeneralDesc v3 身份一致
- validation selection: `128 records / 128 parents`
- selection SHA-256:
  `aa5fdcaf8706d1ed7185b0ecab8018f62dedb2eb1f12ddf01c15b297a5382b43`
- training layout:
  `physical_batch=4 / accumulation=4 / effective_batch=16 / workers=2`
- completed:
  `1000 optimizer steps / 16000 samples / 4905307 input tokens / 538824 supervised tokens / 20571 images`
- peak CUDA: `10.187897205352783 GiB`
- elapsed: `3182.6901238840073 s`
- best checkpoint: `checkpoints/step-00001000`
- best macro task loss: `0.8729393698334892`
- best overall loss: `1.019919321325915`

该输出冻结为候选 RS-General Adapter。teacher-forced validation loss 只能选择训练期
checkpoint，不能证明 Base-vs-Adapter 生成能力提升。Stage 3 Gate B 完成前不得称为
accepted Adapter，也不默认重复训练。

### Mask-Grounded VLM 基础

- repository path: `oa_groundrag/phase4`
- 已有能力:
  `RegionSelector / EvidenceBuilder / Qwen processor-model / prompt-only / LoRA / checkpoint-resume / inference / evaluation / counterfactual`
- 当前 evidence schema:
  `oa_mask_grounded_description.evidence.v1`
- 当前合同仍要求 `rag_context=[]`
- 当前 evaluator 仍拒绝 formal evaluation
- 当前消息为单遍 user 生成合同

这些是 Stage 5 的可复用无 RAG 基线，不是 OA-GroundedEval、两遍式生成或完整
RAG 系统。Stage 0 不改写这些接口。

### RAG_tmp 与知识文档

- repository: `https://github.com/yukun80/RAG_tmp`
- inspected baseline commit:
  `4241140a8005bb79b8d8ebce982c645b096b7aca`
- local knowledge root: `docs/RAG_knowledge`
- local PDF count: `12`
- relationship: `external engineering prototype only`

根据项目负责人说明，RAG_tmp 使用的文档资料与本地 PDF 一致；远程 Git 仓库不包含
PDF 实体，因此 Stage 0 不重新下载或做逐文件 hash 对照。后续只借鉴 PDF/OCR、最小
知识单元、FTS5、dense retrieval、RRF、reranker、authority boost、引用和拒答范式。
不得直接复制代码、导入包、复用其中的 Ollama 最终生成器或把它设为运行时依赖。

## 当前科学任务：Stage 1 / Gate A

Stage 1 目标是正式验收 OA-AuxSeg，不是继续 Phase 3 模型或提前接入 RAG。进入首次
正式 test 前，必须只使用 train/val 预注册并冻结：

- checkpoint 选择规则；
- aggregate 与 positive-only 分割指标门槛；
- no-target FPR 门槛；
- auxiliary 非系统性退化判据；
- source、模态组合和低质量子组报告规则；
- 多随机种子与统计汇总规则。

不得根据 test 或当前 step 211416 的 validation 快照反推门槛。正式长训练由项目负责人
启动。当前同配置恢复入口为：

```bash
cd /home/yukun80/codes/paper7_VLM

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json \
  --resume outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_last.pt
```

恢复前必须严格核对 config、Benchmark、checkpoint identity、日志 cursor、GPU 空闲和
输出根；batch-16 不得从旧 batch-8 checkpoint 恢复。训练完成后依次：

1. 生成最终 `training_report.json` 并严格重载 accepted candidate；
2. 完成 optical-only、proposed、必要融合/quality/dropout 消融与多随机种子；
3. 完成模态组合、低质量模态、source 和 no-target 分层评价；
4. 只在 val 锁定门槛、checkpoint 和推理配置后运行一次 sealed test；
5. Gate A 通过后才导出固定 predicted masks，供 Stage 5 使用。

## 后续冻结顺序

1. **Stage 2：RS-GeneralDesc Benchmark 验收。** 移除
   `oa_component_disabled` 对 RS-GeneralDesc scope 的阻塞；现有 v3 资产保持只读，
   `external_val` 只用于训练监控。
2. **Stage 3：RS-General Adapter。** 使用当前 step-1000 candidate，对 Base Qwen3-VL
   与 Adapter 做固定生成 Gate B；gate 集排除训练报告使用的 128 个 parent。
3. **Stage 4：Landslide Evidence Corpus 与 OA-GroundedEval。** 分开构建 Auto、
   过滤 Silver 和必要 Gold，冻结正式 val/test；付费 API 和 Gold 需要单独授权。
4. **Stage 5：Mask-Grounded Baseline。** 比较 full/crop/overlay/multimodal 与
   GT/fixed/wrong/empty mask；Gate C 失败时先修 Evidence Representation。
5. **Stage 6–7：文本与案例 RAG。** 重新实现 Evidence Retrieval Provider，先文本，
   再正案例/困难负样本/分模态索引；RAG_tmp 不直接集成。
6. **Stage 8：可选 Landslide-Evidence Adapter。** 仅 Gate E 失败时训练，并执行
   RS-General retention Gate F。
7. **Stage 9：统一推理与报告。** 最后实现 Task Controller、两遍式生成、Evidence
   Cards、引用、failure artifact 和端到端评价。

## 已知科学与数据边界

- 当前 OA-AuxSeg 只支持 `dem / insar_velocity / slope` 辅助 registry；SAR 和未审计
  10 通道光学被明确拒绝。
- multimodal_landslide 当前没有 test 样本，不能宣称该 source 的独立 held-out test。
- encoded InSAR 的物理单位和 sign convention 未确认时，只能作为 encoded evidence；
  不得生成定量位移或物理方向结论。
- LMHLD 和 Landslide4Sense 缺少可靠地理 group；不得从文件名或 sample ID 伪造空间
  身份。
- 当前没有 accepted OA-AuxSeg checkpoint、fixed predicted mask、OA-GroundedEval、
  Landslide Evidence Corpus 或正式 mask-grounded test。
- Gate C 通过前不得接入 RAG；RAG 不能为候选 mask 直接寻找支持理由，必须同时检索
  反对证据、混淆对象、困难负样本和传感器限制。

## Stage 0 验收

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| `git diff --check` | 0 | 无 whitespace 错误 |
| canonical/archive 路径 | 0 | canonical 与 v1 archive 均存在，`_重构版` 临时名已消失 |
| v1 归档正文完整性 | 0 | 去除新增历史警告后 SHA-256 仍为 `cbbe3f6f7b964f559429c88e44d397f0a809718bbc85a20e3d7560f857abd4a9` |
| 活动文档旧路径扫描 | 0 | 无 `_重构版` 或已归档旧方案路径引用 |
| Markdown 本地链接 | 0 | `AGENTS.md`、README、进度和 canonical 方案的本地链接目标均存在 |
| OA Benchmark 轻量身份 | 0 | schema 与 `sample_count=53645` 匹配 |
| RS-GeneralDesc 轻量身份 | 0 | schema/scope/build/payload、saved validation `0/0` 匹配 |
| External LoRA 轻量身份 | 0 | `completed`、step 1000、16,000 samples、formal false 匹配 |
| OA-AuxSeg 训练状态 | 0 | 日志末步 213200，最终 training report 仍不存在 |
| Git 变化范围 | 0 | 仅 AGENTS、README、进度、canonical 方案和 v1 archive |

本阶段只修改 Markdown，因此未运行 Python 单元测试、模型 forward 或 GPU smoke；这与
Stage 0 的静态验证边界一致。

## 本次未运行

- GPU、训练、正式评价或长时间任务
- OA-AuxSeg 恢复、test 或 predicted-mask 导出
- External LoRA 重训或 Base-vs-Adapter Gate B
- Benchmark build、full deep validation、payload 重算或源数据扫描
- OA-GroundedEval、Silver/Gold、RAG 或端到端集成
- 数据、模型、依赖或 PDF 下载
- commit 或 push
