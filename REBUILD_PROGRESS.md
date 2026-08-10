# REBUILD_PROGRESS

## 当前状态

- program: `OA_GROUNDRAG_V2`
- authority: `docs/OA-GroundRAG_算法构建方案.md`
- stage: `6`
- stage_name: `EVIDENCE_CONSTRAINED_TEXT_RAG`
- stage_status: `gate_d_development_automatic_only_complete`
- current_task: `STAGE6_GATE_D_DEVELOPMENT_AUTOMATIC_ONLY`
- current_task_status: `frozen_25pair_protocol_gpu_run_and_recomputable_auto_eval_complete`
- next_gate: `A (OA-AuxSeg branch) / C-D (scientific protocols pending)`
- scientific_status: `Stage 6 engineering and Gate D automatic-only development evaluation are complete: the 25-pair run is strict-valid and its descriptive report is recomputable; expert relevance, unsupported-claim rate, retrieval Gold, formal Gate D thresholds, scientific acceptance and sealed test remain pending`
- execution_date: `2026-08-10`
- branch: `main`
- current_head: `eae76528b1fa7aaa1793af5451941d510d4bba29`
- stage0_baseline_head: `1436c9dab5121f8d766bb939d6812334d2ca6409`
- stage1_finalization_baseline_head: `88fec508048b1a8b3bc8dc8085396ba64449d33b`
- stage2_native_migration_baseline_head: `c198f0eb89148032f86c47e5163ac2a05498118d`
- stage3_gate_b_evidence_baseline_head: `2ad01f0723eaf698c0cbaff9bb3e993122bd87e0`
- stage4a_auto_pilot_baseline_head: `ee71e02127476da8e75b6bc9f2ce007fc38f77e5`
- algorithm_design_baseline_head: `eae76528b1fa7aaa1793af5451941d510d4bba29`
- algorithm_design_sha256: `d94a8048f10beca592b972dda3e7d7b744b9242ed163b7a89b7da36c75ffe6f5`
- algorithm_design_document_immutable: `true`
- progress_single_source: `REBUILD_PROGRESS.md`
- active_training_process_found: `false`
- gpu_inference_only_run_performed: `true`
- stage3_training_performed: `true`
- stage3_gate_b_generation_performed: `true`
- stage3_gate_b_evaluated: `true`
- stage3_gate_b_passed: `true`
- stage3_adapter_formal_acceptance: `true`
- stage4a_auto_pilot_built: `true`
- stage4a_auto_pilot_validated: `true`
- stage4_region_corpus_built: `true`
- stage4_region_corpus_validated: `true`
- oa_grounded_eval_dev_built: `true`
- oa_grounded_eval_dev_validated: `true`
- retired_teacher_silver_provider: `true`
- teacher_silver_formal_generation_performed: `false`
- stage4_silver_generated: `false`
- stage4_expert_review_completed: `false`
- stage4_single_expert_tooling_completed: `true`
- stage4_single_expert_model_verified: `true`
- stage4_single_expert_annotation_project_created: `true`
- stage4_single_expert_gpu_generation_performed: `true`
- stage4_single_expert_failed_project_discarded: `true`
- stage4_single_expert_train_annotations_completed: `false`
- stage4_single_expert_dev_references_completed: `false`
- stage4_single_expert_training_messages_exported: `false`
- stage4_model_assisted_extension_built: `true`
- stage4_model_assisted_extension_validated: `true`
- stage4_model_assisted_collection_built: `true`
- stage4_model_assisted_collection_validated: `true`
- stage4_model_assisted_project_created: `true`
- stage4_model_assisted_gpu_generation_performed: `true`
- stage4_model_assisted_supervision_exported: `true`
- stage4_model_assisted_training_messages_exported: `true`
- stage5_compact_training_published: `true`
- stage5_training_completed: `true`
- stage5_region_dev_evaluated: `true`
- stage5_rs_general_retention_reported: `true`
- stage5_formal_acceptance: `false`
- stage5_scientific_acceptance: `false`
- stage6_text_evidence_bank_built: `true`
- stage6_text_evidence_bank_validated: `true`
- stage6_hybrid_retrieval_completed: `true`
- stage6_paired_gpu_smoke_completed: `true`
- stage6_engineering_complete: `true`
- stage6_gate_d_dev_protocol_frozen: `true`
- stage6_gate_d_dev_25pairs_completed: `true`
- stage6_gate_d_automatic_eval_completed: `true`
- stage6_gate_d_development_automatic_only_complete: `true`
- stage6_gate_d_scientific_pass: `false`
- stage6_formal_acceptance: `false`
- stage6_scientific_acceptance: `false`
- oa_grounded_eval_completed: `false`
- stage2_gpu_run_performed: `false`
- stage2_native_full_deep_validation_performed: `true`
- identity_hardening_repackage_performed: `false`
- identity_hardening_deep_validation_performed: `false`
- identity_hardening_gpu_run_performed: `false`
- current_task_training_or_optimizer_step_performed: `false`
- formal_evaluation_performed: `true`
- test_split_evaluated: `false`
- commit_performed: `false`
- push_performed: `false`

Stage 0 权威迁移、Stage 1 工程定版、Stage 2 RS-GeneralDesc Benchmark native v1
重发布与验收、Stage 3 Adapter 重训和 Gate B，以及 Stage 4A deterministic Auto Pilot
均已完成。当前没有活动的 Teacher Silver 实现或正式 Silver 产物；新版 train-only
Region Corpus 与 val-only OA-GroundedEval-dev 工程资产已迁移到统一 Benchmark v1 根并重新验证；
本地模型草稿、固定 `annotator="expert"` 的单专家核验、一键恢复、不可变 package 和
train-only messages 工具链已实现，本地 Qwen3-VL-8B 冻结身份已验证。当前 v1 工作根保留
20 条信息化 calibration 草稿与 5 条 `expert_verified`，原 Gradio 进程已按精确 PID 正常关闭。
在独立 Benchmark v2 根又发布了 train-only 7,950 条扩展 Corpus 和引用 base+extension 的
8,450 条 collection；8B 已完成 8,450 条草稿，其中 6,974 条合格监督被发布为脱离 raw
work/package 的 compact v3。Stage 5 已从冻结 RS-General Adapter warm-start 完成 1,000-step
Mask-Grounded Region LoRA、340 条 automatic-only dev 评价和冻结 Gate B selection 上的
只报告 retention。多人 review、adjudication、Gold、人工 val reference、retention 阈值和
正式协议冻结不属于当前简化方案，科学验收仍未完成。
依赖分为 OA-AuxSeg
工程定版 → 未来 Gate A → formal fixed masks，以及 RS-GeneralDesc Stage 2 → Adapter
重训 → Gate B；两者在 Mask-Grounded 阶段汇合。Gate A 延后不等于通过，也不推翻
独立完成的 Stage 3 Gate B。Gate B 不表示 OA-Grounded、mask-grounded 或系统验收。

## 文档单一来源治理

当前冻结算法方案来自 HEAD
`eae76528b1fa7aaa1793af5451941d510d4bba29`，文件 SHA-256 为
`d94a8048f10beca592b972dda3e7d7b744b9242ed163b7a89b7da36c75ffe6f5`。
它已包含简化后的 Stage 6 Evidence-Constrained Text RAG 设计。本次 Stage 6 实施没有修改
算法方案正文；其中的状态文字仍是设计冻结快照，不用于判断现场进度。此前在本文件记录的
`768d6880...` / `dbf93a4c...` 是旧设计身份，只保留在历史运行段落中。

`README.md` 只保留长期有效的项目结构、合同边界和运行入口；`AGENTS.md`
只保留操作、授权、安全和文档治理规则。阶段状态、运行结果、正式产物
身份、验收证据和下一任务只在本文件维护。

`README.md` 只保留长期有效入口；`AGENTS.md`、算法方案和 `docs/archive/` 本次未修改。
旧算法身份仅作为历史现场记录，不再是当前授权基线。

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
- `checkpoint_best.pt`: final, step `206820`
- `checkpoint_last.pt`: trace only, step `211416`
- logged but not checkpointed: `211417–213200` (`1784` steps)
- final `training_report.json`: present
- report schema: `oa_auxseg.training_report.v1`
- completion mode: `project_owner_manual_stop`
- resume required: `false`
- final checkpoint SHA-256:
  `672d39ab4220d8e1b4f949ca8d1d5dcd34f58898cecd1553dd56cdd9d84fb038`
- visible training process on 2026-07-31: absent

项目负责人确认本次训练是主动手动停止，计划 `max_steps` 仅为预算上限，不要求续训。
既有选择规则冻结的 step `206820` best validation 为：

- Dice: `0.8254104937535791`
- IoU: `0.7027225165592563`
- positive-only Dice: `0.8343047072386582`
- no-target FPR: `0.4239963915200722`
- sample_count: `12375`

离线 fresh val 与上述训练期 selection snapshot 的指标差值为 0。完整 train 工程评价为
36,761 samples、Dice `0.9753234910697791`、IoU `0.9518355135202943`、no-target FPR
`0.0011661807580174927`。严格重载的 mask logits、probability、modality weights 和
四尺度 weight maps 差值均为 0。

该 `checkpoint_best.pt` 是项目负责人定版的当前最终权重，但不是 Gate-A-accepted
checkpoint。Gate A、消融、sealed test 和正式 fixed predicted masks 仍未执行。

### RS-GeneralDesc Benchmark native v1

- root:
  `/home/yukun80/codes/benchmark/rs_generaldesc_v1`
- manifest schema: `rs_generaldesc.manifest.v1`
- canonical schema: `rs_generaldesc.canonical.v1`
- benchmark scope: `rs_generaldesc_external_train_val`
- build_id:
  `build_3ebc09a4daad10e121fc14c2727d9896e10371a95bbaf6b780d15aa42eaf3c03`
- payload SHA-256:
  `549281f296b357bce256e6af71cec7412fe17e36052d6a8674f4876ae2d06e0b`
- semantic config SHA-256:
  `bb9b00ea44fb1c79e9efdfa00fbf73e8d8e9e5c416b8e9ff48d0c0b550e0d162`
- hash manifest SHA-256:
  `55ac26d9771ce8385318fbd23a10b999afb754ac195e823be659f4e49b0a7090`
- records / parents: `274693 / 104954`
- external_train / external_val: `261646 / 13047`
- saved deep validation: `errors=0 / warnings=0`
- source roots embedded: `false`
- formal acceptance eligible: `true`
- formal acceptance blockers: `[]`
- release equivalence schema: `rs_generaldesc.release_equivalence.v1`
- OA-Grounded acceptance: `false`
- Gate B evaluated: `true`（由独立 Stage 3 protocol 完成）
- external_val usage: `Stage 2 training monitoring; Stage 3 independent frozen Gate B selection`

该资产由冻结 payload 确定性重发布；canonical/provenance/ID/SHA/build identity 全部按
native v1 重算，且新树自身具备固定 manifest/hash identity、eligible/空 blockers 和
saved clean deep validation。历史 `release_equivalence.json` 记录了当时的 record、parent、
role 和 asset 等价结论，但旧 repackage 未逐项核验前代 ledger 的全部 record/metadata，
前代 root 又已删除，因此不能追溯性地把该报告升级为严格逐文件证明。目前没有证据表明
native payload 损坏；本次修复后的未来 repackage 才强制完整 ledger 验证。

活动 phase3 builder 只装载 RSGPT、MMRS-1M、DisasterM3，只生成
`rs_generaldesc_external_train_val`；Dataset/exporter 仅接受两类 role 和七类任务。
额外作用域报告、混合分支、旧 validator 与兼容 API 已删除。

### RS-General Adapter

身份迁移前的 phase4 Adapter/checkpoint/training evidence 已永久删除，不保留备份、链接
或 alias。首次 native v1 `batch=4/accumulation=4` 运行在 24 GB 目标显存边界下停止，
不得跨布局恢复；随后使用 physical batch 1、gradient accumulation 16、effective
batch 16 在全新 `rs_vlm_lora_qwen3vl_2b_b1a16` 根完成到 step 1000。best pointer 按
`macro_task_loss` 指向 step 1000（`0.8427927826882716`）。training report 保留
`formal_acceptance=false`，因为它只证明训练闭环；独立 Gate B report 随后以
`formal_acceptance=true` 接受该 Adapter。

最终 Adapter 身份：training report SHA-256
`a4f42e777eaab6e444f04d63b89f482ee31a077bf13006d587863bfa4fb1eb1e`，best pointer
`4d93e2c6c34fe01a10db373c00946166238b8d132bb32b9863b52e305b6f4db6`，step-1000
checkpoint manifest
`aa279659a4c563536f1d7554ed9e51643398365ce11539eb9865f77c1d3a621f`，Adapter 权重
`a367e39c626338a151dad33e6f7a7f9cc9887206dbcd261d147837e6408becc1`。

### Mask-Grounded VLM 基础

- repository path: `oa_groundrag/phase4`
- 已有能力:
  `RegionSelector / EvidenceBuilder / Qwen processor-model / prompt-only / LoRA / checkpoint-resume / inference / evaluation / counterfactual`
- 当前 evidence schema:
  `rs_vlm.evidence.v1`
- 当前合同仍要求 `rag_context=[]`
- 当前 evaluator 仍拒绝 formal evaluation
- 当前消息为单遍 user 生成合同

这些是 Stage 5 的可复用无 RAG 基线，不是 OA-GroundedEval、两遍式生成或完整
RAG 系统。当前 `rs_vlm.config.v2` 只运行 RS-GeneralDesc External；GT/fixed/end-to-end
mask、RegionSelector、EvidenceBuilder、mask-grounded messages、AuxSeg inference 和
反事实评价核心继续保留。临时 Mask-Grounded Dataset 合同已撤回，等待 Stage 4/5 冻结
新数据 schema 后再接入。

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

## Stage 1 工程定版边界

Stage 1 的 proposed 主模型训练已经由负责人定版完成，不再续训。本轮新增的
`finalize` 入口不创建 optimizer、不执行 backward 或 scheduler step，只核对
best/last/log/Benchmark 身份，运行 train/val inference-only 评价、严格重载并原子
生成报告：

```bash
cd /home/yukun80/codes/paper7_VLM

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase2_oa_auxseg/run_oa_auxseg.py finalize \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json \
  --checkpoint outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt \
  --termination-reason project_owner_manual_stop
```

该命令已经成功完成，不得重复覆盖报告。`checkpoint_best.pt`、`checkpoint_last.pt` 和
`train_log.jsonl` 的 size/mtime 在运行前后完全一致，未创建 `checkpoint_final.pt`。

Stage 1 剩余科学任务不是恢复训练。进入首次正式 test 前，必须只使用 train/val
预注册并冻结：

- checkpoint 选择规则；
- aggregate 与 positive-only 分割指标门槛；
- no-target FPR 门槛；
- auxiliary 非系统性退化判据；
- source、模态组合和低质量子组报告规则；
- 多随机种子与统计汇总规则。

不得根据 test 或当前 best validation 快照反推门槛。分割消融和多随机种子实验按项目
负责人指令延后到完整框架搭建后；只有未来 Gate A 通过，才运行一次 sealed test 并
导出供 Stage 5 使用的正式 fixed predicted masks。

## 当前科学任务：Stage 4A deterministic Auto Pilot 已完成

Stage 4A 已从冻结 OA-AuxSeg train split 构建并正式验证 500 条 Landslide Evidence
Corpus Auto records。五源各 100 条，总计 400 target / 100 no-target；没有打开 val/test
HDF5 内容，没有使用 predicted mask。发布根为
`outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500`。

| 身份或规模 | 冻结值 |
| --- | --- |
| manifest SHA-256 | `37aebb9c5f8ceb720e0a1a3c8621212d44562fa6b6786d145c31e11ffa94f9bb` |
| corpus ID | `f4dd4c4e124e112154145a207d67f209ba9a3a1e7cfe56c35ab2f732624951d0` |
| ordered sample IDs SHA-256 | `84e1801fe9ec37284d7bf02b663153d057145726fc78c6d54e87ded11e185936` |
| records | `500`；`records.jsonl` SHA-256 `36f5dcbf8fe6008be09ccb291eb64a8a72398f01337ebb84ce13063d3a870c5d` |
| assets | `2200`；`115528315` bytes；500 optical / 500 mask / 400 overlay / 400 crop / 400 auxiliary |
| ledger | `2201` entries；file SHA-256 `0b0f7a1488e2069ce5d5ed03365d756905f81e742f12d2848dfb9b81de3ef430`；root `bf1f0446e17e6e94ea04859e4e6311226a5684690f44b6426ac64759e32b64c1` |

五种实际 modality signature 各覆盖 100 条。source/target/foreground-ratio quota 全部
达到；唯一组件覆盖不足为 `landslide4sense × large`，18 条均为多连通、单连通为 0，
manifest 记录 `both_covered=false`，未补造样本。所有 `silver_generated`、
`expert_review_completed`、`oa_grounded_eval`、`adapter_training_eligible` 和
`case_kb_eligible` 均为 false。

该 Corpus 是结构化 Auto evidence，不是分割 Benchmark、完整 RAG 知识库或人工测试集。
Silver 仍是假标签，正式生成和专家审核均未执行；OA-GroundedEval 继续作为独立的设计、
人工标注和冻结任务。Landslide-Evidence Adapter 只在后续 Gate E 失败时考虑。
Stage 4A v1 的辅助模态合同只包含 `dem / slope / insar_velocity`；当前 Corpus 不含 SAR，
SAR 是未来显式扩展，不能从现有通道或文件名推断补造。

### Stage 4A 验证记录

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| `/tmp/oa_groundrag_stage4_tests` synthetic unittest | 0 | 11/11；Stage 4A 相关覆盖 train-only/leakage、500 确定性去重、target/no-target、资产路径、科学 claims、manifest/record/ledger 篡改、原子发布与 symlink/path escape |
| Phase 3 单元测试 | 0 | 45/45 |
| 当前 Phase 4 单元测试 | 0 | 82/82；包含保留的 Gate B 媒体定位器测试 |
| 三份活动 RS-VLM preflight | 0 | 3/3 metadata-only；临时 output roots 未创建 |
| 只读 Gate B verifier | 0 | accepted；report SHA-256 仍为 `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |
| 真实 `build-auto` | 0 | staging 内重算通过后原子发布 500 records / 2200 assets |
| 真实 CLI `validate` | 0 | `valid=true`、`source_verified=true`、400/100、五源各 100 |
| Python compile / `git diff --check` | 0 | Stage 4A、phase4、CLI 与临时测试 compile 通过；无 whitespace 错误 |
| 当前 Stage 4A 核心 Python 规模 | 0 | 四个核心模块加薄 CLI 共 1218 physical / 1093 nonblank 行；既有 `phase4/evidence.py` 通用扩展净增 135 physical / 118 nonblank 行 |
| 临时测试清理 | 0 | 已精确删除 `/tmp/oa_groundrag_stage4_tests`；未创建 `tests/stage4_landslide_evidence` |

临时测试命令为：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m unittest discover \
  -s /tmp/oa_groundrag_stage4_tests -p 'test_stage4_landslide_evidence.py' -v
```

真实构建和验证命令为：

```bash
python scripts/stage4_landslide_evidence/run_landslide_evidence.py build-auto \
  --config configs/stage4_landslide_evidence/pilot_500.yaml
python scripts/stage4_landslide_evidence/run_landslide_evidence.py validate \
  --root outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500
```

本任务未运行 GPU、外部 API、正式 Silver、专家审核、OA-GroundedEval、训练、test、
Gate A/C/D/E、predicted mask、RAG、下载、commit 或 push；未修改 Benchmark、checkpoint、
Gate B 正式产物和既有训练 outputs。

## Stage 4B 本地 Provider 退役简记

此前的本地 Qwen Teacher Silver Provider 仅在 `/tmp` 执行有界 GPU 诊断。生成链路能够
运行，但格式与科学质量没有达到预注册 smoke 要求；未运行正式生成，未形成正式 Silver、
过滤结果、审核队列或 OA-GroundedEval。相关配置、运行时、候选合同和 CLI 已从活动仓库
删除，不保留 legacy alias。Stage 4A Corpus 及其冻结身份未修改。

`gate-b-locate-media` 是独立的 Gate B 只读人工检查工具，继续保留其严格身份校验和
10 项永久测试；它不是 Teacher Silver Provider。

## 后续冻结顺序

1. **Teacher Silver 方案重新设计。** 基于算法方案中的 Provider 无关原则，重新确定模型、
   mask/overlay/crop 空间提示、结构化输出、规则过滤和有界验证协议；未经负责人授权不实现
   或运行。
2. **Stage 4 后续人工资产。** Silver 候选通过有界验证后，另行授权专家审核、必要 Gold
   和 OA-GroundedEval 的人工标注与冻结。
3. **Stage 5：Mask-Grounded Baseline。** 比较 full/crop/overlay/multimodal 与
   GT/fixed/wrong/empty mask；Gate C 失败时先修 Evidence Representation。
4. **Stage 6–7：文本与案例 RAG。** 重新实现 Evidence Retrieval Provider，先文本，
   再正案例/困难负样本/分模态索引；RAG_tmp 不直接集成。
5. **Stage 8：可选 Landslide-Evidence Adapter。** 仅 Gate E 失败时训练，并执行
   RS-General retention Gate F。
6. **Stage 9：统一推理与报告。** 最后实现 Task Controller、两遍式生成、Evidence
   Cards、引用、failure artifact 和端到端评价。

## 已知科学与数据边界

- 当前 OA-AuxSeg 只支持 `dem / insar_velocity / slope` 辅助 registry；SAR 和未审计
  10 通道光学被明确拒绝。
- multimodal_landslide 当前没有 test 样本，不能宣称该 source 的独立 held-out test。
- encoded InSAR 的物理单位和 sign convention 未确认时，只能作为 encoded evidence；
  不得生成定量位移或物理方向结论。
- LMHLD 和 Landslide4Sense 缺少可靠地理 group；不得从文件名或 sample ID 伪造空间
  身份。
- 当前有负责人定版的 OA-AuxSeg final checkpoint 和已验证的 Stage 4A Auto Corpus，
  但没有 Gate-A-accepted checkpoint、fixed predicted mask、活动的 Teacher Silver
  实现、正式 Silver/Gold、OA-GroundedEval 或正式 mask-grounded test。
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
| OA-AuxSeg 训练状态（Stage 0 快照） | 0 | 当时日志末步 213200，training report 尚不存在 |
| Git 变化范围 | 0 | 仅 AGENTS、README、进度、canonical 方案和 v1 archive |

以上是 Stage 0 当时的验收记录；此后 Stage 1 已新增代码、测试和人工定版报告。

## Stage 1 工程定版记录（Gate A 未执行）

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 新增 finalization 定向测试 | 0 | 7/7 |
| 不依赖历史 small 的 Phase 2 回归 | 0 | 36/36 |
| Phase 2 全量发现 | 1 | 40 个中 36 通过；4 个仅因 `../benchmark/oa_auxseg_hdf5_v1/small` 不存在而失败 |
| 真实 `finalize` train/val | 0 | train 36,761 / val 12,375，工程检查全部通过 |
| fresh val replay | 0 | selection loss 与 overall 全指标差值均为 0 |
| checkpoint 严格重载 | 0 | 四类输出最大绝对差值均为 0 |
| best/last/log 资产不变 | 0 | size 与 mtime 前后完全一致 |
| sealed test | 未运行 | `test_evaluated=false` |
| formal Gate A | 未运行 | `gate_a_evaluated=false`、`formal_acceptance=false` |

缺失的 historical small 资产不是本次改动造成，也不为通过测试而重建、复制或修改
Benchmark。当前 full Benchmark 和真实 final checkpoint 已完成本任务所需验证。

## Stage 2 native v1 重发布与验收

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Phase 3 全量单元测试 | 0 | 当时 39/39；含 native builder/Dataset/exporter/repackage/validator |
| Phase 4 全量单元测试 | 0 | 当时 48/48；mask/evidence/region/AuxSeg 核心保留 |
| 真实 `repackage` | 0 | 历史报告记录 split/group/content/asset 等价；旧实现未逐项验证全部前代 ledger |
| 新树 full deep validation | 0 | 单次运行；0 error / 0 warning |
| CPU RS-VLM preflight | 0 | schema/build/payload/hash identity 匹配；未创建 output root |
| phase4 活动配置 preflight | 0 | 3/3 直接绑定 native identity |
| 前代 Benchmark 与 Adapter outputs | 已删除 | 验收完成后永久删除；无备份、链接或 alias |
| 源数据读取 / 图像重编码 | 未运行 | 只复制已发布图像字节并重写文本元数据 |
| GPU / training / test / Gate A / Gate B | 未运行 | native 身份迁移不扩张为科学评价 |

## 本次身份绑定加固

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Phase 3 全量单元测试 | 0 | 45/45；新增 ledger、record/metadata/asset 漂移与等价写入时序保护 |
| Phase 4 全量单元测试 | 0 | 53/53；新增 manifest/validation/metadata/shard/asset 运行时绑定保护 |
| 三份真实配置 preflight | 0 | 3/3 metadata-only；manifest/validation/build/payload/ledger identity 匹配 |
| 真实 Benchmark repackage/deep validation | 未运行 | 只使用 `/tmp` 小 fixture，不扫描 40 GB payload |
| Benchmark/checkpoint/outputs 写入 | 未运行 | size/mtime/SHA 与目录清单前后复核 |

## Stage 3 Gate B 正式验收记录

正式输出根为
`outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1`。只读证据复核重新验证了
training root、确定性 selection、Base/Adapter manifests 与 predictions、canonical
records、paired scores、全部指标、10,000 次 bootstrap 和六项判据；未读取图像资产。

| 正式 artifact | SHA-256 |
| --- | --- |
| frozen protocol file | `8378f6f107849439be3b402b0014df4007b10589e93602c1afb99173c2fb2c54` |
| selection file | `98290aaa585b798dcc5a30b9a4d47083e778aa4d48980e24bbce647705b915bd` |
| Base generation manifest | `bad680291426f65fd51dfaa35eca649968e478af963d3c5009cdbce733a699fb` |
| Base predictions | `862759c44400552f40f5211a38ceafd8d1f4712c7d6f870e3a2f0676d8ce8bd6` |
| Adapter generation manifest | `c69314727ba71ef712fd7bbde1990ebd610e6759714641103916adf901787168` |
| Adapter predictions | `a7c791fcfe8f3b94f6b188780bc0aed46dbf2fcf92f1bcca3b6617ca1c4bd98a` |
| paired scores | `64d6802e2b2438305fa7ba560bc4d90ebb4537d1048dc281333c707fb4d3975f` |
| Gate B report | `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |

protocol canonical SHA 为
`05ffb5ddf1940fb2474f13daf9ad7beec844f20d3da70a13c6c1c9f20a6eef0d`，selection
canonical SHA 为
`75285af6d8cee21c84760edcc2f20d71beb283bca8d340d6d3243e22c8537119`。selection
重算与发布 items 逐项一致，256 个 parent 与 training monitoring 的 128 个 parent
交集为 0。Base/Adapter 均为 256 predictions、0 failures，输入顺序、69,798 tokens
和 338 images 一致，paired count 为 256。

- primary task-macro：Base `0.2197314184`，Adapter `0.4559849197`，delta
  `0.2362535013`；95% paired-bootstrap CI 为
  `[0.2061703267, 0.2670255801]`；
- task 数量依次为 `42/41/41/41/11/40/40`；primary delta 依次为
  `0.4429630184 / 0.1316141258 / 0.2377844412 / 0.2792818227 /
  0.0753093476 / 0.2834546518 / 0.2033671016`，七类均为正；
- source 数量为 `disasterm3=128 / mmrs1m=101 / rsgpt=27`；source macro delta
  分别为 `0.2266998826 / 0.2959882661 / 0.0341324147`；
- 六项判据 observed 值分别为 `0.2061703267 / 7 / 0.0753093476 /
  0.0341324147 / 0.2743480215 / 0.0804878049`，全部 PASS；
- report 为 `status=completed`、`gate_b_evaluated=true`、`gate_b_passed=true`、
  `formal_acceptance=true`、`adapter_status=accepted`。

当前 PASS 只证明 RS-GeneralDesc native v1 固定 lexical protocol 下的相对提升，不表示
Gate A、OA-Grounded、mask-grounded、sealed test 或最终系统验收。v1 的 selection loader
本身不重推确定性唯一输出，模型完整权重、tokenizer/generation config、传递实现依赖和
运行环境也未形成完整 ledger；当前复核未发现实际 selection、输入或产物漂移。training
monitoring 与 Gate 共享一个 asset SHA，涉及两条 Gate `spatial_relation` records；删除
它们的非门控事后敏感性分析仍满足六项判据，delta 为 `0.2340555947`，CI 下界为
`0.2038789283`。Gate 内 11 条 spatial records 对应 4 个重复 asset 组；task/source
配额是 capacity-constrained，lexical metric 同时反映答案风格和词面重合，现有产物也
没有真实 finish reason。这些限制不事后改写 v1 判据；未来重跑须升级 v2，预注册完整
模型/tokenizer/runtime ledger、asset-component selection、task-aware metric、grouped
bootstrap 和 finish reason。

### 本次证据闭环验证

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 只读 `gate-b-verify` | 0 | selection、training、256 配对、指标、bootstrap、六判据和 8 个 artifact SHA 全部一致 |
| Phase 3 全量单元测试 | 0 | 45/45 |
| Phase 4 全量单元测试 | 0 | 72/72；含 verifier、public completed pass/fail、真实 exclusion、临时 training tamper 和 CUDA 错误窄分类 |
| 三份 `rs_vlm.config.v2` preflight | 0 | 3/3 metadata-only；临时 output roots 均未创建 |
| Python compile | 0 | `oa_groundrag/phase4/*.py`、phase4 CLI 和测试全部通过 |
| `git diff --check` | 0 | 无 whitespace 错误 |
| 正式 Gate B/训练产物不变 | 0 | 12 个文件的 size、mtime、SHA-256 前后完全一致 |

## Stage 0 当时未运行

- GPU、训练、正式评价或长时间任务
- OA-AuxSeg 恢复、test 或 predicted-mask 导出
- External LoRA 重训或 Base-vs-Adapter Gate B
- 源数据扫描、图像重编码或第二次 full deep validation
- OA-GroundedEval、Silver/Gold、RAG 或端到端集成
- 数据、模型、依赖或 PDF 下载
- commit 或 push

## Stage 1 人工定版当时未运行

- resume、optimizer step、backward、scheduler step 或任何训练
- checkpoint 保存、复制、重命名或 train log 裁剪
- test、Gate A、分割消融、多随机种子或正式 predicted-mask 导出
- Benchmark build、deep validation、payload 重算或源数据扫描
- 当时的 Stage 2–9 算法开发、外部下载、commit 或 push

## 本次身份绑定加固未运行

- 真实 Benchmark build、repackage、export、复制、records/assets 遍历或 payload validator
- GPU、训练、inference、生成评价、test、Gate A 或 Gate B
- OA-GroundedEval、Evidence Corpus、Silver/Gold、RAG 或 Stage 3–9 实施
- 既有 LoRA、checkpoint、training report、validation selection 或模型权重写入
- 数据、模型、依赖下载、commit 或 push

## 本次 Gate B 证据闭环未运行

- Base/Adapter 重新生成、GPU、训练、optimizer/backward 或 checkpoint 写入
- test、Gate A、deep validation、Benchmark payload/asset 重扫或 repackage
- Stage 4 Evidence Corpus/OA-GroundedEval 构建、Mask-Grounded 实验或 RAG
- 正式 Gate B artifact、training report、best pointer、checkpoint、Adapter 权重或
  Benchmark 修改
- 数据、模型、依赖下载、commit 或 push

## Stage 4 Mask-Grounded Region Corpus 与 OA-GroundedEval-dev 工程闭环（2026-08-04）

### 授权基线与写入边界

- 开始分支 / HEAD / upstream：`main` /
  `768d68804d48ab0384f28832a6ee4c838be3e493` / `origin/main`；开始时工作区干净。
- 当前算法文档 SHA-256：
  `dbf93a4ceea1ae973974fb039d5bd2c35d8a62b615ce2add3ab1583f91366d36`；
  本文件中的旧算法身份已在首次写入时修复。
- Stage 4A 根、OA-AuxSeg Benchmark/checkpoint、RS-GeneralDesc Benchmark/Adapter、
  Gate B 产物全程只读；没有修改 `../datasets`、`../benchmark`、`../external`、
  checkpoint、模型权重或既有 outputs。
- 新开发仅处理原始光学 RGB 与人工 GT binary mask。没有读取 test shard，没有使用
  predicted mask、OA-AuxSeg 推理、RAG、Teacher Silver、Gold 或外部多模态 API。

### 实际代码与合同

新增实现文件：

- `oa_groundrag/landslide_evidence/region_contracts.py`：集中定义 Region/Eval schema、
  frozen 数据合同、representation mode、字段 enum 和严格 JSON 工具；
- `oa_groundrag/landslide_evidence/region_pipeline.py`：复核冻结 Stage 4A ordered IDs，
  从 OA Benchmark train GT 原子构建 full/mask/crop/audit-overlay 与 annotation queue；
- `oa_groundrag/landslide_evidence/grounded_eval.py`：val-only 配额选样、边界清晰度 proxy、
  deterministic shift 和反事实组构建；
- `oa_groundrag/landslide_evidence/annotation.py`：annotation queue 导出、人工 JSONL
  严格导入和全新 annotation package 发布；
- `oa_groundrag/landslide_evidence/region_validation.py`：重算 split/parent/source/几何/
  crop/shift/角色、逐资产 SHA/size/mode/pixel，以及 symlink/hardlink/path-escape 防护；
- `oa_groundrag/phase4/grounded_evaluation.py`：开发自动评价、反事实敏感性与可选专家
  聚合；所有报告强制 `formal_acceptance=false`；
- `scripts/stage4_landslide_evidence/run_mask_grounded_region.py`：八个薄子命令；
- `configs/stage4_landslide_evidence/region_corpus_train_v1.yaml` 与
  `oa_grounded_eval_dev_v1.yaml`：独立 train Corpus / val Eval-dev 配置；
- `tests/stage4_landslide_evidence/fixture_helpers.py` 及六个 `test_*.py`：28 个永久 CPU
  回归测试。

增量修改文件：

- `oa_groundrag/phase4/evidence.py`：严格 binary mask、clean/tight crop、audit overlay、
  程序几何与 deterministic shift；
- `oa_groundrag/phase4/messages.py`：独立 v2 有序多图消息和 formal/audit role 隔离；
- `oa_groundrag/phase4/outputs.py`：Stage 4 v2 严格输出 parser、禁止结论检测和版本化
  prediction/failure/provenance；
- `oa_groundrag/phase4/errors.py`、`oa_groundrag/landslide_evidence/__init__.py`：新增集中
  reason code 与公共接口导出；
- `README.md`：只增加长期有效的八个 CLI 入口；本文件记录动态进度。

未修改旧 `contracts.py`、`pipeline.py`、`validation.py` 和
`run_landslide_evidence.py` 的语义；Stage 4A `build-auto` / `validate` 继续通过。

冻结 schema：

- `oa_groundrag.mask_grounded_region.{config,manifest,record,annotation_queue,annotation,annotation_package}.v1`
- `oa_groundrag.oa_grounded_eval_dev.{config,manifest,counterfactual_group}.v1`
- `rs_vlm.mask_grounded_region_output.v2`
- `rs_vlm.mask_grounded_region_{prediction,failure,provenance}.v1`
- `oa_groundrag.oa_grounded_eval_dev.report.v1`

### 首次发布身份（迁移前历史）

以下为 2026-08-04 在仓库 `outputs/` 首次发布时的身份；2026-08-05 当前 Benchmark
路径与重基线 manifest 以“Benchmark 迁移与身份重基线”小节为准，payload SHA 保持不变。

| 资产 | 现场结果 |
| --- | --- |
| `mask_grounded_region_corpus_train_v1_500` | 500 train records；400 target / 100 no-target；五来源各 100；1,800 assets；Silver/专家标注/formal acceptance 均为 false |
| Corpus manifest | `79f8d796d38156ee77dc86bdc2efad2f9992d59090687da2f0bf19d47c52db81` |
| Corpus records | `608df31ae0bd0550caeb4bd7a5c313174647e4ffcb94615b010155a58e173382` |
| Corpus ledger | `c3037a072a448e762f6f157a6cda4517ad3b3ea7496ecaa6126fadc53ab38f82` |
| Corpus annotation queue | `e1ac12fb7573d5426392560c1d9c1a3210ef195bf4a0f93958aee8f03ef876f4` |
| `oa_grounded_eval_dev_v1_100` | 100 unique val baselines；80 target / 20 no-target；340 total records；760 assets；train/val sample 与 parent 交集均为 0；排除 7 个 train-parent-overlap val rows |
| Eval-dev manifest | `bec0bc71972889ef23935e7e00dc9d0b4531b468c42b98e4a2b6a602146a62e5` |
| Eval-dev records | `ca01e944b69fcdc5eb69a8685091044927a984874a36fab67f2b0a2e7ee8ce9d` |
| Eval-dev ledger | `26ce0913510547cb0ca3e4f5967063078b9b8044c83d959d990086c1bb2e7f2a` |
| Eval-dev annotation queue | `261a21527d64cf08c725b9b82dc743e1675c8513e40587ec9c63b8735579b22d` |
| Counterfactual groups | 80 groups；baseline/empty/shift/context-removal 完整；合法 mask-swap 为 0；文件 SHA `bc4d12ef045b6dc7452169dfcf439cdf4e9ce15c3ed3ef8164099d18cc1d0718` |

Eval-dev 明确为 development-only，`sealed_test_accessed=false`、
`expert_annotation_completed=false`、`formal_acceptance=false`。没有生成伪 Gold，
没有用 VLM 填写专家字段。

### 验证结果

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Stage 4 新永久测试 | 0 | 28/28；binary mask、crop 重建、消息顺序、split 隔离、反事实、严格 parser、annotation、tamper/path/link 与 evaluator |
| 既有 Phase 4 回归 | 0 | 82/82；argparse 错误文本来自预期负向用例 |
| 既有 Phase 3 回归 | 0 | 45/45 |
| Python compileall | 0 | `oa_groundrag` 与 `scripts` 全部通过；cache 定向写入任务 `/tmp` 根 |
| `git diff --check` | 0 | 无 whitespace 错误 |
| 冻结 Stage 4A 真实只读 `validate` | 0 | 500 records / 2,200 assets；manifest/records/ledger/ordered-ID 身份一致 |
| 正式 train Corpus build + 独立 validate | 0 | 500 records / 1,800 assets；逐文件及程序事实验证通过 |
| 正式 val Eval-dev build + 独立 validate | 0 | 100 baselines / 340 records / 760 assets；split、parent、反事实验证通过 |
| Gate B 正确参数只读 verifier | 0 | 256 paired；selection、predictions、scores、report 与六项既有判据一致；未重跑 Gate |
| `render-messages` 临时 smoke | 0 | 340 messages；`model_invoked=false`，未调用 Qwen 生成 |
| `export-annotation-queue` 临时 smoke | 0 | 340 items；ledger 绑定导出通过 |

Gate B verifier 的前两次调用仅因 protocol 路径和 expected SHA 参数误传而在验证前拒绝；
修正为已发布 protocol 文件及其 SHA 后只读验证通过，没有写入 Gate B 根。

### 既有资产复核与清理

以下交付后 SHA 与开始时冻结身份一致：

- Stage 4A manifest / records / ledger：`37aebb9c...` / `36f5dcbf...` / `0b0f7a14...`；
- OA Benchmark manifest / index：`9a3b1478...` / `38987722...`；
- OA `checkpoint_best.pt`：`672d39ab...`；
- RS-GeneralDesc `metadata/hashes.json`：`55ac26d9...`；
- RS-General Adapter step-1000 权重：`a367e39c...`；
- Gate B protocol / selection / Base predictions / Adapter predictions / paired scores / report：
  `8378f6f1...` / `98290aaa...` / `862759c4...` / `a7c791fc...` /
  `64d6802e...` / `b150de8e...`。

本任务唯一临时根 `/tmp/oa_groundrag_stage4_v2_20260804` 已精确删除并确认不存在；
其中只含临时 message/annotation smoke 输出和定向 Python cache。仓库根未遗留 debug、
scratch、临时 PNG/JSON 或临时测试目录；永久 tests 和两个正式新输出根保留。

### 当前阻塞、未运行项与下一步

Region/Eval 工程合同、正式 train/val-dev 资产、validator、v2 message/parser 和开发
evaluator 已完成。后续标注已由下节的“单专家最小可训练标注”方案接管：不再要求多人
review、adjudication、Gold、正式阈值或正式 OA-GroundedEval 接受协议。本次工程结论仍
不得解释为专家核验完成、训练语料完成或 Stage 4 科学验收通过。

截至该节 Region/Eval 工程闭环时未运行 GPU、训练、optimizer/backward 或正式 Qwen 生成；
此后首次单专家 calibration 曾运行 GPU 并产生 20 条 invalid 草稿，处置见下节。全程仍未运行
Base/Adapter mask-grounded baseline、sealed test、任何 Gate、RAG、Teacher Silver、Gold、
外部 API、模型/数据/依赖下载、commit 或 push。

## Stage 4 Benchmark 迁移与单专家一键标注闭环（2026-08-05）

### 现场与简化边界

- 当前续作开始分支 / HEAD：`main` / `a1efc73ce8d6e916c6e8aa4342e422e6684b488c`；工作区已包含
  中断前尚未提交的 Stage 4 单专家实现和本地模型目录，本次全部保留，未 commit/push。
- 算法文档仍为冻结 SHA-256
  `dbf93a4ceea1ae973974fb039d5bd2c35d8a62b615ce2add3ab1583f91366d36`；
  本任务未修改算法文档、`AGENTS.md` 或 archive。
- train 结果只称 `expert_verified_train_supervision`；val baseline 结果只称
  `single_expert_dev_reference`。二者都不是 Gold 或专家共识，所有 package/report
  强制 `formal_acceptance=false`、`scientific_acceptance=false`、
  `thresholds_frozen=false`、`sealed_test_evaluated=false`。
- 配置绑定的本地目录 `models_zoo/Qwen3-VL-8B-Instruct-r60595eb` 已就绪，并验证为
  `Qwen/Qwen3-VL-8B-Instruct` revision
  `60595ebc30ec8e3b1d3b9e65d4943ca011c0006a`。本任务没有修改模型目录，也没有自动下载、
  改用 2B 或调用 API。
- 显存查询和容量门槛已从活动工作流删除；显存释放与调度由负责人在运行命令前处理。
  Benchmark 迁移阶段只做 CPU 测试；随后两次草稿运行分别暴露结构 invalid 和模板复制问题，
  旧零核验 work 根均已按授权废弃。当前信息化修复已重新加载本地 8B 并只生成新一轮 20 条
  calibration，实时身份和质量分布见本节末尾。

### 实际代码、合同与入口

新增：

- `oa_groundrag/landslide_evidence/single_expert.py`：严格 project/assignment/draft/
  verified 合同、seed `20260804` 的 20 条 train calibration、可恢复原子工作快照；
- `single_expert_drafting.py`：只允许冻结本地 Qwen3-VL-8B 身份的一次 greedy 草稿，
  复用现有 Qwen processor/model/collator，calibration 完成前拒绝 remaining；
- `single_expert_workbench.py`：仅绑定 `127.0.0.1` 的最小 Gradio 专家核验界面；新增
  `calibration / remaining / all` partition 与 `pending / all` view，pending 只遍历已有草稿
  且尚未核验的记录，invalid 草稿使用严格空模板，核验后自动前进；一键模式在当前分区
  全部核验后自动关闭服务；启动前仅在当前进程精确补全 loopback proxy bypass，启动异常
  转为结构化 reason code 并关闭已部分启动的 app；
- `single_expert_workflow.py`：固定 Benchmark/prompt/config 路径的一键状态机；首次停在
  20 条 calibration 人工边界，第二次推进 480 条，500/500 后自动发布并验证 package 与
  training messages，支持生成和部分发布中断恢复；calibration 使用项目创建时的 prompt，
  仅在负责人第二次调用确认边界才将仓库 prompt 原子同步到 work 根并冻结 prompt/config；
- `single_expert_package.py`：完整 500 train 或 100 val-baseline 单专家 package 的原子
  发布、ledger 和独立验证；
- `single_expert_training.py`：500/500 train-only messages 导出、严格重载及
  `MaskGroundedTrainingMessageDataset`；
- `scripts/stage4_landslide_evidence/run_single_expert_annotation.py`：增加只接受可选
  `--port` 的 `run-train-workflow`，并保留六个细粒度薄子命令；活动单专家入口不再接受
  `annotator_id`；
- `configs/stage4_landslide_evidence/single_expert_prompt_v1.txt` 与
  `single_expert_qwen3vl_8b_v1.yaml`；
- `tests/stage4_landslide_evidence/single_expert_fixture_helpers.py` 及四个
  `test_single_expert_*.py` 永久 CPU 测试。

增量修改：`region_pipeline.py` 暴露共享 asset identity；`region_validation.py` 增加不读
Benchmark shard 的发布 ledger 全文件验证，并重算 Corpus/Eval ID、严格绑定 Eval→Corpus
root/manifest/records/ledger；`phase4/outputs.py` 集中提供可通过严格 parser 的 target/no-target
模板和完整嵌套字段合同，并加固中英文禁止结论局部否定；`phase4/messages.py` 把完整类型、
合法英文 enum、no-target 规则和模板注入 v2 prompt；`phase4/grounded_evaluation.py` 接受
单专家 dev reference，但只计算辅助结构化一致率；
`README.md` 只增加稳定入口；既有 Stage 4A、Gate B 和 external generic 路径未改语义。

冻结 schema：

- `oa_groundrag.mask_grounded_region.expert_verified_annotation.v1`
- `oa_groundrag.mask_grounded_region.expert_verified_package.v1`
- `oa_groundrag.mask_grounded_region.training_message.v1`
- `oa_groundrag.mask_grounded_region.training_messages.v1`
- `oa_groundrag.mask_grounded_region.train_workflow.v1`
- `oa_groundrag.mask_grounded_region.train_workflow_state.v1`
- `oa_groundrag.mask_grounded_region.{annotation_project,annotation_assignment,annotation_work,model_draft,model_draft_failure,model_draft_run,draft_config}.v1`

活动单专家 annotation/work 快照固定写入 `annotator: "expert"`；严格拒绝旧
`annotator_id`、其他 annotator 值和未知字段。既有通用多角色 annotation v1 保持不变，
一键流程不调用它。

唯一 train 一键入口为 `run-train-workflow [--port PORT]`。六个细粒度 CLI 仍为：
`create-annotation-project`、`generate-annotation-drafts`、
`serve-annotation`（要求 `--partition`，`--view` 默认 `pending`）、
`export-verified-annotations`、`validate-annotations`、
`export-training-messages`。不提供 review、adjudication、Gold、protocol freeze 或 API
provider 命令。

### Benchmark 迁移与身份重基线

- 同一设备号 `2096` 上用目录 rename 将两个新版资产移动到：
  - `/home/yukun80/codes/benchmark/oa_grounded_stage4_v1/region_corpus/mask_grounded_region_corpus_train_v1_500`；
  - `/home/yukun80/codes/benchmark/oa_grounded_stage4_v1/eval_dev/oa_grounded_eval_dev_v1_100`。
- 旧 Region/Eval 数据根已消失，未保留副本、symlink 或 alias；冻结 Stage 4A
  `outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500` 原位保留。
- Corpus manifest 从 `79f8d796d38156ee77dc86bdc2efad2f9992d59090687da2f0bf19d47c52db81`
  重基线为 `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e`；
  `corpus_id=74ed1ea9d207c084726781bb3de661a10e1d49a5bb493b58fd5bd1ea95eac342`。
- Eval manifest 从 `bec0bc71972889ef23935e7e00dc9d0b4531b468c42b98e4a2b6a602146a62e5`
  重基线为 `fed7d8b99e4482da1a9e8553c2779cd64007a710fa10404bf2985e96f1ce7492`；
  `eval_id=188b84cc6a1379868d01526df65298fdabb2473865da4a018fa7c649418deea7`，
  并绑定新 Corpus manifest。
- Corpus config file/semantic SHA 为 `fbbc47f7da454cb7492631d2f36fcd01dd0915ffa50b53b97ad509cd1b1a42c7` /
  `08cefb21a3d1ca2b699886b6262d9e8a5eb392244fcf3dc88a120999d5b9e3d2`；
  Eval config file/semantic SHA 为 `5b2d8e3e818fc22b7a512a27bb1029d50608c408bf4058379d02610554116a46` /
  `e2466ec7a87a7fa91c7a21c5968ec34eb6e1971bb30cb61a0adc84d970d81c16`。
- 仅两个 manifest 的配置派生字段发生变化。Corpus records/ledger/queue/guideline SHA 仍为
  `608df31a...` / `c3037a07...` / `e1ac12fb...` / `d08e8638...`；Eval 对应 SHA 仍为
  `ca01e944...` / `26ce0913...` / `261a2152...` / `d08e8638...`。
  排除 manifest 后的文件数与 path/size/mtime 聚合分别保持 `1804`、`842eaaba...` 和
  `765`、`b3dc93fd...`；迁移后 validator 又逐文件重算 SHA/size/pixel 并通过。
- 可恢复 work、annotations 和 training_messages 父目录已建立，但三个正式子根均不存在；
  没有创建 annotation project、package 或 training messages。

### 本地 8B 身份与逐条工作台增量

- 官方 Hugging Face cache verify 检查 16 个发布文件并通过；本地严格 JSON、index 和
  safetensors 结构审计发现 4 个分片、750 个 tensor，未发现 missing、extra、错分片、
  partial/lock、symlink 或 hardlink；离线 `AutoConfig` / processor 加载通过。
- 四个权重分片 SHA-256 依次为：
  `d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa`、
  `8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5`、
  `83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192`、
  `0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5`；
  均与官方身份一致。
- 正式 train Corpus 的 20 条 calibration 在全局 assignment 中分散于 ordinal
  `5, 30, 73, 92, 116, 148, 170, 187, 209, 244, 263, 281, 315, 322, 362, 388, 414, 436, 449, 483`；
  工作台现在按过滤后视图序号逐条导航，不再遍历未生成草稿的其余 assignment。
- `.gitignore` 继续精确排除 `/models_zoo/Qwen3-VL-8B-Instruct-r60595eb/`；已失效的仓库内
  `/annotation_work/` 规则已删除，因为活动工作根位于仓库外的统一 Benchmark 根。

### 验证、身份与未发布项

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| `python -m unittest discover -s tests/stage4_landslide_evidence -v` | 0 | 58/58；新增覆盖模型消息无答案模板、四级草稿质量、表单/canonical/高级 JSON 双向同步、no-target 控件锁定、低信息最终答案拒绝和模板复制 UI 前门控，并保留固定 expert、Benchmark、一键恢复、Region/Eval/Stage 4A 回归 |
| `python -m unittest discover -s tests/phase4_rs_vlm -v` | 0 | 82/82；argparse 输出来自预期负向用例 |
| `python -m unittest discover -s tests/phase3_rs_generaldesc -v` | 0 | 45/45 |
| `python -m compileall oa_groundrag scripts` | 0 | 全部通过；cache 定向写入本任务 `/tmp` 根 |
| `git diff --check` | 0 | 无 whitespace 错误 |
| localhost/Gradio CPU smoke | 0 | 在 `NO_PROXY=no_proxy=127.*` 下自动得到 `127.*,127.0.0.1,localhost`；新版混合界面仅绑定 `127.0.0.1:17869`，启动后立即关闭，未加载模型 |
| 一键 workflow synthetic 回归 | 0 | CPU-only；首次停在 calibration、第二次 remaining、package 后中断恢复、500 messages 发布、重复运行幂等；未启动监听服务或真实模型 |
| 新 Benchmark Corpus 完整验证 | 0 | 500 records / 1,800 assets；源、GT、几何、crop、overlay、ledger 与重算 ID 全部通过 |
| 新 Benchmark Eval-dev 完整验证 | 0 | 100 baseline / 340 records / 760 assets；严格绑定新 Corpus，split/parent/反事实/重算 ID 全部通过 |
| 冻结 Stage 4A 真实只读验证 | 0 | 500 records / 2,200 assets；原路径和冻结身份不变 |

当前 prompt 文件 SHA-256 为
`7a1556e4a4069cb01c366e784bb7fd4c24cc341c21b1c0dd8adc4b0467f86f4e`；draft config 文件
SHA-256 为 `6921decd3766dde578be78ada8a0f6364de8dfebda337237451aac6058524798`。
本任务没有发布 annotation package 或 training messages，因此不存在可报告的正式新产物
manifest/records/ledger 身份；不得把 synthetic 测试或 metadata-only 临时 project 当作产物。
本任务唯一临时根 `/tmp/stage4_benchmark_migration_20260805.BYbXsX`（manifest 备份、迁移
脚本和定向 Python cache）已精确删除并确认不存在；仓库根没有遗留临时 PNG/JSON、debug
脚本、scratch 或 annotation 工作目录。

### 首次 calibration 失败现场、修复与授权处置

- 负责人首次运行 `run-train-workflow` 后，本地 Qwen3-VL-8B 正常加载并逐条完成 20 条
  calibration；运行身份为 `draft_run_1bf65db8ab260e2a7c2ff16d`，当次 prompt SHA-256 为
  `6930dd2f34d878f2bc5afe23775639d1dc96ff0432f2c9e8c4d24f3222ff44f3`。
- 删除前状态严格复核为 `drafted=20 / valid_drafts=0 / failed_drafts=20 / verified=0`，
  `workflow_state.phase=calibration`；20 条首个结构化失败码均为 `INVALID_ENUM`。原始输出均已
  落盘，但消息只声明顶层字段，没有给 8B 完整嵌套结构和英文 enum，故这些草稿不具备训练资格。
- Gradio 已绑定并打印 `127.0.0.1:7860`，随后其 HTTPX `startup-events` 自检因现场
  `NO_PROXY/no_proxy` 只有 `127.*` 而未精确命中 `127.0.0.1`，请求误入 HTTP 代理并返回
  503；这与 CUDA、模型加载、端口占用或 UI callback 无关。
- 删除前确认 `verified/` 无记录，正式 annotation package 与 training messages 根均不存在；
  Corpus/Eval manifest 仍分别为 `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e` /
  `fed7d8b99e4482da1a9e8553c2779cd64007a710fa10404bf2985e96f1ce7492`。
- 负责人已明确选择废弃本次失败 work 根并从修复后的 prompt 重新开始，不保留副本、symlink
  或兼容 alias。删除前再次确认精确路径为
  `/home/yukun80/codes/benchmark/oa_grounded_stage4_v1/work/stage4_train_expert_v1`、普通目录、
  无 symlink、20 draft / 0 verified，且两个发布根不存在；随后只删除该 work 根并确认其
  不再存在。删除不可恢复；当次处置结束时尚未创建新 project。
- 修复后 `phase4/outputs.py`、v2 message、repo prompt 与 UI 空模板共享同一严格合同；target
  与 no-target 模板均可直接通过 parser，enum 保持英文 ASCII，不进行宽松修复。calibration
  prompt 副本不被仓库文件漂移改写；仅在 20/20 后第二次调用时原子同步并冻结。
- Gradio 启动前只修改当前 Python 进程的 `NO_PROXY/no_proxy`，保留既有条目并精确追加
  `127.0.0.1,localhost`；启动失败会返回 `ANNOTATION_UI_START_FAILED` 并关闭 app。真实
  localhost CPU smoke 已在模拟 `127.*` 环境中启动并立即关闭，未再出现 503。
- 删除后 Corpus/Eval manifest SHA 复核仍为
  `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e` /
  `fed7d8b99e4482da1a9e8553c2779cd64007a710fa10404bf2985e96f1ce7492`；正式 annotation
  package 和 training messages 仍不存在。
- 本次修复唯一临时根 `/tmp/stage4_invalid_http_fix.0jdOii`（定向 Python cache 与短暂
  Gradio 临时文件）已按精确路径删除并确认不存在；未在仓库根留下临时脚本、图片或 JSON。

### 第二次 calibration 模板复制现场与信息化草稿修复

- 负责人在上述 JSON/HTTP 修复后第二次运行一键流程，本地 8B 又完成 20 条 calibration；
  运行身份为 `draft_run_4d1ddcb3b9204545514a9ca3`，工作副本 prompt SHA-256 为
  `87e3c1bae916ca957bf56de49e0851dbdf1f3e54d9598138541838b6bbb924ed`。
- 本次 20 条均通过严格 JSON parser，且 `verified=0`；新增语义质量重算得到 16 条 target
  `low_information` 且全部与 target 空模板逐字段等价，4 条 no-target 为合同要求的
  `not_applicable_no_target`。因此 `valid_drafts=20` 只表示结构合法，不能解释为描述质量通过。
- 当时处理器预检确认单条消息实际消费 3 幅图，输入 token 数为 2279，视觉 grid 依次为
  `[1,14,14]`、`[1,14,14]`、`[1,18,4]`；模型确实接收 full、binary mask 和 crop，模板复制
  不是缺图或 tokenizer 截断导致。
- 根因为正式 v2 contract 和单专家 prompt 同时向 greedy 生成暴露完整 target/no-target
  答案模板，并强调“以模板为起点”；8B 选择逐字复制最安全的合规答案，而没有把视觉观察
  写入字段。该问题属于语义草稿失败，不是 parser、CUDA 或 HTTP 失败。
- 当前修复从正式模型消息完全移除 `json_template` 和完整答案，只保留字段、类型、英文 enum、
  禁止结论与逐维观察问题；人工空表单和 invalid 回退仍独立复用严格模板。新增可重算的
  `informative / limited_but_specific / low_information / not_applicable_no_target` 质量诊断；
  target 空模板复制会在启动 UI 前以 `DRAFT_QUALITY_FAILED` 停止，其他低信息草稿仍可由专家
  修正，但低信息最终答案不得标记 `expert_verified`。
- Gradio 专家区改为分组字段表单、始终可见的只读 canonical JSON，以及折叠的高级多行 JSON
  Textbox；支持严格双向同步、数组逐行输入、no-target 控件锁定和窄幅 crop 警告。模型草稿、
  程序事实、质量状态与 audit-only overlay 均保持只读或审计隔离。
- 删除前现场再次严格核对：精确 work 根为普通非链接目录；只有上述一个 run；20 drafted、
  20 parse-valid、16 target template copies、0 verified；正式 annotation package 与 training
  messages 均不存在。仓库中修复后的 prompt SHA-256 为
  `7a1556e4a4069cb01c366e784bb7fd4c24cc341c21b1c0dd8adc4b0467f86f4e`。CPU 回归全部通过后，
  已只精确删除该零核验 work 根且确认不存在；没有保留副本、symlink 或 alias。删除后 Corpus、
  Eval-dev、Stage 4A manifest 仍分别为 `d18e6b4f...`、`fed7d8b9...`、`37aebb9c...`。
- 修复后 processor 只读预检仍为单条 3 图，输入 1925 tokens（上限 4096），视觉 grid 为
  `[1,14,14]`、`[1,14,14]`、`[1,18,4]`。随后一键命令只重新生成 20 条 calibration，
  新 run 为 `draft_run_4dab88a787a3eda6e7056706`，绑定的新 prompt SHA 与仓库当前值一致。
- 新 run 共 20 drafted / 0 verified：6 `informative`、7 `limited_but_specific`、1
  `low_information`、2 `not_applicable_no_target`、4 parse-invalid，且
  `target_template_copies=0`。16 条 parse-valid canonical JSON 长度为 813–1109 字符，中位数
  903；不再是保守模板复制。唯一 low-information target 的具体问题是缺少周围环境观察，仍可
  在 UI 中由专家补充，但在修正前不能核验完成。
- 4 条 failure 均为 `INVALID_MODEL_OUTPUT`：两条 no-target 错误填写了必须为空的区域数组，
  两条输出缺少顶层 `target_status`。按单次生成合同不 repair、不重试；UI 保留 raw output 并为
  这些记录加载严格空表单，由专家核验填写。
- 质量门控确认模板复制数为 0 后，正式核验服务曾启动于 `http://127.0.0.1:7860`，当时命令
  保持运行并等待专家操作。只读访问 live `/config` 确认存在始终可见的 canonical JSON、草稿
  质量 JSON、分组表单和高级 Textbox，`gr.Code` 组件数为 0。未代替专家点击任何核验按钮。
- 本次唯一临时根 `/tmp/stage4_informative_ui.Pg5laD`（定向 compile cache 与 synthetic
  localhost smoke）已精确删除并确认不存在；正式 active work 根必须保留用于断点恢复。

### Stage 4 v2 8,450 条模型辅助监督扩容

#### 服务收尾与只读来源

- 开始时只读确认 PID `2710557` 精确对应
  `run_single_expert_annotation.py run-train-workflow`，cwd 为本仓库且只监听
  `127.0.0.1:7860`；发送一次 `SIGINT` 后进程正常退出，端口已释放，未使用宽泛 `pkill`。
- v1 工作根保留 500 assignments、20 drafts、5 `expert_verified` 和一个 draft run；停止前后
  13 个文件的 path/size/SHA 聚合一致，没有删除、迁移、覆盖或创建 alias。
- 导入来源根的 13 文件身份由 v2 provenance 完整冻结，
  `files_root_sha256=51da12bcf63dced9b85ddfe6219f036218a7d708a056de882eec58107def4a6e`；
  其中 prompt SHA 为 `7a1556e4...`，draft run 为
  `draft_run_4dab88a787a3eda6e7056706`。新增、删除、symlink、hardlink、size 或 SHA 漂移都会使
  v2 project/package validator 失败。

#### 代码与合同

- `expanded_region.py`：train-only 7,950 确定性分层选择、GT Region 资产原子发布、严格
  validator 与不复制旧图像的 8,450 collection；seed 为 `20260805`。
- `region_pipeline.py`：共享 OA Benchmark reader 改为默认最多 8 条样本的 LRU，避免数千条
  image/mask tensor 常驻内存；既有 v1 输出语义未变。
- `model_assisted.py`：collection 消费、旧 20/5 无损导入、单 runtime/batch-1 断点生成、
  expert 优先、模型草稿准入/exclusion、混合 supervision package、动态 training messages
  和 Dataset。
- `model_assisted_workflow.py`：固定 v2 路径的准备、恢复、生成、部分发布恢复和只读复核状态机。
- `run_model_assisted_supervision.py`：仅暴露 `prepare-expanded-corpus` 与
  `run-train-workflow` 的薄 CLI；不查询显存、不调用 API、不启动 UI。
- 新增冻结配置 `region_corpus_train_extension_v2_7950.yaml`，并新增三组永久回归
  `test_expanded_region.py`、`test_model_assisted.py`、`test_model_assisted_workflow.py`。
- 新 schema：
  `oa_groundrag.mask_grounded_region.train_collection.v2`、
  `oa_groundrag.mask_grounded_region.train_supervision_record.v2`、
  `oa_groundrag.mask_grounded_region.train_supervision_package.v2`、
  `oa_groundrag.mask_grounded_region.training_message.v2`、
  `oa_groundrag.mask_grounded_region.training_messages.v2`。模型未复核项只能写成
  `model_generated_unreviewed`；全链路固定 `reference_authority=mixed_model_and_single_expert`、
  `gold=false`、`formal_acceptance=false`、`scientific_acceptance=false`。

#### 正式 CPU 产物

- extension root：
  `/home/yukun80/codes/benchmark/oa_grounded_stage4_v2/region_corpus/mask_grounded_region_corpus_train_extension_v2_7950`
  - manifest `a26f4267ba12fad8ac39481dcd16dd40a65dacdec7133453847cb2e2c71d43fe`
  - corpus ID `3294404bf1e0091e44d84b3cbc815b707307796af11a5d0a1ddde17874f6c193`
  - records `8ff4871d41b7b5d01cdb0b2ae4e4caae159317a35610385aec6120e9f2982304`
  - annotation queue `301add26a10010ecc93bc1fff75c68f038e090d70415e760e5f02a4dfaa2e4cd`
  - ledger `bad88b4b505ca55ca34377c3abc2281829f78c84970376d7b6f17364db16012d`
  - 7,950 records / 30,402 assets / 1,834,018,875 asset bytes；target/no-target 为
    `7,251/699`。
- extension 每来源均为 1,590；target/no-target 分别为：
  `gdcld 1535/55`、`lmhld 1482/108`、`landslidebench_agent 1367/223`、
  `landslide4sense 1277/313`、`multimodal_landslide 1590/0`。最终加上 base 后 no-target
  均不超过 338；`landslidebench_agent` 排除 11 个 Eval parent 后的 1,590 个 eligible
  train sample 全部使用。
- collection root：
  `/home/yukun80/codes/benchmark/oa_grounded_stage4_v2/region_collection/mask_grounded_region_train_collection_v2_8450`
  - manifest `cd2b86f6244f4f5f42d846166f11a34efdb9edd636239039b42444c453e435d2`
  - collection ID `62a67b3d6c5454ffcf58d58c68f6627fe63dee65a2b1a183fcfcd5d058e0aeb1`
  - member index `c4190db5bae68564b0a7b6e05df2445b0e070e7cbb3ff7c8324df36ad199db69`
  - ledger `11784175b9c9793083c0e98e7cc55bbfb5557f8c61ca62f1f6aab0c040a5652b`
  - base/extension 为 `500/7950`；五来源各 1,690；target/no-target 为 `7,651/799`；
    ordered record/sample SHA 为 `965966d2...` / `6d46a4a8...`。
  - 强绑定 Eval-dev manifest `fed7d8b9...` 及 100 个 baseline parent；collection 与其 parent
    零交集，未读取 test 或 sealed test。
- v2 work root：
  `/home/yukun80/codes/benchmark/oa_grounded_stage4_v2/work/mask_grounded_region_model_assisted_train_v2_8450`
  - project ID `model_assisted_project_2808706e63465ea4223f1233`
  - project SHA `3910d7d7d580ef0e10138ec5b5ff2626aa8c7ffd44795a13e07b4301d118475b`
  - import provenance SHA `afcb18ef03d928e13c1694804aa8d7ba2c5248a5014c7dd485b70aacc8e4b96d`
  - 状态为 `total=8450 / drafted=20 / valid=16 / invalid=4 / verified=5 / pending=8430`。
- supervision package 与 training messages 两个正式根均不存在；本次没有用合成内容补造。

#### 验证与边界

- Stage 4 永久测试：`75/75`；其中本次新增扩容/模型辅助/workflow 共 `17/17`，并永久覆盖
  train extension 深验不得打开 val source shard。
- Phase 4 RS-VLM：`82/82`；Phase 3 RS-GeneralDesc：`45/45`。
- 新 extension 在正式发布时以 `verify_source=true` 完整重放选择、GT 几何、PNG/像素、crop、
  overlay、ledger 和 identity；collection 严格验证 base/extension/Eval 绑定。
- 旧 Stage 4A、v1 Region 和 v1 Eval-dev 真实只读 validator 均通过，manifest 仍为
  `37aebb9c...`、`d18e6b4f...`、`fed7d8b9...`。
- `compileall` 与 `git diff --check` 通过；本次唯一 cache 根
  `/tmp/stage4_model_assisted_20260805.tM3yVK` 已精确删除并确认不存在。
- 未运行：剩余 8,430 条 Qwen 生成、训练/optimizer/backward、val/test/sealed-test、RAG、
  Teacher Silver、Gold、外部 API、任何 Gate、正式科学评价、commit 或 push。

### 当前阻塞与下一条命令

当前工程准备完成，唯一待执行长任务是负责人释放显存后启动本地 Qwen3-VL-8B，为缺失的
8,430 条 train record 逐条生成一次草稿。命令会验证并复用当前 collection/work，跳过已导入
20 条；全部草稿落盘后，自动按 expert/model/exclusion 规则发布 supervision package，再按
实际 eligible 数量发布 training messages。它不会打开 Gradio，也不会把未复核模型草稿写成
专家核验或 Gold：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/stage4_landslide_evidence/run_model_assisted_supervision.py \
  run-train-workflow
```

## Stage 4 模型辅助生成完成与 Stage 5 开始（2026-08-08）

负责人已在本地完成剩余 8,430 条逐条 Qwen3-VL-8B 生成。现场只读复核确认：

- draft run：`draft_run_ed39d5ec0d946b025910d63f`；总计 8,450 drafts；
- 新生成 8,430 条中 parse-invalid 1,470；全量最终排除 1,476 条，其中
  `parse_invalid=1474 / generic_low_information=2`；
- supervision package：8,450 source / 6,974 eligible / 1,476 excluded；
- eligible authority：`expert_verified=5 / model_generated_unreviewed=6969`；
- package manifest SHA-256：
  `dd8b58e57e621599028758b65a0001cc35115387329c2371d6bfd3dfedcb5de2`；
- training messages manifest SHA-256：
  `370c0727bd64ea0cb363f2a4efed21d04ccf334cf2c6c3fe8ed3e1cc9304579a`；
- messages JSONL SHA-256：
  `434a2c86abbe6508badc44fdc929ae3d6a8b8bd0984fdcd8aa201fa3742afabf`；
- collection manifest 仍为
  `cd2b86f6244f4f5f42d846166f11a34efdb9edd636239039b42444c453e435d2`；
- prompt SHA 为 `7a1556e4a4069cb01c366e784bb7fd4c24cc341c21b1c0dd8adc4b0467f86f4e`，
  8B revision 为 `60595ebc30ec8e3b1d3b9e65d4943ca011c0006a`。

该 6,974 条资产是 `mixed_model_and_single_expert` 监督，不是 Gold。负责人进一步授权
Stage 5：先发布不依赖 raw work/package 的 compact 训练合同，再精确删除旧 work、package
和被替代的 v2 messages；以 Gate-B-accepted RS-General Adapter step-1000 为 LoRA
warm-start，在全新根使用 90% Region / 10% RS-General replay 训练。Region 数据按 parent
90% train / 10% monitor，含 5 条 expert 记录的 parent 强制进入 train；retention 只报告，
不设置通过阈值。OA-GroundedEval-dev 不建设人工 reference，仅做自动合同与反事实开发评价；
所有新报告继续固定 `formal_acceptance=false / scientific_acceptance=false / sealed_test_evaluated=false`。

本次开发开始时为 `main@912c5ce92a965ffda06f84cd924d3cd9e84c23ea`，工作区干净，
没有活动生成或训练进程。算法文档 SHA-256 仍为
`dbf93a4ceea1ae973974fb039d5bd2c35d8a62b615ce2add3ab1583f91366d36`，正文不修改。

### Stage 5 compact 发布与 raw 清理前冻结证据

已原子发布并全量验证新的独立 compact 根：

- root：`../benchmark/oa_grounded_stage4_v2/training_messages/mask_grounded_region_compact_training_messages_train_v3_6974`；
- schema：`oa_groundrag.mask_grounded_region.compact_training_messages.v3`；
- count：`6974`；compact ID：
  `compact_678ae42424619323115a0f0eb9ea2020304b568c5be725bfcb2d17581bfdcebe`；
- manifest SHA-256：`746f641f1fbe48f4301ffc0c52b586437a1dc0b68a5add4be1e3db50d69a1184`；
- messages SHA-256：`8fcf1dbfb57b6a1e0541cac51e273ba86360280142fc45986d889ff6f3752486`；
- ledger SHA-256：`21c9c3a2bb6f87cc57df48d64a02d4a9a64b2f881f511743402ea3579808e74d`。

compact loader 不读取旧 work/package/v2 messages；它只绑定保留的 8,450 collection，
逐条重算 6,974 个 GT-mask asset identity、full/mask/crop 消息和 canonical assistant JSON。
真实 Qwen processor/collator smoke 为 3 images、1,739 input tokens、442 supervised tokens，
role=`mask_grounded_train`。旧 raw 精确删除前身份如下：

- work：36,054,190 bytes；project SHA-256
  `3910d7d7d580ef0e10138ec5b5ff2626aa8c7ffd44795a13e07b4301d118475b`；
- annotation package：42,500,378 bytes；manifest SHA-256
  `dd8b58e57e621599028758b65a0001cc35115387329c2371d6bfd3dfedcb5de2`；
- v2 messages：55,594,009 bytes；manifest/messages SHA-256 分别为
  `370c0727bd64ea0cb363f2a4efed21d04ccf334cf2c6c3fe8ed3e1cc9304579a` /
  `434a2c86abbe6508badc44fdc929ae3d6a8b8bd0984fdcd8aa201fa3742afabf`。

三根均为普通非 symlink 目录，未发现 symlink、hardlink 或活动生成/训练进程。按负责人
明确授权，下一动作仅精确删除上述三个 raw 根；Region base/extension/collection、图像、
Eval-dev、Stage 4A、RS-General Adapter 与 Gate B 继续保留且只读。

### Stage 5 首次正式运行与 CUDA OOM 阻塞（2026-08-08）

- compact 全量 validator、真实 Dataset/Collator smoke 通过后，已按上述精确清单删除旧
  work、annotation package 和 v2 training messages；删除后再次完整验证 compact，确认其
  不依赖三个 raw 根。保留的 Region base/extension/collection、图像和 Eval-dev 未修改。
- parent split 固定为 `6278 train records / 696 monitor records`、`5784 train parents /
  643 monitor parents`，5 个含专家记录的 parent 均在 train，train/monitor parent 零交集。
  mixed sampler 每个 8,000 micro-sample epoch 精确为 `7200 Region / 800 external_train replay`；
  retention 使用独立冻结的 128 个 `external_val` parent，未把 external_val 用作 replay。
- 正式 Stage 5 工作流根为
  `outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1`。Base 与冻结
  RS-General Adapter 的 340 条 GT-mask 自动 baseline 均已完成并原子发布；两者都未产生可被
  Stage 4 v2 严格 parser 接受的 prediction。Base 为 `0 valid / 340 INVALID_MODEL_OUTPUT`；
  RS-General Adapter 为 `0 valid / 340 failures`，其中 `INVALID_MODEL_OUTPUT=311 /
  UNKNOWN_FIELD=10 / TYPE_MISMATCH=19`。这是真实的零命中开发基线，不是正式科学验收。
- 冻结 RS-General retention teacher-forced loss：Base overall/macro-task 为
  `3.6994192158 / 3.3185156617`；RS-General Adapter 为
  `1.1323244887 / 0.7763413345`。retention 只报告，`retention_gate_frozen=false`，不改写
  Gate B，也不据此阻止 checkpoint 发布。
- warm-start 严格加载旧 step-1000 LoRA
  `a367e39c626338a151dad33e6f7a7f9cc9887206dbcd261d147837e6408becc1`；optimizer、
  scheduler、RNG 和 sampler 均重新初始化，旧 checkpoint 只读。训练完成 step 10 后在下一
  optimizer step 的某个 micro-sample 发生 CUDA OOM；尚未到 step-100 checkpoint，因此没有
  可恢复 checkpoint，Region monitor validation 尚未运行。
- OOM 前 telemetry：step 10 累计 160 samples、236,650 input tokens、451 images；当前 allocated
  约 4.05 GiB、历史 allocated peak 约 7.65 GiB，但 CUDA allocator reserved 已增长到约
  24.91 GiB。CPU 按确定性 sampler 重建 sequence 152--183：step 11 的 Region 样本均为
  1,548--1,703 tokens / 3 images，replay 为 244 tokens / 1 image，全部远低于 4,096-token 与
  5-image 上限。现场证据指向跨可变形状 micro-sample 的 CUDA 缓存/碎片累积，而不是异常样本
  或数据合同越界。
- 负责人已于 2026-08-09 批准“训练前及每个 optimizer step 后精确清理 CUDA cache、
  不改变 4,096-token/5-image/90:10/训练超参数”的内存卫生修复，并明确失败目录后续无消费
  价值时直接删除、不保留 failed-attempt 副本。删除前再次确认 `training/` 只有三个普通
  单链接文件，无 checkpoint、best pointer、validation result 或 training report：
  `config_snapshot.json` 为 4,430 bytes / SHA-256
  `7972cad61b7e889776ba82ef3c32d1a4f00d45035acd6669398e1bdbdc36eda2`，
  `validation_selection.json` 为 171,112 bytes / SHA-256
  `028bbef7c26eababbc7e14c97d0390884cec84ae1b27a91be093e366fe9e1e4d`，
  `train_log.jsonl` 为 1,397 bytes / SHA-256
  `e57d4160ebae8ee85139407f788682feff436dab891922c016d6ea0db69b66d6`。
  该目录不可恢复且不含可用模型权重；下一动作是只精确删除它，然后从同一 warm-start
  step 0 重新训练。未降低输入上限、未更换模型或量化。

### Stage 5 OOM 修复与 20-step GPU smoke（2026-08-09）

- 上述首次失败 `training/` 已按精确绝对路径删除并确认不存在；没有创建 failed-attempt、
  failure manifest、副本或 alias。两个 GT-mask baseline、`retention_losses.json`、compact、
  warm-start checkpoint 和 Gate B 均保留。
- 通用 trainer 新增默认关闭的 CUDA cache cleanup interval；默认 training layout 字节语义
  不变，既有 Phase 3/Gate B checkpoint 继续按原字段验证。Stage 5 固定 interval=1，并将
  `cuda_cache_cleanup_interval_steps=1` 写入自身 checkpoint training layout；不同策略的
  checkpoint/resume 会被拒绝。
- 每个 micro-sample backward 后立即释放 batch、logits/result、loss 和 labels 引用；每个
  optimizer step 后调用一次 `torch.cuda.empty_cache()`。进入 Stage 5 正式训练前另执行
  `gc.collect()` 与 cache cleanup；不修改活跃权重、optimizer 数学、数据顺序或任何冻结
  训练超参数。
- 同一 Qwen3-VL-2B、同一 RS-General step-1000 warm-start、同一 compact split/sampler 的
  20-step GPU smoke 已通过：320 micro-samples，精确 `288 Region / 32 external_train replay`，
  生成临时 step-20 checkpoint。step 10/20 的 CUDA reserved 均约 4.04 GiB；历史 allocated
  peak 约 7.75 GiB，未复现首次运行约 24.91 GiB reserved 后 OOM。临时 checkpoint 不进入
  正式训练或评价，将随本任务唯一 `/tmp` 根精确删除。
- 修复后的 CPU 回归：Stage 4 `77/77`、Phase 4 `92/92`、Phase 3 `45/45`；`compileall` 与
  `git diff --check` 通过。下一动作是复用已完成 Base/RS-General baseline，从相同 warm-start
  重新开始正式 Stage 5 step 0；首个正式可恢复边界仍为 step 100。

### Stage 5 正式训练、自动评价与 retention 闭环（2026-08-09）

- 修复后从冻结 RS-General step-1000 LoRA 重新开始 Stage 5 step 0，正式训练完整到
  `1000/1000`，未再发生 OOM。共消费 16,000 deterministic micro-samples，精确为
  `14,400 Mask-Grounded Region / 1,600 RS-General external_train replay`；replay 七任务计数为
  `global_caption=230 / bbox_region_caption=230 / 其余五类各 228`。累计 23,620,604 input
  tokens、4,375,209 supervised tokens、44,834 images；CUDA allocated peak 为 7.78494 GiB，
  cache cleanup 后 reserved 在训练日志中稳定约 4.04 GiB。
- 每 100 step 保存并在固定 696 条 Region monitor 上验证。step 100--1000 loss 依次为
  `0.395320 / 0.323436 / 0.294082 / 0.277639 / 0.267240 / 0.260312 / 0.255890 /
  0.253285 / 0.252161 / 0.252246`。按最小 Region monitor loss、平手取更早 step 的冻结规则，
  best 为 step 900，而不是最后一步。
- 正式训练身份：training report SHA-256
  `c0751415a2da4ec72c92892975ab713d688aa316a1480b7a82b8ce8e9d5916ab`；best pointer
  `368bc48fec6e5303c80e8b2c0d397f4d55eb8c6d51fe70d9da417833ac8a2c1b`；step-900 manifest
  `c203823597c7b3ecf7f1bce9b3030efe7f5ef2dc5c1ae58bbc58e0982aae5c30`；Adapter
  `858e12ff7e902ce0a3fdfb1a3dfbc2e58ad0892dec870a73fa4fc0a3411f84d7`。checkpoint
  training layout 明确绑定 `cuda_cache_cleanup_interval_steps=1`。
- Mask-Grounded Region Adapter 在 OA-GroundedEval-dev 340 条上得到 `276 valid / 64
  INVALID_MODEL_OUTPUT`。64 条均为严格 no-target 区域数组非空，其中 `11 baseline / 53
  empty_mask`；未自动修复。有效集 schema validity、target-status correctness、binary-mask
  identity、prediction/evidence identity 均为 `1.0`，forbidden claim 与 overlay leakage rate
  均为 `0.0`；但 complete prediction set 为 false，empty refusal、shift/context sensitivity 和
  counterfactual completeness 均为 `0.3375`，不能解释为科学通过。prediction manifest / report
  SHA-256 分别为 `4b090b4392a906817379357d1a8295f8b5eea10339eeb610847bc9bd2ef26a6b` /
  `362e9c0036403382627b677ffedd79125fbcd46ca6ac9d77492f7d9199c07ea1`。
- 128-parent teacher-forced RS-General retention：Base overall/macro 为
  `3.699419 / 3.318516`，RS-General Adapter 为 `1.132324 / 0.776341`，Region Adapter step-900
  为 `1.136405 / 0.804547`。该变化只报告，不阻止 checkpoint。
- 冻结 Gate B selection 上的 256 条 paired retention 已完成；总体 primary delta 相对冻结
  RS-General predictions 为 `-0.00035484`。七任务 delta 为
  `bbox +0.018711 / global +0.004518 / object_count -0.040918 / scene +0.024814 /
  spatial -0.000786 / visible_change -0.010101 / VQA +0.000275`。report / manifest SHA-256 为
  `8f99b28a98bb8d8414e7f9e4e39b6c2ae604eae3cdb7e8c5bd854f4650a9df16` /
  `c35363d3ae6624b6784aa1fe87436c5c3959bb497d5acd62f16b952f0c1bf96e`；ledger 逐文件复核通过。
- retention 首次在 340 条 Region 评价发布后暴露两项路径/身份问题：先误把 frozen protocol
  JSON 当作 static YAML；修正后，正式 Gate B loader 又正确拒绝当前 Stage 5 实现 SHA 与历史
  frozen implementation 不一致。最终保留正式 loader 的严格拒绝语义，新增 Stage 5 专用
  selection consumer：除历史 `implementation_files` fingerprint 外，仍严格验证 frozen
  protocol canonical SHA、selection、Benchmark、shards、monitor exclusion 和 predictions。
  retention 资产显式写入 `selection_authority=frozen_gate_b_selection_only`、
  `historical_gate_b_implementation_match=false`、`historical_gate_b_acceptance_reused=false`；
  不改写或重新宣称历史 Gate B。一次默认沙箱续跑因 GPU 不可见返回 `CUDA_REQUIRED`，没有
  生成资产；随后按已授权 GPU 边界正常续跑完成。
- workflow state SHA-256 为
  `9c5a90743e26576f55a83157e8a1bd3fcf28c1005cc97ff098be1a3b02a62efa`，stage=`complete`；
  `reference_authority=automatic_contract_only`、`expert_metrics_available=false`、
  `retention_gate_frozen=false`、`formal_acceptance=false`、`scientific_acceptance=false`、
  `sealed_test_evaluated=false`。
- 最终回归：Stage 4 `77/77`、Phase 4 `94/94`、Phase 3 `45/45`，总计 `216/216`；
  `compileall`、`git diff --check` 和新评价 ledger 校验通过。compact、Eval-dev、Stage 4A、
  RS-General warm-start 与 Gate B selection 的冻结文件 SHA 未漂移。本任务唯一临时根
  `/tmp/paper7_stage5_oom_fix_20260809.7J4ENs` 已精确删除并确认不存在。
- 未运行：sealed test、RAG、Teacher Silver、Gold、外部 API、Gate A/C/D/E/F、commit 或
  push。当前工程闭环已完成；仍需负责人决定是否针对 64 条 no-target 合同失败开展下一轮
  prompt/supervision 改进，以及未来是否建设专家 dev reference 与冻结 retention/科学阈值。

## Stage 6 Evidence-Constrained Text RAG 工程闭环（2026-08-10）

本次从干净的
`main@eae76528b1fa7aaa1793af5451941d510d4bba29` 开始。Stage 5 的 64 条 no-target
strict-output failure 保持原状，不作为本阶段阻塞项。Stage 6 仅消费 OA-GroundedEval-dev
和 valid Pass-1；未读取 test/sealed test，未运行 OA-AuxSeg，未修改 Stage 5 权重、mask、
Programmatic Facts 或 Pass-1 observation。

### 环境、权重与冻结输入

- `qwen3vl` 环境保持 PyTorch `2.8.0+cu128`、Transformers `5.3.0`、PEFT `0.15.2`、
  NumPy `2.1.2`；Stage 6 新增 PyMuPDF `1.28.2`、RapidOCR `3.9.2`、ONNX Runtime
  `1.28.0`。安装 dry-run 未要求降级或替换 torch/transformers/peft。
- 本地 dense 模型为 `BAAI/bge-m3` 固定 revision
  `5617a9f61b028005a4858fdac845db406aefb181`。使用 `hf 1.26.0` 固定 revision 与
  allowlist 下载；大权重传输停滞后仅通过同一官方 fixed-revision resolve URL 做 byte-range
  续传。`pytorch_model.bin` 为 `2,271,145,830` bytes，SHA-256
  `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`；12 个保留文件的
  identity 为 `8cb129bae94b7ef5f3fe2e8eec3f64527615ba1f5dcebe1ce4f39e4dad26e9a7`。
- BGE 以 `local_files_only=true / trust_remote_code=false` 加载，模型类型
  `xlm-roberta`、dense 维度 `1024`、最大 token `1024`。真实 CUDA smoke 的 3 个中英文
  向量 shape 为 `(3, 1024)`、L2 norm 均为 1、重复编码最大差值为 0，相关段落排名第一。
- RapidOCR 使用安装包自带 ONNX：detector / recognizer / classifier SHA-256 分别为
  `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` /
  `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` /
  `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c`，三模型 identity
  `037b737ea343c5c6997ecdb17d6b5dffe6f3332839f062ebca3c3e05abf1dd8e`；未调用外部 OCR API。
- Stage 5 best 始终从 pointer 动态解析为 step 900。pointer / checkpoint manifest /
  Adapter / workflow state SHA-256 仍为 `368bc48f...` / `c2038235...` / `858e12ff...` /
  `9c5a9074...`。Pass-2 只加载 LoRA trainable state，显式排除 optimizer、scheduler、RNG
  和 sampler。真实本地 Qwen text-only processor smoke 为 0 image / 38 input tokens；原视觉
  inference 的至少一幅图约束保持不变。

### Text Evidence Bank

- Source Registry 显式登记 `docs/RAG_knowledge/` 的 12 个 PDF；registry semantic SHA-256
  `f18a65169af4fa1a8529f8c3d17287f957943080f5801620e391964586070fc0`。原 PDF 全程只读，
  authority/status 不从文件名猜测。
- 全量 page ledger 共 649 页：原生 text-layer 474 页、RapidOCR 可用 163 页、OCR 失败并
  明确排除 12 页，共 637 页可用。12 个来源均产生可用 Evidence Unit。
- 共构建 2,929 units；正式索引 1,283，其中 `interpretation=872 / confounder=138 /
  limitation=273`。另有 1,643 个 unclassified unit 保留审计但不索引；检测 exact duplicate
  22 个，只索引 canonical unit。
- 正式根：`outputs/stage6_text_rag/text_evidence_bank_v1`。Bank ID
  `9322a9139d04be7665feb154153b7dc1c2d35b0871fc32bbd6a6daa942fabb28`；manifest / ledger
  SHA-256 为 `9b891e191581746173a27b80356caa18ec9be5d3c36eaee67444b05a070f0bcc` /
  `1c73fbd6135daaaf4767f8427e7c9e1ba69f09e0223aaad2c3e151a013c1e650`。
- lexical index 为 SQLite FTS5，英文 token 与中文 unigram+bigram 显式分析；dense index 为
  BGE-M3 CLS + L2 normalization 的 NumPy matrix，检索使用 brute-force cosine。Hybrid
  ranking 固定 metadata/modality filter → lexical+dense → RRF (`rrf_k=60`) → 类型 quota。

### 80 条 dev retrieval

- OA-GroundedEval-dev manifest 为 `fed7d8b9...`；Stage 5 Pass-1 manifest / predictions
  SHA-256 为 `4b090b43...` / `84d06ed5...`。只选择 valid 的
  `target_present + baseline_correct_mask`：五来源各 16 条，共 80 条；先按来源排序并
  round-robin 冻结，selection ID
  `fd1d07f6fdd9d66cdbd21d1bd9a5f1045133eb1e47c4f3d5483180840ef95f40`。
- 两个 deterministic builder 共生成 160 queries；80 个 packet 共 480 items，严格为
  `interpretation=160 / confounder=160 / limitation=160`，same-source/page duplicate 为 0，
  全部 source/page/section 可追溯。query 与 rank 重算一致。
- 正式根：`outputs/stage6_text_rag/dev_retrieval_v1`。Retrieval ID
  `e7edfb2ae05a5114c105ce82e7c9bcc87c089dd54d1bd68a5bc43ea860c2f1c2`；manifest / ledger
  SHA-256 为 `da0e207b284bee81824b3542efb8a4b19138c92f07ba09e0afc3f11c4c0b9e7c` /
  `9dd99e6c459ffc91d4e05d604b83a1cc07137dfb4bb532baf7c277c1d839c84d`。
- 没有 retrieval Gold；Recall@K、MRR、nDCG 均明确为 `null`，不据此声称 retrieval 科学通过。

### Text-only paired Pass-2

- Query、Balanced Packet 和 Pass-2 均为版本化严格合同。Task router 只让
  `candidate_interpretation / professional_qa / evidence_constrained_report` 进入 RAG；纯
  scene/region description 不检索，未知任务拒绝。
- Pass-2 输出固定包含 supporting interpretations、alternative explanations、limitations、
  recommended verification、summary，每项为 `{text,evidence_ids}`。生成时 deterministic
  JSON FSM 只约束字段结构和当前 packet 内、类型相容的 citation ID；自然语言 token 仍由
  Stage 5 best generator greedy 生成，validator 直接重解析 raw output，不做输出后修复。
- 从冻结 selection 的前 5 条各取一个来源，每条运行 `no_rag` 与 `text_rag`，共 10 次。
  两模式使用相同 Pass-1、程序事实、问题、generator、prompt 主体、batch 1、greedy 和
  `max_new_tokens=768`，唯一主要差异为 evidence packet。generator identity SHA-256 为
  `9ecd73ce3e823abb3b6054c59f6dcaed954f85d5cd82f6144ccaf2bafd6b0425`。
- 首次 5-pair pre-release 尝试暴露 prompt/结构生成缺陷（9 条 strict JSON invalid、1 条输入
  超限），未发布正式根；随后只用同一冻结首条做 bounded debug，改为生成时 JSON/citation
  约束，debug root 在验收后删除。下一次全量尝试又因 1 条明确否定的 limitation 被否定作用域
  规则误判而拒绝发布；永久测试补入该否定句后，使用同一冻结 5 条从头重跑。全过程没有换
  样本、保留失败正式根或自动修复模型输出。最终为
  schema/citation/evidence-ID `10/10`、text-RAG 四类字段利用 `5/5`、evidence binding rate
  `1.0`、forbidden claim `0`、failure `0`。生成总时长约 `228.97 s`，peak VRAM
  `4,906,324,992` bytes（约 4.57 GiB），GPU 为 RTX 4090 D。
- 正式根：`outputs/stage6_text_rag/pass2_gpu_smoke_v1`。Run ID
  `ab8a0dfd339e71f84fe382b8c19cb093759d75cd1f7ecadb3a7e5620edb9215d`；manifest / ledger
  SHA-256 为 `987b6f7601e4f16632f9c07d8901242717a6551ed3b6758819eb44d0b097a6eb` /
  `91975b47fbc4e49e336bddcf63e7ea5c72a9a353a4e6e3d1648d423fdef68a59`。

### 验证、清理与边界

- Stage 6 永久 synthetic tests `22/22`；Stage 4 `77/77`、Phase 4/Stage 5 `94/94`、
  Phase 3 `45/45`，既有回归合计 `216/216`。Bank、retrieval 和 paired-run validator 均从
  持久化文件重算通过。
- 已精确删除非正式 debug root、BGE `.cache/huggingface`（含停滞传输 incomplete）、
  editable-install 产生的未跟踪 egg-info；未创建一次性程序。正式三根、BGE/OCR 运行权重、
  永久测试和 Stage 0–5 产物均保留。
- 本阶段运行了 bounded BGE/Qwen GPU inference；没有训练、optimizer/backward、权重改写、
  sealed test、Gate A/C/D/E/F、外部付费 API、commit 或 push。
- 当前状态仅为 `stage6_text_rag_engineering_complete`。没有 retrieval Gold 或专家
  no-RAG vs RAG reference，Gate D 未冻结且未执行，因此 `formal_acceptance=false /
  scientific_acceptance=false`。下一步只读复核命令：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/stage6_text_rag/run_text_rag.py \
  --config configs/stage6_text_rag/dev_v1.yaml \
  validate-run
```

## Stage 6 Gate D automatic-only 开发评价（2026-08-10）

本轮从同一 `main@eae76528b1fa7aaa1793af5451941d510d4bba29` 工作树继续实施；结束时
branch 与 HEAD 未改变，未 commit 或 push。现有 5-pair smoke 只作为工程调试样本，不进入
本轮描述性增益结论。本轮没有建立专家工作台、retrieval Gold、正式 Gate D 阈值或 Stage 7，
也没有训练、下载新权重、修改 Stage 5 checkpoint 或访问 sealed test。

### 冻结协议与 prompt 审计

- 新增严格配置 `configs/stage6_text_rag/gate_d_dev_v1.yaml`、业务模块
  `oa_groundrag/text_rag/gate_d.py` 和薄 CLI
  `scripts/stage6_text_rag/run_gate_d_dev.py`，提供 `prepare / generate / evaluate /
  validate`。既有 paired generator 抽成显式 selected-record consumer，原 Stage 6
  `generate-paired --limit 5` 和 v1 validator 身份保持不变。
- `prepare` 先重算 Bank、80-record retrieval 和 5-pair smoke。它从 smoke predictions
  重算并排除 5 个 record ID，再按原 80 条 selection 顺序对
  `gdcld / landslide4sense / landslidebench_agent / lmhld / multimodal_landslide`
  各取最前 5 条剩余记录，并按固定来源顺序 round-robin；25 条与 smoke 零重叠，五来源
  均为 5 条。ordered-record identity 为
  `dedefb405297611b96e45ae9547c4f3e73584808f6ea5ea788b6cd27b1eabc40`。
- 真实 Qwen processor 对 50 个 text-only prompt 的 token 审计范围为 `915–3785`，0 条
  超过 4096，0 条含图输入。
- 正式根 `outputs/stage6_text_rag/gate_d_dev_protocol_v1`；Protocol ID
  `6d54e63f23cc6bb78d4f35b4f6737262e3a600f8b5c2b22d29a4ccf69ee261f9`；manifest / ledger
  SHA-256 为 `8ddf9829c38ae8ab7d583a47f7ba1e4bdf5be0b4dc200618df63669159fa3cd5` /
  `f12f67c3bcc163f887d78da0d09d32cbffdc8bd58d53e91f1ef4fbc84c14d38d`。

### 25-pair GPU run

- 在 `/home/yukun80/miniconda3/envs/qwen3vl/bin/python` 和 RTX 4090 D 上运行；每条均为
  `no_rag + text_rag`，共 50 次，batch size 1、greedy、`max_new_tokens=768`。两模式共享
  Pass-1、Programmatic Facts、问题、Stage 5 step-900 best generator 与 prompt 主体，唯一
  主要差异是 evidence packet。只加载 trainable LoRA state，未加载 optimizer、scheduler、
  RNG 或 sampler。
- 第一次全量尝试得到 48 条 strict-valid，但确定性 forbidden-claim 检测器把“工程活动导致
  局部地表扰动”误判为滑坡诱因，并把“可能被误认为滑坡”误判为候选升级，因此整体拒绝发布。
  修复只区分反例语境与真实“诱因导致滑坡/确认滑坡”断言；新增永久测试仍拒绝“暴雨导致
  滑坡”和转折后的确认升级。失败尝试没有正式根，也没有复用、换样或输出后修复；随后在
  同一 Protocol ID 上从头重跑全部 50 次。
- 最终 run 为 records `25/25`、predictions `50/50`、schema/evidence-ID/citation
  `50/50`、prompt fairness pairs `25/25`、failure `0`、forbidden claim `0`。text-RAG 的
  interpretation/confounder/limitation/recommended-verification 四类字段均为 `25/25`。
  实际生成时长 `1461.909153 s`，peak VRAM `4,917,782,016` bytes（约 4.58 GiB）。
- 正式根 `outputs/stage6_text_rag/gate_d_dev_25pairs_v1`；Run ID
  `0231fd2adfcd24874562198cdfe5304db9205ffbf644bb4ed00a5daaceaf2f86`；manifest / ledger
  SHA-256 为 `170ad8acaadc626e224b3a93e9a7e1758f78d07438b9adcf9ad60027f49673b5` /
  `a0e52e0c5a94f6e1c14f54547f877d5d78db6b2895eeb59913347cd849da1e92`。

### Automatic-only 报告

- 正式根 `outputs/stage6_text_rag/gate_d_dev_auto_eval_v1`；Evaluation ID
  `1fd09e5f536b448642fa524fa85a9f94bc8e82ef8cb2e2b408ead971be2e80d8`；manifest / ledger
  SHA-256 为 `85a0efdda16e582b768128a16348bda22448639cf1c4694dbe37d64e53fc22f6` /
  `791e3253b4402339a4ab5ced68f926460353effdcc15ea6491659cb5be08299f`。
- 25 个 pair 全部完整，schema `50/50`，no-RAG 空 citation `25/25`，text-RAG citation
  valid `25/25`，forbidden/candidate-upgrade 均为 0；两模式五个字段非空率均为 `25/25`，
  25 个 pair 均至少一个字段文本变化。各字段变化数依次为 supporting `19`、alternative
  `24`、limitations `21`、verification `25`、summary `19`；limitations 双模式保留
  `25/25`。
- no-RAG / text-RAG 每 pair 平均字符数为 `240.8 / 239.44`，平均差 `-1.36`；125 个
  citation references 全部具备 source/page/section traceability，涉及 11 个 unique evidence
  IDs，类型计数为 interpretation `50`、confounder `25`、limitation `50`。
- `unsupported_claim_rate / expert_relevance / Recall@K / MRR / nDCG / gate_d_pass`
  全部严格为 `null`；`expert_reference_available=false`、
  `retrieval_gold_available=false`、`formal_acceptance=false`、
  `scientific_acceptance=false`。

### 验证、清理与当前边界

- Stage 6 永久测试更新为 `33/33`；Stage 4 `77/77`、Phase 4/Stage 5 `94/94`、Phase 3
  `45/45`，既有回归仍为 `216/216`。Bank、retrieval、5-pair smoke、Gate D protocol、
  25-pair run 与 automatic evaluation validators 均从持久化文件重算通过；`compileall` 与
  `git diff --check` 通过。Ruff 在 `qwen3vl` 中不可用（`No module named ruff`），未为此
  安装额外工具。
- 清理前检查了 ownership、Git tracking、链接、打开文件和活动进程。没有遗留 Gate D
  `_work`、debug/pre-release 或 `.staging-*` 根；精确删除了
  `oa_groundrag/text_rag/__pycache__`、`scripts/stage6_text_rag/__pycache__`、
  `tests/stage6_text_rag/__pycache__`。没有创建一次性程序。正式六根、BGE/OCR 运行权重、
  永久测试、Stage 0–5 产物和 `docs/archive/` 均保留。
- 当前状态只记录为 `gate_d_development_automatic_only_complete`，不是 Gate D scientific
  pass。专家相关性、unsupported-claim 人工判定、retrieval Gold、正式阈值、sealed test 和
  Gate D scientific acceptance 均未执行。下一步只读复核命令：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/stage6_text_rag/run_gate_d_dev.py \
  --config configs/stage6_text_rag/gate_d_dev_v1.yaml \
  validate
```
