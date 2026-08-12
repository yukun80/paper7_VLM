# REBUILD_PROGRESS

本文件是 OA-GroundRAG 的唯一实时状态源，只记录当前游标、冻结资产身份、科学边界、
最近一次有效验收和下一任务。稳定接口见 `README.md`，操作规则见 `AGENTS.md`，详细
算法见冻结设计 `docs/OA-GroundRAG_算法构建方案_0811.md`。

日期化实施日志、已解决的失败尝试、临时目录、旧命令和重复的“未运行”清单不再保留在
活动文档中。能力重构发布时的完整历史可从
`ac94fc1107b524f37dfbcf529cf4dc09bde27405:REBUILD_PROGRESS.md` 读取；该快照只用于
provenance，不代表当前入口或状态。

## 当前游标

| 字段 | 当前值 |
| --- | --- |
| 更新时间 | `2026-08-12` |
| program | `OA_GROUNDRAG_V3` |
| 当前工程状态 | `P0 / INSTRUCTION_ROUTED_UNIFIED_INFERENCE_CORE + UNIFIED_DEMO_WORKBENCH / engineering_complete` |
| 当前文档任务 | `ALGORITHM_INTERVIEW_GUIDE / completed` |
| 下一任务 | `P1 Multi-Source Grounded Evidence` |
| Git 基线 | `main@041e6a40cc5bc7e6c4bc6416059b490334b4732b` |
| upstream 基线 | 本地 `origin/main` 与上述 HEAD 一致，`ahead=0 / behind=0` |
| 发行接口 | `oa-groundrag==0.2.0`；`oa-groundrag = oa_groundrag.runtime.cli:main` |
| 总体科学状态 | P0 与能力驱动重构仅为工程完成；不升级 Gate A/C/D 或系统科学验收 |
| Benchmark test / sealed test | 未评价 / 未访问 |

本次算法与面试讲解文档编写没有运行训练、模型推理、GPU、正式评价或 artifact 重算，没有 commit 或
push。随后完成的 Unified Demo Workbench 只运行一次 val bounded CUDA inference；没有训练、
正式评价、Gate 或 test/sealed payload 访问。开始 P1 或任何新写入前，必须重新核对现场 Git、
进程、资产身份和负责人授权。

## 权威与发布基线

| 项目 | 冻结值 |
| --- | --- |
| 冻结算法设计 | `docs/OA-GroundRAG_算法构建方案_0811.md` |
| 设计基线提交 | `087ae4b438a26cc0bdcd3c453b339bccadcc9e85` |
| 设计 SHA-256 | `fd088b0a25b3fc8888e7b4c07971ef36858c784ffcfaa735219e8e8514251243` |
| P0 unified runtime 提交 | `6d9cd816495f79bf9b13263d9725d6e159fe448b` |
| 能力路径迁移提交 | `b784c746c7749783739f21e3e810012ac493bd6b` |
| 能力重构发布提交 | `ac94fc1107b524f37dfbcf529cf4dc09bde27405` |
| Unified runtime config | `configs/runtime/inference_v2.yaml`；SHA-256 `7811b6c8bfd217fd3f86f8c5edc6c1e897033036cf4df0844efbdb56d433a631` |
| Unified Demo config | `configs/runtime/demo_v1.yaml`；SHA-256 `a060e69ccb625f55d789e3ff2455ad6c643394834217f89c008ea37ddb19b6f9` |
| Retrieval config | `configs/retrieval/dev_v1.yaml`；SHA-256 `f175a99347184d75592ec9a1c61c88fc7a4b976dd7381cb9afafd209fb1f8b57` |
| Grounded curriculum config | `configs/vlm/grounded/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1.yaml`；SHA-256 `2998cd8c36ad69a703507b8446f3767035819d91f96c964a20969d2f6f3a64e2` |

冻结设计和 `docs/archive/` 只读。README 不保存动态 SHA、checkpoint 或完成状态；
AGENTS 不保存动态运行结果。

## 当前能力状态

| 能力 | 稳定代码位置 | 当前结果 | 尚未越过的边界 |
| --- | --- | --- | --- |
| Spatial Perception / OA-AuxSeg | `oa_groundrag/segmentation`；`oa_groundrag/training/segmentation`；`oa_groundrag/evaluation/segmentation.py` | full Benchmark 与负责人定版 checkpoint 可用 | Gate A、正式 fixed predicted masks、sealed test 均未执行 |
| Shared RS-Geohazard MLLM | `oa_groundrag/vlm`；`oa_groundrag/data/rs_general`；`oa_groundrag/training/vlm` | RS-GeneralDesc native v1、step-1000 Adapter 与 Gate B 冻结证据可用 | Gate B 只接受其冻结 RS-GeneralDesc 作用域 |
| Grounded Multimodal Understanding | `oa_groundrag/grounding`；`oa_groundrag/data/grounded`；`oa_groundrag/training/grounding` | train-only Corpus、Eval-dev、compact supervision 与 step-900 Region Adapter 可用 | 64 条严格 no-target 输出失败未修复；Gate C、专家共识和正式 OA-GroundedEval 未完成 |
| Knowledge Augmentation | `oa_groundrag/retrieval`；`oa_groundrag/evaluation/retrieval` | Text Bank、80-record retrieval、Pass-2 smoke 与 25-pair automatic-only 开发评价可用 | 无 retrieval Gold、专家盲评和正式阈值；Gate D 未科学通过 |
| Unified Inference | `oa_groundrag/runtime` | 六类显式任务、确定性 router、lazy provider，以及只读 Benchmark Browser / Demo Gallery / Frozen Eval Workbench 工程完成 | Demo selection 仅作 qualitative 展示；test 默认锁定；P1 多源 grounded evidence 与统一科学评价均未开始 |

Stage 只保留为 curriculum、schema、output root、checkpoint metadata 和历史 provenance；
活动源码、配置、脚本和测试按能力与工程职责组织。

## 冻结与发布资产身份

下表只保留消费方定位资产所需的根和 identity anchor。更细的统计、ledger、环境和指标以
各正式 manifest/report 为准，不在本文件重复抄录。

### Spatial Perception

| 资产 | 根或文件 | Identity |
| --- | --- | --- |
| OA-AuxSeg full Benchmark | `../benchmark/oa_auxseg_hdf5_v1/full` | manifest `9a3b1478ed844f234e32b839fded67a937c49d202e3d8f5efd7db52596b5a00a`；index `389877226249d2477bdda62d937950339e9fa60df35558b945d02757e8d0da42` |
| OA-AuxSeg final checkpoint | `outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt` | SHA-256 `672d39ab4220d8e1b4f949ca8d1d5dcd34f58898cecd1553dd56cdd9d84fb038`；step `206820` |
| OA-AuxSeg finalization report | 同一输出根的 `training_report.json` | SHA-256 `c92ea5e96152f3bc4f67399396a2f1495ce917968d3b8b6c5a516cf377de42b5`；`project_owner_manual_stop` |

Benchmark 共 `53,645` 条（train/val/test 为 `36,761/12,375/4,509`）。checkpoint 是
负责人定版权重，但 `formal_acceptance=false`、`gate_a_evaluated=false`，不得称为
Gate-A-accepted checkpoint。

### Shared MLLM 与 Gate B

| 资产 | 根 | Identity |
| --- | --- | --- |
| RS-GeneralDesc native v1 | `../benchmark/rs_generaldesc_v1` | manifest `36875efb59619cf10fa614ce5995afb88f7d7888624267f7ed216f29ae385832`；payload `549281f296b357bce256e6af71cec7412fe17e36052d6a8674f4876ae2d06e0b`；hash manifest `55ac26d9771ce8385318fbd23a10b999afb754ac195e823be659f4e49b0a7090` |
| RS-General Adapter | `outputs/phase4_rs_vlm/rs_vlm_lora_qwen3vl_2b_b1a16` | report `a4f42e777eaab6e444f04d63b89f482ee31a077bf13006d587863bfa4fb1eb1e`；best pointer `4d93e2c6c34fe01a10db373c00946166238b8d132bb32b9863b52e305b6f4db6`；step-1000 manifest `aa279659a4c563536f1d7554ed9e51643398365ce11539eb9865f77c1d3a621f`；Adapter `a367e39c626338a151dad33e6f7a7f9cc9887206dbcd261d147837e6408becc1` |

RS-GeneralDesc 包含 `274,693` records / `104,954` parents，保存的 deep validation 为
`0 error / 0 warning`。Adapter training report 本身不作科学接受；Gate B 由独立冻结协议
接受。

Gate B 正式根：
`outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1`。

| Gate B 文件 | SHA-256 |
| --- | --- |
| `selection/gate_b_protocol.json` | `8378f6f107849439be3b402b0014df4007b10589e93602c1afb99173c2fb2c54` |
| `selection/gate_b_selection.json` | `98290aaa585b798dcc5a30b9a4d47083e778aa4d48980e24bbce647705b915bd` |
| `base/generation_manifest.json` | `bad680291426f65fd51dfaa35eca649968e478af963d3c5009cdbce733a699fb` |
| `base/predictions.jsonl` | `862759c44400552f40f5211a38ceafd8d1f4712c7d6f870e3a2f0676d8ce8bd6` |
| `adapter/generation_manifest.json` | `c69314727ba71ef712fd7bbde1990ebd610e6759714641103916adf901787168` |
| `adapter/predictions.jsonl` | `a7c791fcfe8f3b94f6b188780bc0aed46dbf2fcf92f1bcca3b6617ca1c4bd98a` |
| `evaluation/paired_scores.jsonl` | `64d6802e2b2438305fa7ba560bc4d90ebb4537d1048dc281333c707fb4d3975f` |
| `evaluation/gate_b_report.json` | `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |

Gate B 的 `formal_acceptance=true` 只绑定原协议、产物和验收提交。能力重构后的严格 verifier
因 implementation SHA 漂移如实返回 `GATE_B_PROTOCOL_INVALID`；不建立兼容伪协议，也不
据此抹除原提交上的历史 PASS。

### Grounded data 与 Region Adapter

| 资产 | 根 | Manifest SHA-256 / 状态 |
| --- | --- | --- |
| deterministic Auto Pilot | `outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500` | `37aebb9c5f8ceb720e0a1a3c8621212d44562fa6b6786d145c31e11ffa94f9bb`；500 train-only records |
| Region Corpus base | `../benchmark/oa_grounded_stage4_v1/region_corpus/mask_grounded_region_corpus_train_v1_500` | `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e` |
| OA-GroundedEval-dev | `../benchmark/oa_grounded_stage4_v1/eval_dev/oa_grounded_eval_dev_v1_100` | `fed7d8b99e4482da1a9e8553c2779cd64007a710fa10404bf2985e96f1ce7492`；100 val baselines / 340 records |
| Region Corpus extension | `../benchmark/oa_grounded_stage4_v2/region_corpus/mask_grounded_region_corpus_train_extension_v2_7950` | `a26f4267ba12fad8ac39481dcd16dd40a65dacdec7133453847cb2e2c71d43fe` |
| Region collection | `../benchmark/oa_grounded_stage4_v2/region_collection/mask_grounded_region_train_collection_v2_8450` | `cd2b86f6244f4f5f42d846166f11a34efdb9edd636239039b42444c453e435d2` |
| compact training messages | `../benchmark/oa_grounded_stage4_v2/training_messages/mask_grounded_region_compact_training_messages_train_v3_6974` | `746f641f1fbe48f4301ffc0c52b586437a1dc0b68a5add4be1e3db50d69a1184`；6,974 mixed-supervision records |

Region Adapter 正式根：
`outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1`。

| Region Adapter anchor | SHA-256 |
| --- | --- |
| `training/training_report.json` | `c0751415a2da4ec72c92892975ab713d688aa316a1480b7a82b8ce8e9d5916ab` |
| `training/best_checkpoint.json` | `368bc48fec6e5303c80e8b2c0d397f4d55eb8c6d51fe70d9da417833ac8a2c1b` |
| step-900 checkpoint manifest | `c203823597c7b3ecf7f1bce9b3030efe7f5ef2dc5c1ae58bbc58e0982aae5c30` |
| step-900 Adapter | `858e12ff7e902ce0a3fdfb1a3dfbc2e58ad0892dec870a73fa4fc0a3411f84d7` |
| prediction manifest | `4b090b4392a906817379357d1a8295f8b5eea10339eeb610847bc9bd2ef26a6b` |
| evaluation report | `362e9c0036403382627b677ffedd79125fbcd46ca6ac9d77492f7d9199c07ea1` |
| `workflow_state.json` | `9c5a90743e26576f55a83157e8a1bd3fcf28c1005cc97ff098be1a3b02a62efa` |

该 Adapter 在 OA-GroundedEval-dev 上为 `276 valid / 64 INVALID_MODEL_OUTPUT`；64 条均是
严格 no-target 区域数组非空，未做自动修复。workflow 固定
`formal_acceptance=false / scientific_acceptance=false / sealed_test_evaluated=false`。

### Knowledge Augmentation

| 资产 | 根 | Identity anchor |
| --- | --- | --- |
| Text Evidence Bank v1 | `outputs/stage6_text_rag/text_evidence_bank_v1` | Bank ID `9322a9139d04be7665feb154153b7dc1c2d35b0871fc32bbd6a6daa942fabb28`；manifest `9b891e191581746173a27b80356caa18ec9be5d3c36eaee67444b05a070f0bcc`；ledger `1c73fbd6135daaaf4767f8427e7c9e1ba69f09e0223aaad2c3e151a013c1e650` |
| 80-record dev retrieval | `outputs/stage6_text_rag/dev_retrieval_v1` | Retrieval ID `e7edfb2ae05a5114c105ce82e7c9bcc87c089dd54d1bd68a5bc43ea860c2f1c2`；manifest `da0e207b284bee81824b3542efb8a4b19138c92f07ba09e0afc3f11c4c0b9e7c` |
| 5-pair Pass-2 smoke | `outputs/stage6_text_rag/pass2_gpu_smoke_v1` | manifest `987b6f7601e4f16632f9c07d8901242717a6551ed3b6758819eb44d0b097a6eb` |
| Gate D dev protocol | `outputs/stage6_text_rag/gate_d_dev_protocol_v1` | manifest `8ddf9829c38ae8ab7d583a47f7ba1e4bdf5be0b4dc200618df63669159fa3cd5`；protocol `70b13b12711b07b1b74b797e7f230f510cde35c5416a295670ddedc8019ce99d` |
| Gate D 25-pair run | `outputs/stage6_text_rag/gate_d_dev_25pairs_v1` | manifest `170ad8acaadc626e224b3a93e9a7e1758f78d07438b9adcf9ad60027f49673b5` |
| Gate D automatic evaluation | `outputs/stage6_text_rag/gate_d_dev_auto_eval_v1` | manifest `85a0efdda16e582b768128a16348bda22448639cf1c4694dbe37d64e53fc22f6` |

Text Bank 固定消费 `docs/RAG_knowledge/` 的 12 个 PDF；正式索引 1,283 units
（interpretation/confounder/limitation 为 `872/138/273`）。Gate D 结果仅是
automatic-only development evidence；unsupported-claim 人工率、专家相关性、Recall@K、
MRR、nDCG 和 `gate_d_pass` 均为 `null`。

## 科学验收与不可扩张边界

| Gate / 边界 | 当前状态 | 允许的结论 |
| --- | --- | --- |
| Gate A | 未执行 | OA-AuxSeg 只有工程定版 checkpoint；不得导出或宣称正式 fixed predicted masks |
| Gate B | 原冻结作用域正式通过 | 只证明 RS-GeneralDesc native v1 固定协议下 Adapter 相对提升 |
| Gate C | 未执行 | Region Adapter 的 automatic-only dev 结果不能证明模型正确依赖 mask |
| Gate D | automatic-only 开发评价完成，科学未通过 | 只能报告工程合同和描述性差异；不能声称 RAG 有科学增益 |
| Gate E / F | 条件未触发 | 不启动 Landslide-Evidence Adapter；不宣称 retention 通过 |
| sealed test | 未访问 | 任何阈值、协议或实现均不得从 test 反推 |

持续有效的数据与事实边界：

- Corpus 和 supervision 只使用 train；OA-GroundedEval-dev 只使用 val；test/sealed test
  不得进入 Corpus、开发阈值或当前报告。
- 6,974 条 compact supervision 是 `mixed_model_and_single_expert`，不是 Gold 或专家共识。
- OA-AuxSeg 当前辅助 registry 只接受 `dem / insar_velocity / slope`；SAR 尚未接入。
- encoded InSAR 的物理单位与 sign convention 未确认时，只能作为 encoded evidence；不得
  生成定量位移或物理方向结论。
- LMHLD 和 Landslide4Sense 缺少可靠地理 group；不得从文件名或 sample ID 伪造空间身份。
- Text RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1 observation，不把
  候选区域升级为确认滑坡。
- Stage 7 Case RAG 只在正式 Gate D 证明稳定增益后考虑；Stage 8
  Landslide-Evidence Adapter 只在 Gate E 失败时考虑。

## 最近一次有效工程验收

能力驱动重构基线仍保留；Unified Demo Workbench 的当前有效工程证据如下。

| 检查 | 最近结果 |
| --- | --- |
| 全量回归基线 | 能力重构时收集 339 项：331 passed，4 项因缺少 `oa_auxseg_hdf5_v1/small` skip，4 项含 backward/optimizer step 仅收集未执行；本轮未重跑训练相关全集 |
| Runtime / Demo CPU | `tests/runtime` 49/49，其中既有 Unified Runtime 31/31、新 Demo 18/18 |
| Annotation / architecture | annotation Workbench 10/10；architecture 8/8，包含 Demo config 与薄 CLI 检查 |
| Grounding / retrieval / data | 相关只读合同与 pipeline 回归 69/69 |
| Gradio smoke | Blocks 构建通过；最终代码在 `127.0.0.1:8801` 启动并关闭，`share=false`、private callbacks、queue concurrency=1、Demo/Frozen 白名单与 protected roots 黑名单已检查 |
| compile / metadata / diff | `compileall`、Demo CLI help、本地 editable metadata `oa-groundrag==0.2.0`、`demo` extra、console target、`git diff --check` 均通过 |
| CUDA bounded Demo | 单个 val `REGION_INTERPRETATION` 成功，覆盖 segmentation、Pass-1、6 条 evidence 与 Pass-2；peak allocated `4,677,430,784` bytes，provider release 完整；run 为 `outputs/demo/unified_workbench_v1/runs/demo_20260812T034449148419Z_b4bcfc86dfe04039a787e513a5b94b0f` |
| 等价性与保护资产 | Benchmark manifest/index、OA-AuxSeg checkpoint、Region Adapter、Text Bank、Frozen Eval 与正式 prediction/evaluation/RAG output anchors 实施前后 SHA-256 一致；模型数学未修改 |
| Ruff | 未安装；准确结果为 `No module named ruff` |

本次 `2026-08-12` 新增零基础算法与面试讲解：30 道题编号和六类固定栏目完整，115 个
本地 Markdown 引用零缺失，29 个唯一小林题目/专题 URL 已通过页面或站内搜索解析；并复核
活动路径、Git identity、冻结设计 SHA 和 `git diff --check`。没有运行 Python/CUDA 测试
套件、模型 inference、训练、正式评价或 artifact validator；该句只描述面试讲解文档任务。

Unified Demo 的上述 smoke 均为 engineering evidence，不构成 Gate A/C/D、正式 test
evaluation 或 scientific acceptance。`allow_test_demo=false`，真实 test 未读取，Demo root
中不存在 test access receipt；Frozen Evaluation selection 保持 100 条原身份且完全只读。

## 下一任务与授权

下一任务仍为 `P1 Multi-Source Grounded Evidence`；教学文档与 Unified Demo Workbench 都没有开始 P1。

开始前必须：

1. 重新读取本文件、AGENTS 和冻结设计，并核对现场 Git、活动进程与受保护资产。
2. 由项目负责人明确授权本次写入、GPU inference 或评价边界。
3. 保持现有模型数学、checkpoint、Benchmark、Text Evidence Bank 和正式 outputs 只读。
4. 不访问 sealed test，不运行 Gate A/C/D，不启动 Case RAG 或新增训练。

## 维护规则

- 新状态直接更新现有游标、能力表、资产表和边界表，不再按日期追加实施日记。
- 新正式资产只记录稳定根、manifest/report identity 和科学身份；细节留在资产自身
  manifest、ledger、report 和 Git 提交中。
- 已解决的失败、临时 PID、scratch 路径和一次性命令不进入活动文档；仍构成当前 blocker
  时才保留最短必要说明。
- 历史 Stage/phase 名可以保留在 schema、output root、checkpoint metadata 和 Git
  provenance 中，但不得重新成为活动源码、配置、脚本或测试的组织方式。
- 不新增重复的 archive、handoff、audit 或 worklog 文档。
