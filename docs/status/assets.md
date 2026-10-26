# 冻结与发布资产身份

[返回本节目录](README.md) · [文档总目录](../README.md)

路径表中的代码、配置和输出路径均相对仓库根目录；`../benchmark` 指仓库同级目录。

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
| Qwen3.5-4B base candidate | `models_zoo/Qwen3.5-4B-r851bf6e8` | Hub commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；asset ledger `configs/vlm/assets/qwen3_5_4b_r851bf6e8.asset_ledger.json` SHA-256 `ca48af5142c25796898f0f9c44a6720d0656747714565544341f1733ea9dab2f`；payload `23355763f89e7f03e6dc756fede244c5d6761360b065308b2adcbffeb1e5b1ce`；M4 24 GB 工程资源门通过；基模自身不作 Adapter/Gate B/科学接受声明 |
| Qwen3.5-4B RS-General Adapter | `outputs/phase4_rs_vlm/rs_vlm_lora_qwen35_4b_r851bf6e8_b1a16_v1` | config semantic `f2e3a3dc05b5152ddda64e1e88d765c4825b128021d355b9425b35d53220b843`；report `cf3f135bf452a365907c3a4bd3f4d9d07a0f0031be5cab7c01d0ed93401bfec0`；best pointer `9ee78b2c239321ac983ea94e76d254f083e093aebe84928c55438baf8d106a5e`；step-1000 manifest `22cc396e89a8ba84843657b8a9272acc8bcaced7de475c2a831370580cccf98c`；Adapter `0be1b656d5646073c54183837e43cbf0c2c45f24d1c9652f4b7a9ccd9d89f6d5`；training report 自身 `formal_acceptance=false`，独立 4B Gate B 已通过 |

RS-GeneralDesc 包含 `274,693` records / `104,954` parents，保存的 deep validation 为
`0 error / 0 warning`。Adapter training report 本身不作科学接受；Gate B 由独立冻结协议
接受。

Qwen3-VL-2B Gate B 正式根：
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

Qwen3.5-4B Gate B 正式根：
`outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen35_4b_r851bf6e8_v1`。

| Qwen3.5-4B Gate B 文件 | SHA-256 |
| --- | --- |
| `selection/gate_b_protocol.json` | `7c3b94ffc473460f1a1d6652d05eb448d3a5e14ba7ed4eea96a5df423e9f9ba5` |
| `selection/gate_b_selection.json` | `f67b169c1d40cd42ac8591909e1eee49a92ff7179325daf52a67866386c637b1` |
| `base/generation_manifest.json` | `6be25639646fb2aaa2a066dd30f6018ae4351b8845171d5ac8ee39bdfda64fc1` |
| `base/predictions.jsonl` | `87e94fdece610706bb59975c245372885df6251f1f0b615d208fce96bac22687` |
| `adapter/generation_manifest.json` | `cbdaf59efd38e3427c4f2df31e96c18ebf1dea0e65f392ecb7690708c2ad7b1b` |
| `adapter/predictions.jsonl` | `70d730f42fbe8a772b5a6af7e53fdcd2e8ae9e9e0fdceb2bcb41578e972d00be` |
| `evaluation/paired_scores.jsonl` | `c728fd7f2bc64d2783c197c5bc28aea8899e2bc8ba4432a40d1a52668c0d9fc9` |
| `evaluation/gate_b_report.json` | `b7c144eee000222b0556838e5bc17ce14dc409ec7704fa51fb0a4e23ad013cd9` |
| `family_comparison/paired_scores.jsonl` | `cd35ef0161545a606420c04cac8d32efb957e81607f37d8e3f5a1cfc87368fca` |
| `family_comparison/family_comparison_report.json` | `7322eb107b3b1b8fc0f5c713fc9eff6d050fb914d8d6ee9052f0ef1eb0287da5` |

4B Gate B 的 protocol/selection canonical SHA 分别为
`424a3f223a8d30db8138a188c09c731b992ae444cd4e573401ca75443c3eb318` /
`9a16bf06462879549b4e278f652da1aaf55b80908ad8c9db4a857fa6bfee281a`；其
`formal_acceptance=true` 只证明该 4B RS-General Adapter 在独立冻结协议下相对自身 base
通过六项判据。跨家族报告不参与 Gate，不支持 4B 相对 2B 的科学优越性结论。

### Grounded data 与 Region Adapter

| 资产 | 根 | Manifest SHA-256 / 状态 |
| --- | --- | --- |
| deterministic Auto Pilot | `outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500` | `37aebb9c5f8ceb720e0a1a3c8621212d44562fa6b6786d145c31e11ffa94f9bb`；500 train-only records |
| Region Corpus base | `../benchmark/oa_grounded_stage4_v1/region_corpus/mask_grounded_region_corpus_train_v1_500` | `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e` |
| Region Corpus extension | `../benchmark/oa_grounded_stage4_v2/region_corpus/mask_grounded_region_corpus_train_extension_v2_7950` | `a26f4267ba12fad8ac39481dcd16dd40a65dacdec7133453847cb2e2c71d43fe` |
| Region collection | `../benchmark/oa_grounded_stage4_v2/region_collection/mask_grounded_region_train_collection_v2_8450` | `cd2b86f6244f4f5f42d846166f11a34efdb9edd636239039b42444c453e435d2` |
| compact training messages | `../benchmark/oa_grounded_stage4_v2/training_messages/mask_grounded_region_compact_training_messages_train_v3_6974` | `746f641f1fbe48f4301ffc0c52b586437a1dc0b68a5add4be1e3db50d69a1184`；6,974 mixed-supervision records |
| Qwen3.5-4B Grounded bounded smoke（v3） | `outputs/smoke/mask_grounded_region_lora_qwen35_4b_rsinit_r851bf6e8_smoke_v1` | step-1 workflow state/report/checkpoint manifest/Adapter SHA 为 `5181f05233740457c5f3177beee5cbaed1af985016da2113839dbf3660d0c4c9` / `158f48a5ac96b29c6ccd2582f28aa199681ca5a662d2ad66542ebe66d21af755` / `c1fea784bf6e949628286f04da14d402ba6a3f84c14084d0790d0f6abe620532` / `41c6182fbbe7abb8cf533b9a2f6295ad0feda259a3426317b1c54fb12612e040`；v4 资源门未通过，故条件清理未授权触发，目录与旧 v3 配置继续保留且不是可发布 Adapter |
| M7 v4 两 epoch 资源画像 | `outputs/smoke/qwen35_4b_m7_schedule_profile_v4` | 16,000 rows / 1,000 optimizer steps；manifest `94fb0ff87a27e6b219eeb46929c15b73be736be4bb373dba626e3b626d7add33`；schedule `2b25174b9c18437f142ca50b118fee8c045ca97da14a5c7f73b26c06e4329f9c`；selection file/canonical `aba9e06c1adabe66314de0f562d96d37d9bab566e0d3b796fbccc71ba889d271` / `7458e73a091641bace07bbfc9c8cd53b3fa38414904d618959df7d704fdf831a`；`train_only / test_or_sealed_accessed=false` |
| M7 v4 loss/gradient parity 失败根 | `outputs/smoke/qwen35_4b_m7_loss_parity_v4` | report/failure SHA 为 `11953cd3de6940975dff1db517d4d05a985b757c476c384421d20b1931dc1d5a` / `5939162a29dbfa1f68c7d7b73ecb96f25e9428048cbad213e0c95440545a6603`；loss 差 `0`、峰值节省 `1,531,380,224` bytes，但 aggregate gradient relative-L2=`0.009208712726831436 > 0.001`，状态严格为 `failed`，verifier 不可消费 |

Region Adapter 正式根：
`outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1`。

| Region Adapter anchor | SHA-256 |
| --- | --- |
| `training/training_report.json` | `c0751415a2da4ec72c92892975ab713d688aa316a1480b7a82b8ce8e9d5916ab` |
| `training/best_checkpoint.json` | `368bc48fec6e5303c80e8b2c0d397f4d55eb8c6d51fe70d9da417833ac8a2c1b` |
| step-900 checkpoint manifest | `65ecb7b890f38868ae5a1b7af144ff3dc2befad149bdefb6b6f133b0cfa42bb6` |
| step-900 Adapter | `858e12ff7e902ce0a3fdfb1a3dfbc2e58ad0892dec870a73fa4fc0a3411f84d7` |
| `workflow_state.json` | `f1e7fe6b6d3cfe4b86e93fde6d963487b92e696cfd53a3d028cc719f28725155` |

Adapter 权重、best pointer、training report 和 retention 资产字节保持不变；step-900 manifest
和 workflow state 只将模型、processor、Benchmark 与 Adapter 身份改为相对
`<bundle-root>` 的路径。旧 OA-GroundedEval-dev prediction/evaluation 实例已在[退役记录](retired.md)记录
身份后删除，不再作为当前可消费产物。workflow 固定
`formal_acceptance=false / scientific_acceptance=false / sealed_test_evaluated=false`。

### Knowledge Augmentation

| 资产 | 根 | Identity anchor |
| --- | --- | --- |
| Text Evidence Bank v1 | `outputs/stage6_text_rag/text_evidence_bank_v1` | Bank ID `28935a7533a228d0aaa0e8754304549b62b84d067f4653e31b105ec55599284d`；manifest `8d21adefa103f8a53adacc0a82e08e5fdc194b8e780084ab109dc3bbc6c4bb05`；ledger `114a3fffef4067bdfce6f69cc9c7388823773ff5cd54d00c0a1866dd443225ea` |

Text Bank 固定消费 `paper7_VLM/docs/RAG_knowledge/` 的 12 个 PDF；正式索引 1,283 units
（interpretation/confounder/limitation 为 `872/138/273`）。知识源和 BGE-M3 使用相对
`<bundle-root>` 的路径，既有向量与证据内容没有重建。在线 Text RAG 能力和 Bank 身份保留，
旧 development retrieval、Pass-2 smoke 和 Gate-D artifacts 已删除；当前没有 Gate-D 科学结果，
unsupported-claim 人工率、专家相关性、Recall@K、MRR、nDCG 和 `gate_d_pass` 均未建立。
