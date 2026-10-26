# 负责人授权的评价链退役

[返回本节目录](README.md) · [文档总目录](../README.md)

路径表中的代码、配置和输出路径均相对仓库根目录；`../benchmark` 指仓库同级目录。

负责人已明确授权从 Unified Demo 退役 Frozen Evaluation，并删除可由保留代码、配置与上游数据
重建的对应开发评价链。下表是删除前完成的精确路径、普通目录、无 symlink、大小与 manifest
身份核验。所有列出的目录现均为 `DELETED / ABSENT_VERIFIED`；删除不可直接恢复，只能用保留的
代码、配置与上游数据重建。该授权未扩张到未列出的 Benchmark、训练、模型、Text Bank 或正式代码资产。

| 待退役目录 | 删除前大小 | 删除前 identity | 状态 |
| --- | ---: | --- | --- |
| `../benchmark/oa_grounded_stage4_v1/eval_dev/oa_grounded_eval_dev_v1_100` | 28M | manifest `fed7d8b99e4482da1a9e8553c2779cd64007a710fa10404bf2985e96f1ce7492` | `DELETED / ABSENT_VERIFIED` |
| Region Adapter 根下 `base_gt_mask_baseline` | 148K | prediction manifest `6216332b3fef86b0dab751e5237d3524db7927cd5836e8f7c991080b003233af`；evaluation manifest `e4029d21372251766d616c512b3d06c75eb356df7b83e04e445a392493eb0c93` | `DELETED / ABSENT_VERIFIED` |
| Region Adapter 根下 `rs_general_adapter_gt_mask_baseline` | 152K | prediction manifest `dfab33e7b53bb8e7039128cd971c3ee19cc072e211715b5ef2355a753616f7e1`；evaluation manifest `696b369fe1840eea2ec172d259ab7f85649fef65a921decc1c02c047b8ab31c0` | `DELETED / ABSENT_VERIFIED` |
| Region Adapter 根下 `mask_grounded_region_adapter_gt_mask` | 1020K | prediction manifest `4b090b4392a906817379357d1a8295f8b5eea10339eeb610847bc9bd2ef26a6b`；evaluation manifest `8629114aa22fb8b39b9346943f6100b157428c903e295bb67dfb4dfb603e8448` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/dev_retrieval_v1` | 2.0M | manifest `da0e207b284bee81824b3542efb8a4b19138c92f07ba09e0afc3f11c4c0b9e7c` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/pass2_gpu_smoke_v1` | 56K | manifest `987b6f7601e4f16632f9c07d8901242717a6551ed3b6758819eb44d0b097a6eb` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/gate_d_dev_protocol_v1` | 56K | manifest `8ddf9829c38ae8ab7d583a47f7ba1e4bdf5be0b4dc200618df63669159fa3cd5` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/gate_d_dev_25pairs_v1` | 200K | manifest `170ad8acaadc626e224b3a93e9a7e1758f78d07438b9adcf9ad60027f49673b5` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/gate_d_dev_auto_eval_v1` | 136K | manifest `85a0efdda16e582b768128a16348bda22448639cf1c4694dbe37d64e53fc22f6` | `DELETED / ABSENT_VERIFIED` |
| Demo run `demo_20260813T040242784502Z_0fcb1828fc9e488bb2098758da894bc2` | 1.9M | run manifest `9440abb176c6e0bbc7348525af85dd287e90433e4b9d811bc9e118ce19c3f049`；重新枚举确认唯一 Frozen run | `DELETED / ABSENT_VERIFIED` |
