# 权威与发布基线

[返回本节目录](README.md) · [文档总目录](../README.md)

路径表中的代码、配置和输出路径均相对仓库根目录；`../benchmark` 指仓库同级目录。

| 项目 | 冻结值 |
| --- | --- |
| 冻结算法设计 | `docs/archive/OA-GroundRAG_算法构建方案_0829.md`（v3.2，SHA-256 `ddba2c90d9a54fc3561177baec70430d68ac982f08afc02bd87a639595bdf7a0`）；原 v3.1 SHA-256 `1eea63b99b840847440df667899ab26b85faf9f0ea05a947dbcd6a5b67cdacc9` 为 superseded provenance |
| 历史冻结设计 | `docs/archive/OA-GroundRAG_算法构建方案_0811.md`；只读 provenance |
| 设计基线提交 | `087ae4b438a26cc0bdcd3c453b339bccadcc9e85` |
| 0811 历史设计 SHA-256 | `fd088b0a25b3fc8888e7b4c07971ef36858c784ffcfaa735219e8e8514251243` |
| P0 unified runtime 提交 | `6d9cd816495f79bf9b13263d9725d6e159fe448b` |
| 能力路径迁移提交 | `b784c746c7749783739f21e3e810012ac493bd6b` |
| 能力重构发布提交 | `ac94fc1107b524f37dfbcf529cf4dc09bde27405` |
| Unified runtime config | `configs/runtime/inference_v2.yaml`；SHA-256 `14e417a3a7068a13c4aaee7f9e083282df13df417ce8ed6c5a0d335f3af7c883` |
| Unified Demo config | `configs/runtime/demo_v1.yaml`；SHA-256 `540ee12c38130ea20513af61da76cf33b9292f62a7c7ee272f261aab64b5e573`；`frozen_evaluations=[]` |
| Grounded runtime config | `configs/vlm/grounded/runtime_v1.yaml`；SHA-256 `bf5bf69fd2fa77dc8a0daa1fc813075f128459c5819319f2b31ea5063a17a018`；semantic `4620994de82590d763ca1896fb5e5c52a325625cebe6e043bcda1618fc8c4d25` |
| Grounded train-only config | `configs/vlm/grounded/train_v2.yaml`；SHA-256 `16fcfff8ba8ebd8becca48fa57d5711e58a20de25a83f04d1cb92aa6920a52d4` |
| Text RAG runtime config | `configs/retrieval/runtime_v1.yaml`；SHA-256 `b5866e1b26848fdecfcd29ff1bf938702c01149b8dbcd836bfceed14219975d8`；semantic `26f9dc6e375adc443604c27556bb5d809e260d091238498bfce39314514c3034` |

冻结设计和 `docs/archive/` 只读。README 不保存动态 SHA、checkpoint 或完成状态；
AGENTS 不保存动态运行结果。

### 2B 便携元数据重发布

修改前副本与 SHA-256 清单已完成逐项核对。下表只记录此前实施轮次因路径相对化而变化的当前
发布身份；原始模型、Adapter、分割 checkpoint、Benchmark 和向量文件没有改写。

| 配置或元数据 | 修改前 | 当前 |
| --- | --- | --- |
| Unified runtime config | `2cca2a111249be033385348d7e2faf6278e0d39f0ac497e185bbcd3097215979` | `14e417a3a7068a13c4aaee7f9e083282df13df417ce8ed6c5a0d335f3af7c883` |
| Grounded runtime config | `329d1cc57443174dc989a63d58c9613ab79532704c9e73ec0457309a022ab48e` | `bf5bf69fd2fa77dc8a0daa1fc813075f128459c5819319f2b31ea5063a17a018` |
| Text RAG runtime config | `7e732f24d4eb9061465e4ad9ed866ef88869a30943001df9499aea4c65367f33` | `b5866e1b26848fdecfcd29ff1bf938702c01149b8dbcd836bfceed14219975d8` |
| Region step-900 manifest | `c203823597c7b3ecf7f1bce9b3030efe7f5ef2dc5c1ae58bbc58e0982aae5c30` | `65ecb7b890f38868ae5a1b7af144ff3dc2befad149bdefb6b6f133b0cfa42bb6` |
| Region workflow state | `9c5a90743e26576f55a83157e8a1bd3fcf28c1005cc97ff098be1a3b02a62efa` | `f1e7fe6b6d3cfe4b86e93fde6d963487b92e696cfd53a3d028cc719f28725155` |
| Text Bank dense index | `290c8b4a113909f1e8fd6ebc0dac23fc91f6fafe447a65da8248d08abc748652` | `6aba404c33cb39fff4d18552899a0c40f0bb4f64f576fba6e22f870254aa003f` |
| Text Bank environment | `73b9248286264e3d3d28267038cab20743588c0aa5f9fe880368db3e6aa7de6f` | `915deffd313df73b03d5cb9e1f000f9e36f3e49dd15632f7e18d913c4f932624` |
| Text Bank normalized registry | `83bdee00fc18f17e2524acaa93f4a984dfbb881234ead3fe5839a507ce8747df` | `b01564aeb6ba0385a3cf1332de9df40e6b7983049d60c044247b622fa93bfa63` |
| Text Bank ledger | `1c73fbd6135daaaf4767f8427e7c9e1ba69f09e0223aaad2c3e151a013c1e650` | `114a3fffef4067bdfce6f69cc9c7388823773ff5cd54d00c0a1866dd443225ea` |
| Text Bank manifest | `9b891e191581746173a27b80356caa18ec9be5d3c36eaee67444b05a070f0bcc` | `8d21adefa103f8a53adacc0a82e08e5fdc194b8e780084ab109dc3bbc6c4bb05` |
| Text Bank ID | `9322a9139d04be7665feb154153b7dc1c2d35b0871fc32bbd6a6daa942fabb28` | `28935a7533a228d0aaa0e8754304549b62b84d067f4653e31b105ec55599284d` |
