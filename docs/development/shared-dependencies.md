# 共享模型的第三方依赖

[返回开发目录](README.md) · [设计](../architecture/shared-segmentation.md)

Geo-GroundRAG 的分割核心依赖 Xiaomi-Lens-0，版权归 Xiaomi Corporation，采用 Apache-2.0。
固定源码版本：`05106853ed6222763233b4cc31de117b30adef49`。

- [源码与 LICENSE](https://github.com/SeerRay-Lab/Xiaomi-Lens-0/tree/05106853ed6222763233b4cc31de117b30adef49)
- [模型结构](https://github.com/SeerRay-Lab/Xiaomi-Lens-0/blob/05106853ed6222763233b4cc31de117b30adef49/src/xiaomi_lens/model/modeling_xiaomi_lens.py)
- [掩膜损失及其上游来源说明](https://github.com/SeerRay-Lab/Xiaomi-Lens-0/blob/05106853ed6222763233b4cc31de117b30adef49/src/xiaomi_lens/model/losses.py)
- 本地论文：`参考文献/Xiaomi-Lens-0 Technical Report.pdf`。

本项目没有重新实现其完整 decoder 或训练数学；通过固定依赖调用官方实现。
质量残差融合、数据绑定、区域反馈和已有 RAG 的连接是本分支自己的代码。
官方包及其原始许可应随依赖安装保留。模型权重的许可与身份仍须独立核验，代码许可不能代替权重许可。

本次验证使用现有 Python 3.11 / PyTorch 2.8.0+cu128 / Transformers 5.3.0。
官方完整训练配方标注 Transformers 5.4.0；本次仅验证本项目使用的前向、生成与小规模训练路径，
没有宣称官方分布式训练脚本在 5.3.0 上全部兼容，也未修改共享环境或安装可选加速 kernel。
