# 3. 配置与身份

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

### 3.1 RS-VLM 配置

新增严格 `rs_vlm.config.v3`：

```yaml
schema: rs_vlm.config.v3
model:
  backend: qwen3_vl | qwen3_5
  model_root: ...
```

v3 除 `backend` 外继续显式绑定 processor、图像/token 限制、dtype、attention、LoRA、
训练预算、数据 manifest/ledger 与输出根。未知键、缺失键、schema/backend 不一致、
模型目录与 backend 不匹配必须拒绝。

既有 `rs_vlm.config.v2` 只为冻结 2B 资产保留读取语义；原 YAML 及语义 SHA 不修改。

### 3.2 模型与 processor ledger

4B 下载必须使用固定完整 Hub commit revision，进入全新版本化目录。发布 ledger 对每个
实际消费文件记录相对路径、字节数、SHA-256 和角色；分片模型必须列出 index 与每个
safetensors shard，不能只哈希 index。ledger 本身使用 canonical JSON 并记录自身 SHA。

processor 身份至少覆盖：

- `config.json`、`generation_config.json`；
- tokenizer config、tokenizer/vocab/merges 与 special tokens（存在时）；
- preprocessor/video preprocessor；
- `chat_template.jinja` 或家族明确允许的模板文件；
- Hub repo 与固定 revision。

路径必须是普通文件且位于绑定根内；symlink、缺失、额外未登记权重分片、SHA 不符均拒绝。
