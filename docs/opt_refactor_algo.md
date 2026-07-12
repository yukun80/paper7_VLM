# 一、结论先行

当前训练失败**不是数据缓存、benchmark、显存或分割主干的问题，而是 Qwen 的 LoRA 参数没有进入实际反向传播计算图**。集成门禁已经确认 LoRA 参数被成功创建并标记为可训练，否则程序会更早触发“QLoRA 参数隔离失败”；完整 loss 也成功完成 backward，但最终所有 LoRA 梯度之和为 0，因此门禁在启动正式训练前正确地中止了流程。

从当前代码看，**最高概率的根因是 Qwen controller 绕过了 PEFT 包装器的正式 forward 路径**。当前代码通过 `get_base_model().model.language_model` 取得底层语言模型，并直接调用该子模块；而 PEFT 官方的 `PeftModelForCausalLM.forward` 会先启用 `_enable_peft_forward_hooks`，再执行 base model。当前做法没有经过这层包装，因而存在“LoRA 参数已经注入且可训练，但实际 forward 没有激活相应 adapter 路径”的风险。

因此，**不建议通过 `MEMORY_GATE=0` 直接绕过门禁开始正式训练**。这样虽然 SANE、QMEF、PMRD 和 controller 的自定义 embedding 可能继续更新，但 QLoRA 实际上仍然不工作，最终不能支撑“Qwen 经适配后更新 mask queries”的论文主张。

---

# 二、当前算法完成度重新判断

此次重构在算法结构和工程实现上已经相当完整。当前模型不再是简单的遥感多通道分割网络，而是建立了比较严格的三阶段链条：

[
\text{多源遥感实例}
\rightarrow
\text{SANE预训练/物理特征编码}
\rightarrow
\text{Qwen更新mask queries}
\rightarrow
\text{QMEF证据选择}
\rightarrow
\text{PMRD proposal-set分割}
]

Benchmark-v2 已经要求每个模态显式保存 family、sensor、product type、band metadata、GSD、units、signed、orbit、quality、normalization 和物化 valid mask；这使多源输入不再依赖旧 canonical 通道推断。

当前完成度可以重新评估为：

| 部分                       |   完成度 | 当前判断                      |
| ------------------------ | ----: | ------------------------- |
| Benchmark-v2 与数据契约       |   95% | 结构严格，可支撑不同传感器和变长模态组合      |
| Referring/no-target 指令数据 |   85% | 已解决同图不同指令对应不同 mask 的基本问题  |
| Raw SANE/QMEF/PMRD       |   90% | 已有完整 forward、loss、训练和测试闭环 |
| Qwen-ViT 特征缓存            |   85% | 支持多视图、中间层和严格缓存协议          |
| Qwen mask-query 控制器      |   70% | 结构已实现，但 LoRA 反向路径尚未打通     |
| 单卡正式训练路径                 |   50% | 被集成门禁阻断，尚未证明可持续优化         |
| 科学消融协议                   |   80% | 已有严格脚本，但尚缺真实多随机种子结果       |
| 论文实验就绪度                  | 约 45% | 主要障碍已从架构设计变成 Qwen 梯度正确性   |

最新版本仍然具有较强研究潜力。其主要创新链条已经相对清晰：变长传感器—波段建模、subset-synchronized Qwen mask queries、null-aware query-scale-modality fusion，以及无 box 的滑坡 proposal-set refinement。当前不需要再做一次大规模架构重写，应该首先修复 Qwen 的可训练性。

---

# 三、为什么 LoRA 梯度为零

## 1. 当前反向链路理论上是连通的

当前设计中的预期梯度链路为：

[
L_{\text{seg}}
\rightarrow
M_{\text{final}}
\rightarrow
\text{PMRD queries}
\rightarrow
\text{Qwen mask states}
\rightarrow
\text{Qwen LoRA}
]

PMRD 明确要求 controller 提供 `mask_query_states`，并把它们作为 coarse decoder 的 query。若 controller 不提供，则直接报错。

当前 controller 也检查了：

* 动态 padding 后的 `inputs_embeds` 必须具有梯度链路；
* Qwen 输出的 `mask_out` 必须具有梯度链路。

这两个检查没有触发，说明自定义 mask embeddings、view projection、output projection 等控制器参数与 loss 是连通的。

但是，“输出 tensor 有 `requires_grad=True`”只说明它依赖某些可训练参数，不代表它依赖 LoRA。当前 output projection、自定义 mask embeddings、evidence anchors 和 view projection 都是可训练的，因此即使 LoRA 完全未参与 forward，`mask_out.requires_grad` 仍会是 True。

## 2. 当前最可疑的是直接调用底层 language model

当前代码执行的是：

```text
PeftModel
  └─ get_base_model()
       └─ model.language_model(...)
```

而不是：

```text
PeftModel(...)
```

PEFT 官方 forward 会进入 `_enable_peft_forward_hooks`，然后调用 base model；当前实现绕过了这个入口。

标准 LoRA 在部分版本中即使直接调用已注入的子模块也可能生效，但这种做法并不是稳定、受支持的 PEFT 调用协议。结合当前“LoRA 参数存在、其他梯度存在、LoRA 梯度全部为零”的症状，应把它作为第一修复目标。

## 3. 当前测试没有真正覆盖真实 LoRA forward

现有单元测试验证了：

* 动态 padding 能保持输入梯度；
* mask queries 会受语言上下文影响；
* mask embeddings 能收到梯度；
* LoRA target regex 能找到最后四层的 q/k/v/o projection。

但这些测试使用的是模拟 language module，没有加载真实 Qwen、PEFT 和 NF4 adapter，也没有断言真实 `lora_B` 参数在 forward/backward 后获得非零梯度。

所以当前集成门禁实际上是第一个真正发现 LoRA 计算图问题的测试，而不是门禁本身过严。

---

# 四、最优先的代码修改

## 1. 不再直接调用 `_language_model()`

应将 controller 中的 Qwen forward 改为通过 `self.model(...)`，即通过 PEFT 包装后的完整模型执行。

建议的调用逻辑是：

```text
outputs = self.model(
    inputs_embeds=inputs,
    attention_mask=attention,
    output_hidden_states=True,
    return_dict=True,
    use_cache=False,
    logits_to_keep=1,
)
hidden = outputs.hidden_states[-1]
```

Qwen3-VL 的正式 forward 原生支持 `inputs_embeds`，会在进入 language model 前构造 position IDs；同时支持 `logits_to_keep`，可以只计算最后一个位置的词表 logits，避免为整个序列生成巨大词表张量。

当前已经把视觉塔从在线 Qwen 模型中移除。只要不传入 `pixel_values`，完整 Qwen forward仍然可以使用已有的 `inputs_embeds`。使用 `logits_to_keep=1` 后，额外 lm-head开销通常可控。

这一修改有三个好处：

第一，PEFT 的 adapter hooks会按正式路径启用。
第二，Qwen自己的 position-id逻辑会被使用。
第三，模型行为更接近官方 Qwen前向协议，降低版本兼容风险。

## 2. 如果完整 Qwen forward显存过高，直接对 text backbone注入 LoRA

长期更干净的方案是：

1. 加载 Qwen3-VL；
2. 保存 token embedding；
3. 提取 `model.model.language_model`；
4. 只对该 text backbone使用 PEFT；
5. 将其作为 controller 的正式语言模型；
6. 删除顶层 lm head和不用的视觉模块。

此时 LoRA应直接注入到**实际被调用的 text backbone**，而不是先包装完整 VLM，再绕过包装器调用内部子模块。

这种设计更节省显存，但改动比通过完整 PEFT forward更大。建议先用完整 wrapper验证 LoRA梯度，确认问题根因后再做 text-backbone提取。

## 3. 用 layer selection 代替超长正则表达式

当前通过完整模块路径正则选择最后四层 q/k/v/o projection。模型目前有28个语言层。

建议改为明确的 LoRA配置：

```text
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
layers_to_transform = [24, 25, 26, 27]
layers_pattern = "layers"
```

这比依赖完整模块路径正则更稳健。初始化后必须检查：

* 每个目标层的 q/k/v/o均被 LoRA包装；
* active adapter不是空；
* adapter没有 merged；
* adapter没有被 disable；
* 实际 forward时目标模块被调用。

---

# 五、重新设计集成门禁

当前门禁一次性混合了四件事：

1. Qwen adapter是否进入计算图；
2. 完整 segmentation loss是否能回传；
3. full/dropped teacher consistency是否生效；
4. batch 6是否满足显存限制。

这种做法导致当前错误只能显示 `lora=0`，却不能精确说明梯度在哪一层断开。应拆分为三个独立门禁。

## 1. Controller-only 梯度门禁

只运行：

[
L_{\text{probe}}
================

|Q_{\text{mask}}|*2^2
+
0.1|E*{\text{anchor}}|_2^2
]

然后检查：

* `mask_query_states.grad` 是否非零；
* 最后一层 `lora_B` 是否非零；
* 所有 LoRA梯度是否有限；
* 实际执行过的 LoRA目标模块数量。

这个门禁不经过 SANE、QMEF、PMRD，因此可以直接回答：

> Qwen mask-query controller 本身是否真的训练 LoRA？

若该测试仍然为零，问题必然在 Qwen/PEFT前向协议，而不是分割 decoder。

## 2. End-to-end 梯度门禁

controller-only通过后，再运行完整 segmentation loss，并对下列中间变量调用 `retain_grad()`：

* `semantic.mask_query_states`；
* `semantic.evidence_anchors`；
* coarse queries；
* refined queries。

分别报告梯度范数。若 controller-only有 LoRA梯度，而 end-to-end没有，则说明 PMRD或 loss对 Qwen query的依赖过弱或被意外截断。

## 3. 显存门禁

最后再以 batch 6、full/dropped混合和 consistency teacher运行峰值显存测试。

显存测试不应兼任计算图测试。它只需验证：

* forward/backward可完成；
* loss有限；
* LoRA已在前一个门禁证明可训练；
* 峰值显存低于22.5 GiB。

---

# 六、LoRA 门禁应至少运行两个优化步骤

当前只检查一次 backward。标准 LoRA初始化通常会使其中一个低秩矩阵初始为零，因此第一步中不同 LoRA矩阵的梯度行为并不对称。

建议门禁执行两个 optimizer steps，并分开记录：

* `lora_A` 梯度；
* `lora_B` 梯度；
* 每层参数更新量；
* 更新参数数量。

第一步至少要求部分 `lora_B` 获得非零梯度和更新；第二步再检查 `lora_A` 是否开始获得梯度。不应只报告一个聚合的 `lora_gradient_norm_sum`。

当前最新门禁由原来的多步检查缩减成一个代表性 batch的一步检查，这提高了效率，但降低了诊断能力。

---

# 七、当前可立即采用的训练路径

## 1. 不建议直接关闭门禁训练 full model

下面这种方式只能用于定位问题：

```bash
MEMORY_GATE=0
```

因为当前 LoRA梯度已经明确为零。关闭门禁后，模型可能只更新：

* mask embeddings；
* evidence anchors；
* view projection；
* output projection；
* SANE；
* QMEF；
* PMRD。

这能产生一个可训练系统，但不能被称为 QLoRA适配后的 Qwen-PSALM。

## 2. 可以先训练 `pretrained_sane_qmef_pmrd`

为了不让整个项目停滞，可以先运行：

```text
PRESET=pretrained_sane_qmef_pmrd
```

这条路线使用缓存的 Qwen-ViT中间层特征，但不依赖在线 Qwen language decoder的 LoRA。它可以验证：

* benchmark-v2；
* 预训练 SANE；
* QMEF；
* PMRD；
* proposal matching；
* 单卡训练吞吐；
* 分割指标。

这是一条合法而且有价值的强基线，但不能替代最终 `qwen_psalm_full`。

## 3. 建议增加冻结 Qwen 的中间基线

新增一个明确命名的 preset，例如：

```text
qwen_mask_query_frozen
```

其结构为：

* 在线 Qwen language decoder完全冻结；
* 不注入 LoRA；
* 训练 mask embeddings、evidence anchors、view projection和 output projection；
* 训练 SANE/QMEF/PMRD。

这个基线能够回答：

> 仅训练 Qwen周围的软提示和分割模块是否已经足够？

随后 `qwen_psalm_full` 与它对比，才能证明 QLoRA的额外价值。

---

# 八、建议采用分阶段训练，而不是所有模块同时起训

当前第一次 optimizer step同时更新：

* Qwen LoRA；
* mask embeddings；
* evidence anchors；
* view projection；
* SANE adapters；
* raw physical encoder；
* QMEF；
* PMRD。

这会使随机初始化的 PMRD与刚开始适配的 Qwen相互干扰。

建议采用两阶段训练。

第一阶段先冻结 Qwen LoRA，训练 200–500 steps：

* controller新增 embeddings/projection；
* SANE；
* QMEF；
* PMRD。

此阶段让分割 decoder先建立可用梯度路径。

第二阶段再开启最后4层 QLoRA，并将其学习率设为 dense模块的0.1–0.3倍。这样更接近 PSALM和 Qwen3-VL-Seg的分阶段适配思想。PSALM的关键在于 mask tokens经过 LMM更新后参与 proposal生成；Qwen3-VL-Seg则通过预训练中间视觉特征和迭代 mask refinement逐步建立 dense能力。

优化器也不应继续对所有参数使用同一个学习率和 weight decay。当前 trainer将所有可训练参数放进单一 AdamW参数组。

建议参数组为：

| 参数组                      |    相对学习率 | Weight decay |
| ------------------------ | -------: | -----------: |
| Qwen LoRA                | 0.1–0.3× |            0 |
| mask/evidence embeddings |     0.5× |            0 |
| view/output projection   |     0.5× |         0.01 |
| SANE raw/adapters        |       1× |         0.01 |
| QMEF/PMRD                |       1× |         0.01 |
| norm/bias                |      对应组 |            0 |

---

# 九、训练问题修复后仍需优化的设计

## 1. 自定义视觉 token不是 Qwen原生图像 token

当前缓存视觉 token被投影后放在 vision-start/end embedding之间，但没有使用原始 image placeholder、完整 grid信息和 Qwen原生视觉位置协议。它们更准确地是“Qwen视觉塔产生的 evidence tokens”，而不是原生图像 token。

当前代码又直接调用底层 language model，因此连 Qwen3-VL完整的 position-id构造也被绕过。Qwen官方模型会在进入语言模型前计算3D position IDs。

修复为完整 Qwen wrapper forward后，这一问题会得到部分缓解。但论文中仍应把它称为：

> compressed multi-view evidence tokens

而不是完整复现 Qwen原生视觉序列。

## 2. Qwen-ViT空间特征恢复仍需真实图像测试

当前通过 forward hook捕获视觉 block，并通过 `restore_qwen_patch_grid` 恢复二维特征。

合成 token排列测试已经存在，但还需要用真实 Qwen视觉塔验证：

* 高亮方块的位置；
* 左右翻转；
* 上下翻转；
* 棋盘格；
* 方向性条纹。

如果恢复顺序错误，预训练 SANE虽能训练，但空间特征会被打乱。

## 3. Raw 与 pretrained特征融合仍需 family-conditioned

当前 raw residual在所有模态族上共享同一组尺度系数，并且初始化权重很低。

这对 RGB可能合理，但对 SAR、DEM、InSAR和完整多光谱并不合理。Qwen-ViT只看渲染后的三通道视图，而原始物理分支保存更多信息。

应让 raw/pretrained融合至少按 family和scale自适应：

* optical偏向 pretrained；
* multispectral两者并重；
* SAR、terrain、deformation偏向 raw；
* renderer带有质量标记时降低 pretrained权重。

## 4. 多个传感器 view不应直接平均

同一 Sentinel-2产品的真彩色和 SWIR/NIR假彩色包含互补信息。当前 SANE读取缓存特征时会对同一模态的多个 view做简单加权平均。

建议改为轻量 learned view attention，避免真彩色和假彩色的语义被不可学习地混合。

## 5. QMEF 的坐标约定仍不一致

当前对齐模块先用 `align_corners=False` 插值，再用 `align_corners=True` 执行 grid sample。

这会产生半像素偏移风险。应统一约定，并增加 zero-offset identity test。

## 6. Coarse mask细化存在确认偏差

PMRD直接以 sigmoid coarse mask乘 query-specific detail feature。

若第一轮漏掉真实边缘，第二轮无法重新读取 proposal外的特征。建议使用带残差的 soft gate或适度膨胀的 coarse mask，并通过消融确定最优设置。

## 7. Verifier应同时预测 relevance和 mask quality

当前 verifier只判断 proposal语义相关性。建议增加轻量 mask-quality head，预测 proposal IoU/Dice质量，再与 semantic relevance共同决定 union gate。这能避免语义正确但边界很差的 proposal获得高权重。

## 8. 缺失模态评价需要固定参考区域

当前 active subset会参与最终 valid mask构建，不同模态组合可能在不同有效区域上评价。

建议同时保留：

* annotation/reference valid mask；
* active-observable valid mask。

任意模态组合主表应使用固定 reference区域，observable-area指标作为补充。

---

# 十、建议的验收顺序

完成 LoRA修复后，不要直接启动4000步训练。建议按以下顺序验收。

第一，运行 controller-only probe，要求至少一个最后四层 `lora_B` 梯度非零。

第二，运行单样本完整 loss，要求：

* mask-query gradient非零；
* evidence-anchor gradient非零；
* LoRA B梯度非零；
* optimizer step后 LoRA参数发生变化。

第三，运行两个优化步骤，要求第二步开始出现 LoRA A梯度。

第四，运行 batch 2、关闭 consistency的 full model smoke test。

第五，运行 batch 6、full/dropped混合 consistency和22.5 GiB显存门禁。

第六，运行20步短训练，确认：

* loss总体下降；
* LoRA参数持续变化；
* mask embeddings不塌缩；
* null reliability不长期接近1；
* proposal relevance不是全部相同；
* no-target误检没有快速上升。

第七，才开始 small-v2正式4000步训练。

---

# 最终评价

当前版本的模型结构已经比较成熟，创新链条也已经建立。此次无法启动训练并不意味着整体算法设计失败，而是**在线 Qwen QLoRA 路径尚未通过真正的计算图验证**。

最优先修复不是降低 batch、修改 loss权重或关闭门禁，而是：

1. 通过 PEFT包装器执行 Qwen forward；
2. 不再直接调用底层 `language_model`；
3. 增加 controller-only LoRA梯度测试；
4. 将一步门禁改为 controller gate、end-to-end gate和 memory gate；
5. 采用冻结 Qwen到开启 QLoRA的两阶段训练。

其中第一项最可能直接解决当前的 `lora=0`。在它修复之前，其他结构优化都不是训练启动问题的主要矛盾。
