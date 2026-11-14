# rec_models 的新增机制审查：CAN、ComiRec、COLD、Node2vec、AutoRec、GBDT+LR

日期：2026-11-14。范围：评估哪些机制值得进入 ModelEvoHarness 的候选计划，以及怎样和现有骨干组合。本轮仅核对原文和代码，没有训练、导入外部模型包装器、修改组件注册表或证明业务收益。

## 1. 结论与采用策略

最值得继续的是 **CAN 的特征对 micro-MLP** 和 **ComiRec-SA 的多兴趣池化**。两者都能成为有清晰输入输出的小组件。COLD 更适合作为已有特征筛选的成本评估协议；Node2vec 主要补图游走和上下文构造；AutoRec、GBDT+LR 应按数据和任务选择，不能用模型名称数量衡量增量。

这里的 **CLEAR** 表示机制与接入方式已查清，可进入限定范围的设计/对照实验；**WATCH** 表示存在明确增量，但应先满足指定证据；**BLOCK** 表示所列具体做法不应采用。它们不等于已达到生产准入，也不是整个仓库的合并结论。

| 候选 | 相对当前 Harness 的真实增量 | 采用策略 | 优先级与边界 |
|---|---|---|---|
| CAN / Co-Action | 按特征 ID 查表获得 micro-MLP 参数；另一侧特征作为输入；独立交互表、多阶输入、逐层输出 | **CLEAR：限定特征对的原生交互单元**；**BLOCK：宣称已有 APG 等价覆盖，或直接搬整套 CAN** | P1。先做一个有数据支持的 pair，使用已有目标与训练器 |
| ComiRec-SA | 在没有候选商品的条件下，从历史产生 K 个兴趣向量；与 MIND 路由形成可控对照 | **CLEAR：目标无关的多兴趣池化组件** | P1。真正的序列召回或具有多兴趣残差的骨干才启用 |
| ComiRec-DR | 按位置与兴趣分别投影，再作 capsule squash/routing；不同于当前简化 MIND | **WATCH：等参数/等检索预算后再决定** | P2。不得仅改名现有 MIND 作为复现 |
| ComiRec 多样性聚合 | 在去重后的候选集合内，以类别重复惩罚控制准确率/多样性 | **WATCH：单独的集合选择策略** | P2。与已有 DPP 比较；不与编码器效果混测 |
| COLD | 固定实际服务预算，扫描特征组数量并重训，用质量、QPS、RT 共同选型 | **CLEAR：协议补充**；**BLOCK：把现有 `COLD.py` 当作完整 COLD** | 协议 P1；新运行时组件 P3。SE 与筛选基础已存在 |
| Node2vec | 基于上一跳的 p/q 偏置游走，生成不同于普通边采样的上下文 | **WATCH：主机侧图快照/采样能力** | P2/P3。优先复用已有 Item2Vec 目标和预训练向量接入 |
| AutoRec | 观测评分矩阵的非线性重建，损失只计观测项 | **WATCH：显式评分任务基线** | P3。当前任务若没有真实评分矩阵则不启用 |
| GBDT+LR | 用树叶离散编码表达阈值组合，再由 LR 学习评分 | **CLEAR：可选表格基线**；**BLOCK：使用仓库 demo 评估结果或称为可微骨干层** | P2/P3。真实表格数据有阈值残差时再比较 |

## 2. 来源与检查边界

- 聚合仓库固定在 [`lmxhappy/rec_models@966f5f6`](https://github.com/lmxhappy/rec_models/tree/966f5f66f07da78d16f16d44a3978be7235fd7a9)，本地快照 `/private/tmp/rec-models-review-20261114`；完整 Git tree 为 `/private/tmp/rec-models-tree-20261114.json`。
- Harness 固定在 [`CharlesXu-HQ/ModelEvoHarness@5dd85dc`](https://github.com/CharlesXu-HQ/ModelEvoHarness/tree/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac)，本地 `/private/tmp/modelevoharness-components-20261111`。结论以具体组件实现为准，同时检查 method cards，未把卡片中的模型名视为完整论文复现证明。
- `Co-Action-Network` 和 `ComiRec` 在聚合仓库是 mode `160000` 的 gitlink，没有嵌入源文件。经作者仓库 Git tree API 验证，分别精确对应 [`CAN-Paper/Co-Action-Network@98a7219`](https://github.com/CAN-Paper/Co-Action-Network/tree/98a7219105568b9420cf28467877d08a1c56ae69) 与 [`THUDM/ComiRec@7e605f1`](https://github.com/THUDM/ComiRec/tree/7e605f1daabe1ab48e4e10469f4367429bce86c9)。已读取前者 `script/model.py`，后者 `src/model.py` 与 `src/train.py`。
- `collaborative-filtering/AutoRec@bc858b4` 和 `collaborative-filtering/Autorec_tf@d7bc970` 也是 gitlink。**此轮没有解析其确切来源，也没有检查其中代码**；AutoRec 的机制判断仅来自作者论文。
- `cold/COLD.py`、`Node2vec/*/model.py`、`GBDT+LR-Demo/GBDT_LR.py` 是已检查的实际文件。Node2vec 示例 README 明确指向作者 `aditya-grover/node2vec`，实际采样调用 DGL；文件夹名 `deepctr_node2vec` 不能证明其来自 DeepCTR。
- CAN、ComiRec 作者树中未见 LICENSE 文件；这是来源记录，不能据此推定可复制。建议独立实现经确认的必要算子，保留来源与论文/代码差异说明。

## 3. CAN：有价值，但应准确区分 APG、门控和特征对交互

### 原文与作者实现

将一侧特征映射为参数向量 `theta_b`，解析成各层 `W_l,b_l`；另一侧映射为输入 `x_a`。核心为：

```text
h_0 = x_a
h_l = activation(W_l(theta_b) h_(l-1) + b_l(theta_b))
z_ab = concat(h_1, ..., h_L)
```

这里共享的是同一个特征 ID 对应的参数或输入表项，不是为每一个 `(a,b)` 组合存储完整网络。CAN 还使用独立于基础表示的交互参数表，以及 `x_a` 的不同幂次。原文公式 11 将各阶输出相加；已核对的作者代码 `gen_coaction` 则把各阶、各层输出拼接，在历史轴上求和，默认两层 `[16,8],[8,4]`、三阶、无 bias、只在非末层用 tanh。**论文形式与该代码变体必须明确选一种，不能混称逐公式复现。** [CAN 原文 §4](https://arxiv.org/html/2011.05625v2#S4)、[固定作者代码](https://github.com/CAN-Paper/Co-Action-Network/blob/98a7219105568b9420cf28467877d08a1c56ae69/script/model.py#L10)。

### 和现有能力的差别

- Harness `AdaptiveLinear` 已能做完整 `x W(c)` 或低秩 `x U S(c) V`，可在表达能力上构造与单层 co-action 相近的变换。因此不能声称“CAN 首次带来条件参数”。但当前接口通过条件向量生成矩阵；CAN 的可复用单元应显式支持主机提供的参数向量、特征对语义、逐历史位置变换、多阶输入和中间层读出。这些不是现有 APG 调用一次就具备的完整约定。 [AdaptiveLinear](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/ctr_conditioning.py#L24)
- `PriorConditionedGate` 做逐坐标缩放；co-action 可以在坐标间混合并连续非线性变换。DIN 则主要按候选对历史进行权重聚合。它们可共同使用，但应该分开消融。 [先验门](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/ctr_conditioning.py#L84)、[DIN](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/architectures.py#L62)
- 一层、一输出、无激活的 co-action 可退化成类似点积的交互；不能只在这个退化情形验证，然后宣称验证了完整 CAN。

### 最小可行接入与验收

建议未来组件契约为 `CoAction(theta[B,P], feed[B,T,D], valid[B,T]) -> [B,T,C]`，由主机选择掩码求和/均值后，拼到现有 backbone 的某一层或独立交互分支，再进入同一个 head。`theta` 和 `feed` 的表由主机创建；K 个 pair 不必变成 K 个完整模型。首先只做一个有稳定联合支持的 pair，避免默认枚举所有字段。

数据前提：决策前可获得的真实 ID/字段关系；若使用历史则有用户、事件、时间与候选商品对应关系；稀疏组合有足够训练支持。不要把连续数值随意离散后套一个“商品 ID”，也不要为了补齐接口伪造历史。参数表与浮点张量是主机可生成的中间产物，不是要用户另采集的业务字段。

对照至少包括同一骨干无分支、简单 bilinear/点积、现有完整矩阵 APG，以及 CAN 一阶与二/三阶。固定 embedding 预算或同时报告新增表大小；测稀疏/新 ID 切片、序列长度、延迟与显存。另作“共享基础表 vs 独立交互表”消融以确认收益来自何处。原论文 Amazon Book 的两层 AUC 为 `0.7935`、三层 `0.7913`；一阶 `0.7902`、三阶 `0.7935`，仅支持小范围深度/阶数对照，不能外推本任务会提高。 [原文 Table 5](https://arxiv.org/html/2011.05625v2#S5.SS3)

## 4. ComiRec：拆成 SA、DR、集合聚合三个可独立评估的部分

### SA 是首选增量

作者实现用历史加位置向量产生 K 组注意力权重，再对原始历史 embedding 加权。注意力沿历史轴归一化：

```text
A = softmax_over_time(W2 * tanh(W1 * (H + position)))
U = A * H                 # [B,K,D]
```

这是 K 个兴趣池化器，并非一定需要 `T×T` 的 Transformer 自注意力。查询候选商品不参与 `U` 的生成，因此可用于召回；监督训练时通过正目标选择最匹配兴趣，再做 sampled softmax。部署检索不能把真实下一商品作为输入。 [原文公式 5–8](https://arxiv.org/html/2005.09347v2#S3.SS2)、[作者 SA 代码](https://github.com/THUDM/ComiRec/blob/7e605f1daabe1ab48e4e10469f4367429bce86c9/src/model.py#L210)

Harness 现有 MIND 的角色也是多兴趣编码器，DIN 是目标相关历史聚合；SA 新增的是另一种目标无关兴趣抽取方法，值得用同一 `history/mask -> interests` 合约实现。接在现有骨干内部时，可把 `U` 保留为 token 交给后续网络，或以当前候选做目标相关读出后与原分支融合；这属于新组合，应写明不是原 ComiRec 召回链路。 [当前 MIND](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/sequence.py#L36)

优先对照：masked mean、当前 MIND、SA(K=1)、SA(K>1)，保持总候选数、负采样、embedding、时间截断和优化预算一致；SA 再单独开关位置项。检查兴趣间相似度、有效兴趣使用率、不同兴趣丰富度用户的 Recall/NDCG。兴趣数增加造成更多 ANN 搜索，必须报告检索耗时和去重后候选数，不能把检索预算增加当作结构收益。

### DR 不应被现有 MIND 名称掩盖，也不应直接排到 SA 前

作者 ComiRec-DR 的 `bilinear_type=2` 用 `W[t,k]` 变换历史到各兴趣；`s_k=sum_t c_tk W_tk h_t` 后执行 squash，并按 agreement 更新路由 logits。作者代码还在非最终路由迭代停止变换结果的梯度。当前 Harness MIND 使用原型相似度初始化、加权均值加原型、L2 normalize；因此两者具体算子和参数预算都不同。 [ComiRec CapsuleNetwork](https://github.com/THUDM/ComiRec/blob/7e605f1daabe1ab48e4e10469f4367429bce86c9/src/model.py#L108)、[DR 的调用](https://github.com/THUDM/ComiRec/blob/7e605f1daabe1ab48e4e10469f4367429bce86c9/src/model.py#L199)

有真实顺序信息且当前路由对顺序不敏感时，可评估完整 DR；需要明确最大长度、位置投影共享方式、squash 和 stop-gradient 选择。先做相同长度/参数/候选预算的对照，再拆分“位置相关投影”和“路由算法”贡献。论文 Table 3 中 SA、DR 在不同数据/指标各有领先，不支持预设 DR 总是更好。 [原文 Table 3](https://arxiv.org/html/2005.09347v2#S4)

### 聚合是另一种候选集合选择器

先合并各兴趣检索结果并去重，再贪心选择 `score(i) - lambda * selected_count(category(i))` 最高者，这是作者代码的具体形式。类别只是其中一种 diversity 定义，必须来自真实且可用的目录属性。 [作者聚合代码](https://github.com/THUDM/ComiRec/blob/7e605f1daabe1ab48e4e10469f4367429bce86c9/src/train.py#L105)

Harness 已有 DPP greedy 与 slate reranking 家族，因此新价值是简单、可解释的类别重复控制，不是首次支持多样性。固定同一个候选池分别比较 `lambda=0`、类别惩罚、现有 DPP；报告质量与多样性的曲线。不要同时更换编码器和聚合器后只报告一个提升数字。 [已有 DPP 卡片](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/data/supplemental_method_cards.json)

## 5. COLD：优先补成本约束流程，避免新增一个同义 SE 模型

论文的有效信息是：训练较充分的跨特征模型，用组重要性筛选不同 K，重建轻量版本，按 GAUC、QPS、RT 选择服务预算内的配置；还包括在线学习、特征计算布局、并行与半精度等系统设计。论文并没有证明“只乘一个 SE gate 就加速”。 [作者论文 §3.2–3.3](https://dlp-kdd.github.io/dlp-kdd2020/assets/pdf/a13-wang.pdf)

聚合仓库 `COLD.py` 实际只有 `se_block` 加 `1024→512→256→128→64→1` 全连接层；无 top-K 物理裁剪、候选数协议、延迟测量和在线更新实现。SE 计算也是标准两层 excitation，与论文对组权重的简化表达不应混作完全对应。 [被检查文件](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/cold/COLD.py#L4)

Harness 已有 FiBiNET 的 squeeze gate、`ContextFeatureGate`、`gradient_feature_selection`，最后一项要求筛选后匹配重训并保护交互组。COLD 的可行增量是在这条流程上加入真实特征获取成本、并发与候选批量条件，产出质量—延迟 Pareto 配置。 [FiBiNET](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/interactions.py#L91)、[ContextFeatureGate](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/ctr_composition.py#L85)、[筛选卡片](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/data/supplemental_method_cards.json#L855)

只有主机提供真实 serving profile 时才作成本选择：同一硬件、batch/candidate 数、并发、p50/p95、特征读取开销。将某些张量乘零并不会自动减少 dense kernel 或特征读取；应通过移除输入组、重新确定矩阵形状并重训证实收益。可复用已有骨干与平行分支，只比较哪些分支/输入值得保留。无服务测量时可先记录候选协议，不能声称已实现 COLD 的系统收益。

## 6. Node2vec：新增偏置游走，不新增一个神经网络层

原始机制是二阶随机游走：从 `t→v` 后选择邻点 `x`，未归一化概率为 `w(v,x)*alpha(t,x)`，其中 `alpha` 对返回 t、距 t 一跳、距 t 两跳分别为 `1/p, 1, 1/q`。再把游走窗口用于邻域预测。 [作者项目页](https://snap.stanford.edu/node2vec/)、[作者论文](https://arxiv.org/abs/1607.00653)

Harness 已有 Item2Vec 的 skip-gram 负采样、LINE 图边目标、EGES 上下文建模，以及 GraphSAGE/PinSage 聚合。未见 Node2vec 偏置游走构造器；实际增量在“什么节点被采成上下文”，不能用新增 embedding lookup 类冒充。可优先由主机生成 `center/context/negative`，复用 Item2Vec 目标；若要忠实复现 DGL 示例的单表、loss 归一化等具体选择，需另外说明与已有双表目标的差异。 [Item2Vec](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/retrieval.py#L133)、[已检查示例](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/Node2vec/dgb_node2vec/model.py#L1)

必须先有决策时点之前的真实边、节点身份、边权含义和图快照。对照固定边、walk/token 预算、维度和下游 head，比较 `p=q=1`、偏置游走、仅直接邻居；报告稀疏节点、新节点、时间外推与构图成本。预训练向量可通过已有 `PretrainedIDFusion` 注入 backbone，也可形成平行图分支；图的增量应由下游对照证明。新节点没有历史图位置时，需要主机已有 side feature 或默认策略，Node2vec 本身不会产生未见节点表示。 [预训练向量接入](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac/src/model_evo_harness/models/pytorch/ctr_embeddings.py#L6)

## 7. AutoRec、GBDT+LR：条件化补充

**AutoRec。** 原文将评分向量映射为 `r_hat=f(W*g(V*r+mu)+b)`，在观测集合上计算重建误差并正则化参数。与 Harness 的 SDNE 邻接重建、CDL 内容重建有相似计算，但矩阵语义和观测掩码不同；不能把缺失评分当作 0 分监督。 [作者论文](https://akmenon.github.io/papers/autorec/autorec-paper.pdf)、[作者项目](https://mesuvash.github.io/projects/1_autorec/)

若主机有显式评分及观测掩码，可作为矩阵补全基线，对照 BiasSVD/MF 与非线性重建；确保验证评分不会同时出现在输入里。若只有曝光点击或优惠券 outcome，移植需重新定义任务，原 AutoRec 不能直接替代主干目标。隐变量可以作为并行协同特征或辅助任务，但须说明重建损失是否帮助当前目标，且不能直接声称隐式反馈复现。该候选目前没有检查 gitlink 代码，优先级低于 CAN/SA。

**GBDT+LR。** 原论文将每棵树的叶编号转成离散特征，交给 LR；它带来非线性阈值/组合编码，适合作为业务表格数据的独立基线，或训练后固定的叶特征分支。树的分裂和叶路由不可按普通神经层反向传播，不应列为任意骨干内部的可微层。 [作者论文](https://www.quinonero.net/Publications/predicting-clicks-facebook.pdf)

聚合 demo 有明确评估错误：`y_test = df_train['target']` 与 `X_test` 不对应；最终“NE”公式用 `(1+y)/2,(1-y)/2`，对于此处 `objective=binary` 通常使用的 0/1 标签，该公式不适用；本轮未读取该 demo 的完整数据文件；打印量也未除以标签熵，不能当 normalized entropy。代码还物化 dense leaf one-hot。**这些使 demo 的成绩不可用，但不否定机制。** [被检查 demo](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/GBDT%2BLR-Demo/GBDT_LR.py#L22)

如需评估，重新建立正确 train/validation/test 或时间切分，所有树、叶字典和 LR 只拟合训练数据，以真实验证/测试标签计算 logloss、AUC 和校准；采用稀疏叶编码，对照 LR、GBDT、GBDT+LR。若叶特征再进入更大的模型或模型选择流程，明确训练内交叉拟合/独立校准的边界，避免利用验证标签选择树后再把同一验证集作最终结论。无需为这次文档评估实现整套训练器。

## 8. 后续工作应如何界定成功

1. **组件设计层**：先定义 CAN 与 SA 的 typed tensor 合约、接入位置、参数共享和 masking；验证张量/梯度/退化对照，无需训练就能确认基础数值行为。CAN 必须选定论文求和版或作者拼接版；SA 明确位置项只进入权重还是也进入 value。
2. **任务适用层**：只有真实 pair/history/graph/rating 数据满足前提才启用对应路线；中间 embedding、参数向量由主机构造，不向用户索要伪业务字段。始终保留现有 baseline 和已经有效的兼容组件。
3. **有效性层**：在同一真实数据、时间边界、预算和目标上，一次改变一个机制；分开测编码器、融合位置、训练目标与集合选择。质量、成本及关键切片满足预先声明的门槛，才把候选从研究建议转成默认推荐。

本轮通过来源、公式和静态实现交叉检查，确认以上新增机制和边界；未产生任何 CAN、ComiRec 或其他候选在当前业务数据上的实测优劣结论。
