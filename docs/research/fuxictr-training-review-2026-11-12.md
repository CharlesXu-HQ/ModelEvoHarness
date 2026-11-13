# FuxiCTR：训练、embedding 与输入契约增量审查

日期：2026-11-12。FuxiCTR 固定版本 `183da1c944ba8930fcdcd54114837c081959d748`；ModelEvoHarness 审查基线 `0888cd51c184a64da81d75e858d6671a4dc8b72d`（包含待合并工作，不等同于 GitHub main）。本次只读源码并撰写研究文档，未实现组件、运行训练或进行 GPU/效果验证。下述“缺失”指已审查的 Harness 源码、manifest、training patterns 和知识文件中没有相应的具体机制；不表示宿主无法自行实现。

## 判断

对 Agent 最有增量价值的是**把模型输入的表示方式和模型专属损失变成可诊断、可单独消融的改动**。常规优化控制已在 Harness 中有知识与实验协议；迁入 FuxiCTR 的训练器、网格搜索或数据流水线不是当前最优先事项。

FuxiCTR 此处提供的是 CTR 模型、训练与数据处理实现。其 `BaseModel` 接收标签并计算预测损失，autotuner 枚举配置；这些源码没有提供 Agent 的假设形成、因果识别或宿主策略价值保证。不能从 CTR AUC 改善推断 treatment effect 或业务政策收益。[F1][F10]

## 按 Agent 改模价值排序

| 优先级 | 可吸收机制 | Harness 当前状态 | 应补的具体内容与判定实验 |
| --- | --- | --- | --- |
| P1 | 字段/输入角色级 embedding 契约 | **部分已有**：`NumericFieldEmbedding`、分类字段 embedding、表示诊断、分支共享声明均已存在；统一的字段选择、共享词表、维度兼容等具体 recipe 缺失。[H2][H3][H4] | 为模型改造补输入角色→原字段→词表→embedding 模块→输出维度的解释；以现有表示诊断支持“换投影/维度/共享”的假设。在相同数据与参数预算下分别测共享/独立、统一宽度/字段宽度分配、pooling/保留序列。 |
| P1 | 预训练 embedding 的 `init` / `sum` / `concat` 与冻结策略 | **部分已有**：FNN 能加载 FM 向量；知识提到迁移、预训练与冷启动；未见这组三种融合方式及其 OOV 约定的组件/训练模式。[H5][H7][H8] | 增加具体的表示实验 recipe：随机初始化、仅预训练、预训练+ID 残差、concat+投影；冻结和微调另做单因素消融。按 head/tail/new/OOV 切片比较，绑定预训练版本与时间范围。 |
| P1 | FinalNet 双分支一致性/自蒸馏损失 | **具体机制缺失**：已有一般蒸馏知识和 `dmtl_distillation` 模式，但不是 FinalNet 的在线双分支目标。[H1][H7] | 模型卡应同时描述结构和 loss recipe；固定两分支结构，对比只有主 BCE 与主 BCE+一致性项；报告分支差异、主任务和校准。增加分支和增加损失必须分别消融。 |
| P1，有序行为数据条件成立时 | DIEN/DMIN 的下一行为辅助监督 | **结构已有、训练缺失**：Harness DIEN manifest 明确省略辅助下一行为损失，代码只返回主 logit；知识仅提示分开消融。[H6] | 将兴趣提取状态和正/负下一事件、对齐 mask、负样本来源作为显式输入契约；独立比较辅助权重 0 与正值，保持负采样规则和主模型固定。上游当前实现有静态 shape/reduction 问题，不能直接当作已验证代码移植。 |
| P2 | embedding 与非 embedding 分组正则化 | **部分已有**：`weight_decay` 与优化指南覆盖正则化，但未见具体的两组强度实验 recipe。[H1] | 当 sparse ID 表与 dense 网络呈现不同过拟合迹象时，明确参数集合后分别调惩罚强度；保持模型不变并报告 tail/new 切片。不能把源码的显式 L2 惩罚称为 AdamW 式 decoupled weight decay。 |
| P2，已有相应长序列结构时 | SIM 的粗检索/精排双头监督 | **具体 recipe 缺失**；一般多任务平衡知识已存在。[H1] | 同一个观察标签分别监督 GSU 与 ESU：固定序列筛选和两头，测试是否添加粗头监督，再测权重；不可把辅助头当独立标签或负样本任务。 |
| P3 | 训练器、配置枚举、批量实验执行 | **训练知识已存在；执行通常由宿主负责**。[H1][H3] | 有实际 host 集成需求再做适配。不能用更多 YAML 组合替代有诊断证据的单因素实验，也不应将它们计为新增模型优化机制。 |

## 1. embedding 契约能增加哪些具体改模能力

`fuxictr/pytorch/layers/embeddings/feature_embedding.py` 支持 numeric/categorical/sequence/embedding 四种输入，字段级 `embedding_dim`、可选 encoder、按字段名/source/type 过滤以及共享 embedding 模块。numeric 是无偏置 `Linear(1,D)`；已有 dense embedding 默认先投影；sequence 是否仍为序列由 encoder 决定。`dict2tensor` 对字段做 stack 或 concat。[F2]

这不是“任意字段维度都可直接喂给任意模型”：stack 要求输出形状相容；一些模型按 `embedding_dim * num_fields` 固定输入尺寸，例如 FinalMLP。Agent 应先验证当前下游需要统一 `[B,F,D]`，还是允许 concat 的 `[B,ΣD_i]`，再决定维度实验；否则增加字段宽度可能只造成 shape 错误。[F2][F5]

共享还要求**同一 ID 语义和同一映射**。`feature_processor.py` 会合并相关 tokenizer 词表，随后同步 `share_embedding` 字段的映射；embedding 层则复用已经创建的模块。源码依赖被共享目标先存在。target item、history item、negative item 可以具有相同 item ID 域；同为整数的 user/item ID 不能因此共享。Harness 已能声明模块共享，新增价值在于把词表、字段顺序、mask、输出维度与实际共享对象一并绑定，而不是再加一个分支容器。[F2][F3][H3]

输入角色还有实质优化含义：FinalMLP 的 `fs1_context` / `fs2_context` 各自选择上下文来产生乘性门；为空时使用可学习常量上下文。门为 `2*sigmoid`，逐坐标作用于完整 flat embedding，不能描述为实际删除字段的离线特征筛选。两套 context embedding 是分别构造的实例；不能仅因使用相同字段就宣称与主 embedding 共享权重。可比较常量门、相同 context 门、角色不同的 context 门，并在原有表示诊断与分支记录中观察效果。[F5][H3][H4]

## 2. 预训练融合应成为独立的表示实验

`pretrained_embedding.py` 中：

- `init`：使用预训练表，维度不符时线性投影。
- `sum`：预训练向量（必要时投影）加一张可学习 ID 表。
- `concat`：预训练向量与 ID 向量拼接后投影至目标宽度。
- `freeze_emb` 决定预训练表是否训练；ID 表在 sum/concat 下仍为可学习参数。[F4]

需保留其精确 OOV 语义：pretrained vocab 可在训练词表之后追加 key，但保留原 OOV 分界；forward 对 `inputs <= oov_idx` 构造 mask，ID 路径始终使用该 mask，**未冻结**的 pretrained 路径也使用它。因此“冻结/微调”在该实现中可能同时改变训练词表外 token 是否获得预训练表示，不只是梯度开关。若要隔离微调效应，必须在实验中固定词表覆盖与 fallback 行为；不能直接比较两个配置并把差异全部归因于 fine-tuning。[F3][F4]

对 Harness 的实际收益是提供预训练来源、key 对齐、覆盖率、是否冻结、融合方式、投影维度以及 serving fallback 的可执行知识。已有 FNN `load_fm_embeddings` 只检查逐字段矩阵数量和形状，词义对齐与预训练步骤仍由 host 负责；因此此项是补充，不能报告成“首次支持预训练 embedding”。[H5]

## 3. loss recipe 要跟随模型，不能只记录网络名字

### FinalNet：准确的目标与消融

`model_zoo/FinalNet/src/FinalNet.py` 的 2B 路径先取两分支 logit 的平均，再经过 sigmoid 得到 `p=σ((z1+z2)/2)`；它不是两分支概率的平均。使用 BCE 时总目标为 `BCE(p,y) + BCE(σ(z1),stopgrad(p)) + BCE(σ(z2),stopgrad(p))`，三项均为 mean。`detach` 只作用于一致性项的软目标；主 BCE 仍向两分支传播梯度。源码一致性系数固定为 1，若 Harness 引入可调 λ 应标为本地适配。[F6]

这里既无外部教师，也无新的 observed label。应先固定两分支，比较 λ=0 和预先声明的 λ；只有验证主指标/校准/成本得到改善，才能保留该训练项。一般蒸馏知识不能替代这一梯度方向和 reduction 的说明。

### DIEN/DMIN：有增量价值，但源码不能直接视为通过验收

DIEN 用 `interest[:, :-1]` 分别结合正事件与负事件的 `[:,1:]` 做二分类辅助监督，要求 post-padding、真实时序对齐、有效相邻事件 mask 和外部负序列。负样本是训练构造的对比项；源码没有证明这些 item 曾曝光未响应。Harness 应要求 host 声明可用 item 集、采样时间窗、正样本排除和 false-negative 风险。[F7][H6]

静态检查发现：DIEN 的 `aux_prob/aux_label` 将正负批次拼到 `2*B*(T-1)` 行，`pad_mask[:,1:].view(-1,1)` 只有 `B*(T-1)` 行；相乘通常形状不兼容。其沿最后大小为 1 的维度求和，也没有把结果约简成 scalar，却加到主 scalar loss。DMIN 对应代码使用同样形状路径。这是源码可见问题，**没有进行运行重现**；任何后续实现须明确重复 mask、有效计数归一化、全空/无有效转移 batch 的零损失语义，并验证 scalar loss 和梯度。[F7][F8]

SIM 的辅助头则是另一种机制：`alpha * BCE(y_aux,y) + beta * BCE(y_pred,y)`，两个 head 对同一主标签分别监督，不能混称为 DIEN 下一行为训练。FuxiCTR `MultiTaskModel.add_loss` 实现 `EQ` 时求任务损失之和，未见动态任务权重算法；不能把接口上的 `loss_weight` 参数写成已有动态平衡能力。[F9][F13]

## 4. 常规训练控制：已有知识，补精确语义即可

`BaseModel` 将任务损失和显式正则项相加，再 backward、clip、optimizer step。embedding regularizer 对 `FeatureEmbeddingDict` 内的可训练参数生效，可能包含投影/encoder，不限于 `nn.Embedding.weight`；net regularizer 覆盖剩余可训练参数。惩罚形式为 `λ/p * ||θ||_p^p`。它没有照 Harness 现有 `weight_decay` 说明那样自动排除所有 bias/normalization 参数，所以采用时应显式记录参数组。[F1][H1]

默认 `eval_steps` 为每 epoch 一次；patience 数的是评估时未改善次数。未改善时还会缩小 LR，`fit` 最后加载所保存的 checkpoint。比较架构必须固定 eval cadence、checkpoint rule 和真实 update 数；仅说“同样 epochs”不足以排除训练协议差异。其 checkpoint 保存 `state_dict()`，不是完整 optimizer/scheduler/RNG resume 状态。以上适合丰富已有优化说明，无需先替换 host trainer。[F1]

## 5. 数据与实验协议边界

普通输入统计在 train 上拟合；meta 列为避免 group ID 在 valid/test 被 OOV 合并，可以合并三个 split 的 key 建词表。`BaseModel.get_inputs` 排除 label 与 meta，因此不能简单把 all-split meta key 字典等同于训练特征泄漏，也不能据此允许普通预测字段利用 holdout 统计。[F1][F3][F11]

gAUC 是有效组按样本数加权，avgAUC 是有效组等权，单一标签组权重为零；这些具体语义应写进 host evaluator 契约。换用它们不能自动修复 exposure bias 或证明全目录 ranking 能力。[F12]

autotuner 是笛卡尔积配置枚举+GPU 子进程调度，可枚举数据和模型参数。`run_expid.py` 逐实验训练并输出 validation/test 结果。对 Agent 搜索应保持 final holdout 封存；若在每轮看 test 后决定下一步，就破坏 Harness 已有的验证选择协议。源码并未提供自动假设诊断、最小消融选择、可信收益归因或多 seed 统计决策。[F10][H1]

## 建议落点

优先补知识和 method-linked training recipe，覆盖上述输入条件、梯度/shape、最小对照、拒绝条件；之后有实际实验需要时再增加小型 tensor 组件。具体候选为 `field_embedding_sharing_and_projection`、`pretrained_id_fusion`、`two_branch_consistency`、`next_behavior_auxiliary`、`embedding_network_regularization`。这些是研究建议名称，当前没有新增 runtime pattern。

是否有效应由固定宿主任务上的主指标、切片、校准与资源预算决定。字段共享和预训练融合优先用于解释已观察到的表示/冷启动问题；FinalNet 一致性只在对应双分支结构上检验；序列辅助目标以真实且时间安全的历史与负样本契约为前提。

## 一手源码索引

[F1]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/models/rank_model.py
[F2]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/layers/embeddings/feature_embedding.py#L116-L296
[F3]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/preprocess/feature_processor.py#L212-L253
[F4]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/layers/embeddings/pretrained_embedding.py#L64-L189
[F5]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalMLP/src/FinalMLP.py#L77-L191
[F6]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalNet/src/FinalNet.py#L111-L169
[F7]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DIEN/src/DIEN.py#L167-L239
[F8]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMIN/src/DMIN.py#L242-L271
[F9]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/models/multitask_model.py#L146-L175
[F10]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/autotuner.py#L32-L179
[F11]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/preprocess/build_dataset.py#L147-L212
[F12]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/metrics.py#L99-L130
[F13]: https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SIM/SIM.py#L170-L182
[H7]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/knowledge/optimization_and_regularization.md
[H8]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/knowledge/cold_start.md
[H1]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/data/training_patterns.json
[H2]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/models/pytorch/explicit.py#L50-L76
[H3]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/knowledge/horizontal_composition.md
[H4]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/knowledge/representation_quality.md
[H5]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/models/pytorch/paper_ranking.py#L30-L63
[H6]: https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/data/model_implementations.json#L389-L437

另见确切实现：[SIM 双头损失](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SIM/SIM.py#L170-L182)、[FeatureMap 维度/字段覆盖](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/features.py#L54-L154)、[pretrained vocab 扩展](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/preprocess/tokenizer.py#L183-L201)、[单次实验入口](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepFM/DeepFM_torch/run_expid.py#L46-L87)、[Harness 优化知识](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/knowledge/optimization_and_regularization.md)、[Harness 基本训练函数](https://github.com/CharlesXu-HQ/ModelEvoHarness/blob/0888cd51c184a64da81d75e858d6671a4dc8b72d/src/model_evo_harness/models/pytorch/training.py)。
