# TAAC2026 非大模型机制：ModelEvoHarness 融合建议

核查日期：2026-10-21。入口是腾讯广告技术的[赛事综述](https://mp.weixin.qq.com/s/Unf7RyTnIwP2F8tPPfNrMg)，进一步查阅其引用的论文、作者代码和官方资源。仓库对照基线为已合并的 `main`：`d8af70197b69c2d8a4671a9101c82298067eb6c5`。

本文是调研与实施建议。没有新增运行时代码，没有执行作者代码、训练实验或新组件 GPU 测试。下文类名、文件名和优先级均为建议，不能视作已经交付的实现。

后续实施记录（2026-10-22）：首批四个组件已按下述边界落地，接口和来源见[组件说明](../taac-components.md)，实测结果见[验证记录](taac-components-validation-2026-10-22.md)。本文保留调研时的基线、证据缺口和后续候选。

## 1. 融合范围与结论

优先补齐可独立组合的张量机制：字段交互、候选条件门控、历史压缩、查询生成、跨层交互读出。输入由调用方提供，输出为表示向量、token 张量或明确的交互项；不接管生产数据构建、采样、训练入口和服务部署。

这里的 token 通常是连续向量，代表字段、行为、查询或可学习汇聚位置，不要求 Semantic ID，也不要求语言模型。判断依据是组件实际依赖，而非论文是否使用 Transformer 或 scaling 等术语。

建议分三步：

1. **先补四个小组件**：RankMixer 的 token mixing，以及 UniDot 的条件 SwiGLU、多通道历史池化、显式交互 Highway。验证独立张量合同、公式和梯度，再注册方法卡。
2. **再补查询与序列机制**：QueryFormer 查询桥、LENS 的目标条件修正；CRAFT 的 transport 先解决论文与代码差异。分别验证后才组合。
3. **把扩容经验转成实验指南**：比较字段分组、信息来源、token 数、宽度与预算，不把竞赛 AUC 或“越大越好”作为默认选择依据。

当前项目有 69 张方法卡、25 个 family、140 条框架实现记录；下述 TAAC 获奖方法及 RankMixer、Wukong、LONGER、LENS 尚无对应的独立方法卡。知识文档提到论文、具有类似底层算子，均不足以证明已经复现其机制。

## 2. 第一批建议实现的组件

设 B 为 batch，D 为表示宽度，T 为字段 token 数，L 为历史长度，C 为池化通道数。多行为域由调用方显式提供，不固定为比赛的四域。

| 候选组件 | 张量接口与价值 | 现有项目关系 | 复现时不可省略的部分 |
|---|---|---|---|
| `RankMixerBlock` | `[B,T,D] → [B,T,D]`；用通道重排和逐 token FFN 交换字段信息 | 放入 `feature_interactions`；可接在已有字段 embedding 后 | 论文指定的重排、残差、归一化与 FFN 参数共享关系；重排版本的整除条件 |
| `ConditionalSwiGLU` | 历史与候选条件 → 同长度历史表示；候选只调节 gate | 放入 `sequence_ranking`；作为序列编码前后的小块 | value 路径保持由历史产生；不能把候选直接拼入 value 后仍宣称同一机制 |
| `MultiChannelSequencePool` | 当前字段状态、分域历史及 mask → `[B,S·C,D]` | 可作为现有并行分支的压缩输出 | 独立 sigmoid 通道、加权汇聚及归一化；不能替换为多头 softmax |
| `ExplicitInteractionHighway` | 各层用户/候选状态与域检索点积 → 显式交互向量 | 与已有 `GroupedFM` 互补；读出交给调用方 | 用户×候选完整点积、跨层收集，以及明确的输出接入；只计算却不送往下游不构成该机制 |

RankMixer 的依据是[原论文](https://arxiv.org/abs/2507.15551)。后三项有可定位的 [UniDot 作者实现](https://github.com/linrongc/kdd26_unidot/tree/1399702006a1fc2f86329c8c6133f3f3ac1e6d0d)，根许可证为 [MIT](https://github.com/linrongc/kdd26_unidot/blob/1399702006a1fc2f86329c8c6133f3ac1e6d0d/LICENSE)，可以在保留声明、记录修改后抽取目标代码。

### UniDot 的具体抽取位置

固定作者代码 SHA：`1399702006a1fc2f86329c8c6133f3f3ac1e6d0d`。

- [`common.py:197 CondGatedSwiGLU`](https://github.com/linrongc/kdd26_unidot/blob/1399702006a1fc2f86329c8c6133f3f3ac1e6d0d/common.py#L197)。首批只保留必要条件门控，不带竞赛时间特征和额外交互变体。
- [`unified_model.py:481 MultiChannelSeqPool`](https://github.com/linrongc/kdd26_unidot/blob/1399702006a1fc2f86329c8c6133f3ac1e6d0d/unified_model.py#L481)。其旧注释提到 softmax，实际运算使用 sigmoid；实现依据必须记录到运算层面。
- [`unified_model.py:2455 FMHighway`](https://github.com/linrongc/kdd26_unidot/blob/1399702006a1fc2f86329c8c6133f3ac1e6d0d/unified_model.py#L2455)，另需独立抽取同文件 L2243–2272 的域点积计算。FMHighway 类本身不产生全部输入信号。

作者当前代码删去了[论文 §4.4](https://arxiv.org/html/2608.16797v1#S4.SS4)中的 aggregated Gram。因此首批应标为“作者指定版本的核心子集”，不能称完整论文等价复现。整个 `unified_model.py` 导入竞赛 dataset，不能作为 standalone snippet 原样搬入；目标组件应仅依赖框架和标准库。

## 3. 八个获奖方案如何处理

下表区分论文/代码证据和综述级线索。没有取得原文的方案可以进入研究候选清单，但不能登记为已复现方法。

| 方案 | 值得保留的部分 | 建议与证据边界 |
|---|---|---|
| CRAFT | 上下文条件的缩放、偏置与初始状态回流；后续可研究意图检索 | P1，先独立 transport。论文公式和作者运算存在差异，需明确所选版本；不搬蒸馏和整套比赛训练 |
| UniDAC | 域摘要之间的交互及门控注入；长短历史分支 | P2 候选。本轮尚未取得完整原论文/确认作者实现，当前仅有赛事综述；先复现可核查的基础模块 |
| UniTokenFormer | 用户/候选 token 的投影分配、候选条件增强 | 先补 token 构造实验指南；可从 LENS 原论文提取独立增强组件。暂无充分原文证据支持完整复现 |
| UniDot | 条件门控、多通道池化、显式交互 Highway | P0；作者 MIT 源码可定位。全套双总线架构放到独立组件验证之后 |
| QueryFormer | 先字段交互与候选检索，再用行为摘要探针查询字段，生成域查询 | P1。先单视图查询桥；多视图 packing 单独验证。未确认作者仓库 |
| UniBridge | 局部序列处理 → 少量候选条件意图 → 字段混合 | P2 组合配方；综述可解释路径，但不能据此声称论文细节已充分复现 |
| MaskRec | DualQ 双向查询生成与 TopoMask 类型/域拓扑 | P2；分为两个组件。作者仓库没有确认许可，不直接复制；拓扑稀疏不代表稠密 attention 自动加速 |
| CoSFormer | 有信息来源的 token 与容量共同扩展 | 先进入实验指南；未取得原论文/作者代码，不引用同名 cosFormer 代替，也不搬其外部预训练特征生产过程 |

[QueryFormer §4.2](https://arxiv.org/html/2609.16548v1#S4.SS2)的关键是行为摘要到字段集合的显式查询生成，普通 DIN attention 或单个 MLP 不能替代。多视图共享 K/V 的 packed 执行需要与逐视图版本做同权重数值和梯度对照，不能把多个视图强制共享所有参数。

[MaskRec](https://arxiv.org/html/2608.27005v1)明确区分字段、域查询、历史和记忆 token；其记忆 token 是可学习汇聚位置，不是持久用户记忆。作者代码核查版本为 [`84bcc4e44b558b5b0af6d3601de00945b4a09dd7`](https://github.com/zzhlkw-ai/TAAC2026/tree/84bcc4e44b558b5b0af6d3601de00945b4a09dd7)，完整树未见 LICENSE/COPYING/NOTICE，不能按“GitHub 公开即可复制”处理。

[CRAFT 原论文](https://arxiv.org/html/2609.01655v1) Eq(9) 的记忆项是 `η·(Z₀−Z)`；[作者代码](https://github.com/AshleyLuo001/CRAFT/blob/09c9283b811a97b2de089bcd9219bbd344edaa81/craft_model.py#L80)先得到调制状态 A，再使用 `η·(Z₀−A)`。两者一般不等价。作者版本 `09c9283b811a97b2de089bcd9219bbd344edaa81` 未确认软件许可，因此建议先依据论文实现独立 `ContextualFeatureTransport(context, state, initial_state)`，记录公式版本。完整模型的 mixer 还存在静态维度疑点，未运行确认，不应直接承诺整文件可用。

## 4. 基础论文和资源的利用方式

| 来源 | 融合方式 | 范围控制 |
|---|---|---|
| [HyFormer](https://arxiv.org/html/2601.12681v2) | 提供“查询读历史—查询与字段混合—下一层”的结构对照 | 不等于末端一次融合；先两层小模型，暂缓异步训练方案 |
| [InterFormer](https://arxiv.org/html/2411.09852v4) | 指导序列与字段逐层交换信息、保留细粒度状态 | 后续组装参考，避免一次性加入完整大架构 |
| [OneTrans](https://arxiv.org/html/2510.26104v3) | 混合 token、参数共享差异与渐进压缩，可作为后续独立块 | 属于排序模型，不因使用 Transformer 就排除；跨请求缓存服务、分布式优化不进入 snippet |
| [Wukong](https://proceedings.mlr.press/v235/zhang24ao.html) | 后续显式交互候选：FMB 与 LCB | 与现有 FM/DCN 做独立增量比较，不把名字相似视为覆盖 |
| [LONGER](https://arxiv.org/abs/2505.04421) | 研究历史压缩与近期信息保留 | 输入历史与 mask 已由调用方提供；不得重新构建生产日志 |
| [LENS](https://arxiv.org/html/2605.25583v2) | QueryPos、目标条件查询门控 TCQG、低秩位置偏置 TCPB | 历史排序与对齐规则须明确，不能覆盖 padding mask；未确认作者代码 |
| [RankElastor](https://arxiv.org/html/2605.23191v1) | 可学习 full mixing 与 gated P-FFN 分别作为实验候选 | full mixing 参数按 `(T·D)²` 增长；先验预算检查，再做两项消融 |
| [RankUp](https://arxiv.org/html/2604.17878) | 随机字段分组、global token 与表示诊断补入实验指南 | 外部预训练表示、多任务数据依赖不打包进通用层 |
| 官方 [TAAC2026 资源页](https://huggingface.co/TAAC2026) | 了解输入结构；必要时做接口示例 | 两份公开 sample 各 1000 行，CC BY-NC 4.0；不并入默认包，不作为完整竞赛效果评测 |

**LENS 的最小切口。** TCQG 接收原查询及条件向量，计算 `Q₀ = Qraw · 2σ(Wc)`；TCPB 由条件向量产生低秩 query-position bias，加到注意力 logits。两者相应条件投影零初始化，支持先验证与原计算等价，再测训练收益。bias 可能影响 fused attention 路径，参数少不等于没有时延成本。[原论文 §4](https://arxiv.org/html/2605.25583v2)

**RankElastor 的复用边界。** [作者仓库](https://github.com/vasile-paskardlgm/RankElastor/tree/68e5f29c9147b0c2263a1d28216b500940c057fe)采用 [MIT](https://github.com/vasile-paskardlgm/RankElastor/blob/68e5f29c9147b0c2263a1d28216b500940c057fe/LICENSE.txt)。论文的 full mixing 与作者实现的归一化位置/类型不同；代码名为 `SwiGLU` 的块实际用 GELU，须按真实公式登记。以 T=48、D=192 为例，单个 full mixing 矩阵含 84,934,656 个权重，这是维度计算，不是性能实测。[Token_Mixing.py](https://github.com/vasile-paskardlgm/RankElastor/blob/68e5f29c9147b0c2263a1d28216b500940c057fe/src/Token_Mixing.py)、[utils.py](https://github.com/vasile-paskardlgm/RankElastor/blob/68e5f29c9147b0c2263a1d28216b500940c057fe/src/utils.py)

RankMixer 本轮未确认原作者公开仓库，可依据论文独立实现；其他论文作者仓库中的 baseline 或第三方复现不能标为 RankMixer 官方实现。

官方 sample：[第一阶段](https://huggingface.co/datasets/TAAC2026/data_sample_1000)、[第二阶段](https://huggingface.co/datasets/TAAC2026/second_round_sample_1000)。ID 列表与其附属向量需要对齐；一个 list 字段本身不足以证明它是按时间排序的行为历史。

本轮重点核查获奖方案及与可组合机制直接相关的基础论文，没有声称逐篇审阅文章全部 63 条参考文献。未公开或暂无法取得的原文保留为证据缺口。

## 5. 映射到项目中的位置

下列路径相对仓库根目录。现有模型 API 用 AST 提取顶层 class，源码分发会返回整个模块；最小实现应沿用这些约定。

| 层面 | 建议位置与改动 | 验证方式 |
|---|---|---|
| 可抽取代码 | 新增 `src/model_evo_harness/models/{pytorch,tensorflow}/paper_interactions.py` 与 `paper_unified_sequence.py`；分别承载纯字段块和序列条件块 | 独立模块只导入框架/标准库；不依赖作者 dataset、训练包或本仓库其他模型文件 |
| 方法卡 | `data/supplemental_method_cards.json` 登记机制、精确来源、输入前提、对照实验与失败信号 | 每个 ID 均有真实实现；尚在调研的论文不注册成可用候选 |
| 实现映射 | `data/model_implementations.json` 登记两种框架的 class、输出 contract 和 limitations | 现有约定要求每张卡有 PyTorch/TF 对应项，不能只补一边 |
| family 与指南 | 纯字段块归 `feature_interactions`，历史块归 `sequence_ranking`；分别更新相应 `knowledge/*.md` | family 和 card 的 requires 会取并集；避免让纯字段块错误依赖 event_sequence |
| 结构模式 | `data/structure_patterns.json` 标明一个主结构模式；跨机制联系写入指南 | 当前同一方法只能属于一个 pattern，不重复登记 |
| 组合与实验 | 复用 `composition.py`、`knowledge/horizontal_composition.md`，更新 `knowledge/scaling_architecture.md` 的预算对照建议 | 不需要重写 Agent 循环；逐层交互展开为有限 DAG，保留现有无环检查 |
| 来源追踪 | 方法卡使用已有 `source_url/source_references`；更新 `docs/research/source-attribution.md` 和独立 TAAC 来源说明 | 作者 SHA、许可、抽取位置与修改记录完整；不得给 TAAC 论文伪造 reco 编号 |

**现有能力可以作为积木，但不是同一机制：** `NumericFieldEmbedding` 只是数值字段嵌入；`GroupedFM` 是分组二阶交互，不能替代跨层 Highway；`ParallelBranches` 负责调用与融合，不会自动产生逐层字段—历史交互；现有 TransAct 和 LatentCross 也不能直接代称 QueryFormer 或 CRAFT。

**来源清单的最小处理：** `catalog.py` 当前把 `paper_inventory` 固定读取为 `reco_papers.json`，并验证卡片的 `paper_ids`。第一批可使用现成的来源字段及独立 TAAC 文档，不改核心 loader。若后续确需机器可查询的 TAAC 覆盖表，再增加独立来源清单和联合 ID 校验；保留 Reco-papers 的 115 篇口径及原 API，分别报告来源覆盖率。

**框架约束：** 本仓库当前是 PyTorch 与现代 TensorFlow/Keras 双实现，不能称作 TensorFlow 1.15 兼容。上述机制多数可用基础矩阵、归一化、激活和 mask 运算表达，但 TF1.15 需要单独的 API 适配及真实环境验证；现有现代 TF 的 GPU 结果不能替代它。

## 6. 对 Agent 的实验指导应补什么

建议将以下规则写进相关指南与方法卡的 comparison/failure_signals，不增加新的通用 Agent 框架：

1. 字段 token 数、查询数、宽度、深度分别消融；独立投影与简单复制做对照。增加 token 前说明新增信息或表示视角来自什么已有输入。
2. 固定数据划分、标签定义、训练预算，再比较机制；同时报告参数量、长度分布、dtype、batch、延迟与峰值显存。竞赛单点 AUC 不能作为迁移收益保证。
3. 查询桥、gate、Highway、拓扑 mask 都会改变函数，需要质量验证；仅 packed/unpacked 等价执行要求数值与梯度一致。
4. 候选条件进入后的历史表示不可直接跨候选复用；缓存收益必须建立在同一输入依赖关系上。
5. 一次引入一个机制，最后才评估组合；给出提升、不变、退化三种结果的处理条件。不能只展示组合后总提升，无法定位有效组件。
6. 区分表示诊断指标：RankElastor 的 stable rank 是 `||X||F²/||X||2²`，RankUp 使用奇异值分布熵导出的 effective rank；不要合并成一个含糊的 erank。它们是诊断信号，不能替代业务 holdout 指标。

当前 engine 的晋级检查不会因为多报告一个 latency 数值就自动执行成本门槛。若以后需要强制预算约束，必须单独定义宿主可行性检查与对应测试；本次不把它夹带为模型组件改动。

## 7. 实施验收与暂不纳入的内容

首批的完成条件应覆盖：

- 小张量手算/参考运算与反向梯度；Highway 所有声明的层和域信号确实到达输出。
- mask 方向正确，padding 不贡献结果，全空历史行为明确；被屏蔽连接的输入变化不得影响输出。
- 动态 batch/历史长度、训练/推理开关、不同域数；静态字段数或宽度整除要求明确报出。
- 同权重的两框架对照，以及 TensorFlow 图执行；不只比较 shape 或随机输出是否有限。
- `model_api`/源码提取、catalog、方法卡、结构映射和许可声明均通过；抽取后的模块可以单独加载。
- 新组件若宣称 GPU 可用，需要分别运行其新增 GPU 用例；先前 17 项 GPU 检查不能覆盖尚不存在的代码。

不默认加入：LLM 权重、语言/SID 生成过程、外部 embedding 生产、日志重构、负采样、完整训练配方、Muon/DDP/FlashAttention 集成、蒸馏流水线、跨请求缓存服务。若论文辅助 loss 只依赖调用方已经提供的标签/张量，可另行独立实现；需要新增标签或数据流程时则超出本轮边界。

建议首个实施批次保持四个组件、两套框架、完整方法映射与机制测试的闭环。之后再依据实际对照结果扩展查询生成和长序列模块。
