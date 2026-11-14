# ModelEvoHarness

[English](README.md) · [本地技术知识](docs/knowledge/README.md) · [模型参考实现](docs/models.md) · [任务适配接口](docs/adapter-contract.md)

ModelEvoHarness 是面向推荐、搜索、广告和营销的 **Agent 主导离线实验框架**。Agent 读取固定任务、历史实验和本地技术资料，提出可证伪的改动，通过业务方的评估器执行，再根据结果记录经验。可迭代的对象不仅是模型结构，还包括 loss、采样、困难样本挖掘、优化过程、校准、特征表示和预测到决策的规则。

## 给算法工程师提供什么

| 内容 | 作用 |
| --- | --- |
| [29 篇方向指南与 10 篇通用指南](docs/knowledge/README.md) | 覆盖目录中的全部技术方向，原创说明机制、输入条件、对照实验、失败信号及特征缺口如何判断。Agent 直接读取适用方向指南及通用训练、探索和横向组合指南，无需联网获取上游仓库。 |
| 131 张 method card | 统一记录 128 种模型与训练方法，以及 3 个分析工具。[结构选择模式](docs/multi-source-guidance.md)指导模型改动，[分析模式](src/model_evo_harness/data/analysis_patterns.json)指导测量和筛选。`ready` 表示已声明任务前置条件；候选内部张量仍需构建和校验。 |
| [训练策略](src/model_evo_harness/data/training_patterns.json) | 把 loss、负采样、样本加权、正则化、校准和数据增强，与结构实验放在同一套可证伪的实验协议下。 |
| [128 种双框架模型与训练方法](docs/models.md) | [模型实现清单](src/model_evo_harness/data/model_implementations.json)共 258 项，含通用 TwoTower 的双框架实现。`model_api` 提供类路径和契约；训练与评估由业务方执行。 |
| [3 个可供双框架宿主使用的分析工具](docs/research-guided-components.md) | [分析实现清单](src/model_evo_harness/data/analysis_implementations.json)共 6 项，包含表示诊断、梯度特征筛选及成本约束的子集重训；最后一项是双框架宿主共用的标准库实现。`analysis_api` 提供类路径和契约，供宿主显式调用。 |
| 实验日志 | 保存任务、数据集和评估协议指纹，以及假设、已执行实验、失败记录、技术经验与业务经验。数据集或协议改变后，建立新的证据上下文。 |

[来源署名](docs/research/source-attribution.md)记录研究目录参考的原始资料。仓库内的说明和模型代码均为原创，没有镜像上游正文或源码。[研究覆盖说明](docs/source-coverage.md)列出具体范围。

## 根据内部观测选择 backbone 内的改动

新增[研究组件](docs/research-guided-components.md)：逐层有效秩、方差与分支相似性诊断；保护交互字段的梯度特征筛选；逐层开放的交互边；历史事件上下文编码；用户、历史、候选之间独立归一化的注意力。五个组件均有独立 PyTorch 和 TensorFlow 源码，按调用用途分目录：

```text
src/model_evo_harness/
├── analysis/{pytorch,tensorflow}/research_diagnostics.py  # 诊断与筛选
└── models/{pytorch,tensorflow}/
    ├── progressive_interaction.py                       # 候选模型前向计算
    └── contextual_history.py                           # 候选模型前向计算
```

宿主诊断基线或已完成实验时，显式把采集的激活和训练梯度交给分析工具。Agent 据此提出模型改动，宿主接入模型组件后训练并评估候选。两类源码均可通过 `read_references` 读取；发现接口和源码读取不会自动安装 hook、执行工具或训练。分析工具不能作为独立模型实验的 `method_id`。特征筛选保留声明的交互组，删字段前仍需重训验证。

## 在现有 backbone 内增加交互和兴趣结构

新增 [CAN 与 ComiRec-SA/DR 组件](docs/rec-model-components.md)，用于字段对交互与多兴趣历史分支；CIN 激活和 SDIM 归一化可作为显式消融选项。Agent 可以保留当前 backbone，依据已观测瓶颈选择适用组件。独立的成本筛选流程保护交互字段，调用宿主实际裁剪、重训、评估，再在统一协议下比较质量与延迟。每项能力都有输入前提、具体变体和固定对照，是否优先尝试仍由任务证据决定。

## 折扣感知的营销模型迭代

新增 [DANet 机制组件](docs/danet-components.md)：商品折扣历史滤波、候选商品条件下的购买属性注意力、促销上下文门控，以及仅在有效购买样本上监督的折扣辅助回归。组件可以接入现有兴趣 backbone；Agent 分别检查历史输入和真实折扣标签是否具备，并能独立消融结构与 loss。优惠券增量效果仍由接入方的因果评估验证。

## 显式特征交叉：先判断关系，再选择结构

宿主提供 `interaction_views` 并启用 `interaction_plan_required` 后，Agent 区分字段表示与当前交互覆盖，基于实际证据排列实验优先级，并给出同源码的模型配置对照。数值、类别、缺失指示等视图都可参与；没有“某类字段必须先做 FM”的固定规则。也允许暂缓交叉，优先诊断尺度、融合方式或训练问题。

PyTorch 与 TensorFlow 的独立原生 `NumericFieldEmbedding`、`GroupedFM` 提供数值字段向量及可选择的组内/组间二阶交互。Agent 通过 `include_interactions=true` 读取源码，在现有 backbone 中组合字段子集、并行分支与融合；这些组件不是封闭的策略列表。对照是否执行、配对指标如何，均由宿主提供；计划、覆盖声明、代码执行与收益归因分别记录。[设计与合同](docs/explicit-interactions.md)。

## 实验闭环

1. 读取现有字段、目标、评估协议和历史实验，用实际指标诊断问题，查阅适用的本地指南。
2. 提出一个机制、固定对照、预期效果和否定条件。生成候选前，读取所选模型的完整本地源码和实现边界。任务适配器负责在同一验证口径下训练和评估。
3. 结合不确定性分析预测与决策指标，分别记录**技术经验**和**业务经验**。业务结论必须引用任务适配器提供的、带指标和不确定性的 `business_observations`；没有可观测证据时记为 `not_observable`。
4. 继续、停止或请求人工完善数据集。`future_feature_suggestions` 要在两个机制不同的已评估实验之后记录；如果业务方在 `domain_requirements` 中明确声明缺失前提，也可立即记录。正式的 `request_data` 需引用实验和证据，或引用对应的业务前提。

宿主可以在任务快照、基线和实验评估中提供 `evidence` 事实，每条事实包含 ID、内容、来源、`observed`/`declared` 状态和 `task`/`trial` 范围。提供事实后，实验提案须引用实际观测的瓶颈；Agent 自报的 `change_factors` 只是计划，只有独立的宿主 `implementation_check` 与 `change_audit` 才能支持单机制归因。实验型数据请求须引用两个不同机制实验的实测、特定于实验的 `data_gap_candidate` 诊断。仅有元数据声明时，可记录不阻断实验的 `audit_recommendations`。校验器保证引文来源和实验归属，不能证明 Agent 每一句解释都正确。

任务适配器负责原始数据、训练、GPU、切分、目标、不确定性、最终留出集和行动约束。Harness 校验实验记录与适用条件；它不会根据字段名猜测业务语义，也不保证某个模型带来提升。具体接口见[任务适配文档](docs/adapter-contract.md)。

## 提高探索效率

优先目标是在固定实验预算内产出更多可执行、有信息量的假设。与未接 Harness 的 Agent 比较实验完成率、实现失败、有效反思和机制覆盖；验证集提升与独立验证结论分别报告。不同的机制名称不自动等于真正的研究多样性。

默认按分数比较候选，并排除宿主确认实现矛盾的候选。宿主可通过 `require_verified_implementation=True`（CLI 为 `--require-verified-implementation`）启用严格晋级，只允许宿主核验通过的候选成为优胜者；未核验实验仍保留，供诊断和后续探索。每轮记录晋级资格与原因，策略随运行身份固定，恢复运行时不能切换。实现核验、统计确认和组件归因分别判断，严格模式不要求每轮都完成消融。

内置供应商接口支持有限轮次的 `read_reference`，读取完整 PyTorch 或 TensorFlow 模型模块、分析工具及训练辅助代码。Agent 声明选用已收录模型方法却未读源码时，框架先提供源码，再要求生成候选；引擎保存宿主实际读取的内容哈希，并独立记录向 completion 回调交付源码的事件，忽略 Agent 自报的引用记录。自定义适配器通过 `read_references` 与 `call_with_references` 接入同一记录流程。读取和交付均不证明候选已正确使用源码。读取不占训练试验次数。每个实现声明输出契约、宿主责任和原方法中未实现的部分；完整训练流水线由宿主负责。

参见[本轮审计与剩余缺口](docs/research/harness-audit-2026-10-05.md)及[探索策略](src/model_evo_harness/knowledge/exploration_strategy.md)。

## 在 backbone 内部改进，选择性继承局部成果

默认优先围绕适用的 backbone，读取本地参考源码后，在候选内部编写编码器、交叉层、head、目标函数或训练过程的改动。估计方式与表征 backbone 分开记录；证据支持时可以切换，不规定先跑满多少轮，也不要求逐个尝试模型名称。

每个模型组合记录组件及代码位置。下一轮必须对父版本的各个组件逐项选择**保留、适配、放弃或重新验证**，写清来源试验、兼容性和验证方法。因此切换 backbone 时会审查已有局部成果。组件引用与整体模型引用一样触发有次数上限的源码读取；联合实验的收益保留归因限制，组件被迁移并不意味着已经证明有效。

数据条件仍是前提：统计特征可以支持交叉或训练改进；序列、目标感知结构需要真实声明的输入。稳定的 `estimator_id` / `backbone_id` 与可更新的结构描述分开；局部修改省略 ID 时由宿主继承父记录。详见[组件化迭代协议与示例](docs/compositional-evolution.md)。

启用协议后，每轮还评估[横向子网组合](docs/horizontal-composition.md)：依据字段语义和可检验瓶颈，组合任意适用模块的并行实例，明确参数共享与融合。`feature_groups` 声明不赋予序列能力，延期扩展时仍记录已有分支组。通过 `include_composition=true` 可读取原生 PyTorch、TensorFlow 的 `ParallelBranches` 源码，分支类型与候选自写的融合方式均不封闭枚举。内置 provider 默认启用 `model_design_required` 和 `horizontal_expansion_required`，自定义 adapter 可显式启用；旧记录仍按原快照读取。

## 接入

需要 Python 3.12 或更新版本。安装后可检查研究目录：

```bash
pip install git+https://github.com/CharlesXu-HQ/ModelEvoHarness.git
model-evo-harness catalog-check
```

任务适配器实现 `snapshot()`、`baseline()` 和 `evaluate()`；Agent 实现 `propose()` 和 `reflect()`，也可使用内置的 OpenAI 兼容供应商接口：

```bash
export MODEL_AGENT_API_KEY=...  # 密钥只存放在本机环境中
model-evo-harness run \
  --task my_project.experiments:task \
  --agent-config agent.json \
  --output runs/my-task-v1 \
  --max-steps 4
```

日常提案和反思使用 `iteration_effort`；任务评估返回 `review_required: true` 时，反思使用 `review_effort`。

`agent.json` 配置 `provider_url`、`model`、`api_key_env`、迭代的 `iteration_effort`（`high`）和复核的 `review_effort`（`max`）。Python 项目也可调用 `run_search(...)`。Harness 本体不依赖特定模型供应商或训练框架；只有导入可选模型参考实现时才需要 PyTorch 或 TensorFlow。

[CouponEvo](https://github.com/CharlesXu-HQ/CouponEvo) 是首个接入方，提供随机发券任务、候选执行和策略评估。引入 Harness 或模型参考实现本身不构成 uplift 或净收益提升的证据。

## TAAC2026 张量组件

已加入 RankMixer 字段交互、UniDot 条件 SwiGLU、多通道历史池化和跨层显式交互 Highway，均提供独立 PyTorch / TensorFlow-Keras 实现，并映射到方法卡、结构模式及知识指南。输入为已有 embedding 张量，不要求 SID，也不接管数据构造或训练流程。接口、mask 约定、作者代码差异与来源见[组件说明](docs/taac-components.md)。这些实现不代表 TF1.15 兼容或业务收益已经得到验证。

另有四个 [QueryFormer 查询生成组件](docs/queryformer-components.md)，覆盖单视图字段交互、历史检索与查询生成（公式 2–4），提供对应的 PyTorch / TensorFlow-Keras 接口。门控、归一化、池化和空输入处理的工程选择均有说明；下游序列混合、读出与多视图 packed 执行不在此次范围内。

## 许可证

本项目采用 [Apache-2.0](LICENSE)。[来源署名与复用边界](docs/research/source-attribution.md)说明与 FunRec、DeepCTR、Torch-RecHub、RecBole 等研究资料的关系。

## 可组合的 CTR 与序列实验组件

[FuxiCTR 覆盖说明](docs/fuxictr-coverage.md)逐项对应其 57 项目录，本次补充 43 个双框架方法。
Agent 可以在当前 backbone 内调整交叉与融合、字段代数、历史筛选、embedding 融合或辅助目标，
并读取本地完整源码、输入契约和对照实验要求。组件支持组合与局部消融；覆盖范围是机制与配方，
完整论文复现及业务收益需要由宿主实验进一步验证。
