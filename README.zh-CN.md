# ModelEvoHarness

[English](README.md) · [本地技术知识](docs/knowledge/README.md) · [模型参考实现](docs/models.md) · [任务适配接口](docs/adapter-contract.md)

ModelEvoHarness 是面向推荐、搜索、广告和营销的 **Agent 主导离线实验框架**。Agent 读取固定任务、历史实验和本地技术资料，提出可证伪的改动，通过业务方的评估器执行，再根据结果记录经验。可迭代的对象不仅是模型结构，还包括 loss、采样、困难样本挖掘、优化过程、校准、特征表示和预测到决策的规则。

## 给算法工程师提供什么

| 内容 | 作用 |
| --- | --- |
| [22 篇方向指南与 8 篇通用指南](docs/knowledge/README.md) | 覆盖目录中的全部技术方向，原创说明机制、输入条件、对照实验、失败信号及特征缺口如何判断。Agent 直接读取适用方向指南及通用训练、探索指南，无需联网获取上游仓库。 |
| 44 张 method card 和[结构选择模式](docs/multi-source-guidance.md) | 根据实际观察到的问题判断何时值得试某种方法。数据条件为 `ready` 只表示具备输入，不代表已有实现或预计收益。 |
| [训练策略](src/model_evo_harness/data/training_patterns.json) | 把 loss、负采样、样本加权、正则化、校准和数据增强，与结构实验放在同一套可证伪的实验协议下。 |
| [44 种直接基于框架的模型核心实现](docs/models.md) | 每张 method card 都有独立编写的 PyTorch 和 TensorFlow 代码，另附通用 TwoTower。 [实现清单](src/model_evo_harness/data/model_implementations.json)给出准确类路径。输出可能是 logit、检索分数、概率或表示向量；对应 loss 与评估器由业务方提供。不依赖其他模型包装库。 |
| 实验日志 | 保存任务、数据集和评估协议指纹，以及假设、已执行实验、失败记录、技术经验与业务经验。数据集或协议改变后，建立新的证据上下文。 |

[来源署名](docs/research/source-attribution.md)记录研究目录参考的原始资料。仓库内的说明和模型代码均为原创，没有镜像上游正文或源码。[研究覆盖说明](docs/source-coverage.md)列出具体范围。

## 实验闭环

1. 读取现有字段、目标、评估协议和历史实验，用实际指标诊断问题，查阅适用的本地指南。
2. 提出一个机制、固定对照、预期效果和否定条件。生成候选前，读取所选模型的完整本地源码和实现边界。任务适配器负责在同一验证口径下训练和评估。
3. 结合不确定性分析预测与决策指标，分别记录**技术经验**和**业务经验**。业务结论必须引用任务适配器提供的、带指标和不确定性的 `business_observations`；没有可观测证据时记为 `not_observable`。
4. 继续、停止或请求人工完善数据集。`future_feature_suggestions` 要在两个机制不同的已评估实验之后记录；如果业务方在 `domain_requirements` 中明确声明缺失前提，也可立即记录。正式的 `request_data` 需引用实验和证据，或引用对应的业务前提。

宿主可以在任务快照、基线和实验评估中提供 `evidence` 事实，每条事实包含 ID、内容、来源、`observed`/`declared` 状态和 `task`/`trial` 范围。提供事实后，实验提案须引用实际观测的瓶颈；Agent 自报的 `change_factors` 只是计划，只有独立的宿主 `implementation_check` 与 `change_audit` 才能支持单机制归因。实验型数据请求须引用两个不同机制实验的实测、特定于实验的 `data_gap_candidate` 诊断。仅有元数据声明时，可记录不阻断实验的 `audit_recommendations`。校验器保证引文来源和实验归属，不能证明 Agent 每一句解释都正确。

任务适配器负责原始数据、训练、GPU、切分、目标、不确定性、最终留出集和行动约束。Harness 校验实验记录与适用条件；它不会根据字段名猜测业务语义，也不保证某个模型带来提升。具体接口见[任务适配文档](docs/adapter-contract.md)。

## 提高探索效率

优先目标是在固定实验预算内产出更多可执行、有信息量的假设。与未接 Harness 的 Agent 比较实验完成率、实现失败、有效反思和机制覆盖；验证集提升与独立验证结论分别报告。不同的机制名称不自动等于真正的研究多样性。

内置供应商接口支持有限轮次的 `read_reference`，读取完整 PyTorch 或 TensorFlow 模块及训练辅助代码。Agent 声明选用已收录方法却未读源码时，框架先提供源码，再要求生成候选；实验日志保存实际读取内容的哈希。读取不占训练试验次数。每个实现声明输出契约、宿主训练责任和原方法中未实现的部分；当前提供的是机制参考核心，尚不是 44 套完整训练流水线。

参见[本轮审计与剩余缺口](docs/research/harness-audit-2026-10-05.md)及[探索策略](src/model_evo_harness/knowledge/exploration_strategy.md)。

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

## 许可证

本项目采用 [Apache-2.0](LICENSE)。[来源署名与复用边界](docs/research/source-attribution.md)说明与 FunRec、DeepCTR、Torch-RecHub、RecBole 等研究资料的关系。
