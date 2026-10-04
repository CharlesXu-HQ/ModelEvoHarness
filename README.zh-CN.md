# ModelEvoHarness

[English](README.md) · [FunRec 覆盖清单](docs/funrec-coverage.md) · [任务适配接口](docs/adapter-contract.md)

ModelEvoHarness 是面向推荐、搜索、广告和营销的 **Agent 主导离线模型迭代框架**。Agent 读取固定任务和历史结果，选择要检验的机制，说明对照与预期，提交候选模型，并根据评估结果反思。Harness 检查数据条件，调度实验并保存记录。任务适配器负责真实训练、数据切分、指标和运行环境。

研究目录完整核对了 [FunRec](https://github.com/datawhalechina/fun-rec) 当前版本中 55 个实质性章节页面、38 个模型模块、54 个支撑模块和 50 个生产后端模块，归入 19 个可交叉的研究方向。38 个模型模块还对应 method card，写明机制、数据条件、对照实验、失败信号和实现边界。**这些模型是参考实例，并未作为代码复制进本项目。** Agent 可以选择可执行的方向、可执行的 method card，也可以提出目录之外的新方向。

## 实验为什么需要 harness

替换模型名称并不能说明实验有效。一次有用的迭代需要说明：现有结果暴露了什么问题，新机制预计改变什么，使用哪些决策前字段，与哪个稳定基线比较，什么结果会推翻假设。ModelEvoHarness 把这些要求写入执行循环；它区分“考虑过的备选方案”与“实际完成的实验”，经验绑定到具体任务和数据集版本。

| 边界 | 负责方 |
| --- | --- |
| 研究目录、family 与 method 适用条件、实验设计校验、Agent 循环、日志及同一运行身份下的任务经验 | ModelEvoHarness |
| 数据语义、切分、候选执行、GPU、目标指标和不确定性计算 | 任务适配器 |
| 假设、候选代码或配置、数据需求及实验反思 | Agent |

## 使用

需要 Python 3.12 或更新版本：

```bash
pip install git+https://github.com/CharlesXu-HQ/ModelEvoHarness.git
model-evo-harness catalog-check
```

按[任务适配接口](docs/adapter-contract.md)提供数据摘要与评估器，再指定 Agent：

```bash
export MODEL_AGENT_API_KEY=...  # 在本机环境中提供，不写入配置或仓库
model-evo-harness run \
  --task my_project.experiments:task \
  --agent-config agent.json \
  --output runs/my-task-v1 \
  --max-steps 4
```

`agent.json` 配置 `provider_url`、`model`、`api_key_env`，以及迭代的 `high` 和复核的 `max`。也可以用 `--agent module:symbol` 提供自己的 Agent，或从 Python 调用 `run_search(...)`。框架本身不依赖特定训练框架、模型供应商、推荐数据集或 CouponEvo。

任务摘要声明所处阶段、已有数据能力和实际字段。目录会为每个方向和 method card 给出 `ready`、`needs_data` 或 `other_stage` 及原因。例如静态优惠券数据可检验部分特征交叉方法；序列方法需要按决策时点截断的真实行为历史。`ready` 只表示输入条件具备，不表示模型已经预装或指标会提升。

日志保存基线、候选指标、评估失败、研究设计、反思、最佳候选以及任务和目录指纹。更换数据集、目标、包实现或目录后不能混用旧实验。经验绑定到同一运行身份和数据集指纹。最终留出集由任务适配器隔离，Agent 只依据验证结果迭代。

## 首个接入项目

[CouponEvo](https://github.com/CharlesXu-HQ/CouponEvo) 提供随机发券任务、PyTorch 候选沙箱和配对策略评估。ModelEvoHarness 提供独立研究目录和实验设计校验。两个仓库分别发布与测试；CouponEvo 接入前后的对照结果会作为集成验证，不预设指标提升。

## 来源与许可证

目录对 FunRec 的章节和模型路径建立固定版本的映射。FunRec 标注为 [CC BY-NC-SA 4.0](https://github.com/datawhalechina/fun-rec/blob/master/pyproject.toml)。ModelEvoHarness 的指南、method card 和代码独立编写，采用 [Apache-2.0](LICENSE)，未复制 FunRec 模型源码。
