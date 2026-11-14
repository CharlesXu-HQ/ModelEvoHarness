# rec_models：结构增量与交叉验证

> 状态说明：本文保留实施前的评估结论。后续已实现 CAN、ComiRec-SA/DR、CIN/SDIM 选项和 COLD 式分析流程，见[组件说明](../rec-model-components.md)与[发布验证](rec-components-release-validation-2026-11-14.md)。下文“当前”“未新增”等措辞对应 `5dd85dc` 评估基线。

## 结论与核对边界

`lmxhappy/rec_models` 对 Harness 的价值包括缺失的结构机制，以及检查同名模型变体的第二份实现。它是混合来源的学习/实验集合，不能把目录名或两份同源代码一致当成正确性证明。

- 固定来源：[`966f5f6`](https://github.com/lmxhappy/rec_models/tree/966f5f66f07da78d16f16d44a3978be7235fd7a9)，提交时间 2024-04-01。
- Harness 基线：`5dd85dc`，包含 FuxiCTR 全量机制补充；PR #4 的两次 GitHub CI 均已通过。
- 完整目录扫描：299 个普通文件、26 个 Git 引用条目；41 个技术/实验顶级目录不等于 41 个模型。已取得选定的 188 个源码/文档/配置文件及 DSIN 许可证，并核验其 Git blob SHA；模型核心重点审查见分项报告。无根目录 `.gitmodules`，不将未解析 Git 引用算作已读取代码。
- 本次为结构和数值检查；没有运行第三方训练脚本，没有进行数据集/GPU效果实验，也不声明全部 127 个方法已获得新的正确性认证。

## 值得补充的方向

| 方向 | 实际增量 | 建议与输入条件 |
| --- | --- | --- |
| CAN / Co-Action | 一侧 ID 查询出的参数驱动另一侧特征的小网络，结合多阶输入与逐层输出 | 优先作为 backbone 内可组合子网络；与已有 APG 的条件生成矩阵区分。需要有意义的字段/ID、组合支持量和参数预算。 |
| ComiRec-SA / DR | 候选无关多兴趣编码、路由变体，以及多兴趣检索结果的可控聚合 | 有真实行为历史和 item 身份时补充；SA 是明确增量，DR 先与当前 MIND 原语逐项对照。召回聚合协议不能冒充排名层的 DIN 替代。 |
| CIN 的 identity / ReLU | 同一向量交叉机制的不同非线性假设 | 当前固定 ReLU；建议提供显式变体及相同 embeddings/head 下的受控消融。代码已存在不代表论文默认变体已实现。 |
| SDIM 的聚合与 hash 约定 | 未归一化求和与逐 hash L2 归一化、精确零投影约定 | 属于已有组件的语义扩展；固定 hash 投影、样本和下游头后比较，避免把归约差异当成模型质量提升。 |
| COLD 式成本约束特征选择 | SE 门控已有近似原语，额外价值主要是筛选、移除、重训、质量/成本比较 | 优先补实验与训练协议；门控权重不能直接当作因果重要性或删特征证明。现有分析目录继续独立。 |
| Node2vec / AutoRec / GBDT+LR | 图游走偏置、交互矩阵重建、树叶编码 | 有图/交互矩阵或可靠树特征管线时考虑；图采样和树叶构造属于宿主数据/表示契约，不能从普通匿名统计列推断这些输入存在。 |

CAN 与 ComiRec 的作者源码及与现有原语的区别，详见[结构缺口报告](rec-models-gap-review-2026-11-14.md)。这是后续建议，本轮未新增这些模型或改动模型默认行为。

## 已完成的交叉检查

### MMoE：当前核心汇聚公式通过

对同一组手工映射权重，分别在 PyTorch/TensorFlow 验证每个任务独立 softmax gate 对共享专家的加权和。输入和 gate 非均匀，能区分任务 gate 是否错误共享。两个结果均匹配独立标量计算。

来源：[rec_models 的 MMoE_Layer](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/MMoE/src/MMoE.py#L24-L55)、[原作者论文页](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)。

### PLE：不能照来源代码覆盖当前 shared routing

来源 [`CGC_Layer`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/ple_cgc/PLE/src/PLE.py#L43-L82) 的共享 gate 仅混合 shared experts；[原论文](https://doi.org/10.1145/3383313.3412236) §4.2 / Fig.5 则要求中间共享路径综合本层全部 shared/task experts。Harness 已实现后者。

两层解析构造中，第一层专家斜率为 shared=2、task0=3、task1=5，所有 gate 均匀，第二层专家和输出头为恒等。task0 输出应为 `35*x/12`；batch `x=[1,2]` 对 task1 第一层专家权重的梯度应为 `0.5`。双框架均通过，证明了需要保留的跨任务共享路径。这里只验证当前小型结构，未声称复现原论文完整训练配方。

### CIN：收缩与字段顺序通过，激活变体不同

对非对称权重、正负 embedding、两层不同宽度，独立逐字段/逐坐标求和与两份本地实现一致。当前组件在每层施加 ReLU；来源 [`impl_2/CIN`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/xDeepfm/impl_2/xDeepFM/layer.py#L52-L71) 没有该非线性。[原论文 §4.4](https://arxiv.org/html/1803.05170) 主要使用 identity 并比较不同激活。此差异应作为结构变体记录，不能只凭同名要求输出逐元素相等。已补充本地双框架 manifest 和 Agent 知识说明。

### SDIM、ETA、FFM、DeepCrossing

独立检查重点为 hash collision/归约、padding 排除、partner 轴和残差路径，见[重叠结构审查](rec-models-overlap-review-2026-11-14.md)。固定小 tensor 的双框架解析值、掩码及梯度检查通过。对明确的简化或变体保留原行为；DeepCrossing 的前置投影、ETA 的单头读出及 SDIM 的零投影 hash 约定已补充到模型契约。

发现并修复一项接口缺陷：TensorFlow DeepCrossing 首次调用曾忽略声明的 `input_dim`，让错误宽度静默创建参数。新增一行 `InputSpec`；错误宽度回归先出现 **2 failed / 1 passed**，修复后 **3 passed**，合法宽度的输出和梯度保持有效。对应[回归测试](../../tests/test_deepcrossing_input_contract.py)。

## 验证产物与限制

新增 [`test_rec_models_crosscheck.py`](../../tests/test_rec_models_crosscheck.py) 的六项双框架测试及 DeepCrossing 三项回归均通过。修复后运行完整本地测试：**789 passed, 21 skipped, 388 subtests passed**，耗时 29.09 秒。GPU 专用用例在本机跳过；运行产生 3942 条依赖弃用警告（`gast` / Python AST）。日志位于 `/private/tmp/rec-models-final-suite-20261114.log`。重点数值审查的额外探针结果记在分项报告，不计入此测试数。

本轮修改包括上述一行输入契约修复、本地回归测试、研究报告和实现边界说明；尚未提交或推送。没有引入旧版 TensorFlow、DeepCTR/FuxiCTR 包装器，没有宣称业务提升。下一步实现应优先复用原生机制，并为新增变体保留可辨识的结构参数、适用条件和最小消融。

完整目录与来源边界见[清单](rec-models-inventory-2026-11-14.md)。
