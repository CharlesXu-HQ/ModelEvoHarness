# rec_models 重叠机制独立核查：SDIM、ETA、FFM、DeepCrossing

核查日期：2026-11-14。`rec_models` 固定于 [`966f5f66f07da78d16f16d44a3978be7235fd7a9`](https://github.com/lmxhappy/rec_models/tree/966f5f66f07da78d16f16d44a3978be7235fd7a9)；Harness 基线为 `5dd85dc8c9d1f8750b4abb4585b7f39cfc28dfac`，本轮另有明确列出的工作区修复与说明更新。

本报告独立检查公式、张量布局、掩码和梯度边界。没有运行来源训练脚本、下载训练集、训练模型或验证业务收益。没有把 `rec_models` 当作正确性标准，也不据此给出整个仓库或合并批准。MMoE、PLE、CIN 由另一审查分工覆盖。

## 结论与处理

| 项目 | 分类 | 核查结果与处理 |
| --- | --- | --- |
| SDIM 碰撞求和及跨 hash 平均 | 已声明简化 | 两套代码都省略论文逐 hash 的 L2 归一化；Harness 已明确说明，保留当前行为。 |
| ETA Hamming 检索与注意力 | 核心机制一致 | 小 tensor 的注意力值与梯度符合公式；单头 D 维投影是当前组件边界。 |
| 来源 ETA 对 padding 的有限分值屏蔽 | 来源问题，本地已规避 | 最大有效距离与 padding 打平可选出无效位置；Harness 用 `-inf` 屏蔽。 |
| SDIM/ETA 精确零投影 | 哈希约定差异 | 本地用 `< 0`，来源用 `> 0`；非零投影结果等价，精确零投影可能改变检索/碰撞，应明确记录。 |
| FFM 双向 partner embedding 点积 | 核心机制一致 | 成对索引、顺序、双轴掩码与梯度均符合当前参数库契约。来源目录只有说明与 xlearn 调用。 |
| DeepCrossing 输入投影与残差宽度 | 紧凑变体，需要说明 | 本地先投影到 `hidden_dim`，残差块全程保持该宽度；论文与来源保持 stacking 后的输入宽度。主审查补充清单边界，保持模型行为。 |
| TensorFlow DeepCrossing 忽略 `input_dim` | P2 接口缺陷，已修复 | 首次调用错误宽度曾被静默接受；增加一行 Keras `InputSpec`，先红后绿回归通过。 |

## SDIM：公式与布局

[论文 §4.1、式 (9)–(12)](https://arxiv.org/html/2205.10249) 将同签名历史向量相加，逐 hash 做 L2 归一化，再平均多个 hash 输出。Harness 的 `SDIMCollisionPooling` 则实现：

```text
c[b,h,l] = valid[b,l] AND all_bits(hash(history[b,l]) == hash(target[b]))
per_hash[b,h,:] = sum_l c[b,h,l] * history[b,l,:]
representation[b,:] = mean_h per_hash[b,h,:]
```

本地定位：[PyTorch `history_retrieval.py:119`](../../src/model_evo_harness/models/pytorch/history_retrieval.py#L119)、[TensorFlow `history_retrieval.py:121`](../../src/model_evo_harness/models/tensorflow/history_retrieval.py#L121)。对应来源 [`SDIM.py:150`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/SDIM/src/SDIM.py#L150) 的 `embedding_bag(mode='sum')` 后按 hash 平均，亦未归一化。清单 `sdim_collision_pooling` 的限制已写明 unnormalized source option，不能因其不同于论文就误报为未声明错误。

固定 `target=[3,1]`，有效历史依次为 `[1,2],[3,4],[4,1],[-1,2],[-1,-1]`，追加一个无效 NaN 位置。两个 hash 的旋转矩阵分别为 `[[1,0],[0,1]]` 和 `[[1,1],[1,-2]]`：

| 可检查量 | PyTorch / TensorFlow 结果 |
| --- | --- |
| `per_hash` | `[[8,7],[4,1]]` |
| 当前输出 | `[6,4]` |
| 按论文逐 hash L2 后平均 | `[0.8613596,0.4505201]` |
| 输出坐标和对各历史事件的梯度系数 | `[0.5,0.5,1,0,0,0]`，每个事件两坐标相同 |
| target 梯度 | `None`，离散碰撞路径不训练 target |

另用两个 batch（第二个 batch 的 target 和历史全部取反）确认输出为 `[[6,4],[-6,-4]]`。隔离重写来源的 `D,H,bits` rotation、`H,B,L` collision、`H*B,D` embedding-bag 布局，得到相同结果。本地 `H,D,bits` 布局及 `B,H,L` collision 没有串 batch/hash。后续如研究归一化，应保持同一 host/head/数据划分比较 sum 与逐 hash L2 平均；本轮没有做效果实验。

## ETA：距离、掩码与梯度

[ETA 论文式 (4)–(9)、§4.5.1](https://arxiv.org/pdf/2108.04468) 支持固定 hash、按 Hamming 距离选择近邻，再在原始 embedding 上计算注意力并回传梯度。来源 [`ETA.py:153`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/eta/ETA/src/ETA.py#L153) 的负距离 top-k 与本地排序方向一致。本地 [PyTorch `history_retrieval.py:94`](../../src/model_evo_harness/models/pytorch/history_retrieval.py#L94) 和 [TensorFlow `history_retrieval.py:97`](../../src/model_evo_harness/models/tensorflow/history_retrieval.py#L97) 使用单头、D 维 Q/K/V，无独立输出投影；不代表完整多头原模型。

固定 identity hash、Q/K/V，`target=[1,1]`，历史为 `[1,2],[3,4],[-1,2],[-1,-1]` 及无效 NaN，`K=3`：

- 选中位置为 `[0,1,2]`，Hamming 距离为 `[0,0,1,2,0]`（最后的 0 按契约须结合有效掩码解释）。
- 注意力为 `[0.05506019,0.93155378,0.01338605]`；两框架输出均为 `[2.83633542,3.86310744]`，与独立 softmax 计算一致。
- target 梯度非零，两框架分别约 `[0.50385350,0.39595896]` 与 `[0.50385255,0.39595762]`；未选中的有效事件和无效 NaN 事件梯度为零。
- `K` 大于有效数量、全无效历史、`L=0` 均按契约返回零读出；无效槽位为 `-1/False/0`。

**来源问题：** [`ETA.py:160`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/eta/ETA/src/ETA.py#L160) 以 `-hash_bits` 填充无效分数，该数值与最远有效项相同。一个 bit、target `1`、仅第 4 个历史事件 `-1` 有效时，来源四个分数全是 `-1`；隔离表达式在本次 PyTorch 2.12.0 CPU 中 top-1 选择了无效 index `0`。具体平局索引与框架实现有关，缺陷在于无法保证有效项优先。本地 `_select` 在排序前以 `-inf` 屏蔽无效项，`K=6` 返回 `[3,-1,-1,-1,-1,-1]`，无需复制来源行为。

**零投影约定：** 来源 `relu(sign(projection))` 等价于 `projection > 0`；本地为 `projection < 0`，零均为 False。投影全部非零时整体 bit 翻转保持 Hamming 距离和碰撞结果；存在精确零时不等价。固定 `target=0`、history `[-1,1]`、rotation `1`：本地 ETA 选 index `1`、SDIM 输出 `1`；来源约定选 index `0`、输出 `-1`。这是应公开的边界约定，不能称为任意实数输入上的逐值等价。ETA 原 docstring 已说明 `<0`；本轮清单同步补充。

## FFM：partner 参数与输出契约

[FFM 原论文式 (4)](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) 使用双方针对对方 field 的向量点积，并乘对应特征值。来源 [notebook 的公式单元](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/FFM/fm_deep_dive.ipynb#L115) 说明该式；[训练单元](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/FFM/fm_deep_dive.ipynb#L719) 调用 `xlearn.create_ffm()`，没有可独立审查的原生 FFM kernel，不能声称实现级交叉认证。

本地 [PyTorch `field_products.py:35`](../../src/model_evo_harness/models/pytorch/field_products.py#L35)、[TensorFlow `field_products.py:34`](../../src/model_evo_harness/models/tensorflow/field_products.py#L34) 接收 host 提供的 `[B,F,F,D]` 参数相关 embedding。`(i,j)` 表示 field i 针对 field j 的 embedding，按 `(0,1),(0,2),(1,2),...` 输出 `dot(e[i,j],e[j,i])`，不自带 embedding 表、线性项、求和预测头或 trainer。对非单位数值特征，host 必须在输入向量中包含其幅度；把同一 embedding 重复给所有 partner 只得到 FM 对照。

三 field、两坐标的非对称 probe 使用各方向向量 `[1,2]/[3,4]`、`[5,6]/[7,8]`、`[9,10]/[11,12]`，两框架均输出 `[11,83,219]`。输出和的每个 off-diagonal embedding 梯度等于其反向 partner，diagonal 梯度为零。屏蔽中间 field 并将相关行、列替换成 NaN 后，输出 `[0,83,0]`，被屏蔽参数梯度全零且其他梯度有限。当前组件每 field 一个向量的范围，不应扩张为任意多活跃特征、含同 field 特征对的完整 FFM tokenizer。

## DeepCrossing：变体与已修复缺陷

[Deep Crossing 论文 §5.1、§6.1、Fig.4](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf) 的 stacking 是拼接；残差单元内部先映射到隐藏宽度，再映射回输入宽度后加回输入。来源 [`DeepCrossing.py:49`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/DeepCrossing/src/DeepCrossing.py#L49)、[`ResidualBlock:85`](https://github.com/lmxhappy/rec_models/blob/966f5f66f07da78d16f16d44a3978be7235fd7a9/DeepCrossing/src/DeepCrossing.py#L85) 保留这一宽度结构。

本地 [PyTorch `paper_ranking.py:17`](../../src/model_evo_harness/models/pytorch/paper_ranking.py#L17)、[TensorFlow `paper_ranking.py:12`](../../src/model_evo_harness/models/tensorflow/paper_ranking.py#L12) 额外先做 `input_dim -> hidden_dim` 的可学习仿射投影，然后使用固定 `hidden_dim` 残差块。此处 `hidden_dim` 的意义与原论文不同。固定投影保留输入前两维、残差支路全零、读出权重全一，对 `[1,2,999]` 的输出为 `3`、梯度为 `[1,1,0]`；维持原三维输入的零残差单元搭配求和 head 则输出 `1002`。这个例子只说明结构差异，不说明哪一种效果更好。主审查补充了变体说明，未更改该结构。

**P2，已修复：TensorFlow 构造参数 `input_dim` 原先未被使用。** `DeepCrossing(input_dim=4, hidden_dim=2)` 首次接受 `[B,3]` 或 `[B,5]`，Dense 按错误宽度创建参数，而 PyTorch 相同调用失败。特征适配器维度配置错误因此会在两框架产生不同模型行为。

修复仅在 [TensorFlow `paper_ranking.py:11`](../../src/model_evo_harness/models/tensorflow/paper_ranking.py#L11) 添加 `InputSpec(ndim=2, axes={-1: input_dim})`。回归文件为 [`tests/test_deepcrossing_input_contract.py`](../../tests/test_deepcrossing_input_contract.py)：

1. 修复前，宽度 3/5 的两例均为 `DID NOT RAISE ValueError`，结果 **2 failed, 1 passed**。
2. 修复后，结果 **3 passed in 2.37s**。
3. 合法宽度在 batch 1/3 下输出 `[B]`，输入与全部参数梯度存在且有限；固定正参数保持 ReLU 激活，输入梯度严格为正。

本修复不涉及 PyTorch 或其他模型。相关模型数值 probe 再运行仍通过；全套测试交由主审查统一执行，本报告不声称已独立执行全套测试。

## 复现记录

独立固定 tensor 脚本：`/private/tmp/rec_overlap_probe_20261114.py`；完整结果：`/private/tmp/rec_overlap_probe_20261114.json`。这些临时文件不属于发布包。复现命令：

```bash
TF_CPP_MIN_LOG_LEVEL=3 /private/tmp/model-evo-review-venv/bin/python /private/tmp/rec_overlap_probe_20261114.py
TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=src /private/tmp/model-evo-review-venv/bin/python -m pytest -q tests/test_deepcrossing_input_contract.py
```

脚本只导入 Harness 的指定模块、构造固定数值、比较手算公式及梯度，并隔离重写来源 ETA 屏蔽表达式和 SDIM 求和布局；未导入或执行来源 BaseModel/trainer。双框架 SDIM、ETA、FFM、DeepCrossing 的解析值、空历史、NaN padding 与梯度检查完成。精确零投影及归一化差异保留为已解释的变体，不据此宣称来源完全等价。
