# QueryFormer query-generation components

Four independently implemented tensor components are available in PyTorch and TensorFlow/Keras. They consume existing continuous embeddings and expose representations for a host model. The combined `QueryFormerQueryBridge` covers equations 2–4 of [QueryFormer v1, section 4.2](https://arxiv.org/html/2609.16548v1#S4.SS2). The host owns token construction, downstream sequence processing, prediction head, loss and evaluation.

## Components and source retrieval

| Method ID / class | Knowledge / structure pattern | Returned values |
| --- | --- | --- |
| `queryformer_co_transformer` / `CoTransformerBlock` | `feature_interactions` / `queryformer_field_interaction` | updated user `[B,U,D]`, item `[B,I,D]` |
| `queryformer_history_retrieval` / `QuerySeqCrossAttn` | `sequence_ranking` / `queryformer_query_generation` | retrieved `[B,S,D]`, valid domains `[B,S]` |
| `queryformer_query_generator` / `SeqQueryCrossAttn` | `sequence_ranking` / `queryformer_query_generation` | queries `[B,S,2,D]`, query mask `[B,S,2]` |
| `queryformer_query_bridge` / `QueryFormerQueryBridge` | `sequence_ranking` / `queryformer_query_generation` | dictionary with queries, query mask, context tokens and context mask |

Each framework has one self-contained module: [PyTorch](../src/model_evo_harness/models/pytorch/paper_queryformer.py) and [TensorFlow](../src/model_evo_harness/models/tensorflow/paper_queryformer.py). Each imports only its framework and the standard library. All four method IDs fit in one source request, which returns a single complete module, its SHA-256 hash and output contracts:

```python
from model_evo_harness import load_catalog, model_api, read_references

methods = ["queryformer_co_transformer", "queryformer_history_retrieval",
           "queryformer_query_generator", "queryformer_query_bridge"]
catalog = load_catalog()
api = model_api(catalog, framework="pytorch", method_ids=set(methods))
source = read_references(catalog, {"framework": "pytorch", "method_ids": methods})
```

## Tensor contract

`D` is the common width and must be divisible by `num_heads`. `S` is the positive, construction-time `num_domains`; preserve domain ordering across calls. Batch size and user, item, context and history lengths may vary. Every input mask is boolean with **True meaning valid**, and has the input tensor's shape without its final width dimension. Valid values must be finite. Padded values, including NaNs, are sanitized before pooling or projection; output padding is zeroed after affine operations. Zero-length axes and all-masked rows are supported.

Construct and call the standalone components as follows. These positional call signatures are the same in both frameworks; TensorFlow implements them through `call`, PyTorch through `forward`.

- `CoTransformerBlock(dim, num_heads, num_layers=2)(user_tokens, item_tokens, user_mask, item_mask)` returns the two updated token sets. Each layer separately self-attends within user and item tokens, adds the residual and applies LayerNorm. User tokens then query item keys/values. For cross delta `a` and self-refined user `u`, this implementation uses `g = sigmoid(Dense(concat(u, a)))` and `LayerNorm(u + g * a)`. Item tokens receive only their self-attention stage. Masked user/item positions remain zero; an empty counterpart suppresses its cross-attention delta.
- `QuerySeqCrossAttn(dim, num_heads, num_domains)(item_tokens, histories, item_mask, history_masks)` takes a list of `S` history tensors `[B,L_s,D]` with matching masks. A masked item-token mean queries each domain through an independent standard multihead attention module. A domain returns zero and a false availability flag if its history or the item side is empty.
- `SeqQueryCrossAttn(dim, num_heads, num_domains)(context_tokens, histories, context_mask, history_masks)` directly initializes two probes per history in **mean, then max** order. Independent per-domain attention queries the supplied context `[B,T,D]` as keys/values. The output has no extra residual or LayerNorm. There is no extra probe projection before the normal attention Q projection. An empty history domain or empty context produces two zero probes and false masks.
- `QueryFormerQueryBridge(dim, num_heads, num_domains, num_layers=2)(user_tokens, item_tokens, histories, user_mask, item_mask, history_masks)` composes those stages. Its context order is updated user tokens, updated item tokens, then one retrieved token per domain. It returns `queries[B,S,2,D]`, `query_mask[B,S,2]`, `context_tokens[B,U+I+S,D]`, and `context_mask[B,U+I+S]` under those exact dictionary keys. A missing history disables that domain's probes; available field context can still be returned.

The field-only Co-Transformer is applicable with declared `tabular_features`, `user_tokens` and `item_tokens`. The other three methods additionally require the family's `event_sequence` capability plus the method's declared tokens, `domain_histories` and `sequence_mask`. Static field groups do not establish event-history availability.

## Minimal call

```python
import torch
from model_evo_harness.models.pytorch.paper_queryformer import QueryFormerQueryBridge

# Two samples, two user fields, one item field, width four, two history domains.
user = torch.randn(2, 2, 4)
item = torch.randn(2, 1, 4)
histories = [torch.randn(2, 3, 4), torch.empty(2, 0, 4)]
user_mask = torch.tensor([[True, True], [True, False]])
item_mask = torch.ones(2, 1, dtype=torch.bool)
history_masks = [torch.tensor([[True, True, False], [True, False, False]]),
                 torch.empty(2, 0, dtype=torch.bool)]

bridge = QueryFormerQueryBridge(dim=4, num_heads=2, num_domains=2)
result = bridge(user, item, histories, user_mask, item_mask, history_masks)
assert result["queries"].shape == (2, 2, 2, 4)
assert result["context_tokens"].shape == (2, 5, 4)
assert not result["query_mask"][:, 1].any()  # second domain has no history
```

For TensorFlow, import the same class from `model_evo_harness.models.tensorflow.paper_queryformer`, create the corresponding `tf.Tensor` inputs and boolean masks, and use the same argument order and result keys. Consume the returned masks in the host head or subsequent sequence block. These outputs are representations, not logits or calibrated probabilities.

## Fidelity and excluded scope

The paper specifies two Co-Transformer layers, self-attention before gated cross-attention, the user-query/item-key-value direction in Table 2, history retrieval, and two behavior-initialized probes attending to context. This port makes the remaining choices explicit: masked-mean item aggregation; direct mean/max probe initialization; separate per-domain attention; Co-Transformer residual post-LayerNorm and sigmoid concatenation gating; no extra residual or LayerNorm after query generation; affine biases; no dropout; no extra FFN; and zero outputs for unavailable inputs. `num_layers` permits a host ablation away from the paper's two-layer setting. These are inspectable engineering choices rather than a claim of exact author implementation.

This is a single-view query-generation path. Equation 5's downstream sequence encoding, query-to-sequence attention, mixing and readout are outside the component. It also excludes `H>1` tokenization views, packed shared-parameter attention, feature/tokenizer pipelines, temporal embeddings, auxiliary losses, training schedules and a prediction head. It does not reproduce the complete competition model, establish TensorFlow 1.15 compatibility, or imply the paper's speed or business results.

The implementation was written from the paper; the author repository and its code license have not been confirmed, and no third-party code was copied. The [introductory article](https://mp.weixin.qq.com/s/FNdUKN6dkCtIuufu4uQa3g) is discovery context only. Technical attribution is to the versioned paper above.

## Controlled use and diagnostics

Start from an observed bottleneck in how the host combines fields and histories. Keep features, time cutoff, split, head, loss and training budget fixed. Compare the complete path with its removal, then isolate self-attention, cross-side fusion, item-conditioned retrieval and context-aware query generation. Compare retrieval with a masked history mean; compare query generation with a projection-only control using the same pooled histories/context and a declared parameter budget. Separately test mean versus max probe contributions and the number of Co-Transformer layers.

Inspect output and gradient finiteness, invariance to padded-value changes, empty-domain behavior, query diversity, history-length slices, and whether the host actually consumes returned representations/masks. Record held-out quality, parameters, dtype, batch/length distribution, latency and peak memory. Reject leakage, mask dependence, collapsed probes without task benefit, irreproducible gains or unacceptable cost. Candidate-conditioned retrieval and context-aware probes require recomputation when the candidate/context changes.

| Existing component | Distinguishing comparison |
| --- | --- |
| DIN | Candidate attention pools history directly. Test whether constructing context-aware probes helps the host beyond that simpler target-conditioned summary. |
| UniDot gate / pool | Conditions history gates or sigmoid pooling channels on current state. Compare with QueryFormer's attention retrieval and context-attending probes on identical histories. |
| RankMixer | Mixes field-token subspaces with per-token FFNs. It is a field interaction control and can be used without histories. |
| Packed multi-view attention | Unavailable here. Running Python loops or parallel copies of this module does not establish the paper's packed execution or latency scaling. |

Catalog `ready` status means declared input availability. Tensor tests establish computation behavior; task benefit requires the host's controlled experiment.

Executed tensor, source-extraction and package checks, including numerical and masking coverage and the limits of CPU-only validation, are recorded in the [validation report](research/queryformer-validation-2026-10-22.md).
