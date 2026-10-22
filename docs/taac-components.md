# TAAC2026 tensor components

The first implementation batch provides four independently usable mechanisms, with matching PyTorch and TensorFlow/Keras APIs. It accepts existing embedding tensors; the host owns their semantics, prediction head, loss and evaluation. This is not a complete TAAC model, a new data pipeline or a TensorFlow 1.15 implementation.

## Component and knowledge map

Each module is self-contained and imports only its framework and the standard library. Replace `pytorch` with `tensorflow` in the paths below for the matching Keras implementation.

| Method ID / class | Source module | Knowledge / structure pattern | Output |
| --- | --- | --- | --- |
| `rankmixer` / `RankMixerBlock` | [`paper_interactions.py`](../src/model_evo_harness/models/pytorch/paper_interactions.py) | `feature_interactions` / `token_subspace_mixing` | `[B,T,D]` |
| `unidot_conditioned_gate` / `ConditionalSwiGLU` | [`paper_unified_sequence.py`](../src/model_evo_harness/models/pytorch/paper_unified_sequence.py) | `sequence_ranking` / `state_conditioned_history` | `[B,L,D_out]` |
| `unidot_multichannel_pool` / `MultiChannelSequencePool` | [`paper_unified_sequence.py`](../src/model_evo_harness/models/pytorch/paper_unified_sequence.py) | `sequence_ranking` / `state_conditioned_history` | `[B,S*C,D]` |
| `unidot_fm_highway` / `ExplicitInteractionHighway` | [`paper_interactions.py`](../src/model_evo_harness/models/pytorch/paper_interactions.py) | `feature_interactions` / `cross_layer_interaction_readout` | `[B,K*S*I + K*U*I]` |

Use `model_api` for exact constructors/methods and `read_references` for whole modules with SHA-256 hashes and output contracts. The four method IDs fit in a single reference request. The method cards link to original sources; this batch does not add TAAC entries to the separate 115-PDF Reco-papers inventory.

```python
from model_evo_harness import load_catalog, model_api, read_references

catalog = load_catalog()
methods = ["rankmixer", "unidot_conditioned_gate",
           "unidot_multichannel_pool", "unidot_fm_highway"]
api = model_api(catalog, framework="tensorflow", method_ids=set(methods))
source = read_references(catalog, {"framework": "tensorflow", "method_ids": methods})
```

## Input and numerical contracts

Inputs are floating-point continuous representations, not necessarily IDs or semantic tokens. The initial reference and cross-framework checks target float32. No low-precision throughput or training-quality claim is implied.

**RankMixer.** Construct `RankMixerBlock(num_tokens, dim, hidden_dim)` and call it on `[B,T,D]`. T and D are fixed, D must be divisible by T, and all tokens must have defined values. It rewires channel subspaces, applies the residual then LayerNorm, and uses separate GELU FFN parameters for each token followed by another residual/LayerNorm. `mix_tokens` exposes only the permutation. There is no token padding mask, tokenizer, head or sparse MoE.

**Conditional gate.** Construct `ConditionalSwiGLU(sequence_dim, condition_dim, hidden_dim, output_dim)`. Call `gate(history, condition, valid_mask)` with `[B,L,D_s]`, `[B,D_c]` and bool `[B,L]`. The condition is broadcast internally and enters only the gate. True means valid; masked history is sanitized before projection and its output is zero. L can be zero. The caller provides a finite condition vector and valid history values.

**Pool.** Construct `MultiChannelSequencePool(dim, num_state_tokens, num_channels, num_domains)`. Call `pool(state_tokens, histories, valid_masks)`, where state tokens are `[B,T,D]` and each of the S domains supplies `[B,L_s,D]` plus bool `[B,L_s]`. Domain count/order and state token count are fixed; batch and history lengths may vary. Every domain owns a scorer and a projection; output order is domain, then channel. Independent sigmoid gates pool content, followed by L2 normalization, projection and a shared LayerNorm. An empty or entirely masked domain returns exact zeros even with learned affine biases. State tokens are fully defined and finite.

**Highway.** Construct `ExplicitInteractionHighway(num_layers, dim, num_user_tokens, num_item_tokens, num_domains=0)`. Call with post-fusion `user_states[B,K,U,D]`, `item_states[B,K,I,D]` and optional `domain_dots[B,K,S,I]`. The domain bundle is first, flattened in layer/domain/item order; the cross-bus bundle follows in layer/user/item order. Each bundle is LayerNormed separately. Every included bundle needs at least two scalar entries, since LayerNorm of a single scalar would erase the interaction signal. Field-only mode uses S=0 and does not require history.

For domain dots, `domain_interactions(pre_residual_queries[B,I,D], aligned_readouts[B,S,I,D], valid_domains[B,S])` returns `[B,S,I]`. The mask is bool with True=available. The method computes raw same-position dots, without attention or normalization; the caller stacks these across layers. Supply the queries from before the residual update, not the post-fusion item states. Final bundle normalization may recenter an absent domain's zero raw dots. All returned Highway features must be connected to the host prediction head to affect learning.

## Sources, adaptation and fidelity

- **RankMixer:** independent implementation of [v3 equations 1 and 3–8](https://arxiv.org/html/2507.15551v3). Dense block only; no claim about the paper's serving throughput or sparse-MoE system.
- **UniDot:** adapted from [author commit `1399702`](https://github.com/linrongc/kdd26_unidot/tree/1399702006a1fc2f86329c8c6133f3f3ac1e6d0d), specifically `common.py:CondGatedSwiGLU`, `unified_model.py:MultiChannelSeqPool`, `FMHighway` and the per-domain dot operations. Each extractable module retains the full MIT notice.
- The pinned pool code uses sigmoid despite an older softmax docstring. The port follows the executable operation. Our valid-mask direction is the inverse of upstream padding masks; zero empty-domain outputs deliberately override the upstream affine-bias response. Pool L2 normalization accumulates half types in float32 and defines a zero vector's normalized direction and gradient as zero, avoiding FP16 underflow and undefined zero-direction gradients.
- The pinned Highway omits the aggregated Gram described in the [paper](https://arxiv.org/html/2608.16797v1). This implementation follows that code subset, enables cross-bus dots, and omits fused-domain variants. It returns one concatenated feature vector instead of the author's pair of bundles.
- Temporal variants, external embeddings, optimizers, training schedules and author data imports are excluded. Candidate-conditioned history cannot be cached across different candidates without checking the dependency boundary.

## Controlled use

First compare one component with its removal on the same fields, labels, split, head and training budget. For RankMixer, separate token-count, width and per-token/shared-FFN comparisons. For pooling, compare masked mean pooling with state-conditioned sigmoid channels. For Highway, separately ablate cross-bus and domain evidence in the host experiment while preserving the declared output/head contract.

Record held-out quality, parameter count, batch, dtype, length distribution, latency and peak memory. Reject unsupported input semantics, leakage, unstable gradients, absent reproducible improvement or unacceptable cost. A method's `ready` status only establishes declared input availability. QueryFormer, LENS, CRAFT and the remaining candidates in the [research report](research/taac2026-integration-2026-10-21.md) are not implemented by this batch.

Executed CPU, numerical, source-extraction and package checks, together with the pending GPU status, are recorded in the [validation report](research/taac-components-validation-2026-10-22.md).
