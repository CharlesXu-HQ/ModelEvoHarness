# Feature-pair and multi-interest components

This release adds composable mechanisms checked against the [rec_models review](research/rec-models-assessment-2026-11-14.md). The host can retain its estimator, backbone, labels and evaluator while testing one component at a time. Native tensor code is provided in both frameworks; no DeepCTR, DeepMatch or FuxiCTR wrapper is required.

| Method / option | Local source | Intervention and controlled comparison |
| --- | --- | --- |
| `coaction` | `models/{pytorch,tensorflow}/coaction.py` | A feature-pair micro-MLP with per-sample parameters. Compare a simple bilinear/APG branch with one co-action pair; isolate order count, parameter tables and readout. |
| `comirec_sa` | `models/{pytorch,tensorflow}/multi_interest.py` | Candidate-independent multiple history pools. Compare masked mean, K=1 and K>1 at the same total retrieval budget. |
| `comirec_dr` | same | Position/interest-specific projections and capsule routing. Compare against SA and existing MIND with matched parameter and retrieval budgets. |
| `XDeepFM(cin_activation="identity")` | `models/{pytorch,tensorflow}/interactions.py` | Swap only the CIN activation, keeping embeddings, head and training protocol fixed. Default remains `"relu"`. |
| `SDIMCollisionPooling(normalize_per_hash=True)` | `models/{pytorch,tensorflow}/history_retrieval.py` | L2-normalize each collision bucket sum before averaging hashes. Default remains `False`; `per_hash` always reports raw sums. |
| `cold_feature_search` | `analysis/cost_aware_search.py` | Screen field subsets, invoke host physical pruning/retraining, then choose using measured validation quality and latency. Shared stdlib workflow, outside the model forward pass. |

## Co-action inside a backbone

`CoAction(input_dim, layer_dims, order=1, order_reduction="concat", use_bias=False)` consumes `parameters[B,P]`, `feed[B,T,D]`, and `valid[B,T]`, and returns `[B,T,C]`. The host learns a parameter lookup table or generator; these tensors are constructed from existing meaningful fields, not new raw business columns. Per layer the vector stores row-major `W[in,out]`, followed by an optional bias. All orders share those parameters; all layer readouts are kept. Non-final layers use tanh. `order_reduction="sum"` follows paper-style order aggregation; `"concat"` follows the inspected author-code variant. This naming does not claim a complete paper training system.

```python
from model_evo_harness.models.pytorch.coaction import CoAction

cross = CoAction(input_dim=8, layer_dims=[8, 4], order=2)
# Host-owned modules create theta [B,cross.parameter_dim] and feed [B,T,8].
pair_events = cross(theta, feed, valid)
pair_branch = pair_events.sum(dim=1)  # host chooses and records its reduction
# Concatenate pair_branch with the existing backbone representation before its head.
```

For several supported field pairs, instantiate parallel branches or share parameters explicitly using the existing composition contract. Do not enumerate all pairs by default. Sparse IDs, parameter-table growth and unsupported combinations need their own slices and budget checks.

## Multi-interest contracts

Both classes return `interests[B,K,D]`, `interest_mask[B,K]`, `assignments[B,K,T]`. `interest_mask` says that a history exists; it does not certify distinct or occupied clusters. Real histories and item identity are prerequisites. Anonymous aggregate columns do not establish them.

- `ComiRecSA(dim, num_interests, hidden_dim=None, max_length=None)`: assignments normalize over history positions. Optional learned positions affect scores only; the values remain original history vectors. No candidate or future label enters the user encoder.
- `ComiRecDR(dim, num_interests, max_length, routing_iterations=3, stop_gradient_routing=True)`: position/interest projections have shape `[max_length,K,D,D]`. Assignments normalize over interests at each valid event. Non-final routing votes are detached by default; final votes remain differentiable. Zero initial routing logits are an explicit deterministic variant of the author's random initialization.

True means valid. Masked NaN padding, empty histories and entirely masked histories produce finite zero representations. Hosts own embedding tables, objective, negatives, candidate readout and retrieval aggregation. ComiRec's category-diversity reranker is not bundled in these encoder classes. When used inside a ranking backbone, declare that composition separately from an end-to-end ComiRec retrieval experiment.

## Cost-constrained physical subset experiments

`CostAwareFeatureSearch(latency_budget_ms, quality_floor=... or max_quality_drop=...)` uses training-only saliency to propose subsets. The full-input baseline and each unique subset are rebuilt, retrained and evaluated by the supplied host callback. All members of protected interaction groups survive, even with zero individual saliency.

The host callback receives `retained_indices` and a fixed `context`, and returns actual validation quality, measured latency, exact retained indices, training/code artifact IDs and attestations of retraining and physical pruning. Context fixes task/dataset/splits, training budget, seed, metric and serving profile. Invalid or incomparable results fail; missing measurements are never synthesized. See the [complete callable schema](../src/model_evo_harness/analysis/cost_aware_search.py).

The result contains every trial, feasible Pareto candidates and the highest-quality feasible selection. No feasible candidate returns `selected=None`. A feature mask is not physical pruning; lower saliency is not proof that removal is harmless. Declared booleans are host attestations, so preserve artifacts for independent checks. Latency measured without feature lookup is labeled `model_only`. This workflow does not optimize monetary acquisition costs or QPS, and selection still requires independent final validation before a business gain is claimed.

## Agent integration and verification

The four new method cards expose prerequisites, exact source, contracts, comparisons and failure signals. `read_references` delivers complete native modules; framework-neutral analysis source is shared by both framework requests. Model components and the COLD analysis workflow remain in separate APIs and directories.

Validation covers independent numeric oracles, matched-framework output/gradient checks, protected subsets, real callback boundary enforcement and explicit GPU device tests. Tests of a callback protocol are not a real dataset training experiment. See [release validation](research/rec-components-release-validation-2026-11-14.md) for the final results.
