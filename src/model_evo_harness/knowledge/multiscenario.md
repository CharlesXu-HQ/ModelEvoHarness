# Scenario and context specialization

A pooled model assumes the same parameters can serve every context. STAR uses scenario-conditioned components; HMoE routes contexts among experts; PEPNet and APG adapt representations or parameters. Such designs are useful when residuals differ across stable, pre-decision scenarios and each scenario has enough support. They can also overfit small segments and hide weak global calibration.

## Data and conditions

Define scenario keys from fields known before action, such as placement, market, or lifecycle stage. Record segment size, label rate, treatment coverage where relevant, and serving-time availability. A segment built from future outcomes is not a valid context feature. Costs and latency grow with specialization.

## Controlled experiment

Compare the pooled baseline, explicit scenario indicator, separate small heads, and one conditional/gated model. Use the same rows, fields, split and declared model-size budget. Report primary value and calibration per scenario, especially low-support segments; test whether gains persist across time.

## Interpretation and feature needs

If the explicit scenario indicator closes the gap, a complex expert network is unnecessary. If scenario effects are known by business rules but the scenario identifier is absent, request that identifier immediately. If all available scenario variants fail, first review support, label drift and split design before requesting new segmentation fields.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| STAR | Shares a main network while allowing scenario-specific transforms and normalization. | Compare with pooled model plus scenario ID and small local heads. |
| HMoE | Gates shared experts into domain predictions. | Check whether expert specialization persists across time. |
| PEPNet | Gates input embeddings and intermediate layers from context. | Ablate input and layer gates separately. |
| APG | Generates compact context-conditioned layer parameters. | Compare quality and serving cost with simpler scenario indicators. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/5.multi_scenario/1.multi_tower.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/5.multi_scenario/1.multi_tower.html), [docs/chapter_2_ranking/5.multi_scenario/2.dynamic_weight.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/5.multi_scenario/2.dynamic_weight.html), [src/funrec/models/star.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/star.py), [src/funrec/models/apg.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/apg.py).

## FuxiCTR mechanism components

These are independently implemented, composable PyTorch and TensorFlow mechanisms. Read the full local module and output contract before editing the current candidate. A paper name identifies a source of ideas; omitted towers, input adapters, samplers and objectives remain the host’s responsibility. Preserve the current compatible backbone and test one mechanism with its matched control. Multiple meaningful feature groups may use independent or shared instances; justify sharing and fusion from the actual fields. Do not force a complete named-model replacement or exhaust a fixed model list. Record mechanism evidence separately from task/dataset-bound business interpretations.

### `adaptive_linear` — `AdaptiveLinear`

Generate a per-example full linear matrix or low-rank center S(condition) between shared input/output factors, with optional generated bias and overparameterized shared factors.

**When:** Current context/priors are available before decision and supported slices indicate parameter heterogeneity. Compare full/shared/low-rank or detached gate paths separately.

**Input contract:** dense_features, context_features. See `model_api` for the framework constructor and calls. **Output:** `[B,output_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Extension of installed APG core as a reusable hidden-state layer; affine generator, no multilayer APG tower or final predictor. Generated bias follows flattened generated matrix in this independent parameter layout. Rank cannot exceed input/output dimensions; overparameterization requires rank. Host explicitly supplies condition (including layer(x,x) for self-conditioning), scenario semantics and training.

**Bundled code:** `models/pytorch/ctr_conditioning.py`, `models/tensorflow/ctr_conditioning.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/APG/src/APG.py).

### `prior_conditioned_gate` — `PriorConditionedGate`

Scale a hidden layer by twice sigmoid of a ReLU MLP over base features and prior features; optionally detach only the base-feature gate path.

**When:** Current context/priors are available before decision and supported slices indicate parameter heterogeneity. Compare full/shared/low-rank or detached gate paths separately.

**Input contract:** hidden_representation, dense_features, context_features. See `model_api` for the framework constructor and calls. **Output:** `[B,hidden_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Extends existing PEPNet gating with explicit prior inputs, per-depth reuse and a declared stop-gradient boundary. detach_features defaults True; hidden and prior paths remain differentiable. Host creates one gate per chosen depth. No prior embedding tables, EPNet, task tower or automatic stacking; upstream PEPNet.py itself is only a stub.

**Bundled code:** `models/pytorch/ctr_conditioning.py`, `models/tensorflow/ctr_conditioning.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/PEPNet/src/PPNet.py).
