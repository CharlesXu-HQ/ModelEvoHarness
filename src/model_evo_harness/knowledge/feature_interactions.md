# Feature representation and interactions

A linear head adds independent feature effects. FM factorizes second-order pairs; FwFM changes strength by field pair; DeepFM combines pair terms with an MLP; DCN and DCNv2 build explicit higher-order crosses; xDeepFM, attention, and product networks vary how interactions are parameterized. More parameters are not, by themselves, evidence that interactions matter.

A factorization machine uses `z(x) = b + Σ_i w_i x_i + Σ_{i<j} <v_i,v_j> x_i x_j`; the shared embedding vectors permit pair estimates where an explicit one-hot cross would be sparse. A vector DCN layer can be viewed as `x_(l+1) = x_l + x_0 (w_l^T x_l) + b_l`, making the cross degree grow by layer. Matrix-based DCNv2 changes the transform inside that explicit cross; whether it helps is an empirical question under a capacity budget.

## Data and conditions

Declare typed categorical, numeric, multi-hot, and sequence fields; cardinality, missingness, normalization, and the time at which each value becomes known. Audit post-outcome leakage. A cross has a support problem: user-segment × price-band may be plausible but too rare to estimate. Embedding dimensions, sparse-feature handling, and inference cost belong in the comparison.

## Controlled experiment

Use a linear model and plain MLP as controls, then add one interaction mechanism while keeping fields, split, preprocessing, and a declared parameter/latency budget fixed. Ablate the cross branch and compare supported field-pair residuals, calibration, primary decision value, and sparse slices. For explicit crosses, inspect support and sign stability across folds.

## Interpretation and feature needs

If multiple distinct interaction designs fail on the same supported slice, investigate target noise, missing temporal context, or absent semantics before requesting another cross. Ask humans to add a specific aggregate or cross field after controlled attempts reveal an input gap. A business expert can instead name an inherently essential missing pre-decision field, such as recent spending band for a coupon value decision, without waiting for model ablations.

## Choosing the next explicit interaction

Separate raw fields, their representations, and the relations a branch can express. Categorical embeddings preserve field identity; numeric fields can become `x_i * v_i`, train-fitted bins, or remain scalar inputs to a cross network. A flattened embedding coordinate is not an independent raw field. Missing flags are derived from the numeric source and may overlap with imputation effects. A sequence summary is only available if the host supplies that history and an eligible temporal cutoff.

Audit coarse view-pair coverage in the current backbone: implicit MLP, explicit product/cross branch, mixed, none or unknown. Record the code basis. Coverage is a declared structural assessment, not measured feature importance. A DCN over concatenated dense and embedded inputs may already explicitly cover mixed relations even if an earlier FM used categorical fields only.

Rank the next experiment using observed metrics, implementation diagnostics, field support, compute cost and dataset-bound history. Numeric-category crosses are a reasonable hypothesis when those representations are available, but neither numeric fields nor FM have unconditional priority. Check scale and fusion before increasing width/order. Avoid inferring that one failed categorical-FM recipe rejects the entire interaction family. Conversely, an untested pair does not automatically justify an experiment. Defer interactions when loss, leakage checks or other diagnosis have higher expected information value.

Use one discriminating control: same source, fixed protocol, and a meaningful model-config gate removing the selected term. Keep module construction stable when possible. Report whether the control actually completed and the paired candidate-minus-control result. Shared retraining, output scaling and adaptive validation selection still limit attribution. Do not write planned ablations as observed success.

Local source is available through `read_reference` with `include_interactions: true`: `models/pytorch/explicit.py` and `models/tensorflow/explicit.py` contain independent `NumericFieldEmbedding` and `GroupedFM` primitives. Selected within/between disjoint-group terms are returned separately so the Agent can implement parallel scopes, scaling, gates or learned fusion inside its backbone. These are optional building blocks, not a closed strategy list. Full-pair tensors and exhaustive original-field pair testing are unnecessary.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| Wide & Deep | Adds explicit wide crosses to a learned dense path. | Remove wide crosses while keeping the deep branch and fields fixed. |
| FM | Factorizes pairwise feature effects. | Compare with linear and equal-width MLP controls. |
| FwFM | Adds a learned strength per field pair. | Check pair support and compare with FM using the same embeddings. |
| DeepFM | Adds an FM logit to a deep interaction path. | Ablate each branch, rather than comparing only total capacity. |
| NFM | Pools pairwise embedding products before an MLP. | Compare with FM at matched embeddings and budget. |
| AFM | Attention weights the contribution of each feature pair. | Check attention stability and compare with unweighted pair pooling. |
| PNN | Sends inner or outer embedding products to a DNN. | Ablate product terms while retaining the same DNN input size. |
| FiBiNET | Reweights fields and applies bilinear pair interactions. | Ablate field gating and bilinear terms separately. |
| DCN | Repeatedly crosses original inputs with intermediate states. | Compare cross branch with its ablation and a size-matched MLP. |
| DCNv2 (`dcn_v2`) | Uses matrix/low-rank cross transforms for richer explicit crosses. | Compare with vector DCN under parameter and latency limits. |
| xDeepFM | Creates explicit vector-wise higher-order interactions. | Ablate the interaction network with the same deep branch. |
| AutoInt | Uses self-attention to let fields condition on each other. | Compare with a supported explicit-cross or FM baseline. |

## Token mixing and cross-layer readout

Use `rankmixer` when the host provides stable field tokens and the bottleneck concerns communication between their subspaces. Its dense `RankMixerBlock` requires width divisible by token count; compare the same tokens and head against a simpler mixer, then isolate independent versus shared FFNs. Choose token grouping using available field semantics and evidence, not the assumption that every token is a semantic ID. Width and token count are separate experimental axes.

Use `unidot_fm_highway` when useful user/candidate interactions may disappear through depth. Supply aligned user/item states from every declared layer; connect the resulting explicit interaction vector into the host head. Its field-only mode requires no event sequence. If adding domain dots, separately supply actual pre-residual queries and aligned retrieval outputs; post-fusion item states are not interchangeable with those queries. Compare the added evidence with its ablation at the same head and training budget. The pinned author-code subset omits aggregated Gram, and single-scalar normalized bundles are rejected because their signal would vanish.

Both methods are available through the normal method-ID source retrieval. Inspect their API contracts and limitations before composition. Source lineage: [RankMixer equations 1, 3–8](https://arxiv.org/html/2507.15551v3), [UniDot FMHighway](https://github.com/linrongc/kdd26_unidot/blob/1399702006a1fc2f86329c8c6133f3f3ac1e6d0d/unified_model.py#L2455). These are portable tensor mechanisms, not evidence of industrial throughput or an improved task score.

## QueryFormer field interaction

Use `queryformer_co_transformer` when separately encoded user and item fields need a candidate-aware interaction before the host head. This field-only method needs `tabular_features`, `user_tokens` and `item_tokens`; it does not require histories. Each layer refines each side with self-attention, then only user queries attend to item keys/values with gated residual fusion. The item output receives its self-attention stage. Compare against the same fields/head without cross-side fusion, then isolate within-side refinement and compare with a cost-matched RankMixer control.

Both framework modules expose `CoTransformerBlock(dim, num_heads, num_layers=2)` over user/item embeddings and True=valid masks. Padded inputs, including NaNs, are sanitized; padded outputs remain zero. Read the extracted module for residual/LayerNorm and sigmoid-concatenation gate choices. The two-layer setting and attention direction follow [QueryFormer section 4.2](https://arxiv.org/html/2609.16548v1#S4.SS2); normalization, biases and empty-side behavior are explicit engineering choices. This independent paper implementation adds no tokenizer or task head and establishes no task gain. For history retrieval and generated probes, use the related methods in the [sequence-ranking guide](sequence_ranking.md).

## Progressive interaction schedule

`progressive_interaction` keeps a field/group token axis and ranks directed edges
once from scaled initial raw dot products. Each successive learned attention/FFN
layer opens a nondecreasing edge budget using that original ranking. Try it when
observed residuals or early noisy mixing justify controlling when interactions
occur. Compare full, fixed-sparse and progressive schedules with the same fields,
parameters, head and training budget; inspect task metrics and cost. Self edges
count in the budget; empty rows get zero context. This is an original local
[MGDIN-inspired mechanism](https://arxiv.org/html/2603.12586v1), without learned
initial Q0/K0, multigranularity branches or semantic-ID preprocessing. Read the
exact native source and use observed evidence; readiness is not expected gain.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/1.wide_and_deep.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/1.wide_and_deep.html), [docs/chapter_2_ranking/2.feature_crossing/1.second_order.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/2.feature_crossing/1.second_order.html), [src/funrec/models/wide_deep.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/wide_deep.py), [src/funrec/models/fm.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/fm.py).

Additional pinned source for `dcn_v2`: [torch_rechub/models/ranking/dcn_v2.py](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/ranking/dcn_v2.py).

Additional pinned source for `fwfm`: [deepctr/models/fwfm.py](https://github.com/shenweichen/DeepCTR/blob/1b5fe40e158d1ee6af8b1d9df217a5ed5aea9136/deepctr/models/fwfm.py).

## Reco-papers component map

These entries are independent computational components, not full training systems. Read each exact signature and limitation through `model_api` or `read_references`; use the same input tensors and evaluator when comparing them.

- `deepcrossing` / `DeepCrossing`: Stacked dense feature vectors processed by dimension-preserving residual MLP units. Output: `binary_logit`, `[B]`. Embedding/field encoding supplied by caller; compact residual units rather than the original distributed advertising training system.
- `fnn` / `FNN`: Initialize field embeddings from a supplied pretrained FM, then train an MLP over the concatenated fields. Output: `binary_logit`, `[B]`. FM pretraining and feature-ID alignment are host responsibilities; randomly initialized MLP is not a faithful FNN baseline.
- `predictive_bilinear` / `PredictiveBilinear`: Bilinear user-content interaction x_user^T W x_item for changing item features. Output: `bilinear_score`, `[B]`. Only the predictor; host must build as-of dynamic features and fit the intended objective.

The machine-readable paper inventory links these methods to their PDF, family, structural pattern and training objective. Explicitly declared capabilities only establish input availability; they do not establish that an experiment will improve the target task.

## FuxiCTR mechanism components

These are independently implemented, composable PyTorch and TensorFlow mechanisms. Read the full local module and output contract before editing the current candidate. A paper name identifies a source of ideas; omitted towers, input adapters, samplers and objectives remain the host’s responsibility. Preserve the current compatible backbone and test one mechanism with its matched control. Multiple meaningful feature groups may use independent or shared instances; justify sharing and fusion from the actual fields. Do not force a complete named-model replacement or exhaust a fixed model list. Record mechanism evidence separately from task/dataset-bound business interpretations.

### `gated_cross` — `GatedCross`

At each layer, add x0*(W*x+b)*sigmoid(G*x) to the current hidden state, retaining the initial feature vector.

**When:** Context or field-slice residuals suggest selective multiplicative processing. First compare with the existing local cross/MLP control.

**Input contract:** dense_features. See `model_api` for the framework constructor and calls. **Output:** `[B,input_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Coordinate-gated matrix cross block only; no learned FDO embedding dimensions, deep tower or classifier. All dense inputs must be defined; host owns feature preparation and training.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/GDCN/src/GDCN.py).

### `instance_mask` — `MaskBlock`

Generate an unbounded nonlinear coordinate mask from original features, multiply current hidden state, then project, optionally normalize and ReLU.

**When:** Context or field-slice residuals suggest selective multiplicative processing. First compare with the existing local cross/MLP control.

**Input contract:** dense_features, hidden_representation. See `model_api` for the framework constructor and calls. **Output:** `[B,output_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One reusable MaskBlock; serial composition reuses original features, parallel composition uses explicitly declared independent/shared instances and host fusion. No automatic feature embedding layer, batch norm, dropout or classifier.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/MaskNet/src/MaskNet.py).

### `context_feature_gate` — `ContextFeatureGate`

Scale input features by twice sigmoid of a context-conditioned two-layer ReLU gate, independently instantiated per stream.

**When:** Two existing streams show complementary supported residuals. Test mid-layer exchange or fusion while preserving both input branches.

**Input contract:** dense_features, context_features. See `model_api` for the framework constructor and calls. **Output:** `[B,feature_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Host supplies the context; constant contexts produce global gates. Uses existing gating primitive as an explicit stream interface; no pair of MLP towers is created.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalMLP/src/FinalMLP.py).

### `multihead_bilinear_fusion` — `MultiHeadBilinearFusion`

Fuse two representations with affine contributions and separate bilinear parameter tensors for matching contiguous head partitions.

**When:** Two existing streams show complementary supported residuals. Test mid-layer exchange or fusion while preserving both input branches.

**Input contract:** branch_representations. See `model_api` for the framework constructor and calls. **Output:** `[B,output_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Both stream widths must divide by heads; cross-head bilinear terms are excluded. Extends existing single bilinear scoring with a fusion contract; no softmax attention, sigmoid output or branch construction.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalMLP/src/FinalMLP.py).

### `cross_deep_bridge` — `CrossDeepBridge`

Communicate cross/deep intermediate states through product, sum, projected concatenation or two independently parameterized attention gates.

**When:** Two existing streams show complementary supported residuals. Test mid-layer exchange or fusion while preserving both input branches.

**Input contract:** branch_representations. See `model_api` for the framework constructor and calls. **Output:** `[B,dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Single bridge at a host-selected depth; no entire EDCN network. Attention deliberately uses distinct gates for the two streams, correcting pinned upstream reuse of attention1. Gate MLP final biases are enabled.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EDCN/src/EDCN.py).

### `field_regulation` — `FieldRegulation`

Apply two learned temperature-softmax field weight vectors, each shared across embedding coordinates, to form cross/deep inputs.

**When:** Context or field-slice residuals suggest selective multiplicative processing. First compare with the existing local cross/MLP control.

**Input contract:** field_tokens. See `model_api` for the framework constructor and calls. **Output:** `tuple([B,num_fields,embedding_dim], [B,num_fields,embedding_dim])`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Global field regulation rather than sample-adaptive routing; preserves field axis. No batch normalization; host pairs this block with independently evaluated bridge/cross/deep updates.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EDCN/src/EDCN.py).

### `wukong_block` — `WuKongBlock`

Normalize low-rank Gram features X*(X^T*P), transform them into tokens with a ReLU MLP, concatenate linear field compression and add a projected residual.

**When:** Context or field-slice residuals suggest selective multiplicative processing. First compare with the existing local cross/MLP control.

**Input contract:** field_tokens. See `model_api` for the framework constructor and calls. **Output:** `[B,interaction_tokens+linear_tokens,embedding_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** A single FMB+LCB block with a two-layer FMB MLP; no tokenizer, model stack or head. Compressed Gram avoids forming the full field-pair matrix; no measured speed or scaling-law claim. Fully defined field tokens required; optional final normalization, Gram normalization always enabled.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/WuKong/src/WuKong.py).

### `field_matrix_interaction` — `FieldMatrixInteraction`

Compute lexicographic i<j field-pair scores using a distinct matrix e_i^T*M_ij*e_j or vector-weighted product for each pair.

**When:** Known field identities and supported pair/group errors justify distinct partner parameters. Compare shared-bank and pair-specific encodings under a memory ceiling.

**Input contract:** field_tokens. See `model_api` for the framework constructor and calls. **Output:** `[B,num_fields*(num_fields-1)/2]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Uses host embeddings and returns pair scores without first-order terms or a classifier. Adds explicit field-pair parameter sharing policy to existing bilinear primitives; matrix storage grows quadratically in field count and embedding dimension.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FmFM/src/FmFM.py).

### `grouped_field_interaction` — `GroupedFieldInteraction`

Combine weighted within-group FM vectors with weighted products between pooled semantic groups.

**When:** Known field identities and supported pair/group errors justify distinct partner parameters. Compare shared-bank and pair-specific encodings under a memory ceiling.

**Input contract:** field_tokens, feature_groups. See `model_api` for the framework constructor and calls. **Output:** `[B,embedding_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Group IDs per field must be contiguous nonnegative integers and cover at least two nonempty groups; singleton within-group terms are zero. Generalizes the pinned three user/item/context groups; host supplies meaningful groups, linear terms, deep path and classifier.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FLEN/src/FLEN.py).

### `factorized_interaction` — `FactorizedInteraction`

Split an affine projection into a,b and emit a+a*b or concat(a,a*b), preserving factorized multiplicative interactions.

**When:** Lower-order controls underfit an interaction hypothesis on current fields. Increase order or factorization locally before enlarging the entire backbone.

**Input contract:** dense_features. See `model_api` for the framework constructor and calls. **Output:** `[B,output_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Sum form is already expressible with LatentCross plus a projection; this is a convenience component and recipe boundary. Concat needs even output width; no FinalNet tower ensemble or detached consistency loss.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalNet/src/FinalNet.py).

### `quadratic_interaction` — `QuadraticInteraction`

Split field coordinates into heads, multiply them by multiple ReLU-projected rows, sum rows, restore field-coordinate order and add the residual.

**When:** Lower-order controls underfit an interaction hypothesis on current fields. Increase order or factorization locally before enlarging the entire backbone.

**Input contract:** dense_features, field_layout. See `model_api` for the framework constructor and calls. **Output:** `[B,input_dim]`.

**Controlled test:** Keep host embeddings, head, objective, data split and budget fixed; compare this local block with its ablation and a parameter/latency-matched simpler control. Report supported slices, gradients and branch redundancy. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Input width must divide into fields and each embedding width must divide into heads. Restores original coordinate order before residual, deliberately correcting pinned source direct flattening of head-major products. Projected row storage uses row/field/head/coordinate order. No batch norm, dropout or self-ensemble objective; compare head/row parameterization against existing multiplicative gates.

**Bundled code:** `models/pytorch/ctr_composition.py`, `models/tensorflow/ctr_composition.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/QNN/src/QNN_alpha.py).

### `ffm_pairs` — `FieldAwareFactorization`

Dot each field i embedded for partner j with field j embedded for partner i, keeping the i<j pair axis.

**When:** Known field identities and supported pair/group errors justify distinct partner parameters. Compare shared-bank and pair-specific encodings under a memory ceiling.

**Input contract:** field_embeddings, partner_specific_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,F*(F-1)/2]`.

**Controlled test:** Compare genuine partner-specific embeddings with a repeated shared embedding bank at fixed head, reporting parameter and memory changes. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Host creates partner-specific lookup/projection banks; this parameter-free core is not an FFM trainer or automatic ID tokenizer. Diagonal entries are unused by the pair output. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FFM/src/FFM.py).

### `onn_operations` — `OperationAwareInteraction`

Concatenate diagonal copy-operation embeddings with off-diagonal field-aware pair dot values.

**When:** Known field identities and supported pair/group errors justify distinct partner parameters. Compare shared-bank and pair-specific encodings under a memory ceiling.

**Input contract:** field_embeddings, partner_specific_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,F*D+F*(F-1)/2]`.

**Controlled test:** Compare operation-specific copy/pair banks with shared embeddings while fixing downstream DNN width and reporting parameter/storage cost. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** ONNv2 tensor convention is explicit: diagonal is the copy operation, off-diagonal is the partner operation. No lookup bank, DNN or prediction head is added. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/ONN/ONN_torch/src/ONNv2.py).

### `higher_order_fm` — `HigherOrderFactorization`

Dynamic programming computes exact elementary symmetric products over distinct fields for one selected order.

**When:** Lower-order controls underfit an interaction hypothesis on current fields. Increase order or factorization locally before enlarging the entire backbone.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,D]`.

**Controlled test:** Compare order 2 with a prespecified higher order using identical embeddings/head; report any independent per-order embedding banks separately. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One tensor bank and one order only; host combines independent orders/banks and linear terms. Products can overflow for extreme activations and high order. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/HOFM/src/HOFM.py).

### `deep_interaction_machine` — `InteractionMachine`

Power sums and Newton identities produce a concatenated readout of all elementary symmetric interaction orders 1..K.

**When:** Lower-order controls underfit an interaction hypothesis on current fields. Increase order or factorization locally before enlarging the entire backbone.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,K*D]`.

**Controlled test:** Compare all-order readout with matched selected-order HOFM or the order-2 control; keep the host head/deep path fixed. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Same polynomials as HOFM, with an all-order composition and power-sum implementation; not a new generic MLP. K<=min(5,F). No BatchNorm, head or parallel deep branch; cancellation is possible at large scales. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepIM/src/DeepIM.py).

### `holographic_interaction` — `HolographicInteraction`

Real FFT pair multiplication followed by inverse FFT yields circular convolution, or correlation by conjugating the left spectrum.

**When:** An explicit product or geometric hypothesis remains unexplained by simple pairwise controls. Compare the new algebra with the same representation width and numerical scale.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,F*(F-1)/2,D]`.

**Controlled test:** Compare circular convolution/correlation against Hadamard pair products with the same embedding dimension and host aggregation. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Circular embedding-coordinate interactions, not chronological convolution. Explicit inverse length supports odd D; half inputs promote. No source classifier or HFM projection head. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/HFM/src/HFM.py).

### `dlrm_composition` — `DenseSparseInteraction`

Append the projected dense representation as a field, compute all sparse/dense pair dots, then append the dense vector itself.

**When:** Use an explicit simple control or combine actual dense and sparse representations without inventing token semantics.

**Input contract:** field_embeddings, dense_representation. See `model_api` for the framework constructor and calls. **Output:** `[B,(F+1)*F/2+D]`.

**Controlled test:** Compare dot composition with dense/sparse concatenation under identical bottom representation and top-head budget. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Composition of existing dot products; host supplies dense bottom MLP, sparse embeddings and top prediction head. Dense token is always observed; only sparse fields have a mask. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DLRM/src/DLRM.py).

### `lorentz_interaction` — `LorentzInteraction`

Lift each embedding to the unit hyperboloid time component and return LorentzFM triangle interaction values for field pairs.

**When:** An explicit product or geometric hypothesis remains unexplained by simple pairwise controls. Compare the new algebra with the same representation width and numerical scale.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,F*(F-1)/2]`.

**Controlled test:** Compare Lorentz triangle terms with Euclidean dot pairs using the same embeddings/head and controlled embedding norms. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Curvature fixed at one, no manifold optimizer or learned geometric projection. Float64 intermediates protect float32 squares, but extreme float64 values can overflow and performance is unmeasured. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_products.py`, `models/tensorflow/field_products.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LorentzFM/src/LorentzFM.py).

### `logarithmic_interaction` — `LogarithmicInteraction`

Learn field-axis interaction orders in log(abs(x)) space and exponentiate their bounded linear combinations.

**When:** An explicit product or geometric hypothesis remains unexplained by simple pairwise controls. Compare the new algebra with the same representation width and numerical scale.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,O,D]`.

**Controlled test:** Compare learned orders with fixed pair products or a linear field mixer at matched hidden width; inspect clipping frequency. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Sign is discarded; epsilon floors magnitudes and exponent clipping saturates gradients. Missing fields are multiplicative-neutral, all-missing output zero. Omits AFN log/exp BatchNorm and ensemble DNN. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_algebra.py`, `models/tensorflow/field_algebra.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AFN/src/AFN.py).

### `euler_interaction` — `EulerInteraction`

Apply shared learned field orders to log magnitudes and phases, reconstruct real/imaginary outputs and add a shared implicit ReLU projection.

**When:** An explicit product or geometric hypothesis remains unexplained by simple pairwise controls. Compare the new algebra with the same representation width and numerical scale.

**Input contract:** complex_field_representations. See `model_api` for the framework constructor and calls. **Output:** `(real [B,O,D], imaginary [B,O,D])`.

**Controlled test:** Compare explicit polar products alone, implicit projection alone and their sum with matched representations and host head. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Host supplies real/imaginary tensors; no initial semantic embedding conversion, dropout, normalization or head. Zero phase is explicitly defined with zero gradient near zero, magnitudes floored, exponents clipped. Missing fields are multiplicative-neutral. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_algebra.py`, `models/tensorflow/field_algebra.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EulerNet/src/EulerNet.py).

### `generalized_interaction` — `GeneralizedInteraction`

Fuse full initial-field/current-subspace outer products with learned alpha, multiply by output-specific channel matrices, and contract a learned channel vector.

**When:** An explicit product or geometric hypothesis remains unexplained by simple pairwise controls. Compare the new algebra with the same representation width and numerical scale.

**Input contract:** field_embeddings, current_subspace_tokens. See `model_api` for the framework constructor and calls. **Output:** `[B,O,D]`.

**Controlled test:** Compare learned subspace fusion and full channel matrices against fixed fusion and diagonal matrices, holding token counts/head fixed. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One generalized interaction layer only. Initial and current token axes remain explicit; no parallel DNN or complete AOANet stack. Full O*D*D matrices can be costly. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_algebra.py`, `models/tensorflow/field_algebra.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AOANet/src/AOANet.py).

### `hierarchical_field_aggregation` — `HierarchicalInteractionAggregation`

Pool each interaction order, update fields with the pooled vector multiplied by the fixed initial contextual tokens, then attention-pool across orders.

**When:** Field tokens are available and context-dependent connectivity or order aggregation is a testable bottleneck. Measure field-order and attention-path ablations.

**Input contract:** contextualized_field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,D]`.

**Controlled test:** Compare one order with recurrent orders and uniform versus learned pooling using the same contextualized input and prediction head. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Consumes host contextual X1; omits upstream initial self-attention/FFN and classifier. Learned local aggregators use hidden width D. No claim that higher orders improve held-out outcomes. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_attention.py`, `models/tensorflow/field_attention.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/InterHAt/src/InterHAt.py).

### `field_graph_interaction` — `FieldGraphInteraction`

Build a learned directed field graph without self edges, send through per-field output matrices and receive through per-field input matrices, with residual.

**When:** Field tokens are available and context-dependent connectivity or order aggregation is a testable bottleneck. Measure field-order and attention-path ablations.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,F,D]; optional tuple with graph [B,F,F]`.

**Controlled test:** Compare learned graph weights with uniform valid-neighbor weights and shared versus field-specific matrices at fixed propagation depth. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One non-GRU propagation step, no extra initial residual, multi-step reuse or task head; bias omitted from messages. A graph over fields is not a user-item graph. Isolated valid fields retain their residual. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_attention.py`, `models/tensorflow/field_attention.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FiGNN/src/FiGNN.py).

### `disentangled_field_attention` — `DisentangledFieldAttention`

Center Q/K over valid fields per head and add separately normalized pair attention and unary key attention.

**When:** Field tokens are available and context-dependent connectivity or order aggregation is a testable bottleneck. Measure field-order and attention-path ablations.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `[B,F,D]; optional tuple with attention [B,H,F,F]`.

**Controlled test:** Compare pair-only, unary-only and their sum at fixed head count and projection width. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Correct [B,H,F,d] head transpose replaces source direct view; both distributions sum to one so combined row mass is two. No residual, dropout, optional pre-attention ReLU or classifier. Valid centering respects padding. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_attention.py`, `models/tensorflow/field_attention.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DESTINE/src/DESTINE.py).

### `sam_interaction` — `SelfAttentionInteraction`

Multiply raw pair dots by learned pair vectors (A) or elementwise pair vectors (E); order-3 variants transform keys, aggregate neighbors and optionally add a projection residual.

**When:** Field tokens are available and context-dependent connectivity or order aggregation is a testable bottleneck. Measure field-order and attention-path ablations.

**Input contract:** field_embeddings. See `model_api` for the framework constructor and calls. **Output:** `2A/2E: [B,F,F,D]; 3A/3E: [B,F,D]`.

**Controlled test:** Compare 2A/2E and 3A/3E at declared pair/vector storage cost, separating residual and output aggregation choices. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** All four local variants supported; no softmax, dropout or final weighted/sum/concat aggregation. Self pairs are included, unlike strict upper-triangle FM pair outputs. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_attention.py`, `models/tensorflow/field_attention.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/SAM/src/SAM.py).

### `ordered_field_convolution` — `OrderedKMaxConvolution`

Wide convolution along the ordered field axis, independently select largest k positions per channel/coordinate, restore their source order, then tanh.

**When:** A defensible stable field order exists; compare order permutations, generated-field removal and a non-convolutional matched-capacity control.

**Input contract:** field_embeddings, ordered_field_layout. See `model_api` for the framework constructor and calls. **Output:** `[B,k*D*C] flattened position-coordinate-channel`.

**Controlled test:** Compare ordered convolution/k-max with no convolution or ordinary max pooling, including a field-order shuffle control. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One stage with explicit k, no source dynamic multi-stage schedule or task head. Host must justify field order. Missing inputs are zero-filled; empty receptive fields cannot win and unavailable top-k slots become zero. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_convolution.py`, `models/tensorflow/field_convolution.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/CCPM/src/CCPM.py).

### `convolutional_field_generation` — `ConvolutionalFieldGenerator`

Length-preserving field convolution, tanh and max pooling feed a learned flattened recombination that creates new field tokens.

**When:** A defensible stable field order exists; compare order permutations, generated-field removal and a non-convolutional matched-capacity control.

**Input contract:** field_embeddings, ordered_field_layout. See `model_api` for the framework constructor and calls. **Output:** `[B,G,D]`.

**Controlled test:** Compare original fields with original plus generated fields at matched downstream interaction budget, ablating convolution and recombination separately. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One stage, odd kernels, right-padded final pooling window, generated tokens only. No BatchNorm, source multistage schedule, original-field concatenation or head. All-absent samples yield zero. Independent tensor mechanism reference; host owns preprocessing, embeddings, training, prediction head and dataset evaluation. No task-specific lift or speed claim.

**Bundled code:** `models/pytorch/field_convolution.py`, `models/tensorflow/field_convolution.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FGCNN/src/FGCNN.py).

### `response_mlp` — `ResponseMLP`

Apply a raw-logit affine head when hidden_dims is empty, or a ReLU MLP for a declared sequence of hidden widths.

**When:** Use an explicit simple control or combine actual dense and sparse representations without inventing token semantics.

**Input contract:** dense_features. See `model_api` for the framework constructor and calls. **Output:** `[B]`.

**Controlled test:** Keep input encoding and objective fixed; use this as the no-extra-structure or capacity-matched control for a candidate component. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Generic LR/DNN control or host prediction head; no tokenizer, implicit sigmoid or causal interpretation.

**Bundled code:** `models/pytorch/ctr_embeddings.py`, `models/tensorflow/ctr_embeddings.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DNN/DNN_torch/src/DNN.py).


## CIN activation is a structural choice

The bundled `xdeepfm` exposes `cin_activation="relu"|"identity"`. The default ReLU
preserves existing behavior; both choices concatenate every layer's sum-pooled channels.
Its field-pair contraction can match an identity CIN algebraically before activation,
while its outputs differ when coordinates
are negative. Do not treat identity and ReLU variants as interchangeable or attribute
their difference to an incorrect einsum layout. The original xDeepFM paper uses identity
CIN activation and compares alternatives in Section 4.4; see
[the paper](https://arxiv.org/html/1803.05170). An identity-versus-ReLU ablation with fixed
embeddings, head and budget is a useful local experiment. Both native frameworks expose
the option and retain all-channel readout; split-half routing is not bundled. A nonlinear
variant does not inherit the exact global polynomial interpretation unchanged.


## Feature-pair micro-MLP: CAN / CoAction

`coaction` provides `CoAction(input_dim, layer_dims, order=1,
order_reduction="concat", use_bias=False)` in both native frameworks. The host
supplies `parameters[B,P]`, `feed[B,T,D]` and a True=valid mask `[B,T]`; the output
is `[B,T,C]`. A single position (`T=1`) supports ordinary feature pairs without
inventing an event sequence. Parameters are host-created trainable table lookups
or composed tensors, not business fields that need new data collection.

For each sample, flatten each row-major layer matrix `[d_in,d_out]`, followed
immediately by its optional bias, in layer order. `parameter_dim` gives the total
required width. Each power of the feed traverses the same micro-MLP, with tanh on
nonfinal layers and readout from every layer. `order_reduction="concat"` joins
orders in ascending order and follows the pinned author's readout choice;
`"sum"` follows the paper's order aggregation choice. `output_dim` is
`sum(layer_dims) * order` for concat and `sum(layer_dims)` for sum. The module
owns no trainable parameters and performs no history pooling; gradients flow to
both host inputs. Padding, including masked NaNs, is sanitized and output is zero.

Use one meaningful, pre-decision pair with adequate joint support, then pool or
fuse its result inside the current backbone. The catalog requires
`tabular_features` and `meaningful_feature_pairs`; lookup tables and masks belong
to the host tensor contract. Do not enumerate every pair or fabricate history
from aggregate columns. CAN differs from a conditional APG layer in its explicit
parameter layout, pair semantics, feed powers and layer readout; it does not
introduce the first possible conditional matrix transform.

Compare the current backbone, a bilinear/dot-product branch, full-matrix APG and
CoAction with one versus several orders under the same fields, split, head and
training budget. Match or report table size; isolate shared versus independent
interaction tables, order aggregation and fusion. Inspect sparse/new-ID slices,
branch gradients, quality and latency. Tensor correctness does not demonstrate
task improvement. This is an independent core, without the whole CAN training
system. Sources: [CAN section 4](https://arxiv.org/html/2011.05625v2#S4) and
[pinned author code](https://github.com/CAN-Paper/Co-Action-Network/blob/98a7219105568b9420cf28467877d08a1c56ae69/script/model.py).
