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
