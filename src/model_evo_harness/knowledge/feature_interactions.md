# Feature representation and interactions

A linear head adds independent feature effects. FM factorizes second-order pairs; FwFM changes strength by field pair; DeepFM combines pair terms with an MLP; DCN and DCNv2 build explicit higher-order crosses; xDeepFM, attention, and product networks vary how interactions are parameterized. More parameters are not, by themselves, evidence that interactions matter.

A factorization machine uses `z(x) = b + Σ_i w_i x_i + Σ_{i<j} <v_i,v_j> x_i x_j`; the shared embedding vectors permit pair estimates where an explicit one-hot cross would be sparse. A vector DCN layer can be viewed as `x_(l+1) = x_l + x_0 (w_l^T x_l) + b_l`, making the cross degree grow by layer. Matrix-based DCNv2 changes the transform inside that explicit cross; whether it helps is an empirical question under a capacity budget.

## Data and conditions

Declare typed categorical, numeric, multi-hot, and sequence fields; cardinality, missingness, normalization, and the time at which each value becomes known. Audit post-outcome leakage. A cross has a support problem: user-segment × price-band may be plausible but too rare to estimate. Embedding dimensions, sparse-feature handling, and inference cost belong in the comparison.

## Controlled experiment

Use a linear model and plain MLP as controls, then add one interaction mechanism while keeping fields, split, preprocessing, and a declared parameter/latency budget fixed. Ablate the cross branch and compare supported field-pair residuals, calibration, primary decision value, and sparse slices. For explicit crosses, inspect support and sign stability across folds.

## Interpretation and feature needs

If multiple distinct interaction designs fail on the same supported slice, investigate target noise, missing temporal context, or absent semantics before requesting another cross. Ask humans to add a specific aggregate or cross field after controlled attempts reveal an input gap. A business expert can instead name an inherently essential missing pre-decision field, such as recent spending band for a coupon value decision, without waiting for model ablations.

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

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/1.wide_and_deep.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/1.wide_and_deep.html), [docs/chapter_2_ranking/2.feature_crossing/1.second_order.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/2.feature_crossing/1.second_order.html), [src/funrec/models/wide_deep.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/wide_deep.py), [src/funrec/models/fm.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/fm.py).

Additional pinned source for `dcn_v2`: [torch_rechub/models/ranking/dcn_v2.py](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/ranking/dcn_v2.py).

Additional pinned source for `fwfm`: [deepctr/models/fwfm.py](https://github.com/shenweichen/DeepCTR/blob/1b5fe40e158d1ee6af8b1d9df217a5ed5aea9136/deepctr/models/fwfm.py).
