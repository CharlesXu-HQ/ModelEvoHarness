# Negative sampling and hard-example mining

Training examples are shaped by the sampling policy. In implicit-feedback retrieval, an unclicked item may be unseen rather than disliked. Hard-negative mining chooses items the current model scores highly while treating them as negatives; this can sharpen boundaries, but it increases false-negative risk and can overfit exposure patterns. For randomized coupon outcomes, low-response users are **not** negative treatment effects: the two potential outcomes are only partly observed.

## Source of negatives

Distinguish observed non-response after exposure, uniformly sampled unexposed items, popularity-sampled items, in-batch negatives, and mined near-misses. Record the sampling probability and whether a candidate was eligible at the historical decision time. Keep positives from the same user or near-duplicate item out of the negative pool when they would be false negatives. Never mine from validation or test labels.

## Mining strategies and controls

Start with uniform or exposure-matched sampling, then a mixture of easy and hard candidates from a frozen previous checkpoint. Vary hardness or mined fraction as the single experiment factor. A temperature or curriculum can gradually increase hardness, but compare against a compute-matched random-sampling run. For ranking, preserve candidate count and loss; for retrieval, preserve batch size and index version. If using an adaptive sampler, freeze its version per experiment and log its distribution.

## Executable conditional proposals

`conditional_negative_sampling` provides native `ConditionalNegativeSampler` references. Catalog positions share a versioned ID/count mapping. For uniform sampling use a flat catalog proposal; for popularity use normalized `count ** alpha` with finite `alpha >= 0`. The optional uniform mixture is formed over the catalog **before** each row's eligibility/positive exclusion. Renormalize the remaining mass and draw **with replacement**, returning `item_ids` and their actual conditional `sampling_probs`. The mixture coefficient is not the final per-row uniform share. Duplicates are valid; deduplicating later changes the probability semantics.

The host supplies counts, eligibility and known-positive masks at the training cutoff. Do not use later outcomes to clean training negatives. A sampled unobserved item is not an observed dislike. Zero-support rows fail explicitly, rather than switching distributions or returning fewer examples. Derive any sampled-softmax or importance correction for the actual objective: per-draw q is not an inclusion probability under sampling without replacement and is not an exposure propensity. Keep the evaluator's candidate universe fixed when changing the sampler.

These references use dense `[B,N]` masks/distributions and a local/stateless seed. They do not implement an industrial catalog sampler or promise identical draws between frameworks. The [DSSM-Lookalike reference](https://github.com/lmxhappy/DSSM-Lookalike/tree/07dd9442a8a721aa6db678c5dfc4aa65e5708e71) motivates the comparison, but its frequency/rank formulas and filtering behavior are not copied or labeled as a faithful RALM implementation.

## Evaluation and rejection

Assess full-catalog or fixed-candidate recall, tail coverage, calibration if scores become probabilities, and utility after decision rules. Include false-negative audits and slices by item popularity and history density. A better metric under the *same sampled negatives used for training* is weak evidence; a fixed evaluator is essential. In coupon policy tasks, treat high-uncertainty or discordant treatment rows as analysis slices, not supervised hard negatives, unless the causal estimation method defines a valid weighted objective.

## When to request data

Absent impression logs do not prevent all retrieval experiments. A declared implicit-feedback task can compare uniform, popularity, in-batch and model-mined non-interactions while acknowledging that these are sampled unobserved items, not verified negatives. Keep the candidate evaluator fixed and report false-negative risk. Exposure-conditioned response modeling requires actual exposure records; an immediate terminal data request still needs an explicit host domain requirement. Missing inputs for one sampler do not show that other experiments are impossible.

## Research lineage

This guide is independently written. Reference context: [RecBole evaluation settings](https://recbole.io/docs/user_guide/config/evaluation_settings.html), [YouTubeSBC reference](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/matching/youtube_sbc.py).
