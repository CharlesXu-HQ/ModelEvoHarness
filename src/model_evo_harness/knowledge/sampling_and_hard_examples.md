# Negative sampling and hard-example mining

Training examples are shaped by the sampling policy. In implicit-feedback retrieval, an unclicked item may be unseen rather than disliked. Hard-negative mining chooses items the current model scores highly while treating them as negatives; this can sharpen boundaries, but it increases false-negative risk and can overfit exposure patterns. For randomized coupon outcomes, low-response users are **not** negative treatment effects: the two potential outcomes are only partly observed.

## Source of negatives

Distinguish observed non-response after exposure, uniformly sampled unexposed items, popularity-sampled items, in-batch negatives, and mined near-misses. Record the sampling probability and whether a candidate was eligible at the historical decision time. Keep positives from the same user or near-duplicate item out of the negative pool when they would be false negatives. Never mine from validation or test labels.

## Mining strategies and controls

Start with uniform or exposure-matched sampling, then a mixture of easy and hard candidates from a frozen previous checkpoint. Vary hardness or mined fraction as the single experiment factor. A temperature or curriculum can gradually increase hardness, but compare against a compute-matched random-sampling run. For ranking, preserve candidate count and loss; for retrieval, preserve batch size and index version. If using an adaptive sampler, freeze its version per experiment and log its distribution.

## Evaluation

Assess full-catalog or fixed-candidate recall, tail coverage, calibration if scores become probabilities, and utility after decision rules. Include false-negative audits and slices by item popularity and history density. A better metric under the *same sampled negatives used for training* is weak evidence; a fixed evaluator is essential. In coupon policy tasks, treat high-uncertainty or discordant treatment rows as analysis slices, not supervised hard negatives, unless the causal estimation method defines a valid weighted objective.

## When to request data

Absent impression logs do not prevent all retrieval experiments. A declared implicit-feedback task can compare uniform, popularity, in-batch and model-mined non-interactions while acknowledging that these are sampled unobserved items, not verified negatives. Keep the candidate evaluator fixed and report false-negative risk. Exposure-conditioned response modeling requires actual exposure records; an immediate terminal data request still needs an explicit host domain requirement. Missing inputs for one sampler do not show that other experiments are impossible.

## Research lineage

This guide is independently written. Reference context: [RecBole evaluation settings](https://recbole.io/docs/user_guide/config/evaluation_settings.html), [YouTubeSBC reference](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/matching/youtube_sbc.py).
