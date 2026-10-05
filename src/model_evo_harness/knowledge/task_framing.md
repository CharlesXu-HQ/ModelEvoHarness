# Task framing and measurement

A model experiment begins with a decision, not an architecture. Specify the unit on which an action is chosen (request, user, session, or user-item pair), the available actions, the prediction target, and the business utility. For a coupon action, an outcome model and a policy are distinct: a model can estimate responses under treatment and control, while a policy decides whether the estimated increment justifies coupon cost.

## Data and evaluation contract

Record the randomization or exposure process, label horizon, pre-decision feature cutoff, dataset version, split keys, treatment support, action constraints, and the exact validation and holdout metrics. AUC, recall, calibration, expected value, and realized incremental value answer different questions. Retrieval additionally needs a fixed candidate universe and top-K; sampled negatives change the meaning of retrieval metrics.

## Experiment

Freeze the dataset, split, objective, metric code, budget, and selection rule before searching. Compare each candidate with a simple baseline on identical eligible units; store paired per-unit outcomes so uncertainty can be estimated. Use validation to choose and a final holdout once for the selected proposal. Repeatedly selecting on the holdout invalidates its interpretation.

## Interpretation

A better proxy loss does not establish better decisions. If labels or action costs needed for the declared objective do not exist, state the missing contract immediately as a domain fact. Otherwise, run simple baselines and investigate failure slices before requesting new features. Do not infer an unobserved business cause from a metric alone.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_0_introduction/1.intro.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_0_introduction/1.intro.html), [docs/chapter_0_introduction/2.outline.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_0_introduction/2.outline.html).
