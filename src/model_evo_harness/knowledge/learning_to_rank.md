# Query-group ranking objectives

A search or recommendation ranking task often evaluates the order of candidates within a query rather than the calibration of each individual score. RankNet and LambdaRank provide independent scalar objectives over score, relevance and mask tensors shaped `[B,N]`. Each batch row is a query group. They require no particular feature encoder, so the host can reuse an existing scorer and change only its objective.

## Pair construction and metric weighting

RankNet uses one directed pair for each strict relevance inequality within a query. Ties, self-pairs and padded candidates are excluded. The reference computes softplus of the negative score difference and averages across valid pairs in the batch. Queries with more labeled pairs therefore receive more total weight; query-balanced averaging is a separate experiment.

LambdaRank multiplies those pair losses by the absolute change in NDCG from swapping the predicted positions of the two candidates. Gains use `2**relevance - 1`, discounts use reciprocal log base two of rank plus one, and the ideal DCG is computed from valid labels. Metric weights and sorting are detached. The reduction still divides by valid pair count, preserving the influence of the metric weights instead of normalizing it away. This is a gradient surrogate, not LambdaMART tree boosting. Valid relevance values should be finite and nonnegative.

## Controlled experiment

Freeze query groups, labels, backbone, candidate eligibility and data split. Compare pointwise loss, RankNet and weighted pairs with identical optimizer budgets. Report query-level NDCG and retrieval/ranking guardrails, as well as query-size slices. Do not combine candidates from unrelated queries or reinterpret missing feedback as negative relevance. The implementation sanitizes masked score/label entries before pair arithmetic, including negative-infinity padding, but does not repair nonfinite valid scores.

## Research lineage

The components follow the RankNet-to-LambdaRank overview linked in `reco-007`. Method cards, framework manifests and the `query_delta_ndcg` training pattern expose their exact scope. Neither component claims to reproduce the original datasets, experiments or a tree-learning pipeline.
