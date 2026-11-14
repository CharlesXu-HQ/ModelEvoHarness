# Representation diagnostics and protected feature screening

## Mechanism and inputs

Measure intermediate representations before choosing a repair. `RepresentationDiagnostics`
accepts a supplied `[batch, tokens, channels]` layer output and valid-token mask.
Entropy effective rank and stable rank are different quantities: the first uses
the singular-value probability distribution, the second is squared Frobenius norm
divided by squared spectral norm. Both are computed per sample. State whether
token means were subtracted; zero input has rank zero by convention. Output
variance is population variance across valid tokens, averaged across channels.
It is not prediction variance across users or uncertainty in a treatment effect.

Use `inspect` on named layer tensors to compare the input, interaction blocks and
readout. Paired branch similarity requires corresponding token/channel axes. A
low-dimensional representation can be sufficient for a simple target; a high rank
can be noise. Describe the observed signal, competing explanations and the
intervention needed to distinguish them. Do not turn a rank or cosine threshold
into an automatic model recommendation.

`GradientFeatureSelector` consumes activations and aligned gradients from a
declared training objective. Its score averages the absolute channel-summed
activation-times-gradient contribution across valid examples. Inputs can describe
fields or already constructed cross features; their mapping back to original
fields must be preserved. Redundancy is absolute cosine on pairwise-common valid
observations, with overlap counts. Small overlap, saturated activations and noisy
gradients weaken an interpretation; no redundancy score removes an input by itself.

## Controlled experiments

First collect a fixed diagnostic cohort, checkpoint, split identity, mask convention
and resource cost without changing predictions. If repeated evidence supports a
specific bottleneck, change one grouping, projection, sharing rule or interaction
branch inside the current backbone. Compare with its removal under the same data,
training budget, objective and decision rule. Report task metrics, slices and cost
alongside the diagnostic; a healthier spectrum alone is not a success criterion.

Screen features using training rows only. `select` preserves `protected_indices`
and every member of `protected_groups`; an impossible protected budget is an error.
Protection must be supplied from known interactions or explicit interaction tests.
The selector cannot discover all interaction-only fields from marginal scores.
Repeat screening over training batches or seeds, then compare full-input and
screened candidates after matched retraining and group ablation. Keep the final
holdout sealed and retain source-field lineage when selecting cross tensors.

## Failure signals and feature requests

Reject pruning when it removes supported interactions, destabilizes a critical
slice, or loses task performance at matched cost. Missing gradients or hidden
states are missing observations, not evidence that features are missing from the
dataset. Try appropriate encoding and local structure alternatives before asking
humans for more fields, except when an explicit expert business requirement shows
that a necessary pre-decision input is absent. Bind both technical and business
lessons to the actual task, dataset version and evaluation protocol.

## Research lineage

Original native utilities informed by [RankUp](https://arxiv.org/html/2604.17878v3)
and [TayFCS](https://arxiv.org/html/2507.03895v1). They are not full reproductions:
there is no RankUp token construction, learned pruning policy, cross enumeration,
LO-FAR estimator or causal feature-importance guarantee.


## COLD-inspired physical feature subset search

`cold_feature_search` is an analysis operation available through `analysis_api`
and `read_reference` for either framework. Both labels return the same stdlib
module `analysis/cost_aware_search.py`, with no deep-learning runtime import.
`CostAwareFeatureSearch(latency_budget_ms, *, quality_floor=None,
max_quality_drop=None)` requires at least one quality constraint. The maximum
quality drop is an absolute difference from the retrained full-input baseline,
not a percentage. Quality is a declared maximize metric; adapt minimize metrics
explicitly and keep that definition fixed.

Call it with `saliency`, retained-count `budgets`, `retrain_evaluate`, and keyword
arguments `context`, `saliency_split`, `protected_indices=()` and
`protected_groups=()`. Saliency must come from the context's training split.
The analysis uses stable score/index ordering to propose subsets and always
retains every protected group member, including low-saliency interaction-only
fields. Budgets below that requirement are rejected. Keep group/cross-to-source
field lineage in the host; the orchestrator does not discover protected groups
or causal feature value from marginal scores.

The callback receives exact `retained_indices` and `context`, first for the full
baseline and then once per distinct proposed subset. The host must remove
omitted feature reads and model input dimensions, rebuild the candidate, run its
real trainer under the fixed budget, and measure validation quality and latency.
Return `retrained=True`, `physical_pruning=True`, the exact retained tuple and
unchanged context, `implementation_id`, a unique `train_run_id`, finite
`validation_quality`, and positive `latency_ms`. These fields attest what the
host did; retain auditable code/training artifacts because the function cannot
prove that an external trainer actually ran. Zeroing inputs or recording gate
weights does not physically prune a feature or establish speedup.

Context fixes `task_id`, `dataset_id`, `train_split_id`, `validation_split_id`,
`quality_metric`, `quality_direction="maximize"`, `training_budget_id`, `seed`,
and `serving_profile`. The profile declares `hardware`, `batch_size`,
`candidate_count`, `concurrency`, `latency_statistic` and
`includes_feature_lookup`; include immutable dtype/software/timing details when
relevant. All context fields must match every trial. Train and validation split
identities must differ. Preserve the final holdout for a later confirmation;
repeated validation selection does not make that validation result unbiased.

The result includes baseline, every trial, feasible `pareto_candidates`, and
`selected` (None when no trial meets the constraints). Selection maximizes
feasible quality, then minimizes latency and subset size. It reports whether
latency includes feature lookup; a model-only profile cannot establish overall
service savings. This protocol does not measure QPS, implement online learning,
or reproduce the complete [COLD system](https://dlp-kdd.github.io/dlp-kdd2020/assets/pdf/a13-wang.pdf).

Use the five explicit prerequisites: `training_feature_saliency`,
`feature_group_lineage`, `protected_interaction_groups`,
`retrain_evaluate_callback`, and `serving_profile`. Reuse the current backbone and
compare physically smaller retrained variants with the full-input control,
including critical slices and group ablations. The analysis ID is not a model
component or an optimization-trial method: the actual retrained candidate is
the experiment, with its own source, measurements and task-bound evidence.
