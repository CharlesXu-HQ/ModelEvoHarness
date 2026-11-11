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
