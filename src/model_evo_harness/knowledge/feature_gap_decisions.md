# When an Agent should request human feature work

Most production training datasets already contain many fields. The Agent should first learn what the present fields can express. A missing-feature request is a claim about unavailable decision-time information, not a generic reaction to a weak metric.

## Two valid paths

1. **Expert prerequisite:** business knowledge directly establishes that an input is essential to the decision and absent from the dataset. Examples include actual coupon cost for a net-value objective, timestamped history for a sequence hypothesis, or historical impressions for exposure-aware negative sampling. The Agent can request the field immediately, without running models to prove a definitional absence.
2. **Empirical residual:** the field gap is not obvious. The Agent compares simple baselines, feature transformations, interactions, sequence models where data permit, losses, and training choices on fixed splits. It records the residual slice, checks leakage and support, and explains why existing fields cannot encode the hypothesized cause. Only then does it propose a precise human data change.

## A useful request artifact

Name the business mechanism, field or aggregate, entity and time grain, decision-time cutoff, allowed lookback window, population coverage, expected missingness, and the specific future experiment that would test value. For a proposed `recent_3d_price_band × recent_15d_activity` feature, specify which events define price and activity, how late events are handled, and a comparison against separate raw fields and existing crosses. The human may reject the request if it cannot be collected safely or economically.

## Guardrails

Do not generate absent user facts from language-model guesses; do not compute features using post-action outcomes; do not call a weak model proof of a data gap. If the dataset version changes after humans add the field, start a new evidence context and re-evaluate earlier conclusions under the changed schema. Keep prior experience attached to its dataset fingerprint.

## Research lineage

This guide is independently written. Reference context: [Feature input contracts in DeepCTR](https://github.com/shenweichen/DeepCTR/blob/1b5fe40e158d1ee6af8b1d9df217a5ed5aea9136/docs/source/Features.md).
