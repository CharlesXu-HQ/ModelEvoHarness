# Evidence-driven training references

## Intent

Follow up the four reviewed repositories with non-duplicative local mechanisms that an Agent can inspect and apply inside its current backbone. Improve experimental hypotheses and diagnosis; this change does not establish a task-metric improvement.

## Decisions

- Implement a PEP-inspired `PrunableEmbedding` independently in PyTorch and TensorFlow: learn soft thresholds, export the resulting mask, and construct a separate fixed-mask layer from the original initialization for retraining. Dense masked storage does not demonstrate memory or latency savings.
- Implement a conditional uniform/popularity/mixture negative sampler in both frameworks. The host supplies time-valid eligibility and known positives. Sampling is with replacement; return the actual conditional probability of each draw after exclusion. Do not treat that probability as exposure propensity.
- Add a standard-library `BinaryPredictionDiagnostics` in `analysis/`, shared by both framework hosts. Require exact sample-ID pairing; measure weighted AUC, LogLoss, class contributions and paired loss changes. Describe disagreement without inventing its cause or significance.
- Extend existing sampling and business-insight guidance with time-safe repeat/exploration hypotheses, fixed evaluation universes and validation-only routing. Keep technical findings separate from dataset-bound business observations.

## Approach

Adding only links would leave executable gaps. Importing upstream trainers would add framework wrappers and dataset assumptions. Use independent local components, existing capability screening and complete-source delivery; keep dataset preparation, training, evaluation and deployment in the host.

## Boundaries

No new backbone, automatic pruning scheduler, competition trainer, causal estimator, data acquisition or Agent training. No CouponEvo change. The target GPU machine was shut down at the user's request; local deterministic tensor tests are not dataset/model experiments, and GPU validation remains pending.

## Acceptance

Numerical fixtures establish threshold/mask behavior, conditional sampling probabilities and paired metric decomposition. Invalid inputs fail explicitly. Both framework implementations appear in the existing API and reference-delivery paths. Analysis stays outside model components. Existing tests and catalog coverage pass. English and Chinese documentation state source differences and validation limits.
