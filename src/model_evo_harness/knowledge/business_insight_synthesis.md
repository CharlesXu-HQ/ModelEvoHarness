# From experiments to business insight

An experiment can teach two different things: how to build a better predictor or policy, and what the observed business response suggests about users, products, or incentives. Store those conclusions separately. A model-structure result may generalize within a fixed dataset; a business statement needs its own population, horizon, treatment, cost, and uncertainty context.

## Derive a useful observation

Start from a hypothesis stated before running the experiment. Compare the baseline and candidate on the same eligible units. Identify the measured slice, direction, effect size, interval, and sample support. Then state the narrowest business interpretation consistent with the design. Example: “Among randomized users with low recent activity, the proposed coupon rule raised estimated incremental gross margin over the baseline, with a wide interval”; that is more honest than “inactive users love coupons.”

## Causal and descriptive boundaries

Random assignment supports comparisons of assigned actions within overlap; it does not automatically explain *why* a segment responds. Observational feature importance is descriptive and can reflect selection, correlated fields, or leakage. A change in model score across segments is not a change in treatment effect. Budget interference, delayed outcomes, and unobserved redemption cost can limit the conclusion.

## Output contract

Record the business hypothesis, dataset fingerprint, action and outcome definitions, audience segment, compared policies, estimate and uncertainty, supporting/contradicting slices, limitations, and a suggested follow-up. Mark the conclusion `supported`, `tentative`, or `refuted` using the predeclared decision rule. Keep a separate technical note for architecture, loss, sampling, and feature results. Both are local to the dataset and task; a changed dataset requires review before reuse.

## Repeat and exploration hypotheses

The [Antai e-commerce solution](https://github.com/lmxhappy/Tianchi-AntaiCup-International-E-commerce-Artificial-Intelligence-Challenge/tree/0baf46ff442bfb2dfa52eb1393c9256d732f94d1) suggests a useful business question: are errors concentrated in recommending previously interacted items or discovering previously unseen items? Derive history strictly before the decision cutoff. A target's eventual repeat/exploration membership may define a label or an evaluation slice, but cannot become a serving input. A user who next chooses an unseen item is not necessarily a new or cold-start user.

First separate candidate recall coverage from ranking quality conditional on retrieval. More expressive ranking cannot recover a target absent from its candidate pool. Compare a global model with a history/association route under the same catalog, candidate budget and metric. Learn routing or stacking features from training-only out-of-fold predictions (time-respecting folds for temporal tasks); choose thresholds and mixture weights on validation, then freeze them before the final holdout. The competition's particular thresholds, split choices and reported scores are not transferable evidence.

Record a falsifiable business hypothesis before the experiment, the history definition, slice support, uncertainty and opposing evidence. A supported observation may be that an association route improves discovery coverage on this dataset; it does not establish why a customer prefers novelty or how they would respond to a coupon. Keep that business interpretation separate from the technical conclusion about routing or feature interactions. If the current data permit additional time-safe feature/structure tests, continue them; do not turn a plausible business story into an immediate request for new fields. Existing evidence requirements for human feature suggestions still apply.

## Research lineage

This guide is independently written. Reference context: [RecBole evaluation settings](https://recbole.io/docs/user_guide/config/evaluation_settings.html).
