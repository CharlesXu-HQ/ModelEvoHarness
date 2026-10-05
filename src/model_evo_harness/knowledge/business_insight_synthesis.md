# From experiments to business insight

An experiment can teach two different things: how to build a better predictor or policy, and what the observed business response suggests about users, products, or incentives. Store those conclusions separately. A model-structure result may generalize within a fixed dataset; a business statement needs its own population, horizon, treatment, cost, and uncertainty context.

## Derive a useful observation

Start from a hypothesis stated before running the experiment. Compare the baseline and candidate on the same eligible units. Identify the measured slice, direction, effect size, interval, and sample support. Then state the narrowest business interpretation consistent with the design. Example: “Among randomized users with low recent activity, the proposed coupon rule raised estimated incremental gross margin over the baseline, with a wide interval”; that is more honest than “inactive users love coupons.”

## Causal and descriptive boundaries

Random assignment supports comparisons of assigned actions within overlap; it does not automatically explain *why* a segment responds. Observational feature importance is descriptive and can reflect selection, correlated fields, or leakage. A change in model score across segments is not a change in treatment effect. Budget interference, delayed outcomes, and unobserved redemption cost can limit the conclusion.

## Output contract

Record the business hypothesis, dataset fingerprint, action and outcome definitions, audience segment, compared policies, estimate and uncertainty, supporting/contradicting slices, limitations, and a suggested follow-up. Mark the conclusion `supported`, `tentative`, or `refuted` using the predeclared decision rule. Keep a separate technical note for architecture, loss, sampling, and feature results. Both are local to the dataset and task; a changed dataset requires review before reuse.

## Research lineage

This guide is independently written. Reference context: [RecBole evaluation settings](https://recbole.io/docs/user_guide/config/evaluation_settings.html).
