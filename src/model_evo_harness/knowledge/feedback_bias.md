# Feedback bias and causal support

Observed feedback is filtered by prior exposure and policy decisions. A high predictor score among shown items may reflect where the old system chose to show them. Randomized treatment assignment can identify a treatment effect within its support; logged recommendation exposure generally needs additional assumptions or randomization to evaluate a changed policy.

## Data and estimand

Record assignment or exposure probability, treatment/action, observed outcome, eligible population, time horizon, and any interference or budget constraints. Define whether the goal is response prediction, treatment effect, or policy value. For two actions, uplift is the difference between potential-outcome expectations; only one outcome is observed per unit. Propensity scores are valid only when they describe the actual behavior policy.

For a policy `π` over logged randomized actions, a basic inverse-propensity estimate is `V̂(π) = (1/n) Σ_i 1[A_i = π(X_i)] Y_i / p(A_i | X_i)`. It requires positive probability for every evaluated action (`0 < p < 1` in the binary case); large weights increase variance. A net-value outcome should subtract the action's realized or expected cost according to the declared estimand.

## Controlled experiment

For a randomized coupon dataset, compare policies on the same held-out units using a predeclared estimator and paired uncertainty interval; inspect overlap and effective sample size. For observational logs, compare unadjusted and supported propensity-based estimates with diagnostics. Change one modeling or policy component at a time; keep outcome windows and cost definitions fixed.

## Interpretation and feature needs

Extreme importance weights, treatment imbalance, or missing action support weaken offline conclusions. Do not interpret heterogeneous response from an observational predictor as causal without design support. A known absent assignment probability or missing randomized control is an immediate data-contract gap; asking for more ordinary user features cannot repair missing support by itself.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_4_trends/1.debias.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_4_trends/1.debias.html).
