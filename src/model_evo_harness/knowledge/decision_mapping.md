# Prediction-to-decision mapping

A predictor estimates a quantity; a decision rule chooses an action. For a coupon, a simple policy may issue when estimated incremental margin minus expected redemption cost is positive, subject to budget and eligibility. Thresholds, quotas, fairness, and capacity can change realized value without changing the model weights. This is a separate optimization direction from network architecture.

## Data and conditions

Require calibrated predictions for each available action or a supported uplift estimate, action cost, eligibility, budget, and an evaluator that estimates policy value. Distinguish offered face value from actual expected cost. If a cost field is unavailable, define a sensitivity range rather than silently treating cost as zero.

With action-conditional expected margins `μ_1(x)` and `μ_0(x)` and expected coupon cost `c(x)`, an unconstrained local rule offers the coupon when `μ_1(x) - μ_0(x) - c(x) > 0`. A budget cap changes this from independent thresholds into a selection problem; the offline evaluator must match that constraint.

## Controlled experiment

Freeze predictions and validation data. Compare current rule, unconstrained net-value threshold, and one budget-constrained rule under the same objective; report action rate, cost, incremental value, and paired uncertainty. Tune thresholds on validation, then evaluate the chosen rule once on the holdout. If constraints couple users, a row-wise policy estimator may not capture all effects.

## Interpretation and feature needs

A policy gain with identical predictions identifies a decision-rule bottleneck. If cost or eligibility is absent but intrinsic to business rules, request it immediately. If existing inputs support several rules and all fail on stable slices, investigate model information or action support before suggesting new fields.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [wangshusen-recommendersystem](https://github.com/wangshusen/RecommenderSystem/blob/8c8796c560c13059b19b51f5b5b963137f4ef287/Slides/03_Rank_03.pdf).
