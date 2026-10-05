# Adaptive piecewise response models

A mixture of local response functions can approximate different regimes without training a completely separate model for every segment. In MLR-style formulations, a gate gives soft weights to local experts and the weighted sum produces the prediction. Learned regions describe predictive heterogeneity; they are not treatment effects unless a causal design supports that interpretation.

A simple formulation is `μ(x) = Σ_k g_k(x) h_k(x)` with `g_k(x) ≥ 0` and `Σ_k g_k(x) = 1`; MLR uses local linear response functions for `h_k`. The gate's soft regions can be inspected for stability, but their labels are not automatically business segments.

## Data and conditions

Require observed labels, pre-decision tabular fields, enough examples per effective expert, and a serving path for the gate. A gate can collapse to one expert or fragment into tiny unstable regions. Measure expert occupancy across time and treatment arms if a treatment model is built.

## Controlled experiment

Compare one linear model, a size-matched MLP, explicit known-segment heads, and one small mixture on the same rows and split. Track per-segment calibration and primary utility, not just aggregate loss. Regularize or cap expert count based on validation only; inspect latency and stability of learned assignments.

## Interpretation and feature needs

If known business segments explain the gain, a direct scenario feature may be simpler. If learned experts change arbitrarily across folds, the apparent heterogeneity may be noise. Only after stable errors persist under existing fields should the Agent request richer segment descriptors, unless experts can name an obvious absent segment variable from the business process.

## Method choice

| Method | What changes | First discriminating test |
| --- | --- | --- |
| MLR | A soft gate mixes several local linear response functions. | Compare with one global linear model and known-segment heads. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [deepctr](https://github.com/shenweichen/DeepCTR/blob/1b5fe40e158d1ee6af8b1d9df217a5ed5aea9136/deepctr/models/mlr.py), [mlr](https://github.com/shenweichen/DeepCTR/blob/1b5fe40e158d1ee6af8b1d9df217a5ed5aea9136/deepctr/models/mlr.py).
