# Training objectives and losses

Changing the objective can improve a fixed model more than adding layers. Choose a loss from the *observed label process* and the *decision objective*, then compare it under the same evaluator. A lower training loss under a new objective is not itself a better model.

## Common objective families

- **Pointwise probability:** binary cross-entropy on an observed response supports calibrated response modeling if examples and weights reflect the target population. Focal loss or class weighting emphasizes rare/hard rows, but may alter probability calibration; recalibrate before using expected value.
- **Pairwise ranking:** BPR-like loss increases the score margin between an observed positive and a sampled negative. Its meaning depends on how negatives were sampled. It targets ordering, not an absolute response probability.
- **Listwise and sampled softmax:** compare a positive to a set of candidates. If candidates come from a nonuniform sampler, document or correct the sampling distribution; otherwise the learned score can absorb popularity bias.
- **Multi-task losses:** a weighted sum of task losses is a policy choice. Measure gradient conflict and primary-task value; adjusting weights is an experiment separate from changing shared-bottom, MMoE, or PLE structure.
- **Treatment and policy objectives:** in randomized data, estimate outcomes by action or a treatment-effect pseudo-outcome using the known assignment probability. For binary treatment, a simple transformed outcome is `Y*T/p - Y*(1-T)/(1-p)`; its conditional expectation is the treatment effect under valid randomization, but variance can be high. A policy's value must still be estimated on held-out randomized units. Net business value also needs cost and action constraints.

For a positive item `i+` and sampled negative `i-`, a BPR-style loss is `-log σ(s(u,i+) - s(u,i-))`. It only asserts an ordering within the sampled pair. For randomized treatment, the transformed outcome formula below requires `0 < p < 1` and a recorded probability for the actual assignment mechanism.

## Controlled experiment

Keep dataset, preprocessing, architecture, optimizer budget, and split fixed while changing one loss. Compare optimization stability, calibration, ranking or uplift metric, final policy value, and per-slice variance. Use training data only to fit any reweighting or pseudo-label transformation. Select the loss on validation and reserve the final holdout.

## Failure signals

AUC gain with calibration collapse can worsen threshold decisions; a sampled-ranking gain may vanish on the full catalog; high-variance inverse-propensity weights can make treatment gains unstable. When the objective requires an unavailable cost or action-propensity field, request it as a known data-contract gap. Do not ask for arbitrary new features merely because a harder loss was unstable.

## Research lineage

This guide is independently written. Reference context: [FunRec loss module](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/training/loss.py), [RecBole evaluation settings](https://recbole.io/docs/user_guide/config/evaluation_settings.html).
