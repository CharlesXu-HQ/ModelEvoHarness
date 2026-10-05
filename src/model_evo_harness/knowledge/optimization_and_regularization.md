# Optimization, regularization, and training controls

An architecture comparison is uninterpretable when training budgets or basic optimization differ. Learning rate, batch size, embedding dimensions, weight decay, dropout, gradient clipping, normalization, scheduler, early stopping, and precision can change quality and cost without changing the network family. These are legitimate Agent experiment directions when the run record points to underfitting, overfitting, or instability.

## Diagnose before changing a knob

Compare training and validation curves, per-slice errors, gradient norms, calibration, and repeated seeds. Underfitting with both losses high suggests insufficient representation, optimization, or label signal. A widening train-validation gap suggests capacity, leakage, selection, or regularization issues. Highly unstable results call for seed and batch diagnostics before interpreting small metric differences.

## Controlled experiments

Change one training factor at a time while fixing the data and model code. For learning rate, use a bounded log-scale search with equal update budget. For embedding size or dropout, compare total parameter count, training time, and serving cost. Early-stop only on validation and record the selected epoch; never use the final holdout to choose it. If repeating seeds, report mean and uncertainty rather than the best seed.

## Advanced options

Distillation may compress an already validated teacher into a cheaper student; it does not create new ground truth. Curriculum learning can change example order but must keep exposure and label semantics. Ensembling can reduce variance but consumes serving budget and may obscure component-level learning. Mixed precision and sparse optimizers can speed training, yet numerical stability must be checked. Treat each as a falsifiable cost-quality hypothesis.

## Other training directions worth testing

| Direction | Signal that motivates it | Necessary control or caveat |
| --- | --- | --- |
| Sample weighting | Important populations are underrepresented or exposed at different rates. | Compare unweighted training; distinguish business weights from propensity weights and inspect effective sample size. |
| Self-supervised pretraining | Labeled outcomes are sparse but valid pre-decision events are abundant. | Compare from-scratch training with equal labeled-data and serving budgets; pretraining cannot include test-period events. |
| Transfer or fine-tuning | A related scene has shared input semantics and much more support. | Compare pooled, transferred, and local-only models on the target population; watch negative transfer. |
| Contrastive representation learning | Retrieval embeddings collapse or miss semantic neighbors. | Hold positive/negative definitions and candidate evaluation fixed; audit false negatives. |
| Calibration | Ranking improves but thresholds or expected-value decisions fail. | Fit calibration only on validation/calibration data, then evaluate the fixed rule on holdout. |
| Data augmentation | Rare but valid configurations have too little training support. | Compare with simple resampling, and measure on untouched real validation and holdout. |
| Adversarial or noise robustness | Small plausible input perturbations cause unstable decisions. | Define allowable perturbations from the product domain; ensure robustness does not erase real signal. |
| Ensemble or distillation | Independent seeds are noisy or a strong teacher is too costly to serve. | Compare value per serving cost, and keep teacher/student training splits separate from evaluation. |

A direction is eligible only when its required fields and labels are actually available. More training variants are not a substitute for an estimand, fixed evaluator, or business decision rule.

## Feature requests

A stalled optimizer is not evidence of missing features. Request a new input only when a domain prerequisite is obvious or multiple sound models and training regimes leave a stable, explainable residual tied to unobserved information.

## Research lineage

This guide is independently written. Reference context: [FunRec trainer module](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/training/trainer.py), [FunRec loss module](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/training/loss.py).
