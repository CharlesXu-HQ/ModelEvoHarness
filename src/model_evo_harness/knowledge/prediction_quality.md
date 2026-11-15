# Paired prediction quality

AUC asks whether positives rank above negatives; binary LogLoss measures the probabilities assigned to the observed outcomes. An unchanged ordering can have a different loss after probability rescaling. An improved AUC can coexist with a worse loss if some confident errors become more costly. Neither observation alone establishes miscalibration, an incorrect objective or a need for a larger backbone.

## Observe before changing the model

Use `binary_prediction_diagnostics` / `BinaryPredictionDiagnostics` from the analysis API. Supply probabilities, binary labels, one fixed weighting scheme and the exact ordered unique sample IDs for both predictors. Declare dataset and split identity. The tool rejects mismatched IDs instead of silently comparing different populations. The host still owns correct joins, label maturity and split provenance.

The tool computes AUC from original probabilities with half credit for tied scores. AUC is undefined when either class has zero effective weight. LogLoss uses the declared epsilon only for numerical clipping; inspect the clipping count. Report each class's conditional loss and its contribution to the total separately: different class frequencies can change the total without changing within-class performance. Paired loss changes describe which rows improved or worsened; they are not uncertainty intervals or evidence of a causal business mechanism.

## Decide the next experiment

1. Check row alignment, label window, class support, weights, probability range and clipping before spending a training trial.
2. If ordering is stable while probability quality deteriorates, test a calibration hypothesis on a permitted calibration split. Keep the evaluator and final holdout fixed; do not fit on the rows used to claim final improvement.
3. If ordering deteriorates on a supported slice, investigate its representation, sampling or objective with one controlled change. AUC/LogLoss disagreement alone does not identify which explanation is correct.
4. If only a few observations dominate loss changes, inspect label maturity and confident errors. Do not automatically remove them or relabel difficult examples.
5. Keep the task's predeclared primary metric. A more favorable secondary metric does not automatically promote a candidate or justify a metric switch.

For randomized coupon experiments, response-prediction AUC and LogLoss diagnose nuisance or outcome models. Policy value, uplift and cost need their own causal evaluation and paired uncertainty; this analysis cannot substitute for them.

## Local implementation boundary and source

The standard-library module `analysis/prediction_diagnostics.py` is shared by PyTorch and TensorFlow hosts and retrieved as complete source. It does not fit models, change predictions or execute automatically when an Agent reads it. Record actual host results before using them as evidence.

## Research lineage

The motivating [AUC/LogLoss notebook](https://github.com/lmxhappy/why_loss_auc_unconsistent/blob/a6aea3647affa9d952c5b975d0180c387c584c54/why_loss_auc_unconsistent.ipynb) compares class losses but is not a reproducible benchmark: its saved output includes a non-finite loss and its labels are not defined in the inspected notebook. This guide and implementation are independent; no notebook code or outputs are copied as validation evidence.
