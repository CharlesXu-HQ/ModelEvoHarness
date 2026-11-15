# Evidence-driven training and prediction references

These additions address specific experimental gaps found while reviewing PEP,
DSSM-Lookalike and an AUC/LogLoss notebook. They
extend the current backbone's representation and training options. They do not
require a backbone switch or make every direction mandatory.

| Observed problem | Local component or guidance | Controlled next step |
| --- | --- | --- |
| Embedding capacity or sparse-ID overfitting | `prunable_embedding` / `PrunableEmbedding` | Dense vs matched random mask vs learned mask, with identical initialization and a declared total training budget |
| Easy or popularity-biased retrieval negatives | `conditional_negative_sampling` / `ConditionalNegativeSampler` | Change only the proposal distribution, preserving positives, negative draws and the evaluator's candidate universe |
| Conflicting AUC and LogLoss | `binary_prediction_diagnostics` / `BinaryPredictionDiagnostics` | Check paired rows, class losses and clipping before selecting a calibration, representation or training hypothesis |

## Native embedding pruning

Both `models.pytorch.embedding_pruning` and
`models.tensorflow.embedding_pruning` expose:

```python
PrunableEmbedding(initial_weights, threshold_type="global",
                  threshold_init=-3.0, padding_idx=None, *, fixed_mask=None)
PrunableEmbedding.from_mask(initial_weights, mask, padding_idx=None)
```

The search table is `sign(V) * relu(abs(V) - sigmoid(s))`; sharing options are
`global`, `dimension`, `feature` (vocabulary row) and `feature_dim`. Integer IDs
of shape `[...]` yield embeddings `[...,D]`. `effective_weight()`,
`export_mask()` and `nonzero_count()` expose the current representation and
support. The finite default threshold is not the authors' tuned search schedule.

Save initial weights before search. Use the selected mask with the original
weights in `from_mask`, and reset the optimizer and other backbone state as
required by the matched experiment. Fixed-mask retraining has no trainable
threshold. A masked cell cannot return; an allowed cell that happens to become
zero can still learn. Padding remains zero. Checkpoint loading requires a layer
constructed in the same mode; the host retains initialization and optimizer state.

The layer still allocates and transforms a dense table. Nonzero counts are not
physical memory, latency or field-dimension compression measurements. Full
training, mask-budget selection and sparse deployment belong to the host.

## Native conditional sampling

Both `models.pytorch.negative_sampling` and
`models.tensorflow.negative_sampling` expose:

```python
sampler = ConditionalNegativeSampler(distribution="popularity",
                                     alpha=0.75, uniform_mix=0.1)
draws = sampler.sample(counts, eligible_mask, positive_mask,
                       num_samples=negative_count, seed=step_seed)
# draws["item_ids"] and draws["sampling_probs"] both have shape [B,K].
```

Counts `[N]` and both boolean masks `[B,N]` use one fixed item-ID mapping. The
catalog proposal is uniform or proportional to `count ** alpha`, optionally
mixed with uniform. Each row then excludes ineligible and known-positive items
and renormalizes. Thus `uniform_mix` is a catalog coefficient, not the final
uniform share in each row. Zero support fails explicitly.

Draws are **with replacement**. The returned probability is the actual
conditional probability of that independent draw. Do not deduplicate while
retaining the same probability interpretation, or equate q with an exposure
propensity. Any log-q correction requires an objective-specific derivation.
The host supplies time-safe counts/masks, changes seeds between steps and
audits false negatives. Local/stateless seeds leave the global RNG untouched;
draws need not match between frameworks. Dense `[B,N]` work is a bounded
reference, not an industrial catalog sampler or a claimed RALM reproduction.

## Shared prediction analysis

`analysis.prediction_diagnostics.BinaryPredictionDiagnostics` uses only the
standard library. Both framework manifests point to this one implementation:

```python
report = BinaryPredictionDiagnostics(eps=1e-15)(
    labels, baseline_probabilities, candidate_probabilities,
    sample_ids=label_ids, baseline_sample_ids=baseline_ids,
    candidate_sample_ids=candidate_ids,
    dataset_id=dataset_fingerprint, split_id=validation_fingerprint,
    sample_weight=evaluation_weights,
)
```

IDs must be unique and match exactly in order. Labels are numeric 0/1 and scores
are probabilities, not logits. Optional weights are shared, finite and
nonnegative, with positive total mass. AUC uses original probabilities and
half credit for ties; it is `None` for a zero-effective-weight class. LogLoss
uses declared clipping and reports clipping counts, per-class conditional loss
and contributions to overall loss. Deltas are candidate minus baseline.
The upper clip uses the actual floating-point value of `1 - eps`; predictions
at and above it have equal loss. If that bound rounds to one, the exact endpoint
uses the finite virtual-endpoint convention documented in the implementation.

The report describes paired loss changes and metric disagreement. It does not
fit calibration, identify causes, compute confidence intervals, select a
champion or demonstrate uplift. Read the
[prediction-quality guide](../src/model_evo_harness/knowledge/prediction_quality.md)
for evidence-driven next steps.

## Agent and host responsibilities

Cards, capability screening, model/analysis APIs, patterns and complete-source
retrieval expose these components through existing interfaces. A declared
model/training component triggers source delivery before the provider accepts
the candidate. A prediction diagnostic is an analysis tool: the host explicitly
runs it and records observations; reading its source is not execution or a
successful optimization trial.

Source differences and local verification are recorded in the
[review](research/training-reference-gap-review-2026-11-14.md).
