# Training Interventions

Does an isolated objective or regularization change improve learning with the current backbone and inputs?

## Research lineage

Original tensor implementations informed by the pinned FuxiCTR source inventory. No FuxiCTR model or training wrapper is imported.

## FuxiCTR mechanism components

These are independently implemented, composable PyTorch and TensorFlow mechanisms. Read the full local module and output contract before editing the current candidate. A paper name identifies a source of ideas; omitted towers, input adapters, samplers and objectives remain the host’s responsibility. Preserve the current compatible backbone and test one mechanism with its matched control. Multiple meaningful feature groups may use independent or shared instances; justify sharing and fusion from the actual fields. Do not force a complete named-model replacement or exhaust a fixed model list. Record mechanism evidence separately from task/dataset-bound business interpretations.

### `ensemble_consistency` — `EnsembleConsistency`

Supervise aggregated branch logits or probabilities and add separately summed branch BCE against a detached ensemble teacher.

**When:** Host can expose the required logits/states/parameters. A learning or gradient observation motivates changing an objective while retaining the backbone.

**Input contract:** branch_logits, binary_outcome_labels. See `model_api` for the framework constructor and calls. **Output:** `dict(loss,supervised,consistency,teacher)`.

**Controlled test:** Hold branch architecture and predictions fixed; ablate consistency weight and compare logit versus probability aggregation separately, reporting each loss and calibration. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Teacher is detached; aggregate supervised path remains differentiable. The weight multiplies a SUM of branch means, so branch count changes its scale. Probability averaging uses stable log probabilities; neither aggregation is a causal effect estimator.

**Bundled code:** `models/pytorch/ctr_training.py`, `models/tensorflow/ctr_training.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalNet/src/FinalNet.py).

### `next_behavior_auxiliary` — `NextBehaviorObjective`

Score aligned state-positive and state-negative pairs with a shared MLP; average BCE over both classes and valid transitions only.

**When:** Host can expose the required logits/states/parameters. A learning or gradient observation motivates changing an objective while retaining the backbone.

**Input contract:** aligned_sequence_states, positive_negative_next_events, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict(loss,positive_logits,negative_logits,valid_count)`.

**Controlled test:** Keep main model and sampler fixed; compare main loss only with a weighted auxiliary term, then test negative strategy separately. Audit false negatives and future leakage. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Host aligns state t with event t+1 and supplies time-safe negatives. Inputs are already shifted; this module does not shift a second time. Mask is applied to unreduced positive and negative losses. Empty/all-masked transitions give differentiable zero. Host must expose sequence states from the candidate.

**Bundled code:** `models/pytorch/ctr_training.py`, `models/tensorflow/ctr_training.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DIEN/src/DIEN.py).

### `coarse_fine_supervision` — `CoarseFineObjective`

Apply independently weighted binary supervision to coarse and fine response logits for the same observed target.

**When:** Host can expose the required logits/states/parameters. A learning or gradient observation motivates changing an objective while retaining the backbone.

**Input contract:** coarse_fine_logits, binary_outcome_labels. See `model_api` for the framework constructor and calls. **Output:** `dict(loss,coarse,fine)`.

**Controlled test:** Keep selector, fine head and retained history fixed; isolate coarse loss weight and measure retrieval coverage alongside final task quality. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Host builds both prediction heads. Retrieval scores alone are not response logits. Does not make hard top-k differentiable; report which selection parameters receive gradients.

**Bundled code:** `models/pytorch/ctr_training.py`, `models/tensorflow/ctr_training.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SIM/SIM.py).

### `grouped_regularization` — `GroupedRegularization`

Add explicit lambda/p times the sum of absolute parameter values to power p, separately for embedding and network groups.

**When:** Host can expose the required logits/states/parameters. A learning or gradient observation motivates changing an objective while retaining the backbone.

**Input contract:** parameter_groups. See `model_api` for the framework constructor and calls. **Output:** `dict(loss,embedding,network)`.

**Controlled test:** Keep optimizer and architecture fixed; vary one regularization group at a time, recording loss contribution, gradient scale and slice support. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** This is an explicit loss penalty, not decoupled AdamW. Host supplies disjoint parameter objects and chooses whether to exclude bias and normalization. Never double-count the same penalty in optimizer weight decay; group identity checks do not detect storage aliases.

**Bundled code:** `models/pytorch/ctr_training.py`, `models/tensorflow/ctr_training.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/models/rank_model.py).
