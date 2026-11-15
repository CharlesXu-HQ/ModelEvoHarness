# Native CTR refinements

Original PyTorch and TensorFlow components informed by the pinned DeepCTR and
DeepCTR-Torch sources. They use only their own framework and Python's standard
library. No upstream model library, trainer or feature-column framework is needed.
The host keeps its backbone, data adapter, optimizer and evaluator.

## Component map

The relative paths below exist under both `src/model_evo_harness/models/pytorch/`
and `src/model_evo_harness/models/tensorflow/`.

| Method / existing component | API or explicit option | Role |
| --- | --- | --- |
| `crossnet_mix` | `interaction_variants.CrossNetMix(input_dim, rank=8, num_experts=4, num_layers=2)` | Nonlinear low-rank cross experts; `[B,D]` representation and optional routing. |
| `ifm` | `interaction_variants.InputAwareFM(num_fields, embedding_dim, hidden_dims=(32,))` | One field factor jointly weights first-order terms and FM embeddings. |
| `difm` | `interaction_variants.DualInputAwareFM(num_fields, embedding_dim, hidden_dims=(32,), num_heads=2)` | Attention and MLP field factors; no final IFM softmax. |
| `dice_activation` | `sequence_attention.MaskedDice(dim, momentum=.9, epsilon=1e-5)` | Activation with valid-position moments and frozen evaluation statistics. |
| `local_activation_pooling` | `sequence_attention.LocalActivationPooling(dim, hidden_dim, activation='relu', normalization='sigmoid')` | Returns representation, attention and scores; `dice`, `raw`, `softmax` are explicit alternatives. |
| `masked_sequence_pooling` | `sequence_inputs.MaskedSequencePooling(mode='mean', weight_normalization='none', require_prefix=False)` | Validated sum/mean/max with optional weights; zero empty output. |
| `onn_operations` | `output_mode='pair_vectors'` | Preserve `[B,pairs,D]`; default `copy_and_dots` stays ONNv2-style. |
| `fibinet` | `bilinear_type='all'/'field'/'pair'`, `separate_branches=True` | Isolate parameter sharing; defaults preserve previous shared behavior. |
| `field_matrix_interaction` | `symmetric=True` | Constrain matrix mode to `W+W.T`; vector mode remains separate. |
| `dien` | `evolution='gru'/'aigru'/'agru'/'augru'` | Interest evolution control; default AUGRU retained. |
| `dsin` | `bias_encoding=True`, `max_sessions`, `max_session_len` | Learned session, position and coordinate biases. |
| `bst_history_encoder` | `readout='history_attention'`, injected `pooling` | History-only Transformer followed by candidate pooling; default candidate-token path retained. |

These are mechanism references with deliberate parameterizations, not exact
upstream model reproductions. In particular the local AUGRU reset convention and
FiBiNET squeeze activation remain documented variants. Raw attention is signed;
attention/gating weights do not measure causal effects.

## Native DIEN training composition

PyTorch:

```python
from model_evo_harness.models.pytorch.sequence import DIEN
from model_evo_harness.models.pytorch.ctr_training import NextBehaviorObjective

model = DIEN(dim=16, hidden_dim=32,
             auxiliary_objective=NextBehaviorObjective(16, 16, 32))
details = model(target, history, valid, return_details=True)
aux = model.auxiliary_next_event_loss(
    history, valid, negative_history, interests=details['interests'])
main = torch.nn.functional.binary_cross_entropy_with_logits(details['logit'], labels)
loss = main + alpha * aux['loss']
loss.backward()
```

TensorFlow:

```python
from model_evo_harness.models.tensorflow.sequence import DIEN
from model_evo_harness.models.tensorflow.ctr_training import NextBehaviorObjective

model = DIEN(dim=16, hidden_dim=32,
             auxiliary_objective=NextBehaviorObjective(16, 16, 32))
with tf.GradientTape() as tape:
    details = model(target, history, valid, return_details=True, training=True)
    aux = model.auxiliary_next_event_loss(
        history, valid, negative_history, interests=details['interests'])
    main = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=details['logit']))
    loss = main + alpha * aux['loss']
grads = tape.gradient(loss, model.trainable_variables)
```

The host imports its framework, creates tensors/optimizer and chooses `alpha`.
`negative_history` is **unshifted** `[B,T,D]`: negative at position t corresponds
to the event at position t. The method pairs state t with event t+1 internally.
Use only pre-decision historical events and legal training negatives. A valid
transition needs both adjacent positions. Auxiliary BCE averages across positive
and negative classes and valid transitions; zero transitions return zero.
Compare `alpha=0` against a positive alpha with all structure and sampling fixed.

`GroupedRegularization` retains `weight/p * sum(abs(w)**p)`, including the one-half
factor for p=2. Record main reduction, valid transition count, regularization and
auxiliary weights separately. Different wide/deep optimizers remain a host recipe
with explicit nonoverlapping parameter groups, learning rates and update cadence.

## Input contracts

DIEN requires Boolean contiguous-prefix masks. DIN accepts arbitrary Boolean
masks. Both sanitize masked padding before arithmetic and reject nonfinite valid
values. DIEN's empty extracted states are zero; final logits can include the
prediction head's bias and target-only contribution.

`sequence_inputs.validate_sequence_mask` checks shape, Boolean masks, integer
lengths and count agreement; length-only input creates prefixes. An optional
`max_length` bounds the padded time axis. `validate_aligned_sequences` checks
same-event attributes and masks, and compares valid event IDs when supplied.
Without IDs it verifies layout, not actual join correctness.

For weighted pooling, `none` means nonnegative raw weights; mean divides by their
sum. `softmax` means the supplied weights are logits normalized across valid
positions. Max is the coordinatewise maximum of weighted valid vectors. Empty
support is zero. The schema never infers these meanings from field names.

Optional `embedding_roles[].sequence_contract` declares `max_length`,
`mask_layout`, `pooling`, `weight_normalization`, `alignment_group` and task fields
for masks/lengths/event IDs. Separate histories can have different alignment
groups. `encoding_transform` participates in shared embedding identity alongside
ID domain, vocabulary digest, width and padding/OOV semantics. Old role records
without the optional fields remain valid.

## Agent use and validation boundary

Six new method cards have dual-framework source contracts and applicable
structure patterns. Existing cards describe the new options and their controlled
comparisons. Selecting DIEN also delivers attention, pooling and auxiliary loss
source modules through `companion_files`; selected sources retain hashes in the
existing reference ledger. Each extracted module remains independently importable.
The packaged [guide](../src/model_evo_harness/knowledge/native_ctr_refinements.md)
is included in common Agent knowledge, including input checks, loss scaling,
method applicability, ablation controls and rejection conditions.

Implementation tests verify formulas, gradients, padding invariance, state and
source delivery. Real-data improvement and Agent exploration efficiency require
separate host experiments; no dataset-level gain follows from these checks.

## Sources and original implementation boundaries

- [DeepCTR](https://github.com/shenweichen/DeepCTR/tree/1b5fe40e158d1ee6af8b1d9df217a5ed5aea9136): fixed source for CrossNetMix, IFM/DIFM, Dice/local activation, ONN vectors, FiBiNET sharing, FEFM symmetry and DSIN bias.
- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch/tree/9eccef78e5810c61d6a04ddc7d96279e3db9c970): independent comparison of interaction formulas, DIEN extraction/evolution and loss/regularization conventions.

No upstream code, data, weights or model-wrapper dependencies are incorporated
in these independent additions. Existing adapted components retain their own
notices. PyTorch and TensorFlow implementations are kept in separate directories.

See [verification evidence](research/native-ctr-validation-2026-11-15.md) for the full local suite and 122 GPU cases. DSIN compacts valid sessions before its bidirectional recurrence, preserving original bias indices and safely handling empty users.
