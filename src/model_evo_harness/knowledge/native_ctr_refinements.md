# Local CTR refinements inside a backbone

## Choose from evidence

Start with input correctness, then change **one mechanism** within the current
backbone. Rank a candidate by measured residuals, available typed inputs and a
matched control. The following mechanisms have no fixed superiority ordering.
Do not replace a two_tower retrieval encoder with candidate-dependent attention:
a target-aware history component changes indexability and serving cost. For a
coupon task these components predict response representations; incrementality
still requires the host's causal evaluation.

## CrossNetMix and field importance

CrossNetMix maps each layer state through low-rank projections V, tanh, C, tanh,
and U for each expert, multiplies by the original input, and mixes cross residuals
with per-example expert softmax weights. It returns a representation for the
existing prediction head. Compare full-matrix crosses, linear low-rank crosses,
one nonlinear expert and multiple experts separately. Track rank, expert count,
parameter count, runtime and gate concentration. Multiple experts are justified
by supported context-dependent errors, not by the method's name.

IFM generates `F * softmax(MLP(flatten(field_tokens)))`. The SAME `[B,F]` factors
reweight supplied first-order field terms and `[B,F,D]` embeddings before FM.
DIFM uses an attention path and a flattened-MLP path, each projected to `[B,F]`,
and sums them without a final IFM-style softmax. The final factors are field-level;
"bit" and "vector" refer to modeling paths, not final coordinate-wise weights.
Hold the existing lookup tables and prediction task fixed; isolate attention and
capacity effects. Generic field gating is a useful overlap control. Neither gates
nor attention establish causal feature importance.

Existing components also expose local variants: ONN preserves un-summed pair
vectors with `output_mode='pair_vectors'`; FiBiNET varies bilinear sharing and
whether its raw/reweighted branches have separate matrices; field-matrix
interaction can impose `W + W.T` symmetry. Reuse the same head when possible and
record changed representation width. These variants do not require a new backbone.

## Sequence boundaries and pooling

DIN supports arbitrary Boolean valid-position masks. DIEN requires a contiguous
valid prefix followed by padding; both frameworks reject internal holes. Invalid
padding is removed before nonlinear/recurrent arithmetic. Values at valid
positions must be finite. Empty rows produce finite neutral history contributions;
a downstream bias or target-only branch can still produce a nonzero logit.

`MaskedSequencePooling` accepts masks, lengths or both; supplied lengths must
agree with the mask. Sum, mean and max are explicit alternatives. With raw
nonnegative weights, weighted mean divides by weight sum, not event count;
softmax-normalized weights are logits over valid positions. All-empty support
returns zero. This is different from automatically interpreting an arbitrary
feature column as a sequence.

For multiple event attributes, position t must describe the same event across
item/category/price/time tensors. Validate masks and, when provided, event IDs.
Each independent history has its own alignment group, length and mask. Shared
candidate/history embeddings require the same ID domain, vocabulary and encoding
transformation; equal embedding width alone is insufficient. Metadata checks
cannot prove actual event identity or pre-decision availability: the host supplies
those declarations and verifies source provenance.

## Local activation and sequence variants

`LocalActivationPooling` scores query/history/difference/product with an MLP and
returns scores, attention weights and the pooled representation. Raw scores may
be negative; sigmoid scores are bounded; masked softmax normalizes across valid
events. Change activation and normalization independently. `MaskedDice` maintains
valid-position statistics for training and running statistics for evaluation,
with a learned coordinate-wise alpha. Dice activation is unrelated to the DICE
interest/conformity disentanglement method.

Inject the pooling component into DIN or history-only BST; modules remain
standalone and the Agent receives their companion sources. BST's default
candidate-token readout changes history encoding by candidate; the optional
history-only encoder followed by target pooling isolates where candidate
conditioning enters. DSIN's optional session/position/coordinate bias adds order
information; ablate it while preserving actual session boundaries.

DIEN exposes GRU/AIGRU/AGRU/AUGRU evolution as alternatives. Hold attention,
initialization protocol, auxiliary weight and sampling fixed when varying
interest evolution. Preserve the documented local reset-gate convention rather
than assuming identical seeds make two libraries equivalent.

## Wire the existing next-event objective

Supply `NextBehaviorObjective` as DIEN's `auxiliary_objective`. `interest_states`
exposes extracted states and `auxiliary_next_event_loss` pairs state t with event
t+1 and the corresponding supplied negative. A valid transition requires both
positions to be valid. Positives and negatives must lie within the training
history cutoff and follow the host's eligibility/sampling rules.

The objective reports `valid_count` and averages BCE across valid positive and
negative examples: `(sum positive BCE + sum negative BCE)/(2*valid_count)`.
No valid transitions gives zero. The host combines:

`total = mean_main_loss + alpha * auxiliary_loss + grouped_regularization`

Compare **alpha=0** with a declared positive alpha while holding model structure
and sampler fixed. Log each term independently. The grouped regularizer retains
`weight/p * sum(abs(parameter)**p)` per disjoint group; p=2 therefore includes a
one-half factor. A coefficient copied from a library using `weight*sum(w**2)`
would have a different strength. Sum/mean reductions, task aggregation, weight
decay and auxiliary coefficients must be recorded before interpreting a gain.
Separate wide/deep optimizers are a host training recipe: declare disjoint parameter
groups, update cadence and learning rates and compare under a fixed budget; the
Harness does not instantiate an upstream trainer.

## Evidence and rejection

Source/formula tests and GPU gradients check component behavior. A successful
real-data trial additionally needs stable training, a held-out primary metric,
uncertainty, supported slices and serving-cost checks. Reject a mechanism when the
matched control performs equally well, improvement disappears across seeds or
required guardrails regress. Keep technical findings tied to task and dataset
identity; propose human feature changes only under the existing evidence policy.
