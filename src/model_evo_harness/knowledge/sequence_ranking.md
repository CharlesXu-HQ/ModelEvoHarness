# Behavior sequence ranking

DIN attends to historical events conditioned on the candidate; DIEN adds interest evolution; DSIN introduces session structure. Candidate-conditioned attention asks which past actions matter for this target, unlike a single pooled user vector. The extra recurrent/session layers are justified only if time order or session changes add repeatable signal.

## Data and conditions

Each training example needs a candidate and a history cut strictly before its decision timestamp. Define event type, history length, masking, deduplication, session boundaries, and identity joins. Serving must be able to construct the same history at similar latency. For treatment experiments, avoid using behavior generated after coupon assignment as a baseline covariate.

## Controlled experiment

Compare static aggregate history, masked mean pooling, candidate attention, and then one temporal/session extension. Use identical events and sequence length, with an order-shuffled control for evolution claims. Report calibration and business value by history density and recency, plus memory and serving latency.

## Interpretation and feature needs

A DIN gain with no DIEN gain suggests relevance to the candidate but weak incremental order signal. A sequence gain that disappears under honest time cutoffs indicates leakage. When the product domain clearly requires event history but the dataset has only current state, request event logs directly; otherwise exhaust simpler history representations before claiming new behavioral fields are required.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| DIN | Candidate-conditioned attention selects relevant past behavior. | Compare with time-masked mean pooling on identical histories. |
| DIEN | Models behavior-state evolution before candidate activation. | Ablate evolution and auxiliary behavior loss separately. |
| DSIN | Encodes within-session and between-session behavior. | Compare with DIN only when reliable session boundaries exist. |

## State-conditioned gates and pooling

Use `unidot_conditioned_gate` to test whether the candidate should select history content through the gate while leaving the value projection history-only. The API takes existing history embeddings, a per-sample condition and a bool valid-position mask. Use `unidot_multichannel_pool` when a fixed history average may lose state-dependent evidence: current field tokens condition independent sigmoid channels for each supplied domain. It returns a fixed set of domain/channel vectors for a host-defined downstream block. Domain count is explicit, not fixed to the competition's four domains.

For both components, True means valid and padded values are sanitized before arithmetic. Pool outputs for entirely absent domains are zero even after affine transforms; preserve domain/channel order and masks when combining them. The host must supply real, time-safe behavior histories; static field groups do not establish this capability. The conditioned representations depend on the current candidate or state and cannot be reused across changed conditions without recomputation.

First compare the gate with the same sequence encoder without conditioning, or pool with a masked mean on the same histories. Fix token counts, downstream head and training budget. Then isolate channel count and conditioning with matched cost controls. Reject improvements that depend on future events, unknown field semantics or unacceptable latency. Tensor shape and gradient tests establish implementation behavior, not task benefit.

The source is the [UniDot author version](https://github.com/linrongc/kdd26_unidot/tree/1399702006a1fc2f86329c8c6133f3f3ac1e6d0d); the extracted modules carry its MIT notice. They omit temporal variants, auxiliary training recipes and embedding/data construction. The manifest's output contracts describe representations, so the host supplies a prediction head and task loss.

## QueryFormer history retrieval and query generation

Use `queryformer_history_retrieval` when a masked average loses item-relevant domain evidence: `QuerySeqCrossAttn` aggregates item tokens by a masked mean and retrieves one vector per history domain. Use `queryformer_query_generator` when a host sequence block already consumes probes but projection-only queries lose context: `SeqQueryCrossAttn` initializes mean/max probes from each history and attends to supplied context tokens as keys/values. `queryformer_query_bridge` composes these with field interaction and returns probes plus context ordered as updated users, updated items, retrieved domains. All three require genuine `event_sequence`, `domain_histories`, `sequence_mask` and their declared item/context/user inputs.

Every mask uses True=valid. Padding is sanitized before arithmetic, including padded NaNs; domain absence is carried by explicit output masks. Empty history domains return zero probes, and query generation also requires nonempty context. Pooling and attention operate per example, with fixed domain count/order and variable history lengths. Preserve masks when connecting the output to the host. Candidate/context changes invalidate cached conditioned representations.

First compare history retrieval with masked mean pooling, and query generation with a projection-only control using identical history summaries and context. Ablate field interaction, retrieval and query generation separately before attributing a composed gain. Check finite gradients, invariance to padded content, probe diversity and history-density slices alongside held-out metrics, parameters, latency and memory. DIN provides a simpler target-aware history summary; UniDot gates or pools history with state-conditioned sigmoid channels. These comparisons test different mechanisms and still need the same data, head and budget.

The independent implementation follows the query-generation portion of [QueryFormer section 4.2](https://arxiv.org/html/2609.16548v1#S4.SS2), with exact constructors and engineering boundaries available through `model_api` and whole-module `read_references`. It covers single-view equations 2–4 only. The host supplies downstream sequence encoding/mixing/readout, a prediction head and training; H>1 views and packed shared-parameter execution are unavailable. Mean/max initialization is direct, with no extra probe projection before attention; per-domain parameters, Co-Transformer residual post-LayerNorm, biases and empty-input behavior are declared in the source. Query generation adds no residual or LayerNorm after attention. The author repository/license is unconfirmed and no third-party code was copied.

## Context-rich events and separate relation normalization

`contextual_history_encoding` fuses aligned per-event item and context features
before the downstream sequence encoder. Use it when the dataset already contains
pre-decision event attributes that current tokens omit; compare with item-only
and context-shuffled controls. It preserves order but adds no chronology, codebook
or data. `relation_history_attention` separately normalizes user-to-history,
target-to-history and target-to-user attention, with disjoint learned projections.
Compare this local branch with joint pooling and ablate relations independently;
keep the same task head and training budget. Masked NaNs are sanitized and absent
relations remain zero with false availability. Candidate targets do not attend
to one another. These original components draw on
[SIF](https://arxiv.org/html/2604.15650v3) and
[CCFormer](https://arxiv.org/html/2607.28070v2); they omit SIF quantization and the
CCFormer hierarchical compression pipeline. Event context must exist at the
historical cutoff; outcome or future exposure information is not a valid input.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/3.sequence.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/3.sequence.html), [src/funrec/models/din.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/din.py), [src/funrec/models/dien.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/dien.py).

## Reco-papers component map

These entries are independent computational components, not full training systems. Read each exact signature and limitation through `model_api` or `read_references`; use the same input tensors and evaluator when comparing them.

- `latent_cross` / `LatentCross`: Residual modulation `hidden * (1 + projected_context)` preserves the hidden state at zero projected context. Output: `modulated_hidden_state`, `same as hidden input`. Tensor interaction only; host owns recurrent encoder, context encoding and padding. No sequence construction.
- `transact` / `TransAct`: Candidate-conditioned action Transformer (`history, candidate, mask`) followed by concatenated recent token outputs and masked max pooling. Output: `ranking_representation`, `[B,(K+1)*D]`. Compact real-time action encoder. No random time-window augmentation, batch PinnerFormer fusion or production serving stack.
- `dicm` / `DICM`: Independent image-query and ID-query attention channels pool the same behavior-image history. Output: `ranking_representation`, `[B,2D]`. DICM rank-stage pooling only; compact scoring MLP choice is declared. No image descriptor/AMS; candidate-dependent output is not an independent retrieval tower.
- `mmrec` / `MMRec`: Four separately normalized text/image candidate-history attention streams, summed before joint candidate scoring. Output: `ranking_score`, `[B]`. Exact attention/scoring head over supplied embeddings; excludes ViLBERT, image ROI extraction and multimodal pretraining.

The machine-readable paper inventory links these methods to their PDF, family, structural pattern and training objective. Explicitly declared capabilities only establish input availability; they do not establish that an experiment will improve the target task.

## FuxiCTR mechanism components

These are independently implemented, composable PyTorch and TensorFlow mechanisms. Read the full local module and output contract before editing the current candidate. A paper name identifies a source of ideas; omitted towers, input adapters, samplers and objectives remain the host’s responsibility. Preserve the current compatible backbone and test one mechanism with its matched control. Multiple meaningful feature groups may use independent or shared instances; justify sharing and fusion from the actual fields. Do not force a complete named-model replacement or exhaust a fixed model list. Record mechanism evidence separately from task/dataset-bound business interpretations.

### `bst_history_encoder` — `BSTHistoryEncoder`

Append the candidate token to history, add learned positions, then apply bidirectional multihead self-attention and residual feed-forward refinement. Expose transformed history and candidate readout.

**When:** Real ordered histories and candidate features are available. Test candidate inclusion, causal refinement or separate user/item matching rather than treating numeric columns as events.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** Keep the same events, embedding width and training budget; compare candidate-in-encoder with history-only encoding followed by target attention. Test positional contribution separately. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** One Transformer block with target-token pooling; excludes source concat/mean/sum readouts and causal-mask option. No-history target readout is deliberately zero rather than a target-only prediction; retain target-only information in a host branch. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_matching.py`, `models/tensorflow/history_matching.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/BST/src/BST.py).

### `dmin_interest_refiner` — `DMINInterestRefiner`

Causally refine events, compute a second head-split self-attention, expand each head separately, then use independent target/position-conditioned DIN-style attentions to form multiple interests.

**When:** Real ordered histories and candidate features are available. Test candidate inclusion, causal refinement or separate user/item matching rather than treating numeric columns as events.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** Compare one head and multiple heads at matched budget, then ablate the causal behavior refiner while retaining target attentions. Auxiliary transition loss is a separate experiment. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Independent two-normalization refiner and D-dimensional positional context; no dropout or upstream exact hidden-width choices. The host constructs auxiliary next-event/negative supervision; intermediate refined/head states are exposed and remain event-aligned. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_matching.py`, `models/tensorflow/history_matching.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMIN/src/DMIN.py).

### `dmr_history_matching` — `DMRHistoryMatching`

Context/position-conditioned scores form causal prefix user states; their final valid state matches the target by a dot product. A separate target/context-conditioned item path returns both softmax history pooling and the sum of raw relevance scores.

**When:** Real ordered histories and candidate features are available. Test candidate inclusion, causal refinement or separate user/item matching rather than treating numeric columns as events.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** On aligned events and context, compare user path only, item path only and both. Keep the loss fixed, then test next-event user-state supervision separately. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Optional side context must be event-aligned; omission supplies zero side context, not fabricated business features. Separate input/output item embedding tables, calibrated heads and next-event objective are host-owned. Uses the last valid prefix rather than the last padded array slot. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_matching.py`, `models/tensorflow/history_matching.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMR/src/DMR.py).

### `sim_history_selector` — `SIMHistorySelector`

Learned target/history dot scores create an unnormalized coarse weighted sum, including negative contributions; valid-only top-K retrieval feeds a separate learned target-attention fine readout.

**When:** Measured history cost or signal dilution motivates selection. Compare exact scoring, recency and random-at-equal-k controls; measure useful-event coverage and latency as well as response.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** Compare the same fine attention under random/recency/learned selection at fixed K; then add coarse-head supervision without changing selectors or fine structure. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Hard indices are nondifferentiable; the coarse representation and scores expose a trainable route for selector supervision. Masks rank invalid events below every finite valid score, correcting source padding that can beat negative scores. Coarse/fine heads, losses and short-history branch remain host-owned. Dense O(L) score scan; fixed K outputs, stable ties favor earlier event indices. Invalid indices are -1. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_retrieval.py`, `models/tensorflow/history_retrieval.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SIM/SIM.py).

### `eta_history_selector` — `ETAHistorySelector`

Fixed checkpointed random-hyperplane signs yield Hamming distance from target; retrieve nearest K valid history events and apply learned target attention.

**When:** Measured history cost or signal dilution motivates selection. Compare exact scoring, recency and random-at-equal-k controls; measure useful-event coverage and latency as well as response.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** Fix K, history and fine readout; compare Hamming retrieval with learned similarity, recency and random eligible retrieval. Treat hash seed as an experimental seed. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Dense Hamming scan, not an ANN serving index. Hash bits are discrete and non-trainable; no stochastic per-batch rehashing. Stable ties use original order; fixed K invalid slots are -1/False/zero. Hash distance reports zero at invalid events and must be interpreted with the input valid mask. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_retrieval.py`, `models/tensorflow/history_retrieval.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/ETA/ETA.py).

### `sdim_collision_pooling` — `SDIMCollisionPooling`

Sum history embeddings whose full random-hyperplane bit code equals the target code independently for each hash; average the hash-specific sums.

**When:** Measured history cost or signal dilution motivates selection. Compare exact scoring, recency and random-at-equal-k controls; measure useful-event coverage and latency as well as response.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** At fixed history and projection seed, compare exact collision pooling with top-K Hamming selection and masked mean; measure empty-bucket rate and cost. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Preserves the unnormalized source option; no optional per-hash L2 normalization. Collision sums are not event means or learned attention. No trainable parameters inside this component: gradients flow to collided history embeddings; discrete bucket membership does not train target/hash rotations. No ANN index. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_retrieval.py`, `models/tensorflow/history_retrieval.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SDIM/SDIM.py).

### `twin_headwise_selector` — `TWINHeadwiseSelector`

Each learned query/key head chooses its own top-K history events; exactly those scores normalize corresponding value heads, then concatenated head outputs are projected.

**When:** Measured history cost or signal dilution motivates selection. Compare exact scoring, recency and random-at-equal-k controls; measure useful-event coverage and latency as well as response.

**Input contract:** event_sequence, target_item_features, sequence_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** Compare headwise selections against a common shared-K selection with matched q/k/v width, then ablate retrieval/fine-score consistency. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Dense search, without production ANN caches or the full short/long dual-branch ranker. Hard selection indices are nondifferentiable. Fixed K per head; every head reports original event indices, masks and weights. Invalid indices are -1, and all-masked heads remain zero. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_retrieval.py`, `models/tensorflow/history_retrieval.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/TWIN/TWIN.py).

### `mirrn_history_refiner` — `MIRRNHistoryRefiner`

Target, masked recent-history mean and global-history mean independently query fixed LSH top-K; original indices are restored to temporal order. Each selected stream receives masked complex block-channel FFT refinement, pooled interests and final target attention.

**When:** Measured history cost or signal dilution motivates selection. Compare exact scoring, recency and random-at-equal-k controls; measure useful-event coverage and latency as well as response.

**Input contract:** event_sequence, target_item_features, sequence_mask, recent_history_mask. See `model_api` for the framework constructor and calls. **Output:** `dict`.

**Controlled test:** Compare target-only with three selector queries, then compare identity/no-filter control against frequency refinement at identical selected indices. Keep K, hashes and serving cost recorded. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Corrects unused selection masks and diagonal-only einsum: full off-diagonal channel mixing is deliberate and independently implemented. Orthogonal rFFT/irFFT with explicit n=K supports odd K; complex matrices are shared across frequencies. Zero-padded invalid slots are part of the declared K-point transform, masked both before FFT and after residual LayerNorm. Omits upstream recency-position additions, dropout and short-history attention branch. Recent mask must be a subset of valid; missing recent history produces no recent interest. Dense hash scans, no ANN cache. Fixed K with chronological valid selections and invalid -1 slots. Host supplies time-safe embeddings, ordered histories, prediction heads, task loss and evaluator; this is an independent mechanism reference, not a complete production clone. True=valid; invalid history values are sanitized, all-masked/empty readouts are zero. No training experiment or effectiveness claim.

**Bundled code:** `models/pytorch/history_retrieval.py`, `models/tensorflow/history_retrieval.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/MIRRN/MIRRN.py).
