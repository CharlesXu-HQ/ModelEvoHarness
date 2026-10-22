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

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/3.sequence.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/3.sequence.html), [src/funrec/models/din.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/din.py), [src/funrec/models/dien.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/dien.py).

## Reco-papers component map

These entries are independent computational components, not full training systems. Read each exact signature and limitation through `model_api` or `read_references`; use the same input tensors and evaluator when comparing them.

- `latent_cross` / `LatentCross`: Residual modulation `hidden * (1 + projected_context)` preserves the hidden state at zero projected context. Output: `modulated_hidden_state`, `same as hidden input`. Tensor interaction only; host owns recurrent encoder, context encoding and padding. No sequence construction.
- `transact` / `TransAct`: Candidate-conditioned action Transformer (`history, candidate, mask`) followed by concatenated recent token outputs and masked max pooling. Output: `ranking_representation`, `[B,(K+1)*D]`. Compact real-time action encoder. No random time-window augmentation, batch PinnerFormer fusion or production serving stack.
- `dicm` / `DICM`: Independent image-query and ID-query attention channels pool the same behavior-image history. Output: `ranking_representation`, `[B,2D]`. DICM rank-stage pooling only; compact scoring MLP choice is declared. No image descriptor/AMS; candidate-dependent output is not an independent retrieval tower.
- `mmrec` / `MMRec`: Four separately normalized text/image candidate-history attention streams, summed before joint candidate scoring. Output: `ranking_score`, `[B]`. Exact attention/scoring head over supplied embeddings; excludes ViLBERT, image ROI extraction and multimodal pretraining.

The machine-readable paper inventory links these methods to their PDF, family, structural pattern and training objective. Explicitly declared capabilities only establish input availability; they do not establish that an experiment will improve the target task.
