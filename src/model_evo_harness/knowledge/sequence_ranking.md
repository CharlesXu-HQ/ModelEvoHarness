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

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/3.sequence.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/3.sequence.html), [src/funrec/models/din.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/din.py), [src/funrec/models/dien.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/dien.py).

## Reco-papers component map

These entries are independent computational components, not full training systems. Read each exact signature and limitation through `model_api` or `read_references`; use the same input tensors and evaluator when comparing them.

- `latent_cross` / `LatentCross`: Residual modulation `hidden * (1 + projected_context)` preserves the hidden state at zero projected context. Output: `modulated_hidden_state`, `same as hidden input`. Tensor interaction only; host owns recurrent encoder, context encoding and padding. No sequence construction.
- `transact` / `TransAct`: Candidate-conditioned action Transformer (`history, candidate, mask`) followed by concatenated recent token outputs and masked max pooling. Output: `ranking_representation`, `[B,(K+1)*D]`. Compact real-time action encoder. No random time-window augmentation, batch PinnerFormer fusion or production serving stack.
- `dicm` / `DICM`: Independent image-query and ID-query attention channels pool the same behavior-image history. Output: `ranking_representation`, `[B,2D]`. DICM rank-stage pooling only; compact scoring MLP choice is declared. No image descriptor/AMS; candidate-dependent output is not an independent retrieval tower.
- `mmrec` / `MMRec`: Four separately normalized text/image candidate-history attention streams, summed before joint candidate scoring. Output: `ranking_score`, `[B]`. Exact attention/scoring head over supplied embeddings; excludes ViLBERT, image ROI extraction and multimodal pretraining.

The machine-readable paper inventory links these methods to their PDF, family, structural pattern and training objective. Explicitly declared capabilities only establish input availability; they do not establish that an experiment will improve the target task.
