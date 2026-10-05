# Structure hypotheses beyond the current model cards

This guide describes **candidate mechanisms for an Agent to implement and test**, not models already provided by ModelEvoHarness. The current 44 method cards and their PyTorch/TensorFlow references do not include the structures below. A source name is orientation, not a drop-in API or a claim of paper reproduction. If a host has a valid task contract, the Agent may write an original implementation directly in PyTorch or TensorFlow, add a matched control, and run it through that host's trainer. It must record the exact variant and training recipe it actually tested.

## Contract before selecting a new structure

Identify the target (CTR, conversion, next item, or value), unit of evaluation, feature availability at decision time, typed inputs, label horizon, loss, sampler, output meaning, and fixed validation protocol. A richer network is a reasonable hypothesis only when an observed residual or a domain requirement suggests the corresponding missing *mechanism*. Keep preprocessing, split, candidate universe, training budget and primary metric fixed while changing one mechanism. Inspect supported slices, calibration, compute and serving cost. If a model cannot be trained or scored by the host, record `needs_contract` rather than claiming an experiment. See [feature interaction](feature_interactions.md), [sequence ranking](sequence_ranking.md), [interest retrieval](interest_retrieval.md), [training objectives](training_objectives.md), and [feature-gap decisions](feature_gap_decisions.md).

## Input-conditioned feature importance: IFM and DIFM

**Computation.** FM uses the same feature embedding in every example. IFM generates a sample-dependent importance weight from the current field embeddings and applies it to each active feature before the linear/pair interaction terms. DIFM adds two weight paths: one acts on individual embedding coordinates (bit-wise), the other on whole field vectors (vector-wise). This asks whether the *same* field should matter differently for different contexts, beyond static field-pair weights. It is distinct from FwFM's field-pair scalar and FiBiNET's particular squeeze/excitation plus bilinear path.

**When and inputs.** Try this when residuals suggest context-dependent field relevance, such as coupon price band mattering differently across recent purchase states, while the required context fields already exist. Supply typed sparse/dense fields, consistent embedding dimensions, missing-value handling, and enough support within context slices. All values must precede the decision. Compare IFM with FM or FiBiNET at the same fields, then test DIFM's coordinate path against its field-only path with a parameter/latency budget.

**Falsification.** If gains vanish when capacity is matched, weights are unstable across seeds, or slice errors do not improve where relevance was hypothesized, do not infer that adaptive field weighting solved the problem. Attention weights alone are not a causal explanation. An absent essential pre-decision context field is a data request, not an invitation to infer it from post-action behavior.

**Code status.** Design guidance only; no local IFM/DIFM method card or implementation. [DeepCTR's model explanations](https://deepctr-doc.readthedocs.io/en/latest/Features.html#ifm-input-aware-factorization-machine) describe the two mechanisms.

## Field- and operation-aware pairs: FFM, DeepFFM, FEFM, ONN

**Computation.** Ordinary FM shares one embedding for feature `i` across all partner fields. FFM instead uses a partner-field-specific vector, so a pair contributes approximately `<v_(i,field(j)), v_(j,field(i))>`. DeepFFM sends these pair representations to a neural head; FatDeepFFM adds learned emphasis over pair representations. FEFM keeps one vector per feature and uses a symmetric interaction matrix for each field pair, `e_iᵀ M_(field(i),field(j)) e_j`. ONN keeps embeddings aware of the operation that consumes them and feeds preserved pairwise interaction information to a deep head. These differ from FwFM's *scalar* field-pair weight and from the current compact FiBiNET's shared bilinear transform.

**When and inputs.** Consider these when supported residuals concentrate in a few field pairs and simpler FM/FwFM/DCN variants underfit those pairs. Require stable field identities, categorical vocabularies, pair support and a bounded number of fields; FFM's embedding count grows with possible partner fields. Compare FFM/FEFM with FM and FwFM under the same embedding/parameter budget. Test the deep head separately from the field-aware interaction so a gain is not credited to the wrong part.

**Falsification.** A lift confined to extremely rare pairs, disappearing under cross-fold support checks, or accompanied by severe calibration/latency regression is weak evidence for richer pair parameterization. Do not invent explicit cross fields from outcomes or aggregates beyond the decision time.

**Code status.** Design guidance only; no local FFM, DeepFFM, FatDeepFFM, FEFM or ONN card/reference. Source mechanisms: [DeepCTR FEFM and ONN descriptions](https://deepctr-doc.readthedocs.io/en/latest/Features.html#deepfefm-deep-field-embedded-factorization-machine) and [Torch-RecHub DeepFFM source](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/ranking/deepffm.py).

## Cross/deep information exchange: EDCN

**Computation.** DCN concatenates an explicit cross path with an implicit deep path at the end. EDCN exchanges information **between layers** through a bridge and regulates field contributions before later cross/deep computation. The hypothesis is that intermediate signals from one path help the other, rather than that another cross layer alone helps. This is also separate from DCNv2's matrix/low-rank cross and DCNMix's expert routing.

**When and inputs.** Try only after a cross branch and an MLP each show complementary, supported residuals on the same typed fields. Hold cross depth, deep width, parameters and step budget as close as possible to a DCN control; ablate bridge and regulation independently. Inspect whether the apparent gain persists by field group and under an inference-cost ceiling.

**Falsification.** If a wider DCN/MLP matches EDCN, the exchange mechanism has not been isolated. If the bridge adds no gain after regulation or vice versa, attribute the result to the working component. Disappearing support in sparse groups is not evidence of a missing business feature.

**Code status.** Design guidance only; no local EDCN card/reference. [DeepCTR description and source](https://deepctr-doc.readthedocs.io/en/latest/Features.html#edcn-enhancing-explicit-and-implicit-feature-interactions-dcn) identify bridge and regulation as discriminating components.

## Field locality and grouping: FGCNN, CCPM, FLEN

**Computation.** FGCNN generates additional feature patterns with convolutions over an ordered field-embedding grid before interaction scoring. CCPM also applies local convolution/pooling to a field sequence. FLEN instead uses field groups to reduce unwanted coupling and capture grouped interaction signals; it does not rely on a CNN field order. These are distinct hypotheses, even though all need a defensible field layout.

**When and inputs.** FGCNN/CCPM require a meaningful and stable field neighborhood (for example, a structured group/order from the data contract), not alphabetical column order. Use a field-order permutation control: a gain that persists under arbitrary permutation does not support a locality story. FLEN needs explicit, stable field groups and enough examples per group; compare with FwFM or grouped FM under matched fields and capacity.

**Falsification.** If no valid field order exists, do not build a CNN simply because it appears in an upstream list. If a random order performs equally well, the effect may be capacity or regularization. If groups are inferred from the validation outcomes, the protocol is contaminated.

**Code status.** Design guidance only; none has a local method card/reference. [DeepCTR model descriptions](https://deepctr-doc.readthedocs.io/en/latest/Features.html#fgcnn-feature-generation-by-convolutional-neural-network) distinguish generated local patterns from field grouping.

## Candidate-aware sequence Transformer: BST

**Computation.** BST encodes past behavior together with the candidate item using self-attention, position/time information and a padding mask, then uses the candidate-position representation for ranking. It tests whether pairwise dependencies *inside* a user's history improve a candidate score. DIN tests candidate-conditioned weighting of history; SASRec chiefly learns a sequential next-item representation. A BST test must preserve its ranking target and candidate path rather than relabel any Transformer block as BST.

**When and inputs.** Require ordered, timestamped pre-decision events, the candidate item's features, a consistent history cutoff, padding mask, and an impression/decision label. Start after mean-pooled history and DIN controls; use the same history length and features. Compare positional encoding with order-shuffled history and report behavior by history length, recency, calibration and serving latency.

**Falsification.** If order shuffling leaves the gain intact, the sequence-dependency hypothesis is unsupported. If only target leakage or a longer history explains the gain, do not credit self-attention. A post-coupon event in the input invalidates the uplift comparison.

**Code status.** Design guidance only; no local BST card/reference. The [Torch-RecHub BST source](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/ranking/bst.py) makes its history, target and padding contracts explicit.

## Sparse, recent and diverse interests: SINE, STAMP, ComiRec

**Computation.** SINE selects a small set of relevant interests from a larger preference space and aggregates behavior toward them; its test is sparse-interest selection, not merely more attention heads. STAMP combines a session representation with a path emphasizing the latest action, testing short-term intent. ComiRec learns multiple interest vectors and selects/diversifies them for retrieval; the relevant output is a set of user vectors and candidate scores, not one scalar user embedding. MIND already supplies a local multi-interest reference, but its fixed-K core does not test all of these selection or diversity mechanisms.

**When and inputs.** SINE/ComiRec need sufficiently long, multi-topic histories and stable item identities; STAMP needs session boundaries and meaningful last-event order. A retrieval adapter must define positive targets, eligible negative pool, training objective, item table, full/fixed-catalog evaluator and serving aggregation across interests. Compare SINE with simple attention and MIND, STAMP with NARM/last-item and session-average controls, and ComiRec with MIND at equal interest count and retrieval budget. A GRU4Rec next-item baseline can help distinguish recurrent order modeling from these interest mechanisms; it is also absent as a dedicated local card.

**Falsification.** If histories are short or single-topic, multi-interest capacity has no sound premise. If STAMP gains survive randomizing the last-event position, its recency hypothesis is not supported. If multi-interest recall rises only because more candidates are retrieved, hold the candidate count fixed. Measure coverage/diversity jointly with relevance and cost.

**Code status.** Design guidance only; no local SINE, STAMP, ComiRec or GRU4Rec method card/reference. Sources: [Torch-RecHub matching inventory](https://github.com/datawhalechina/torch-rechub/blob/main/README.md#-supported-models), [SINE](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/sine.py), [STAMP](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/stamp.py), and [ComiRec](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/comirec.py).

## Generative item tokenizer and decoder boundary

**Computation.** A semantic-token path first maps a versioned item representation into discrete codes, then trains a sequence model to predict code sequences that decode to legal item IDs. Residual quantization, as in an RQ-VAE tokenizer, represents an item by successive codebook choices over remaining representation error. TIGER is an example of training a generator to predict those semantic IDs. HLLM is a different hierarchical language-model route for item/user sequence representation; it should not be treated as a synonym for the tokenizer. The current local HSTU is a sequence block, not an end-to-end tokenizer/generator/evaluator.

**When and inputs.** Require item content/embeddings available at the intended serving time; a frozen, versioned catalog; train-only tokenizer fitting; item-to-code and code-to-item maps; handling for collisions/new items; legal-ID constrained decoding; history/target sequences; and a fixed full-catalog or declared candidate evaluator. Compare first with a strong two-tower or sequence retrieval baseline under matched candidate count, latency and training budget. Ablate tokenizer versus generator changes separately. In a two-action coupon decision, justify why discrete item generation is preferable to a calibrated policy model before spending this budget.

**Falsification.** High token accuracy with low legal-item rate, poor catalog coverage, unstable code assignment, or worse value under equal inference cost does not establish better recommendation. Never score an unconstrained token string as an eligible item. Do not evaluate with a tokenizer fitted on future catalog/test interactions.

**Code status.** The existing `hstu` card/reference does **not** provide RQ-VAE, TIGER or HLLM; these are design guidance only. Source orientation: [Torch-RecHub generative inventory](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/generative/__init__.py), [RQ-VAE quantizer](https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/generative/rqvae.py), and [TIGER integration description](https://github.com/datawhalechina/torch-rechub/issues/149).

## Interpretation and provenance

Record a technical experience (which mechanism, fields, loss/sampler, measured residual, control, support and failure mode) separately from a business experience (what user behavior or coupon response the result suggests, with uncertainty and counterexamples). Repeated failed structures may justify a human feature request only when the missing information is specific and pre-decision. A documented domain requirement can identify an essential absent business field immediately; it still does not make an unrun model experiment successful.

This page is original synthesis, not copied source code or a wrapper around DeepCTR or Torch-RecHub. It was checked against the linked first-party source/docs on 2026-10-05. No effectiveness claim follows from these design hypotheses.
