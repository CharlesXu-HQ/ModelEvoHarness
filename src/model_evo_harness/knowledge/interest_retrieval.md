# Interest and sequence retrieval

A single user vector can average incompatible interests. MIND routes behavior into multiple interest vectors; a sequence encoder such as SASRec or SDM retains order and recency; NARM emphasizes the current session's short-term intent. These mechanisms address distinct residuals: multiple clusters, temporal transitions, and session-local shifts.

## Data and conditions

Require ordered events with reliable timestamps, item IDs, sequence truncation at each decision, and a catalog. Session methods need defensible session boundaries. Repeated events, delayed event ingestion, and action-triggered observations must be handled consistently. A coupon-only table with no item or event history does not satisfy this contract.

## Controlled experiment

First compare mean pooling and recency-weighted pooling under the same candidate universe. Then vary only one mechanism: attention/order, multiple vectors, or session encoder. Keep maximum history length and total retrieved K fixed; multi-vector recall must be deduplicated fairly. Add a shuffled-order control for order claims. Report short-history and dense-history slices, latency, and catalog coverage.

## Interpretation and feature needs

A gain from order-sensitive models over shuffled-order controls is evidence for temporal structure. If event order is known to matter operationally but timestamps are absent, request timestamped history immediately. Otherwise, repeated failures across available history encoders and slice diagnostics should precede a request for richer behavior attributes.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| MIND | Compact routing creates multiple user-interest vectors. | Compare one versus several vectors at the same total retrieved K. |
| ComiRec-SA (`comirec_sa`) | Produces candidate-independent interests with attention over time. | Compare K=1/K>1 and position on/off with MIND and masked mean. |
| ComiRec-DR (`comirec_dr`) | Routes position-specific projected votes into squashed interest capsules. | Compare with SA and MIND at matched parameter and retrieval budgets. |
| SASRec | Causal self-attention encodes ordered item prefixes. | Compare with masked mean pooling and an order-shuffled control. |
| SDM | Combines short-term sequence signal with pooled long-term behavior. | Ablate the long or short branch on stable horizon slices. |
| NARM | Uses global session context plus attention to current intent. | Compare with recency pooling on sessions with defensible boundaries. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_1_retrieval/4.sequence/1.mind.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/4.sequence/1.mind.html), [docs/chapter_1_retrieval/4.sequence/2.sdm.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/4.sequence/2.sdm.html), [src/funrec/models/mind.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/mind.py), [src/funrec/models/sdm.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/sdm.py).

Additional pinned source for `narm`: [torch_rechub/models/matching/narm.py](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/matching/narm.py).

## Reco-papers component map

These entries are independent computational components, not full training systems. Read each exact signature and limitation through `model_api` or `read_references`; use the same input tensors and evaluator when comparing them.

- `pinnerformer` / `PinnerFormer`: Causal user-action encoder with normalized position representations and a dense future-positive contrastive objective. Output: `user_representation`, `[B,D]` (`encode()` returns `[B,T,D]`). Host supplies future positives, negatives and valid prefix mask; no K-day target sampling, timestamp validation or full Pinterest feature encoder.

The machine-readable paper inventory links these methods to their PDF, family, structural pattern and training objective. Explicitly declared capabilities only establish input availability; they do not establish that an experiment will improve the target task.


## Native ComiRec-SA and ComiRec-DR cores

Both native `models/{pytorch,tensorflow}/multi_interest.py` modules accept only
`history[B,T,D]` and a boolean True=valid mask `[B,T]`. They return a dictionary:
`interests[B,K,D]`, `interest_mask[B,K]`, and `assignments[B,K,T]`. No candidate or
future target is needed to encode a user. Padding, including masked NaNs, is
sanitized; empty histories return zero vectors/assignments and false masks.
`interest_mask` indicates history presence for every slot, not distinct cluster
occupancy. Measure interest similarity and effective usage separately.

`ComiRecSA(dim, num_interests, hidden_dim=None, max_length=None)` uses a bias-free
projection, tanh, and K attention scores normalized over time. The hidden width
defaults to `dim`; supplying `max_length` adds trainable position embeddings only
to the score input. The weighted values remain the original history. This
score/value distinction makes position on/off a controlled local test.

`ComiRecDR(dim, num_interests, max_length, routing_iterations=3,
stop_gradient_routing=True)` learns a `[max_length,K,D,D]` projection. Each event's
routing assignments normalize over interests, then projected votes sum over
history and undergo capsule squash. Nonfinal votes and agreement updates stop
gradients by default; the final sum uses live votes. Setting the option false
permits gradients through all iterations. Zero initial routing logits are an
explicit deterministic variation from the author's random initialization.
Position-specific projection and routing differ from the bundled compact MIND;
report their parameter and maximum-history-length costs.

Standalone method screening stays in the `interest_retrieval` family and needs
real `event_sequence` and `item_catalog` capabilities. For a backbone with real
history, the interests can also be intermediate tokens or feed a separate
candidate-aware readout. Declare this as a local composition and ablate its
fusion; it is not the original retrieval pipeline. The host owns time-safe
embedding construction, positive targets, loss and negative sampling, training
readout, catalog search and candidate deduplication. The encoder does not bundle
ComiRec's category-penalty diversity aggregator.

Use masked mean, MIND, SA(K=1), SA(K>1), and DR as hypotheses selected by observed
residuals. Keep embeddings, time cuts, training budget and total deduplicated
candidate count fixed; match or report parameter and retrieval costs. Separate
SA positions, DR projections/routing and downstream readout. Report Recall/NDCG,
short/diverse-history slices and latency. More ANN queries cannot by themselves
establish a better encoder. These are independent tensor implementations, with
no task-performance claim. Sources: [ComiRec](https://arxiv.org/html/2005.09347v2)
and [pinned author implementation](https://github.com/THUDM/ComiRec/blob/7e605f1daabe1ab48e4e10469f4367429bce86c9/src/model.py).
