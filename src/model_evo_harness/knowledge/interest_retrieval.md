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
| MIND | Dynamic routing creates multiple user-interest vectors. | Compare one versus several vectors at the same total retrieved K. |
| SASRec | Causal self-attention encodes ordered item prefixes. | Compare with masked mean pooling and an order-shuffled control. |
| SDM | Combines short-term sequence signal with pooled long-term behavior. | Ablate the long or short branch on stable horizon slices. |
| NARM | Uses global session context plus attention to current intent. | Compare with recency pooling on sessions with defensible boundaries. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_1_retrieval/4.sequence/1.mind.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/4.sequence/1.mind.html), [docs/chapter_1_retrieval/4.sequence/2.sdm.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/4.sequence/2.sdm.html), [src/funrec/models/mind.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/mind.py), [src/funrec/models/sdm.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/sdm.py).

Additional pinned source for `narm`: [torch_rechub/models/matching/narm.py](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/matching/narm.py).
