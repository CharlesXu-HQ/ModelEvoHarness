# Two-tower matching

A user encoder and an item encoder produce vectors whose dot product or cosine similarity supports indexed nearest-neighbor retrieval. The item tower can be computed in advance; the user tower runs at request time. This separation is the operational reason to build a two-tower model, while its limited cross-tower interaction is a quality tradeoff. FM-style matching preserves some feature interactions; YouTubeDNN and DSSM differ in how histories and features form the query vector.

The usual score is `s(u,i) = f_user(u)^T f_item(i)`. Because `f_item(i)` does not depend on the current user, item vectors can be indexed before a request; any feature that requires simultaneous access to both sides belongs in a later reranker or a different retrieval method.

## Data and conditions

Require a catalog, positive user-item pairs, pre-decision user context, item features, and an explicit negative-sampling distribution. Store the serving vector dimension, normalization, index version, and item freshness. In-batch negatives are sampled from training traffic, often with a popularity-skewed distribution; a correction needs actual sampling probabilities, not guessed weights.

## Controlled experiment

Compare a simple popularity/collaborative retriever, a small two-tower model, and one proposed encoder or loss change on the same pairs and negatives. Evaluate full-catalog or fixed-candidate recall and downstream value at a fixed candidate count. Record index build cost, lookup latency, and stale-vector degradation. For sampling correction, compare corrected and uncorrected logits on identical batches.

## Interpretation

If recall improves only on sampled evaluation, the negative distribution may be the cause. If recall is good but final decisions are poor, inspect the ranker or policy. Request new user/item fields only after time-safe existing features and encoder ablations fail, unless business knowledge immediately identifies an essential absent field.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| DSSM | Encodes two sides independently and compares normalized vectors. | Compare a small tower with a linear or collaborative retriever at equal K. |
| FM recall | Rearranges an FM-like score into separable user and item vectors. | Check whether retained pair terms beat a plain tower under the same index. |
| YouTubeDNN | Encodes user context and pooled history for sampled-softmax item matching. | Ablate history while holding negatives and catalog fixed. |
| YouTubeSBC (`youtube_sbc`) | Adjusts in-batch logits using recorded item sampling weights. | Compare corrected and uncorrected logits on identical batches. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_1_retrieval/3.two_tower/1.fm.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/3.two_tower/1.fm.html), [docs/chapter_1_retrieval/3.two_tower/2.dssm.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/3.two_tower/2.dssm.html), [src/funrec/models/fm_recall.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/fm_recall.py), [src/funrec/models/dssm.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/dssm.py).

Additional pinned source for `youtube_sbc`: [torch_rechub/models/matching/youtube_sbc.py](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/matching/youtube_sbc.py).
