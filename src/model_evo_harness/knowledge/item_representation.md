# Item embeddings and graph retrieval

Item2Vec treats co-occurring items as contextual neighbors; graph-style embedding aggregates user-item or item-side relations. The resulting vectors offer candidate recall from behavior structure without scoring every catalog item. A session co-view relation and a purchase relation mean different things, so edge type and time window are part of the model hypothesis.

## Data and conditions

Require item identity, context windows or graph edges available before evaluation, and a catalog with enough repeated support. EGES-like side information can mix item ID and attribute embeddings, which helps sparse IDs only when those attributes exist at serving time. Do not construct edges from validation or future interactions.

## Controlled experiment

Hold training windows, seed items, index, candidate budget, and ranker fixed. Compare co-occurrence counts, plain item embeddings, and an attribute-aware variant; ablate each side attribute. Evaluate recall, coverage, tail behavior, vector freshness, and latency. Compare a shuffled-context control when sequence context is the claimed source of gain.

## Interpretation and feature needs

An attribute-aware gain only on sparse items supports a cold-item explanation. If item descriptions or categories are absent but are intrinsic to the business catalog, a domain expert can request them immediately. If attributes exist yet gains plateau, check representation and edge construction before claiming the dataset lacks more fields.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| Item2Vec | Learns an item vector from nearby events in histories. | Shuffle context windows to test whether local co-occurrence adds signal. |
| EGES | Mixes item-ID and side-attribute embeddings by item-specific weights. | Compare ID-only and side-only retrieval, especially for low-history items. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_1_retrieval/2.i2i/1.word2vec.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/2.i2i/1.word2vec.html), [docs/chapter_1_retrieval/2.i2i/2.item2vec.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/2.i2i/2.item2vec.html), [src/funrec/models/item2vec.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/item2vec.py), [src/funrec/models/eges.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/eges.py).
